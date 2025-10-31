import jax
import jax.numpy as jnp
from typing import NamedTuple


def _calculate_gae(config, traj_batch, last_val):
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        done, value, reward = (
            transition.done,
            transition.value,
            transition.reward,
        )
        delta = reward + config["GAMMA"] * next_value * (1 - done) - value
        gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + traj_batch.value


def _calculate_gae_s(config, traj_batch, last_val, last_val_s):
    def _get_advantages(gae_and_next_value, transition):
        gae, gae_s, next_value, next_value_s = gae_and_next_value
        done, value, value_s, reward, reward_s = (
            transition.done,
            transition.value,
            transition.value_s,
            transition.reward,
            transition.reward_s,
        )
        delta = reward + config["GAMMA"] * next_value * (1 - done) - value
        gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
        delta_s = (
            reward_s + config["GAMMA"] * next_value_s * (1 - done) - value_s
        )
        gae_s = (
            delta_s
            + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae_s
        )
        return (gae, gae_s, value, value_s), (gae, gae_s)

    _, (advantages, advantages_s) = jax.lax.scan(
        _get_advantages,
        (
            jnp.zeros_like(last_val),
            jnp.zeros_like(last_val),
            last_val,
            last_val_s,
        ),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    return (
        advantages,
        advantages_s,
        advantages + traj_batch.value,
        advantages_s + traj_batch.value_s,
    )


class RolloutTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


class RolloutSensTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    value_s: jnp.ndarray
    reward: jnp.ndarray
    reward_s: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


class RolloutSensAugTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    value_s: jnp.ndarray
    reward: jnp.ndarray
    reward_s: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    gru_hidden: jnp.ndarray  # GRU hidden state instead of sens_state
    info: jnp.ndarray


class ReplayTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray


class ReplayBufferState(NamedTuple):
    transitions: ReplayTransition  # shape [buffer_size, ...]
    position: jnp.ndarray  # scalar int32
    size: jnp.ndarray  # scalar int32


@jax.jit
def add_batch_to_buffer(
    buffer_state: ReplayBufferState, new_transitions: ReplayTransition
) -> ReplayBufferState:
    batch_size = new_transitions.obs.shape[0]
    buffer_size = buffer_state.transitions.obs.shape[0]

    indices = (jnp.arange(batch_size) + buffer_state.position) % buffer_size

    # update each field in transitions
    updated_transitions = jax.tree_util.tree_map(
        lambda buf, new: buf.at[indices].set(new),
        buffer_state.transitions,
        new_transitions,
    )

    new_position = (buffer_state.position + batch_size) % buffer_size
    new_size = jnp.minimum(buffer_state.size + batch_size, buffer_size)

    return ReplayBufferState(updated_transitions, new_position, new_size)


def sample_batch_from_buffer(
    buffer_state: ReplayBufferState, batch_size: int, rng: jax.Array
) -> ReplayTransition:
    max_index = buffer_state.size
    rng, key = jax.random.split(rng)
    indices = jax.random.randint(key, (batch_size,), 0, max_index)

    sampled_transitions = jax.tree_util.tree_map(
        lambda data: data[indices], buffer_state.transitions
    )
    return sampled_transitions


def create_empty_buffer(
    buffer_size: int, example_transition: ReplayTransition
) -> ReplayBufferState:
    """Create an empty buffer state given a sample transition for shape inference"""
    empty_transitions = jax.tree_util.tree_map(
        lambda x: jnp.zeros((buffer_size,) + x.shape[1:], dtype=x.dtype),
        example_transition,
    )
    return ReplayBufferState(
        empty_transitions, jnp.array(0, jnp.int32), jnp.array(0, jnp.int32)
    )
