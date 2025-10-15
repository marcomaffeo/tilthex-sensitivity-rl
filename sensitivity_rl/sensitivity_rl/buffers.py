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
