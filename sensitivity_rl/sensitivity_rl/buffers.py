import jax
import jax.numpy as jnp
from typing import NamedTuple


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


# Runner state NamedTuples for different algorithms
class RunnerState(NamedTuple):
    """Runner state for single-network algorithms (e.g., PPO)."""
    train_state: object  # Flax TrainState
    env_state: object
    last_obs: jnp.ndarray
    rng: jnp.ndarray


class RunnerStateDual(NamedTuple):
    """Runner state for dual-network algorithms (e.g., SPPO-aug with GRU)."""
    train_state: object  # Main network TrainState
    train_state_vs: object  # Sensitivity value network TrainState
    env_state: object
    last_obs: jnp.ndarray
    gru_hidden: jnp.ndarray
    rng: jnp.ndarray


# Update state NamedTuples for different algorithms
class UpdateState(NamedTuple):
    """Update state for single-network algorithms (e.g., PPO)."""
    train_state: object
    traj_batch: object
    advantages: jnp.ndarray
    targets: jnp.ndarray
    rng: jnp.ndarray


class UpdateStateDual(NamedTuple):
    """Update state for dual-network algorithms (e.g., SPPO-aug)."""
    train_state: object
    train_state_vs: object
    traj_batch: object
    advantages: jnp.ndarray
    advantages_s: jnp.ndarray
    targets: jnp.ndarray
    targets_s: jnp.ndarray
    rng: jnp.ndarray


class RolloutGRUTransition(NamedTuple):
    """Transition for SPPO with single GRU-based ActorCritic."""
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray  
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    gru_hidden: jnp.ndarray
    info: jnp.ndarray


class RunnerStateGRU(NamedTuple):
    """Runner state for GRU-based single-network algorithms (e.g., SPPO)."""
    train_state: object  # Flax TrainState
    env_state: object
    last_obs: jnp.ndarray
    gru_hidden: jnp.ndarray
    rng: jnp.ndarray


class ReplayTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray

    def init_mean(self):
        return jnp.array(0.0)

    def init_var(self):
        return jnp.array(0.0)

    def compute_mean_std(self, current_mean, current_var, new_size):
        delta = current_mean[0] - self.reward
        new_mean = current_mean[0] + delta * self.reward.shape[0] / new_size
        m_a = current_var[0] * new_size
        m_b = jnp.var(self.reward) * self.reward.shape[0]
        M2 = (
            m_a
            + m_b
            + jnp.square(delta) * (new_size - self.reward.shape[0]) / new_size
        )
        new_var = M2 / new_size

        return jnp.array(new_mean), jnp.array(new_var)


class ReplaySensTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    
    def init_mean(self):
        return jnp.array([0.0])

    def init_var(self):
        return jnp.array([0.0])

    def compute_mean_std(self, current_mean, current_var, new_size):
        delta = current_mean[0] - self.reward
        new_mean = current_mean[0] + jnp.mean(delta) * self.reward.shape[0] / new_size
        m_a = current_var[0] * new_size
        m_b = jnp.var(self.reward) * self.reward.shape[0]
        M2 = (
            m_a
            + m_b
            + jnp.mean(jnp.square(delta)) * (new_size - self.reward.shape[0]) / new_size
        )
        new_var = M2 / new_size

        return jnp.array([new_mean]), jnp.array([new_var])


class ReplaySensOnlyTransition(NamedTuple):
    """Transition containing only sensitivity-related data"""
    done: jnp.ndarray
    action: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    sens: jnp.ndarray
    sens_next: jnp.ndarray
    F_sens_x: jnp.ndarray
    F_sens_x_next: jnp.ndarray
    df_du: jnp.ndarray
    df_du_next: jnp.ndarray
    dobs_dx: jnp.ndarray
    dobs_dx_next: jnp.ndarray
    log_sens_norm: jnp.ndarray
    log_sens_next_norm: jnp.ndarray

    def init_mean(self):
        return jnp.array([0.0])

    def init_var(self):
        return jnp.array([0.0])

    def compute_mean_std(self, current_mean, current_var, new_size):
        # current_mean and current_var are shape [1] arrays
        # self.log_sens_norm is shape (batch_size,) 
        delta = current_mean[0] - self.log_sens_norm
        # Compute mean of delta for the update
        mean_delta = jnp.mean(delta)
        new_mean_sens = current_mean[0] + mean_delta * self.log_sens_norm.shape[0] / new_size
        
        m_a = current_var[0] * new_size
        m_b = jnp.var(self.log_sens_norm) * self.log_sens_norm.shape[0]
        M2 = (
            m_a
            + m_b
            + jnp.mean(jnp.square(delta)) * (new_size - self.log_sens_norm.shape[0]) / new_size
        )
        new_var_sens = M2 / new_size

        # Return shape [1] arrays to match init_mean() and init_var()
        return jnp.array([new_mean_sens], dtype=jnp.float32), jnp.array([new_var_sens], dtype=jnp.float32)


class ReplayBufferState(NamedTuple):
    transitions: (
        ReplayTransition | ReplaySensTransition | ReplaySensOnlyTransition
    )  # shape [buffer_size, ...]
    position: jnp.ndarray  # scalar int32
    size: jnp.ndarray  # scalar int32
    mean: jnp.ndarray  # mean of the transitions
    var: jnp.ndarray  # standard deviation of the transitions


@jax.jit
def add_batch_to_buffer(
    buffer_state: ReplayBufferState,
    new_transitions: ReplayTransition | ReplaySensTransition | ReplaySensOnlyTransition,
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
    # Update mean and std of the transitions
    new_mean, new_var = new_transitions.compute_mean_std(
        buffer_state.mean, buffer_state.var, new_size
    )

    return ReplayBufferState(
        updated_transitions, new_position, new_size, new_mean, new_var
    )


def sample_batch_from_buffer(
    buffer_state: ReplayBufferState, batch_size: int, rng: jax.Array
) -> ReplayTransition | ReplaySensTransition | ReplaySensOnlyTransition:
    max_index = buffer_state.size
    rng, key = jax.random.split(rng)
    indices = jax.random.randint(key, (batch_size,), 0, max_index)

    sampled_transitions = jax.tree_util.tree_map(
        lambda data: data[indices], buffer_state.transitions
    )
    return sampled_transitions


def create_empty_buffer(
    buffer_size: int,
    example_transition: ReplayTransition | ReplaySensTransition | ReplaySensOnlyTransition,
) -> ReplayBufferState:
    """Create an empty buffer state given a sample transition for shape inference"""
    empty_transitions = jax.tree_util.tree_map(
        lambda x: jnp.zeros((buffer_size,) + x.shape[1:], dtype=x.dtype),
        example_transition,
    )

    return ReplayBufferState(
        empty_transitions,
        jnp.array(0, jnp.int32),
        jnp.array(0, jnp.int32),
        example_transition.init_mean(),
        example_transition.init_var(),
    )
