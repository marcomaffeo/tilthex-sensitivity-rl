import jax
import jax.numpy as jnp
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Unified rollout types for all PPO-family algorithms.
#
# Algorithm-specific fields live in the `extras` dict, which is a valid JAX
# pytree container.  Each algorithm populates a fixed set of keys so the
# structure stays constant within a single JIT-compiled training function.
#
# Examples:
#   PPO / RAPPO  : extras = {}
#   SPPO / SPPO_GRAD : extras = {"gru_hidden": ...}
#   SPPO_AUG     : extras = {"value_s": ..., "reward_s": ..., "gru_hidden": ...}
#   CPPO         : extras = {"nu": ..., "cvarlam": ...}   (runner/update only)
# ---------------------------------------------------------------------------


class Transition(NamedTuple):
    """Rollout transition for all PPO-family algorithms."""
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: dict
    extras: dict = {}


class RunnerState(NamedTuple):
    """Runner state for all PPO-family algorithms."""
    train_state: object  # Flax TrainState
    env_state: object
    last_obs: jnp.ndarray
    rng: jnp.ndarray
    extras: dict = {}


class UpdateState(NamedTuple):
    """Update state for all PPO-family algorithms."""
    train_state: object
    traj_batch: object
    advantages: jnp.ndarray
    targets: jnp.ndarray
    rng: jnp.ndarray
    extras: dict = {}


