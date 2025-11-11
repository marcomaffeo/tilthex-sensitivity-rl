import jax
from flax import struct


@struct.dataclass
class Sensitivity:
    log_norm: jax.Array
    reward_norm: jax.Array
    state: jax.Array
    action: jax.Array
    policy_gains: jax.Array
