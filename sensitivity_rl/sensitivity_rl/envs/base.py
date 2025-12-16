import jax
from flax import struct


@struct.dataclass
class Sensitivity:
    log_norm: jax.Array
    state: jax.Array
    action: jax.Array
    policy_gains: jax.Array


@struct.dataclass
class SensitivityOff:
    log_norm: jax.Array
    log_norm_old: jax.Array
    state: jax.Array
    F_sens_x: jax.Array
    df_du: jax.Array
    dobs_dx: jax.Array
