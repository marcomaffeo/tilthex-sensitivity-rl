import jax
from flax import struct

@struct.dataclass
class Sensitivity():
    state: jax.Array
    action: jax.Array
    policy_gains: jax.Array

