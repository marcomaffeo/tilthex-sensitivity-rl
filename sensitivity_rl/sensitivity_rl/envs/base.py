import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class Sensitivity:
    log_norm: jax.Array
    state: jax.Array
    action: jax.Array
    policy_gains: jax.Array

    def default(env):
        return Sensitivity(
            log_norm=jnp.array(jnp.log(1e-6)),
            state=jnp.zeros((env.state_dim, env.param_dim)),  # ds/dp
            action=jnp.zeros((env.action_dim, env.param_dim)),  # du/dp
            policy_gains=jnp.zeros((env.action_dim, env.obs_dim)),  # dmu/dobs
        )


@struct.dataclass
class SensitivityOff:
    log_norm: jax.Array
    log_norm_old: jax.Array
    state: jax.Array
    F_sens_x: jax.Array
    df_du: jax.Array
    dobs_dx: jax.Array
    
    def default(env):
        return SensitivityOff(
            log_norm=jnp.array(jnp.log(1e-6)),
            log_norm_old=jnp.array(jnp.log(1e-6)),
            state=jnp.zeros((env.state_dim, env.param_dim)),  # ds/dp
            F_sens_x=jnp.zeros((env.state_dim, env.param_dim)),  # df/dx @ S + df/dp (normalised)
            df_du=jnp.zeros((env.state_dim, env.action_dim)),  # df/du
            dobs_dx=jnp.zeros((env.obs_dim, env.param_dim)),  # dobs/dx @ S
        )

