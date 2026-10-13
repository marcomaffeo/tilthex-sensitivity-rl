"""Containers of the parametric sensitivity state, carried along the trajectory by the sensitivity wrappers."""

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class Sensitivity:
    """Parametric sensitivity of the closed loop, carried along the trajectory.

    Attributes:
        log_norm (jax.Array): Log of the weighted Frobenius norm of the state sensitivity.
        state (jax.Array): State sensitivity ``dx/dp``, shape ``(state_dim, param_dim)``.
        action (jax.Array): Action sensitivity ``du/dp``, shape ``(action_dim, param_dim)``.
        policy_gains (jax.Array): Policy Jacobian ``dmu/dobs``, shape ``(action_dim, obs_dim)``.
    """
    log_norm: jax.Array
    state: jax.Array
    action: jax.Array
    policy_gains: jax.Array

    def default(env):
        """Zero sensitivity state for an environment.

        Args:
            env (environment.Environment): Environment that provides ``state_dim``, ``action_dim``, ``obs_dim`` and ``param_dim``.

        Returns:
            Sensitivity: Sensitivity with zero matrices and ``log_norm = log(1e-6)``.
        """
        return Sensitivity(
            log_norm=jnp.array(jnp.log(1e-6)),
            state=jnp.zeros((env.state_dim, env.param_dim)),  # ds/dp
            action=jnp.zeros((env.action_dim, env.param_dim)),  # du/dp
            policy_gains=jnp.zeros((env.action_dim, env.obs_dim)),  # dmu/dobs
        )


@struct.dataclass
class SensitivityOff:
    """Parametric sensitivity for the off-policy algorithms, with the Jacobians needed by their losses.

    Attributes:
        log_norm (jax.Array): Log of the weighted Frobenius norm of the state sensitivity.
        log_norm_old (jax.Array): The same value at the previous step.
        state (jax.Array): State sensitivity ``dx/dp``, shape ``(state_dim, param_dim)``.
        F_sens_x (jax.Array): Policy-independent part of the update, ``df/dx S + df/dp``.
        df_du (jax.Array): Jacobian of the dynamics with respect to the action, shape ``(state_dim, action_dim)``.
        dobs_dx (jax.Array): Observation Jacobian times the state sensitivity, shape ``(obs_dim, param_dim)``.
    """
    log_norm: jax.Array
    log_norm_old: jax.Array
    state: jax.Array
    F_sens_x: jax.Array
    df_du: jax.Array
    dobs_dx: jax.Array
    
    def default(env):
        """Zero state of the off-policy sensitivity.

        Args:
            env (environment.Environment): Environment that provides ``state_dim``, ``action_dim``, ``obs_dim`` and ``param_dim``.

        Returns:
            SensitivityOff: Sensitivity with zero matrices and ``log_norm = log(1e-6)``.
        """
        return SensitivityOff(
            log_norm=jnp.array(jnp.log(1e-6)),
            log_norm_old=jnp.array(jnp.log(1e-6)),
            state=jnp.zeros((env.state_dim, env.param_dim)),  # ds/dp
            F_sens_x=jnp.zeros((env.state_dim, env.param_dim)),  # df/dx @ S + df/dp (normalised)
            df_du=jnp.zeros((env.state_dim, env.action_dim)),  # df/du
            dobs_dx=jnp.zeros((env.obs_dim, env.param_dim)),  # dobs/dx @ S
        )
