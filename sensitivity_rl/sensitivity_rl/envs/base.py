import jax
import jax.flatten_util

import chex
from flax import struct
from typing import Union

from gymnax.environments import environment

from abc import ABC, abstractmethod

@struct.dataclass
class Sensitivity():
    state: jax.Array
    action: jax.Array
    policy_gains: jax.Array

class EnvSensitivity(environment.Environment, ABC):
    """Add sensitivity computation to compatible environments.
    This base class assumes that the environment has a method `step_dynamics` that computes
    the next state given the current state, action, and uncertain parameters.
    """
    def __init__(self):
        super().__init__()

        self.state_unravel = jax.flatten_util.ravel_pytree(
            self.reset(jax.random.PRNGKey(0))[1].state)[1]

        self.param_unravel = jax.flatten_util.ravel_pytree(
            self.default_params)[1]

        self.obs_unravel = jax.flatten_util.ravel_pytree(
            self.get_obs(self.reset(jax.random.PRNGKey(0))[1].state))[1]

        self.df_dx_flat = jax.jacfwd(self.step_dynamics_flattened, argnums=1)
        self.df_du_flat = jax.jacfwd(self.step_dynamics_flattened, argnums=2)
        self.df_dp_flat = jax.jacfwd(self.step_dynamics_flattened, argnums=3)

        self.dobs_dx_flat = jax.jacfwd(self.get_obs_flattened)

    @abstractmethod
    def step_dynamics(self, key, state, action, params):
        """Compute the next state given current state, action, and parameters.
        This method should be implemented by the specific environment.
        """
        pass

    def step_dynamics_flattened(
        self,
        key: chex.PRNGKey,
        state_flat: Union[float, jax.Array],
        action: Union[float, jax.Array],
        params_flat: Union[float, jax.Array],
    ) -> Union[float, jax.Array]:

        state = self.state_unravel(state_flat)
        params = self.param_unravel(params_flat)
        state_next = self.step_dynamics(key, state, action, params)
        return jax.flatten_util.ravel_pytree(state_next)[0]

    def df_dx(self, key, state, action, params):
        state_flat = jax.flatten_util.ravel_pytree(state)[0]
        params_flat = jax.flatten_util.ravel_pytree(params)[0]
        return self.df_dx_flat(key, state_flat, action, params_flat)

    def df_du(self, key, state, action, params):
        state_flat = jax.flatten_util.ravel_pytree(state)[0]
        params_flat = jax.flatten_util.ravel_pytree(params)[0]
        return self.df_du_flat(key, state_flat, action, params_flat)

    def df_dp(self, key, state, action, params):
        state_flat = jax.flatten_util.ravel_pytree(state)[0]
        params_flat = jax.flatten_util.ravel_pytree(params)[0]
        return self.df_dp_flat(key, state_flat, action, params_flat)

    def get_obs_flattened(self, state_flat):
        state = self.state_unravel(state_flat)
        obs = self.get_obs(state)
        return jax.flatten_util.ravel_pytree(obs)[0]

    def dobs_dx(self, state):
        state_flat = jax.flatten_util.ravel_pytree(state)[0]
        return self.dobs_dx_flat(state_flat)

    def step_sensitivity(
        self,
        key: chex.PRNGKey,
        env_state,
        action: Union[int, float, jax.Array],
        params,
    ):
        """Compute the next environment state and update sensitivities."""

        # Compute Jacobians
        df_dx = self.df_dx(key, env_state.state, action, params)
        df_du = self.df_du(key, env_state.state, action, params)
        df_dp = self.df_dp(key, env_state.state, action, params)
        dobs_dx = self.dobs_dx(env_state.state)

        # Unpack sensitivities
        sens = env_state.sensitivity
        sens_s = sens.state  # ds/dp
        dmu_dobs = sens.policy_gains  # dmu/dobs
        # Update action sensitivity
        sens_a = dmu_dobs @ dobs_dx @ sens_s  # du/dp
        # Update sensitivities using chain rule
        sens_s_next = df_dx @ sens_s + df_du @ sens_a + df_dp

        # Pack updated sensitivitie
        return Sensitivity(
            state=sens_s_next,
            action=sens_a,
            policy_gains=dmu_dobs,
        )
