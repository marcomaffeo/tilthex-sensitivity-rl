import jax
import jax.numpy as jnp
import jax.flatten_util
import chex
import numpy as np
from flax import struct
from typing import Optional, Tuple, Union
from gymnax.environments import environment, spaces
from envs.base import Sensitivity, SensitivityOff
from utils import (
    angle_normalize,
    normalize,
    default_sensitivity,
    default_sensitivity_off,
)


def dataclass_with_custom_replace(cls):
    cls = struct.dataclass(cls)
    cls.replace = (
        cls._replace  # reuse the same _replace function (or ReplaceMixin.replace)
    )
    return cls


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class FlattenObservationWrapper(GymnaxWrapper):
    """Flatten the observations of the environment."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    def observation_space(self, params) -> spaces.Box:
        assert isinstance(
            self._env.observation_space(params), spaces.Box
        ), "Only Box spaces are supported for now."
        return spaces.Box(
            low=self._env.observation_space(params).low,
            high=self._env.observation_space(params).high,
            shape=(np.prod(self._env.observation_space(params).shape),),
            dtype=self._env.observation_space(params).dtype,
        )

    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, state = self._env.reset(key, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state

    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(
            key, state, action, params
        )
        obs = jnp.reshape(obs, (-1,))
        return obs, state, reward, done, info


@dataclass_with_custom_replace
class LogEnvState:
    env_state: environment.EnvState
    episode_returns_true: float
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_returns_true: float
    returned_episode_lengths: int
    timestep: int

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self.env_state, name)

    def _replace(self, **kwargs):
        """Return a copy with updated fields (Flax-style replace)."""
        return LogEnvState(
            self.env_state.replace(**kwargs),
            self.episode_returns_true,
            self.episode_returns,
            self.episode_lengths,
            self.returned_episode_returns,
            self.returned_episode_returns_true,
            self.returned_episode_lengths,
            self.timestep,
        )


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0.0, 0.0, 0, 0.0, 0.0, 0, 0)
        return obs, state

    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, LogEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return_true = state.episode_returns_true + info.get(
            "trw", reward
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns_true=new_episode_return_true * (1 - done),
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns_true=state.returned_episode_returns_true
            * (1 - done)
            + new_episode_return_true * done,
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns_true"] = (
            state.returned_episode_returns_true
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info


class ClipAction(GymnaxWrapper):
    def __init__(self, env, low=-1.0, high=1.0):
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, key, state, action, params=None):
        action = jnp.clip(
            action,
            self._env.action_space(params).low,
            self._env.action_space(params).high,
        )
        return self._env.step(key, state, action, params)


class TransformObservation(GymnaxWrapper):
    def __init__(self, env, transform_obs):
        super().__init__(env)
        self.transform_obs = transform_obs

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        return self.transform_obs(obs), state

    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(
            key, state, action, params
        )
        return self.transform_obs(obs), state, reward, done, info


class TransformReward(GymnaxWrapper):
    def __init__(self, env, transform_reward):
        super().__init__(env)
        self.transform_reward = transform_reward

    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(
            key, state, action, params
        )
        return obs, state, self.transform_reward(reward), done, info


class VecEnv(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))


@dataclass_with_custom_replace
class NormalizeVecRewSensEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    mean_sens: jnp.ndarray
    var_sens: jnp.ndarray
    count: float
    return_val: float
    return_val_sens: jnp.ndarray
    env_state: environment.EnvState

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self.env_state, name)

    def _replace(self, **kwargs):
        """Return a copy with updated fields (Flax-style replace)."""
        return struct.replace(self.env_state, **kwargs)


class NormalizeVecRewardSensitivity(GymnaxWrapper):
    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        batch_count = obs.shape[0]
        state = NormalizeVecRewSensEnvState(
            mean=0.0,
            var=1.0,
            mean_sens=0.0,
            var_sens=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            return_val_sens=jnp.zeros((batch_count,)),
            env_state=state,
        )
        return obs, state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        return_val = state.return_val * self.gamma * (1 - done) + reward
        return_val_sens = state.return_val_sens * self.gamma * (
            1 - done
        ) + info.get("sensitivity_state_rw", 0.0)

        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_mean_sens = jnp.mean(return_val_sens, axis=0)
        batch_var_sens = jnp.var(return_val_sens, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = (
            m_a
            + m_b
            + jnp.square(delta) * state.count * batch_count / tot_count
        )
        new_var = M2 / tot_count
        new_count = tot_count

        delta_sens = batch_mean_sens - state.mean_sens
        new_mean_sens = state.mean_sens + delta_sens * batch_count / tot_count
        m_a_sens = state.var_sens * state.count
        m_b_sens = batch_var_sens * batch_count
        M2_sens = (
            m_a_sens
            + m_b_sens
            + jnp.square(delta_sens) * state.count * batch_count / tot_count
        )
        new_var_sens = M2_sens / tot_count
        state = NormalizeVecRewSensEnvState(
            mean=new_mean,
            var=new_var,
            mean_sens=new_mean_sens,
            var_sens=new_var_sens,
            count=new_count,
            return_val=return_val,
            return_val_sens=return_val_sens,
            env_state=env_state,
        )

        info["sensitivity_state_rw"] = info.get(
            "sensitivity_state_rw", 0.0
        ) / jnp.sqrt(state.var_sens + 1e-8)

        return obs, state, reward / jnp.sqrt(state.var + 1e-8), done, info


@dataclass_with_custom_replace
class NormalizeVecRewEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: float
    env_state: environment.EnvState

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self.env_state, name)

    def _replace(self, **kwargs):
        """Return a copy with updated fields (Flax-style replace)."""
        return struct.replace(self.env_state, **kwargs)


class NormalizeVecReward(GymnaxWrapper):
    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        batch_count = obs.shape[0]
        state = NormalizeVecRewEnvState(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            env_state=state,
        )
        return obs, state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        return_val = state.return_val * self.gamma * (1 - done) + reward

        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = (
            m_a
            + m_b
            + jnp.square(delta) * state.count * batch_count / tot_count
        )
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecRewEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            env_state=env_state,
        )
        return obs, state, reward / jnp.sqrt(state.var + 1e-8), done, info


@dataclass_with_custom_replace
class SensitivityEnvState:
    env_state: environment.EnvState
    sensitivity: Sensitivity

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self.env_state, name)

    def _replace(self, **kwargs):
        """Return a copy with updated fields (Flax-style replace)."""
        return struct.replace(self.env_state, **kwargs)


@dataclass_with_custom_replace
class SensitivityOffEnvState:
    env_state: environment.EnvState
    sensitivity: SensitivityOff

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self.env_state, name)

    def _replace(self, **kwargs):
        """Return a copy with updated fields (Flax-style replace)."""
        return struct.replace(self.env_state, **kwargs)


class SensitivityWrapper(GymnaxWrapper):
    def __init__(
        self, env, regularization_weight=0.0, add_sensitivity_to_info=False
    ):
        super().__init__(env)

        self.state_unravel = jax.flatten_util.ravel_pytree(
            self._env.reset(jax.random.PRNGKey(0))[1].state
        )[1]

        self.param_unravel = jax.flatten_util.ravel_pytree(
            self._env.default_params
        )[1]

        self.regularization_weight = regularization_weight
        self.add_sensitivity_to_info = add_sensitivity_to_info

        self.action_gains = jax.jacfwd(self.compute_action, argnums=1)
        self.df_dx = jax.jacfwd(self._step_to_state_wrapped, argnums=2)
        self.df_du = jax.jacfwd(self._step_to_state_wrapped, argnums=3)
        self.df_dp = jax.jacfwd(self._step_to_state_wrapped, argnums=4)
        self.dobs_dx = jax.jacfwd(self._step_to_obs_wrapped, argnums=2)

    def set_actor(self, actor, network_params):
        # Should add a check for dimension compatibility
        self._actor = actor
        self.network_params = network_params

    def update_network_params(self, network_params):
        self.network_params = network_params

    def sensitivity_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.state_dim * self._env.param_dim,),
        )

    def compute_action(self, rng, obs, network_params):
        pi, _ = self._actor.apply(network_params, obs)
        action = pi.sample(seed=rng)
        return action

    def step_sensitivity(
        self,
        key: chex.PRNGKey,
        sens_state: SensitivityEnvState,
        action: Union[int, float, jax.Array],
        params,
        obs,
        done,
    ):
        """Compute the next environment state and update sensitivities."""
        state = jax.flatten_util.ravel_pytree(sens_state.state)[0]
        params = jax.flatten_util.ravel_pytree(params)[0]
        # Compute Jacobians
        df_dx = self.df_dx(key, sens_state.env_state, state, action, params)
        df_du = self.df_du(key, sens_state.env_state, state, action, params)
        df_dp = self.df_dp(key, sens_state.env_state, state, action, params)
        dobs_dx = self.dobs_dx(key, sens_state.env_state, state, action, params)

        """
        jax.debug.callback(
            lambda x, y, z, w, d: (
                print(f"Sensitivity jacobians: {x},{y},{z},{w}") if d else None
            ),
            jnp.linalg.matrix_norm(df_dx, ord="fro"),
            jnp.linalg.matrix_norm(df_du, ord="fro"),
            jnp.linalg.matrix_norm(df_dp, ord="fro"),
            jnp.linalg.matrix_norm(dobs_dx, ord="fro"),
            done,
        )
        """
        df_dx = jnp.clip(df_dx, -1e3, 1e3)
        df_du = jnp.clip(df_du, -1e3, 1e3)
        df_dp = jnp.clip(df_dp, -1e3, 1e3)
        dobs_dx = jnp.clip(dobs_dx, -1e3, 1e3)

        # Unpack sensitivities
        sens_s = sens_state.sensitivity.state  # ds/dp
        dmu_dobs = self.action_gains(key, obs, self.network_params)  # dmu/dobs

        dmu_dobs = jnp.clip(dmu_dobs, -1e3, 1e3)

        # Update action sensitivity
        sens_a = dmu_dobs @ dobs_dx @ sens_s  # du/dp
        # Update sensitivities using chain rule
        sens_s_next = df_dx @ sens_s + df_du @ sens_a + df_dp/(jnp.exp(jnp.clip(sens_state.sensitivity.log_norm, -1e4, 1e10)+1e-4))
        
        norm_next = jnp.linalg.matrix_norm(sens_s_next, ord="fro")
        sens_s_next = sens_s_next / (norm_next + 1e-6)

        # Pack updated sensitivitie
        return Sensitivity(
            log_norm=sens_state.sensitivity.log_norm
            + jnp.log(norm_next + 1e-6),
            state=sens_s_next,
            action=sens_a,
            policy_gains=dmu_dobs,
        )

    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ):
        obs, env_state = self._env.reset(key, params)
        sensitivity = default_sensitivity(self._env)
        state = SensitivityEnvState(
            env_state=env_state, sensitivity=sensitivity
        )
        return obs, state

    def step(
        self,
        key: chex.PRNGKey,
        sens_state: SensitivityEnvState,
        action: Union[int, float, jax.Array],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:

        obs, env_state, reward, done, info = self._env.step(
            key, sens_state.env_state, action, params
        )

        next_sensitivity = self.step_sensitivity(
            key, sens_state, action, params, obs, done
        )

        info["trw"] = reward
        info["sensitivity_state"] = next_sensitivity.log_norm

        # Final sensitivity reward regularization
        # reward = jax.lax.select(
        #   done,
        #    reward
        #    - self.regularization_weight
        #    * jnp.linalg.matrix_norm(next_sensitivity.state, ord="fro"),
        #    reward,
        # )

        # Sensitivity difference reward regularization (next sensitivity whould be smaller than previous one)
        #reward = reward - self.regularization_weight * (
        #    next_sensitivity.log_norm - sens_state.sensitivity.log_norm
        #)

        sens_state = SensitivityEnvState(
            env_state=env_state, sensitivity=next_sensitivity
        )

        return obs, sens_state, reward, done, info

    # We need to wrap these functions to convert from flat state/param to pytree and differentiate through env.step
    def _step_to_state_wrapped(
        self, key, env_state, state_flat, action, params_flat
    ):
        state = self.state_unravel(state_flat)
        params = self.param_unravel(params_flat)
        env_state = env_state.replace(state=state)
        obs, env_state, reward, done, info = self._env.step(
            key, env_state, action, params
        )
        return jax.flatten_util.ravel_pytree(env_state.state)[0]

    def _step_to_obs_wrapped(
        self, key, env_state, state_flat, action, params_flat
    ):
        state = self.state_unravel(state_flat)
        params = self.param_unravel(params_flat)
        env_state = env_state.replace(state=state)
        obs, env_state, reward, done, info = self._env.step(
            key, env_state, action, params
        )
        return jax.flatten_util.ravel_pytree(obs)[0]


class MujocoSensitivityWrapper(SensitivityWrapper):
    def __init__(
        self, env, regularization_weight=0.0, add_sensitivity_to_info=False
    ):
        super().__init__(env, regularization_weight, add_sensitivity_to_info)

    # We need to wrap these functions to convert from flat state/param to pytree and differentiate through env.step
    def _step_to_state_wrapped(
        self, key, env_state, state_flat, action, params_flat
    ):
        state = self.state_unravel(state_flat)
        params = self.param_unravel(params_flat)
        next_state = self._env.step_dynamics(key, state, action, params)
        return jax.flatten_util.ravel_pytree(next_state)[0]

    def _step_to_obs_wrapped(
        self, key, env_state, state_flat, action, params_flat
    ):
        state = self.state_unravel(state_flat)
        params = self.param_unravel(params_flat)
        next_state = self._env.step_dynamics(key, state, action, params)
        # Create a dummy EnvState to pass to get_obs
        dummy_env_state = env_state.replace(state=next_state)
        obs = self._env.get_obs(dummy_env_state)
        return jax.flatten_util.ravel_pytree(obs)[0]


class SensitivityExtWrapper(GymnaxWrapper):
    def __init__(
        self, env, regularization_weight=0.0, add_sensitivity_to_info=False
    ):
        super().__init__(env)

        self.state_unravel = jax.flatten_util.ravel_pytree(
            self._env.reset(jax.random.PRNGKey(0))[1].state
        )[1]

        self.param_unravel = jax.flatten_util.ravel_pytree(
            self._env.default_params
        )[1]

        self.regularization_weight = regularization_weight
        self.add_sensitivity_to_info = add_sensitivity_to_info

        self.action_gains = jax.jacfwd(self.compute_action, argnums=1)
        self.df_dx = jax.jacfwd(self._step_to_state_wrapped, argnums=2)
        self.df_du = jax.jacfwd(self._step_to_state_wrapped, argnums=3)
        self.df_dp = jax.jacfwd(self._step_to_state_wrapped, argnums=4)
        self.dobs_dx = jax.jacfwd(self._step_to_obs_wrapped, argnums=2)

    def set_actor(self, actor, network_params):
        # Should add a check for dimension compatibility
        self._actor = actor
        self.network_params = network_params

    def sensitivity_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.state_dim * self._env.param_dim,),
        )

    def update_network_params(self, network_params):
        self.network_params = network_params

    def compute_action(self, rng, obs, network_params):
        pi, _ = self._actor.apply(network_params, obs)
        action = pi.sample(seed=rng)
        return action

    def step_sensitivity(
        self,
        key: chex.PRNGKey,
        sens_state: SensitivityEnvState,
        action: Union[int, float, jax.Array],
        params,
        obs,
        done,
    ):
        """Compute the next environment state and update sensitivities."""
        state = jax.flatten_util.ravel_pytree(sens_state.state)[0]
        params = jax.flatten_util.ravel_pytree(params)[0]
        # Compute Jacobians
        df_dx = self.df_dx(key, sens_state.env_state, state, action, params)
        df_du = self.df_du(key, sens_state.env_state, state, action, params)
        df_dp = self.df_dp(key, sens_state.env_state, state, action, params)
        dobs_dx = self.dobs_dx(key, sens_state.env_state, state, action, params)

        """
        jax.debug.callback(
            lambda x, y, z, w, d: (
                print(f"Sensitivity jacobians: {x},{y},{z},{w}") if d else None
            ),
            jnp.linalg.matrix_norm(df_dx, ord="fro"),
            jnp.linalg.matrix_norm(df_du, ord="fro"),
            jnp.linalg.matrix_norm(df_dp, ord="fro"),
            jnp.linalg.matrix_norm(dobs_dx, ord="fro"),
            done,
        )
        """
        df_dx = jnp.clip(df_dx, -1e3, 1e3)
        df_du = jnp.clip(df_du, -1e3, 1e3)
        df_dp = jnp.clip(df_dp, -1e3, 1e3)
        dobs_dx = jnp.clip(dobs_dx, -1e3, 1e3)
        
        # Replace NaN with 1e3
        df_dx = jnp.where(jnp.isnan(df_dx), 1e3, df_dx)
        df_du = jnp.where(jnp.isnan(df_du), 1e3, df_du)
        df_dp = jnp.where(jnp.isnan(df_dp), 1e3, df_dp)
        dobs_dx = jnp.where(jnp.isnan(dobs_dx), 1e3, dobs_dx)

        # Unpack sensitivities
        sens_s = sens_state.sensitivity.state  # ds/dp
        dmu_dobs = self.action_gains(key, obs, self.network_params)  # dmu/dobs
        
        dmu_dobs = jnp.clip(dmu_dobs, -1e3, 1e3)
        dmu_dobs = jnp.where(jnp.isnan(dmu_dobs), 1e3, dmu_dobs)
        
        # Update action sensitivity
        sens_a = dmu_dobs @ dobs_dx @ sens_s  # du/dp
        sens_a = jnp.where(jnp.isnan(sens_a), 1e3, sens_a)
        
        # Update sensitivities using chain rule
        sens_s_next = df_dx @ sens_s + df_du @ sens_a + df_dp/(jnp.exp(jnp.clip(sens_state.sensitivity.log_norm, -1e4, 1e10) +1e-4))
        sens_s_next = jnp.where(jnp.isnan(sens_s_next), 1e3, sens_s_next)
        
        # not normalized
        norm_next = jnp.linalg.matrix_norm(sens_s_next, ord="fro")
        
        # normalize
        sens_s_next = sens_s_next / (norm_next + 1e-6)
        # Pack updated sensitivitie
        return Sensitivity(
            log_norm=sens_state.sensitivity.log_norm
            + jnp.log(norm_next + 1e-6),
            state=sens_s_next,
            action=sens_a,
            policy_gains=dmu_dobs,
        )

    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ):
        obs, env_state = self._env.reset(key, params)
        sensitivity = default_sensitivity(self._env)
        state = SensitivityEnvState(
            env_state=env_state, sensitivity=sensitivity
        )
        return obs, state

    def step(
        self,
        key: chex.PRNGKey,
        sens_state: SensitivityEnvState,
        action: Union[int, float, jax.Array],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:

        obs, env_state, reward, done, info = self._env.step(
            key, sens_state.env_state, action, params
        )

        next_sensitivity = self.step_sensitivity(
            key, sens_state, action, params, obs, done
        )

        info["trw"] = reward

        info["sensitivity_state"] = next_sensitivity.log_norm
        info["sensitivity_state_rw"] = -(
            next_sensitivity.log_norm - sens_state.sensitivity.log_norm
        )

        sens_state = SensitivityEnvState(
            env_state=env_state,
            sensitivity=next_sensitivity,
        )

        return obs, sens_state, reward, done, info

    # We need to wrap these functions to convert from flat state/param to pytree and differentiate through env.step
    def _step_to_state_wrapped(
        self, key, env_state, state_flat, action, params_flat
    ):
        state = self.state_unravel(state_flat)
        params = self.param_unravel(params_flat)
        env_state = env_state.replace(state=state)
        obs, env_state, reward, done, info = self._env.step(
            key, env_state, action, params
        )
        return jax.flatten_util.ravel_pytree(env_state.state)[0]

    def _step_to_obs_wrapped(
        self, key, env_state, state_flat, action, params_flat
    ):
        state = self.state_unravel(state_flat)
        params = self.param_unravel(params_flat)
        env_state = env_state.replace(state=state)
        obs, env_state, reward, done, info = self._env.step(
            key, env_state, action, params
        )
        return jax.flatten_util.ravel_pytree(obs)[0]


class MujocoSensitivityExtWrapper(SensitivityExtWrapper):
    def __init__(
        self, env, regularization_weight=0.0, add_sensitivity_to_info=False
    ):
        super().__init__(env, regularization_weight, add_sensitivity_to_info)

    # We need to wrap these functions to convert from flat state/param to pytree and differentiate through env.step
    def _step_to_state_wrapped(
        self, key, env_state, state_flat, action, params_flat
    ):
        state = self.state_unravel(state_flat)
        params = self.param_unravel(params_flat)
        next_state = self._env.step_dynamics(key, state, action, params)
        return jax.flatten_util.ravel_pytree(next_state)[0]

    def _step_to_obs_wrapped(
        self, key, env_state, state_flat, action, params_flat
    ):
        state = self.state_unravel(state_flat)
        params = self.param_unravel(params_flat)
        next_state = self._env.step_dynamics(key, state, action, params)
        # Create a dummy EnvState to pass to get_obs
        dummy_env_state = env_state.replace(state=next_state)
        obs = self._env.get_obs(dummy_env_state)
        return jax.flatten_util.ravel_pytree(obs)[0]


class SensitivityOffWrapper(GymnaxWrapper):
    def __init__(
        self, env, regularization_weight=0.0, add_sensitivity_to_info=False
    ):
        super().__init__(env)

        self.state_unravel = jax.flatten_util.ravel_pytree(
            self._env.reset(jax.random.PRNGKey(0))[1].state
        )[1]

        self.param_unravel = jax.flatten_util.ravel_pytree(
            self._env.default_params
        )[1]

        self.regularization_weight = regularization_weight
        self.add_sensitivity_to_info = add_sensitivity_to_info

        self.action_gains = jax.jacfwd(self.compute_action, argnums=1)
        self._dynamics_jacs_fn = jax.jacfwd(
            self._step_to_state_and_obs_wrapped, argnums=(2, 3, 4)
        )

    def set_actor(
        self, actor, network_params, get_action, action_scale, action_bias
    ):
        # Should add a check for dimension compatibility
        self._actor = actor
        self.network_params = network_params
        self.get_action = get_action
        self.action_scale = action_scale
        self.action_bias = action_bias

    def update_network_params(self, network_params):
        self.network_params = network_params

    def compute_action(self, rng, obs, network_params):
        pi = self._actor.apply(network_params, obs)
        action, _, _, _ = self.get_action(
            pi, self.action_scale, self.action_bias, rng, jnp.zeros(self._env.action_space(self._env.default_params).shape), False
        )
        return action

    # TODO: Understand when doing this, in off-policy each step use a different policy
    def step_sensitivity(
        self,
        key: chex.PRNGKey,
        sens_state: SensitivityOffEnvState,
        action: Union[int, float, jax.Array],
        params,
        obs,
        done,
    ):
        """Compute the next environment state and update sensitivities."""
        state = jax.flatten_util.ravel_pytree(sens_state.state)[0]
        params_flat = jax.flatten_util.ravel_pytree(params)[0]
        # Compute Jacobians
        # _dynamics_jacs_fn returns: (jacs_of_next_state, jacs_of_obs)
        # where each is a tuple: (d_output/d_state, d_output/d_action, d_output/d_params)
        (jacs_next_state, jacs_obs) = self._dynamics_jacs_fn(
            key, sens_state.env_state, state, action, params_flat
        )

        # Unpack Jacobians for next_state
        df_dx = jacs_next_state[0]  # d_next_state/d_state
        df_du = jacs_next_state[1]  # d_next_state/d_action
        df_dp = jacs_next_state[2]  # d_next_state/d_params

        # Unpack Jacobians for obs
        dobs_dx = jacs_obs[0]  # d_obs/d_state

        df_dx = jnp.clip(df_dx, -1e3, 1e3)
        df_du = jnp.clip(df_du, -1e3, 1e3)
        df_dp = jnp.clip(df_dp, -1e3, 1e3)
        dobs_dx = jnp.clip(dobs_dx, -1e3, 1e3)
        
        # Replace NaN with 1e3
        df_dx = jnp.where(jnp.isnan(df_dx), 1e3, df_dx)
        df_du = jnp.where(jnp.isnan(df_du), 1e3, df_du)
        df_dp = jnp.where(jnp.isnan(df_dp), 1e3, df_dp)
        dobs_dx = jnp.where(jnp.isnan(dobs_dx), 1e3, dobs_dx)

        # Unpack sensitivities
        sens_s = sens_state.sensitivity.state  # ds/dp
        dmu_dobs = self.action_gains(key, obs, self.network_params)  # dmu/dobs
        
        dmu_dobs = jnp.clip(dmu_dobs, -1e3, 1e3)
        dmu_dobs = jnp.where(jnp.isnan(dmu_dobs), 1e3, dmu_dobs)
        
        # Update action sensitivity
        F_sens_u = df_du @ dmu_dobs @ dobs_dx @ sens_s
        F_sens_u = jnp.where(jnp.isnan(F_sens_u), 1e3, F_sens_u)
        
        F_sens_x = df_dx @ sens_s + df_dp/(jnp.exp(jnp.clip(sens_state.sensitivity.log_norm, -1e4, 1e10)+1e-4))
        F_sens_x = jnp.where(jnp.isnan(F_sens_x), 1e3, F_sens_x)
        
        # Update sensitivities using chain rule
        sens_s_next = F_sens_x + F_sens_u
        sens_s_next = jnp.where(jnp.isnan(sens_s_next), 1e3, sens_s_next)
        
        norm_next = jnp.linalg.matrix_norm(sens_s_next, ord="fro")
        
        sens_s_next = sens_s_next / (norm_next + 1e-6)
        # Pack updated sensitivitie
        return SensitivityOff(
            log_norm=sens_state.sensitivity.log_norm
            + jnp.log(norm_next + 1e-6),
            log_norm_old=sens_state.sensitivity.log_norm,
            state=sens_s_next,
            F_sens_x=F_sens_x,
            df_du=df_du,
            dobs_dx=dobs_dx @ sens_s,
        )

    def update_sensitivity(
        self,
        key: chex.PRNGKey,
        network_params,
        df_du,
        dobs_dx,
        obs,
        sens_log_norm,
        F_sens_x,
    ):

        dmu_dobs = self.action_gains(key, obs, network_params)  # dmu/dobs
        dmu_dobs = jax.lax.stop_gradient(dmu_dobs)
        sens_a = df_du @ dmu_dobs @ dobs_dx  # du/dp
        sens_a = jnp.where(jnp.isnan(sens_a), 1e3, sens_a)

        sens_s_next = F_sens_x + sens_a
        sens_s_next = jnp.where(jnp.isnan(sens_s_next), 1e3, sens_s_next)

        norm_next = jnp.linalg.matrix_norm(sens_s_next, ord="fro")
        
        sens_s_next = sens_s_next / (norm_next + 1e-6)
        
        # Return the incremental log norm change
        return sens_s_next, sens_log_norm + jnp.log(norm_next + 1e-6)

    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ):
        obs, env_state = self._env.reset(key, params)
        sensitivity = default_sensitivity_off(self._env)
        state = SensitivityOffEnvState(
            env_state=env_state, sensitivity=sensitivity
        )
        return obs, state

    def step(
        self,
        key: chex.PRNGKey,
        sens_state: SensitivityOffEnvState,
        action: Union[int, float, jax.Array],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:

        obs, env_state, reward, done, info = self._env.step(
            key, sens_state.env_state, action, params
        )

        next_sensitivity = self.step_sensitivity(
            key, sens_state, action, params, obs, done
        )

        info["trw"] = reward
        info["sensitivity_state"] = next_sensitivity.log_norm
        info["sensitivity_state_rw"] = -(
            next_sensitivity.log_norm 
        )

        sens_state = SensitivityOffEnvState(
            env_state=env_state, sensitivity=next_sensitivity
        )

        return obs, sens_state, reward, done, info

    # We need to wrap these functions to convert from flat state/param to pytree and differentiate through env.step
    def _step_to_state_and_obs_wrapped(
        self, key, env_state, state_flat, action, params_flat
    ):
        state = self.state_unravel(state_flat)
        params = self.param_unravel(params_flat)
        env_state = env_state.replace(state=state)

        # Base environment version (differentiates through env.step)
        obs, env_state, reward, done, info = self._env.step(
            key, env_state, action, params
        )
        next_state_flat = jax.flatten_util.ravel_pytree(env_state.state)[0]
        obs_flat = jax.flatten_util.ravel_pytree(obs)[0]
        return next_state_flat, obs_flat


class MujocoSensitivityOffWrapper(SensitivityOffWrapper):
    def __init__(
        self, env, regularization_weight=0.0, add_sensitivity_to_info=False
    ):
        super().__init__(env, regularization_weight, add_sensitivity_to_info)

    # We need to wrap these functions to convert from flat state/param to pytree and differentiate through env.step
    def _step_to_state_and_obs_wrapped(
        self, key, env_state, state_flat, action, params_flat
    ):
        state = self.state_unravel(state_flat)
        params = self.param_unravel(params_flat)

        # Call dynamics and get_obs
        next_state = self._env.step_dynamics(key, state, action, params)
        dummy_env_state = env_state.replace(state=next_state)
        obs = self._env.get_obs(dummy_env_state)

        # Return flattened results
        next_state_flat = jax.flatten_util.ravel_pytree(next_state)[0]
        obs_flat = jax.flatten_util.ravel_pytree(obs)[0]
        return next_state_flat, obs_flat


class MinMaxObservationWrapper(GymnaxWrapper):
    """Normalize the observations of the environment to [-1, 1]."""

    def __init__(self, env, params: Optional[environment.EnvParams] = None):
        super().__init__(env)
        self._obs_min = jnp.array(self._env.observation_space(params).low)
        self._obs_max = jnp.array(self._env.observation_space(params).high)
        # check for infinities
        assert jnp.isinf(self._obs_max).sum() == 0, "Obs space has infinities"
        assert jnp.isinf(self._obs_min).sum() == 0, "Obs space has infinities"

    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ):
        obs, env_state = self._env.reset(key, params)
        obs = normalize(obs, self._obs_min, self._obs_max)
        return obs, env_state

    def step(
        self,
        key: chex.PRNGKey,
        state,
        action: Union[int, float, jax.Array],
        params: Optional[environment.EnvParams] = None,
    ):
        obs, env_state, reward, done, info = self._env.step(
            key, state, action, params
        )
        obs = normalize(obs, self._obs_min, self._obs_max)
        return obs, env_state, reward, done, info


class ResetEnvWrapper(GymnaxWrapper):
    """Call reset when done is True."""

    def __init__(self, env, params: Optional[environment.EnvParams] = None):
        super().__init__(env)

    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ):
        obs, env_state = self._env.reset(key, params)
        return obs, env_state

    def step(
        self,
        key: chex.PRNGKey,
        state,
        action: Union[int, float, jax.Array],
        params: Optional[environment.EnvParams] = None,
    ):
        obs_st, state_st, reward, done, info = self._env.step(
            key, state, action, params
        )
        obs_re, state_re = self.reset(key, params)

        # Auto-reset environment based on termination
        state = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        obs = jax.lax.select(done, obs_re, obs_st)

        return obs, state, reward, done, info
