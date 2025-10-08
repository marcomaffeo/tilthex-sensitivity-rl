import jax
import jax.numpy as jnp
import jax.flatten_util
import chex
import numpy as np
from flax import struct
from typing import Optional, Tuple, Union
from gymnax.environments import environment, spaces
from envs.base import Sensitivity
from utils import angle_normalize, normalize, default_sensitivity


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


class SensitivityWrapper(GymnaxWrapper):
    def __init__(self, env, regularization_weight=0.0):
        super().__init__(env)

        self.state_unravel = jax.flatten_util.ravel_pytree(
            self._env.reset(jax.random.PRNGKey(0))[1].state
        )[1]

        self.param_unravel = jax.flatten_util.ravel_pytree(
            self._env.default_params
        )[1]

        self.regularization_weight = regularization_weight

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
        # Unpack sensitivities
        sens_s = sens_state.sensitivity.state  # ds/dp
        dmu_dobs = self.action_gains(key, obs, self.network_params)  # dmu/dobs
        # Update action sensitivity
        sens_a = dmu_dobs @ dobs_dx @ sens_s  # du/dp
        # Update sensitivities using chain rule
        sens_s_next = df_dx @ sens_s + df_du @ sens_a + df_dp
        sens_s_next = jnp.clip(sens_s_next, -10, 10)
        # Pack updated sensitivitie
        return Sensitivity(
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

        info["trw"] = reward
        info["sensitivity_state"] = jnp.linalg.matrix_norm(
            sens_state.sensitivity.state, ord="fro"
        )

        next_sensitivity = self.step_sensitivity(
            key, sens_state, action, params, obs, done
        )

        # Final sensitivity reward regularization
        # reward = jax.lax.select(
        #   done,
        #    reward
        #    - self.regularization_weight
        #    * jnp.linalg.matrix_norm(next_sensitivity.state, ord="fro"),
        #    reward,
        # )

        # Sensitivity difference reward regularization (next sensitivity whould be smaller than previous one)
        reward = reward - self.regularization_weight * (
            jnp.linalg.matrix_norm(next_sensitivity.state, ord="fro")
            - jnp.linalg.matrix_norm(sens_state.sensitivity.state, ord="fro")
        )

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
