"""Gymnax wrappers: action clipping, vectorisation, reward normalisation, sensitivity propagation, domain randomisation, constraints and logging."""

import jax
import jax.numpy as jnp
import jax.flatten_util
import chex
import numpy as np
from flax import nnx
from typing import NamedTuple, Optional, Tuple, Union
from gymnax.environments import environment, spaces
from envs.base import Sensitivity, SensitivityOff
from utils import angle_normalize, normalize


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        """Wrap an environment.

        Args:
            env (GymnaxWrapper | environment.Environment): Environment to wrap.
        """
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        """Forward attribute access to the wrapped environment.

        Args:
            name (str): Attribute name.

        Returns:
            Any: The attribute of the wrapped environment.
        """
        return getattr(self._env, name)


class FlattenObservationWrapper(GymnaxWrapper):
    """Flatten the observations of the environment."""

    def __init__(self, env: environment.Environment):
        """Wrap an environment with Box observations.

        Args:
            env (GymnaxWrapper | environment.Environment): Environment to wrap.
        """
        super().__init__(env)

    def observation_space(self, params) -> spaces.Box:
        """Flattened observation space.

        Args:
            params (EnvParams): Environment parameters.

        Returns:
            spaces.Box: Box with the observation flattened to one dimension.
        """
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
        """Reset and flatten the first observation.

        Args:
            key (jax.Array): PRNG key.
            params (EnvParams, optional): Environment parameters.

        Returns:
            tuple[jax.Array, EnvState]: First observation and initial state.
        """
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
        """Step and flatten the observation.

        Args:
            key (jax.Array): PRNG key.
            state (EnvState): Current state.
            action (jax.Array): Action of the policy.
            params (EnvParams, optional): Environment parameters.

        Returns:
            tuple[jax.Array, EnvState, jax.Array, jax.Array, dict]: Observation, next state, reward, done flag and info.
        """
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state, reward, done, info


class LogEnvState(NamedTuple):
    """State of ``LogWrapper``.

    Attributes:
        env_state (EnvState): State of the wrapped environment.
        episode_returns_true (float): Running return of the task reward alone.
        episode_returns (float): Running return of the reward seen by the agent.
        episode_lengths (int): Running episode length.
        returned_episode_returns (float): Return of the last finished episode.
        returned_episode_returns_true (float): Task return of the last finished episode.
        returned_episode_lengths (int): Length of the last finished episode.
        timestep (int): Total number of steps.
    """
    env_state: environment.EnvState
    episode_returns_true: float
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_returns_true: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment):
        """Wrap an environment to log episode returns and lengths.

        Args:
            env (GymnaxWrapper | environment.Environment): Environment to wrap.
        """
        super().__init__(env)

    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        """Reset the environment and the episode counters.

        Args:
            key (jax.Array): PRNG key.
            params (EnvParams, optional): Environment parameters.

        Returns:
            tuple[jax.Array, LogEnvState]: First observation and initial state.
        """
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
        """Step and update the episode return and length.

        At the end of an episode the info dict carries ``returned_episode_returns``,
        ``returned_episode_lengths`` and ``returned_episode``. The true return uses
        ``info['trw']`` when a sensitivity wrapper adds it.

        Args:
            key (jax.Array): PRNG key.
            state (LogEnvState): Current state.
            action (jax.Array): Action of the policy.
            params (EnvParams, optional): Environment parameters.

        Returns:
            tuple[jax.Array, LogEnvState, jax.Array, jax.Array, dict]: Observation, next state, reward, done flag and info.
        """
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return_true = state.episode_returns_true + info.get("trw", reward)
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
        info["returned_episode_returns_true"] = state.returned_episode_returns_true
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info


class ClipAction(GymnaxWrapper):
    """Clip the actions to the action space of the environment."""
    def __init__(self, env, low=-1.0, high=1.0):
        """Wrap an environment to clip the actions to its action space.

        Args:
            env (GymnaxWrapper | environment.Environment): Environment to wrap.
            low (float): Unused lower bound, kept for compatibility.
            high (float): Unused upper bound, kept for compatibility.
        """
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, key, state, action, params=None):
        """Clip the action to the action space of the environment and step.

        Args:
            key (jax.Array): PRNG key.
            state (EnvState): Current state.
            action (jax.Array): Action of the policy.
            params (EnvParams, optional): Environment parameters.

        Returns:
            tuple[jax.Array, EnvState, jax.Array, jax.Array, dict]: Observation, next state, reward, done flag and info.
        """
        action = jnp.clip(
            action,
            self._env.action_space(params).low,
            self._env.action_space(params).high,
        )
        return self._env.step(key, state, action, params)


class VecEnv(GymnaxWrapper):
    """Run a batch of environments in parallel."""
    def __init__(self, env):
        """Vectorise ``reset`` and ``step`` over a batch of environments with ``jax.vmap``.

        Args:
            env (GymnaxWrapper | environment.Environment): Environment to wrap.
        """
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))


class NormalizeVecRewSensEnvState(NamedTuple):
    """State of ``NormalizeVecRewardSensitivity``.

    Attributes:
        mean (jax.Array): Running mean of the task return.
        var (jax.Array): Running variance of the task return.
        mean_sens (jax.Array): Running mean of the sensitivity return.
        var_sens (jax.Array): Running variance of the sensitivity return.
        count (float): Number of samples of the running statistics.
        return_val (float): Discounted task return of each environment.
        return_val_sens (jax.Array): Discounted sensitivity return of each environment.
        env_state (EnvState): State of the wrapped environment.
    """
    mean: jnp.ndarray
    var: jnp.ndarray
    mean_sens: jnp.ndarray
    var_sens: jnp.ndarray
    count: float
    return_val: float
    return_val_sens: jnp.ndarray
    env_state: environment.EnvState


class NormalizeVecRewardSensitivity(GymnaxWrapper):
    """Normalise the task and sensitivity rewards separately."""
    def __init__(self, env, gamma):
        """Normalise the task and sensitivity rewards by running estimates of the return variance.

        Args:
            env (VecEnv): Vectorised environment.
            gamma (float): Discount factor of the returns.
        """
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key, params=None):
        """Reset the environments and the running statistics.

        Args:
            key (jax.Array): PRNG key.
            params (EnvParams, optional): Environment parameters.

        Returns:
            tuple[jax.Array, NormalizeVecRewSensEnvState]: First observation and initial state.
        """
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
        """Step and divide the rewards by the running return standard deviation.

        The sensitivity reward ``info['sensitivity_state_rw']`` is normalised with its own statistics.

        Args:
            key (jax.Array): PRNG key.
            state (NormalizeVecRewSensEnvState): Current state.
            action (jax.Array): Action of the policy.
            params (EnvParams, optional): Environment parameters.

        Returns:
            tuple[jax.Array, NormalizeVecRewSensEnvState, jax.Array, jax.Array, dict]: Observation, next state, reward, done flag and info.
        """
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        return_val = state.return_val * self.gamma * (1 - done) + reward
        return_val_sens = state.return_val_sens * self.gamma * (1 - done) + info.get(
            "sensitivity_state_rw", 0.0
        )

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
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
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

        norm_factor = jnp.sqrt(state.var_sens + 1e-8)
        info["sensitivity_state_rw"] = (
            info.get("sensitivity_state_rw", 0.0) / norm_factor
        )
        info["sens_norm_factor"] = jnp.full_like(reward, norm_factor)

        return obs, state, reward / jnp.sqrt(state.var + 1e-8), done, info


class NormalizeVecRewEnvState(NamedTuple):
    """State of ``NormalizeVecReward``.

    Attributes:
        mean (jax.Array): Running mean of the return.
        var (jax.Array): Running variance of the return.
        count (float): Number of samples of the running statistics.
        return_val (float): Discounted return of each environment.
        env_state (EnvState): State of the wrapped environment.
    """
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: float
    env_state: environment.EnvState


class NormalizeVecReward(GymnaxWrapper):
    """Normalise the reward."""
    def __init__(self, env, gamma):
        """Normalise the reward by a running estimate of the return variance.

        Args:
            env (VecEnv): Vectorised environment.
            gamma (float): Discount factor of the returns.
        """
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key, params=None):
        """Reset the environments and the running statistics.

        Args:
            key (jax.Array): PRNG key.
            params (EnvParams, optional): Environment parameters.

        Returns:
            tuple[jax.Array, NormalizeVecRewEnvState]: First observation and initial state.
        """
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
        """Step and divide the reward by the running return standard deviation.

        Args:
            key (jax.Array): PRNG key.
            state (NormalizeVecRewEnvState): Current state.
            action (jax.Array): Action of the policy.
            params (EnvParams, optional): Environment parameters.

        Returns:
            tuple[jax.Array, NormalizeVecRewEnvState, jax.Array, jax.Array, dict]: Observation, next state, reward, done flag and info.
        """
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
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
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


class SensitivityEnvState(NamedTuple):
    """State of ``SensitivityWrapper``.

    Attributes:
        env_state (EnvState): State of the wrapped environment.
        sensitivity (Sensitivity): Sensitivity carried along the trajectory.
        sens_steps (int): Steps since the last reset of the sensitivity.
    """
    env_state: environment.EnvState
    sensitivity: Sensitivity
    sens_steps: int


class SensitivityWrapper(GymnaxWrapper):
    """Propagate the parametric sensitivity of the closed loop."""
    def __init__(self, env, config=None, expose_jacobians: bool = False):
        """Propagate the parametric sensitivity of the closed loop along the trajectory.

        The Jacobians of the dynamics and of the observation are computed with
        ``jax.jacfwd``, through ``step_dynamics`` when the environment provides it.

        Args:
            env (GymnaxWrapper | environment.Environment): Environment to wrap.
            config (dict, optional): Training configuration (``SENSITIVITY_NUM_STEPS``).
            expose_jacobians (bool): Whether to add the Jacobians needed by the gradient-based losses to the info dict.
        """
        super().__init__(env)
        self.config = config or {}
        self._expose_jacobians = expose_jacobians

        state_flat, self.state_unravel = jax.flatten_util.ravel_pytree(
            self._env.reset(jax.random.PRNGKey(0))[1].state
        )
        # Default to identity if the environment doesn't specify weights
        self.W = getattr(self._env, "sens_weights", jnp.eye(state_flat.shape[0]))

        nominal_flat, self.param_unravel = jax.flatten_util.ravel_pytree(
            self._env.default_params
        )
        self._nominal_params_flat = nominal_flat

        # Use step_dynamics for Jacobians when available.  This avoids tracing
        # jacfwd through obs/reward/done/reset, which blows up the XLA graph
        # when domain randomisation is active.
        self._use_step_dynamics = hasattr(self._env, "step_dynamics")
        self._has_rescale = hasattr(self._env, "rescale_action")
        self._has_get_obs = hasattr(self._env, "get_obs")

        self.action_gains = jax.jacfwd(self.compute_action, argnums=1)
        self.action_gains_mean = jax.jacfwd(self.compute_action_mean, argnums=1)
        self.df_dx = jax.jacfwd(self._step_to_state_wrapped, argnums=2)
        self.df_du = jax.jacfwd(self._step_to_state_wrapped, argnums=3)
        self.df_dp = jax.jacfwd(self._step_to_state_wrapped, argnums=4)
        self.dobs_dx = jax.jacfwd(self._step_to_obs_wrapped, argnums=2)

    def set_actor(self, graphdef, network_params):
        # Should add a check for dimension compatibility
        """Set the policy network used to compute the action gains.

        Args:
            graphdef (nnx.GraphDef): Graph definition of the network.
            network_params (nnx.State): Network parameters.
        """
        self.graphdef = graphdef
        self.network_params = network_params

    def update_network_params(self, network_params):
        """Replace the network parameters used for the action gains.

        Args:
            network_params (nnx.State): New network parameters.
        """
        self.network_params = network_params

    def compute_action(self, rng, obs, network_params):
        """Sample an action of the policy.

        Args:
            rng (jax.Array): PRNG key.
            obs (jax.Array): Observation.
            network_params (nnx.State): Network parameters.

        Returns:
            jax.Array: Sampled action.
        """
        model = nnx.merge(self.graphdef, network_params)
        pi = model.actor(obs)
        action = pi.sample(seed=rng)
        return action

    def compute_action_mean(self, rng, obs, network_params):
        """Deterministic action of the policy (mode of the distribution).

        Args:
            rng (jax.Array): PRNG key, unused.
            obs (jax.Array): Observation.
            network_params (nnx.State): Network parameters.

        Returns:
            jax.Array: Mean action.
        """
        model = nnx.merge(self.graphdef, network_params)
        pi = model.actor(obs)
        action = pi.mode()
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
        """Propagate the state sensitivity by one step with the chain rule.

        ``S' = df/dx S + df/du (dmu/dobs dobs/dx S) + df/dp``, normalised by its
        weighted Frobenius norm, whose logarithm is accumulated in ``log_norm``.

        Args:
            key (jax.Array): PRNG key.
            sens_state (SensitivityEnvState): Current state with the sensitivity.
            action (jax.Array): Action applied in this step.
            params (EnvParams): Environment parameters at which the Jacobians are taken.
            obs (jax.Array): Observation after the step.
            done (jax.Array): Done flag of the step.

        Returns:
            tuple[Sensitivity, dict]: Updated sensitivity, and the Jacobian terms
            ``sens_bias``, ``df_du`` and ``dobs_dx_sens`` used by the gradient-based losses.
        """
        state = jax.flatten_util.ravel_pytree(sens_state.env_state.state)[0]
        params_flat = jax.flatten_util.ravel_pytree(params)[0]
        # params_flat = self._nominal_params_flat
        # Compute Jacobians (at nominal params)
        df_dx = self.df_dx(key, sens_state.env_state, state, action, params_flat)
        df_du = self.df_du(key, sens_state.env_state, state, action, params_flat)
        df_dp = self.df_dp(key, sens_state.env_state, state, action, params_flat)
        dobs_dx = self.dobs_dx(key, sens_state.env_state, state, action, params_flat)

        df_dx = jnp.clip(df_dx, -1e3, 1e3)
        df_du = jnp.clip(df_du, -1e3, 1e3)
        df_dp = jnp.clip(df_dp, -1e3, 1e3)
        dobs_dx = jnp.clip(dobs_dx, -1e3, 1e3)

        # # Replace NaN with 1e3
        df_dx = jnp.where(jnp.isnan(df_dx), 1e3, df_dx)
        df_du = jnp.where(jnp.isnan(df_du), 1e3, df_du)
        df_dp = jnp.where(jnp.isnan(df_dp), 1e3, df_dp)
        dobs_dx = jnp.where(jnp.isnan(dobs_dx), 1e3, dobs_dx)

        df_dp_scaled = df_dp / jnp.clip(
            jnp.exp(sens_state.sensitivity.log_norm), -1e6, 1e3
        )

        # Unpack sensitivities
        sens_s = sens_state.sensitivity.state  # ds/dp
        dmu_dobs = self.action_gains(key, obs, self.network_params)  # dmu/dobs

        dmu_dobs = jnp.clip(dmu_dobs, -1e3, 1e3)
        dmu_dobs = jnp.where(jnp.isnan(dmu_dobs), 1e3, dmu_dobs)

        # Update action sensitivity
        sens_a = dmu_dobs @ dobs_dx @ sens_s  # du/dp
        sens_a = jnp.where(jnp.isnan(sens_a), 1e3, sens_a)

        # Update sensitivities using chain rule
        sens_s_next = df_dx @ sens_s + df_du @ sens_a + df_dp_scaled
        sens_s_next = jnp.where(jnp.isnan(sens_s_next), 1e3, sens_s_next)

        # not normalized
        norm_next = jnp.linalg.matrix_norm(self.W @ sens_s_next, ord="fro")

        # normalize
        sens_s_next = sens_s_next / (norm_next + 1e-6)
        # Pack updated sensitivities
        sensitivity = Sensitivity(
            log_norm=sens_state.sensitivity.log_norm + jnp.log(norm_next + 1e-7),
            state=sens_s_next,
            action=sens_a,
            policy_gains=dmu_dobs,
        )
        # Store Jacobians for gradient-based sensitivity loss in loss_fn
        jac_info = {
            "sens_bias": df_dx @ sens_s + df_dp_scaled,  # policy-independent part of M
            "df_du": df_du,
            "dobs_dx_sens": dobs_dx @ sens_s,
        }
        return sensitivity, jac_info

    def reset(self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None):
        """Reset the environment and set the sensitivity to zero.

        Args:
            key (jax.Array): PRNG key.
            params (EnvParams, optional): Environment parameters.

        Returns:
            tuple[jax.Array, SensitivityEnvState]: First observation and initial state.
        """
        obs, env_state = self._env.reset(key, params)
        sensitivity = Sensitivity.default(self._env)
        state = SensitivityEnvState(env_state=env_state, sensitivity=sensitivity, sens_steps=0)
        return obs, state

    def step(
        self,
        key: chex.PRNGKey,
        sens_state: SensitivityEnvState,
        action: Union[int, float, jax.Array],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:

        """Step the environment and propagate the sensitivity.

        The sensitivity is reset at the end of an episode and every
        ``SENSITIVITY_NUM_STEPS`` steps (``info['done_sens']``). The info dict also
        receives ``trw`` (task reward) and ``sensitivity_state`` (weighted log-norm).

        Args:
            key (jax.Array): PRNG key.
            sens_state (SensitivityEnvState): Current state.
            action (jax.Array): Action of the policy.
            params (EnvParams, optional): Environment parameters.

        Returns:
            tuple[jax.Array, SensitivityEnvState, jax.Array, jax.Array, dict]: Observation, next state, reward, done flag and info.
        """
        obs, env_state, reward, done, info = self._env.step(
            key, sens_state.env_state, action, params
        )

        next_sensitivity, jac_info = self.step_sensitivity(
            key, sens_state, action, params, obs, done
        )

        next_sens_steps = sens_state.sens_steps + 1
        sens_num_steps = self.config.get("SENSITIVITY_NUM_STEPS", 50) if hasattr(self, "config") else 50
        done_sens = jnp.logical_or(done, next_sens_steps >= sens_num_steps)
        info["done_sens"] = done_sens

        def _reset_sens(sens):
            return Sensitivity.default(self._env)
            
        def _no_reset(sens):
            return sens
            
        next_sensitivity = jax.lax.cond(done_sens, _reset_sens, _no_reset, next_sensitivity)
        next_sens_steps = jnp.where(done_sens, 0, next_sens_steps)

        # Use default weight matrix if it's not set
        sens_state_weighted = self.W @ next_sensitivity.state
        weighted_norm = jnp.linalg.matrix_norm(sens_state_weighted, ord="fro")
        weighted_log_norm = next_sensitivity.log_norm + jnp.log(weighted_norm + 1e-6)

        info["trw"] = reward
        info["sensitivity_state"] = weighted_log_norm
        if self._expose_jacobians:
            info["sens_bias"] = jac_info["sens_bias"]
            info["df_du"] = jac_info["df_du"]
            info["dobs_dx_sens"] = jac_info["dobs_dx_sens"]

        sens_state = SensitivityEnvState(
            env_state=env_state, sensitivity=next_sensitivity, sens_steps=next_sens_steps
        )

        return obs, sens_state, reward, done, info

    # We need to wrap these functions to convert from flat state/param to pytree and differentiate through env.step
    def _step_to_state_wrapped(self, key, env_state, state_flat, action, params_flat):
        """Next flattened state as a function of flattened state and parameters, for ``jax.jacfwd``.

        Args:
            key (jax.Array): PRNG key.
            env_state (EnvState): Environment state that provides the non-differentiated fields.
            state_flat (jax.Array): Flattened model state.
            action (jax.Array): Action.
            params_flat (jax.Array): Flattened environment parameters.

        Returns:
            jax.Array: Flattened next model state.
        """
        state = self.state_unravel(state_flat)
        params = self.param_unravel(params_flat)
        env_state = env_state.replace(state=state)
        if self._use_step_dynamics:
            # Optimised: trace only through pure dynamics (skip obs/reward/done)
            if self._has_rescale:
                action = jnp.clip(action, params.min_action, params.max_action)
                action = self._env.rescale_action(action, params)
            next_state = self._env.step_dynamics(key, env_state, action, params)
        else:
            env_state = self._env.step(key, env_state, action, params)[
                1
            ]  # take next_env_state from step
            next_state = env_state.state
        return jax.flatten_util.ravel_pytree(next_state)[0]

    def _step_to_obs_wrapped(self, key, env_state, state_flat, action, params_flat):
        """Flattened observation as a function of flattened state and parameters, for ``jax.jacfwd``.

        Args:
            key (jax.Array): PRNG key.
            env_state (EnvState): Environment state that provides the non-differentiated fields.
            state_flat (jax.Array): Flattened model state.
            action (jax.Array): Action.
            params_flat (jax.Array): Flattened environment parameters.

        Returns:
            jax.Array: Flattened observation.
        """
        state = self.state_unravel(state_flat)
        params = self.param_unravel(params_flat)

        if self._has_get_obs:
            obs = self._env.get_obs(env_state.replace(state=state), key, params)
        else:
            env_state = env_state.replace(state=state)
            obs, env_state, _, _, _ = self._env.step(key, env_state, action, params)
        return jax.flatten_util.ravel_pytree(obs)[0]


class ComputeSensitivityReward(GymnaxWrapper):
    """Compute the sensitivity reward and store it in the info dict."""
    def __init__(self, env):
        """Compute the sensitivity reward without adding it to the task reward.

        Args:
            env (SensitivityWrapper): Environment wrapped with the sensitivity.
        """
        super().__init__(env)

    def step(
        self,
        key: chex.PRNGKey,
        current_state: SensitivityEnvState,
        action: Union[int, float, jax.Array],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:

        """Step and store ``-log ||W S||_F`` in ``info['sensitivity_state_rw']``.

        Args:
            key (jax.Array): PRNG key.
            current_state (SensitivityEnvState): Current state.
            action (jax.Array): Action of the policy.
            params (EnvParams, optional): Environment parameters.

        Returns:
            tuple[jax.Array, SensitivityEnvState, jax.Array, jax.Array, dict]: Observation, next state, reward, done flag and info.
        """
        obs, next_state, reward, done, info = self._env.step(
            key, current_state, action, params
        )

        norm_diff = jnp.exp(
            next_state.sensitivity.log_norm - current_state.sensitivity.log_norm
        )
        sens_state_weighted = self.W @ (norm_diff * next_state.sensitivity.state)
        weighted_norm = jnp.linalg.matrix_norm(sens_state_weighted, ord="fro")

        info["sensitivity_state_rw"] = -jnp.log(weighted_norm + 1e-6)

        return obs, next_state, reward, done, info


class AddSensitivityToReward(GymnaxWrapper):
    """Add the weighted sensitivity reward to the task reward."""
    def __init__(self, env, regularization_weight=0.0):
        """Add a weighted sensitivity penalty to the task reward.

        Args:
            env (SensitivityWrapper): Environment wrapped with the sensitivity.
            regularization_weight (float): Weight of the sensitivity term.
        """
        super().__init__(env)
        self.regularization_weight = regularization_weight

    def step(
        self,
        key: chex.PRNGKey,
        current_state: SensitivityEnvState,
        action: Union[int, float, jax.Array],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:

        """Step and add the weighted sensitivity reward to the task reward.

        Args:
            key (jax.Array): PRNG key.
            current_state (SensitivityEnvState): Current state.
            action (jax.Array): Action of the policy.
            params (EnvParams, optional): Environment parameters.

        Returns:
            tuple[jax.Array, SensitivityEnvState, jax.Array, jax.Array, dict]: Observation, next state, reward, done flag and info.
        """
        obs, next_state, reward, done, info = self._env.step(
            key, current_state, action, params
        )

        norm_diff = jnp.exp(
            next_state.sensitivity.log_norm - current_state.sensitivity.log_norm
        )
        sens_state_weighted = self.W @ (norm_diff * next_state.sensitivity.state)
        weighted_norm = jnp.linalg.matrix_norm(sens_state_weighted, ord="fro")

        reward += -self.regularization_weight * jnp.log(weighted_norm + 1e-6)

        return obs, next_state, reward, done, info


class MinMaxObservationWrapper(GymnaxWrapper):
    """Normalize the observations of the environment to [-1, 1]."""

    def __init__(self, env, params: Optional[environment.EnvParams] = None):
        """Rescale the observations to ``[-1, 1]`` with the bounds of the observation space.

        Args:
            env (GymnaxWrapper | environment.Environment): Environment to wrap.
            params (EnvParams, optional): Parameters used to read the observation space.
        """
        super().__init__(env)
        self._obs_min = jnp.array(self._env.observation_space(params).low)
        self._obs_max = jnp.array(self._env.observation_space(params).high)
        # check for infinities
        assert jnp.isinf(self._obs_max).sum() == 0, "Obs space has infinities"
        assert jnp.isinf(self._obs_min).sum() == 0, "Obs space has infinities"

    def reset(self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None):
        """Reset and rescale the first observation.

        Args:
            key (jax.Array): PRNG key.
            params (EnvParams, optional): Environment parameters.

        Returns:
            tuple[jax.Array, EnvState]: First observation and initial state.
        """
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
        """Step and rescale the observation.

        Args:
            key (jax.Array): PRNG key.
            state (EnvState): Current state.
            action (jax.Array): Action of the policy.
            params (EnvParams, optional): Environment parameters.

        Returns:
            tuple[jax.Array, EnvState, jax.Array, jax.Array, dict]: Observation, next state, reward, done flag and info.
        """
        obs, env_state, reward, done, info = self._env.step(key, state, action, params)
        obs = normalize(obs, self._obs_min, self._obs_max)
        return obs, env_state, reward, done, info


class ResetEnvWrapper(GymnaxWrapper):
    """Call reset when done is True."""

    def __init__(self, env, params: Optional[environment.EnvParams] = None):
        """Reset the environment automatically at the end of an episode.

        Args:
            env (GymnaxWrapper | environment.Environment): Environment to wrap.
            params (EnvParams, optional): Unused.
        """
        super().__init__(env)

    def reset(self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None):
        """Reset the wrapped environment.

        Args:
            key (jax.Array): PRNG key.
            params (EnvParams, optional): Environment parameters.

        Returns:
            tuple[jax.Array, EnvState]: First observation and initial state.
        """
        obs, env_state = self._env.reset(key, params)
        return obs, env_state

    def step(
        self,
        key: chex.PRNGKey,
        state,
        action: Union[int, float, jax.Array],
        params: Optional[environment.EnvParams] = None,
    ):
        """Step, and return the reset state and observation when the episode ends.

        Args:
            key (jax.Array): PRNG key.
            state (EnvState): Current state.
            action (jax.Array): Action of the policy.
            params (EnvParams, optional): Environment parameters.

        Returns:
            tuple[jax.Array, EnvState, jax.Array, jax.Array, dict]: Observation, next state, reward, done flag and info.
        """
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


# ---------------------------------------------------------------------------
# Constraint evaluation wrapper (for CaT batch processing)
# ---------------------------------------------------------------------------

class ConstraintEvalWrapper(GymnaxWrapper):
    """Thin wrapper that evaluates hard/soft constraints and stores raw values in info.

    The environment must implement ``hard_constraints(state, params)`` and
    ``soft_constraints(state, params)`` methods.  Each returns a 1-D array of
    constraint values (positive = violated, zero or negative = satisfied).

    Must be placed **below** ``ResetEnvWrapper`` and **below**
    ``SensitivityWrapper`` in the wrapper stack so that it operates on the
    base ``EnvState``.
    """

    def step(
        self,
        key: chex.PRNGKey,
        state,
        action: Union[int, float, jax.Array],
        params: Optional[environment.EnvParams] = None,
    ):
        """Step and store the hard and soft constraint values in the info dict.

        Args:
            key (jax.Array): PRNG key.
            state (EnvState): Current state.
            action (jax.Array): Action of the policy.
            params (EnvParams, optional): Environment parameters.

        Returns:
            tuple[jax.Array, EnvState, jax.Array, jax.Array, dict]: Observation, next state, reward, done flag and info.
        """
        obs, state_st, reward, done, info = self._env.step(
            key, state, action, params
        )
        # Evaluate constraints on the post-step (pre-reset) state.
        hard_c = self._env.hard_constraints(state_st, params)
        soft_c = self._env.soft_constraints(state_st, params)
        # Only store non-empty constraint arrays; zero-length arrays cause
        # division-by-zero in minibatch reshaping downstream.
        if hard_c.shape[0] > 0:
            info["hard_constraints"] = hard_c
        if soft_c.shape[0] > 0:
            info["soft_constraints"] = soft_c
        return obs, state_st, reward, done, info


class ConstraintPenaltyWrapper(GymnaxWrapper):
    """Subtract a fixed penalty from the reward when any constraint is violated.

    Must be placed **after** ``ConstraintEvalWrapper`` in the wrapper stack so
    that ``info`` already contains ``hard_constraints`` / ``soft_constraints``.
    """

    def __init__(self, env, penalty: float = 1.0):
        """Set the wrapped environment and the penalty.

        Args:
            env (ConstraintEvalWrapper): Environment with the constraint values in the info dict.
            penalty (float): Penalty per violating step.
        """
        super().__init__(env)
        self._penalty = penalty

    def step(
        self,
        key: chex.PRNGKey,
        state,
        action: Union[int, float, jax.Array],
        params: Optional[environment.EnvParams] = None,
    ):
        """Step and subtract the penalty when any constraint is positive.

        The info dict receives ``constraint_violated`` and ``constraint_violation`` (largest violation).

        Args:
            key (jax.Array): PRNG key.
            state (EnvState): Current state.
            action (jax.Array): Action of the policy.
            params (EnvParams, optional): Environment parameters.

        Returns:
            tuple[jax.Array, EnvState, jax.Array, jax.Array, dict]: Observation, next state, reward, done flag and info.
        """
        obs, state_st, reward, done, info = self._env.step(
            key, state, action, params
        )
        # Check if any constraint dimension is violated (> 0)
        hard_c = info.get("hard_constraints", jnp.zeros((0,)))
        soft_c = info.get("soft_constraints", jnp.zeros((0,)))
        hard_violated = hard_c.max(initial=0.0) > 0.0
        soft_violated = soft_c.max(initial=0.0) > 0.0
        violated = hard_violated | soft_violated
        reward = reward - self._penalty * violated.astype(jnp.float32)

        # Constraint violation metrics
        all_c = jnp.concatenate([hard_c, soft_c])
        max_violation = jnp.where(all_c.shape[0] > 0, all_c.max(initial=0.0), 0.0)
        info["constraint_violated"] = violated.astype(jnp.float32)
        info["constraint_violation"] = jnp.maximum(max_violation, 0.0)
        return obs, state_st, reward, done, info


class DomainRandomizationEnvState(NamedTuple):
    """State of ``DomainRandomizationWrapper``.

    Attributes:
        env_state (EnvState): State of the wrapped environment.
        params (EnvParams): Parameters sampled at the last reset.
    """
    env_state: environment.EnvState
    params: environment.EnvParams


class DomainRandomizationWrapper(GymnaxWrapper):
    """Resample the uncertain parameters at every reset."""

    def __init__(self, env, params: Optional[environment.EnvParams] = None):
        """Resample the uncertain parameters at every reset.

        Args:
            env (GymnaxWrapper | environment.Environment): Environment to wrap. It must define ``params_min`` and ``params_max``.
            params (EnvParams, optional): Unused.
        """
        super().__init__(env)

    def reset(self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None):
        """Sample new parameters and reset the environment with them.

        Args:
            key (jax.Array): PRNG key.
            params (EnvParams, optional): Environment parameters.

        Returns:
            tuple[jax.Array, DomainRandomizationEnvState]: First observation and initial state.
        """
        min_params = self._env.params_min
        max_params = self._env.params_max

        key, sample_key = jax.random.split(key)

        # Sample the randomised parameters
        randomized_params = self.sample_params(
            sample_key, params, min_params, max_params
        )

        obs, env_state = self._env.reset(key, randomized_params)
        state = DomainRandomizationEnvState(
            env_state=env_state, params=randomized_params
        )
        return obs, state

    def sample_params(self, key, params, low_list, high_list):
        """Sample each parameter leaf uniformly between its bounds.

        Args:
            key (jax.Array): PRNG key.
            params (EnvParams): Parameters that give the tree structure and the leaf shapes.
            low_list (list): Lower bounds, in the leaf order of ``params``.
            high_list (list): Upper bounds, in the same order.

        Returns:
            EnvParams: Sampled parameters.
        """
        # Flatten the params to get the tree structure and the leaf shapes
        leaves, treedef = jax.tree_util.tree_flatten(params)

        # The bounds must match the number of uncertain parameters
        assert (
            len(leaves) == len(low_list) == len(high_list)
        ), f"Mismatch: Params has {len(leaves)} nodes, but low/high lists have {len(low_list)}"

        # One key per parameter array
        keys = jax.random.split(key, len(leaves))

        new_leaves = []
        for rng, leaf, l, h in zip(keys, leaves, low_list, high_list):
            sampled_val = jax.random.uniform(
                rng, shape=jnp.shape(leaf), minval=l, maxval=h
            )
            new_leaves.append(sampled_val)

        # Rebuild the params dataclass from the sampled leaves
        new_params = jax.tree_util.tree_unflatten(treedef, new_leaves)

        return new_params

    def step(
        self,
        key: chex.PRNGKey,
        state: DomainRandomizationEnvState,
        action: Union[int, float, jax.Array],
        params: Optional[environment.EnvParams] = None,
    ):
        """Step with the parameters of the episode, and resample them at the end of the episode.

        Args:
            key (jax.Array): PRNG key.
            state (DomainRandomizationEnvState): Current state.
            action (jax.Array): Action of the policy.
            params (EnvParams, optional): Environment parameters.

        Returns:
            tuple[jax.Array, DomainRandomizationEnvState, jax.Array, jax.Array, dict]: Observation, next state, reward, done flag and info.
        """
        obs_st, env_state_st, reward, done, info = self._env.step(
            key, state.env_state, action, state.params
        )

        state_st = DomainRandomizationEnvState(
            env_state=env_state_st, params=state.params
        )

        obs_re, state_re = self.reset(key, params)

        # Auto-reset environment based on termination
        state = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        obs = jax.lax.select(done, obs_re, obs_st)

        return obs, state, reward, done, info


"""
Off-policy sensitivity wrappers
"""


class SensitivityOffEnvState(NamedTuple):
    """State of ``SensitivityOffWrapper``.

    Attributes:
        env_state (EnvState): State of the wrapped environment.
        sensitivity (SensitivityOff): Sensitivity carried along the trajectory.
    """
    env_state: environment.EnvState
    sensitivity: SensitivityOff


class SensitivityOffWrapper(GymnaxWrapper):
    """Propagate the parametric sensitivity for the off-policy algorithms."""
    def __init__(self, env, regularization_weight=0.0, add_sensitivity_to_info=False):
        """Sensitivity propagation for off-policy algorithms with a squashed Gaussian actor.

        Args:
            env (GymnaxWrapper | environment.Environment): Environment to wrap.
            regularization_weight (float): Weight of the sensitivity term.
            add_sensitivity_to_info (bool): Whether to report the sensitivity in the info dict.
        """
        super().__init__(env)

        self.state_unravel = jax.flatten_util.ravel_pytree(
            self._env.reset(jax.random.PRNGKey(0))[1].state
        )[1]

        self.param_unravel = jax.flatten_util.ravel_pytree(self._env.default_params)[1]

        self.regularization_weight = regularization_weight
        self.add_sensitivity_to_info = add_sensitivity_to_info

        self.action_gains = jax.jacfwd(self.compute_action, argnums=1)
        self.action_gains_mean = jax.jacfwd(self.compute_action_mean, argnums=1)
        self._dynamics_jacs_fn = jax.jacfwd(
            self._step_to_state_and_obs_wrapped, argnums=(2, 3, 4)
        )

    def set_actor(
        self, graphdef, network_params, get_action, action_scale, action_bias
    ):
        # Should add a check for dimension compatibility
        """Set the actor network and the function that maps its output to an action.

        Args:
            graphdef (nnx.GraphDef): Graph definition of the network.
            network_params (nnx.State): Network parameters.
            get_action (Callable): Function that returns the sampled action, log-probability and mean.
            action_scale (jax.Array): Scale of the action space.
            action_bias (jax.Array): Offset of the action space.
        """
        self.graphdef = graphdef
        self.network_params = network_params
        self.get_action = get_action
        self.action_scale = action_scale
        self.action_bias = action_bias

    def update_network_params(self, network_params):
        """Replace the network parameters used for the action gains.

        Args:
            network_params (nnx.State): New network parameters.
        """
        self.network_params = network_params

    def compute_action(self, rng, obs, network_params):
        """Sample an action of the actor.

        Args:
            rng (jax.Array): PRNG key.
            obs (jax.Array): Observation.
            network_params (nnx.State): Network parameters.

        Returns:
            jax.Array: Sampled action.
        """
        model = nnx.merge(self.graphdef, network_params)
        pi = model.actor(obs)
        action, _, _, _ = self.get_action(
            pi,
            self.action_scale,
            self.action_bias,
            rng,
            jnp.zeros(self._env.action_space(self._env.default_params).shape),
            False,
        )
        return action

    def compute_action_mean(self, rng, obs, network_params):
        """Mean action of the actor.

        Args:
            rng (jax.Array): PRNG key.
            obs (jax.Array): Observation.
            network_params (nnx.State): Network parameters.

        Returns:
            jax.Array: Mean action.
        """
        model = nnx.merge(self.graphdef, network_params)
        pi = model.actor(obs)
        _, _, mean, _ = self.get_action(
            pi,
            self.action_scale,
            self.action_bias,
            rng,
            jnp.zeros(self._env.action_space(self._env.default_params).shape),
            False,
        )
        return mean

    def step_sensitivity(
        self,
        key: chex.PRNGKey,
        sens_state: SensitivityOffEnvState,
        action: Union[int, float, jax.Array],
        params,
        obs,
        done,
    ):
        """Propagate the state sensitivity by one step.

        Args:
            key (jax.Array): PRNG key.
            sens_state (SensitivityOffEnvState): Current state with the sensitivity.
            action (jax.Array): Action applied in this step.
            params (EnvParams): Environment parameters.
            obs (jax.Array): Observation after the step.
            done (jax.Array): Done flag of the step.

        Returns:
            SensitivityOff: Updated sensitivity.
        """
        state = jax.flatten_util.ravel_pytree(sens_state.env_state.state)[0]
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

        F_sens_x = df_dx @ sens_s + df_dp / (
            jnp.exp(jnp.clip(sens_state.sensitivity.log_norm, -1e4, 1e3) + 1e-4)
        )
        F_sens_x = jnp.where(jnp.isnan(F_sens_x), 1e3, F_sens_x)

        # Update sensitivities using chain rule
        sens_s_next = F_sens_x + F_sens_u
        sens_s_next = jnp.where(jnp.isnan(sens_s_next), 1e3, sens_s_next)

        norm_next = jnp.linalg.matrix_norm(sens_s_next, ord="fro")

        sens_s_next = sens_s_next / (norm_next + 1e-6)
        # Pack updated sensitivities
        return SensitivityOff(
            log_norm=sens_state.sensitivity.log_norm + jnp.log(norm_next + 1e-6),
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

        """Sensitivity update for given network parameters, differentiable with respect to them.

        Args:
            key (jax.Array): PRNG key.
            network_params (nnx.State): Network parameters.
            df_du (jax.Array): Jacobian of the dynamics with respect to the action.
            dobs_dx (jax.Array): Observation Jacobian times the state sensitivity.
            obs (jax.Array): Observation.
            sens_log_norm (jax.Array): Accumulated log-norm of the sensitivity.
            F_sens_x (jax.Array): Policy-independent part of the update.

        Returns:
            tuple[jax.Array, jax.Array]: Next sensitivity and its accumulated log-norm.
        """
        dmu_dobs = self.action_gains_mean(key, obs, network_params)  # dmu/dobs
        dmu_dobs = jax.lax.stop_gradient(dmu_dobs)
        sens_a = df_du @ dmu_dobs @ dobs_dx  # du/dp
        sens_a = jnp.where(jnp.isnan(sens_a), 1e3, sens_a)

        sens_s_next = F_sens_x + sens_a
        sens_s_next = jnp.where(jnp.isnan(sens_s_next), 1e3, sens_s_next)

        norm_next = jnp.linalg.matrix_norm(sens_s_next, ord="fro")

        sens_s_next = sens_s_next / (norm_next + 1e-6)

        # Return the incremental log norm change
        return sens_s_next, sens_log_norm + jnp.log(norm_next + 1e-6)

    def reset(self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None):
        """Reset the environment and set the sensitivity to zero.

        Args:
            key (jax.Array): PRNG key.
            params (EnvParams, optional): Environment parameters.

        Returns:
            tuple[jax.Array, SensitivityOffEnvState]: First observation and initial state.
        """
        obs, env_state = self._env.reset(key, params)
        sensitivity = SensitivityOff.default(self._env)
        state = SensitivityOffEnvState(env_state=env_state, sensitivity=sensitivity)
        return obs, state

    def step(
        self,
        key: chex.PRNGKey,
        sens_state: SensitivityOffEnvState,
        action: Union[int, float, jax.Array],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:

        """Step the environment and propagate the sensitivity.

        Args:
            key (jax.Array): PRNG key.
            sens_state (SensitivityOffEnvState): Current state.
            action (jax.Array): Action of the policy.
            params (EnvParams, optional): Environment parameters.

        Returns:
            tuple[jax.Array, SensitivityOffEnvState, jax.Array, jax.Array, dict]: Observation, next state, reward, done flag and info.
        """
        obs, env_state, reward, done, info = self._env.step(
            key, sens_state.env_state, action, params
        )

        next_sensitivity = self.step_sensitivity(
            key, sens_state, action, params, obs, done
        )

        info["trw"] = reward
        info["sensitivity_state"] = next_sensitivity.log_norm
        info["sensitivity_state_rw"] = -(next_sensitivity.log_norm)

        sens_state = SensitivityOffEnvState(
            env_state=env_state, sensitivity=next_sensitivity
        )

        return obs, sens_state, reward, done, info

    # We need to wrap these functions to convert from flat state/param to pytree and differentiate through env.step
    def _step_to_state_and_obs_wrapped(
        self, key, env_state, state_flat, action, params_flat
    ):
        """Flattened next state and observation as a function of flattened state and parameters.

        Args:
            key (jax.Array): PRNG key.
            env_state (EnvState): Environment state that provides the non-differentiated fields.
            state_flat (jax.Array): Flattened model state.
            action (jax.Array): Action.
            params_flat (jax.Array): Flattened environment parameters.

        Returns:
            tuple[jax.Array, jax.Array]: Flattened next state and observation.
        """
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
