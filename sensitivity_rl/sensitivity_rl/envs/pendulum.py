"""Pendulum swing-up in JAX, with the parametric sensitivity used by SPPO.

Adapted from github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
"""

from typing import Any, List
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from gymnax.environments import environment, spaces

from envs.base import Sensitivity

from utils import angle_normalize


@struct.dataclass
class ModelState:
    """State of the pendulum.

    Attributes:
        theta (jax.Array): Angle [rad].
        theta_dot (jax.Array): Angular velocity [rad/s].
    """
    theta: jax.Array
    theta_dot: jax.Array


@struct.dataclass
class EnvState(environment.EnvState):
    """State of the pendulum environment.

    Attributes:
        state (ModelState): Pendulum state.
        last_u (jax.Array): Last torque applied.
        time (int): Step index in the episode.
    """
    state: ModelState
    last_u: jax.Array  # Only needed for rendering
    time: int

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        """Forward attribute access to the wrapped model state.

        Args:
            name (str): Attribute name, e.g. ``pos`` or ``attitude``.

        Returns:
            Any: The attribute of ``self.state``.
        """
        return getattr(self.state, name)


@struct.dataclass
class EnvParams(environment.EnvParams):
    """Parameters of the pendulum environment; mass ``m`` and length ``l`` are the uncertain parameters."""
    max_speed: float = struct.field(pytree_node=False, default=8.0)
    max_torque: float = struct.field(pytree_node=False, default=2.0)
    dt: float = struct.field(pytree_node=False, default=0.05)
    g: float = struct.field(pytree_node=False, default=10.0)
    m: float = 1.0
    l: float = 1.0  # noqa: E741
    max_steps_in_episode: int = struct.field(pytree_node=False, default=200)


class Pendulum(environment.Environment):
    """Pendulum-v0 swing-up task."""

    def __init__(self):
        """Set the dimensions and the ranges of the uncertain parameters (mass and length)."""
        self.type = ""
        self.state_dim = 2
        self.obs_dim = 3
        self.action_dim = self.action_space(self.default_params).shape[0]
        self.param_dim = len(jax.tree.leaves(self.default_params))
        self.params_min = [
            0.1,  # mass
            0.5,  # length
        ]
        self.params_max = [
            10.0,  # mass
            2.0,  # length
        ]
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        """Default parameters of the pendulum environment.

        Returns:
            EnvParams: Nominal parameters.
        """
        return EnvParams()

    def step_env(
        self,
        key: jax.Array,
        env_state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Apply a torque, integrate the pendulum and compute the reward.

        The reward is ``-(theta^2 + 0.1 theta_dot^2 + 0.001 u^2)``, with ``theta``
        wrapped to ``[-pi, pi)``.

        Args:
            key (jax.Array): PRNG key.
            env_state (EnvState): Current environment state.
            action (jax.Array): Torque, shape ``(1,)``.
            params (EnvParams): Environment parameters.

        Returns:
            tuple[jax.Array, EnvState, jax.Array, jax.Array, dict]: Observation, next state, reward, done flag and info.
        """

        u = jnp.clip(action, -params.max_torque, params.max_torque)

        state = env_state.state

        state_next = self.step_dynamics(key, env_state, action, params)

        reward = -(
            angle_normalize(state_next.theta) ** 2
            + 0.1 * state_next.theta_dot**2
            + 0.001 * (u**2)
        )

        reward = reward.squeeze()

        env_state = EnvState(
            state=state_next,
            last_u=u.reshape(),
            time=env_state.time + 1,
        )

        done = self.is_terminal(env_state, params)
        return (
            self._obs_from_model_state(state_next),
            env_state,
            reward,
            done,
            {"discount": self.discount(env_state, params)},
        )

    def step_dynamics(
        self,
        key: jax.Array,
        env_state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> ModelState:
        """Next model state from the environment state and the actuator input.

        Args:
            key (jax.Array): PRNG key.
            env_state (EnvState): Current environment state.
            action (jax.Array): Actuator input.
            params (EnvParams): Environment parameters.

        Returns:
            ModelState: Next model state.
        """
        return self._step_dynamics(key, env_state.state, action, params)

    def _step_dynamics(
        self,
        key: jax.Array,
        state: ModelState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> ModelState:
        """One integration step of the pendulum.

        Args:
            key (jax.Array): PRNG key.
            state (ModelState): Current angle and angular velocity.
            action (jax.Array): Torque.
            params (EnvParams): Environment parameters (mass, length, ``dt``).

        Returns:
            ModelState: Next angle and angular velocity.
        """

        u = jnp.clip(action, -params.max_torque, params.max_torque)

        newthdot = state.theta_dot + (
            (
                3 * params.g / (2 * params.l) * jnp.sin(state.theta)
                + 3.0 / (params.m * params.l**2) * u
            )
            * params.dt
        )

        newthdot = jnp.clip(newthdot, -params.max_speed, params.max_speed)
        newth = state.theta + newthdot * params.dt

        state = ModelState(
            theta=newth.squeeze(),
            theta_dot=newthdot.squeeze(),
        )
        return state

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset the pendulum to a random angle and angular velocity.

        Args:
            key (jax.Array): PRNG key.
            params (EnvParams): Environment parameters.

        Returns:
            tuple[jax.Array, EnvState]: First observation and initial state.
        """
        high = jnp.array([jnp.pi, 1])
        state = jax.random.uniform(key, shape=(2,), minval=-high, maxval=high)
        state = ModelState(theta=state[0], theta_dot=state[1])
        env_state = EnvState(state=state, last_u=jnp.array(0.0), time=0)
        return self._obs_from_model_state(state), env_state

    def get_obs(
        self,
        env_state: EnvState,
        key: jax.Array | None = None,
        params: EnvParams | None = None,
    ) -> jax.Array:
        """Observation of an environment state.

        The key and params arguments keep the signature expected by the sensitivity wrappers.

        Args:
            env_state (EnvState): Environment state.
            key (jax.Array, optional): Unused.
            params (EnvParams, optional): Unused.

        Returns:
            jax.Array: ``[cos(theta), sin(theta), theta_dot]``.
        """
        return self._obs_from_model_state(env_state.state)

    def _obs_from_model_state(self, state: ModelState) -> jax.Array:
        """Observation of a model state.

        Args:
            state (ModelState): Angle and angular velocity.

        Returns:
            jax.Array: ``[cos(theta), sin(theta), theta_dot]``.
        """
        return jnp.array(
            [
                jnp.cos(state.theta),
                jnp.sin(state.theta),
                state.theta_dot,
            ]
        ).squeeze()

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Whether the episode has reached its maximum length.

        Args:
            state (EnvState): Environment state.
            params (EnvParams): Environment parameters (``max_steps_in_episode``).

        Returns:
            jax.Array: Boolean done flag.
        """
        # Check number of steps in episode termination condition
        done = state.time >= params.max_steps_in_episode
        return jnp.array(done)

    def eval_metrics(self, env_state: EnvState, params: EnvParams) -> dict[str, jax.Array]:
        """Evaluation metrics of the current state, logged during the rollouts.

        Args:
            env_state (EnvState): Current environment state.
            params (EnvParams): Environment parameters.

        Returns:
            dict[str, jax.Array]: Metric name to scalar value.
        """
        state = env_state.state
        
        # Angle error from upright position (target is theta=0)
        angle_error = jnp.abs(state.theta)
        
        # Angular velocity magnitude
        angular_velocity = jnp.abs(state.theta_dot)
        
        return {
            "angle_error": angle_error,
            "angular_velocity": angular_velocity,
        }

    @property
    def name(self) -> str:
        """Environment name.

        Returns:
            str: Name used in the logs.
        """
        return "Pendulum-Sensitivity"

    @property
    def num_actions(self) -> int:
        """Number of action components.

        Returns:
            int: Size of the action vector.
        """
        return 1

    def action_space(self, params: EnvParams | None = None) -> spaces.Box:
        """Action space of the environment.

        Args:
            params (EnvParams, optional): Environment parameters; the defaults when ``None``.

        Returns:
            spaces.Box: Box between ``min_action`` and ``max_action``.
        """
        if params is None:
            params = self.default_params
        return spaces.Box(
            low=-params.max_torque,
            high=params.max_torque,
            shape=(1,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment.

        Args:
            params (EnvParams): Environment parameters.

        Returns:
            spaces.Box: Box with one entry per observation component.
        """
        high = jnp.array([1.0, 1.0, params.max_speed], dtype=jnp.float32)
        return spaces.Box(-high, high, shape=(self.obs_dim,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment.

        Args:
            params (EnvParams): Environment parameters.

        Returns:
            spaces.Dict: Spaces of the state components.
        """
        return spaces.Dict(
            {
                "state": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (2,),
                    jnp.float32,
                ),
                "last_u": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

    @staticmethod
    def plot_rollouts(series_list: List[dict], output: Path | None = None) -> None:
        """Plot the trajectories of a set of rollouts.

        Args:
            series_list (list[dict]): One dict per rollout, with ``state`` (shape ``(T+1, 2)``,
                angle and angular velocity) and ``actions`` (shape ``(T, 1)``).
            output (pathlib.Path, optional): File to save the figure to; shown interactively when ``None``.
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)

        for series in series_list:
            state = series.get("state", None)
            if state is None:
                continue
            state = np.asarray(state)
            theta = state[:, 0]
            theta_dot = state[:, 1]
            timesteps = np.arange(theta.shape[0])
            
            axes[0, 0].plot(timesteps, theta, alpha=0.4)
            axes[0, 1].plot(timesteps, theta_dot, alpha=0.4)
            
            actions = np.asarray(series.get("actions", []))
            if actions.size > 0:
                action = actions[:, 0] if actions.ndim > 1 else actions
                axes[1, 0].plot(np.arange(action.shape[0]), action, alpha=0.4)
            
            axes[1, 1].plot(theta, theta_dot, alpha=0.4)

        axes[0, 0].set_title("Theta")
        axes[0, 0].set_xlabel("t")
        axes[0, 0].set_ylabel("theta")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set_title("Theta dot")
        axes[0, 1].set_xlabel("t")
        axes[0, 1].set_ylabel("theta_dot")
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].set_title("Action")
        axes[1, 0].set_xlabel("t")
        axes[1, 0].set_ylabel("u")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].set_title("Phase portrait")
        axes[1, 1].set_xlabel("theta")
        axes[1, 1].set_ylabel("theta_dot")
        axes[1, 1].grid(True, alpha=0.3)

        fig.suptitle("Pendulum trajectories")

        if output is None:
            plt.show()
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=150)
            plt.close(fig)
