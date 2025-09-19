"""Pendulum-Sensitivity
JAX implementation of Pendulum-v0 OpenAI gym environment with added
parametric sensitivity metric optimization.

Source: github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
"""

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces

from envs.base import EnvSensitivity, Sensitivity

@struct.dataclass
class ModelState():
    theta: jax.Array
    theta_dot: jax.Array



@struct.dataclass
class EnvState(environment.EnvState):
    state: ModelState
    sensitivity: Sensitivity
    last_u: jax.Array # Only needed for rendering
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    max_speed: float = struct.field(pytree_node=False, default=8.0)
    max_torque: float = struct.field(pytree_node=False, default=2.0)
    dt: float = struct.field(pytree_node=False, default=0.05)
    g: float = struct.field(pytree_node=False, default=10.0)
    m: float = 1.0
    l: float = 1.0  # noqa: E741
    max_steps_in_episode: int = struct.field(pytree_node=False, default=200)


class Pendulum(EnvSensitivity):
    """JAX Compatible version of Pendulum-v0 OpenAI gym environment."""
    def __init__(self):
        self.state_dim = 2
        self.obs_dim = 3
        self.action_dim = self.action_space(self.default_params).shape[0]
        self.param_dim = len(jax.tree.leaves(self.default_params))
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for Pendulum-v0."""
        return EnvParams()

    def default_sensitivity(self) -> Sensitivity:
        """Default sensitivity parameters."""
        return Sensitivity(
            state=jnp.zeros((self.state_dim, self.param_dim)), # ds/dp
            action=jnp.zeros((self.action_dim, self.param_dim )), # du/dp
            policy_gains=jnp.zeros((self.action_dim, self.obs_dim)), # dmu/dobs
        )

    def step_env(
        self,
        key: jax.Array,
        env_state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Integrate pendulum ODE and return transition."""

        u = jnp.clip(action, -params.max_torque, params.max_torque)

        state = env_state.state

        reward = - (
            angle_normalize(state.theta) ** 2
            + 0.1 * state.theta_dot**2
            + 0.001 * (u**2)
            #+ 0.1*jnp.linalg.matrix_norm(env_state.sensitivity.action, ord='fro')
        )
        # reward = jax.lax.select(
        #     jnp.abs(angle_normalize(state.theta) < 0.1),
        #     reward - 0.1*jnp.linalg.matrix_norm(env_state.sensitivity.state, ord='fro'),
        #     reward
        # )

        reward = reward.squeeze()

        state_next = self.step_dynamics(key, state, action, params)

        sens_next = self.step_sensitivity(key, env_state, action, params)

        env_state = EnvState(
            state=state_next,
            sensitivity=sens_next, # Placeholder, to be implemented
            last_u=u.reshape(),
            time=env_state.time + 1,
        )

        done = self.is_terminal(env_state, params)
        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(env_state),
            reward,
            done,
            {"discount": self.discount(env_state, params)},
        )

    def step_dynamics(
        self,
        key: jax.Array,
        state: ModelState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> ModelState:
        """Integrate pendulum ODE and return next state."""

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
        """Reset environment state by sampling theta, theta_dot."""
        high = jnp.array([jnp.pi, 1])
        state = jax.random.uniform(key, shape=(2,), minval=-high, maxval=high)
        state = ModelState(theta=state[0], theta_dot=state[1])
        env_state = EnvState(
            state=state, sensitivity=self.default_sensitivity(), last_u=jnp.array(0.0), time=0
        )
        return self.get_obs(state), env_state

    def get_obs(self, state: ModelState) -> jax.Array:
        """Return angle in polar coordinates and change."""
        return jnp.array(
            [
                jnp.cos(state.theta),
                jnp.sin(state.theta),
                state.theta_dot,
            ]
        ).squeeze()

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = state.time >= params.max_steps_in_episode
        return jnp.array(done)

    @property
    def name(self) -> str:
        """Environment name."""
        return "Pendulum-Sensitivity"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 1

    def action_space(self, params: EnvParams | None = None) -> spaces.Box:
        """Action space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Box(
            low=-params.max_torque,
            high=params.max_torque,
            shape=(1,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        high = jnp.array([1.0, 1.0, params.max_speed], dtype=jnp.float32)
        return spaces.Box(-high, high, shape=(self.obs_dim,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
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


def angle_normalize(x: jax.Array) -> jax.Array:
    """Normalize the angle - radians."""
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi
