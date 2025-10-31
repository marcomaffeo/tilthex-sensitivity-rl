"""Pendulum-Sensitivity
JAX implementation of Pendulum-v0 OpenAI gym environment with added
parametric sensitivity metric optimization.

Source: github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
"""

from typing import Any
from functools import partial

import jax
import jax.numpy as jnp
from flax import struct

import mujoco
import mujoco.mjx as mjx

from pathlib import Path

from gymnax.environments import environment, spaces
from gymnax.environments.environment import TEnvState, TEnvParams
import mujoco

from envs.base import Sensitivity

from utils import angle_normalize


@struct.dataclass
class ModelState:
    qpos: jax.Array
    qvel: jax.Array


@struct.dataclass
class EnvState(environment.EnvState):
    state: ModelState
    target_pos: jax.Array
    last_u: jax.Array  # Only needed for rendering
    time: int

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self.state, name)


@struct.dataclass
class EnvParams(environment.EnvParams):
    min_torque: jax.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([0, -2.5, -2.5, -2]),
    )
    max_torque: jax.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([1, 2.5, 2.5, 2])
    )
    env_max_limits: jax.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([1.0, 1.0, 1.0])
    )
    env_min_limits: jax.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([-1.0, -1.0, 0.0])
    )
    ctrl_cost_weight: float = struct.field(pytree_node=False, default=0.1)
    forward_reward_weight: float = struct.field(pytree_node=False, default=1.0)
    dt: int = struct.field(pytree_node=False, default=5)
    damping: jax.Array = struct.field(
        pytree_node=True,
        default_factory=lambda: jnp.array([6, 4.5, 3, 4.5, 3, 1.5]),
    )
    # TODO: check this
    friction: jax.Array = struct.field(
        pytree_node=True,
        default_factory=lambda: jnp.array([0.4, 0.1, 0.1]),
    )
    max_steps_in_episode: int = struct.field(pytree_node=False, default=200)
    is_testing: bool = struct.field(pytree_node=False, default=False)


class HalfCheetahV5(environment.Environment):
    """JAX Compatible version of HalfCheetah-v5."""

    def __init__(self):

        model_path = Path(__file__).parent / "assets" / "half_cheetah.xml"
        mj_model = mujoco.MjModel.from_xml_path(str(model_path))
        mj_data = mujoco.MjData(mj_model)

        # Put on JAX device
        self.base_model = mjx.put_model(mj_model)
        self.base_data = mjx.put_data(mj_model, mj_data)

        self.obs_dim = self.base_data.qpos.size + self.base_data.qvel.size

        self.state_dim = self.obs_dim

        self.action_dim = self.action_space(self.default_params).shape[0]
        self.param_dim = 4  # len(jax.tree.leaves(self.default_params))
        self.target_pos = jnp.array([0.0, 0.0, 0.5])
        self.hovering_action = jnp.array(
            [self.default_params.m * self.default_params.g, 0.0, 0.0, 0.0]
        )

        self.params_min = [
            jnp.array([2, 1.5, 2, 1.5, 2, 1.5]),  # damping
            jnp.array([0.2, 0.05, 0.05]),  # friction
        ]
        self.params_max = [
            jnp.array([8, 6.5, 4, 6.5, 4, 4.5]),  # damping
            jnp.array([0.8, 0.2, 0.2]),  # friction
        ]
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for Cheetah-v5."""
        return EnvParams()

    @partial(jax.jit, static_argnames=("self",))
    def step(
        self,
        key: jax.Array,
        state: TEnvState,
        action: int | float | jax.Array,
        params: TEnvParams | None = None,
    ) -> tuple[jax.Array, TEnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Overload step to remove the reset when done."""

        if params is None:
            params = self.default_params

        # Step
        key_step, _ = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(
            key_step, state, action, params
        )
        return obs_st, state_st, reward, done, info

    @partial(jax.jit, static_argnums=(0))
    def step_env(
        self,
        key: jax.Array,
        env_state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Integrate pendulum ODE and return transition."""

        u = jnp.clip(action, params.min_torque, params.max_torque)

        state = env_state.state

        state_next = self.step_dynamics(key, state, action, params)
        x_velocity = state_next.qvel[0]
        forward_reward = params.forward_reward_weight * x_velocity
        ctrl_cost = params.ctrl_cost_weight * jnp.sum(jnp.square(action))

        reward = forward_reward - ctrl_cost

        reward = reward.squeeze()

        env_state = EnvState(
            state=state_next,
            last_u=u.reshape(-1),
            time=env_state.time + 1,
        )

        done = self.is_terminal(env_state, params)
        return (
            self.get_obs(env_state),
            env_state,
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
        """
        Simple quadrotor dynamics model with CoM placed at the geometric center

        Parameters
        ----------
        state : jnp.array
            state vector [pos (world frame),
                        attitude (unit quaternion [w, x, y, z]),
                        vel (world frame),
                        angular_velocity (body frame)]
        inputs : jnp.array):
            input vector [thrust (along the body-frame z axis), torque (body frame)]
        Returns
        -------
        state_next : ModelState
            next state after time dt
        """

        friction = params.friction.reshape((1, -1)).repeat(
            self.base_model.geom_friction.shape[0], axis=0
        )
        damping = jnp.concatenate(
            [jnp.array([0.0, 0.0, 0.0]), params.damping], axis=0
        )
        model = self.base_model.replace(
            geom_friction=friction, dof_damping=damping
        )
        data = self.base_data.replace(
            ctrl=action, qpos=state.qpos, qvel=state.qvel
        )
        data = mjx.step(model, data)

        state_new = ModelState(
            qpos=data.qpos.squeeze(),
            qvel=data.qvel.squeeze(),
        )

        return state_new

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset environment state by sampling theta, theta_dot."""
        pos = jax.random.uniform(
            key,
            shape=(3,),
            minval=params.env_min_limits,
            maxval=params.env_max_limits,
        )
        attitude = jnp.array([1.0, 0.0, 0.0, 0.0])
        vel = jnp.array([0.0, 0.0, 0.0])
        omega = jnp.array([0.0, 0.0, 0.0])
        state = ModelState(pos=pos, attitude=attitude, vel=vel, omega=omega)

        key, _ = jax.random.split(key)
        target_pos = jax.lax.select(
            params.is_testing,
            self.target_pos,
            jax.random.uniform(
                key,
                shape=(3,),
                minval=params.env_min_limits,
                maxval=params.env_max_limits,
            ),
        )

        env_state = EnvState(
            state=state,
            target_pos=target_pos,
            last_u=self.hovering_action,
            time=0,
        )
        return self.get_obs(env_state), env_state

    def get_obs(self, env_state: EnvState) -> jax.Array:
        """Return a vector of observations from the state."""
        state = env_state.state

        pos = state.pos
        vel = state.vel
        to_target = env_state.target_pos - pos
        q = state.attitude
        ang_vel = state.omega

        return jnp.concatenate(
            [
                to_target.reshape(-1),
                vel.reshape(-1),
                q.reshape(-1),
                ang_vel.reshape(-1),
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
            low=params.min_torque,
            high=params.max_torque,
            shape=(4,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(-1, 1, shape=(self.obs_dim,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "state": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (13,),
                    jnp.float32,
                ),
                "target_pos": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (3,),
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
