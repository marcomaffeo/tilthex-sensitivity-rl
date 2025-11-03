"""HalfCheetah-Sensitivity
JAX implementation of HalfCheetah-v5 MuJoCo environment with added
parametric sensitivity metric optimization.

Based on the MuJoCo HalfCheetah environment.
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

from envs.base import Sensitivity

model_path = Path(__file__).parent / "assets" / "half_cheetah.xml"
mj_model = mujoco.MjModel.from_xml_path(str(model_path))
mj_data = mujoco.MjData(mj_model)

# Put on JAX device
base_model = mjx.put_model(mj_model)
base_data = mjx.put_data(mj_model, mj_data)


@struct.dataclass
class ModelState:
    qpos: jax.Array
    qvel: jax.Array


@struct.dataclass
class EnvState(environment.EnvState):
    state: ModelState
    last_u: jax.Array  # Only needed for rendering
    time: int

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self.state, name)


@struct.dataclass
class EnvParams(environment.EnvParams):
    min_torque: jax.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
    )
    max_torque: jax.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    )
    ctrl_cost_weight: float = struct.field(pytree_node=False, default=0.1)
    forward_reward_weight: float = struct.field(pytree_node=False, default=1.0)
    frame_skip: int = struct.field(pytree_node=False, default=1)
    damping: jax.Array = struct.field(
        pytree_node=True,
        default_factory=lambda: jnp.array([6, 4.5, 3, 4.5, 3, 1.5]),
    )
    friction: jax.Array = struct.field(
        pytree_node=True,
        default_factory=lambda: jnp.array([0.4, 0.1, 0.1]),
    )
    max_steps_in_episode: int = struct.field(pytree_node=False, default=1000)
    is_testing: bool = struct.field(pytree_node=False, default=False)


class HalfCheetahV5(environment.Environment):
    """JAX Compatible version of HalfCheetah-v5."""

    def __init__(self):

        self.obs_dim = base_data.qpos.size + base_data.qvel.size

        self.state_dim = self.obs_dim

        self.action_dim = self.action_space(self.default_params).shape[0]
        # Only count pytree parameters (damping and friction), not static fields
        self.param_dim = len(self.default_params.damping) + len(
            self.default_params.friction
        )

        # Parameter bounds for damping and friction
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
        """Integrate HalfCheetah dynamics and return transition."""

        u = jnp.clip(action, params.min_torque, params.max_torque)

        state = env_state.state

        state_next = self.step_dynamics(key, state, u, params)
        x_velocity = state_next.qvel[0]
        forward_reward = params.forward_reward_weight * x_velocity
        ctrl_cost = params.ctrl_cost_weight * jnp.sum(jnp.square(u))

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
        HalfCheetah dynamics using MuJoCo MJX

        Parameters
        ----------
        state : ModelState
            state vector with qpos and qvel
        action : jax.Array
            control inputs (joint torques)
        params : EnvParams
            environment parameters including damping and friction

        Returns
        -------
        state_next : ModelState
            next state after frame_skip steps
        """

        friction = params.friction.reshape((1, -1)).repeat(
            base_model.geom_friction.shape[0], axis=0
        )
        damping = jnp.concatenate(
            [jnp.array([0.0, 0.0, 0.0]), params.damping], axis=0
        )
        model = base_model.replace(geom_friction=friction, dof_damping=damping)
        data = base_data.replace(ctrl=action, qpos=state.qpos, qvel=state.qvel)

        jax.debug.print("Step HalfCheetah with params:1")

        # Apply frame skip
        # for _ in range(params.frame_skip):
        data = mjx.step(model, data)

        jax.debug.print("Step HalfCheetah with params:2")

        state_new = ModelState(
            qpos=data.qpos.squeeze(),
            qvel=data.qvel.squeeze(),
        )

        return state_new

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset environment state with small random noise."""
        qpos_noise = jax.random.uniform(
            key, base_data.qpos.shape, minval=-0.1, maxval=0.1
        )
        qpos = base_data.qpos + qpos_noise
        qvel = jnp.zeros_like(base_data.qvel)

        state = ModelState(qpos=qpos, qvel=qvel)

        env_state = EnvState(
            state=state,
            last_u=jnp.zeros(self.action_dim),
            time=0,
        )
        return self.get_obs(env_state), env_state

    def get_obs(self, env_state: EnvState) -> jax.Array:
        """Return a vector of observations from the state."""
        state = env_state.state

        return jnp.concatenate(
            [
                state.qpos[:].reshape(-1),
                state.qvel.reshape(-1),
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
        return "HalfCheetah-v5"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 6

    def action_space(self, params: EnvParams | None = None) -> spaces.Box:
        """Action space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Box(
            low=params.min_torque,
            high=params.max_torque,
            shape=(6,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(-1, 1, shape=(self.obs_dim,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        qpos_dim = base_data.qpos.size
        qvel_dim = base_data.qvel.size

        return spaces.Dict(
            {
                "state": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (qpos_dim + qvel_dim,),
                    jnp.float32,
                ),
                "last_u": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (6,),
                    jnp.float32,
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )
