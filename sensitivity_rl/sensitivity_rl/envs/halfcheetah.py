"""HalfCheetah-Sensitivity
JAX implementation of HalfCheetah-v5 MuJoCo environment with added
parametric sensitivity metric optimization.

Based on the MuJoCo HalfCheetah environment.
"""

from typing import Any, List
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.flatten_util
from flax import struct

import mujoco
import mujoco.mjx as mjx

from pathlib import Path

from gymnax.environments import environment, spaces
from gymnax.environments.environment import TEnvState, TEnvParams

from envs.base import Sensitivity

# model_path = Path(__file__).parent / "assets" / "half_cheetah.xml"
# mj_model = mujoco.MjModel.from_xml_path(str(model_path))
# mj_data = mujoco.MjData(mj_model)


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
        model_path = Path(__file__).parent / "assets" / "half_cheetah.xml"
        self.mj_model = mujoco.MjModel.from_xml_path(str(model_path))
        self.mj_data = mujoco.MjData(self.mj_model)
        self.model = mjx.put_model(self.mj_model)
        self.data = mjx.put_data(self.mj_model, self.mj_data)
        # to handle in a different way
        self.type = "mujoco"
        base_data = mjx.put_data(self.mj_model, self.mj_data)

        flat_vector, unravel_fn = jax.flatten_util.ravel_pytree(base_data)
        total_size = flat_vector.shape[0]
        index_vector = jnp.arange(total_size)
        index_map = unravel_fn(index_vector.astype(jnp.float32))
        index_qpos = index_map.qpos
        index_qvel = index_map.qvel

        self.state_idx = (int(index_qpos[0]), int(index_qvel[-1]))
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

        x_velocity = (
            (state_next.qpos[0] - state.qpos[0]) / params.frame_skip / 0.01
        )

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
            self.model.geom_friction.shape[0], axis=0
        )
        damping = jnp.concatenate(
            [jnp.array([0.0, 0.0, 0.0]), params.damping], axis=0
        )
        model = self.model.replace(geom_friction=friction, dof_damping=damping)
        data = self.data.replace(qpos=state.qpos, qvel=state.qvel, ctrl=action)

        # Apply frame skip
        # Define the function for a single step
        def step_fn(data_carry, _):
            # 'model' is captured from the outer scope
            data_new = mjx.step(model, data_carry)
            return data_new, None

        # Apply frame skip using jax.lax.scan
        # This is fast to compile, even for large frame_skip
        data, _ = jax.lax.scan(step_fn, data, None, length=params.frame_skip)

        state_new = ModelState(
            qpos=data.qpos,
            qvel=data.qvel,
        )

        return state_new

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset environment state with small random noise."""
        rng, rng_qpos = jax.random.split(key, 2)

        qpos_noise = jax.random.uniform(
            rng_qpos, self.data.qpos.shape, minval=-0.1, maxval=0.1
        )
        new_qpos = self.data.qpos + qpos_noise

        state = ModelState(
            qpos=new_qpos,
            qvel=jnp.zeros_like(self.data.qvel),
        )

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
                state.qpos.reshape(-1),
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
        qpos_dim = self.data.qpos.size
        qvel_dim = self.data.qvel.size

        return spaces.Dict(
            {
                "state": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (self.state_dim,),
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

    @staticmethod
    def plot_rollouts(series_list: List[dict], output: Path | None = None) -> None:
        """Plot HalfCheetah rollout trajectories.
        
        Args:
            series_list: List of trajectory dictionaries.
            output: Optional path to save the figure.
        """
        print("[plot_rollouts] HalfCheetah plotting is not implemented yet.")
