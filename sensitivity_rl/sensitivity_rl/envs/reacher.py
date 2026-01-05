"""Reacher-Sensitivity
JAX implementation of Reacher-v5 MuJoCo environment with added
parametric sensitivity metric optimization.

Based on the MuJoCo Reacher environment.
"""

from typing import Any, List
from functools import partial
from pathlib import Path
from xml.parsers.expat import model

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
    fingertip: jax.Array


@struct.dataclass
class EnvState(environment.EnvState):
    state: ModelState
    target: jax.Array
    last_u: jax.Array  # Only needed for rendering
    time: int

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self.state, name)


@struct.dataclass
class EnvParams(environment.EnvParams):
    min_torque: jax.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([-1.0, -1.0]),
    )
    max_torque: jax.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([1.0, 1.0]),
    )
    reward_control_weight: float = struct.field(pytree_node=False, default=1.0)
    reward_dist_weight: float = struct.field(pytree_node=False, default=1.0)
    frame_skip: int = struct.field(pytree_node=False, default=1)
    damping: jax.Array = struct.field(
        pytree_node=True,
        default_factory=lambda: jnp.array([1.0, 1.0]),
    )
    friction: jax.Array = struct.field(
        pytree_node=True,
        default_factory=lambda: jnp.array([1.0, 0.1, 0.1]),
    )
    mass: jax.Array = struct.field(
        pytree_node=True,
        default_factory=lambda: jnp.array([0.03560472, 0.03560472, 0.00418879]),
    )
    inertia: jax.Array = struct.field(
        pytree_node=True,
        default_factory=lambda: jnp.array(
            [
                [3.91756604e-05, 3.91756604e-05, 1.73834793e-06],
                [3.91756604e-05, 3.91756604e-05, 1.73834793e-06],
                [1.67551608e-07, 1.67551608e-07, 1.67551608e-07],
            ]
        ),
    )
    max_steps_in_episode: int = struct.field(pytree_node=False, default=50)
    is_testing: bool = struct.field(pytree_node=False, default=False)


class ReacherV5(environment.Environment):
    """JAX Compatible version of Reacher-v5."""

    def __init__(self):
        model_path = Path(__file__).parent / "assets" / "reacher.xml"
        self.mj_model = mujoco.MjModel.from_xml_path(str(model_path))

        self.fingertip_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "fingertip"
        )
        self.target_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )

        self.mj_data = mujoco.MjData(self.mj_model)
        self.model = mjx.put_model(self.mj_model)
        self.data = mjx.put_data(self.mj_model, self.mj_data)

        # Pre-compute Subtree Mask
        subtree_mask = jnp.eye(self.mj_model.nbody)
        parents = self.mj_model.body_parentid
        for i in range(self.mj_model.nbody - 1, -1, -1):
            parent_id = parents[i]
            if parent_id >= 0:
                subtree_mask = subtree_mask.at[parent_id, :].add(subtree_mask[i, :])
        self.SUBTREE_MASK = jnp.array(subtree_mask)

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
        self.obs_dim = base_data.qpos.size + 6

        self.state_dim = base_data.qpos.size + base_data.qvel.size + 3

        self.action_dim = self.action_space(self.default_params).shape[0]
        # Only count pytree parameters (damping and friction), not static fields
        self.param_dim = (
            len(self.default_params.damping)
            + len(self.default_params.friction)
            + len(self.default_params.mass)
            + jnp.prod(jnp.array(self.default_params.inertia.shape))
        )

        # Parameter bounds for damping and friction
        self.params_min = [
            jnp.array([0.1, 0.1]),  # damping
            jnp.array([0.5, 0.05, 0.05]),  # friction
            jnp.array([0.01, 0.01, 0.001]),  # mass
            jnp.array(
                [
                    [2.0e-05, 2.0e-05, 0.5e-06],
                    [1.0e-05, 1.0e-05, 0.5e-06],
                    [0.5e-07, 0.5e-07, 0.5e-07],
                ]
            ),  # inertia
        ]
        self.params_max = [
            jnp.array([2.0, 2.0]),  # damping
            jnp.array([2.0, 0.2, 0.2]),  # friction
            jnp.array([0.07, 0.07, 0.01]),  # mass
            jnp.array(
                [
                    [8.0e-05, 8.0e-05, 3.0e-06],
                    [4.0e-05, 4.0e-05, 3.0e-06],
                    [3.0e-07, 3.0e-07, 3.0e-07],
                ]
            ),  # inertia
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

        target = env_state.target

        reward_dist = -params.reward_dist_weight * jnp.sum(
            jnp.square(state_next.fingertip[:2] - target[:2])
        )
        ctrl_cost = -params.reward_control_weight * jnp.sum(jnp.square(u))

        reward = reward_dist + ctrl_cost

        reward = reward.squeeze()

        env_state = EnvState(
            state=state_next,
            target=target,
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

        model = self.rebuild_model(self.model, params)

        xpos = self.data.xpos
        xpos = xpos.at[self.fingertip_id].set(state.fingertip)
        data = self.data.replace(
            qpos=state.qpos, qvel=state.qvel, xpos=xpos, ctrl=action
        )

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
            fingertip=data.xpos[self.fingertip_id],
        )

        return state_new

    def rebuild_model(self, base_model: mjx.Model, params: EnvParams) -> mjx.Model:
        """
        1. Takes the base model's FULL mass/inertia.
        2. Overwrites the [1:-1] slice with values from params.
        3. Recalculates dependencies (Subtree Mass, Solver Weights).
        """

        friction = params.friction.reshape((1, -1)).repeat(
            self.model.geom_friction.shape[0], axis=0
        )
        damping = jnp.concatenate([params.damping, jnp.array([0.0, 0.0])], axis=0)
        cuurent_mass = base_model.body_mass.at[1:-1].set(params.mass)
        current_inertia = base_model.body_inertia.at[1:-1].set(params.inertia)
        subtree_mass = jnp.matmul(self.SUBTREE_MASK, cuurent_mass)
        inv_mass = 1.0 / jnp.maximum(1e-9, cuurent_mass)
        inv_inertia = 1.0 / jnp.maximum(1e-9, jnp.mean(current_inertia, axis=1))
        inv_weight0 = jnp.stack([inv_mass, inv_inertia], axis=1)
        return base_model.replace(
            body_mass=cuurent_mass,
            body_inertia=current_inertia,
            body_subtreemass=subtree_mass,
            body_invweight0=inv_weight0,
            geom_friction=friction,
            dof_damping=damping,
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset environment state with small random noise."""
        rng, rng_qpos = jax.random.split(key, 2)

        model = self.rebuild_model(self.model, params)

        qpos_noise = jax.random.uniform(
            rng_qpos, self.data.qpos.shape, minval=-0.1, maxval=0.1
        )
        new_qpos = self.data.qpos + qpos_noise

        rng, rng_goal = jax.random.split(rng, 2)
        goal = jax.random.uniform(rng_goal, shape=(2,), minval=-0.2, maxval=0.2)

        norm = jnp.linalg.norm(goal)
        goal = jnp.where(norm < 0.2, goal / (norm + 1e-8) * 0.2, goal)

        new_qpos = new_qpos.at[-2:].set(goal)

        rng, rng_qvel = jax.random.split(rng, 2)
        qvel = jax.random.uniform(
            rng_qvel, self.data.qvel.shape, minval=-0.005, maxval=0.005
        )
        qvel = qvel.at[-2:].set(0.0)

        data = self.data.replace(qpos=new_qpos, qvel=qvel)
        data = mjx.forward(model, data)

        state = ModelState(
            qpos=new_qpos,
            qvel=qvel,
            fingertip=data.xpos[self.fingertip_id],
        )

        env_state = EnvState(
            state=state,
            target=goal,
            last_u=jnp.zeros(self.action_dim),
            time=0,
        )
        return self.get_obs(env_state), env_state

    def get_obs(self, env_state: EnvState) -> jax.Array:
        """Return a vector of observations from the state."""
        state = env_state.state
        theta = state.qpos.flatten()[:2]

        return jnp.concatenate(
            [
                jnp.cos(theta).reshape(-1),
                jnp.sin(theta).reshape(-1),
                state.qpos.flatten()[2:].reshape(-1),
                state.qvel.flatten()[:2].reshape(-1),
                (state.fingertip.reshape(-1)[:2] - env_state.target.reshape(-1)[:2]),
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
        return "Reacher-v5"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(self, params: EnvParams | None = None) -> spaces.Box:
        """Action space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Box(
            low=params.min_torque,
            high=params.max_torque,
            shape=(2,),
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
                "target": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (2,),
                    jnp.float32,
                ),
                "last_u": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (2,),
                    jnp.float32,
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

    @staticmethod
    def plot_rollouts(series_list: List[dict], output: Path | None = None) -> None:
        """Plot Reacher rollout trajectories showing full body motion.

        Args:
            series_list: List of trajectory dictionaries containing qpos and qvel.
            output: Optional path to save the figure.
        """
        pass
