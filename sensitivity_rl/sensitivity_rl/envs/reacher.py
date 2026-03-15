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
from yaml import warnings

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
        default_factory=lambda: jnp.array([0.01, 0.01]),
    )
    friction: jax.Array = struct.field(
        pytree_node=True,
        default_factory=lambda: jnp.array([1.0, 5.0e-03, 1.0e-04]),
    )
    mass: jax.Array = struct.field(
        pytree_node=True,
        default_factory=lambda: jnp.array([0.0418879, 0.03560472, 0.00418879]),
    )
    inertia: jax.Array = struct.field(
        pytree_node=True,
        default_factory=lambda: jnp.array(
            [
                [6.33135639e-05, 6.33135639e-05, 2.05250720e-06],
                [3.91756604e-05, 3.91756604e-05, 1.73834793e-06],
                [1.67551608e-07, 1.67551608e-07, 1.67551608e-07],
            ]
        ),
    )
    max_steps_in_episode: int = struct.field(pytree_node=False, default=1000)
    is_testing: bool = struct.field(pytree_node=False, default=False)


def tolerance(
    x, bounds=(0.0, 0.0), margin=0.0, sigmoid="gaussian", value_at_margin=0.1
):
    """Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.

    Args:
      x: A scalar or numpy array.
      bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
        the target interval. These can be infinite if the interval is unbounded
        at one or both ends, or they can be equal to one another if the target
        value is exact.
      margin: Float. Parameter that controls how steeply the output decreases as
        `x` moves out-of-bounds.
        * If `margin == 0` then the output will be 0 for all values of `x`
          outside of `bounds`.
        * If `margin > 0` then the output will decrease sigmoidally with
          increasing distance from the nearest bound.
      sigmoid: String, choice of sigmoid type. Valid values are: 'gaussian',
         'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.
      value_at_margin: A float between 0 and 1 specifying the output value when
        the distance from `x` to the nearest bound is equal to `margin`. Ignored
        if `margin == 0`.

    Returns:
      A float or numpy array with values between 0.0 and 1.0.

    Raises:
      ValueError: If `bounds[0] > bounds[1]`.
      ValueError: If `margin` is negative.
    """
    lower, upper = bounds
    if lower > upper:
        raise ValueError("Lower bound must be <= upper bound.")
    if margin < 0:
        raise ValueError("`margin` must be non-negative.")

    in_bounds = jnp.logical_and(lower <= x, x <= upper)
    if margin == 0:
        value = jnp.where(in_bounds, 1.0, 0.0)
    else:
        d = jnp.where(x < lower, lower - x, x - upper) / margin
        value = jnp.where(in_bounds, 1.0, _sigmoids(d, value_at_margin, sigmoid))
    return jnp.array(value) if jnp.isscalar(x) else value


def _sigmoids(x, value_at_1, sigmoid):
    """Returns 1 when `x` == 0, between 0 and 1 otherwise.

    Args:
      x: A scalar or numpy array.
      value_at_1: A float between 0 and 1 specifying the output when `x` == 1.
      sigmoid: String, choice of sigmoid type.

    Returns:
      A numpy array with values between 0.0 and 1.0.

    Raises:
      ValueError: If not 0 < `value_at_1` < 1, except for `linear`, `cosine` and
        `quadratic` sigmoids which allow `value_at_1` == 0.
      ValueError: If `sigmoid` is of an unknown type.
    """
    if sigmoid in ("cosine", "linear", "quadratic"):
        if not 0 <= value_at_1 < 1:
            raise ValueError(
                "`value_at_1` must be nonnegative and smaller than 1, "
                "got {}.".format(value_at_1)
            )
    else:
        if not 0 < value_at_1 < 1:
            raise ValueError(
                "`value_at_1` must be strictly between 0 and 1, "
                "got {}.".format(value_at_1)
            )

    if sigmoid == "gaussian":
        scale = jnp.sqrt(-2 * jnp.log(value_at_1))
        return jnp.exp(-0.5 * (x * scale) ** 2)

    elif sigmoid == "hyperbolic":
        scale = jnp.arccosh(1 / value_at_1)
        return 1 / jnp.cosh(x * scale)

    elif sigmoid == "long_tail":
        scale = jnp.sqrt(1 / value_at_1 - 1)
        return 1 / ((x * scale) ** 2 + 1)

    elif sigmoid == "reciprocal":
        scale = 1 / value_at_1 - 1
        return 1 / (abs(x) * scale + 1)

    elif sigmoid == "cosine":
        scale = jnp.arccos(2 * value_at_1 - 1) / jnp.pi
        scaled_x = x * scale
        cos_pi_scaled_x = jnp.cos(jnp.pi * scaled_x)
        return jnp.where(abs(scaled_x) < 1, (1 + cos_pi_scaled_x) / 2, 0.0)

    elif sigmoid == "linear":
        scale = 1 - value_at_1
        scaled_x = x * scale
        return jnp.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)

    elif sigmoid == "quadratic":
        scale = jnp.sqrt(1 - value_at_1)
        scaled_x = x * scale
        return jnp.where(abs(scaled_x) < 1, 1 - scaled_x**2, 0.0)

    elif sigmoid == "tanh_squared":
        scale = jnp.arctanh(jnp.sqrt(1 - value_at_1))
        return 1 - jnp.tanh(x * scale) ** 2

    else:
        raise ValueError("Unknown sigmoid type {!r}.".format(sigmoid))


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
        self.mj_model.opt.timestep = 0.02
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

        self.target_pos = jnp.array([0.15, 0.15])

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
            jnp.array([0.001, 0.001]),  # damping
            jnp.array([1.0, 0.5e-03, 1e-05]),  # friction
            jnp.array([0.005, 0.005, 0.001]),  # mass
            jnp.array(
                [
                    [0.2e-05, 0.2e-05, 0.1e-06],
                    [0.2e-05, 0.2e-05, 0.1e-06],
                    [0.1e-07, 0.1e-07, 0.1e-07],
                ]
            ),  # inertia
        ]
        self.params_max = [
            jnp.array([0.06, 0.06]),  # damping
            jnp.array([1.0, 4e-04, 6e-04]),  # friction
            jnp.array([0.15, 0.15, 0.02]),  # mass
            jnp.array(
                [
                    [5.0e-04, 5.0e-04, 1.0e-05],
                    [1.0e-04, 1.0e-04, 1.0e-05],
                    [1.0e-06, 1.0e-06, 1.0e-06],
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
        state_next = self.step_dynamics(key, env_state, u, params)

        target = env_state.target

        dist = jnp.linalg.norm(state_next.fingertip[:2] - target[:2], ord=2)
        reward_dist = -params.reward_dist_weight * dist
        reward_dist_1 = tolerance(dist, bounds=(0, 0.05))
        # ctrl_cost = -params.reward_control_weight * jnp.sum(jnp.square(u))

        reward = reward_dist_1  # + reward_dist  # + ctrl_cost

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
        env_state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> ModelState:
        """Step dynamics extracting state from env_state."""
        return self._step_dynamics(key, env_state.state, action, params)

    def _step_dynamics(
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
            rng_qpos, self.data.qpos.shape, minval=-2.7, maxval=2.7
        )
        qpos_noise = qpos_noise.at[-2:].set(0.0)  # Only noise joints, not target
        new_qpos = self.data.qpos + qpos_noise

        rng, rng_goal = jax.random.split(rng, 2)
        goal = jax.lax.select(
            params.is_testing,
            self.target_pos,
            jax.random.uniform(rng_goal, shape=(2,), minval=-0.17, maxval=0.17),
        )
        norm = jnp.linalg.norm(goal)
        goal = jnp.where(norm < 0.2, goal / (norm + 1e-8) * 0.2, goal)

        new_qpos = new_qpos.at[-2:].set(goal)

        rng, rng_qvel = jax.random.split(rng, 2)
        qvel = jnp.zeros_like(self.data.qvel)

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

    def eval_metrics(self, env_state: EnvState, params: EnvParams) -> dict[str, jax.Array]:
        """Compute evaluation metrics for the current state.
        
        Args:
            env_state: Current environment state
            params: Environment parameters
            
        Returns:
            Dictionary of metric names to scalar values
        """
        state = env_state.state
        
        # Distance from fingertip to target
        distance_to_target = jnp.linalg.norm(
            state.fingertip.reshape(-1)[:2] - env_state.target.reshape(-1)[:2]
        )
        
        # Joint velocities magnitude
        joint_velocity = jnp.linalg.norm(state.qvel.flatten()[:2])
        
        return {
            "distance_to_target": distance_to_target,
            "joint_velocity": joint_velocity,
        }

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
        """Plot Reacher rollout trajectories showing manipulator motion.

        Args:
            series_list: List of trajectory dictionaries containing state, target, and action.
            output: Optional path to save the figure.
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm
        import numpy as np

        # Load MuJoCo model to compute forward kinematics
        model_path = Path(__file__).parent / "assets" / "reacher.xml"
        mj_model = mujoco.MjModel.from_xml_path(str(model_path))
        mj_data = mujoco.MjData(mj_model)

        # Get body IDs
        fingertip_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, "fingertip"
        )

        fig = plt.figure(figsize=(18, 10))

        # Create subplots
        ax_traj = fig.add_subplot(2, 3, 1)  # 2D trajectories
        ax_error = fig.add_subplot(2, 3, 2)  # Error to target
        ax_joints = fig.add_subplot(2, 3, 3)  # Joint trajectories
        ax_input = fig.add_subplot(2, 3, 4)  # Actions/inputs
        ax_joint_vel = fig.add_subplot(2, 3, 5)  # Joint velocities
        ax_fingertip = fig.add_subplot(2, 3, 6)  # Fingertip position over time

        colors = cm.viridis(np.linspace(0, 1, len(series_list)))

        # Collect all trajectory data for computing means and stds
        all_fingertip_x = []
        all_fingertip_y = []
        all_error = []
        all_qpos = []  # Joint positions
        all_qvel = []  # Joint velocities
        all_actions = []
        all_targets = []

        for idx, series in enumerate(series_list):
            # Extract state from series - format: [qpos (4), qvel (4), fingertip (3)]
            state_flat = series["state"]  # Shape: (timesteps, 11)
            target = series["target"]  # Shape: (timesteps, 2)
            action = series["last_u"]  # Shape: (timesteps, 2)

            # Split state
            qpos = state_flat[:, :4]  # First 4: joint angles + target pos
            qvel = state_flat[:, 4:8]  # Next 4: joint velocities + target vel (0)
            fingertip = state_flat[:, 8:11]  # Last 3: fingertip x, y, z

            timesteps = qpos.shape[0]
            steps = np.arange(timesteps)

            # Use the fingertip positions directly from the state (already computed)
            ee_positions = fingertip[:, :2]  # Only x, y (timesteps, 2)

            # Compute error to target
            error = np.linalg.norm(ee_positions - target[:, :2], axis=1)

            # Store data for mean/std computation
            all_fingertip_x.append(ee_positions[:, 0])
            all_fingertip_y.append(ee_positions[:, 1])
            all_error.append(error)
            all_qpos.append(qpos[:, :2])  # Only joint angles
            all_qvel.append(qvel[:, :2])  # Only joint velocities
            all_actions.append(action)
            all_targets.append(target[:, :2])

            # Plot 2D trajectories - show the manipulator motion
            ax_traj.plot(
                ee_positions[:, 0],
                ee_positions[:, 1],
                color=colors[idx],
                alpha=0.3,
                linewidth=1.5,
                label=f"Rollout {idx+1}",
            )

            # Plot target position
            ax_traj.scatter(
                target[0, 0],
                target[0, 1],
                color=colors[idx],
                marker="x",
                s=100,
                alpha=0.8,
            )

            # Plot final position with a star
            ax_traj.scatter(
                ee_positions[-1, 0],
                ee_positions[-1, 1],
                color=colors[idx],
                marker="*",
                s=200,
                alpha=0.8,
                edgecolors="black",
                linewidths=1.5,
            )

            # Draw the arm configuration at a few timesteps
            sample_times = [
                0,
                timesteps // 4,
                timesteps // 2,
                3 * timesteps // 4,
                timesteps - 1,
            ]
            for st in sample_times:
                if st < timesteps:
                    mj_data.qpos[:] = qpos[st]
                    mj_data.qvel[:] = qvel[st]
                    mujoco.mj_forward(mj_model, mj_data)

                    # Base, link1, fingertip
                    base_pos = mj_data.xpos[0][:2]
                    link1_pos = mj_data.xpos[1][:2]
                    finger_pos = mj_data.xpos[fingertip_id][:2]

                    alpha_val = 0.1 + 0.15 * (st / timesteps)
                    ax_traj.plot(
                        [base_pos[0], link1_pos[0], finger_pos[0]],
                        [base_pos[1], link1_pos[1], finger_pos[1]],
                        color=colors[idx],
                        alpha=alpha_val,
                        linewidth=2,
                    )

        # Compute mean and std across all series
        min_length = min(len(traj) for traj in all_error)

        # Truncate all trajectories to minimum length
        error_array = np.array([traj[:min_length] for traj in all_error])
        qpos_array = np.array([qpos[:min_length] for qpos in all_qpos])
        qvel_array = np.array([qvel[:min_length] for qvel in all_qvel])
        action_array = np.array([action[:min_length] for action in all_actions])
        fingertip_x_array = np.array([traj[:min_length] for traj in all_fingertip_x])
        fingertip_y_array = np.array([traj[:min_length] for traj in all_fingertip_y])

        steps = np.arange(min_length)

        # Plot error with mean and std
        error_mean = np.mean(error_array, axis=0)
        error_std = np.std(error_array, axis=0)
        ax_error.plot(
            steps, error_mean, color="red", linewidth=2.5, label="Mean Error", alpha=0.8
        )
        ax_error.fill_between(
            steps,
            error_mean - error_std,
            error_mean + error_std,
            color="red",
            alpha=0.3,
            label="± 1 std",
        )

        # Plot joint trajectories with mean and std
        joint_colors = ["blue", "green"]
        joint_labels = ["Joint 1", "Joint 2"]
        for dim in range(2):
            joint_mean = np.mean(qpos_array[:, :, dim], axis=0)
            joint_std = np.std(qpos_array[:, :, dim], axis=0)
            ax_joints.plot(
                steps,
                joint_mean,
                color=joint_colors[dim],
                linewidth=2.5,
                label=joint_labels[dim],
                alpha=0.8,
            )
            ax_joints.fill_between(
                steps,
                joint_mean - joint_std,
                joint_mean + joint_std,
                color=joint_colors[dim],
                alpha=0.3,
            )

        # Plot joint velocities with mean and std
        for dim in range(2):
            vel_mean = np.mean(qvel_array[:, :, dim], axis=0)
            vel_std = np.std(qvel_array[:, :, dim], axis=0)
            ax_joint_vel.plot(
                steps,
                vel_mean,
                color=joint_colors[dim],
                linewidth=2.5,
                label=f"{joint_labels[dim]} Vel",
                alpha=0.8,
            )
            ax_joint_vel.fill_between(
                steps,
                vel_mean - vel_std,
                vel_mean + vel_std,
                color=joint_colors[dim],
                alpha=0.3,
            )

        # Plot actions/inputs with mean and std
        action_colors = ["purple", "orange"]
        action_labels = ["Action 1", "Action 2"]
        for dim in range(2):
            action_mean = np.mean(action_array[:, :, dim], axis=0)
            action_std = np.std(action_array[:, :, dim], axis=0)
            ax_input.plot(
                steps,
                action_mean,
                color=action_colors[dim],
                linewidth=2.5,
                label=action_labels[dim],
                alpha=0.8,
            )
            ax_input.fill_between(
                steps,
                action_mean - action_std,
                action_mean + action_std,
                color=action_colors[dim],
                alpha=0.3,
            )

        # Plot fingertip position over time
        fingertip_x_mean = np.mean(fingertip_x_array, axis=0)
        fingertip_x_std = np.std(fingertip_x_array, axis=0)
        fingertip_y_mean = np.mean(fingertip_y_array, axis=0)
        fingertip_y_std = np.std(fingertip_y_array, axis=0)

        ax_fingertip.plot(
            steps,
            fingertip_x_mean,
            color="cyan",
            linewidth=2.5,
            label="Fingertip X",
            alpha=0.8,
        )
        ax_fingertip.fill_between(
            steps,
            fingertip_x_mean - fingertip_x_std,
            fingertip_x_mean + fingertip_x_std,
            color="cyan",
            alpha=0.3,
        )
        ax_fingertip.plot(
            steps,
            fingertip_y_mean,
            color="magenta",
            linewidth=2.5,
            label="Fingertip Y",
            alpha=0.8,
        )
        ax_fingertip.fill_between(
            steps,
            fingertip_y_mean - fingertip_y_std,
            fingertip_y_mean + fingertip_y_std,
            color="magenta",
            alpha=0.3,
        )

        # Configure trajectory plot
        ax_traj.set_xlabel("X Position", fontsize=11, fontweight="bold")
        ax_traj.set_ylabel("Y Position", fontsize=11, fontweight="bold")
        ax_traj.set_title(
            "Reacher 2D Manipulator Trajectories\n(lines: paths, x: targets, stars: final positions)",
            fontsize=11,
            fontweight="bold",
        )
        ax_traj.grid(True, alpha=0.3)
        ax_traj.axis("equal")

        # Configure error plot
        ax_error.set_xlabel("Step", fontsize=11, fontweight="bold")
        ax_error.set_ylabel("Distance to Target", fontsize=11, fontweight="bold")
        ax_error.set_title(
            "Mean Error to Target Across All Rollouts", fontsize=11, fontweight="bold"
        )
        ax_error.legend(fontsize=9)
        ax_error.grid(True, alpha=0.3)

        # Configure joint trajectories plot
        ax_joints.set_xlabel("Step", fontsize=11, fontweight="bold")
        ax_joints.set_ylabel("Joint Angle (rad)", fontsize=11, fontweight="bold")
        ax_joints.set_title("Mean Joint Angles", fontsize=11, fontweight="bold")
        ax_joints.legend(fontsize=9)
        ax_joints.grid(True, alpha=0.3)

        # Configure joint velocities plot
        ax_joint_vel.set_xlabel("Step", fontsize=11, fontweight="bold")
        ax_joint_vel.set_ylabel(
            "Joint Velocity (rad/s)", fontsize=11, fontweight="bold"
        )
        ax_joint_vel.set_title("Mean Joint Velocities", fontsize=11, fontweight="bold")
        ax_joint_vel.legend(fontsize=9)
        ax_joint_vel.grid(True, alpha=0.3)

        # Configure input plot
        ax_input.set_xlabel("Step", fontsize=11, fontweight="bold")
        ax_input.set_ylabel("Action (Torque)", fontsize=11, fontweight="bold")
        ax_input.set_title(
            "Mean Control Inputs (Torques)", fontsize=11, fontweight="bold"
        )
        ax_input.legend(fontsize=9)
        ax_input.grid(True, alpha=0.3)

        # Configure fingertip position plot
        ax_fingertip.set_xlabel("Step", fontsize=11, fontweight="bold")
        ax_fingertip.set_ylabel("Position", fontsize=11, fontweight="bold")
        ax_fingertip.set_title(
            "Mean Fingertip Position Over Time", fontsize=11, fontweight="bold"
        )
        ax_fingertip.legend(fontsize=9)
        ax_fingertip.grid(True, alpha=0.3)

        plt.tight_layout()

        if output is not None:
            plt.savefig(output, dpi=150, bbox_inches="tight")
            print(f"[plot_rollouts] Figure saved to {output}")
        else:
            plt.show()

        plt.close()
