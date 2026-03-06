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

        state_next = self.step_dynamics(key, env_state, u, params)

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

    def eval_metrics(self, env_state: EnvState, params: EnvParams) -> dict[str, jax.Array]:
        """Compute evaluation metrics for the current state.
        
        Args:
            env_state: Current environment state
            params: Environment parameters
            
        Returns:
            Dictionary of metric names to scalar values
        """
        state = env_state.state
        
        # Forward velocity (x-velocity)
        forward_velocity = state.qvel.flatten()[0]
        
        # Height (z position of torso)
        height = state.qpos.flatten()[1]
        
        # Total velocity magnitude
        velocity_magnitude = jnp.linalg.norm(state.qvel)
        
        return {
            "forward_velocity": forward_velocity,
            "height": height,
            "velocity_magnitude": velocity_magnitude,
        }

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
        """Plot HalfCheetah rollout trajectories showing full body motion.
        
        Args:
            series_list: List of trajectory dictionaries containing qpos and qvel.
            output: Optional path to save the figure.
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm
        import numpy as np
        from mpl_toolkits.mplot3d import Axes3D
        
        # Load MuJoCo model to compute forward kinematics
        model_path = Path(__file__).parent / "assets" / "half_cheetah.xml"
        mj_model = mujoco.MjModel.from_xml_path(str(model_path))
        mj_data = mujoco.MjData(mj_model)
        
        fig = plt.figure(figsize=(18, 12))
        
        # Create 3D subplot for full body trajectory
        ax_3d = fig.add_subplot(2, 4, (1,4), projection='3d')
        
        # Create 2D subplots for mean velocities and states
        ax_vel_mean = fig.add_subplot(2, 4, (5,6))
        ax_states_mean = fig.add_subplot(2, 4, (7,8))
        
        colors = cm.viridis(np.linspace(0, 1, len(series_list)))
        
        # Collect all trajectory data for computing means and stds
        all_qpos = []  # List of (timesteps, 9) arrays
        all_qvel = []  # List of (timesteps, 9) arrays
        all_vel_x = []  # List of (timesteps,) arrays - forward velocity
        
        for idx, series in enumerate(series_list):
            # Extract state from series - it's stored as a flattened array
            # The state field contains concatenated qpos and qvel
            state_flat = series['state']  # Shape: (timesteps, state_dim)
                        
            # For HalfCheetah: state = [qpos (9), qvel (9)] = 18 total
            # Split into qpos and qvel
            qpos = state_flat[:, :9]
            qvel = state_flat[:, 9:18]
            
            timesteps = qpos.shape[0]
            steps = np.arange(timesteps)
            
            # Store data for mean/std computation
            all_qpos.append(qpos)
            all_qvel.append(qvel)
            all_vel_x.append(qvel[:, 0])
            
            # Compute body positions for all timesteps using forward kinematics
            body_positions = []  # Will store (timesteps, n_bodies, 3)
            
            for t in range(timesteps):
                mj_data.qpos[:] = qpos[t]
                mj_data.qvel[:] = qvel[t]
                mujoco.mj_forward(mj_model, mj_data)
                # xpos contains 3D positions of all bodies (n_bodies, 3)
                # Skip body 0 (world/floor) and only get actual robot bodies
                body_positions.append(mj_data.xpos[1:].copy())  # Skip world body
            
            body_positions = np.array(body_positions)  # (timesteps, n_bodies-1, 3)
            n_bodies = body_positions.shape[1]
            
            # Plot body parts as individual points with fading along trajectory
            # Show all bodies but with varying transparency
            for body_idx in range(n_bodies):
                x = body_positions[:, body_idx, 0]
                y = body_positions[:, body_idx, 1]
                z = body_positions[:, body_idx, 2]
                
                # Plot points with varying alpha - fade from transparent to opaque
                ax_3d.plot(x, y, z, color=colors[idx], alpha=0.2, linestyle='--',
                           label=f'Rollout {idx+1}' if (body_idx == 0) else '')
            
            
            # Compute joint positions at final timestep
            final_time = timesteps - 1
            mj_data.qpos[:] = qpos[final_time]
            mj_data.qvel[:] = qvel[final_time]
            mujoco.mj_forward(mj_model, mj_data)
            
            # Get joint anchor positions (xipos contains joint positions in world frame)
            # Each joint connects a parent body to a child body
            joint_positions = []
            for jnt_idx in range(mj_model.njnt):
                # Get the joint position in world coordinates
                joint_pos = mj_data.xanchor[jnt_idx].copy()
                joint_positions.append(joint_pos)
            joint_positions = np.array(joint_positions)  # (n_joints, 3)
            
            # Draw full skeleton at final position
            for body_idx in range(n_bodies):
                ax_3d.scatter([body_positions[final_time, body_idx, 0]], 
                             [body_positions[final_time, body_idx, 1]], 
                             [body_positions[final_time, body_idx, 2]], 
                             color=colors[idx], marker='o', s=100, alpha=0.2, 
                             edgecolors='black', linewidths=2,
                             label=f'Rollout {idx+1}' if body_idx == 0 else '')
            
            # For each body, draw line to parent joint (if exists)
            for body_idx in range(1, mj_model.nbody):  # Skip world body
                parent_id = mj_model.body_parentid[body_idx]
                if parent_id > 0:  # If parent is not world
                    # Find joint connecting this body to its parent
                    # body_jntadr gives the starting joint index for each body
                    # body_jntnum gives the number of joints for each body
                    if mj_model.body_jntnum[body_idx] > 0:
                        jnt_idx = mj_model.body_jntadr[body_idx]
                        
                        # Adjust body indices (skip world body)
                        child_body_idx = body_idx - 1
                        parent_body_idx = parent_id - 1
                        
                        # Draw from parent body to joint
                        ax_3d.plot(
                            [body_positions[final_time, parent_body_idx, 0], 
                             joint_positions[jnt_idx, 0]],
                            [body_positions[final_time, parent_body_idx, 1], 
                             joint_positions[jnt_idx, 1]],
                            [body_positions[final_time, parent_body_idx, 2], 
                             joint_positions[jnt_idx, 2]],
                            color=colors[idx], linewidth=3, alpha=0.2, linestyle='-'
                        )
                        
                        # Draw from joint to child body
                        ax_3d.plot(
                            [joint_positions[jnt_idx, 0],
                             body_positions[final_time, child_body_idx, 0]],
                            [joint_positions[jnt_idx, 1],
                             body_positions[final_time, child_body_idx, 1]],
                            [joint_positions[jnt_idx, 2],
                             body_positions[final_time, child_body_idx, 2]],
                            color=colors[idx], linewidth=3, alpha=0.2, linestyle='-'
                        )
        
        # Compute mean and std across all series
        # Ensure all trajectories have the same length (pad or truncate if needed)
        min_length = min(len(traj) for traj in all_vel_x)
        
        # Truncate all trajectories to minimum length
        vel_x_array = np.array([traj[:min_length] for traj in all_vel_x])  # (n_series, timesteps)
        qpos_array = np.array([qpos[:min_length] for qpos in all_qpos])  # (n_series, timesteps, 9)
        qvel_array = np.array([qvel[:min_length] for qvel in all_qvel])  # (n_series, timesteps, 9)
        
        steps = np.arange(min_length)
        
        # Compute mean and std for forward velocity (qvel[:, 0])
        vel_x_mean = np.mean(vel_x_array, axis=0)
        vel_x_std = np.std(vel_x_array, axis=0)
        
        # Plot mean forward velocity with std bands
        ax_vel_mean.plot(steps, vel_x_mean, color='blue', linewidth=2.5, 
                        label='Mean Forward Velocity', alpha=0.8)
        ax_vel_mean.fill_between(steps, vel_x_mean - vel_x_std, vel_x_mean + vel_x_std,
                                 color='blue', alpha=0.3, label='± 1 std')
        
        # Compute mean and std for all other states
        # qpos has 9 dimensions, qvel has 9 dimensions (but we skip qvel[:, 0])
        # So we'll plot: qpos (9) + qvel[:, 1:9] (8) = 17 total curves
        
        state_labels_qpos = ['qpos_x', 'qpos_z', 'qpos_rot', 'qpos_j1', 'qpos_j2', 
                            'qpos_j3', 'qpos_j4', 'qpos_j5', 'qpos_j6']
        state_labels_qvel = ['qvel_z', 'qvel_rot', 'qvel_j1', 'qvel_j2', 
                            'qvel_j3', 'qvel_j4', 'qvel_j5', 'qvel_j6']
        
        state_colors = cm.tab20(np.linspace(0, 1, 17))
        
        # Plot qpos dimensions
        for dim in range(9):
            state_mean = np.mean(qpos_array[:, :, dim], axis=0)
            state_std = np.std(qpos_array[:, :, dim], axis=0)
            ax_states_mean.plot(steps, state_mean, color=state_colors[dim], 
                              linewidth=1.5, label=state_labels_qpos[dim], alpha=0.7)
            ax_states_mean.fill_between(steps, state_mean - state_std, state_mean + state_std,
                                       color=state_colors[dim], alpha=0.2)
        
        # Plot qvel dimensions (skip first one - forward velocity)
        for dim in range(1, 9):
            state_mean = np.mean(qvel_array[:, :, dim], axis=0)
            state_std = np.std(qvel_array[:, :, dim], axis=0)
            ax_states_mean.plot(steps, state_mean, color=state_colors[8 + dim], 
                              linewidth=1.5, label=state_labels_qvel[dim-1], alpha=0.7)
            ax_states_mean.fill_between(steps, state_mean - state_std, state_mean + state_std,
                                       color=state_colors[8 + dim], alpha=0.2)
        
        # Remove old individual velocity plotting code
        # (The code below this point for vel_x, vel_z, vel_rot, etc. is no longer needed)
        
        # Configure 3D plot
        ax_3d.set_xlabel('X Position (Forward)', fontsize=11, fontweight='bold')
        ax_3d.set_ylabel('Y Position (Lateral)', fontsize=11, fontweight='bold')
        ax_3d.set_zlabel('Z Position (Height)', fontsize=11, fontweight='bold')
        ax_3d.set_title('HalfCheetah Full Body 3D Motion\n(trajectories: faded paths, final pose: solid)', 
                       fontsize=13, fontweight='bold')
        ax_3d.grid(True, alpha=0.3)
        ax_3d.view_init(elev=15, azim=45)
        
        # Configure forward velocity mean plot
        ax_vel_mean.set_xlabel('Step', fontsize=11, fontweight='bold')
        ax_vel_mean.set_ylabel('Forward Velocity', fontsize=11, fontweight='bold')
        ax_vel_mean.set_title('Mean Forward Velocity Across All Rollouts', 
                            fontsize=12, fontweight='bold')
        ax_vel_mean.legend(fontsize=9)
        ax_vel_mean.grid(True, alpha=0.3)
        
        # Configure other states mean plot
        ax_states_mean.set_xlabel('Step', fontsize=11, fontweight='bold')
        ax_states_mean.set_ylabel('State Value', fontsize=11, fontweight='bold')
        ax_states_mean.set_title('Mean of All Other States (Position & Velocity Components)', 
                            fontsize=12, fontweight='bold')
        ax_states_mean.legend(fontsize=7, ncol=3, loc='best')
        ax_states_mean.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output is not None:
            plt.savefig(output, dpi=150, bbox_inches='tight')
            print(f"[plot_rollouts] Figure saved to {output}")
        else:
            plt.show()
        
        plt.close()
