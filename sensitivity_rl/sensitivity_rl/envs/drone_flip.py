"""Drone V2 Environment.
JAX implementation of the Mikrokopter drone environment considering uncertainties on the mass, inertia,
position of the center of mass.
"""

from functools import partial
from typing import Any, List
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import environment, spaces
from gymnax.environments.environment import TEnvParams, TEnvState

from .math import quat2rotm, quat_product, skew, quat_inverse


@struct.dataclass
class ModelState:
    pos: jax.Array
    attitude: jax.Array  # (unit quaternion [w, x, y, z])
    vel: jax.Array
    omega: jax.Array


@struct.dataclass
class EnvState(environment.EnvState):
    state: ModelState
    target_pos: jax.Array
    target_yaw: jax.Array  # Desired yaw angle (rotation around z-axis)
    last_u: jax.Array  # Only needed for rendering
    time: int
    accumulated_flip_angle: jax.Array

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self.state, name)


@struct.dataclass
class EnvParams(environment.EnvParams):
    min_action: jax.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([-1.0, -1.0, -1.0, -1.0]),
    )
    max_action: jax.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([1.0, 1.0, 1.0, 1.0])
    )

    min_input: jax.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([4.0, -3.0, -3.0, -1.0]),
    )
    max_input: jax.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([25.0, 3.0, 3.0, 1.0])
    )
    env_max_limits: jax.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([1.0, 1.0, 1.0])
    )
    env_min_limits: jax.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([-1.0, -1.0, 0.0])
    )
    target_flips: int = struct.field(pytree_node=False, default=1)
    dt: float = struct.field(pytree_node=False, default=0.02)
    g: float = struct.field(pytree_node=False, default=9.81)
    m: float = 1.4
    I: jax.Array = struct.field(
        pytree_node=True,
        default_factory=lambda: jnp.array([0.022, 0.022, 0.020]),
    )
    com_pos: jax.Array = struct.field(
        pytree_node=True, default_factory=lambda: jnp.array([0.0, 0.0, 0.0])
    )
    delay: float = struct.field(pytree_node=True, default=0.002)

    max_steps_in_episode: int = struct.field(pytree_node=False, default=300)
    is_testing: bool = struct.field(pytree_node=False, default=False)


class DroneFlipV0(environment.Environment):
    """JAX Compatible version of drone to make flips."""

    def __init__(self):
        self.type = ""
        self.state_dim = 13
        self.obs_dim = 15  # CHANGED: Was 14 in drone_v2, added accumulated_flip_angle
        self.action_dim = self.action_space(self.default_params).shape[0]
        self.param_dim = 8  # len(jax.tree.leaves(self.default_params))
        self.target_pos = jnp.array([0.0, 0.0, 0.5])
        self.hovering_action = jnp.array(
            [
                self.default_params.m
                * self.default_params.g
                / self.default_params.max_input[0],
                0.0,
                0.0,
                0.0,
            ]
        )

        self.params_min = [
            1.0,  # mass
            jnp.array([0.01, 0.01, 0.01]),  # inertia
            jnp.array([-0.02, -0.02, -0.02]),  # com_pos
            0.002,  # delay
        ]
        self.params_max = [
            1.8,  # mas
            jnp.array([0.05, 0.05, 0.05]),  # inertia
            jnp.array([0.02, 0.02, 0.02]),  # com_pos
            0.015,  # delay
        ]
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for Drone-v2."""
        return EnvParams()

    def rescale_action(self, action: jax.Array, params: EnvParams) -> jax.Array:
        action_span = params.max_action - params.min_action
        input_span = params.max_input - params.min_input
        normalized = (action - params.min_action) / action_span
        return params.min_input + normalized * input_span

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

        action_clipped = jnp.clip(action, params.min_action, params.max_action)
        action_rescaled = self.rescale_action(action_clipped, params)

        state = env_state.state

        state_next = self.step_dynamics_delay(
            key, state, action_rescaled, env_state.last_u, params
        )

        target_angle = params.target_flips * 2.0 * jnp.pi
        delta_angle = state_next.omega[1] * params.dt
        new_accumulated_angle = env_state.accumulated_flip_angle + delta_angle

        # 1. FLIP PROGRESS [0.0 to 1.0]
        flip_ratio = jnp.clip(
            new_accumulated_angle / jnp.maximum(target_angle, 1e-5), 0.0, 1.0
        )

        # THE PHASE SHIFT MULTIPLIER (Maintains the exploit fix to prevent tumbling)
        angle_shortfall = jnp.maximum(0.0, target_angle - new_accumulated_angle)
        phase_stabilize = jnp.exp(-0.5 * jnp.square(angle_shortfall))
        stabilize = jnp.exp(-2.0 * jnp.square(angle_shortfall))

        # 2. DENSE FLIP STATE REWARD
        reward_flip = 15.0 * flip_ratio

        # 3. STRICTLY BOUNDED TRACKING REWARD
        pos_diff = state_next.pos - env_state.target_pos
        pos_error = jnp.linalg.norm(pos_diff)

        # A wide, strictly positive exponential.
        # It never drops below zero, providing a safe, guiding gradient from any distance.
        reward_pos = 50.0 * jnp.exp(-0.5 * pos_error) * stabilize

        # Dynamic velocity damping. Only acts as a heavy brake when approaching the center.
        vel_error = jnp.linalg.norm(state_next.vel)
        braking_zone = jnp.exp(-1.0 * pos_error)
        reward_vel = -2.0 * braking_zone * vel_error * stabilize

        # 4. EXPLICIT ATTITUDE STABILIZATION
        q = state_next.attitude
        upright_projection = 1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)
        reward_upright = 3.0 * jnp.exp(-0.5 * (1.0 - upright_projection)) * stabilize

        # 5. SMOOTHED YAW ALIGNMENT AND SPIN PENALTY
        current_yaw = jnp.arctan2(
            2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)
        )
        yaw_error = jnp.abs(
            jnp.arctan2(
                jnp.sin(current_yaw - env_state.target_yaw),
                jnp.cos(current_yaw - env_state.target_yaw),
            )
        )
        reward_yaw = 2.0 * jnp.exp(-0.5 * yaw_error) * stabilize

        # A unified spin penalty to calm the noisy thrashing
        omega_sq = jnp.square(state_next.omega)
        spin_penalty = -0.05 * jnp.sum(omega_sq) * stabilize

        # 6. SMOOTH CONTROL EFFORT
        thrust_normalized_input = action_rescaled.at[0].set(
            action_rescaled[0] / (params.max_input[0])
        )
        thrust_normalized_input_last = env_state.last_u.at[0].set(
            env_state.last_u[0] / (params.max_input[0])
        )
        ctrl_penalty = (
            -0.15
            * jnp.linalg.norm(thrust_normalized_input - thrust_normalized_input_last)
            ** 2
        )

        # 7. CRASH PENALTY
        crash_penalty = jax.lax.select(state_next.pos[2] < 0.35, -50.0, 0.0)

        # COMBINE REWARDS
        reward = (
            reward_flip
            + reward_pos
            + reward_vel
            + reward_upright
            + reward_yaw
            + spin_penalty
            + ctrl_penalty
            + crash_penalty
            + phase_stabilize * 5.0
        )

        reward = reward.squeeze()

        env_state = EnvState(
            state=state_next,
            target_pos=env_state.target_pos,
            target_yaw=env_state.target_yaw,
            last_u=action_rescaled.reshape(-1),
            time=env_state.time + 1,
            accumulated_flip_angle=new_accumulated_angle,
        )

        done = self.is_terminal(env_state, params)
        return (
            self.get_obs(env_state),
            env_state,
            reward,
            done,
            {"discount": self.discount(env_state, params)},
        )

    def step_dynamics_delay(
        self,
        key: jax.Array,
        state: ModelState,
        action: int | float | jax.Array,
        last_action: int | float | jax.Array,
        params: EnvParams,
    ) -> ModelState:
        """Step dynamics with input delay."""

        dt1 = jnp.clip(params.delay, 0.0, params.dt)
        dt2 = params.dt - dt1

        params = params.replace(dt=dt1)
        state_intermediate = self.step_dynamics(key, state, last_action, params)
        params = params.replace(dt=dt2)
        state_final = self.step_dynamics(key, state_intermediate, action, params)

        return state_final

    def step_dynamics(
        self,
        key: jax.Array,
        state: ModelState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> ModelState:

        u = jnp.clip(action, params.min_input, params.max_input)

        quat = state.attitude
        ang_vel = state.omega
        ang_vel_skew = skew(ang_vel)

        orientation_mat = quat2rotm(quat)

        total_force = (
            jnp.array([0.0, 0.0, u[0]])
            - params.m * params.g * orientation_mat[2, :]
            - params.m * ang_vel_skew @ ang_vel_skew @ params.com_pos
        )  # transpose + 3rd col = 3rd row

        total_torque = (
            u[1:4]
            - ang_vel_skew @ params.I @ ang_vel
            - params.m * params.g * skew(params.com_pos) @ orientation_mat[2, :]
        )

        # Spatial Inertia Matrix
        si_mat = jnp.diag(
            jnp.concatenate([params.m * jnp.ones(3, dtype=jnp.float32), params.I])
        )
        # Off diagonal terms due to CoM offset
        si_mat = si_mat.at[0:3, 3:6].set(-params.m * skew(params.com_pos))
        si_mat = si_mat.at[3:6, 0:3].set(params.m * skew(params.com_pos))

        acc = jnp.linalg.solve(si_mat, jnp.concatenate([total_force, total_torque]))

        # Semi-implicit Euler integration
        vel = state.vel + params.dt * orientation_mat @ acc[:3]
        omega = state.omega + params.dt * acc[3:6]

        pos = state.pos + params.dt * vel
        omega_quat = jnp.array([0.0, omega[0], omega[1], omega[2]])
        q = state.attitude + params.dt * 0.5 * quat_product(quat, omega_quat)

        q = q / jnp.linalg.norm(q)

        # clip position
        # pos = jnp.clip(pos, params.env_min_limits, params.env_max_limits)

        state_new = ModelState(
            pos=pos.squeeze(),
            attitude=q.squeeze(),
            vel=vel.squeeze(),
            omega=omega.squeeze(),
        )

        return state_new

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset environment state by sampling theta, theta_dot."""
        pos = jax.random.uniform(
            key,
            shape=(3,),
            minval=-0.1,
            maxval=0.1,
        )
        attitude = jnp.array([1.0, 0.0, 0.0, 0.0])
        vel = jnp.array([0.0, 0.0, 0.0])
        omega = jnp.array([0.0, 0.0, 0.0])
        state = ModelState(
            pos=self.target_pos + pos, attitude=attitude, vel=vel, omega=omega
        )

        target_pos = self.target_pos

        # Sample target yaw uniformly between -pi and pi
        target_yaw = jnp.array(0.0)

        env_state = EnvState(
            state=state,
            target_pos=target_pos,
            target_yaw=target_yaw,
            last_u=self.hovering_action,
            time=0,
            accumulated_flip_angle=jnp.array(0.0),
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

        # Compute current yaw and yaw error
        current_yaw = jnp.arctan2(
            2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)
        )
        yaw_error = jnp.arctan2(
            jnp.sin(current_yaw - env_state.target_yaw),
            jnp.cos(current_yaw - env_state.target_yaw),
        )

        return jnp.concatenate(
            [
                to_target.reshape(-1),
                vel.reshape(-1),
                q.reshape(-1),
                ang_vel.reshape(-1),
                jnp.array([yaw_error]),
                jnp.array([env_state.accumulated_flip_angle]),
            ]
        ).squeeze()

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = state.time >= params.max_steps_in_episode
        return jnp.array(done)

    def eval_metrics(
        self, env_state: EnvState, params: EnvParams
    ) -> dict[str, jax.Array]:
        """Compute evaluation metrics for the current state.

        Args:
            env_state: Current environment state
            params: Environment parameters

        Returns:
            Dictionary of metric names to scalar values
        """
        state = env_state.state

        # Distance to target
        distance_to_target = jnp.linalg.norm(state.pos - env_state.target_pos)

        # Velocity magnitude
        velocity_magnitude = jnp.linalg.norm(state.vel)

        # Angular velocity magnitude
        angular_velocity_magnitude = jnp.linalg.norm(state.omega)

        # Attitude error (deviation from upright quaternion [1, 0, 0, 0])
        upright_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
        quat_error = quat_product(quat_inverse(state.attitude), upright_quat)
        attitude_error = 2.0 * jnp.arccos(jnp.clip(jnp.abs(quat_error[0]), 0.0, 1.0))

        # Yaw error
        q = state.attitude
        current_yaw = jnp.arctan2(
            2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)
        )
        yaw_error = jnp.abs(
            jnp.arctan2(
                jnp.sin(current_yaw - env_state.target_yaw),
                jnp.cos(current_yaw - env_state.target_yaw),
            )
        )

        return {
            "distance_to_target": distance_to_target,
            "velocity_magnitude": velocity_magnitude,
            "angular_velocity_magnitude": angular_velocity_magnitude,
            "attitude_error": attitude_error,
            "yaw_error": yaw_error,
            "current_yaw": current_yaw,
            "target_yaw": env_state.target_yaw,
            "accumulated_flip_angle": env_state.accumulated_flip_angle,
        }

    @property
    def name(self) -> str:
        """Environment name."""
        return "Drone-v3"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 4

    def action_space(self, params: EnvParams | None = None) -> spaces.Box:
        """Action space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Box(
            low=params.min_action,
            high=params.max_action,
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

    @staticmethod
    def plot_rollouts(series_list: List[dict], output: Path | None = None) -> None:
        """Plot quadrotor rollout trajectories.

        Args:
            series_list: List of trajectory dictionaries with keys:
                - 'state': array of shape (T+1, 13) with [pos(3), attitude(4), vel(3), omega(3)]
                - 'actions': array of shape (T, 4)
                - 'target_pos': optional target position array
                - 'eval_metrics': optional dict with 'distance_to_target' array
            output: Optional path to save the figure. If None, displays interactively.
        """
        import matplotlib.pyplot as plt

        plt.style.use("seaborn-v0_8")
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(2, 4, height_ratios=[2, 1], width_ratios=[1, 1, 1, 1])

        ax_path = fig.add_subplot(gs[0, :2], projection="3d")
        ax_dist = fig.add_subplot(gs[0, 2])
        axes_act = [fig.add_subplot(gs[1, i]) for i in range(4)]
        ax_pitch = fig.add_subplot(gs[0, 3])

        labels = ["Thrust", "Torque X", "Torque Y", "Torque Z"]

        # Collect distance_to_target and yaw_error from all rollouts
        all_distances = []
        all_yaw_errors = []
        all_pitch_angles = []

        for series in series_list:
            state = series.get("state", None)
            if state is None:
                continue
            state = np.asarray(state)
            # state format: [pos(3), attitude(4), vel(3), omega(3)]
            pos = state[:, :3]

            ax_path.plot(pos[:, 0], pos[:, 1], pos[:, 2], alpha=0.6)

            if "target_pos" in series:
                target = np.asarray(series["target_pos"])
                if target.ndim >= 2:
                    target = target[0]
                ax_path.scatter(
                    target[0], target[1], target[2], marker="x", color="red", s=30
                )

            ax_path.scatter(
                pos[0, 0], pos[0, 1], pos[0, 2], marker="o", color="green", s=30
            )

            actions = np.asarray(series.get("actions", []))
            if actions.size > 0:
                for idx, ax in enumerate(axes_act):
                    if idx < actions.shape[1]:
                        ax.plot(np.arange(actions.shape[0]), actions[:, idx], alpha=0.6)
                    ax.set_xlabel("t")
                    ax.set_ylabel(labels[idx])
                    ax.grid(True, alpha=0.3)

            # Collect distance_to_target and yaw_error if available
            eval_metrics = series.get("eval_metrics", {})
            if "distance_to_target" in eval_metrics:
                dist = np.asarray(eval_metrics["distance_to_target"]).flatten()
                all_distances.append(dist)
            if "yaw_error" in eval_metrics:
                yaw = np.asarray(eval_metrics["yaw_error"]).flatten()
                all_yaw_errors.append(yaw)
            if "accumulated_flip_angle" in eval_metrics:
                pitch = np.asarray(eval_metrics["accumulated_flip_angle"]).flatten()
                all_pitch_angles.append(pitch)

        ax_path.set_title("Quadrotor position trajectories")
        ax_path.set_xlabel("x [m]")
        ax_path.set_ylabel("y [m]")
        ax_path.set_zlabel("z [m]")
        ax_path.grid(True, alpha=0.3)

        # Plot mean distance to target and yaw error with std shading
        if all_distances or all_yaw_errors:
            # Create twin axis for yaw error
            ax_yaw = ax_dist.twinx()

            # Plot distance to target
            if all_distances:
                max_len = max(d.shape[0] for d in all_distances)
                padded = np.array(
                    [
                        np.pad(d, (0, max_len - d.shape[0]), mode="edge")
                        for d in all_distances
                    ]
                )
                mean_dist = np.mean(padded, axis=0)
                std_dist = np.std(padded, axis=0)
                timesteps = np.arange(max_len)

                ax_dist.plot(
                    timesteps, mean_dist, color="blue", label="Distance", linewidth=2
                )
                ax_dist.fill_between(
                    timesteps,
                    mean_dist - std_dist,
                    mean_dist + std_dist,
                    alpha=0.3,
                    color="blue",
                )
                ax_dist.set_xlabel("Timestep")
                ax_dist.set_ylabel("Distance to target [m]", color="blue")
                ax_dist.tick_params(axis="y", labelcolor="blue")

                # Display final mean and std
                final_mean = mean_dist[-1]
                final_std = std_dist[-1]
                ax_dist.text(
                    0.02,
                    0.98,
                    f"Distance: {final_mean:.3f} ± {final_std:.3f} m",
                    transform=ax_dist.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
                )

            # Plot yaw error
            if all_yaw_errors:
                max_len_yaw = max(y.shape[0] for y in all_yaw_errors)
                padded_yaw = np.array(
                    [
                        np.pad(y, (0, max_len_yaw - y.shape[0]), mode="edge")
                        for y in all_yaw_errors
                    ]
                )
                mean_yaw = np.mean(padded_yaw, axis=0)
                std_yaw = np.std(padded_yaw, axis=0)
                timesteps_yaw = np.arange(max_len_yaw)

                ax_yaw.plot(
                    timesteps_yaw, mean_yaw, color="red", label="Yaw error", linewidth=2
                )
                ax_yaw.fill_between(
                    timesteps_yaw,
                    mean_yaw - std_yaw,
                    mean_yaw + std_yaw,
                    alpha=0.3,
                    color="red",
                )
                ax_yaw.set_ylabel("Yaw error [rad]", color="red")
                ax_yaw.tick_params(axis="y", labelcolor="red")

                # Display final yaw mean and std
                final_yaw_mean = mean_yaw[-1]
                final_yaw_std = std_yaw[-1]
                ax_dist.text(
                    0.02,
                    0.88,
                    f"Yaw error: {final_yaw_mean:.3f} ± {final_yaw_std:.3f} rad",
                    transform=ax_dist.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="lightsalmon", alpha=0.5),
                )

            ax_dist.set_title("Distance and Yaw Error over time")
            ax_dist.grid(True, alpha=0.3)

            # Add legend
            lines1, labels1 = ax_dist.get_legend_handles_labels()
            lines2, labels2 = ax_yaw.get_legend_handles_labels()
            ax_dist.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        else:
            ax_dist.text(
                0.5,
                0.5,
                "No eval_metrics available",
                ha="center",
                va="center",
                transform=ax_dist.transAxes,
            )
            ax_dist.set_title("Distance and Yaw Error over time")

        if all_pitch_angles:
            max_len_pitch = max(p.shape[0] for p in all_pitch_angles)
            padded_pitch = np.array(
                [
                    np.pad(p, (0, max_len_pitch - p.shape[0]), mode="edge")
                    for p in all_pitch_angles
                ]
            )
            mean_pitch = np.mean(padded_pitch, axis=0)
            std_pitch = np.std(padded_pitch, axis=0)
            timesteps_pitch = np.arange(max_len_pitch)

            ax_pitch.plot(
                timesteps_pitch,
                mean_pitch,
                color="purple",
                label="Accumulated Pitch",
                linewidth=2,
            )
            ax_pitch.fill_between(
                timesteps_pitch,
                mean_pitch - std_pitch,
                mean_pitch + std_pitch,
                alpha=0.3,
                color="purple",
            )

            # Add a dashed line at 2*pi (one full flip) to easily visualize completion
            ax_pitch.axhline(
                y=2 * np.pi,
                color="green",
                linestyle="--",
                alpha=0.7,
                label="1 Full Flip (2π rad)",
            )

            ax_pitch.set_title("Accumulated Pitch Angle (Flip Progress)")
            ax_pitch.set_xlabel("Timestep")
            ax_pitch.set_ylabel("Angle [rad]")
            ax_pitch.grid(True, alpha=0.3)
            ax_pitch.legend(loc="lower right")

        plt.tight_layout()

        if output is None:
            plt.show()
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=150)
            plt.close(fig)
