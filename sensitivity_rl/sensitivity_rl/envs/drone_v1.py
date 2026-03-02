"""Drone V1 Environment.
JAX implementation of the Mikrokopter drone environment.
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
    last_u: jax.Array  # Only needed for rendering
    time: int

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
    dt: float = struct.field(pytree_node=False, default=0.02)
    g: float = struct.field(pytree_node=False, default=9.81)
    m: float = 1.4
    I: jax.Array = struct.field(
        pytree_node=True,
        default_factory=lambda: jnp.array([0.022, 0.022, 0.020]),
    )
    max_steps_in_episode: int = struct.field(pytree_node=False, default=200)
    is_testing: bool = struct.field(pytree_node=False, default=False)


class DroneV1(environment.Environment):
    """JAX Compatible version of mikrokopter drone converging to a target."""

    def __init__(self):
        self.type = ""
        self.state_dim = 13
        self.obs_dim = 13
        self.action_dim = self.action_space(self.default_params).shape[0]
        self.param_dim = 4  # len(jax.tree.leaves(self.default_params))
        self.target_pos = jnp.array([0.0, 0.0, 0.5])
        self.hovering_action = jnp.array(
            [self.default_params.m * self.default_params.g / self.default_params.max_input[0], 0.0, 0.0, 0.0]
        )

        self.params_min = [
            1.0,  # mass
            jnp.array([0.01, 0.01, 0.01])  # inertia
        ]
        self.params_max = [
            1.8,  # mass
            jnp.array([0.05, 0.05, 0.05])  # inertia
        ]
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for Drone-v1."""
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

        clipped_action = jnp.clip(action, params.min_action, params.max_action)
        input = self.rescale_action(clipped_action, params)

        state = env_state.state

        state_next = self.step_dynamics(key, env_state, input, params)

        pos_diff = state_next.pos - env_state.target_pos

        reward_app = jax.lax.select(
            jnp.linalg.norm(pos_diff) < 0.02,
            10.0,  # reward when close to target
            1.0 * (-jnp.linalg.norm(pos_diff)),  # distance improvement reward
        )

        thrust_normalized_input = input.at[0].set(input[0] / (params.max_input[0]))
        thrust_normalized_input_last = env_state.last_u.at[0].set(env_state.last_u[0] / (params.max_input[0]))

        reward = (
            -0.1 * jnp.linalg.norm(state_next.omega)
            + reward_app
            - 0.01 * (jnp.linalg.norm((thrust_normalized_input - self.hovering_action)) ** 2)
            - 0.05 * (jnp.linalg.norm(thrust_normalized_input - thrust_normalized_input_last) ** 2)
        )

        reward = reward.squeeze()

        env_state = EnvState(
            state=state_next,
            target_pos=env_state.target_pos,
            last_u=input.reshape(-1),
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

        u = jnp.clip(action, params.min_input, params.max_input)

        quat = state.attitude
        ang_vel = state.omega

        orientation_mat = quat2rotm(quat)
        ang_vel_quat = jnp.array([0.0, ang_vel[0], ang_vel[1], ang_vel[2]])

        total_force = (
            jnp.array([0.0, 0.0, u[0]]) - params.m * params.g * orientation_mat[2, :]
        )  # transpose + 3rd col = 3rd row

        total_torque = (
            u[1:4] - skew(ang_vel) @ params.I @ ang_vel
        ) 

        si_mat = jnp.diag(
            jnp.concatenate([params.m * jnp.ones(3, dtype=jnp.float32), params.I])
        )
        si_mat_inv = jnp.linalg.inv(si_mat)

        si_mat = jnp.diag(
            jnp.concatenate([params.m * jnp.ones(3, dtype=jnp.float32), params.I])
        )
        acc = si_mat_inv @ jnp.concatenate([total_force, total_torque])

        pos = state.pos + params.dt * state.vel
        vel = state.vel + params.dt * orientation_mat @ acc[:3]
        q = state.attitude + params.dt * 0.5 * quat_product(quat, ang_vel_quat)
        omega = state.omega + params.dt * acc[3:6]

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

    def eval_metrics(self, env_state: EnvState, params: EnvParams) -> dict[str, jax.Array]:
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
        
        return {
            "distance_to_target": distance_to_target,
            "velocity_magnitude": velocity_magnitude,
            "angular_velocity_magnitude": angular_velocity_magnitude,
            "attitude_error": attitude_error,
        }

    @property
    def name(self) -> str:
        """Environment name."""
        return "Drone-v1"

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
        ax_dist = fig.add_subplot(gs[0, 2:])
        axes_act = [fig.add_subplot(gs[1, i]) for i in range(4)]

        labels = ["Thrust", "Torque X", "Torque Y", "Torque Z"]

        # Collect distance_to_target from all rollouts
        all_distances = []

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

            # Collect distance_to_target if available
            eval_metrics = series.get("eval_metrics", {})
            if "distance_to_target" in eval_metrics:
                dist = np.asarray(eval_metrics["distance_to_target"]).flatten()
                all_distances.append(dist)

        ax_path.set_title("Quadrotor position trajectories")
        ax_path.set_xlabel("x [m]")
        ax_path.set_ylabel("y [m]")
        ax_path.set_zlabel("z [m]")
        ax_path.grid(True, alpha=0.3)

        # Plot mean distance to target with std shading
        if all_distances:
            # Pad to same length if needed
            max_len = max(d.shape[0] for d in all_distances)
            padded = np.array([
                np.pad(d, (0, max_len - d.shape[0]), mode='edge') 
                for d in all_distances
            ])
            mean_dist = np.mean(padded, axis=0)
            std_dist = np.std(padded, axis=0)
            timesteps = np.arange(max_len)

            ax_dist.plot(timesteps, mean_dist, color="blue", label="Mean distance")
            ax_dist.fill_between(
                timesteps,
                mean_dist - std_dist,
                mean_dist + std_dist,
                alpha=0.3,
                color="blue",
                label="± 1 std"
            )
            ax_dist.set_xlabel("Timestep")
            ax_dist.set_ylabel("Distance to target [m]")
            ax_dist.set_title("Distance to target over time")
            ax_dist.legend(loc="upper right")
            ax_dist.grid(True, alpha=0.3)
            
            # Display final mean and std
            final_mean = mean_dist[-1]
            final_std = std_dist[-1]
            ax_dist.text(
                0.02, 0.98, 
                f"Final: {final_mean:.3f} ± {final_std:.3f} m",
                transform=ax_dist.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
        else:
            ax_dist.text(0.5, 0.5, "No eval_metrics available", 
                        ha='center', va='center', transform=ax_dist.transAxes)
            ax_dist.set_title("Distance to target over time")

        plt.tight_layout()

        if output is None:
            plt.show()
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=150)
            plt.close(fig)
