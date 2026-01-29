"""Tilthex-Sensitivity 
JAX implementation of Tilthex OpenAI gym environment with added
parametric sensitivity metric optimization."""

from functools import partial
from typing import Any, List
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from envs.base import Sensitivity
from flax import struct
from gymnax.environments import environment, spaces
from gymnax.environments.environment import TEnvParams, TEnvState
from utils import angle_normalize

from math import pi
import numpy as np
from numpy import matrix as mat

from .utils_allocation import *
from .utils_operations import *

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
    target_yaw: jax.Array # Desired yaw angle in radians
    last_u: jax.Array  # Only needed for rendering
    time: int

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self.state, name)


@struct.dataclass
class EnvParams(environment.EnvParams):
    min_action: jax.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
    )
    max_action: jax.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    )

    min_input: jax.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([0.3072, 0.3072, 0.3072, 0.3072, 0.3072, 0.3072]),
    )
    max_input: jax.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([12.0, 12.0, 12.0, 12.0, 12.0, 12.0]),
    )
    env_max_limits: jax.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([2.0, 2.0, 1.0])
    )
    env_min_limits: jax.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([-2.0, -2.0, 0.0])
    )

    max_target_yaw: float = struct.field(pytree_node=False, default=180.0)
    min_target_yaw: float = struct.field(pytree_node=False, default=-180.0)
    
    dt: float = struct.field(pytree_node=False, default=0.02)
    g: float = struct.field(pytree_node=False, default=9.81)
    n: float = struct.field(pytree_node=False, default=6)  # number of rotors
    arm_length : float = struct.field(pytree_node=False, default=0.39) # length of single arm
    c_tau: float = struct.field(pytree_node=False, default=2.4e-5) # squared propeller speed to drag torque coefficient [Nm/Hz^2]
    beta: float = struct.field(pytree_node=False, default=0.0) # beta tilting angle
    wmin: float = struct.field(pytree_node=False, default=16) # min prop rotation speed in Hz
    wmax: float = struct.field(pytree_node=False, default=100) # max prop rotation speed in Hz

    max_steps_in_episode: int = struct.field(pytree_node=False, default=200)
    is_testing: bool = struct.field(pytree_node=False, default=False)
  
    
    # Uncertain parameters
    m: float = 3.072 #implicit pytree_node=True
    I: jax.Array = struct.field(pytree_node=True,default_factory=lambda: jnp.array([0.076276, 0.078072, 0.147915]),) # inertia diagonal [kg*m^2]
    com_default: jax.Array = struct.field(pytree_node=True,default_factory=lambda: jnp.array([0.0, 0.0, 0.0]),) # center of mass offset from geometric center [m]
    alpha: jax.Array = struct.field(pytree_node=True,default_factory=lambda: jnp.full((6,), -20.0)) # tilt_angle [deg] radial inclination angle of rotors
    c_f: float = struct.field(pytree_node=True, default=12.0e-4) # squared propeller speed to thrust force coefficient [N/hz^2]
    delay: float = struct.field(pytree_node=True, default=0.002) # input delay [s]


class TilthexV0(environment.Environment):
    """JAX Compatible version of Tilthex-v0 target approaching."""

    def __init__(self):
        self.type = ""
        self.state_dim = 13
        self.obs_dim = 14 # 13 + 1 (yaw)
        self.action_dim = self.action_space(self.default_params).shape[0]
        self.param_dim = 15  # len(jax.tree.leaves(self.default_params)): 1 (m) + 3 (I) + 3 (com_default) + 6(alpha) + 1 (c_f) + 1 (delay)
        self.target_pos = jnp.array([0.0, 0.0, 0.5]) 

        # hovering action
        efficiency = jnp.cos(jnp.deg2rad(self.default_params.alpha))
        hover_force = (self.default_params.m * self.default_params.g) / (self.default_params.n * efficiency)
        self.hovering_action = jnp.full((self.default_params.n,), hover_force)

        self.params_min = [
            2.8,                                            # mass
            jnp.array([0.069, 0.071, 0.135]),               # inertia
            jnp.array([-0.02, -0.02, -0.02]),               # com offset
            jnp.full((6,), -25.0),                          # alpha tilt angles
            jnp.array(11.8e-4),                             # c_f
            0.002,                                          # delay
        ]
        self.params_max = [
            3.2,                                            # mass
            jnp.array([0.079, 0.081, 0.156]),               # inertia
            jnp.array([0.02, 0.02, 0.02]),                  # com offset
            jnp.full((6,), -15.0),                          # alpha tilt angles
            jnp.array(12.5e-4),                             # c_f
            0.004,                                          # delay
        ]
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for Tilthex-v0."""
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

        state_next = self.step_dynamics_delay(key, state, action_rescaled, env_state.last_u, params)

        pos_diff = state_next.pos - env_state.target_pos

        q = state_next.attitude
        current_yaw = jnp.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2]**2 + q[3]**2))
        
        yaw_error = jnp.abs(jnp.arctan2(jnp.sin(current_yaw - env_state.target_yaw), 
                                        jnp.cos(current_yaw - env_state.target_yaw)))

        reward_yaw = jax.lax.select(
            yaw_error < 0.1,
            1.0, 
            -1.0 * yaw_error,
        )
        
        reward_app = jax.lax.select(
            jnp.linalg.norm(pos_diff) < 0.02,
            10.0,  # reward when close to target
            1.0 * (-jnp.linalg.norm(pos_diff)),  # distance improvement reward
        )

        thrust_normalized_input = action_rescaled / params.max_input
        thrust_normalized_input_last = env_state.last_u / params.max_input

        reward = (
            -0.1 * jnp.linalg.norm(state_next.omega)
            + reward_app
            + reward_yaw
            - 0.001 * (jnp.linalg.norm((thrust_normalized_input - self.hovering_action)) ** 2)
            - 0.005 * (jnp.linalg.norm(thrust_normalized_input - thrust_normalized_input_last) ** 2)
        )

        reward = reward.squeeze()
    
        env_state = EnvState(
            state=state_next,
            target_pos=env_state.target_pos,
            target_yaw=env_state.target_yaw,
            last_u=action_rescaled.reshape(-1),
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
        """
        Tilthex dynamics model with CoM placed at the geometric center
        """

        u = jnp.clip(action, params.min_input, params.max_input)
   
        quat = state.attitude
        ang_vel = state.omega
        
        orientation_mat = quat2rotm(quat)
        
        GF, GT = compute_allocation(6, params.arm_length, params.alpha, params.beta, params.c_f, params.c_tau, params.com_default)
        
        F_body = GF @ u       # [Fx, Fy, Fz] 
        tau_body = GT @ u     # [Tx, Ty, Tz] 

        lin_acc_body = F_body / params.m
        J_vec = params.I
        gyro_term = jnp.cross(ang_vel, J_vec * ang_vel)
        ang_acc_body = (1.0 / J_vec) * (tau_body - gyro_term)
        acc = jnp.hstack([lin_acc_body, ang_acc_body]) 


        gravity_vec = jnp.array([0.0, 0.0, params.g]) 
        ang_vel_quat = jnp.array([0.0, ang_vel[0], ang_vel[1], ang_vel[2]])

        pos = state.pos + params.dt * state.vel
        vel = state.vel + params.dt * (orientation_mat @ acc[:3] - gravity_vec)
        q = state.attitude + params.dt * 0.5 * quat_product(quat, ang_vel_quat)
        omega = state.omega + params.dt * acc[3:6]

        q = q / jnp.linalg.norm(q) 
    
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

        key, key_target, key_yaw = jax.random.split(key, 3)
        target_pos = jax.lax.select(
            params.is_testing,
            self.target_pos,
            jax.random.uniform(
                key_target,
                shape=(3,),
                minval=params.env_min_limits,
                maxval=params.env_max_limits,
            ),
        )

        # Sample target yaw uniformly between -pi and pi
        target_yaw = jax.lax.select(
            params.is_testing,
            jnp.array(0.0),  # Point forward when testing
            jax.random.uniform(key_yaw, minval=-jnp.pi, maxval=jnp.pi),
        )

        env_state = EnvState(
            state=state,
            target_pos=target_pos,
            target_yaw=target_yaw,
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


        # Compute current yaw and yaw error
        current_yaw = jnp.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2]**2 + q[3]**2))
        yaw_error = jnp.arctan2(jnp.sin(current_yaw - env_state.target_yaw), 
                                jnp.cos(current_yaw - env_state.target_yaw))

        return jnp.concatenate(
            [
                to_target.reshape(-1),
                vel.reshape(-1),
                q.reshape(-1),
                ang_vel.reshape(-1),
                jnp.array([yaw_error]),
            ]
        ).squeeze()

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state is terminal."""
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
        
        # Yaw error
        q = state.attitude
        current_yaw = jnp.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2]**2 + q[3]**2))
        yaw_error = jnp.abs(jnp.arctan2(jnp.sin(current_yaw - env_state.target_yaw), 
                                         jnp.cos(current_yaw - env_state.target_yaw)))
        
        return {
            "distance_to_target": distance_to_target,
            "velocity_magnitude": velocity_magnitude,
            "angular_velocity_magnitude": angular_velocity_magnitude,
            "attitude_error": attitude_error,
            "yaw_error": yaw_error,
            "current_yaw": current_yaw,
            "target_yaw": env_state.target_yaw,
        }

    @property
    def name(self) -> str:
        """Environment name."""
        return "Tilthex-v0"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 6

    def action_space(self, params: EnvParams | None = None) -> spaces.Box:
        """Action space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Box(
            low=params.min_action,
            high=params.max_action,
            shape=(6,),
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
                    (6,),
                    jnp.float32,
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

    @staticmethod
    def plot_rollouts(series_list: List[dict], output: Path | None = None) -> None:
        """Plot tilthex rollout trajectories.
        
        Args:
            series_list: List of trajectory dictionaries with keys:
                - 'state': array of shape (T+1, 13) with [pos(3), attitude(4), vel(3), omega(3)]
                - 'actions': array of shape (T, 6)
                - 'target_pos': optional target position array
                - 'eval_metrics': optional dict with 'distance_to_target' array
            output: Optional path to save the figure. If None, displays interactively.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.style.use("seaborn-v0_8")
        fig = plt.figure(figsize=(24, 20))
        
        gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1], width_ratios=[1, 1, 1])

        ax_path = fig.add_subplot(gs[0:2, 0:2], projection="3d")
        ax_dist = fig.add_subplot(gs[0, 2])
        ax_yaw_err = fig.add_subplot(gs[1, 2])

        ax_m1 = fig.add_subplot(gs[2, 0])
        ax_m2 = fig.add_subplot(gs[2, 1])
        ax_m3 = fig.add_subplot(gs[2, 2])
        ax_m4 = fig.add_subplot(gs[3, 0])
        ax_m5 = fig.add_subplot(gs[3, 1]) 
        ax_m6 = fig.add_subplot(gs[3, 2])
        axes_act = [ax_m1, ax_m2, ax_m3, ax_m4, ax_m5, ax_m6]
        labels = ["Force p1 [N]", "Force p2 [N]", "Force p3 [N]", "Force p4 [N]", "Force p5 [N]", "Force p6 [N]"]

        # Collect data
        all_distances = []
        all_yaw_errors = []

        for series in series_list:
            state = series.get("state", None)
            if state is None:
                continue
            state = np.asarray(state)
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

            if "target_pos" in series:
                target = np.asarray(series["target_pos"])
                if target.ndim >= 2: target = target[0]
                dist = np.linalg.norm(pos - target, axis=1).flatten()
                all_distances.append(dist)

            if "target_yaw" in series:
                t_yaw = np.asarray(series["target_yaw"]).flatten()
                if t_yaw.size == 1:
                    t_yaw = np.full(pos.shape[0], t_yaw[0])
                
                q = state[:, 3:7] 
                w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
                current_yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
                
                diff = current_yaw - t_yaw
                yaw_error = np.abs(np.arctan2(np.sin(diff), np.cos(diff)))
                all_yaw_errors.append(yaw_error)

        ax_path.set_title("Tilthex position trajectories")
        ax_path.set_xlabel("x [m]")
        ax_path.set_ylabel("y [m]")
        ax_path.set_zlabel("z [m]")
        ax_path.grid(True, alpha=0.3)

        # Plot Error Pos
        if all_distances:
            max_len = max(d.shape[0] for d in all_distances)
            padded = np.array([
                np.pad(d, (0, max_len - d.shape[0]), mode='edge') 
                for d in all_distances
            ])
            mean_dist = np.mean(padded, axis=0)
            std_dist = np.std(padded, axis=0)
            timesteps = np.arange(max_len)

            ax_dist.plot(timesteps, mean_dist, color="blue", label="Mean Error", linewidth=2)
            ax_dist.fill_between(
                timesteps, mean_dist - std_dist, mean_dist + std_dist,
                alpha=0.3, color="blue"
            )
            ax_dist.set_title("Position Error [m]")
            ax_dist.set_xlabel("Timestep")
            ax_dist.set_ylabel("Error [m]")
            ax_dist.grid(True, alpha=0.3)
            
            final_mean = mean_dist[-1]
            final_std = std_dist[-1]
            ax_dist.text(
                0.5, 0.9, 
                f"Final: {final_mean:.3f} ± {final_std:.3f} m",
                transform=ax_dist.transAxes, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
            )
        else:
            ax_dist.text(0.5, 0.5, "No pos data", ha='center')

        # Plot Errore Yaw
        if all_yaw_errors:
            max_len_yaw = max(y.shape[0] for y in all_yaw_errors)
            padded_yaw = np.array([
                np.pad(y, (0, max_len_yaw - y.shape[0]), mode='edge') 
                for y in all_yaw_errors
            ])
            mean_yaw = np.mean(padded_yaw, axis=0)
            std_yaw = np.std(padded_yaw, axis=0)
            timesteps_yaw = np.arange(max_len_yaw)

            ax_yaw_err.plot(timesteps_yaw, mean_yaw, color="red", label="Mean Error", linewidth=2)
            ax_yaw_err.fill_between(
                timesteps_yaw, mean_yaw - std_yaw, mean_yaw + std_yaw,
                alpha=0.3, color="red"
            )
            ax_yaw_err.set_title("Yaw Error [rad]")
            ax_yaw_err.set_ylabel("Error [rad]")
            ax_yaw_err.set_xlabel("Timestep")
            ax_yaw_err.grid(True, alpha=0.3)
            
            final_yaw_mean = mean_yaw[-1]
            final_yaw_std = std_yaw[-1]
            ax_yaw_err.text(
                0.5, 0.9, 
                f"Final: {final_yaw_mean:.3f} ± {final_yaw_std:.3f} rad",
                transform=ax_yaw_err.transAxes, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.5)
            )
        else:
            ax_yaw_err.text(0.5, 0.5, "No yaw data", ha='center')

        plt.tight_layout()

        if output is None:
            plt.show()
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=150)
            plt.close(fig)