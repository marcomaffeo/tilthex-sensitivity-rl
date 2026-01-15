"""Tilthex-Sensitivity 
JAX implementation of Tilthex OpenAI gym environment with added
parametric sensitivity metric optimization."""

# LR, CLIP_EPS

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


@jax.jit
def skew(vec):
    return jnp.array(
        [[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]]
    )


@jax.jit
def quat2rotm(quat):
    eta = quat[0]
    vec = quat[1:4]
    return jnp.array(
        [
            [
                2 * (eta * eta + vec[0] * vec[0]) - 1,
                2 * (vec[0] * vec[1] - eta * vec[2]),
                2 * (vec[0] * vec[2] + eta * vec[1]),
            ],
            [
                2 * (vec[0] * vec[1] + eta * vec[2]),
                2 * (eta * eta + vec[1] * vec[1]) - 1,
                2 * (vec[1] * vec[2] - eta * vec[0]),
            ],
            [
                2 * (vec[0] * vec[2] - eta * vec[1]),
                2 * (vec[1] * vec[2] + eta * vec[0]),
                2 * (eta * eta + vec[2] * vec[2]) - 1,
            ],
        ]
    )


@jax.jit
def quat_product(quat1, quat2):
    vec1 = quat1[1:4]
    vec2 = quat2[1:4]

    return jnp.concatenate(
        [
            jnp.array([quat1[0] * quat2[0] - vec1.dot(vec2)]),
            quat1[0] * vec2 + quat2[0] * vec1 + skew(vec1) @ vec2,
        ]
    )


@jax.jit
def quat_inverse(q):
    """For unit quaternions"""
    res = jnp.array([q[0], -q[1], -q[2], -q[3]])
    return res


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
    target_attitude: jax.Array
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
    
    dt: float = struct.field(pytree_node=False, default=0.02)
    g: float = struct.field(pytree_node=False, default=9.81)
    n: float = struct.field(pytree_node=False, default=6)  # number of rotors
    arm_length : float = struct.field(pytree_node=False, default=0.39) # length of single arm
    c_tau: float = struct.field(pytree_node=False, default=2.4e-5) # squared propeller speed to drag torque coefficient [Nm/Hz^2]
    beta: float = struct.field(pytree_node=False, default=0.0) # beta tilting angle
    wmin: float = struct.field(pytree_node=False, default=16) # min prop rotation speed in Hz
    wmax: float = struct.field(pytree_node=False, default=100) # max prop rotation speed in Hz
    
    m: float = 3.072 #implicit pytree_node=True
    I: jax.Array = struct.field(pytree_node=True,default_factory=lambda: jnp.array([0.076276, 0.078072, 0.147915]),) # inertia diagonal [kg*m^2]
    com_default: jax.Array = struct.field(pytree_node=True,default_factory=lambda: jnp.array([0.0, 0.0, 0.0]),) # center of mass offset from geometric center [m]
    alpha: jax.Array = struct.field(pytree_node=True,default_factory=lambda: jnp.full((6,), -20.0)) # tilt_angle [deg] radial inclination angle of rotors
    c_f: float = struct.field(pytree_node=True, default=12.0e-4) # squared propeller speed to thrust force coefficient [N/hz^2]

    max_steps_in_episode: int = struct.field(pytree_node=False, default=200)
    is_testing: bool = struct.field(pytree_node=False, default=False)


class TilthexV1(environment.Environment):
    """JAX Compatible version of Tilthex-v1 target approaching."""

    def __init__(self):
        self.type = ""
        self.state_dim = 13
        self.obs_dim = 17 # 13 + 4 (quaternion to target)
        self.action_dim = self.action_space(self.default_params).shape[0]
        self.param_dim = 14  # len(jax.tree.leaves(self.default_params)): 1 (m) + 3 (I) + 3 (com_default) + 6(alpha) + 1 (c_f)
        self.target_pos = jnp.array([0.0, 0.0, 0.5]) 

        target_roll_deg = 0.0       # between -10 and 10 degrees
        target_pitch_deg = 0.0      # between -10 and 10 degrees
        target_yaw_deg = 0.0        # between -180 and 180 degrees

        target_rpy = jnp.deg2rad(jnp.array([target_roll_deg, target_pitch_deg, target_yaw_deg]))
        self.target_attitude = euler_to_quat(target_rpy)

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
        ]
        self.params_max = [
            3.2,                                            # mass
            jnp.array([0.079, 0.081, 0.156]),               # inertia
            jnp.array([0.02, 0.02, 0.02]),                  # com offset
            jnp.full((6,), -15.0),                          # alpha tilt angles
            jnp.array(12.5e-4),                             # c_f
        ]
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for Tilthex-v1."""
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

        state_next = self.step_dynamics(key, state, input, params)

        pos_diff = state_next.pos - env_state.target_pos

        q_w = state_next.attitude[0]
        reward_upright = -10 * (1.0 - (q_w )** 2)

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
            + reward_upright
            - 0.001 * (jnp.linalg.norm((thrust_normalized_input - self.hovering_action)) ** 2)
            - 0.005 * (jnp.linalg.norm(thrust_normalized_input - thrust_normalized_input_last) ** 2)
        )

        # lower entropy version
        # -0.001 * (jnp.linalg.norm((thrust_normalized_input - self.hovering_action)) ** 2)
        # -0.01 * (jnp.linalg.norm(thrust_normalized_input - thrust_normalized_input_last) ** 2)

        reward = reward.squeeze()
    
        env_state = EnvState(
            state=state_next,
            target_pos=env_state.target_pos,
            target_attitude=env_state.target_attitude,
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
        state: ModelState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> ModelState:
        """
        Tilthex dynamics model with CoM placed at the geometric center

        Parameters
        ----------
        state : jnp.array
            state vector [pos (world frame),
                        attitude (unit quaternion [w, x, y, z]),
                        vel (world frame),
                        angular_velocity (body frame)]
        inputs : jnp.array):
            input vector [actuator forces (f1, f2, f3, f4, f5, f6)]
        Returns
        -------
        state_next : ModelState
            next state after time dt
        """

        u = jnp.clip(action, params.min_input, params.max_input)
   
        quat = state.attitude
        ang_vel = state.omega
        
        orientation_mat = quat2rotm(quat)
        
        GF, GT = compute_allocation(6, params.arm_length, params.alpha, params.beta, params.c_f, params.c_tau, params.com_default)
        
        # Wrench in Body Frame
        F_body = GF @ u       # [Fx, Fy, Fz] 
        tau_body = GT @ u     # [Tx, Ty, Tz] 

        # Acceleration in Body Frame
        lin_acc_body = F_body / params.m
        J_vec = params.I
        gyro_term = jnp.cross(ang_vel, J_vec * ang_vel)
        ang_acc_body = (1.0 / J_vec) * (tau_body - gyro_term)
        acc = jnp.hstack([lin_acc_body, ang_acc_body]) 


        gravity_vec = jnp.array([0.0, 0.0, params.g]) # O params.g * e3
        ang_vel_quat = jnp.array([0.0, ang_vel[0], ang_vel[1], ang_vel[2]])

        pos = state.pos + params.dt * state.vel
        vel = state.vel + params.dt * (orientation_mat @ acc[:3] - gravity_vec)
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
            target_attitude=self.target_attitude,
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

        #target attitude error
        q_target_inv = quat_inverse(env_state.target_attitude)
        q_error = quat_product(q_target_inv, q)

        return jnp.concatenate(
            [
                to_target.reshape(-1),
                vel.reshape(-1),
                q.reshape(-1),
                ang_vel.reshape(-1),
                q_error.reshape(-1),
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
        return "Tilthex-v1"

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
                "target_attitude": spaces.Box(
                    -1.0, 1.0, (4,), 
                    jnp.float32
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
            output: Optional path to save the figure. If None, displays interactively.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        def quat_to_euler_deg(q_array):
            w, x, y, z = q_array[:, 0], q_array[:, 1], q_array[:, 2], q_array[:, 3]
            
            # Roll (x-axis rotation)
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            roll = np.arctan2(sinr_cosp, cosr_cosp)

            # Pitch (y-axis rotation)
            sinp = 2 * (w * y - z * x)
            # Clip per evitare errori numerici fuori da [-1, 1]
            sinp = np.clip(sinp, -1.0, 1.0)
            pitch = np.arcsin(sinp)
            # Yaw (z-axis rotation)
            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)

            return np.degrees(np.stack([roll, pitch, yaw], axis=1))
        
        plt.style.use("seaborn-v0_8")
        fig = plt.figure(figsize=(14, 12))
        gs = fig.add_gridspec(4, 3, height_ratios=[2, 1, 1, 1])

        ax_path = fig.add_subplot(gs[0, :], projection="3d")

        ax_m1 = fig.add_subplot(gs[1, 0])
        ax_m2 = fig.add_subplot(gs[1, 1])
        ax_m3 = fig.add_subplot(gs[1, 2])
        ax_m4 = fig.add_subplot(gs[2, 0])
        ax_m5 = fig.add_subplot(gs[2, 1]) 
        ax_m6 = fig.add_subplot(gs[2, 2])

        axes_act = [ax_m1, ax_m2, ax_m3, ax_m4, ax_m5, ax_m6]

        labels = ["Force p1", "Force p2", "Force p3", "Force p4", "Force p5", "Force p6"]

        ax_roll = fig.add_subplot(gs[3, 0])
        ax_pitch = fig.add_subplot(gs[3, 1])
        ax_yaw = fig.add_subplot(gs[3, 2])
        axes_att = [ax_roll, ax_pitch, ax_yaw]
        att_labels = ["Roll (x) [deg]", "Pitch (y) [deg]", "Yaw (z) [deg]"]

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

            quats = state[:, 3:7] 
            rpy_deg = quat_to_euler_deg(quats)

            for i, ax in enumerate(axes_att):
                ax.plot(np.arange(rpy_deg.shape[0]), rpy_deg[:, i], alpha=0.8)
                ax.set_xlabel("t")
                ax.set_ylabel(att_labels[i])
                ax.grid(True, alpha=0.3)

            actions = np.asarray(series.get("actions", []))
            if actions.size > 0:
                for idx, ax in enumerate(axes_act):
                    if idx < actions.shape[1]:
                        ax.plot(np.arange(actions.shape[0]), actions[:, idx], alpha=0.6)
                    ax.set_xlabel("t")
                    ax.set_ylabel(labels[idx])
                    ax.grid(True, alpha=0.3)

        ax_path.set_title("Tilthex position trajectories")
        ax_path.set_xlabel("x [m]")
        ax_path.set_ylabel("y [m]")
        ax_path.set_zlabel("z [m]")
        ax_path.grid(True, alpha=0.3)

        if output is None:
            plt.show()
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=150)
            plt.close(fig)

    #         @staticmethod
    # def plot_rollouts(series_list: List[dict], output: Path | None = None) -> None:
    #     """Plot tilthex rollout trajectories.
        
    #     Args:
    #         series_list: List of trajectory dictionaries with keys:
    #             - 'state': array of shape (T+1, 13) with [pos(3), attitude(4), vel(3), omega(3)]
    #             - 'actions': array of shape (T, 6)
    #             - 'target_pos': optional target position array
    #         output: Optional path to save the figure. If None, displays interactively.
    #     """
    #     import matplotlib.pyplot as plt
        
    #     plt.style.use("seaborn-v0_8")
    #     fig = plt.figure(figsize=(14, 8))
    #     gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1])

    #     ax_path = fig.add_subplot(gs[0, :], projection="3d")

    #     ax_m1 = fig.add_subplot(gs[1, 0])
    #     ax_m2 = fig.add_subplot(gs[1, 1])
    #     ax_m3 = fig.add_subplot(gs[1, 2])
    #     ax_m4 = fig.add_subplot(gs[2, 0])
    #     ax_m5 = fig.add_subplot(gs[2, 1]) 
    #     ax_m6 = fig.add_subplot(gs[2, 2])

    #     axes_act = [ax_m1, ax_m2, ax_m3, ax_m4, ax_m5, ax_m6]

    #     labels = ["Force p1", "Force p2", "Force p3", "Force p4", "Force p5", "Force p6"]

    #     for series in series_list:
    #         state = series.get("state", None)
    #         if state is None:
    #             continue
    #         state = np.asarray(state)
    #         # state format: [pos(3), attitude(4), vel(3), omega(3)]
    #         pos = state[:, :3]
            
    #         ax_path.plot(pos[:, 0], pos[:, 1], pos[:, 2], alpha=0.6)

    #         if "target_pos" in series:
    #             target = np.asarray(series["target_pos"])
    #             if target.ndim >= 2:
    #                 target = target[0]
    #             ax_path.scatter(
    #                 target[0], target[1], target[2], marker="x", color="red", s=30
    #             )

    #         ax_path.scatter(
    #             pos[0, 0], pos[0, 1], pos[0, 2], marker="o", color="green", s=30
    #         )

    #         actions = np.asarray(series.get("actions", []))
    #         if actions.size > 0:
    #             for idx, ax in enumerate(axes_act):
    #                 if idx < actions.shape[1]:
    #                     ax.plot(np.arange(actions.shape[0]), actions[:, idx], alpha=0.6)
    #                 ax.set_xlabel("t")
    #                 ax.set_ylabel(labels[idx])
    #                 ax.grid(True, alpha=0.3)

    #     ax_path.set_title("Tilthex position trajectories")
    #     ax_path.set_xlabel("x [m]")
    #     ax_path.set_ylabel("y [m]")
    #     ax_path.set_zlabel("z [m]")
    #     ax_path.grid(True, alpha=0.3)

    #     if output is None:
    #         plt.show()
    #     else:
    #         output.parent.mkdir(parents=True, exist_ok=True)
    #         fig.savefig(output, dpi=150)
    #         plt.close(fig)

