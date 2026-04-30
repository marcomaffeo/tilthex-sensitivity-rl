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
    target_attitude: jax.Array
    u_optimal_ref: jax.Array  # optimal action for dynamic hovering
    last_u: jax.Array  # Only needed for rendering
    time: int

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self.state, name)


@struct.dataclass
class EnvParams(environment.EnvParams):
    min_action: jax.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
    )
    max_action: jax.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    )
    min_input: jax.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([0.3072, 0.3072, 0.3072, 0.3072, 0.3072, 0.3072]), # F = c_f*wmin^2
    )
    max_input: jax.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([12.0, 12.0, 12.0, 12.0, 12.0, 12.0]),             # F = c_f*wmax^2
    )
    env_max_limits: jax.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([2.0, 2.0, 2.0])
    )
    env_min_limits: jax.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([-2.0, -2.0, 0.0])
    )

    min_target_roll: float = struct.field(pytree_node=False, default=-10.0)
    min_target_pitch: float = struct.field(pytree_node=False, default=-10.0)
    min_target_yaw: float = struct.field(pytree_node=False, default=-180.0)

    max_target_roll: float = struct.field(pytree_node=False, default=10.0)
    max_target_pitch: float = struct.field(pytree_node=False, default=10.0)
    max_target_yaw: float = struct.field(pytree_node=False, default=180.0)

    dt: float = struct.field(pytree_node=False, default=0.02)
    g: float = struct.field(pytree_node=False, default=9.81)
    n: float = struct.field(pytree_node=False, default=6)                                                                                       # number of rotors
    arm_length : float = struct.field(pytree_node=False, default=0.39)                                                                          # length of single arm
    c_tau: float = struct.field(pytree_node=False, default=2.4e-5)                                                                              # squared propeller speed to drag torque coefficient [Nm/Hz^2]
    beta: float = struct.field(pytree_node=False, default=0.0)                                                                                  # beta tilting angle
    wmin: float = struct.field(pytree_node=False, default=16)                                                                                   # min prop rotation speed in Hz
    wmax: float = struct.field(pytree_node=False, default=100)                                                                                  # max prop rotation speed in Hz
    I_off_diag: jax.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([0.00175324, 1.3135e-06, -5.1512e-05]))           # inertia off-diagonal terms [kg*m^2]
    action_noise_scale: float = struct.field(pytree_node=False, default=0.03)                                                                   # action noise scale for exploration
    obs_noise_stds: jax.Array = struct.field(
        pytree_node=False, 
        default_factory=lambda: jnp.array([
            0.007124, 0.006210, 0.001297,  # Position (x, y, z)
            0.024369, 0.014837, 0.005416,  # Velocity (vx, vy, vz)
            0.002755, 0.004249, 0.002538, 0.003132,  # Quaternion (w, x, y, z)
            0.054904, 0.078569, 0.031948,  # Angular Vel (wx, wy, wz)
            0.002755, 0.004249, 0.002538, 0.003132   # Quaternion Target Error
        ])
    )                                                                                   # observation noise standard deviations for each state component

    gain_stab: float = struct.field(pytree_node=False, default=0.2)             # reward gain for angular velocity penalty to encourage stability
    gain_att: float = struct.field(pytree_node=False, default=2.0)              # reward gain for attitude error to encourage correct orientation
    gain_ref: float = struct.field(pytree_node=False, default=1.0)              # reward gain to encourage following the ideal action for hovering at target
    gain_sm: float = struct.field(pytree_node=False, default=2.0)              # reward gain to encourage smoother actions and reduce chattering
    reward_pos_tolerance: float = struct.field(pytree_node=False, default=0.04)

    # Uncertain parameters
    m: float = 2.936                                                                                                                            # mass [kg]
    I: jax.Array = struct.field(pytree_node=True,default_factory=lambda: jnp.array([0.082529, 0.0846092, 0.154814]),)                           # inertia diagonal [kg*m^2]
    com_default: jax.Array = struct.field(pytree_node=True,default_factory=lambda: jnp.array([0.000122973, 0.00013006, 0.0248586]),)            # center of mass offset from geometric center [m]
    alpha: jax.Array = struct.field(pytree_node=True,default_factory=lambda: jnp.full((6,), -20.0))                                             # tilt_angle [deg] radial inclination angle of rotors
    c_f: float = struct.field(pytree_node=True, default=12.0e-4)                                                                                # squared propeller speed to thrust force coefficient [N/hz^2]
    delay: float = struct.field(pytree_node=True, default=0.002)                                                                                # actuation delay [s]

    max_steps_in_episode: int = struct.field(pytree_node=False, default=200)
    is_testing: bool = struct.field(pytree_node=False, default=False)


class TilthexV1(environment.Environment):
    """JAX Compatible version of Tilthex-v1 target approaching."""

    def __init__(self):
        self.type = ""
        self.state_dim = 13
        self.obs_dim = 17 # 13 + 4 (quaternion to target)
        self.action_dim = self.action_space(self.default_params).shape[0]
        self.param_dim = 15  # len(jax.tree.leaves(self.default_params)): 1 (m) + 3 (I) + 3 (com_default) + 6(alpha) + 1 (c_f) + 1 (delay)
        self.target_pos = jnp.array([0.0, 0.0, 1.0]) 
        self.target_attitude = euler_to_quat(jnp.deg2rad(jnp.array([0.0, 0.0, 0.0])))

        # hovering action
        efficiency = jnp.cos(jnp.deg2rad(self.default_params.alpha))
        hover_force = (self.default_params.m * self.default_params.g) / (self.default_params.n * efficiency)
        self.hovering_action = jnp.full((self.default_params.n,), hover_force)

        self.params_min = [
            2.6,                                            # mass -10%
            jnp.array([0.074, 0.076, 0.14]),                # inertia -10%
            jnp.array([0.0, 0.0, 0.01]),                    # com offset [-110%, -110%, -10%]
            jnp.full((6,), -22.0),                          # alpha tilt angles -2deg
            jnp.array(1.08e-3),                             # c_f -10%
            0.001,                                          # delay 
        ]
        self.params_max = [
            3.2,                                            # mass +10%
            jnp.array([0.090, 0.093, 0.17]),                # inertia +10%
            jnp.array([0.001, 0.001, 0.04]),                # com offset [+110%, +110%, +10%]
            jnp.full((6,), -18.0),                          # alpha tilt angles +2deg
            jnp.array(1.32e-3),                             # c_f +10%
            0.01,                                           # delay
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

    def solve_allocation_analytical(self, W_des: jax.Array, M: jax.Array, params: EnvParams) -> jax.Array:
        """Solve Mu = W_des using Least Squares"""
        M_pinv = jnp.linalg.pinv(M)
        u = M_pinv @ W_des
        u_clamped = jnp.clip(u, params.min_input, params.max_input)
        
        return u_clamped
    
    def compute_optimal_u_for_target(self, target_quat: jax.Array, params: EnvParams) -> jax.Array:
        """Calculate the ideal motor input to maintain balance at target_quat"""
        R_target = quat2rotm(target_quat)
        g_world = jnp.array([0.0, 0.0, params.m * params.g])
        F_body_req = R_target.T @ g_world
        W_des = jnp.hstack([F_body_req, jnp.array([0.0, 0.0, 0.0])])
        
        GF, GT = compute_allocation(
            params.n, params.arm_length, params.alpha, 
            params.beta, params.c_f, params.c_tau, params.com_default
        )
        M = jnp.vstack([GF, GT])
        
        return self.solve_allocation_analytical(W_des, M, params)

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

        key_step, key_noise, key_obs = jax.random.split(key, 3)

        action_clipped = jnp.clip(action, params.min_action, params.max_action)
        action_rescaled = self.rescale_action(action_clipped, params)

        noise = jax.random.normal(key_noise, shape=action_rescaled.shape) * params.action_noise_scale
        action_noisy = jnp.clip(action_rescaled + noise, params.min_input, params.max_input)

        state = env_state.state
        state_next = self.step_dynamics(key_step, env_state, action_noisy, params)

        pos_diff = state_next.pos - env_state.target_pos

        q_state_inv = quat_inverse(state_next.attitude)
        q_error_body = quat_product(q_state_inv, env_state.target_attitude)
        q_error_body = jax.lax.select(q_error_body[0] < 0.0, -q_error_body, q_error_body)

        q_err_w = q_error_body[0]
        q_err_vec = jnp.linalg.norm(q_error_body[1:])
        attitude_error_rad = 2.0 * jnp.arctan2(q_err_vec, q_err_w)

        reward_orientation = params.gain_att * jnp.exp(-1.0 * attitude_error_rad)   
        reward_app = jax.lax.select(
            jnp.linalg.norm(pos_diff) < params.reward_pos_tolerance,
            10.0,                                                       # reward when close to target
            1.0 * (-jnp.linalg.norm(pos_diff)),                         # distance improvement reward
        )

        thrust_normalized_input = action_rescaled / params.max_input
        thrust_normalized_input_last = env_state.last_u / params.max_input

        reward = (
            - params.gain_stab * jnp.linalg.norm(state_next.omega)
            + reward_app
            + reward_orientation
            - params.gain_ref * jnp.linalg.norm(thrust_normalized_input - (env_state.u_optimal_ref / params.max_input)) ** 2
            - params.gain_sm * (jnp.linalg.norm(thrust_normalized_input - thrust_normalized_input_last) ** 2)
        )

        reward = reward.squeeze()
    
        env_state = EnvState(
            state=state_next,
            target_pos=env_state.target_pos,
            target_attitude=env_state.target_attitude,
            u_optimal_ref=env_state.u_optimal_ref,
            last_u=action_rescaled.reshape(-1),
            time=env_state.time + 1,
        )

        done = self.is_terminal(env_state, params)
        return (
            self.get_obs(env_state, key_obs, params),
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
        """Step dynamics handling input delay via intermediate integration."""
        
        state = env_state.state
        last_action = env_state.last_u
        
        dt1 = jnp.clip(params.delay, 0.0, params.dt)
        dt2 = params.dt - dt1

        params = params.replace(dt=dt1)
        state_intermediate = self._step_dynamics(key, state, last_action, params)
        params = params.replace(dt=dt2)
        state_final = self._step_dynamics(key, state_intermediate, action, params)

        return state_final

    def _step_dynamics(
        self,
        key: jax.Array,
        state: ModelState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> ModelState:
        """Core tilthex dynamics model with CoM placed at the geometric center."""

        u = jnp.clip(action, params.min_input, params.max_input)
   
        quat = state.attitude
        ang_vel = state.omega
        
        orientation_mat = quat2rotm(quat)
        
        GF, GT = compute_allocation(6, params.arm_length, params.alpha, params.beta, params.c_f, params.c_tau, params.com_default)
        
        F_body = GF @ u       
        tau_body = GT @ u     

        lin_acc_body = F_body / params.m

        ixy, ixz, iyz = params.I_off_diag
        J_mat = jnp.array([[params.I[0], ixy, ixz],[ixy, params.I[1], iyz],[ixz, iyz, params.I[2]]])

        gyro_term = jnp.cross(ang_vel, J_mat @ ang_vel)
        ang_acc_body = jnp.linalg.solve(J_mat, tau_body - gyro_term)
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
        """Reset environment state by sampling theta, theta_dot.
            key_pos: drone initial position
            key_att: drone initial attitude
            key_vel: drone initial velocity
            key_omega: drone initial angular velocity
            key_target_pos: target position
            key_target_att: target attitude
        """
        key, key_pos, key_att, key_vel, key_omega, key_target_pos, key_target_att, key_obs = jax.random.split(key, 8)
        
        pos = jax.random.uniform(
            key_pos,
            shape=(3,),
            minval=params.env_min_limits,
            maxval=params.env_max_limits,
        )

        min_rpy_rad = jnp.deg2rad(jnp.array([-2.0, -2.0, -180.0])) 
        max_rpy_rad = jnp.deg2rad(jnp.array([2.0, 2.0, 180.0]))

        att_rad = jax.random.uniform(
            key_att,
            shape=(3,),
            minval=min_rpy_rad,
            maxval=max_rpy_rad,
        )
        attitude = euler_to_quat(att_rad)

        vel = jax.random.uniform(key_vel, (3,), minval=-0.1, maxval=0.1) 
        omega = jax.random.uniform(key_omega, (3,), minval=-0.2, maxval=0.2)
        state = ModelState(pos=pos, attitude=attitude, vel=vel, omega=omega)

        target_pos = jax.lax.select(
            params.is_testing,
            self.target_pos,
            jax.random.uniform(
                key_target_pos,
                shape=(3,),
                minval=params.env_min_limits,
                maxval=params.env_max_limits,
            ),
        )

        min_target_rpy_rad = jnp.deg2rad(jnp.array([params.min_target_roll, params.min_target_pitch, params.min_target_yaw]))
        max_target_rpy_rad = jnp.deg2rad(jnp.array([params.max_target_roll, params.max_target_pitch, params.max_target_yaw]))

        target_att_rad_rnd = jax.random.uniform(key_target_att, (3,), minval=min_target_rpy_rad, maxval=max_target_rpy_rad)
        target_attitude = jax.lax.select(params.is_testing, self.target_attitude, euler_to_quat(target_att_rad_rnd))

        u_optimal_ref = self.compute_optimal_u_for_target(target_attitude, params) 

        env_state = EnvState(
            state=state,
            target_pos=target_pos,
            target_attitude=target_attitude,
            u_optimal_ref=u_optimal_ref,
            last_u=self.hovering_action,
            time=jnp.array(0, dtype=jnp.int32),
        )
        return self.get_obs(env_state, key_obs, params), env_state

    def get_obs(self, env_state: EnvState, key: jax.Array, params: EnvParams) -> jax.Array:
        """Return a vector of observations from the state."""
        state = env_state.state

        pos = state.pos
        vel_world = state.vel
        to_target_world = env_state.target_pos - pos
        q = state.attitude
        ang_vel = state.omega

        rot_mat = quat2rotm(q)
        to_target_body = rot_mat.T @ to_target_world
        vel_body = rot_mat.T @ vel_world

        q_state_inv = quat_inverse(q)
        q_error_body = quat_product(q_state_inv, env_state.target_attitude)

        q_error_body = jax.lax.select(
            q_error_body[0] < 0.0,
            -q_error_body,
            q_error_body
        )

        obs_clean = jnp.concatenate(
            [
                to_target_body.reshape(-1),
                vel_body.reshape(-1),
                q.reshape(-1),
                ang_vel.reshape(-1),
                q_error_body.reshape(-1),
            ]
        ).squeeze()

        # Add noise to the observations
        obs_noisy = obs_clean + jax.random.normal(key, shape=obs_clean.shape) * params.obs_noise_stds

        return obs_noisy

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state is terminal."""
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
                - 'target_attitude': optional target quaternion array
            output: Optional path to save the figure. If None, displays interactively.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        plt.style.use("seaborn-v0_8")
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(4, 3, height_ratios=[2, 2, 1, 1], width_ratios=[1, 1, 1])

        ax_path = fig.add_subplot(gs[0:2, 0], projection="3d")

        ax_dist_err = fig.add_subplot(gs[0, 1:3])
        ax_att_err = fig.add_subplot(gs[1, 1:3])

        ax_m1 = fig.add_subplot(gs[2, 0])
        ax_m2 = fig.add_subplot(gs[2, 1])
        ax_m3 = fig.add_subplot(gs[2, 2])
        ax_m4 = fig.add_subplot(gs[3, 0])
        ax_m5 = fig.add_subplot(gs[3, 1]) 
        ax_m6 = fig.add_subplot(gs[3, 2])
        axes_act = [ax_m1, ax_m2, ax_m3, ax_m4, ax_m5, ax_m6]
        labels = ["Force p1 [N]", "Force p2 [N]", "Force p3 [N]", "Force p4 [N]", "Force p5 [N]", "Force p6 [N]"]

        # Collect data
        all_dist_errors = []
        all_att_errors = []

        for series in series_list:
            state = series.get("state", None)
            if state is None:
                continue
            state = np.asarray(state) 
            
            pos = state[:, :3]
            quats = state[:, 3:7]

            ax_path.plot(pos[:, 0], pos[:, 1], pos[:, 2], alpha=0.6)
            ax_path.scatter(pos[0, 0], pos[0, 1], pos[0, 2], marker="o", color="green", s=30)

            # Target Pos
            if "target_pos" in series:
                target = np.asarray(series["target_pos"])
                if target.ndim >= 2: target = target[0]
                target_pos_val = target
                ax_path.scatter(target[0], target[1], target[2], marker="x", color="red", s=30)
                
                dist = np.linalg.norm(pos - target_pos_val, axis=1).flatten()
                all_dist_errors.append(dist)

            # Target Attitude 
            if "target_attitude" in series:
                t_a = np.asarray(series["target_attitude"])
                
                if t_a.ndim == 1: 
                    target_att = np.tile(t_a, (quats.shape[0], 1))
                elif t_a.ndim >= 2 and t_a.shape[0] != quats.shape[0]:
                    target_att = np.tile(t_a[0], (quats.shape[0], 1))
                else:
                    target_att = t_a
                
                dot_prod = np.sum(quats * target_att, axis=1)
                dot_prod = np.clip(np.abs(dot_prod), -1.0, 1.0)
                total_error_rad = 2.0 * np.arccos(dot_prod)
                
                all_att_errors.append(total_error_rad.flatten())

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

        # Helper function
        def process_error_data(errors):
            if not errors: return None, None, None
            max_len = max(len(e) for e in errors)
            padded = np.array([np.pad(e, (0, max_len - len(e)), mode='edge') for e in errors])
            mean = np.mean(padded, axis=0).flatten()
            std = np.std(padded, axis=0).flatten()
            t = np.arange(max_len)
            return t, mean, std

        # Plot Position Error 
        t_pos, mean_pos, std_pos = process_error_data(all_dist_errors)
        if t_pos is not None:
            ax_dist_err.plot(t_pos, mean_pos, color="blue", linewidth=2, label="Mean Error")
            ax_dist_err.fill_between(t_pos, mean_pos - std_pos, mean_pos + std_pos, color="blue", alpha=0.2)
            
            final_err = mean_pos[-1]
            ax_dist_err.text(0.5, 0.9, f"Final: {final_err:.3f} ± {std_pos[-1]:.3f} m", 
                             transform=ax_dist_err.transAxes, ha="right", 
                             bbox=dict(boxstyle="round", fc="lightblue", alpha=0.5))
            ax_dist_err.set_title("Position Error [m]")
            ax_dist_err.set_ylabel("Error [m]")
            ax_dist_err.grid(True, alpha=0.3)
        else:
            ax_dist_err.text(0.5, 0.5, "No pos data", ha='center')

        # Plot Attitude Error
        t_att, mean_att, std_att = process_error_data(all_att_errors)

        if t_att is not None:
            ax_att_err.plot(t_att, mean_att, color="red", linewidth=2, label="Mean Error")
            ax_att_err.fill_between(t_att, mean_att - std_att, mean_att + std_att, color="red", alpha=0.2)

            final_att_err = mean_att[-1]
            ax_att_err.text(0.5, 0.9, f"Final: {final_att_err:.3f} ± {std_att[-1]:.3f} rad", 
                            transform=ax_att_err.transAxes, ha="right", 
                            bbox=dict(boxstyle="round", fc="lightsalmon", alpha=0.5))

            ax_att_err.set_title("Attitude Error [rad]")
            ax_att_err.set_ylabel("Error [rad]")
            ax_att_err.set_xlabel("Timestep")
            ax_att_err.grid(True, alpha=0.3)
        else:
            ax_att_err.text(0.5, 0.5, "No attitude data", ha='center')

        plt.tight_layout()

        if output is None:
            plt.show()
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=150)
            plt.close(fig)