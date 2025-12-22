"""Tilthex-Sensitivity 
JAX implementation of Tilthex OpenAI gym environment with added
parametric sensitivity metric optimization."""

# LR, CLIP_EPS

from typing import Any
from functools import partial

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces
from gymnax.environments.environment import TEnvState, TEnvParams

from envs.base import Sensitivity

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
        pytree_node=False, default_factory=lambda: jnp.array([1.0, 1.0, 1.0])
    )
    env_min_limits: jax.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([-1.0, -1.0, 0.0])
    )

    m: float = 3.072
    
    dt: float = struct.field(pytree_node=False, default=0.02)
    g: float = struct.field(pytree_node=False, default=9.81)
    n: float = struct.field(pytree_node=False, default=6)  # number of rotors
    arm_length : float = struct.field(pytree_node=False, default=0.39)
    c_f: float = struct.field(pytree_node=False, default=12.0e-4) # squared propeller speed to thrust force coefficient [N/hz^2]
    c_tau: float = struct.field(pytree_node=False, default=2.4e-5) # squared propeller speed to drag torque coefficient [Nm/Hz^2]
    alpha: float = struct.field(pytree_node=False, default=-20.0) # tilt_angle [deg] radial inclination angle of rotors
    beta: float = struct.field(pytree_node=False, default=0.0) # beta tilting angle
    wmin: float = struct.field(pytree_node=False, default=16) # min prop rotation speed in Hz
    wmax: float = struct.field(pytree_node=False, default=100) # max prop rotation speed in Hz
    
    com_default: jax.Array = struct.field(pytree_node=False,default_factory=lambda: jnp.array([0.0, 0.0, 0.0]),)
    I: jax.Array = struct.field(pytree_node=True,default_factory=lambda: jnp.array([0.076276, 0.078072, 0.147915]),)
    max_steps_in_episode: int = struct.field(pytree_node=False, default=200)
    is_testing: bool = struct.field(pytree_node=False, default=False)


class TilthexV0(environment.Environment):
    """JAX Compatible version of Tilthex-v0 target approaching."""

    def __init__(self):
        self.type = ""
        self.state_dim = 13
        self.obs_dim = 13
        self.action_dim = self.action_space(self.default_params).shape[0]
        self.param_dim = 4  # len(jax.tree.leaves(self.default_params))
        self.target_pos = jnp.array([0.0, 0.0, 0.5]) 

        # hovering action
        efficiency = jnp.cos(jnp.deg2rad(self.default_params.alpha))
        hover_force = (self.default_params.m * self.default_params.g) / (self.default_params.n * efficiency)
        self.hovering_action = jnp.full((self.default_params.n,), hover_force)

        self.params_min = [
            2.4,  # mass
            jnp.array([0.059, 0.061, 0.115]),  # inertia
        ]
        self.params_max = [
            3.6,  # mass
            jnp.array([0.089, 0.091, 0.173]),  # inertia
        ]
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for Tilthex-v0."""
        return EnvParams()

    def rescale_action(self, action: jax.Array, params: EnvParams) -> jax.Array:
        """
        Rescale action: map 0.0 to hovering thrust to fix initialization bias.
        """
        efficiency = jnp.cos(jnp.deg2rad(params.alpha))
        hover_force_single = (params.m * params.g) / (params.n * efficiency)

        action_scale = 3.5  
        
        u_unclipped = hover_force_single + (action * action_scale)
        return jnp.clip(u_unclipped, params.min_input, params.max_input)

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
        # ... inizio funzione ...

        # DEBUG: Stampiamo l'azione GREZZA che esce dalla rete
        # Se vedi numeri > 1.0 o < -1.0, abbiamo trovato il colpevole.
        jax.debug.print("RAW Action corretta 3.5 : {x}", x=action)

        clipped_action = jnp.clip(action, params.min_action, params.max_action)
        # ... resto del codice ...
        input = self.rescale_action(clipped_action, params)

        state = env_state.state

        state_next = self.step_dynamics(key, state, input, params)

        pos_diff = state_next.pos - env_state.target_pos

        R = quat2rotm(state_next.attitude)
        z_proj = R[2, 2] # 1.0 = Dritto, -1.0 = Capovolto
        reward_upright = -1.0* (1.0 - z_proj)

        reward_app = jax.lax.select(
            jnp.linalg.norm(pos_diff) < 0.2,
            10.0,  # reward when close to target
            1.0 * (-jnp.linalg.norm(pos_diff)),  # distance improvement reward
        )

        thrust_normalized_input = input / params.max_input
        thrust_normalized_input_last = env_state.last_u / params.max_input

        reward = (
            -0.01 * jnp.linalg.norm(state_next.omega)
            + reward_app
            + reward_upright
            - 0.001 * (jnp.linalg.norm((thrust_normalized_input - self.hovering_action)) ** 2)
            - 0.005 * (jnp.linalg.norm(thrust_normalized_input - thrust_normalized_input_last) ** 2)
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
        
        # Wrench nel Body Frame
        F_body = GF @ u       # [Fx, Fy, Fz] 
        tau_body = GT @ u     # [Tx, Ty, Tz] 

        # Acceleration (IN BODY FRAME)
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
            low=params.min_input,
            high=params.max_input,
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
