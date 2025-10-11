"""Drone_v1


Source: github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
"""

from typing import Any
from functools import partial

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces
from gymnax.environments.environment import TEnvState, TEnvParams

from envs.base import Sensitivity

from utils import angle_normalize

def l1_smooth(x):
    delta = 1.0
    error_norm = jnp.linalg.norm(x + 1e-6)
    quadratic = jnp.minimum(error_norm, delta)
    linear = error_norm - quadratic
    return 0.5 * quadratic**2 + delta * linear

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
    min_torque: jax.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([0, -2.5, -2.5, -2]),
    )
    max_torque: jax.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([1, 2.5, 2.5, 2])
    )
    env_max_limits: jax.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([1.0, 1.0, 1.0])
    )
    env_min_limits: jax.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([-1.0, -1.0, 0.0])
    )
    dt: float = struct.field(pytree_node=False, default=0.02)
    g: float = struct.field(pytree_node=False, default=9.81)
    m: float = 0.027
    # TODO: check this
    I: jax.Array = struct.field(
        pytree_node=True,
        default_factory=lambda: jnp.array([2.3951e-5, 2.3951e-5, 3.2347e-5]),
    )
    max_steps_in_episode: int = struct.field(pytree_node=False, default=200)
    is_testing: bool = struct.field(pytree_node=False, default=False)


class DroneV1(environment.Environment):
    """JAX Compatible version of Drone-v1 target approaching."""

    def __init__(self):
        self.state_dim = 13
        self.obs_dim = 13
        self.action_dim = self.action_space(self.default_params).shape[0]
        self.param_dim = 4  # len(jax.tree.leaves(self.default_params))
        self.target_pos = jnp.array([0.0, 0.0, 0.5])
        self.hovering_thrust = self.default_params.m * self.default_params.g

        self.params_min = [
            0.02,  # mass
            jnp.array([1e-5, 1e-5, 1e-5]),  # inertia
        ]
        self.params_max = [
            0.05,  # mass
            jnp.array([1e-4, 1e-4, 1e-4]),  # inertia 
        ]  
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for Drone-v1."""
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
        """Integrate pendulum ODE and return transition."""

        u = jnp.clip(action, params.min_torque, params.max_torque)

        state = env_state.state
        pos_diff_old = state.pos - env_state.target_pos



        state_next = self.step_dynamics(key, state, action, params)

        pos_diff = state_next.pos - env_state.target_pos

        reward_app = jax.lax.select(
            jnp.linalg.norm(pos_diff) < 0.01,
            5.0,  # reward when close to target
            1.0 * (- l1_smooth( 5.0 * (state_next.pos -  env_state.target_pos))), # distance penalty
        )

        reward = (
            - 0.1 * l1_smooth(state_next.omega)
            - 0.1 * l1_smooth(state_next.vel)
            + reward_app
            - 0.5 * (l1_smooth(u - self.hovering_thrust))
            # - 0.005 * (l1_smooth(u - env_state.last_u) ** 2)
        )

        reward = reward.squeeze()

        env_state = EnvState(
            state=state_next,
            target_pos=env_state.target_pos,
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

        u = jnp.clip(action, params.min_torque, params.max_torque)

        quat = state.attitude
        ang_vel = state.omega

        orientation_mat = quat2rotm(quat)
        ang_vel_quat = jnp.array([0.0, ang_vel[0], ang_vel[1], ang_vel[2]])

        total_force = (
            jnp.array([0.0, 0.0, u[0]])
            - params.m * params.g * orientation_mat[2, :]
        )  # transpose + 3rd col = 3rd row

        total_torque = (
            1e-3 * u[1:4] - skew(ang_vel) @ params.I @ ang_vel
        )  # multiplication by normalization factor

        si_mat = jnp.diag(
            jnp.concatenate(
                [params.m * jnp.ones(3, dtype=jnp.float32), params.I]
            )
        )
        si_mat_inv = jnp.linalg.inv(si_mat)

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
            )
        )

        env_state = EnvState(
            state=state,
            target_pos=target_pos,
            last_u=jnp.array([0.0, 0.0, 0.0, 0.0]),
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
        return "Pendulum-Sensitivity"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 1

    def action_space(self, params: EnvParams | None = None) -> spaces.Box:
        """Action space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Box(
            low=params.min_torque,
            high=params.max_torque,
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