"""Quadrotor environment in JAX, with uncertain mass, inertia, centre of mass
position and actuator time constant."""

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
    """Rigid-body state of the quadrotor.

    Attributes:
        pos (jax.Array): Position in the world frame [m].
        attitude (jax.Array): Unit quaternion (w, x, y, z), body to world.
        vel (jax.Array): Linear velocity in the world frame [m/s].
        omega (jax.Array): Angular velocity in the body frame [rad/s].
    """
    pos: jax.Array
    attitude: jax.Array  # (unit quaternion [w, x, y, z])
    vel: jax.Array
    omega: jax.Array


@struct.dataclass
class EnvState(environment.EnvState):
    """State of the quadrotor environment: rigid body, target, inputs and episode counter.

    Attributes:
        state (ModelState): Rigid-body state.
        target_pos (jax.Array): Target position [m].
        target_yaw (jax.Array): Target yaw [rad].
        last_input (jax.Array): Input applied to the system after the actuator low-pass filter.
        last_commanded_action (jax.Array): Last action commanded by the policy.
        time (int): Step index in the episode.
    """
    state: ModelState
    target_pos: jax.Array
    target_yaw: jax.Array  # Desired yaw angle (rotation around z-axis)
    last_input: jax.Array # This is the actual input applied to the system after low-pass filtering, which is what affects the next state
    last_commanded_action: jax.Array # This is the commanded action

    time: int

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        """Forward attribute access to the wrapped model state.

        Args:
            name (str): Attribute name, e.g. ``pos`` or ``attitude``.

        Returns:
            Any: The attribute of ``self.state``.
        """
        return getattr(self.state, name)


@struct.dataclass
class EnvParams(environment.EnvParams):
    """Parameters of the quadrotor environment.

    The pytree fields (mass, inertia, centre of mass and actuator time
    constant) are the uncertain parameters; the fields with
    ``pytree_node=False`` are static.
    """
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
        pytree_node=False, default_factory=lambda: jnp.array([2.0, 2.0, 1.0])
    )
    env_min_limits: jax.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([-2.0, -2.0, -1.0])
    )
    dt: float = struct.field(pytree_node=False, default=0.02)
    g: float = struct.field(pytree_node=False, default=9.81)
    m: float = 1.4
    I: jax.Array = struct.field(
        pytree_node=True,
        default_factory=lambda: jnp.array([0.012, 0.012, 0.020]),
    )
    com_pos: jax.Array = struct.field(
        pytree_node=True, default_factory=lambda: jnp.array([0.0, 0.0, 0.0])
    )
    actuator_tau: float = 0.04  # Actuator time constant [s] (low-pass filter)

    max_steps_in_episode: int = struct.field(pytree_node=False, default=200)
    is_testing: bool = struct.field(pytree_node=False, default=False)


class DroneV3(environment.Environment):
    """Quadrotor reaching a target position and yaw."""

    def __init__(self):
        """Set the dimensions, the nominal target, the sensitivity weights, the hover input and the parameter ranges."""
        self.type = ""
        self.state_dim = 13
        self.obs_dim = 14  # 13 + yaw error
        self.action_dim = self.action_space(self.default_params).shape[0]
        self.param_dim = 8  # len(jax.tree.leaves(self.default_params))
        self.target_pos = jnp.array([0.0, 0.0, 0.5])

        self.sens_weights = jnp.zeros((self.state_dim, self.state_dim))
        self.sens_weights = self.sens_weights.at[0:3, 0:3].set(5.0)  # position sensitivity
        self.sens_weights = self.sens_weights.at[3:7, 3:7].set(0.5)  # attitude sensitivity

        # Hovering input corresponds to the input required to counteract gravity at the default mass, with zero torques
        self.hovering_input = jnp.array(
            [self.default_params.m * self.default_params.g, 0.0, 0.0, 0.0]
        )

        self.params_min = [
            1.0,  # mass
            jnp.array([0.01, 0.01, 0.01]),  # inertia
            jnp.array([-0.025, -0.025, -0.02]),  # com_pos
            0.01,  # actuator_tau
        ]
        self.params_max = [
            1.8,  # mass
            jnp.array([0.05, 0.05, 0.05]),  # inertia
            jnp.array([0.025, 0.025, 0.02]),  # com_pos
            0.08,  # actuator_tau
        ]
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        """Default parameters of the quadrotor environment.

        Returns:
            EnvParams: Nominal parameters.
        """
        return EnvParams()

    def rescale_action(self, action: jax.Array, params: EnvParams) -> jax.Array:
        """Map an action from ``[min_action, max_action]`` to ``[min_input, max_input]``.

        Args:
            action (jax.Array): Action of the policy.
            params (EnvParams): Environment parameters.

        Returns:
            jax.Array: Actuator input.
        """
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
        """Environment step without the automatic reset at the end of an episode.

        The reset is applied by ``ResetEnvWrapper`` or ``DomainRandomizationWrapper``.

        Args:
            key (jax.Array): PRNG key.
            state (EnvState): Current environment state.
            action (jax.Array): Action of the policy.
            params (EnvParams, optional): Environment parameters; the defaults when ``None``.

        Returns:
            tuple[jax.Array, EnvState, jax.Array, jax.Array, dict]: Observation, next state, reward, done flag and info.
        """

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
        """Apply an action, integrate the dynamics and compute the reward.

        The reward combines the distance to the target (with a bonus when close and
        slow), the yaw error (with a bonus below 0.05 rad), the distance from the
        hover input and the variation of the input.

        Args:
            key (jax.Array): PRNG key.
            env_state (EnvState): Current environment state.
            action (jax.Array): Action of the policy, shape ``(4,)``.
            params (EnvParams): Environment parameters.

        Returns:
            tuple[jax.Array, EnvState, jax.Array, jax.Array, dict]: Observation, next state, reward, done flag and info.
        """

        action_clipped = jnp.clip(action, params.min_action, params.max_action)
        action_rescaled = self.rescale_action(action_clipped, params)

        state_next = self.step_dynamics(key, env_state, action_rescaled, params)

        # Compute the filtered actuator input (must match step_dynamics)
        alpha = jnp.clip(params.dt / params.actuator_tau, 0.0, 1.0)
        u_actual = env_state.last_input + alpha * (action_rescaled - env_state.last_input)

        pos_diff_norm = jnp.linalg.norm(state_next.pos - env_state.target_pos)
        vel_norm = jnp.linalg.norm(state_next.vel)

        reward_app = -1.0 * pos_diff_norm + jax.lax.select(
            jnp.logical_and(pos_diff_norm < 0.05, vel_norm < 0.1),
            10.0,  # reward when close to target
            0.0,  # distance improvement reward
        )

        # Compute yaw error from quaternion
        # Yaw can be extracted from quaternion as: yaw = atan2(2*(q0*q3 + q1*q2), 1 - 2*(q2^2 + q3^2))
        q = state_next.attitude
        current_yaw = jnp.arctan2(
            2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)
        )
        yaw_error = jnp.abs(
            jnp.arctan2(
                jnp.sin(current_yaw - env_state.target_yaw),
                jnp.cos(current_yaw - env_state.target_yaw),
            )
        )

        reward_yaw = -1.0 * yaw_error + jax.lax.select(
            yaw_error < 0.05,
            1.0,  # reward when close to target
            0.0,  # no reward otherwise
        )

        reward = (
            # -0.01 * jnp.linalg.norm(state_next.omega)
            + reward_app
            + reward_yaw
            - 1.0
            * (
                jnp.linalg.norm(
                    (action_rescaled - self.hovering_input) / params.max_input
                )
            )
            - 0.2
            * (jnp.linalg.norm((action_rescaled - env_state.last_input) / params.max_input))
        )

        reward = reward.squeeze()

        env_state = EnvState(
            state=state_next,
            target_pos=env_state.target_pos,
            target_yaw=env_state.target_yaw,
            last_input=u_actual.reshape(-1),
            last_commanded_action=action_rescaled.reshape(-1),
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
        """Next model state, with a first-order low-pass filter on the actuators.

        The actuator output is filtered as::

            u_actual = last_input + alpha * (u_commanded - last_input)

        with ``alpha = dt / tau`` clamped to ``[0, 1]``.

        Args:
            key (jax.Array): PRNG key.
            env_state (EnvState): Current environment state (holds the last input).
            action (jax.Array): Commanded input (thrust and three torques).
            params (EnvParams): Environment parameters (``dt``, ``actuator_tau``).

        Returns:
            ModelState: Next model state.
        """
        state = env_state.state
        last_input = env_state.last_input

        # First-order low-pass filter coefficient
        alpha = jnp.clip(params.dt / params.actuator_tau, 0.0, 1.0)
        u_actual = last_input + alpha * (action - last_input)

        return self._step_dynamics(key, state, u_actual, params)

    def _step_dynamics(
        self,
        key: jax.Array,
        state: ModelState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> ModelState:

        """Rigid-body dynamics of the quadrotor, with the centre of mass offset.

        Args:
            key (jax.Array): PRNG key.
            state (ModelState): Current model state.
            action (jax.Array): Filtered input (thrust and three torques).
            params (EnvParams): Environment parameters.

        Returns:
            ModelState: Next model state.
        """
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
        """Reset the environment by sampling the initial state and the target.

        Args:
            key (jax.Array): PRNG key.
            params (EnvParams): Environment parameters (sampling ranges).

        Returns:
            tuple[jax.Array, EnvState]: First observation and initial state.
        """
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
            last_input=self.hovering_input,
            last_commanded_action=self.hovering_input,
            time=0,
        )
        return self.get_obs(env_state), env_state

    def get_obs(
        self,
        env_state: EnvState,
        key: jax.Array | None = None,
        params: EnvParams | None = None,
    ) -> jax.Array:
        """Observation of the policy, without noise.

        The 14 components are the vector to the target, the velocity, the attitude
        quaternion, the angular velocity and the yaw error. The key and params
        arguments keep the signature expected by the sensitivity wrappers.

        Args:
            env_state (EnvState): Environment state.
            key (jax.Array, optional): Unused.
            params (EnvParams, optional): Unused.

        Returns:
            jax.Array: Observation, shape ``(14,)``.
        """
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
            ]
        ).squeeze()

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Whether the episode has reached its maximum length.

        Args:
            state (EnvState): Environment state.
            params (EnvParams): Environment parameters (``max_steps_in_episode``).

        Returns:
            jax.Array: Boolean done flag.
        """
        # Check number of steps in episode termination condition
        done = state.time >= params.max_steps_in_episode
        return jnp.array(done)

    def eval_metrics(
        self, env_state: EnvState, params: EnvParams
    ) -> dict[str, jax.Array]:
        """Evaluation metrics of the current state, logged during the rollouts.

        Args:
            env_state (EnvState): Current environment state.
            params (EnvParams): Environment parameters.

        Returns:
            dict[str, jax.Array]: Metric name to scalar value.
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
        }

    @property
    def name(self) -> str:
        """Environment name.

        Returns:
            str: Name used in the logs.
        """
        return "Drone-v2"

    @property
    def num_actions(self) -> int:
        """Number of action components.

        Returns:
            int: Size of the action vector.
        """
        return 4

    def action_space(self, params: EnvParams | None = None) -> spaces.Box:
        """Action space of the environment.

        Args:
            params (EnvParams, optional): Environment parameters; the defaults when ``None``.

        Returns:
            spaces.Box: Box between ``min_action`` and ``max_action``.
        """
        if params is None:
            params = self.default_params
        return spaces.Box(
            low=params.min_action,
            high=params.max_action,
            shape=(4,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment.

        Args:
            params (EnvParams): Environment parameters.

        Returns:
            spaces.Box: Box with one entry per observation component.
        """
        return spaces.Box(-1, 1, shape=(self.obs_dim,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment.

        Args:
            params (EnvParams): Environment parameters.

        Returns:
            spaces.Dict: Spaces of the state components.
        """
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
                "last_input": spaces.Box(
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
        """Plot the trajectories of a set of rollouts.

        Args:
            series_list (list[dict]): One dict per rollout, with ``state`` (shape ``(T+1, 13)``),
                ``actions`` (shape ``(T, 4)``) and optionally ``target_pos`` and ``eval_metrics``.
            output (pathlib.Path, optional): File to save the figure to; shown interactively when ``None``.
                A second figure with the state components is saved to ``str(output) + "1"``.
        """
        import matplotlib.pyplot as plt

        plt.rcdefaults()
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(2, 4, height_ratios=[2, 1], width_ratios=[1, 1, 1, 1])

        ax_path = fig.add_subplot(gs[0, :2], projection="3d")
        ax_dist = fig.add_subplot(gs[0, 2:])
        axes_act = [fig.add_subplot(gs[1, i]) for i in range(4)]

        labels = ["Thrust", "Torque X", "Torque Y", "Torque Z"]

        _defaults = EnvParams()
        min_input = np.asarray(_defaults.min_input)
        max_input = np.asarray(_defaults.max_input)

        # Collect distance_to_target and yaw_error from all rollouts
        distance_series = []
        yaw_series = []
        state_series = []
        default_dt = 0.02

        for series in series_list:
            state = series.get("state", None)
            if state is None:
                continue
            state = np.asarray(state)
            series_dt = float(series.get("dt", default_dt))
            # state format: [pos(3), attitude(4), vel(3), omega(3)]
            pos = state[:, :3]

            is_nominal = series.get("is_nominal", False)
            traj_color = "red" if is_nominal else "grey"
            traj_alpha = 0.9 if is_nominal else 0.4
            traj_lw = 1.5 if is_nominal else 0.8
            traj_zorder = 6 if is_nominal else 3

            if state.ndim == 2 and state.shape[1] >= 13:
                state_series.append({"values": state[:, :13], "dt": series_dt, "is_nominal": is_nominal})

            ax_path.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=traj_color, alpha=traj_alpha, linewidth=traj_lw, zorder=traj_zorder)

            if "target_pos" in series:
                target = np.asarray(series["target_pos"])
                if target.ndim >= 2:
                    target = target[0]
                ax_path.scatter(
                    target[0], target[1], target[2], marker="x", color="red", s=30, zorder=10
                )

            ax_path.scatter(
                pos[0, 0], pos[0, 1], pos[0, 2], marker="o", color="green", s=30, zorder=10
            )
            ax_path.scatter(
                pos[-1, 0], pos[-1, 1], pos[-1, 2], marker="s", color="blue", s=30, zorder=10
            )

            actions = np.asarray(series.get("actions", []))
            if actions.size > 0:
                for idx, ax in enumerate(axes_act):
                    if idx < actions.shape[1]:
                        ax.plot(np.arange(actions.shape[0]), actions[:, idx], color=traj_color, alpha=traj_alpha, linewidth=traj_lw, zorder=traj_zorder)
                    ax.set_xlabel("t")
                    ax.set_ylabel(labels[idx])
                    ax.set_ylim(float(min_input[idx]), float(max_input[idx]))
                    ax.grid(True, alpha=0.3)

            # Collect distance_to_target and yaw_error if available
            eval_metrics = series.get("eval_metrics", {})
            if "distance_to_target" in eval_metrics:
                dist = np.asarray(eval_metrics["distance_to_target"]).flatten()
                distance_series.append({"values": dist, "dt": series_dt})
            if "yaw_error" in eval_metrics:
                yaw = np.asarray(eval_metrics["yaw_error"]).flatten()
                yaw_series.append({"values": yaw, "dt": series_dt})

        ax_path.set_title("Quadrotor position trajectories")
        ax_path.set_xlabel("x [m]")
        ax_path.set_ylabel("y [m]")
        ax_path.set_zlabel("z [m]")
        ax_path.grid(True, alpha=0.3)

        # Plot mean distance to target and yaw error with std shading
        if distance_series or yaw_series:
            # Create twin axis for yaw error
            ax_yaw = ax_dist.twinx()

            # Plot distance to target
            if distance_series:
                max_len = max(d["values"].shape[0] for d in distance_series)
                padded = np.array(
                    [
                        np.pad(
                            d["values"],
                            (0, max_len - d["values"].shape[0]),
                            mode="edge",
                        )
                        for d in distance_series
                    ]
                )
                mean_dist = np.mean(padded, axis=0)
                std_dist = np.std(padded, axis=0)
                reference_dt = (
                    distance_series[0]["dt"] if distance_series else default_dt
                )
                timesteps = np.arange(max_len) * reference_dt

                for dist_entry in distance_series:
                    ax_dist.plot(
                        np.arange(dist_entry["values"].shape[0]) * dist_entry["dt"],
                        dist_entry["values"],
                        color="blue",
                        alpha=0.2,
                        linewidth=0.8,
                        zorder=1,
                    )

                ax_dist.plot(
                    timesteps,
                    mean_dist,
                    color="blue",
                    label="Distance",
                    linewidth=2,
                    zorder=2,
                )
                ax_dist.fill_between(
                    timesteps,
                    mean_dist - std_dist,
                    mean_dist + std_dist,
                    alpha=0.3,
                    color="blue",
                )
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
            if yaw_series:
                max_len_yaw = max(y["values"].shape[0] for y in yaw_series)
                padded_yaw = np.array(
                    [
                        np.pad(
                            y["values"],
                            (0, max_len_yaw - y["values"].shape[0]),
                            mode="edge",
                        )
                        for y in yaw_series
                    ]
                )
                mean_yaw = np.mean(padded_yaw, axis=0)
                std_yaw = np.std(padded_yaw, axis=0)
                reference_dt = (
                    distance_series[0]["dt"] if distance_series else yaw_series[0]["dt"]
                )
                timesteps_yaw = np.arange(max_len_yaw) * reference_dt

                for yaw_entry in yaw_series:
                    ax_yaw.plot(
                        np.arange(yaw_entry["values"].shape[0]) * yaw_entry["dt"],
                        yaw_entry["values"],
                        color="red",
                        alpha=0.2,
                        linewidth=0.8,
                        zorder=1,
                    )

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
            ax_dist.set_xlabel("Time [s]")
            ax_dist.grid(True, alpha=0.3)
            dist_top = max(ax_dist.get_ylim()[1], 0)
            ax_dist.set_ylim(0, dist_top)
            yaw_top = max(ax_yaw.get_ylim()[1], 0)
            ax_yaw.set_ylim(0, yaw_top)

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

        plt.tight_layout()

        if output is None:
            plt.show()
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=150)
            plt.close(fig)

        # Extra figure: each state component trajectory with mean/std shading
        if state_series:
            state_labels = [
                "pos_x",
                "pos_y",
                "pos_z",
                "q_w",
                "q_x",
                "q_y",
                "q_z",
                "vel_x",
                "vel_y",
                "vel_z",
                "omega_x",
                "omega_y",
                "omega_z",
            ]

            fig_state, axes_state = plt.subplots(4, 4, figsize=(20, 12), sharex=False)
            axes_state = axes_state.flatten()

            for idx, label in enumerate(state_labels):
                ax = axes_state[idx]
                max_len = max(entry["values"].shape[0] for entry in state_series)
                padded = np.array(
                    [
                        np.pad(
                            entry["values"][:, idx],
                            (0, max_len - entry["values"].shape[0]),
                            mode="edge",
                        )
                        for entry in state_series
                    ]
                )

                mean_state = np.mean(padded, axis=0)
                std_state = np.std(padded, axis=0)
                reference_dt = state_series[0]["dt"] if state_series else default_dt
                timesteps = np.arange(max_len) * reference_dt

                for entry in state_series:
                    s_nominal = entry.get("is_nominal", False)
                    s_color = "red" if s_nominal else "grey"
                    s_alpha = 0.9 if s_nominal else 0.2
                    s_lw = 1.5 if s_nominal else 0.8
                    s_zorder = 3 if s_nominal else 1
                    ax.plot(
                        np.arange(entry["values"].shape[0]) * entry["dt"],
                        entry["values"][:, idx],
                        color=s_color,
                        alpha=s_alpha,
                        linewidth=s_lw,
                        zorder=s_zorder,
                    )

                ax.plot(
                    timesteps,
                    mean_state,
                    color="tab:blue",
                    linewidth=2.0,
                    zorder=2,
                )
                ax.fill_between(
                    timesteps,
                    mean_state - std_state,
                    mean_state + std_state,
                    color="tab:blue",
                    alpha=0.25,
                    zorder=1,
                )
                ax.set_title(label)
                ax.set_xlabel("Time [s]")
                ax.grid(True, alpha=0.3)

            for idx in range(len(state_labels), len(axes_state)):
                axes_state[idx].axis("off")

            fig_state.suptitle("State trajectories over time")
            fig_state.tight_layout(rect=[0.0, 0.03, 1.0, 0.97])

            if output is None:
                plt.show()
            else:
                output_state = Path(output.parent, "state_" + output.name)
                output_state.parent.mkdir(parents=True, exist_ok=True)
                fig_state.savefig(output_state, dpi=150)
                plt.close(fig_state)
