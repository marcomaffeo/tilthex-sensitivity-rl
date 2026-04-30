"""Drone Obstacle Environment.

JAX implementation of the Mikrokopter drone navigating from start to target
while avoiding a square pillar obstacle in the room.  Based on drone_v3
dynamics (actuator low-pass filter, CoM offset, etc.).

The obstacle is a vertical square pillar centred near (0, 0).  During
training, the pillar centre is randomised within a small region around the
origin.  During testing it is fixed at the origin.

Constraint interface (for CaT):
    - hard_constraints: penetration depth into the pillar (1-D),
      positive when the drone is inside the obstacle (violated).
    - soft_constraints: empty (no soft constraints).
"""

from functools import partial
from typing import Any, List
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import environment, spaces

from .math import quat_product, quat_inverse
from .drone_v3 import DroneV3, ModelState

# ──────────────────────────────────────────────────────────────────────────────
# State / Params dataclasses  (obstacle-specific fields only)
# ──────────────────────────────────────────────────────────────────────────────

@struct.dataclass
class EnvState(environment.EnvState):
    state: ModelState
    target_pos: jax.Array
    target_yaw: jax.Array
    obstacle_pos: jax.Array         # (2,) XY centre of the pillar
    obstacle_half_size: jax.Array   # scalar, half-width of the square pillar
    last_input: jax.Array           # filtered actuator output
    last_commanded_action: jax.Array  # raw rescaled action before filter
    time: int

    def __getattr__(self, name):
        return getattr(self.state, name)


@struct.dataclass
class EnvParams(environment.EnvParams):
    # Action / input limits — same as drone_v3
    min_action: jax.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([-1.0, -1.0, -1.0, -1.0]),
    )
    max_action: jax.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([1.0, 1.0, 1.0, 1.0]),
    )
    min_input: jax.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([4.0, -3.0, -3.0, -1.0]),
    )
    max_input: jax.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([25.0, 3.0, 3.0, 1.0]),
    )
    env_max_limits: jax.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([2.0, 2.0, 1.0])
    )
    env_min_limits: jax.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([-2.0, -2.0, -1.0])
    )

    # Physical parameters — slightly different inertia default vs drone_v3
    dt: float = struct.field(pytree_node=False, default=0.02)
    g: float = struct.field(pytree_node=False, default=9.81)
    m: float = 1.4
    I: jax.Array = struct.field(
        pytree_node=True,
        default_factory=lambda: jnp.array([0.012, 0.012, 0.020]),
    )
    com_pos: jax.Array = struct.field(
        pytree_node=True,
        default_factory=lambda: jnp.array([0.0, 0.0, 0.0]),
    )
    actuator_tau: float = 0.04

    # Obstacle parameters (not present in drone_v3)
    obstacle_half_size: float = struct.field(pytree_node=False, default=0.25)
    obstacle_size_noise: float = struct.field(pytree_node=False, default=0.1)
    obstacle_noise: float = struct.field(pytree_node=False, default=0.3)

    max_steps_in_episode: int = struct.field(pytree_node=False, default=200)
    is_testing: bool = struct.field(pytree_node=False, default=False)


# ──────────────────────────────────────────────────────────────────────────────
# Environment class
# ──────────────────────────────────────────────────────────────────────────────

class DroneObstacle(DroneV3):
    """Drone navigating to a target while avoiding a square pillar.

    Inherits all dynamics from DroneV3 (_step_dynamics, rescale_action, step,
    is_terminal, action_space, observation_space, num_actions).
    Overrides only obstacle-specific behaviour.
    """

    def __init__(self):
        self.type = ""
        self.state_dim = 13
        # obs: to_target(3), vel(3), quat(4), omega(3), yaw_error(1),
        #      to_obstacle(2), obstacle_half_size(1)
        self.obs_dim = 17
        self.action_dim = self.action_space(self.default_params).shape[0]
        self.param_dim = 8


        self.start_pos = jnp.array([-1.5, 0.0, 0.5])
        self.target_pos = jnp.array([1.5, 0.0, 0.5])
        self.hovering_input = jnp.array(
            [self.default_params.m * self.default_params.g, 0.0, 0.0, 0.0]
        )

        self.params_min = [
            1.0,                                    # mass
            jnp.array([0.01, 0.01, 0.01]),          # inertia
            jnp.array([-0.02, -0.02, -0.02]),       # com_pos
            0.01,                                   # actuator_tau
        ]
        self.params_max = [
            1.8,
            jnp.array([0.05, 0.05, 0.05]),
            jnp.array([0.02, 0.02, 0.02]),
            0.08,
        ]
        # Skip DroneV3.__init__ — it sets fields (sens_weights, priv_obs_dim)
        # that don't apply here.  Call the gymnax base directly.
        environment.Environment.__init__(self)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    # ── step ─────────────────────────────────────────────────────────────

    @partial(jax.jit, static_argnums=(0))
    def step_env(
        self,
        key: jax.Array,
        env_state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        action_clipped = jnp.clip(action, params.min_action, params.max_action)
        action_rescaled = self.rescale_action(action_clipped, params)

        state_next = self.step_dynamics(key, env_state, action_rescaled, params)

        alpha = jnp.clip(params.dt / params.actuator_tau, 0.0, 1.0)
        u_actual = env_state.last_input + alpha * (action_rescaled - env_state.last_input)

        pos_diff_norm = jnp.linalg.norm(state_next.pos - env_state.target_pos)
        vel_norm = jnp.linalg.norm(state_next.vel)

        reward_app = -1.0 * pos_diff_norm + jax.lax.select(
            jnp.logical_and(pos_diff_norm < 0.05, vel_norm < 0.1),
            10.0,
            0.0,
        )

        q = state_next.attitude
        current_yaw = jnp.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2]**2 + q[3]**2))
        yaw_error = jnp.abs(jnp.arctan2(jnp.sin(current_yaw - env_state.target_yaw),
                                         jnp.cos(current_yaw - env_state.target_yaw)))

        reward_yaw = -1.0 * yaw_error + jax.lax.select(
            yaw_error < 0.05, 1.0, 0.0,
        )

        reward = (
            reward_app
            + reward_yaw
            - 1.0 * (jnp.linalg.norm((action_rescaled - self.hovering_input) / params.max_input))
            - 0.2 * (jnp.linalg.norm((action_rescaled - env_state.last_input) / params.max_input))
        ).squeeze()

        env_state = EnvState(
            state=state_next,
            target_pos=env_state.target_pos,
            target_yaw=env_state.target_yaw,
            obstacle_pos=env_state.obstacle_pos,
            obstacle_half_size=env_state.obstacle_half_size,
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

    # step_dynamics inherited from DroneV3 (reads last_input, same field name now)

    # ── reset ─────────────────────────────────────────────────────────────

    def reset_env(self, key, params):
        key, key_pos, key_target, key_yaw, key_obs, key_size = jax.random.split(key, 6)

        pos = jax.lax.select(
            params.is_testing,
            self.start_pos + jax.random.uniform(key_pos, shape=(3,), minval=-0.25, maxval=0.25),
            jax.random.uniform(key_pos, shape=(3,),
                               minval=params.env_min_limits, maxval=params.env_max_limits),
        )
        state = ModelState(
            pos=pos,
            attitude=jnp.array([1.0, 0.0, 0.0, 0.0]),
            vel=jnp.zeros(3),
            omega=jnp.zeros(3),
        )

        target_pos = jax.lax.select(
            params.is_testing,
            self.target_pos,
            jax.random.uniform(key_target, shape=(3,),
                               minval=params.env_min_limits, maxval=params.env_max_limits),
        )
        target_yaw = jax.lax.select(
            params.is_testing,
            jnp.array(0.0),
            jax.random.uniform(key_yaw, minval=-jnp.pi, maxval=jnp.pi),
        )

        obstacle_pos = jax.lax.select(
            params.is_testing,
            jnp.array([0.0, 0.0]),
            jax.random.uniform(key_obs, shape=(2,),
                               minval=-params.obstacle_noise, maxval=params.obstacle_noise),
        )
        obstacle_half_size = jax.lax.select(
            params.is_testing,
            jnp.array(params.obstacle_half_size),
            jnp.array(params.obstacle_half_size) + jax.random.uniform(
                key_size, minval=-params.obstacle_size_noise, maxval=params.obstacle_size_noise,
            ),
        )

        env_state = EnvState(
            state=state,
            target_pos=target_pos,
            target_yaw=target_yaw,
            obstacle_pos=obstacle_pos,
            obstacle_half_size=obstacle_half_size,
            last_input=self.hovering_input,
            last_commanded_action=self.hovering_input,
            time=0,
        )
        return self.get_obs(env_state), env_state

    # ── observation ───────────────────────────────────────────────────────

    def get_obs(self, env_state):
        state = env_state.state
        q = state.attitude
        current_yaw = jnp.arctan2(
            2.0 * (q[0] * q[3] + q[1] * q[2]),
            1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2),
        )
        yaw_error = jnp.arctan2(
            jnp.sin(current_yaw - env_state.target_yaw),
            jnp.cos(current_yaw - env_state.target_yaw),
        )
        to_obstacle = env_state.obstacle_pos - state.pos[:2]
        return jnp.concatenate([
            (env_state.target_pos - state.pos).reshape(-1),   # 3
            state.vel.reshape(-1),                            # 3
            q.reshape(-1),                                    # 4
            state.omega.reshape(-1),                          # 3
            jnp.array([yaw_error]),                           # 1
            to_obstacle.reshape(-1),                          # 2
            jnp.array([env_state.obstacle_half_size]),        # 1
        ]).squeeze()

    # ── terminal / constraints ────────────────────────────────────────────

    # is_terminal inherited from DroneV3 (same logic)

    def hard_constraints(self, env_state, params):
        """Penetration depth into the square pillar (L∞).

        Returns > 0 when inside the obstacle (violated), ≤ 0 when safe.
        """
        diff = jnp.abs(env_state.state.pos[:2] - env_state.obstacle_pos)
        penetration = env_state.obstacle_half_size - jnp.max(diff)
        return jnp.array([penetration])

    def soft_constraints(self, env_state, params):
        return jnp.zeros(0)

    # ── eval metrics ──────────────────────────────────────────────────────

    def eval_metrics(self, env_state, params):
        state = env_state.state
        distance_to_target = jnp.linalg.norm(state.pos - env_state.target_pos)
        velocity_magnitude = jnp.linalg.norm(state.vel)
        angular_velocity_magnitude = jnp.linalg.norm(state.omega)

        upright_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
        quat_error = quat_product(quat_inverse(state.attitude), upright_quat)
        attitude_error = 2.0 * jnp.arccos(jnp.clip(jnp.abs(quat_error[0]), 0.0, 1.0))

        q = state.attitude
        current_yaw = jnp.arctan2(
            2.0 * (q[0] * q[3] + q[1] * q[2]),
            1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2),
        )
        yaw_error = jnp.abs(jnp.arctan2(
            jnp.sin(current_yaw - env_state.target_yaw),
            jnp.cos(current_yaw - env_state.target_yaw),
        ))

        diff = jnp.abs(state.pos[:2] - env_state.obstacle_pos)
        in_collision = (env_state.obstacle_half_size - jnp.max(diff) > 0.0).astype(jnp.float32)

        return {
            "distance_to_target": distance_to_target,
            "velocity_magnitude": velocity_magnitude,
            "angular_velocity_magnitude": angular_velocity_magnitude,
            "attitude_error": attitude_error,
            "yaw_error": yaw_error,
            "in_collision": in_collision,
            "pos_x": state.pos[0],
            "pos_y": state.pos[1],
        }

    # ── spaces / metadata ─────────────────────────────────────────────────

    @property
    def name(self):
        return "Drone-obstacle"

    # num_actions, action_space, observation_space inherited from DroneV3

    def state_space(self, params):
        return spaces.Dict({
            "state": spaces.Box(-jnp.finfo(jnp.float32).max, jnp.finfo(jnp.float32).max, (13,), jnp.float32),
            "target_pos": spaces.Box(-jnp.finfo(jnp.float32).max, jnp.finfo(jnp.float32).max, (3,), jnp.float32),
            "obstacle_pos": spaces.Box(-jnp.finfo(jnp.float32).max, jnp.finfo(jnp.float32).max, (2,), jnp.float32),
            "last_input": spaces.Box(-jnp.finfo(jnp.float32).max, jnp.finfo(jnp.float32).max, (4,), jnp.float32),
            "last_commanded_action": spaces.Box(-jnp.finfo(jnp.float32).max, jnp.finfo(jnp.float32).max, (4,), jnp.float32),
            "time": spaces.Discrete(params.max_steps_in_episode),
        })

    # ── plotting ──────────────────────────────────────────────────────────

    @staticmethod
    def plot_rollouts(series_list: List[dict], output: Path | None = None) -> None:
        """Top-down XY plot with obstacle and collision statistics."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        plt.rcdefaults()
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(2, 4, height_ratios=[2, 1])

        ax_xy = fig.add_subplot(gs[0, :2])
        ax_dist = fig.add_subplot(gs[0, 2:])
        axes_act = [fig.add_subplot(gs[1, i]) for i in range(4)]
        act_labels = ["Thrust", "Torque X", "Torque Y", "Torque Z"]

        _defaults = EnvParams()
        min_input = np.asarray(_defaults.min_input)
        max_input = np.asarray(_defaults.max_input)

        distance_series = []
        collision_counts = []
        default_dt = 0.02

        for series in series_list:
            _raw_state = series.get("state", None)
            state_data = np.asarray(_raw_state) if _raw_state is not None else np.array([])
            if state_data.size == 0:
                continue
            pos = state_data[:, :3]

            is_nominal = series.get("is_nominal", False)
            traj_color = "red" if is_nominal else "grey"
            traj_alpha = 0.9 if is_nominal else 0.4
            traj_lw = 1.5 if is_nominal else 0.8
            traj_zorder = 6 if is_nominal else 3

            ax_xy.plot(pos[:, 0], pos[:, 1], color=traj_color, alpha=traj_alpha,
                       linewidth=traj_lw, zorder=traj_zorder)
            ax_xy.scatter(pos[0, 0], pos[0, 1], marker="o", color="green", s=20, zorder=10)
            ax_xy.scatter(pos[-1, 0], pos[-1, 1], marker="s", color="blue", s=20, zorder=10)

            if "target_pos" in series:
                target = np.asarray(series["target_pos"])
                if target.ndim >= 2:
                    target = target[0]
                ax_xy.scatter(target[0], target[1], marker="x", color="red", s=50, zorder=10)

            if "obstacle_pos" in series:
                obs_pos = np.asarray(series["obstacle_pos"])
                if obs_pos.ndim >= 2:
                    obs_pos = obs_pos[0]
                half = float(np.asarray(series.get("obstacle_half_size", 0.25)).flat[0])
                rect = Rectangle(
                    (obs_pos[0] - half, obs_pos[1] - half), 2 * half, 2 * half,
                    linewidth=1.5, edgecolor="black", facecolor="gray", alpha=0.6, zorder=4,
                )
                ax_xy.add_patch(rect)

            actions = np.asarray(series.get("actions", []))
            if actions.size > 0:
                for idx, ax in enumerate(axes_act):
                    if idx < actions.shape[1]:
                        ax.plot(np.arange(actions.shape[0]), actions[:, idx],
                                color=traj_color, alpha=traj_alpha, linewidth=traj_lw, zorder=traj_zorder)
                    ax.set_xlabel("t")
                    ax.set_ylabel(act_labels[idx])
                    ax.set_ylim(float(min_input[idx]), float(max_input[idx]))
                    ax.grid(True, alpha=0.3)

            eval_metrics = series.get("eval_metrics", {})
            series_dt = float(series.get("dt", default_dt))
            if "distance_to_target" in eval_metrics:
                dist = np.asarray(eval_metrics["distance_to_target"]).flatten()
                distance_series.append({"values": dist, "dt": series_dt})
            if "in_collision" in eval_metrics:
                coll = np.asarray(eval_metrics["in_collision"]).flatten()
                collision_counts.append(int(coll.sum()))

        ax_xy.set_aspect("equal")
        ax_xy.set_xlim(float(_defaults.env_min_limits[0]), float(_defaults.env_max_limits[0]))
        ax_xy.set_ylim(float(_defaults.env_min_limits[1]), float(_defaults.env_max_limits[1]))
        ax_xy.set_title("Top-down view (XY)")
        ax_xy.set_xlabel("x [m]")
        ax_xy.set_ylabel("y [m]")
        ax_xy.grid(True, alpha=0.3)

        if collision_counts:
            n_ep = len(collision_counts)
            n_coll = sum(1 for c in collision_counts if c > 0)
            ax_xy.text(
                0.02, 0.98,
                f"Collisions: {n_coll}/{n_ep} episodes\n"
                f"Mean collision steps: {np.mean(collision_counts):.1f}",
                transform=ax_xy.transAxes, verticalalignment="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
            )

        if distance_series:
            max_len = max(d["values"].shape[0] for d in distance_series)
            padded = np.array([
                np.pad(d["values"], (0, max_len - d["values"].shape[0]), mode="edge")
                for d in distance_series
            ])
            mean_dist = np.mean(padded, axis=0)
            std_dist = np.std(padded, axis=0)
            ref_dt = distance_series[0]["dt"]
            timesteps = np.arange(max_len) * ref_dt
            for entry in distance_series:
                ax_dist.plot(np.arange(entry["values"].shape[0]) * entry["dt"],
                             entry["values"], color="blue", alpha=0.2, linewidth=0.8)
            ax_dist.plot(timesteps, mean_dist, color="blue", linewidth=2)
            ax_dist.fill_between(timesteps, mean_dist - std_dist, mean_dist + std_dist,
                                 alpha=0.3, color="blue")
            ax_dist.set_ylabel("Distance to target [m]")
            ax_dist.text(
                0.02, 0.98,
                f"Final: {mean_dist[-1]:.3f} ± {std_dist[-1]:.3f} m",
                transform=ax_dist.transAxes, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
            )
        ax_dist.set_title("Distance to target")
        ax_dist.set_xlabel("Time [s]")
        ax_dist.grid(True, alpha=0.3)

        plt.tight_layout()
        if output is None:
            plt.show()
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=150)
            plt.close(fig)

