#!/usr/bin/env python3
"""Sequential rollout script for trained PPO policies.

- Reconstruct the evaluation environment from the saved config.
- Load the trained ActorCritic parameters.
- Run rollouts.
- Save the environment state trajectory and the action sequence for
    each episode under the respective run directory.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import distrax
from flax import serialization
from flax.linen.initializers import constant, orthogonal
from envs import Pendulum, DroneV0
from envs.wrappers import GymnaxWrapper, ClipAction, SensitivityWrapper


ENV_REGISTRY = {
    "pendulum": Pendulum,
    "drone_v0": DroneV0,
}


@dataclass
class RolloutConfig:
    num_rollouts: int
    max_steps: int
    deterministic: bool
    seed: int


class ActorCriticEval(nn.Module):
    action_dim: int
    activation: str = "tanh"
    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256

    @nn.compact
    def __call__(self, x):
        activation_fn = nn.relu if self.activation == "relu" else nn.tanh

        actor = nn.Dense(
            self.actor_hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="actor_dense1",
        )(x)
        actor = activation_fn(actor)
        actor = nn.Dense(
            self.actor_hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="actor_dense2",
        )(actor)
        actor = activation_fn(actor)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_output",
        )(actor)
        actor_log_std = self.param(
            "actor_log_std",
            nn.initializers.zeros,
            (self.action_dim,),
        )
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_log_std))

        critic = nn.Dense(
            self.critic_hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_dense1",
        )(x)
        critic = activation_fn(critic)
        critic = nn.Dense(
            self.critic_hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_dense2",
        )(critic)
        critic = activation_fn(critic)
        critic_value = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic_output",
        )(critic)

        return pi, jnp.squeeze(critic_value, axis=-1)


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_run_artifacts(run_dir: Path) -> Tuple[Dict[str, Any], bytes]:
    cfg_path = run_dir / "config.json"
    params_path = run_dir / "params.msgpack"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json in {run_dir}")
    if not params_path.exists():
        raise FileNotFoundError(f"Missing params.msgpack in {run_dir}")
    with cfg_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    params_bytes = params_path.read_bytes()
    return config, params_bytes


def _build_env_from_config(
    config: Dict[str, Any],
) -> Tuple[GymnaxWrapper, Any, str]:
    env_name = config.get("ENV_NAME", "pendulum").lower()
    if env_name not in ENV_REGISTRY:
        raise ValueError(
            f"Unknown environment '{env_name}'. Available: {sorted(ENV_REGISTRY)}"
        )

    env_cls = ENV_REGISTRY[env_name]
    env = env_cls()
    env_params = env.default_params

    wrapped_env = GymnaxWrapper(env)
    wrapped_env = ClipAction(wrapped_env)

    # if bool(config.get("SENSITIVITY", False)):
    #     regularization = float(
    #         config.get("SENSITIVITY_REGULARIZATION_WEIGHT", 0.0)
    #     )
    #     wrapped_env = SensitivityWrapper(
    #         wrapped_env, regularization_weight=regularization
    #     )

    return wrapped_env, env_params, env_name


def _unwrap_base_env_state(env_state: Any) -> Any:
    while hasattr(env_state, "env_state"):
        env_state = env_state.env_state
    return env_state


def _record_state(state_tracks, env_state: Any) -> None:
    base_state = _unwrap_base_env_state(env_state)
    components: Dict[str, np.ndarray] = {}

    model_state = getattr(base_state, "state", None)
    if model_state is not None:
        if hasattr(model_state, "__dataclass_fields__"):
            for name in model_state.__dataclass_fields__:
                components[f"state_{name}"] = np.asarray(
                    getattr(model_state, name)
                )
        else:
            components["state"] = np.asarray(model_state)

    if hasattr(base_state, "time"):
        components["time"] = np.asarray(base_state.time)

    if hasattr(base_state, "target_pos"):
        components["target_pos"] = np.asarray(base_state.target_pos)

    for key, value in components.items():
        state_tracks[key].append(np.asarray(value))


def _convert_state_tracks(state_tracks) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    for key, values in state_tracks.items():
        arrays[key] = np.stack([np.asarray(v) for v in values], axis=0)
    return arrays


def _rollout_episode(
    env,
    env_params,
    model,
    params,
    rng_key: jax.Array,
    max_steps: int,
    deterministic: bool,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], int]:
    rng_reset, rng = jax.random.split(rng_key)
    obs, env_state = env.reset(rng_reset, env_params)

    env_state = env_state.replace(target_pos=jnp.array([0.0, 0.0, 0.5]))

    horizon = (
        max_steps
        if max_steps > 0
        else int(getattr(env_params, "max_steps_in_episode", 200))
    )

    state_tracks = defaultdict(list)
    _record_state(state_tracks, env_state)

    action_shape = env.action_space(env_params).shape
    actions_list = []

    done = False

    for steps in range(horizon):
        pi, _ = model.apply(params, obs)
        if deterministic:
            action = pi.mean()
        else:
            rng, act_key = jax.random.split(rng)
            action = pi.sample(seed=act_key)

        rng, step_key = jax.random.split(rng)
        obs, env_state, reward, done_flag, _ = env.step(
            step_key, env_state, action, env_params
        )

        actions_list.append(env_state.last_u)

        done = bool(np.asarray(done_flag))
        _record_state(state_tracks, env_state)

    if actions_list:
        actions = np.stack(actions_list, axis=0)
    else:
        actions = np.empty((0,) + action_shape, dtype=np.float32)

    state_arrays = _convert_state_tracks(state_tracks)
    return actions, state_arrays, steps + 1


def _save_trajectory(
    path: Path, actions: np.ndarray, state_arrays: Dict[str, np.ndarray]
) -> None:
    payload = {"actions": actions}
    payload.update(state_arrays)
    np.savez_compressed(path, **payload)


def _init_model_and_params(
    config: Dict[str, Any], env, env_params, params_bytes: bytes
):
    params = serialization.msgpack_restore(params_bytes)

    actor_hidden = 256
    critic_hidden = 256
    try:
        actor_hidden = int(params["params"]["actor_dense1"]["kernel"].shape[1])
    except (KeyError, AttributeError, TypeError, IndexError):
        actor_hidden = 256
    try:
        critic_hidden = int(
            params["params"]["critic_dense1"]["kernel"].shape[1]
        )
    except (KeyError, AttributeError, TypeError, IndexError):
        critic_hidden = 256

    action_dim = env.action_space(env_params).shape[0]
    model = ActorCriticEval(
        action_dim=action_dim,
        activation=config.get("ACTIVATION", "tanh"),
        actor_hidden_dim=actor_hidden,
        critic_hidden_dim=critic_hidden,
    )
    return model, params


def run_rollouts_for_run(run_dir: Path, rcfg: RolloutConfig) -> None:
    config, params_bytes = _load_run_artifacts(run_dir)
    env, env_params, env_name = _build_env_from_config(config)
    model, params = _init_model_and_params(
        config, env, env_params, params_bytes
    )

    if bool(config.get("SENSITIVITY", False)) and hasattr(env, "set_actor"):
        env.set_actor(model, params)

    rollouts_dir = run_dir / "rollouts"
    _ensure_dir(rollouts_dir)

    meta = {
        "run_dir": str(run_dir),
        "timestamp": _now_stamp(),
        "seed": rcfg.seed,
        "num_rollouts": rcfg.num_rollouts,
        "max_steps": rcfg.max_steps,
        "deterministic": rcfg.deterministic,
        "env_name": env_name,
        "format_version": 1,
        "config": config,
    }
    with (rollouts_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    summary_path = rollouts_dir / "rollouts_summary.csv"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("traj_id,length\n")

    rng = jax.random.PRNGKey(rcfg.seed)

    for idx in range(rcfg.num_rollouts):
        rng, episode_key = jax.random.split(rng)
        actions, state_arrays, steps = _rollout_episode(
            env=env,
            env_params=env_params,
            model=model,
            params=params,
            rng_key=episode_key,
            max_steps=rcfg.max_steps,
            deterministic=rcfg.deterministic,
        )

        traj_id = idx + 1
        traj_file = rollouts_dir / f"traj_{traj_id:05d}.npz"
        _save_trajectory(traj_file, actions, state_arrays)

        with summary_path.open("a", encoding="utf-8") as f:
            f.write(f"{traj_id},{steps}\n")

        print(
            f"[rollouts] Saved trajectory {traj_id}/{rcfg.num_rollouts} -> {traj_file}"
        )

    print(
        f"[rollouts] Completed {rcfg.num_rollouts} trajectories for {run_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate sequential rollouts from learned policies."
    )
    parser.add_argument(
        "--run_dirs",
        nargs="+",
        default=["runs/latest"],
        help="One or more run directories containing params.msgpack and config.json.",
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=10,
        help="Number of rollouts to generate per run directory.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Maximum steps per episode (0 uses the environment default).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use policy mean action instead of sampling from the distribution.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base PRNG seed for rollouts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rcfg = RolloutConfig(
        num_rollouts=args.num_rollouts,
        max_steps=args.max_steps,
        deterministic=args.deterministic,
        seed=args.seed,
    )

    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str).resolve()
        if not run_dir.exists():
            print(f"[rollouts] Skipping missing run directory: {run_dir}")
            continue
        try:
            run_rollouts_for_run(run_dir, rcfg)
        except Exception as exc:  # pragma: no cover - guardrail for CLI use
            print(f"[rollouts] Error processing {run_dir}: {exc}")


if __name__ == "__main__":
    main()
