#!/usr/bin/env python3
"""
Run many rollouts of the learned PPO policy from different initial conditions
and save full system trajectories.

This script:
- Loads policy parameters and config from one or more run directories
  (e.g., runs/latest or runs/ppo_jax__YYYYMMDD-HHMM).
- Reconstructs the ActorCritic model from ppo.py.
- Builds an evaluation environment that mirrors training wrappers (incl. normalization).
- Generates multiple episodes (rollouts) from random initial conditions
  by using different PRNG keys for env.reset.
  Writes a summary CSV, under <run_dir>/rollouts/<timestamp>/.

Usage:
  python -m sensitivity_rl.sensitivity_rl.rollouts \
      --run_dirs runs/latest \
      --num_rollouts 100 \
      --batch_size 256 \
      --max_steps 0 \
      --deterministic \
      --seed 0

Notes:
- max_steps=0 uses the environment default max_steps_in_episode.
- Deterministic actions (policy mean) by default. If omitted, actions are sampled.
- Requires that the run directory contains:
  - params.msgpack
  - config.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Local imports from the package
from ppo import ActorCritic
from envs import Pendulum
from envs.wrappers import (
    GymnaxWrapper,
    VecEnv,
    ClipAction,
    NormalizeVecObservation,
    NormalizeVecReward,
)


@dataclass
class RolloutConfig:
    num_rollouts: int
    batch_size: int
    max_steps: int  # 0 means use env default
    deterministic: bool
    seed: int
    plot: bool
    plot_format: str  # e.g., "png"


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_run_artifacts(run_dir: Path) -> Tuple[Dict, bytes]:
    cfg_path = run_dir / "config.json"
    params_path = run_dir / "params.msgpack"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json in {run_dir}")
    if not params_path.exists():
        raise FileNotFoundError(f"Missing params.msgpack in {run_dir}")
    with cfg_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    with params_path.open("rb") as f:
        params_bytes = f.read()
    return config, params_bytes


def _build_env_from_config(config: Dict) -> Tuple[GymnaxWrapper, object]:
    """Construct evaluation env with wrappers consistent with training."""
    env = Pendulum()
    env_params = env.default_params
    env = GymnaxWrapper(env)
    # No LogWrapper for evaluation rollouts to avoid metric interference
    env = ClipAction(env)
    env = VecEnv(env)
    # Apply normalization wrappers if used during training
    if bool(config.get("NORMALIZE_ENV", False)):
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, float(config.get("GAMMA", 0.99)))
    return env, env_params


def _unwrap_base_env_state(env_state):
    """
    Unwrap nested wrapper env_state to reach the base environment's EnvState.
    Works with wrapper dataclasses that carry `env_state` as an attribute.
    """
    # Some wrappers wrap the state into dataclasses with an env_state field
    # Drill down until we reach the base env state (which has `state` and `time`).
    while hasattr(env_state, "env_state"):
        env_state = env_state.env_state
    return env_state


def _init_model_and_params(config: Dict, env, env_params, params_bytes: bytes):
    # Build model with activation from training config
    action_dim = env.action_space(env_params).shape[0]
    model = ActorCritic(action_dim=action_dim, activation=config.get("ACTIVATION", "tanh"))
    # Create a dummy input to initialize parameter structure
    dummy_obs = jnp.zeros(env.observation_space(env_params).shape)
    params = model.init(jax.random.PRNGKey(0), dummy_obs)
    # Load trained parameters
    params = serialization.from_bytes(params, params_bytes)
    return model, params


def _plot_all_trajectories(
    out_dir: Path,
    all_theta: List[np.ndarray],
    all_theta_dot: List[np.ndarray],
    all_action: List[np.ndarray],
    all_reward: List[np.ndarray],
    all_value: List[np.ndarray],
    fmt: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)

    # Theta over time
    for theta in all_theta:
        t = np.arange(theta.shape[0])
        axes[0, 0].plot(t, theta, alpha=0.5)
    axes[0, 0].set_title("Theta")
    axes[0, 0].set_xlabel("t")
    axes[0, 0].set_ylabel("theta")
    axes[0, 0].grid(True, alpha=0.3)

    # Theta dot over time
    for thd in all_theta_dot:
        t = np.arange(thd.shape[0])
        axes[0, 1].plot(t, thd, alpha=0.5, color="tab:orange")
    axes[0, 1].set_title("Theta dot")
    axes[0, 1].set_xlabel("t")
    axes[0, 1].set_ylabel("theta_dot")
    axes[0, 1].grid(True, alpha=0.3)

    # Action over time (1D)
    for a in all_action:
        a1d = a[:, 0] if a.ndim > 1 else a
        t = np.arange(a1d.shape[0])
        axes[0, 2].plot(t, a1d, alpha=0.5, color="tab:green")
    axes[0, 2].set_title("Action")
    axes[0, 2].set_xlabel("t")
    axes[0, 2].set_ylabel("u")
    axes[0, 2].grid(True, alpha=0.3)

    # Reward over time
    for r in all_reward:
        t = np.arange(r.shape[0])
        axes[1, 0].plot(t, r, alpha=0.5, color="tab:red")
    axes[1, 0].set_title("Reward")
    axes[1, 0].set_xlabel("t")
    axes[1, 0].set_ylabel("r")
    axes[1, 0].grid(True, alpha=0.3)

    # Value estimate over time
    for v in all_value:
        t = np.arange(v.shape[0])
        axes[1, 1].plot(t, v, alpha=0.5, color="tab:purple")
    axes[1, 1].set_title("Value estimate")
    axes[1, 1].set_xlabel("t")
    axes[1, 1].set_ylabel("V")
    axes[1, 1].grid(True, alpha=0.3)

    # Phase portrait
    for theta, thd in zip(all_theta, all_theta_dot):
        axes[1, 2].plot(theta, thd, alpha=0.5, color="tab:brown")
    axes[1, 2].set_title("Phase portrait")
    axes[1, 2].set_xlabel("theta")
    axes[1, 2].set_ylabel("theta_dot")
    axes[1, 2].grid(True, alpha=0.3)

    fig.suptitle("Trajectories")
    fig.savefig(out_dir / f"trajectories.{fmt}", dpi=150)
    plt.show()
    plt.close(fig)

def _rollout_batch(
    model,
    params,
    env,
    env_params,
    batch_size: int,
    max_steps: int,
    rng_key: jax.Array,
    deterministic: bool,
    global_traj_offset: int,
    out_dir: Path,
    all_theta: list,
    all_theta_dot: list,
    all_action: list,
    all_reward: list,
    all_value: list,
) -> Tuple[int, List[Tuple[int, float, int, float, float, float, float]]]:
    """
    Run a batch of parallel rollouts and save per-trajectory .npz files.
    Returns:
      - number of trajectories saved
      - list of summary tuples:
        (traj_id, ep_return, ep_length, start_theta, start_theta_dot, end_theta, end_theta_dot)
    """
    # Reset environments with different keys
    reset_keys = jax.random.split(rng_key, batch_size)
    obs, env_state = env.reset(reset_keys, env_params)

    # Convert max_steps=0 to env default horizon
    if max_steps <= 0:
        max_steps = int(getattr(env_params, "max_steps_in_episode", 200))

    # Storage per env (ragged)
    per_env_obs: List[List[np.ndarray]] = [[] for _ in range(batch_size)]
    per_env_act: List[List[np.ndarray]] = [[] for _ in range(batch_size)]
    per_env_rew: List[List[float]] = [[] for _ in range(batch_size)]
    per_env_val: List[List[float]] = [[] for _ in range(batch_size)]
    per_env_theta: List[List[float]] = [[] for _ in range(batch_size)]
    per_env_theta_dot: List[List[float]] = [[] for _ in range(batch_size)]

    # Track start state for summary
    base_state0 = _unwrap_base_env_state(env_state)
    # base_state0.state fields are arrays with shape (batch_size,)
    theta0 = np.asarray(base_state0.state.theta)
    theta_dot0 = np.asarray(base_state0.state.theta_dot)

    done_mask = np.zeros((batch_size,), dtype=bool)
    steps = 0

    rng = rng_key
    while steps < (max_steps-1) and not bool(done_mask.all()):
        # Policy forward
        pi, value = model.apply(params, obs)
        if deterministic:
            action = pi.mean()
        else:
            rng, ak = jax.random.split(rng)
            action = pi.sample(seed=ak)

        # Step env
        rng, sk = jax.random.split(rng)
        step_keys = jax.random.split(sk, batch_size)
        next_obs, next_state, reward, done, info = env.step(step_keys, env_state, action, env_params)

        # Unwrap base state to access true system state
        base_state = _unwrap_base_env_state(next_state)
        # Convert tensors to numpy for saving
        obs_np = np.asarray(obs)
        act_np = np.asarray(action)
        rew_np = np.asarray(reward)
        val_np = np.asarray(value)
        th_np = np.asarray(base_state.state.theta)
        thd_np = np.asarray(base_state.state.theta_dot)
        done_np = np.asarray(done).astype(bool)

        # Append only for active envs
        for i in range(batch_size):
            if not done_mask[i]:
                per_env_obs[i].append(obs_np[i])
                per_env_act[i].append(act_np[i])
                per_env_rew[i].append(float(rew_np[i]))
                # value is per-env scalar
                per_env_val[i].append(float(val_np[i]))
                per_env_theta[i].append(float(th_np[i]))
                per_env_theta_dot[i].append(float(thd_np[i]))

        # Update masks and states
        done_mask = np.logical_or(done_mask, done_np)
        obs, env_state = next_obs, next_state
        steps += 1

    # Save per-trajectory files
    summaries: List[Tuple[int, float, int, float, float, float, float]] = []
    saved = 0
    for i in range(batch_size):
        traj_id = global_traj_offset + i + 1
        # Skip empty trajectories (shouldn't happen unless batch_size > remaining)
        if len(per_env_rew[i]) == 0:
            continue
        traj_dir = out_dir
        _ensure_dir(traj_dir)
        traj_path = traj_dir / f"traj_{traj_id:05d}.npz"

        obs_arr = np.stack(per_env_obs[i], axis=0)
        act_arr = np.stack(per_env_act[i], axis=0)
        rew_arr = np.array(per_env_rew[i], dtype=float)
        val_arr = np.array(per_env_val[i], dtype=float)
        theta_arr = np.array(per_env_theta[i], dtype=float)
        theta_dot_arr = np.array(per_env_theta_dot[i], dtype=float)

        # np.savez_compressed(
        #     traj_path,
        #     obs=obs_arr,
        #     action=act_arr,
        #     reward=rew_arr,
        #     value=val_arr,
        #     theta=theta_arr,
        #     theta_dot=theta_dot_arr,
        # )
        # Accumulate for combined plot
        all_theta.append(theta_arr)
        all_theta_dot.append(theta_dot_arr)
        all_action.append(act_arr[:, 0] if act_arr.ndim > 1 else act_arr)
        all_reward.append(rew_arr)
        all_value.append(val_arr)

        ep_return = float(rew_arr.sum())
        ep_length = int(rew_arr.shape[0])
        summaries.append(
            (
                traj_id,
                ep_return,
                ep_length,
                float(theta0[i]),
                float(theta_dot0[i]),
                float(theta_arr[-1]),
                float(theta_dot_arr[-1]),
            )
        )
        saved += 1

    return saved, summaries


def run_rollouts_for_run(run_dir: Path, rcfg: RolloutConfig) -> None:
    config, params_bytes = _load_run_artifacts(run_dir)
    env, env_params = _build_env_from_config(config)
    model, params = _init_model_and_params(config, env, env_params, params_bytes)

    # Prepare output directory: <run_dir>/rollouts/YYYYMMDD-HHMM
    rollouts_dir = run_dir / "rollouts" / _now_stamp()
    _ensure_dir(rollouts_dir)

    # Persist metadata
    meta = {
        "run_dir": str(run_dir),
        "timestamp": _now_stamp(),
        "seed": rcfg.seed,
        "num_rollouts": rcfg.num_rollouts,
        "batch_size": rcfg.batch_size,
        "max_steps": rcfg.max_steps,
        "deterministic": rcfg.deterministic,
        "plot": rcfg.plot,
        "plot_format": rcfg.plot_format,
        "config": config,
    }
    with (rollouts_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # CSV summary header
    summary_path = rollouts_dir / "rollouts_summary.csv"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("traj_id,return,length,start_theta,start_theta_dot,end_theta,end_theta_dot\n")

    all_theta: List[np.ndarray] = []
    all_theta_dot: List[np.ndarray] = []
    all_action: List[np.ndarray] = []
    all_reward: List[np.ndarray] = []
    all_value: List[np.ndarray] = []
    total = rcfg.num_rollouts
    batch = rcfg.batch_size
    rng = jax.random.PRNGKey(rcfg.seed)
    saved_total = 0
    all_summaries: List[Tuple[int, float, int, float, float, float, float]] = []

    print(f"[rollouts] Starting: run={run_dir} total={total} batch={batch} deterministic={rcfg.deterministic}")
    for offset in range(0, total, batch):
        n_this = min(batch, total - offset)
        rng, subkey = jax.random.split(rng)
        # Run one batch; note that we pass batch_size=n_this for the last partial batch
        saved, summaries = _rollout_batch(
            model=model,
            params=params,
            env=env,
            env_params=env_params,
            batch_size=n_this,
            max_steps=rcfg.max_steps,
            rng_key=subkey,
            deterministic=rcfg.deterministic,
            global_traj_offset=offset,
            out_dir=rollouts_dir,
            all_theta=all_theta,
            all_theta_dot=all_theta_dot,
            all_action=all_action,
            all_reward=all_reward,
            all_value=all_value,
        )
        saved_total += saved
        all_summaries.extend(summaries)
        print(f"[rollouts] Progress: {saved_total}/{total} saved")

    # Write summaries
    with summary_path.open("a", encoding="utf-8") as f:
        for s in all_summaries:
            traj_id, ep_ret, ep_len, th0, thd0, thN, thdN = s
            f.write(f"{traj_id},{ep_ret},{ep_len},{th0},{thd0},{thN},{thdN}\n")

    if rcfg.plot:
        _plot_all_trajectories(rollouts_dir, all_theta, all_theta_dot, all_action, all_reward, all_value, rcfg.plot_format)
    print(f"[rollouts] Done. Saved {saved_total} trajectories to {rollouts_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate many rollouts from learned policies across random initial conditions.")
    p.add_argument(
        "--run_dirs",
        nargs="+",
        default=["runs/latest"],
        help="One or more run directories containing params.msgpack and config.json (default: runs/latest).",
    )
    p.add_argument(
        "--num_rollouts",
        type=int,
        default=100,
        help="Total number of trajectories to generate per run directory.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Number of parallel environments per step.",
    )
    p.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Max steps per episode (0 uses env default max_steps_in_episode).",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Use policy mean action instead of sampling.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base PRNG seed for generating initial conditions and actions.",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="If set, save per-trajectory plots (theta, theta_dot, action, reward, value, phase portrait).",
    )

    p.add_argument(
        "--plot_format",
        type=str,
        default="png",
        help="Image format for trajectory plots (e.g., png, pdf).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rcfg = RolloutConfig(
        num_rollouts=args.num_rollouts,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        deterministic=args.deterministic,
        seed=args.seed,
        plot=args.plot,
        plot_format=args.plot_format,
    )

    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str).resolve()
        if not run_dir.exists():
            print(f"[rollouts] Skipping: run_dir does not exist: {run_dir}")
            continue
        try:
            run_rollouts_for_run(run_dir, rcfg)
        except Exception as e:
            print(f"[rollouts] Error processing {run_dir}: {e}")


if __name__ == "__main__":
    main()
