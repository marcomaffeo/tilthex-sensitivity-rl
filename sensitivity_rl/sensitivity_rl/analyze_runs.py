#!/usr/bin/env python3
"""
Analyze PPO JAX training runs and generate plots/CSVs from saved artifacts.

This script expects runs produced by sensitivity_rl/sensitivity_rl/ppo.py,
which saves:
- params.msgpack (Flax model params; not used here)
- config.json (contains training configuration, including NUM_ENVS)
- metrics_raw.npz (raw metrics recorded during training):
  - returned_episode_returns
  - returned_episode_lengths
  - returned_episode (boolean flags)
  - timestep

Per-run outputs (saved next to the artifacts by default):
- episode_stats.csv: per-episode return, length, and global_step
- returns.png: return vs episode index (+ optional rolling mean)
- lengths.png: length vs episode index (+ optional rolling mean)
- returns_vs_steps.png: return vs global training step
- lengths_vs_steps.png: length vs global training step
- summary.json: basic statistics over episodic returns and lengths

If multiple runs are provided and --compare is set, the script also produces:
- returns_compare.png
- lengths_compare.png
- returns_vs_steps_compare.png
- lengths_vs_steps_compare.png

Usage examples:
- Analyze one run:
  python sensitivity_rl/sensitivity_rl/analyze_runs.py runs/ppo_jax__1700000000
- Analyze and smooth:
  python sensitivity_rl/sensitivity_rl/analyze_runs.py runs/ppo_jax__1700000000 --smooth 21
- Compare multiple runs:
  python sensitivity_rl/sensitivity_rl/analyze_runs.py runs/ppo_jax__*/ --compare --out_dir runs/compare

Dependencies:
- matplotlib
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")  # Headless backend for servers/CI
import matplotlib.pyplot as plt


@dataclass
class RunData:
    run_dir: Path
    config: Dict
    episode_returns: np.ndarray  # shape: [num_episodes]
    episode_lengths: np.ndarray  # shape: [num_episodes]
    episode_steps: np.ndarray    # global step at episode termination, [num_episodes]


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.size == 0:
        return x
    w = min(window, len(x))
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(x, kernel, mode="same")


def _load_run(run_dir: Path) -> RunData:
    # Validate paths
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    metrics_path = run_dir / "metrics_raw.npz"
    config_path = run_dir / "config.json"

    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics_raw.npz not found in {run_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {run_dir}")

    # Load config
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    # Load raw metrics
    data = np.load(metrics_path)
    ret = np.array(data["returned_episode_returns"]).reshape(-1)
    elen = np.array(data["returned_episode_lengths"]).reshape(-1)
    done = np.array(data["returned_episode"]).reshape(-1).astype(bool)
    tsteps = np.array(data["timestep"]).reshape(-1)

    # Filter completed episodes
    ep_returns = ret[done]
    ep_lengths = elen[done]
    # Convert per-env timesteps to global steps
    num_envs = int(config.get("NUM_ENVS", 1))
    ep_steps = tsteps[done] * num_envs

    return RunData(
        run_dir=run_dir,
        config=config,
        episode_returns=ep_returns.astype(float),
        episode_lengths=ep_lengths.astype(float),
        episode_steps=ep_steps.astype(int),
    )


def _save_episode_csv(run: RunData, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "episode_stats.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("episode,return,length,global_step\n")
        for i, (r, l, s) in enumerate(
            zip(run.episode_returns, run.episode_lengths, run.episode_steps), start=1
        ):
            f.write(f"{i},{float(r)},{int(l)},{int(s)}\n")
    return csv_path


def _save_summary(run: RunData, out_dir: Path) -> Path:
    summary = {
        "num_episodes": int(run.episode_returns.size),
        "avg_return": float(run.episode_returns.mean()) if run.episode_returns.size else None,
        "std_return": float(run.episode_returns.std(ddof=1)) if run.episode_returns.size > 1 else 0.0,
        "min_return": float(run.episode_returns.min()) if run.episode_returns.size else None,
        "max_return": float(run.episode_returns.max()) if run.episode_returns.size else None,
        "avg_length": float(run.episode_lengths.mean()) if run.episode_lengths.size else None,
        "min_length": int(run.episode_lengths.min()) if run.episode_lengths.size else None,
        "max_length": int(run.episode_lengths.max()) if run.episode_lengths.size else None,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary_path


def _plot_series(
    xs: np.ndarray,
    ys: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
    smooth_window: int = 0,
    color: Optional[str] = None,
    label: Optional[str] = None,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o", linewidth=1, label=label or ylabel, color=color)
    if smooth_window and len(ys) >= 3:
        rm = _rolling_mean(ys, smooth_window)
        plt.plot(xs, rm, linestyle="--", label=f"Rolling mean (w={smooth_window})")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_compare(
    runs: List[RunData],
    xs_accessor,
    ys_accessor,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
    smooth_window: int = 0,
) -> None:
    plt.figure(figsize=(9, 6))
    for run in runs:
        xs = xs_accessor(run)
        ys = ys_accessor(run)
        label = run.run_dir.name
        plt.plot(xs, ys, linewidth=1, alpha=0.85, label=label)
        if smooth_window and len(ys) >= 3:
            rm = _rolling_mean(ys, smooth_window)
            plt.plot(xs, rm, linestyle="--", alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def analyze_run(run_dir: Path, out_dir: Optional[Path], smooth: int) -> Path:
    run = _load_run(run_dir)
    save_dir = out_dir or run_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # CSV + summary
    _save_episode_csv(run, save_dir)
    _save_summary(run, save_dir)

    # Plots per-episode index
    ep_idx = np.arange(1, run.episode_returns.size + 1)
    _plot_series(
        ep_idx,
        run.episode_returns,
        xlabel="Episode",
        ylabel="Return",
        title=f"Returns ({run.run_dir.name})",
        out_path=save_dir / "returns.png",
        smooth_window=smooth,
    )
    print(f"[analyze] Saved artifacts to: {save_dir}")
    return save_dir


def compare_runs(run_dirs: List[Path], out_dir: Path, smooth: int) -> Path:
    runs = []
    for rd in run_dirs:
        try:
            runs.append(_load_run(rd))
        except Exception as e:
            print(f"[compare] Skipping {rd}: {e}")

    if not runs:
        raise RuntimeError("No valid runs to compare.")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Compare by episode index
    _plot_compare(
        runs,
        xs_accessor=lambda r: np.arange(1, r.episode_returns.size + 1),
        ys_accessor=lambda r: r.episode_returns,
        xlabel="Episode",
        ylabel="Return",
        title="Returns (per-episode comparison)",
        out_path=out_dir / "returns_compare.png",
        smooth_window=smooth,
    )
    print(f"[compare] Saved comparison artifacts to: {out_dir}")
    return out_dir


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze PPO JAX runs and generate plots/CSVs.")
    p.add_argument(
        "--run_dirs",
        nargs="+",
        default=["runs/latest"],
        help="One or more run directories (each containing metrics_raw.npz and config.json)",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output directory. If analyzing a single run, defaults to the run dir. "
             "If comparing runs, defaults to this path (or 'runs/compare' if omitted).",
    )
    p.add_argument(
        "--smooth",
        type=int,
        default=0,
        help="Window size for rolling mean (applied to per-episode plots). 0 disables smoothing.",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="If set and multiple run_dirs are provided, also generate comparison plots across runs.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    run_dirs = [Path(rd).resolve() for rd in args.run_dirs]

    # Per-run analysis
    if len(run_dirs) == 1 and not args.compare:
        out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dirs[0]
        analyze_run(run_dirs[0], out_dir, args.smooth)
        return

    # Analyze each run individually to ensure per-run outputs exist
    for rd in run_dirs:
        try:
            analyze_run(rd, None, args.smooth)
        except Exception as e:
            print(f"[analyze] Warning: failed to analyze {rd}: {e}")

    # Multi-run comparison
    if args.compare and len(run_dirs) > 1:
        if args.out_dir:
            compare_dir = Path(args.out_dir).resolve()
        else:
            compare_dir = Path("runs") / "compare"
        compare_runs(run_dirs, compare_dir, args.smooth)


if __name__ == "__main__":
    main()
