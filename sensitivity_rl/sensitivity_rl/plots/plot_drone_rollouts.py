#!/usr/bin/env python3
"""Plot quadrotor rollouts saved by sensitivity_rl.rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from flax import serialization

import matplotlib.pyplot as plt
import numpy as np

POSITION_KEYS = ("state_pos",)
TARGET_KEYS = ("target_pos",)


def _iter_trajectory_files(rollouts_dir: Path) -> Iterable[Path]:
    return sorted(rollouts_dir.glob("traj_*.msgpack"))


def _get_array(
    data, candidates: Iterable[str], label: str
) -> np.ndarray:
    for key in candidates:
        if key in data:
            return np.asarray(data[key])
    raise KeyError(
        f"Required quadrotor data '{label}' not found. Checked keys: {list(candidates)}"
    )


def _extract_series(to_load: Path) -> dict:

    data = serialization.msgpack_restore(to_load.read_bytes())

    positions = np.asarray(data["state"][:, :3])
    actions = np.asarray(data["actions"])

    series = {
        "positions": positions,
        "actions": actions,
    }

    try:
        target_traj = _get_array(data, TARGET_KEYS, "target position")
        series["target"] = (
            target_traj[0] if target_traj.ndim >= 2 else target_traj
        )
    except KeyError:
        pass

    return series


def plot_drone_rollouts(
    series_list: List[dict], rollout_dir: Path | None = None, output: Path | None = None
) -> None:
    plt.style.use("seaborn-v0_8")
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[2, 1])

    ax_path = fig.add_subplot(gs[0, :], projection="3d")
    axes_act = [fig.add_subplot(gs[1, i]) for i in range(4)]

    labels = ["Thrust", "Torque X", "Torque Y", "Torque Z"]

    for series in series_list:
        pos = series["positions"]
        timesteps = np.arange(pos.shape[0])
        ax_path.plot(pos[:, 0], pos[:, 1], pos[:, 2], alpha=0.6)

        if "target" in series:
            target = series["target"]
            ax_path.scatter(
                target[0], target[1], target[2], marker="x", color="red", s=30
            )

        ax_path.scatter(
            pos[0, 0],
            pos[0, 1],
            pos[0, 2],
            marker="o",
            color="green",
            s=30,
        )

        actions = series["actions"]
        for idx, ax in enumerate(axes_act):
            if idx < actions.shape[1]:
                ax.plot(np.arange(actions.shape[0]), actions[:, idx], alpha=0.6)
            ax.set_xlabel("t")
            ax.set_ylabel(labels[idx])
            ax.grid(True, alpha=0.3)

    ax_path.set_title("Quadrotor position trajectories")
    ax_path.set_xlabel("x [m]")
    ax_path.set_ylabel("y [m]")
    ax_path.set_zlabel("z [m]")
    ax_path.grid(True, alpha=0.3)

    if output is None:
        plt.show()
    else:
        rollout_dir.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(rollout_dir.parent / output, dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot quadrotor rollouts produced by sensitivity_rl.rollouts."
    )
    parser.add_argument(
        "--rollouts_dir",
        type=Path,
        default=Path("runs/latest/rollouts"),
        help="Directory containing trajectory .msgpack files and config.json",
    )
    parser.add_argument(
        "--max_traj",
        type=int,
        default=100,
        help="Maximum number of trajectories to include (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file to save the plot (if not provided, plots in with the same directory name as rollouts_dir)"
    )

    args = parser.parse_args()

    rollouts_dir = args.rollouts_dir
    if not rollouts_dir.exists():
        raise FileNotFoundError(f"Rollouts directory not found: {rollouts_dir}")

    env_name = "drone_v0"
    if env_name.lower() not in {"drone_v0", "quadrotor", "drone", "quad"}:
        raise ValueError(
            f"This plotting script is for the quadrotor environment. Found env_name={env_name!r} in meta.json"
        )

    files = list(_iter_trajectory_files(rollouts_dir))
    if not files:
        raise FileNotFoundError(f"No trajectory files found in {rollouts_dir}")

    series_to_plot = []
    for path in files[: args.max_traj]:
        try:
            series_to_plot.append(_extract_series(path))
        except (
            KeyError
        ) as exc:  # pragma: no cover - guardrail for mixed directories
            print(f"[warn] Skipping {path.name}: {exc}")

    if not series_to_plot:
        raise RuntimeError(
            "No valid quadrotor trajectories were loaded. Nothing to plot."
        )

    plot_drone_rollouts(series_to_plot, rollouts_dir, args.output)


if __name__ == "__main__":
    main()
