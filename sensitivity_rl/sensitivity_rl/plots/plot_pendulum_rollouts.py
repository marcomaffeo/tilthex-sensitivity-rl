#!/usr/bin/env python3
"""Plot pendulum rollouts saved by sensitivity_rl.rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
PREFERRED_STATE_KEYS = {
    "theta": ("model_state_theta", "state_theta"),
    "theta_dot": ("model_state_theta_dot", "state_theta_dot"),
}


def _load_meta(rollouts_dir: Path) -> dict:
    meta_path = rollouts_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {rollouts_dir}")
    with meta_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _iter_trajectory_files(rollouts_dir: Path) -> Iterable[Path]:
    return sorted(rollouts_dir.glob("traj_*.npz"))


def _get_array(data: np.lib.npyio.NpzFile, candidates: Iterable[str], label: str) -> np.ndarray:
    for key in candidates:
        if key in data:
            return np.asarray(data[key])
    raise KeyError(
        f"Required pendulum state '{label}' not found. Checked keys: {list(candidates)}"
    )


def _extract_series(npz_file: Path) -> dict:
    data = np.load(npz_file)

    theta = _get_array(data, PREFERRED_STATE_KEYS["theta"], "theta")
    theta_dot = _get_array(data, PREFERRED_STATE_KEYS["theta_dot"], "theta_dot")
    actions = np.asarray(data["actions"])

    # Older rollouts may still store rewards/values; include when present.
    rewards = np.asarray(data["rewards"]) if "rewards" in data else None
    values = np.asarray(data["values"]) if "values" in data else None

    return {
        "theta": theta,
        "theta_dot": theta_dot,
        "actions": actions,
        "rewards": rewards,
        "values": values,
    }


def plot_pendulum_rollouts(series_list: List[dict], output: Path | None = None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)

    for series in series_list:
        timesteps = np.arange(series["theta"].shape[0])
        axes[0, 0].plot(timesteps, series["theta"], alpha=0.4)
        axes[0, 1].plot(timesteps, series["theta_dot"], alpha=0.4)
        action = series["actions"][:, 0] if series["actions"].ndim > 1 else series["actions"]
        axes[1, 0].plot(np.arange(action.shape[0]), action, alpha=0.4)
        axes[1, 1].plot(series["theta"], series["theta_dot"], alpha=0.4)

    axes[0, 0].set_title("Theta")
    axes[0, 0].set_xlabel("t")
    axes[0, 0].set_ylabel("theta")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title("Theta dot")
    axes[0, 1].set_xlabel("t")
    axes[0, 1].set_ylabel("theta_dot")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title("Action")
    axes[1, 0].set_xlabel("t")
    axes[1, 0].set_ylabel("u")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title("Phase portrait")
    axes[1, 1].set_xlabel("theta")
    axes[1, 1].set_ylabel("theta_dot")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Pendulum trajectories")

    if output is None:
        plt.show()
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot pendulum rollouts produced by sensitivity_rl.rollouts.")
    parser.add_argument("rollouts_dir", type=Path, help="Directory containing trajectory .npz files and meta.json")
    parser.add_argument(
        "--max_traj",
        type=int,
        default=50,
        help="Maximum number of trajectories to include (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the figure instead of displaying it.",
    )
    args = parser.parse_args()

    rollouts_dir = args.rollouts_dir
    if not rollouts_dir.exists():
        raise FileNotFoundError(f"Rollouts directory not found: {rollouts_dir}")

    meta = _load_meta(rollouts_dir)
    env_name = meta.get("env_name", "pendulum")
    if env_name.lower() != "pendulum":
        raise ValueError(
            f"This plotting script is for the pendulum environment. Found env_name={env_name!r} in meta.json"
        )

    files = list(_iter_trajectory_files(rollouts_dir))
    if not files:
        raise FileNotFoundError(f"No trajectory files found in {rollouts_dir}")

    series_to_plot = []
    for path in files[: args.max_traj]:
        try:
            series_to_plot.append(_extract_series(path))
        except KeyError as exc:  # pragma: no cover - guardrail for mixed directories
            print(f"[warn] Skipping {path.name}: {exc}")

    if not series_to_plot:
        raise RuntimeError("No valid pendulum trajectories were loaded. Nothing to plot.")

    plot_pendulum_rollouts(series_to_plot, args.output)


if __name__ == "__main__":
    main()
