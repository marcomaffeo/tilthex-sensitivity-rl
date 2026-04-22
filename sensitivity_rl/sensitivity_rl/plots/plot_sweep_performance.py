"""
Compare rollout trajectories across seeds and conditions from a sweeps directory.

Scans a sweeps directory (default: sensitivity_rl/plots/sweeps/), groups experiment
directories by their configuration (excluding seed), and produces two figures:

  1. sweep_error_comparison.png
       Mean ± std of position error and yaw error over time, one line per condition.

  2. sweep_input_trajectories.png
       All action trajectories as thin lines, one subplot per action dimension,
       coloured by condition.

Directory names are parsed generically to discover algorithm, env, domain-
randomisation flag, seed, and other hyper-parameters.  Experiments are grouped by
all parameters except seed.

USAGE:
    pixi run python sensitivity_rl/plots/plot_sweep_performance.py
    pixi run python sensitivity_rl/plots/plot_sweep_performance.py \\
        --sweeps_dir sensitivity_rl/plots/sweeps --dt 0.02
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from flax import serialization


# ─── Trajectory loading ──────────────────────────────────────────────────────

def _load_traj(path: Path) -> Optional[Dict]:
    """Load a single trajectory msgpack file saved by flax.serialization."""
    try:
        return serialization.msgpack_restore(path.read_bytes())
    except Exception as exc:
        print(f"  Warning: could not load {path.name}: {exc}")
        return None


def load_rollouts_for_exp(exp_dir: Path) -> List[Dict]:
    """Return all trajectory dicts found in <exp_dir>/rollouts/."""
    rollouts_dir = exp_dir / "rollouts"
    if not rollouts_dir.exists():
        return []
    trajs = []
    for f in sorted(rollouts_dir.glob("*.msgpack")):
        data = _load_traj(f)
        if data is not None:
            trajs.append(data)
    return trajs


# ─── Directory-name parsing ──────────────────────────────────────────────────

# Known environment base names and version tokens
_ENV_BASES = {"drone", "halfcheetah", "pendulum", "reacher"}
_ENV_VERSIONS = {"v0", "v1", "v2", "v3", "v4", "v5", "obstacle", "flip"}


def parse_exp_dir_name(name: str) -> Dict:
    """Parse an experiment directory name into a flat parameter dict.

    Handles patterns like:
      exp_008_sppo_aug_grad_param_dual_drone_v3_srw0.1_sensitivity_num_steps50_seed0_domain_randomizationFalse
      exp_025_ppo_drone_v3_domain_randomizationTrue_seed0
    """
    params: Dict = {}

    # Strip leading exp_<number>_ prefix
    stripped = re.sub(r"^exp_\d+_", "", name)
    tokens = stripped.split("_")

    # Walk tokens to identify method (before env) and env, then parse the rest
    i = 0
    method_tokens: List[str] = []

    while i < len(tokens):
        tok = tokens[i]
        # Detect env: base_version pattern
        if tok in _ENV_BASES and (i + 1) < len(tokens) and tokens[i + 1] in _ENV_VERSIONS:
            params["env"] = f"{tok}_{tokens[i + 1]}"
            i += 2
            break
        method_tokens.append(tok)
        i += 1

    params["method"] = "_".join(method_tokens) if method_tokens else "unknown"

    # Parse remaining tokens
    while i < len(tokens):
        tok = tokens[i]

        # seed<N>
        m = re.fullmatch(r"seed(\d+)", tok)
        if m:
            params["seed"] = int(m.group(1))
            i += 1
            continue

        # domain_randomization<True|False>  (split across two tokens by '_')
        if tok == "domain" and (i + 1) < len(tokens):
            m2 = re.fullmatch(r"randomization(True|False)", tokens[i + 1])
            if m2:
                params["domain_randomization"] = m2.group(1) == "True"
                i += 2
                continue

        # <key><True|False>  (single token, e.g. sensitivityTrue)
        m = re.fullmatch(r"([a-zA-Z_]+)(True|False)", tok)
        if m:
            params[m.group(1)] = m.group(2) == "True"
            i += 1
            continue

        # <key><numeric>  (e.g. srw0.1, seed0, num_steps50)
        m = re.fullmatch(r"([a-zA-Z_]+)([-+]?\d*\.?\d+)", tok)
        if m:
            val_str = m.group(2)
            try:
                params[m.group(1)] = float(val_str) if "." in val_str else int(val_str)
            except ValueError:
                pass
            i += 1
            continue

        i += 1

    return params


def group_experiments(sweeps_dir: Path) -> Dict[Tuple, List[Path]]:
    """Group experiment directories by their config, excluding the seed."""
    groups: Dict[Tuple, List[Path]] = defaultdict(list)
    for exp_dir in sorted(sweeps_dir.iterdir()):
        if not exp_dir.is_dir() or not exp_dir.name.startswith("exp_"):
            continue
        params = parse_exp_dir_name(exp_dir.name)
        group_key = tuple(sorted((k, v) for k, v in params.items() if k != "seed"))
        groups[group_key].append(exp_dir)
    return groups


# ─── Label generation ────────────────────────────────────────────────────────

_METHOD_SHORT: Dict[str, str] = {
    "sppo_aug_grad_param_dual": "SPPO",
    "sppo_aug_grad_param": "SPPO",
    "sppo_aug": "SPPO-Aug",
    "sppo": "SPPO",
    "ppo": "PPO",
    "sac": "SAC",
    "ddpg": "DDPG",
}


def make_label(param_key: Tuple) -> str:
    """Human-readable label for a parameter group."""
    params = dict(param_key)
    method = params.get("method", "unknown")
    label = _METHOD_SHORT.get(method, method.replace("_", "-").upper())

    dr = params.get("domain_randomization", None)
    if dr is True:
        label += " + DR"

    return label


# ─── Metric helpers ───────────────────────────────────────────────────────────

def _collect_metric(trajs: List[Dict], key: str) -> List[np.ndarray]:
    """Extract a flat 1-D time series from eval_metrics across all trajectories."""
    series: List[np.ndarray] = []
    for traj in trajs:
        em = traj.get("eval_metrics", {})
        if key in em:
            arr = np.asarray(em[key]).flatten()
            series.append(arr)
    return series


def _collect_actions(trajs: List[Dict]) -> List[np.ndarray]:
    """Extract action arrays (T, n_act) from all trajectories."""
    out: List[np.ndarray] = []
    for traj in trajs:
        actions = traj.get("actions", None)
        if actions is None:
            # Try last_u as fallback
            actions = traj.get("last_u", None)
        if actions is not None:
            arr = np.asarray(actions)
            if arr.ndim == 2:
                out.append(arr)
    return out


def _pad_stack(series: List[np.ndarray]) -> Optional[np.ndarray]:
    """Pad all arrays to the same length and stack → (N, T)."""
    if not series:
        return None
    max_len = max(s.shape[0] for s in series)
    padded = np.stack(
        [np.pad(s, (0, max_len - s.shape[0]), mode="edge") for s in series]
    )
    return padded


def _rmse_with_std(stacked: np.ndarray):
    """Per-timestep RMSE and its std across rollouts.

    stacked: (N, T) array of scalar error values.
    Returns (rmse, std) each of shape (T,), where
      rmse[t] = sqrt(mean(stacked[:, t]**2))
      std[t]  = std of the per-rollout RMS contributions
    """
    sq = stacked ** 2
    rmse = np.sqrt(np.mean(sq, axis=0))
    # Std of individual squared errors, propagated through sqrt via delta method:
    # std(sqrt(mean)) ≈ std(x) / (2 * sqrt(mean(x)))  — but simpler to just show
    # the std of the per-rollout root values so the shading is interpretable.
    std = np.std(np.sqrt(sq), axis=0)
    return rmse, std


# ─── Plots ───────────────────────────────────────────────────────────────────

_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
_LAST_N = 10  # number of trailing samples used for the summary table


def _sort_key(param_key: Tuple):
    """Sort groups so SPPO-based methods always appear last."""
    label = make_label(param_key)
    is_sppo = label.upper().startswith("SPPO")
    return (int(is_sppo),) + param_key


def _end_stats(series: List[np.ndarray], last_n: int = _LAST_N) -> Tuple[float, float]:
    """Mean and std (ddof=1) of per-trajectory tail means (last `last_n` samples)."""
    tail_means = [float(np.mean(s[-min(last_n, len(s)):])) for s in series if len(s) > 0]
    if not tail_means:
        return float("nan"), float("nan")
    return float(np.mean(tail_means)), float(np.std(tail_means, ddof=min(1, len(tail_means) - 1)))


def _per_seed_mean(seed_trajs_list: List[List[Dict]], key: str) -> Optional[np.ndarray]:
    """Compute per-seed mean time series for a metric, then stack → (n_seeds, T).

    Each element of seed_trajs_list is the list of trajectories for one seed.
    Returns array of shape (n_seeds, T) or None if no data.
    """
    seed_means = []
    for trajs in seed_trajs_list:
        series = _collect_metric(trajs, key)
        stacked = _pad_stack(series)
        if stacked is not None:
            seed_means.append(np.mean(stacked, axis=0))
    if not seed_means:
        return None
    # Align lengths across seeds
    max_len = max(m.shape[0] for m in seed_means)
    aligned = np.stack(
        [np.pad(m, (0, max_len - m.shape[0]), mode="edge") for m in seed_means]
    )
    return aligned  # (n_seeds, T)


def _per_seed_action_mean(seed_trajs_list: List[List[Dict]], dim: int) -> Optional[np.ndarray]:
    """Compute per-seed mean for one action dimension → (n_seeds, T)."""
    seed_means = []
    for trajs in seed_trajs_list:
        series = [
            np.asarray(traj.get("actions", traj.get("last_u", [])))[:, dim]
            for traj in trajs
            if np.asarray(traj.get("actions", traj.get("last_u", []))).ndim == 2
            and np.asarray(traj.get("actions", traj.get("last_u", []))).shape[1] > dim
        ]
        stacked = _pad_stack(series)
        if stacked is not None:
            seed_means.append(np.mean(stacked, axis=0))
    if not seed_means:
        return None
    max_len = max(m.shape[0] for m in seed_means)
    return np.stack(
        [np.pad(m, (0, max_len - m.shape[0]), mode="edge") for m in seed_means]
    )  # (n_seeds, T)


def plot_error_comparison(
    groups: Dict[Tuple, List[Path]],
    output_path: Path,
    dt: float = 0.02,
) -> None:
    """Position error and yaw error: mean ± std across seeds, one line per condition."""
    fig, (ax_pos, ax_yaw) = plt.subplots(1, 2, figsize=(13, 5))

    _pos_inset_data: List[Tuple] = []  # (t, mean, std, color) for inset zoom

    for idx, (param_key, exp_dirs) in enumerate(sorted(groups.items(), key=lambda kv: _sort_key(kv[0]))):
        color = _COLORS[idx % len(_COLORS)]
        label = make_label(param_key)

        # One list of trajs per seed
        seed_trajs_list = [load_rollouts_for_exp(d) for d in exp_dirs]
        n_trajs = sum(len(t) for t in seed_trajs_list)
        if n_trajs == 0:
            print(f"  Warning: no trajectories for group '{label}', skipping.")
            continue
        print(f"  [{label}]: {len(exp_dirs)} seed(s), {n_trajs} trajectories total")

        # ── Position error ────────────────────────────────────────────────
        per_seed_dist = _per_seed_mean(seed_trajs_list, "distance_to_target")
        all_trajs = [t for trajs in seed_trajs_list for t in trajs]
        dist_series_all = _collect_metric(all_trajs, "distance_to_target")
        stacked_dist_all = _pad_stack(dist_series_all)
        if per_seed_dist is not None and stacked_dist_all is not None:
            mean = np.mean(per_seed_dist, axis=0)
            std = np.std(stacked_dist_all, axis=0, ddof=1)
            t = np.arange(len(mean)) * dt
            m, s = _end_stats(dist_series_all)
            leg = f"{label}  [{m:.3f} \u00b1 {s:.3f} m]"
            ax_pos.plot(t, mean, color=color, label=leg, linewidth=2)
            ax_pos.fill_between(t, mean - std, mean + std, color=color, alpha=0.2)
            _pos_inset_data.append((t, mean, std, color))

        # ── Yaw error ─────────────────────────────────────────────────────
        per_seed_yaw = _per_seed_mean(seed_trajs_list, "yaw_error")
        yaw_series_all = _collect_metric(all_trajs, "yaw_error")
        stacked_yaw_all = _pad_stack(yaw_series_all)
        if per_seed_yaw is not None and stacked_yaw_all is not None:
            mean = np.mean(per_seed_yaw, axis=0)
            std = np.std(stacked_yaw_all, axis=0, ddof=1)
            t = np.arange(len(mean)) * dt
            m, s = _end_stats(yaw_series_all)
            leg = f"{label}  [{m:.3f} \u00b1 {s:.3f} rad]"
            ax_yaw.plot(t, mean, color=color, label=leg, linewidth=2)
            ax_yaw.fill_between(t, mean - std, mean + std, color=color, alpha=0.2)

    # ── Inset zoom on position error (last 30% of trajectory) ────────────────
    if _pos_inset_data:
        t_max = max(d[0][-1] for d in _pos_inset_data)
        zoom_start = t_max * 0.70
        axins = ax_pos.inset_axes([0.38, 0.28, 0.58, 0.45])
        for t_i, mean_i, std_i, color_i in _pos_inset_data:
            mask = t_i >= zoom_start
            axins.plot(t_i[mask], mean_i[mask], color=color_i, linewidth=1.5)
            axins.fill_between(t_i[mask], (mean_i - std_i)[mask], (mean_i + std_i)[mask],
                               color=color_i, alpha=0.2)
        axins.set_xlim(zoom_start, t_max)
        axins.tick_params(labelsize=6)
        axins.grid(True, alpha=0.3)
        ax_pos.indicate_inset_zoom(axins, edgecolor="gray", alpha=0.5)

    for ax, title, ylabel in [
        (ax_pos, "Position Error (Mean \u00b1 Std across seeds)", "Position error [m]"),
        (ax_yaw, "Yaw Error (Mean \u00b1 Std across seeds)", "Yaw error [rad]"),
    ]:
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8, title=f"Algorithm [final mean \u00b1 std across seeds]",
                  title_fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_input_trajectories(
    groups: Dict[Tuple, List[Path]],
    output_path: Path,
    action_labels: Optional[List[str]] = None,
) -> None:
    """Mean ± std of action trajectories, one shaded band per condition."""
    # Discover action dimensionality from the first available trajectory
    n_act = None
    for _, exp_dirs in sorted(groups.items()):
        for exp_dir in exp_dirs:
            for traj in load_rollouts_for_exp(exp_dir):
                actions = np.asarray(traj.get("actions", traj.get("last_u", [])))
                if actions.ndim == 2 and actions.shape[1] > 0:
                    n_act = actions.shape[1]
                    break
            if n_act is not None:
                break
        if n_act is not None:
            break

    if n_act is None:
        print("  Warning: no action data found — skipping input trajectory plot.")
        return

    labels = action_labels or [f"Action {i}" for i in range(n_act)]

    fig, axes = plt.subplots(1, n_act, figsize=(4 * n_act, 4), squeeze=False)
    axes = axes[0]

    group_order: List[Tuple] = sorted(groups.keys(), key=_sort_key)

    for g_idx, param_key in enumerate(group_order):
        exp_dirs = groups[param_key]
        color = _COLORS[g_idx % len(_COLORS)]
        group_label = make_label(param_key)

        seed_trajs_list = [load_rollouts_for_exp(d) for d in exp_dirs]

        if not any(seed_trajs_list):
            continue

        for dim, ax in enumerate(axes):
            per_seed = _per_seed_action_mean(seed_trajs_list, dim)
            if per_seed is None:
                continue
            # std over all individual rollouts (total variance, ddof=1)
            all_trajs_flat = [t for trajs in seed_trajs_list for t in trajs]
            dim_series_all = [
                np.asarray(traj.get("actions", traj.get("last_u", [])))[:, dim]
                for traj in all_trajs_flat
                if np.asarray(traj.get("actions", traj.get("last_u", []))).ndim == 2
                and np.asarray(traj.get("actions", traj.get("last_u", []))).shape[1] > dim
            ]
            stacked_all = _pad_stack(dim_series_all)
            mean = np.mean(per_seed, axis=0)
            std = np.std(stacked_all, axis=0, ddof=1) if stacked_all is not None else np.zeros_like(mean)
            t = np.arange(len(mean))
            ax.plot(t, mean, color=color, linewidth=2,
                    label=group_label if dim == 0 else "_nolegend_")
            ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.2)

    for dim, ax in enumerate(axes):
        lbl = labels[dim] if dim < len(labels) else f"Action {dim}"
        ax.set_xlabel("Timestep")
        ax.set_ylabel(lbl)
        ax.set_title(lbl)
        ax.grid(True, alpha=0.3)

    handles, leg_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            leg_labels,
            loc="upper center",
            ncol=len(handles),
            fontsize=9,
            bbox_to_anchor=(0.5, 1.02),
        )

    plt.suptitle("Input Trajectories (Mean ± Std)", fontsize=13, fontweight="bold", y=1.06)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sweeps_dir",
        type=Path,
        default=Path(__file__).parent / "sweeps",
        help="Path to the sweeps directory (default: plots/sweeps next to this script)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.02,
        help="Control timestep in seconds used for the time axis (default: 0.02)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Where to save the output figures (default: same as --sweeps_dir)",
    )
    parser.add_argument(
        "--action_labels",
        nargs="+",
        default=None,
        help=(
            "Labels for each action dimension (default: drone labels "
            "[Thrust, Torque X, Torque Y, Torque Z])"
        ),
    )
    args = parser.parse_args()

    sweeps_dir = args.sweeps_dir.resolve()
    output_dir = (args.output_dir or sweeps_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not sweeps_dir.exists():
        parser.error(f"Sweeps directory not found: {sweeps_dir}")

    print(f"Scanning: {sweeps_dir}")
    groups = group_experiments(sweeps_dir)

    if not groups:
        print("No experiment directories (exp_*) found — nothing to plot.")
        return

    print(f"\nFound {len(groups)} condition group(s):")
    for param_key, exp_dirs in sorted(groups.items(), key=lambda kv: _sort_key(kv[0])):
        lbl = make_label(param_key)
        seeds = sorted(parse_exp_dir_name(d.name).get("seed", "?") for d in exp_dirs)
        print(f"  [{lbl}]  seeds={seeds}  ({len(exp_dirs)} dir(s))")

    print("\n--- Error comparison plot ---")
    plot_error_comparison(
        groups,
        output_dir / "sweep_error_comparison.png",
        dt=args.dt,
    )

    print("\n--- Input trajectory plot ---")
    action_labels = args.action_labels or ["Thrust", "Torque X", "Torque Y", "Torque Z"]
    plot_input_trajectories(
        groups,
        output_dir / "sweep_input_trajectories.png",
        action_labels=action_labels,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
