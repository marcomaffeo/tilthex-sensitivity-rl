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

USAGE::

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


# --- Trajectory loading ---

def _load_traj(path: Path) -> Optional[Dict]:
    """Load one trajectory saved with ``flax.serialization``.

    Args:
        path (pathlib.Path): ``traj_*.msgpack`` file.

    Returns:
        dict | None: The trajectory, or ``None`` if the file cannot be read.
    """
    try:
        return serialization.msgpack_restore(path.read_bytes())
    except Exception as exc:
        print(f"  Warning: could not load {path.name}: {exc}")
        return None


def load_rollouts_for_exp(exp_dir: Path) -> List[Dict]:
    """All the trajectories in ``<exp_dir>/rollouts/``.

    Args:
        exp_dir (pathlib.Path): Experiment folder.

    Returns:
        list[dict]: Loaded trajectories.
    """
    rollouts_dir = exp_dir / "rollouts"
    if not rollouts_dir.exists():
        return []
    trajs = []
    for f in sorted(rollouts_dir.glob("*.msgpack")):
        data = _load_traj(f)
        if data is not None:
            trajs.append(data)
    return trajs


def load_nominal_rollouts_for_exp(exp_dir: Path) -> List[Dict]:
    """Only the nominal trajectories in ``<exp_dir>/rollouts/``.

    Nominal trajectories are ``traj_XXXXX.msgpack`` (single-sample runs) and
    ``traj_XXXXX_sample_00.msgpack`` (multi-sample runs).

    Args:
        exp_dir (pathlib.Path): Experiment folder.

    Returns:
        list[dict]: Loaded nominal trajectories.
    """
    rollouts_dir = exp_dir / "rollouts"
    if not rollouts_dir.exists():
        return []
    trajs = []
    for f in sorted(rollouts_dir.glob("*.msgpack")):
        name = f.stem  # strip .msgpack
        if "_sample_" in name and not name.endswith("_sample_00"):
            continue  # skip non-nominal perturbed samples
        data = _load_traj(f)
        if data is not None:
            trajs.append(data)
    return trajs


# --- Directory-name parsing ---

# Known environment base names and version tokens
_ENV_BASES = {"drone", "halfcheetah", "pendulum", "reacher"}
_ENV_VERSIONS = {"v0", "v1", "v2", "v3", "v4", "v5", "obstacle", "flip"}


def parse_exp_dir_name(name: str) -> Dict:
    """Parameters encoded in an experiment folder name.

    Handles names such as
    ``exp_008_sppo_aug_grad_param_dual_drone_v3_srw0.1_seed0_domain_randomizationFalse``.

    Args:
        name (str): Folder name.

    Returns:
        dict: Method, environment, seed and the other parameters.
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
    """Group the experiment folders by their parameters, ignoring the seed.

    Args:
        sweeps_dir (pathlib.Path): Folder with the ``exp_*`` subfolders.

    Returns:
        dict[tuple, list[pathlib.Path]]: Parameter tuple to the folders of its seeds.
    """
    groups: Dict[Tuple, List[Path]] = defaultdict(list)
    for exp_dir in sorted(sweeps_dir.iterdir()):
        if not exp_dir.is_dir() or not exp_dir.name.startswith("exp_"):
            continue
        params = parse_exp_dir_name(exp_dir.name)
        group_key = tuple(sorted((k, v) for k, v in params.items() if k != "seed"))
        groups[group_key].append(exp_dir)
    return groups


# --- Label generation ---

_METHOD_SHORT: Dict[str, str] = {
    "sppo_aug_grad_param_dual": "SPPO",
    "sppo_aug_grad_param": "SPPO",
    "sppo_aug": "SPPO-Aug",
    "sppo": "SPPO",
    "ppo": "PPO",
    "sac": "SAC",
    "ddpg": "DDPG",
}

# Physical input limits (min_input, max_input) for known environments.
# Used to normalise actions before computing total-variation / jerk so that
# all input dimensions are on the same [0, 1] scale.
_DRONE_INPUT_LIMITS: Tuple[np.ndarray, np.ndarray] = (
    np.array([4.0, -3.0, -3.0, -1.0]),   # min_input
    np.array([25.0,  3.0,  3.0,  1.0]),   # max_input
)
_ENV_INPUT_LIMITS: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    "drone_v0":       _DRONE_INPUT_LIMITS,
    "drone_v1":       _DRONE_INPUT_LIMITS,
    "drone_v2":       _DRONE_INPUT_LIMITS,
    "drone_v3":       _DRONE_INPUT_LIMITS,
    "drone_v4":       _DRONE_INPUT_LIMITS,
    "drone_obstacle": _DRONE_INPUT_LIMITS,
    "drone_flip":     _DRONE_INPUT_LIMITS,
}


def make_label(param_key: Tuple) -> str:
    """Readable label of a parameter group, used in the legends.

    Args:
        param_key (tuple): Parameter tuple of the group.

    Returns:
        str: Label, e.g. ``SPPO srw=0.1``.
    """
    params = dict(param_key)
    method = params.get("method", "unknown")
    label = _METHOD_SHORT.get(method, method.replace("_", "-").upper())

    dr = params.get("domain_randomization", None)
    if dr is True:
        label += " + DR"

    return label


# --- Metric helpers ---

def _collect_metric(trajs: List[Dict], key: str) -> List[np.ndarray]:
    """Time series of one evaluation metric from every trajectory.

    Args:
        trajs (list[dict]): Trajectories.
        key (str): Key in ``eval_metrics``.

    Returns:
        list[numpy.ndarray]: One 1D series per trajectory that has the metric.
    """
    series: List[np.ndarray] = []
    for traj in trajs:
        em = traj.get("eval_metrics", {})
        if key in em:
            arr = np.asarray(em[key]).flatten()
            series.append(arr)
    return series


def _collect_actions(trajs: List[Dict]) -> List[np.ndarray]:
    """Action arrays of every trajectory.

    Args:
        trajs (list[dict]): Trajectories.

    Returns:
        list[numpy.ndarray]: One array of shape ``(T, n_act)`` per trajectory.
    """
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
    """Pad series to the same length with their last value and stack them.

    Args:
        series (list[numpy.ndarray]): 1D series.

    Returns:
        numpy.ndarray | None: Array of shape ``(N, T)``, or ``None`` for an empty list.
    """
    if not series:
        return None
    max_len = max(s.shape[0] for s in series)
    padded = np.stack(
        [np.pad(s, (0, max_len - s.shape[0]), mode="edge") for s in series]
    )
    return padded


def _rmse_with_std(stacked: np.ndarray):
    """Per-time-step RMSE over rollouts and its spread.

    Args:
        stacked (numpy.ndarray): Errors, shape ``(N, T)``.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: ``rmse[t] = sqrt(mean(stacked[:, t]**2))`` and the
        standard deviation of the per-rollout absolute errors, shape ``(T,)`` each.
    """
    sq = stacked ** 2
    rmse = np.sqrt(np.mean(sq, axis=0))
    # The shading shows the std of the per-rollout RMS values, which is
    # easier to read than the delta-method std of the RMSE.
    std = np.std(np.sqrt(sq), axis=0)
    return rmse, std


# --- Plot helpers ---

def _savefig(fig: plt.Figure, output_path: Path) -> None:
    """Save a figure as PNG and PDF and close it.

    Args:
        fig (matplotlib.figure.Figure): Figure.
        output_path (pathlib.Path): Output path; the suffix is replaced.
    """
    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


# --- Plots ---

_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
_LAST_N = 10  # number of trailing samples used for the summary table


def _sort_key(param_key: Tuple):
    """Sort key that puts the SPPO-based groups last.

    Args:
        param_key (tuple): Parameter tuple of the group.

    Returns:
        tuple: Sort key.
    """
    label = make_label(param_key)
    is_sppo = label.upper().startswith("SPPO")
    return (int(is_sppo),) + param_key


def _end_stats(series: List[np.ndarray], last_n: int = _LAST_N) -> Tuple[float, float]:
    """Mean and standard deviation over trajectories of the mean of the last samples.

    Args:
        series (list[numpy.ndarray]): One series per trajectory.
        last_n (int): Number of final samples averaged per trajectory.

    Returns:
        tuple[float, float]: Mean and standard deviation (ddof = 1).
    """
    tail_means = [float(np.mean(s[-min(last_n, len(s)):])) for s in series if len(s) > 0]
    if not tail_means:
        return float("nan"), float("nan")
    return float(np.mean(tail_means)), float(np.std(tail_means, ddof=min(1, len(tail_means) - 1)))


def _per_seed_mean(seed_trajs_list: List[List[Dict]], key: str) -> Optional[np.ndarray]:
    """Mean time series of a metric for each seed.

    Args:
        seed_trajs_list (list[list[dict]]): Trajectories of each seed.
        key (str): Key in ``eval_metrics``.

    Returns:
        numpy.ndarray | None: Array of shape ``(n_seeds, T)``, or ``None`` without data.
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
    """Mean time series of one action component for each seed.

    Args:
        seed_trajs_list (list[list[dict]]): Trajectories of each seed.
        dim (int): Action component.

    Returns:
        numpy.ndarray | None: Array of shape ``(n_seeds, T)``, or ``None`` without data.
    """
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
    output_dir: Path,
    dt: float = 0.02,
) -> None:
    """Plot position and yaw error over time, mean and standard deviation over seeds, one curve per group.

    Args:
        groups (dict[tuple, list[pathlib.Path]]): Output of ``group_experiments``.
        output_dir (pathlib.Path): Output folder.
        dt (float): Time step of the rollouts [s].
    """
    fig_pos, ax_pos = plt.subplots(figsize=(7, 5))
    fig_yaw, ax_yaw = plt.subplots(figsize=(7, 5))

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

        all_trajs = [t for trajs in seed_trajs_list for t in trajs]

        # Position error
        per_seed_dist = _per_seed_mean(seed_trajs_list, "distance_to_target")
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

        # Yaw error
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

    # Inset zoom on the position error (last 30% of the trajectory)
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

    # Finalise the position figure
    ax_pos.set_xlabel("Time [s]")
    ax_pos.set_ylabel("Position error [m]")
    ax_pos.set_title("Position Error (Mean \u00b1 Std across seeds)")
    ax_pos.legend(fontsize=8, title="Algorithm [final mean \u00b1 std across seeds]",
                  title_fontsize=8)
    ax_pos.grid(True, alpha=0.3)
    fig_pos.tight_layout()
    _savefig(fig_pos, output_dir / "sweep_position_error")
    plt.close(fig_pos)

    # Finalise the yaw figure
    ax_yaw.set_xlabel("Time [s]")
    ax_yaw.set_ylabel("Yaw error [rad]")
    ax_yaw.set_title("Yaw Error (Mean \u00b1 Std across seeds)")
    ax_yaw.legend(fontsize=8, title="Algorithm [final mean \u00b1 std across seeds]",
                  title_fontsize=8)
    ax_yaw.grid(True, alpha=0.3)
    ymin, ymax = ax_yaw.get_ylim()
    ax_yaw.set_ylim(bottom=min(ymin, 0.2), top=ymax)
    fig_yaw.tight_layout()
    _savefig(fig_yaw, output_dir / "sweep_yaw_error")
    plt.close(fig_yaw)


def plot_input_trajectories(
    groups: Dict[Tuple, List[Path]],
    output_path: Path,
    action_labels: Optional[List[str]] = None,
) -> None:
    """Plot the mean and standard deviation of each action component, one band per group.

    Args:
        groups (dict[tuple, list[pathlib.Path]]): Output of ``group_experiments``.
        output_path (pathlib.Path): Output figure.
        action_labels (list[str], optional): Labels of the action components.
    """
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
    fig = plt.gcf()
    _savefig(fig, output_path)
    plt.close(fig)


# --- Metrics summary (terminal) ---

def _action_jerk_stats(
    trajs: List[Dict],
    action_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Optional[Dict]:
    """Statistics of the step-to-step variation of the actions.

    The jerk at each step is ``||(a[t+1] - a[t]) / span||_2``, with ``span``
    the input range of each component (no normalisation when
    ``action_limits`` is ``None``); its sum over the episode is the total
    variation.

    Args:
        trajs (list[dict]): Trajectories.
        action_limits (tuple[numpy.ndarray, numpy.ndarray], optional): ``(min_input, max_input)`` per component.

    Returns:
        dict | None: ``normalised``, ``span``, ``mean_jerk``, ``std_jerk``, ``integral_jerk``,
        ``std_integral``, the per-component values (``per_dim_*``) and ``n_trajs``;
        ``None`` without actions.
    """
    actions_list = _collect_actions(trajs)
    if not actions_list:
        return None

    n_dims = actions_list[0].shape[1]

    # Build normalisation span
    if action_limits is not None:
        min_a, max_a = action_limits
        span = (max_a - min_a).astype(float)  # shape (n_act,)
        span = np.where(span == 0, 1.0, span)  # guard division by zero
    else:
        span = None

    per_traj_mean_jerk: List[float] = []
    per_traj_integral: List[float] = []
    per_dim_mean: List[List[float]] = [[] for _ in range(n_dims)]
    per_dim_integral: List[List[float]] = [[] for _ in range(n_dims)]

    for act in actions_list:  # act: (T, n_act)
        if act.shape[0] < 2:
            continue
        diff = np.diff(act, axis=0)  # (T-1, n_act)
        if span is not None:
            diff_norm = diff / span[np.newaxis, :]
        else:
            diff_norm = diff
        jerk = np.linalg.norm(diff_norm, axis=1)  # (T-1,)
        per_traj_mean_jerk.append(float(np.mean(jerk)))
        per_traj_integral.append(float(np.sum(jerk)))
        for d in range(n_dims):
            abs_diff_d = np.abs(diff_norm[:, d])
            per_dim_mean[d].append(float(np.mean(abs_diff_d)))
            per_dim_integral[d].append(float(np.sum(abs_diff_d)))

    if not per_traj_mean_jerk:
        return None

    def _agg(vals: List[float]) -> Tuple[float, float]:
        return (
            float(np.mean(vals)),
            float(np.std(vals, ddof=min(1, len(vals) - 1))),
        )

    result: Dict = {
        "normalised": span is not None,
        "span": span.tolist() if span is not None else None,
        "n_trajs": len(per_traj_mean_jerk),
        "mean_jerk": _agg(per_traj_mean_jerk)[0],
        "std_jerk":  _agg(per_traj_mean_jerk)[1],
        "integral_jerk": _agg(per_traj_integral)[0],
        "std_integral":  _agg(per_traj_integral)[1],
        "per_dim_mean_jerk":    [_agg(v)[0] for v in per_dim_mean],
        "per_dim_std_jerk":     [_agg(v)[1] for v in per_dim_mean],
        "per_dim_integral":     [_agg(v)[0] for v in per_dim_integral],
        "per_dim_std_integral": [_agg(v)[1] for v in per_dim_integral],
    }
    return result


def _get_action_limits(param_key: Tuple) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Input limits of the environment of a parameter group.

    Args:
        param_key (tuple): Parameter tuple of the group.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray] | None: ``(min_input, max_input)``, or ``None`` for an unknown environment.
    """
    params = dict(param_key)
    env = params.get("env", "")
    return _ENV_INPUT_LIMITS.get(env, None)


def _time_to_goal(
    dist_series: List[np.ndarray],
    goal_radius: float = 0.05,
    dt: float = 0.02,
) -> Dict:
    """Time to reach the goal, over a set of trajectories.

    Args:
        dist_series (list[numpy.ndarray]): Distance to the target of each trajectory.
        goal_radius (float): Distance below which the goal is reached [m].
        dt (float): Time step [s].

    Returns:
        dict: ``n_reached``, ``mean_time``, ``std_time``, ``min_time`` and ``max_time`` [s].
    """
    times: List[float] = []
    for s in dist_series:
        idx = np.argmax(s <= goal_radius)  # first True, or 0 if never
        if s[idx] <= goal_radius:
            times.append(float(idx) * dt)
    n = len(times)
    if n == 0:
        return {"n_reached": 0, "mean_time": float("nan"), "std_time": float("nan"),
                "min_time": float("nan"), "max_time": float("nan")}
    return {
        "n_reached": n,
        "mean_time": float(np.mean(times)),
        "std_time": float(np.std(times, ddof=min(1, n - 1))),
        "min_time": float(np.min(times)),
        "max_time": float(np.max(times)),
    }


def print_metrics_summary(
    groups: Dict[Tuple, List[Path]],
    action_labels: Optional[List[str]] = None,
    action_limits_override: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    last_n: int = _LAST_N,
    dt: float = 0.02,
) -> None:
    """Print accuracy, time to goal and action variation for each group.

    Args:
        groups (dict[tuple, list[pathlib.Path]]): Output of ``group_experiments``.
        action_labels (list[str], optional): Labels of the action components.
        action_limits_override (tuple[numpy.ndarray, numpy.ndarray], optional): Input limits used instead of the known ones.
        last_n (int): Final samples used for the steady-state values.
        dt (float): Time step [s].
    """
    args_dt = dt  # alias used inside the loop
    sep = "=" * 80
    thin = "-" * 80

    print(f"\n{sep}")
    print("METRICS SUMMARY")
    print(sep)

    for idx, (param_key, exp_dirs) in enumerate(
        sorted(groups.items(), key=lambda kv: _sort_key(kv[0]))
    ):
        label = make_label(param_key)
        all_trajs = [t for d in exp_dirs for t in load_rollouts_for_exp(d)]

        print(f"\n  Condition: {label}  ({len(exp_dirs)} seeds, {len(all_trajs)} trajs)")
        print(thin)

        # Accuracy
        dist_series = _collect_metric(all_trajs, "distance_to_target")
        yaw_series = _collect_metric(all_trajs, "yaw_error")

        if dist_series:
            # Final-value stats (last `last_n` timesteps of each trajectory)
            m_final, s_final = _end_stats(dist_series, last_n)
            # RMSE over full trajectory (per rollout → mean ± std across rollouts)
            per_traj_rmse = [float(np.sqrt(np.mean(s ** 2))) for s in dist_series]
            rmse_mean = float(np.mean(per_traj_rmse))
            rmse_std = float(np.std(per_traj_rmse, ddof=min(1, len(per_traj_rmse) - 1)))
            print(f"  Position error:")
            print(f"    Final (last {last_n} steps): {m_final:.4f} ± {s_final:.4f} m")
            print(f"    RMSE over trajectory:        {rmse_mean:.4f} ± {rmse_std:.4f} m")
        else:
            print("  Position error: no data")

        # Time to goal
        _ttg = _time_to_goal(dist_series, goal_radius=0.05, dt=args_dt)
        if _ttg["n_reached"] > 0:
            n_tot = len(dist_series)
            print(
                f"  Time to goal (r=5 cm):  "
                f"{_ttg['mean_time']:.3f} ± {_ttg['std_time']:.3f} s  "
                f"[{_ttg['n_reached']}/{n_tot} trajs reached goal  "
                f"({100 * _ttg['n_reached'] / n_tot:.1f}%)]"
            )
        else:
            print(f"  Time to goal (r=5 cm):  goal never reached in any trajectory")

        if yaw_series:
            m_final, s_final = _end_stats(yaw_series, last_n)
            per_traj_rmse = [float(np.sqrt(np.mean(s ** 2))) for s in yaw_series]
            rmse_mean = float(np.mean(per_traj_rmse))
            rmse_std = float(np.std(per_traj_rmse, ddof=min(1, len(per_traj_rmse) - 1)))
            print(f"  Yaw error:")
            print(f"    Final (last {last_n} steps): {m_final:.4f} ± {s_final:.4f} rad")
            print(f"    RMSE over trajectory:        {rmse_mean:.4f} ± {rmse_std:.4f} rad")
        else:
            print("  Yaw error: no data")

        # Action jerk
        limits = action_limits_override or _get_action_limits(param_key)
        jerk = _action_jerk_stats(all_trajs, action_limits=limits)
        nominal_trajs = [t for d in exp_dirs for t in load_nominal_rollouts_for_exp(d)]
        has_perturbed = len(nominal_trajs) < len(all_trajs)
        jerk_nom = _action_jerk_stats(nominal_trajs, action_limits=limits) if has_perturbed else None
        if jerk is not None:
            al = action_labels or [f"Action {i}" for i in range(len(jerk["per_dim_mean_jerk"]))]
            norm_tag = "normalised by input span" if jerk["normalised"] else "raw (no normalisation)"
            print(f"  Action jerk (||Δa_norm||₂, {norm_tag}):  ({jerk['n_trajs']} trajs, all)")
            print(f"    Mean jerk per step:    {jerk['mean_jerk']:.6f} ± {jerk['std_jerk']:.6f}")
            print(f"    Integrated jerk (TV):  {jerk['integral_jerk']:.4f} ± {jerk['std_integral']:.4f}")
            if jerk_nom is not None:
                print(f"  Action jerk (nominal only, {jerk_nom['n_trajs']} trajs):")
                print(f"    Mean jerk per step:    {jerk_nom['mean_jerk']:.6f} ± {jerk_nom['std_jerk']:.6f}")
                print(f"    Integrated jerk (TV):  {jerk_nom['integral_jerk']:.4f} ± {jerk_nom['std_integral']:.4f}")
            span = jerk["span"]
            print(f"  Per-dimension mean jerk per step (|Δa_i / span_i|):")
            for d, (dm, ds) in enumerate(
                zip(jerk["per_dim_mean_jerk"], jerk["per_dim_std_jerk"])
            ):
                dim_label = al[d] if d < len(al) else f"dim {d}"
                span_str = f"span={span[d]:.2g}" if span is not None else "no span"
                nom_str = ""
                if jerk_nom is not None and d < len(jerk_nom["per_dim_mean_jerk"]):
                    nom_str = f"  (nominal: {jerk_nom['per_dim_mean_jerk'][d]:.6f} ± {jerk_nom['per_dim_std_jerk'][d]:.6f})"
                print(f"    [{dim_label:>12s}]  {span_str:>12s}  |  mean/step: {dm:.6f} ± {ds:.6f}{nom_str}")
        else:
            print("  Action jerk: no action data found")

    print(f"\n{sep}\n")


# --- Main ---

def main() -> None:
    """Command-line entry point: group the experiments of a sweep folder, plot them and print the summary."""
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
    parser.add_argument(
        "--action_min",
        nargs="+",
        type=float,
        default=None,
        metavar="VAL",
        help=(
            "Override minimum physical input limits for jerk normalisation, "
            "one value per action dimension. Inferred from ENV_NAME otherwise."
        ),
    )
    parser.add_argument(
        "--action_max",
        nargs="+",
        type=float,
        default=None,
        metavar="VAL",
        help=(
            "Override maximum physical input limits for jerk normalisation, "
            "one value per action dimension. Inferred from ENV_NAME otherwise."
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

    print("\n--- Error comparison plots ---")
    plot_error_comparison(
        groups,
        output_dir,
        dt=args.dt,
    )

    print("\n--- Input trajectory plot ---")
    action_labels = args.action_labels or ["Thrust", "Torque X", "Torque Y", "Torque Z"]
    plot_input_trajectories(
        groups,
        output_dir / "sweep_input_trajectories",
        action_labels=action_labels,
    )

    print("\n--- Metrics summary ---")
    # Build optional limits override from CLI args
    limits_override: Optional[Tuple[np.ndarray, np.ndarray]] = None
    if args.action_min is not None and args.action_max is not None:
        limits_override = (np.array(args.action_min), np.array(args.action_max))
    elif args.action_min is not None or args.action_max is not None:
        print("  Warning: both --action_min and --action_max must be specified together. Ignoring.")
    print_metrics_summary(groups, action_labels=action_labels, action_limits_override=limits_override, dt=args.dt)

    print("Done.")


if __name__ == "__main__":
    main()
