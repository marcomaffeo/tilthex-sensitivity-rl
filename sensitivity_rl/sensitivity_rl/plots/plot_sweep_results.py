"""
Plot sweep results from TensorBoard logs.

This script aggregates results across different seeds and creates plots showing
mean and standard deviation for key metrics:
- Episodic return (true reward)
- Sensitivity state mean
- Sensitivity state max

Can be used standalone or called from the tuner after a sweep completes.

USAGE:
    # Standalone - specify sweep directory
    python plot_sweep_results.py --sweep_dir runs/sweep_20251209-123456

    # Or from tuner (called automatically)
    python tuner.py
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorboard.backend.event_processing import event_accumulator


def save_table_image_latex_style(
    headers: List[str],
    rows: List[List[str]],
    title: str,
    output_file: Path,
):
    """Save a table as a PNG with a LaTeX-like (booktabs) visual style."""
    if not rows:
        print(f"Warning: No rows to render for table image: {output_file}")
        return

    fig_height = max(2.8, 0.42 * (len(rows) + 1) + 1.4)
    fig_width = max(10.0, 2.1 * len(headers))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.35)

    # Booktabs-like look: no inner cell borders, serif text, bold header.
    for (r, _), cell in table.get_celld().items():
        cell.set_linewidth(0.0)
        cell.set_edgecolor("white")
        cell.set_facecolor("white")
        cell.get_text().set_fontfamily("serif")
        if r == 0:
            cell.get_text().set_weight("bold")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    table_bbox = table.get_window_extent(renderer=renderer).transformed(
        ax.transAxes.inverted()
    )
    x0, y0, x1, y1 = table_bbox.x0, table_bbox.y0, table_bbox.x1, table_bbox.y1

    header_bbox = (
        table[(0, 0)]
        .get_window_extent(renderer=renderer)
        .transformed(ax.transAxes.inverted())
    )
    y_header_bottom = header_bbox.y0

    ax.hlines(y1, x0, x1, colors="black", linewidth=1.4, transform=ax.transAxes)
    ax.hlines(
        y_header_bottom,
        x0,
        x1,
        colors="black",
        linewidth=0.9,
        transform=ax.transAxes,
    )
    ax.hlines(y0, x0, x1, colors="black", linewidth=1.4, transform=ax.transAxes)

    ax.set_title(title, fontsize=13, fontweight="bold", fontfamily="serif", pad=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved table image: {output_file}")


def load_tensorboard_data(log_dir: Path, tags: List[str]) -> Dict[str, np.ndarray]:
    """
    Load data from TensorBoard event files.

    Args:
        log_dir: Path to tensorboard log directory
        tags: List of metric tags to extract

    Returns:
        Dictionary mapping tag names to (steps, values) tuples
    """
    ea = event_accumulator.EventAccumulator(str(log_dir))
    ea.Reload()

    data = {}
    for tag in tags:
        try:
            events = ea.Scalars(tag)
            steps = np.array([e.step for e in events])
            values = np.array([e.value for e in events])
            data[tag] = (steps, values)
        except KeyError:
            print(f"Warning: Tag '{tag}' not found in {log_dir}")
            data[tag] = (np.array([]), np.array([]))

    return data


def parse_experiment_name(exp_name: str) -> Dict[str, any]:
    """
    Parse experiment name to extract parameters.

    Example: exp_001_sppovsa_halfcheetah_v5_seed0_srw0.5_total_timesteps100000
    Returns: {'method': 'sppovsa', 'env': 'halfcheetah_v5', 'seed': 0, 'srw': 0.5, ...}
    """
    parts = exp_name.split("_")
    params = {}

    i = 0
    while i < len(parts):
        # Extract method (after exp_XXX)
        if parts[i].startswith("exp") and i + 1 < len(parts):
            # Skip exp_XXX numbering
            i += 1
            if i < len(parts) and parts[i].isdigit():
                i += 1
            # Next part is method name
            if i < len(parts):
                params["method"] = parts[i]
                i += 1
            continue

        # Extract environment (look for common env names)
        if parts[i] in ["halfcheetah", "drone", "pendulum", "reacher"] and i + 1 < len(
            parts
        ):
            params["env"] = f"{parts[i]}_{parts[i+1]}"
            i += 2
            continue

        # Extract parameters that start with known prefixes followed by value
        # Check if current part contains letters followed by digits (e.g., seed0, srw0.5)
        # or letters followed by boolean strings (e.g., domain_randomizationTrue)
        import re

        # First try numeric pattern
        match = re.match(r"^([a-zA-Z_]+)([-+]?[0-9]*\.?[0-9]+)$", parts[i])
        if match:
            param_name = match.group(1)
            param_value_str = match.group(2)
            try:
                param_value = (
                    float(param_value_str)
                    if "." in param_value_str
                    else int(param_value_str)
                )
                params[param_name] = param_value
                i += 1
                continue
            except ValueError:
                pass

        # Try boolean pattern
        bool_match = re.match(r"^([a-zA-Z_]+)(True|False)$", parts[i])
        if bool_match:
            param_name = bool_match.group(1)
            param_value_str = bool_match.group(2)
            params[param_name] = param_value_str == "True"
            i += 1
            continue

        i += 1

    return params


def group_experiments_by_params(sweep_dir: Path) -> Dict[Tuple, List[Path]]:
    """
    Group experiment directories by their parameter settings (excluding seed).

    Args:
        sweep_dir: Path to sweep directory

    Returns:
        Dictionary mapping parameter tuples to lists of experiment directories
    """
    experiments = defaultdict(list)

    for exp_dir in sweep_dir.iterdir():
        if not exp_dir.is_dir() or not exp_dir.name.startswith("exp_"):
            continue

        params = parse_experiment_name(exp_dir.name)

        # Create key from all params except seed
        param_key = tuple(sorted([(k, v) for k, v in params.items() if k != "seed"]))
        experiments[param_key].append(exp_dir)

    return experiments


def interpolate_to_common_steps(
    data_list: List[Tuple[np.ndarray, np.ndarray]], num_points: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate multiple time series to common step values.

    Args:
        data_list: List of (steps, values) tuples
        num_points: Number of interpolation points

    Returns:
        Tuple of (common_steps, interpolated_values_array)
    """
    if not data_list:
        return np.array([]), np.array([])

    # Find common step range
    min_step = max(steps[0] for steps, _ in data_list if len(steps) > 0)
    max_step = min(steps[-1] for steps, _ in data_list if len(steps) > 0)

    if min_step >= max_step:
        return np.array([]), np.array([])

    common_steps = np.linspace(min_step, max_step, num_points)
    interpolated_values = []

    for steps, values in data_list:
        if len(steps) > 0 and len(values) > 0:
            interp_values = np.interp(common_steps, steps, values)
            interpolated_values.append(interp_values)

    return common_steps, np.array(interpolated_values)


def plot_metric_comparison(
    sweep_dir: Path,
    grouped_experiments: Dict[Tuple, List[Path]],
    metric_tag: str,
    metric_name: str,
    output_dir: Path,
):
    """
    Create plots comparing a metric across different parameter settings.

    Args:
        sweep_dir: Path to sweep directory
        grouped_experiments: Dictionary of parameter groups to experiment directories
        metric_tag: TensorBoard tag to plot
        metric_name: Human-readable metric name for plot title
        output_dir: Directory to save plots
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for param_key, exp_dirs in sorted(grouped_experiments.items()):
        if len(exp_dirs) == 0:
            continue

        # Load data from all seeds
        all_data = []
        for exp_dir in exp_dirs:
            data = load_tensorboard_data(exp_dir, [metric_tag])
            if metric_tag in data and len(data[metric_tag][0]) > 0:
                all_data.append(data[metric_tag])

        if not all_data:
            continue

        # Interpolate to common steps
        common_steps, interpolated_values = interpolate_to_common_steps(all_data)

        if len(interpolated_values) == 0:
            continue

        # Compute mean and std across seeds
        mean_values = np.mean(interpolated_values, axis=0)
        std_values = np.std(interpolated_values, axis=0)

        # Create label from parameters (excluding seed, method, env)
        params_dict = dict(param_key)
        label_parts = []

        # Extract all varying parameters (not method or env)
        for k, v in sorted(params_dict.items()):
            if k not in ["timesteps", "env", "seed"]:
                # Format parameter for label
                param_label = k.upper() if len(k) <= 3 else k.replace("_", " ").title()
                if param_label == "Randomization":
                    param_label = "DR"
                if param_label != "Method":
                    if isinstance(v, float):
                        label_parts.append(f"{param_label}={v:.3g}")
                    else:
                        label_parts.append(f"{param_label}={v}")
                else:
                    label_parts.append(f"{v}")
        # If no varying parameters, show number of seeds
        if not label_parts:
            label = f"Default seeds)"
        else:
            label = ", ".join(label_parts)

        # Plot mean with shaded std
        line = ax.plot(common_steps, mean_values, label=label, linewidth=2, alpha=0.8)
        color = line[0].get_color()
        ax.fill_between(
            common_steps,
            mean_values - std_values,
            mean_values + std_values,
            alpha=0.15,
            color=color,
        )

    # Format plot
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)

    # Try to extract method and env from first group for title
    method_name = "Unknown"
    env_name = "Unknown"
    if grouped_experiments:
        first_key = list(grouped_experiments.keys())[0]
        first_params = dict(first_key)
        method_name = first_params.get("method", "Unknown")
        env_name = first_params.get("env", "Unknown")

    ax.set_title(
        f"{metric_name}\n(Mean ± Std across seeds)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize=9,
        framealpha=0.9,
    )
    ax.grid(True, alpha=0.3)

    # Save plot
    output_file = output_dir / f"{metric_tag.replace('/', '_')}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {output_file}")


def create_summary_table(
    sweep_dir: Path, grouped_experiments: Dict[Tuple, List[Path]], output_dir: Path
):
    """
    Create a summary table with final metric values.

    Args:
        sweep_dir: Path to sweep directory
        grouped_experiments: Dictionary of parameter groups to experiment directories
        output_dir: Directory to save summary
    """
    metrics = [
        "returns/episodic_return_true",
        "returns/sensitivity_state_mean",
        "returns/sensitivity_state_max",
    ]

    summary = []

    for param_key, exp_dirs in sorted(grouped_experiments.items()):
        if len(exp_dirs) == 0:
            continue

        params_dict = dict(param_key)
        row = {"params": params_dict, "num_seeds": len(exp_dirs)}

        for metric_tag in metrics:
            final_values = []

            for exp_dir in exp_dirs:
                data = load_tensorboard_data(exp_dir, [metric_tag])
                if metric_tag in data and len(data[metric_tag][1]) > 0:
                    # Take last 10% of values and average
                    values = data[metric_tag][1]
                    n = max(1, len(values) // 10)
                    final_values.append(np.mean(values[-n:]))

            if final_values:
                row[f"{metric_tag}_mean"] = np.mean(final_values)
                row[f"{metric_tag}_std"] = np.std(final_values)
            else:
                row[f"{metric_tag}_mean"] = np.nan
                row[f"{metric_tag}_std"] = np.nan

        summary.append(row)

    # Save as JSON
    output_file = output_dir / "summary.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary: {output_file}")

    # Save as CSV with explicit approach/environment columns.
    csv_file = output_dir / "summary_table.csv"
    csv_headers = [
        "approach",
        "environment",
        "configuration",
        "num_seeds",
        "episodic_return_mean",
        "episodic_return_std",
        "sensitivity_mean",
        "sensitivity_mean_std",
        "sensitivity_max",
        "sensitivity_max_std",
    ]
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()

        for row in summary:
            params = row.get("params", {})
            method = params.get("method", "Unknown")
            env = params.get("env", "Unknown")

            config_parts = []
            for k, v in sorted(params.items()):
                if k in ["method", "env", "seed"]:
                    continue
                if isinstance(v, float):
                    config_parts.append(f"{k}={v:.3g}")
                else:
                    config_parts.append(f"{k}={v}")

            writer.writerow(
                {
                    "approach": method,
                    "environment": env,
                    "configuration": (
                        ", ".join(config_parts) if config_parts else "Default"
                    ),
                    "num_seeds": row.get("num_seeds", ""),
                    "episodic_return_mean": row.get(
                        "returns/episodic_return_true_mean", np.nan
                    ),
                    "episodic_return_std": row.get(
                        "returns/episodic_return_true_std", np.nan
                    ),
                    "sensitivity_mean": row.get(
                        "returns/sensitivity_state_mean_mean", np.nan
                    ),
                    "sensitivity_mean_std": row.get(
                        "returns/sensitivity_state_mean_std", np.nan
                    ),
                    "sensitivity_max": row.get(
                        "returns/sensitivity_state_max_mean", np.nan
                    ),
                    "sensitivity_max_std": row.get(
                        "returns/sensitivity_state_max_std", np.nan
                    ),
                }
            )

    print(f"Saved CSV table: {csv_file}")

    # Print table to console
    print("\n" + "=" * 80)
    print("SUMMARY TABLE (Final 10% average)")
    print("=" * 80)
    for row in summary:
        print(f"\nParameters: {row['params']}")
        for metric_tag in metrics:
            mean_key = f"{metric_tag}_mean"
            std_key = f"{metric_tag}_std"
            if mean_key in row and not np.isnan(row[mean_key]):
                print(f"  {metric_tag}: {row[mean_key]:.4f} ± {row[std_key]:.4f}")
    print("=" * 80 + "\n")

    # Save a publication-style table image.
    headers = [
        "Configuration",
        "Episodic Return",
        "Sensitivity Mean",
        "Sensitivity Max",
    ]
    image_rows = []

    for row in summary:
        params = row.get("params", {})
        label_parts = []
        for k, v in sorted(params.items()):
            if k in ["method", "env", "seed"]:
                continue
            if isinstance(v, float):
                label_parts.append(f"{k}={v:.3g}")
            else:
                label_parts.append(f"{k}={v}")
        config_label = ", ".join(label_parts) if label_parts else "Default"

        def fmt_metric(mean_key: str, std_key: str) -> str:
            if np.isnan(row.get(mean_key, np.nan)):
                return "N/A"
            return f"{row[mean_key]:.3f} ± {row[std_key]:.3f}"

        image_rows.append(
            [
                config_label,
                fmt_metric(
                    "returns/episodic_return_true_mean",
                    "returns/episodic_return_true_std",
                ),
                fmt_metric(
                    "returns/sensitivity_state_mean_mean",
                    "returns/sensitivity_state_mean_std",
                ),
                fmt_metric(
                    "returns/sensitivity_state_max_mean",
                    "returns/sensitivity_state_max_std",
                ),
            ]
        )

    table_image_file = output_dir / "summary_table_latex.png"
    save_table_image_latex_style(
        headers,
        image_rows,
        "Sweep Summary (Final 10% Mean ± Std)",
        table_image_file,
    )


def compute_pareto_front(objectives: np.ndarray) -> np.ndarray:
    """
    Compute Pareto front from multi-objective data.

    Args:
        objectives: Array of shape (n_points, 2) where both columns are to be maximized:
                   - First column: maximize episodic return
                   - Second column: maximize negative sensitivity (i.e., minimize sensitivity)

    Returns:
        Boolean array indicating which points are on the Pareto front
    """
    is_pareto = np.ones(objectives.shape[0], dtype=bool)
    for i, point in enumerate(objectives):
        # A point is dominated if there exists another point that is >= in all objectives
        # and strictly > in at least one objective
        dominated_by = np.all(objectives >= point, axis=1) & np.any(
            objectives > point, axis=1
        )
        if np.any(dominated_by):
            is_pareto[i] = False
    return is_pareto


def plot_pareto_front(
    sweep_dir: Path, grouped_experiments: Dict[Tuple, List[Path]], output_dir: Path
):
    """
    Create Pareto front plot for episodic return vs sensitivity.

    Args:
        sweep_dir: Path to sweep directory
        grouped_experiments: Dictionary of parameter groups to experiment directories
        output_dir: Directory to save plots
    """
    metrics = ["returns/episodic_return_true", "returns/sensitivity_state_mean"]

    # Collect data for all parameter combinations
    data_points = []
    param_labels = []

    for param_key, exp_dirs in sorted(grouped_experiments.items()):
        if len(exp_dirs) == 0:
            continue

        params_dict = dict(param_key)

        # Collect final values for both metrics
        return_values = []
        sensitivity_values = []

        for exp_dir in exp_dirs:
            data = load_tensorboard_data(exp_dir, metrics)

            # Get final values (average of last 10%)
            if (
                "returns/episodic_return_true" in data
                and len(data["returns/episodic_return_true"][1]) > 0
            ):
                values = data["returns/episodic_return_true"][1]
                n = max(1, len(values) // 10)
                return_values.append(np.mean(values[-n:]))

            if (
                "returns/sensitivity_state_mean" in data
                and len(data["returns/sensitivity_state_mean"][1]) > 0
            ):
                values = data["returns/sensitivity_state_mean"][1]
                n = max(1, len(values) // 10)
                # Store positive sensitivity value (will negate later for Pareto computation)
                sensitivity_values.append(np.mean(values[-n:]))

        if return_values and sensitivity_values:
            mean_return = np.mean(return_values)
            mean_sensitivity = np.mean(sensitivity_values)

            # Store for Pareto computation: [maximize return, minimize sensitivity]
            # We negate sensitivity to convert minimization to maximization
            data_points.append([mean_return, -mean_sensitivity])

            # Create label
            label_parts = []
            for k, v in sorted(params_dict.items()):
                if k not in ["method", "env", "seed"]:
                    if isinstance(v, float):
                        label_parts.append(f"{k}={v:.3g}")
                    else:
                        label_parts.append(f"{k}={v}")

            if label_parts:
                param_labels.append(", ".join(label_parts))
            else:
                param_labels.append("Default")

    if not data_points:
        print("Warning: No data points found for Pareto front")
        return None

    # Convert to numpy array
    data_points = np.array(data_points)

    # Compute Pareto front
    is_pareto = compute_pareto_front(data_points)
    pareto_points = data_points[is_pareto]
    pareto_labels = [param_labels[i] for i in range(len(param_labels)) if is_pareto[i]]

    # Sort Pareto points by return value for nice plotting
    sort_idx = np.argsort(pareto_points[:, 0])
    pareto_points = pareto_points[sort_idx]
    pareto_labels = [pareto_labels[i] for i in sort_idx]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot all points (using negative sensitivity for y-axis, so lower is better = more negative)
    ax.scatter(
        data_points[:, 0],
        data_points[:, 1],
        c="lightgray",
        s=100,
        alpha=0.6,
        label="All configurations",
        zorder=1,
    )

    # Plot Pareto front points
    ax.scatter(
        pareto_points[:, 0],
        pareto_points[:, 1],
        c="red",
        s=200,
        marker="*",
        edgecolors="darkred",
        linewidth=1.5,
        label="Pareto front",
        zorder=3,
    )

    # Plot Pareto front line
    ax.plot(
        pareto_points[:, 0],
        pareto_points[:, 1],
        "r--",
        linewidth=2,
        alpha=0.5,
        zorder=2,
    )

    # Find best trade-off point (compute trade-off scores)
    return_vals = [p[0] for p in pareto_points]
    sensitivity_vals = [-p[1] for p in pareto_points]  # Convert to positive for scoring

    best_idx = 0
    best_score = -1
    for i in range(len(pareto_points)):
        norm_return = (return_vals[i] - min(return_vals)) / (
            max(return_vals) - min(return_vals) + 1e-10
        )
        norm_sensitivity = 1 - (sensitivity_vals[i] - min(sensitivity_vals)) / (
            max(sensitivity_vals) - min(sensitivity_vals) + 1e-10
        )
        tradeoff_score = (norm_return + norm_sensitivity) / 2
        if tradeoff_score > best_score:
            best_score = tradeoff_score
            best_idx = i

    # Highlight best trade-off point
    best_point = pareto_points[best_idx]
    best_label = pareto_labels[best_idx]
    ax.scatter(
        [best_point[0]],
        [best_point[1]],
        c="gold",
        s=400,
        marker="*",
        edgecolors="orange",
        linewidth=2.5,
        label="Best trade-off",
        zorder=4,
    )
    ax.annotate(
        f"Best: {best_label}",
        (best_point[0], best_point[1]),
        xytext=(15, 15),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="gold", alpha=0.8),
        arrowprops=dict(
            arrowstyle="->", connectionstyle="arc3,rad=0.2", color="orange", lw=2
        ),
    )

    ax.set_xlabel("Episodic Return (Higher is Better)", fontsize=14, fontweight="bold")
    ax.set_ylabel(
        "Negative Sensitivity Mean (Higher is Better = Lower Sensitivity)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_title(
        "Pareto Front: Maximize Return vs Minimize Sensitivity\n(Optimal trade-offs)",
        fontsize=16,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Save plot
    output_file = output_dir / "pareto_front.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved Pareto front plot: {output_file}")

    # Return Pareto front data for table
    pareto_data = []
    for i, (point, label) in enumerate(zip(pareto_points, pareto_labels)):
        pareto_data.append(
            {
                "config": label,
                "episodic_return": point[0],
                "sensitivity_mean_negative": point[1],  # Keep negative for clarity
                "sensitivity_mean": -point[1],  # Also store positive for reference
            }
        )

    return pareto_data


def create_pareto_table(pareto_data: List[Dict], output_dir: Path):
    """
    Create a table with Pareto front configurations showing trade-offs.

    Args:
        pareto_data: List of dictionaries with Pareto front points
        output_dir: Directory to save table
    """
    if not pareto_data:
        print("No Pareto front data to create table")
        return

    # Save as JSON
    output_file = output_dir / "pareto_front_table.json"
    with open(output_file, "w") as f:
        json.dump(pareto_data, f, indent=2)

    print(f"Saved Pareto front table: {output_file}")

    # Print table to console
    print("\n" + "=" * 110)
    print(
        "PARETO FRONT TABLE - BEST TRADE-OFFS (Maximize Return & Minimize Sensitivity)"
    )
    print("=" * 110)
    print(
        f"{'Configuration':<40} {'Return':>15} {'-Sensitivity':>15} {'Sensitivity':>15} {'Trade-off Score':>15}"
    )
    print("-" * 110)

    for item in pareto_data:
        # Normalize values for trade-off score (higher return, lower sensitivity = better)
        # Simple approach: normalize to [0,1] and compute weighted sum
        return_vals = [d["episodic_return"] for d in pareto_data]
        sensitivity_vals = [d["sensitivity_mean"] for d in pareto_data]

        # Normalize return (higher is better)
        norm_return = (item["episodic_return"] - min(return_vals)) / (
            max(return_vals) - min(return_vals) + 1e-10
        )
        # Normalize sensitivity (lower is better, so invert)
        norm_sensitivity = 1 - (item["sensitivity_mean"] - min(sensitivity_vals)) / (
            max(sensitivity_vals) - min(sensitivity_vals) + 1e-10
        )

        # Weighted trade-off score (equal weights)
        tradeoff_score = (norm_return + norm_sensitivity) / 2

        print(
            f"{item['config']:<40} {item['episodic_return']:>15.2f} {item['sensitivity_mean_negative']:>15.6f} {item['sensitivity_mean']:>15.6f} {tradeoff_score:>15.4f}"
        )

    print("=" * 110)
    print(
        "\nNOTE: -Sensitivity column shows negative values (higher = better = lower actual sensitivity)"
    )
    print("      Sensitivity column shows positive values (lower = better)")
    print(
        "Trade-off Score: Balanced metric considering both high return and low sensitivity (range: 0-1, higher is better)"
    )
    print("=" * 110 + "\n")


def plot_sweep_results(sweep_dir: Path, output_subdir: str = "plots"):
    """
    Main function to generate all plots and summaries for a sweep.

    Args:
        sweep_dir: Path to sweep directory containing experiment subdirectories
        output_subdir: Subdirectory name for plots (default: "plots")
    """
    sweep_dir = Path(sweep_dir)

    if not sweep_dir.exists():
        print(f"Error: Sweep directory not found: {sweep_dir}")
        return

    print(f"\n{'='*80}")
    print(f"PLOTTING SWEEP RESULTS")
    print(f"{'='*80}")
    print(f"Sweep directory: {sweep_dir}")

    # Create output directory
    output_dir = sweep_dir / output_subdir
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Group experiments by parameters
    print("Grouping experiments by parameters...")
    grouped_experiments = group_experiments_by_params(sweep_dir)
    print(f"Found {len(grouped_experiments)} unique parameter combinations\n")

    # Print parameter groups for debugging
    for i, (param_key, exp_dirs) in enumerate(sorted(grouped_experiments.items()), 1):
        params_dict = dict(param_key)
        print(f"  Group {i}: {params_dict}")
        print(
            f"    Experiments: {len(exp_dirs)} (seeds: {sorted([parse_experiment_name(d.name).get('seed', 'N/A') for d in exp_dirs])})"
        )
    print()

    # Define metrics to plot
    metrics = [
        ("returns/episodic_return_true", "Episodic Return (True Reward)"),
        ("returns/sensitivity_state_mean", "Sensitivity State Mean"),
        ("returns/sensitivity_state_max", "Sensitivity State Max"),
    ]

    # Create plots for each metric
    print("Generating plots...")
    for metric_tag, metric_name in metrics:
        print(f"  Plotting: {metric_name}")
        plot_metric_comparison(
            sweep_dir, grouped_experiments, metric_tag, metric_name, output_dir
        )

    # Create summary table
    print("\nGenerating summary table...")
    create_summary_table(sweep_dir, grouped_experiments, output_dir)

    # Create Pareto front plot and table
    print("\nGenerating Pareto front analysis...")
    pareto_data = plot_pareto_front(sweep_dir, grouped_experiments, output_dir)
    if pareto_data:
        create_pareto_table(pareto_data, output_dir)

    print(f"\n{'='*80}")
    print(f"PLOTTING COMPLETED")
    print(f"{'='*80}")
    print(f"All plots saved in: {output_dir}")
    print(f"{'='*80}\n")


def main():
    """Command-line interface for standalone usage."""
    parser = argparse.ArgumentParser(
        description="Plot sweep results from TensorBoard logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot results from a specific sweep
  python plot_sweep_results.py --sweep_dir runs/sweep_20251209-123456
  
  # Plot results from latest sweep
  python plot_sweep_results.py --sweep_dir runs/latest
  
  # Specify custom output directory
  python plot_sweep_results.py --sweep_dir runs/sweep_20251209-123456 --output_dir my_plots
        """,
    )
    parser.add_argument(
        "--sweep_dir",
        type=Path,
        required=True,
        help="Path to sweep directory containing experiment subdirectories",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots",
        help="Subdirectory name for output plots (default: plots)",
    )

    args = parser.parse_args()

    plot_sweep_results(args.sweep_dir, args.output_dir)


if __name__ == "__main__":
    main()