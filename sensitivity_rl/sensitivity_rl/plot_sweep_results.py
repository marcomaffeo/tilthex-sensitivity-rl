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
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorboard.backend.event_processing import event_accumulator


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
    parts = exp_name.split('_')
    params = {}
    
    i = 0
    while i < len(parts):
        # Extract method (after exp_XXX)
        if parts[i].startswith('exp') and i + 1 < len(parts):
            # Skip exp_XXX numbering
            i += 1
            if i < len(parts) and parts[i].isdigit():
                i += 1
            # Next part is method name
            if i < len(parts):
                params['method'] = parts[i]
                i += 1
            continue
        
        # Extract environment (look for common env names)
        if parts[i] in ['halfcheetah', 'drone', 'pendulum', 'hopper', 'walker'] and i + 1 < len(parts):
            params['env'] = f"{parts[i]}_{parts[i+1]}"
            i += 2
            continue
        
        # Extract parameters that start with known prefixes followed by value
        # Check if current part contains letters followed by digits (e.g., seed0, srw0.5)
        import re
        match = re.match(r'^([a-zA-Z]+)([-+]?[0-9]*\.?[0-9]+)$', parts[i])
        if match:
            param_name = match.group(1)
            param_value_str = match.group(2)
            try:
                param_value = float(param_value_str) if '.' in param_value_str else int(param_value_str)
                params[param_name] = param_value
                i += 1
                continue
            except ValueError:
                pass
        
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
        if not exp_dir.is_dir() or not exp_dir.name.startswith('exp_'):
            continue
        
        params = parse_experiment_name(exp_dir.name)
        
        # Create key from all params except seed
        param_key = tuple(sorted([(k, v) for k, v in params.items() if k != 'seed']))
        experiments[param_key].append(exp_dir)
    
    return experiments


def interpolate_to_common_steps(data_list: List[Tuple[np.ndarray, np.ndarray]], 
                                num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
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


def plot_metric_comparison(sweep_dir: Path, 
                           grouped_experiments: Dict[Tuple, List[Path]],
                           metric_tag: str,
                           metric_name: str,
                           output_dir: Path):
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
            if k not in ['method', 'env', 'seed']:
                # Format parameter for label
                param_label = k.upper() if len(k) <= 3 else k.replace('_', ' ').title()
                if isinstance(v, float):
                    label_parts.append(f"{param_label}={v:.3g}")
                else:
                    label_parts.append(f"{param_label}={v}")
        
        # If no varying parameters, show number of seeds
        if not label_parts:
            label = f"Default (n={len(exp_dirs)} seeds)"
        else:
            label = ", ".join(label_parts) + f" (n={len(exp_dirs)})"
        
        # Plot mean with shaded std
        line = ax.plot(common_steps, mean_values, label=label, linewidth=2, alpha=0.8)
        color = line[0].get_color()
        ax.fill_between(common_steps, 
                        mean_values - std_values, 
                        mean_values + std_values, 
                        alpha=0.15,
                        color=color)
    
    # Format plot
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    
    # Try to extract method and env from first group for title
    method_name = "Unknown"
    env_name = "Unknown"
    if grouped_experiments:
        first_key = list(grouped_experiments.keys())[0]
        first_params = dict(first_key)
        method_name = first_params.get('method', 'Unknown')
        env_name = first_params.get('env', 'Unknown')
    
    ax.set_title(f'{metric_name} - {method_name.upper()} on {env_name}\n(Mean ± Std across seeds)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Save plot
    output_file = output_dir / f"{metric_tag.replace('/', '_')}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {output_file}")


def create_summary_table(sweep_dir: Path, 
                         grouped_experiments: Dict[Tuple, List[Path]],
                         output_dir: Path):
    """
    Create a summary table with final metric values.
    
    Args:
        sweep_dir: Path to sweep directory
        grouped_experiments: Dictionary of parameter groups to experiment directories
        output_dir: Directory to save summary
    """
    metrics = [
        'returns/episodic_return_true',
        'returns/sensitivity_state_mean',
        'returns/sensitivity_state_max'
    ]
    
    summary = []
    
    for param_key, exp_dirs in sorted(grouped_experiments.items()):
        if len(exp_dirs) == 0:
            continue
        
        params_dict = dict(param_key)
        row = {'params': params_dict}
        
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
                row[f'{metric_tag}_mean'] = np.mean(final_values)
                row[f'{metric_tag}_std'] = np.std(final_values)
            else:
                row[f'{metric_tag}_mean'] = np.nan
                row[f'{metric_tag}_std'] = np.nan
        
        summary.append(row)
    
    # Save as JSON
    output_file = output_dir / 'summary.json'
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved summary: {output_file}")
    
    # Print table to console
    print("\n" + "="*80)
    print("SUMMARY TABLE (Final 10% average)")
    print("="*80)
    for row in summary:
        print(f"\nParameters: {row['params']}")
        for metric_tag in metrics:
            mean_key = f'{metric_tag}_mean'
            std_key = f'{metric_tag}_std'
            if mean_key in row and not np.isnan(row[mean_key]):
                print(f"  {metric_tag}: {row[mean_key]:.4f} ± {row[std_key]:.4f}")
    print("="*80 + "\n")


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
        print(f"    Experiments: {len(exp_dirs)} (seeds: {sorted([parse_experiment_name(d.name).get('seed', 'N/A') for d in exp_dirs])})")
    print()
    
    # Define metrics to plot
    metrics = [
        ('returns/episodic_return_true', 'Episodic Return (True Reward)'),
        ('returns/sensitivity_state_mean', 'Sensitivity State Mean'),
        ('returns/sensitivity_state_max', 'Sensitivity State Max'),
    ]
    
    # Create plots for each metric
    print("Generating plots...")
    for metric_tag, metric_name in metrics:
        print(f"  Plotting: {metric_name}")
        plot_metric_comparison(sweep_dir, grouped_experiments, metric_tag, metric_name, output_dir)
    
    # Create summary table
    print("\nGenerating summary table...")
    create_summary_table(sweep_dir, grouped_experiments, output_dir)
    
    print(f"\n{'='*80}")
    print(f"PLOTTING COMPLETED")
    print(f"{'='*80}")
    print(f"All plots saved in: {output_dir}")
    print(f"{'='*80}\n")


def main():
    """Command-line interface for standalone usage."""
    parser = argparse.ArgumentParser(
        description='Plot sweep results from TensorBoard logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot results from a specific sweep
  python plot_sweep_results.py --sweep_dir runs/sweep_20251209-123456
  
  # Plot results from latest sweep
  python plot_sweep_results.py --sweep_dir runs/latest
  
  # Specify custom output directory
  python plot_sweep_results.py --sweep_dir runs/sweep_20251209-123456 --output_dir my_plots
        """
    )
    parser.add_argument(
        '--sweep_dir',
        type=Path,
        required=True,
        help='Path to sweep directory containing experiment subdirectories'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='plots',
        help='Subdirectory name for output plots (default: plots)'
    )
    
    args = parser.parse_args()
    
    plot_sweep_results(args.sweep_dir, args.output_dir)


if __name__ == "__main__":
    main()
