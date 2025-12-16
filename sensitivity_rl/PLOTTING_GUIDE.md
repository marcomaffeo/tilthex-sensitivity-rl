# Sweep Results Plotting Guide

## Overview

The `plot_sweep_results.py` script automatically generates plots and summary tables from your parameter sweep experiments. It aggregates results across different random seeds and creates visualizations showing mean ± standard deviation for key metrics.

## Features

- **Automatic aggregation** across random seeds
- **Mean and standard deviation** plots with confidence bands
- **Multiple metrics** tracked:
  - Episodic Return (True Reward)
  - Sensitivity State Mean
  - Sensitivity State Max
- **Summary table** with final performance metrics
- **Flexible usage**: Can be run from tuner or standalone

## Usage

### Method 1: Automatic (via Tuner)

When running sweeps with `tuner.py`, plots are automatically generated after completion if `GENERATE_PLOTS = True`:

```python
# In tuner.py
GENERATE_PLOTS = True  # Enable automatic plotting

# Run the tuner
python tuner.py
```

Plots will be saved in: `runs/sweep_TIMESTAMP/plots/`

### Method 2: Standalone

Generate plots for an existing sweep directory:

```bash
# Plot specific sweep
python plot_sweep_results.py --sweep_dir runs/sweep_20251209-123456

# Plot latest sweep
python plot_sweep_results.py --sweep_dir runs/latest

# Specify custom output directory
python plot_sweep_results.py --sweep_dir runs/sweep_20251209-123456 --output_dir my_plots
```

## Output Files

After running, you'll find in the `plots/` subdirectory:

1. **`returns_episodic_return_true.png`** - Learning curves for true episodic returns
2. **`returns_sensitivity_state_mean.png`** - Sensitivity state mean over training
3. **`returns_sensitivity_state_max.png`** - Maximum sensitivity state over training
4. **`summary.json`** - Final metrics table (JSON format)

## Plot Interpretation

Each plot shows:
- **Solid lines**: Mean value across all seeds for each parameter setting
- **Shaded regions**: ±1 standard deviation confidence band
- **X-axis**: Training steps
- **Y-axis**: Metric value
- **Legend**: Parameter settings that differ between experiments

## Example Workflow

```bash
# 1. Configure sweep parameters in tuner.py
# Edit SWEEP_PARAMS, METHODS, ENVIRONMENTS

# 2. Run the sweep
python tuner.py

# 3. Plots are automatically generated in runs/sweep_TIMESTAMP/plots/

# 4. (Optional) Regenerate plots later
python plot_sweep_results.py --sweep_dir runs/latest
```

## Requirements

Make sure you have the required packages:

```bash
pip install matplotlib tensorboard numpy
```

Or if using pixi:

```bash
pixi add matplotlib tensorboard numpy
```

## Troubleshooting

**Issue: "plot_sweep_results module not found"**
- Make sure `plot_sweep_results.py` is in the same directory as `tuner.py`
- Or add the directory to your Python path

**Issue: "Tag not found in TensorBoard logs"**
- Check that your experiments logged the required metrics
- Verify TensorBoard event files exist in experiment directories

**Issue: Empty plots**
- Ensure experiments completed successfully
- Check that experiments have the same metric tags
- Verify TensorBoard logs are not corrupted

## Customization

To plot additional metrics, edit `plot_sweep_results.py`:

```python
# Add your metric to the list
metrics = [
    ('returns/episodic_return_true', 'Episodic Return (True Reward)'),
    ('returns/sensitivity_state_mean', 'Sensitivity State Mean'),
    ('returns/sensitivity_state_max', 'Sensitivity State Max'),
    ('your/custom/metric', 'Your Custom Metric'),  # Add here
]
```

## Summary Table Format

The `summary.json` file contains final metric values:

```json
[
  {
    "params": {
      "method": "sppovsa",
      "env": "halfcheetah_v5",
      "srw": 0.5,
      ...
    },
    "returns/episodic_return_true_mean": 4250.5,
    "returns/episodic_return_true_std": 125.3,
    "returns/sensitivity_state_mean_mean": 0.045,
    "returns/sensitivity_state_mean_std": 0.008,
    ...
  }
]
```

These values represent the average of the final 10% of training for each metric.
