# Tuner Guide

The tuner is a powerful tool for running hyperparameter sweeps across multiple methods and environments, with automatic result aggregation and visualization.

## Overview

The tuner allows you to:
- Run multiple experiments with different hyperparameter combinations
- Sweep across different algorithms (PPO, SAC, SPPO variants)
- Test across multiple environments
- Execute experiments in parallel or sequentially
- Automatically organize results in timestamped sweep directories
- Generate plots and Pareto front analysis after completion

## Quick Start

1. **Edit the tuner configuration** in [`sensitivity_rl/tuner.py`](sensitivity_rl/tuner.py):

```python
# Choose methods to test
METHODS = ["sppo_aug"]

# Choose environments
ENVIRONMENTS = ["drone_v1"]

# Define parameter sweep
SWEEP_PARAMS = {
    "SEED": [0, 1, 2],
    "SENSITIVITY_REGULARIZATION_WEIGHT": [0.0, 0.5, 0.7],
    "GAMMA": [0.96, 0.98, 0.99],
}
```

2. **Run the tuner**:

```bash
cd sensitivity_rl
pixi run python sensitivity_rl/tuner.py
```

3. **View results** in `runs/sweep_YYYYMMDD-HHMMSS/` or use the `runs/latest` symlink

## Hydra Configuration Hierarchy

The tuner uses [Hydra](https://hydra.cc/) for flexible configuration composition. Understanding the hierarchy helps you customize experiments effectively.

### Configuration Structure

Configs are organized in layers:

```
configs/hydra/
├── base_config.yaml       # Base defaults and composition
├── method/                # Method-specific configs
│   ├── ppo.yaml          # PPO/SPPO parameters
│   └── sac.yaml          # SAC parameters
└── env/                   # Environment-specific configs
    ├── drone_v0.yaml
    ├── drone_v1.yaml
    └── halfcheetah_v5.yaml
```

### Merge Order (Priority: Low → High)

The tuner composes configurations in the following order:

```
1. base_config.yaml          (lowest priority - defaults)
   ↓ merged with
2. method/{method}.yaml      (method-specific overrides)
   ↓ merged with
3. env/{environment}.yaml    (environment-specific overrides)
   ↓ merged with
4. SWEEP_PARAMS              (highest priority - sweep overrides)
```

**Later configs override earlier ones for the same parameter.**

### Example Configuration Merge

Let's trace how `GAMMA` is determined for an experiment:

#### 1. Base Config (`base_config.yaml`)
```yaml
defaults:
  - method: ppo
  - env: drone_v1
  - _self_

SEED: 0
```
No `GAMMA` defined here.

#### 2. Method Config (`method/ppo.yaml`)
```yaml
# @package _global_
GAMMA: 0.99                    # ← Default GAMMA from method
LR: 0.001
UPDATE_EPOCHS: 10
CLIP_EPS: 0.1
SENSITIVITY_REGULARIZATION_WEIGHT: 0.5
# ... more parameters
```
`GAMMA = 0.99` (from PPO defaults)

#### 3. Environment Config (`env/drone_v1.yaml`)
```yaml
# @package _global_
ENV_NAME: drone_v1
NUM_ENVS: 2048
GAMMA: 0.98                    # ← Overrides method GAMMA
CLIP_EPS: 0.2                  # ← Overrides method CLIP_EPS
SENSITIVITY_REGULARIZATION_WEIGHT: 0.0  # ← Overrides method value
# ... more parameters
```
`GAMMA = 0.98` (environment override wins)

#### 4. Sweep Parameters (in `tuner.py`)
```python
SWEEP_PARAMS = {
    "GAMMA": [0.96, 0.98, 0.99],  # ← Final override
    "CLIP_EPS": [0.1, 0.2],
}
```
`GAMMA = 0.96` (or 0.98, or 0.99 depending on sweep iteration)

**Final Result**: For a given sweep iteration, the parameter values are:
- `GAMMA`: `0.96` (from sweep, overriding environment's `0.98`)
- `LR`: `0.001` (from method, not overridden)
- `CLIP_EPS`: `0.1` or `0.2` (from sweep, overriding environment's `0.2`)
- `SENSITIVITY_REGULARIZATION_WEIGHT`: `0.0` (from environment, overriding method's `0.5`, unless swept)
- `ENV_NAME`: `drone_v1` (from environment, not overridden)

### Hydra Special Directives

#### `@package _global_`
At the top of method and env config files:
```yaml
# @package _global_
```
This tells Hydra to merge these parameters at the global config level (not nested under `method.` or `env.`).

#### `defaults` list
In `base_config.yaml`:
```yaml
defaults:
  - method: ppo      # Load configs/hydra/method/ppo.yaml
  - env: drone_v1    # Load configs/hydra/env/drone_v1.yaml
  - _self_           # Apply base_config.yaml itself (lowest priority)
```

The `_self_` marker determines when base_config's own parameters are applied:
- At the end of list → base_config overrides composed configs (higher priority)
- At the start of list → composed configs override base_config (lower priority)

### Best Practices for Config Organization

#### 1. **Base Config** - Minimal Composition Rules
Only include:
- Defaults list (which configs to compose)
- Universal parameters (like `SEED`)
- No method or environment-specific values

#### 2. **Method Configs** - Algorithm Defaults
Include:
- Learning hyperparameters (`LR`, `UPDATE_EPOCHS`)
- Algorithm-specific parameters (`GAMMA`, `GAE_LAMBDA`, `CLIP_EPS`)
- Network architecture defaults
- Sensitivity settings

#### 3. **Environment Configs** - Task-Specific Tuning
Include:
- `ENV_NAME` (required)
- Environment-specific overrides (e.g., different `NUM_ENVS`, `GAMMA`)
- Task-specific network architectures
- Optimized hyperparameters for this environment

#### 4. **Sweep Parameters** - Experimental Variables
Include:
- Parameters you want to compare
- Multiple seeds for statistical robustness
- Hyperparameters you're tuning

### Adding New Configurations

#### To add a new method:

1. Create `configs/hydra/method/my_method.yaml`:
```yaml
# @package _global_
# My Method configuration

LR: 0.0003
UPDATE_EPOCHS: 5
# ... your parameters
```

2. Add mapping in `tuner.py`:
```python
METHOD_CONFIG_NAMES = {
    "my-method-script": "my_method",
}
```

3. Use in sweep:
```python
METHODS = ["my-method-script"]
```

#### To add a new environment:

1. Create `configs/hydra/env/my_env.yaml`:
```yaml
# @package _global_
ENV_NAME: my_env

NUM_ENVS: 100
TOTAL_TIMESTEPS: 5e6
# ... environment-specific overrides
```

2. Use in sweep:
```python
ENVIRONMENTS = ["my_env"]
```

### Debugging Configuration

To see the final merged configuration for an experiment:

1. Check the generated `config.json` in each experiment directory:
```bash
cat runs/latest/exp_001_*/config.json | jq
```

2. Add debug printing in `tuner.py`:
```python
# After config merging
print("Final config:", json.dumps(config, indent=2))
```

3. Use Hydra's composition order to trace parameter sources:
```
base_config.yaml → method/ppo.yaml → env/drone_v1.yaml → SWEEP_PARAMS
```

## Configuration Options

### Methods

Available methods:
- `"ppo"` - Standard Proximal Policy Optimization
- `"sppo-vs-estimation"` - SPPO with value-state sensitivity estimation
- `"sppo-vs-estimation-aug"` - SPPO with augmented sensitivity estimation (recommended)
- `"sac"` - Soft Actor-Critic
- `"sac-estimation"` - SAC with sensitivity estimation

### Environments

Common environments (ensure config files exist in `configs/hydra/env/`):
- `"drone_v0"` - Drone control task (v0)
- `"drone_v1"` - Drone control task (v1)
- `"halfcheetah_v5"` - HalfCheetah locomotion
- `"pendulum_v0"` - Inverted pendulum

### Sweep Parameters

Common parameters to sweep:

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `SEED` | Random seed | `[0, 1, 2, 3, 4]` |
| `SENSITIVITY_REGULARIZATION_WEIGHT` | Weight for sensitivity regularization | `[0.0, 0.3, 0.5, 0.7, 0.9]` |
| `GAMMA` | Discount factor | `[0.96, 0.98, 0.99]` |
| `LR` or `LEARNING_RATE` | Learning rate | `[0.001, 0.0005, 0.0001]` |
| `CLIP_EPS` | PPO clipping epsilon | `[0.1, 0.2, 0.3]` |
| `UPDATE_EPOCHS` | Number of update epochs per step | `[5, 10, 20]` |
| `ENT_COEF` | Entropy coefficient | `[0.0, 0.001, 0.005, 0.01]` |
| `NUM_ENVS` | Number of parallel environments | `[25, 50, 100]` |
| `GAE_LAMBDA` | GAE lambda parameter | `[0.9, 0.95, 0.99]` |

### Execution Settings

```python
# Number of experiments to run in parallel
# Set to 1 for sequential execution (safer, less memory)
# Set to N for parallel execution (faster, requires more resources)
NUM_PARALLEL_JOBS = 5

# Enable/disable evaluation rollouts after training
RUN_EVAL_ROLLOUTS = True

# Enable/disable automatic plot generation
GENERATE_PLOTS = True

# Evaluation configuration
EVAL_CONFIG = {
    "NUM_ROLLOUTS": 10,      # Number of evaluation episodes
    "MAX_STEPS": 1000,       # Max steps per episode
    "PARAM_SAMPLES": 10,     # Number of parameter samples for sensitivity
}
```

## Output Structure

After running a sweep, results are organized as:

```
runs/
├── sweep_20251217-143022/
│   ├── sweep_config.json           # Sweep configuration
│   ├── exp_001_sppovsa_drone_v1_seed0_srw0.0_gamma0.96/
│   │   ├── config.json             # Experiment configuration
│   │   ├── events.out.tfevents.*   # TensorBoard logs
│   │   └── checkpoints/            # Model checkpoints
│   ├── exp_002_sppovsa_drone_v1_seed0_srw0.5_gamma0.96/
│   ├── ...
│   └── plots/                      # Generated plots
│       ├── returns_episodic_return_true.png
│       ├── returns_sensitivity_state_mean.png
│       ├── returns_sensitivity_state_max.png
│       ├── pareto_front.png        # Pareto front visualization
│       ├── pareto_front_table.json # Best trade-off configurations
│       └── summary.json            # Summary statistics
└── latest -> sweep_20251217-143022  # Symlink to latest sweep
```

## Experiment Naming Convention

Experiments are automatically named based on the configuration:

```
exp_{number}_{method}_{environment}_{param1}{value1}_{param2}{value2}...
```

Example:
```
exp_001_sppovsa_drone_v1_seed0_srw0.5_gamma0.98_clip_eps0.2_update_epochs10
```

Parameter name abbreviations:
- `srw` = SENSITIVITY_REGULARIZATION_WEIGHT
- `lr` = LEARNING_RATE
- All parameters are converted to lowercase in the name

## Analyzing Results

### 1. TensorBoard

View training progress in real-time or after completion:

```bash
tensorboard --logdir runs/latest
```

Navigate to http://localhost:6006

### 2. Generated Plots

After sweep completion, plots are automatically generated in `runs/latest/plots/`:

- **Metric plots**: Show mean ± std across seeds for each configuration
  - Episodic return
  - Sensitivity state mean
  - Sensitivity state max

- **Pareto front plot**: Shows optimal trade-offs between return and sensitivity
  - Gray points: All configurations
  - Red stars: Pareto-optimal configurations
  - Gold star: Best trade-off (highest balanced score)

### 3. Summary Tables

**`plots/summary.json`**: Final metric values (last 10% average) for all configurations

**`plots/pareto_front_table.json`**: Pareto-optimal configurations with:
- Configuration parameters
- Episodic return
- Sensitivity mean (both positive and negative representations)
- Trade-off score (normalized balanced metric)

## Tips and Best Practices

### Memory Management

- **Sequential execution** (`NUM_PARALLEL_JOBS = 1`): Safest, minimal memory usage
- **Parallel execution**: Faster but requires more RAM and GPU memory
  - Each parallel job gets `1/NUM_PARALLEL_JOBS` of GPU memory
  - Monitor GPU memory usage: `watch -n 1 nvidia-smi`

### Debugging Failed Experiments

If experiments fail:

1. Check error messages in terminal output
2. Reduce `NUM_PARALLEL_JOBS` to 1 for sequential debugging
3. Verify config files exist for your methods/environments
4. Check GPU memory availability
5. Review individual experiment directories for logs

### Efficient Sweeping

1. **Start small**: Test with 2-3 combinations first
2. **Use multiple seeds**: At least 3-5 seeds for statistical significance
3. **Incremental sweeps**: Sweep one parameter at a time initially
4. **GPU utilization**: Adjust `NUM_PARALLEL_JOBS` based on GPU memory
5. **Resume capability**: Failed experiments can be re-run individually

### Customizing Plots

Modify [`sensitivity_rl/plot_sweep_results.py`](sensitivity_rl/plot_sweep_results.py) to:
- Change plot styles
- Add new metrics
- Adjust Pareto front computation
- Modify trade-off scoring

Run plotting separately:
```bash
python sensitivity_rl/plot_sweep_results.py --sweep_dir runs/latest
```

## Example Workflows

### Example 1: Quick sensitivity weight comparison

```python
METHODS = ["sppo-vs-estimation-aug"]
ENVIRONMENTS = ["drone_v1"]
SWEEP_PARAMS = {
    "SEED": [0, 1, 2],
    "SENSITIVITY_REGULARIZATION_WEIGHT": [0.0, 0.3, 0.5, 0.7, 0.9],
}
NUM_PARALLEL_JOBS = 5
```

Result: 3 × 5 = 15 experiments

### Example 2: Full hyperparameter search

```python
METHODS = ["sppo-vs-estimation-aug"]
ENVIRONMENTS = ["drone_v1"]
SWEEP_PARAMS = {
    "SEED": [0, 1, 2, 3, 4],
    "SENSITIVITY_REGULARIZATION_WEIGHT": [0.0, 0.5, 0.7],
    "GAMMA": [0.96, 0.98, 0.99],
    "CLIP_EPS": [0.1, 0.2],
    "UPDATE_EPOCHS": [5, 10],
    "ENT_COEF": [0.0, 0.001, 0.005],
}
NUM_PARALLEL_JOBS = 10
```

Result: 5 × 3 × 3 × 2 × 2 × 3 = 540 experiments

### Example 3: Multi-environment comparison

```python
METHODS = ["sppo-vs-estimation-aug", "ppo"]
ENVIRONMENTS = ["drone_v1", "halfcheetah_v5"]
SWEEP_PARAMS = {
    "SEED": [0, 1, 2],
    "SENSITIVITY_REGULARIZATION_WEIGHT": [0.0, 0.5],
}
NUM_PARALLEL_JOBS = 4
```

Result: 2 × 2 × 3 × 2 = 24 experiments

## Interpreting Pareto Front Results

The Pareto front analysis helps identify optimal trade-offs between:
- **Episodic return** (maximize): Task performance
- **Sensitivity** (minimize): Robustness to parameter changes

Key insights:
- **Points on the Pareto front**: No other configuration dominates them
- **Best trade-off**: Configuration with highest balanced score
- **High return, high sensitivity**: Peak performance but less robust
- **Low return, low sensitivity**: More robust but lower performance
- **Sweet spot**: Look for configurations with high trade-off scores

## Troubleshooting

### Issue: Out of memory errors

**Solution**: 
- Reduce `NUM_PARALLEL_JOBS`
- Reduce `NUM_ENVS` in sweep parameters
- Use smaller batch sizes

### Issue: No plots generated

**Solution**:
- Check that `GENERATE_PLOTS = True`
- Ensure `plot_sweep_results.py` is in the same directory
- Verify at least one experiment succeeded

### Issue: Experiments not starting

**Solution**:
- Check that config files exist for your methods/environments
- Verify Python environment has all dependencies
- Check `sweep_config.json` for parameter names

### Issue: Very slow execution

**Solution**:
- Increase `NUM_PARALLEL_JOBS` (if memory allows)
- Use GPU instead of CPU (check `JAX_PLATFORMS`)
- Reduce total training steps in base config

## Advanced Usage

### Custom Parameter Mappings

Edit `METHOD_CONFIG_NAMES` dictionary to map new methods to config groups:

```python
METHOD_CONFIG_NAMES = {
    "my-custom-method": "ppo",  # Use PPO config structure
}
```

### Programmatic Sweep Definition

Generate sweeps programmatically:

```python
import numpy as np

SWEEP_PARAMS = {
    "SEED": list(range(5)),
    "SENSITIVITY_REGULARIZATION_WEIGHT": np.linspace(0, 1, 11).tolist(),
    "GAMMA": [0.98, 0.99],
}
```

### Post-Processing Hooks

Add custom analysis after sweep completion by modifying the end of `tuner.py`:

```python
if __name__ == "__main__":
    successful_runs = run_sweep_experiments()
    
    # Add your custom analysis here
    for run_dir in successful_runs:
        # Analyze results, generate custom plots, etc.
        pass
```

## See Also

- [PLOTTING_GUIDE.md](PLOTTING_GUIDE.md) - Detailed guide for visualization
- [README.md](README.md) - Main project documentation
- TensorBoard documentation: https://www.tensorflow.org/tensorboard
