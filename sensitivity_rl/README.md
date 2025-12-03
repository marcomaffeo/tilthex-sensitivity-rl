# sensitivity-rl

## Requirements

Install `pixi` following [https://pixi.sh/latest/installation/](https://pixi.sh/latest/installation/)

Once Pixi is installed, all commands below are run from the repository root (the folder containing `pixi.toml`).

## Environments

This project defines:
- A default CPU environment
- A CUDA (CUDA 12) environment named `cuda` (Linux only)

JAX is installed by default in the CPU environment. The CUDA environment uses the official GPU-enabled JAX wheel (`jax[cuda12]`) on Linux.

### Create the CPU environment (default)

Install dependencies:
`pixi install`

### Create the CUDA environment (Linux, CUDA 12)

Install dependencies for the `cuda` env:
`pixi install -e cuda`

Verify JAX sees your GPU:
`pixi run -e cuda python -c "import jax; print(jax.devices())"`

Notes:
- The CUDA wheels typically bundle the necessary CUDA runtime components; you still need a compatible NVIDIA driver on the host.
- macOS should use the default CPU environment.

## Common tasks

Pixi tasks are defined in `pixi.toml`. Use `pixi run <task>` to execute them, optionally with `-e cuda` to select the CUDA environment.

Train PPO (writes latest artifacts to `runs/latest`):
- CPU: `pixi run train`
- CUDA: `pixi run -e cuda train`

### Configuration files

- Training hyperparameters are stored as JSON files in the `configs/` directory (for example `configs/drone_v0.json`).
- The `train` task loads `configs/default.json` by default. Choose a different config via `pixi run train --config configs/<name>.json`.
- Whatever config you pass is copied into the output run directory as `config.json`, alongside the learned parameters.

Generate rollouts from a saved model:
- `pixi run rollouts --run_dirs runs/latest --num_rollouts 100 --deterministic`

Generate rollouts from a saved model (with perturbed parameters):
- `pixi run rollouts --run_dirs runs/latest --num_rollouts 100 --param_samples 10 --deterministic`

Sensitivity/Jacobian checks:
`pixi run test-sensitivity`

Package/build (optional):
- Build wheel/sdist: `pixi run build`
- Editable install of this package: `pixi run install`



## Quickstart

### Training
CPU:
1) `pixi install`
2) `pixi run train`

CUDA (Linux, CUDA 12):
1) `pixi install -e cuda`
2) `pixi run -e cuda python -c "import jax; print(jax.devices())"`  (confirm GPU)
3) `pixi run -e cuda train`

### Testing

1) `pixi run rollouts`
2) `pixi run plot_drone`
3) `pixi run clean` (deletes all training results)


## Usage
### Training
The training of the non-robustified policy with ppo can be launched with 
`pixi run -e cuda train_ppo --config configs/drone_v0_ppo.json`

This will create logs in the runs directory and plots of the drone position in perturbed and nominal case

The robustified version can then be run with 
`pixi run -e cuda train_sppo --config configs/drone_v0.json`


### Visualize training logs
In a second shell, open the connection with tensorboard and follow the link printed on the terminal to open a web browser
`pixi run tensorboard --logdir ./runs/drone_v0/ --bind_all`

## Notes

- Default environment (no `-e`) installs CPU JAX and should work on macOS and Linux without a GPU.
- CUDA environment (`-e cuda`) installs `jax[cuda12]` and targets Linux with a compatible NVIDIA driver.
