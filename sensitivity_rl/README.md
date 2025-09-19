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
`pixi run -e cuda python -- -c "import jax; print(jax.devices())"`

Notes:
- The CUDA wheels typically bundle the necessary CUDA runtime components; you still need a compatible NVIDIA driver on the host.
- macOS should use the default CPU environment.

## Common tasks

Pixi tasks are defined in `pixi.toml`. Use `pixi run <task>` to execute them, optionally with `-e cuda` to select the CUDA environment.

Train PPO (writes latest artifacts to `runs/latest`):
- CPU: `pixi run train`
- CUDA: `pixi run -e cuda train`

Analyze a run (generate plots/CSV/summary):
- `pixi run analyze -- runs/latest`
- Add smoothing: `pixi run analyze -- runs/latest --smooth 21`
- Compare multiple runs: `pixi run analyze -- runs/ppo_* --compare --out_dir runs/compare`

Generate rollouts from a saved model:
- `pixi run rollouts --run_dirs runs/latest --num_rollouts 100 --batch_size 256 --deterministic`
- Add per-trajectory plots: append `--plot --plot_format png` (or pdf)

Sensitivity/Jacobian checks:
`pixi run test-sensitivity`

Package/build (optional):
- Build wheel/sdist: `pixi run build`
- Editable install of this package: `pixi run install`



## Quickstart

CPU:
1) `pixi install`
2) `pixi run train`
3) `pixi run analyze -- runs/latest`
4) `pixi run rollouts -- --run_dirs runs/latest --num_rollouts 100 --batch_size 256 --deterministic`

CUDA (Linux, CUDA 12):
1) `pixi install -e cuda`
2) `pixi run -e cuda python -- -c "import jax; print(jax.devices())"`  (confirm GPU)
3) `pixi run -e cuda train`
4) `pixi run analyze -- runs/latest`
5) `pixi run rollouts -- --run_dirs runs/latest --num_rollouts 100 --batch_size 256 --deterministic`

## Notes

- Default environment (no `-e`) installs CPU JAX and should work on macOS and Linux without a GPU.
- CUDA environment (`-e cuda`) installs `jax[cuda12]` and targets Linux with a compatible NVIDIA driver.
