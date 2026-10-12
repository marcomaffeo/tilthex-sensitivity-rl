# sensitivity_rl — training

JAX framework to train control policies with PPO and with sensitivity-aware variants of PPO (SPPO).
Environments, algorithms and the training loop are written in JAX, so a whole training run
(thousands of parallel environments, rollouts and updates) is compiled and runs on CPU or GPU.

Part of [Tilthex Sensitivity RL](../README.md).

## Installation

Requires [pixi](https://pixi.sh). From this folder:

```bash
pixi install                 # default environment: CPU, macOS (Apple Silicon) and Linux x86_64
pixi install -e cuda         # optional, Linux only: JAX with CUDA 12 (needs an NVIDIA driver)
pixi run -e cuda python -c "import jax; print(jax.devices())"   # check that the GPU is visible
```

Resolved versions (same on macOS and Linux): Python 3.12, JAX 0.9.1, Flax 0.12, distrax 0.1.8+.

## Quick check

```bash
pixi run python sensitivity_rl/train.py ALGORITHM=ppo env=pendulum
```

About 3 minutes on a laptop CPU. The evaluation return printed during training goes from about −1200 to about −150.

## Environments

| Name | Description |
|---|---|
| `tilthex_v1` | The Tilthex of the thesis: hexarotor with tilted propellers, fully actuated, six rotor forces as actions, quaternion attitude, sensitivity with respect to mass, inertia, centre of mass, propeller tilt angles, thrust coefficient and actuation delay. Model parameters from the LAAS Tilthex. |
| `drone_v3` | Generic quadrotor (Mikrokopter): uncertain mass, inertia and centre of mass, first-order actuator dynamics. |
| `pendulum` | Pendulum swing-up, the minimal example. |

All environments are in `sensitivity_rl/envs/` and are registered in `envs/__init__.py` (`ENV_REGISTRY`).
`envs/wrappers.py` holds the wrappers shared by all algorithms: sensitivity computation, reward
normalisation, domain randomisation, constraints, logging.

## Algorithms

Selected with `ALGORITHM=<name>`; all are in `sensitivity_rl/algorithms/` and registered in `algorithms/__init__.py`.

| Name | Description |
|---|---|
| `ppo` | Standard PPO. With `SENSITIVITY: true` the sensitivity is computed and logged, but it does not enter the loss. |
| `sppo_aug_grad_param_dual` | **The SPPO of the thesis.** A second, recurrent (GRU) critic estimates the sensitivity return; the policy advantage combines task and sensitivity advantages with weight `SENSITIVITY_REGULARIZATION_WEIGHT`, adapted online (`XI`, `RHO`), plus a proximity term on the actor (`TAU`). |
| `sppo`, `sppo_aug`, `sppo_grad`, `sppo_grad_param`, `sppo_aug_grad_param` | Earlier sensitivity-aware variants explored in the framework. |
| `cppo`, `rappo` | Risk-aware baselines: CPPO with a CVaR constraint, RAPPO with a distributional (quantile) critic. |

## Configuration (Hydra)

Configurations are in `configs/hydra/` and are composed by [Hydra](https://hydra.cc):

```
configs/hydra/
├── config.yaml        # entry point: default method and environment, seed, evaluation settings
├── method/ppo.yaml    # algorithm hyperparameters (default ALGORITHM: sppo_aug_grad_param_dual)
├── method/cppo.yaml, method/rappo.yaml
└── env/               # one file per environment: tilthex_v1.yaml, drone_v3.yaml, pendulum.yaml
```

`config.yaml` selects `method: ppo` and `env: tilthex_v1`, and its own values are applied last.
Any value can be overridden on the command line:

```bash
pixi run python sensitivity_rl/train.py env=drone_v3 ALGORITHM=ppo NUM_ENVS=1024 LR=0.0005
```

Note: `SEED` is set in `config.yaml` (30) and therefore overrides the one in the environment file;
pass `SEED=...` explicitly when the seed matters.

## Training

```bash
pixi run python sensitivity_rl/train.py                      # SPPO on tilthex_v1 (the defaults)
pixi run -e cuda python sensitivity_rl/train.py              # same, on GPU
```

Each run writes a folder `runs/<env>/<ALGORITHM>_<timestamp>/` (and updates the `runs/latest` link) with:

| File | Content |
|---|---|
| `config.json` | the full configuration used |
| `policy_params.msgpack` | the trained network parameters |
| `evaluation_summary.json` | evaluation return and sensitivity, every `EVAL_EVERY_UPDATES` updates |
| `events.out.tfevents.*` | TensorBoard logs |
| `rollouts/`, `rollout_data.npz`, `rollouts_nominal.png`, `rollouts_perturbed.png` | final evaluation rollouts on the nominal model and on perturbed models |

Follow a training with TensorBoard:

```bash
pixi run tensorboard --logdir runs
```

### Reproducing the thesis policies

The two flown policies were trained on Linux with an NVIDIA GPU (50 million steps, 2048 parallel environments):

```bash
pixi run -e cuda python sensitivity_rl/train.py env=tilthex_v1 ALGORITHM=sppo_aug_grad_param_dual SEED=14   # SPPO_14
pixi run -e cuda python sensitivity_rl/train.py env=tilthex_v1 ALGORITHM=ppo SEED=14                        # PPO_14
```

The configuration in this repository matches the one saved with SPPO_14. The saved PPO_14 configuration
also has `SENSITIVITY_REGULARIZATION_WEIGHT: 0.05`, which PPO does not use.
The trained policies are in [`../sensitivity_rl_telekyb/configs/policy/`](../sensitivity_rl_telekyb/configs/policy/).

## Evaluation rollouts

`rollouts.py` runs a trained policy and saves trajectories and plots. It runs automatically at the end of
training; it can also be run on any run folder:

```bash
pixi run rollouts --run_dirs runs/latest --num_rollouts 100 --plot                    # nominal model
pixi run rollouts --run_dirs runs/latest --num_rollouts 100 --param_samples 10 --plot # 10 perturbed models per rollout
```

## Hyperparameter sweeps

`sensitivity_rl/tuner.py` runs grids of trainings over methods, environments and parameters, and
`sensitivity_rl/plots/plot_sweep_results.py` aggregates them (mean and standard deviation over seeds,
return–sensitivity Pareto front). The sweep is defined at the top of `tuner.py`
(`METHODS`, `ENVIRONMENTS`, `SWEEP_PARAMS`):

```bash
pixi run python sensitivity_rl/tuner.py
```

More details in [`TUNER_README.md`](TUNER_README.md) and [`Autotuner_guide.md`](Autotuner_guide.md),
written for an earlier version of the framework: some environments named there no longer exist.

## Folder layout

```
sensitivity_rl/
├── train.py          # training entry point (Hydra)
├── rollouts.py       # evaluation rollouts of a trained policy
├── tuner.py          # hyperparameter sweeps
├── algorithms/       # PPO, SPPO variants, CPPO, RAPPO
├── envs/             # environments and wrappers
├── networks.py, net_utils.py, buffers.py, utils.py
├── plots/            # sweep plots, rollout export and ranking
└── benchmark/        # sensitivity checks on the pendulum
configs/hydra/        # configurations
runs/                 # outputs (ignored by git)
```
