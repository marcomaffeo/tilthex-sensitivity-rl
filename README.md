# Tilthex Sensitivity RL

Sensitivity-aware reinforcement learning for the robust control of a fully actuated aerial robot.

This repository contains the code and the experimental data of the MSc thesis
*Sensitivity-Aware Reinforcement Learning for Fully Actuated Aerial Robot Control*
(MSc in Automation and Robotics Engineering, University of Naples Federico II), developed at IRISA, Rennes.

The platform is the **Tilthex**, a hexarotor with tilted propellers, which makes it fully actuated.
Two policies are trained on the nominal model of the vehicle and compared:

- **PPO**, the standard on-policy algorithm;
- **SPPO**, a sensitivity-aware variant of PPO that also penalises how much the closed-loop
  trajectory changes when the model parameters (mass, inertia, centre of mass, ...) change.

Both policies are validated in a Gazebo simulation and in real flights.

## Repository structure

| Folder | What it contains | Runs on |
|---|---|---|
| [`sensitivity_rl/`](sensitivity_rl/) | Training framework in JAX: environments, PPO/SPPO algorithms, evaluation rollouts | macOS (Apple Silicon) and Linux x86_64, optional NVIDIA GPU |
| [`sensitivity_rl_telekyb/`](sensitivity_rl_telekyb/) | Deployment: ROS 2 node that runs a trained policy on the Tilthex through TeleKyb3, Gazebo simulation, the flown policies | Linux x86_64 (simulation), Linux aarch64 (onboard computer) |
| [`data_analysis/`](data_analysis/) | Analysis of the flight campaigns: flight data, scripts, result tables of the thesis | macOS and Linux (any Python 3 with numpy, scipy, matplotlib) |

Each folder has its own README with the detailed instructions.

## From training to flight

1. **Training.** `sensitivity_rl/sensitivity_rl/train.py` trains a policy on an environment
   (`tilthex_v1` for the thesis) and saves a run folder with `policy_params.msgpack` and `config.json`.
2. **Deployment.** The run folder is copied into `sensitivity_rl_telekyb/configs/policy/`.
   The ROS 2 node loads it, reads the vehicle state from the TeleKyb3 components and sends
   the rotor commands at 100 Hz, in simulation or on the real vehicle.
3. **Logging.** Every flight is saved by the node in a folder with a `data.npz` file
   (states, references, actions, timing).
4. **Analysis.** The scripts in `data_analysis/` read the `data.npz` files of a campaign and produce
   the metrics, the tables and the figures used in the thesis.

## Quick start

All the Python environments are managed with [pixi](https://pixi.sh), which installs every
dependency (Python, JAX, ROS 2, Gazebo, compilers) inside the project folder, without touching the system.

```bash
# 1. Install pixi (see https://pixi.sh for other options)
curl -fsSL https://pixi.sh/install.sh | sh

# 2. Clone the repository
git clone https://github.com/marcomaffeo/tilthex-sensitivity-rl.git
cd tilthex-sensitivity-rl
```

### Train a policy

The pendulum is the quickest check that the installation works: about 3 minutes on a laptop CPU.

```bash
cd sensitivity_rl
pixi install
pixi run python sensitivity_rl/train.py ALGORITHM=ppo env=pendulum
```

Training on the Tilthex, with the algorithm and settings of the thesis:

```bash
pixi run python sensitivity_rl/train.py ALGORITHM=sppo_aug_grad_param_dual env=tilthex_v1   # SPPO
pixi run python sensitivity_rl/train.py ALGORITHM=ppo env=tilthex_v1                        # PPO
```

Results are written to `sensitivity_rl/runs/<env>/<ALGORITHM>_<timestamp>/`.
On Linux with an NVIDIA GPU, use the CUDA environment: `pixi run -e cuda python sensitivity_rl/train.py ...`.
See [`sensitivity_rl/README.md`](sensitivity_rl/README.md) for the environments, the algorithms and all the options.

### Run a policy in simulation

Linux x86_64 only. TeleKyb3 and its components are built from source the first time:

```bash
cd sensitivity_rl_telekyb
pixi run checkout    # download the TeleKyb3 sources
pixi run build       # build them inside the pixi environment
```

The simulation needs several terminals (Gazebo, TeleKyb3 components, policy node, mission commands).
The full procedure is in [`sensitivity_rl_telekyb/README.md`](sensitivity_rl_telekyb/README.md).

### Reproduce the analysis of the thesis

```bash
cd data_analysis
pixi install
pixi run tables      # regenerate every table and figure of the thesis from the flight data (about 2 minutes)
```

See [`data_analysis/README.md`](data_analysis/README.md) for every campaign and script.

## Documentation website

The README files and an API reference generated from the source code are also available as a
website: <https://marcomaffeo.github.io/tilthex-sensitivity-rl/>. It is rebuilt at every push to
`main`. To build it locally:

```bash
cd docs
pixi install
pixi run build      # then open docs/_build/html/index.html
```

## Reproducing the thesis results

| What | Where |
|---|---|
| Code flown in the real-flight campaign (May 2026) | git tag [`flight-campaign`](../../tree/flight-campaign) |
| The two flown policies, PPO_14 and SPPO_14 | `sensitivity_rl_telekyb/configs/policy/PPO/PPO_14/`, `.../SPPO/SPPO_14/` |
| Flight data of the five campaigns (one `data.npz` per flight) | `data_analysis/<campaign>/` |
| Result tables used in the thesis | `data_analysis/<campaign>/data_analysis/tables/`, `data_analysis/*_comparison/tables/` |

PPO_14 and SPPO_14 differ only in the algorithm and in the sensitivity weight: same seed (14),
same hyperparameters, no domain randomisation. The thesis compares them as a paired experiment.

## TeleKyb3

The vehicle-side software is [TeleKyb3](https://git.openrobots.org/projects/telekyb3/pages/index),
the open-source framework of LAAS-CNRS for aerial robots, based on GenoM3 components
(`rotorcraft` for the motors, `pom` for state estimation, `nhfc` for the baseline controller, mocap drivers).
Its official documentation is the place to start to understand the components used here.

A more recent but work-in-progress version of the documentation is in the Inria mirror
[rainbow/software/telekyb3](https://gitlab.inria.fr/rainbow/software/telekyb3/telekyb3)
(access may require an Inria account). It is built locally:

```bash
git clone https://gitlab.inria.fr/rainbow/software/telekyb3/telekyb3.git
cd telekyb3
autoreconf -vi
mkdir build && cd build
../configure --prefix=/any/install/path
make install
# open /any/install/path/share/doc/telekyb3/index.html
# example scripts: /any/install/path/share/doc/telekyb3/examples (bin/ and etc/)
```

## Credits

Thesis work of Marco Maffeo. The training framework in `sensitivity_rl/` was started by
Tommaso Belvedere and Antonio Marino (IRISA), whose commits are kept in the history.
The TeleKyb3 integration builds on the setup by Tommaso Belvedere and Olivier Roussel (Inria).
TeleKyb3 and GenoM3 are developed at LAAS-CNRS.
