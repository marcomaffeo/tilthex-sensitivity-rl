# data_analysis — flight campaigns

Flight data of the Tilthex experiments and the scripts that turn them into the metrics, tables and
figures of the thesis. Everything here can be regenerated from the `data.npz` files alone.

Part of [Tilthex Sensitivity RL](../README.md).

## Installation

Requires [pixi](https://pixi.sh). From this folder:

```bash
pixi install
pixi run tables      # regenerate every table and figure (about 2 minutes)
```

Any Python 3 with `numpy`, `scipy` and `matplotlib` works too: run the scripts from this folder.

## Campaigns

All flights are real flights of the Tilthex in the IRISA flight arena, with the policies PPO and SPPO.

| Folder | Mission | Condition | Flights |
|---|---|---|---|
| `hovering_campaign/` | hold a position for 10 s | nominal | 28: one PPO and one SPPO flight for each of the 14 trained pairs (`PPO_04` … `PPO_18`, `SPPO_04` … `SPPO_18`) |
| `p2p_nominal/` | point-to-point: sequence of position steps | nominal | 2: PPO_14 and SPPO_14 |
| `p2p_perturbed/` | point-to-point | perturbed | 2: PPO_14 and SPPO_14 |
| `rpy_nominal/` | attitude tracking ("tilt mission" in the thesis): sequence of roll, pitch and yaw steps | nominal | 2: PPO_14 and SPPO_14 |
| `rpy_perturbed/` | attitude tracking | perturbed | 2: PPO_14 and SPPO_14 |
| `p2p_comparison/`, `rpy_comparison/` | PPO vs SPPO, nominal vs perturbed | — | results only, no flights |

The hovering campaign is the screening of all the trained pairs; the pair 14 was then selected for the
two missions. In the **perturbed** condition a 200 g weight is attached under the body (about +7 % of
the mass, changing mass and inertia) and an external fan blows on the vehicle.

## Flight data format

Each flight is a folder `<POLICY>_<ID>_<YYYYMMDD>_<HHMMSS>/` with one `data.npz`, written by the
policy node of [`sensitivity_rl_telekyb`](../sensitivity_rl_telekyb/) at the end of the flight
(one sample per control step, 100 Hz):

| Key | Shape | Content |
|---|---|---|
| `t` | (N,) | time [s] |
| `pos`, `vel` | (N, 3) | position and linear velocity |
| `quat` | (N, 4) | attitude quaternion |
| `avel` | (N, 3) | angular velocity |
| `pos_ref`, `quat_ref` | (N, 3), (N, 4) | position and attitude references |
| `action` | (N, 6) | commanded rotor forces [N], one per propeller |
| `inf_time` | (N,) | policy inference time |
| `policy_name` | () | name of the flown policy |

Only `data.npz` is tracked by git; plots, raw GenoM logs and per-flight images are kept outside the repository.

## Scripts

| Script | Input | Output |
|---|---|---|
| `analyze_trajectory.py --input <campaign>` | a mission folder (`p2p_*`, `rpy_*`) | `<campaign>/data_analysis/`: `flight/` (whole flight), `maneuvers/` (flight split by manoeuvre: takeoff, hold, steps, landing), `tables/` |
| `compare_trajectory.py -n <nominal> -p <perturbed>` | the two folders of one mission, already analysed | `<mission>_comparison/`: PPO vs SPPO in both conditions, by mission phase and by manoeuvre category (error, total variation, step response, degradation) |
| `export_rotor_tv.py p2p` / `rpy` | the two folders of one mission | `<mission>_comparison/tables/rotor_tv_regime.csv`: total variation per rotor |
| `analyze_campaign.py --input hovering_campaign` | the hovering campaign | `hovering_campaign/data_analysis/`: RMSE and total variation per flight and per policy, paired PPO/SPPO statistics (Wilcoxon), success rate |

`compare_trajectory.py` does not recompute the metrics: it reads the tables written by
`analyze_trajectory.py`, so the trajectories must be analysed first (`pixi run tables` does it in order).
Run the scripts from this folder. Comments and help messages in the scripts are in Italian.

## Where the thesis numbers come from

| Thesis content | File |
|---|---|
| Hovering screening (Section 7.2) | `hovering_campaign/data_analysis/tables/` (`results.tex`, `paired_stats.csv`, `summary_*.csv`) |
| Mission results per manoeuvre category, PPO vs SPPO | `p2p_comparison/tables/`, `rpy_comparison/tables/` (`comparison_categories.csv`, `results_comparison.tex`) |
| Step response | `*_comparison/tables/step_metrics_range.tex` |
| Total variation per rotor (Fig. 7.11–7.12, right panel) | `*_comparison/tables/rotor_tv_regime.csv` |
| Single-flight metrics | `<campaign>/data_analysis/tables/` |

In every PPO/SPPO comparison the first value is PPO and the second is SPPO
(for example the columns `ppo_nom` and `sppo_nom` of `comparison_categories.csv`).

## Verification

On 12 October 2026 all the tables were regenerated from scratch on Linux from the `data.npz` files
(numpy 2.5, scipy 1.18, matplotlib 3.11). Every value equals the tracked one, up to floating-point rounding
(relative differences below 1e-12).
