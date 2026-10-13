#!/usr/bin/env python3
"""Total variation of the rotor speed, rotor by rotor, over the steady-state categories
(Step excluded): right panel of Fig. 7.11-7.12 of the thesis.

The categories and the sample-weighted mean are those of analyze_trajectory.metrics_for_category,
so the sum over the six rotors of one category equals the 'tv' row of comparison_categories.csv.
Here the steady-state categories are merged, each block weighted by its number of samples.

Usage, from the data_analysis folder::

  python export_rotor_tv.py p2p      -> p2p_comparison/tables/rotor_tv_regime.csv
  python export_rotor_tv.py rpy      -> rpy_comparison/tables/rotor_tv_regime.csv

It reads <mission>_nominal and <mission>_perturbed.
"""
import contextlib
import csv
import io
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_trajectory as at  # noqa: E402

CFG = {'t_ss': 2.0, 'tol_pos': 0.05, 'tol_att_deg': 5.0, 'settle_band': 0.05,
       'settle_min_ang_deg': 5.0, 'step_tol_pos': at.STEP_TOL_POS, 'step_tol_ang': at.STEP_TOL_ANG,
       'min_step_norm': at.MIN_STEP_NORM, 'v_thr': at.V_THR, 'w_thr': at.W_THR,
       'min_block': at.MIN_BLOCK, 'step_pos': at.STEP_POS, 'step_ang': np.deg2rad(2.0),
       'cat_delta': 2.5, 'cat_clip': False, 'cat_exclude': []}


def per_rotor(folder):
    """Total variation per rotor over the categories other than Step.

    Runs ``analyze_trajectory`` on the folder and collects, for each block of
    each category, the total variation rate of each rotor; the blocks are then
    averaged with their number of samples as weights.

    Args:
        folder (pathlib.Path): Mission folder.

    Returns:
        dict[str, numpy.ndarray]: Total variation rate of each rotor [rad/s^2], per group.
    """
    blocks = {}
    orig = at.metrics_for_category

    def patched(f, bl, cat):
        M = orig(f, bl, cat)
        if M is not None and cat != "Step" and f["forces"] is not None:
            for b in (b for b in bl if b[0] == cat):
                idx = np.flatnonzero(at.window(f, b[2], b[3]))
                if idx.size < 2:
                    continue
                w = np.sqrt(np.clip(f["forces"][idx], at.F_MIN, at.F_MAX) / at.CF)
                dur = float(f["t"][idx][-1] - f["t"][idx][0])
                blocks.setdefault(f["group"], []).append(
                    (np.sum(np.abs(np.diff(w, axis=0)), axis=0) / dur, idx.size))
        return M

    at.metrics_for_category = patched
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            at.export_analysis(folder, CFG)
    finally:
        at.metrics_for_category = orig
    return {g: np.average([p for p, _ in b], axis=0, weights=[n for _, n in b])
            for g, b in blocks.items()}


def main():
    """Write ``<mission>_comparison/tables/rotor_tv_regime.csv`` for the mission given on the command line."""
    m = sys.argv[1]
    nom, pert = per_rotor(HERE / f"{m}_nominal"), per_rotor(HERE / f"{m}_perturbed")
    out = HERE / f"{m}_comparison" / "tables" / "rotor_tv_regime.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rotor", "unit", "ppo_nom", "ppo_pert", "sppo_nom", "sppo_pert"])
        for i in range(at.N_ROTORS):
            w.writerow([i + 1, "rad/s^2", f"{nom['PPO'][i]:.4f}", f"{pert['PPO'][i]:.4f}",
                        f"{nom['SPPO'][i]:.4f}", f"{pert['SPPO'][i]:.4f}"])
    print("written", out)


if __name__ == "__main__":
    main()
