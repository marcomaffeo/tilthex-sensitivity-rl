#!/usr/bin/env python3
"""
Nominal vs perturbed comparison for one mission.

Two readings (who has the lower error, who degrades less) on two splits of
the flight: the mission phases and the control task categories.

The script takes the two folders of the same mission (e.g. p2p_nominal and
p2p_perturbed) and writes a comparison folder with tables and figures.

Where the numbers come from
---------------------------
The metrics are not recomputed. The script reads the CSV files written by
analyze_trajectory.py in <folder>/data_analysis/tables/::

    phase_metrics.csv     metrics per phase, per flight
    tv_windows.csv        total variation, per window and per motor
    category_metrics.csv  metrics per manoeuvre and per step
    params.csv            the parameters used for the folder

If the CSV files are missing the script stops and prints the command to run.
The overlay figures over time need the raw series and reload the data.npz
files, with the same error definitions.

The two questions
-----------------
With four flights (PPO nominal, PPO perturbed, SPPO nominal, SPPO perturbed)
two different questions can be asked, with different answers:

  1. WHO HAS THE LOWER ERROR UNDER THE DISTURBANCE:
     direct comparison SPPO vs PPO in the perturbed case.

  2. WHO DEGRADES LESS:
     D = perturbed / nominal, computed per algorithm and then compared.

A policy can have a lower error in both conditions and still degrade more
in relative terms, because it starts lower. The script always reports the
two readings side by side, in the tables and in the figures.

Why the ratio, and in this order:

  * ratio: the two algorithms start from different nominal errors, so a
    difference in metres is not comparable between them; the ratio is
    dimensionless;
  * perturbed / nominal: D > 1 reads as a degradation, and the nominal
    value, the cleaner of the two, is the denominator.

Limits
------
D is the ratio of two single flights: with n = 1 its uncertainty cannot be
estimated from these data. A fixed threshold is used instead (--noise-floor,
default 5%): below that difference the script writes '~' and assigns no
winner. The threshold is a convention, not a statistical estimate; the
consistency of the sign across phases and missions carries more weight than
a single number.

Usage::

    python compare_trajectory.py --nominal p2p_nominal --perturbed p2p_perturbed
    python compare_trajectory.py -n rpy_nominal -p rpy_perturbed \
                                 --noise-floor 5 --output rpy_comparison
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

RAD2DEG = 180.0 / math.pi

C_PPO, C_SPPO = '#ff7f0e', '#1f77b4'  # orange / blue
COL = {'PPO': C_PPO, 'SPPO': C_SPPO}
GROUPS = ('PPO', 'SPPO')
CONDS = ('nominal', 'perturbed')
HATCH = {'nominal': '', 'perturbed': '///'}
ALPHA = {'nominal': 0.90, 'perturbed': 0.45}

NAN = float('nan')

# (key in the csv, label, unit, scale factor, defined on every phase)
PHASE_METRICS = [
    ('rmse_pos_steady',         r'steady-state RMSE $\|e_p\|$',        'm',       1.0,     True),
    ('rmse_att_steady',         r'steady-state RMSE $\|e_R\|$',        'rad',     1.0,     True),
    ('peak_pos_norm',           r'peak $\|e_p\|$ / step',      '-',       1.0,     False),
    ('rmse_pos_transient_norm', r'transient RMSE / step',     '-',       1.0,     False),
    ('peak_att_norm',           r'peak $\|e_R\|$ / step',      'rad/m',   1.0,     False),
    ('tv_total',                r'phase total variation',        'Hz/s',      1.0 / (2 * math.pi),  True),
]

# short labels for the console (mathtext is not printed in a terminal)
SHORT = {
    'rmse_pos_steady': 'steady pos [m]',
    'rmse_att_steady': 'steady att [rad]',
    'peak_pos_norm': 'peak pos / step [-]',
    'rmse_pos_transient_norm': 'transient RMSE / step [-]',
    'peak_att_norm': 'peak att / step [rad/m]',
    'tv_total': 'phase TV [Hz/s]',
}


# =========================================================== reading the CSV files
def fnum(s):
    """Convert a CSV cell to a float; an empty or non-numeric cell gives ``nan``.

    Args:
        s (object): Cell value.

    Returns:
        float: Numeric value or ``nan``.
    """
    try:
        v = float(s)
    except (TypeError, ValueError):
        return NAN
    return v


def tables_dir(folder: Path) -> Path:
    """Folder of the CSV tables written by ``analyze_trajectory``.

    Args:
        folder (pathlib.Path): Mission folder analysed by ``analyze_trajectory``.

    Returns:
        pathlib.Path: ``<folder>/data_analysis/tables``.
    """
    return folder / "data_analysis" / "tables"


def read_phase_metrics(folder: Path):
    """Read ``phase_metrics.csv`` of a mission folder.

    Args:
        folder (pathlib.Path): Mission folder analysed by ``analyze_trajectory``.

    Returns:
        dict: ``{(group, phase): {field: value}}``.
    """
    path = tables_dir(folder) / "phase_metrics.csv"
    if not path.is_file():
        sys.exit(f"Missing {path}\n"
                 f"  Run first:  python analyze_trajectory.py --input {folder}")
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            key = (r['group'], r['phase'])
            out[key] = {k: (v if k in ('run', 'group', 'phase') else fnum(v))
                        for k, v in r.items()}
    if not out:
        sys.exit(f"{path} is empty.")
    return out


def read_tv_flight(folder: Path):
    """Read the total variation over the whole flight from ``tv_windows.csv``.

    The whole flight is the only window that does not depend on a cut, so it
    can be compared between two folders without checking where the cut fell.

    Args:
        folder (pathlib.Path): Mission folder analysed by ``analyze_trajectory``.

    Returns:
        dict[str, float]: Total variation per group; empty when the file is missing.
    """
    path = tables_dir(folder) / "tv_windows.csv"
    if not path.is_file():
        return {}
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get('window') == 'flight' and r.get('channel') == 'TOTAL':
                out[r['group']] = fnum(r['tv'])
    return out


# ================================================ raw series for the curves
# The tables all come from the CSV files. The curves over time need the
# raw series, and the two error definitions are repeated here:
#
#     |e_p| = || pos - pos_ref ||
#     |e_R| = || rotvec(R_ref^-1 R_act) ||          quaternions in WXYZ
#
# After loading, the script recomputes the RMSE of a phase already in
# phase_metrics.csv and compares it with the value written there. If the
# two differ, it says so and draws nothing.

def load_series(npz_path: Path):
    """Load one flight and recompute its error series.

    Args:
        npz_path (pathlib.Path): ``data.npz`` of the flight.

    Returns:
        tuple: ``(group, series)``, with the time, the error norms, the raw
        position and attitude and, when logged, the velocity norms; or
        ``(None, reason)`` when the file is discarded.
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:                                   # noqa: BLE001
        return None, f"load_error: {e}"
    keys = set(data.files)
    if not {'t', 'pos', 'pos_ref', 'quat', 'quat_ref'} <= keys:
        return None, "missing fields"

    pname = str(data['policy_name']) if 'policy_name' in keys else ""
    tag = (npz_path.parent.name + "_" + pname).upper()
    group = "SPPO" if "SPPO" in tag else ("PPO" if "PPO" in tag else None)
    if group is None:
        return None, "unknown group"

    def to_xyzw(q):
        q = np.asarray(q, float)
        return np.column_stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]])

    t = np.asarray(data['t'], float)
    t = t - t[0]
    pos = np.asarray(data['pos'], float)
    pos_ref = np.asarray(data['pos_ref'], float)
    e_w = pos - pos_ref
    R_act = Rotation.from_quat(to_xyzw(data['quat']))
    R_ref = Rotation.from_quat(to_xyzw(data['quat_ref']))
    rv = (R_ref.inv() * R_act).as_rotvec()
    out = {'t': t, 'e_pos_norm': np.linalg.norm(e_w, axis=1),
           'e_att_norm': np.linalg.norm(rv, axis=1),
           # Raw series, besides the norms: fig_step_profile_pair needs the
           # direction of the command for the step progress p(t)
           # (eq:step_progress).
           'pos': pos, 'pos_ref': pos_ref, 'R_act': R_act, 'R_ref': R_ref}
    # Linear and angular velocity, as in analyze_trajectory (process_log):
    # norm only, no reference is logged.
    if 'vel' in keys:
        out['v_norm'] = np.linalg.norm(np.asarray(data['vel'], float), axis=1)
    if 'avel' in keys:
        out['w_norm'] = np.linalg.norm(np.asarray(data['avel'], float), axis=1)
    return group, out


def check_against_csv(f, pm, group, tol=0.02):
    """Check that the series recomputed here give the RMSE written by ``analyze_trajectory``.

    The check is done on the first comparable phase of the group.

    Args:
        f (dict): Series from ``load_series``.
        pm (dict): Phase metrics from ``read_phase_metrics``.
        group (str): ``'PPO'`` or ``'SPPO'``.
        tol (float): Relative tolerance.

    Returns:
        tuple[bool, str]: Outcome and message.
    """
    for (g, ph), r in pm.items():
        if g != group or r.get('comparable_transient') != 1:
            continue
        exp = r.get('rmse_pos_all', NAN)
        t0, t1 = r.get('t0', NAN), r.get('t1', NAN)
        if exp != exp or t0 != t0 or t1 != t1 or exp == 0:
            continue
        sel = (f['t'] >= t0) & (f['t'] <= t1)
        if sel.sum() < 3:
            continue
        got = float(np.sqrt(np.mean(f['e_pos_norm'][sel] ** 2)))
        if abs(got - exp) / exp > tol:
            return False, (f"{group}: RMSE recomputed on '{ph}' = {got:.5f}, "
                           f"in the CSV = {exp:.5f}")
        return True, f"{group}: consistent on '{ph}' ({got:.5f} vs {exp:.5f})"
    return True, f"{group}: no phase suitable for the check"


# =========================================================== comparisons
def pct(new, old):
    """Percentage change of ``new`` with respect to ``old``.

    Args:
        new (float): New value.
        old (float): Reference value.

    Returns:
        float: ``(new - old) / old * 100``, ``nan`` when not defined.
    """
    if new != new or old != old or old == 0:
        return NAN
    return (new - old) / old * 100.0


def ratio(num, den):
    """Ratio of two values.

    Args:
        num (float): Numerator.
        den (float): Denominator.

    Returns:
        float: ``num / den``, ``nan`` when not defined.
    """
    if num != num or den != den or den == 0:
        return NAN
    return num / den


def winner(delta_pct, floor):
    """Policy with the lower value of a metric where lower is better.

    Below the threshold no winner is assigned.

    Args:
        delta_pct (float): ``(SPPO - PPO) / PPO * 100``.
        floor (float): Relative difference [%] below which no winner is assigned.

    Returns:
        str: ``'PPO'``, ``'SPPO'``, ``'~'`` below the threshold, ``''`` when not defined.
    """
    if delta_pct != delta_pct:
        return ''
    if abs(delta_pct) < floor:
        return '~'
    return 'SPPO' if delta_pct < 0 else 'PPO'


def compare_cell(nom, pert, floor):
    """All the derived numbers for one metric and one phase.

    Args:
        nom (dict[str, float]): Nominal value per group.
        pert (dict[str, float]): Perturbed value per group.
        floor (float): Relative difference [%] below which no winner is assigned.

    Returns:
        dict: Values, absolute and relative differences, degradation ratios
        and winners.
    """
    c = {}
    for g in GROUPS:
        c[f'{g.lower()}_nom'] = nom.get(g, NAN)
        c[f'{g.lower()}_pert'] = pert.get(g, NAN)
        c[f'D_{g.lower()}'] = ratio(pert.get(g, NAN), nom.get(g, NAN))
    # reading 1: who has the lower error, condition by condition
    c['delta_nom_pct'] = pct(c['sppo_nom'], c['ppo_nom'])
    c['delta_pert_pct'] = pct(c['sppo_pert'], c['ppo_pert'])
    # reading 2: who has the lower D. "Degrades less" holds only when both
    # D are above 1. When they are below, the perturbed flight did better
    # than the nominal one and the lower D means "improved more", which with
    # an added load is flight-to-flight variability.
    c['delta_D_pct'] = pct(c['D_sppo'], c['D_ppo'])
    # reading 3: DELTA, the absolute shift caused by the disturbance.
    #
    # The disturbance is the SAME for the two policies, so the more robust
    # policy is the one whose error shifts less: a difference, not a ratio.
    # D divides by the nominal value, so the policy with the lower nominal
    # error has a smaller denominator and is penalised by the arithmetic.
    # Delta has no such bias and is the quantity to report for the
    # sensitivity.
    #
    # A NEGATIVE Delta with an added load is not physical: there the
    # flight-to-flight variability is measured, and its absolute value
    # gives the noise scale below which no difference between the two
    # policies can be read.
    for g in GROUPS:
        c[f'delta_abs_{g.lower()}'] = (c[f'{g.lower()}_pert']
                                       - c[f'{g.lower()}_nom'])
    c['win_nom'] = winner(c['delta_nom_pct'], floor)
    c['win_pert'] = winner(c['delta_pert_pct'], floor)
    c['win_D'] = winner(c['delta_D_pct'], floor)
    # who shifts less: on the absolute value of the shift, so that a
    # negative Delta (noise) is not rewarded as the smallest shift
    da, ds = c['delta_abs_ppo'], c['delta_abs_sppo']
    c['win_delta'] = (None if da != da or ds != ds or abs(da) < 1e-12
                      else winner(pct(abs(ds), abs(da)), floor))
    return c


# =========================================================== figures
def set_style():
    """Set the matplotlib style of the figures (serif fonts, light grid, minimal frame)."""
    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.serif": ["cmr10", "STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "axes.formatter.use_mathtext": True,
        "font.size": 15,
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "axes.titlepad": 12,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#cccccc",
        "figure.titlesize": 17,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.4,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "savefig.bbox": "tight",
        "axes.titleweight": "normal",
    })


def save(fig, path):
    """Save a figure as a transparent PDF and close it.

    Args:
        fig (matplotlib.figure.Figure): Figure.
        path (pathlib.Path): Output file.
    """
    fig.savefig(path, format='pdf', transparent=True, bbox_inches='tight')
    plt.close(fig)


# Position and attitude, one panel each: the same two quantities as in
# the mid-point and range plot of fig_cat_range.
OVERLAY_PANELS = [
    ('e_pos_norm', r'$\|e_p\|$   [m]'),
    ('e_att_norm', r'$\|e_R\|$   [rad]'),
]


def fig_overlay_grid(flights, out_path, bounds=(), smooth_s=0.5):
    """1x2 grid: position on the left, attitude on the right, nominal against
    perturbed, one figure and one legend. Same layout as the rollout grids of
    analyze_trajectory (larger fonts, 1.8 lines, no title), so that the print
    scale is the same when the figure spans the page width.

    Dashes are not used: in the other figures of the chapter (full_pose,
    TRACK) a dashed line is the commanded reference. Nominal and perturbed
    are both solid and differ in transparency: nominal faded, perturbed full.
    Each flight is drawn as its moving average only.

    Args:
        flights (dict): Series from ``load_series``, keyed by ``(group, condition)``.
        out_path (pathlib.Path): Output file.
        bounds (list[float]): Phase boundaries [s], drawn as vertical lines.
        smooth_s (float): Width of the moving average [s].
    """
    with plt.rc_context({
        "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
        "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9,
        "xtick.major.size": 4.5, "ytick.major.size": 4.5, "grid.linewidth": 0.7,
    }):
        fig_w, fig_h = 11, 4.6
        fig, axs = plt.subplots(1, 2, figsize=(fig_w, fig_h), squeeze=False, sharex=True)
        t_lo, t_hi = np.inf, -np.inf
        handles_by_label = {}
        for i, (key, ylabel) in enumerate(OVERLAY_PANELS):
            ax = axs[0][i]
            for (g, cond), f in sorted(flights.items()):
                if key not in f:
                    continue
                t, v = f['t'], np.abs(f[key])
                if t.size < 3:
                    continue
                t_lo, t_hi = min(t_lo, float(t[0])), max(t_hi, float(t[-1]))
                dt = float(np.median(np.diff(t))) if t.size > 2 else 0.01
                k = max(1, int(round(smooth_s / max(dt, 1e-6))))
                if k > 1:
                    vs = np.convolve(v, np.ones(k) / k, mode='same')
                    vs[:k], vs[-k:] = v[:k], v[-k:]
                else:
                    vs = v
                alpha = 0.45 if cond == 'nominal' else 0.95
                label = f'{g} {cond}'
                line, = ax.plot(t, vs, color=COL[g], lw=1.8, alpha=alpha, label=label)
                handles_by_label[label] = line
            for b in bounds:
                ax.axvline(b, color='#999', lw=0.9, ls=':')
            ax.set_ylabel(ylabel)
            ax.set_ylim(bottom=0)
            ax.set_xlabel(r'$t$   [s]')
        for ax in axs.ravel():
            tight_x(ax, t_lo, t_hi)
        # One legend inside the position panel (upper right, where the curves
        # stay low after the second peak), valid for both panels.
        fig.tight_layout()
        order = sorted(handles_by_label)
        axs[0][0].legend([handles_by_label[l] for l in order], order,
                         loc='upper right', ncol=1, frameon=True, fontsize=14)
        save(fig, out_path)


# Phase of the step C->D (Point-to-point in p2p, Tilt in the tilt
# mission): the SECOND hold phase of the mission, 'SETPOINT 2' in the
# phase_metrics.csv of analyze_trajectory for both missions, so the
# function below treats p2p and tilt in the same way.
STEP_PROFILE_PHASE = 'SETPOINT 2'

# Window of the profile, different from cat_delta (which separates Step
# from Hold for the category metrics and is usually shorter than the
# transient). SETPOINT 2 lasts about 5-6 s in both missions
# (phase_metrics.csv, t1-t0): 5 s shows almost the whole transient
# without reaching the next phase. The value is fixed, as in
# subsec:window of the thesis.
STEP_PROFILE_WINDOW_S = 5.0

STEP_PROFILE_PANELS = [('pos', r'$p(t)$   [-]'), ('att', r'$p(t)$   [-]')]
STEP_PROFILE_CHANNELS = {'pos': ('pos', 'pos_ref'), 'att': ('R_act', 'R_ref')}
STEP_MIN_AMP = {'pos': 0.02, 'att': float(np.deg2rad(2.0))}   # STEP_POS/STEP_ANG

# p(t) channel of each mission: the one with a nonzero step in C->D
# (the other keeps a constant reference during the transition, and
# _step_progress discards it through STEP_MIN_AMP). Keys = name of the
# nominal folder without '_nominal' (see run()): 'p2p' for the
# point-to-point, 'rpy' for the tilt mission.
MISSION_STEP_CHANNEL = {'p2p': 'pos', 'rpy': 'att'}


def _step_progress(t, sig, sig_ref, t0, delta, kind):
    """Progress variable ``p(t)`` of a step, on the window ``[t0, t0 + delta]``.

    ``p`` is 0 at the instant of the step (one sample before the phase
    boundary, as in ``step_response`` of ``analyze_trajectory``, so that the
    first sample is zero), 1 on the target and above 1 when the vehicle goes
    past it. The formula is the same as ``step_response``; the amplitude is
    checked against ``step_pos_m`` and ``step_ang_deg`` of ``phase_metrics.csv``.

    Args:
        t (numpy.ndarray): Time stamps [s].
        sig (numpy.ndarray | scipy.spatial.transform.Rotation): Position, shape
            ``(N, 3)``, for ``kind='pos'``; attitude stack for ``kind='att'``.
        sig_ref (numpy.ndarray | scipy.spatial.transform.Rotation): Reference, same type as ``sig``.
        t0 (float): Start of the phase [s].
        delta (float): Duration of the window after the step [s].
        kind (str): ``'pos'`` or ``'att'``.

    Returns:
        tuple | None: ``(t_rel, p, amp)``, or ``None`` when the step on this
        channel is below the threshold or the window has too few samples.
    """
    # tolerance: t0 comes from phase_metrics.csv rounded to 6 decimals, up
    # to ~1e-6 s after the actual sample of the jump. Without a margin,
    # 't >= t0' could select the sample AFTER the right one, making i_prev
    # already post-jump and the amplitude zero. 1e-4 s is well above the
    # rounding error and well below the sampling period (~0.01 s).
    idx = np.flatnonzero(t >= t0 - 1e-4)
    if idx.size == 0 or idx[0] == 0:
        return None
    i0 = int(idx[0])
    i_prev = i0 - 1
    tail = np.flatnonzero(t > t0 + delta)
    i1 = int(tail[0]) if tail.size else t.size
    if i1 - i_prev < 3:
        return None
    sl = slice(i_prev, i1)
    tt = t[sl] - t[i_prev]
    if kind == 'pos':
        d = sig_ref[i0] - sig_ref[i_prev]
        amp = float(np.linalg.norm(d))
        if not (amp > STEP_MIN_AMP['pos']):
            return None
        e = sig[sl] - sig_ref[i0]
        sgn = e @ (d / amp)
    else:
        dr = (sig_ref[i_prev].inv() * sig_ref[i0]).as_rotvec()
        amp = float(np.linalg.norm(dr))
        if not (amp > STEP_MIN_AMP['att']):
            return None
        R_t = sig_ref[i0]
        rv = (R_t.inv() * sig[sl]).as_rotvec()
        sgn = rv @ (dr / amp)
    prog = (sgn + amp) / amp
    return tt, prog, amp


def _step_progress_band(flights, pm_nom, pm_pert, key, delta,
                        smooth_s=0.10, n_grid=251, amp_tol=0.05, tag=''):
    """Mid-point and range band of ``p(t)`` for one channel, one per group.

    The band covers ``[0, delta]`` after the step of ``STEP_PROFILE_PHASE``.
    It is shared by ``fig_step_profile_pair`` and ``fig_step_profile_single``;
    see ``_step_progress`` for the formula.

    Args:
        flights (dict): Series from ``load_series``, keyed by ``(group, condition)``.
        pm_nom (dict): Phase metrics of the nominal folder, from ``read_phase_metrics``.
        pm_pert (dict): Phase metrics of the perturbed folder, from ``read_phase_metrics``.
        key (str): Channel, ``'pos'`` or ``'att'``.
        delta (float): Duration of the window after the step [s].
        smooth_s (float): Width of the moving average [s].
        n_grid (int): Number of points of the common time grid.
        amp_tol (float): Relative tolerance between the step amplitude recomputed here and the one in the CSV.
        tag (str): Prefix of the console warnings, to tell the callers apart.

    Returns:
        tuple: ``(t_grid, bands)``, with ``bands[g] = (mid, lo, hi)`` on
        ``t_grid``. ``bands[g]`` is ``None`` when a flight of that group is
        missing or the channel has no step above the threshold in one of the two
        conditions.
    """
    sig_key, ref_key = STEP_PROFILE_CHANNELS[key]
    pm_by_cond = {'nominal': pm_nom, 'perturbed': pm_pert}
    t_grid = np.linspace(0.0, delta, n_grid)
    bands = {}
    for g in GROUPS:
        curves = []
        ok = True
        for cond in CONDS:
            pm = pm_by_cond[cond]
            row = pm.get((g, STEP_PROFILE_PHASE))
            f = flights.get((g, cond))
            if row is None or f is None or sig_key not in f or ref_key not in f:
                ok = False
                break
            t0 = row.get('t0', NAN)
            if t0 != t0:
                ok = False
                break
            res = _step_progress(f['t'], f[sig_key], f[ref_key], t0, delta, key)
            if res is None:
                ok = False
                break
            tt, prog, amp = res
            csv_amp = row.get('step_pos_m' if key == 'pos' else 'step_ang_deg', NAN)
            if csv_amp == csv_amp:
                csv_amp_v = csv_amp if key == 'pos' else np.deg2rad(csv_amp)
                if csv_amp_v > 0 and abs(amp - csv_amp_v) / csv_amp_v > amp_tol:
                    print(f"  [step p(t){tag}] !! {g} {cond} {key}: amplitude "
                          f"recomputed {amp:.4f} vs CSV {csv_amp_v:.4f}")
            vg = np.interp(t_grid, tt, prog)
            dt = float(np.median(np.diff(t_grid)))
            k = max(1, int(round(smooth_s / max(dt, 1e-6))))
            if k > 1:
                vs = np.convolve(vg, np.ones(k) / k, mode='same')
                vs[:k], vs[-k:] = vg[:k], vg[-k:]
            else:
                vs = vg
            curves.append(vs)
        if ok and len(curves) == 2:
            lo = np.minimum(curves[0], curves[1])
            hi = np.maximum(curves[0], curves[1])
            mid = (curves[0] + curves[1]) / 2.0
            bands[g] = (mid, lo, hi)
        else:
            bands[g] = None
    return t_grid, bands


def fig_step_profile_pair(flights, pm_nom, pm_pert, delta, out_path,
                          smooth_s=0.10, n_grid=251, amp_tol=0.05):
    """Step progress of position and attitude, in a 1x2 grid.

    ``p(t)`` is drawn on the window ``[t0, t0 + delta]`` of
    ``STEP_PROFILE_PHASE``, with one mid-point and range band per policy (see
    ``_step_progress_band``). The convention is the one of
    ``fig_cat_range_pair``, over time instead of by category: the centre of
    each band is the mid-point between the nominal and the perturbed flight,
    the width is the interval they span (two values are not a sample, so no
    standard deviation is used). ``t0`` comes from ``phase_metrics.csv``.

    Each mission has a step on a single channel in C->D (see
    ``MISSION_STEP_CHANNEL``), so one of the two panels is empty; the figure
    used in the text is ``fig_step_profile_single``.

    Args:
        flights (dict): Series from ``load_series``, keyed by ``(group, condition)``.
        pm_nom (dict): Phase metrics of the nominal folder, from ``read_phase_metrics``.
        pm_pert (dict): Phase metrics of the perturbed folder, from ``read_phase_metrics``.
        delta (float): Duration of the window after the step [s] (``STEP_PROFILE_WINDOW_S``).
        out_path (pathlib.Path): Output file.
        smooth_s (float): Width of the moving average [s].
        n_grid (int): Number of points of the common time grid.
        amp_tol (float): Relative tolerance between the step amplitude recomputed here and the one in the CSV.

    Returns:
        bool: ``True`` when the figure is written.
    """
    panels = {}
    missing = []
    t_grid = None
    for key, _ in STEP_PROFILE_PANELS:
        t_grid, bands = _step_progress_band(flights, pm_nom, pm_pert, key, delta,
                                            smooth_s, n_grid, amp_tol,
                                            tag=' pair')
        panels[key] = bands
        missing += [f"{g}/{key}" for g, b in bands.items() if b is None]

    if all(b is None for bands in panels.values() for b in bands.values()):
        print(f"  [step profile] no data, {out_path} not written")
        return False
    if missing:
        print(f"  [step profile] band missing for: {', '.join(missing)}")

    with plt.rc_context({
        "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
        "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9,
        "xtick.major.size": 4.5, "ytick.major.size": 4.5, "grid.linewidth": 0.7,
    }):
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), squeeze=False)
        titles = {'pos': 'Position', 'att': 'Attitude'}
        for i, (key, ylabel) in enumerate(STEP_PROFILE_PANELS):
            ax = axs[0][i]
            ax.axhline(1.0, color='#999999', lw=1.0, ls=':')
            for g in GROUPS:
                b = panels[key].get(g)
                if b is None:
                    continue
                mid, lo, hi = b
                ax.fill_between(t_grid, lo, hi, color=COL[g], alpha=0.25, lw=0)
                ax.plot(t_grid, mid, color=COL[g], lw=1.8, label=g)
            ax.set_xlabel(r'$t$   [s]')
            ax.set_ylabel(ylabel)
            ax.set_title(titles[key], fontsize=15)
            tight_x(ax, 0.0, delta)
        fig.tight_layout()
        axs[0][0].legend(loc='lower right', frameon=True, fontsize=14)
        save(fig, out_path)
    return True


def fig_step_profile_single(flights, pm_nom, pm_pert, mission, delta, out_path,
                            smooth_s=0.10, n_grid=251, amp_tol=0.05):
    """Step progress on the channel with a step, one panel per mission.

    ``p(t)`` is drawn on the window ``[t0, t0 + delta]`` of
    ``STEP_PROFILE_PHASE``, with one mid-point and range band per policy, only
    on the channel with a step in C->D for this mission (position for p2p,
    attitude for the tilt mission, see ``MISSION_STEP_CHANNEL``).

    Args:
        flights (dict): Series from ``load_series``, keyed by ``(group, condition)``.
        pm_nom (dict): Phase metrics of the nominal folder, from ``read_phase_metrics``.
        pm_pert (dict): Phase metrics of the perturbed folder, from ``read_phase_metrics``.
        mission (str): Key of ``MISSION_STEP_CHANNEL``, ``'p2p'`` or ``'rpy'``.
        delta (float): Duration of the window after the step [s].
        out_path (pathlib.Path): Output file.
        smooth_s (float): Width of the moving average [s].
        n_grid (int): Number of points of the common time grid.
        amp_tol (float): Relative tolerance between the step amplitude recomputed here and the one in the CSV.

    Returns:
        bool: ``True`` when the figure is written.
    """
    key = MISSION_STEP_CHANNEL[mission]
    ylabel = dict(STEP_PROFILE_PANELS)[key]
    t_grid, bands = _step_progress_band(flights, pm_nom, pm_pert, key, delta,
                                        smooth_s, n_grid, amp_tol,
                                        tag=' single')

    if all(b is None for b in bands.values()):
        print(f"  [step profile single] no data for "
              f"'{mission}', {out_path} not written")
        return False

    with plt.rc_context({
        "axes.labelsize": 18, "xtick.labelsize": 15, "ytick.labelsize": 15,
        "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9,
        "xtick.major.size": 4.5, "ytick.major.size": 4.5, "grid.linewidth": 0.7,
    }):
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        ax.axhline(1.0, color='#999999', lw=1.0, ls=':')
        for g in GROUPS:
            b = bands.get(g)
            if b is None:
                continue
            mid, lo, hi = b
            ax.fill_between(t_grid, lo, hi, color=COL[g], alpha=0.25, lw=0)
            ax.plot(t_grid, mid, color=COL[g], lw=1.8, label=g)
        ax.set_xlabel(r' $t$   [s]')
        ax.set_ylabel(ylabel)
        tight_x(ax, 0.0, delta)
        fig.tight_layout()
        ax.legend(loc='lower right', frameon=True, fontsize=13)
        save(fig, out_path)
    return True


# ================================================== step response table,
# mid-point and range
#
# The four quantities (overshoot, rise time, settling time, steady-state
# error) are read from assets.csv (tables/ of each folder), written by
# step_response()/export_assets() of analyze_trajectory.py. The
# steady-state error of the attitude is in DEGREES, as in
# _step_table_figure() of analyze_trajectory.py.
STEP_METRICS_SECTION = {'pos': 'step_pos', 'att': 'step_att'}
STEP_METRICS = [
    ('overshoot_pct', r'overshoot $M_p$', '%', 1.0),
    ('rise_10_90_s', r'rise time $t_r$', 's', 1.0),
    ('settle_s', r'settling time $t_s$', 's', 1.0),
    ('final_err_abs', r'steady-state error $e_{ss}$', None, None),
]


def read_assets(folder: Path):
    """Read ``assets.csv`` of a mission folder.

    Args:
        folder (pathlib.Path): Mission folder analysed by ``analyze_trajectory``.

    Returns:
        dict: ``{(section, phase, group, metric): value}``.
    """
    path = tables_dir(folder) / "assets.csv"
    if not path.is_file():
        sys.exit(f"Missing {path}\n"
                 f"  Run first:  python analyze_trajectory.py --input {folder}")
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out[(r['section'], r['phase'], r['group'], r['metric'])] = fnum(r['value'])
    return out


def step_metrics_range_table(nom_dir, pert_dir, mission, out_path):
    """Write the LaTeX table of the step metrics, nominal against perturbed.

    The table holds the four quantities of ``STEP_METRICS`` on
    ``STEP_PROFILE_PHASE``, as mid-point and range between nominal and
    perturbed, PPO and SPPO side by side: the counterpart of
    ``fig_cat_range_pair`` for the step.

    Args:
        nom_dir (pathlib.Path): Nominal mission folder.
        pert_dir (pathlib.Path): Perturbed mission folder.
        mission (str): Key of ``MISSION_STEP_CHANNEL``; it selects the
            ``step_pos`` or ``step_att`` rows of ``assets.csv``.
        out_path (pathlib.Path): Output ``.tex`` file.

    Returns:
        bool: ``True`` when the table is written.
    """
    key = MISSION_STEP_CHANNEL[mission]
    section = STEP_METRICS_SECTION[key]
    a_nom, a_pert = read_assets(nom_dir), read_assets(pert_dir)
    deg = key == 'att'

    rows = []
    for metric, label, unit, scale in STEP_METRICS:
        if unit is None:                      # steady-state error: m or deg
            unit = 'deg' if deg else 'm'
            scale = RAD2DEG if deg else 1.0
        cells = {}
        ok_any = False
        for g in GROUPS:
            vn = a_nom.get((section, STEP_PROFILE_PHASE, g, metric), NAN)
            vp = a_pert.get((section, STEP_PROFILE_PHASE, g, metric), NAN)
            if vn == vn and vp == vp:
                vn, vp = vn * scale, vp * scale
                mid = (vn + vp) / 2.0
                lo, hi = min(vn, vp), max(vn, vp)
                cells[g] = (mid, lo, hi)
                ok_any = True
            else:
                cells[g] = None
        if ok_any:
            rows.append((label, unit, cells))

    if not rows:
        print(f"  [step table] no data for '{mission}', {out_path} not written")
        return False

    def fmt(cell):
        if cell is None:
            return '--'
        mid, lo, hi = cell
        return f"{mid:.3f}  [{lo:.3f}, {hi:.3f}]"

    with open(out_path, "w") as fh:
        fh.write(f"% {mission}: {section}, {STEP_PROFILE_PHASE}, "
                 "mid-point and range (nominal, perturbed)\n")
        fh.write("\\begin{tabular}{lccc}\n\\toprule\n")
        fh.write("Quantity & Unit & PPO & SPPO \\\\\n\\midrule\n")
        for label, unit, cells in rows:
            fh.write(f"{label} & {unit} & {fmt(cells['PPO'])} & "
                     f"{fmt(cells['SPPO'])} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    print(f"  written {out_path}")
    return True



def tight_x(ax, lo, hi):
    """No horizontal margin: the trace starts at the axis.

    Args:
        ax (matplotlib.axes.Axes): Axis.
        lo (float): Left limit.
        hi (float): Right limit.
    """
    if hi > lo:
        ax.set_xlim(lo, hi)
    ax.margins(x=0)


# =========================================================== main

# =========================================================== MISSION ONLY
# Until the vehicle reaches the world frame, the two flights do not
# share the same reference: the takeoff starts above the ground pose,
# which changes in every flight, so the approach step has a different
# amplitude (up to 11% between flights). What is measured there depends
# on that difference and not on the disturbance.
#
# From the arrival at (0,0,z) on, the references of the four flights are
# identical, instant by instant: in this window D measures only the
# effect of the perturbation.
#
# The cut is anchored to the REFERENCE (instant of the step plus a fixed
# settling margin), not to the actual arrival of the vehicle: aligning on
# the arrival would condition on the outcome and remove from the slower
# flight the part where it does worse.

N_ROTORS, F_MIN, F_MAX, CF = 6, 0.3072, 12.0, 12.0e-4

# Same names and order as in analyze_trajectory.
# numpy >= 2.0 calls it trapezoid, older versions trapz
_TRAPZ = getattr(np, 'trapezoid', None) or np.trapz


def seg_mean_t(t, y, mask=None):
    """Time average of y: trapezoidal integral divided by the duration, summed
    over contiguous stretches. Same as analyze_trajectory.seg_mean_t, so that
    the two scripts give the same number on the same window.

    Args:
        t (numpy.ndarray): Time stamps [s].
        y (numpy.ndarray): Signal.
        mask (numpy.ndarray | None): Boolean mask of the samples; ``None`` uses all of them.

    Returns:
        float: Time average of ``y``, ``nan`` for an empty mask.
    """
    t = np.asarray(t, float); y = np.asarray(y, float)
    idx = np.arange(t.size) if mask is None else np.flatnonzero(mask)
    if idx.size == 0:
        return NAN
    num = den = 0.0
    for blk in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        if blk.size < 2:
            continue
        num += float(_TRAPZ(y[blk], t[blk]))
        den += float(t[blk[-1]] - t[blk[0]])
    return num / den if den > 0 else float(np.mean(y[idx]))


def rms_t(t, x, mask=None):
    """RMS over time, ``sqrt((1/T) * integral of x^2 dt)``.

    Args:
        t (numpy.ndarray): Time stamps [s].
        x (numpy.ndarray): Signal.
        mask (numpy.ndarray | None): Boolean mask of the samples; ``None`` uses all of them.

    Returns:
        float: RMS of ``x`` over time.
    """
    v = seg_mean_t(t, np.asarray(x, float) ** 2, mask)
    return float(np.sqrt(v)) if v == v and v >= 0 else NAN


def mae_t(t, x, mask=None):
    """Mean absolute value over time, ``(1/T) * integral of |x| dt``.

    Args:
        t (numpy.ndarray): Time stamps [s].
        x (numpy.ndarray): Signal.
        mask (numpy.ndarray | None): Boolean mask of the samples; ``None`` uses all of them.

    Returns:
        float: Mean absolute value of ``x`` over time.
    """
    return seg_mean_t(t, np.abs(np.asarray(x, float)), mask)


# The two error statistics, as in analyze_trajectory: TIME averages.
# With --full-flight summary_rmse.csv is read, which has the RMSE only.
STATS = (("RMSE", rms_t),
         ("MAE",  mae_t))

def read_params(folder):
    """Read ``params.csv``, the parameters used for the analysis of a folder.

    Args:
        folder (pathlib.Path): Mission folder analysed by ``analyze_trajectory``.

    Returns:
        dict[str, float] | None: Parameter values, ``None`` when the file is missing.
    """
    p = tables_dir(folder) / "params.csv"
    if not p.exists():
        return None
    out = {}
    with open(p, newline="") as fh:
        for r in csv.DictReader(fh):
            out[r['param']] = fnum(r['value'])
    return out


# Parameters that define the WINDOWS: when they differ between nominal
# and perturbed, the two folders cut the flight at different points and
# every comparison also measures the difference of the cuts.
WINDOW_PARAMS = ('cat_delta', 't_ss', 'step_tol_pos', 'step_tol_ang')


def check_params(nom_dir, pert_dir):
    """Were the two folders analysed with the same cuts?

    The figures of maneuvers/ and step/ read category_metrics.csv as it is:
    the windows are inherited, not recomputed. With different --delta values
    'Step' and 'Hover' would cover different intervals in the two folders.

    Args:
        nom_dir (pathlib.Path): Nominal mission folder.
        pert_dir (pathlib.Path): Perturbed mission folder.

    Returns:
        bool: ``True`` when the parameters that define the windows are the same.
    """
    a, b = read_params(nom_dir), read_params(pert_dir)
    print(f"\n{'='*78}\nPARAMETERS OF THE TWO FOLDERS\n{'='*78}")
    if a is None or b is None:
        miss = [str(d) for d, v in ((nom_dir, a), (pert_dir, b)) if v is None]
        print("  !! tables/params.csv missing in: " + ", ".join(miss))
        print("     Run analyze_trajectory again on those folders: without that")
        print("     file the cuts cannot be checked.")
        return False
    ok = True
    print(f"  {'parameter':20s}{'nominal':>14s}{'perturbed':>14s}   result")
    for k in WINDOW_PARAMS:
        va, vb = a.get(k, NAN), b.get(k, NAN)
        same = (va != va and vb != vb) or abs(va - vb) < 1e-9
        ok &= same
        def s(v):
            return '--' if v != v else f"{v:.4g}"
        print(f"  {k:20s}{s(va):>14s}{s(vb):>14s}   "
              + ("ok" if same else "!! DIFFERENT"))
    if not ok:
        print("\n  !! the two folders cut the flight at different points.")
        print("     The figures in maneuvers/ and maneuvers/step/ use those")
        print("     windows as they are: regenerate both folders with the")
        print("     same parameters before reading the comparison.")
    if ok:
        print("\n  -> same cuts in the two folders.")
    return ok


# Unit of each global metric, for the figures of the RMSE folder. The
# attitude stays in radians, the unit of e_R (norm of the log map). The
# values are around 0.02-0.1, so the bar labels need enough digits.
GLOBAL_UNITS = {
    'Position Error X (ref frame)': ('m', 1.0),
    'Position Error Y (ref frame)': ('m', 1.0),
    'Position Error Z (ref frame)': ('m', 1.0),
    'Position Error X (body)': ('m', 1.0),
    'Position Error Y (body)': ('m', 1.0),
    'Position Error Z (body)': ('m', 1.0),
    'Attitude Error Roll': ('rad', 1.0),
    'Attitude Error Pitch': ('rad', 1.0),
    'Attitude Error Yaw': ('rad', 1.0),
    'Total Position Error': ('m', 1.0),
    'Total Attitude Error': ('rad', 1.0),
    'Linear Velocity Norm': ('m/s', 1.0),
    'Angular Velocity Norm': ('rad/s', 1.0),
    'Total variation - all motors': ('Hz/s', 1.0 / (2 * math.pi)),
}


def _slug(name):
    """Turn a metric name into a file name (lower case, underscores).

    Args:
        name (str): Metric name.

    Returns:
        str: File name.
    """
    keep = "".join(ch if ch.isalnum() else "_" for ch in name.lower())
    while "__" in keep:
        keep = keep.replace("__", "_")
    return keep.strip("_")


def fig_metric_bars(metric, c, out_path, floor, stat='RMSE'):
    """One metric, four bars: the two policies in the two conditions, without
    ratios or normalisation.

    Args:
        metric (str): Name of the metric.
        c (dict): Derived numbers from ``compare_cell``.
        out_path (pathlib.Path): Output file.
        floor (float): Relative difference [%] below which no winner is assigned.
        stat (str): Statistic, ``'RMSE'`` or ``'MAE'``.

    Returns:
        bool: ``True`` when the figure is written.
    """
    unit, scale = GLOBAL_UNITS.get(metric, ('', 1.0))
    order = [('ppo_nom', 'PPO', 'nominal'), ('ppo_pert', 'PPO', 'perturbed'),
             ('sppo_nom', 'SPPO', 'nominal'), ('sppo_pert', 'SPPO', 'perturbed')]
    vals = [c.get(k, NAN) * scale for k, _g, _cd in order]
    if not any(v == v for v in vals):
        return False
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    x = np.arange(4)
    for xi, (k, g, cond) in zip(x, order):
        v = c.get(k, NAN) * scale
        ax.bar(xi, v, 0.62, color=COL[g],
               alpha=(0.92 if cond == 'nominal' else 0.45),
               hatch=('' if cond == 'nominal' else '///'),
               edgecolor='white', linewidth=1.4, zorder=2)
        if v == v:
            ax.text(xi, v, f"{v:.4g}", ha='center', va='bottom',
                    fontsize=11, color='#222222', zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([g + "\n" + cond for _k, g, cond in order])
    ax.set_ylabel(stat + "   [" + unit + "]" if unit else stat)
    ax.set_title(metric)
    ax.set_ylim(0, max(v for v in vals if v == v) * 1.42)
    ax.grid(axis='x', visible=False)

    fig.tight_layout()
    save(fig, out_path)
    return True




# ===========================================================================
# CONTROL TASK CATEGORIES
# ===========================================================================
# The same two readings as above (who has the lower error, who degrades
# less) on the categories instead of the phases, so that the rows are
# comparable across missions.

CATEGORIES = ["Takeoff", "Step", "Hover", "Tilt", "Land"]
# same meaning as in analyze_trajectory: tells which categories are
# normalised in the figures instead of shown in absolute value.
KIND = {"Takeoff": "ramp", "Land": "ramp", "Step": "step",
        "Hover": "hold", "Tilt": "hold"}

# Same mission-specific names as in analyze_trajectory.py (Tables
# 5.5/5.6 of the thesis): 'stem' ('p2p'/'rpy') selects the list of
# categories used to read category_metrics.csv.
CATEGORIES_BY_MISSION = {
    "p2p": ["Takeoff", "Step", "Origin", "Point-to-point", "Land"],
    "rpy": ["Takeoff", "Step", "Origin", "Tilt", "Level", "Land"],
}
for _mission_cats in CATEGORIES_BY_MISSION.values():
    for _cat in _mission_cats:
        KIND.setdefault(_cat, "hold")
del _mission_cats, _cat
CAT_HATCH = {"nom": "", "pert": "///"}
CAT_ALPHA = {"nom": 0.92, "pert": 0.45}
COND_LAB = {"nom": "nominal", "pert": "perturbed"}
C_INK, C_MUTED = "#222222", "#9a9a9a"

# SIGNED metrics: the ratio D between values of different sign has no
# meaning, and the percentage change explodes when the denominator is
# close to zero. For these the four values and the ABSOLUTE differences
# are reported, in the unit of the metric.
CAT_SIGNED = {"lag_z"}


def read_category_metrics(folder):
    """Read ``category_metrics.csv`` of a mission folder.

    Args:
        folder (pathlib.Path): Mission folder analysed by ``analyze_trajectory``.

    Returns:
        dict: Values keyed by ``(group, category, statistic, metric)``, and the
        units under ``'_units'``.
    """
    p = tables_dir(Path(folder)) / "category_metrics.csv"
    if not p.is_file():
        raise SystemExit(f"Missing {p}\n"
                         f"  Run first:  python analyze_trajectory.py --input {folder}")
    out = {}
    for r in csv.DictReader(open(p, newline="")):
        try:
            v = float(r["value"])
        except (TypeError, ValueError):
            continue
        out[(r["group"], r["category"], r.get("stat", "-"), r["metric"])] = v
        out.setdefault("_units", {})[r["metric"]] = r["unit"]
    return out




# The ten quantities, same keys and file names as in analyze_trajectory
CAT_FIGS = [
    ("e_x",     r"$e_x^{\,r}$",   "m",     "1_err_bx"),
    ("e_y",     r"$e_y^{\,r}$",   "m",     "2_err_by"),
    ("e_z",     r"$e_z^{\,r}$",   "m",     "3_err_bz"),
    ("e_roll",  r"$e_{roll}$",    "rad",   "4_err_roll"),
    ("e_pitch", r"$e_{pitch}$",   "rad",   "5_err_pitch"),
    ("e_yaw",   r"$e_{yaw}$",     "rad",   "6_err_yaw"),
    ("e_p",     r"$\|e_p\|$",     "m",     "7_err_pos"),
    ("e_R",     r"$\|e_R\|$",     "rad",   "8_err_att"),
    ("v_norm",  r"$\|v\|$",       "m/s",   "9_vel"),
    ("w_norm",  r"$\|\omega\|$",  "rad/s", "10_avel"),
]
CAT_MOTION = ("v_norm", "w_norm")


def fig_cat_nom_vs_pert(cats, V, stat, key, sym, unit, out_path, stem):
    """One quantity: the manoeuvres on the x axis, four bars per manoeuvre,
    without ratios or normalisation.

    Args:
        cats (list[str]): Manoeuvre categories on the x axis.
        V (dict): Values keyed by ``(group, condition, category, statistic, metric)``, condition ``'nom'`` or ``'pert'``.
        stat (str): Statistic, ``'RMSE'`` or ``'MAE'``.
        key (str): Quantity, as in ``CAT_FIGS``.
        sym (str): Symbol of the quantity (mathtext).
        unit (str): Unit of the quantity.
        out_path (pathlib.Path): Output file.
        stem (str): Mission name (e.g. ``p2p``), used in the titles.

    Returns:
        bool: ``True`` when the figure is written.
    """
    order = [("PPO", "nom"), ("PPO", "pert"), ("SPPO", "nom"), ("SPPO", "pert")]
    allv = [V.get((g, c, cat, stat, key), NAN) for g, c in order for cat in cats]
    finite = [v for v in allv if v == v]
    if not finite:
        return False
    top = max(finite) * 1.30
    fmt = "{:.3f}" if unit in ("m", "m/s") else "{:.4f}"

    fig, ax = plt.subplots(figsize=(2.1 * len(cats) + 5.0, 6.4))
    x = np.arange(len(cats))
    pitch, w = 0.215, 0.20
    for i, (g, c) in enumerate(order):
        vals = [V.get((g, c, cat, stat, key), NAN) for cat in cats]
        ax.bar(x + (i - 1.5) * pitch, vals, w, color=COL[g], alpha=CAT_ALPHA[c],
               hatch=CAT_HATCH[c], edgecolor="white", linewidth=1.1,
               label=f"{g} {COND_LAB[c]}", zorder=2)
        for xi, vi in zip(x, vals):
            if vi == vi:
                ax.text(xi + (i - 1.5) * pitch, vi, fmt.format(vi), ha="center",
                        va="bottom", fontsize=9, rotation=90,
                        color=C_INK, zorder=4)
    ax.set_ylim(0, top)
    ax.set_ylabel(f"{stat} {sym}   [{unit}]")
    ax.set_title(f"{sym}  -  {stat}  -  nominal vs perturbed  -  {stem}", pad=10)
    ax.grid(axis="x", visible=False)
    ax.set_xticks(x)
    ax.set_xticklabels(list(cats))
    ax.set_xlim(-0.65, len(cats) - 0.35)
    ax.legend(loc="best", ncol=2, frameon=True, fontsize=9.5)
    fig.tight_layout()
    save(fig, out_path)
    print("  written", out_path)
    return True


def fig_cat_degradation(cats, V, stat, key, sym, out_path, stem):
    """One quantity: D = perturbed / nominal, two bars per manoeuvre.

    Args:
        cats (list[str]): Manoeuvre categories on the x axis.
        V (dict): Values keyed by ``(group, condition, category, statistic, metric)``, condition ``'nom'`` or ``'pert'``.
        stat (str): Statistic, ``'RMSE'`` or ``'MAE'``.
        key (str): Quantity, as in ``CAT_FIGS``.
        sym (str): Symbol of the quantity (mathtext).
        out_path (pathlib.Path): Output file.
        stem (str): Mission name (e.g. ``p2p``), used in the titles.

    Returns:
        bool: ``True`` when the figure is written.
    """
    d_all = {}
    for g in ("PPO", "SPPO"):
        d_all[g] = [ratio(V.get((g, "pert", cat, stat, key), NAN),
                          V.get((g, "nom", cat, stat, key), NAN)) for cat in cats]
    finite = [v for vs in d_all.values() for v in vs if v == v]
    if not finite:
        return False

    fig, ax = plt.subplots(figsize=(2.1 * len(cats) + 5.0, 6.0))
    x = np.arange(len(cats))
    pitch, w = 0.38, 0.34
    for i, g in enumerate(("PPO", "SPPO")):
        ax.bar(x + (i - 0.5) * pitch, d_all[g], w, color=COL[g],
               edgecolor="white", linewidth=1.4, label=g, zorder=2)
        for xi, vi in zip(x, d_all[g]):
            if vi == vi:
                ax.text(xi + (i - 0.5) * pitch, vi, f"{vi:.2f}", ha="center",
                        va="bottom", fontsize=9, color=C_INK)
    ax.axhline(1.0, color="#333333", lw=1.2, ls="--", zorder=3,
               label="$D=1$ (no degradation)")
    if min(finite) < 0.995:
        ax.axhspan(0, 1.0, color="#999999", alpha=0.07, zorder=0)
    ax.set_ylim(0, max(finite + [1.0]) * 1.26)
    ax.set_ylabel(r"$D$ = perturbed / nominal   [-]")
    ax.set_title(f"{sym}  -  {stat}  -  degradation  -  {stem}", pad=10)
    ax.grid(axis="x", visible=False)
    ax.set_xticks(x)
    ax.set_xticklabels(list(cats))
    ax.set_xlim(-0.65, len(cats) - 0.35)
    ax.legend(loc="best", ncol=2, frameon=True, fontsize=9.5)
    fig.tight_layout()
    save(fig, out_path)
    print("  written", out_path)
    return True


def fig_cat_delta(cats, V, stat, key, sym, unit, out_path, stem):
    """DELTA = perturbed - nominal, two bars per manoeuvre.

    The reading for ROBUSTNESS. The disturbance is the same for the two
    policies: the more robust one is the one that shifts less, and a shift is
    a difference. The ratio D, in the next folder, divides by the nominal
    value and penalises the policy that starts lower, so on the same data the
    two readings often disagree.

    Below zero the perturbed flight did BETTER than the nominal one. With an
    added load this has no physical explanation: it is flight-to-flight
    variability, and its size gives the threshold below which no difference
    between the two policies can be read. The negative region is shaded.

    Args:
        cats (list[str]): Manoeuvre categories on the x axis.
        V (dict): Values keyed by ``(group, condition, category, statistic, metric)``, condition ``'nom'`` or ``'pert'``.
        stat (str): Statistic, ``'RMSE'`` or ``'MAE'``.
        key (str): Quantity, as in ``CAT_FIGS``.
        sym (str): Symbol of the quantity (mathtext).
        unit (str): Unit of the quantity.
        out_path (pathlib.Path): Output file.
        stem (str): Mission name (e.g. ``p2p``), used in the titles.

    Returns:
        bool: ``True`` when the figure is written.
    """
    d_all = {}
    for g in ("PPO", "SPPO"):
        d_all[g] = [V.get((g, "pert", cat, stat, key), NAN)
                    - V.get((g, "nom", cat, stat, key), NAN) for cat in cats]
    finite = [v for vs in d_all.values() for v in vs if v == v]
    if not finite:
        return False

    fig, ax = plt.subplots(figsize=(2.1 * len(cats) + 5.0, 6.0))
    x = np.arange(len(cats))
    pitch, w = 0.38, 0.34
    fmt = "{:+.3f}" if unit in ("m", "m/s") else "{:+.4f}"
    for i, g in enumerate(("PPO", "SPPO")):
        ax.bar(x + (i - 0.5) * pitch, d_all[g], w, color=COL[g],
               edgecolor="white", linewidth=1.4, label=g, zorder=2)
        for xi, vi in zip(x, d_all[g]):
            if vi == vi:
                ax.text(xi + (i - 0.5) * pitch, vi, fmt.format(vi),
                        ha="center", va="bottom" if vi >= 0 else "top",
                        fontsize=9, color=C_INK)
    lo, hi = min(finite + [0.0]), max(finite + [0.0])
    pad = 0.30 * max(hi - lo, 1e-9)
    ax.set_ylim(lo - pad, hi + pad)
    if lo < -1e-12:
        ax.axhspan(lo - pad, 0.0, color="#999999", alpha=0.07, zorder=0)
    ax.axhline(0.0, color="#333333", lw=1.2, zorder=3)
    ax.set_ylabel(rf"$\Delta$ {sym}   [{unit}]")
    ax.set_title(f"{sym}  -  {stat}  -  shift under the disturbance  -  {stem}")
    ax.grid(axis="x", visible=False)
    ax.set_xticks(x)
    ax.set_xticklabels(list(cats))
    ax.set_xlim(-0.65, len(cats) - 0.35)
    ax.legend(loc="best", ncol=2, frameon=True, fontsize=9.5)
    fig.tight_layout()
    save(fig, out_path)
    print("  written", out_path)
    return True


def _draw_cat_range(ax, cats, V, stat, key, sym, unit):
    """Draw on a given axis the mid-point and range plot of one quantity: two
    columns per manoeuvre (PPO, SPPO), the column at the mid-point between
    nominal and perturbed, the black whisker from the lower to the higher of
    the two. The whisker is the INTERVAL spanned by the tested model
    uncertainty, not a statistical deviation: two numbers are not a sample.
    The plot shows centre and width, without marking which end is nominal.

    The y label holds only symbol and unit; the statistic goes in the
    caption.

    Used by fig_cat_range (one quantity, one figure) and by
    fig_cat_range_pair (position and attitude in a 1x2 grid).
    Return False, without touching the axis, when there is nothing to draw.

    Args:
        ax (matplotlib.axes.Axes): Axis.
        cats (list[str]): Manoeuvre categories on the x axis.
        V (dict): Values keyed by ``(group, condition, category, statistic, metric)``, condition ``'nom'`` or ``'pert'``.
        stat (str): Statistic, ``'RMSE'`` or ``'MAE'``.
        key (str): Quantity, as in ``CAT_FIGS``.
        sym (str): Symbol of the quantity (mathtext).
        unit (str): Unit of the quantity.

    Returns:
        bool: ``True`` when the plot is drawn.
    """
    mid, lo, hi = {}, {}, {}
    for g in ("PPO", "SPPO"):
        nom = [V.get((g, "nom", cat, stat, key), NAN) for cat in cats]
        pert = [V.get((g, "pert", cat, stat, key), NAN) for cat in cats]
        mid[g] = [(a + b) / 2 if a == a and b == b else NAN
                  for a, b in zip(nom, pert)]
        lo[g] = [min(a, b) if a == a and b == b else NAN
                 for a, b in zip(nom, pert)]
        hi[g] = [max(a, b) if a == a and b == b else NAN
                 for a, b in zip(nom, pert)]
    finite = [v for g in ("PPO", "SPPO") for v in hi[g] if v == v]
    if not finite:
        return False
    top = max(finite) * 1.18

    x = np.arange(len(cats))
    pitch, w = 0.38, 0.34
    for i, g in enumerate(("PPO", "SPPO")):
        xi = x + (i - 0.5) * pitch
        yerr_lo = [m - l if m == m and l == l else 0.0
                   for m, l in zip(mid[g], lo[g])]
        yerr_hi = [h - m if m == m and h == h else 0.0
                   for m, h in zip(mid[g], hi[g])]
        ax.bar(xi, mid[g], w, color=COL[g], edgecolor="none", label=g,
               zorder=2)
        ax.errorbar(xi, mid[g], yerr=[yerr_lo, yerr_hi], fmt="none",
                    ecolor="#222222", elinewidth=1.3, capsize=5,
                    capthick=1.3, zorder=3)
    ax.set_ylim(0, top)
    ax.set_ylabel(f"{sym}   [{unit}]")
    ax.grid(axis="x", visible=False)
    ax.set_xticks(x)
    ax.set_xticklabels(list(cats), fontsize=16)
    ax.set_xlim(-0.65, len(cats) - 0.35)
    return True


def fig_cat_range(cats, V, stat, key, sym, unit, out_path, stem):
    """One quantity, one figure: see _draw_cat_range for the drawing and the
    rationale of the mid-point and range plot. The raw values
    (fig_cat_nom_vs_pert) and D/Delta are also written; this figure and
    fig_cat_range_pair are the ones used in the thesis.

    Args:
        cats (list[str]): Manoeuvre categories on the x axis.
        V (dict): Values keyed by ``(group, condition, category, statistic, metric)``, condition ``'nom'`` or ``'pert'``.
        stat (str): Statistic, ``'RMSE'`` or ``'MAE'``.
        key (str): Quantity, as in ``CAT_FIGS``.
        sym (str): Symbol of the quantity (mathtext).
        unit (str): Unit of the quantity.
        out_path (pathlib.Path): Output file.
        stem (str): Mission name (e.g. ``p2p``), used in the titles.

    Returns:
        bool: ``True`` when the figure is written.
    """
    fig, ax = plt.subplots(figsize=(1.9 * len(cats) + 4.6, 6.2))
    if not _draw_cat_range(ax, cats, V, stat, key, sym, unit):
        plt.close(fig)
        return False
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    save(fig, out_path)
    print("  written", out_path)
    return True


def fig_cat_range_pair(cats, V, stat, out_path, stem):
    """Position and attitude side by side in a 1x2 grid: the same mid-point
    and range plot as fig_cat_range (see _draw_cat_range), one quantity per
    panel and one shared legend.

    Args:
        cats (list[str]): Manoeuvre categories on the x axis.
        V (dict): Values keyed by ``(group, condition, category, statistic, metric)``, condition ``'nom'`` or ``'pert'``.
        stat (str): Statistic, ``'RMSE'`` or ``'MAE'``.
        out_path (pathlib.Path): Output file.
        stem (str): Mission name (e.g. ``p2p``), used in the titles.

    Returns:
        bool: ``True`` when the figure is written.
    """
    specs = [(k, sm, u) for k, sm, u, _f in CAT_FIGS if k in ("e_p", "e_R")]
    fig_w, fig_h = 11, 4.6
    fig, axs = plt.subplots(1, 2, figsize=(fig_w, fig_h))
    ok = False
    for ax, (key, sym, unit) in zip(axs, specs):
        if _draw_cat_range(ax, cats, V, stat, key, sym, unit):
            ok = True
    if not ok:
        plt.close(fig)
        return False
    # As in fig_overlay_grid: legend inside the position panel (upper
    # right, where the columns are lower).
    axs[0].legend(loc="upper right", frameon=True)
    fig.tight_layout()
    save(fig, out_path)
    print("  written", out_path)
    return True


def fig_tv_rotors_range(cats, V, stat, key, sym, unit, out_path, stem):
    """TV per rotor, with the style and height of pos_att_pair_RMSE.pdf.

    Args:
        cats (list[str]): Rotor labels on the x axis.
        V (dict): Values keyed by ``(group, condition, category, statistic, metric)``, condition ``'nom'`` or ``'pert'``.
        stat (str): Statistic, ``'RMSE'`` or ``'MAE'``.
        key (str): Quantity, as in ``CAT_FIGS``.
        sym (str): Symbol of the quantity (mathtext).
        unit (str): Unit of the quantity.
        out_path (pathlib.Path): Output file.
        stem (str): Mission name (e.g. ``p2p``), used in the titles.

    Returns:
        bool: ``True`` when the figure is written.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    if not _draw_cat_range(ax, cats, V, stat, key, sym, unit):
        plt.close(fig)
        return False
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    save(fig, out_path)
    print("  written", out_path)
    return True


def read_tv_rotors(folder: Path):
    """{(group, rotor): TV over the whole flight per rotor} from tv_windows.csv,
    window=='flight', channels M1..M6 (the TOTAL row is read by
    read_tv_flight). The keys stay those of the CSV (M1..M6); the thesis
    labels (P1..P6) are applied in tv_rotors_range_section.

    Args:
        folder (pathlib.Path): Mission folder analysed by ``analyze_trajectory``.

    Returns:
        dict: ``{(group, rotor): value}``; empty when the file is missing.
    """
    path = tables_dir(folder) / "tv_windows.csv"
    if not path.is_file():
        return {}
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get('window') == 'flight' and r.get('channel', '') != 'TOTAL' \
                    and r.get('channel', '').startswith('M'):
                out[(r['group'], r['channel'])] = fnum(r['tv'])
    return out


def tv_rotors_range_section(nom_dir, pert_dir, dirs, stem):
    """Total variation of the command per rotor (eq:total_variation), whole
    flight, nominal/perturbed comparison with the same mid-point and range
    plot as compare_categories_section: here 'cats' are the six rotors,
    labelled P1..P6 as in Section 2.4.1.

    Write tables/tv_comparison_rotors.csv and
    propellers/03_tv_rotors_range.pdf.

    Args:
        nom_dir (pathlib.Path): Nominal mission folder.
        pert_dir (pathlib.Path): Perturbed mission folder.
        dirs (dict[str, pathlib.Path]): Output folders.
        stem (str): Mission name, used in the titles.
    """
    rotor_labels = {f"M{i}": f"P{i}" for i in range(1, 7)}
    Tn, Tp = read_tv_rotors(nom_dir), read_tv_rotors(pert_dir)
    if not Tn or not Tp:
        print("  [tv rotors] tv_windows.csv without per-motor rows in one "
              "of the two folders, skipped")
        return

    rotors = [f"M{i}" for i in range(1, 7)
              if all((g, f"M{i}") in Tn and (g, f"M{i}") in Tp for g in GROUPS)]
    if not rotors:
        print("  [tv rotors] no rotor in common, skipped")
        return
    labels = [rotor_labels[m] for m in rotors]

    scale = 1.0 / (2 * math.pi)

    V = {}
    for g in GROUPS:
        for cond, T in (('nom', Tn), ('pert', Tp)):
            for m, lab in zip(rotors, labels):
                V[(g, cond, lab, '-', 'tv')] = T[(g, m)] * scale

    rows = []
    for m, lab in zip(rotors, labels):
        row = {'rotor': lab}
        for g in GROUPS:
            vn, vp = Tn[(g, m)] * scale, Tp[(g, m)] * scale
            row[f'{g.lower()}_nom'] = vn
            row[f'{g.lower()}_pert'] = vp
            row[f'{g.lower()}_mid'] = (vn + vp) / 2
            row[f'{g.lower()}_lo'] = min(vn, vp)
            row[f'{g.lower()}_hi'] = max(vn, vp)
        rows.append(row)
    csv_path = dirs['tables'] / "tv_comparison_rotors.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  written {csv_path}")

    out_path = dirs['propellers'] / "03_tv_rotors_range.pdf"
    if fig_tv_rotors_range(labels, V, '-', 'tv', r'$\mathrm{TV}_i$', 'Hz/s',
                           out_path, stem):
        print(f"  written {out_path}")


def compare_categories_section(nom, pert, dirs, stem, exclude):
    """Comparison by category: tables/comparison_categories.csv and the
    category figures.

    Args:
        nom (pathlib.Path): Nominal mission folder.
        pert (pathlib.Path): Perturbed mission folder.
        dirs (dict[str, pathlib.Path]): Output folders.
        stem (str): Mission name (e.g. ``p2p``); it selects the categories.
        exclude (list[str]): Categories left out of the figures (kept in the CSV).
    """
    Mn, Mp = read_category_metrics(nom), read_category_metrics(pert)
    units = {**Mn.get("_units", {}), **Mp.get("_units", {})}

    V = {}
    for src, cond in ((Mn, "nom"), (Mp, "pert")):
        for k, v in src.items():
            if k == "_units":
                continue
            g, cat, st, met = k
            V[(g, cond, cat, st, met)] = v

    cats_all = CATEGORIES_BY_MISSION.get(stem, CATEGORIES)
    cats = [c for c in cats_all
            if all((g, cnd, c, "-", "rmse_pos") in V
                   for g in ("PPO", "SPPO") for cnd in ("nom", "pert"))]
    if not cats:
        raise SystemExit("No category present in both conditions.")
    # (statistic, metric): '-' marks the metrics that already contain a
    # statistic (lag_z is a mean, t10_s a time, tv a total variation).
    # The _abs counters are bookkeeping, not metrics.
    mets = [x for x in dict.fromkeys(
        (k[3], k[4]) for k in V if k[2] in cats and k[4] not in
        ("n_samples", "n_subblocks", "dur_s",
         "n_samples_abs", "n_subblocks_abs",
         "n_subblocks_real", "n_subblocks_real_att"))]
    stats_av = [st for st, _a in STATS if any(x[0] == st for x in mets)]

    print(f"\n{'='*100}\nnominal vs perturbed by category  -  {stem}\n{'='*100}")
    print(f"  nominal  : {nom.name}\n  perturbed: {pert.name}")
    print(f"  categories:  {', '.join(cats)}")

    rows, n_sub1 = [], 0
    for cat in cats:
        print(f"\n{'-'*100}\n  [{cat}]\n{'-'*100}")
        print(f"    {'metric':14s}{'PPO nom':>11s}{'PPO pert':>11s}{'SPPO nom':>11s}"
              f"{'SPPO pert':>11s}{'D_PPO':>8s}{'D_SPPO':>8s}"
              f"{'DELTA_PPO':>11s}{'DELTA_SPPO':>11s}   result")
        for st, met in mets:
            get = lambda g, c: V.get((g, c, cat, st, met), NAN)
            pn, pp, sn, sp = get("PPO", "nom"), get("PPO", "pert"), \
                             get("SPPO", "nom"), get("SPPO", "pert")
            if not any(v == v for v in (pn, pp, sn, sp)):
                continue
            signed = met in CAT_SIGNED
            dp = ds = NAN if signed else ratio(pp, pn)
            if not signed:
                ds = ratio(sp, sn)
            r = {"category": cat, "stat": st, "metric": met,
                 "unit": units.get(met, ""), "signed": int(signed),
                 "ppo_nom": pn, "ppo_pert": pp, "sppo_nom": sn, "sppo_pert": sp,
                 "D_ppo": dp, "D_sppo": ds,
                 "delta_nom_pct": NAN if signed else pct(sn, pn),
                 "delta_pert_pct": NAN if signed else pct(sp, pp),
                 "delta_D_pct": NAN if signed else pct(ds, dp),
                 "diff_nom_abs": (sn - pn) if signed else NAN,
                 "diff_pert_abs": (sp - pp) if signed else NAN,
                 # DELTA: the shift caused by the disturbance, the reading for
                 # robustness: the disturbance is the same for the two policies, so
                 # the question is how much the error shifts, and a shift is a
                 # difference. See compare_cell and the maneuvers/delta/ folder.
                 "delta_abs_ppo": pp - pn,
                 "delta_abs_sppo": sp - sn}
            rows.append(r)
            if not signed:
                for v in (dp, ds):
                    if v == v and v < 0.95:
                        n_sub1 += 1
            lab_m = met if st == '-' else f"{met} [{st}]"
            if signed:
                u = units.get(met, "")
                print(f"    {lab_m:20s}{pn:11.4f}{pp:11.4f}{sn:11.4f}{sp:11.4f}"
                      f"{'--':>8s}{'--':>8s}"
                      f"{sn-pn:+8.4f} {sp-pp:+8.4f} {u:<6s}  (signed)")
            else:
                # the three readings often disagree: D penalises the policy that
                # starts lower, so it may say "degrades more" for the policy with the
                # lower error in both conditions. When this happens it is written on
                # the row.
                da, dl = r['delta_abs_ppo'], r['delta_abs_sppo']
                w_pert = 'SPPO' if sp < pp else 'PPO'
                w_D = 'SPPO' if ds < dp else 'PPO'
                w_del = 'SPPO' if abs(dl) < abs(da) else 'PPO'
                note = ""
                if w_D != w_del:
                    note = f"  !! D says {w_D}, DELTA says {w_del}"
                elif w_D != w_pert:
                    note = f"  !! lower error {w_pert}, lower degradation {w_D}"
                if min(da, dl) < 0:
                    note += "  (DELTA<0: noise)"
                print(f"    {lab_m:20s}{pn:11.4f}{pp:11.4f}{sn:11.4f}{sp:11.4f}"
                      f"{dp:8.2f}{ds:8.2f}{da:+11.4f}{dl:+11.4f}{note}")

    csv_path = dirs['tables'] / "comparison_categories.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  written {csv_path}")

    # -------------------------------------------- size of the noise
    # The disturbance is an added load, so the error can only grow. Every
    # negative DELTA is flight-to-flight variability, and the largest one
    # gives the threshold below which no difference between the two
    # policies can be read. Comparing it with the difference between the
    # DELTA of the two policies tells whether there is a signal.
    _pos = [r for r in rows if not r['signed']
            and r['delta_abs_ppo'] == r['delta_abs_ppo']
            and r['delta_abs_sppo'] == r['delta_abs_sppo']
            and r['metric'] in ('e_p', 'rmse_pos')]
    if _pos:
        neg = [(r, v) for r in _pos
               for v in (r['delta_abs_ppo'], r['delta_abs_sppo']) if v < 0]
        gap = [abs(r['delta_abs_sppo'] - r['delta_abs_ppo']) for r in _pos]
        print(f"\n{'-'*100}\n  ROBUSTNESS: IS THERE A SIGNAL?\n{'-'*100}")
        print("  With an added load the error can only grow: every negative DELTA")
        print("  is flight-to-flight variability, and measures its scale.")
        if neg:
            worst = max(abs(v) for _r, v in neg)
            print(f"    {len(neg)} negative DELTA, the largest "
                  f"{worst*1000:.1f} mm -> noise >= {worst*1000:.0f} mm")
        else:
            worst = 0.0
            print("    no negative DELTA: the noise cannot be estimated from these data.")
        print(f"    difference between the DELTA of PPO and SPPO: "
              f"{min(gap)*1000:.1f}-{max(gap)*1000:.1f} mm "
              f"(median {sorted(gap)[len(gap)//2]*1000:.1f} mm)")
        if worst > 0 and sorted(gap)[len(gap)//2] < worst:
            print("    -> the effect is SMALLER than the noise shown by the data:")
            print("       with one flight per condition it cannot be told apart from")
            print("       zero. A robustness claim needs more disturbance levels")
            print("       (a slope, not two points)")
            print("       or paired repetitions over several policies.")
        elif worst > 0:
            print("    -> the effect exceeds the estimated noise: the difference")
            print("       between the policies is readable, with one flight per condition.")

    shown = [c for c in cats if c not in exclude]

    # --- NORMALISATION, steps and ramps.
    # Steps: the four flights do not cover the same distance, so the
    # highest bar would only be the longest step.
    # Ramps: the vehicle does not land from the same height, so the
    # perturbed landing descends 0.78 m where the nominal one descends
    # 0.83 m. For both, analyze_trajectory writes the same columns
    # (*_pooled_n_*, *_pooled_den): the error normalised sample by sample
    # by the distance to cover.
    # Here they are scaled back to a denominator COMMON to the four
    # flights, so the number stays in metres and shares the axis with the
    # hold categories, which have no distance to cover. The absolute value
    # is kept in comparison_categories.csv.
    ORD4 = [("PPO", "nom"), ("PPO", "pert"), ("SPPO", "nom"), ("SPPO", "pert")]
    for cat in shown:
        if KIND.get(cat) not in ('step', 'ramp'):
            continue
        for fld, skey in (('pos', 'e_p'), ('att', 'e_R')):
            den = [V.get((g, c, cat, "-", f"{fld}_pooled_den"), NAN)
                   for g, c in ORD4]
            if any(v != v for v in den):
                continue
            dbar = float(sum(den) / len(den))
            vals, bad = {}, False
            for st in stats_av:
                for g, c in ORD4:
                    v = V.get((g, c, cat, "-", f"{fld}_pooled_n_{st}"), NAN)
                    if v != v:
                        bad = True
                        break
                    vals[(g, c, st)] = v * dbar
                if bad:
                    break
            if bad or not vals:
                continue
            for (g, c, st), v in vals.items():
                V[(g, c, cat, st, skey)] = v
            u = 'm' if fld == 'pos' else 'rad'
            print(f"  FIGURE  {cat} ({fld}): denominators "
                  + " / ".join(f"{v:.3f}" for v in den)
                  + f" {u} -> common {dbar:.3f} {u}")

    # maneuvers/ = takeoff, hold, landing. The step is in
    # maneuvers/step/, as in the single-mission analysis.
    shown_man = [c for c in shown if KIND.get(c) != 'step']
    shown_step = [c for c in shown if KIND.get(c) == 'step']
    for tag, cats_sel, k_abs, k_deg, k_del, k_range in (
            ('maneuvers', shown_man, 'man_', 'man_degradation', 'man_delta',
             'man_range'),
            ('step', shown_step, 'step_', 'step_degradation', 'step_delta',
             'step_range')):
        if not cats_sel:
            continue
        for st in stats_av:
            d = dirs[k_abs + st]
            n_fig = 0
            for key, sym, unit, fname in CAT_FIGS:
                if fig_cat_nom_vs_pert(cats_sel, V, st, key, sym, unit,
                                       d / f"{fname}.pdf", stem):
                    n_fig += 1
            print(f"  {n_fig} figures written in {d}")
        for k_dir, fn in ((k_deg, fig_cat_degradation), (k_del, fig_cat_delta)):
            d = dirs[k_dir]
            n_fig = 0
            for st in stats_av:
                for key, sym, unit, fname in CAT_FIGS:
                    args = ((cats_sel, V, st, key, sym, d / f"{fname}_{st}.pdf",
                             stem) if fn is fig_cat_degradation else
                            (cats_sel, V, st, key, sym, unit,
                             d / f"{fname}_{st}.pdf", stem))
                    if fn(*args):
                        n_fig += 1
            print(f"  {n_fig} figures written in {d}")
        d = dirs[k_range]
        n_fig = 0
        for st in stats_av:
            for key, sym, unit, fname in CAT_FIGS:
                if fig_cat_range(cats_sel, V, st, key, sym, unit,
                                 d / f"{fname}_{st}.pdf", stem):
                    n_fig += 1
        for st in stats_av:
            if fig_cat_range_pair(cats_sel, V, st, d / f"pos_att_pair_{st}.pdf",
                                  stem):
                n_fig += 1
        print(f"  {n_fig} figures written in {d}")

    # ---- summary
    #
    # Two metrics are not counted as they are:
    #
    #  - in 'Step' rmse_pos is proportional to the step amplitude, which
    #    is not the same in the two flights: the normalised value is counted.
    #  - rmse_att is dominated by the yaw, which on this vehicle carries an
    #    offset from the mounting, common to the two policies. The regulated
    #    part (att_rp) and the yaw part (att_yaw) are reported separately.
    SUMMARY_KEY = {"Step": "pos_pooled_n_RMSE"}   # position: what to count where

    def tally(getter, keyfn, cats_sel):
        sel = []
        for cat in cats_sel:
            met = keyfn(cat)
            r = next((r for r in rows if r["category"] == cat
                      and r["metric"] == met and r["stat"] == "-"
                      and not r["signed"]), None)
            if r is not None and getter(r) == getter(r):
                sel.append((cat, met, getter(r)))
        w = sum(1 for _c, _m, v in sel if v < 0)
        return w, len(sel), sel

    print(f"\n{'='*100}\nSUMMARY\n{'='*100}")
    print("  POSITION  (in 'Step' pos_pooled_n_RMSE is counted, normalised "
          "by the distance to cover)")
    for lab, g in (("nominal ", lambda r: r["delta_nom_pct"]),
                   ("perturbed", lambda r: r["delta_pert_pct"]),
                   ("lower degr.", lambda r: r["delta_D_pct"])):
        w, n, sel = tally(g, lambda c: SUMMARY_KEY.get(c, "rmse_pos"), cats)
        det = "  ".join(f"{c.split()[0][:6]}:{v:+.0f}%" for c, _m, v in sel)
        print(f"    {lab:14s} SPPO better {w}/{n}    {det}")
    print("  HORIZONTAL COMPONENT  (lateral drift: no commanded amplitude)")
    for lab, g in (("nominal ", lambda r: r["delta_nom_pct"]),
                   ("perturbed", lambda r: r["delta_pert_pct"])):
        w, n, sel = tally(g, lambda c: "err_xy", cats)
        det = "  ".join(f"{c.split()[0][:6]}:{v:+.0f}%" for c, _m, v in sel)
        print(f"    {lab:14s} SPPO better {w}/{n}    {det}")
    print("  ATTITUDE  (split: regulated part and yaw offset)")
    for nm, met in (("roll+pitch", "att_rp"), ("yaw", "att_yaw")):
        for lab, g in (("nominal ", lambda r: r["delta_nom_pct"]),
                       ("perturbed", lambda r: r["delta_pert_pct"])):
            w, n, sel = tally(g, lambda c: met, cats)
            if not n:
                continue
            det = "  ".join(f"{c.split()[0][:6]}:{v:+.0f}%" for c, _m, v in sel)
            print(f"    {nm:11s}{lab:11s} SPPO better {w}/{n}    {det}")
    n_tot = sum(1 for r in rows if not r["signed"]) * 2
    if n_sub1:
        print(f"\n  {n_sub1} values of D out of {n_tot} below 0.95: the perturbed flight did better")
        print("  than the nominal one, which has no physical explanation with an")
        print("  added load: it is flight-to-flight variability.")
    print("\n  One flight per condition: single values, without uncertainty estimates.")


def fig_metric_degradation(metric, c, out_path, stat='RMSE'):
    """One metric, two bars: the degradation of each policy.

    D = perturbed / nominal. It is read next to the absolute values, since it
    penalises the policy that starts from a lower error.

    Args:
        metric (str): Name of the metric.
        c (dict): Derived numbers from ``compare_cell``.
        out_path (pathlib.Path): Output file.
        stat (str): Statistic, ``'RMSE'`` or ``'MAE'``.

    Returns:
        bool: ``True`` when the figure is written.
    """
    dp, ds = c.get('D_ppo', NAN), c.get('D_sppo', NAN)
    if dp != dp and ds != ds:
        return False
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for xi, (v, g) in enumerate(((dp, 'PPO'), (ds, 'SPPO'))):
        ax.bar(xi, v, 0.5, color=COL[g], edgecolor='white', linewidth=1.4,
               zorder=2, label=g)
        if v == v:
            ax.text(xi, v, f"{v:.3f}", ha='center', va='bottom', fontsize=11,
                    color='#222222', zorder=4)
    lo = min([1.0] + [v for v in (dp, ds) if v == v])
    ax.axhline(1.0, color='k', lw=1.2, ls='--', zorder=3,
               label='$D = 1$ (no degradation)')
    if lo < 0.995:
        ax.axhspan(0, 1.0, color='#999999', alpha=0.08, zorder=0)
    top = max([1.0] + [v for v in (dp, ds) if v == v])
    ax.set_ylim(0, top * 1.28)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['PPO', 'SPPO'])
    ax.set_ylabel(r'$D$ = perturbed / nominal   [-]')
    ax.set_title(f"{metric}  -  {stat}")
    ax.grid(axis='x', visible=False)
    ax.set_xlim(-0.65, 1.35)
    ax.legend(loc='best', frameon=True, fontsize=9.5)
    fig.tight_layout()
    save(fig, out_path)
    return True


# ===========================================================================
def run(nom_dir: Path, pert_dir: Path, out_dir: Path, floor: float,
        cat_exclude=()):
    """Compare the nominal and perturbed flights of one mission.

    Reads the tables of the two folders, checks that they were analysed with
    the same parameters, and writes the comparison tables and figures.

    Args:
        nom_dir (pathlib.Path): Nominal mission folder.
        pert_dir (pathlib.Path): Perturbed mission folder.
        out_dir (pathlib.Path): Output folder.
        floor (float): Relative difference [%] below which no winner is assigned.
        cat_exclude (list[str]): Categories left out of the figures (kept in the CSV).
    """
    set_style()
    # Same tree as the single-mission analysis, plus the dimension that
    # only the comparison has: the degradation.
    # The comparison is made on MANOEUVRES (takeoff, hold, landing and step
    # response), each with a window identified from the reference.
    dirs = {'tables': out_dir / "tables",
            'roll': out_dir / "Rollouts",
            'propellers': out_dir / "propellers",
            'man_RMSE': out_dir / "maneuvers" / "RMSE",
            'man_MAE': out_dir / "maneuvers" / "MAE",
            'man_degradation': out_dir / "maneuvers" / "degradation",
            # delta/ next to degradation/: the same comparison read as a
            # difference instead of a ratio. The two readings often disagree on
            # the data, so they are in two folders.
            'man_delta': out_dir / "maneuvers" / "delta",
            # range/: nominal-perturbed mid-point with a whisker between the
            # two, the reading used in the thesis (see fig_cat_range).
            'man_range': out_dir / "maneuvers" / "range",
            'step_delta': out_dir / "maneuvers" / "step" / "delta",
            'step_range': out_dir / "maneuvers" / "step" / "range",
            # as in the single-mission analysis: the manoeuvres of the mission
            # on one side, the step on the other
            'step_RMSE': out_dir / "maneuvers" / "step" / "RMSE",
            'step_MAE': out_dir / "maneuvers" / "step" / "MAE",
            'step_degradation': out_dir / "maneuvers" / "step" / "degradation"}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print(f"nominal   : {nom_dir}")
    print(f"perturbed : {pert_dir}")
    print(f"output    : {out_dir}")
    print(f"threshold : {floor:.1f}%  (no winner is assigned below it)\n")

    check_params(nom_dir, pert_dir)

    pm_nom, pm_pert = read_phase_metrics(nom_dir), read_phase_metrics(pert_dir)
    raw_nom, raw_pert = dict(pm_nom), dict(pm_pert)

    # --------------------------------------------- consistency check
    ph_nom = list(dict.fromkeys(p for _, p in pm_nom))
    ph_pert = list(dict.fromkeys(p for _, p in pm_pert))
    phases = [p for p in ph_nom if p in ph_pert]
    if not phases:
        sys.exit("No phase in common between the two folders: is it the same mission?")
    print(f"{'='*78}\nCONSISTENCY OF THE TWO FOLDERS\n{'='*78}")
    print(f"  nominal phases  :  {', '.join(ph_nom)}")
    print(f"  perturbed phases:  {', '.join(ph_pert)}")
    if ph_nom != ph_pert:
        print("  !! the two phase sequences differ: comparison limited to")
        print(f"     the common ones ({', '.join(phases)}). Check that the two")
        print("     flights ran the same mission.")
    else:
        print("  -> same phase sequence.")

    # The MISSION SETPOINTS must also coincide between nominal and
    # perturbed, otherwise D compares two different missions. Takeoff and
    # landing are not checked: their reference is the ground pose, which
    # changes in every flight; there the commanded displacement counts,
    # checked by analyze_trajectory within each folder.
    worst = 0.0
    for p in [x for x in phases if x.startswith('SETPOINT')]:
        for g in GROUPS:
            a, b = pm_nom.get((g, p)), pm_pert.get((g, p))
            if not a or not b:
                continue
            d = max(abs(a[k] - b[k]) for k in ('ref_x', 'ref_y', 'ref_z')
                    if a[k] == a[k] and b[k] == b[k])
            worst = max(worst, d)
    print(f"  maximum difference between the mission setpoints of the two conditions: "
          f"{worst*1000:.1f} mm")
    if worst > 0.01:
        print("  !! above 1 cm: nominal and perturbed do not track the same")
        print("     reference, so D is not only the degradation.")
    else:
        print("  -> same mission setpoints: D measures only the effect of the")
        print("     disturbance.")

    # RAMPS. Takeoff and landing start from the ground pose, which changes
    # in every flight, so the ramp length differs between the two folders.
    # The 'Takeoff' and 'Land' categories of maneuvers/ are the ramp
    # window, so a shorter ramp shortens the window and the comparison
    # also measures that difference. It is printed here.
    print(f"\n  {'ramp':<12}{'grp':<6}{'dz nominal':>14}{'dz perturbed':>16}"
          f"{'diff':>10}")
    ramp_bad = []
    for p in ('TAKEOFF', 'LANDING'):
        for g in GROUPS:
            a, b = raw_nom.get((g, p)), raw_pert.get((g, p))
            if not a or not b:
                continue
            za, zb = a.get('ramp_dz_m', NAN), b.get('ramp_dz_m', NAN)
            if za != za or zb != zb:
                print(f"  {p:<12}{g:<6}{'--':>14}{'--':>16}{'--':>10}"
                      "   (regenerate data_analysis: ramp_dz_m missing)")
                continue
            d = abs(zb - za)
            flag = ""
            if d > 0.02:
                flag = "   !! not comparable"
                ramp_bad.append((p, g, d))
            print(f"  {p:<12}{g:<6}{za:>13.3f}m{zb:>15.3f}m{d*1000:>7.0f}mm{flag}")
    if ramp_bad:
        labs = ", ".join(sorted({p for p, _g, _d in ramp_bad}))
        print(f"\n  !! {labs}: the commanded ramp has a different length in the two")
        print("     folders (the vehicle does not land from the same height).")
        print("     In maneuvers/ the bars of that category cover windows of")
        print("     different length. The figures normalise that category by the")
        print("     commanded height difference, so the bars are comparable; the")
        print("     ABSOLUTE value in comparison_categories.csv is not.")
    else:
        print("\n  -> ramps of equal length: takeoff and landing are")
        print("     comparable between the two folders.")

    # --------------------------------------------- comparisons per phase
    all_cells = {}
    rows = []
    for key, label, unit, scale, all_phases in PHASE_METRICS:
        cells = {}
        for p in phases:
            nom = {g: pm_nom[(g, p)][key] * scale
                   for g in GROUPS if (g, p) in pm_nom and key in pm_nom[(g, p)]}
            pert = {g: pm_pert[(g, p)][key] * scale
                     for g in GROUPS if (g, p) in pm_pert and key in pm_pert[(g, p)]}
            cells[p] = compare_cell(nom, pert, floor)
            rows.append(dict(metric=SHORT.get(key, key), phase=p,
                             **{k: (round(v, 6) if isinstance(v, float) else v)
                                for k, v in cells[p].items()}))
        all_cells[key] = cells

    # --------------------------------------------- console: the two readings
    for key, label, unit, scale, all_phases in PHASE_METRICS:
        cells = all_cells[key]
        usable = [p for p in phases if cells[p]['ppo_nom'] == cells[p]['ppo_nom']
                  and cells[p]['sppo_pert'] == cells[p]['sppo_pert']]
        if not usable:
            continue
        print(f"\n{'='*94}\n{SHORT.get(key, key).upper()}\n{'='*94}")
        print(f"  {'phase':<12}{'PPO nom':>10}{'PPO per':>10}{'SPPO nom':>10}"
              f"{'SPPO per':>10}{'D PPO':>8}{'D SPPO':>8}"
              f"{'lower error':>20}{'lower D':>20}")
        for p in usable:
            c = cells[p]
            def s(v, w=10, d=4):
                return f"{'--':>{w}}" if v != v else f"{v:>{w}.{d}f}"
            w_pert = (f"{c['win_pert']} ({c['delta_pert_pct']:+.0f}%)"
                      if c['win_pert'] else '--')
            w_D = (f"{c['win_D']} ({c['delta_D_pct']:+.0f}%)"
                   if c['win_D'] else '--')
            print(f"  {p:<12}{s(c['ppo_nom'])}{s(c['ppo_pert'])}"
                  f"{s(c['sppo_nom'])}{s(c['sppo_pert'])}"
                  f"{s(c['D_ppo'], 8, 2)}{s(c['D_sppo'], 8, 2)}"
                  f"{w_pert:>20}{w_D:>20}")
        # count, per phase
        n_p = sum(1 for p in usable if cells[p]['win_pert'] == 'SPPO')
        n_d = sum(1 for p in usable if cells[p]['win_D'] == 'SPPO')
        n_t = sum(1 for p in usable if cells[p]['win_pert'])
        print(f"  -> under the disturbance SPPO has the lower error in {n_p}/{n_t} phases; "
              f"the lower D in {n_d}/{n_t}.")

    # ------------------------------------------- control effort
    # TV over the whole flight, from tv_windows.csv: the only global number
    # kept, since it does not depend on a cut, and it tells the command cost
    # of the differences seen in the manoeuvres.
    glob_rows = []
    tv_nom, tv_pert = read_tv_flight(nom_dir), read_tv_flight(pert_dir)
    tv_cell = None
    if tv_nom and tv_pert:
        tv_cell = compare_cell(tv_nom, tv_pert, floor)
        glob_rows.append(dict(metric='Total variation - all motors',
                              **{k: (round(v, 6) if isinstance(v, float) else v)
                                 for k, v in tv_cell.items()}))
        c = tv_cell
        print(f"\n{'='*94}\nCONTROL EFFORT OVER THE WHOLE FLIGHT\n{'='*94}")
        print(f"  {'metric':<28}{'PPO nom':>10}{'PPO per':>10}{'SPPO nom':>10}"
              f"{'SPPO per':>10}{'D PPO':>8}{'D SPPO':>8}"
              f"{'lower error':>16}{'lower D':>16}")
        print(f"  {'Total Variation':<28}{c['ppo_nom']:>10.1f}{c['ppo_pert']:>10.1f}"
              f"{c['sppo_nom']:>10.1f}{c['sppo_pert']:>10.1f}"
              f"{c['D_ppo']:>8.2f}{c['D_sppo']:>8.2f}"
              f"{c['win_pert'] or '--':>16}{c['win_D'] or '--':>16}")

    # --------------------------------------------- figures
    # The comparison per phase is in the tables and in the console output.
    if tv_cell is not None:
        mtv = 'Total variation - all motors'
        n_fig = 0
        # the TV is not an error: the axis label is its symbol
        if fig_metric_bars(mtv, tv_cell,
                           dirs['propellers'] / "01_total_variation.pdf", floor,
                           stat=r'$TV_{\omega}$'):
            n_fig += 1
        if fig_metric_degradation(mtv, tv_cell,
                                  dirs['propellers'] / "02_total_variation_D.pdf",
                                  stat=r'$TV_{\omega}$'):
            n_fig += 1
        print(f"\n  {n_fig} figures written in {dirs['propellers']}")

    # TV over the whole flight per rotor (P1..P6), with the same mid-point
    # and range style as the other figures (see tv_rotors_range_section)
    rotor_stem = nom_dir.name.replace("_nominal", "").replace("_nom", "")
    tv_rotors_range_section(nom_dir, pert_dir, dirs, rotor_stem)

    # overlay over time: needs the raw series (see load_series)
    print(f"\n{'='*78}\nRAW SERIES FOR THE CURVES\n{'='*78}")
    flights, all_ok = {}, True
    for cond, folder, pm in (('nominal', nom_dir, pm_nom),
                             ('perturbed', pert_dir, pm_pert)):
        for npz in sorted(folder.rglob("data.npz")):
            group, res = load_series(npz)
            if group is None:
                print(f"  [skip] {npz}: {res}")
                continue
            ok, msg = check_against_csv(res, pm, group)
            print(f"  {cond:<11}{msg}")
            if not ok:
                all_ok = False
            flights[(group, cond)] = res
    if not flights:
        print("  no readable data.npz: no overlay figures.")
    elif not all_ok:
        print("  !! the series recomputed here differ from the CSV files: the")
        print("     overlay figures are not drawn. Check that the CSV files were")
        print("     regenerated after the last change to the data.")
    else:
        # phase boundaries from the PPO nominal flight: the mission is the same
        # for all four, so they serve as a visual reference
        bounds = sorted({pm_nom[('PPO', p)]['t0'] for p in phases
                         if ('PPO', p) in pm_nom
                         and pm_nom[('PPO', p)]['t0'] == pm_nom[('PPO', p)]['t0']})[1:]
        missing = [key for key, _ in OVERLAY_PANELS
                  if not any(key in f for f in flights.values())]
        if missing:
            print(f"  [warning] missing in data.npz: {', '.join(missing)} -- "
                  "those panels stay empty")
        fig_overlay_grid(flights, dirs['roll'] / "overlay_grid.pdf", bounds=bounds)
        fig_step_profile_pair(flights, pm_nom, pm_pert, STEP_PROFILE_WINDOW_S,
                              dirs['roll'] / "step_profile_pair.pdf")
        mission_stem = nom_dir.name.replace("_nominal", "").replace("_nom", "")
        if mission_stem in MISSION_STEP_CHANNEL:
            fig_step_profile_single(
                flights, pm_nom, pm_pert, mission_stem, STEP_PROFILE_WINDOW_S,
                dirs['roll'] / "step_profile_single.pdf")
            step_metrics_range_table(
                nom_dir, pert_dir, mission_stem,
                dirs['tables'] / "step_metrics_range.tex")
        else:
            print(f"  [step profile] mission '{mission_stem}' not in "
                  "MISSION_STEP_CHANNEL, figure and table not written")

    # --------------------------------------------- export
    if rows:
        with open(dirs['tables'] / "comparison_phase.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({k: ('' if isinstance(v, float) and v != v else v)
                            for k, v in r.items()})
    if glob_rows:
        with open(dirs['tables'] / "comparison_global.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(glob_rows[0].keys()))
            w.writeheader()
            for r in glob_rows:
                w.writerow({k: ('' if isinstance(v, float) and v != v else v)
                            for k, v in r.items()})

    with open(dirs['tables'] / "results_comparison.tex", "w") as fh:
        fh.write(f"% {nom_dir.name} vs {pert_dir.name}\n")
        fh.write("% D = perturbed / nominal. D > 1 = degradation.\n")
        fh.write(f"% One flight per condition: differences below the {floor:.0f}% "
                 "threshold are not\n")
        fh.write("% discussed. The threshold is a convention, not "
                 "a statistical estimate.\n")
        for key, label, unit, scale, all_phases in PHASE_METRICS[:2]:
            cells = all_cells[key]
            usable = [p for p in phases
                      if cells[p]['ppo_nom'] == cells[p]['ppo_nom']
                      and cells[p]['sppo_pert'] == cells[p]['sppo_pert']]
            if not usable:
                continue
            fh.write(f"\n% --- {SHORT.get(key, key)} ---\n")
            fh.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
            fh.write("Phase & PPO nom & PPO pert & SPPO nom & SPPO pert & "
                     "$D_{\\mathrm{PPO}}$ & $D_{\\mathrm{SPPO}}$ \\\\\n\\midrule\n")
            for p in usable:
                c = cells[p]
                def t(v, d=4):
                    return '--' if v != v else f"{v:.{d}f}"
                fh.write(f"{p} & {t(c['ppo_nom'])} & {t(c['ppo_pert'])} & "
                         f"{t(c['sppo_nom'])} & {t(c['sppo_pert'])} & "
                         f"{t(c['D_ppo'], 2)} & {t(c['D_sppo'], 2)} \\\\\n")
            fh.write("\\bottomrule\n\\end{tabular}\n")

    # ---------------------------------------------- comparison by category
    stem = nom_dir.name.replace("_nominal", "").replace("_nom", "")
    compare_categories_section(nom_dir, pert_dir, dirs, stem, cat_exclude)

    print(f"\n  D is the ratio of two single flights. The {floor:.0f}% threshold")
    print(f"  is a convention, not an uncertainty estimate: the consistency")
    print( "  of the sign over several phases and missions carries more weight")
    print( "  than a single number.")
    print(f"\nComparison done. Output in: {out_dir}")


def default_output(nom_dir: Path, pert_dir: Path) -> Path:
    """Default output folder, e.g. ``p2p_comparison`` for ``p2p_nominal`` and ``p2p_perturbed``.

    The folder is next to the two inputs.

    Args:
        nom_dir (pathlib.Path): Nominal mission folder.
        pert_dir (pathlib.Path): Perturbed mission folder.

    Returns:
        pathlib.Path: Output folder.
    """
    a, b = nom_dir.name, pert_dir.name
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    prefix = a[:i].rstrip('_- ') or f"{a}_vs_{b}"
    return nom_dir.parent / f"{prefix}_comparison"


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Nominal vs perturbed comparison for one Tilthex mission.")
    p.add_argument("-n", "--nominal", required=True, type=str,
                   help="nominal folder (e.g. p2p_nominal)")
    p.add_argument("-p", "--perturbed", required=True, type=str,
                   help="perturbed folder")
    p.add_argument("-o", "--output", type=str, default=None,
                   help="output folder (default: <prefix>_comparison "
                        "next to the two input folders)")
    p.add_argument("--cat-exclude", nargs="*", default=[], metavar="CAT",
                   help="categories left out of the figures of "
                        "categories/ (kept in the CSV)")
    p.add_argument("--noise-floor", type=float, default=5.0,
                   help="percentage threshold below which no winner "
                        "is assigned (default 5)")
    args = p.parse_args()

    nom_dir, pert_dir = Path(args.nominal), Path(args.perturbed)
    for d in (nom_dir, pert_dir):
        if not d.is_dir():
            sys.exit(f"Error: folder '{d}' not found.")
    out_dir = Path(args.output) if args.output else default_output(nom_dir, pert_dir)
    run(nom_dir, pert_dir, out_dir, args.noise_floor,
        cat_exclude=args.cat_exclude)