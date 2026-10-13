#!/usr/bin/env python3
"""
Analysis of the trajectory missions: PPO vs SPPO.

The script writes a data_analysis folder with two branches and the tables::

    flight/      the whole log: error curves over time, rollouts, TV. It
                 describes the flight, including the takeoff from the
                 ground pose, which is different in every flight.
    maneuvers/   the flight split by manoeuvre: takeoff, hold, landing, and
                 maneuvers/step/ for the step response. The comparison is
                 made here: every number refers to a window identified from
                 the reference.
    tables/      the CSV files, including params.csv with the cuts used.

It applies to the folders with two flights, PPO and SPPO of the same policy
index, tracking the same reference::

    p2p_nominal   p2p_perturbed   rpy_nominal   rpy_perturbed

Nature of the data
------------------
One flight per cell: 2 missions x 2 algorithms x 2 conditions = 8 flights.
These are case studies. The phases of a mission come from the same flight
and are not independent samples, so no statistical test, p-value or
uncertainty band is computed here: only single values.

Differences with the hovering campaign
--------------------------------------
  * One flight per algorithm: no averages over policies, no +/- sigma bands,
    no p-values.
  * The reference changes in steps at every phase of the mission, so the
    flight is split into phases. The boundaries are detected from the
    reference; the staging windows between publish_absolute_setpoint and
    /policy/start are merged into the takeoff or the landing.
  * Besides the RMSE over the whole flight, the metrics are computed per
    phase, separating transient and steady state.

Phases with different steps
---------------------------
The vehicle cannot be put down at exactly the same point before every
flight, so the step into the first setpoint of the mission has a slightly
different amplitude for PPO and SPPO (1.35 vs 1.53 m in the data). A longer
step gives a larger error peak regardless of the controller, so the
transient of that phase cannot be compared in absolute terms.

The script compares the amplitude of the command entering each phase in the
two flights, and for the phases where it differs:

  * the absolute RMSE over the phase, transient RMSE, peak, settling time and
    phase TV are left blank (n.c.);
  * the steady-state error is kept, since it depends on the final setpoint,
    which is the same;
  * that transient window is excluded from the global metrics over the
    whole flight.

The curves are still drawn in full.

Normalised transient
--------------------
The transient is also reported divided by the step amplitude. For a linear
system the step response scales with the amplitude, so e(t)/Delta does not
depend on Delta, as for the overshoot in percent or the rise and settling
times. Next to the absolute values the script computes peak / Delta,
transient RMSE / Delta and IAE / Delta, which remain comparable on the phases
flagged above.

The invariance holds in the linear regime: with saturated actuators a longer
step degrades more also in relative terms. Here saturation is marginal and
the steps differ little, and the steady-state values, which need no such
assumption, are reported next to the normalised ones.

The normalisation applies only where the transient is a step response, that
is to the hold phases reached through a jump of the reference. Takeoff and
landing are ramps, with an entering jump close to zero.

Data format (as in analyze_campaign.py)
---------------------------------------
  * data['action'] holds the 6 rotor forces in newtons, one per rotor.
  * data['quat'] and data['quat_ref'] are in WXYZ order.

Error definitions, the same as in the hovering campaign::

  e_p = pos - pos_ref                 |e_p| independent of the frame
  e_b = R_ref^-1 (pos - pos_ref)      components in the reference frame
  e_R = rotvec(R_ref^-1 R_act)        log map in the reference frame;
                                      |e_R| is the geodesic distance

The world-frame components are not used: the reference yaw changes between
phases, so e_x, e_y, e_z would not be comparable axis by axis.

e_b is the error in the REFERENCE frame. Where the reference rotates during
the mission (in the attitude mission the commanded yaw goes from 0 to 40
degrees) the x and y axes of e_b rotate with it, and the global metric per
axis sums components along different physical directions. The comparison
between the two flights holds (same frame for both); across missions the
norm ``|e_p|`` is used.

Usage::

    python analyze_trajectory.py --input p2p_nominal
    python analyze_trajectory.py --input rpy_perturbed --t-ss 2.5
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  (registers the 3d projection)
from scipy.spatial.transform import Rotation

# --------------------------------------------------------------- constants
CF = 12.0e-4                   # thrust coefficient [N / (rad/s)^2]
F_MIN, F_MAX = 0.3072, 12.0    # force limits per rotor [N]
# The mission logs reach exactly 0.3072 and 12.0000 N: these are the
# actuator limits of these flights.
N_ROTORS = 6

C_PPO, C_SPPO = '#ff7f0e', '#1f77b4'  # orange / blue
C_REF = '#222222'
COL = {'PPO': C_PPO, 'SPPO': C_SPPO}
# Offset dashes for the two reference curves: the references coincide
# from SETPOINT 1 on, and with the same dash pattern one would hide
# the other.
DASH_REF = [(0, (7, 4)), (4, (7, 4))]
GROUPS = ('PPO', 'SPPO')

RAD2DEG = 180.0 / math.pi
_ESC_SPACE = "\\ "


def tex(s):
    r"""Escape the spaces of a label for matplotlib mathtext.

    The escape is done outside the f-strings, which accept a backslash only
    from Python 3.12.

    Args:
        s (str): Label, e.g. ``'Position Error X'``.

    Returns:
        str: Label with escaped spaces, e.g. ``'Position\ Error\ X'``.
    """
    return s.replace(" ", _ESC_SPACE)


def nan_to_blank(v):
    """Turn ``nan`` into an empty CSV cell, which reads as "not computable".

    Args:
        v (object): Cell value.

    Returns:
        object: ``''`` for ``nan``, otherwise ``v`` unchanged.
    """
    return '' if isinstance(v, float) and v != v else v


# ===========================================================================
# Metrics plotted over time.
# ===========================================================================
# key, symbol (without norm and unit), unit, title, file name
METRICS = [
    ('e_bx',       r'e_x^{\,r}',   'm',     'Position Error X (ref frame)', '1_err_bx'),
    ('e_by',       r'e_y^{\,r}',   'm',     'Position Error Y (ref frame)', '2_err_by'),
    ('e_bz',       r'e_z^{\,r}',   'm',     'Position Error Z (ref frame)', '3_err_bz'),
    ('e_roll',     r'e_\phi',      'rad',   'Attitude Error Roll',          '4_err_roll'),
    ('e_pitch',    r'e_\theta',    'rad',   'Attitude Error Pitch',         '5_err_pitch'),
    ('e_yaw',      r'e_\psi',      'rad',   'Attitude Error Yaw',           '6_err_yaw'),
    ('e_pos_norm', r'\|e_p\|',     'm',     'Total Position Error',         '7_err_pos'),
    ('e_att_norm', r'\|e_R\|',     'rad',   'Total Attitude Error',         '8_err_att'),
    ('v_norm',     r'\|v\|',       'm/s',   'Linear Velocity Norm',         '9_vel'),
    ('w_norm',     r'\|\omega\|',  'rad/s', 'Angular Velocity Norm',        '10_avel'),
]

# metrics whose reference is zero only in steady state: during a transition
# the vehicle has to move, so a high value there is not an error
MOTION_METRICS = {'v_norm', 'w_norm'}

# long title -> symbol, for the x axis of the summaries
SHORT = {t: rf'${s}$' for _k, s, _u, t, _f in METRICS}

# fields of phase_metrics that depend on the transient in ABSOLUTE value:
# they are not comparable for a phase entered with steps of different
# amplitude. The normalised counterparts (suffix _norm) stay valid and
# are not in this list.
TRANSIENT_FIELDS = ('rmse_pos_all', 'rmse_pos_transient', 'peak_pos',
                    'iae_pos', 'settle_pos_s', 'rmse_att_all',
                    'rmse_att_transient', 'peak_att', 'iae_att',
                    'settle_att_s', 'tv_total')

# minimum step amplitude for the normalisation
MIN_STEP_NORM = 0.05         # [m]


# ===========================================================================
# DETECTION OF THE MISSION PHASES
#
# The missions are defined in policy_control_node_tilthex.py:
#
#   point-to-point  (--takeoff-zero-point-land)
#       PHASE 1  takeoff (ramp)                                wait_s
#       PHASE 2  first point  [0, 0, takeoff_z]                wait_s
#       PHASE 3  second point [x, y, takeoff_z + z]            wait_s
#       PHASE 4  landing (ramp)                                wait_s
#
#   attitude manoeuvre (--takeoff-zero-attitude-zero-land)
#       PHASE 1  takeoff (ramp)                                wait_s
#       PHASE 2  zero point, level attitude                    wait_s
#       PHASE 3  target attitude [roll, pitch, yaw]            wait_s
#       PHASE 4  zero point with the target yaw                wait_s
#       PHASE 5  landing (ramp)                                wait_s
#
# The boundaries are read from the reference. Two kinds of events:
#   * STEP: publish_absolute_setpoint changes the setpoint from one
#     sample to the next. It lasts one sample, so it is detected on the
#     increment between consecutive samples.
#   * RAMP: takeoff and landing use set_ramp_rate, so the reference
#     changes continuously for a few seconds.
# ===========================================================================

V_THR = 1e-3                 # [m/s]   above this the reference is "moving"
W_THR = 1e-3                 # [rad/s]
MIN_BLOCK = 0.25             # [s]     shorter ramps are merged into the neighbour
STEP_POS = 0.02              # [m]     jump within a single sample
STEP_ANG = np.deg2rad(2.0)   # [rad]

# thresholds above which the command entering a phase is considered
# DIFFERENT between the two flights, so that the transient is not comparable
STEP_TOL_POS = 0.02          # [m]
STEP_TOL_ANG = 2.0           # [deg]


def to_scipy_quat(q_wxyz):
    """Reorder quaternions from the log order (w, x, y, z) to the SciPy order (x, y, z, w).

    Args:
        q_wxyz (numpy.ndarray): Quaternions, shape ``(N, 4)``.

    Returns:
        numpy.ndarray: Quaternions (x, y, z, w), shape ``(N, 4)``.
    """
    q = np.asarray(q_wxyz, dtype=float)
    return np.column_stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]])


def _blocks(mask, t, min_dur):
    """Split a boolean mask into contiguous blocks.

    Blocks shorter than ``min_dur`` are merged into the previous one, and
    adjacent blocks with the same value are joined.

    Args:
        mask (numpy.ndarray): Boolean mask over the samples.
        t (numpy.ndarray): Time stamps [s].
        min_dur (float): Minimum block duration [s].

    Returns:
        list[tuple[int, int, bool]]: ``(start, end, value)`` of each block, with ``end`` exclusive.
    """
    idx = np.flatnonzero(np.diff(mask.astype(int))) + 1
    edges = np.concatenate([[0], idx, [len(mask)]])
    raw = [(edges[i], edges[i + 1], bool(mask[edges[i]])) for i in range(len(edges) - 1)]
    merged = []
    for a, b, v in raw:
        dur = t[min(b, len(t) - 1)] - t[a]
        if merged and dur < min_dur:
            merged[-1] = (merged[-1][0], b, merged[-1][2])
        else:
            merged.append((a, b, v))
    out = []
    for a, b, v in merged:
        if out and out[-1][2] == v:
            out[-1] = (out[-1][0], b, v)
        else:
            out.append((a, b, v))
    return out


def detect_phases(f, cfg):
    """Detect the phases of a mission from the reference.

    Args:
        f (dict): Flight from ``process_log``.
        cfg (dict): Analysis settings (thresholds ``v_thr``, ``w_thr``, ``min_block``, ``step_pos``, ``step_ang``).

    Returns:
        list[dict]: One dict per phase with type (``'hold'`` or ``'move'``),
        label, sample indices, times, duration, target and, for the ramps, the
        commanded displacement.
    """
    t = f['t']
    dt = np.diff(t)
    R = f['R_ref']

    d_pos = np.linalg.norm(np.diff(f['pos_ref'], axis=0), axis=1)   # [m] per sample
    d_ang = (R[:-1].inv() * R[1:]).magnitude()                      # [rad] per sample
    v_ref, w_ref = d_pos / dt, d_ang / dt

    steps = np.flatnonzero((d_pos > cfg['step_pos']) | (d_ang > cfg['step_ang'])) + 1

    moving = (v_ref > cfg['v_thr']) | (w_ref > cfg['w_thr'])
    moving = np.concatenate([[moving[0]], moving])
    moving[steps] = False                       # a step is not a ramp
    ramp_edges = [b[0] for b in _blocks(moving, t, cfg['min_block'])]

    bounds = sorted({0, len(t)} | set(int(i) for i in steps) | set(ramp_edges))
    phases = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        # the segment ends at the sample BEFORE the next boundary: the boundary
        # index already carries the new setpoint, and including it would make a
        # hold phase look like a moving one
        e_ = min(b - 1, len(t) - 1)
        if e_ <= a:
            continue
        sl = slice(a, e_ + 1)
        is_move = bool(np.mean(v_ref[a:e_]) > cfg['v_thr'] or
                       np.mean(w_ref[a:e_]) > cfg['w_thr']) if e_ > a else False
        ph = {'type': 'move' if is_move else 'hold', 'i0': a, 'i1': e_,
              't0': float(t[a]), 't1': float(t[e_]), 'dur': float(t[e_] - t[a])}
        if is_move:
            ph['d_pos'] = f['pos_ref'][e_] - f['pos_ref'][a]
            # target at the end of the phase, used by the metrics, which treat
            # holds and ramps in the same way
            ph['pos'] = f['pos_ref'][e_].copy()
            ph['rpy_deg'] = np.rad2deg(f['rpy_ref'][e_])
        else:
            ph['pos'] = f['pos_ref'][sl].mean(axis=0)
            ph['rpy_deg'] = np.rad2deg(f['rpy_ref'][sl].mean(axis=0))
        phases.append(ph)
    phases = [p for p in phases if p['dur'] >= 0.2] or phases

    moves = [i for i, p in enumerate(phases) if p['type'] == 'move']
    if moves:
        i0 = moves[0]
        if phases[i0]['d_pos'][2] > 0.05 and i0 <= 1:
            phases[i0]['label'] = 'TAKEOFF'
        i1 = moves[-1]
        if phases[i1]['d_pos'][2] < -0.05 and i1 >= len(phases) - 2:
            phases[i1]['label'] = 'LANDING'

    # Very short holds next to takeoff or landing are the staging window of
    # the node (0.5 + 0.2 s between publish_absolute_setpoint and
    # /policy/start), where the reference is still above the ground pose.
    # They are merged into the takeoff or the landing, so that the phases
    # are exactly those of the mission: takeoff, the setpoints, landing.
    hold_durs = [p['dur'] for p in phases if p['type'] == 'hold']
    if hold_durs:
        typ = float(np.median(hold_durs))
        absorb = []
        for i, p in enumerate(phases):
            if p['type'] != 'hold' or p['dur'] >= 0.5 * typ:
                continue
            prev_lab = phases[i - 1].get('label') if i > 0 else None
            next_lab = phases[i + 1].get('label') if i + 1 < len(phases) else None
            if prev_lab in ('TAKEOFF', 'LANDING'):
                absorb.append((i, i - 1))       # merge into the previous ramp
            elif next_lab in ('TAKEOFF', 'LANDING'):
                absorb.append((i, i + 1))       # merge into the next ramp
        for idx, host in sorted(absorb, reverse=True):
            h, q = phases[host], phases[idx]
            a = min(h['i0'], q['i0'])
            b = max(h['i1'], q['i1'])
            h['i0'], h['i1'] = a, b
            h['t0'], h['t1'] = float(t[a]), float(t[b])
            h['dur'] = float(t[b] - t[a])
            h['d_pos'] = f['pos_ref'][b] - f['pos_ref'][a]
            h['pos'] = f['pos_ref'][b].copy()
            h['rpy_deg'] = np.rad2deg(f['rpy_ref'][b])
            phases.pop(idx)

    n_sp = 0
    for p in phases:
        if p['type'] == 'hold' and 'label' not in p:
            n_sp += 1
            p['label'] = f'SETPOINT {n_sp}'
        elif 'label' not in p:
            p['label'] = 'transition'
    return phases


def setpoint_phases(phases):
    """Only the commanded setpoints, without takeoff and landing.

    Args:
        phases (list[dict]): Phases from ``detect_phases``.

    Returns:
        list[dict]: Phases labelled ``SETPOINT``.
    """
    return [p for p in phases if p['label'].startswith('SETPOINT')]


def analysed_phases(phases):
    """All the phases where the metrics are computed: takeoff, the setpoints,
    landing. The policy also flies the takeoff and the landing, which are the
    most demanding phases. Unclassified fragments are left out.

    Args:
        phases (list[dict]): Phases from ``detect_phases``.

    Returns:
        list[dict]: Takeoff, setpoint and landing phases, in time order.
    """
    return [p for p in phases
            if p['label'] in ('TAKEOFF', 'LANDING')
            or p['label'].startswith('SETPOINT')]


def step_into(f, sp, j):
    """Amplitude of the command entering a phase.

    For the first phase, a ramp from the ground, it is the displacement
    commanded by the ramp.

    Args:
        f (dict): Flight from ``process_log``.
        sp (list[dict]): Analysed phases of the flight.
        j (int): Index of the phase in ``sp``.

    Returns:
        tuple[float, float]: Displacement [m] and rotation [deg]; ``nan`` where not defined.
    """
    ph = sp[j]
    if j == 0:
        if ph['type'] != 'move':
            return float('nan'), float('nan')
        return float(np.linalg.norm(ph['d_pos'])), float('nan')
    prev = sp[j - 1]
    dp = float(np.linalg.norm(f['pos_ref'][ph['i0']] - f['pos_ref'][prev['i1']]))
    da = float(np.rad2deg(
        (f['R_ref'][prev['i1']].inv() * f['R_ref'][ph['i0']]).magnitude()))
    return dp, da


def flag_comparability(flights, cfg):
    """Flag each phase as comparable or not in the TRANSIENT.

    When the command entering the phase has a different amplitude in the two
    flights (beyond the tolerances), the transient is not comparable: a longer
    step gives a larger peak. The steady state depends only on the final
    setpoint, which is the same, and stays valid.

    The flags come from the data, flight by flight and folder by folder.

    Sets the key ``comparable`` of every phase of every flight.

    Args:
        flights (list[dict]): Flights from ``process_log``, with their phases.
        cfg (dict): Analysis settings (``step_tol_pos`` [m], ``step_tol_ang`` [deg]).

    Returns:
        tuple[dict, list[dict]]: The phases with different steps, keyed by label
        with the displacements, rotations and worst differences, and the step of
        each phase in each flight.
    """
    per = []
    for f in flights:
        sp = analysed_phases(f['phases'])
        per.append({p['label']: step_into(f, sp, j) for j, p in enumerate(sp)})

    bad = {}
    if len(per) >= 2:
        common = [lab for lab in per[0] if all(lab in d for d in per)]
        for lab in common:
            dps = [d[lab][0] for d in per if d[lab][0] == d[lab][0]]
            das = [d[lab][1] for d in per if d[lab][1] == d[lab][1]]
            worst_p = (max(dps) - min(dps)) if len(dps) == len(per) else 0.0
            worst_a = (max(das) - min(das)) if len(das) == len(per) else 0.0
            if worst_p > cfg['step_tol_pos'] or worst_a > cfg['step_tol_ang']:
                bad[lab] = (dps, das, worst_p, worst_a)

    for f in flights:
        for p in f['phases']:
            p['comparable'] = p.get('label') not in bad
    return bad, per


# ===========================================================================
def process_log(npz_path: Path, cfg):
    """Load one flight log and compute the tracking errors.

    The position error is expressed in the reference frame, the attitude error
    as the rotation vector of ``R_ref^T R``.

    Args:
        npz_path (pathlib.Path): ``data.npz`` of the flight.
        cfg (dict): Analysis settings (not used).

    Returns:
        tuple[dict | None, str | None]: ``(flight, None)``, or ``(None, reason)``
        when the file is discarded. Every discarded file is listed in the manifest.
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        return None, f"load_error: {e}"

    keys = set(data.files)
    need = {'t', 'pos', 'pos_ref', 'quat', 'quat_ref'}
    if not need <= keys:
        return None, f"missing {sorted(need - keys)}"

    t = np.asarray(data['t'], dtype=float)
    if t.size < 3:
        return None, "time series too short"
    t = t - t[0]
    if np.any(np.diff(t) <= 0):
        return None, "time stamps not increasing"

    folder = npz_path.parent.name
    pname = str(data['policy_name']) if 'policy_name' in keys else ""
    tag = (folder + "_" + pname).upper()
    if "SPPO" in tag:
        group = "SPPO"
    elif "PPO" in tag:
        group = "PPO"
    else:
        return None, "unknown group (neither PPO nor SPPO)"

    pos = np.asarray(data['pos'], float)
    pos_ref = np.asarray(data['pos_ref'], float)
    R_act = Rotation.from_quat(to_scipy_quat(data['quat']))
    R_ref = Rotation.from_quat(to_scipy_quat(data['quat_ref']))

    e_w = pos - pos_ref
    e_b = R_ref.inv().apply(e_w)
    rv = (R_ref.inv() * R_act).as_rotvec()
    dts = np.diff(t)

    res = {
        'file': str(npz_path), 'run': folder, 'group': group, 'policy_name': pname,
        't': t, 'duration': float(t[-1]), 'n_samples': int(t.size),
        'dt_mean': float(dts.mean()),
        'pos': pos, 'pos_ref': pos_ref,
        'rpy': R_act.as_euler('xyz'), 'rpy_ref': R_ref.as_euler('xyz'),
        'R_ref': R_ref, 'R_act': R_act, 'warnings': [],
        'e_bx': e_b[:, 0], 'e_by': e_b[:, 1], 'e_bz': e_b[:, 2],
        'e_pos_norm': np.linalg.norm(e_w, axis=1),
        'e_roll': rv[:, 0], 'e_pitch': rv[:, 1], 'e_yaw': rv[:, 2],
        'e_att_norm': np.linalg.norm(rv, axis=1),
    }

    if 'vel' in keys:
        res['v_norm'] = np.linalg.norm(np.asarray(data['vel'], float), axis=1)
    else:
        res['warnings'].append("missing 'vel'")
    if 'avel' in keys:
        res['w_norm'] = np.linalg.norm(np.asarray(data['avel'], float), axis=1)
    else:
        res['warnings'].append("missing 'avel'")
    if 'inf_time' in keys:
        it = np.asarray(data['inf_time'], float)
        res['inf_time_mean_ms'] = float(np.mean(it) * 1e3)
        res['inf_time_max_ms'] = float(np.max(it) * 1e3)

    if 'action' in keys:
        a = np.atleast_2d(np.asarray(data['action'], float))
        if a.shape[0] != t.size:
            res['warnings'].append(f"action {a.shape} not aligned with t ({t.size})")
        elif a.shape[1] < N_ROTORS:
            res['warnings'].append(f"action has {a.shape[1]} channels, {N_ROTORS} expected")
        else:
            # raw forces: the clip is applied only inside the square root
            res['forces'] = a[:, :N_ROTORS]
            res['sat_frac'] = float(np.mean((a[:, :N_ROTORS] <= F_MIN * 1.001) |
                                            (a[:, :N_ROTORS] >= F_MAX * 0.999)))
    else:
        res['warnings'].append("missing 'action': no total variation")
    return res, None


def compute_tv(f, mask):
    """Total variation of the rotor speed over the selected samples, divided by
    the time span. The mask may have gaps (the windows excluded from the
    comparison): the variation is summed within each contiguous stretch, never
    across a gap, where the jump between the two edges would be artificial.

    Args:
        f (dict): Flight from ``process_log``.
        mask (numpy.ndarray): Boolean mask of the selected samples.

    Returns:
        float | None: Total variation rate [rad/s^2], or ``None`` without rotor data or with fewer than two samples.
    """
    if 'forces' not in f:
        return None
    idx = np.flatnonzero(mask)
    if idx.size < 2:
        return None
    w = np.sqrt(np.clip(f['forces'], F_MIN, F_MAX) / CF)           # [rad/s]
    tv, dur = np.zeros(N_ROTORS), 0.0
    for blk in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        if blk.size < 2:
            continue
        tv += np.sum(np.abs(np.diff(w[blk], axis=0)), axis=0)
        dur += f['t'][blk[-1]] - f['t'][blk[0]]
    return tv / dur if dur > 0 else None


def rms(x):
    """RMS of a set of values without time: used to aggregate scalars (e.g. a
    bias per sub-block), not time series.

    Args:
        x (array_like): Values.

    Returns:
        float: Root mean square, ``nan`` for an empty input.
    """
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x ** 2))) if x.size else float('nan')


# numpy >= 2.0 calls it trapezoid, older versions trapz
_TRAPZ = getattr(np, 'trapezoid', None) or np.trapz


def seg_mean_t(t, y, mask=None):
    """TIME AVERAGE of y: trapezoidal integral divided by the duration.

    The logging period is not exactly constant nor the same in every flight
    (9.998-10.032 ms here), so a mean over the samples would weight the
    densely sampled stretches more and compare two flights with different
    weights. The time integral divided by the duration does not depend on how
    the samples are spread, and is the usual definition of the control
    figures of merit (IAE, ISE).

    The mask may have gaps (the windows excluded from the comparison): the
    integral is taken within each contiguous stretch and numerators and
    durations are summed, never across a gap, as in compute_tv().

    Args:
        t (numpy.ndarray): Time stamps [s].
        y (numpy.ndarray): Signal.
        mask (numpy.ndarray | None): Boolean mask of the samples; ``None`` uses all of them.

    Returns:
        float: Time average of ``y``, ``nan`` for an empty mask.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    idx = np.arange(t.size) if mask is None else np.flatnonzero(mask)
    if idx.size == 0:
        return float('nan')
    num = den = 0.0
    for blk in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        if blk.size < 2:
            continue
        num += float(_TRAPZ(y[blk], t[blk]))
        den += float(t[blk[-1]] - t[blk[0]])
    if den <= 0:                      # degenerate window: plain mean
        return float(np.mean(y[idx]))
    return num / den


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
    return float(np.sqrt(v)) if v == v and v >= 0 else float('nan')


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


def settling_time(t, e, t0, tol, smooth_s=0.25):
    """Settling time: first instant after t0 after which the deviation `e`
    stays below `tol`.

    The caller passes the deviation from the FINAL value. The policy has no
    integral action and settles on a nonzero steady-state error, so a band
    centred on zero might never be reached; the band centred on the final
    value is also the convention of classical control (stepinfo). The error
    is first smoothed with a moving average over smooth_s, so that a single
    noise spike at the end of the phase does not set the settling time to
    the whole phase.

    Args:
        t (numpy.ndarray): Time stamps of the phase [s].
        e (numpy.ndarray): Deviation from the final value.
        t0 (float): Start of the measurement [s].
        tol (float): Half-width of the band; ``nan`` when the channel has no step.
        smooth_s (float): Width of the moving average [s].

    Returns:
        float: Settling time from ``t0`` [s], ``nan`` when not reached or not defined.
    """
    if len(t) < 3 or tol != tol:       # nan band: no step in this
        return float('nan')              # channel, nothing to
    dt = float(np.median(np.diff(t)))    # measure
    k = max(1, int(round(smooth_s / max(dt, 1e-6))))
    if k > 1:
        es = np.convolve(e, np.ones(k) / k, mode='same')
        es[:k], es[-k:] = e[:k], e[-k:]        # the edges stay raw
    else:
        es = e
    below = es <= tol
    if not below.any():
        return float('nan')
    above = np.flatnonzero(~below)
    if above.size == 0:
        return 0.0
    idx = above[-1] + 1
    return float(t[idx] - t0) if idx < len(t) else float('nan')


def phase_metrics(f, sp, j, cfg):
    """Metrics of a single phase. When the transient is not comparable (step of
    different amplitude in the two flights) every field that depends on it is
    nan, written as an empty cell.

    Args:
        f (dict): Flight from ``process_log``.
        sp (list[dict]): Analysed phases of the flight.
        j (int): Index of the phase in ``sp``.
        cfg (dict): Analysis settings.

    Returns:
        dict: Metrics of the phase, one key per column of the phase table.
    """
    ph = sp[j]
    i0, i1 = ph['i0'], ph['i1']
    sl = slice(i0, i1 + 1)
    t = f['t'][sl]
    ep, ea = f['e_pos_norm'][sl], f['e_att_norm'][sl]

    ss = t >= (t[-1] - cfg['t_ss'])            # steady state: last t_ss seconds
    tr = ~ss                                   # transient: the rest

    d_pos, d_ang = step_into(f, sp, j)
    ph_mask = np.zeros(f['n_samples'], dtype=bool)
    ph_mask[i0:i1 + 1] = True
    tv = compute_tv(f, ph_mask)

    # SETTLING BAND, centred on the final value of the phase.
    #
    # The settling time is defined only when THAT channel has a step: in a
    # pure rotation the position reference does not move, and the settling
    # time of the position would come out as 0 s. Same conditions as the
    # normalisation below: a hold phase reached through a jump of the
    # reference above the minimum threshold.
    #
    # Band width: a fraction of the step, as in classical control. A
    # threshold tied to the noise of each flight would differ between the
    # two flights. With a comparable step the band is the same in both
    # flights. When the band is narrower than the noise the error never
    # stays inside and the time is nan.
    e_ss_pos, e_ss_att = float(np.mean(ep[ss])), float(np.mean(ea[ss]))
    nan_ = float('nan')
    is_hold = ph['type'] == 'hold'
    band_pos = (cfg['settle_band'] * d_pos
                if is_hold and d_pos == d_pos and d_pos > cfg['min_step_norm']
                else nan_)
    band_att = (cfg['settle_band'] * math.radians(d_ang)
                if is_hold and d_ang == d_ang
                and d_ang > cfg['settle_min_ang_deg']
                else nan_)

    m = {
        'run': f['run'], 'group': f['group'], 'phase': ph['label'],
        # 6 decimals: compare_trajectory.py opens the mission window at this
        # t0 read from the CSV, and at 100 Hz a rounding to 10 ms would move
        # the window by one sample.
        't0': round(ph['t0'], 6), 't1': round(ph['t1'], 6), 'dur': round(ph['dur'], 2),
        'comparable_transient': int(bool(ph.get('comparable', True))),
        'step_pos_m': round(d_pos, 4), 'step_ang_deg': round(d_ang, 2),
        # VERTICAL displacement commanded by the ramp, for takeoff and
        # landing. step_pos_m measures the jump ENTERING the phase (about 0
        # for a ramp), while here the length of the ramp is needed. The
        # vehicle does not start from the same height in every flight, so
        # this length changes between flights, and compare_trajectory uses it
        # to decide whether the 'Land' category is comparable.
        'ramp_dz_m': (round(float(ph['d_pos'][2]), 4)
                      if ph.get('type') == 'move' else ''),
        'ref_x': round(float(ph['pos'][0]), 3), 'ref_y': round(float(ph['pos'][1]), 3),
        'ref_z': round(float(ph['pos'][2]), 3),
        'ref_roll_deg': round(float(ph['rpy_deg'][0]), 2),
        'ref_pitch_deg': round(float(ph['rpy_deg'][1]), 2),
        'ref_yaw_deg': round(float(ph['rpy_deg'][2]), 2),
        'rmse_pos_all': round(rms_t(t, ep), 5),
        'rmse_pos_transient': round(rms_t(t, ep, tr), 5),
        'rmse_pos_steady': round(rms_t(t, ep, ss), 5),
        'peak_pos': round(float(ep.max()), 5),
        # iae_*: mean absolute error over time, not the raw integral, so that
        # it stays in metres and is comparable across phases of different length
        'iae_pos': round(mae_t(t, ep), 5),
        'rmse_att_all': round(rms_t(t, ea), 5),
        'rmse_att_transient': round(rms_t(t, ea, tr), 5),
        'rmse_att_steady': round(rms_t(t, ea, ss), 5),
        'peak_att': round(float(ea.max()), 5),
        'iae_att': round(mae_t(t, ea), 5),
        'settle_pos_s': round(settling_time(t, np.abs(ep - e_ss_pos),
                                            t[0], band_pos), 3),
        'settle_att_s': round(settling_time(t, np.abs(ea - e_ss_att),
                                            t[0], band_att), 3),
        'settle_band_pos_m': round(band_pos, 4),
        'settle_band_att_rad': round(band_att, 5),
        'tv_total': round(float(np.sum(tv)), 2) if tv is not None else float('nan'),
    }
    # ||v|| and ||omega||: the velocity reference is zero only in steady state
    for k, name in (('v_norm', 'v_norm_steady'), ('w_norm', 'w_norm_steady')):
        if k in f:
            m[name] = round(rms_t(t, f[k][sl], ss), 5)

    # ---------------------------------------------------------------------
    # TRANSIENT NORMALISED by the step amplitude.
    # A longer step gives a larger error: dividing by the amplitude makes
    # the transient comparable between flights that started from different
    # ground positions (see the module docstring for the validity limits).
    # Only for the hold phases reached through a jump of the reference:
    # takeoff and landing are ramps, with an entering jump close to zero.
    # The attitude is normalised by the same position step: in a pure
    # translation the attitude error is coupling induced by the motion, so
    # the ratio is a gain [rad per commanded metre].
    # ---------------------------------------------------------------------
    nan = float('nan')
    norm_ok = (ph['type'] == 'hold' and d_pos == d_pos
               and d_pos > cfg['min_step_norm'])
    m['norm_base_m'] = round(d_pos, 4) if norm_ok else nan
    m['rmse_pos_all_norm'] = round(m['rmse_pos_all'] / d_pos, 5) if norm_ok else nan
    m['rmse_pos_transient_norm'] = (round(m['rmse_pos_transient'] / d_pos, 5)
                                    if norm_ok else nan)
    m['peak_pos_norm'] = round(m['peak_pos'] / d_pos, 5) if norm_ok else nan
    m['iae_pos_norm'] = round(m['iae_pos'] / d_pos, 5) if norm_ok else nan
    m['peak_att_norm'] = (round(m['peak_att'] / d_pos, 6)
                          if norm_ok else nan)
    m['rmse_att_transient_norm'] = (round(m['rmse_att_transient'] / d_pos, 6)
                                    if norm_ok else nan)

    # ---------------------------------------------------------------------
    # TRANSIENT NORMALISED by the ACTUAL DISTANCE TO COVER.
    #
    # At the instant of the step the vehicle is not exactly on the previous
    # setpoint but carries its steady-state error, which adds to or
    # subtracts from the distance to cover. On the second setpoint of p2p
    # the command is 0.7348 m for both flights, while the actual distance
    # is 0.6984 m for one and 0.7220 m for the other (3.4% more).
    #
    # Dividing by the actual distance removes this effect without cutting
    # samples or choosing thresholds, and holds for every step, including
    # those with a different commanded amplitude. For this reason these
    # columns are not in TRANSIENT_FIELDS and are kept for the phases that
    # are not comparable.
    #
    # The stepinfo metrics (overshoot, rise, settling) normalise the
    # progress on the actual range [p0, p_fin] in step_response(), where
    # p0 is this same offset.
    # ---------------------------------------------------------------------
    if j > 0:
        i_prev = sp[j - 1]['i1']
        d_real = float(np.linalg.norm(f['pos'][i_prev] - f['pos_ref'][i0]))
        a_real = float(np.linalg.norm(
            (f['R_ref'][i0].inv() * f['R_act'][i_prev]).as_rotvec()))
    else:
        d_real = a_real = nan
    # Conditioned on the COMMANDED step: in a pure rotation the position
    # command is zero and the actual distance is only the residual error,
    # which would give meaningless ratios.
    real_ok = (ph['type'] == 'hold' and d_real == d_real and d_real > 1e-6
               and d_pos == d_pos and d_pos > cfg['min_step_norm'])
    real_ok_a = (ph['type'] == 'hold' and a_real == a_real and a_real > 1e-6
                 and d_ang == d_ang
                 and d_ang > cfg['settle_min_ang_deg'])
    m['norm_real_m'] = round(d_real, 4) if real_ok else nan
    m['norm_real_deg'] = round(float(np.rad2deg(a_real)), 3) if real_ok_a else nan
    m['rmse_pos_transient_real'] = (round(m['rmse_pos_transient'] / d_real, 5)
                                    if real_ok else nan)
    # peak/d is 1 by construction (the peak of the error norm is the
    # initial distance), so the IAE is reported instead: it tells "reaches
    # and stays" from "drags".
    m['iae_pos_real'] = (round(m['iae_pos'] / d_real, 5) if real_ok else nan)
    m['rmse_att_transient_real'] = (round(m['rmse_att_transient'] / a_real, 5)
                                    if real_ok_a else nan)
    m['iae_att_real'] = (round(m['iae_att'] / a_real, 5) if real_ok_a else nan)

    # The ABSOLUTE transient values are dropped when the phase is not
    # comparable; the normalised ones, computed above, are kept.
    if not ph.get('comparable', True):
        for k in TRANSIENT_FIELDS:
            m[k] = nan
    return m


def _smooth(t, y, win_s):
    """Moving average over win_s seconds, edges left raw.

    Args:
        t (numpy.ndarray): Time stamps [s].
        y (numpy.ndarray): Signal.
        win_s (float): Window width [s].

    Returns:
        numpy.ndarray: Smoothed signal.
    """
    if win_s <= 0 or y.size < 3:
        return y
    dt = float(np.median(np.diff(t)))
    k = max(1, int(round(win_s / max(dt, 1e-6))))
    if k < 2:
        return y
    ys = np.convolve(y, np.ones(k) / k, mode='same')
    ys[:k], ys[-k:] = y[:k], y[-k:]
    return ys


def step_response(f, sp, j, cfg, kind, smooth_s=0.10):
    """Step response of a phase: overshoot, 10-90% rise time and settling time.

    The error is projected on the direction of the command. The overshoot is
    not visible on the norm ``||e||``, which starts from the step amplitude
    ``Delta`` and decreases, so that its maximum is ``Delta`` by construction.

    Along the command direction ``u`` the progress variable
    ``p(t) = (e(t) . u + Delta) / Delta`` is 0 at the instant of the step and 1
    on the target; ``p > 1`` means that the vehicle went past the target, which
    is the overshoot.

    Args:
        f (dict): Flight from ``process_log``.
        sp (list[dict]): Analysed phases of the flight.
        j (int): Index of the phase in ``sp``.
        cfg (dict): Analysis settings.
        kind (str): ``'pos'`` for a position step, ``'att'`` for a commanded rotation.
        smooth_s (float): Width of the moving average applied to the progress [s].

    Returns:
        dict | None: Step metrics and the progress profile (``t``, ``prog``),
        or ``None`` when the phase has no step in that channel.
    """
    ph = sp[j]
    if j == 0 or ph['type'] != 'hold':
        return None
    prev = sp[j - 1]
    i0, i1 = ph['i0'], ph['i1']
    # The window starts ONE SAMPLE BEFORE the phase, at the last sample
    # with the old reference: the instant of the step. The phase detection
    # may fall one or two samples after the jump, and on a fast rotation
    # (11 deg in 0.14 s, 14 samples) the progress would already be past 10%
    # at the first sample, giving t10 = 0.
    #
    # The target is the NEW reference held fixed: within a hold phase this
    # is the same as the sampled reference, and on the sample before the
    # jump it gives p = 0.
    i_s = max(prev['i1'], 0)
    sl = slice(i_s, i1 + 1)
    t = f['t'][sl]

    if kind == 'pos':
        d = f['pos_ref'][i0] - f['pos_ref'][prev['i1']]
        amp = float(np.linalg.norm(d))
        if not (amp > cfg['step_pos']):
            return None
        e = f['pos'][sl] - f['pos_ref'][i0]
        sgn = e @ (d / amp)
        unit, amp_out = 'm', amp
    else:
        dr = (f['R_ref'][prev['i1']].inv() * f['R_ref'][i0]).as_rotvec()
        amp = float(np.linalg.norm(dr))
        if not (np.rad2deg(amp) > cfg['settle_min_ang_deg']):
            return None
        R_t = f['R_ref'][i0]
        rv = (R_t.inv() * f['R_act'][sl]).as_rotvec()
        sgn = rv @ (dr / amp)
        unit, amp_out = 'rad', amp

    prog = (sgn + amp) / amp                     # 0 at the start, 1 on the target
    ps = _smooth(t, prog, smooth_s)

    ss = t >= (t[-1] - cfg['t_ss'])
    p_fin = float(np.mean(prog[ss])) if ss.any() else float('nan')

    over = float(np.max(ps)) - 1.0                # >0 = went past the target

    # RISE TIME on the ACTUAL range [p0, p_fin].
    # The vehicle starts from the previous setpoint with a steady-state
    # error, which may cover part of the next step. In rpy SETPOINT 3 the
    # residual of 3.3 deg on a command of 11.2 deg is 30%: with the 10%
    # level on [0, 1] the threshold would fall inside the initial condition
    # and t10 would be zero. On [p0, p_fin] the 10-90% measures the rise,
    # as for a step response with a nonzero initial condition.
    p0 = float(np.mean(ps[:3])) if ps.size >= 3 else float(ps[0])
    span = p_fin - p0
    if abs(span) > 1e-9:
        pn = (ps - p0) / span
        def first_at(level):
            k = np.flatnonzero(pn >= level)
            return float(t[k[0]] - t[0]) if k.size else float('nan')
        t10, t90 = first_at(0.10), first_at(0.90)
        rise = t90 - t10 if (t10 == t10 and t90 == t90) else float('nan')
    else:
        t10 = t90 = rise = float('nan')

    # settling within 5% of Delta around the final value: on the progress
    # variable the band is cfg['settle_band']
    settle = settling_time(t, np.abs(prog - p_fin), t[0], cfg['settle_band'])

    return {'phase': ph['label'], 'group': f['group'], 'kind': kind,
            'amp': amp_out, 'unit': unit,
            'p0_frac': p0,
            'overshoot_pct': over * 100.0,
            'rise_10_90_s': rise, 't10_s': t10, 't90_s': t90,
            'settle_s': settle,
            'final_err_frac': p_fin - 1.0,
            'final_err_abs': (p_fin - 1.0) * amp,
            'comparable': int(bool(ph.get('comparable', True))),
            't': t - t[0], 'prog': prog, 'p_fin': p_fin}


def touchdown_metrics(f, sp, cfg, tail_s=0.20):
    """Error at the instant the landing ramp stops.

    After the reference stops descending the vehicle is on the ground, so the
    end of the log is not used. Mean over the last tail_s of the ramp.

    Args:
        f (dict): Flight from ``process_log``.
        sp (list[dict]): Analysed phases of the flight (not used).
        cfg (dict): Analysis settings (not used).
        tail_s (float): Averaging window before the end of the ramp [s].

    Returns:
        dict | None: Horizontal, vertical and total position error [m] at
        touchdown, or ``None`` without a landing phase.
    """
    lan = [p_ for p_ in f['phases'] if p_['label'] == 'LANDING']
    if not lan:
        return None
    ph = lan[0]
    i0, i1 = ph['i0'], ph['i1']
    t = f['t']
    dref = np.concatenate([[0.0],
                           np.linalg.norm(np.diff(f['pos_ref'], axis=0), axis=1)])
    idx = np.flatnonzero((np.arange(t.size) >= i0) & (np.arange(t.size) <= i1)
                         & (dref > REF_MOVE_TOL))
    if idx.size < 2:
        return None
    t_td = float(t[idx[-1]])
    m = (t >= t_td - tail_s) & (t <= t_td)
    e = f['pos'][m] - f['pos_ref'][m]
    return {'group': f['group'], 't_touchdown_s': t_td,
            'err_xy_m': float(np.mean(np.linalg.norm(e[:, :2], axis=1))),
            'err_z_m': float(np.mean(e[:, 2])),
            'err_pos_m': float(np.mean(np.linalg.norm(e, axis=1))),
            'n_samples': int(m.sum())}


def coupling_metrics(f, sp, j, cfg):
    """Attitude -> position coupling in a phase of PURE rotation.

    The position reference does not move, so all the position error in the
    phase is induced by the commanded rotation. The position step is zero in
    both flights, so this transient metric is always comparable.

    Args:
        f (dict): Flight from ``process_log``.
        sp (list[dict]): Analysed phases of the flight.
        j (int): Index of the phase in ``sp``.
        cfg (dict): Analysis settings (``t_ss``).

    Returns:
        dict | None: Peak and steady-state position error [m] and the same
        values per degree of rotation [mm/deg], or ``None`` when the phase is
        not a pure rotation.
    """
    ph = sp[j]
    if j == 0 or ph['type'] != 'hold':
        return None
    prev = sp[j - 1]
    i0, i1 = ph['i0'], ph['i1']
    d_pos = float(np.linalg.norm(f['pos_ref'][i0] - f['pos_ref'][prev['i1']]))
    dr = (f['R_ref'][prev['i1']].inv() * f['R_ref'][i0]).as_rotvec()
    ang_deg = float(np.rad2deg(np.linalg.norm(dr)))
    if d_pos > cfg['step_pos'] or ang_deg <= cfg['settle_min_ang_deg']:
        return None
    sl = slice(i0, i1 + 1)
    t = f['t'][sl]
    ep = f['e_pos_norm'][sl]
    ss = t >= (t[-1] - cfg['t_ss'])
    return {'phase': ph['label'], 'group': f['group'], 'ang_deg': ang_deg,
            'peak_pos_m': float(np.max(ep)),
            'peak_per_deg_mm': float(np.max(ep)) / ang_deg * 1000.0,
            'steady_pos_m': rms_t(t, ep, ss),
            'steady_per_deg_mm': rms_t(t, ep, ss) / ang_deg * 1000.0}


# ===========================================================================
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
        "legend.fontsize": 15,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#cccccc",
        "figure.titlesize": 17,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.4,
        "grid.linestyle": "--",
        # minimal frame: only the axes that carry the numbers
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "savefig.bbox": "tight",
        # cmr10 has no bold weight: a bold title makes matplotlib print
        # "findfont: Failed to find font weight bold" for every figure
        "axes.titleweight": "normal",
    })


def save(fig, path, pad_inches=None):
    """Save a figure as a transparent PDF and close it.

    Args:
        fig (matplotlib.figure.Figure): Figure.
        path (pathlib.Path): Output file.
        pad_inches (float | None): Padding around the figure; ``None`` keeps the default.
    """
    kw = {} if pad_inches is None else {'pad_inches': pad_inches}
    fig.savefig(path, format='pdf', transparent=True, bbox_inches='tight', **kw)
    plt.close(fig)


def tight_x(ax, lo, hi):
    """No horizontal margin: the trace starts at the axis.

    The limits are passed explicitly: axhline and axvline have xdata in axes
    coordinates and would distort limits inferred from the drawn lines.

    Args:
        ax (matplotlib.axes.Axes): Axis.
        lo (float): Left limit.
        hi (float): Right limit.
    """
    if hi > lo:
        ax.set_xlim(lo, hi)
    ax.margins(x=0)


def mark_phases(ax, ref, lw=0.9):
    """Vertical lines at the phase boundaries, over the whole flight.

    Args:
        ax (matplotlib.axes.Axes): Axis.
        ref (dict): Flight whose phases are marked.
        lw (float): Line width.
    """
    for p in analysed_phases(ref['phases'])[1:]:
        ax.axvline(p['t0'], color='#999', lw=lw, ls=':')


# ===========================================================================
# CONTROL TASK CATEGORIES
# ===========================================================================
# Besides the mission phases, the flight is split into categories that
# describe the control problem the vehicle is solving at each instant:
#
#    Takeoff    vertical ramp at takeoff
#    Step       step response transients
#    Hover      hold at a fixed point (also with a nonzero attitude)
#    Land       vertical ramp at landing
#
# The rows are the same in every mission, so they can be compared.
#
# HOW THE BLOCKS ARE BUILT
#   Each detected phase is classified from the REFERENCE:
#     - moving reference (takeoff, landing) -> ramp, taken from the start
#       until the reference stops; the tail where the vehicle waits at the
#       top of the ramp or is already on the ground is left out;
#     - step of the reference -> the first DELTA seconds are transient,
#       the rest is hold at a fixed point (Hover).
#   The cut is anchored to the REFERENCE, not to the arrival of the vehicle.
#
# POOLING
#   The RMSE metrics pool the SAMPLES of the windows of the same category,
#   not the RMSE values. The vector bias is not pooled: the mean errors of
#   two holds at different points point in different directions.
#
# STEPS THAT ARE NOT COMPARABLE
#   In the 'Step' category a sub-block enters the ABSOLUTE metrics
#   (rmse_pos, rmse_att, err_xy, err_z, tv) only when its phase is flagged
#   comparable by flag_comparability, i.e. when the commanded amplitude is
#   the same in the two flights within tolerance. A longer step gives a
#   larger error, which does not depend on the controller. The NORMALISED
#   columns (peak_n, rmse_n, att_peak_n) use ALL the sub-blocks, since they
#   divide by the amplitude. Same rule as phase_metrics, where the fields
#   in TRANSIENT_FIELDS become nan.

REF_MOVE_TOL = 1e-3         # [m/sample] below this the reference is still
C_INK, C_MUTED = "#222222", "#9a9a9a"

CATEGORIES = ["Takeoff", "Step", "Hover", "Land"]
KIND = {"Takeoff": "ramp", "Land": "ramp", "Step": "step", "Hover": "hold"}

# Mission-specific phase names (Tables 5.5/5.6 of the thesis): the
# generic 'Hover' bucket is relabelled with the name of the phase
# (SETPOINT k) it corresponds to. 'Takeoff'/'Step'/'Land' stay generic.
HOLD_PHASES = {
    "p2p": {"SETPOINT 1": "Origin", "SETPOINT 2": "Point-to-point"},
    "rpy": {"SETPOINT 1": "Origin", "SETPOINT 2": "Tilt", "SETPOINT 3": "Level"},
}
CATEGORIES_BY_MISSION = {
    "p2p": ["Takeoff", "Step", "Origin", "Point-to-point", "Land"],
    "rpy": ["Takeoff", "Step", "Origin", "Tilt", "Level", "Land"],
}
for _mission_cats in CATEGORIES_BY_MISSION.values():
    for _cat in _mission_cats:
        KIND.setdefault(_cat, "hold")
del _mission_cats, _cat

CAT_UNITS = {"rmse_pos": "m", "rmse_att": "rad", "lag_z": "m", "err_xy": "m",
             "e_x": "m", "e_y": "m", "e_z": "m",
             "e_roll": "rad", "e_pitch": "rad", "e_yaw": "rad",
             "err_z": "m", "peak_n": "-", "rmse_n": "-", "t10_s": "s",
             "att_peak_n": "-", "step_amp_mean": "m", "e_mean": "m",
             "e_std": "m", "e_bias": "m", "att_mean": "rad", "att_yaw": "rad",
             "att_rp": "rad", "w_norm": "rad/s", "tv": "rad/s^2",
             "n_samples": "-", "n_subblocks": "-", "dur_s": "s",
             "rmse_n_real": "-", "iae_n_real": "-", "dist_real_mean": "m",
             "n_subblocks_real": "-",
             "rmse_n_real_att": "-", "iae_n_real_att": "-",
             "ang_real_mean": "rad", "n_subblocks_real_att": "-",
             "pos_pooled_n_RMSE": "-", "pos_pooled_n_MAE": "-",
             "pos_pooled_den": "m", "ramp_len_mean": "m",
             "att_pooled_n_RMSE": "-", "att_pooled_n_MAE": "-",
             "att_pooled_den": "rad"}


def cellf(s):
    """Convert a CSV cell to a float; a non-numeric cell gives ``nan``.

    Args:
        s (object): Cell value.

    Returns:
        float: Numeric value or ``nan``.
    """
    try:
        return float(s)
    except (TypeError, ValueError):
        return float('nan')


def cat_view(f):
    """Adapt a flight from ``process_log`` to the category section.

    No error is recomputed: the errors and their components are those of
    ``process_log``, so there is a single error definition in the file.

    Args:
        f (dict): Flight from ``process_log``.

    Returns:
        dict: Time, errors, velocities and rotor forces with the keys used by the category section.
    """
    e_w = f['pos'] - f['pos_ref']
    dref = np.linalg.norm(np.diff(f['pos_ref'], axis=0), axis=1)
    return {
        't': f['t'], 'dt': float(np.median(np.diff(f['t']))),
        'run': f['run'], 'group': f['group'],
        'pos': f['pos'], 'pos_ref': f['pos_ref'], 'e_w': e_w,
        'e_pos': f['e_pos_norm'], 'e_att': f['e_att_norm'],
        'e_yaw': np.abs(f['e_yaw']),
        'e_rp': np.hypot(f['e_roll'], f['e_pitch']),
        # single components, as in the RMSE/ and Rollouts/ folders: position
        # in the REFERENCE frame, attitude as a rotation vector
        'e_b': np.column_stack([f['e_bx'], f['e_by'], f['e_bz']]),
        'rv': np.column_stack([f['e_roll'], f['e_pitch'], f['e_yaw']]),
        'avel': f.get('w_norm'), 'v': f.get('v_norm'),
        'forces': f.get('forces'),
        'ref_moving': np.concatenate([[False], dref > REF_MOVE_TOL]),
    }


def tv_rotor(t_blk, forces):
    """Total variation of the rotor speed over a contiguous window, summed over
    the six rotors.

    The duration is the ELAPSED TIME of the window, t[-1] - t[0]: the loop
    period is not constant (9.998-10.032 ms), and (n-1)*mean_dt would differ by
    0.5% on the same window. Same convention as compute_tv(), used in the
    tables.

    Args:
        t_blk (numpy.ndarray): Time stamps of the window [s].
        forces (numpy.ndarray | None): Rotor forces [N], shape ``(N, 6)``.

    Returns:
        float: Total variation rate [rad/s^2], ``nan`` when not computable.
    """
    if forces is None or forces.shape[0] < 2:
        return float("nan")
    w = np.sqrt(np.clip(forces, F_MIN, F_MAX) / CF)
    dur = float(t_blk[-1] - t_blk[0])
    return float(np.sum(np.abs(np.diff(w, axis=0))) / dur) if dur > 0 else float("nan")


def build_blocks(f, phase_rows, delta, mission=None):
    """Split a flight into manoeuvre categories.

    ``mission`` is ``'p2p'`` or ``'rpy'`` and selects the ``HOLD_PHASES``
    names for the hold category; ``None`` or an unknown mission keeps the
    generic name ``Hover``.

    Args:
        f (dict): Flight from ``cat_view``.
        phase_rows (list[dict]): Rows of the phase table of this flight.
        delta (float): Duration of the transient after a step [s].
        mission (str | None): Mission type.

    Returns:
        list[tuple[str, str, float, float, dict]]: ``(category, label, t0, t1, info)`` of each block.
    """
    t = f["t"]
    out = []
    for r in phase_rows:
        lab = r["phase"]
        t0, t1 = cellf(r["t0"]), cellf(r["t1"])
        inside = (t >= t0) & (t <= t1)
        moving = np.flatnonzero(inside & f["ref_moving"])
        ramp_frac = float((inside & f["ref_moving"]).sum() / max(inside.sum(), 1))

        if ramp_frac > 0.5:                       # ramp phase
            t_stop = t[moving[-1]] if moving.size else t1
            cat = "Takeoff" if cellf(r["ref_z"]) > f["pos_ref"][inside][0][2] else "Land"
            # the sign of the commanded vertical displacement
            dz = f["pos_ref"][min(moving[-1], len(t) - 1)][2] - f["pos_ref"][moving[0]][2] \
                if moving.size else 0.0
            cat = "Takeoff" if dz > 0 else "Land"
            out.append((cat, lab, t0, t_stop, {"ramp_dz": dz}))
            continue

        # step phase: transient + hold
        step_pos = cellf(r.get("step_pos_m", "nan"))
        step_ang = cellf(r.get("step_ang_deg", "nan"))
        # the transient is comparable only if its phase is: the flag comes
        # from flag_comparability, on the data
        cmp_flag = cellf(r.get("comparable_transient", "1"))
        cmp_flag = True if cmp_flag != cmp_flag else bool(cmp_flag)
        # the steady-state error of the phase is needed for the rise time: the
        # 'Step' window holds only the transient, while the steady state is in
        # 'Hover', so the final value is passed from here
        out.append(("Step", lab, t0, min(t0 + delta, t1),
                    {"step_pos_m": step_pos, "step_ang_deg": step_ang,
                     "comparable": cmp_flag,
                     "e_ss_pos": cellf(r.get("rmse_pos_steady", "nan")),
                     "e_ss_att": cellf(r.get("rmse_att_steady", "nan"))}))
        if t1 > t0 + delta:
            hold_cat = HOLD_PHASES.get(mission, {}).get(lab, "Hover")
            out.append((hold_cat, lab, t0 + delta, t1, {}))

    return out


def window(f, t0, t1):
    """Boolean mask of the samples between two instants, ends included.

    Args:
        f (dict): Flight from ``cat_view``.
        t0 (float): Start [s].
        t1 (float): End [s].

    Returns:
        numpy.ndarray: Boolean mask over the samples.
    """
    m = (f["t"] >= t0) & (f["t"] <= t1)
    return m


# minimum distance below which no normalisation is done (m). Same value
# as --min-step-norm: metrics_for_category does not receive cfg.
cfg_min_step = MIN_STEP_NORM
# angular counterpart: below ~2 deg the ratio is noise.
cfg_min_step_ang = float(np.deg2rad(2.0))   # [rad]


def metrics_for_category(f, blocks, cat):
    """Metrics of one manoeuvre category of one flight.

    Args:
        f (dict): Flight from ``cat_view``.
        blocks (list[tuple]): Blocks from ``build_blocks``.
        cat (str): Category.

    Returns:
        dict | None: Value of each metric, or ``None`` when the category is absent.
    """
    sel = [b for b in blocks if b[0] == cat]
    if not sel:
        return None
    masks = [window(f, b[2], b[3]) for b in sel]
    allm = np.zeros_like(masks[0])
    for m in masks:
        allm |= m
    n = int(allm.sum())
    if n < 10:
        return None

    # Mask for the ABSOLUTE metrics: it excludes the sub-blocks whose step
    # is not comparable between the flights (see STEPS THAT ARE NOT
    # COMPARABLE above). For the other categories it equals allm.
    sel_cmp = [(b, m) for b, m in zip(sel, masks)
               if b[4].get("comparable", True)]
    absm = np.zeros_like(allm)
    for _b, m in sel_cmp:
        absm |= m
    n_abs = int(absm.sum())
    if n_abs < 10:                       # no comparable step
        absm = np.zeros_like(allm)
        n_abs = 0

    M = {"n_samples": n, "n_subblocks": len(sel),
         "n_samples_abs": n_abs, "n_subblocks_abs": len(sel_cmp),
         "dur_s": round(float(sum(b[3] - b[2] for b in sel)), 2)}

    # --- common header: the two quantities shown in the figures
    M["rmse_pos"] = rms_t(f["t"], f["e_pos"], absm) if n_abs else float("nan")
    M["rmse_att"] = rms_t(f["t"], f["e_att"], absm) if n_abs else float("nan")
    # decomposition of the position error, which always satisfies
    #     rmse_pos^2 = err_xy^2 + err_z^2
    # Horizontal and vertical errors have different causes (drift against
    # height lag or thrust error).
    M["err_xy"] = (rms_t(f["t"], np.linalg.norm(f["e_w"][:, :2], axis=1), absm)
                   if n_abs else float("nan"))
    M["err_z"] = rms_t(f["t"], f["e_w"][:, 2], absm) if n_abs else float("nan")

    kind = KIND[cat]

    if kind == "ramp":
        # vertical reference: signed height lag, which tells whether the
        # vehicle fell behind or went ahead of the ramp
        M["lag_z"] = float(np.mean(f["e_w"][allm][:, 2]))

        # NORMALISATION BY THE DISTANCE COMMANDED BY THE RAMP.
        #
        # The vehicle does not land at the same height in every flight, so the
        # landing ramp has a different length: 0.83 m in one folder and
        # 0.78 m in another, a window of 4.71 s against one of 4.44 s. As for
        # the steps, the error is divided by the amplitude, here the height
        # difference commanded by the ramp.
        #
        # The columns have the same names as those of the steps
        # (*_pooled_n_*, *_pooled_den) and the same meaning: error normalised
        # sample by sample, then a single integral. The attitude has no command
        # on a vertical ramp: it stays nan, and the attitude figure of
        # Takeoff/Land stays in absolute value.
        num_sq = num_ab = dur = wd = 0.0
        for b, m in zip(sel, masks):
            if m.sum() < 5:
                continue
            d = abs(float(b[4].get("ramp_dz", float("nan"))))
            if not (d == d and d > cfg_min_step):
                continue
            tt, en = f["t"][m], f["e_pos"][m] / d
            T = float(tt[-1] - tt[0])
            if T <= 0:
                continue
            num_sq += float(_TRAPZ(en ** 2, tt))
            num_ab += float(_TRAPZ(np.abs(en), tt))
            dur += T
            wd += d * T
        if dur > 0:
            M["pos_pooled_n_RMSE"] = float(np.sqrt(num_sq / dur))
            M["pos_pooled_n_MAE"] = float(num_ab / dur)
            M["pos_pooled_den"] = float(wd / dur)
            M["ramp_len_mean"] = float(wd / dur)
        else:
            for st, _a in STATS:
                M[f"pos_pooled_n_{st}"] = float("nan")
            M["pos_pooled_den"] = float("nan")
            M["ramp_len_mean"] = float("nan")
        for st, _a in STATS:
            M[f"att_pooled_n_{st}"] = float("nan")
        M["att_pooled_den"] = float("nan")

    elif kind == "step":
        # The normalised columns also use only the comparable sub-blocks. The
        # amplitude guard (sp > 0.02, sa > 1.0) is per flight, so on a step of
        # different amplitude it could pass in one group and fail in the
        # other; 'comparable' is decided by comparing the flights, so the sets
        # are symmetric. This handles several comparable steps of different
        # amplitude.
        # NORMALISATION BY THE ACTUAL DISTANCE, on ALL the sub-blocks.
        # It divides by the distance the vehicle had to cover at the instant
        # of the step, so it absorbs both the commanded amplitude and the
        # steady-state error carried from the previous setpoint. It runs on
        # 'sel', not on 'sel_cmp'.
        rm_r, iae_r, dr_l = [], [], []
        rm_ra, iae_ra, dr_la = [], [], []
        for b, m in zip(sel, masks):
            if m.sum() < 5:
                continue
            tt = f["t"][m]
            info = b[4]
            # The gate is on the COMMANDED step: when the reference has no step in
            # that quantity (in rpy the position does not move), the distance to
            # cover is only residual error, and dividing an error by another error
            # would reward the worse tracking.
            # The denominator must also be dominated by the command: when the
            # actual distance is less than half of the commanded amplitude, the
            # ratio measures the previous error.
            sp = info.get("step_pos_m", float("nan"))
            sa = info.get("step_ang_deg", float("nan"))
            sa_r = float(np.deg2rad(sa)) if sa == sa else float("nan")
            ep = f["e_pos"][m]
            d_real = float(ep[0])          # distance to cover at the instant of the step
            if (sp == sp and sp > cfg_min_step
                    and d_real > max(cfg_min_step, 0.5 * sp)):
                rm_r.append(rms_t(tt, ep) / d_real)
                iae_r.append(mae_t(tt, ep) / d_real)
                dr_l.append(d_real)
            ea = f["e_att"][m]
            a_real = float(ea[0])          # angle to cover at the instant of the step
            if (sa_r == sa_r and sa > 5.0
                    and a_real > max(cfg_min_step_ang, 0.5 * sa_r)):
                rm_ra.append(rms_t(tt, ea) / a_real)
                iae_ra.append(mae_t(tt, ea) / a_real)
                dr_la.append(a_real)
        # --- the same normalisation, POOLED LIKE THE ABSOLUTE VALUE.
        # The *_n_real columns above are means of ratios, one per sub-block;
        # the absolute bar puts all the samples in a single RMS. The pooled
        # form normalises the error SAMPLE BY SAMPLE by the distance of its
        # step, then takes one RMS (or MAE) over all the samples of all the
        # steps, SETPOINT 1 included. It is dimensionless; the rescaling to a
        # common distance, for an axis in metres, is done in compare_trajectory,
        # where both flights are available.
        for fld, ser, den, gate, dmin in (
                ("pos", "e_pos", "step_pos_m", cfg_min_step, cfg_min_step),
                ("att", "e_att", "step_ang_deg", 5.0, cfg_min_step_ang)):
            # the pooling is on the INTEGRALS, not on concatenated samples, which
            # would add a spurious trapezoid across the gap between two steps.
            num_sq = num_ab = dur = wd = 0.0
            for b, m in zip(sel, masks):
                if m.sum() < 5:
                    continue
                s = b[4].get(den, float("nan"))
                if not (s == s and s > gate):
                    continue
                e = f[ser][m]
                d = float(e[0])
                if not (d > dmin):
                    continue
                tt, en = f["t"][m], e / d
                T = float(tt[-1] - tt[0])
                if T <= 0:
                    continue
                num_sq += float(_TRAPZ(en ** 2, tt))
                num_ab += float(_TRAPZ(np.abs(en), tt))
                dur += T
                wd += d * T
            if dur <= 0:
                for st, _a in STATS:
                    M[f"{fld}_pooled_n_{st}"] = float("nan")
                M[f"{fld}_pooled_den"] = float("nan")
                continue
            M[f"{fld}_pooled_n_RMSE"] = float(np.sqrt(num_sq / dur))
            M[f"{fld}_pooled_n_MAE"] = float(num_ab / dur)
            M[f"{fld}_pooled_den"] = float(wd / dur)

        M["rmse_n_real"] = float(np.mean(rm_r)) if rm_r else float("nan")
        M["iae_n_real"] = float(np.mean(iae_r)) if iae_r else float("nan")
        M["dist_real_mean"] = float(np.mean(dr_l)) if dr_l else float("nan")
        M["n_subblocks_real"] = len(rm_r)
        M["rmse_n_real_att"] = float(np.mean(rm_ra)) if rm_ra else float("nan")
        M["iae_n_real_att"] = float(np.mean(iae_ra)) if iae_ra else float("nan")
        M["ang_real_mean"] = float(np.mean(dr_la)) if dr_la else float("nan")
        M["n_subblocks_real_att"] = len(rm_ra)

        pk, rm, t10, apk, amp = [], [], [], [], []
        for b, m in sel_cmp:
            if m.sum() < 5:
                continue
            info = b[4]
            sp, sa = info.get("step_pos_m", float("nan")), info.get("step_ang_deg", float("nan"))
            ep, ea = f["e_pos"][m], f["e_att"][m]
            tt = f["t"][m]
            if sp == sp and sp > 0.02:
                pk.append(float(ep.max()) / sp)
                rm.append(rms_t(tt, ep) / sp)
                amp.append(sp)
                # RISE TIME, on the deviation from the FINAL VALUE.
                # The policy settles on a nonzero steady-state error: on the absolute
                # error, when that error exceeds the threshold (10% of the step) the
                # condition is never met and the time equals the whole window.
                e_ss = info.get("e_ss_pos", float("nan"))
                dev = np.abs(ep - e_ss) if e_ss == e_ss else ep
                above = np.flatnonzero(dev > 0.10 * sp)
                t10.append(float(tt[above[-1] + 1] - tt[0])
                           if above.size and above[-1] + 1 < tt.size else float("nan"))
            if sa == sa and sa > 1.0:
                apk.append(float(ea.max()) / math.radians(sa))
        M["peak_n"] = float(np.mean(pk)) if pk else float("nan")
        M["rmse_n"] = float(np.mean(rm)) if rm else float("nan")
        M["t10_s"] = float(np.nanmean(t10)) if t10 else float("nan")
        M["att_peak_n"] = float(np.mean(apk)) if apk else float("nan")
        M["step_amp_mean"] = float(np.mean(amp)) if amp else float("nan")

    elif kind == "hold":
        M["e_mean"] = seg_mean_t(f["t"], f["e_pos"], allm)
        # e_std is a dispersion over the samples, not a time average
        M["e_std"] = float(np.std(f["e_pos"][allm]))
        bias = [float(np.linalg.norm(f["e_w"][m].mean(axis=0)))
                for m in masks if m.sum() > 5]
        M["e_bias"] = rms(bias) if bias else float("nan")
        M["att_mean"] = seg_mean_t(f["t"], f["e_att"], allm)
        M["att_yaw"] = rms_t(f["t"], f["e_yaw"], allm)
        M["att_rp"] = rms_t(f["t"], f["e_rp"], allm)
        if f["avel"] is not None:
            M["w_norm"] = rms_t(f["t"], f["avel"], allm)

    # --- control effort, for every category: weighted sum over the sub-blocks
    if f["forces"] is not None:
        # the TV is an absolute effort: a longer step needs more, so it uses
        # the same mask as the other absolute metrics
        tv_masks = [m for _b, m in sel_cmp] if kind == "step" else masks
        tvs, ws = [], []
        for m in tv_masks:
            idx = np.flatnonzero(m)
            if idx.size < 2:
                continue
            tvs.append(tv_rotor(f["t"][idx], f["forces"][idx]))
            ws.append(idx.size)
        if tvs:
            M["tv"] = float(np.average(tvs, weights=ws))
    return M


def _scale_limit(cat_max, factor=2.5):
    """y limit when ONE CATEGORY dominates the others.

    The comparison is between the maxima PER CATEGORY: on the single values a
    continuous distribution without outliers would cut half of the plot. The
    axis is cut only when the highest category exceeds the second by `factor`.

    Args:
        cat_max (list[float]): Maximum value of each category.
        factor (float): Ratio between the two highest categories that triggers the cut.

    Returns:
        float | None: Upper y limit, or ``None`` when the axis is not cut.
    """
    v = sorted((x for x in cat_max if x == x), reverse=True)
    if len(v) < 2 or v[1] <= 0:
        return None
    return v[1] * 1.35 if v[0] > factor * v[1] else None


# One figure per quantity and per error statistic. The horizontal and
# vertical components (err_xy / err_z) are in the CSV only: the figures
# show the six single components, defined as in the RMSE/ and Rollouts/
# folders, and the two norms.
#
# TWO STATISTICS
#   RMSE  size of the error, weighting the peaks
#   MAE   size of the typical error, without weighting the peaks

CAT_SERIES = [
    ("e_x",     lambda f: f["e_b"][:, 0], r"$e_x^{\,r}$",    "m",   "1_err_bx",    "{:.3f}"),
    ("e_y",     lambda f: f["e_b"][:, 1], r"$e_y^{\,r}$",    "m",   "2_err_by",    "{:.3f}"),
    ("e_z",     lambda f: f["e_b"][:, 2], r"$e_z^{\,r}$",    "m",   "3_err_bz",    "{:.3f}"),
    ("e_roll",  lambda f: f["rv"][:, 0],  r"$e_{roll}$",     "rad", "4_err_roll",  "{:.4f}"),
    ("e_pitch", lambda f: f["rv"][:, 1],  r"$e_{pitch}$",    "rad", "5_err_pitch", "{:.4f}"),
    ("e_yaw",   lambda f: f["rv"][:, 2],  r"$e_{yaw}$",      "rad", "6_err_yaw",   "{:.4f}"),
    ("e_p",     lambda f: f["e_pos"],     r"$\|e_p\|$",      "m",   "7_err_pos",   "{:.3f}"),
    ("e_R",     lambda f: f["e_att"],     r"$\|e_R\|$",      "rad", "8_err_att",   "{:.4f}"),
    # motion, not errors: in Takeoff, Step and Landing the reference
    # moves, so a high value is expected.
    ("v_norm",  lambda f: f["v"],         r"\|v\|",          "m/s", "9_vel",       "{:.3f}"),
    ("w_norm",  lambda f: f["avel"],      r"\|\omega\|",     "rad/s", "10_avel",   "{:.3f}"),
]
# Figure titles
CAT_TITLE = {"e_x": "Position error X (reference frame)",
             "e_y": "Position error Y (reference frame)",
             "e_z": "Position error Z (reference frame)",
             "e_roll": "Attitude error - roll",
             "e_pitch": "Attitude error - pitch",
             "e_yaw": "Attitude error - yaw",
             "e_p": "Position error (total)",
             "e_R": "Attitude error (total)",
             "v_norm": "Linear velocity norm",
             "w_norm": "Angular velocity norm"}
CAT_MOTION = ("v_norm", "w_norm")

# The two error statistics, with signature (t, v, mask): TIME averages,
# see seg_mean_t.
STATS = (("RMSE", rms_t),
         ("MAE",  mae_t))


def category_series_stats(f, blocks, cats):
    """RMSE and MAE of each quantity of ``CAT_SERIES``, per category, for one flight.

    As in ``metrics_for_category``, the blocks with a non-comparable step are
    left out, since RMSE and MAE of these series are absolute values.

    Args:
        f (dict): Flight from ``cat_view``.
        blocks (list[tuple]): Blocks from ``build_blocks``.
        cats (list[str]): Categories.

    Returns:
        dict: ``{statistic: {category: {quantity: value}}}``.
    """
    out = {st: {} for st, _fn in STATS}
    for cat in cats:
        sel = [b for b in blocks
               if b[0] == cat and b[4].get("comparable", True)]
        if not sel:
            continue
        allm = np.zeros(f["t"].size, bool)
        for b in sel:
            allm |= window(f, b[2], b[3])
        if allm.sum() < 10:
            continue
        for key, get, _sym, _u, _fn, _fmt in CAT_SERIES:
            ser = get(f)
            if ser is None:                       # 'vel'/'avel' missing from the log
                continue
            v = np.asarray(ser, float)
            for st, agg in STATS:
                out[st].setdefault(cat, {})[key] = agg(f["t"], v, allm)
    return out


def cat_bar_figure(cats, stats, groups, key, stat, out_path, clip=False):
    """One quantity and one statistic: the manoeuvres on the x axis, one bar per
    group.

    Every manoeuvre is shown. Where one dominates the others (typically Step,
    with errors five times larger) --cat-clip keeps the axis on the small bars
    and CUTS the bar out of scale, writing its value above it. By default the
    proportions are kept.

    Args:
        cats (list[str]): Categories on the x axis.
        stats (dict): ``{group: category_series_stats(...)}``.
        groups (list[str]): Groups, one bar each (PPO, SPPO).
        key (str): Quantity of ``CAT_SERIES``.
        stat (str): ``'RMSE'`` or ``'MAE'``.
        out_path (pathlib.Path): Output PDF.
        clip (bool): Cut the bar of a dominating category.

    Returns:
        bool: ``True`` when the figure is written.
    """
    spec = next(x for x in CAT_SERIES if x[0] == key)
    _k, _get, sym, unit, _fname, fmt = spec
    vals = [stats[g][stat].get(c, {}).get(key, float("nan"))
            for g in groups for c in cats]
    finite = [v for v in vals if v == v]
    if not finite:
        return False
    signed = min(finite) < 0                      # only the mean error can be negative

    if signed:
        m = max(abs(v) for v in finite) * 1.32
        bot, top = -m, m
    else:
        bot = 0.0
        cat_max = [max((stats[g][stat].get(c, {}).get(key, float("nan"))
                        for g in groups), default=float("nan")) for c in cats]
        lim = _scale_limit(cat_max) if clip else None
        top = lim if lim else max(finite) * 1.30

    fig, ax = plt.subplots(figsize=(1.9 * len(cats) + 4.6, 6.2))
    x = np.arange(len(cats))
    pitch, w = 0.38, 0.34
    for i, g in enumerate(groups):
        v = [stats[g][stat].get(c, {}).get(key, float("nan")) for c in cats]
        ax.bar(x + (i - 0.5) * pitch, v, w, color=COL[g], edgecolor="none",
               label=g, zorder=2)
    ax.set_ylim(bot, top)
    if signed:
        ax.axhline(0.0, color="#333333", lw=1.1, zorder=3)
    ax.set_ylabel(f"{sym}   [{unit}]")
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", visible=True)
    ax.set_xticks(x)
    ax.set_xticklabels(list(cats), fontsize=16)
    ax.set_xlim(-0.65, len(cats) - 0.35)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    save(fig, out_path)
    return True


def cat_bar_figure_pair(cats, stats, groups, stat, out_path, clip=False):
    """1x2 grid: position (e_p) and attitude (e_R) side by side.

    Args:
        cats (list[str]): Categories on the x axis.
        stats (dict): ``{group: category_series_stats(...)}``.
        groups (list[str]): Groups, one bar each (PPO, SPPO).
        stat (str): ``'RMSE'`` or ``'MAE'``.
        out_path (pathlib.Path): Output PDF.
        clip (bool): Cut the bar of a dominating category.

    Returns:
        bool: ``True`` when the figure is written.
    """
    keys = ["e_p", "e_R"]
    specs = [next(x for x in CAT_SERIES if x[0] == k) for k in keys]

    has_data = False
    for spec in specs:
        vals = [stats[g][stat].get(c, {}).get(spec[0], float("nan")) for g in groups for c in cats]
        if any(v == v for v in vals):
            has_data = True
    if not has_data:
        return False

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    x = np.arange(len(cats))
    pitch, w = 0.38, 0.34

    for ax, spec in zip(axs, specs):
        key, _get, sym, unit, _fname, fmt = spec
        
        cat_max = [max((stats[g][stat].get(c, {}).get(key, float("nan"))
                        for g in groups), default=float("nan")) for c in cats]
        lim = _scale_limit(cat_max) if clip else None
        
        vals_for_top = [stats[g][stat].get(c, {}).get(key, float("nan")) for g in groups for c in cats]
        finite = [v for v in vals_for_top if v == v]
        top = lim if lim else (max(finite) * 1.30 if finite else 1.0)
        
        for i, g in enumerate(groups):
            v = [stats[g][stat].get(c, {}).get(key, float("nan")) for c in cats]
            ax.bar(x + (i - 0.5) * pitch, v, w, color=COL[g], edgecolor="none", label=g, zorder=2)
        
        ax.set_ylim(0, top)
        ax.set_ylabel(f"{sym}   [{unit}]")
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", visible=True)
        ax.set_xticks(x)
        ax.set_xticklabels(list(cats), fontsize=16)
        ax.set_xlim(-0.65, len(cats) - 0.35)
        
    axs[0].legend(loc="upper right", frameon=True)
    fig.tight_layout()
    save(fig, out_path)
    return True


def export_categories(flights, rows, dirs, folder_name, cfg):
    """Write tables/category_metrics.csv and the category figures.

    Args:
        flights (list[dict]): Flights from ``process_log``.
        rows (list[dict]): Rows of the phase table.
        dirs (dict[str, pathlib.Path]): Output folders.
        folder_name (str): Name of the mission folder; its prefix (``p2p`` or ``rpy``) selects the categories.
        cfg (dict): Analysis settings (``cat_delta``, ``cat_clip``, ``cat_exclude``).
    """
    if not rows or len(flights) < 2:
        return
    mission = ("p2p" if folder_name.startswith("p2p") else
               "rpy" if folder_name.startswith("rpy") else None)
    cats_all = CATEGORIES_BY_MISSION.get(mission, CATEGORIES)
    views = {f['group']: cat_view(f) for f in flights}
    blocks = {g: build_blocks(v, [r for r in rows if r['group'] == g],
                              cfg['cat_delta'], mission)
              for g, v in views.items()}
    groups = [g for g in GROUPS if g in views]
    if len(groups) < 2:
        return

    print('=' * 94)
    print('CONTROL TASK CATEGORIES'
          f"   (Delta = {cfg['cat_delta']} s)")
    print('=' * 94)
    print(f"  {'category':16s}{'phase':12s}"
          + ''.join(f"{g + '  t0-t1':>16s}{'dur':>7s}" for g in groups))
    for cat in cats_all:
        per_g = [[b for b in blocks[g] if b[0] == cat] for g in groups]
        for tup in zip(*per_g):
            print(f"  {cat:16s}{tup[0][1]:12s}"
                  + ''.join(f"{f'{b[2]:.2f}-{b[3]:.2f}':>16s}{b[3]-b[2]:7.2f}"
                            for b in tup))

    data = {g: {} for g in groups}
    for g in groups:
        for cat in cats_all:
            m = metrics_for_category(views[g], blocks[g], cat)
            if m:
                data[g][cat] = m
    cats = [c for c in cats_all if all(c in data[g] for g in groups)]
    if not cats:
        print("  no category in common between the flights: section skipped.\n")
        return

    a_, b_ = groups[0], groups[1]
    print(f"\n{'-'*94}\nMETRICS  (delta = {b_} with respect to {a_}; "
          f"negative = {b_} better)\n{'-'*94}")
    rows_csv = []
    for cat in cats:
        nb = data[a_][cat]['n_subblocks']
        nba = data[a_][cat].get('n_subblocks_abs', nb)
        print(f"\n  [{cat}]   {data[a_][cat]['dur_s']:.1f} s, "
              f"{nb} sub-block(s), "
              f"{data[a_][cat]['n_samples']} samples"
              + ("" if nba == nb else
                 f"  --  {nba}/{nb} comparable: the absolute metrics "
                 f"use only those, {data[a_][cat].get('n_samples_abs', 0)} samples"))
        for k in [k for k in data[a_][cat]
                  if k not in ('n_samples', 'n_subblocks', 'dur_s',
                               'n_samples_abs', 'n_subblocks_abs')]:
            va, vb = data[a_][cat].get(k), data[b_][cat].get(k)
            if va is None or vb is None or va != va or vb != vb:
                continue
            d = (vb - va) / va * 100.0 if va else float('nan')
            star = '   <<<' if abs(d) >= 10 else ''
            print(f"      {k:14s}{va:12.4f}{vb:12.4f}  "
                  f"{CAT_UNITS.get(k, ''):<7s}{d:+8.1f}%{star}")
        for g in groups:
            for k, v in data[g][cat].items():
                rows_csv.append({'run': views[g]['run'], 'group': g,
                                 'folder': folder_name, 'category': cat,
                                 'stat': '-', 'metric': k, 'value': v,
                                 'unit': CAT_UNITS.get(k, ''),
                                 'n_samples': data[g][cat]['n_samples'],
                                 'n_subblocks': data[g][cat]['n_subblocks']})

    if 'Step' in cats:
        aa = data[a_]['Step'].get('step_amp_mean', float('nan'))
        bb = data[b_]['Step'].get('step_amp_mean', float('nan'))
        nb = data[a_]['Step']['n_subblocks']
        nba = data[a_]['Step'].get('n_subblocks_abs', nb)
        if nba < nb:
            print(f"\n  STEPS  {nb - nba} sub-block(s) of 'Step' with a different amplitude "
                  f"in the two flights, excluded from the absolute metrics.")
            if nba == 0:
                print("  No comparable step: rmse_pos, rmse_att, err_xy, "
                      "err_z and tv of 'Step' are empty.")
                print("  peak_n, rmse_n and att_peak_n are kept, normalised "
                      "by the amplitude and computed on all the sub-blocks.")
        if aa == aa and bb == bb and aa and abs(bb - aa) / aa > 1e-3:
            print(f"\n  NOTE  mean step amplitude (all sub-blocks): "
                  f"{a_} {aa:.3f} m, {b_} {bb:.3f} m  ({(bb-aa)/aa*100:+.1f}%).")
            print("  It applies to the normalised columns; the absolute ones are already filtered.")

    # The error statistics are computed from the samples (the MAE cannot be
    # derived from the RMSE). They also go into the CSV, flagged in the
    # 'stat' column, because compare_trajectory reads them from there.
    stats = {g: category_series_stats(views[g], blocks[g], cats)
             for g in groups}
    for g in groups:
        for stat, _agg in STATS:
            for cat, mm in stats[g][stat].items():
                for k, v in mm.items():
                    rows_csv.append({'run': views[g]['run'], 'group': g,
                                     'folder': folder_name, 'category': cat,
                                     'stat': stat, 'metric': k, 'value': v,
                                     'unit': CAT_UNITS.get(k, ''),
                                     'n_samples': data[g][cat]['n_samples'],
                                     'n_subblocks': data[g][cat]['n_subblocks']})

    csv_path = dirs['tables'] / 'category_metrics.csv'
    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['run', 'group', 'folder', 'category',
                                           'stat', 'metric', 'value', 'unit',
                                           'n_samples', 'n_subblocks'])
        w.writeheader()
        w.writerows(rows_csv)
    print(f"\n  written {csv_path}")

    shown = [c for c in cats if c not in cfg['cat_exclude']]
    if not shown:
        print("  --cat-exclude removed every category: figures skipped.")
        return
    if len(shown) < len(cats):
        print(f"  figure without: {', '.join(c for c in cats if c not in shown)}"
              f"  (still in the CSV)")
    # maneuvers/ follows the MISSION: takeoff, hold, landing. The step is
    # an experiment on the controller rather than a manoeuvre of the
    # mission, with its own reading (the error normalised by the distance
    # to cover) and its three numbers (rise, overshoot, settling), so it
    # is in maneuvers/step/.
    shown_man = [c for c in shown if KIND.get(c) != 'step']
    shown_step = [c for c in shown if KIND.get(c) == 'step']

    # ------------------------------------- maneuvers/: one quantity, one figure
    for stat, _agg in STATS:
        d = dirs['man_' + stat]
        n_fig = 0
        for key, _get, _sym, _u, fname, _fmt in CAT_SERIES:
            if cat_bar_figure(shown_man, stats, groups, key, stat,
                              d / f"{fname}.pdf", clip=cfg['cat_clip']):
                n_fig += 1
                
        if cat_bar_figure_pair(shown_man, stats, groups, stat,
                               d / "pos_att_pair.pdf", clip=cfg['cat_clip']):
            n_fig += 1

        print(f"  {n_fig} figures written in {d}")

    # --- maneuvers/propellers: control effort PER PHASE.
    tvcat = {}
    for g in groups:
        v = views[g]
        if v.get('forces') is None:
            continue
        for cat in shown_man:
            sel = [b for b in blocks[g]
                   if b[0] == cat and b[4].get('comparable', True)]
            if not sel:
                continue
            mk = np.zeros(v['t'].size, bool)
            for b in sel:
                mk |= window(v, b[2], b[3])
            tv = compute_tv(v, mk)
            if tv is not None:
                tvcat[(g, cat)] = tv

    n_fig = 0
    if tvcat:
        width = 0.35
        xs = np.arange(len(shown_man))
        for i in range(N_ROTORS):
            fig, ax = plt.subplots(figsize=(max(7, 1.7 * len(shown_man)), 6))
            for gi, g in enumerate(groups):
                vals = [tvcat.get((g, c), [float('nan')] * N_ROTORS)[i]
                        for c in shown_man]
                ax.bar(xs + (gi - 0.5) * width, vals, width, label=g,
                       color=COL[g], edgecolor='none')
            ax.set_ylabel(r'$TV_{\omega}$   [rad/s$^2$]')
            ax.set_xticks(xs)
            ax.set_xticklabels(shown_man, fontsize=16)
            ax.grid(axis='x', visible=False)
            ax.grid(axis='y', visible=True)
            ax.legend(loc='best', frameon=True)
            fig.tight_layout()
            save(fig, dirs['man_tv'] / f"tv_propeller_{i+1}.pdf")
            n_fig += 1

        fig, ax = plt.subplots(figsize=(max(8, 2.0 * len(shown_man)), 6))
        for gi, g in enumerate(groups):
            vals = [float(np.sum(tvcat[(g, c)])) if (g, c) in tvcat
                    else float('nan') for c in shown_man]
            ax.bar(xs + (gi - 0.5) * width, vals, width, label=g,
                   color=COL[g], edgecolor='none')
        ax.set_ylabel(r'$TV_{\omega}$   [rad/s$^2$]')
        ax.set_xticks(xs)
        ax.set_xticklabels(shown_man, fontsize=16)
        ax.grid(axis='x', visible=False)
        ax.grid(axis='y', visible=True)
        ax.legend(loc='best', frameon=True)
        fig.tight_layout()
        save(fig, dirs['man_tv'] / "summary_tv.pdf")
        n_fig += 1
        print(f"  {n_fig} figures written in {dirs['man_tv']}")

    with open(dirs['tables'] / 'tv_maneuvers.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['category', 'group', 'channel', 'tv'])
        for (g, cat), tv in sorted(tvcat.items()):
            for i in range(N_ROTORS):
                w.writerow([cat, g, f'M{i+1}', f"{tv[i]:.4f}"])
            w.writerow([cat, g, 'TOTAL', f"{float(np.sum(tv)):.4f}"])


def _bars_on_ax(ax, labels, vals, title, ylabel, hatch_flags=None,
                fmt='{:.3f}', zero_line=False, legend=True):
    """PPO/SPPO bars side by side on a given axis, one pair per label.

    Args:
        ax (matplotlib.axes.Axes): Axis.
        labels (list[str]): Labels on the x axis.
        vals (dict[str, list[float]]): Values per group, one per label.
        title (str): Axis title.
        ylabel (str): Label of the y axis.
        hatch_flags (list[bool] | None): Comparability of each label; the bars of a label set to ``False`` are hatched.
        fmt (str): Format of the values written above the bars.
        zero_line (bool): Draw a horizontal line at zero.
        legend (bool): Draw the legend.

    Returns:
        bool: ``True`` when at least one bar is drawn.
    """
    have = [g for g in GROUPS if g in vals and any(v == v for v in vals[g])]
    if not labels or not have:
        return False
    x, pitch, width = np.arange(len(labels)), 0.38, 0.34
    for gi, g in enumerate(have):
        xb = x + (gi - (len(have) - 1) / 2) * pitch
        bars = ax.bar(xb, [0 if v != v else v for v in vals[g]], width,
                      label=g, color=COL[g], edgecolor='none')
        for k, (b, v) in enumerate(zip(bars, vals[g])):
            if v != v:
                b.set_visible(False)
                continue
            if hatch_flags is not None and not hatch_flags[k]:
                b.set_hatch('//')
    if zero_line:
        ax.axhline(0.0, color='k', lw=1.0, ls=':')
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    long_lab = max((len(str(l)) for l in labels), default=0) > 11
    ax.set_xticklabels(labels, rotation=18 if long_lab else 0,
                       ha='right' if long_lab else 'center', fontsize=16)
    ax.set_xlim(-0.65, len(labels) - 0.35)
    ax.grid(axis='x', visible=False)
    ax.grid(axis='y', visible=True)
    vmax = max((v for g in have for v in vals[g] if v == v), default=0.0)
    if vmax > 0:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, max(hi, vmax * (1.30 if legend else 1.10)))
    if legend:
        ax.legend(loc='upper right', frameon=True)
    return True


def _panel_figure(out_path, suptitle, panels, labels):
    """One figure, several panels side by side on the same labels.

    Args:
        out_path (pathlib.Path): Output PDF.
        suptitle (str): Title of the figure.
        panels (list[dict]): Panels, each with ``vals`` and the arguments of ``_bars_on_ax``.
        labels (list[str]): Labels on the x axis.

    Returns:
        int: Number of figures written (0 or 1).
    """
    panels = [q for q in panels
              if any(v == v for g in q['vals'] for v in q['vals'][g])]
    if not labels or not panels:
        return 0
    n = len(panels)
    fig, axs = plt.subplots(1, n, figsize=(max(4.6, 1.9 * len(labels)) * n, 5.6),
                            squeeze=False)
    for i, q in enumerate(panels):
        _bars_on_ax(axs[0][i], labels, q['vals'], q['title'], q['ylabel'],
                    None, q.get('fmt', '{:.3f}'),
                    q.get('zero_line', False), legend=(i == 0))
    fig.tight_layout()
    save(fig, out_path)
    return 1


def _slug(label):
    """Turn a phase label into a file name, e.g. ``'SETPOINT 2'`` into ``'SETPOINT_2'``.

    Args:
        label (str): Phase label.

    Returns:
        str: Label with underscores in place of spaces.
    """
    return "_".join(str(label).split())


def _step_profile_ax(ax, per, label, finals=True, bands=True, title=True):
    """Progress profile of ONE phase, PPO against SPPO.

    Args:
        ax (matplotlib.axes.Axes): Axis.
        per (dict): Step responses keyed by ``(label, group)``.
        label (str): Phase label.
        finals (bool): Draw a line at the final progress of each flight.
        bands (bool): Draw the 10% and 90% lines of the rise time.
        title (bool): Write the axis title.

    Returns:
        bool: ``True`` when at least one profile is drawn.
    """
    drawn = False
    for g in GROUPS:
        if (label, g) not in per:
            continue
        r = per[(label, g)]
        ax.plot(r['t'], r['prog'], color=COL[g], lw=1.6, label=g)
        if finals:
            ax.axhline(r['p_fin'], color=COL[g], lw=0.9, ls=':')
        drawn = True
    ax.axhline(1.0, color='k', lw=1.0, ls='--')
    if bands:
        ax.axhline(0.9, color='#2e8b57', lw=0.9, ls='-.')
        ax.axhline(0.1, color='#2e8b57', lw=0.9, ls='-.')
    ax.set_xlabel(r'$t - t_{\mathrm{step}}$   [s]')
    ax.set_ylim(-0.05, max(1.25, ax.get_ylim()[1]))
    ax.margins(x=0)
    if drawn:
        ax.legend(loc='lower right', frameon=True)
    return drawn


def _step_table_figure(out_path, per, label, kind, tag, cfg):
    """Table of the step response numbers, PPO against SPPO.

    Args:
        out_path (pathlib.Path): Output PDF.
        per (dict): Step responses keyed by ``(label, group)``.
        label (str): Phase label.
        kind (str): ``'pos'`` or ``'att'``.
        tag (str): Name of the channel used in the title.
        cfg (dict): Analysis settings.

    Returns:
        int: Number of figures written (0 or 1).
    """
    have = {g: per[(label, g)] for g in GROUPS if (label, g) in per}
    if len(have) < 2:
        return 0
    a, b = have[GROUPS[0]], have[GROUPS[1]]
    unit = 'deg' if kind == 'att' else 'm'
    k_abs = RAD2DEG if kind == 'att' else 1.0
    band = f"{cfg['settle_band']*100:.0f}%"

    spec = [
        ('overshoot  [% of $\\Delta$]',
         a['overshoot_pct'], b['overshoot_pct'], '{:+.1f}', 'p.p.', False),
        ('rise time  $t_{90}-t_{10}$  [s]',
         a['rise_10_90_s'], b['rise_10_90_s'], '{:.3f}', 's', True),
        (f"settling time  $t_s$  [s]   ({band} band)",
         a['settle_s'], b['settle_s'], '{:.3f}', 's', True),
        (f'steady-state error  [{unit}]',
         a['final_err_abs'] * k_abs, b['final_err_abs'] * k_abs,
         '{:+.4f}' if kind == 'pos' else '{:+.2f}', unit, True),
    ]

    def cells(v0, v1, fmt, du, rel):
        f = (lambda v: '--' if v != v else fmt.format(v))
        rs = '--'
        if rel and v0 == v0 and v1 == v1 and abs(v0) > 1e-9:
            rs = f"{(abs(v1) - abs(v0)) / abs(v0) * 100.0:+.1f} %"
        elif not rel and v0 == v0 and v1 == v1:
            rs = f"{v1 - v0:+.1f} p.p."
        return [f(v0), f(v1), rs]

    body = [[s[0]] + cells(*s[1:]) for s in spec]
    head = ['', GROUPS[0], GROUPS[1], 'variation']

    fig, ax = plt.subplots(figsize=(11.2, 0.46 * (len(body) + 1)))
    ax.axis('off')
    ax.grid(False)
    tb = ax.table(cellText=body, colLabels=head, loc='center',
                  cellLoc='center', colWidths=[0.40, 0.17, 0.17, 0.18])
    tb.auto_set_font_size(False)
    tb.set_fontsize(13)
    tb.scale(1, 1.75)
    for (i, j), c in tb.get_celld().items():
        c.set_linewidth(0.9)
        c.set_edgecolor(C_INK)
        if i == 0:                                                           # header: rules only
            c.visible_edges = 'TB'
            c.set_facecolor('#f2f2f2')
            if j in (1, 2):                                                  # PPO / SPPO in their colours
                c.get_text().set_color(COL[GROUPS[j - 1]])
        else:
            c.visible_edges = 'B' if i == len(body) else ''
        if j == 0:
            c.get_text().set_ha('left')
            c.PAD = 0.03

    save(fig, out_path)
    return 1


def export_assets(flights, dirs, cfg, phase_rows=()):
    """Manoeuvre-specific metrics, per phase, PPO against SPPO.

    Args:
        flights (list[dict]): Flights from ``process_log``.
        dirs (dict[str, pathlib.Path]): Output folders.
        cfg (dict): Analysis settings.
        phase_rows (list[dict]): Rows of the phase table.

    Returns:
        list[dict]: Rows of the manoeuvre table.
    """
    rows = []
    n_fig = 0

    # ---------------------------------------------- step response
    for kind, tag, unit_amp, amp_fmt in (('pos', 'position', 'm', '{:.3f}'),
                                         ('att', 'attitude', 'deg', '{:.1f}')):
        per = {}
        for f in flights:
            sp = analysed_phases(f['phases'])
            for j in range(len(sp)):
                r = step_response(f, sp, j, cfg, kind)
                if r is not None:
                    per[(r['phase'], f['group'])] = r
        labels = list(dict.fromkeys(k[0] for k in per))
        if not labels:
            continue
        def col(key, scale=1.0):
            return {g: [per[(l, g)][key] * scale if (l, g) in per else float('nan')
                        for l in labels] for g in GROUPS}

        pre = 1 if kind == 'pos' else 3
        n_fig += _panel_figure(
            dirs['step'] / f"{pre}_step_{kind}_response.pdf",
            f'Step response - {tag}',
            [{'vals': col('overshoot_pct'), 'title': 'overshoot',
              'ylabel': r'$\hat{s}$   [% of $\Delta$]', 'fmt': '{:+.1f}',
              'zero_line': True},
             {'vals': col('rise_10_90_s'), 'title': 'rise time',
              'ylabel': r'$t_{90}-t_{10}$   [s]'},
             {'vals': col('settle_s'),
              'title': f"settling time ({cfg['settle_band']*100:.0f}% band)",
              'ylabel': r'$t_s$   [s]'}],
            labels)

        fig, axs = plt.subplots(1, len(labels), figsize=(5.6 * len(labels), 5.2),
                                squeeze=False)
        for i, l in enumerate(labels):
            _step_profile_ax(axs[0][i], per, l)
        axs[0][0].set_ylabel(r'progress $p(t)$   [1 = target]')
        fig.tight_layout()
        save(fig, dirs['step'] / f"{pre+1}_step_{kind}_profile.pdf")
        n_fig += 1

        for l in labels:
            slug = _slug(l)
            fig, ax = plt.subplots(figsize=(6.4, 5.2))
            _step_profile_ax(ax, per, l, finals=False, bands=False, title=False)
            ax.set_ylabel(r'$\bar{p}(t)$   [1 = target]')
            fig.tight_layout()
            save(fig, dirs['step']
                 / f"{pre+1}_step_{kind}_profile_{slug}.pdf")
            n_fig += 1

            n_fig += _step_table_figure(
                dirs['step'] / f"{pre+6}_step_{kind}_table_{slug}.pdf",
                per, l, kind, tag, cfg)

        for (l, g), r in per.items():
            for k in ('amp', 'p0_frac', 'overshoot_pct', 'rise_10_90_s',
                      't10_s', 't90_s', 'settle_s', 'final_err_frac',
                      'final_err_abs'):
                rows.append({'section': f'step_{kind}', 'phase': l, 'group': g,
                             'metric': k, 'value': r[k],
                             'comparable': r['comparable']})

    # ------------------------- transient normalised by the actual distance
    for kind, pre, base, ttl in (('pos', 5, 'norm_real_m', 'position'),
                                 ('att', 6, 'norm_real_deg', 'attitude')):
        keys = (('rmse_pos_transient_real', 'iae_pos_real') if kind == 'pos'
                else ('rmse_att_transient_real', 'iae_att_real'))
        per = {}
        for r in phase_rows:
            if not r['phase'].startswith('SETPOINT'):
                continue
            if r.get(base) != r.get(base):          # nan: no step here
                continue
            per[(r['phase'], r['group'])] = r
        labels = list(dict.fromkeys(k[0] for k in per))
        if not labels:
            continue

        def rcol(key):
            return {g: [per[(l, g)][key] if (l, g) in per else float('nan')
                        for l in labels] for g in GROUPS}

        n_fig += _panel_figure(
            dirs['step'] / f"{pre}_step_{kind}_normalised.pdf",
            f'Transient normalised by actual distance - {ttl}',
            [{'vals': rcol(keys[0]), 'title': 'RMSE / distance',
              'ylabel': r'RMSE$_{tr}$ / $d$   [-]', 'fmt': '{:.3f}'},
             {'vals': rcol(keys[1]), 'title': 'IAE / distance',
              'ylabel': r'IAE / $d$   [-]', 'fmt': '{:.3f}'}],
            labels)
        for (l, g), r in per.items():
            for k in (base,) + keys:
                rows.append({'section': f'norm_real_{kind}', 'phase': l,
                             'group': g, 'metric': k, 'value': r[k],
                             'comparable': 1})

    # ------------------------------ control effort WITHIN the phase
    tvph = {}
    for f in flights:
        if 'forces' not in f:
            continue
        for ph in analysed_phases(f['phases']):
            mk = np.zeros(f['t'].size, bool)
            mk[ph['i0']:ph['i1'] + 1] = True
            tv = compute_tv(f, mk)
            if tv is not None:
                tvph[(ph['label'], f['group'])] = tv

    for l in list(dict.fromkeys(k[0] for k in tvph)):
        have = [g for g in GROUPS if (l, g) in tvph]
        if not have:
            continue
        xs, width = np.arange(N_ROTORS), 0.35
        fig, ax = plt.subplots(figsize=(9.0, 5.6))
        vmax = 0.0
        for gi, g in enumerate(have):
            vals = tvph[(l, g)]
            xb = xs + (gi - (len(have) - 1) / 2) * width
            ax.bar(xb, vals, width, color=COL[g], edgecolor='none',
                   label=f"{g}")
            vmax = max(vmax, float(np.max(vals)))
        ax.set_ylabel(r'$TV_{\omega}$   [rad/s$^2$]')
        ax.set_xlabel('propeller', fontsize=16)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(i + 1) for i in range(N_ROTORS)], fontsize=16)
        ax.grid(axis='x', visible=False)
        ax.grid(axis='y', visible=True)
        if vmax > 0:
            ax.set_ylim(0, vmax * 1.30)
        ax.legend(loc='upper right', frameon=True)
        fig.tight_layout()
        _dk = 'step_tv' if str(l).upper().startswith('SETPOINT') else 'man_tv'
        save(fig, dirs[_dk] / f"tv_phase_{_slug(l)}.pdf")
        n_fig += 1
        for g in have:
            for i, v in enumerate(tvph[(l, g)]):
                rows.append({'section': 'tv_phase', 'phase': l, 'group': g,
                             'metric': f'tv_rotor_{i+1}', 'value': float(v),
                             'comparable': 1})

    # ---------------------------------------------- landing
    td = {}
    for f in flights:
        r = touchdown_metrics(f, analysed_phases(f['phases']), cfg)
        if r is not None:
            td[f['group']] = r
    if td:
        for g, r in td.items():
            for k in ('t_touchdown_s', 'err_xy_m', 'err_z_m', 'err_pos_m'):
                rows.append({'section': 'touchdown', 'phase': 'LANDING',
                             'group': g, 'metric': k, 'value': r[k],
                             'comparable': 1})

    # ---------------------------------------------- coupling
    cp = {}
    for f in flights:
        sp = analysed_phases(f['phases'])
        for j in range(len(sp)):
            r = coupling_metrics(f, sp, j, cfg)
            if r is not None:
                cp[(r['phase'], f['group'])] = r
    labels = list(dict.fromkeys(k[0] for k in cp))
    if labels:
        for (l, g), r in cp.items():
            for k in ('ang_deg', 'peak_pos_m', 'peak_per_deg_mm',
                      'steady_pos_m', 'steady_per_deg_mm'):
                rows.append({'section': 'coupling', 'phase': l, 'group': g,
                             'metric': k, 'value': r[k], 'comparable': 1})

    if rows:
        with open(dirs['tables'] / "assets.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=['section', 'phase', 'group',
                                               'metric', 'value', 'comparable'])
            w.writeheader()
            for r in rows:
                w.writerow({**r, 'value': nan_to_blank(r['value'])})
    print(f"\n  {n_fig} figures written in maneuvers/step")
    return rows


# ===========================================================================
def export_analysis(input_dir: Path, cfg):
    """Run the whole analysis of one mission folder.

    Loads the flights, detects the phases, computes the metrics and writes the
    figures and the tables under ``<input_dir>/data_analysis``.

    Args:
        input_dir (pathlib.Path): Mission folder, e.g. ``p2p_nominal``.
        cfg (dict): Analysis settings from the command line.
    """
    set_style()

    out_dir = input_dir / "data_analysis"
    dirs = {
        'man_RMSE': out_dir / "maneuvers" / "RMSE",
        'man_MAE': out_dir / "maneuvers" / "MAE",
        'man_tv': out_dir / "maneuvers" / "propellers",
        'step': out_dir / "maneuvers" / "step",
        'step_RMSE': out_dir / "maneuvers" / "step" / "RMSE",
        'step_MAE': out_dir / "maneuvers" / "step" / "MAE",
        'step_tv': out_dir / "maneuvers" / "step" / "propellers",
        'flight_RMSE': out_dir / "flight" / "RMSE",
        'flight_MAE': out_dir / "flight" / "MAE",
        'flight_tv': out_dir / "flight" / "propellers",
        'roll': out_dir / "flight" / "Rollouts",
        'tables': out_dir / "tables",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print(f"Scanning data.npz files in: {input_dir} ...")
    npz_files = sorted(input_dir.rglob("data.npz"))
    print(f"  {len(npz_files)} files found.\n")

    # ----------------------------------------------------- reading + manifest
    flights, manifest = [], []
    for p in npz_files:
        res, why = process_log(p, cfg)
        if res is None:
            manifest.append({'run': p.parent.name, 'group': '', 'included': 0,
                             'duration': '', 'dt_mean_ms': '', 'n_samples': '',
                             'n_phases': '', 'sat_frac': '', 'inf_time_mean_ms': '',
                             'note': why})
            continue
        res['phases'] = detect_phases(res, cfg)
        flights.append(res)

    if not flights:
        print("No readable flight."); return
    flights.sort(key=lambda f: (f['group'] != 'PPO', f['run']))

    for f in flights:
        print(f"  {f['group']:<5} {f['run']:<34} {f['duration']:6.2f} s, "
              f"{f['n_samples']} samples ({f['n_samples']/f['duration']:.1f} Hz), "
              f"{len(setpoint_phases(f['phases']))} mission setpoints")

    ref = flights[0]

    # ------------------------------------------------------- detected phases
    print(f"\n{'='*78}\nPHASES DETECTED FROM THE REFERENCE\n{'='*78}")
    for f in flights:
        print(f"\n  --- {f['group']}  {f['run']} ---")
        print(f"  {'phase':<14}{'t0':>7}{'t1':>7}{'dur':>7}   reference")
        for ph in f['phases']:
            if ph['type'] == 'hold':
                info = (f"pos=[{ph['pos'][0]:+.2f},{ph['pos'][1]:+.2f},"
                        f"{ph['pos'][2]:+.2f}] m  "
                        f"rpy=[{ph['rpy_deg'][0]:+6.1f},{ph['rpy_deg'][1]:+6.1f},"
                        f"{ph['rpy_deg'][2]:+6.1f}] deg")
            else:
                info = (f"dpos=[{ph['d_pos'][0]:+.2f},{ph['d_pos'][1]:+.2f},"
                        f"{ph['d_pos'][2]:+.2f}] m")
            print(f"  {ph['label']:<14}{ph['t0']:>7.2f}{ph['t1']:>7.2f}"
                  f"{ph['dur']:>7.2f}   {info}")
        sp = setpoint_phases(f['phases'])
        if sp:
            durs = [p['dur'] for p in sp]
            print(f"      mission setpoints: {', '.join(f'{d:.2f}' for d in durs)} s"
                  f"   ->  wait_s ~ {np.median(durs) - 0.7:.1f} s "
                  f"(minus 0.7 s of staging)")

    same_ref = True
    if len(flights) >= 2:
        print(f"\n{'='*78}\nCHECK: SAME REFERENCE IN THE TWO FLIGHTS?\n{'='*78}")
        fa, fb = flights[0], flights[1]
        pa, pb = setpoint_phases(fa['phases']), setpoint_phases(fb['phases'])
        if len(pa) != len(pb):
            same_ref = False
            print(f"  !! different number of setpoints: {len(pa)} vs {len(pb)}.")
            print("     The phase-by-phase comparison is not valid.")
        else:
            print(f"  {'setpoint':<12}{'pos diff':>12}{'rpy diff':>12}"
                  f"{'time shift':>12}")
            worst_p = worst_r = worst_t = 0.0
            for x, y in zip(pa, pb):
                dp = float(np.max(np.abs(x['pos'] - y['pos'])))
                dr = float(np.max(np.abs(x['rpy_deg'] - y['rpy_deg'])))
                dt_ = abs(x['t0'] - y['t0'])
                worst_p, worst_r = max(worst_p, dp), max(worst_r, dr)
                worst_t = max(worst_t, dt_)
                print(f"  {x['label']:<12}{dp*1000:>9.1f} mm{dr:>9.2f} deg"
                      f"{dt_:>9.2f} s")
            if worst_p < 0.01 and worst_r < 1.0:
                print("  -> the mission setpoints coincide: the comparison is on the")
                print("     same reference.")
            else:
                same_ref = False
                print("  -> !! the setpoints differ: the comparison is not on the")
                print("        same reference.")
            if worst_t > 0.5:
                print("  -> !! phases shifted by more than 0.5 s: compare by phase, not by")
                print("        absolute time.")

        ma = [p for p in fa['phases'] if p['label'] in ('TAKEOFF', 'LANDING')]
        mb = [p for p in fb['phases'] if p['label'] in ('TAKEOFF', 'LANDING')]
        if ma and len(ma) == len(mb):
            print(f"\n  {'manoeuvre':<12}{fa['group']:>26}{fb['group']:>26}")
            for x, y in zip(ma, mb):
                print(f"  {x['label']:<12}"
                      f"   dpos=[{x['d_pos'][0]:+.2f},{x['d_pos'][1]:+.2f},"
                      f"{x['d_pos'][2]:+.2f}] m"
                      f"   dpos=[{y['d_pos'][0]:+.2f},{y['d_pos'][1]:+.2f},"
                      f"{y['d_pos'][2]:+.2f}] m")
            dmax = max(float(np.max(np.abs(x['d_pos'] - y['d_pos'])))
                       for x, y in zip(ma, mb))
            if dmax < 0.02:
                print(f"  -> same commanded manoeuvre (max difference {dmax*1000:.1f} mm):")
                print("     takeoff and landing are comparable.")
            else:
                print(f"  !! different commanded displacements (max difference {dmax*1000:.0f} mm):")
                print("     the takeoff and landing comparison is indicative only.")

    bad, steps_per_flight = flag_comparability(flights, cfg)
    print(f"\n{'='*78}\nCOMPARABILITY OF THE TRANSIENTS\n{'='*78}")
    print(f"  {'phase':<14}" + "".join(f"{f['group']:>12}" for f in flights)
          + f"{'spread':>10}   result")
    for lab in [p['label'] for p in analysed_phases(ref['phases'])]:
        vals = [d.get(lab, (float('nan'),))[0] for d in steps_per_flight]
        if not all(v == v for v in vals):
            continue
        spread = max(vals) - min(vals)
        verdict = ("STEADY ONLY" if lab in bad else "ok")
        print(f"  {lab:<14}" + "".join(f"{v:>10.3f} m" for v in vals)
              + f"{spread*1000:>7.0f} mm   {verdict}")
    if bad:
        print("\n  The phases flagged STEADY ONLY have a step of different amplitude")
        print("  in the two flights: the vehicle cannot be put down at exactly the")
        print("  same point. On a longer step the error peak is larger by")
        print("  construction, so transient, peak, settling time, phase RMSE")
        print("  and phase TV are not comparable and are left blank. The")
        print("  STEADY-STATE error is kept: it depends on the final setpoint,")
        print("  which is the same, and not on the path to reach it. The same")
        print("  window is also excluded from the global metrics of the flight.")
    else:
        print("\n  Every phase has the same entering command in the two flights:")
        print("  transients comparable, nothing excluded.")

    print(f"\n  Whole-flight analysis: takeoff, setpoints and landing are all")
    print(f"  flown by the policy.")
    print(f"  The phases are evaluated one by one: a single RMSE would average")
    print(f"  ramp tracking, step response and disturbance rejection at a")
    print(f"  fixed point, weighted by the durations chosen for the mission.")

    for f in flights:
        m = np.ones(f['n_samples'], dtype=bool)
        for ph in analysed_phases(f['phases']):
            if ph.get('comparable', True):
                continue
            idx = np.arange(ph['i0'], ph['i1'] + 1)
            tt = f['t'][idx]
            m[idx[tt < (tt[-1] - cfg['t_ss'])]] = False
        f['mask'] = m
        tv = compute_tv(f, m)
        if tv is not None:
            f['tv_per_motor'] = tv

    for f in flights:
        manifest.append({
            'run': f['run'], 'group': f['group'], 'included': 1,
            'duration': f"{f['duration']:.4f}",
            'dt_mean_ms': f"{f['dt_mean']*1e3:.3f}", 'n_samples': f['n_samples'],
            'n_phases': len(setpoint_phases(f['phases'])),
            'sat_frac': f"{f['sat_frac']:.5f}" if 'sat_frac' in f else "",
            'inf_time_mean_ms': (f"{f['inf_time_mean_ms']:.3f}"
                                 if 'inf_time_mean_ms' in f else ""),
            'note': "; ".join(f['warnings']),
        })
    with open(dirs['tables'] / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
        w.writeheader(); w.writerows(manifest)

    with open(dirs['tables'] / "params.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(['param', 'value'])
        for k in ('cat_delta', 't_ss',
                  'step_tol_pos', 'step_tol_ang', 'settle_band',
                  'tol_pos', 'tol_att_deg', 'min_step_norm'):
            v = cfg.get(k)
            w.writerow([k, '' if v is None or v != v else v])
    warned = [m for m in manifest if m['included'] and m['note']]
    if warned:
        print(f"  !! {len(warned)} flights with warnings (see tables/manifest.csv)")

    # ====================================================== RMSE + Rollouts
    per_flight_rows = []
    t_lo = min(float(f['t'][0]) for f in flights)
    t_hi = max(float(f['t'][-1]) for f in flights)

    flight_rows = []

    for key, sym, unit, title, fname in METRICS:
        if not any(key in f for f in flights):
            print(f"  [skip] metric '{key}': missing in every flight.")
            continue

        for stat, agg in STATS:
            fig, ax = plt.subplots(figsize=(11, 6))
            for f in flights:
                if key not in f:
                    continue
                val = np.asarray(f[key], float)
                mk = np.ones(f['n_samples'], dtype=bool)
                r = agg(f['t'], val, mk)
                flight_rows.append({'metric': title, 'group': f['group'],
                                    'stat': stat, 'rmse': r,
                                    'n_samples': int(mk.sum())})
                ax.plot(f['t'], np.abs(val), color=COL[f['group']], lw=1.4,
                        label=rf'$\mathrm{{{f["group"]}}}\ '
                              rf'(\mathrm{{{stat}}}\ {r:.4f})$')
            mark_phases(ax, ref)
            ax.set_title(f'Absolute {title}')
            _s = sym if key.endswith('norm') else rf'\left|{sym}\right|'
            ax.set_ylabel(rf'${_s}$   [{unit}]')
            ax.set_xlabel(r'$t$   [s]')
            ax.set_ylim(bottom=0)
            ax.legend(loc='best', frameon=True)
            tight_x(ax, t_lo, t_hi)
            fig.tight_layout()
            save(fig, dirs[f'flight_{stat}'] / f"{fname}.pdf")

        for f in flights:
            if key not in f:
                continue
            r = rms_t(f['t'], np.asarray(f[key], float), f['mask'])
            per_flight_rows.append({'run': f['run'], 'group': f['group'],
                                    'metric': title, 'rmse': f"{r:.6f}",
                                    'n_samples': int(f['mask'].sum())})


    # =============================================== Rollouts: tracking
    TRACK = [
        ('pos', 'pos_ref', 0, r'$x$   [m]',        'X',     '1_x'),
        ('pos', 'pos_ref', 1, r'$y$   [m]',        'Y',     '2_y'),
        ('pos', 'pos_ref', 2, r'$z$   [m]',        'Z',     '3_z'),
        ('rpy', 'rpy_ref', 0, r'$\phi$   [rad]',   'roll',  '4_roll'),
        ('rpy', 'rpy_ref', 1, r'$\theta$   [rad]', 'pitch', '5_pitch'),
        ('rpy', 'rpy_ref', 2, r'$\psi$   [rad]',   'yaw',   '6_yaw'),
    ]
    for ka, kr, idx, ylab, comp, fn in TRACK:
        fig, ax = plt.subplots(figsize=(11, 6))
        solid_handles, ref_handles = [], []
        for f in flights:
            line, = ax.plot(f['t'], f[ka][:, idx], color=COL[f['group']], lw=2.0,
                             label=f['group'], alpha=0.92)
            solid_handles.append(line)
        for i_f, f in enumerate(flights):
            line, = ax.plot(f['t'], f[kr][:, idx], color=COL[f['group']], lw=2.4,
                             ls=DASH_REF[i_f % len(DASH_REF)], alpha=0.80,
                             zorder=3, label=f"{f['group']} ref")
            ref_handles.append(line)
        mark_phases(ax, ref)
        ax.set_ylabel(ylab)
        ax.set_xlabel(r'$t$   [s]')
        handles = solid_handles + ref_handles
        ax.legend(handles=handles, labels=[h.get_label() for h in handles],
                  loc='best', ncol=2, frameon=True)
        tight_x(ax, t_lo, t_hi)
        fig.tight_layout()
        save(fig, dirs['roll'] / f"{fn}.pdf")

    for mkey, ylab, comp, fn in (('v_norm', r'$\|v\|$   [m/s]', 'Linear velocity', '7_vel'),
                                 ('w_norm', r'$\|\omega\|$   [rad/s]', 'Angular velocity', '8_avel')):
        if not any(mkey in f for f in flights):
            continue
        fig, ax = plt.subplots(figsize=(11, 6))
        for f in flights:
            if mkey not in f:
                continue
            ax.plot(f['t'], f[mkey], color=COL[f['group']], lw=2.0,
                    label=f['group'], alpha=0.92)
        mark_phases(ax, ref)
        ax.set_ylabel(ylab)
        ax.set_xlabel(r'$t$   [s]')
        ax.set_ylim(bottom=0)
        ax.legend(loc='best', frameon=True)
        tight_x(ax, t_lo, t_hi)
        fig.tight_layout()
        save(fig, dirs['roll'] / f"{fn}.pdf")

    if any('forces' in f for f in flights):
        with plt.rc_context({
            "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
            "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9,
            "xtick.major.size": 4.5, "ytick.major.size": 4.5, "grid.linewidth": 0.7,
        }):
            fig, axs = plt.subplots(3, 2, figsize=(11, 10), squeeze=False, sharex=True)
            handles_by_group = {}
            for i in range(N_ROTORS):
                ax = axs[i % 3][i // 3]
                for f in flights:
                    if 'forces' not in f:
                        continue
                    line, = ax.plot(f['t'], f['forces'][:, i], color=COL[f['group']], lw=1.8,
                                     label=f['group'], alpha=0.9)
                    handles_by_group[f['group']] = line
                mark_phases(ax, ref, lw=0.8)
                ax.set_ylabel(rf'$f_{{{i+1}}}$   [N]')
                ax.set_ylim(0, F_MAX)
                ax.set_yticks(np.arange(0, F_MAX + 1e-6, 2))
                if i % 3 == 2:
                    ax.set_xlabel(r'$t$   [s]')
            for ax in axs.ravel():
                tight_x(ax, t_lo, t_hi)
            fig.tight_layout(rect=(0, 0, 1, 0.93))
            order = [g for g in ('PPO', 'SPPO') if g in handles_by_group]
            fig.legend([handles_by_group[g] for g in order], order,
                       loc='upper center', bbox_to_anchor=(0.5, 0.995),
                       ncol=2, frameon=True, fontsize=19)
            save(fig, dirs['roll'] / "9_rollout_forces.pdf")

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    ref_handles = []
    for i_f, f in enumerate(flights):
        line, = ax.plot(f['pos_ref'][:, 0], f['pos_ref'][:, 1], f['pos_ref'][:, 2],
                         color=COL[f['group']], lw=2.0,
                         ls=DASH_REF[i_f % len(DASH_REF)], alpha=0.80,
                         label=f"{f['group']} ref")
        ref_handles.append(line)
    sp = setpoint_phases(ref['phases'])
    uniq = []
    for p in sp:
        if not any(np.linalg.norm(p['pos'] - q['pos']) < 0.03 for q in uniq):
            uniq.append(p)
    letters = ['C', 'D'] if len(uniq) >= 2 else ['D']
    for p, lab in zip(uniq, letters):
        ax.scatter(*p['pos'], color='k', s=55, marker='o', depthshade=False)
        ax.text(p['pos'][0], p['pos'][1], p['pos'][2] + 0.03, lab, fontsize=13)
    solid_handles = []
    for f in flights:
        line, = ax.plot(f['pos'][:, 0], f['pos'][:, 1], f['pos'][:, 2],
                         color=COL[f['group']], lw=1.4, label=f['group'], alpha=0.9)
        solid_handles.append(line)
    ax.set_xlabel('$x$   [m]', labelpad=12)
    ax.set_ylabel('$y$   [m]', labelpad=12)
    ax.set_zlabel('$z$   [m]', labelpad=8)
    ax.tick_params(axis='both', which='major', labelsize=10)
    handles = solid_handles + ref_handles
    ax.legend(handles=handles, labels=[h.get_label() for h in handles],
              loc='upper left')
    save(fig, dirs['roll'] / "10_trajectory_3d.pdf", pad_inches=0.3)

    sp_ref = setpoint_phases(ref['phases'])
    trans = [p for p in sp_ref[1:] if p.get('comparable', True)]
    if trans:
        n = len(trans)
        fig, axs = plt.subplots(2, n, figsize=(5.4 * n, 8.5), squeeze=False)
        for j, p_ref in enumerate(trans):
            for f in flights:
                p = next((q for q in setpoint_phases(f['phases'])
                          if q['label'] == p_ref['label']), None)
                if p is None:
                    continue
                sl = slice(p['i0'], p['i1'] + 1)
                tau = f['t'][sl] - p['t0']
                axs[0][j].plot(tau, f['e_pos_norm'][sl], color=COL[f['group']],
                               lw=1.6, label=f['group'])
                axs[1][j].plot(tau, f['e_att_norm'][sl],
                               color=COL[f['group']], lw=1.6, label=f['group'])
            axs[0][j].axhline(cfg['tol_pos'], color='#2e8b57', lw=1.1, ls='-.')
            axs[1][j].axhline(np.deg2rad(cfg['tol_att_deg']),
                              color='#2e8b57', lw=1.1, ls='-.')
            axs[0][j].set_title(
                f"{p_ref['label']}\nref=[{p_ref['pos'][0]:+.2f},"
                f"{p_ref['pos'][1]:+.2f},{p_ref['pos'][2]:+.2f}] m  "
                f"rpy=[{p_ref['rpy_deg'][0]:+.0f},{p_ref['rpy_deg'][1]:+.0f},"
                f"{p_ref['rpy_deg'][2]:+.0f}]", fontsize=12)
            for r_ in (0, 1):
                axs[r_][j].set_xlabel(r'$t - t_{\mathrm{transition}}$   [s]')
                axs[r_][j].set_ylim(bottom=0)
                axs[r_][j].margins(x=0)
                axs[r_][j].legend(loc='best')
        axs[0][0].set_ylabel(r'$\|e_p\|$   [m]')
        axs[1][0].set_ylabel(r'$\|e_R\|$   [rad]')
        fig.suptitle('Reference step responses (transitions realigned)',
                     y=1.01)
        fig.tight_layout()
        save(fig, dirs['roll'] / "11_step_response.pdf")

    with plt.rc_context({
        "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
        "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9,
        "xtick.major.size": 4.5, "ytick.major.size": 4.5, "grid.linewidth": 0.7,
    }):
        fig, axs = plt.subplots(3, 2, figsize=(11, 10), squeeze=False, sharex=True)
        legend_handles = None
        for i, (ka, kr, idx, ylab, comp, fn) in enumerate(TRACK):
            ax = axs[i % 3][i // 3]
            solid_handles, ref_handles = [], []
            for f in flights:
                line, = ax.plot(f['t'], f[ka][:, idx], color=COL[f['group']], lw=1.8,
                                 label=f['group'], alpha=0.92)
                solid_handles.append(line)
            for i_f, f in enumerate(flights):
                line, = ax.plot(f['t'], f[kr][:, idx], color=COL[f['group']], lw=2.2,
                                 ls=DASH_REF[i_f % len(DASH_REF)], alpha=0.80,
                                 zorder=3, label=f"{f['group']} ref")
                ref_handles.append(line)
            mark_phases(ax, ref, lw=0.8)
            ax.set_ylabel(ylab)
            if i % 3 == 2:
                ax.set_xlabel(r'$t$   [s]')
            legend_handles = solid_handles + ref_handles
        for ax in axs.ravel():
            tight_x(ax, t_lo, t_hi)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        labels = [h.get_label() for h in legend_handles]
        fig.legend(legend_handles, labels, loc='upper center',
                   bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=True, fontsize=19)
        save(fig, dirs['roll'] / "12_full_pose.pdf")

    # ================================================== TOTAL VARIATION
    tv_rows = []

    def tv_bars(per_motor, out_dir_key, wname):
        """Six figures, one per motor, and the summary, on one window."""
        if not all(g in per_motor and per_motor[g] is not None
                   for g in (f['group'] for f in flights[:2])):
            return 0
        width, n = 0.35, 0
        for i in range(N_ROTORS):
            fig, ax = plt.subplots(figsize=(6, 6))
            for gi, f in enumerate(flights[:2]):
                ax.bar([(gi - 0.5) * width], [per_motor[f['group']][i]], width,
                       label=f['group'], color=COL[f['group']],
                       edgecolor='none')
            ax.set_ylabel(r'$TV_{\omega}$   [rad/s$^2$]')
            ax.set_xticks([0]);
            ax.set_xticklabels([f'P{i+1}'], fontsize=16)
            ax.grid(axis='x', visible=False)
            ax.grid(axis='y', visible=True)
            ax.legend()
            fig.tight_layout()
            save(fig, dirs[out_dir_key] / f"tv_propeller_{i+1}.pdf")
            n += 1

        x = np.arange(N_ROTORS)
        fig, ax = plt.subplots(figsize=(13, 7))
        for gi, f in enumerate(flights[:2]):
            ax.bar(x + (gi - 0.5) * width, per_motor[f['group']], width,
                   label=f['group'], color=COL[f['group']], edgecolor='none')
        ax.set_ylabel(r'$TV_{\omega}$   [rad/s$^2$]')
        ax.set_xticks(x)
        ax.set_xticklabels([f'P{i+1}' for i in range(N_ROTORS)], fontsize=16)
        ax.grid(axis='x', visible=False)
        ax.grid(axis='y', visible=True)
        ax.legend()
        fig.tight_layout()
        save(fig, dirs[out_dir_key] / "summary_tv.pdf")
        for f in flights[:2]:
            for i in range(N_ROTORS):
                tv_rows.append({'window': wname, 'group': f['group'],
                                'channel': f'M{i+1}',
                                'tv': f"{per_motor[f['group']][i]:.4f}"})
            tv_rows.append({'window': wname, 'group': f['group'],
                            'channel': 'TOTAL',
                            'tv': f"{float(np.sum(per_motor[f['group']])):.4f}"})
        return n + 1

    if len(flights) >= 2 and all('forces' in f for f in flights[:2]):
        n_tv = 0
        n_tv += tv_bars({f['group']: compute_tv(f, np.ones(f['n_samples'], bool))
                         for f in flights}, 'flight_tv', 'flight')
        print(f"\n  {n_tv} TV figures written in flight/propellers")

        for f in flights[:2]:
            for i in range(N_ROTORS):
                tv_rows.append({'window': 'comparable_flight',
                                'group': f['group'], 'channel': f'M{i+1}',
                                'tv': f"{f['tv_per_motor'][i]:.4f}"})
            tv_rows.append({'window': 'comparable_flight', 'group': f['group'],
                            'channel': 'TOTAL',
                            'tv': f"{float(np.sum(f['tv_per_motor'])):.4f}"})

        print(f"  TV (sum over {N_ROTORS} propellers)"
              + (", non-comparable transients excluded:" if bad else ":"))
        for f in flights[:2]:
            print(f"    {f['group']:<5} {float(np.sum(f['tv_per_motor'])):8.2f}")

    # ============================================ metrics per phase
    rows = []
    for f in flights:
        sp = analysed_phases(f['phases'])
        for j in range(len(sp)):
            rows.append(phase_metrics(f, sp, j, cfg))

    def fnum(v, w=0, p=2):
        if isinstance(v, float) and v != v:
            return f"{'--':>{w}}" if w else '--'
        return f"{v:>{w}.{p}f}" if w else f"{v:.{p}f}"

    if rows:
        print(f"\n{'='*94}")
        print(f"METRICS PER PHASE   (steady state = last {cfg['t_ss']:.1f} s; "
              f"tolerances {cfg['tol_pos']*100:.0f} cm / {cfg['tol_att_deg']:.0f} deg)")
        print("  '--' = not comparable (step of different amplitude in the two flights)")
        print('=' * 94)
        print(f"  {'phase':<12}{'grp':<6}{'RMSE pos':>10}{'steady':>9}{'peak':>8}"
              f"{'set.[s]':>9}{'RMSE att':>10}{'steady':>9}{'peak':>8}{'set.[s]':>9}")
        print(f"  {'':<12}{'':<6}{'[m]':>10}{'[m]':>9}{'[m]':>8}{'':>9}"
              f"{'[rad]':>10}{'[rad]':>9}{'[rad]':>8}{'':>9}")
        for ph in dict.fromkeys(r['phase'] for r in rows):
            for r in [x for x in rows if x['phase'] == ph]:
                print(f"  {r['phase']:<12}{r['group']:<6}"
                      f"{fnum(r['rmse_pos_all'], 10, 4)}"
                      f"{r['rmse_pos_steady']:>9.4f}"
                      f"{fnum(r['peak_pos'], 8, 3)}"
                      f"{fnum(r['settle_pos_s'], 9, 2)}"
                      f"{fnum(r['rmse_att_all'], 10, 4)}"
                      f"{r['rmse_att_steady']:>9.4f}"
                      f"{fnum(r['peak_att'], 8, 4)}"
                      f"{fnum(r['settle_att_s'], 9, 2)}")
            print()

    norm_rows = [r for r in rows if r['norm_base_m'] == r['norm_base_m']]
    if norm_rows:
        print(f"{'='*94}")
        print("TRANSIENT NORMALISED BY THE STEP AMPLITUDE")
        print("  A longer step gives a larger error by construction. Divided")
        print("  by the amplitude the transient becomes comparable: in the")
        print("  linear regime the step response scales with the amplitude,")
        print("  so e(t)/Delta does not depend on Delta.")
        print("  This holds while the actuators do not saturate; the")
        print("  steady-state error is reported next to it.")
        print('=' * 94)
        print(f"  {'phase':<12}{'grp':<6}{'step':>10}{'peak/D':>10}"
              f"{'RMSEtr/D':>10}{'IAE/D':>10}{'peakAtt/D':>12}")
        print(f"  {'':<12}{'':<6}{'[m]':>10}{'[-]':>10}{'[-]':>10}{'[-]':>10}"
              f"{'[rad/m]':>12}")
        for ph in dict.fromkeys(r['phase'] for r in norm_rows):
            sel = [x for x in norm_rows if x['phase'] == ph]
            for r in sel:
                print(f"  {r['phase']:<12}{r['group']:<6}"
                      f"{r['norm_base_m']:>10.3f}{r['peak_pos_norm']:>10.4f}"
                      f"{r['rmse_pos_transient_norm']:>10.4f}"
                      f"{r['iae_pos_norm']:>10.4f}{r['peak_att_norm']:>12.3f}")
            if len(sel) == 2:
                a_, b_ = sel[0], sel[1]
                d = {k: ((b_[k] - a_[k]) / a_[k] * 100 if a_[k] else float('nan'))
                     for k in ('peak_pos_norm', 'rmse_pos_transient_norm',
                               'iae_pos_norm', 'peak_att_norm')}
                print(f"  {'':<12}{b_['group']+'/'+a_['group']:<6}{'':>10}"
                      f"{d['peak_pos_norm']:>+9.1f}%"
                      f"{d['rmse_pos_transient_norm']:>+9.1f}%"
                      f"{d['iae_pos_norm']:>+9.1f}%{d['peak_att_norm']:>+11.1f}%")
            print()

    phases_l = list(dict.fromkeys(r['phase'] for r in rows))
    groups_l = list(dict.fromkeys(r['group'] for r in rows))

    export_categories(flights, rows, dirs, input_dir.name, cfg)

    print('=' * 94)
    print("MANOEUVRES  -  step response, landing, coupling"
          "   (landing and coupling: tables/assets.csv only)")
    print('=' * 94)
    asset_rows = export_assets(flights, dirs, cfg, rows)
    if asset_rows:
        print(f"  {'section':14s}{'phase':13s}{'metric':20s}"
              + "".join(f"{g:>12s}" for g in GROUPS))
        seen = []
        for r in asset_rows:
            k = (r['section'], r['phase'], r['metric'])
            if k in seen:
                continue
            seen.append(k)
            vv = {x['group']: x['value'] for x in asset_rows
                  if (x['section'], x['phase'], x['metric']) == k}
            cells = "".join(
                (f"{vv[g]:12.4f}" if g in vv and vv[g] == vv[g] else f"{'--':>12s}")
                for g in GROUPS)
            print(f"  {r['section']:14s}{r['phase']:13s}{r['metric']:20s}{cells}")

    def dump(name, rws, fields):
        if not rws:
            return
        with open(dirs['tables'] / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            for r in rws:
                w.writerow({k: nan_to_blank(v) for k, v in r.items()})

    dump("per_flight_rmse.csv", per_flight_rows,
         ['run', 'group', 'metric', 'rmse', 'n_samples'])
    if rows:
        dump("phase_metrics.csv", rows, list(rows[0].keys()))

    for _rws, _name in ((flight_rows, "summary_flight.csv"),):
      if _rws:
        with open(dirs['tables'] / _name, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(['metric', 'stat', 'ppo', 'sppo', 'delta_pct',
                        'n_samples_ppo', 'n_samples_sppo',
                        't0_s', 't1_s'])
            _lo, _hi = (t_lo, t_hi)
            for t_ in dict.fromkeys(r['metric'] for r in _rws):
                for st_ in [st for st, _a in STATS]:
                    g = {r['group']: r for r in _rws
                         if r['metric'] == t_ and r['stat'] == st_}
                    if 'PPO' not in g or 'SPPO' not in g:
                        continue
                    a, b = g['PPO']['rmse'], g['SPPO']['rmse']
                    dp = ((b - a) / a * 100) if a == a and a else float('nan')
                    w.writerow([t_, st_, f"{a:.6f}", f"{b:.6f}", f"{dp:.2f}",
                                g['PPO']['n_samples'], g['SPPO']['n_samples'],
                                f"{_lo:.3f}", f"{_hi:.3f}"])

    if tv_rows:
        with open(dirs['tables'] / "tv_windows.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=['window', 'group', 'channel', 'tv'])
            w.writeheader(); w.writerows(tv_rows)

    with open(dirs['tables'] / "results.tex", "w") as fh:
        fh.write(f"% {input_dir.name}: one flight per algorithm, same reference.\n")
        fh.write(f"% Steady state = last {cfg['t_ss']:.1f} s of each phase.\n")
        fh.write("% Single values: n = 1 per condition.\n")
        if bad:
            fh.write("% '--': transient not comparable (step amplitude\n")
            fh.write("%       different in the two flights). Phases: "
                     + ", ".join(sorted(bad)) + "\n")
        fh.write("\\begin{tabular}{llcccc}\n\\toprule\n")
        fh.write("Phase & Policy & RMSE $\\|e_p\\|$ [m] & peak [m] & "
                 "RMSE $\\|e_R\\|$ [rad] & peak [rad] \\\\\n")
        fh.write(" & & (steady) & & (steady) & \\\\\n\\midrule\n")
        for iph, ph in enumerate(phases_l):
            for r in [x for x in rows if x['phase'] == ph]:
                fh.write(f"{ph} & {r['group']} & {r['rmse_pos_steady']:.4f} & "
                         f"{fnum(r['peak_pos'], 0, 3)} & "
                         f"{r['rmse_att_steady']:.4f} & "
                         f"{fnum(r['peak_att'], 0, 4)} \\\\\n")
            if iph < len(phases_l) - 1:
                fh.write("\\midrule\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    if norm_rows:
        with open(dirs['tables'] / "results_transient_norm.tex", "w") as fh:
            fh.write(f"% {input_dir.name}: transient normalised "
                     "by the step amplitude.\n")
            fh.write("% e(t)/Delta does not depend on the step amplitude "
                     "in the linear\n")
            fh.write("% regime. "
                     "Only phases reached through a step\n")
            fh.write("% (takeoff and landing are ramps).\n")
            fh.write("\\begin{tabular}{llccccc}\n\\toprule\n")
            fh.write("Phase & Policy & $\\Delta$ [m] & peak/$\\Delta$ [-] & "
                     "RMSE$_{tr}$/$\\Delta$ [-] & IAE/$\\Delta$ [-] & "
                     "peak $\\|e_R\\|$/$\\Delta$ [rad/m] \\\\\n\\midrule\n")
            phl = list(dict.fromkeys(r['phase'] for r in norm_rows))
            for iph, ph in enumerate(phl):
                for r in [x for x in norm_rows if x['phase'] == ph]:
                    fh.write(f"{ph} & {r['group']} & {r['norm_base_m']:.3f} & "
                             f"{r['peak_pos_norm']:.4f} & "
                             f"{r['rmse_pos_transient_norm']:.4f} & "
                             f"{r['iae_pos_norm']:.4f} & "
                             f"{r['peak_att_norm']:.3f} \\\\\n")
                if iph < len(phl) - 1:
                    fh.write("\\midrule\n")
            fh.write("\\bottomrule\n\\end{tabular}\n")

    if len(flights) == 2 and same_ref and rows:
        a, b = flights[0]['group'], flights[1]['group']
        print(f"{'='*78}\n{b} WITH RESPECT TO {a}\n{'='*78}")
        print("  steady-state error, per phase:")
        for ph in phases_l:
            ra = next((r for r in rows if r['phase'] == ph and r['group'] == a), None)
            rb = next((r for r in rows if r['phase'] == ph and r['group'] == b), None)
            if not ra or not rb or not ra['rmse_pos_steady']:
                continue
            dp = (rb['rmse_pos_steady'] - ra['rmse_pos_steady']) / ra['rmse_pos_steady'] * 100
            da = (rb['rmse_att_steady'] - ra['rmse_att_steady']) / ra['rmse_att_steady'] * 100
            print(f"    {ph:<12} position {dp:+6.1f}%   attitude {da:+6.1f}%")
        print("\n  over the whole flight (RMSE, takeoff and ground tail included;")
        print("  descriptive only, the comparison is in maneuvers/):")
        for t_ in dict.fromkeys(r['metric'] for r in flight_rows):
            g = {r['group']: r['rmse'] for r in flight_rows
                 if r['metric'] == t_ and r['stat'] == 'RMSE'}
            if a in g and b in g and g[a]:
                print(f"    {t_:<28} {(g[b]-g[a])/g[a]*100:+6.1f}%")

    itm = {f['group']: f.get('inf_time_mean_ms') for f in flights}
    if itm and all(v is not None for v in itm.values()):
        dt_ms = np.mean([f['dt_mean'] for f in flights]) * 1e3
        print("\n  Inference time: "
              + ", ".join(f"{g} {v:.3f} ms" for g, v in itm.items())
              + f"  (control period {dt_ms:.2f} ms)")

    print("\n  One flight per condition: single values, without averages,")
    print("      bands or p-values. The phases are not independent samples.")
    print(f"\nAnalysis done. Output in: {out_dir}")
    print(f"  tables and manifest -> {dirs['tables']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Tilthex trajectory missions: PPO vs SPPO.")
    p.add_argument("--input", type=str, required=True,
                   help="mission folder (e.g. p2p_nominal)")
    p.add_argument("--t-ss", type=float, default=2.0,
                   help="duration [s] of the steady-state window at the end of each phase")
    p.add_argument("--tol-pos", type=float, default=0.05,
                   help="position tolerance [m] for the settling time")
    p.add_argument("--settle-band", type=float, default=0.05,
                   help="half-width of the settling band, as a fraction "
                        "of the entering step (default %(default)s)")
    p.add_argument("--settle-min-ang-deg", type=float, default=5.0,
                   help="commanded rotation below which the attitude settling "
                        "time is not measured (default %(default)s)")
    p.add_argument("--tol-att-deg", type=float, default=5.0,
                   help="attitude tolerance [deg] for the settling time")
    p.add_argument("--step-tol-pos", type=float, default=STEP_TOL_POS,
                   help="difference [m] above which the step of a phase is "
                        "considered different in the two flights")
    p.add_argument("--step-tol-ang-deg", type=float, default=STEP_TOL_ANG,
                   help="same, for the angular part [deg]")
    p.add_argument("--min-step-norm", type=float, default=MIN_STEP_NORM,
                   help="minimum step amplitude [m] for the normalisation "
                        "of the transient")
    p.add_argument("--v-thr", type=float, default=V_THR)
    p.add_argument("--w-thr", type=float, default=W_THR)
    p.add_argument("--min-block", type=float, default=MIN_BLOCK)
    p.add_argument("--step-pos", type=float, default=STEP_POS)
    p.add_argument("--step-ang-deg", type=float, default=2.0)
    p.add_argument("--delta", type=float, default=2.5, metavar="S",
                   help="seconds of transient after a step, for the "
                        "category section (default %(default)s)")
    p.add_argument("--cat-clip", action="store_true",
                   help="cut the y axis of the category figure when one "
                        "category dominates the others")
    p.add_argument("--cat-exclude", nargs="*", default=[], metavar="CAT",
                   help="categories removed from the figure (kept in the CSV)")
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.is_dir():
        print(f"Error: folder '{input_path}' not found.")
        raise SystemExit(1)

    export_analysis(input_path, {
        't_ss': args.t_ss, 'tol_pos': args.tol_pos, 'tol_att_deg': args.tol_att_deg,
        'settle_band': args.settle_band,
        'settle_min_ang_deg': args.settle_min_ang_deg,
        'step_tol_pos': args.step_tol_pos, 'step_tol_ang': args.step_tol_ang_deg,
        'min_step_norm': args.min_step_norm,
        'v_thr': args.v_thr, 'w_thr': args.w_thr, 'min_block': args.min_block,
        'step_pos': args.step_pos, 'step_ang': np.deg2rad(args.step_ang_deg),
        'cat_delta': args.delta, 'cat_clip': args.cat_clip,
        'cat_exclude': args.cat_exclude})