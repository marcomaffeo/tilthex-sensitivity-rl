#!/usr/bin/env python3
"""
Analysis of the hovering campaign: PPO vs SPPO on the Tilthex.

Data format
-----------
  * data['action'] holds the 6 rotor forces in newtons, one per rotor.
    In the campaign they stay in [0.3072, 10.9057] N, inside the limits
    [0.288, 11.25] N. The rotors of the Tilthex are tilted mechanically,
    so the action has no tilt channels.

  * data['quat'] and data['quat_ref'] are in WXYZ order (scalar first).
    Decoded this way, roll and pitch stay below 2 degrees in hovering.

  * The flights are position holds: pos_ref and quat_ref are constant
    within a flight (the setpoint is captured when the policy starts), so
    the RMSE is computed over the whole window.

  * The setpoint changes from flight to flight (x, y, z and yaw): errors
    are averaged across flights, absolute positions and angles are not.

Usage::

    python analyze_campaign.py --input hovering_campaign
    python analyze_campaign.py --input hovering_campaign --t-crop 9.99
    python analyze_campaign.py --input hovering_campaign --band iqr
"""

import argparse
import csv
import re
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy import stats

# --------------------------------------------------------------- constants
CF = 11.25e-4            # thrust coefficient [N / (rad/s)^2]
F_MIN, F_MAX = 0.288, 11.25   # force limits per rotor [N]
N_ROTORS = 6

C_PPO, C_SPPO = '#ff7f0e', '#1f77b4'  # orange / blue

# Evaluation horizon.
# None  -> taken from the data as the shortest included flight, and printed.
# float -> fixed value, so that the numbers do not change when a flight
# is added to or removed from the folder.
T_EVAL_FIXED = None
EVAL_RATE_HZ = 100.0

# Typical folder name:  PPO_10_20260511_174030
RUN_RE = re.compile(r'^(SPPO|PPO)[_-](.+?)[_-](\d{8})[_-](\d{6})$', re.IGNORECASE)

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


def to_scipy_quat(q_wxyz):
    """Reorder quaternions from the log order (w, x, y, z) to the SciPy order (x, y, z, w).

    Args:
        q_wxyz (numpy.ndarray): Quaternions, shape ``(N, 4)``.

    Returns:
        numpy.ndarray: Quaternions (x, y, z, w), shape ``(N, 4)``.
    """
    q = np.asarray(q_wxyz, dtype=float)
    return np.column_stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]])


def wilson(k, n, z=1.96):
    """Wilson confidence interval of a proportion. With few flights per group
    and p close to 1 the normal approximation does not hold.

    Args:
        k (int): Number of successes.
        n (int): Number of trials.
        z (float): Quantile of the normal distribution (1.96 for 95%).

    Returns:
        tuple[float, float]: Lower and upper bound of the interval, in ``[0, 1]``.
    """
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ===========================================================================
# Metrics.
# ===========================================================================
# key, symbol (without norm and unit), unit, title, file, official
METRICS = [
    # --- position in the REFERENCE frame ---
    ('e_bx',     r'e_x^{\,b}',    'm',     'Position Error X (body)',  '1_err_bx',    True),
    ('e_by',     r'e_y^{\,b}',    'm',     'Position Error Y (body)',  '2_err_by',    True),
    ('e_bz',     r'e_z^{\,b}',    'm',     'Position Error Z (body)',  '3_err_bz',    True),
    # --- attitude: rotation vector of the relative error, in the body frame ---
    ('e_roll',   r'e_\phi',       'rad',   'Attitude Error Roll',      '4_err_roll',  True),
    ('e_pitch',  r'e_\theta',     'rad',   'Attitude Error Pitch',     '5_err_pitch', True),
    ('e_yaw',    r'e_\psi',       'rad',   'Attitude Error Yaw',       '6_err_yaw',   True),
    # --- norms: independent of the frame ---
    ('e_pos_norm', r'\|e_p\|',    'm',     'Total Position Error',     '7_err_pos',   True),
    ('e_att_norm', r'\|e_R\|',    'rad',   'Total Attitude Error',     '8_err_att',   True),
    # --- velocities: zero reference in hold, independent of the setpoint ---
    ('v_norm',   r'\|v\|',        'm/s',   'Linear Velocity Norm',     '9_vel',       True),
    ('w_norm',   r'\|\omega\|',   'rad/s', 'Angular Velocity Norm',    '10_avel',     True),
]

# long title -> symbol, for the x axis of the summaries
SHORT = {t: rf'${s}$' for _k, s, _u, t, _f, _o in METRICS}


# ===========================================================================
def process_log(npz_path: Path, cfg):
    """Load one hovering flight and compute the tracking errors.

    The position error is expressed in the reference frame, the attitude
    error as the rotation vector of ``R_ref^T R``. Every discarded file has a
    reason, written in the manifest and in the printed summary.

    Args:
        npz_path (pathlib.Path): ``data.npz`` of the flight.
        cfg (dict): Analysis settings (``t_success``).

    Returns:
        tuple[dict | None, str | None]: ``(flight, None)``, or ``(None, reason)``
        when the file is discarded.
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        return None, f"load_error: {e}"

    keys = set(data.files)
    if 't' not in keys:
        return None, "missing key 't'"

    t = np.asarray(data['t'], dtype=float)
    if t.size < 2:
        return None, "time series with fewer than 2 samples"
    t = t - t[0]
    if np.any(np.diff(t) <= 0):
        return None, "time stamps not increasing (np.interp cannot be used)"

    folder = npz_path.parent.name
    pname = str(data['policy_name']) if 'policy_name' in keys else ""
    tag = (folder + "_" + pname).upper()
    if "SPPO" in tag:
        group = "SPPO"
    elif "PPO" in tag:
        group = "PPO"
    else:
        return None, "unknown group (neither PPO nor SPPO)"

    # policy_id: needed for the per-policy aggregation and the paired test
    m = RUN_RE.match(folder)
    if m:
        policy_id = m.group(2)
    else:
        policy_id = re.sub(r'^S?PPO[_-]?', '', pname, flags=re.IGNORECASE) or "?"

    dts = np.diff(t)
    res = {
        'file': str(npz_path), 'run': folder, 'group': group,
        'policy_id': policy_id, 'policy_name': pname,
        't': t, 'duration': float(t[-1]),
        'dt_mean': float(dts.mean()), 'n_samples': int(t.size),
        'success': bool(t[-1] >= cfg['t_success']),
        'warnings': [],
    }

    need = {'pos', 'quat', 'pos_ref', 'quat_ref'}
    if need <= keys:
        pos = np.asarray(data['pos'], dtype=float)
        pos_ref = np.asarray(data['pos_ref'], dtype=float)
        e_w = pos - pos_ref

        R_act = Rotation.from_quat(to_scipy_quat(data['quat']))
        R_ref = Rotation.from_quat(to_scipy_quat(data['quat_ref']))

        # Position error rotated into the reference frame. The reference yaw
        # spans about 290 degrees across the flights, so in the world frame
        # e_x and e_y point in different physical directions from flight to
        # flight and cannot be averaged.
        e_b = R_ref.inv().apply(e_w)
        res['e_bx'], res['e_by'], res['e_bz'] = e_b.T
        res['e_pos_norm'] = np.linalg.norm(e_w, axis=1)   # frame-invariant

        # Attitude error as the log map of the relative rotation. The reference
        # has yaw up to 170 degrees and is not level (roll 2.8 degrees, pitch
        # -0.9), so Euler angle differences would be wrong. The rotation vector
        # lives in the body frame and its norm is the geodesic distance.
        rv = (R_ref.inv() * R_act).as_rotvec()
        res['e_roll'], res['e_pitch'], res['e_yaw'] = rv.T
        res['e_att_norm'] = np.linalg.norm(rv, axis=1)

        # absolute state, used only for the plot of the representative flight
        res['pos_abs'] = pos
        res['setpoint'] = pos_ref[0].copy()
        res['ref_yaw_deg'] = float(np.rad2deg(R_ref[0].as_euler('xyz')[2]))
        res['ref_is_constant'] = bool(pos_ref.std(axis=0).max() < 1e-9)
    else:
        res['warnings'].append(f"missing {sorted(need - keys)}: no error metrics")

    # vel and avel come from the state estimator (the same quantities the policy
    # observes), so the position is not differentiated numerically. In a hold
    # the velocity reference is zero for every flight: these are the only
    # metrics independent of the setpoint. In trajectory tracking the velocity
    # reference is not zero and the logs have no 'vel_ref'.
    if 'vel' in keys:
        res['v_norm'] = np.linalg.norm(np.asarray(data['vel'], dtype=float), axis=1)
    else:
        res['warnings'].append("missing 'vel'")
    if 'avel' in keys:
        res['w_norm'] = np.linalg.norm(np.asarray(data['avel'], dtype=float), axis=1)
    else:
        res['warnings'].append("missing 'avel'")
    if 'inf_time' in keys:
        it = np.asarray(data['inf_time'], dtype=float)
        res['inf_time_mean_ms'] = float(np.mean(it) * 1e3)
        res['inf_time_max_ms'] = float(np.max(it) * 1e3)

    if 'action' in keys:
        a = np.atleast_2d(np.asarray(data['action'], dtype=float))
        if a.shape[0] != t.size:
            res['warnings'].append(f"action {a.shape} not aligned with t ({t.size})")
        elif a.shape[1] < N_ROTORS:
            res['warnings'].append(f"action has {a.shape[1]} channels, {N_ROTORS} expected")
        else:
            # Raw forces: saturation is computed on them, the clip is applied only
            # inside the square root for the rotor speed.
            res['forces'] = a[:, :N_ROTORS]
            res['sat_frac'] = float(np.mean((a[:, :N_ROTORS] <= F_MIN * 1.001) |
                                            (a[:, :N_ROTORS] >= F_MAX * 0.999)))
    else:
        res['warnings'].append("missing 'action': no total variation")

    return res, None


def compute_tv(d, t_grid):
    """Total variation of the rotor speed, divided by the time span.

    Computed on the common time grid: the logging period varies between
    flights (6.47 to 10.00 ms), and on the native samples a flight logged
    at 154 Hz would look noisier than the same flight logged at 100 Hz.

    Args:
        d (dict): Flight from ``process_log``.
        t_grid (numpy.ndarray): Common time grid [s].

    Returns:
        numpy.ndarray | None: Total variation rate of each rotor [rad/s^2], or
        ``None`` without rotor data or with fewer than two samples on the grid.
    """
    if 'forces' not in d:
        return None
    f = np.column_stack([np.interp(t_grid, d['t'], d['forces'][:, i],
                                   left=np.nan, right=np.nan)
                         for i in range(d['forces'].shape[1])])
    ok = ~np.any(np.isnan(f), axis=1)
    if ok.sum() < 2:
        return None
    w = np.sqrt(np.clip(f[ok], F_MIN, F_MAX) / CF)
    tt = t_grid[ok]
    return np.sum(np.abs(np.diff(w, axis=0)), axis=0) / (tt[-1] - tt[0] + 1e-9)


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
        "legend.fontsize": 13,
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
        # cmr10 has no bold weight: a bold title makes matplotlib print
        # "findfont: Failed to find font weight bold" for every figure.
        "axes.titleweight": "normal",
    })


def save(fig, path):
    """Save a figure as PDF with a tight bounding box, so that labels are not cut.

    Args:
        fig (matplotlib.figure.Figure): Figure.
        path (pathlib.Path): Output file.
    """
    fig.savefig(path, format='pdf', transparent=True, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
def export_unified_analysis(input_dir: Path, cfg):
    """Run the whole analysis of the hovering campaign.

    Loads the flights, computes RMSE, total variation and success rate per
    policy, compares PPO and SPPO paired by policy index, and writes figures
    and tables under ``<input_dir>/data_analysis``.

    Args:
        input_dir (pathlib.Path): Campaign folder.
        cfg (dict): Analysis settings from the command line (``t_crop``,
            ``t_success``, ``band``).
    """
    set_style()

    out_dir = input_dir / "data_analysis"
    dirs = {
        'rmse': out_dir / "RMSE",
        'tv': out_dir / "TV",
        'roll': out_dir / "Rollouts",
        'paired': out_dir / "paired",
        'summary': out_dir / "summary",
        'tables': out_dir / "tables",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print(f"Scanning data.npz files in: {input_dir} ...")
    npz_files = sorted(input_dir.rglob("data.npz"))
    print(f"  {len(npz_files)} files found.\n")

    # ----------------------------------------------------- reading + manifest
    flights, manifest = [], []
    for f in npz_files:
        res, why = process_log(f, cfg)
        if res is None:
            manifest.append({'run': f.parent.name, 'group': '', 'policy_id': '',
                             'included': 0, 'success': '', 'duration': '',
                             'dt_mean_ms': '', 'n_samples': '', 'covers_grid': '',
                             'setpoint_x': '', 'setpoint_y': '', 'setpoint_z': '',
                             'ref_yaw_deg': '', 'sat_frac': '',
                             'inf_time_mean_ms': '', 'note': why})
            continue
        flights.append(res)

    if not flights:
        print("No valid flight."); return

    successful = [d for d in flights if d['success']]
    if not successful:
        print("No successful flight."); return

    # ------------------------------------------- common horizon
    common = min(d['duration'] for d in successful)
    if cfg['t_crop'] is not None:
        t_eval = cfg['t_crop']
        src = "from the command line"
    elif T_EVAL_FIXED is not None:
        t_eval = T_EVAL_FIXED
        src = "constant T_EVAL_FIXED"
    else:
        t_eval = np.floor(common * 100) / 100      # rounded down to 10 ms
        src = "from the data"

    n_pts = int(round(t_eval * EVAL_RATE_HZ)) + 1
    t_grid = np.linspace(0.0, t_eval, n_pts)

    not_covering = [d for d in successful if d['duration'] < t_eval]
    print(f"  Evaluation horizon T = {t_eval:.4f} s ({src}, "
          f"{EVAL_RATE_HZ:.0f} Hz, {n_pts} samples)")
    print(f"  Shortest included flight: {common:.4f} s")
    if not_covering:
        print(f"  !! {len(not_covering)}/{len(successful)} flights are shorter than T: the tail")
        print(f"     of the mean curves would be computed on a biased subset.")
        for d in not_covering[:5]:
            print(f"       {d['run']}  ({d['duration']:.4f} s)")
        print(f"     Suggested: --t-crop {np.floor(common*100)/100:.2f}")
    else:
        print(f"  All {len(successful)} included flights cover T: n(t) is constant.")
    if T_EVAL_FIXED is None and cfg['t_crop'] is None:
        print(f"  >> To keep the numbers fixed, set")
        print(f"     T_EVAL_FIXED = {t_eval:.2f}  at the top of the file.")

    # long flights: cropped like the others, but reported
    long_runs = [d for d in successful if d['duration'] > t_eval * 1.5]
    if long_runs:
        print(f"\n  {len(long_runs)} flights with an unusual duration, cropped to T like the others:")
        for d in long_runs:
            print(f"    {d['duration']:6.2f}s  {d['group']:<5} policy={d['policy_id']:<4} {d['run']}")

    # TV on the common grid
    for d in successful:
        tv = compute_tv(d, t_grid)
        if tv is not None:
            d['tv_per_motor'] = tv

    # ------------------------------------------------ full manifest
    for d in flights:
        sp = d.get('setpoint', [None] * 3)
        manifest.append({
            'run': d['run'], 'group': d['group'], 'policy_id': d['policy_id'],
            'included': 1, 'success': int(d['success']),
            'duration': f"{d['duration']:.4f}",
            'dt_mean_ms': f"{d['dt_mean']*1e3:.3f}", 'n_samples': d['n_samples'],
            'covers_grid': int(d['duration'] >= t_eval),
            'setpoint_x': f"{sp[0]:.4f}" if sp[0] is not None else "",
            'setpoint_y': f"{sp[1]:.4f}" if sp[1] is not None else "",
            'setpoint_z': f"{sp[2]:.4f}" if sp[2] is not None else "",
            'ref_yaw_deg': f"{d['ref_yaw_deg']:.2f}" if 'ref_yaw_deg' in d else "",
            'sat_frac': f"{d['sat_frac']:.5f}" if 'sat_frac' in d else "",
            'inf_time_mean_ms': f"{d['inf_time_mean_ms']:.3f}" if 'inf_time_mean_ms' in d else "",
            'note': "; ".join(d['warnings']),
        })
    manifest.sort(key=lambda r: (r['group'], str(r['policy_id']), r['run']))
    with open(dirs['tables'] / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
        w.writeheader(); w.writerows(manifest)

    n_skip = sum(1 for m in manifest if not m['included'])
    if n_skip:
        print(f"\n  !! {n_skip} files discarded (reasons in tables/manifest.csv)")
    warned = [m for m in manifest if m['included'] and m['note']]
    if warned:
        print(f"  !! {len(warned)} files included with warnings (see manifest.csv)")

    # -------------------------------------------- setpoint diagnostics
    sps = np.array([d['setpoint'] for d in successful if 'setpoint' in d])
    if len(sps):
        yaws = np.array([d['ref_yaw_deg'] for d in successful if 'ref_yaw_deg' in d])
        print(f"\n  Setpoints (one per flight):")
        print(f"    x [{sps[:,0].min():6.2f}, {sps[:,0].max():6.2f}] m   "
              f"y [{sps[:,1].min():6.2f}, {sps[:,1].max():6.2f}] m   "
              f"z [{sps[:,2].min():6.2f}, {sps[:,2].max():6.2f}] m")
        print(f"    reference yaw [{yaws.min():.1f}, {yaws.max():.1f}] deg "
              f"(range {yaws.max()-yaws.min():.0f} deg)")
        
        n_var_flights = [d['run'] for d in successful if not d.get('ref_is_constant', True)]
        if n_var_flights:
            print(f"    !! {len(n_var_flights)} flights with a varying reference: not pure holds.")
            for run_name in n_var_flights:
                print(f"       - {run_name}")

        # confounder check: is the height balanced between the two groups?
        z_ppo = np.array([d['setpoint'][2] for d in successful
                          if d['group'] == 'PPO' and 'setpoint' in d])
        z_spo = np.array([d['setpoint'][2] for d in successful
                          if d['group'] == 'SPPO' and 'setpoint' in d])
        if len(z_ppo) > 2 and len(z_spo) > 2:
            try:
                _, p_z = stats.mannwhitneyu(z_ppo, z_spo)
            except Exception:
                p_z = float('nan')
            print(f"    mean height: PPO {z_ppo.mean():.3f} m, SPPO {z_spo.mean():.3f} m "
                  f"(Mann-Whitney p={p_z:.3f})")
            if p_z < 0.05:
                print("    !! heights differ between the groups: the ground effect differs,")
                print("       and part of the observed difference may come from it.")

    # ============================================== SUCCESS RATE
    print(f"\n{'='*68}\nSUCCESS RATE\n{'='*68}")
    cnt = {g: sum(1 for d in flights if d['group'] == g) for g in ('PPO', 'SPPO')}
    cok = {g: sum(1 for d in successful if d['group'] == g) for g in ('PPO', 'SPPO')}
    ci = {}
    for g in ('PPO', 'SPPO'):
        lo, hi = wilson(cok[g], cnt[g]); ci[g] = (lo, hi)
        print(f"  {g:<5} {cok[g]:2d}/{cnt[g]:2d} = {cok[g]/max(cnt[g],1)*100:5.1f}%"
              f"   Wilson 95% CI [{lo*100:5.1f}%, {hi*100:6.1f}%]")
    try:
        _, p_fisher = stats.fisher_exact([[cok['PPO'], cnt['PPO'] - cok['PPO']],
                                          [cok['SPPO'], cnt['SPPO'] - cok['SPPO']]])
    except Exception:
        p_fisher = float('nan')
    print(f"  Fisher exact: p = {p_fisher:.4f}  "
          f"{'(significant)' if p_fisher < 0.05 else '(not significant)'}")
    if p_fisher >= 0.05:
        print("  !! The confidence intervals overlap: report the difference as a")
        print("     trend, together with the intervals.")

    fails = [d for d in flights if not d['success']]
    if fails:
        print("\n  Failed flights:")
        for d in sorted(fails, key=lambda x: x['duration']):
            print(f"    {d['duration']:6.2f}s  {d['group']:<5} "
                  f"policy={d['policy_id']:<4} {d['run']}")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    vals = [cok[g] / max(cnt[g], 1) * 100 for g in ('PPO', 'SPPO')]
    yerr = np.array([[vals[i] - ci[g][0] * 100 for i, g in enumerate(('PPO', 'SPPO'))],
                     [ci[g][1] * 100 - vals[i] for i, g in enumerate(('PPO', 'SPPO'))]])
    ax.bar(['PPO', 'SPPO'], vals, yerr=yerr, color=[C_PPO, C_SPPO], edgecolor='black',
           width=0.5, alpha=0.85, capsize=9)
    for i, g in enumerate(('PPO', 'SPPO')):
        ax.annotate(f'{vals[i]:.1f}%\n({cok[g]}/{cnt[g]})', xy=(i, vals[i]),
                    xytext=(0, 9), textcoords="offset points", ha='center')
    ax.set_ylabel('Success rate   [%]')
    ax.set_ylim(0, 122)
    ax.grid(axis='x', visible=False)
    ax.set_title(f'Success rate (Wilson 95% CI),  $p={p_fisher:.3f}$')
    fig.tight_layout()
    save(fig, dirs['summary'] / "summary_success_rate.pdf")

    # ============================== per-policy aggregation
    # The number of flights per policy varies (1, 2 or 4): a plain mean over
    # the flights would give more weight to the policies flown more often.
    # The mean is taken first within each policy, then across policies.
    def curves_by_policy(key, group):
        acc = defaultdict(list)
        for d in successful:
            if d['group'] == group and key in d:
                acc[d['policy_id']].append(
                    np.interp(t_grid, d['t'], d[key], left=np.nan, right=np.nan))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return {p: np.nanmean(np.array(v), axis=0) for p, v in acc.items()}

    def rmse_by_policy(key, group):
        acc = defaultdict(list)
        for d in successful:
            if d['group'] == group and key in d:
                v = np.interp(t_grid, d['t'], d[key], left=np.nan, right=np.nan)
                v = v[~np.isnan(v)]
                if v.size:
                    acc[d['policy_id']].append(float(np.sqrt(np.mean(v ** 2))))
        return {p: float(np.mean(r)) for p, r in acc.items()}

    def band(arr):
        """Centre and limits of the band, with ddof=1 as in the tables."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if cfg['band'] == 'iqr':
                c = np.nanmedian(arr, axis=0)
                lo = np.nanpercentile(arr, 25, axis=0)
                hi = np.nanpercentile(arr, 75, axis=0)
            else:
                c = np.nanmean(arr, axis=0)
                s = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(c)
                lo, hi = c - s, c + s
        return c, lo, hi

    band_lbl = 'median, IQR' if cfg['band'] == 'iqr' else r'mean \pm \sigma'

    per_flight_rows, per_policy_rows, summary_rows, paired_rows = [], [], [], []

    # ======================================================= curves + paired test
    for key, sym, unit, title, fname, official in METRICS:
        # A metric is computed when at least one flight has it.
        if not any(key in d for d in successful):
            print(f"  [skip] metric '{key}': missing in every flight.")
            continue

        d_rmse, d_curves = {}, {}
        for group in ('PPO', 'SPPO'):
            d_curves[group] = curves_by_policy(key, group)
            d_rmse[group] = rmse_by_policy(key, group)

        # ---------- 1) absolute value curve (RMSE folder) ----------
        fig, ax = plt.subplots(figsize=(10, 6))
        for group, color in (('PPO', C_PPO), ('SPPO', C_SPPO)):
            if not d_curves[group]:
                continue
            c, lo, hi = band(np.abs(np.array(list(d_curves[group].values()))))
            pr = d_rmse[group]
            mu = float(np.mean(list(pr.values())))
            sd = float(np.std(list(pr.values()), ddof=1)) if len(pr) > 1 else 0.0
            summary_rows.append({'metric': title, 'group': group, 'official': int(official),
                                 'n_policies': len(pr), 'mean': mu, 'std': sd})
            for pid, r in pr.items():
                per_policy_rows.append({'policy_id': pid, 'group': group, 'metric': title,
                                        'official': int(official), 'rmse': f"{r:.6f}"})
            ax.plot(t_grid, c, color=color, linewidth=2.5,
                    label=rf'$\mathrm{{{group}}}\ (\mathrm{{RMSE}}\ {mu:.4f}\pm{sd:.4f},'
                          rf'\ n_{{pol}}={len(pr)})$')
            # the quantity is non-negative: the lower band stops at zero
            ax.fill_between(t_grid, np.maximum(lo, 0.0), hi, color=color, alpha=0.2)
        # The curve is the absolute value averaged over the policies;
        # the RMSE per flight is in the legend.
        ax.set_title(f'Absolute {title}')
        _s = sym if key.endswith('norm') else rf'\left|{sym}\right|'
        ax.set_ylabel(rf'${_s}$   [{unit}]')
        ax.set_xlabel(r'$t$   [s]')
        ax.set_xlim([0, t_eval]); ax.set_ylim(bottom=0)
        ax.legend(loc='best', frameon=True)
        save(fig, dirs['rmse'] / f"{fname}.pdf")

        # ---------- 2) signed error (Rollouts folder) ----------
        # The signed error is comparable across flights because it is taken
        # with respect to the setpoint of each flight.
        fig, ax = plt.subplots(figsize=(10, 6))
        for group, color in (('PPO', C_PPO), ('SPPO', C_SPPO)):
            if not d_curves[group]:
                continue
            c, lo, hi = band(np.array(list(d_curves[group].values())))
            ax.plot(t_grid, c, color=color, linewidth=2.5, label=group)
            ax.fill_between(t_grid, lo, hi, color=color, alpha=0.2)
        ax.axhline(0.0, color='k', lw=1.0, ls=':')
        ax.set_title(f'Rollout - {title}')
        ax.set_ylabel(rf'${sym}$   [{unit}]')
        ax.set_xlabel(r'$t$   [s]'); ax.set_xlim([0, t_eval])
        ax.legend(loc='best', frameon=True)
        save(fig, dirs['roll'] / f"{fname}.pdf")

        # ---------- 3) paired comparison ----------
        # Wilcoxon signed-rank test on the PPO/SPPO pairs of the same policy
        # index, which accounts for the variability between policies.
        shared = sorted(set(d_rmse['PPO']) & set(d_rmse['SPPO']))
        if len(shared) >= 3:
            a = np.array([d_rmse['PPO'][p] for p in shared])
            b = np.array([d_rmse['SPPO'][p] for p in shared])
            try:
                _, pw = stats.wilcoxon(a, b)
            except Exception:
                pw = float('nan')
            med = float(np.median((b - a) / a * 100.0))
            nbet = int(np.sum(b < a))
            paired_rows.append({'metric': title, 'official': int(official),
                                'n_pairs': len(shared),
                                'ppo_mean': round(float(a.mean()), 6),
                                'sppo_mean': round(float(b.mean()), 6),
                                'median_delta_pct': round(med, 2),
                                'sppo_better': nbet,
                                'p_wilcoxon': float(f"{pw:.6g}"),
                                'policies': " ".join(map(str, shared))})

            fig, ax = plt.subplots(figsize=(6.6, 6.6))
            # Axes centred on the data: what matters is the position with respect
            # to the diagonal.
            v_lo = min(a.min(), b.min())
            v_hi = max(a.max(), b.max())
            pad = max((v_hi - v_lo) * 0.18, v_hi * 0.04)
            lo_ax, hi_ax = max(0.0, v_lo - pad), v_hi + pad
            ax.plot([lo_ax, hi_ax], [lo_ax, hi_ax], 'k--', lw=1.2, zorder=1)
            ax.fill_between([lo_ax, hi_ax], [lo_ax, lo_ax], [lo_ax, hi_ax],
                            color=C_SPPO, alpha=.06, zorder=0)
            ax.scatter(a, b, s=75, color='#3a3a3a', edgecolor='white', zorder=3)
            for p, xx, yy in zip(shared, a, b):
                ax.annotate(str(p), (xx, yy), fontsize=9, xytext=(5, 5),
                            textcoords='offset points')
            ax.set_xlim(lo_ax, hi_ax); ax.set_ylim(lo_ax, hi_ax)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel(f'PPO   {title}   [{unit}]')
            ax.set_ylabel(f'SPPO   {title}   [{unit}]')
            ax.set_title(f'{nbet}/{len(shared)} policies in favour of SPPO,'
                         f'  $p={pw:.3f}$')
            fig.tight_layout()
            save(fig, dirs['paired'] / f"paired_{fname}.pdf")

        # ---------- 4) rows per flight ----------
        for d in successful:
            if key not in d:
                continue
            v = np.interp(t_grid, d['t'], d[key], left=np.nan, right=np.nan)
            v = v[~np.isnan(v)]
            if v.size:
                per_flight_rows.append({
                    'run': d['run'], 'group': d['group'], 'policy_id': d['policy_id'],
                    'metric': title, 'official': int(official),
                    'rmse': f"{float(np.sqrt(np.mean(v**2))):.6f}", 'n_samples': v.size})

    # ================================================== TOTAL VARIATION
    tv = {g: np.array([d['tv_per_motor'] for d in successful
                       if d['group'] == g and 'tv_per_motor' in d])
          for g in ('PPO', 'SPPO')}
    have_tv = all(len(tv[g]) for g in tv)
    if have_tv:
        m_tv = {g: tv[g].mean(axis=0) for g in tv}
        s_tv = {g: (tv[g].std(axis=0, ddof=1) if len(tv[g]) > 1 else np.zeros(N_ROTORS))
                for g in tv}
        width = 0.35

        for i in range(N_ROTORS):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.bar([-width/2], [m_tv['PPO'][i]], width, yerr=[s_tv['PPO'][i]],
                   label='PPO', color=C_PPO, edgecolor='black', capsize=5, alpha=0.85)
            ax.bar([width/2], [m_tv['SPPO'][i]], width, yerr=[s_tv['SPPO'][i]],
                   label='SPPO', color=C_SPPO, edgecolor='black', capsize=5, alpha=0.85)
            ax.set_title(f'Total variation - motor {i+1}')
            ax.set_ylabel(r'$TV_{\omega}$   [rad/s$^2$]')
            ax.set_xticks([0]); ax.set_xticklabels([f'M{i+1}']); ax.legend()
            ax.grid(axis='x', visible=False)
            fig.tight_layout()
            save(fig, dirs['tv'] / f"tv_motor_{i+1}.pdf")

        x = np.arange(N_ROTORS)
        fig, ax = plt.subplots(figsize=(13, 7))
        ax.bar(x - width/2, m_tv['PPO'], width, yerr=s_tv['PPO'], label='PPO',
               color=C_PPO, edgecolor='black', capsize=5, alpha=0.85)
        ax.bar(x + width/2, m_tv['SPPO'], width, yerr=s_tv['SPPO'], label='SPPO',
               color=C_SPPO, edgecolor='black', capsize=5, alpha=0.85)
        ax.set_title('Total variation - all motors')
        ax.set_ylabel(r'$TV_{\omega}$   [rad/s$^2$]')
        ax.grid(axis='x', visible=False)
        ax.set_xticks(x); ax.set_xticklabels([f'M{i+1}' for i in range(N_ROTORS)])
        ax.legend()
        save(fig, dirs['summary'] / "summary_tv.pdf")

        tot = {g: tv[g].sum(axis=1) for g in tv}
        try:
            _, p_tv = stats.mannwhitneyu(tot['PPO'], tot['SPPO'])
        except Exception:
            p_tv = float('nan')
        print(f"\n  Total TV (sum over {N_ROTORS} motors):")
        print(f"    PPO  {tot['PPO'].mean():8.2f} +/- {tot['PPO'].std(ddof=1):6.2f}")
        print(f"    SPPO {tot['SPPO'].mean():8.2f} +/- {tot['SPPO'].std(ddof=1):6.2f}"
              f"    Mann-Whitney p = {p_tv:.4f}")

    # ================================================== ROLLOUT: forces
    if any('forces' in d for d in successful):
        fig, axs = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
        axs_flat = axs.flatten()
        for i in range(N_ROTORS):
            for group, color in (('PPO', C_PPO), ('SPPO', C_SPPO)):
                acc = defaultdict(list)
                for d in successful:
                    if d['group'] == group and 'forces' in d:
                        acc[d['policy_id']].append(
                            np.interp(t_grid, d['t'], d['forces'][:, i],
                                      left=np.nan, right=np.nan))
                if not acc:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    pol = np.array([np.nanmean(np.array(v), axis=0) for v in acc.values()])
                c, lo, hi = band(pol)
                axs_flat[i].plot(t_grid, c, color=color, linewidth=2.0, label=group)
                axs_flat[i].fill_between(t_grid, lo, hi, color=color, alpha=0.2)
            axs_flat[i].axhline(F_MAX, color='k', ls=':', lw=1)
            axs_flat[i].axhline(F_MIN, color='k', ls=':', lw=1)
            axs_flat[i].set_title(f'Propeller {i+1} force')
            axs_flat[i].set_ylabel(r'$f$   [N]')
            axs_flat[i].set_xlabel(r'$t$   [s]')
            axs_flat[i].set_xlim([0, t_eval]); axs_flat[i].legend(loc='best')
        fig.tight_layout()
        save(fig, dirs['roll'] / "11_rollout_forces.pdf")

    # ====================== ROLLOUT: inter-run dispersion
    # Standard deviation ACROSS FLIGHTS of the position error: dispersion
    # between runs (air flow, battery, initial conditions). It is not the
    # sensitivity of the training objective; the empirical counterpart of
    # the sensitivity is the nominal -> perturbed degradation.
    if any('e_pos_norm' in d for d in successful):
        fig, ax = plt.subplots(figsize=(10, 6))
        for group, color in (('PPO', C_PPO), ('SPPO', C_SPPO)):
            arr = np.array([np.interp(t_grid, d['t'], d['e_pos_norm'],
                                      left=np.nan, right=np.nan)
                            for d in successful
                            if d['group'] == group and 'e_pos_norm' in d])
            if arr.size == 0:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                sd = np.nanstd(arr, axis=0, ddof=1)
            ax.plot(t_grid, sd, color=color, linewidth=2.5,
                    label=rf'$\mathrm{{{group}}}\ (n={arr.shape[0]})$')
            ax.fill_between(t_grid, np.zeros_like(sd), sd, color=color, alpha=0.12)
        # plain-text title: mathtext has no \mbox, and in \mathrm the hyphen
        # would become a minus sign
        ax.set_title('Inter-run dispersion of position error')
        ax.set_ylabel(r'$\sigma_{\mathrm{runs}}(\|e_p\|)$   [m]')
        ax.set_xlabel(r'$t$   [s]')
        ax.set_xlim([0, t_eval]); ax.set_ylim(bottom=0); ax.legend(loc='best')
        save(fig, dirs['roll'] / "12_interrun_dispersion.pdf")

    # ====================== ROLLOUT: representative flight (absolute state)
    # The absolute state cannot be averaged across flights, so a single
    # flight is shown: the one with the median position RMSE of its group.
    reps = {}
    for group in ('PPO', 'SPPO'):
        cand = []
        for d in successful:
            if 'e_pos_norm' in d and 'pos_abs' in d and d['duration'] >= t_eval:
                v = np.interp(t_grid, d['t'], d['e_pos_norm'])
                cand.append((float(np.sqrt(np.mean(v ** 2))), d))
        cand = [c for c in cand if c[1]['group'] == group]
        if cand:
            cand.sort(key=lambda c: c[0])
            reps[group] = cand[len(cand) // 2]
    if reps:
        print("\n  Representative flight (median position RMSE of the group):")
        fig, axs = plt.subplots(len(reps), 1, figsize=(10, 4.6 * len(reps)), squeeze=False)
        for ax, (group, (r, d)) in zip(axs.flatten(), reps.items()):
            print(f"    {group:<5} {d['run']}  (RMSE {r:.4f} m, setpoint "
                  f"[{d['setpoint'][0]:.2f}, {d['setpoint'][1]:.2f}, {d['setpoint'][2]:.2f}] m)")
            for j, (axis, col) in enumerate(zip("xyz", ('#1b6ca8', '#e08214', '#2e8b57'))):
                ax.plot(d['t'], d['pos_abs'][:, j], color=col, lw=1.8, label=f'${axis}$')
                ax.axhline(d['setpoint'][j], color=col, lw=1.0, ls='--', alpha=0.7)
            ax.set_xlim([0, t_eval]); ax.set_xlabel(r'$t$   [s]')
            ax.set_ylabel(r'position   [m]')
            ax.set_title(f'{group} - {d["run"]}   '
                         rf'(dashed = setpoint)', fontsize=13)
            ax.legend(loc='best', ncol=3)
        fig.tight_layout()
        save(fig, dirs['roll'] / "13_representative_flight.pdf")

    # ====================================== SUMMARY: global RMSE comparison
    official_titles = [m[3] for m in METRICS if m[5]]
    uniq = [t for t in official_titles if any(r['metric'] == t for r in summary_rows)]
    if uniq:
        get = lambda t_, g_, f_: next((r[f_] for r in summary_rows
                                       if r['metric'] == t_ and r['group'] == g_), 0.0)
        xm, width = np.arange(len(uniq)), 0.35
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.bar(xm - width/2, [get(t_, 'PPO', 'mean') for t_ in uniq], width,
               yerr=[get(t_, 'PPO', 'std') for t_ in uniq], label='PPO',
               color=C_PPO, capsize=5, alpha=0.85, edgecolor='black')
        ax.bar(xm + width/2, [get(t_, 'SPPO', 'mean') for t_ in uniq], width,
               yerr=[get(t_, 'SPPO', 'std') for t_ in uniq], label='SPPO',
               color=C_SPPO, capsize=5, alpha=0.85, edgecolor='black')
        ax.set_ylabel('RMSE (mean over the policies)')
        ax.set_title('Global RMSE comparison - PPO vs SPPO')
        ax.grid(axis='x', visible=False)
        ax.set_xticks(xm)
        ax.set_xticklabels([SHORT.get(t_, t_) for t_ in uniq])
        ax.legend(loc='best', frameon=True)
        save(fig, dirs['summary'] / "summary_rmse.pdf")

    # ================================================= EXPORT
    def dump(name, rows, fields):
        if not rows:
            return
        with open(dirs['tables'] / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
            w.writeheader(); w.writerows(rows)

    dump("per_flight_rmse.csv", per_flight_rows,
         ['run', 'group', 'policy_id', 'metric', 'official', 'rmse', 'n_samples'])
    dump("per_policy_rmse.csv", per_policy_rows,
         ['policy_id', 'group', 'metric', 'official', 'rmse'])
    dump("paired_stats.csv", paired_rows,
         ['metric', 'official', 'n_pairs', 'ppo_mean', 'sppo_mean',
          'median_delta_pct', 'sppo_better', 'p_wilcoxon', 'policies'])

    with open(dirs['tables'] / "summary_rmse.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(['metric', 'official', 'n_policies', 'ppo_mean', 'ppo_std',
                    'sppo_mean', 'sppo_std', 'delta_pct', 'sppo_better', 'p_wilcoxon'])
        for t_ in dict.fromkeys(r['metric'] for r in summary_rows):
            g = {r['group']: r for r in summary_rows if r['metric'] == t_}
            if 'PPO' not in g or 'SPPO' not in g:
                continue
            pr = next((r for r in paired_rows if r['metric'] == t_), None)
            dpct = ((g['SPPO']['mean'] - g['PPO']['mean']) / g['PPO']['mean'] * 100
                    if g['PPO']['mean'] else float('nan'))
            w.writerow([t_, g['PPO']['official'], g['PPO']['n_policies'],
                        f"{g['PPO']['mean']:.6f}", f"{g['PPO']['std']:.6f}",
                        f"{g['SPPO']['mean']:.6f}", f"{g['SPPO']['std']:.6f}",
                        f"{dpct:.2f}",
                        f"{pr['sppo_better']}/{pr['n_pairs']}" if pr else "",
                        f"{pr['p_wilcoxon']:.4f}" if pr else ""])

    if have_tv:
        with open(dirs['tables'] / "summary_tv.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(['channel', 'ppo_mean', 'ppo_std', 'sppo_mean', 'sppo_std'])
            for i in range(N_ROTORS):
                w.writerow([f'M{i+1}', f"{m_tv['PPO'][i]:.4f}", f"{s_tv['PPO'][i]:.4f}",
                            f"{m_tv['SPPO'][i]:.4f}", f"{s_tv['SPPO'][i]:.4f}"])
            w.writerow(['TOTAL', f"{tot['PPO'].mean():.4f}", f"{tot['PPO'].std(ddof=1):.4f}",
                        f"{tot['SPPO'].mean():.4f}", f"{tot['SPPO'].std(ddof=1):.4f}"])
            w.writerow(['p_mannwhitney_total', f"{p_tv:.6f}", '', '', ''])

    with open(dirs['tables'] / "success_rate.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(['group', 'n_total', 'n_success', 'rate_pct', 'wilson_lo', 'wilson_hi',
                    'p_fisher', 'mean_sat_frac', 'inf_time_mean_ms', 'inf_time_max_ms'])
        for g in ('PPO', 'SPPO'):
            sat = [d['sat_frac'] for d in flights if d['group'] == g and 'sat_frac' in d]
            itm = [d['inf_time_mean_ms'] for d in flights
                   if d['group'] == g and 'inf_time_mean_ms' in d]
            itx = [d['inf_time_max_ms'] for d in flights
                   if d['group'] == g and 'inf_time_max_ms' in d]
            w.writerow([g, cnt[g], cok[g], f"{cok[g]/max(cnt[g],1)*100:.2f}",
                        f"{ci[g][0]*100:.2f}", f"{ci[g][1]*100:.2f}", f"{p_fisher:.4f}",
                        f"{np.mean(sat):.5f}" if sat else "",
                        f"{np.mean(itm):.3f}" if itm else "",
                        f"{np.max(itx):.3f}" if itx else ""])

    # LaTeX table: official metrics only
    with open(dirs['tables'] / "results.tex", "w") as fh:
        fh.write("% RMSE averaged within each policy, then mean +/- std over the policies.\n")
        fh.write("% p: paired Wilcoxon signed-rank test on the PPO/SPPO pairs.\n")
        fh.write("\\begin{tabular}{lccrc}\n\\toprule\n")
        fh.write("Metric & PPO & SPPO & $\\Delta$ [\\%] & $p$ \\\\\n\\midrule\n")
        for t_ in uniq:
            g = {r['group']: r for r in summary_rows if r['metric'] == t_}
            if 'PPO' not in g or 'SPPO' not in g:
                continue
            pr = next((r for r in paired_rows if r['metric'] == t_), None)
            dpct = ((g['SPPO']['mean'] - g['PPO']['mean']) / g['PPO']['mean'] * 100
                    if g['PPO']['mean'] else float('nan'))
            ps = f"{pr['p_wilcoxon']:.3f}" if pr else "--"
            star = "$^{*}$" if pr and pr['p_wilcoxon'] < 0.05 else ""
            fh.write(f"{t_} & ${g['PPO']['mean']:.4f} \\pm {g['PPO']['std']:.4f}$ & "
                     f"${g['SPPO']['mean']:.4f} \\pm {g['SPPO']['std']:.4f}$ & "
                     f"${dpct:+.1f}$ & {ps}{star} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    # ------------------------------------------------------- printed summary
    print(f"\n{'='*68}\nPAIRED COMPARISON (Wilcoxon signed-rank over the policies)\n{'='*68}")
    print(f"  {'metric':<27}{'PPO':>9}{'SPPO':>10}{'delta%':>9}{'better':>9}{'p':>9}")
    for r in paired_rows:
        if not r['official']:
            continue
        star = " *" if r['p_wilcoxon'] < 0.05 else ""
        print(f"  {r['metric']:<27}{r['ppo_mean']:>9.4f}{r['sppo_mean']:>10.4f}"
              f"{r['median_delta_pct']:>+9.1f}"
              f"{r['sppo_better']:>6}/{r['n_pairs']:<3}{r['p_wilcoxon']:>8.4f}{star}")

    n_pol_tot = len({d['policy_id'] for d in flights})
    n_pairs = paired_rows[0]['n_pairs'] if paired_rows else 0
    if n_pairs < n_pol_tot:
        missing = sorted({d['policy_id'] for d in flights} -
                         set((paired_rows[0]['policies'].split() if paired_rows else [])))
        print(f"\n  !! {n_pol_tot - n_pairs} policies left out of the paired comparison "
              f"({', '.join(map(str, missing))}):")
        print("     one of the two groups has no successful flight. These are the")
        print("     policies where PPO fails, so leaving them out favours PPO in the")
        print("     RMSE. The success rate counts every flight.")

    itm = {g: [d['inf_time_mean_ms'] for d in flights
               if d['group'] == g and 'inf_time_mean_ms' in d] for g in ('PPO', 'SPPO')}
    if all(itm.values()):
        dt_ms = np.mean([d['dt_mean'] for d in successful]) * 1e3
        print(f"\n  Inference time: "
              f"PPO {np.mean(itm['PPO']):.3f} ms, SPPO {np.mean(itm['SPPO']):.3f} ms")
        print(f"    mean control period {dt_ms:.2f} ms -> "
              f"{np.mean(itm['SPPO'])/dt_ms*100:.1f}% of the period")

    print(f"\nAnalysis done. Output in: {out_dir}")
    print(f"  tables and manifest -> {dirs['tables']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Tilthex hovering campaign: RMSE, TV, rollouts, paired comparison.")
    p.add_argument("--input", type=str, required=True,
                   help="campaign folder (e.g. hovering_campaign)")
    p.add_argument("--t-crop", type=float, default=None,
                   help="evaluation horizon [s]; taken from the data when omitted")
    p.add_argument("--t-success", type=float, default=9.5,
                   help="minimum duration of a successful flight [s]")
    p.add_argument("--band", choices=['std', 'iqr'], default='std',
                   help="band around the curve: mean +/- sigma or median + IQR")
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.is_dir():
        print(f"Error: folder '{input_path}' not found.")
        raise SystemExit(1)

    export_unified_analysis(input_path, {'t_crop': args.t_crop,
                                         't_success': args.t_success,
                                         'band': args.band})