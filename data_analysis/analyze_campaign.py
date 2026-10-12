#!/usr/bin/env python3
"""
analyze_campaign.py  --  Analisi campagna TiltHex: PPO vs SPPO.

Riscrittura dello script originale con tutte le correzioni concordate.
Ogni modifica sostanziale e' marcata con  # [FIX n]  per il diff.

ASSUNZIONI SUL FORMATO DEI DATI (verificate su hovering_nominal, 44 voli)
-------------------------------------------------------------------------
  * data['action'] contiene 6 FORZE in NEWTON, una per rotore.
    Verificato: range [0.3072, 10.9057] N, nessuna saturazione ai limiti
    [0.288, 11.25] N. Il TiltHex ha i rotori inclinati meccanicamente,
    quindi non ci sono canali di tilt nell'azione.
    >>> Se un giorno rigeneri le policy con un'interfaccia diversa (azione
    >>> normalizzata in [-1,1], o con canali di tilt), QUESTO SCRIPT PRODUCE
    >>> NUMERI SBAGLIATI IN SILENZIO. Ricontrolla qui.

  * data['quat'] e data['quat_ref'] sono in convenzione WXYZ (scalare per
    primo). Verificato su tutti i 44 voli con test fisico: decodificando in
    wxyz roll e pitch restano sotto i 2 gradi, in xyzw risulterebbero decine
    di gradi di roll costante in hovering.
    >>> Stessa avvertenza: se cambia il nodo di logging, ricontrolla qui.

  * I voli sono HOLD, non reach: pos_ref e quat_ref sono costanti entro il
    volo (setpoint catturato all'attivazione della policy). Non c'e' quindi
    un transitorio iniziale da separare dal regime, e l'RMSE sull'intera
    finestra e' legittimo.

  * Il setpoint CAMBIA da volo a volo (x,y,z e yaw). Di conseguenza posizioni
    e angoli ASSOLUTI non sono mediabili tra voli: vedi [FIX 6].

Uso:
    python analyze_flights.py --input plots/hovering_nominal
    python analyze_flights.py --input plots/hovering_nominal --t-crop 9.99
    python analyze_flights.py --input plots/hovering_nominal --band iqr

NB: compare_conditions.py importa da questo file. Se lo usi, cambia la riga
    "from analyze_flights_v3 import ..."  ->  "from analyze_flights import ..."
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

# --------------------------------------------------------------- costanti
CF = 11.25e-4            # coefficiente di spinta [N / (rad/s)^2]
F_MIN, F_MAX = 0.288, 11.25   # limiti di forza per rotore [N]
N_ROTORS = 6

C_PPO, C_SPPO = '#ff7f0e', '#1f77b4'  # arancione / blu, come export_campaign.py

# [FIX 1] Orizzonte di valutazione.
# None  -> dedotto dai dati come min(durata dei voli inclusi), stampato a video.
# float -> valore congelato: usalo per i run definitivi della tesi, cosi' i
#          numeri non cambiano se aggiungi o togli un volo dalla cartella.
T_EVAL_FIXED = None
EVAL_RATE_HZ = 100.0

# Nome cartella tipico:  PPO_10_20260511_174030
RUN_RE = re.compile(r'^(SPPO|PPO)[_-](.+?)[_-](\d{8})[_-](\d{6})$', re.IGNORECASE)

_ESC_SPACE = "\\ "


def tex(s):
    """'Position Error X' -> 'Position\\ Error\\ X'.
    [FIX 9] Nell'originale il replace stava DENTRO l'f-string: sintassi valida
    solo da Python 3.12 (PEP 701), SyntaxError su 3.11 e precedenti."""
    return s.replace(" ", _ESC_SPACE)


def to_scipy_quat(q_wxyz):
    """wxyz (formato dei log) -> xyzw (formato che vuole scipy)."""
    q = np.asarray(q_wxyz, dtype=float)
    return np.column_stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]])


def wilson(k, n, z=1.96):
    """Intervallo di confidenza di Wilson per una proporzione.
    Con n~22 e p vicino a 1 l'intervallo normale e' inutilizzabile."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ===========================================================================
# Metriche.
# ===========================================================================
#   chiave, simbolo (senza modulo e senza unita'), unita', titolo, file, ufficiale
METRICS = [
    # --- posizione nel frame del RIFERIMENTO: ufficiali ---
    ('e_bx',     r'e_x^{\,b}',    'm',     'Position Error X (body)',  '1_err_bx',    True),
    ('e_by',     r'e_y^{\,b}',    'm',     'Position Error Y (body)',  '2_err_by',    True),
    ('e_bz',     r'e_z^{\,b}',    'm',     'Position Error Z (body)',  '3_err_bz',    True),
    # --- assetto: rotvec dell'errore relativo, gia' nel body frame ---
    ('e_roll',   r'e_\phi',       'rad',   'Attitude Error Roll',      '4_err_roll',  True),
    ('e_pitch',  r'e_\theta',     'rad',   'Attitude Error Pitch',     '5_err_pitch', True),
    ('e_yaw',    r'e_\psi',       'rad',   'Attitude Error Yaw',       '6_err_yaw',   True),
    # --- norme: invarianti rispetto al frame ---
    ('e_pos_norm', r'\|e_p\|',    'm',     'Total Position Error',     '7_err_pos',   True),
    ('e_att_norm', r'\|e_R\|',    'rad',   'Total Attitude Error',     '8_err_att',   True),
    # --- [FIX 4] velocita': riferimento nullo in hold, indipendenti dal setpoint ---
    ('v_norm',   r'\|v\|',        'm/s',   'Linear Velocity Norm',     '9_vel',       True),
    ('w_norm',   r'\|\omega\|',   'rad/s', 'Angular Velocity Norm',    '10_avel',     True),
]

# titolo lungo -> simbolo, per l'asse x dei riepiloghi
SHORT = {t: rf'${s}$' for _k, s, _u, t, _f, _o in METRICS}


# ===========================================================================
def process_log(npz_path: Path, cfg):
    """Ritorna (res, None) oppure (None, motivo_dello_scarto).

    [FIX 8] Nessun file sparisce in silenzio: ogni scarto ha un motivo che
    finisce nel manifest e nel riepilogo a video."""
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        return None, f"load_error: {e}"

    keys = set(data.files)
    if 't' not in keys:
        return None, "manca la chiave 't'"

    t = np.asarray(data['t'], dtype=float)
    if t.size < 2:
        return None, "serie temporale con meno di 2 campioni"
    t = t - t[0]
    if np.any(np.diff(t) <= 0):
        return None, "tempi non monotoni (np.interp non applicabile)"

    folder = npz_path.parent.name
    pname = str(data['policy_name']) if 'policy_name' in keys else ""
    tag = (folder + "_" + pname).upper()
    if "SPPO" in tag:
        group = "SPPO"
    elif "PPO" in tag:
        group = "PPO"
    else:
        return None, "gruppo non riconosciuto (ne' PPO ne' SPPO)"

    # [FIX 2] policy_id: indispensabile per l'aggregazione e il test appaiato
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

        # [FIX 5] errore di posizione ruotato nel frame del riferimento.
        # Lo yaw di riferimento spazia su ~290 gradi tra i voli: nel frame
        # mondo e_x ed e_y puntano in direzioni fisiche diverse volo per volo
        # e mediarli tra voli non ha significato.
        e_b = R_ref.inv().apply(e_w)
        res['e_bx'], res['e_by'], res['e_bz'] = e_b.T
        res['e_pos_norm'] = np.linalg.norm(e_w, axis=1)   # invariante

        # [FIX 3] errore di assetto = log-map dell'errore relativo.
        # L'originale faceva euler_act - euler_ref componente per componente:
        # valido solo per errori piccoli attorno a un assetto livellato, mentre
        # qui il riferimento ha yaw fino a 170 gradi e non e' livellato
        # (roll 2.8 gradi, pitch -0.9). In piu' quelle componenti erano
        # incoerenti con la norma, che era gia' calcolata via quaternioni.
        # rotvec vive nel body frame e la sua norma E' la distanza geodetica.
        rv = (R_ref.inv() * R_act).as_rotvec()
        res['e_roll'], res['e_pitch'], res['e_yaw'] = rv.T
        res['e_att_norm'] = np.linalg.norm(rv, axis=1)

        # stato assoluto: utile SOLO per il grafico del volo rappresentativo
        res['pos_abs'] = pos
        res['setpoint'] = pos_ref[0].copy()
        res['ref_yaw_deg'] = float(np.rad2deg(R_ref[0].as_euler('xyz')[2]))
        res['ref_is_constant'] = bool(pos_ref.std(axis=0).max() < 1e-9)
    else:
        res['warnings'].append(f"mancano {sorted(need - keys)}: nessuna metrica di errore")

    # [FIX 4] vel/avel erano presenti nei .npz e semplicemente ignorati.
    # Vengono dallo stimatore di stato (le stesse quantita' che osserva la
    # policy), quindi niente derivazione numerica della posizione e niente
    # amplificazione del rumore. In HOLD il riferimento di velocita' e' zero
    # per tutti i voli: sono le uniche metriche indipendenti dal setpoint.
    # ATTENZIONE: questo NON vale nel trajectory tracking, dove la velocita'
    # di riferimento non e' nulla e nei log non c'e' alcun 'vel_ref'.
    if 'vel' in keys:
        res['v_norm'] = np.linalg.norm(np.asarray(data['vel'], dtype=float), axis=1)
    else:
        res['warnings'].append("manca 'vel'")
    if 'avel' in keys:
        res['w_norm'] = np.linalg.norm(np.asarray(data['avel'], dtype=float), axis=1)
    else:
        res['warnings'].append("manca 'avel'")
    if 'inf_time' in keys:
        it = np.asarray(data['inf_time'], dtype=float)
        res['inf_time_mean_ms'] = float(np.mean(it) * 1e3)
        res['inf_time_max_ms'] = float(np.max(it) * 1e3)

    if 'action' in keys:
        a = np.atleast_2d(np.asarray(data['action'], dtype=float))
        if a.shape[0] != t.size:
            res['warnings'].append(f"action {a.shape} non allineata con t ({t.size})")
        elif a.shape[1] < N_ROTORS:
            res['warnings'].append(f"action ha {a.shape[1]} canali, attesi {N_ROTORS}")
        else:
            # [FIX 7] forze salvate GREZZE: l'originale salvava gia' il segnale
            # clippato, perdendo l'informazione sulla saturazione. Il clip serve
            # solo dentro la sqrt per la velocita' di rotore.
            res['forces'] = a[:, :N_ROTORS]
            res['sat_frac'] = float(np.mean((a[:, :N_ROTORS] <= F_MIN * 1.001) |
                                            (a[:, :N_ROTORS] >= F_MAX * 0.999)))
    else:
        res['warnings'].append("manca 'action': nessuna Total Variation")

    return res, None


def compute_tv(d, t_grid):
    """Total Variation della velocita' di rotore, normalizzata sul tempo.

    [FIX 10] Calcolata sulla griglia comune, non sui campioni nativi.
    Il passo di logging varia tra i voli da 6.47 a 10.00 ms: sommando |dw| sui
    campioni nativi, un volo a 154 Hz risulta piu' nervoso di uno identico a
    100 Hz. In piu' l'originale normalizzava per times[-1] sommando su TUTTA la
    durata, quindi i voli da 20-30 s avevano un orizzonte diverso dagli altri.
    """
    if 'forces' not in d:
        return None
    f = np.column_stack([np.interp(t_grid, d['t'], d['forces'][:, i],
                                   left=np.nan, right=np.nan)
                         for i in range(d['forces'].shape[1])])
    ok = ~np.any(np.isnan(f), axis=1)
    if ok.sum() < 2:
        return None
    w = np.sqrt(np.clip(f[ok], F_MIN, F_MAX) / CF)      # [rad/s]
    tt = t_grid[ok]
    return np.sum(np.abs(np.diff(w, axis=0)), axis=0) / (tt[-1] - tt[0] + 1e-9)


def set_style():
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
        # [FIX 11] cmr10 non ha il peso bold: con fontweight='bold' matplotlib
        # stampa "findfont: Failed to find font weight bold" a ogni titolo.
        "axes.titleweight": "normal",
    })


def save(fig, path):
    """[FIX 11] bbox_inches='tight' ovunque: nell'originale mancava sulle
    figure RMSE e Rollouts e le label venivano tagliate nel PDF."""
    fig.savefig(path, format='pdf', transparent=True, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
def export_unified_analysis(input_dir: Path, cfg):
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

    print(f"Scansione di data.npz in: {input_dir} ...")
    npz_files = sorted(input_dir.rglob("data.npz"))
    print(f"  {len(npz_files)} file trovati.\n")

    # ----------------------------------------------------- lettura + manifest
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
        print("Nessun volo valido."); return

    successful = [d for d in flights if d['success']]
    if not successful:
        print("Nessun volo riuscito."); return

    # ------------------------------------------- [FIX 1] orizzonte comune
    common = min(d['duration'] for d in successful)
    if cfg['t_crop'] is not None:
        t_eval = cfg['t_crop']
        src = "da riga di comando"
    elif T_EVAL_FIXED is not None:
        t_eval = T_EVAL_FIXED
        src = "costante T_EVAL_FIXED"
    else:
        t_eval = np.floor(common * 100) / 100      # arrotonda per difetto a 10 ms
        src = "dedotto dai dati"

    n_pts = int(round(t_eval * EVAL_RATE_HZ)) + 1
    t_grid = np.linspace(0.0, t_eval, n_pts)

    not_covering = [d for d in successful if d['duration'] < t_eval]
    print(f"  Orizzonte di valutazione T = {t_eval:.4f} s ({src}, "
          f"{EVAL_RATE_HZ:.0f} Hz, {n_pts} campioni)")
    print(f"  Durata minima tra i voli inclusi: {common:.4f} s")
    if not_covering:
        print(f"  !! {len(not_covering)}/{len(successful)} voli NON coprono T: la coda")
        print(f"     delle curve medie verrebbe calcolata su un sottoinsieme distorto.")
        for d in not_covering[:5]:
            print(f"       {d['run']}  ({d['duration']:.4f} s)")
        print(f"     Consigliato: --t-crop {np.floor(common*100)/100:.2f}")
    else:
        print(f"  Tutti i {len(successful)} voli inclusi coprono T: n(t) costante.")
    if T_EVAL_FIXED is None and cfg['t_crop'] is None:
        print(f"  >> Per i run definitivi della tesi, congela questo valore in")
        print(f"     T_EVAL_FIXED = {t_eval:.2f}  in cima al file.")

    # voli lunghi: croppati come tutti gli altri, ma segnalati
    long_runs = [d for d in successful if d['duration'] > t_eval * 1.5]
    if long_runs:
        print(f"\n  {len(long_runs)} voli con durata anomala, croppati a T come gli altri:")
        for d in long_runs:
            print(f"    {d['duration']:6.2f}s  {d['group']:<5} policy={d['policy_id']:<4} {d['run']}")

    # TV sulla griglia comune
    for d in successful:
        tv = compute_tv(d, t_grid)
        if tv is not None:
            d['tv_per_motor'] = tv

    # ------------------------------------------------ manifest completo
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
        print(f"\n  !! {n_skip} file scartati (motivi in tables/manifest.csv)")
    warned = [m for m in manifest if m['included'] and m['note']]
    if warned:
        print(f"  !! {len(warned)} file inclusi con avvisi (vedi manifest.csv)")

    # -------------------------------------------- diagnostica sui setpoint
    sps = np.array([d['setpoint'] for d in successful if 'setpoint' in d])
    if len(sps):
        yaws = np.array([d['ref_yaw_deg'] for d in successful if 'ref_yaw_deg' in d])
        print(f"\n  Setpoint (uno per volo, NON condiviso):")
        print(f"    x [{sps[:,0].min():6.2f}, {sps[:,0].max():6.2f}] m   "
              f"y [{sps[:,1].min():6.2f}, {sps[:,1].max():6.2f}] m   "
              f"z [{sps[:,2].min():6.2f}, {sps[:,2].max():6.2f}] m")
        print(f"    yaw di riferimento [{yaws.min():.1f}, {yaws.max():.1f}] deg "
              f"(escursione {yaws.max()-yaws.min():.0f} deg)")
        
        n_var_flights = [d['run'] for d in successful if not d.get('ref_is_constant', True)]
        if n_var_flights:
            print(f"    !! {len(n_var_flights)} voli con riferimento NON costante: non sono hold puri.")
            for run_name in n_var_flights:
                print(f"       - {run_name}")

        # controllo confondente: la quota e' bilanciata tra i due gruppi?
        z_ppo = np.array([d['setpoint'][2] for d in successful
                          if d['group'] == 'PPO' and 'setpoint' in d])
        z_spo = np.array([d['setpoint'][2] for d in successful
                          if d['group'] == 'SPPO' and 'setpoint' in d])
        if len(z_ppo) > 2 and len(z_spo) > 2:
            try:
                _, p_z = stats.mannwhitneyu(z_ppo, z_spo)
            except Exception:
                p_z = float('nan')
            print(f"    quota media: PPO {z_ppo.mean():.3f} m, SPPO {z_spo.mean():.3f} m "
                  f"(Mann-Whitney p={p_z:.3f})")
            if p_z < 0.05:
                print("    !! quote NON bilanciate tra i gruppi: effetto suolo diverso,")
                print("       parte della differenza osservata potrebbe dipendere da questo.")

    # ============================================== [FIX 12] SUCCESS RATE
    print(f"\n{'='*68}\nSUCCESS RATE\n{'='*68}")
    cnt = {g: sum(1 for d in flights if d['group'] == g) for g in ('PPO', 'SPPO')}
    cok = {g: sum(1 for d in successful if d['group'] == g) for g in ('PPO', 'SPPO')}
    ci = {}
    for g in ('PPO', 'SPPO'):
        lo, hi = wilson(cok[g], cnt[g]); ci[g] = (lo, hi)
        print(f"  {g:<5} {cok[g]:2d}/{cnt[g]:2d} = {cok[g]/max(cnt[g],1)*100:5.1f}%"
              f"   IC95% Wilson [{lo*100:5.1f}%, {hi*100:6.1f}%]")
    try:
        _, p_fisher = stats.fisher_exact([[cok['PPO'], cnt['PPO'] - cok['PPO']],
                                          [cok['SPPO'], cnt['SPPO'] - cok['SPPO']]])
    except Exception:
        p_fisher = float('nan')
    print(f"  Fisher exact: p = {p_fisher:.4f}  "
          f"{'(significativo)' if p_fisher < 0.05 else '(NON significativo)'}")
    if p_fisher >= 0.05:
        print("  !! Da riportare come tendenza con l'intervallo, non come risultato")
        print("     dimostrato: gli IC si sovrappongono.")

    fails = [d for d in flights if not d['success']]
    if fails:
        print("\n  Voli falliti:")
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

    # ============================== [FIX 2] aggregazione per policy
    # Le ripetizioni per policy non sono uniformi (1, 2 o 4 run): una media
    # semplice sui voli farebbe pesare la policy con 4 run quattro volte piu'
    # delle altre. Si media prima DENTRO la policy, poi tra policy.
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
        """Centro e limiti della banda. ddof=1 ovunque ([FIX 11]: l'originale
        usava ddof=0 nella banda e ddof=1 nella tabella)."""
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

    band_lbl = 'mediana, IQR' if cfg['band'] == 'iqr' else r'media \pm \sigma'

    per_flight_rows, per_policy_rows, summary_rows, paired_rows = [], [], [], []

    # ======================================================= curve + appaiato
    for key, sym, unit, title, fname, official in METRICS:
        # [FIX 8] la presenza della metrica si valuta su TUTTI i voli.
        # L'originale usava "if key not in successful_data[0]": il primo file
        # scansionato decideva quali metriche calcolare per l'intera campagna.
        if not any(key in d for d in successful):
            print(f"  [skip] metrica '{key}': assente in tutti i voli.")
            continue

        d_rmse, d_curves = {}, {}
        for group in ('PPO', 'SPPO'):
            d_curves[group] = curves_by_policy(key, group)
            d_rmse[group] = rmse_by_policy(key, group)

        # ---------- 1) curva del valore assoluto (cartella RMSE) ----------
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
            # la grandezza e' non negativa: la banda inferiore non scende sotto zero
            ax.fill_between(t_grid, np.maximum(lo, 0.0), hi, color=color, alpha=0.2)
        # [FIX 11] titolo onesto: la curva e' il valore ASSOLUTO mediato sulle
        # policy, l'RMSE per volo sta in legenda. Nell'originale il titolo
        # diceva RMSE mentre la curva era la media di |e|.
        ax.set_title(f'Absolute {title}')
        _s = sym if key.endswith('norm') else rf'\left|{sym}\right|'
        ax.set_ylabel(rf'${_s}$   [{unit}]')
        ax.set_xlabel(r'$t$   [s]')
        ax.set_xlim([0, t_eval]); ax.set_ylim(bottom=0)
        ax.legend(loc='best', frameon=True)
        save(fig, dirs['rmse'] / f"{fname}.pdf")

        # ---------- 2) rollout con segno (cartella Rollouts) ----------
        # [FIX 6] L'originale plottava lo STATO ASSOLUTO mediato tra voli.
        # Ogni volo ha il proprio setpoint (e il proprio yaw): quelle medie
        # non avevano significato fisico. Qui si plotta l'errore con segno,
        # che e' confrontabile perche' riferito al setpoint di ciascun volo.
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

        # ---------- 3) confronto appaiato ----------
        # [FIX 13] Wilcoxon signed-rank sulle coppie PPO/SPPO della STESSA
        # policy: l'unico confronto che controlla la variabilita' tra policy.
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
            # Finestra centrata sui dati, non sull'origine: qui conta la
            # posizione rispetto alla diagonale, non la distanza da zero.
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
            ax.set_title(f'{nbet}/{len(shared)} policy a favore di SPPO,'
                         f'  $p={pw:.3f}$')
            fig.tight_layout()
            save(fig, dirs['paired'] / f"paired_{fname}.pdf")

        # ---------- 4) righe per volo ----------
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
        print(f"\n  TV totale (somma {N_ROTORS} motori):")
        print(f"    PPO  {tot['PPO'].mean():8.2f} +/- {tot['PPO'].std(ddof=1):6.2f}")
        print(f"    SPPO {tot['SPPO'].mean():8.2f} +/- {tot['SPPO'].std(ddof=1):6.2f}"
              f"    Mann-Whitney p = {p_tv:.4f}")

    # ================================================== ROLLOUT: forze
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

    # ====================== ROLLOUT: dispersione inter-run (ex "sensitivity")
    # [FIX 14] Rinominato. Questa e' la deviazione standard TRA VOLI dell'errore
    # di posizione: dispersione inter-run (vento, batteria, condizioni iniziali),
    # NON la sensitivity nel senso del funzionale di training. Il proxy empirico
    # della sensitivity e' la degradazione nominale -> perturbato.
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
        # titolo in testo normale: mathtext non conosce \mbox e in \mathrm il
        # trattino diventerebbe un segno meno
        ax.set_title('Inter-run dispersion of position error')
        ax.set_ylabel(r'$\sigma_{\mathrm{runs}}(\|e_p\|)$   [m]')
        ax.set_xlabel(r'$t$   [s]')
        ax.set_xlim([0, t_eval]); ax.set_ylim(bottom=0); ax.legend(loc='best')
        save(fig, dirs['roll'] / "12_interrun_dispersion.pdf")

    # ====================== ROLLOUT: volo rappresentativo (stato assoluto)
    # [FIX 6b] Lo stato assoluto non e' mediabile tra voli, ma su UN volo solo
    # e' la figura piu' leggibile che puoi mettere in tesi. Si sceglie il volo
    # con RMSE di posizione mediano nel suo gruppo, e lo si dichiara.
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
        print("\n  Volo rappresentativo (RMSE di posizione mediano nel gruppo):")
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
                         rf'(tratteggio = setpoint)', fontsize=13)
            ax.legend(loc='best', ncol=3)
        fig.tight_layout()
        save(fig, dirs['roll'] / "13_representative_flight.pdf")

    # ====================================== SUMMARY: confronto RMSE globale
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
        ax.set_ylabel('RMSE (media sulle policy)')
        ax.set_title('Global RMSE comparison - PPO vs SPPO')
        ax.grid(axis='x', visible=False)
        ax.set_xticks(xm)
        ax.set_xticklabels([SHORT.get(t_, t_) for t_ in uniq])
        ax.legend(loc='best', frameon=True)
        save(fig, dirs['summary'] / "summary_rmse.pdf")

    # ================================================= [FIX 15] EXPORT
    # L'originale importava csv senza mai usarlo e costruiva table_data per poi
    # buttarlo via: uscivano solo PDF, quindi nessuna tabella per la tesi e
    # nessun modo di ricontrollare un numero senza rilanciare tutto.
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

    # tabella LaTeX: solo metriche ufficiali
    with open(dirs['tables'] / "results.tex", "w") as fh:
        fh.write("% RMSE mediato dentro ciascuna policy, poi media +/- std sulle policy.\n")
        fh.write("% p: Wilcoxon signed-rank appaiato sulle coppie PPO/SPPO.\n")
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

    # ------------------------------------------------------- riepilogo video
    print(f"\n{'='*68}\nCONFRONTO APPAIATO (Wilcoxon signed-rank sulle policy)\n{'='*68}")
    print(f"  {'metrica':<27}{'PPO':>9}{'SPPO':>10}{'delta%':>9}{'meglio':>9}{'p':>9}")
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
        print(f"\n  !! {n_pol_tot - n_pairs} policy fuori dal confronto appaiato "
              f"({', '.join(map(str, missing))}):")
        print("     un gruppo non ha voli riusciti. Sono le policy in cui PPO fallisce,")
        print("     quindi escluderle FAVORISCE PPO nell'RMSE. Da dichiarare nella")
        print("     didascalia della tabella. Il success rate, che li conta tutti, no.")

    itm = {g: [d['inf_time_mean_ms'] for d in flights
               if d['group'] == g and 'inf_time_mean_ms' in d] for g in ('PPO', 'SPPO')}
    if all(itm.values()):
        dt_ms = np.mean([d['dt_mean'] for d in successful]) * 1e3
        print(f"\n  Tempo di inferenza (appendice): "
              f"PPO {np.mean(itm['PPO']):.3f} ms, SPPO {np.mean(itm['SPPO']):.3f} ms")
        print(f"    periodo di controllo medio {dt_ms:.2f} ms -> "
              f"{np.mean(itm['SPPO'])/dt_ms*100:.1f}% del budget")

    print(f"\nAnalisi completata. Output in: {out_dir}")
    print(f"  tabelle e manifest -> {dirs['tables']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Analisi campagna TiltHex: RMSE, TV, rollout, confronto appaiato.")
    p.add_argument("--input", type=str, required=True,
                   help="cartella dei log (es. plots/hovering_nominal)")
    p.add_argument("--t-crop", type=float, default=None,
                   help="orizzonte di valutazione [s]; se omesso viene dedotto dai dati")
    p.add_argument("--t-success", type=float, default=9.5,
                   help="durata minima perche' un volo sia considerato riuscito")
    p.add_argument("--band", choices=['std', 'iqr'], default='std',
                   help="banda attorno alla curva: media+/-sigma oppure mediana+IQR")
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.is_dir():
        print(f"Errore: cartella '{input_path}' non trovata.")
        raise SystemExit(1)

    export_unified_analysis(input_path, {'t_crop': args.t_crop,
                                         't_success': args.t_success,
                                         'band': args.band})