#!/usr/bin/env python3
"""
compare_trajectory.py  --  Confronto nominal vs perturbed per un task.

Accorpa i vecchi compare_comparison.py e compare_categories.py: stesse due
letture (chi sbaglia meno, chi degrada meno) su due divisioni del volo, le
fasi della missione e le cinque categorie di compito di controllo.

Prende le due cartelle dello stesso task (es. p2p_nominal e p2p_perturbed) e
produce una cartella di confronto con tabelle e figure.

DA DOVE PRENDE I NUMERI
-----------------------
NON ricalcola le metriche. Legge i CSV che analyze_trajectory.py ha gia'
prodotto dentro <cartella>/data_analysis/tables/:
    phase_metrics.csv   metriche per fase, per volo
    tv_windows.csv      Total Variation, per finestra e per motore
    category_metrics.csv  metriche per manovra e per gradino
    params.csv          i parametri con cui e' stata prodotta la cartella
Una sola implementazione delle metriche, un solo posto dove puo' esserci un
errore. Se i CSV mancano lo script si ferma e dice quale comando lanciare.
(Unica eccezione: le figure di sovrapposizione nel tempo, che hanno bisogno
delle serie grezze. Per quelle riusa process_log di analyze_trajectory, quindi
anche li' le definizioni di errore restano le sue.)

LE DUE DOMANDE
--------------
Con quattro voli -- PPO nominal, PPO perturbed, SPPO nominal, SPPO
perturbed -- si possono fare due domande diverse, che NON hanno la stessa
risposta:

  1. CHI SBAGLIA MENO COL DISTURBO
     confronto diretto SPPO vs PPO nel caso perturbed.

  2. CHI DEGRADA MENO
     D = perturbed / nominal, calcolato per algoritmo e poi confrontato.

Una policy puo' sbagliare meno in assoluto in entrambe le condizioni e al
tempo stesso degradare di piu' in rapporto, semplicemente perche' parte da
piu' in basso. Mostrare solo la (1) o solo la (2) sarebbe scegliere la lettura
comoda: lo script le riporta SEMPRE affiancate, in tabella e in figura.

PERCHE' IL RAPPORTO E NON LA DIFFERENZA, E IN QUEST'ORDINE
  * rapporto: i due algoritmi partono da errori nominali diversi, quindi una
    differenza in metri non e' confrontabile fra loro; il rapporto e'
    adimensionale;
  * perturbed / nominal: D > 1 si legge subito come peggioramento, e il
    valore nominal -- il piu' pulito dei due -- sta al denominatore.

IL LIMITE, DA DICHIARARE IN TESI
--------------------------------
D e' il rapporto fra due voli singoli: la sua incertezza e' quella di un volo
moltiplicata per quella di un altro, e con n = 1 non e' quantificabile da
questi dati. Si usa quindi una SOGLIA ASSUNTA (--noise-floor, default 5%):
sotto quella differenza lo script scrive '~' e non assegna un vincitore.
La soglia e' una convenzione dichiarata, NON una stima statistica: va scritta
nel testo per quello che e'. Quello che regge davvero non e' il singolo numero
ma la COERENZA del segno su piu' fasi e su piu' traiettorie.

Uso:
    python compare_comparison.py --nominal plots/p2p_nominal \
                                 --perturbed plots/p2p_perturbed
    python compare_comparison.py -n plots/rpy_nominal -p plots/rpy_perturbed \
                                 --noise-floor 5 --output plots/rpy_comparison
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

C_PPO, C_SPPO = '#ff7f0e', '#1f77b4'  # arancione / blu, come export_campaign.py
COL = {'PPO': C_PPO, 'SPPO': C_SPPO}
GROUPS = ('PPO', 'SPPO')
CONDS = ('nominal', 'perturbed')
HATCH = {'nominal': '', 'perturbed': '///'}
ALPHA = {'nominal': 0.90, 'perturbed': 0.45}

NAN = float('nan')

# (chiave nel csv, etichetta, unita', fattore di scala, definita su tutte le fasi)
PHASE_METRICS = [
    ('rmse_pos_steady',         r'RMSE $\|e_p\|$ a regime',        'm',       1.0,     True),
    ('rmse_att_steady',         r'RMSE $\|e_R\|$ a regime',        'rad',     1.0,     True),
    ('peak_pos_norm',           r'picco $\|e_p\|$ / gradino',      '-',       1.0,     False),
    ('rmse_pos_transient_norm', r'RMSE transitorio / gradino',     '-',       1.0,     False),
    ('peak_att_norm',           r'picco $\|e_R\|$ / gradino',      'rad/m',   1.0,     False),
    ('tv_total',                r'Total Variation di fase',        'Hz/s',      1.0 / (2 * math.pi),  True),
]

# etichette brevi per la console (il mathtext non si stampa a terminale)
SHORT = {
    'rmse_pos_steady': 'pos a regime [m]',
    'rmse_att_steady': 'att a regime [rad]',
    'peak_pos_norm': 'picco pos / gradino [-]',
    'rmse_pos_transient_norm': 'RMSE transit / gradino [-]',
    'peak_att_norm': 'picco att / gradino [rad/m]',
    'tv_total': 'TV di fase [Hz/s]',
}


# =========================================================== lettura dei CSV
def fnum(s):
    """Cella di CSV -> float. Vuota o non numerica -> nan."""
    try:
        v = float(s)
    except (TypeError, ValueError):
        return NAN
    return v


def tables_dir(folder: Path) -> Path:
    return folder / "data_analysis" / "tables"


def read_phase_metrics(folder: Path):
    """{(group, phase): {campo: valore}} da phase_metrics.csv."""
    path = tables_dir(folder) / "phase_metrics.csv"
    if not path.is_file():
        sys.exit(f"Manca {path}\n"
                 f"  Lancia prima:  python analyze_trajectory.py --input {folder}")
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            key = (r['group'], r['phase'])
            out[key] = {k: (v if k in ('run', 'group', 'phase') else fnum(v))
                        for k, v in r.items()}
    if not out:
        sys.exit(f"{path} e' vuoto.")
    return out


def read_tv_flight(folder: Path):
    """{group: TV totale sul volo intero} da tv_windows.csv.

    Il volo intero, non un ritaglio: e' la sola finestra che non dipende da
    una scelta e che quindi si puo' confrontare fra due cartelle senza prima
    verificare che il taglio sia caduto nello stesso posto."""
    path = tables_dir(folder) / "tv_windows.csv"
    if not path.is_file():
        return {}
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get('window') == 'flight' and r.get('channel') == 'TOTAL':
                out[r['group']] = fnum(r['tv'])
    return out


# ================================================ serie grezze per le curve
# Le tabelle arrivano tutte dai CSV. Per disegnare le curve nel tempo servono
# pero' le serie grezze, e queste due definizioni di errore sono ripetute qui:
#
#     |e_p| = || pos - pos_ref ||
#     |e_R| = || rotvec(R_ref^-1 R_act) ||          quaternioni in WXYZ
#
# Ripetere una definizione e' il modo classico di ritrovarsi due numeri diversi
# per la stessa cosa. Per questo NON ci si fida: dopo il caricamento lo script
# ricalcola l'RMSE di una fase gia' presente in phase_metrics.csv e lo confronta
# con il valore scritto li'. Se non coincide, lo dice e non disegna.
# (Prima queste curve arrivavano importando analyze_trajectory. Se quel file non
# sta accanto a questo -- o e' stato rinominato -- l'import falliva e la cartella
# Rollouts restava vuota senza che fosse ovvio il perche'.)

def load_series(npz_path: Path):
    """(group, dict con t, e_pos_norm, e_att_norm) oppure (None, motivo)."""
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:                                   # noqa: BLE001
        return None, f"load_error: {e}"
    keys = set(data.files)
    if not {'t', 'pos', 'pos_ref', 'quat', 'quat_ref'} <= keys:
        return None, "campi mancanti"

    pname = str(data['policy_name']) if 'policy_name' in keys else ""
    tag = (npz_path.parent.name + "_" + pname).upper()
    group = "SPPO" if "SPPO" in tag else ("PPO" if "PPO" in tag else None)
    if group is None:
        return None, "gruppo non riconosciuto"

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
           # Serie grezze, non solo le norme: servono a fig_step_profile_pair
           # per il progresso di gradino p(t) (eq:step_progress), che ha
           # bisogno della direzione del comando, non solo della sua norma.
           'pos': pos, 'pos_ref': pos_ref, 'R_act': R_act, 'R_ref': R_ref}
    # Velocita' lineare e angolare: stessa convenzione di analyze_trajectory
    # (process_log), solo la norma, nessun riferimento loggato per confrontarle.
    if 'vel' in keys:
        out['v_norm'] = np.linalg.norm(np.asarray(data['vel'], float), axis=1)
    if 'avel' in keys:
        out['w_norm'] = np.linalg.norm(np.asarray(data['avel'], float), axis=1)
    return group, out


def check_against_csv(f, pm, group, tol=0.02):
    """Verifica che le serie ricalcolate qui diano lo stesso RMSE che
    analyze_trajectory ha scritto nel CSV, su una fase confrontabile.
    Ritorna (ok, messaggio)."""
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
            return False, (f"{group}: RMSE ricalcolato su '{ph}' = {got:.5f}, "
                           f"nel CSV = {exp:.5f}")
        return True, f"{group}: coerente su '{ph}' ({got:.5f} vs {exp:.5f})"
    return True, f"{group}: nessuna fase adatta alla verifica"


# =========================================================== confronti
def pct(new, old):
    """Variazione percentuale di new rispetto a old."""
    if new != new or old != old or old == 0:
        return NAN
    return (new - old) / old * 100.0


def ratio(num, den):
    if num != num or den != den or den == 0:
        return NAN
    return num / den


def winner(delta_pct, floor):
    """delta_pct = (SPPO - PPO) / PPO * 100, su una metrica dove meno e' meglio.
    Sotto la soglia assunta non si assegna nessun vincitore."""
    if delta_pct != delta_pct:
        return ''
    if abs(delta_pct) < floor:
        return '~'
    return 'SPPO' if delta_pct < 0 else 'PPO'


def compare_cell(nom, pert, floor):
    """Tutti i numeri derivati per una metrica e una fase.
    nom/pert sono dict {group: valore}."""
    c = {}
    for g in GROUPS:
        c[f'{g.lower()}_nom'] = nom.get(g, NAN)
        c[f'{g.lower()}_pert'] = pert.get(g, NAN)
        c[f'D_{g.lower()}'] = ratio(pert.get(g, NAN), nom.get(g, NAN))
    # lettura 1: chi sbaglia meno, condizione per condizione
    c['delta_nom_pct'] = pct(c['sppo_nom'], c['ppo_nom'])
    c['delta_pert_pct'] = pct(c['sppo_pert'], c['ppo_pert'])
    # lettura 2: chi ha il D piu' basso. ATTENZIONE: "degrada meno" e' vero
    # solo se entrambi i D sono sopra 1. Se sono sotto, il volo perturbed e'
    # andato meglio del nominal e il D piu' basso vuol dire "e' migliorato di
    # piu'", che con un carico aggiunto non e' un risultato ma variabilita'.
    c['delta_D_pct'] = pct(c['D_sppo'], c['D_ppo'])
    # lettura 3: DELTA, lo spostamento assoluto causato dal disturbo.
    #
    # Il disturbo e' lo STESSO per le due policy, quindi "chi e' piu' robusto"
    # e' "a chi si sposta meno l'errore": una differenza, non un rapporto.
    # D divide per il valore nominal, e chi parte da un errore piu' basso ha
    # un denominatore piu' piccolo -- il rapporto lo penalizza per una
    # proprieta' dell'aritmetica, non del controllore. Delta non ha quel bias
    # ed e' la grandezza da riportare quando si parla di sensibilita'.
    #
    # Delta NEGATIVO con un carico aggiunto e' fisicamente impossibile: li'
    # non si sta misurando il disturbo ma la variabilita' volo-su-volo, e il
    # suo valore assoluto da' la scala del rumore sotto cui nessuna
    # differenza fra le due policy e' leggibile.
    for g in GROUPS:
        c[f'delta_abs_{g.lower()}'] = (c[f'{g.lower()}_pert']
                                       - c[f'{g.lower()}_nom'])
    c['win_nom'] = winner(c['delta_nom_pct'], floor)
    c['win_pert'] = winner(c['delta_pert_pct'], floor)
    c['win_D'] = winner(c['delta_D_pct'], floor)
    # chi si sposta meno: sul valore assoluto dello spostamento, cosi' un
    # Delta negativo (rumore) non viene premiato come "spostamento minimo"
    da, ds = c['delta_abs_ppo'], c['delta_abs_sppo']
    c['win_delta'] = (None if da != da or ds != ds or abs(da) < 1e-12
                      else winner(pct(abs(ds), abs(da)), floor))
    return c


# =========================================================== figure
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
    fig.savefig(path, format='pdf', transparent=True, bbox_inches='tight')
    plt.close(fig)


# Posizione e assetto, un pannello ciascuno. Velocita' lineare/angolare
# tolte il 5/10 su richiesta di Marco: la griglia torna 1x2 come lo era
# prima di D26/del giro overlay, per restare sulle due sole grandezze che
# contano anche nel grafico punto medio+range di fig_cat_range.
OVERLAY_PANELS = [
    ('e_pos_norm', r'$\|e_p\|$   [m]'),
    ('e_att_norm', r'$\|e_R\|$   [rad]'),
]


def fig_overlay_grid(flights, out_path, bounds=(), smooth_s=0.5):
    """Griglia 1x2: posizione a sinistra, assetto a destra, nominal contro
    perturbed, un'unica figura e un'unica legenda -- stessa logica delle
    griglie di rollout di analyze_trajectory (font piu' grandi, linee a 1.8,
    legenda fuori dal pannello sopra la griglia, niente titolo), cosi' la
    scala di stampa coincide quando la figura va a tutta larghezza pagina.
    Dimensione (11, 10) invariata rispetto alla 2x2 precedente, la stessa
    del rollout 'forces' di analyze_trajectory (griglia 3x2): due pannelli
    affiancati invece di quattro, stessa dimensione complessiva.

    I pannelli di velocita' lineare/angolare, presenti nella 2x2 di un giro
    precedente, sono tolte il 5/10 su richiesta di Marco: restano solo le
    due grandezze che contano anche nel grafico punto medio+range (vedi
    fig_cat_range), per restare coerenti fra le figure del capitolo.

    Il tratteggio qui non si usa: nelle altre figure del capitolo (full_pose,
    TRACK) il tratteggio indica il riferimento comandato, e riusarlo per
    'perturbed' si confonderebbe con quello. nominal e perturbed restano
    sullo stesso tratto continuo e si distinguono per trasparenza: nominal
    sfumato, perturbed pieno. Niente traccia grezza in trasparenza sotto la
    media mobile (come nelle altre figure del capitolo): l'unica traccia
    visibile per ogni volo e' gia' la media mobile."""
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
        # Legenda dentro il pannello invece che sopra la griglia: su una
        # griglia 1x2 (non piu' 2x2) la banda riservata sopra la figura per
        # la legenda valeva quanto i pannelli stessi -- Marco, 5/10. Un solo
        # riquadro (nel pannello di posizione, in alto a destra: dopo il
        # secondo picco le curve restano basse li', verificato a vista) vale
        # per entrambi i pannelli, stessa mappa colori/etichette.
        fig.tight_layout()
        order = sorted(handles_by_label)
        axs[0][0].legend([handles_by_label[l] for l in order], order,
                         loc='upper right', ncol=1, frameon=True, fontsize=14)
        save(fig, out_path)


# Fase del gradino C->D (Point-to-point nel p2p, Tilt nel tilt): e' la
# SECONDA fase di hold della missione, in entrambi i casi 'SETPOINT 2' nelle
# phase_metrics.csv prodotte da analyze_trajectory -- stessa etichetta per
# le due missioni, quindi la funzione sotto non distingue p2p da tilt.
STEP_PROFILE_PHASE = 'SETPOINT 2'

# Finestra del profilo: NON e' cat_delta (quello segmenta Step da Hold per
# le metriche a categoria, tipicamente piu' corto del transitorio). La fase
# SETPOINT 2 dura circa 5-6 s in entrambe le missioni (phase_metrics.csv,
# colonna t1-t0): 5 s mostra il transitorio quasi per intero senza sforare
# nella fase successiva. Dichiarata e fissa, non calcolata volo per volo,
# stessa convenzione di subsec:window in tesi.
STEP_PROFILE_WINDOW_S = 5.0

STEP_PROFILE_PANELS = [('pos', r'$p(t)$   [-]'), ('att', r'$p(t)$   [-]')]
STEP_PROFILE_CHANNELS = {'pos': ('pos', 'pos_ref'), 'att': ('R_act', 'R_ref')}
STEP_MIN_AMP = {'pos': 0.02, 'att': float(np.deg2rad(2.0))}   # STEP_POS/STEP_ANG

# Canale p(t) rilevante per ciascuna missione: l'unico, dei due, che ha
# davvero un gradino non nullo in C->D (l'altro resta a riferimento
# costante nella transizione -- vedi STEP_MIN_AMP/_step_progress, che
# infatti scarta quel canale). Chiavi = stem di nom_dir.name dopo aver
# tolto '_nominal' (vedi run()): 'p2p' per il point-to-point, 'rpy' per il
# tilt (nome della cartella, non della manovra -- in tesi resta "Tilt").
MISSION_STEP_CHANNEL = {'p2p': 'pos', 'rpy': 'att'}


def _step_progress(t, sig, sig_ref, t0, delta, kind):
    """Variabile di progresso p(t) di eq:step_progress sulla finestra
    [t0, t0+delta]: 0 all'istante del gradino (un campione prima del
    confine di fase, come in step_response() di analyze_trajectory, cosi'
    il primo campione e' davvero zero), 1 sul bersaglio, >1 se il drone lo
    supera. kind='pos': sig/sig_ref sono gli array pos/pos_ref (N,3).
    kind='att': sig/sig_ref sono gli stack R_act/R_ref di scipy Rotation.
    Stessa formula di step_response(), copiata a mano perche' qui non si
    importa analyze_trajectory (vedi load_series): ricalcolare e controllare
    contro step_pos_m/step_ang_deg di phase_metrics.csv e' la stessa
    disciplina di check_against_csv.
    Ritorna (t_rel, p, amp) oppure None se il gradino su questo canale e'
    sotto soglia o la finestra non ha abbastanza campioni."""
    # tolleranza: t0 arriva da phase_metrics.csv arrotondato a 6 decimali,
    # fino a ~1e-6 s sopra il campione vero del salto -- senza margine,
    # su un campione sfortunato 't >= t0' sceglie il campione DOPO quello
    # giusto, rendendo i_prev gia' post-salto e l'ampiezza nulla (scartata
    # da STEP_MIN_AMP in silenzio). 1e-4 s e' molto sopra l'errore di
    # arrotondamento e molto sotto il periodo di campionamento (~0.01 s),
    # quindi non sposta quale campione e' 'quello giusto'. Bug trovato e
    # corretto il 5/10 generando le prime figure/tabelle reali di 5.3.3
    # (D29): su otto combinazioni missione/gruppo a SETPOINT 2, sei
    # cadevano nel caso sfortunato, verificato confrontando l'ampiezza
    # ricalcolata con step_pos_m/step_ang_deg di phase_metrics.csv.
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
    """Banda punto medio+range di p(t) per un canale (key='pos'/'att'),
    una per gruppo, su [0, delta] di STEP_PROFILE_PHASE -- fattorizzata da
    fig_step_profile_pair e fig_step_profile_single, stessa logica in un
    solo posto (vedi _step_progress per la formula).

    flights: {(group, cond): dict di load_series, con pos/pos_ref/R_act/R_ref}.
    pm_nom, pm_pert: dict da read_phase_metrics, chiave (group, phase).
    tag: prefisso dei messaggi di avviso in console, per distinguere la
    chiamante quando piu' di una e' attiva nello stesso run.

    Ritorna (t_grid, bands) con bands[g] = (mid, lo, hi) sulla griglia
    t_grid, o None se a quel gruppo manca un volo o il canale non ha un
    gradino sopra soglia su almeno una delle due condizioni."""
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
                    print(f"  [step p(t){tag}] !! {g} {cond} {key}: ampiezza "
                          f"ricalcolata {amp:.4f} vs CSV {csv_amp_v:.4f}")
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
    """Griglia 1x2 (posizione, assetto): p(t) sulla finestra [t0, t0+delta]
    di STEP_PROFILE_PHASE, una banda punto medio+range per policy (vedi
    _step_progress_band). Stessa convenzione di fig_cat_range_pair, qui nel
    tempo invece che per categoria: il centro di ogni banda e' il punto
    medio fra il volo nominal e quello perturbed, l'ampiezza e' l'intervallo
    che i due coprono -- non una deviazione standard (subsec:nom_pert_range:
    due soli valori non sono un campione). t0 viene da phase_metrics.csv
    (stesso file che fig_overlay_grid legge per i confini verticali).

    Griglia completa pos+att per entrambe le missioni, utile a controllo:
    ciasccuna missione ha un gradino vero su un solo canale in C->D (vedi
    MISSION_STEP_CHANNEL), quindi uno dei due pannelli resta vuoto nel
    file prodotto qui. La figura pensata per il testo e' invece
    fig_step_profile_single, un solo pannello sul canale col gradino.

    flights, pm_nom, pm_pert: come _step_progress_band.
    delta: durata dichiarata della finestra (STEP_PROFILE_WINDOW_S)."""
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
        print(f"  [step profile] nessun dato disponibile, {out_path} non scritto")
        return False
    if missing:
        print(f"  [step profile] banda mancante per: {', '.join(missing)}")

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
    """Un solo pannello: p(t) sulla finestra [t0, t0+delta] di
    STEP_PROFILE_PHASE, banda punto medio+range per policy, SOLO sul canale
    che in questa missione ha un gradino vero in C->D ((posizione per il p2p,
    assetto per il tilt -- vedi MISSION_STEP_CHANNEL)). E' la figura pensata
    per il testo di Sezione 5.3.3: una per sottosezione (Point-to-point,
    Tilt), niente traiettoria 3D ne' secondo pannello vuoto -- quella
    combinazione (fig_trajectory3d_step_pair) e' stata scartata il 5/10 sera
    su richiesta di Marco (troppe voci in legenda), vedi idea.md.

    flights, pm_nom, pm_pert, delta: come fig_step_profile_pair.
    mission: chiave di MISSION_STEP_CHANNEL ('p2p' o 'rpy'; vedi stem in
    run())."""
    key = MISSION_STEP_CHANNEL[mission]
    ylabel = dict(STEP_PROFILE_PANELS)[key]
    t_grid, bands = _step_progress_band(flights, pm_nom, pm_pert, key, delta,
                                        smooth_s, n_grid, amp_tol,
                                        tag=' single')

    if all(b is None for b in bands.values()):
        print(f"  [step profile single] nessun dato disponibile per "
              f"'{mission}', {out_path} non scritto")
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


# ================================================== tabella risposta al
# gradino, punto medio+range (Sezione 5.3.3)
#
# Le quattro grandezze (overshoot, rise time, settling time, steady-state
# error) non si ricalcolano qui: vengono da assets.csv (tables/ di ciascuna
# cartella), scritto da step_response()/export_assets() di
# analyze_trajectory.py -- unica implementazione delle formule, stessa
# disciplina dichiarata in testa a questo file. Steady-state error in GRADI
# per l'assetto (non radianti), come _step_table_figure() dello stesso
# analyze_trajectory.py, per restare coerenti con 03_method.tex (righe
# 659-660: assetto in gradi nelle figure/tabelle del Capitolo 5).
STEP_METRICS_SECTION = {'pos': 'step_pos', 'att': 'step_att'}
STEP_METRICS = [
    ('overshoot_pct', r'overshoot $M_p$', '%', 1.0),
    ('rise_10_90_s', r'rise time $t_r$', 's', 1.0),
    ('settle_s', r'settling time $t_s$', 's', 1.0),
    ('final_err_abs', r'steady-state error $e_{ss}$', None, None),
]


def read_assets(folder: Path):
    """{(section, phase, group, metric): valore} da assets.csv -- stessa
    forma chiave-valore del file sorgente, tables_dir() e' la stessa
    funzione gia' usata da read_phase_metrics/read_tv_flight."""
    path = tables_dir(folder) / "assets.csv"
    if not path.is_file():
        sys.exit(f"Manca {path}\n"
                 f"  Lancia prima:  python analyze_trajectory.py --input {folder}")
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out[(r['section'], r['phase'], r['group'], r['metric'])] = fnum(r['value'])
    return out


def step_metrics_range_table(nom_dir, pert_dir, mission, out_path):
    """Scrive una tabella LaTeX pronta (\\begin{tabular}...) con le quattro
    grandezze di STEP_METRICS su STEP_PROFILE_PHASE, punto medio+range fra
    nominal e perturbed (D28), PPO ed SPPO affiancate -- l'equivalente di
    fig_cat_range_pair per il gradino, in forma di tabella invece che di
    barre (Marco, 5/10 sera: per 5.3.3 solo p(t) come figura, i quattro
    numeri restano una tabella, non un secondo grafico).

    mission: chiave di MISSION_STEP_CHANNEL, sceglie se leggere le righe
    'step_pos' o 'step_att' di assets.csv."""
    key = MISSION_STEP_CHANNEL[mission]
    section = STEP_METRICS_SECTION[key]
    a_nom, a_pert = read_assets(nom_dir), read_assets(pert_dir)
    deg = key == 'att'

    rows = []
    for metric, label, unit, scale in STEP_METRICS:
        if unit is None:                      # steady-state error: m o deg
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
        print(f"  [step table] nessun dato per '{mission}', {out_path} non scritto")
        return False

    def fmt(cell):
        if cell is None:
            return '--'
        mid, lo, hi = cell
        return f"{mid:.3f}  [{lo:.3f}, {hi:.3f}]"

    with open(out_path, "w") as fh:
        fh.write(f"% {mission}: {section}, {STEP_PROFILE_PHASE}, "
                 "punto medio e range (nominal, perturbed)\n")
        fh.write("\\begin{tabular}{lccc}\n\\toprule\n")
        fh.write("Quantity & Unit & PPO & SPPO \\\\\n\\midrule\n")
        for label, unit, cells in rows:
            fh.write(f"{label} & {unit} & {fmt(cells['PPO'])} & "
                     f"{fmt(cells['SPPO'])} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    print(f"  scritto {out_path}")
    return True



def tight_x(ax, lo, hi):
    """Niente margine orizzontale: la traccia attacca all'asse."""
    if hi > lo:
        ax.set_xlim(lo, hi)
    ax.margins(x=0)


# =========================================================== main

# =========================================================== SOLO LA MISSIONE
# Fino al momento in cui il drone raggiunge il world frame, i due voli NON sono
# confrontabili a parita' di riferimento: il decollo avviene sulla verticale
# della posa a terra, che cambia a ogni volo, e il gradino di avvicinamento ha
# quindi un'ampiezza diversa (fino all'11% fra un volo e l'altro). Tutto quello
# che si misura li' dentro risente di quella differenza e non del disturbo.
#
# Da quando il drone e' arrivato a (0,0,z) in poi i riferimenti sono identici
# nei quattro voli, istante per istante: e' l'unica finestra in cui D misura
# solo l'effetto della perturbazione.
#
# Il taglio e' ancorato al RIFERIMENTO (istante del gradino + un margine di
# assestamento dichiarato), non all'arrivo effettivo del drone: allineare
# sull'arrivo significherebbe condizionare sull'esito e togliere al volo piu'
# lento proprio la parte in cui va peggio.

N_ROTORS, F_MIN, F_MAX, CF = 6, 0.3072, 12.0, 12.0e-4

# Stessi nomi e stesso ordine di analyze_trajectory, cosi' tabelle e figure
# restano leggibili allo stesso modo.
# Le due statistiche d'errore, come in analyze_trajectory. Con --full-flight
# si legge summary_rmse.csv, che porta il solo RMSE: li' il MAE non c'e'.
# numpy >= 2.0 la chiama trapezoid, prima trapz
_TRAPZ = getattr(np, 'trapezoid', None) or np.trapz


def seg_mean_t(t, y, mask=None):
    """Media temporale di y: integrale per trapezi diviso per la durata,
    sommando per tratti contigui. Copia fedele di analyze_trajectory.seg_mean_t:
    i due script devono dare lo stesso numero sulla stessa finestra."""
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
    v = seg_mean_t(t, np.asarray(x, float) ** 2, mask)
    return float(np.sqrt(v)) if v == v and v >= 0 else NAN


def mae_t(t, x, mask=None):
    return seg_mean_t(t, np.abs(np.asarray(x, float)), mask)


# Le due statistiche d'errore, come in analyze_trajectory: medie TEMPORALI,
# non sui campioni. Con --full-flight si legge summary_rmse.csv, che porta il
# solo RMSE: li' il MAE non c'e'.
STATS = (("RMSE", rms_t),
         ("MAE",  mae_t))

def read_params(folder):
    """tables/params.csv: i parametri con cui e' stata prodotta la cartella.
    Manca nelle cartelle generate prima che analyze_trajectory lo scrivesse,
    e in quel caso il controllo si limita a dirlo."""
    p = tables_dir(folder) / "params.csv"
    if not p.exists():
        return None
    out = {}
    with open(p, newline="") as fh:
        for r in csv.DictReader(fh):
            out[r['param']] = fnum(r['value'])
    return out


# Parametri che definiscono le FINESTRE: se non coincidono fra nominal e
# perturbed, le due cartelle tagliano il volo in punti diversi e ogni
# confronto fra loro misura anche la differenza di taglio.
WINDOW_PARAMS = ('cat_delta', 't_ss', 'step_tol_pos', 'step_tol_ang')


def check_params(nom_dir, pert_dir):
    """Le due cartelle sono state analizzate con gli stessi tagli?

    Le figure di maneuvers/ e step/ leggono category_metrics.csv cosi' com'e':
    le finestre non si ricalcolano qui, si ereditano. Se le due cartelle sono
    state prodotte con --delta diversi, 'Step' e 'Hover' coprono intervalli
    diversi nei due casi e il confronto lo nasconde."""
    a, b = read_params(nom_dir), read_params(pert_dir)
    print(f"\n{'='*78}\nPARAMETRI DELLE DUE CARTELLE\n{'='*78}")
    if a is None or b is None:
        miss = [str(d) for d, v in ((nom_dir, a), (pert_dir, b)) if v is None]
        print("  !! manca tables/params.csv in: " + ", ".join(miss))
        print("     Rilancia analyze_trajectory su quelle cartelle: senza quel")
        print("     file non posso verificare che i tagli coincidano.")
        return False
    ok = True
    print(f"  {'parametro':20s}{'nominal':>14s}{'perturbed':>14s}   esito")
    for k in WINDOW_PARAMS:
        va, vb = a.get(k, NAN), b.get(k, NAN)
        same = (va != va and vb != vb) or abs(va - vb) < 1e-9
        ok &= same
        def s(v):
            return '--' if v != v else f"{v:.4g}"
        print(f"  {k:20s}{s(va):>14s}{s(vb):>14s}   "
              + ("ok" if same else "!! DIVERSO"))
    if not ok:
        print("\n  !! le due cartelle tagliano il volo in punti diversi.")
        print("     Le figure in maneuvers/ e maneuvers/step/ leggono quelle")
        print("     finestre cosi' come sono: rigenera entrambe le cartelle con")
        print("     gli stessi parametri prima di leggere il confronto.")
    if ok:
        print("\n  -> stessi tagli nelle due cartelle.")
    return ok


# Unita' di ciascuna metrica globale, per le figure della cartella RMSE.
# L'assetto resta in radianti: e' l'unita' in cui e_R e' definito (norma del
# log-map) ed e' quella usata in tesi. I valori sono dell'ordine di 0.02-0.1,
# quindi le etichette sulle barre vanno scritte con abbastanza cifre.
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
    'Total variation - tutti i motori': ('Hz/s', 1.0 / (2 * math.pi)),
}


def _slug(name):
    keep = "".join(ch if ch.isalnum() else "_" for ch in name.lower())
    while "__" in keep:
        keep = keep.replace("__", "_")
    return keep.strip("_")


def fig_metric_bars(metric, c, out_path, floor, stat='RMSE'):
    """Una metrica, quattro barre: le due policy nelle due condizioni.

    E' la vista piu' diretta di tutte -- nessun rapporto, nessuna
    normalizzazione, solo i quattro numeri misurati -- e serve proprio a
    poterli leggere senza passare da D."""
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
# CATEGORIE DI COMPITO DI CONTROLLO
# ===========================================================================
# Sezione ex compare_categories.py, accorpata qui. Stesse due letture della
# parte sopra -- chi sbaglia meno, chi degrada meno -- ma con le cinque
# categorie al posto delle fasi, e quindi righe confrontabili fra scenari.

CATEGORIES = ["Takeoff", "Step", "Hover", "Tilt", "Land"]
# stesso significato che in analyze_trajectory: serve per sapere quali
# categorie nelle figure vanno normalizzate invece che in valore assoluto.
KIND = {"Takeoff": "rampa", "Land": "rampa", "Step": "gradino",
        "Hover": "hold", "Tilt": "hold"}

# Stessa rietichettatura per missione di analyze_trajectory.py (Tabelle
# 5.5/5.6 della tesi): 'stem' ('p2p'/'rpy') seleziona la lista di categorie
# con cui leggere category_metrics.csv, gia' scritto con questi nomi.
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

# Metriche CON SEGNO: il rapporto D fra due valori di segno diverso non
# significa niente, e la variazione percentuale esplode quando il denominatore
# passa vicino allo zero. Per queste si riportano i quattro valori e le
# differenze ASSOLUTE, nell'unita' della metrica.
CAT_SIGNED = {"lag_z"}


def read_category_metrics(folder):
    p = tables_dir(Path(folder)) / "category_metrics.csv"
    if not p.is_file():
        raise SystemExit(f"Manca {p}\n"
                         f"  Lancia prima:  python analyze_trajectory.py --input {folder}")
    out = {}
    for r in csv.DictReader(open(p, newline="")):
        try:
            v = float(r["value"])
        except (TypeError, ValueError):
            continue
        out[(r["group"], r["category"], r.get("stat", "-"), r["metric"])] = v
        out.setdefault("_units", {})[r["metric"]] = r["unit"]
    return out




# Le dieci grandezze, stesse chiavi e stessi nomi di file di analyze_trajectory
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
    """Una grandezza: le manovre sull'asse x, quattro barre per manovra.

    Nessun rapporto e nessuna normalizzazione: solo i quattro numeri misurati,
    che e' la vista da cui si parte prima di guardare D."""
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
    print("  scritto", out_path)
    return True


def fig_cat_degradation(cats, V, stat, key, sym, out_path, stem):
    """Una grandezza: D = perturbed / nominal, due barre per manovra."""
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
               label="$D=1$ (nessuna degradazione)")
    if min(finite) < 0.995:
        ax.axhspan(0, 1.0, color="#999999", alpha=0.07, zorder=0)
    ax.set_ylim(0, max(finite + [1.0]) * 1.26)
    ax.set_ylabel(r"$D$ = perturbed / nominal   [-]")
    ax.set_title(f"{sym}  -  {stat}  -  degradazione  -  {stem}", pad=10)
    ax.grid(axis="x", visible=False)
    ax.set_xticks(x)
    ax.set_xticklabels(list(cats))
    ax.set_xlim(-0.65, len(cats) - 0.35)
    ax.legend(loc="best", ncol=2, frameon=True, fontsize=9.5)
    fig.tight_layout()
    save(fig, out_path)
    print("  scritto", out_path)
    return True


def fig_cat_delta(cats, V, stat, key, sym, unit, out_path, stem):
    """DELTA = perturbed - nominal, due barre per manovra.

    La lettura da preferire quando si parla di ROBUSTEZZA. Il disturbo e' lo
    stesso per le due policy: chi e' piu' robusto e' chi si sposta meno, e uno
    spostamento e' una differenza. Il rapporto D, che sta nella cartella
    accanto, divide per il valore nominal e quindi penalizza chi parte gia'
    basso -- sugli stessi dati le due letture si contraddicono spesso, ed e'
    il motivo per cui questa cartella esiste.

    Sotto lo zero il volo perturbed e' andato MEGLIO del nominal. Con un
    carico aggiunto non ha spiegazione fisica: quella e' variabilita'
    volo-su-volo, e quanto scende dice sotto quale soglia nessuna differenza
    fra le due policy e' leggibile. Per questo la fascia negativa e'
    ombreggiata invece che lasciata passare per "spostamento minore".
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
    ax.set_title(f"{sym}  -  {stat}  -  spostamento col disturbo  -  {stem}")
    ax.grid(axis="x", visible=False)
    ax.set_xticks(x)
    ax.set_xticklabels(list(cats))
    ax.set_xlim(-0.65, len(cats) - 0.35)
    ax.legend(loc="best", ncol=2, frameon=True, fontsize=9.5)
    fig.tight_layout()
    save(fig, out_path)
    print("  scritto", out_path)
    return True


def _draw_cat_range(ax, cats, V, stat, key, sym, unit):
    """Disegna in un asse dato il grafico punto medio+range di una singola
    grandezza: due colonne per manovra (PPO, SPPO), la colonna al punto
    medio fra nominal e perturbed, il baffo nero dal piu' basso al piu'
    alto dei due -- l'INTERVALLO esplorato dall'incertezza di modello
    testata, non una deviazione statistica: restano due soli numeri, non un
    campione, quindi niente media o std nel senso proprio del termine.
    Nessuna distinzione visiva di quale estremo sia nominal e quale
    perturbed: il grafico mostra centro e ampiezza, non una direzione.

    L'etichetta dell'asse Y e' solo simbolo e unita' (niente prefisso
    RMSE/MAE): la statistica va nella didascalia della figura in tesi, non
    nel grafico, su richiesta di Marco il 5/10.

    Usata sia da fig_cat_range (una grandezza, una figura) sia da
    fig_cat_range_pair (posizione e assetto affiancate in una griglia 1x2).
    Ritorna False se non c'e' nulla da disegnare (nessun dato finito), senza
    toccare l'asse."""
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
    """Una grandezza, una figura: vedi _draw_cat_range per il disegno e il
    razionale completo del grafico punto medio+range. Sostituisce, per la
    Robustness di Sezione 5.3, sia il confronto a valori grezzi
    (fig_cat_nom_vs_pert) sia D/Delta: quelli restano nello script per uso
    proprio, questa (e fig_cat_range_pair) sono le sole pensate per finire
    in tesi."""
    fig, ax = plt.subplots(figsize=(1.9 * len(cats) + 4.6, 6.2))
    if not _draw_cat_range(ax, cats, V, stat, key, sym, unit):
        plt.close(fig)
        return False
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    save(fig, out_path)
    print("  scritto", out_path)
    return True


def fig_cat_range_pair(cats, V, stat, out_path, stem):
    """Posizione e assetto affiancate in un'unica griglia 1x2, invece di due
    figure separate: stesso grafico punto medio+range di fig_cat_range (vedi
    _draw_cat_range), una colonna per pannello e un'unica legenda condivisa.
    Richiesta da Marco il 5/10 per avere un'unica figura d'apertura per le
    `\subsubsection{Robustness}` di 5.3.2-5.3.4, dove servono entrambe le
    grandezze viste insieme."""
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
    # Stesso principio di fig_overlay_grid: legenda dentro il pannello di
    # posizione (in alto a destra, le colonne piu' basse stanno li') invece
    # che sopra la griglia -- Marco, 5/10.
    axs[0].legend(loc="upper right", frameon=True)
    fig.tight_layout()
    save(fig, out_path)
    print("  scritto", out_path)
    return True


def fig_tv_rotors_range(cats, V, stat, key, sym, unit, out_path, stem):
    """Dedicated figure for TV rotors to match the style and dimensions
    of pos_att_pair_RMSE.pdf (height 4.6)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    if not _draw_cat_range(ax, cats, V, stat, key, sym, unit):
        plt.close(fig)
        return False
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    save(fig, out_path)
    print("  scritto", out_path)
    return True


def read_tv_rotors(folder: Path):
    """{(group, rotore): TV sul volo intero per singolo rotore} da
    tv_windows.csv, window=='flight', canali M1..M6 (TOTAL escluso: la somma
    sui sei rotori resta in read_tv_flight, qui serve il dettaglio per
    motore). Le chiavi restano quelle del CSV (M1..M6); la conversione alle
    etichette di tesi (P1..P6, Sezione 2.4.1) avviene in
    tv_rotors_range_section, non qui."""
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
    """Variazione totale del comando per singolo rotore (eq:total_variation),
    volo intero, confronto nominal/perturbed con lo stesso disegno punto
    medio+range di compare_categories_section (_draw_cat_range/
    fig_cat_range, D29) -- qui 'cats' sono i sei rotori invece delle
    categorie di manovra, su richiesta di Marco il 6/10 (adatta
    flight/propellers/summary_tv.pdf, nominal-only e senza range, allo
    stile delle altre figure di Sezione 5.3). Le etichette in figura sono
    P1..P6 (Sezione 2.4.1): quelle del CSV grezzo (M1..M6) restano solo in
    tv_comparison_rotors.csv.

    Scrive tables/tv_comparison_rotors.csv e la figura
    propellers/03_tv_rotors_range.pdf."""
    rotor_labels = {f"M{i}": f"P{i}" for i in range(1, 7)}
    Tn, Tp = read_tv_rotors(nom_dir), read_tv_rotors(pert_dir)
    if not Tn or not Tp:
        print("  [tv rotori] tv_windows.csv senza righe per motore in una "
              "delle due cartelle, salto")
        return

    rotors = [f"M{i}" for i in range(1, 7)
              if all((g, f"M{i}") in Tn and (g, f"M{i}") in Tp for g in GROUPS)]
    if not rotors:
        print("  [tv rotori] nessun rotore in comune, salto")
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
        row = {'rotore': lab}
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
    print(f"  scritto {csv_path}")

    out_path = dirs['propellers'] / "03_tv_rotors_range.pdf"
    if fig_tv_rotors_range(labels, V, '-', 'tv', r'$\mathrm{TV}_i$', 'Hz/s',
                           out_path, stem):
        print(f"  scritto {out_path}")


def compare_categories_section(nom, pert, dirs, stem, exclude):
    """Confronto per categoria: tables/comparison_categories.csv
    e le due figure in categories/."""
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
        raise SystemExit("Nessuna categoria presente in entrambe le condizioni.")
    # (statistica, metrica): '-' sono quelle che una statistica ce l'hanno gia'
    # dentro (lag_z e' una media, t10_s un tempo, tv una Total Variation)
    # Anche i contatori _abs sono bookkeeping, non metriche: con le colonne
    # D e DELTA accanto finivano per avere un "esito" su un numero di
    # campioni, che non vuol dire niente.
    mets = [x for x in dict.fromkeys(
        (k[3], k[4]) for k in V if k[2] in cats and k[4] not in
        ("n_samples", "n_subblocks", "dur_s",
         "n_samples_abs", "n_subblocks_abs",
         "n_subblocks_real", "n_subblocks_real_att"))]
    stats_av = [st for st, _a in STATS if any(x[0] == st for x in mets)]

    print(f"\n{'='*100}\nnominal vs perturbed per categoria  -  {stem}\n{'='*100}")
    print(f"  nominal  : {nom.name}\n  perturbed: {pert.name}")
    print(f"  categorie : {', '.join(cats)}")

    rows, n_sub1 = [], 0
    for cat in cats:
        print(f"\n{'-'*100}\n  [{cat}]\n{'-'*100}")
        print(f"    {'metrica':14s}{'PPO nom':>11s}{'PPO pert':>11s}{'SPPO nom':>11s}"
              f"{'SPPO pert':>11s}{'D_PPO':>8s}{'D_SPPO':>8s}"
              f"{'DELTA_PPO':>11s}{'DELTA_SPPO':>11s}   esito")
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
                 # DELTA: lo spostamento causato dal disturbo. E' la lettura
                 # da usare per la robustezza -- il disturbo e' lo stesso per
                 # le due policy, quindi la domanda e' di quanto si sposta
                 # l'errore, e uno spostamento e' una differenza. Vedi
                 # compare_cell e la cartella maneuvers/delta/.
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
                      f"{sn-pn:+8.4f} {sp-pp:+8.4f} {u:<6s}  (con segno)")
            else:
                # le tre letture, e cosa dicono. Si contraddicono spesso:
                # D penalizza chi parte gia' basso, quindi puo' dare
                # "degrada di piu'" proprio a chi sbaglia meno in entrambe le
                # condizioni. Quando succede va detto sulla riga, non lasciato
                # trovare a chi legge.
                da, dl = r['delta_abs_ppo'], r['delta_abs_sppo']
                w_pert = 'SPPO' if sp < pp else 'PPO'
                w_D = 'SPPO' if ds < dp else 'PPO'
                w_del = 'SPPO' if abs(dl) < abs(da) else 'PPO'
                note = ""
                if w_D != w_del:
                    note = f"  !! D dice {w_D}, DELTA dice {w_del}"
                elif w_D != w_pert:
                    note = f"  !! sbaglia meno {w_pert}, degrada meno {w_D}"
                if min(da, dl) < 0:
                    note += "  (DELTA<0: rumore)"
                print(f"    {lab_m:20s}{pn:11.4f}{pp:11.4f}{sn:11.4f}{sp:11.4f}"
                      f"{dp:8.2f}{ds:8.2f}{da:+11.4f}{dl:+11.4f}{note}")

    csv_path = dirs['tables'] / "comparison_categories.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  scritto {csv_path}")

    # -------------------------------------------- quanto vale il rumore
    # Il disturbo e' un carico aggiunto: l'errore puo' solo crescere. Ogni
    # DELTA negativo e' quindi variabilita' volo-su-volo, non un risultato, e
    # il piu' grande di quelli da' la soglia sotto cui nessuna differenza fra
    # le due policy e' leggibile. Confrontarla con la differenza fra i DELTA
    # delle due policy dice, in una riga, se c'e' un segnale o no.
    _pos = [r for r in rows if not r['signed']
            and r['delta_abs_ppo'] == r['delta_abs_ppo']
            and r['delta_abs_sppo'] == r['delta_abs_sppo']
            and r['metric'] in ('e_p', 'rmse_pos')]
    if _pos:
        neg = [(r, v) for r in _pos
               for v in (r['delta_abs_ppo'], r['delta_abs_sppo']) if v < 0]
        gap = [abs(r['delta_abs_sppo'] - r['delta_abs_ppo']) for r in _pos]
        print(f"\n{'-'*100}\n  ROBUSTEZZA: C'E' UN SEGNALE?\n{'-'*100}")
        print("  Con un carico aggiunto l'errore puo' solo crescere: ogni DELTA")
        print("  negativo e' variabilita' volo-su-volo, e ne misura la scala.")
        if neg:
            worst = max(abs(v) for _r, v in neg)
            print(f"    {len(neg)} DELTA negativi, il piu' grande "
                  f"{worst*1000:.1f} mm -> rumore >= {worst*1000:.0f} mm")
        else:
            worst = 0.0
            print("    nessun DELTA negativo: il rumore non e' stimabile da qui.")
        print(f"    differenza fra i DELTA di PPO e SPPO: "
              f"{min(gap)*1000:.1f}-{max(gap)*1000:.1f} mm "
              f"(mediana {sorted(gap)[len(gap)//2]*1000:.1f} mm)")
        if worst > 0 and sorted(gap)[len(gap)//2] < worst:
            print("    -> l'effetto e' PIU' PICCOLO del rumore che i dati stessi")
            print("       mostrano: con un volo per condizione non e' distinguibile")
            print("       da zero. Per un'affermazione sulla robustezza servono")
            print("       piu' livelli di disturbo (una pendenza, non due punti)")
            print("       oppure repliche appaiate su piu' policy.")
        elif worst > 0:
            print("    -> l'effetto supera il rumore stimato: la differenza fra le")
            print("       due policy e' leggibile, ma resta un volo per condizione.")

    shown = [c for c in cats if c not in exclude]

    # --- stessa sostituzione di analyze_trajectory: nelle FIGURE le categorie
    # a gradino non vanno in valore assoluto, perche' i quattro voli non
    # percorrono la stessa distanza e la barra piu' alta sarebbe solo la piu'
    # lunga. Si disegna l'errore normalizzato campione per campione sulla
    # distanza da coprire (tutti i setpoint, SETPOINT 1 compreso) riportato a
    # una distanza comune ai QUATTRO voli, cosi' resta in metri. Il valore
    # assoluto resta in comparison_categories.csv.
    # NORMALIZZAZIONE, gradini E rampe.
    #
    # Il gradino: i quattro voli non percorrono la stessa distanza, la barra
    # piu' alta sarebbe solo la piu' lunga.
    # La rampa: il drone non viene riposato a terra alla stessa quota, quindi
    # l'atterraggio del perturbed scende 0.78 m dove il nominal ne scende
    # 0.83 -- stessa identica trappola. analyze_trajectory scrive per
    # entrambe le stesse colonne (*_pooled_n_*, *_pooled_den): l'errore
    # normalizzato campione per campione sulla distanza da coprire.
    #
    # Qui le si riporta a un denominatore COMUNE ai quattro voli, cosi' il
    # numero resta in metri e sta sullo stesso asse delle categorie di
    # mantenimento, che una distanza da coprire non ce l'hanno. Il valore
    # assoluto non si perde: resta in comparison_categories.csv.
    ORD4 = [("PPO", "nom"), ("PPO", "pert"), ("SPPO", "nom"), ("SPPO", "pert")]
    for cat in shown:
        if KIND.get(cat) not in ('gradino', 'rampa'):
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
            print(f"  FIGURE  {cat} ({fld}): denominatori "
                  + " / ".join(f"{v:.3f}" for v in den)
                  + f" {u} -> {dbar:.3f} {u} comuni")

    # maneuvers/ = decollo, mantenimento, atterraggio. Il gradino sta in
    # maneuvers/step/, come in data_analysis.
    shown_man = [c for c in shown if KIND.get(c) != 'gradino']
    shown_step = [c for c in shown if KIND.get(c) == 'gradino']
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
            print(f"  {n_fig} figure scritte in {d}")
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
            print(f"  {n_fig} figure scritte in {d}")
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
        print(f"  {n_fig} figure scritte in {d}")

    # ---- sintesi onesta
    #
    # Due metriche NON vanno contate cosi' come sono, e contarle falsava il
    # riepilogo che stampavo prima:
    #
    #  - in 'Step' rmse_pos e' proporzionale all'ampiezza del gradino, che
    #    NON e' identica fra i due voli. Li' si conta rmse_n, normalizzata.
    #  - rmse_att e' dominata dallo yaw, che su questo velivolo porta un offset
    #    da errore di montaggio comune alle due policy. Si riportano separate
    #    la componente regolata (att_rp) e quella di imbardata (att_yaw).
    SUMMARY_KEY = {"Step": "pos_pooled_n_RMSE"}   # posizione: cosa contare dove

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

    print(f"\n{'='*100}\nSINTESI\n{'='*100}")
    print("  POSIZIONE  (in 'Step' si conta pos_pooled_n_RMSE, normalizzata "
          "sulla distanza da coprire)")
    for lab, g in (("nominal ", lambda r: r["delta_nom_pct"]),
                   ("perturbed", lambda r: r["delta_pert_pct"]),
                   ("degrada meno", lambda r: r["delta_D_pct"])):
        w, n, sel = tally(g, lambda c: SUMMARY_KEY.get(c, "rmse_pos"), cats)
        det = "  ".join(f"{c.split()[0][:6]}:{v:+.0f}%" for c, _m, v in sel)
        print(f"    {lab:14s} SPPO meglio {w}/{n}    {det}")
    print("  COMPONENTE ORIZZONTALE  (deriva laterale: nessuna ampiezza comandata)")
    for lab, g in (("nominal ", lambda r: r["delta_nom_pct"]),
                   ("perturbed", lambda r: r["delta_pert_pct"])):
        w, n, sel = tally(g, lambda c: "err_xy", cats)
        det = "  ".join(f"{c.split()[0][:6]}:{v:+.0f}%" for c, _m, v in sel)
        print(f"    {lab:14s} SPPO meglio {w}/{n}    {det}")
    print("  ASSETTO  (separato: parte regolata e offset di imbardata)")
    for nm, met in (("roll+pitch", "att_rp"), ("yaw", "att_yaw")):
        for lab, g in (("nominal ", lambda r: r["delta_nom_pct"]),
                       ("perturbed", lambda r: r["delta_pert_pct"])):
            w, n, sel = tally(g, lambda c: met, cats)
            if not n:
                continue
            det = "  ".join(f"{c.split()[0][:6]}:{v:+.0f}%" for c, _m, v in sel)
            print(f"    {nm:11s}{lab:11s} SPPO meglio {w}/{n}    {det}")
    n_tot = sum(1 for r in rows if not r["signed"]) * 2
    if n_sub1:
        print(f"\n  {n_sub1} valori di D su {n_tot} sotto 0.95: il volo perturbed e' andato meglio")
        print("  del nominal. Non ha spiegazione fisica con un carico aggiunto:")
        print("  e' la variabilita' volo-su-volo, e va dichiarata come tale.")
    print("\n  NB: un volo per condizione. Valori singoli, non stime con incertezza.")


def fig_metric_degradation(metric, c, out_path, stat='RMSE'):
    """Una metrica, due barre: quanto degrada ciascuna policy.

    D = perturbed / nominal. Va letto accanto ai valori assoluti, mai da
    solo: penalizza strutturalmente chi parte da un errore piu' basso."""
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
               label='$D = 1$ (nessuna degradazione)')
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
    set_style()
    # Stesso albero di data_analysis, piu' la dimensione che il confronto ha
    # e l'analisi di un singolo scenario non ha: la degradazione.
    #
    # Niente mission/ ne' summary/. Erano la stessa domanda posta su un
    # ritaglio temporale -- "come e' andata da t0 a t1" -- e un ritaglio non
    # ha un nome: per leggere quei numeri bisognava prima sapere dove cadeva
    # la finestra e verificare che cadesse nello stesso posto nelle due
    # cartelle. Qui si confrontano MANOVRE: decollo, mantenimento,
    # atterraggio e risposta al gradino. Ognuna ha una finestra che si
    # riconosce dal riferimento, non dall'orologio.
    dirs = {'tables': out_dir / "tables",
            'roll': out_dir / "Rollouts",
            'propellers': out_dir / "propellers",
            'man_RMSE': out_dir / "maneuvers" / "RMSE",
            'man_MAE': out_dir / "maneuvers" / "MAE",
            'man_degradation': out_dir / "maneuvers" / "degradation",
            # delta/ accanto a degradation/: lo stesso confronto letto come
            # differenza invece che come rapporto. Sono due letture diverse e
            # sui dati si contraddicono spesso, quindi stanno in due cartelle.
            'man_delta': out_dir / "maneuvers" / "delta",
            # range/: punto medio nominal-perturbed piu' baffo fra i due,
            # la sola delle quattro letture pensata per finire in tesi
            # (Robustness, Sezione 5.3) -- vedi fig_cat_range.
            'man_range': out_dir / "maneuvers" / "range",
            'step_delta': out_dir / "maneuvers" / "step" / "delta",
            'step_range': out_dir / "maneuvers" / "step" / "range",
            # come in data_analysis: le manovre della missione da una parte,
            # il gradino dall'altra
            'step_RMSE': out_dir / "maneuvers" / "step" / "RMSE",
            'step_MAE': out_dir / "maneuvers" / "step" / "MAE",
            'step_degradation': out_dir / "maneuvers" / "step" / "degradation"}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print(f"nominal   : {nom_dir}")
    print(f"perturbed : {pert_dir}")
    print(f"uscita     : {out_dir}")
    print(f"soglia assunta: {floor:.1f}%  (sotto questa non si assegna un vincitore)\n")

    check_params(nom_dir, pert_dir)

    pm_nom, pm_pert = read_phase_metrics(nom_dir), read_phase_metrics(pert_dir)
    raw_nom, raw_pert = dict(pm_nom), dict(pm_pert)

    # --------------------------------------------- controllo di coerenza
    ph_nom = list(dict.fromkeys(p for _, p in pm_nom))
    ph_pert = list(dict.fromkeys(p for _, p in pm_pert))
    phases = [p for p in ph_nom if p in ph_pert]
    if not phases:
        sys.exit("Nessuna fase in comune fra le due cartelle: sono lo stesso task?")
    print(f"{'='*78}\nCOERENZA FRA LE DUE CARTELLE\n{'='*78}")
    print(f"  fasi nominal  : {', '.join(ph_nom)}")
    print(f"  fasi perturbed: {', '.join(ph_pert)}")
    if ph_nom != ph_pert:
        print("  !! le due sequenze di fasi NON coincidono: confronto limitato a")
        print(f"     quelle in comune ({', '.join(phases)}). Verifica che i due")
        print("     voli abbiano eseguito la stessa missione.")
    else:
        print("  -> stessa sequenza di fasi.")

    # I SETPOINT DI MISSIONE devono coincidere anche fra nominal e perturbed,
    # altrimenti D confronta due missioni diverse. Decollo e atterraggio restano
    # fuori dal controllo: il loro riferimento e' la posa a terra del drone, che
    # cambia a ogni volo per costruzione -- li' conta lo spostamento comandato,
    # gia' verificato da analyze_trajectory dentro ciascuna cartella.
    worst = 0.0
    for p in [x for x in phases if x.startswith('SETPOINT')]:
        for g in GROUPS:
            a, b = pm_nom.get((g, p)), pm_pert.get((g, p))
            if not a or not b:
                continue
            d = max(abs(a[k] - b[k]) for k in ('ref_x', 'ref_y', 'ref_z')
                    if a[k] == a[k] and b[k] == b[k])
            worst = max(worst, d)
    print(f"  scarto massimo fra i setpoint di missione delle due condizioni: "
          f"{worst*1000:.1f} mm")
    if worst > 0.01:
        print("  !! oltre 1 cm: nominal e perturbed non inseguono lo stesso")
        print("     riferimento, D non e' interpretabile come sola degradazione.")
    else:
        print("  -> stessi setpoint di missione: D misura solo l'effetto del")
        print("     disturbo.")

    # RAMPE. Decollo e atterraggio partono dalla posa a terra, che cambia a
    # ogni volo: la rampa non e' lunga uguale nelle due cartelle. Finora il
    # controllo le saltava -- ma le categorie 'Takeoff' e 'Land' delle figure
    # in maneuvers/ sono proprio la finestra della rampa, quindi una rampa
    # piu' corta accorcia la finestra, e il confronto fra le due colonne
    # misura anche quella differenza. Va detto, non nascosto.
    print(f"\n  {'rampa':<12}{'grp':<6}{'dz nominal':>14}{'dz perturbed':>16}"
          f"{'scarto':>10}")
    ramp_bad = []
    for p in ('TAKEOFF', 'LANDING'):
        for g in GROUPS:
            a, b = raw_nom.get((g, p)), raw_pert.get((g, p))
            if not a or not b:
                continue
            za, zb = a.get('ramp_dz_m', NAN), b.get('ramp_dz_m', NAN)
            if za != za or zb != zb:
                print(f"  {p:<12}{g:<6}{'--':>14}{'--':>16}{'--':>10}"
                      "   (rigenera data_analysis: manca ramp_dz_m)")
                continue
            d = abs(zb - za)
            flag = ""
            if d > 0.02:
                flag = "   !! non confrontabile"
                ramp_bad.append((p, g, d))
            print(f"  {p:<12}{g:<6}{za:>13.3f}m{zb:>15.3f}m{d*1000:>7.0f}mm{flag}")
    if ramp_bad:
        labs = ", ".join(sorted({p for p, _g, _d in ramp_bad}))
        print(f"\n  !! {labs}: la rampa comandata e' lunga diversamente nelle due")
        print("     cartelle (il drone non si riposa a terra alla stessa quota).")
        print("     Nelle figure di maneuvers/ le barre di quella categoria")
        print("     coprono finestre di durata diversa: una barra piu' alta puo'")
        print("     essere solo una rampa piu' lunga. Leggile con questo in mente,")
        print("     Le figure normalizzano gia' quella categoria sul dislivello")
        print("     comandato, quindi la barra e' confrontabile; e' il valore")
        print("     ASSOLUTO in comparison_categories.csv che non lo e'.")
    else:
        print("\n  -> rampe di pari lunghezza: decollo e atterraggio sono")
        print("     confrontabili fra le due cartelle.")

    # --------------------------------------------- confronti per fase
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

    # --------------------------------------------- console: le due letture
    for key, label, unit, scale, all_phases in PHASE_METRICS:
        cells = all_cells[key]
        usable = [p for p in phases if cells[p]['ppo_nom'] == cells[p]['ppo_nom']
                  and cells[p]['sppo_pert'] == cells[p]['sppo_pert']]
        if not usable:
            continue
        print(f"\n{'='*94}\n{SHORT.get(key, key).upper()}\n{'='*94}")
        print(f"  {'fase':<12}{'PPO nom':>10}{'PPO per':>10}{'SPPO nom':>10}"
              f"{'SPPO per':>10}{'D PPO':>8}{'D SPPO':>8}"
              f"{'chi sbaglia meno':>20}{'chi ha D minore':>20}")
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
        # conteggio, per fase
        n_p = sum(1 for p in usable if cells[p]['win_pert'] == 'SPPO')
        n_d = sum(1 for p in usable if cells[p]['win_D'] == 'SPPO')
        n_t = sum(1 for p in usable if cells[p]['win_pert'])
        print(f"  -> col disturbo SPPO sbaglia meno in {n_p}/{n_t} fasi; "
              f"ha D minore in {n_d}/{n_t}.")

    # ------------------------------------------- sforzo di controllo
    # La TV sul volo intero, da tv_windows.csv. E' l'unico numero "globale"
    # che resta: non dipende da un ritaglio scelto a mano, e serve a dire
    # quanto costa in comando la differenza che si legge nelle manovre.
    glob_rows = []
    tv_nom, tv_pert = read_tv_flight(nom_dir), read_tv_flight(pert_dir)
    tv_cell = None
    if tv_nom and tv_pert:
        tv_cell = compare_cell(tv_nom, tv_pert, floor)
        glob_rows.append(dict(metric='Total variation - tutti i motori',
                              **{k: (round(v, 6) if isinstance(v, float) else v)
                                 for k, v in tv_cell.items()}))
        c = tv_cell
        print(f"\n{'='*94}\nSFORZO DI CONTROLLO SUL VOLO INTERO\n{'='*94}")
        print(f"  {'metrica':<28}{'PPO nom':>10}{'PPO per':>10}{'SPPO nom':>10}"
              f"{'SPPO per':>10}{'D PPO':>8}{'D SPPO':>8}"
              f"{'sbaglia meno':>16}{'D minore':>16}")
        print(f"  {'Total Variation':<28}{c['ppo_nom']:>10.1f}{c['ppo_pert']:>10.1f}"
              f"{c['sppo_nom']:>10.1f}{c['sppo_pert']:>10.1f}"
              f"{c['D_ppo']:>8.2f}{c['D_sppo']:>8.2f}"
              f"{c['win_pert'] or '--':>16}{c['win_D'] or '--':>16}")

    # --------------------------------------------- figure
    # Le figure per fase (cartella 'paired') non vengono piu' prodotte: il
    # confronto per fase resta nelle tabelle e a console.
    if tv_cell is not None:
        mtv = 'Total variation - tutti i motori'
        n_fig = 0
        # la TV non e' un errore: l'etichetta dell'asse e' il suo simbolo,
        # non 'RMSE'
        if fig_metric_bars(mtv, tv_cell,
                           dirs['propellers'] / "01_total_variation.pdf", floor,
                           stat=r'$TV_{\omega}$'):
            n_fig += 1
        if fig_metric_degradation(mtv, tv_cell,
                                  dirs['propellers'] / "02_total_variation_D.pdf",
                                  stat=r'$TV_{\omega}$'):
            n_fig += 1
        print(f"\n  {n_fig} figure scritte in {dirs['propellers']}")

    # confronto per rotore (P1..P6) della TV sul volo intero,
    # stesso stile punto medio+range delle altre figure di 5.3
    # (Marco, 6/10, vedi tv_rotors_range_section)
    rotor_stem = nom_dir.name.replace("_nominal", "").replace("_nom", "")
    tv_rotors_range_section(nom_dir, pert_dir, dirs, rotor_stem)

    # sovrapposizione nel tempo: serve la serie grezza (vedi load_series)
    print(f"\n{'='*78}\nSERIE GREZZE PER LE CURVE\n{'='*78}")
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
        print("  nessun data.npz leggibile: niente figure di sovrapposizione.")
    elif not all_ok:
        print("  !! le serie ricalcolate qui non coincidono con i CSV: le figure")
        print("     di sovrapposizione NON vengono disegnate. Controlla che i CSV")
        print("     siano stati rigenerati dopo l'ultima modifica ai dati.")
    else:
        # confini di fase presi dal volo nominal PPO: stessa missione per
        # tutti e quattro, quindi bastano quelli come riferimento visivo
        bounds = sorted({pm_nom[('PPO', p)]['t0'] for p in phases
                         if ('PPO', p) in pm_nom
                         and pm_nom[('PPO', p)]['t0'] == pm_nom[('PPO', p)]['t0']})[1:]
        missing = [key for key, _ in OVERLAY_PANELS
                  if not any(key in f for f in flights.values())]
        if missing:
            print(f"  [attenzione] mancano in data.npz: {', '.join(missing)} -- "
                  "quei pannelli restano vuoti nella griglia")
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
            print(f"  [step profile] missione '{mission_stem}' non in "
                  "MISSION_STEP_CHANNEL, figura/tabella non scritte")

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
        fh.write("% D = perturbed / nominal. D > 1 = peggioramento.\n")
        fh.write(f"% Un volo per condizione: differenze sotto il {floor:.0f}% "
                 "assunto non sono\n")
        fh.write("% commentate. La soglia e' una convenzione dichiarata, non "
                 "una stima statistica.\n")
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

    # ---------------------------------------------- confronto per categoria
    stem = nom_dir.name.replace("_nominal", "").replace("_nom", "")
    compare_categories_section(nom_dir, pert_dir, dirs, stem, cat_exclude)

    print(f"\n  Promemoria per il testo: D e' il rapporto fra due voli singoli.")
    print(f"  La soglia del {floor:.0f}% e' una convenzione dichiarata, non una")
    print( "  stima di incertezza. Quello che regge e' la coerenza del segno su")
    print( "  piu' fasi e piu' traiettorie, non il singolo numero.")
    print(f"\nConfronto completato. Output in: {out_dir}")


def default_output(nom_dir: Path, pert_dir: Path) -> Path:
    """p2p_nominal + p2p_perturbed -> p2p_comparison, accanto alle due."""
    a, b = nom_dir.name, pert_dir.name
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    prefix = a[:i].rstrip('_- ') or f"{a}_vs_{b}"
    return nom_dir.parent / f"{prefix}_comparison"


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Confronto nominal vs perturbed per un task TiltHex.")
    p.add_argument("-n", "--nominal", required=True, type=str,
                   help="cartella del caso nominal (es. plots/p2p_nominal)")
    p.add_argument("-p", "--perturbed", required=True, type=str,
                   help="cartella del caso perturbed")
    p.add_argument("-o", "--output", type=str, default=None,
                   help="cartella di uscita (default: <prefisso>_comparison "
                        "accanto alle due di ingresso)")
    p.add_argument("--cat-exclude", nargs="*", default=[], metavar="CAT",
                   help="categorie da NON disegnare nelle figure di "
                        "categories/ (restano nel CSV)")
    p.add_argument("--noise-floor", type=float, default=5.0,
                   help="soglia percentuale assunta sotto la quale non si "
                        "assegna un vincitore (default 5)")
    args = p.parse_args()

    nom_dir, pert_dir = Path(args.nominal), Path(args.perturbed)
    for d in (nom_dir, pert_dir):
        if not d.is_dir():
            sys.exit(f"Errore: cartella '{d}' non trovata.")
    out_dir = Path(args.output) if args.output else default_output(nom_dir, pert_dir)
    run(nom_dir, pert_dir, out_dir, args.noise_floor,
        cat_exclude=args.cat_exclude)