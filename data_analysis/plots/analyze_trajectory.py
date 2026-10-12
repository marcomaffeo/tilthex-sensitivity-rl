#!/usr/bin/env python3
"""
analyze_trajectory.py  --  Analisi delle missioni di traiettoria: PPO vs SPPO.

Produce una cartella data_analysis con DUE rami e le tabelle:

    flight/      il log intero: curve d'errore nel tempo, Rollouts, TV.
                 DESCRIVE il volo. Contiene il decollo, che parte dalla posa
                 a terra -- diversa a ogni volo -- quindi non confronta.
    maneuvers/   il volo diviso per manovra: decollo, mantenimento,
                 atterraggio, piu' maneuvers/step/ per la risposta al
                 gradino. E' qui che si CONFRONTA: ogni numero e' riferito a
                 una finestra che si riconosce dal riferimento, non a un
                 ritaglio dell'orologio.
    tables/      i CSV, compreso params.csv con i tagli usati.

Si applica alle cartelle che contengono DUE voli -- PPO e SPPO della stessa
policy -- che inseguono lo STESSO riferimento:

    p2p_nominal   p2p_perturbed   rpy_nominal   rpy_perturbed

NATURA DEI DATI  (da tenere a mente leggendo qualunque numero prodotto qui)
--------------------------------------------------------------------------
Un volo per cella: 2 traiettorie x 2 algoritmi x 2 condizioni = 8 voli.
Sono CASI DI STUDIO, non un campione. Le fasi di una missione non sono
campioni indipendenti -- vengono dallo stesso volo -- quindi contarle come
tali sarebbe pseudo-replicazione. Di conseguenza qui non si calcola nessun
test statistico, nessun p-value, nessuna banda di incertezza: solo valori
singoli, da riportare in tesi con linguaggio descrittivo.

DIFFERENZE RISPETTO ALLA CAMPAGNA DI HOVERING
---------------------------------------------
  * Un volo per algoritmo, non 15 policy appaiate: niente medie fra policy,
    niente bande +/- sigma, niente p-value.
  * Il riferimento NON e' costante: cambia a gradino a ogni fase della
    missione. Il volo va quindi SEZIONATO, non croppato a un orizzonte fisso.
    I confini si rilevano dal riferimento; le finestre di staging fra
    publish_absolute_setpoint e /policy/start vengono assorbite nel decollo o
    nell'atterraggio.
  * Oltre all'RMSE sull'intero volo, le metriche sono calcolate PER FASE
    separando transitorio e regime.

FASI NON CONFRONTABILI NEL TRANSITORIO
--------------------------------------
Il drone non puo' essere riposizionato a terra esattamente nello stesso punto
a ogni volo, quindi il gradino che porta al primo setpoint di missione ha
ampiezza leggermente diversa fra PPO e SPPO (nei dati: 1.35 vs 1.53 m). Su un
gradino piu' lungo il picco di errore e' piu' grande per costruzione, non per
merito del controllore: il transitorio di quella fase NON e' confrontabile.

Lo script se ne accorge da solo -- confronta l'ampiezza del comando in
ingresso a ogni fase fra i due voli -- e per le fasi che risultano diverse:
  * azzera (n.c.) RMSE sull'intera fase, RMSE di transitorio, picco, tempo di
    assestamento e TV di fase, nei valori ASSOLUTI;
  * tiene l'errore A REGIME, che dipende dal setpoint finale (identico) e non
    dal cammino percorso per arrivarci;
  * ESCLUDE quella finestra di transitorio dalle metriche globali sull'intero
    volo, che altrimenti la conterrebbero.
Le curve restano disegnate per intero: si esclude dal CONTO, non dal grafico.

TRANSITORIO NORMALIZZATO
------------------------
Buttare il transitorio pero' costa informazione, e il controllo classico ha una
risposta migliore: si divide l'errore per l'ampiezza del gradino. In un sistema
lineare la risposta al gradino scala con l'ampiezza, quindi e(t)/Delta non
dipende da Delta -- e' lo stesso motivo per cui la sovraelongazione si riporta
in percentuale e non in metri, e per cui tempo di salita e di assestamento sono
specifiche indipendenti dall'ampiezza.

Lo script calcola quindi, accanto ai valori assoluti, le versioni normalizzate
(picco / Delta, RMSE di transitorio / Delta, IAE / Delta), che restano
confrontabili anche sulle fasi marcate non confrontabili in assoluto.

LIMITE DA DICHIARARE IN TESI: l'invarianza rispetto all'ampiezza vale nel
regime lineare. Con gli attuatori in saturazione un gradino piu' lungo degrada
di piu' anche in rapporto. Qui la saturazione e' marginale e i gradini
differiscono di poco, quindi l'approssimazione regge -- ma e' un'assunzione,
e va scritta. Per questo la normalizzazione AFFIANCA il regime, non lo
sostituisce: il regime non assume nulla, e se le due letture concordano la
conclusione non dipende dalla metrica scelta (analisi di sensibilita').

La normalizzazione si applica solo dove il transitorio e' davvero una risposta
al gradino, cioe' alle fasi di hold raggiunte da un salto del riferimento.
Decollo e atterraggio sono RAMPE: il loro salto in ingresso e' ~0 e dividere
per quello darebbe numeri enormi e privi di senso.

ASSUNZIONI SUL FORMATO DEI DATI (le stesse di analyze_campaign.py)
------------------------------------------------------------------
  * data['action'] contiene 6 FORZE in NEWTON, una per rotore.
  * data['quat'] e data['quat_ref'] sono in convenzione WXYZ.
    >>> Se cambia il nodo di logging, ricontrolla qui: lo script produrrebbe
    >>> numeri sbagliati in silenzio.

Definizioni di errore identiche a quelle della campagna:
  e_p = pos - pos_ref                 |e_p| invariante rispetto al frame
  e_b = R_ref^-1 (pos - pos_ref)      componenti nel frame del riferimento
  e_R = rotvec(R_ref^-1 R_act)        log-map nel frame del RIFERIMENTO;
                                      |e_R| e' geodetica, quindi invariante

Le componenti nel frame MONDO non compaiono: lo yaw di riferimento cambia
fra le fasi, quindi e_x, e_y, e_z non sono confrontabili per asse.

ATTENZIONE al significato di e_b: e' l'errore nel frame del RIFERIMENTO, non
del corpo. Dove il riferimento ruota durante la missione -- p.es. lo scenario
di assetto, dove lo yaw comandato passa da 0 a 40 gradi -- gli assi x, y di
e_b ruotano con lui, e la metrica globale per asse somma componenti misurate
lungo direzioni fisiche diverse. Il confronto fra due voli resta valido (stesso
frame per entrambi), ma 'errore in X' non e' una direzione fissa dello spazio:
per il confronto fra scenari usare la norma |e_p|, che e' invariante.

Lo script e' AUTONOMO: non richiede nessun altro file.

Uso:
    python analyze_trajectory.py --input plots/p2p_nominal
    python analyze_trajectory.py --input plots/rpy_perturbed --t-ss 2.5
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  (registra il 3d)
from scipy.spatial.transform import Rotation

# --------------------------------------------------------------- costanti
CF = 12.0e-4                   # coefficiente di spinta [N / (rad/s)^2]
F_MIN, F_MAX = 0.3072, 12.0    # limiti di forza per rotore [N]
# NB: i log delle missioni toccano esattamente 0.3072 e 12.0000 N, quindi
# questi SONO i limiti reali dell'attuatore per questi voli. I valori
# precedenti (0.288 / 11.25 / 11.25e-4, tuttora in analyze_campaign.py)
# venivano dalla campagna di hovering e qui tagliavano comandi veri.
N_ROTORS = 6

C_PPO, C_SPPO = '#ff7f0e', '#1f77b4'  # arancione / blu, come export_campaign.py
C_REF = '#222222'
COL = {'PPO': C_PPO, 'SPPO': C_SPPO}
# Tratteggi sfalsati per le curve di riferimento: i due riferimenti
# coincidono da SETPOINT 1 in poi, e con lo stesso tratteggio uno
# nasconderebbe l'altro. Cosi' restano visibili entrambi.
DASH_REF = [(0, (7, 4)), (4, (7, 4))]
GROUPS = ('PPO', 'SPPO')

RAD2DEG = 180.0 / math.pi
_ESC_SPACE = "\\ "


def tex(s):
    """'Position Error X' -> 'Position\\ Error\\ X'. Il backslash sta fuori
    dall'f-string: dentro e' sintassi valida solo da Python 3.12 (PEP 701)."""
    return s.replace(" ", _ESC_SPACE)


def nan_to_blank(v):
    """nan -> '' nei CSV: una cella vuota si legge come 'non calcolabile',
    la stringa 'nan' si legge come un bug."""
    return '' if isinstance(v, float) and v != v else v


# ===========================================================================
# Metriche tracciate nel tempo.  (chiave, etichetta y, titolo, nome file)
# ===========================================================================
#   chiave, simbolo (senza modulo, senza unita'), unita', titolo, nome file
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

# metriche il cui riferimento e' nullo SOLO a regime: durante una transizione
# il drone deve muoversi, quindi li' un valore alto non e' un errore
MOTION_METRICS = {'v_norm', 'w_norm'}

# titolo lungo -> simbolo, per l'asse x dei riepiloghi. Dieci nomi per esteso
# ruotati di 35 gradi occupano meta' figura e non si leggono comunque.
SHORT = {t: rf'${s}$' for _k, s, _u, t, _f in METRICS}

# campi di phase_metrics che dipendono dal transitorio IN VALORE ASSOLUTO: per
# una fase con gradino di ampiezza diversa fra i due voli non sono
# confrontabili. I corrispettivi normalizzati (suffisso _norm) NON sono in
# questa lista: sono proprio quelli che restano validi.
TRANSIENT_FIELDS = ('rmse_pos_all', 'rmse_pos_transient', 'peak_pos',
                    'iae_pos', 'settle_pos_s', 'rmse_att_all',
                    'rmse_att_transient', 'peak_att', 'iae_att',
                    'settle_att_s', 'tv_total')

# ampiezza minima del gradino sotto la quale non ha senso normalizzare
MIN_STEP_NORM = 0.05         # [m]


# ===========================================================================
# RILEVAMENTO DELLE FASI DI MISSIONE
#
# Le missioni sono definite in policy_control_node_tilthex.py:
#
#   point-to-point  (--takeoff-zero-point-land)
#       FASE 1  takeoff (rampa)                                wait_s
#       FASE 2  primo punto   [0, 0, takeoff_z]                wait_s
#       FASE 3  secondo punto [x, y, takeoff_z + z]            wait_s
#       FASE 4  landing (rampa)                                wait_s
#
#   attitude manoeuvre (--takeoff-zero-attitude-zero-land)
#       FASE 1  takeoff (rampa)                                wait_s
#       FASE 2  zero point, assetto nullo                      wait_s
#       FASE 3  assetto target [roll, pitch, yaw]              wait_s
#       FASE 4  zero point con yaw target                      wait_s
#       FASE 5  landing (rampa)                                wait_s
#
# Il sezionamento NON e' a tempo fisso: i confini si leggono dal riferimento.
# Due tipi di evento, da trattare separatamente:
#   * SALTO A GRADINO: publish_absolute_setpoint cambia il setpoint da un
#     campione all'altro. Dura un campione solo, quindi un filtro basato sulla
#     durata lo cancellerebbe: va rilevato sull'incremento fra campioni
#     consecutivi.
#   * RAMPA sostenuta: decollo e atterraggio usano set_ramp_rate, quindi il
#     riferimento varia con continuita' per alcuni secondi.
# ===========================================================================

V_THR = 1e-3                 # [m/s]   sopra questa il riferimento "si muove"
W_THR = 1e-3                 # [rad/s]
MIN_BLOCK = 0.25             # [s]     rampe piu' corte assorbite nel vicino
STEP_POS = 0.02              # [m]     salto su un singolo campione
STEP_ANG = np.deg2rad(2.0)   # [rad]

# soglie oltre le quali il comando in ingresso a una fase e' considerato
# DIVERSO fra i due voli, e quindi il transitorio non confrontabile
STEP_TOL_POS = 0.02          # [m]
STEP_TOL_ANG = 2.0           # [deg]


def to_scipy_quat(q_wxyz):
    """wxyz (formato dei log) -> xyzw (formato che vuole scipy)."""
    q = np.asarray(q_wxyz, dtype=float)
    return np.column_stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]])


def _blocks(mask, t, min_dur):
    """mask booleana -> lista di (start, end, value), con i blocchi piu' corti
    di min_dur assorbiti nel precedente."""
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
    """Rileva le fasi dal riferimento. Ritorna una lista di dict con tipo
    ('hold'/'move'), indici, tempi, durata, bersaglio e (per le rampe)
    spostamento comandato."""
    t = f['t']
    dt = np.diff(t)
    R = f['R_ref']

    d_pos = np.linalg.norm(np.diff(f['pos_ref'], axis=0), axis=1)   # [m] per campione
    d_ang = (R[:-1].inv() * R[1:]).magnitude()                      # [rad] per campione
    v_ref, w_ref = d_pos / dt, d_ang / dt

    steps = np.flatnonzero((d_pos > cfg['step_pos']) | (d_ang > cfg['step_ang'])) + 1

    moving = (v_ref > cfg['v_thr']) | (w_ref > cfg['w_thr'])
    moving = np.concatenate([[moving[0]], moving])
    moving[steps] = False                       # il gradino non e' una rampa
    ramp_edges = [b[0] for b in _blocks(moving, t, cfg['min_block'])]

    bounds = sorted({0, len(t)} | set(int(i) for i in steps) | set(ramp_edges))
    phases = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        # il segmento arriva al campione PRIMA del confine successivo: l'indice
        # di confine porta gia' il setpoint nuovo, e includerlo farebbe
        # risultare "in movimento" una fase di hold
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
            # bersaglio a fine fase: serve alle metriche, che trattano hold e
            # rampe allo stesso modo
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

    # Hold brevissimi adiacenti a decollo o atterraggio: sono la finestra di
    # staging del nodo (0.5 + 0.2 s fra publish_absolute_setpoint e
    # /policy/start), in cui il riferimento e' ancora la verticale del punto di
    # posa. Non sono fasi a se': vengono ASSORBITE nel decollo o
    # nell'atterraggio, cosi' le fasi risultano esattamente quelle della
    # missione -- decollo, i setpoint, atterraggio -- senza frammenti spuri.
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
                absorb.append((i, i - 1))       # fondi nella rampa precedente
            elif next_lab in ('TAKEOFF', 'LANDING'):
                absorb.append((i, i + 1))       # fondi nella rampa seguente
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
    """Solo i setpoint comandati, senza decollo e atterraggio.

    Si chiamava mission_phases: il nome veniva dalla finestra di 'missione',
    che non esiste piu'. La funzione non c'entrava con quella finestra --
    seleziona le fasi a riferimento costante -- quindi resta, col nome giusto.
    """
    return [p for p in phases if p['label'].startswith('SETPOINT')]


def analysed_phases(phases):
    """Tutte le fasi su cui si calcolano le metriche: decollo, i setpoint,
    atterraggio. La policy pilota anche decollo e atterraggio, quindi sono
    prestazioni sue a tutti gli effetti -- e sono le fasi piu' severe.
    Restano fuori solo eventuali frammenti non classificati."""
    return [p for p in phases
            if p['label'] in ('TAKEOFF', 'LANDING')
            or p['label'].startswith('SETPOINT')]


def step_into(f, sp, j):
    """Ampiezza del comando che porta NELLA fase j: (spostamento [m], rotazione
    [deg]). Per la prima fase, che e' una rampa da terra, vale lo spostamento
    comandato dalla rampa stessa."""
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
    """Marca ogni fase come confrontabile o no nel TRANSITORIO.

    Criterio: se il comando che porta nella fase ha ampiezza diversa fra i due
    voli (oltre le tolleranze), il transitorio non e' confrontabile -- un
    gradino piu' lungo produce un picco piu' grande per costruzione. Il regime
    invece dipende solo dal setpoint finale, che e' identico, e resta valido.

    Nessuna fase e' scritta a mano nel codice: chi non e' confrontabile lo
    decidono i dati, volo per volo, cartella per cartella.
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
    """Ritorna (res, None) oppure (None, motivo_dello_scarto).
    Nessun file sparisce in silenzio: ogni scarto finisce nel manifest."""
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        return None, f"load_error: {e}"

    keys = set(data.files)
    need = {'t', 'pos', 'pos_ref', 'quat', 'quat_ref'}
    if not need <= keys:
        return None, f"mancano {sorted(need - keys)}"

    t = np.asarray(data['t'], dtype=float)
    if t.size < 3:
        return None, "serie temporale troppo corta"
    t = t - t[0]
    if np.any(np.diff(t) <= 0):
        return None, "tempi non monotoni"

    folder = npz_path.parent.name
    pname = str(data['policy_name']) if 'policy_name' in keys else ""
    tag = (folder + "_" + pname).upper()
    if "SPPO" in tag:
        group = "SPPO"
    elif "PPO" in tag:
        group = "PPO"
    else:
        return None, "gruppo non riconosciuto (ne' PPO ne' SPPO)"

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
        res['warnings'].append("manca 'vel'")
    if 'avel' in keys:
        res['w_norm'] = np.linalg.norm(np.asarray(data['avel'], float), axis=1)
    else:
        res['warnings'].append("manca 'avel'")
    if 'inf_time' in keys:
        it = np.asarray(data['inf_time'], float)
        res['inf_time_mean_ms'] = float(np.mean(it) * 1e3)
        res['inf_time_max_ms'] = float(np.max(it) * 1e3)

    if 'action' in keys:
        a = np.atleast_2d(np.asarray(data['action'], float))
        if a.shape[0] != t.size:
            res['warnings'].append(f"action {a.shape} non allineata con t ({t.size})")
        elif a.shape[1] < N_ROTORS:
            res['warnings'].append(f"action ha {a.shape[1]} canali, attesi {N_ROTORS}")
        else:
            # forze salvate GREZZE: il clip serve solo dentro la sqrt
            res['forces'] = a[:, :N_ROTORS]
            res['sat_frac'] = float(np.mean((a[:, :N_ROTORS] <= F_MIN * 1.001) |
                                            (a[:, :N_ROTORS] >= F_MAX * 0.999)))
    else:
        res['warnings'].append("manca 'action': nessuna Total Variation")
    return res, None


def compute_tv(f, mask):
    """TV della velocita' di rotore sui campioni selezionati, normalizzata sul
    tempo. La maschera puo' avere buchi (le finestre escluse dal confronto):
    la variazione si somma DENTRO ogni tratto contiguo, mai a cavallo di un
    buco, altrimenti il salto artificiale fra i due bordi entrerebbe nel conto.
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
    """RMS su un insieme di valori SENZA tempo: serve per aggregare scalari
    (p.es. un bias per sotto-blocco), non serie temporali."""
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x ** 2))) if x.size else float('nan')


# numpy >= 2.0 la chiama trapezoid, prima trapz
_TRAPZ = getattr(np, 'trapezoid', None) or np.trapz


def seg_mean_t(t, y, mask=None):
    """MEDIA TEMPORALE di y: integrale per trapezi diviso per la durata.

    Perche' non la media sui campioni. Il periodo del loop di logging non e'
    esattamente costante ne' identico fra i voli (qui 9.998-10.032 ms), quindi
    una media sui campioni pesa piu' i tratti campionati piu' densamente e due
    voli finiscono confrontati con pesi diversi. L'integrale nel tempo diviso
    per la durata e' la definizione che non dipende da come sono distribuiti i
    campioni, ed e' la prassi delle cifre di merito del controllo (IAE, ISE).

    La maschera puo' avere buchi -- le finestre escluse dal confronto: si
    integra DENTRO ogni tratto contiguo e si sommano numeratore e durata, mai
    a cavallo di un buco, altrimenti il trapezio fra i due bordi inventerebbe
    area su un intervallo che non e' stato misurato. Stessa regola di
    compute_tv().
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
    if den <= 0:                      # finestra degenere: resta la media
        return float(np.mean(y[idx]))
    return num / den


def rms_t(t, x, mask=None):
    """RMS nel tempo: sqrt( (1/T) * integrale di x^2 dt )."""
    v = seg_mean_t(t, np.asarray(x, float) ** 2, mask)
    return float(np.sqrt(v)) if v == v and v >= 0 else float('nan')


def mae_t(t, x, mask=None):
    """Errore assoluto medio nel tempo: (1/T) * integrale di |x| dt."""
    return seg_mean_t(t, np.abs(np.asarray(x, float)), mask)


def settling_time(t, e, t0, tol, smooth_s=0.25):
    """Tempo di assestamento: primo istante dopo t0 oltre il quale lo scarto
    passato in `e` resta sotto `tol`.

    Chi chiama passa lo scarto dal VALORE FINALE, non l'errore assoluto: una
    policy feed-forward non ha azione integrale e si assesta su un errore a
    regime non nullo, quindi una banda centrata su zero non viene mai
    raggiunta e il tempo di assestamento uscirebbe nan sempre. La banda
    centrata sul valore finale e' anche la convenzione del controllo classico
    (stepinfo). L'errore viene prima lisciato con una media mobile
    di smooth_s: sul segnale grezzo un singolo picco di rumore a fine fase
    farebbe risultare un assestamento pari all'intera durata della fase, che e'
    un artefatto del rumore e non un comportamento del controllore."""
    if len(t) < 3 or tol != tol:       # banda nan: la fase non ha gradino
        return float('nan')              # in questo canale, non c'e' nulla da
    dt = float(np.median(np.diff(t)))    # misurare
    k = max(1, int(round(smooth_s / max(dt, 1e-6))))
    if k > 1:
        es = np.convolve(e, np.ones(k) / k, mode='same')
        es[:k], es[-k:] = e[:k], e[-k:]        # i bordi restano grezzi
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
    """Metriche di una singola fase. Se il transitorio non e' confrontabile
    (gradino di ampiezza diversa fra i due voli) tutti i campi che ne dipendono
    escono nan: meglio una cella vuota che un numero che non significa quello
    che sembra."""
    ph = sp[j]
    i0, i1 = ph['i0'], ph['i1']
    sl = slice(i0, i1 + 1)
    t = f['t'][sl]
    ep, ea = f['e_pos_norm'][sl], f['e_att_norm'][sl]

    ss = t >= (t[-1] - cfg['t_ss'])            # regime: ultimi t_ss secondi
    tr = ~ss                                   # transitorio: il resto

    d_pos, d_ang = step_into(f, sp, j)
    ph_mask = np.zeros(f['n_samples'], dtype=bool)
    ph_mask[i0:i1 + 1] = True
    tv = compute_tv(f, ph_mask)

    # BANDA DI ASSESTAMENTO, centrata sul valore finale della fase stessa.
    #
    # Il tempo di assestamento esiste solo se in QUEL canale c'e' davvero un
    # gradino: in una rotazione pura il riferimento di posizione non si muove,
    # e "quanto ci mette la posizione ad assestarsi" non e' una domanda -- la
    # risposta uscirebbe 0 s, che si legge come un assestamento istantaneo e
    # non lo e'. Stesse condizioni della normalizzazione qui sotto: fase di
    # hold raggiunta da un salto del riferimento sopra la soglia minima.
    #
    # Ampiezza della banda: frazione del gradino, come nel controllo classico.
    # Nessuna soglia legata al rumore del singolo volo: la renderebbe diversa
    # fra i due voli e il piu' rumoroso verrebbe giudicato con un criterio
    # piu' severo. Con un gradino confrontabile l'ampiezza e' identica fra i
    # voli per costruzione, quindi la banda lo e' di conseguenza. Se la banda
    # e' piu' stretta del rumore l'errore non ci resta dentro e il tempo esce
    # nan: una cella vuota, non un numero deciso dal rumore.
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
        # 4 decimali, non 2: compare_trajectory.py apre la finestra di
        # missione a questo t0 letto dal CSV, e a 100 Hz un arrotondamento a
        # 10 ms sposta la finestra di un campione -- abbastanza per far
        # divergere i due script nella quinta cifra sulle stesse serie.
        't0': round(ph['t0'], 6), 't1': round(ph['t1'], 6), 'dur': round(ph['dur'], 2),
        'comparable_transient': int(bool(ph.get('comparable', True))),
        'step_pos_m': round(d_pos, 4), 'step_ang_deg': round(d_ang, 2),
        # Spostamento VERTICALE comandato dalla rampa, per decollo e
        # atterraggio. Non e' ricavabile da step_pos_m, che per una rampa
        # misura il salto IN INGRESSO alla fase (~0): qui serve la lunghezza
        # della rampa stessa. Il drone non viene riposizionato a terra alla
        # stessa quota a ogni volo, quindi questa lunghezza cambia da volo a
        # volo ed e' il confondente che rende la categoria 'Land' non
        # confrontabile fra due cartelle. Senza scriverla nel CSV,
        # compare_trajectory non ha modo di accorgersene.
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
        # iae_*: errore assoluto MEDIO nel tempo, non l'integrale grezzo --
        # cosi' resta in metri e confrontabile fra fasi di durata diversa
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
    # ||v|| e ||omega||: il riferimento di velocita' e' zero solo a REGIME
    for k, name in (('v_norm', 'v_norm_steady'), ('w_norm', 'w_norm_steady')):
        if k in f:
            m[name] = round(rms_t(t, f[k][sl], ss), 5)

    # ---------------------------------------------------------------------
    # TRANSITORIO NORMALIZZATO sull'ampiezza del gradino.
    # Un gradino piu' lungo produce un errore piu' grande per costruzione:
    # dividendo per l'ampiezza il transitorio torna confrontabile anche fra
    # voli partiti da posizioni a terra diverse (vedi docstring, compreso il
    # limite di validita' in saturazione).
    # Solo per le fasi di hold raggiunte da un vero salto del riferimento:
    # decollo e atterraggio sono rampe, il loro salto in ingresso e' ~0.
    # L'assetto si normalizza sullo STESSO gradino di posizione: in una fase
    # di pura traslazione l'errore di assetto e' accoppiamento indotto dalla
    # traslazione, quindi il rapporto e' un guadagno [rad per metro comandato]
    # -- non e' adimensionale e va letto per quello che e'.
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
    # TRANSITORIO NORMALIZZATO sulla DISTANZA REALE DA COPRIRE.
    #
    # Normalizzare sull'ampiezza comandata toglie un confondente e ne lascia
    # uno piu' sotto: all'istante del gradino il drone non e' sul setpoint
    # precedente, ci sta con il suo errore a regime, e quell'errore si somma o
    # si sottrae alla strada da fare. Sul secondo setpoint di p2p il comando e'
    # 0.7348 m per entrambi i voli, ma la distanza vera da coprire e' 0.6984 m
    # per uno e 0.7220 m per l'altro: il 3.4% in piu', che e' esattamente lo
    # scarto di RMSE che si misura nel primo mezzo secondo.
    #
    # Dividendo per la distanza vera il confondente cade alla radice, senza
    # tagliare campioni e senza scegliere soglie. E vale per TUTTI i gradini,
    # anche quelli con ampiezza comandata diversa: la normalizzazione assorbe
    # sia l'ampiezza sia l'errore a regime precedente. Per questo queste
    # colonne NON entrano in TRANSIENT_FIELDS e sopravvivono alle fasi non
    # confrontabili.
    #
    # NB: le metriche di stepinfo (sovraelongazione, salita, assestamento)
    # erano gia' immuni, perche' step_response() normalizza l'avanzamento
    # sull'escursione effettiva [p0, p_fin] -- e p0 e' proprio questo scarto.
    # ---------------------------------------------------------------------
    if j > 0:
        i_prev = sp[j - 1]['i1']
        d_real = float(np.linalg.norm(f['pos'][i_prev] - f['pos_ref'][i0]))
        a_real = float(np.linalg.norm(
            (f['R_ref'][i0].inv() * f['R_act'][i_prev]).as_rotvec()))
    else:
        d_real = a_real = nan
    # Condizionato al gradino COMANDATO, non alla distanza reale: in una fase
    # di sola rotazione il comando di posizione e' nullo e la distanza reale e'
    # solo l'errore residuo con cui il drone arriva -- normalizzare su quello
    # darebbe rapporti enormi e privi di significato, e con la soglia applicata
    # per volo tornerebbe anche l'asimmetria fra i due gruppi.
    real_ok = (ph['type'] == 'hold' and d_real == d_real and d_real > 1e-6
               and d_pos == d_pos and d_pos > cfg['min_step_norm'])
    real_ok_a = (ph['type'] == 'hold' and a_real == a_real and a_real > 1e-6
                 and d_ang == d_ang
                 and d_ang > cfg['settle_min_ang_deg'])
    m['norm_real_m'] = round(d_real, 4) if real_ok else nan
    m['norm_real_deg'] = round(float(np.rad2deg(a_real)), 3) if real_ok_a else nan
    m['rmse_pos_transient_real'] = (round(m['rmse_pos_transient'] / d_real, 5)
                                    if real_ok else nan)
    # NB: peak/d vale 1 per costruzione -- il picco della norma dell'errore
    # E' la distanza iniziale -- quindi non si riporta. Al suo posto l'IAE,
    # che e' la metrica che discrimina fra "arriva e resta" e "si trascina".
    m['iae_pos_real'] = (round(m['iae_pos'] / d_real, 5) if real_ok else nan)
    m['rmse_att_transient_real'] = (round(m['rmse_att_transient'] / a_real, 5)
                                    if real_ok_a else nan)
    m['iae_att_real'] = (round(m['iae_att'] / a_real, 5) if real_ok_a else nan)

    # I valori ASSOLUTI del transitorio cadono se la fase non e' confrontabile.
    # I normalizzati restano: sono calcolati qui sopra, prima di questa riga.
    if not ph.get('comparable', True):
        for k in TRANSIENT_FIELDS:
            m[k] = nan
    return m


def _smooth(t, y, win_s):
    """Media mobile su win_s secondi, bordi lasciati grezzi."""
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
    """stepinfo di una fase: sovraelongazione, salita 10-90%, assestamento.

    L'errore viene PROIETTATO sulla direzione del comando. E' l'unico modo di
    vedere la sovraelongazione: sulla norma ||e|| non si vede, perche' la norma
    parte da Delta e scende, quindi il suo massimo E' Delta per costruzione e
    'picco/Delta' vale ~0.96 in ogni volo senza dire nulla sul controllore.

    Sulla direzione del comando la variabile di avanzamento

        p(t) = (e(t) . u + Delta) / Delta

    vale 0 all'istante del gradino e 1 sul bersaglio. p > 1 significa che il
    drone ha SUPERATO il bersaglio: quella e' la sovraelongazione vera.

    kind: 'pos' (gradino di posizione) o 'att' (rotazione comandata).
    Ritorna None se in quel canale la fase non ha un gradino.
    """
    ph = sp[j]
    if j == 0 or ph['type'] != 'hold':
        return None
    prev = sp[j - 1]
    i0, i1 = ph['i0'], ph['i1']
    # La finestra parte UN CAMPIONE PRIMA dell'inizio fase, cioe' dall'ultimo
    # campione in cui vale ancora il riferimento vecchio: e' l'istante del
    # gradino. Partendo da i0 il rilevamento della fase puo' cadere uno o due
    # campioni dopo il salto, e su una rotazione rapida (11 deg in 0.14 s, 14
    # campioni) l'avanzamento e' gia' oltre il 10% al primo campione -- da cui
    # un t10 pari a zero, che e' un artefatto della griglia, non una risposta
    # istantanea.
    #
    # E il bersaglio e' il riferimento NUOVO tenuto fisso, non quello campione
    # per campione: dentro una fase di hold sono la stessa cosa, ma sul
    # campione prima del salto solo il primo dei due da' p = 0.
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

    prog = (sgn + amp) / amp                     # 0 all'inizio, 1 sul bersaglio
    ps = _smooth(t, prog, smooth_s)

    ss = t >= (t[-1] - cfg['t_ss'])
    p_fin = float(np.mean(prog[ss])) if ss.any() else float('nan')

    over = float(np.max(ps)) - 1.0                # >0 = ha superato il bersaglio

    # TEMPO DI SALITA sull'escursione EFFETTIVA, non su [0, 1].
    # Il drone non parte dal setpoint precedente: ci sta con un errore a
    # regime, e quell'errore puo' coprire da solo una parte del gradino
    # successivo. In rpy SETPOINT 3 il residuo di 3.3 deg su un comando di
    # 11.2 deg vale il 30%: misurando il 10% su [0, 1] la soglia cadrebbe
    # dentro la condizione iniziale e t10 uscirebbe zero. Normalizzando su
    # [p0, p_fin] il 10-90% torna a misurare la salita, come si fa per una
    # risposta al gradino con condizione iniziale non nulla.
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

    # assestamento nella banda del 5% di Delta attorno al valore finale:
    # sulla variabile di avanzamento la banda e' proprio cfg['settle_band']
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
    """Errore all'istante in cui la rampa di atterraggio si ferma.

    Non a fine log: dopo che il riferimento smette di scendere il drone e' a
    terra e si misura dove lo tiene il suolo. Media sugli ultimi tail_s della
    rampa, perche' un singolo campione e' rumore.
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
    """Accoppiamento assetto -> posizione in una fase di SOLA rotazione.

    Il riferimento di posizione non si muove, quindi tutto l'errore di
    posizione che compare li' e' indotto dalla rotazione comandata. E' l'unica
    metrica del transitorio immune al confondente dell'ampiezza: il gradino di
    posizione e' zero per costruzione in entrambi i voli.
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
        # minimal: niente cornice sui quattro lati, restano gli assi che
        # portano i numeri
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "savefig.bbox": "tight",
        # cmr10 non ha il peso bold: con fontweight='bold' matplotlib stampa
        # "findfont: Failed to find font weight bold" a ogni titolo
        "axes.titleweight": "normal",
    })


def save(fig, path, pad_inches=None):
    kw = {} if pad_inches is None else {'pad_inches': pad_inches}
    fig.savefig(path, format='pdf', transparent=True, bbox_inches='tight', **kw)
    plt.close(fig)


def tight_x(ax, lo, hi):
    """Niente margine orizzontale: la traccia attacca all'asse.

    I limiti si passano espliciti, non si deducono dalle linee gia' disegnate:
    axhline e axvline hanno xdata in coordinate d'assi e falserebbero il conto.
    """
    if hi > lo:
        ax.set_xlim(lo, hi)
    ax.margins(x=0)


def mark_phases(ax, ref, lw=0.9):
    """Linee verticali ai confini di fase. Nessuna ombreggiatura: si guarda
    il volo intero, decollo e atterraggio inclusi."""
    for p in analysed_phases(ref['phases'])[1:]:
        ax.axvline(p['t0'], color='#999', lw=lw, ls=':')


# ===========================================================================
# CATEGORIE DI COMPITO DI CONTROLLO
# ===========================================================================
# Sezione ex analyze_categories.py, accorpata qui. Invece di dividere il volo
# nelle fasi della missione lo divide in categorie che dicono che
# problema di controllo il drone sta risolvendo in quell'istante:
#
#    Takeoff    rampa verticale in decollo
#    Step       transitori di risposta al gradino
#    Hover      mantenimento a punto fisso (anche con assetto non nullo)
#    Land       rampa verticale in atterraggio
#
# Le righe sono le stesse in tutti gli scenari, quindi si confrontano fra loro.
#
# COME NASCONO I BLOCCHI
#   Ogni fase rilevata sopra viene classificata dal RIFERIMENTO:
#     - riferimento in movimento (decollo, atterraggio) -> rampa, presa
#       dall'inizio fino a quando il riferimento si ferma; la coda in cui il
#       drone e' fermo in cima alla rampa o gia' posato resta FUORI;
#     - riferimento a gradino -> i primi DELTA secondi sono transitorio, il
#       resto e' mantenimento a punto fisso (Hover).
#   Il taglio e' ancorato al RIFERIMENTO, non all'arrivo del drone.
#
# ACCORPAMENTO
#   Le metriche RMSE accorpano i CAMPIONI delle finestre della stessa
#   categoria, non le RMSE. Il bias vettoriale no: la media dell'errore di due
#   hover in punti diversi punta in direzioni diverse e si cancellerebbe.
#
# GRADINI NON CONFRONTABILI
#   Nella categoria 'Step' un sotto-blocco entra nelle metriche ASSOLUTE
#   (rmse_pos, rmse_att, err_xy, err_z, tv) solo se la fase da cui viene e'
#   marcata confrontabile da flag_comparability, cioe' se l'ampiezza comandata
#   coincide fra i voli entro tolleranza. Un gradino piu' lungo produce un
#   errore piu' grande per costruzione, e accorparlo agli altri sposta la media
#   di una quantita' che non dipende dal controllore. Le colonne NORMALIZZATE
#   (peak_n, rmse_n, att_peak_n) restano calcolate su TUTTI i sotto-blocchi:
#   sono valide proprio perche' dividono per l'ampiezza. Stessa regola di
#   phase_metrics, dove i campi in TRANSIENT_FIELDS cadono a nan.

REF_MOVE_TOL = 1e-3         # [m/campione] sotto cui il riferimento e' fermo
C_INK, C_MUTED = "#222222", "#9a9a9a"

CATEGORIES = ["Takeoff", "Step", "Hover", "Land"]
KIND = {"Takeoff": "rampa", "Land": "rampa", "Step": "gradino", "Hover": "hold"}

# Nomi di fase specifici per missione, Tabelle 5.5/5.6 della tesi: il bucket
# 'Hover' generico viene rietichettato con il nome della fase (SETPOINT k) a
# cui corrisponde, per missione. 'Takeoff'/'Step'/'Land' restano generici.
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
    """Cella di CSV o valore gia' numerico -> float; non numerico -> nan."""
    try:
        return float(s)
    except (TypeError, ValueError):
        return float('nan')


def cat_view(f):
    """Adatta il dizionario di volo di process_log a quello che si aspetta la
    sezione categorie. NON ricalcola nessun errore: e_pos, e_att e le
    componenti sono quelle gia' definite sopra, cosi' resta una sola
    definizione di errore in tutto il file."""
    e_w = f['pos'] - f['pos_ref']
    dref = np.linalg.norm(np.diff(f['pos_ref'], axis=0), axis=1)
    return {
        't': f['t'], 'dt': float(np.median(np.diff(f['t']))),
        'run': f['run'], 'group': f['group'],
        'pos': f['pos'], 'pos_ref': f['pos_ref'], 'e_w': e_w,
        'e_pos': f['e_pos_norm'], 'e_att': f['e_att_norm'],
        'e_yaw': np.abs(f['e_yaw']),
        'e_rp': np.hypot(f['e_roll'], f['e_pitch']),
        # componenti singole, le stesse delle cartelle RMSE/ e Rollouts/:
        # posizione nel frame del RIFERIMENTO, assetto come rotation vector
        'e_b': np.column_stack([f['e_bx'], f['e_by'], f['e_bz']]),
        'rv': np.column_stack([f['e_roll'], f['e_pitch'], f['e_yaw']]),
        'avel': f.get('w_norm'), 'v': f.get('v_norm'),
        'forces': f.get('forces'),
        'ref_moving': np.concatenate([[False], dref > REF_MOVE_TOL]),
    }


def tv_rotor(t_blk, forces):
    """Total Variation della velocita' di rotore su una finestra contigua,
    sommata sui sei rotori.

    La durata e' il TEMPO TRASCORSO nella finestra, t[-1] - t[0], non
    (n-1)*dt_medio: il periodo del loop non e' costante (9.998-10.032 ms), e
    le due formule davano numeri diversi allo 0.5% sulla stessa finestra --
    abbastanza da spostare un confronto fra due policy che ne distano meno.
    Stessa convenzione di compute_tv(), che e' quella che finisce nelle
    tabelle."""
    if forces is None or forces.shape[0] < 2:
        return float("nan")
    w = np.sqrt(np.clip(forces, F_MIN, F_MAX) / CF)
    dur = float(t_blk[-1] - t_blk[0])
    return float(np.sum(np.abs(np.diff(w, axis=0))) / dur) if dur > 0 else float("nan")


def build_blocks(f, phase_rows, delta, mission=None):
    """[(categoria, etichetta, t0, t1, info)] per un volo.

    'mission' e' 'p2p' o 'rpy': seleziona il rietichettamento di HOLD_PHASES
    per il bucket di mantenimento. None (o missione non riconosciuta) lascia
    il nome generico 'Hover'."""
    t = f["t"]
    out = []
    for r in phase_rows:
        lab = r["phase"]
        t0, t1 = cellf(r["t0"]), cellf(r["t1"])
        inside = (t >= t0) & (t <= t1)
        moving = np.flatnonzero(inside & f["ref_moving"])
        ramp_frac = float((inside & f["ref_moving"]).sum() / max(inside.sum(), 1))

        if ramp_frac > 0.5:                       # fase di rampa
            t_stop = t[moving[-1]] if moving.size else t1
            cat = "Takeoff" if cellf(r["ref_z"]) > f["pos_ref"][inside][0][2] else "Land"
            # piu' robusto: il segno dello spostamento verticale comandato
            dz = f["pos_ref"][min(moving[-1], len(t) - 1)][2] - f["pos_ref"][moving[0]][2] \
                if moving.size else 0.0
            cat = "Takeoff" if dz > 0 else "Land"
            out.append((cat, lab, t0, t_stop, {"ramp_dz": dz}))
            continue

        # fase di gradino: transitorio + mantenimento
        step_pos = cellf(r.get("step_pos_m", "nan"))
        step_ang = cellf(r.get("step_ang_deg", "nan"))
        # il transitorio e' confrontabile solo se lo e' la fase da cui viene:
        # il flag lo decide flag_comparability sui dati, non il codice
        cmp_flag = cellf(r.get("comparable_transient", "1"))
        cmp_flag = True if cmp_flag != cmp_flag else bool(cmp_flag)
        # l'errore a regime della fase serve al tempo di salita: la finestra
        # 'Step' e' il solo transitorio e non contiene il regime, che finisce
        # in 'Hover', quindi il valore finale va passato da qui
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
    m = (f["t"] >= t0) & (f["t"] <= t1)
    return m


# soglia minima di distanza sotto cui non si normalizza (m). Stesso valore di
# --min-step-norm: metrics_for_category non riceve cfg, quindi vive qui.
cfg_min_step = MIN_STEP_NORM
# analogo angolare: sotto ~2 deg di angolo da coprire il rapporto e' rumore.
cfg_min_step_ang = float(np.deg2rad(2.0))   # [rad]


def metrics_for_category(f, blocks, cat):
    """dict metrica -> valore, per una categoria di un volo."""
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

    # Maschera per le metriche ASSOLUTE: esclude i sotto-blocchi il cui
    # gradino non e' confrontabile fra i voli (vedi GRADINI NON CONFRONTABILI
    # nella docstring di sezione). Per le categorie che non sono gradini
    # coincide con allm, quindi nulla cambia altrove.
    sel_cmp = [(b, m) for b, m in zip(sel, masks)
               if b[4].get("comparable", True)]
    absm = np.zeros_like(allm)
    for _b, m in sel_cmp:
        absm |= m
    n_abs = int(absm.sum())
    if n_abs < 10:                       # niente gradino confrontabile
        absm = np.zeros_like(allm)
        n_abs = 0

    M = {"n_samples": n, "n_subblocks": len(sel),
         "n_samples_abs": n_abs, "n_subblocks_abs": len(sel_cmp),
         "dur_s": round(float(sum(b[3] - b[2] for b in sel)), 2)}

    # --- intestazione comune: le due grandezze che finiscono in figura
    M["rmse_pos"] = rms_t(f["t"], f["e_pos"], absm) if n_abs else float("nan")
    M["rmse_att"] = rms_t(f["t"], f["e_att"], absm) if n_abs else float("nan")
    # decomposizione dell'errore di posizione: vale sempre
    #     rmse_pos^2 = err_xy^2 + err_z^2
    # L'orizzontale e la verticale hanno cause diverse -- deriva contro
    # ritardo di quota / errore di spinta -- e sommarle nella norma le nasconde.
    M["err_xy"] = (rms_t(f["t"], np.linalg.norm(f["e_w"][:, :2], axis=1), absm)
                   if n_abs else float("nan"))
    M["err_z"] = rms_t(f["t"], f["e_w"][:, 2], absm) if n_abs else float("nan")

    kind = KIND[cat]

    if kind == "rampa":
        # il riferimento si muove in verticale: separo ritardo e deriva
        # solo per le rampe: il ritardo di quota CON SEGNO, che dice se il
        # drone e' rimasto indietro o e' andato avanti rispetto alla rampa
        M["lag_z"] = float(np.mean(f["e_w"][allm][:, 2]))

        # NORMALIZZAZIONE SULLA DISTANZA COMANDATA DALLA RAMPA.
        #
        # Il drone non viene riposato a terra alla stessa quota a ogni volo,
        # quindi la rampa di atterraggio non e' lunga uguale: 0.83 m in una
        # cartella e 0.78 m in un'altra, cioe' una finestra di 4.71 s contro
        # una di 4.44 s. Confrontare i due errori assoluti mette a confronto
        # due manovre diverse, ed e' lo stesso confondente che per i gradini
        # si risolve dividendo per l'ampiezza: qui il denominatore e' il
        # dislivello che la rampa comanda.
        #
        # Le colonne hanno gli stessi nomi di quelle dei gradini
        # (*_pooled_n_*, *_pooled_den) perche' hanno lo stesso significato --
        # errore normalizzato campione per campione, poi un solo integrale --
        # e cosi' chi legge, qui o in compare_trajectory, non deve sapere se
        # la categoria era una rampa o un gradino per trattarla nel modo
        # giusto. L'assetto non ha un comando su una rampa verticale: resta
        # nan, e la figura d'assetto di Takeoff/Land resta in valore assoluto.
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

    elif kind == "gradino":
        # Anche le colonne normalizzate girano sui soli sotto-blocchi
        # confrontabili. Non e' ridondante con il filtro sugli assoluti: la
        # guardia di ampiezza (sp > 0.02, sa > 1.0) e' per volo, quindi su un
        # gradino di ampiezza diversa fra i voli puo' passare in un gruppo e
        # non nell'altro, e le due medie finirebbero calcolate su insiemi di
        # manovre diversi. 'comparable' invece e' deciso confrontando i voli,
        # quindi rende gli insiemi simmetrici per costruzione. Resta il caso
        # in cui normalizzare serve davvero: piu' gradini confrontabili di
        # ampiezza diversa fra loro, che accorpati in valore assoluto
        # darebbero la media di manovre incommensurabili.
        # NORMALIZZAZIONE SULLA DISTANZA VERA, su TUTTI i sotto-blocchi.
        # E' l'unica sotto cui anche i gradini di ampiezza comandata diversa
        # sono confrontabili: divide per la distanza che il drone aveva
        # davvero davanti all'istante del gradino, quindi assorbe sia
        # l'ampiezza comandata sia l'errore a regime che si porta dentro dal
        # setpoint precedente. Gira su 'sel', non su 'sel_cmp': e' il motivo
        # per cui questa colonna esiste.
        rm_r, iae_r, dr_l = [], [], []
        rm_ra, iae_ra, dr_la = [], [], []
        for b, m in zip(sel, masks):
            if m.sum() < 5:
                continue
            tt = f["t"][m]
            info = b[4]
            # il gate e' sul gradino COMANDATO, non sulla distanza vera: se il
            # riferimento non fa un gradino su quella grandezza (in rpy la
            # posizione resta ferma), la "distanza da coprire" e' solo errore
            # residuo, e dividere un errore per un altro errore premia chi
            # inseguiva peggio. E' l'artefatto che questa colonna doveva
            # togliere, non reintrodurre.
            # e non basta che il gradino comandato esista: il denominatore deve
            # essere DOMINATO dal comando, non dall'errore residuo. Se la
            # distanza vera e' meno di meta' dell'ampiezza comandata, quel
            # rapporto sta misurando l'errore precedente, non la manovra.
            sp = info.get("step_pos_m", float("nan"))
            sa = info.get("step_ang_deg", float("nan"))
            sa_r = float(np.deg2rad(sa)) if sa == sa else float("nan")
            ep = f["e_pos"][m]
            d_real = float(ep[0])          # distanza da coprire all'istante
            if (sp == sp and sp > cfg_min_step
                    and d_real > max(cfg_min_step, 0.5 * sp)):
                rm_r.append(rms_t(tt, ep) / d_real)
                iae_r.append(mae_t(tt, ep) / d_real)
                dr_l.append(d_real)
            ea = f["e_att"][m]
            a_real = float(ea[0])          # angolo da coprire all'istante
            if (sa_r == sa_r and sa > 5.0
                    and a_real > max(cfg_min_step_ang, 0.5 * sa_r)):
                rm_ra.append(rms_t(tt, ea) / a_real)
                iae_ra.append(mae_t(tt, ea) / a_real)
                dr_la.append(a_real)
        # --- la stessa normalizzazione, ma AGGREGATA COME L'ASSOLUTO.
        # Le colonne *_n_real sopra sono medie di rapporti, un sotto-blocco
        # conta uno; la barra assoluta invece mette in un unico RMS tutti i
        # campioni. Per sostituire la barra assoluta serve la forma pooled:
        # si normalizza l'errore CAMPIONE PER CAMPIONE sulla distanza che quel
        # gradino aveva da coprire, poi si fa un solo RMS (o MAE) su tutti i
        # campioni di tutti i gradini -- SETPOINT 1 compreso, che e' il motivo
        # per cui questa colonna esiste: normalizzato, torna confrontabile.
        # Resta adimensionale; la riscalatura a una distanza comune, che serve
        # solo per disegnarla su un asse in metri, si fa in export_categories
        # dove si vedono entrambi i voli.
        for fld, ser, den, gate, dmin in (
                ("pos", "e_pos", "step_pos_m", cfg_min_step, cfg_min_step),
                ("att", "e_att", "step_ang_deg", 5.0, cfg_min_step_ang)):
            # l'aggregazione e' sugli INTEGRALI, non sui campioni concatenati:
            # concatenare e integrare di seguito aggiungerebbe un trapezio
            # spurio nel salto fra la fine di un gradino e l'inizio del
            # successivo, pesato dal buco temporale fra i due.
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
                # TEMPO DI SALITA, sullo scarto dal VALORE FINALE.
                # Sull'errore assoluto questo numero non misura la velocita':
                # una policy feed-forward si assesta su un errore a regime non
                # nullo, e quando quell'errore supera la soglia (il 10% del
                # gradino) la condizione non si verifica mai e il tempo esce
                # pari alla finestra intera -- cioe' il volo con piu' errore a
                # regime risulta piu' lento anche se e' arrivato prima.
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
        # e_std resta una dispersione sui campioni: non e' una media nel
        # tempo, e pesarla darebbe una grandezza senza un nome riconoscibile
        M["e_std"] = float(np.std(f["e_pos"][allm]))
        bias = [float(np.linalg.norm(f["e_w"][m].mean(axis=0)))
                for m in masks if m.sum() > 5]
        M["e_bias"] = rms(bias) if bias else float("nan")
        M["att_mean"] = seg_mean_t(f["t"], f["e_att"], allm)
        M["att_yaw"] = rms_t(f["t"], f["e_yaw"], allm)
        M["att_rp"] = rms_t(f["t"], f["e_rp"], allm)
        if f["avel"] is not None:
            M["w_norm"] = rms_t(f["t"], f["avel"], allm)

    # --- sforzo di controllo, per ogni categoria: somma pesata sui sotto-blocchi
    if f["forces"] is not None:
        # la TV e' uno sforzo assoluto: un gradino piu' lungo ne richiede di
        # piu', quindi segue la stessa maschera delle altre metriche assolute
        tv_masks = [m for _b, m in sel_cmp] if kind == "gradino" else masks
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
    """Limite y quando UNA CATEGORIA domina le altre.

    Il confronto va fatto fra i massimi PER CATEGORIA, non fra i singoli
    valori: guardando i valori sciolti, una distribuzione continua senza veri
    outlier finisce per far tagliare mezzo grafico. Si taglia solo se la
    categoria piu' alta supera di `factor` la seconda."""
    v = sorted((x for x in cat_max if x == x), reverse=True)
    if len(v) < 2 or v[1] <= 0:
        return None
    return v[1] * 1.35 if v[0] > factor * v[1] else None


# Una figura per grandezza e per statistica d'errore. Le componenti
# orizzontale e verticale (err_xy / err_z) restano nel CSV ma non si disegnano
# piu': al loro posto ci sono le sei componenti singole, definite come nelle
# cartelle RMSE/ e Rollouts/, piu' le due norme.
#
# DUE STATISTICHE, DUE DOMANDE DIVERSE
#   RMSE  quanto e' grande l'errore, pesando i picchi
#   MAE   quanto e' grande l'errore tipico, senza pesare i picchi

CAT_SERIES = [
    ("e_x",     lambda f: f["e_b"][:, 0], r"$e_x^{\,r}$",    "m",   "1_err_bx",    "{:.3f}"),
    ("e_y",     lambda f: f["e_b"][:, 1], r"$e_y^{\,r}$",    "m",   "2_err_by",    "{:.3f}"),
    ("e_z",     lambda f: f["e_b"][:, 2], r"$e_z^{\,r}$",    "m",   "3_err_bz",    "{:.3f}"),
    ("e_roll",  lambda f: f["rv"][:, 0],  r"$e_{roll}$",     "rad", "4_err_roll",  "{:.4f}"),
    ("e_pitch", lambda f: f["rv"][:, 1],  r"$e_{pitch}$",    "rad", "5_err_pitch", "{:.4f}"),
    ("e_yaw",   lambda f: f["rv"][:, 2],  r"$e_{yaw}$",      "rad", "6_err_yaw",   "{:.4f}"),
    ("e_p",     lambda f: f["e_pos"],     r"$\|e_p\|$",      "m",   "7_err_pos",   "{:.3f}"),
    ("e_R",     lambda f: f["e_att"],     r"$\|e_R\|$",      "rad", "8_err_att",   "{:.4f}"),
    # moto: NON sono errori. In Takeoff, Step e Landing il riferimento si
    # muove per costruzione, quindi un valore alto e' atteso, non un difetto.
    ("v_norm",  lambda f: f["v"],         r"\|v\|",          "m/s", "9_vel",       "{:.3f}"),
    ("w_norm",  lambda f: f["avel"],      r"\|\omega\|",     "rad/s", "10_avel",   "{:.3f}"),
]
# Testo delle FIGURE in inglese: sono quelle che finiscono in tesi. L'output
# a console resta in italiano, e' lo strumento di lavoro.
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

# Le due statistiche d'errore. Firma (t, v, mask): sono medie TEMPORALI, non
# sui campioni -- vedi seg_mean_t per il perche'.
STATS = (("RMSE", rms_t),
         ("MAE",  mae_t))


def category_series_stats(f, blocks, cats):
    """{statistica: {categoria: {grandezza: valore}}} per un volo.

    Come metrics_for_category: i sotto-blocchi con gradino non confrontabile
    restano fuori, perche' RMSE e MAE di queste serie sono valori assoluti."""
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
            if ser is None:                       # 'vel'/'avel' assenti dal log
                continue
            v = np.asarray(ser, float)
            for st, agg in STATS:
                out[st].setdefault(cat, {})[key] = agg(f["t"], v, allm)
    return out


def cat_bar_figure(cats, stats, groups, key, stat, out_path, clip=False):
    """Una grandezza e una statistica: le manovre sull'asse x, una barra per
    gruppo.

    Nessuna manovra viene esclusa. Dove una domina le altre -- tipicamente
    Step, con errori 5 volte piu' grandi -- con --cat-clip l'asse resta tarato
    sulle barre piccole e quella fuori scala viene TAGLIATA, con il suo valore
    scritto sopra. Di default invece le proporzioni restano vere."""
    spec = next(x for x in CAT_SERIES if x[0] == key)
    _k, _get, sym, unit, _fname, fmt = spec
    vals = [stats[g][stat].get(c, {}).get(key, float("nan"))
            for g in groups for c in cats]
    finite = [v for v in vals if v == v]
    if not finite:
        return False
    signed = min(finite) < 0                      # solo il ME puo' esserlo

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
    """Griglia 1x2: Posizione (e_p) e Assetto (e_R) affiancate."""
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
    """Scrive tables/category_metrics.csv e categories/paired_categories.pdf."""
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
    print('CATEGORIE DI COMPITO DI CONTROLLO'
          f"   (Delta = {cfg['cat_delta']} s)")
    print('=' * 94)
    print(f"  {'categoria':16s}{'fase':12s}"
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
        print("  nessuna categoria in comune fra i voli: sezione saltata.\n")
        return

    a_, b_ = groups[0], groups[1]
    print(f"\n{'-'*94}\nMETRICHE  (delta = {b_} rispetto a {a_}; "
          f"negativo = {b_} meglio)\n{'-'*94}")
    rows_csv = []
    for cat in cats:
        nb = data[a_][cat]['n_subblocks']
        nba = data[a_][cat].get('n_subblocks_abs', nb)
        print(f"\n  [{cat}]   {data[a_][cat]['dur_s']:.1f} s, "
              f"{nb} sotto-blocco/hi, "
              f"{data[a_][cat]['n_samples']} campioni"
              + ("" if nba == nb else
                 f"  --  {nba}/{nb} confrontabile/i: le metriche assolute "
                 f"usano solo quello/i, {data[a_][cat].get('n_samples_abs', 0)} campioni"))
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
            print(f"\n  GRADINI  {nb - nba} sotto-blocco/hi di 'Step' ha ampiezza "
                  f"diversa fra i voli ed e' escluso dalle metriche assolute.")
            if nba == 0:
                print("  Nessun gradino confrontabile: rmse_pos, rmse_att, err_xy, "
                      "err_z e tv di 'Step' escono vuote.")
                print("  Restano peak_n, rmse_n e att_peak_n, normalizzate "
                      "sull'ampiezza e calcolate su tutti i sotto-blocchi.")
        if aa == aa and bb == bb and aa and abs(bb - aa) / aa > 1e-3:
            print(f"\n  NOTA  ampiezza media dei gradini (tutti i sotto-blocchi): "
                  f"{a_} {aa:.3f} m, {b_} {bb:.3f} m  ({(bb-aa)/aa*100:+.1f}%).")
            print("  Vale per le colonne normalizzate; gli assoluti sono gia' filtrati.")

    # Le statistiche d'errore si calcolano dai campioni, non dal CSV: il MAE
    # non e' ricavabile dall'RMSE. Vanno anche nel CSV, marcate nella colonna
    # 'stat', perche' compare_trajectory legge da li' e altrimenti potrebbe
    # produrre solo le figure RMSE.
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
    print(f"\n  scritto {csv_path}")

    shown = [c for c in cats if c not in cfg['cat_exclude']]
    if not shown:
        print("  --cat-exclude ha escluso tutte le categorie: figure saltate.")
        return
    if len(shown) < len(cats):
        print(f"  figura senza: {', '.join(c for c in cats if c not in shown)}"
              f"  (restano nel CSV)")
    # maneuvers/ e' la tassonomia della MISSIONE: decollo, mantenimento,
    # atterraggio. Il gradino non e' una manovra della missione ma un
    # esperimento sul controllore: ha una sua unita' di lettura -- l'errore
    # normalizzato sulla distanza da percorrere -- e i suoi tre numeri,
    # salita, sovraelongazione, assestamento. Sta in maneuvers/step/, da solo.
    shown_man = [c for c in shown if KIND.get(c) != 'gradino']
    shown_step = [c for c in shown if KIND.get(c) == 'gradino']

    # ------------------------------------- maneuvers/: una grandezza, una figura
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

        print(f"  {n_fig} figure scritte in {d}")

    # --- maneuvers/propellers: lo sforzo di controllo PER FASE.
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
        print(f"  {n_fig} figure scritte in {dirs['man_tv']}")

    with open(dirs['tables'] / 'tv_maneuvers.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['category', 'group', 'channel', 'tv'])
        for (g, cat), tv in sorted(tvcat.items()):
            for i in range(N_ROTORS):
                w.writerow([cat, g, f'M{i+1}', f"{tv[i]:.4f}"])
            w.writerow([cat, g, 'TOTAL', f"{float(np.sum(tv)):.4f}"])


def _bars_on_ax(ax, labels, vals, title, ylabel, hatch_flags=None,
                fmt='{:.3f}', zero_line=False, legend=True):
    """Barre PPO/SPPO affiancate su un asse dato, una coppia per etichetta."""
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
    """Una figura, piu' pannelli affiancati sulle stesse etichette."""
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
    """'SETPOINT 2' -> 'SETPOINT_2'. Nome file, non etichetta."""
    return "_".join(str(label).split())


def _step_profile_ax(ax, per, label, finals=True, bands=True, title=True):
    """Il profilo di avanzamento di UNA fase, PPO contro SPPO."""
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
    """Tabella dei tre numeri della risposta al gradino, PPO contro SPPO."""
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
        if i == 0:                                                           # intestazione: filetti
            c.visible_edges = 'TB'
            c.set_facecolor('#f2f2f2')
            if j in (1, 2):                                                  # PPO / SPPO nel loro colore
                c.get_text().set_color(COL[GROUPS[j - 1]])
        else:
            c.visible_edges = 'B' if i == len(body) else ''
        if j == 0:
            c.get_text().set_ha('left')
            c.PAD = 0.03

    save(fig, out_path)
    return 1


def export_assets(flights, dirs, cfg, phase_rows=()):
    """Metriche specifiche della manovra. Per fase, PPO contro SPPO."""
    rows = []
    n_fig = 0

    # ---------------------------------------------- risposta al gradino
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

    # ------------------------- transitorio normalizzato sulla distanza vera
    for kind, pre, base, ttl in (('pos', 5, 'norm_real_m', 'position'),
                                 ('att', 6, 'norm_real_deg', 'attitude')):
        keys = (('rmse_pos_transient_real', 'iae_pos_real') if kind == 'pos'
                else ('rmse_att_transient_real', 'iae_att_real'))
        per = {}
        for r in phase_rows:
            if not r['phase'].startswith('SETPOINT'):
                continue
            if r.get(base) != r.get(base):          # nan: niente gradino qui
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

    # ------------------------------ sforzo di controllo DENTRO la fase
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

    # ---------------------------------------------- atterraggio
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

    # ---------------------------------------------- accoppiamento
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
    print(f"\n  {n_fig} figure scritte in maneuvers/step")
    return rows


# ===========================================================================
def export_analysis(input_dir: Path, cfg):
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

    print(f"Scansione di data.npz in: {input_dir} ...")
    npz_files = sorted(input_dir.rglob("data.npz"))
    print(f"  {len(npz_files)} file trovati.\n")

    # ----------------------------------------------------- lettura + manifest
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
        print("Nessun volo leggibile."); return
    flights.sort(key=lambda f: (f['group'] != 'PPO', f['run']))

    for f in flights:
        print(f"  {f['group']:<5} {f['run']:<34} {f['duration']:6.2f} s, "
              f"{f['n_samples']} campioni ({f['n_samples']/f['duration']:.1f} Hz), "
              f"{len(setpoint_phases(f['phases']))} setpoint di missione")

    ref = flights[0]

    # ------------------------------------------------------- fasi rilevate
    print(f"\n{'='*78}\nFASI RILEVATE DAL RIFERIMENTO\n{'='*78}")
    for f in flights:
        print(f"\n  --- {f['group']}  {f['run']} ---")
        print(f"  {'fase':<14}{'t0':>7}{'t1':>7}{'dur':>7}   riferimento")
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
            print(f"      setpoint di missione: {', '.join(f'{d:.2f}' for d in durs)} s"
                  f"   ->  wait_s ~ {np.median(durs) - 0.7:.1f} s "
                  f"(tolti 0.7 s di staging)")

    same_ref = True
    if len(flights) >= 2:
        print(f"\n{'='*78}\nCONTROLLO: STESSO RIFERIMENTO PER I DUE VOLI?\n{'='*78}")
        fa, fb = flights[0], flights[1]
        pa, pb = setpoint_phases(fa['phases']), setpoint_phases(fb['phases'])
        if len(pa) != len(pb):
            same_ref = False
            print(f"  !! numero di setpoint diverso: {len(pa)} vs {len(pb)}.")
            print("     Il confronto fase per fase NON e' valido.")
        else:
            print(f"  {'setpoint':<12}{'scarto pos':>12}{'scarto rpy':>12}"
                  f"{'sfasamento':>12}")
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
                print("  -> i setpoint di missione coincidono: confronto a parita'")
                print("     di riferimento.")
            else:
                same_ref = False
                print("  -> !! i setpoint NON coincidono: il confronto non e' a parita'")
                print("        di riferimento.")
            if worst_t > 0.5:
                print("  -> !! fasi sfasate di oltre 0.5 s: confronta per fase, non a")
                print("        tempo assoluto.")

        ma = [p for p in fa['phases'] if p['label'] in ('TAKEOFF', 'LANDING')]
        mb = [p for p in fb['phases'] if p['label'] in ('TAKEOFF', 'LANDING')]
        if ma and len(ma) == len(mb):
            print(f"\n  {'manovra':<12}{fa['group']:>26}{fb['group']:>26}")
            for x, y in zip(ma, mb):
                print(f"  {x['label']:<12}"
                      f"   dpos=[{x['d_pos'][0]:+.2f},{x['d_pos'][1]:+.2f},"
                      f"{x['d_pos'][2]:+.2f}] m"
                      f"   dpos=[{y['d_pos'][0]:+.2f},{y['d_pos'][1]:+.2f},"
                      f"{y['d_pos'][2]:+.2f}] m")
            dmax = max(float(np.max(np.abs(x['d_pos'] - y['d_pos'])))
                       for x, y in zip(ma, mb))
            if dmax < 0.02:
                print(f"  -> stessa manovra comandata (scarto max {dmax*1000:.1f} mm):")
                print("     decollo e atterraggio sono confrontabili.")
            else:
                print(f"  !! spostamenti comandati diversi (scarto max {dmax*1000:.0f} mm):")
                print("     il confronto su decollo e atterraggio va preso con cautela.")

    bad, steps_per_flight = flag_comparability(flights, cfg)
    print(f"\n{'='*78}\nCONFRONTABILITA' DEI TRANSITORI\n{'='*78}")
    print(f"  {'fase':<14}" + "".join(f"{f['group']:>12}" for f in flights)
          + f"{'scarto':>10}   esito")
    for lab in [p['label'] for p in analysed_phases(ref['phases'])]:
        vals = [d.get(lab, (float('nan'),))[0] for d in steps_per_flight]
        if not all(v == v for v in vals):
            continue
        spread = max(vals) - min(vals)
        verdict = ("SOLO REGIME" if lab in bad else "ok")
        print(f"  {lab:<14}" + "".join(f"{v:>10.3f} m" for v in vals)
              + f"{spread*1000:>7.0f} mm   {verdict}")
    if bad:
        print("\n  Le fasi marcate SOLO REGIME hanno un gradino di ampiezza diversa")
        print("  fra i due voli: il drone non puo' essere riposizionato a terra nello")
        print("  stesso identico punto. Su un gradino piu' lungo il picco di errore e'")
        print("  piu' grande PER COSTRUZIONE, quindi transitorio, picco, assestamento,")
        print("  RMSE di fase e TV di fase non sono confrontabili e vengono lasciati")
        print("  vuoti. Resta l'errore A REGIME, che dipende dal setpoint finale --")
        print("  identico -- e non dal cammino percorso per arrivarci. La stessa")
        print("  finestra e' esclusa anche dalle metriche globali sull'intero volo.")
    else:
        print("\n  Tutte le fasi hanno lo stesso comando in ingresso nei due voli:")
        print("  transitori confrontabili, nessuna esclusione.")

    print(f"\n  Analisi sull'intero volo: decollo, setpoint e atterraggio sono")
    print(f"  tutti pilotati dalla policy, quindi sono tutte prestazioni sue.")
    print(f"  Le fasi vengono valutate una per una: un RMSE unico mediarebbe")
    print(f"  inseguimento di rampa, risposta al gradino e reiezione a punto")
    print(f"  fisso, con pesi dati dalle durate scelte in fase di missione.")

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
        print(f"  !! {len(warned)} voli con avvisi (vedi tables/manifest.csv)")

    # ====================================================== RMSE + Rollouts
    per_flight_rows = []
    t_lo = min(float(f['t'][0]) for f in flights)
    t_hi = max(float(f['t'][-1]) for f in flights)

    flight_rows = []

    for key, sym, unit, title, fname in METRICS:
        if not any(key in f for f in flights):
            print(f"  [skip] metrica '{key}': assente in tutti i voli.")
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
        """Sei figure per motore piu' il riepilogo, su una finestra."""
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
        print(f"\n  {n_tv} figure di TV scritte in flight/propellers")

        for f in flights[:2]:
            for i in range(N_ROTORS):
                tv_rows.append({'window': 'comparable_flight',
                                'group': f['group'], 'channel': f'M{i+1}',
                                'tv': f"{f['tv_per_motor'][i]:.4f}"})
            tv_rows.append({'window': 'comparable_flight', 'group': f['group'],
                            'channel': 'TOTAL',
                            'tv': f"{float(np.sum(f['tv_per_motor'])):.4f}"})

        print(f"  TV (somma {N_ROTORS} propeller)"
              + (", transitori non confrontabili esclusi:" if bad else ":"))
        for f in flights[:2]:
            print(f"    {f['group']:<5} {float(np.sum(f['tv_per_motor'])):8.2f}")

    # ============================================ metriche per fase
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
        print(f"METRICHE PER FASE   (regime = ultimi {cfg['t_ss']:.1f} s; "
              f"tolleranze {cfg['tol_pos']*100:.0f} cm / {cfg['tol_att_deg']:.0f} deg)")
        print("  '--' = non confrontabile (gradino di ampiezza diversa fra i voli)")
        print('=' * 94)
        print(f"  {'fase':<12}{'grp':<6}{'RMSE pos':>10}{'regime':>9}{'picco':>8}"
              f"{'set.[s]':>9}{'RMSE att':>10}{'regime':>9}{'picco':>8}{'set.[s]':>9}")
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
        print("TRANSITORIO NORMALIZZATO SULL'AMPIEZZA DEL GRADINO")
        print("  Un gradino piu' lungo produce un errore piu' grande per")
        print("  costruzione. Dividendo per l'ampiezza il transitorio torna")
        print("  confrontabile: in regime lineare la risposta al gradino scala")
        print("  con l'ampiezza, quindi e(t)/Delta non dipende da Delta.")
        print("  Vale finche' gli attuatori non saturano: assunzione DA")
        print("  DICHIARARE in tesi. Affianca l'errore a regime, non lo sostituisce.")
        print('=' * 94)
        print(f"  {'fase':<12}{'grp':<6}{'gradino':>10}{'picco/D':>10}"
              f"{'RMSEtr/D':>10}{'IAE/D':>10}{'piccoAtt/D':>12}")
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
    print("MANOVRE  -  risposta al gradino, atterraggio, accoppiamento"
          "   (atterraggio e accoppiamento: solo tables/assets.csv)")
    print('=' * 94)
    asset_rows = export_assets(flights, dirs, cfg, rows)
    if asset_rows:
        print(f"  {'sezione':14s}{'fase':13s}{'metrica':20s}"
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
        fh.write(f"% {input_dir.name}: un volo per algoritmo, stesso riferimento.\n")
        fh.write(f"% Regime = ultimi {cfg['t_ss']:.1f} s di ciascuna fase.\n")
        fh.write("% Valori singoli, non stime con incertezza: n = 1 per condizione.\n")
        if bad:
            fh.write("% '--': transitorio non confrontabile (ampiezza del gradino\n")
            fh.write("%       diversa fra i due voli). Fasi: "
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
            fh.write(f"% {input_dir.name}: transitorio normalizzato "
                     "sull'ampiezza del gradino.\n")
            fh.write("% e(t)/Delta e' invariante rispetto all'ampiezza del "
                     "gradino nel regime\n")
            fh.write("% lineare: assunzione da dichiarare nel testo. "
                     "Solo fasi raggiunte da un gradino\n")
            fh.write("% (decollo e atterraggio sono rampe ed escono).\n")
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
        print(f"{'='*78}\n{b} RISPETTO A {a}\n{'='*78}")
        print("  errore a regime, per fase:")
        for ph in phases_l:
            ra = next((r for r in rows if r['phase'] == ph and r['group'] == a), None)
            rb = next((r for r in rows if r['phase'] == ph and r['group'] == b), None)
            if not ra or not rb or not ra['rmse_pos_steady']:
                continue
            dp = (rb['rmse_pos_steady'] - ra['rmse_pos_steady']) / ra['rmse_pos_steady'] * 100
            da = (rb['rmse_att_steady'] - ra['rmse_att_steady']) / ra['rmse_att_steady'] * 100
            print(f"    {ph:<12} posizione {dp:+6.1f}%   assetto {da:+6.1f}%")
        print("\n  sull'intero volo (RMSE, decollo e coda a terra compresi:")
        print("  descrive, non confronta -- il confronto sta in maneuvers/):")
        for t_ in dict.fromkeys(r['metric'] for r in flight_rows):
            g = {r['group']: r['rmse'] for r in flight_rows
                 if r['metric'] == t_ and r['stat'] == 'RMSE'}
            if a in g and b in g and g[a]:
                print(f"    {t_:<28} {(g[b]-g[a])/g[a]*100:+6.1f}%")

    itm = {f['group']: f.get('inf_time_mean_ms') for f in flights}
    if itm and all(v is not None for v in itm.values()):
        dt_ms = np.mean([f['dt_mean'] for f in flights]) * 1e3
        print("\n  Tempo di inferenza: "
              + ", ".join(f"{g} {v:.3f} ms" for g, v in itm.items())
              + f"  (periodo di controllo {dt_ms:.2f} ms)")

    print("\n  NB: un volo per condizione. Questi sono valori singoli, non stime")
    print("      con incertezza: in tesi vanno riportati come tali, senza medie,")
    print("      bande o p-value. Le fasi non sono campioni indipendenti.")
    print(f"\nAnalisi completata. Output in: {out_dir}")
    print(f"  tabelle e manifest -> {dirs['tables']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Analisi missioni di traiettoria TiltHex: PPO vs SPPO.")
    p.add_argument("--input", type=str, required=True,
                   help="cartella dei log (es. plots/p2p_nominal)")
    p.add_argument("--t-ss", type=float, default=2.0,
                   help="durata [s] della finestra di regime a fine fase")
    p.add_argument("--tol-pos", type=float, default=0.05,
                   help="tolleranza di posizione [m] per il tempo di assestamento")
    p.add_argument("--settle-band", type=float, default=0.05,
                   help="semiampiezza della banda di assestamento, frazione "
                        "del gradino in ingresso (default %(default)s)")
    p.add_argument("--settle-min-ang-deg", type=float, default=5.0,
                   help="rotazione comandata sotto la quale non si misura "
                        "l'assestamento d'assetto (default %(default)s)")
    p.add_argument("--tol-att-deg", type=float, default=5.0,
                   help="tolleranza di assetto [deg] per il tempo di assestamento")
    p.add_argument("--step-tol-pos", type=float, default=STEP_TOL_POS,
                   help="scarto [m] oltre il quale il gradino di una fase e' "
                        "considerato diverso fra i due voli")
    p.add_argument("--step-tol-ang-deg", type=float, default=STEP_TOL_ANG,
                   help="idem, per la parte angolare [deg]")
    p.add_argument("--min-step-norm", type=float, default=MIN_STEP_NORM,
                   help="ampiezza minima [m] del gradino sotto la quale non si "
                        "normalizza il transitorio")
    p.add_argument("--v-thr", type=float, default=V_THR)
    p.add_argument("--w-thr", type=float, default=W_THR)
    p.add_argument("--min-block", type=float, default=MIN_BLOCK)
    p.add_argument("--step-pos", type=float, default=STEP_POS)
    p.add_argument("--step-ang-deg", type=float, default=2.0)
    p.add_argument("--delta", type=float, default=2.5, metavar="S",
                   help="secondi di transitorio dopo un gradino, per la "
                        "sezione categorie (default %(default)s)")
    p.add_argument("--cat-clip", action="store_true",
                   help="taglia l'asse y nella figura per categoria quando una "
                        "categoria domina le altre")
    p.add_argument("--cat-exclude", nargs="*", default=[], metavar="CAT",
                   help="categorie da togliere dalla figura (restano nel CSV)")
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.is_dir():
        print(f"Errore: cartella '{input_path}' non trovata.")
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