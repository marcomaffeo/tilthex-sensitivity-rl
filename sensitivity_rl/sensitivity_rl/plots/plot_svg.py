#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process_error_data(errors):
    if not errors: return None, None, None
    max_len = max(len(e) for e in errors)
    padded = np.array([np.pad(e, (0, max_len - len(e)), mode='edge') for e in errors])
    mean = np.mean(padded, axis=0).flatten()
    std = np.std(padded, axis=0).flatten()
    t = np.arange(max_len)
    return t, mean, std

def generate_svg_plots(data_dict, output_dir: Path, base_name="publication_plot"):
    # ==========================================
    # PUBLICATION-READY LATEX STYLE SETTINGS
    # ==========================================
    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.serif": ["cmr10", "STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "axes.formatter.use_mathtext": True,
        "font.size": 14,
        "axes.labelsize": 16,
        "legend.fontsize": 15, # AUMENTATO: Legenda più grande
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.grid": True,
        "grid.alpha": 0.4,
        "grid.linestyle": "--",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2
    })

    # Estrazione array dal dizionario .npz
    state_data = data_dict["state"]
    actions_data = data_dict["actions"]
    target_pos_data = data_dict.get("target_pos")
    target_att_data = data_dict.get("target_attitude")

    num_series = state_data.shape[0]

    # Colori Unificati
    traj_color = '#1f77b4'   # Blu standard per le traiettorie e azioni
    start_color = '#2ca02c'  # Verde per i punti di partenza
    target_color = '#d62728' # Rosso per il target

    fig_3d = plt.figure(figsize=(8, 8))
    ax_path = fig_3d.add_subplot(111, projection="3d")
    ax_path.xaxis.pane.fill = False; ax_path.yaxis.pane.fill = False; ax_path.zaxis.pane.fill = False
    
    fig_err, (ax_dist_err, ax_att_err) = plt.subplots(2, 1, figsize=(9, 8))
    fig_act, axes_act = plt.subplots(2, 3, figsize=(20, 8))
    axes_act_flat = axes_act.flatten()
    labels_act = [rf'$f_{i+1}$ [N]' for i in range(6)]

    all_dist_errors, all_att_errors = [], []

    for i in range(num_series):
        state = state_data[i]
        
        pos = state[:, :3]
        quats = state[:, 3:7]

        # Evitiamo di duplicare le label nella legenda iterando
        label_traj = r'$\mathrm{Trajectory}$' if i == 0 else None
        label_start = r'$\mathrm{Start}$' if i == 0 else None
        label_target = r'$\mathrm{Target}$' if i == 0 else None

        # BUGFIX MATPLOTLIB 3D SVG: NO trasparenza (alpha) nelle linee 3D per evitare errori di compositing.
        ax_path.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=traj_color, linewidth=1.5, label=label_traj)
        ax_path.scatter(pos[0, 0], pos[0, 1], pos[0, 2], marker="o", color=start_color, s=40, label=label_start) 

        if target_pos_data is not None:
            target = target_pos_data[i]
            if target.ndim >= 2: target = target[0]
            if i == 0:
                ax_path.scatter(target[0], target[1], target[2], marker="X", color=target_color, s=100, label=label_target)
            all_dist_errors.append(np.linalg.norm(pos - target, axis=1).flatten())

        if target_att_data is not None:
            t_a = target_att_data[i]
            target_att = np.tile(t_a[0] if t_a.ndim >= 2 else t_a, (quats.shape[0], 1))
            dot_prod = np.clip(np.sum(quats * target_att, axis=1), -1.0, 1.0)
            all_att_errors.append((2.0 * np.arccos(np.abs(dot_prod))).flatten())

        actions = actions_data[i]
        if actions.size > 0:
            t_steps = np.arange(actions.shape[0])
            for idx, ax in enumerate(axes_act_flat):
                if idx < actions.shape[1]:
                    # Usiamo alpha=0.4 per sovrapporre il fascio di azioni nei plot perturbati
                    ax.plot(t_steps, actions[:, idx], color=traj_color, alpha=0.4, linewidth=1.5)

    # Errore di Posizione
    t_pos, mean_pos, std_pos = process_error_data(all_dist_errors)
    if t_pos is not None:
        ax_dist_err.plot(t_pos, mean_pos, color="blue", linewidth=2.5, label=r'$\bar{e}_p \pm \sigma$')
        ax_dist_err.fill_between(t_pos, mean_pos - std_pos, mean_pos + std_pos, color="blue", alpha=0.2)
    
    # Errore di Attitudine
    t_att, mean_att, std_att = process_error_data(all_att_errors)
    if t_att is not None:
        ax_att_err.plot(t_att, mean_att, color="red", linewidth=2.5, label=r'$\bar{e}_a \pm \sigma$')
        ax_att_err.fill_between(t_att, mean_att - std_att, mean_att + std_att, color="red", alpha=0.2)

    # --- FORMATTAZIONE 3D ---
    ax_path.set(xlabel=r'$x$ [m]', ylabel=r'$y$ [m]', zlabel=r'$z$ [m]')
    handles, labels = ax_path.get_legend_handles_labels()
    if labels:
        ax_path.legend(loc='best', frameon=True, edgecolor='black')
    fig_3d.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    # --- FORMATTAZIONE ERRORI ---
    ax_dist_err.set_ylabel(r'$e_p$ [m]')
    handles, labels = ax_dist_err.get_legend_handles_labels()
    if labels:
        ax_dist_err.legend(loc='upper right', frameon=True, edgecolor='black', fontsize=20)
    ax_dist_err.tick_params(labelbottom=False) 

    ax_att_err.set(ylabel=r'$e_a$ [rad]', xlabel=r'$t$ [steps]')
    handles, labels = ax_att_err.get_legend_handles_labels()
    if labels:
        ax_att_err.legend(loc='upper right', frameon=True, edgecolor='black', fontsize=20)
    fig_err.align_ylabels()
    fig_err.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.2)

    # --- FORMATTAZIONE AZIONI ---
    for idx, ax in enumerate(axes_act_flat):
        ax.set_ylabel(labels_act[idx])
        ax.set_ylim(0, 12)
        if idx >= 3: 
            ax.set_xlabel(r'$t$ [steps]')
        else: 
            ax.tick_params(labelbottom=False)
    fig_act.align_ylabels()
    fig_act.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.1, hspace=0.3, wspace=0.3)

    # Salvataggio
    output_dir.mkdir(parents=True, exist_ok=True)
    # BUGFIX MATPLOTLIB 3D SVG: rimosso transparent=True per evitare compositing error
    fig_3d.savefig(output_dir / f"{base_name}_3d.svg", format='svg', transparent=False)
    fig_err.savefig(output_dir / f"{base_name}_errors.svg", format='svg', transparent=True)
    fig_act.savefig(output_dir / f"{base_name}_inputs.svg", format='svg', transparent=True)

    plt.close('all')
    print(f"SVG plots salvati in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SVG plots da dati di rollout in formato .npz.")
    parser.add_argument("--data", type=str, required=True, help="Percorso del file .npz contenente i dati del rollout")
    parser.add_argument("--outdir", type=str, default="logs/svg_plots", help="Cartella di output")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Errore: File {data_path} non trovato.")
        exit(1)

    print(f"Caricamento dati da {data_path}...")
    # Caricamento del file numpy zip
    data = np.load(data_path, allow_pickle=True)
    
    generate_svg_plots(data, Path(args.outdir), base_name=data_path.stem)