#!/usr/bin/env python3
"""
Post-processing script to generate Publication-Ready SVG plots from telemetry_full.npz files.
"""
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

def generate_svg_plots(npz_path: Path):
    if not npz_path.exists():
        raise FileNotFoundError(f"Telemetry file not found: {npz_path}")

    print(f"Loading telemetry from: {npz_path}")
    data = np.load(npz_path)

    policy_name = str(data['policy_name'])
    
    out_dir = Path("plot_svg") / policy_name
    out_dir.mkdir(parents=True, exist_ok=True)

    times = data['t']
    pos_act = data['pos']
    vel_act = data['vel']
    quats_act = data['quat']
    avel_act = data['avel']
    pos_ref = data['pos_ref']
    quats_ref = data['quat_ref']
    actions = data['action']
    inf_times = data['inf_time']

    euler_act = Rotation.from_quat(quats_act, scalar_first=True).as_euler('xyz')
    euler_ref = Rotation.from_quat(quats_ref, scalar_first=True).as_euler('xyz')

    pos_err = np.linalg.norm(pos_act - pos_ref, axis=1)
    vel_err = np.linalg.norm(vel_act, axis=1)
    avel_err = np.linalg.norm(avel_act, axis=1)

    dot_prod = np.sum(quats_act * quats_ref, axis=1)
    dot_prod = np.clip(np.abs(dot_prod), -1.0, 1.0)
    total_error_rad = 2.0 * np.arccos(dot_prod)

    # ==========================================
    # PUBLICATION-READY LATEX STYLE SETTINGS
    # ==========================================
    plt.rcParams.update({
        "text.usetex": True,                   # Abilita il rendering LaTeX nativo
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"], # Font standard LaTeX
        "font.size": 14,
        "axes.labelsize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.grid": True,
        "grid.alpha": 0.4,                     # Griglia molto leggera
        "grid.linestyle": "--",
        "axes.facecolor": "white",             # Niente sfondi grigi
        "figure.facecolor": "white",
        "savefig.bbox": "tight",               # Rimuove i bordi bianchi in eccesso
        "savefig.pad_inches": 0.05
    })

    # --- FIG 1: Positions & Attitude ---
    fig1, axs1 = plt.subplots(2, 3, figsize=(16, 7))
    
    ylabels_pos = [r'$x$ [m]', r'$y$ [m]', r'$z$ [m]']
    ylabels_euler = [r'$\phi$ [rad]', r'$\theta$ [rad]', r'$\psi$ [rad]']
    
    for i in range(3):
        # Position (Row 0)
        axs1[0, i].plot(times, pos_act[:, i], label=r'$\mathrm{Actual}$', color='blue', linewidth=1.5)
        axs1[0, i].plot(times, pos_ref[:, i], '--', label=r'$\mathrm{Reference}$', color='black', alpha=0.8, linewidth=1.5)
        axs1[0, i].set_ylabel(ylabels_pos[i])
        axs1[0, i].tick_params(labelbottom=False) # Nasconde le label X nella riga superiore
        
        # Attitude (Row 1)
        axs1[1, i].plot(times, euler_act[:, i], color='red', linewidth=1.5)
        axs1[1, i].plot(times, euler_ref[:, i], '--', color='black', alpha=0.8, linewidth=1.5)
        axs1[1, i].set_ylabel(ylabels_euler[i])
        axs1[1, i].set_xlabel(r'$t$ [s]')
    
    # Legend solo nel primo subplot per pulizia visiva
    axs1[0, 0].legend(loc='best', frameon=True, edgecolor='black')
    fig1.align_ylabels()
    fig1.tight_layout()

    # --- FIG 2: Inputs (Rotor Commands) ---
    fig2, axs2 = plt.subplots(2, 3, figsize=(16, 7))
    for i in range(min(6, actions.shape[1])):
        row, col = i // 3, i % 3
        axs2[row, col].plot(times, actions[:, i], color='goldenrod', linewidth=1.5)
        axs2[row, col].set_ylabel(rf'$f_{i+1}$ [N]')
        axs2[row, col].set_ylim(0, 12)
        
        if row == 0:
            axs2[row, col].tick_params(labelbottom=False)
        else:
            axs2[row, col].set_xlabel(r'$t$ [s]')
            
    fig2.align_ylabels()
    fig2.tight_layout()

    # --- FIG 3: Tracking Errors ---
    fig3, axs3 = plt.subplots(2, 2, figsize=(12, 7))
    err_data = [
        (pos_err, r'$e_p$ [m]', "blue"), 
        (total_error_rad, r'$e_R$ [rad]', "red"), 
        (vel_err, r'$e_v$ [m/s]', "green"), 
        (avel_err, r'$e_\omega$ [rad/s]', "purple")
    ]
    for i, (data_arr, ylabel, col) in enumerate(err_data):
        r, c = i // 2, i % 2
        axs3[r, c].plot(times, data_arr, color=col, linewidth=1.5)
        axs3[r, c].set_ylabel(ylabel)
        
        if r == 0:
            axs3[r, c].tick_params(labelbottom=False)
        else:
            axs3[r, c].set_xlabel(r'$t$ [s]')
            
    fig3.align_ylabels()
    fig3.tight_layout()

    # --- FIG 4: JAX Inference Latency ---
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    ax4.plot(times, inf_times * 1000.0, color='forestgreen', linewidth=1.5)
    ax4.set_ylabel(r'$t_{\mathrm{inf}}$ [ms]')
    ax4.set_xlabel(r'$t$ [s]')
    fig4.tight_layout()

    # --- SVG Export ---
    figures = {
        'positions_attitude': fig1,
        'rotor_forces': fig2,
        'tracking_errors': fig3,
        'inference_latency': fig4
    }

    print(f"Exporting publication-ready SVG plots to: {out_dir}")
    for name, fig in figures.items():
        save_path = out_dir / f"{name}.svg"
        # transparent=True assicura che il background sia trasparente per sovrapporlo in LaTeX
        fig.savefig(save_path, format='svg', transparent=True)
        plt.close(fig)

    print("SVG export complete.")

if __name__ == "__main__":
    TARGET_NPZ_FILE = "logs/SPPO_20260417_183227/telemetry_full.npz"
    
    npz_path = Path(TARGET_NPZ_FILE)
    
    try:
        generate_svg_plots(npz_path)
    except Exception as e:
        print(f"FAILED: {e}")