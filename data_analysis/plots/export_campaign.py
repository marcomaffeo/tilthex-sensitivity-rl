#!/usr/bin/env python3
"""
Comparison script to generate Publication-Ready SVG plots comparing two telemetry files.
Includes tracking errors and rotor inputs comparison with extra-large legends and thick lines.
"""
import argparse
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

def load_data(path: Path):
    if path.is_dir():
        npz_path = path / "telemetry_full.npz"
    else:
        npz_path = path
        
    if not npz_path.exists():
        raise FileNotFoundError(f"File not found: {npz_path}")
        
    data = np.load(npz_path)
    
    # Pre-processing
    times = data['t']
    pos_act = data['pos']
    pos_ref = data['pos_ref']
    quats_act = data['quat']
    quats_ref = data['quat_ref']
    actions = data['action']
    
    # Errors
    pos_err = np.linalg.norm(pos_act - pos_ref, axis=1)
    dot_prod = np.sum(quats_act * quats_ref, axis=1)
    dot_prod = np.clip(np.abs(dot_prod), -1.0, 1.0)
    att_err = 2.0 * np.arccos(dot_prod)
    
    return times, pos_err, att_err, actions

def generate_comparison_plots(path1: Path, label1: str, path2: Path, label2: str):
    # Caricamento dati
    t1, p_err1, a_err1, act1 = load_data(path1)
    t2, p_err2, a_err2, act2 = load_data(path2)

    # Output directory
    out_dir = Path("comparison_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # PUBLICATION-READY LATEX STYLE SETTINGS
    # ==========================================
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 18,            
        "axes.labelsize": 22,       
        "legend.fontsize": 24, 
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
        "lines.antialiased": True
    })

    C_PPO, C_SPPO, C_FALLBACK = '#ff7f0e', '#1f77b4', '#2ca02c'  # arancione / blu, come analyze_trajectory.py

    def color_for(label: str) -> str:
        key = label.strip().upper()
        if key == 'PPO':
            return C_PPO
        if key == 'SPPO':
            return C_SPPO
        return C_FALLBACK

    c1, c2 = color_for(label1), color_for(label2)

    # --- FIG 1: Comparison Tracking Errors ---
    fig_err, axs_err = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Position Error
    axs_err[0].plot(t1, p_err1, color=c1, label=label1, linewidth=4.0) # Linea molto spessa
    axs_err[0].plot(t2, p_err2, color=c2, label=label2, linewidth=4.0, alpha=0.8)
    axs_err[0].set_ylabel(r'$e_p$ [m]')
    axs_err[0].legend(loc='upper right', frameon=True, shadow=True)
    
    # Attitude Error
    axs_err[1].plot(t1, a_err1, color=c1, label=label1, linewidth=4.0)
    axs_err[1].plot(t2, a_err2, color=c2, label=label2, linewidth=4.0, alpha=0.8)
    axs_err[1].set_ylabel(r'$e_R$ [rad]')
    axs_err[1].set_xlabel(r'$t$ [s]')
            
    fig_err.align_ylabels()
    fig_err.tight_layout()
    fig_err.savefig(out_dir / f"comparison_errors_{label1}_{label2}.svg", format='svg', transparent=True)

    # --- FIG 2: Comparison Rotor Inputs ---
    fig_inp, axs_inp = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    num_rotors = min(6, act1.shape[1], act2.shape[1])
    
    for i in range(num_rotors):
        row, col = i // 3, i % 3
        axs_inp[row, col].plot(t1, act1[:, i], color=c1, label=label1, linewidth=3.0)
        axs_inp[row, col].plot(t2, act2[:, i], color=c2, label=label2, linewidth=3.0, alpha=0.7)
        axs_inp[row, col].set_ylabel(rf'$f_{i+1}$ [N]')
        axs_inp[row, col].set_ylim(0, 12)
        
        if row == 1:
            axs_inp[row, col].set_xlabel(r'$t$ [s]')
        if i == 0: 
            axs_inp[row, col].legend(loc='upper right', frameon=True, fontsize=20)

    fig_inp.align_ylabels()
    fig_inp.tight_layout()
    fig_inp.savefig(out_dir / f"comparison_inputs_{label1}_{label2}.svg", format='svg', transparent=True)
    
    plt.close('all')
    print(f"Comparison plots (Extra Thick Lines) saved in: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two telemetry logs.")
    parser.add_argument("path1", type=str, help="Path to first log")
    parser.add_argument("label1", type=str, help="Label for first log")
    parser.add_argument("path2", type=str, help="Path to second log")
    parser.add_argument("label2", type=str, help="Label for second log")
    
    args = parser.parse_args()

    try:
        generate_comparison_plots(Path(args.path1), args.label1, Path(args.path2), args.label2)
    except Exception as e:
        print(f"FAILED: {e}")