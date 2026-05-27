#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def load_actions(data_path: Path):
    data = np.load(data_path, allow_pickle=True)
    return data['t'], data['action']

def compute_total_variation(actions):
    # TV = sum(|u_{k+1} - u_k|) along the time axis for each rotor
    return np.sum(np.abs(np.diff(actions, axis=0)), axis=0)

def export_smoothness_plots(ppo_sim: Path, ppo_real: Path, sppo_sim: Path, sppo_real: Path, output_filename="control_smoothness_report.pdf"):
    plt.rcParams.update({ 
        "text.usetex": False, 
        "mathtext.fontset": "cm", 
        "font.family": "serif", 
        "font.serif": ["cmr10", "STIXGeneral", "Times New Roman", "DejaVu Serif"], 
        "axes.formatter.use_mathtext": True, 
        "font.size": 12, 
        "axes.titlesize": 14,
        "axes.labelsize": 12, 
        "legend.fontsize": 11, 
        "xtick.labelsize": 11, 
        "ytick.labelsize": 11, 
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

    print("Caricamento dati degli attuatori...")
    t_ppo_sim, act_ppo_sim = load_actions(ppo_sim)
    t_ppo_real, act_ppo_real = load_actions(ppo_real)
    t_sppo_sim, act_sppo_sim = load_actions(sppo_sim)
    t_sppo_real, act_sppo_real = load_actions(sppo_real)

    # Compute Total Variation metrics
    tv_ppo_sim = compute_total_variation(act_ppo_sim)
    tv_ppo_real = compute_total_variation(act_ppo_real)
    tv_sppo_sim = compute_total_variation(act_sppo_sim)
    tv_sppo_real = compute_total_variation(act_sppo_real)

    num_rotors = act_ppo_real.shape[1]
    output_pdf = ppo_real.parent / output_filename

    with PdfPages(output_pdf) as pdf:
        print("Generazione dei grafici di smoothness ed efficienza energetica...")

        # --- PAGE 1: Total Variation Bar Chart ---
        fig1, ax1 = plt.subplots(figsize=(14, 8))
        fig1.suptitle(r'$\mathrm{{Actuator\ Total\ Variation\ (TV)\ Comparison}}$', fontweight='bold', fontsize=16)
        
        x = np.arange(num_rotors)
        width = 0.2
        
        # Plotting the 4 conditions side by side for each rotor
        rects1 = ax1.bar(x - 1.5*width, tv_ppo_sim, width, label=r'$\mathrm{{PPO\ Simulation}}$', color='#a6cee3')
        rects2 = ax1.bar(x - 0.5*width, tv_ppo_real, width, label=r'$\mathrm{{PPO\ Real\ World}}$', color='#1f77b4')
        rects3 = ax1.bar(x + 0.5*width, tv_sppo_sim, width, label=r'$\mathrm{{SPPO\ Simulation}}$', color='#fb9a99')
        rects4 = ax1.bar(x + 1.5*width, tv_sppo_real, width, label=r'$\mathrm{{SPPO\ Real\ World}}$', color='#d62728')
        
        ax1.set_ylabel(r'$\mathrm{{Total\ Variation\ (TV)\ [N]}}$', fontsize=13)
        ax1.set_title(r'$\mathrm{{Lower\ TV\ indicates\ smoother\ control\ inputs\ and\ reduced\ actuator\ chattering}}$', fontsize=12, style='italic', pad=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels([rf'$f_{{{i+1}}}$' for i in range(num_rotors)])
        ax1.legend(loc='upper left')

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax1.annotate(f'{height:.1f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)
        autolabel(rects4)

        fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig1)
        plt.close(fig1)

        # --- PAGE 2: Control Input Time Series Snippet (Real World) ---
        # Plotting 3 representative rotors to inspect chattering behavior over time
        fig2, axs2 = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        fig2.suptitle(r'$\mathrm{{Real-World\ Control\ Input\ Signal\ Quality\ (Chattering\ Inspection)}}$', fontweight='bold', fontsize=16)
        
        # Inspecting a subset of time to highlight high frequency details clearly
        start_idx = int(len(t_ppo_real) * 0.2)
        end_idx = int(len(t_ppo_real) * 0.5) if len(t_ppo_real) * 0.5 < len(t_ppo_real) else len(t_ppo_real) - 1
        
        t_slice_ppo = t_ppo_real[start_idx:end_idx]
        t_slice_sppo = t_sppo_real[start_idx:end_idx]

        rotor_indices = [0, 1, 2] # Rotors 1, 2, and 3
        for i, r_idx in enumerate(rotor_indices):
            axs2[i].plot(t_slice_ppo, act_ppo_real[start_idx:end_idx, r_idx], color='#1f77b4', alpha=0.8, 
                         label=rf'$\mathrm{{PPO\ Real\ (Rotor\ TV: {tv_ppo_real[r_idx]:.1f}\ [N])}}$', linewidth=1.5)
            axs2[i].plot(t_slice_sppo, act_sppo_real[start_idx:end_idx, r_idx], color='#d62728', alpha=0.9, 
                         label=rf'$\mathrm{{SPPO\ Real\ (Rotor\ TV: {tv_sppo_real[r_idx]:.1f}\ [N])}}$', linewidth=1.5)
            
            axs2[i].set_ylabel(rf'$f_{{{r_idx+1}}}\ \mathrm{{[N]}}$', fontsize=12)
            axs2[i].legend(loc='upper right')
            axs2[i].set_ylim(0, 12)
            
        axs2[-1].set_xlabel(r'$t\ \mathrm{{[s]}}$', fontsize=13)
        fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig2)
        plt.close(fig2)

    print(f"[PDF] Analisi della smoothness completata! File salvato in: {output_pdf}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze control input smoothness and compute Total Variation (TV) between PPO and SPPO.")
    parser.add_argument("--ppo-sim", type=str, required=True, help="Path to PPO simulation data.npz")
    parser.add_argument("--ppo-real", type=str, required=True, help="Path to PPO real-world data.npz")
    parser.add_argument("--sppo-sim", type=str, required=True, help="Path to SPPO simulation data.npz")
    parser.add_argument("--sppo-real", type=str, required=True, help="Path to SPPO real-world data.npz")
    parser.add_argument("--output", type=str, default="control_smoothness_report.pdf", help="Output PDF filename")
    
    args = parser.parse_args()
    
    paths = {
        "PPO Sim": Path(args.ppo_sim),
        "PPO Real": Path(args.ppo_real),
        "SPPO Sim": Path(args.sppo_sim),
        "SPPO Real": Path(args.sppo_real)
    }
    
    for name, p in paths.items():
        if not p.exists():
            print(f"Errore: File '{name}' in {p} non trovato.")
            exit(1)
            
    export_smoothness_plots(paths["PPO Sim"], paths["PPO Real"], paths["SPPO Sim"], paths["SPPO Real"], output_filename=args.output)