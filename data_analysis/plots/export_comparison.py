#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

def quat_wxyz_to_euler_xyz(q_wxyz):
    q_xyzw = np.empty_like(q_wxyz)
    q_xyzw[:, 0] = q_wxyz[:, 1]  
    q_xyzw[:, 1] = q_wxyz[:, 2]  
    q_xyzw[:, 2] = q_wxyz[:, 3]  
    q_xyzw[:, 3] = q_wxyz[:, 0]  
    
    rot = Rotation.from_quat(q_xyzw)
    return rot.as_euler('xyz')

def load_and_process(data_path: Path):
    # Load data from npz file
    data = np.load(data_path, allow_pickle=True)
    
    times = data['t']
    pos_act = data['pos']
    vel_act = data['vel']
    quats_act = data['quat']
    avel_act = data['avel']
    pos_ref = data['pos_ref']
    quats_ref = data['quat_ref']
    actions = data['action']
    inf_times = data['inf_time']

    euler_act = quat_wxyz_to_euler_xyz(quats_act)
    euler_ref = quat_wxyz_to_euler_xyz(quats_ref)

    err_pos_split = pos_act - pos_ref
    err_euler_split = euler_act - euler_ref
    err_euler_split = (err_euler_split + np.pi) % (2 * np.pi) - np.pi

    pos_err = np.linalg.norm(err_pos_split, axis=1) 
    vel_err = np.linalg.norm(vel_act, axis=1)  
    avel_err = np.linalg.norm(avel_act, axis=1) 
    dot_prod = np.clip(np.abs(np.sum(quats_act * quats_ref, axis=1)), -1.0, 1.0)
    total_error_rad = 2.0 * np.arccos(dot_prod)

    return {
        't': times, 'pos': pos_act, 'vel': vel_act, 'euler': euler_act, 'avel': avel_act,
        'pos_ref': pos_ref, 'euler_ref': euler_ref, 'action': actions, 'inf_time': inf_times,
        'err_pos_split': err_pos_split, 'err_euler_split': err_euler_split,
        'err_vel_split': vel_act, 'err_avel_split': avel_act,
        'pos_err': pos_err, 'vel_err': vel_err, 'avel_err': avel_err, 'total_error_rad': total_error_rad
    }

def export_telemetry_plots(ppo_path: Path, sppo_path: Path, ext="svg", freq=100.0):
    plt.rcParams.update({ 
        "text.usetex": False, 
        "mathtext.fontset": "cm", 
        "font.family": "serif", 
        "font.serif": ["cmr10", "STIXGeneral", "Times New Roman", "DejaVu Serif"], 
        "axes.formatter.use_mathtext": True, 
        "font.size": 14, 
        "axes.titlesize": 16,
        "axes.labelsize": 14, 
        "legend.fontsize": 12, 
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

    print(f"Caricamento dati PPO da {ppo_path} e SPPO da {sppo_path}...")
    try:
        d_ppo = load_and_process(ppo_path)
        d_sppo = load_and_process(sppo_path)
    except Exception as e:
        print(f"Errore nel caricamento dei file npz: {e}")
        return

    c_ppo = '#1f77b4'  # Blue for PPO
    c_sppo = '#d62728' # Red for SPPO
    c_ref = '#000000'  # Black for Reference

    output_dir = ppo_path.parent / "exported_plot"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Generazione dei 7 grafici ad alta risoluzione in sovrapposizione...")

    # --- FIG 1: Position Tracking ---
    fig1, axs1 = plt.subplots(2, 3, figsize=(18, 10)) 
    fig1.suptitle(r'$\mathrm{PPO\ vs\ SPPO\ Position\ Tracking}$', fontweight='bold') 
    titles_pos = [r'$x$ [m]', r'$y$ [m]', r'$z$ (Height) [m]']
    titles_euler = [r'$\phi$ (Roll) [rad]', r'$\theta$ (Pitch) [rad]', r'$\psi$ (Yaw) [rad]']
    
    for i in range(3):
        axs1[0, i].plot(d_ppo['t'], d_ppo['pos'][:, i], label=r'$\mathrm{PPO}$', color=c_ppo, linewidth=2.0)
        axs1[0, i].plot(d_sppo['t'], d_sppo['pos'][:, i], label=r'$\mathrm{SPPO}$', color=c_sppo, linewidth=2.0)
        axs1[0, i].plot(d_ppo['t'], d_ppo['pos_ref'][:, i], '--', label=r'$\mathrm{Desired}$', color=c_ref, alpha=0.8, linewidth=2.0)
        axs1[0, i].set_title(titles_pos[i])
        axs1[0, i].set_xlabel(r'$t$ [s]')
        if i == 0: axs1[0, i].legend(loc='best')

        axs1[1, i].plot(d_ppo['t'], d_ppo['euler'][:, i], label=r'$\mathrm{PPO}$', color=c_ppo, linewidth=2.0)
        axs1[1, i].plot(d_sppo['t'], d_sppo['euler'][:, i], label=r'$\mathrm{SPPO}$', color=c_sppo, linewidth=2.0)
        axs1[1, i].plot(d_ppo['t'], d_ppo['euler_ref'][:, i], '--', label=r'$\mathrm{Desired}$', color=c_ref, alpha=0.8, linewidth=2.0)
        axs1[1, i].set_title(titles_euler[i])
        axs1[1, i].set_xlabel(r'$t$ [s]')
        if i == 0: axs1[1, i].legend(loc='best')
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- FIG 2: Rotor Forces ---
    fig2, axs2 = plt.subplots(2, 3, figsize=(18, 8)) 
    fig2.suptitle(r'$\mathrm{Rotor\ Forces\ [N]}$', fontweight='bold') 
    for i in range(min(6, d_ppo['action'].shape[1])):
        row, col = i // 3, i % 3
        axs2[row, col].plot(d_ppo['t'], d_ppo['action'][:, i], color=c_ppo, label=r'$\mathrm{PPO}$', linewidth=2.0)
        axs2[row, col].plot(d_sppo['t'], d_sppo['action'][:, i], color=c_sppo, label=r'$\mathrm{SPPO}$', alpha=0.8, linewidth=2.0)
        axs2[row, col].set_title(rf'$f_{i+1}$')
        axs2[row, col].set_xlabel(r'$t$ [s]')
        axs2[row, col].set_ylim(0, 12)
        if i == 0: axs2[row, col].legend(loc='best')
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- FIG 3: Control Metrics ---
    fig3, axs3 = plt.subplots(2, 2, figsize=(16, 10))
    fig3.suptitle(r'$\mathrm{Control\ Performance\ Metrics}$', fontweight='bold')
    
    metrics_keys = [('pos_err', r'$e_p\ \mathrm{[m]}$'), 
                    ('total_error_rad', r'$e_a\ \mathrm{[rad]}$'), 
                    ('vel_err', r'$e_v\ \mathrm{[m/s]}$'), 
                    ('avel_err', r'$e_\omega\ \mathrm{[rad/s]}$')]
    
    for i, (key, label) in enumerate(metrics_keys):
        r, c = i // 2, i % 2
        
        mae_ppo = np.mean(np.abs(d_ppo[key])) if len(d_ppo[key]) > 0 else 0.0
        mae_sppo = np.mean(np.abs(d_sppo[key])) if len(d_sppo[key]) > 0 else 0.0
            
        axs3[r, c].plot(d_ppo['t'], d_ppo[key], color=c_ppo, label=rf'$\mathrm{{PPO\ MAE}}:\ {mae_ppo:.4f}$', linewidth=2.0)
        axs3[r, c].plot(d_sppo['t'], d_sppo[key], color=c_sppo, label=rf'$\mathrm{{SPPO\ MAE}}:\ {mae_sppo:.4f}$', alpha=0.8, linewidth=2.0)
        axs3[r, c].set_title(label)
        axs3[r, c].set_xlabel(r'$t$ [s]')
        axs3[r, c].legend(loc="upper right")
    fig3.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- FIG 4: Inference Latency ---
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.plot(d_ppo['t'], d_ppo['inf_time'] * 1000.0, color=c_ppo, label=r'$\mathrm{PPO}$', linewidth=2.0)
    ax4.plot(d_sppo['t'], d_sppo['inf_time'] * 1000.0, color=c_sppo, label=r'$\mathrm{SPPO}$', alpha=0.8, linewidth=2.0)
    ax4.axhline(y=(1.0/freq)*1000.0, color='#2ca02c', linestyle='--', label=r'$\mathrm{Loop\ Limit}$', linewidth=2.0)
    ax4.set_title(r'$\mathrm{JAX\ Inference\ Latency}$', fontweight='bold')
    ax4.set_ylabel(r'$\mathrm{Time\ [ms]}$')
    ax4.set_xlabel(r'$t$ [s]')
    ax4.legend(loc='best')
    fig4.tight_layout()

    # --- FIG 5: Individual State Errors ---
    fig5, axs5 = plt.subplots(2, 3, figsize=(18, 10))
    fig5.suptitle(r'$\mathrm{Individual\ States\ Tracking\ Error}$', fontweight='bold')
    err_titles_pos = [r'$e_x\ \mathrm{[m]}$', r'$e_y\ \mathrm{[m]}$', r'$e_z\ \mathrm{[m]}$']
    err_titles_euler = [r'$e_\phi\ \mathrm{[rad]}$', r'$e_\theta\ \mathrm{[rad]}$', r'$e_\psi\ \mathrm{[rad]}$']

    for i in range(3):
        mae_pos_ppo = np.mean(np.abs(d_ppo['err_pos_split'][:, i]))
        mae_pos_sppo = np.mean(np.abs(d_sppo['err_pos_split'][:, i]))
        axs5[0, i].plot(d_ppo['t'], d_ppo['err_pos_split'][:, i], color=c_ppo, label=rf'$\mathrm{{PPO\ MAE}}:\ {mae_pos_ppo:.4f}$', linewidth=2.0)
        axs5[0, i].plot(d_sppo['t'], d_sppo['err_pos_split'][:, i], color=c_sppo, label=rf'$\mathrm{{SPPO\ MAE}}:\ {mae_pos_sppo:.4f}$', alpha=0.8, linewidth=2.0)
        axs5[0, i].set_title(err_titles_pos[i])
        axs5[0, i].set_xlabel(r'$t$ [s]')
        axs5[0, i].legend(loc='best')

        mae_euler_ppo = np.mean(np.abs(d_ppo['err_euler_split'][:, i]))
        mae_euler_sppo = np.mean(np.abs(d_sppo['err_euler_split'][:, i]))
        axs5[1, i].plot(d_ppo['t'], d_ppo['err_euler_split'][:, i], color=c_ppo, label=rf'$\mathrm{{PPO\ MAE}}:\ {mae_euler_ppo:.4f}$', linewidth=2.0)
        axs5[1, i].plot(d_sppo['t'], d_sppo['err_euler_split'][:, i], color=c_sppo, label=rf'$\mathrm{{SPPO\ MAE}}:\ {mae_euler_sppo:.4f}$', alpha=0.8, linewidth=2.0)
        axs5[1, i].set_title(err_titles_euler[i])
        axs5[1, i].set_xlabel(r'$t$ [s]')
        axs5[1, i].legend(loc='best')
    fig5.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- FIG 6: Velocities ---
    fig6, axs6 = plt.subplots(2, 3, figsize=(18, 10))
    fig6.suptitle(r'$\mathrm{Pose\ Velocity\ components}$', fontweight='bold')
    titles_vel = [r'$v_x\ \mathrm{[m/s]}$', r'$v_y\ \mathrm{[m/s]}$', r'$v_z\ \mathrm{[m/s]}$']
    titles_avel = [r'$\omega_x\ \mathrm{[rad/s]}$', r'$\omega_y\ \mathrm{[rad/s]}$', r'$\omega_z\ \mathrm{[rad/s]}$']
    
    des_vel = np.zeros_like(d_ppo['t'])
    
    for i in range(3):
        axs6[0, i].plot(d_ppo['t'], d_ppo['vel'][:, i], color=c_ppo, label=r'$\mathrm{PPO}$', linewidth=2.0)
        axs6[0, i].plot(d_sppo['t'], d_sppo['vel'][:, i], color=c_sppo, label=r'$\mathrm{SPPO}$', alpha=0.8, linewidth=2.0)
        axs6[0, i].plot(d_ppo['t'], des_vel, '--', color=c_ref, alpha=0.8, label=r'$\mathrm{Desired}$', linewidth=2.0)
        axs6[0, i].set_title(titles_vel[i])
        axs6[0, i].set_xlabel(r'$t$ [s]')
        if i == 0: axs6[0, i].legend(loc='best')

        axs6[1, i].plot(d_ppo['t'], d_ppo['avel'][:, i], color=c_ppo, label=r'$\mathrm{PPO}$', linewidth=2.0)
        axs6[1, i].plot(d_sppo['t'], d_sppo['avel'][:, i], color=c_sppo, label=r'$\mathrm{SPPO}$', alpha=0.8, linewidth=2.0)
        axs6[1, i].plot(d_ppo['t'], des_vel, '--', color=c_ref, alpha=0.8, label=r'$\mathrm{Desired}$', linewidth=2.0)
        axs6[1, i].set_title(titles_avel[i])
        axs6[1, i].set_xlabel(r'$t$ [s]')
        if i == 0: axs6[1, i].legend(loc='best')
    fig6.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- FIG 7: Velocity Errors ---
    fig7, axs7 = plt.subplots(2, 3, figsize=(18, 10))
    fig7.suptitle(r'$\mathrm{Velocity\ Error\ components}$', fontweight='bold')
    err_titles_vel = [r'$e_{v_x}\ \mathrm{[m/s]}$', r'$e_{v_y}\ \mathrm{[m/s]}$', r'$e_{v_z}\ \mathrm{[m/s]}$']
    err_titles_avel = [r'$e_{\omega_x}\ \mathrm{[rad/s]}$', r'$e_{\omega_y}\ \mathrm{[rad/s]}$', r'$e_{\omega_z}\ \mathrm{[rad/s]}$']
    
    for i in range(3):
        mae_vel_ppo = np.mean(np.abs(d_ppo['err_vel_split'][:, i]))
        mae_vel_sppo = np.mean(np.abs(d_sppo['err_vel_split'][:, i]))
        axs7[0, i].plot(d_ppo['t'], d_ppo['err_vel_split'][:, i], color=c_ppo, label=rf'$\mathrm{{PPO\ MAE}}:\ {mae_vel_ppo:.4f}$', linewidth=2.0)
        axs7[0, i].plot(d_sppo['t'], d_sppo['err_vel_split'][:, i], color=c_sppo, label=rf'$\mathrm{{SPPO\ MAE}}:\ {mae_vel_sppo:.4f}$', alpha=0.8, linewidth=2.0)
        axs7[0, i].set_title(err_titles_vel[i])
        axs7[0, i].set_xlabel(r'$t$ [s]')
        axs7[0, i].legend(loc='best')

        mae_avel_ppo = np.mean(np.abs(d_ppo['err_avel_split'][:, i]))
        mae_avel_sppo = np.mean(np.abs(d_sppo['err_avel_split'][:, i]))
        axs7[1, i].plot(d_ppo['t'], d_ppo['err_avel_split'][:, i], color=c_ppo, label=rf'$\mathrm{{PPO\ MAE}}:\ {mae_avel_ppo:.4f}$', linewidth=2.0)
        axs7[1, i].plot(d_sppo['t'], d_sppo['err_avel_split'][:, i], color=c_sppo, label=rf'$\mathrm{{SPPO\ MAE}}:\ {mae_avel_sppo:.4f}$', alpha=0.8, linewidth=2.0)
        axs7[1, i].set_title(err_titles_avel[i])
        axs7[1, i].set_xlabel(r'$t$ [s]')
        axs7[1, i].legend(loc='best')
    fig7.tight_layout(rect=[0, 0.03, 1, 0.95])

    export_map = {
        '1_position_comparison': fig1,
        '2_inputs_comparison': fig2,
        '3_tracking_errors_comparison': fig3,
        '4_inference_latency_comparison': fig4,
        '5_position_errors_comparison': fig5,
        '6_velocities_comparison': fig6,
        '7_velocity_error_comparison': fig7
    }

    for name, fig in export_map.items():
        save_path = output_dir / f"{name}.{ext}"
        fig.savefig(save_path, format=ext, transparent=True)

    for fig in export_map.values():
        plt.close(fig)

    print(f"[{ext.upper()}] Esportazione completata! Trovi i 7 grafici in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ROS2 telemetry plots comparing PPO and SPPO to PDF/SVG.")
    parser.add_argument("--ppo", type=str, required=True, help="Path to PPO data.npz")
    parser.add_argument("--sppo", type=str, required=True, help="Path to SPPO data.npz")
    parser.add_argument("--format", type=str, choices=['svg', 'pdf'], default='pdf', help="Export format (default: pdf)")
    parser.add_argument("--freq", type=float, default=100.0, help="Control frequency in Hz for latency plot (default: 100.0)")
    
    args = parser.parse_args()
    ppo_path = Path(args.ppo)
    sppo_path = Path(args.sppo)
    
    if not ppo_path.exists():
        print(f"Errore: File PPO '{ppo_path}' non trovato.")
        exit(1)
    if not sppo_path.exists():
        print(f"Errore: File SPPO '{sppo_path}' non trovato.")
        exit(1)
        
    export_telemetry_plots(ppo_path, sppo_path, ext=args.format, freq=args.freq)