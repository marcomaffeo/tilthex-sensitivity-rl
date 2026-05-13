#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

def quat_wxyz_to_euler_xyz(q_wxyz):
    """
    Converte un array di quaternioni [N, 4] dal formato WXYZ (usato nel tuo nodo ROS)
    al formato angoli di Eulero XYZ (Roll, Pitch, Yaw).
    """
    # scipy usa xyzw, quindi riordiniamo le colonne
    q_xyzw = np.empty_like(q_wxyz)
    q_xyzw[:, 0] = q_wxyz[:, 1]  # x
    q_xyzw[:, 1] = q_wxyz[:, 2]  # y
    q_xyzw[:, 2] = q_wxyz[:, 3]  # z
    q_xyzw[:, 3] = q_wxyz[:, 0]  # w
    
    rot = Rotation.from_quat(q_xyzw)
    return rot.as_euler('xyz')

def export_telemetry_plots(data_path: Path, ext="svg", freq=100.0):
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

    print(f"Caricamento dati da {data_path}...")
    try:
        data = np.load(data_path, allow_pickle=True)
    except Exception as e:
        print(f"Errore nel caricamento del file npz: {e}")
        return

    # Estrazione dati
    times = data['t']
    pos_act = data['pos']
    vel_act = data['vel']
    quats_act = data['quat']
    avel_act = data['avel']
    pos_ref = data['pos_ref']
    quats_ref = data['quat_ref']
    actions = data['action']
    inf_times = data['inf_time']
    policy_name = str(data['policy_name']) if 'policy_name' in data else "Control Policy"

    # Ricostruzione Eulero (assenti nel file npz originale)
    euler_act = quat_wxyz_to_euler_xyz(quats_act)
    euler_ref = quat_wxyz_to_euler_xyz(quats_ref)

    # Calcolo errori
    err_pos_split = pos_act - pos_ref
    err_euler_split = euler_act - euler_ref
    err_euler_split = (err_euler_split + np.pi) % (2 * np.pi) - np.pi

    err_vel_split = vel_act
    err_avel_split = avel_act

    pos_err = np.linalg.norm(pos_act - pos_ref, axis=1) 
    vel_err = np.linalg.norm(vel_act, axis=1)  
    avel_err = np.linalg.norm(avel_act, axis=1) 
    dot_prod = np.clip(np.abs(np.sum(quats_act * quats_ref, axis=1)), -1.0, 1.0)
    total_error_rad = 2.0 * np.arccos(dot_prod)

    # Colori
    c_act = '#1f77b4' 
    c_ref = '#000000' 
    c_att = '#d62728' 
    c_mot = '#ff7f0e' 

    output_dir = data_path.parent / "exported_plot"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Generazione dei 7 grafici ad alta risoluzione...")

    # --- FIG 1: Position Tracking ---
    fig1, axs1 = plt.subplots(2, 3, figsize=(18, 10)) 
    fig1.suptitle(rf'$\mathrm{{{policy_name}}}\mathrm{{\ Position\ Tracking\ ({times[-1]:.1f}s)}}$', fontweight='bold') 
    titles_pos = [r'$x$ [m]', r'$y$ [m]', r'$z$ (Height) [m]']
    titles_euler = [r'$\phi$ (Roll) [rad]', r'$\theta$ (Pitch) [rad]', r'$\psi$ (Yaw) [rad]']
    
    for i in range(3):
        axs1[0, i].plot(times, pos_act[:, i], label=r'$\mathrm{Actual}$', color=c_act, linewidth=2.0)
        axs1[0, i].plot(times, pos_ref[:, i], '--', label=r'$\mathrm{Desired}$', color=c_ref, alpha=0.8, linewidth=2.0)
        axs1[0, i].set_title(titles_pos[i])
        axs1[0, i].set_xlabel(r'$t$ [s]')
        if i == 0: axs1[0, i].legend(loc='best')

        axs1[1, i].plot(times, euler_act[:, i], label=r'$\mathrm{Actual}$', color=c_att, linewidth=2.0)
        axs1[1, i].plot(times, euler_ref[:, i], '--', label=r'$\mathrm{Desired}$', color=c_ref, alpha=0.8, linewidth=2.0)
        axs1[1, i].set_title(titles_euler[i])
        axs1[1, i].set_xlabel(r'$t$ [s]')
        if i == 0: axs1[1, i].legend(loc='best')
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- FIG 2: Rotor Forces ---
    fig2, axs2 = plt.subplots(2, 3, figsize=(18, 8)) 
    fig2.suptitle(r'$\mathrm{Rotor\ Forces\ [N]}$', fontweight='bold') 
    for i in range(min(6, actions.shape[1])):
        row, col = i // 3, i % 3
        axs2[row, col].plot(times, actions[:, i], color=c_mot, linewidth=2.0)
        axs2[row, col].set_title(rf'$f_{i+1}$')
        axs2[row, col].set_xlabel(r'$t$ [s]')
        axs2[row, col].set_ylim(0, 12)
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- FIG 3: Control Metrics ---
    fig3, axs3 = plt.subplots(2, 2, figsize=(16, 10))
    fig3.suptitle(r'$\mathrm{Control\ Performance\ Metrics}$', fontweight='bold')
    err_data = [(pos_err, r'$e_p\ \mathrm{[m]}$', c_act), 
                (total_error_rad, r'$e_a\ \mathrm{[rad]}$', c_att), 
                (vel_err, r'$e_v\ \mathrm{[m/s]}$', "#2ca02c"), 
                (avel_err, r'$e_\omega\ \mathrm{[rad/s]}$', "#9467bd")]
    
    for i, (edata, label, col) in enumerate(err_data):
        r, c = i // 2, i % 2
        if len(edata) > 0:
            final_val = edata[-1]
            mae_val = np.mean(np.abs(edata))
        else:
            final_val = 0.0
            mae_val = 0.0
            
        legend_label = rf'$\mathrm{{MAE}}:\ {mae_val:.4f}\ |\ \mathrm{{Final}}:\ {final_val:.4f}$'
        axs3[r, c].plot(times, edata, color=col, label=legend_label, linewidth=2.0)
        axs3[r, c].set_title(label)
        axs3[r, c].set_xlabel(r'$t$ [s]')
        axs3[r, c].legend(loc="upper right")
    fig3.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- FIG 4: Inference Latency ---
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.plot(times, inf_times * 1000.0, color='#2ca02c', linewidth=2.0)
    ax4.axhline(y=(1.0/freq)*1000.0, color=c_att, linestyle='--', label=r'$\mathrm{Loop\ Limit}$', linewidth=2.0)
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
        mae_pos = np.mean(np.abs(err_pos_split[:, i]))
        axs5[0, i].plot(times, err_pos_split[:, i], color=c_act, label=rf'$\mathrm{{MAE}}:\ {mae_pos:.4f}$', linewidth=2.0)
        axs5[0, i].set_title(err_titles_pos[i])
        axs5[0, i].set_xlabel(r'$t$ [s]')
        axs5[0, i].legend(loc='best')

        mae_euler = np.mean(np.abs(err_euler_split[:, i]))
        axs5[1, i].plot(times, err_euler_split[:, i], color=c_att, label=rf'$\mathrm{{MAE}}:\ {mae_euler:.4f}$', linewidth=2.0)
        axs5[1, i].set_title(err_titles_euler[i])
        axs5[1, i].set_xlabel(r'$t$ [s]')
        axs5[1, i].legend(loc='best')
    fig5.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- FIG 6: Velocities ---
    fig6, axs6 = plt.subplots(2, 3, figsize=(18, 10))
    fig6.suptitle(r'$\mathrm{Pose\ Velocity\ components}$', fontweight='bold')
    titles_vel = [r'$v_x\ \mathrm{[m/s]}$', r'$v_y\ \mathrm{[m/s]}$', r'$v_z\ \mathrm{[m/s]}$']
    titles_avel = [r'$\omega_x\ \mathrm{[rad/s]}$', r'$\omega_y\ \mathrm{[rad/s]}$', r'$\omega_z\ \mathrm{[rad/s]}$']
    
    des_vel = np.zeros_like(times)
    
    for i in range(3):
        axs6[0, i].plot(times, vel_act[:, i], color=c_act, label=r'$\mathrm{Actual}$', linewidth=2.0)
        axs6[0, i].plot(times, des_vel, '--', color=c_ref, alpha=0.8, label=r'$\mathrm{Desired}$', linewidth=2.0)
        axs6[0, i].set_title(titles_vel[i])
        axs6[0, i].set_xlabel(r'$t$ [s]')
        if i == 0: axs6[0, i].legend(loc='best')

        axs6[1, i].plot(times, avel_act[:, i], color=c_att, label=r'$\mathrm{Actual}$', linewidth=2.0)
        axs6[1, i].plot(times, des_vel, '--', color=c_ref, alpha=0.8, label=r'$\mathrm{Desired}$', linewidth=2.0)
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
        mae_vel = np.mean(np.abs(err_vel_split[:, i]))
        axs7[0, i].plot(times, err_vel_split[:, i], color=c_act, label=rf'$\mathrm{{MAE}}:\ {mae_vel:.4f}$', linewidth=2.0)
        axs7[0, i].set_title(err_titles_vel[i])
        axs7[0, i].set_xlabel(r'$t$ [s]')
        axs7[0, i].legend(loc='best')

        mae_avel = np.mean(np.abs(err_avel_split[:, i]))
        axs7[1, i].plot(times, err_avel_split[:, i], color=c_att, label=rf'$\mathrm{{MAE}}:\ {mae_avel:.4f}$', linewidth=2.0)
        axs7[1, i].set_title(err_titles_avel[i])
        axs7[1, i].set_xlabel(r'$t$ [s]')
        axs7[1, i].legend(loc='best')
    fig7.tight_layout(rect=[0, 0.03, 1, 0.95])

    export_map = {
        '1_position': fig1,
        '2_inputs': fig2,
        '3_tracking_errors': fig3,
        '4_inference_latency': fig4,
        '5_position_errors': fig5,
        '6_velocities': fig6,
        '7_velocity_error': fig7
    }

    # Salvataggio nel formato richiesto
    for name, fig in export_map.items():
        save_path = output_dir / f"{name}.{ext}"
        fig.savefig(save_path, format=ext, transparent=True)

    for fig in export_map.values():
        plt.close(fig)

    print(f"[{ext.upper()}] Esportazione completata! Trovi i 7 grafici in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ROS2 telemetry plots from data.npz to PDF/SVG.")
    parser.add_argument("--data", type=str, required=True, help="Percorso del file data.npz")
    parser.add_argument("--format", type=str, choices=['svg', 'pdf'], default='pdf', help="Formato di esportazione (default: pdf)")
    parser.add_argument("--freq", type=float, default=100.0, help="Frequenza di controllo in Hz per il plot della latenza (default: 100.0)")
    
    args = parser.parse_args()
    data_path = Path(args.data)
    
    if not data_path.exists():
        print(f"Errore: File '{data_path}' non trovato.")
        exit(1)
        
    export_telemetry_plots(data_path, ext=args.format, freq=args.freq)