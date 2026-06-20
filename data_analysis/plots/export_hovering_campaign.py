#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
import matplotlib
import warnings
import csv
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

def process_log(npz_path: Path):
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"Errore nel caricamento di {npz_path}: {e}")
        return None

    if 't' not in data:
        print(f"-> File {npz_path.name} ignorato: manca la chiave 't'.")
        return None
        
    times = data['t']
    times = times - times[0] 
    
    policy_name = str(data['policy_name']) if 'policy_name' in data else ""
    folder_name = npz_path.parent.name.upper()
    
    if "SPPO" in folder_name or "SPPO" in policy_name.upper():
        group = "SPPO"
    elif "PPO" in folder_name or "PPO" in policy_name.upper():
        group = "PPO"
    else:
        group = "Unknown"

    res = {'t': times, 'group': group}
    
    if 'pos' in data and 'quat' in data and 'pos_ref' in data and 'quat_ref' in data:
        pos_act = data['pos']
        pos_ref = data['pos_ref']
        quats_act = data['quat']
        quats_ref = data['quat_ref']

        euler_act = quat_wxyz_to_euler_xyz(quats_act)
        euler_ref = quat_wxyz_to_euler_xyz(quats_ref)

        err_pos = pos_act - pos_ref
        err_euler = euler_act - euler_ref
        err_euler = (err_euler + np.pi) % (2 * np.pi) - np.pi

        res['e_x'] = err_pos[:, 0]
        res['e_y'] = err_pos[:, 1]
        res['e_z'] = err_pos[:, 2]
        res['e_roll'] = err_euler[:, 0]
        res['e_pitch'] = err_euler[:, 1]
        res['e_yaw'] = err_euler[:, 2]
        res['e_pos_norm'] = np.linalg.norm(err_pos, axis=1)
        dot_prod = np.clip(np.abs(np.sum(quats_act * quats_ref, axis=1)), -1.0, 1.0)
        res['e_att_norm'] = 2.0 * np.arccos(dot_prod)
        
    elif 'pos_err' in data and 'att_err' in data:
        if len(data['pos_err'].shape) > 1 and data['pos_err'].shape[1] == 3:
            res['e_x'] = data['pos_err'][:, 0]
            res['e_y'] = data['pos_err'][:, 1]
            res['e_z'] = data['pos_err'][:, 2]
            res['e_pos_norm'] = np.linalg.norm(data['pos_err'], axis=1)
        else:
            res['e_pos_norm'] = data['pos_err']
            
        if len(data['att_err'].shape) > 1 and data['att_err'].shape[1] == 3:
            res['e_roll'] = data['att_err'][:, 0]
            res['e_pitch'] = data['att_err'][:, 1]
            res['e_yaw'] = data['att_err'][:, 2]
            res['e_att_norm'] = np.linalg.norm(data['att_err'], axis=1)
        else:
            res['e_att_norm'] = data['att_err']
            
        if 'vel_err' in data:
            res['e_vel_norm'] = np.linalg.norm(data['vel_err'], axis=1) if len(data['vel_err'].shape) > 1 else data['vel_err']
        if 'avel_err' in data:
            res['e_avel_norm'] = np.linalg.norm(data['avel_err'], axis=1) if len(data['avel_err'].shape) > 1 else data['avel_err']
    else:
        print(f"-> File {npz_path.name} in {npz_path.parent.name} ignorato: chiavi mancanti. Chiavi: {list(data.keys())}")
        return None

    if 'action' in data:
        motor_cmds = data['action']
        if motor_cmds.ndim == 2 and motor_cmds.shape[0] > 1:
            a_span = 2.0 
            delta_a = np.diff(motor_cmds, axis=0) / a_span
            
            inst_tv = np.linalg.norm(delta_a, axis=1)
            inst_tv = np.insert(inst_tv, 0, 0.0)
            
            res['inst_tv'] = inst_tv
            res['tv_a_global'] = np.sum(inst_tv)

    return res

def export_campaign_plots(input_dir: Path, output_dir: Path):
    plt.rcParams.update({ 
        "text.usetex": False, 
        "mathtext.fontset": "cm", 
        "font.family": "serif", 
        "font.serif": ["cmr10", "STIXGeneral", "Times New Roman", "DejaVu Serif"], 
        "axes.formatter.use_mathtext": True, 
        "font.size": 16, 
        "axes.titlesize": 18,
        "axes.labelsize": 16, 
        "legend.fontsize": 14, 
        "xtick.labelsize": 14, 
        "ytick.labelsize": 14, 
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

    print(f"Ricerca dei file data.npz in: {input_dir}...")
    npz_files = list(input_dir.rglob("data.npz"))
    
    if not npz_files:
        print("Nessun file data.npz trovato.")
        return
        
    print(f"Trovati {len(npz_files)} file. Estrazione in corso...")
    
    all_data = []
    for f in npz_files:
        res = process_log(f)
        if res and res['group'] in ["PPO", "SPPO"]:
            all_data.append(res)
            
    count_ppo = sum(1 for d in all_data if d['group'] == "PPO")
    count_sppo = sum(1 for d in all_data if d['group'] == "SPPO")
    print(f"Acquisiti con successo: {count_ppo} voli PPO, {count_sppo} voli SPPO.")

    if not all_data:
        print("Nessun dato valido estratto. Uscita.")
        return

    max_time_global = max(d['t'][-1] for d in all_data)
    print(f"Durata massima identificata nella campagna: {max_time_global:.1f} s")

    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_def = [
        ('e_x', r'$e_x$ [m]', 'Position Error X', '1_error_pos_x.pdf'),
        ('e_y', r'$e_y$ [m]', 'Position Error Y', '2_error_pos_y.pdf'),
        ('e_z', r'$e_z$ [m]', 'Position Error Z', '3_error_pos_z.pdf'),
        ('e_roll', r'$e_\phi$ [rad]', 'Attitude Error Roll', '4_error_att_roll.pdf'),
        ('e_pitch', r'$e_\theta$ [rad]', 'Attitude Error Pitch', '5_error_att_pitch.pdf'),
        ('e_yaw', r'$e_\psi$ [rad]', 'Attitude Error Yaw', '6_error_att_yaw.pdf'),
        ('e_pos_norm', r'$||e_p||$ [m]', 'Total Position Error', '7_error_pos_total.pdf'),
        ('e_att_norm', r'$||e_R||$ [rad]', 'Total Attitude Error', '8_error_att_total.pdf'),
        ('e_vel_norm', r'$||e_v||$ [m/s]', 'Total Velocity Error', '9_error_vel_total.pdf'),
        ('e_avel_norm', r'$||e_\omega||$ [rad/s]', 'Total Angular Vel Error', '10_error_avel_total.pdf'),
        ('inst_tv', r'$||\Delta a / a_{\mathrm{span}}||_2$', 'Instantaneous Action Variation', '11_action_variation.pdf')
    ]
    
    metrics_to_plot = [m for m in metrics_def if m[0] in all_data[0]]

    c_ppo = '#1f77b4'  
    c_sppo = '#d62728' 
    
    for idx, (key, ylabel, title, filename) in enumerate(metrics_to_plot):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ppo_label_added = False
        sppo_label_added = False
        
        for d in all_data:
            if d['group'] == 'PPO':
                lbl = r'$\mathrm{PPO}$' if not ppo_label_added else None
                color = c_ppo
                ppo_label_added = True
            else:
                lbl = r'$\mathrm{SPPO}$' if not sppo_label_added else None
                color = c_sppo
                sppo_label_added = True
                
            ax.plot(d['t'], d[key], color=color, alpha=0.3, linewidth=2.0, label=lbl)
            
            if d['t'][-1] < max_time_global - 0.5:
                ax.plot(d['t'][-1], d[key][-1], marker='x', markersize=8, color=color, 
                        markeredgewidth=2, alpha=0.8, zorder=10)
                
        ax.axhline(0, color='black', linewidth=1.0, linestyle='--', alpha=0.5)

        formatted_title = title.replace(" ", r"\ ")
        ax.set_title(rf'$\mathrm{{{formatted_title}}}$', fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_xlabel(r'$t$ [s]')
        
        leg = ax.legend(loc='best', frameon=True)
        for lh in leg.legend_handles:
            lh.set_alpha(1.0)
            lh.set_linewidth(3.0)
            
        fig.savefig(output_dir / filename, format='pdf', transparent=True)
        plt.close(fig)
        print(f"Generato: {filename}")

    print("Generazione dei grafici di Media + Deviazione Standard (primi 10s)...")
    t_max = 10.0
    t_grid = np.linspace(0, t_max, 1000)
    
    t_ss_start = 3.0
    ss_mask = t_grid >= t_ss_start
    
    table_data = []
    
    for idx, (key, ylabel, title, filename) in enumerate(metrics_to_plot):
        ms_filename = filename.replace('.pdf', '_mean_std.pdf')
        ms_title = f"Mean Std {title}"
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for group, color in [('PPO', c_ppo), ('SPPO', c_sppo)]:
            group_data = [d for d in all_data if d['group'] == group]
            if not group_data:
                continue
                
            interp_vals = []
            for d in group_data:
                vals = np.interp(t_grid, d['t'], d[key], left=np.nan, right=np.nan)
                interp_vals.append(vals)
                
            interp_vals = np.array(interp_vals)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_vals = np.nanmean(interp_vals, axis=0)
                std_vals = np.nanstd(interp_vals, axis=0)
                
            ss_data = interp_vals[:, ss_mask]
            
            ss_mean = np.nanmean(ss_data)
            ss_std = np.nanstd(ss_data)
            ss_rmse = np.sqrt(np.nanmean(np.square(ss_data)))
            ss_mae = np.nanmean(np.abs(ss_data))
            
            table_data.append({
                'Metric': title,
                'Group': group,
                'SS_Mean': ss_mean,
                'SS_Std': ss_std,
                'SS_RMSE': ss_rmse,
                'SS_MAE': ss_mae
            })
            
            lbl = rf'$\mathrm{{{group}}}\ (\mathrm{{}}\ {ss_mean:.3f} \pm {ss_std:.3f})$'
            ax.plot(t_grid, mean_vals, color=color, linewidth=2.5, label=lbl)
            ax.fill_between(t_grid, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=0.2)
            
        ax.axhline(0, color='black', linewidth=1.0, linestyle='--', alpha=0.5)

        formatted_ms_title = ms_title.replace(" ", r"\ ")
        ax.set_title(rf'$\mathrm{{{formatted_ms_title}}}$', fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_xlabel(r'$t$ [s]')
        ax.set_xlim([0, t_max])
        
        ax.legend(loc='best', frameon=True)
        
        fig.savefig(output_dir / ms_filename, format='pdf', transparent=True)
        plt.close(fig)
        print(f"Generato: {ms_filename}")
        
    ppo_tv_values = [d['tv_a_global'] for d in all_data if d['group'] == 'PPO' and 'tv_a_global' in d]
    sppo_tv_values = [d['tv_a_global'] for d in all_data if d['group'] == 'SPPO' and 'tv_a_global' in d]

    if ppo_tv_values or sppo_tv_values:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        plot_data = []
        labels = []
        colors = []
        
        if ppo_tv_values:
            plot_data.append(ppo_tv_values)
            labels.append('PPO')
            colors.append(c_ppo)
        if sppo_tv_values:
            plot_data.append(sppo_tv_values)
            labels.append('SPPO')
            colors.append(c_sppo)

        bplot = ax.boxplot(plot_data, patch_artist=True, vert=True)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)

        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_title(r'$\mathrm{Control\ Effort\ Smoothness\ (Total\ Variation)}$', fontweight='bold')
        ax.set_ylabel(r'Total Variation ($\mathrm{TV}_{\mathrm{a}}$)')
        
        tv_filename = 'total_variation_boxplot.pdf'
        fig.savefig(output_dir / tv_filename, format='pdf', transparent=True)
        plt.close(fig)
        print(f"Generato: {tv_filename}")
    else:
        print("\nAVVISO: Dati per la Total Variation non trovati in nessun file .npz.")

    table_path = output_dir / 'summary_table.csv'
    with open(table_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Group', 'SteadyState_Mean', 'SteadyState_Std', 'SteadyState_RMSE', 'SteadyState_MAE'])
        for row in table_data:
            writer.writerow([
                row['Metric'], 
                row['Group'], 
                f"{row['SS_Mean']:.4f}", 
                f"{row['SS_Std']:.4f}",
                f"{row['SS_RMSE']:.4f}",
                f"{row['SS_MAE']:.4f}"
            ])
            
    print(f"Generata tabella riassuntiva: {table_path.name}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export combined telemetry plots for a hovering campaign (PPO vs SPPO).")
    parser.add_argument("--input", type=str, required=True, help="Cartella contenente tutte le run (es. plots/hovering)")
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = input_dir / "campaign_results"
    
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Errore: Cartella di input '{input_dir}' non trovata.")
        exit(1)
        
    export_campaign_plots(input_dir, output_dir)