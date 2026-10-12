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

def export_campaign_plots(input_dir: Path, output_dir: Path, error_metric: str, t_crop: float = None):
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

    actual_max_time = max(d['t'][-1] for d in all_data)
    
    if t_crop is not None:
        t_max = min(actual_max_time, t_crop)
        print(f"Durata massima limitata a {t_max:.1f} s tramite parametro --t-crop (max originale: {actual_max_time:.1f} s)")
    else:
        t_max = actual_max_time
        print(f"Durata massima identificata nella campagna: {t_max:.1f} s")

    output_dir.mkdir(parents=True, exist_ok=True)

    # N.B. Ho aggiornato i titoli e le label per riflettere il fatto che plottiamo il Valore Assoluto
    metrics_def = [
        ('e_x', r'$|e_x|$ [m]', 'Absolute Position Error X', '1_error_pos_x.pdf'),
        ('e_y', r'$|e_y|$ [m]', 'Absolute Position Error Y', '2_error_pos_y.pdf'),
        ('e_z', r'$|e_z|$ [m]', 'Absolute Position Error Z', '3_error_pos_z.pdf'),
        ('e_roll', r'$|e_\phi|$ [rad]', 'Absolute Attitude Error Roll', '4_error_att_roll.pdf'),
        ('e_pitch', r'$|e_\theta|$ [rad]', 'Absolute Attitude Error Pitch', '5_error_att_pitch.pdf'),
        ('e_yaw', r'$|e_\psi|$ [rad]', 'Absolute Attitude Error Yaw', '6_error_att_yaw.pdf'),
        ('e_pos_norm', r'$||e_p||$ [m]', 'Total Position Error', '7_error_pos_total.pdf'),
        ('e_att_norm', r'$||e_R||$ [rad]', 'Total Attitude Error', '8_error_att_total.pdf'),
        ('e_vel_norm', r'$||e_v||$ [m/s]', 'Total Velocity Error', '9_error_vel_total.pdf'),
        ('e_avel_norm', r'$||e_\omega||$ [rad/s]', 'Total Angular Vel Error', '10_error_avel_total.pdf')
    ]
    
    metrics_to_plot = [m for m in metrics_def if m[0] in all_data[0]]
    c_ppo = '#ff7f0e'  # arancione
    c_sppo = '#1f77b4' # blu

    print(f"Generazione dei grafici aggregati utilizzando la metrica {error_metric}...")
    
    t_grid = np.linspace(0, t_max, 1000)
    t_start = 0.0
    mask = t_grid >= t_start
    
    table_data = []
    
    for idx, (key, ylabel, title, filename) in enumerate(metrics_to_plot):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for group, color in [('PPO', c_ppo), ('SPPO', c_sppo)]:
            group_data = [d for d in all_data if d['group'] == group]
            if not group_data:
                continue
                
            interp_vals = []
            flight_rmses = []
            flight_maes = []
            flight_maxs = []
            
            for d in group_data:
                vals = np.interp(t_grid, d['t'], d[key], left=np.nan, right=np.nan)
                
                valid_data = vals[mask]
                if len(valid_data) > 0:
                    flight_rmses.append(np.sqrt(np.nanmean(np.square(valid_data))))
                    flight_maes.append(np.nanmean(np.abs(valid_data)))
                    flight_maxs.append(np.nanmax(np.abs(valid_data)))
                
                # CRUCIALE: Plottiamo la magnitudo dell'errore per allineare il grafico all'RMSE/MAE calcolato
                interp_vals.append(np.abs(vals))
                
            interp_vals = np.array(interp_vals)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_vals = np.nanmean(interp_vals, axis=0)
                std_vals = np.nanstd(interp_vals, axis=0)
                
            mean_of_rmse = np.nanmean(flight_rmses)
            mean_of_mae  = np.nanmean(flight_maes)
            mean_of_max  = np.nanmean(flight_maxs)

            std_of_rmse = np.nanstd(flight_rmses, ddof=1)
            std_of_mae  = np.nanstd(flight_maes,  ddof=1)
            std_of_max  = np.nanstd(flight_maxs,  ddof=1)
            
            table_data.append({
                'Metric': title.replace("Absolute ", ""), # Rimuoviamo "Absolute" per mantenere pulita la tabella
                'Group': group,
                'Mean_RMSE': mean_of_rmse,
                'Std_RMSE': std_of_rmse,
                'Mean_MAE': mean_of_mae,
                'Std_MAE': std_of_mae,
                'Mean_MAX': mean_of_max,
                'Std_MAX': std_of_max
            })
            
            if error_metric.upper() == 'MAE':
                metric_mean = mean_of_mae
                metric_std = std_of_mae
            else:
                metric_mean = mean_of_rmse
                metric_std = std_of_rmse
            
            if len(group_data) == 1:
                lbl = rf'$\mathrm{{{group}}}\ (\mathrm{{{error_metric}:}}\ {metric_mean:.3f})$'
            else:
                lbl = rf'$\mathrm{{{group}}}\ (\mathrm{{{error_metric}:}}\ {metric_mean:.3f} \pm {metric_std:.3f})$'
            
            ax.plot(t_grid, mean_vals, color=color, linewidth=2.5, label=lbl)
            
            if len(group_data) > 1:
                ax.fill_between(t_grid, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=0.2)
            
        ax.axhline(0, color='black', linewidth=1.0, linestyle='--', alpha=0.5)

        formatted_title = title.replace(" ", r"\ ")
        ax.set_title(rf'$\mathrm{{{formatted_title}}}$', fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_xlabel(r'$t$ [s]')
        ax.set_xlim([0, t_max])
        
        # Facciamo partire l'asse Y da 0 per gli errori assoluti
        bottom, top = ax.get_ylim()
        ax.set_ylim(0, top)
        
        ax.legend(loc='best', frameon=True)
        
        fig.savefig(output_dir / filename, format='pdf', transparent=True)
        plt.close(fig)
        print(f"Generato: {filename}")

    # Generate CSV Summary Table
    table_path = output_dir / 'summary_table.csv'
    with open(table_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Group', 'Mean_RMSE', 'Std_RMSE', 'Mean_MAE', 'Std_MAE', 'Mean_MAX', 'Std_MAX'])
        for row in table_data:
            writer.writerow([
                row['Metric'], 
                row['Group'], 
                f"{row['Mean_RMSE']:.4f}", 
                f"{row['Std_RMSE']:.4f}",
                f"{row['Mean_MAE']:.4f}",
                f"{row['Std_MAE']:.4f}",
                f"{row['Mean_MAX']:.4f}",
                f"{row['Std_MAX']:.4f}"
            ])
            
    print(f"Generata tabella riassuntiva: {table_path.name}")
    
    # Generate Summary Bar Chart
    print(f"Generazione del Bar Chart riassuntivo per la metrica {error_metric}...")
    
    unique_metrics = []
    for row in table_data:
        if row['Metric'] not in unique_metrics:
            unique_metrics.append(row['Metric'])
            
    ppo_means = []
    ppo_stds = []
    sppo_means = []
    sppo_stds = []
    
    mean_key = f'Mean_{error_metric.upper()}'
    std_key = f'Std_{error_metric.upper()}'
    
    for m in unique_metrics:
        ppo_val = next((item for item in table_data if item['Metric'] == m and item['Group'] == 'PPO'), None)
        sppo_val = next((item for item in table_data if item['Metric'] == m and item['Group'] == 'SPPO'), None)
        
        ppo_means.append(ppo_val[mean_key] if ppo_val else 0.0)
        ppo_stds.append(ppo_val[std_key] if ppo_val and ppo_val[std_key] > 0 else 0.0)
        sppo_means.append(sppo_val[mean_key] if sppo_val else 0.0)
        sppo_stds.append(sppo_val[std_key] if sppo_val and sppo_val[std_key] > 0 else 0.0)
        
    x = np.arange(len(unique_metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ppo_has_err = any(s > 0 for s in ppo_stds)
    sppo_has_err = any(s > 0 for s in sppo_stds)
    
    rects1 = ax.bar(x - width/2, ppo_means, width, 
                    yerr=ppo_stds if ppo_has_err else None, 
                    label='PPO', color=c_ppo, capsize=5, alpha=0.8, edgecolor='black')
    rects2 = ax.bar(x + width/2, sppo_means, width, 
                    yerr=sppo_stds if sppo_has_err else None, 
                    label='SPPO', color=c_sppo, capsize=5, alpha=0.8, edgecolor='black')
    
    ax.set_ylabel(rf'${error_metric.upper()}$ Value')
    ax.set_title(rf'$\mathrm{{Comparison\ ({error_metric.upper()})\ -\ PPO\ vs\ SPPO}}$', fontweight='bold')
    ax.set_xticks(x)
    
    # Clean up metric names for better x-axis readability
    clean_metrics = [m.replace(" Error ", " ") for m in unique_metrics]
    ax.set_xticklabels(clean_metrics, rotation=45, ha='right')
    
    ax.legend(loc='best', frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    bar_filename = f'summary_barchart_{error_metric.lower()}.pdf'
    fig.savefig(output_dir / bar_filename, format='pdf', transparent=True)
    plt.close(fig)
    print(f"Generato: {bar_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export combined telemetry plots for a hovering campaign (PPO vs SPPO).")
    parser.add_argument("--input", type=str, required=True, help="Cartella contenente tutte le run (es. plots/hovering)")
    parser.add_argument("--metric", type=str, choices=['RMSE', 'MAE'], default='RMSE', help="Evaluation metric to display in legends (RMSE or MAE)")
    parser.add_argument("--t-crop", type=float, default=None, help="Maximum time to evaluate (e.g., 10.0 for cropping 20s flights)")
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = input_dir / "campaign_results"
    
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Errore: Cartella di input '{input_dir}' non trovata.")
        exit(1)
        
    export_campaign_plots(input_dir, output_dir, args.metric, args.t_crop)