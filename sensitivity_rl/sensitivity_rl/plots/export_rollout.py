#!/usr/bin/env python3
"""Plot the nominal and perturbed rollouts saved in ``rollout_data.npz``: 3D trajectories, tracking errors and inputs, as PDF or SVG figures."""

import argparse 
import numpy as np 
from pathlib import Path 
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 

def process_error_data(errors): 
    """Mean and standard deviation over rollouts of error series of different length.

    Shorter series are padded with their last value.

    Args:
        errors (list[numpy.ndarray]): One error series per rollout.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: Time steps, mean and standard deviation;
        ``(None, None, None)`` when the list is empty.
    """
    if not errors: return None, None, None 
    max_len = max(len(e) for e in errors) 
    padded = np.array([np.pad(e, (0, max_len - len(e)), mode='edge') for e in errors]) 
    mean = np.mean(padded, axis=0).flatten() 
    std = np.std(padded, axis=0).flatten() 
    t = np.arange(max_len) 
    return t, mean, std 

def generate_plots(data_dict, output_dir: Path, base_name="plot", nominal_dict=None, ext="svg"): 
    """Save the 3D trajectory, error and input figures of a set of rollouts.

    Args:
        data_dict (dict): Rollout data (``state``, ``actions``, targets) of one condition.
        output_dir (pathlib.Path): Output folder.
        base_name (str): Prefix of the file names.
        nominal_dict (dict, optional): Nominal rollouts, drawn as an overlay on the perturbed ones.
        ext (str): File format, ``svg`` or ``pdf``.
    """
    # ========================================== 
    # Plot style for the thesis figures
    # ========================================== 
    plt.rcParams.update({ 
        "text.usetex": False, 
        "mathtext.fontset": "cm", 
        "font.family": "serif", 
        "font.serif": ["cmr10", "STIXGeneral", "Times New Roman", "DejaVu Serif"], 
        "axes.formatter.use_mathtext": True, 
        "font.size": 14, 
        "axes.labelsize": 16, 
        "legend.fontsize": 15, 
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

    state_data = data_dict["state"] 
    actions_data = data_dict["actions"] 
    target_pos_data = data_dict.get("target_pos") 
    target_att_data = data_dict.get("target_attitude") 

    num_series = state_data.shape[0] 
    is_comparison = nominal_dict is not None

    # Colours
    traj_color = '#1f77b4'         # Blue (nominal, or perturbed bundle)
    traj_color_light = '#9ecae1'   # Light blue for the perturbed bundle in 3D
    nom_color = '#ff7f0e'          # Orange for the nominal overlay
    start_color = '#2ca02c'  
    target_color = '#d62728' 

    fig_3d = plt.figure(figsize=(8, 8)) 
    ax_path = fig_3d.add_subplot(111, projection="3d") 
    ax_path.xaxis.pane.fill = False; ax_path.yaxis.pane.fill = False; ax_path.zaxis.pane.fill = False 
     
    fig_err, (ax_dist_err, ax_att_err) = plt.subplots(2, 1, figsize=(9, 8)) 
    fig_act, axes_act = plt.subplots(2, 3, figsize=(20, 8)) 
    axes_act_flat = axes_act.flatten() 
    labels_act = [rf'$f_{i+1}$ [N]' for i in range(6)] 

    all_dist_errors, all_att_errors = [], [] 

    # 1. Nominal overlay
    if is_comparison:
        nom_state = nominal_dict["state"]
        nom_act = nominal_dict["actions"]
        for i in range(nom_state.shape[0]):
            pos = nom_state[i, :, :3]
            actions = nom_act[i]
            
            lbl_nom = r'$\mathrm{Nominal}$' if i == 0 else None
            ax_path.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=nom_color, linewidth=2.5, zorder=10, label=lbl_nom)
            ax_path.scatter(pos[0, 0], pos[0, 1], pos[0, 2], marker="o", color=nom_color, s=50, zorder=12)

            if actions.size > 0:
                t_steps = np.arange(actions.shape[0])
                for idx, ax in enumerate(axes_act_flat):
                    if idx < actions.shape[1]:
                        ax.plot(t_steps, actions[:, idx], color=nom_color, alpha=1.0, linewidth=2.0, zorder=10)

    # 2. Main data
    for i in range(num_series): 
        state = state_data[i] 
        pos = state[:, :3] 
        quats = state[:, 3:7] 

        if is_comparison:
            c_3d = traj_color_light
            lw_3d = 1.0
            zo_3d = 1
            lbl_3d = r'$\mathrm{Perturbed}$' if i == 0 else None
            c_act = traj_color
            alpha_act = 0.2
            lw_act = 1.0
        else:
            c_3d = traj_color
            lw_3d = 1.5
            zo_3d = 5
            lbl_3d = r'$\mathrm{Trajectory}$' if i == 0 else None
            c_act = traj_color
            alpha_act = 0.4
            lw_act = 1.5

        label_start = r'$\mathrm{Start}$' if i == 0 else None 
        label_target = r'$\mathrm{Target}$' if i == 0 else None 

        ax_path.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=c_3d, linewidth=lw_3d, zorder=zo_3d, label=lbl_3d) 
        ax_path.scatter(pos[0, 0], pos[0, 1], pos[0, 2], marker="o", color=start_color, s=40, zorder=11, label=label_start)  

        if target_pos_data is not None: 
            target = target_pos_data[i] 
            if target.ndim >= 2: target = target[0] 
            ax_path.scatter(target[0], target[1], target[2], marker="X", color=target_color, s=100, zorder=11, label=label_target) 
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
                    ax.plot(t_steps, actions[:, idx], color=c_act, alpha=alpha_act, linewidth=lw_act, zorder=zo_3d)
                    
                    if is_comparison and i == 0 and idx == 0:
                        ax.plot([], [], color=nom_color, linewidth=2.0, label=r'$\mathrm{Nominal}$')
                        ax.plot([], [], color=c_act, alpha=0.5, linewidth=1.0, label=r'$\mathrm{Perturbed}$')
                        ax.legend(loc='upper right', fontsize=12)

    # --- Errors ---
    t_pos, mean_pos, std_pos = process_error_data(all_dist_errors) 
    if t_pos is not None: 
        ax_dist_err.plot(t_pos, mean_pos, color="blue", linewidth=2.5, label=r'$\bar{e}_p \pm \sigma$') 
        ax_dist_err.fill_between(t_pos, mean_pos - std_pos, mean_pos + std_pos, color="blue", alpha=0.2) 
     
    t_att, mean_att, std_att = process_error_data(all_att_errors) 
    if t_att is not None: 
        ax_att_err.plot(t_att, mean_att, color="red", linewidth=2.5, label=r'$\bar{e}_a \pm \sigma$') 
        ax_att_err.fill_between(t_att, mean_att - std_att, mean_att + std_att, color="red", alpha=0.2) 

    # --- 3D formatting ---
    ax_path.set(xlabel=r'$x$ [m]', ylabel=r'$y$ [m]', zlabel=r'$z$ [m]') 
    handles, labels = ax_path.get_legend_handles_labels() 
    if labels: 
        ax_path.legend(loc='best', frameon=True, edgecolor='black') 
    fig_3d.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05) 

    # --- Error formatting ---
    ax_dist_err.set_ylabel(r'$e_p$ [m]') 
    handles, labels = ax_dist_err.get_legend_handles_labels() 
    if labels: 
        ax_dist_err.legend(loc='upper right', frameon=True, edgecolor='black', fontsize=26) 
    ax_dist_err.tick_params(labelbottom=False)  

    ax_att_err.set(ylabel=r'$e_a$ [rad]', xlabel=r'$t$ [steps]') 
    handles, labels = ax_att_err.get_legend_handles_labels() 
    if labels: 
        ax_att_err.legend(loc='upper right', frameon=True, edgecolor='black', fontsize=26) 
    fig_err.align_ylabels() 
    fig_err.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.2) 

    # --- Input formatting ---
    for idx, ax in enumerate(axes_act_flat): 
        ax.set_ylabel(labels_act[idx]) 
        ax.set_ylim(0, 12) 
        if idx >= 3:  
            ax.set_xlabel(r'$t$ [steps]') 
        else:  
            ax.tick_params(labelbottom=False) 
    fig_act.align_ylabels() 
    fig_act.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.1, hspace=0.3, wspace=0.3) 

    # Save in the selected format
    output_dir.mkdir(parents=True, exist_ok=True) 
    fig_3d.savefig(output_dir / f"{base_name}_3d.{ext}", format=ext, transparent=False) 
    fig_err.savefig(output_dir / f"{base_name}_errors.{ext}", format=ext, transparent=True) 
    fig_act.savefig(output_dir / f"{base_name}_inputs.{ext}", format=ext, transparent=True) 

    plt.close('all') 
    print(f"[{base_name.upper()}] Plots saved in: {output_dir}") 

def extract_data_dict(data, suffix):
    """Nominal or perturbed data from the unified ``rollout_data.npz`` file.

    Args:
        data (numpy.lib.npyio.NpzFile): Content of the file.
        suffix (str): ``nom`` or ``pert``.

    Returns:
        dict | None: The arrays of that condition, or ``None`` if absent.
    """
    if f"state_{suffix}" not in data:
        return None
    res = {
        "state": data[f"state_{suffix}"],
        "actions": data[f"actions_{suffix}"]
    }
    if f"target_pos_{suffix}" in data:
        res["target_pos"] = data[f"target_pos_{suffix}"]
    if f"target_att_{suffix}" in data:
        res["target_attitude"] = data[f"target_att_{suffix}"]
    return res

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Generate PDF/SVG plots from unificated .npz rollout data.") 
    parser.add_argument("--data", type=str, required=True, help="Path to the rollout_data.npz file") 
    parser.add_argument("--format", type=str, choices=['svg', 'pdf'], default='svg', help="Figure format (default: svg)")
    args = parser.parse_args() 

    data_path = Path(args.data) 
    if not data_path.exists(): 
        print(f"Error: file {data_path} not found.") 
        exit(1) 

    print(f"Loading unified data from {data_path}...") 
    main_data = np.load(data_path, allow_pickle=True) 
    
    base_outdir = data_path.parent / "exported_plots"
    
    # Nominal data
    nom_dict = extract_data_dict(main_data, "nom")
    # Perturbed data
    pert_dict = extract_data_dict(main_data, "pert")
    
    if nom_dict is not None:
        out_nom = base_outdir / "nominal"
        print("Plotting the nominal rollouts...")
        generate_plots(nom_dict, out_nom, base_name="nominal", nominal_dict=None, ext=args.format)
    else:
        print("Nominal data not found in the .npz file.")

    if pert_dict is not None:
        out_pert = base_outdir / "perturbed"
        print("Plotting the perturbed rollouts...")
        # The nominal dict draws the orange overlay in the perturbed plots
        generate_plots(pert_dict, out_pert, base_name="perturbed", nominal_dict=nom_dict, ext=args.format)
    else:
        print("Perturbed data not found in the .npz file.")
        