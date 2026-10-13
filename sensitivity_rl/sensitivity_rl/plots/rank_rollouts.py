"""
Rank sweep results based on FINAL rollout trajectories (MsgPack files).

This script reads the actual rollout data generated after training, 
giving you the true deterministic performance at the final timestep.
It splits the evaluation into two distinct rankings: Nominal and Perturbed.
Works gracefully even if the sweep is currently running and experiments are incomplete.

USAGE:
    python rank_rollouts.py --sweep_dir runs/latest --top_n 10
"""

import argparse
import json
from pathlib import Path
import numpy as np
from flax import serialization

def get_final_errors(traj_data):
    """Position and attitude error at the last time step of the trajectory."""
    # Safety check: the file may be partial and miss some keys
    if not all(k in traj_data for k in ("state", "target_pos", "target_attitude")):
        raise ValueError("Missing keys in the trajectory data")

    state = np.asarray(traj_data["state"])
    target_pos = np.asarray(traj_data["target_pos"])
    target_att = np.asarray(traj_data["target_attitude"])

    # Controllo di sicurezza sulle dimensioni
    if len(state) == 0:
        raise ValueError("The state array is empty")

    final_pos = state[-1, :3]
    final_quat = state[-1, 3:7]
    
    if target_pos.ndim > 1: 
        target_pos = target_pos[-1]
    if target_att.ndim > 1: 
        target_att = target_att[-1]
        
    pos_err = np.linalg.norm(final_pos - target_pos)
    
    dot = np.sum(final_quat * target_att)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    att_err = 2.0 * np.arccos(dot)
    
    return float(pos_err), float(att_err)

def analyze_rollouts(sweep_dir: Path, top_n: int = 10):
    if not sweep_dir.exists():
        print(f"Error: Directory {sweep_dir} does not exist.")
        return

    print(f"\nAnalizzando le traiettorie di rollout in: {sweep_dir}")
    
    results = []
    skipped_exps = 0
    processed_exps = 0

    for exp_dir in sweep_dir.iterdir():
        if not exp_dir.is_dir() or not exp_dir.name.startswith("exp_"):
            continue

        rollouts_dir = exp_dir / "rollouts"
        
        # Skip if the rollouts folder does not exist yet
        if not rollouts_dir.exists():
            skipped_exps += 1
            continue

        nom_pos_errs, nom_att_errs = [], []
        pert_pos_errs, pert_att_errs = [], []

        msgpack_files = list(rollouts_dir.glob("*.msgpack"))
        
        # Skip if the folder exists but has no files
        if not msgpack_files:
            skipped_exps += 1
            continue

        for msgpack_file in msgpack_files:
            try:
                # Check that the file is not empty (0 bytes) before opening it
                if msgpack_file.stat().st_size == 0:
                    continue

                with open(msgpack_file, "rb") as f:
                    traj_data = serialization.msgpack_restore(f.read())
                
                pos_err, att_err = get_final_errors(traj_data)

                # Nominal or perturbed
                if "sample_" in msgpack_file.name and "sample_00" not in msgpack_file.name:
                    pert_pos_errs.append(pos_err)
                    pert_att_errs.append(att_err)
                else:
                    nom_pos_errs.append(pos_err)
                    nom_att_errs.append(att_err)
            except Exception:
                # Ignore the error: it happens for files still being written (corrupted/incomplete)
                pass

        # Skip the experiment if no valid file could be read
        if not nom_pos_errs and not pert_pos_errs:
            skipped_exps += 1
            continue

        # Averages. An empty list gets infinity, so it is ignored in the ranking
        nom_pos_mean = np.mean(nom_pos_errs) if nom_pos_errs else float('inf')
        nom_att_mean = np.mean(nom_att_errs) if nom_att_errs else float('inf')
        
        pert_pos_mean = np.mean(pert_pos_errs) if pert_pos_errs else float('inf')
        pert_att_mean = np.mean(pert_att_errs) if pert_att_errs else float('inf')

        score_nom = nom_pos_mean + nom_att_mean
        score_pert = pert_pos_mean + pert_att_mean

        seed = "N/A"
        for part in exp_dir.name.split("_"):
            if part.startswith("seed"):
                seed = part.replace("seed", "")

        results.append({
            "exp_name": exp_dir.name,
            "seed": seed,
            "nom_pos": nom_pos_mean,
            "nom_att": nom_att_mean,
            "score_nom": score_nom,
            "pert_pos": pert_pos_mean,
            "pert_att": pert_att_mean,
            "score_pert": score_pert
        })
        
        processed_exps += 1

    if not results:
        print(f"\nNo completed rollout data found. {skipped_exps} experiments skipped because incomplete.")
        return

    print(f"\n[INFO] Processed {processed_exps} experiments. Skipped {skipped_exps} running or incomplete experiments.")

    # --- CLASSIFICA NOMINALE ---
    valid_nom = [r for r in results if r["score_nom"] != float('inf') and not np.isnan(r["score_nom"])]
    valid_nom.sort(key=lambda x: x["score_nom"])
    top_nom = valid_nom[:top_n]

    if top_nom:
        print("\n" + "=" * 110)
        print(f"TOP {len(top_nom)} RUNS: AMBIENTE NOMINALE (Ideale)")
        print("=" * 110)
        print(f"{'Experiment Name':<55} | {'Seed':<5} | {'Pos [m]':<10} | {'Att [rad]':<10} | {'SCORE':<10}")
        print("-" * 110)

        for res in top_nom:
            exp_name = res['exp_name']
            if len(exp_name) > 52: exp_name = exp_name[:49] + "..."
            print(f"{exp_name:<55} | {res['seed']:<5} | "
                  f"{res['nom_pos']:<10.4f} | {res['nom_att']:<10.4f} | "
                  f"{res['score_nom']:<10.4f}")
    else:
        print("\nNo valid data for the nominal environment.")

    # --- CLASSIFICA PERTURBATA ---
    valid_pert = [r for r in results if r["score_pert"] != float('inf') and not np.isnan(r["score_pert"])]
    valid_pert.sort(key=lambda x: x["score_pert"])
    top_pert = valid_pert[:top_n]

    if top_pert:
        print("\n" + "=" * 110)
        print(f"TOP {len(top_pert)} RUNS: AMBIENTE PERTURBATO (Robustezza)")
        print("=" * 110)
        print(f"{'Experiment Name':<55} | {'Seed':<5} | {'Pos [m]':<10} | {'Att [rad]':<10} | {'SCORE':<10}")
        print("-" * 110)

        for res in top_pert:
            exp_name = res['exp_name']
            if len(exp_name) > 52: exp_name = exp_name[:49] + "..."
            print(f"{exp_name:<55} | {res['seed']:<5} | "
                  f"{res['pert_pos']:<10.4f} | {res['pert_att']:<10.4f} | "
                  f"{res['score_pert']:<10.4f}")
    else:
        print("\nNo valid data for the perturbed environment.")

    print("=" * 110 + "\n")

    # Ordinamento finale per JSON
    def sort_key(x):
        val = x["score_nom"]
        return val if (val != float('inf') and not np.isnan(val)) else 1e9

    results.sort(key=sort_key)

    output_file = sweep_dir / "best_rollout_rankings.json"
    
    clean_results = []
    for r in results:
        clean_r = r.copy()
        for k in ["nom_pos", "nom_att", "score_nom", "pert_pos", "pert_att", "score_pert"]:
            val = clean_r[k]
            if val == float('inf') or (isinstance(val, float) and np.isnan(val)):
                clean_r[k] = None
        clean_results.append(clean_r)

    with open(output_file, "w") as f:
        json.dump(clean_results, f, indent=2)
    
    print(f"Sorted data saved in: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank sweep results by actual rollout trajectories.")
    parser.add_argument("--sweep_dir", type=Path, required=True, help="Path to sweep directory")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top experiments to display")
    args = parser.parse_args()
    
    analyze_rollouts(args.sweep_dir, args.top_n)