#!/usr/bin/env python3
import argparse
import csv
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_summary_table(csv_path: Path):
    """Reads a summary CSV table and organizes data by metric and group."""
    data = {}
    with open(csv_path, mode='r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = row['Metric']
            group = row['Group']
            if metric not in data:
                data[metric] = {}
            data[metric][group] = {
                'Mean_RMSE': float(row['Mean_RMSE']),
                'Std_RMSE': float(row['Std_RMSE']),
                'Mean_MAE': float(row['Mean_MAE']),
                'Std_MAE': float(row['Std_MAE'])
            }
    return data

def generate_comparison_plots(nominal_path: Path, perturbed_path: Path, output_dir: Path, error_metric: str):
    # Matplotlib professional style configuration
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

    print(f"Caricamento dati nominali da: {nominal_path.name}")
    nominal_data = load_summary_table(nominal_path)
    
    print(f"Caricamento dati perturbati da: {perturbed_path.name}")
    perturbed_data = load_summary_table(perturbed_path)
    
    # Find common metrics present in both files
    common_metrics = [m for m in nominal_data if m in perturbed_data]
    if not common_metrics:
        print("Errore: Nessuna metrica in comune trovata tra i due file CSV.")
        return
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mean_key = f'Mean_{error_metric.upper()}'
    std_key = f'Std_{error_metric.upper()}'
    
    # Consistent color scheme: Nominal (Blue), Perturbed (Red)
    c_nominal = '#1f77b4'
    c_perturbed = '#d62728'
    
    groups = ['PPO', 'SPPO']
    x = np.arange(len(groups))  # Positions for PPO and SPPO on X axis
    width = 0.35  # Bar width
    
    for metric in common_metrics:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Extract values for Nominal case
        nom_ppo = nominal_data[metric].get('PPO', None)
        nom_sppo = nominal_data[metric].get('SPPO', None)
        
        # Extract values for Perturbed case
        pert_ppo = perturbed_data[metric].get('PPO', None)
        pert_sppo = perturbed_data[metric].get('SPPO', None)
        
        # Skip if any policy data is missing for this metric
        if not (nom_ppo and nom_sppo and pert_ppo and pert_sppo):
            continue
            
        nominal_means = [nom_ppo[mean_key], nom_sppo[mean_key]]
        nominal_stds = [nom_ppo[std_key], nom_sppo[std_key]]
        
        perturbed_means = [pert_ppo[mean_key], pert_sppo[mean_key]]
        perturbed_stds = [pert_ppo[std_key], pert_sppo[std_key]]
        
        # Check if error bars are needed (std > 0)
        nom_has_err = any(s > 0 for s in nominal_stds)
        pert_has_err = any(s > 0 for s in perturbed_stds)
        
        # Plot Nominal bars
        ax.bar(x - width/2, nominal_means, width, 
               yerr=nominal_stds if nom_has_err else None,
               label='Nominal', color=c_nominal, capsize=6, alpha=0.85, edgecolor='black', lw=1.2)
               
        # Plot Perturbed bars
        ax.bar(x + width/2, perturbed_means, width, 
               yerr=perturbed_stds if pert_has_err else None,
               label='Perturbed', color=c_perturbed, capsize=6, alpha=0.85, edgecolor='black', lw=1.2)
        
        # Set titles and labels using LaTeX syntax style
        formatted_title = metric.replace(" ", r"\ ")
        ax.set_title(rf'$\mathrm{{{formatted_title}\ -\ Campaign\ Comparison}}$', fontweight='bold')
        ax.set_ylabel(rf'${error_metric.upper()}$ Value')
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        
        ax.legend(loc='best', frameon=True)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Save individual plot
        clean_filename = metric.lower().replace(" ", "_") + "_comparison.pdf"
        fig.savefig(output_dir / clean_filename, format='pdf', transparent=True)
        plt.close(fig)
        print(f"Generato: {clean_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Nominal vs Perturbed campaigns using bar charts with error bars.")
    parser.add_argument("--nominal", type=str, required=True, help="Path to the nominal summary_table.csv")
    parser.add_argument("--perturbed", type=str, required=True, help="Path to the perturbed summary_table.csv")
    parser.add_argument("--metric", type=str, choices=['RMSE', 'MAE'], default='RMSE', help="Evaluation metric to parse and plot")
    args = parser.parse_args()
    
    nominal_path = Path(args.nominal)
    perturbed_path = Path(args.perturbed)
    output_dir = perturbed_path.parent / "cross_campaign_comparison"
    
    if not nominal_path.exists() or not perturbed_path.exists():
        print("Errore: Uno o entrambi i file CSV di input non esistono.")
        exit(1)
        
    generate_comparison_plots(nominal_path, perturbed_path, output_dir, args.metric)