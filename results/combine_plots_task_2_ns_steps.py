"""
Combine NS depth comparison plots from CSV files.
Reads ns_0.csv, ns_1.csv, ns_3.csv, ns_5.csv, ns_7.csv and creates a combined plot.
"""

import csv
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def load_csv_data(csv_path: Path):
    """Load epoch and lambda_max data from CSV file."""
    epochs = []
    lambda_max_values = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            lambda_max_values.append(float(row['lambda_max']))
    
    return epochs, lambda_max_values


def combine_ns_depth_plots(csv_dir: Path, save_path: Path):
    """Combine all NS depth CSV files into one plot."""
    ns_depths = [0, 1, 3, 5, 7]
    colors = ['red', 'orange', 'blue', 'green', 'purple']
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for ns_depth, color in zip(ns_depths, colors):
        csv_path = csv_dir / f"ns_{ns_depth}.csv"
        
        if not csv_path.exists():
            print(f"WARNING: {csv_path.name} not found, skipping...")
            continue
        
        epochs, lambda_max_values = load_csv_data(csv_path)
        
        ax.plot(
            epochs,
            lambda_max_values,
            linewidth=3.5,
            label=f"NS depth = {ns_depth}",
            color=color,
            marker='o',
            markersize=5
        )
        print(f"Loaded {len(epochs)} data points from {csv_path.name}")
    
    ax.set_xlabel("Epoch", fontsize=16, fontweight='medium')
    ax.set_ylabel("λ_max(H) (Largest Hessian Eigenvalue)", fontsize=16, fontweight='medium')
    ax.set_title("NS Depth Impact on Sharpness (Full Preset)", fontsize=18, fontweight="bold")
    ax.legend(fontsize=14, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nCombined plot saved to: {save_path}")


if __name__ == "__main__":
    # CSV directory
    csv_dir = Path("/Users/maximilian/ETH/Master/3_Semester/Deep Learning/muon/results/task2_new")
    
    # Output plot path
    save_path = csv_dir / "ns_depth_comparison.png"
    
    print("=" * 80)
    print("Combining NS Depth Comparison Plots")
    print("=" * 80)
    print(f"CSV directory: {csv_dir}")
    print(f"Output plot: {save_path}")
    print("=" * 80)
    
    combine_ns_depth_plots(csv_dir, save_path)
    
    print("=" * 80)