"""
Combine Task 2 plots from CSV files.
- NS depth comparison: ns_0.csv, ns_1.csv, ns_3.csv, ns_5.csv, ns_7.csv
- Preset comparison: full.csv, no_ortho.csv, no_rms.csv, none.csv
- Preset ratio comparison: full_ratio.csv, no_ortho_ratio.csv, no_rms_ratio.csv, none_ratio.csv
"""

import csv
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def load_csv_data(csv_path: Path, metric_name: str = "lambda_max"):
    """Load epoch and metric data from CSV file."""
    epochs = []
    values = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            values.append(float(row[metric_name]))
    
    return epochs, values


def combine_ns_depth_plots(csv_dir: Path, save_path: Path):
    """Combine all NS depth CSV files into one plot."""
    ns_depths = [0, 1, 3, 5, 7]
    colors = ['red', 'orange', 'blue', 'green', 'purple']
    
    fig, ax = plt.subplots(figsize=(10, 4.5))
    
    for ns_depth, color in zip(ns_depths, colors):
        csv_path = csv_dir / f"ns_{ns_depth}.csv"
        
        if not csv_path.exists():
            print(f"WARNING: {csv_path.name} not found, skipping...")
            continue
        
        epochs, lambda_max_values = load_csv_data(csv_path, "lambda_max")
        
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


def combine_preset_plots(csv_dir: Path, save_path: Path):
    """Combine all preset CSV files into one plot."""
    presets = ["full", "no_ortho", "no_rms", "none"]
    colors = ['green', 'orange', 'blue', 'red']
    labels = {
        "full": "Full (RMS + Ortho)",
        "no_ortho": "No Ortho (RMS only)",
        "no_rms": "No RMS (Ortho only)",
        "none": "None (baseline)"
    }
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for preset, color in zip(presets, colors):
        csv_path = csv_dir / f"{preset}.csv"
        
        if not csv_path.exists():
            print(f"WARNING: {csv_path.name} not found, skipping...")
            continue
        
        epochs, lambda_max_values = load_csv_data(csv_path, "lambda_max")
        
        ax.plot(
            epochs,
            lambda_max_values,
            linewidth=3.5,
            label=labels[preset],
            color=color,
            marker='o',
            markersize=5
        )
        print(f"Loaded {len(epochs)} data points from {csv_path.name}")
    
    ax.set_xlabel("Epoch", fontsize=16, fontweight='medium')
    ax.set_ylabel("λ_max(H) (Largest Hessian Eigenvalue)", fontsize=16, fontweight='medium')
    ax.set_title("Preset Comparison: Component Impact on Sharpness (NS depth=5)", fontsize=18, fontweight="bold")
    ax.legend(fontsize=14, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nCombined plot saved to: {save_path}")


def combine_preset_ratio_plots(csv_dir: Path, save_path: Path):
    """Combine all preset lambda_eff_ratio CSV files into one plot."""
    presets = ["full", "no_ortho", "no_rms", "none"]
    colors = ['green', 'orange', 'blue', 'red']
    labels = {
        "full": "Full (RMS + Ortho)",
        "no_ortho": "No Ortho (RMS only)",
        "no_rms": "No RMS (Ortho only)",
        "none": "None (baseline)"
    }
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for preset, color in zip(presets, colors):
        csv_path = csv_dir / f"{preset}_ratio.csv"
        
        if not csv_path.exists():
            print(f"WARNING: {csv_path.name} not found, skipping...")
            continue
        
        epochs, ratio_values = load_csv_data(csv_path, "lambda_eff_ratio")
        
        ax.plot(
            epochs,
            ratio_values,
            linewidth=3.5,
            label=labels[preset],
            color=color,
            marker='o',
            markersize=5
        )
        print(f"Loaded {len(epochs)} data points from {csv_path.name}")
    
    # Add horizontal line at ratio = 1 for reference
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Ratio = 1 (no suppression)')
    
    ax.set_xlabel("Epoch", fontsize=16, fontweight='medium')
    ax.set_ylabel("λ_eff_ratio (λ_muon / λ_grad)", fontsize=16, fontweight='medium')
    ax.set_title("Preset Comparison: Curvature Suppression Mechanism (NS depth=5)", fontsize=18, fontweight="bold")
    ax.legend(fontsize=14, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    
    # Add text annotation explaining ratio < 1, positioned on the right
    ax.text(0.55, 0.98, "Ratio < 1: Muon moves in flatter directions\nRatio ≈ 1: No suppression", 
            transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nCombined plot saved to: {save_path}")


if __name__ == "__main__":
    # CSV directory
    csv_dir = Path("/Users/maximilian/ETH/Master/3_Semester/Deep Learning/muon/results/task2_new")
    
    print("=" * 80)
    print("Combining Task 2 Plots")
    print("=" * 80)
    print(f"CSV directory: {csv_dir}")
    print("=" * 80)
    
    # Combine NS depth plots
    print("\n1. Combining NS Depth Comparison Plots...")
    ns_depth_save_path = csv_dir / "ns_depth_comparison.png"
    combine_ns_depth_plots(csv_dir, ns_depth_save_path)
    
    # Combine preset plots (lambda_max)
    print("\n2. Combining Preset Comparison Plots (λ_max)...")
    preset_save_path = csv_dir / "preset_comparison.png"
    combine_preset_plots(csv_dir, preset_save_path)
    
    # Combine preset ratio plots (lambda_eff_ratio)
    print("\n3. Combining Preset Ratio Comparison Plots (λ_eff_ratio)...")
    preset_ratio_save_path = csv_dir / "preset_ratio_comparison.png"
    combine_preset_ratio_plots(csv_dir, preset_ratio_save_path)
    
    print("\n" + "=" * 80)
    print("All plots created successfully!")
    print("=" * 80)