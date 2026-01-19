"""
Task 2: Preset Comparison for Muon

This script trains different Muon presets with ns_depth=5, weight_decay=0.0
and saves CSV files for lambda_max and lambda_eff_ratio data to be used for plotting.

Presets:
- full: RMS + orthogonalization
- no_ortho: RMS only (no orthogonalization)
- no_rms: orthogonalization only (no RMS)
- none: baseline (neither RMS nor orthogonalization)
"""

import torch
import yaml
from pathlib import Path
import argparse
import subprocess
import sys
from typing import Dict, Any, Optional
import copy
import csv
import numpy as np
import re
import time


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_result_directory(base_config: Dict[str, Any], run_suffix: str, results_base_dir: Path) -> Optional[Path]:
    """Find the result directory for a given run suffix."""
    # The run name format is: t2_{run_suffix}_{model}_{dataset}_{optimizer}_{timestamp}
    pattern = f"t2_{run_suffix}_*"
    matching_dirs = list(results_base_dir.glob(pattern))
    
    if matching_dirs:
        # Sort by modification time, get the most recent
        matching_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return matching_dirs[0]
    return None


def extract_csv_data(result_dir: Path) -> Optional[Dict[str, list]]:
    """Extract CSV data from results.pt or checkpoint file."""
    # Try to find results.pt file
    results_files = list(result_dir.glob("*_results.pt"))
    
    # If not found, try checkpoint files
    if not results_files:
        checkpoint_files = list(result_dir.glob("checkpoint_epoch_*.pt"))
        if checkpoint_files:
            # Sort by epoch number and use the latest
            def get_epoch_num(path):
                match = re.search(r'checkpoint_epoch_(\d+)\.pt', path.name)
                return int(match.group(1)) if match else 0
            checkpoint_files.sort(key=get_epoch_num, reverse=True)
            results_files = [checkpoint_files[0]]
    
    if not results_files:
        print(f"  WARNING: No results file found in {result_dir}")
        return None
    
    try:
        data = torch.load(results_files[0], map_location='cpu')
        history = data.get("history", {})
        
        if "lambda_max" not in history or not history["lambda_max"]:
            print(f"  WARNING: No lambda_max data in {result_dir}")
            return None
        
        # Extract lambda_max epochs and values
        if "lambda_max_epochs" in history and history["lambda_max_epochs"]:
            lambda_max_epochs = [
                epoch for val, epoch in zip(history["lambda_max"], history["lambda_max_epochs"])
                if val is not None and not (isinstance(val, float) and np.isnan(val))
            ]
            lambda_max_values = [
                val for val, epoch in zip(history["lambda_max"], history["lambda_max_epochs"])
                if val is not None and not (isinstance(val, float) and np.isnan(val))
            ]
        else:
            lambda_max_epochs = [i for i, val in enumerate(history["lambda_max"]) 
                     if val is not None and not (isinstance(val, float) and np.isnan(val))]
            lambda_max_values = [val for val in history["lambda_max"] 
                                if val is not None and not (isinstance(val, float) and np.isnan(val))]
        
        # Extract lambda_eff_ratio epochs and values
        lambda_eff_ratio_epochs = []
        lambda_eff_ratio_values = []
        if "lambda_eff_ratio" in history and history["lambda_eff_ratio"]:
            if "lambda_eff_epochs" in history and history["lambda_eff_epochs"]:
                lambda_eff_ratio_epochs = [
                    epoch for val, epoch in zip(history["lambda_eff_ratio"], history["lambda_eff_epochs"])
                    if val is not None and not (isinstance(val, float) and np.isnan(val))
                ]
                lambda_eff_ratio_values = [
                    val for val, epoch in zip(history["lambda_eff_ratio"], history["lambda_eff_epochs"])
                    if val is not None and not (isinstance(val, float) and np.isnan(val))
                ]
        
        return {
            "lambda_max_epoch": lambda_max_epochs,
            "lambda_max": lambda_max_values,
            "lambda_eff_ratio_epoch": lambda_eff_ratio_epochs,
            "lambda_eff_ratio": lambda_eff_ratio_values,
        }
    except Exception as e:
        print(f"  ERROR extracting data from {result_dir}: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_csv(csv_data: Dict[str, list], csv_path: Path, metric_name: str = "lambda_max"):
    """Save CSV file with epoch and metric values."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    if metric_name == "lambda_max":
        epochs = csv_data["lambda_max_epoch"]
        values = csv_data["lambda_max"]
    elif metric_name == "lambda_eff_ratio":
        epochs = csv_data["lambda_eff_ratio_epoch"]
        values = csv_data["lambda_eff_ratio"]
    else:
        raise ValueError(f"Unknown metric: {metric_name}")
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", metric_name])
        for epoch, val in zip(epochs, values):
            writer.writerow([epoch, val])
    
    print(f"  Saved CSV: {csv_path}")


def preset_to_config(preset_name: str) -> Dict[str, bool]:
    """Convert preset name to Muon config flags."""
    if preset_name == "full":
        return {"use_rms": True, "use_orthogonalization": True}
    elif preset_name == "no_ortho":
        return {"use_rms": True, "use_orthogonalization": False}
    elif preset_name == "no_rms":
        return {"use_rms": False, "use_orthogonalization": True}
    elif preset_name == "none":
        return {"use_rms": False, "use_orthogonalization": False}
    else:
        raise ValueError(f"Unknown preset: {preset_name}")


def run_preset_training(base_config: Dict[str, Any], preset_name: str, results_dir: Path, csv_dir: Path) -> bool:
    """Run training for a specific preset and save CSV."""
    # Create Muon config for preset with ns_depth=5, weight_decay=0.0
    muon_base = base_config.get("muon_base_config", {})
    preset_flags = preset_to_config(preset_name)
    
    muon_config = {
        "lr": muon_base.get("lr", 0.00095),
        "momentum": muon_base.get("momentum", 0.95),
        "ns_depth": 5,  # Fixed at 5
        "use_rms": preset_flags["use_rms"],
        "use_orthogonalization": preset_flags["use_orthogonalization"],
        "weight_decay": 0.0,  # Fixed at 0.0
        "adamw_lr": muon_base.get("adamw_lr", 0.001),
    }
    
    run_suffix = f"{preset_name}_ns5_wd0p0"
    
    # Create config for this run
    config = copy.deepcopy(base_config)
    config["optimizer"] = {
        "type": "muon",
        "config": muon_config
    }
    config["experiment"]["name"] = f"t2_{run_suffix}"
    
    # Override save directory - basic_training.py uses logging.save_dir
    if "logging" not in config:
        config["logging"] = {}
    config["logging"]["save_dir"] = str(results_dir)
    
    # Disable wandb
    if "wandb" not in config["logging"]:
        config["logging"]["wandb"] = {}
    config["logging"]["wandb"]["enabled"] = False
    
    temp_config_path = Path(f"configs/temp_task2_{run_suffix}.yaml")
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n{'='*80}")
    print(f"Running Preset Comparison: {preset_name}")
    print(f"Muon config: {muon_config}")
    print(f"{'='*80}\n")
    
    try:
        # Run training
        subprocess.run(
            [sys.executable, "-m", "src.experiments.basic_training", "--config", str(temp_config_path)],
            check=True
        )
        success = True
        
        # Wait a bit for file system to sync
        time.sleep(2)
        
        # Extract and save CSV
        result_dir = find_result_directory(config, run_suffix, results_dir)
        if result_dir and result_dir.exists():
            print(f"  Extracting CSV data from {result_dir.name}...")
            csv_data = extract_csv_data(result_dir)
            if csv_data:
                # Save lambda_max CSV
                csv_path_max = csv_dir / f"{preset_name}.csv"
                save_csv(csv_data, csv_path_max, "lambda_max")
                
                # Save lambda_eff_ratio CSV if available
                if csv_data["lambda_eff_ratio"]:
                    csv_path_ratio = csv_dir / f"{preset_name}_ratio.csv"
                    save_csv(csv_data, csv_path_ratio, "lambda_eff_ratio")
                else:
                    print(f"  WARNING: No lambda_eff_ratio data available")
            else:
                print(f"  WARNING: Could not extract CSV data from {result_dir.name}")
        else:
            print(f"  WARNING: Result directory not found for {run_suffix}")
            if result_dir:
                print(f"    Expected: {result_dir}")
            else:
                print(f"    Looking in: {results_dir}")
                print(f"    Pattern: t2_{run_suffix}_*")
                # List what's actually there
                existing_dirs = list(results_dir.glob("t2_*"))
                if existing_dirs:
                    print(f"    Found directories: {[d.name for d in existing_dirs[:5]]}")
            
    except subprocess.CalledProcessError as e:
        print(f"ERROR running preset {preset_name}: {e}")
        success = False
    finally:
        if temp_config_path.exists():
            temp_config_path.unlink()
    
    return success


def main():
    parser = argparse.ArgumentParser(description="Task 2: Preset Comparison (NS depth=5, Weight Decay=0.0)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/task2_ablation.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Set results directory
    results_dir = Path("/Users/maximilian/ETH/Master/3_Semester/Deep Learning/muon/results/task2_new")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV directory (same as results directory)
    csv_dir = results_dir
    
    print("=" * 80)
    print("Task 2: Preset Comparison (NS depth=5, Weight Decay=0.0)")
    print("=" * 80)
    print(f"Model: {config['model']['type']}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Presets: ['full', 'no_ortho', 'no_rms', 'none']")
    print(f"Results directory: {results_dir}")
    print(f"CSV files will be saved to: {csv_dir}")
    print("=" * 80)
    
    # Presets to test
    presets = ["full", "no_ortho", "no_rms", "none"]
    
    results = {}
    for preset_name in presets:
        ok = run_preset_training(config, preset_name, results_dir, csv_dir)
        results[preset_name] = "success" if ok else "failed"
    
    print("\n" + "=" * 80)
    print("Preset Comparison Summary")
    print("=" * 80)
    failed = [preset for preset, status in results.items() if status != "success"]
    print(f"Success: {len(results) - len(failed)}/{len(results)}")
    if failed:
        print("Failed presets:")
        for preset in failed:
            print(f"  - {preset}")
    else:
        print("All preset trainings completed successfully!")
        print(f"\nCSV files saved in: {csv_dir}")
        print("Files created:")
        for preset in presets:
            csv_file = csv_dir / f"{preset}.csv"
            csv_file_ratio = csv_dir / f"{preset}_ratio.csv"
            if csv_file.exists():
                print(f"  ✓ {csv_file.name}")
            else:
                print(f"  ✗ {csv_file.name} (missing)")
            if csv_file_ratio.exists():
                print(f"  ✓ {csv_file_ratio.name}")
            else:
                print(f"  ✗ {csv_file_ratio.name} (missing)")
    print("=" * 80)


if __name__ == "__main__":
    main()