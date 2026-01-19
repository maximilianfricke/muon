"""
Task 2: NS Depth Comparison for Full Muon Preset

This script trains the "full" Muon preset with weight_decay=0.0
for different NS depths [0, 1, 3, 5, 7] and saves CSV files
for lambda_max data to be used for plotting.
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
        
        # Extract epochs and values
        if "lambda_max_epochs" in history and history["lambda_max_epochs"]:
            epochs = [
                epoch for val, epoch in zip(history["lambda_max"], history["lambda_max_epochs"])
                if val is not None and not (isinstance(val, float) and np.isnan(val))
            ]
            lambda_max_values = [
                val for val, epoch in zip(history["lambda_max"], history["lambda_max_epochs"])
                if val is not None and not (isinstance(val, float) and np.isnan(val))
            ]
        else:
            epochs = [i for i, val in enumerate(history["lambda_max"]) 
                     if val is not None and not (isinstance(val, float) and np.isnan(val))]
            lambda_max_values = [val for val in history["lambda_max"] 
                                if val is not None and not (isinstance(val, float) and np.isnan(val))]
        
        return {
            "epoch": epochs,
            "lambda_max": lambda_max_values,
        }
    except Exception as e:
        print(f"  ERROR extracting data from {result_dir}: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_csv(csv_data: Dict[str, list], csv_path: Path):
    """Save CSV file with epoch and lambda_max."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lambda_max"])
        for epoch, lam_max in zip(csv_data["epoch"], csv_data["lambda_max"]):
            writer.writerow([epoch, lam_max])
    
    print(f"  Saved CSV: {csv_path}")


def run_ns_depth_training(base_config: Dict[str, Any], ns_depth: int, results_dir: Path, csv_dir: Path) -> bool:
    """Run training for a specific NS depth and save CSV."""
    # Create Muon config for "full" preset with weight_decay=0.0
    muon_base = base_config.get("muon_base_config", {})
    muon_config = {
        "lr": muon_base.get("lr", 0.00095),
        "momentum": muon_base.get("momentum", 0.95),
        "ns_depth": ns_depth,
        "use_rms": True,  # "full" preset
        "use_orthogonalization": True if ns_depth > 0 else False,  # "full" preset
        "weight_decay": 0.0,  # Fixed at 0.0
        "adamw_lr": muon_base.get("adamw_lr", 0.001),
    }
    
    run_suffix = f"full_ns{ns_depth}_wd0p0"
    
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
    print(f"Running NS Depth Comparison: NS depth = {ns_depth}")
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
                csv_path = csv_dir / f"ns_{ns_depth}.csv"
                save_csv(csv_data, csv_path)
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
        print(f"ERROR running NS depth {ns_depth}: {e}")
        success = False
    finally:
        if temp_config_path.exists():
            temp_config_path.unlink()
    
    return success


def main():
    parser = argparse.ArgumentParser(description="Task 2: NS Depth Comparison (Full Preset)")
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
    print("Task 2: NS Depth Comparison (Full Preset, Weight Decay=0.0)")
    print("=" * 80)
    print(f"Model: {config['model']['type']}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"NS Depths: [0, 1, 3, 5, 7]")
    print(f"Results directory: {results_dir}")
    print(f"CSV files will be saved to: {csv_dir}")
    print("=" * 80)
    
    # NS depths to test
    ns_depths = [0, 1, 3, 5, 7]
    
    results = {}
    for ns_depth in ns_depths:
        ok = run_ns_depth_training(config, ns_depth, results_dir, csv_dir)
        results[ns_depth] = "success" if ok else "failed"
    
    print("\n" + "=" * 80)
    print("NS Depth Comparison Summary")
    print("=" * 80)
    failed = [ns for ns, status in results.items() if status != "success"]
    print(f"Success: {len(results) - len(failed)}/{len(results)}")
    if failed:
        print("Failed NS depths:")
        for ns in failed:
            print(f"  - NS depth {ns}")
    else:
        print("All NS depth trainings completed successfully!")
        print(f"\nCSV files saved in: {csv_dir}")
        print("Files created:")
        for ns in ns_depths:
            csv_file = csv_dir / f"ns_{ns}.csv"
            if csv_file.exists():
                print(f"  ✓ {csv_file.name}")
            else:
                print(f"  ✗ {csv_file.name} (missing)")
    print("=" * 80)


if __name__ == "__main__":
    main()