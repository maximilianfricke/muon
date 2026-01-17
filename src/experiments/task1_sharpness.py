"""
Task 1: Track λ_max (largest Hessian eigenvalue) for Muon, SGD, AdamW
during full-batch training.

This script runs the same training with different optimizers and compares
the λ_max trajectories. It uses basic_training.py internally.
"""

import yaml
import argparse
from pathlib import Path
import subprocess
import sys
from typing import Dict
import copy


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_training_with_optimizer(base_config: Dict, optimizer_type: str, optimizer_config: Dict):
    config = copy.deepcopy(base_config)
    config["optimizer"] = {
        "type": optimizer_type,
        "config": optimizer_config
    }
    
    base_name = base_config["experiment"]["name"]
    config["experiment"]["name"] = base_name
    
    temp_config_path = Path(f"configs/temp_task1_{optimizer_type}.yaml")
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\n{'='*60}")
    print(f"Running training with {optimizer_type.upper()} optimizer")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.experiments.basic_training", "--config", str(temp_config_path)],
            check=True
        )
        success = result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running training with {optimizer_type}: {e}")
        success = False
    finally:
        if temp_config_path.exists():
            temp_config_path.unlink()
    
    return success


def main():
    """
    Runs training with Muon, SGD, and AdamW, tracking λ_max for each.
    """
    parser = argparse.ArgumentParser(description="Task 1: Sharpness Tracking")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/task1_sharpness.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    optimizers_config = config.get("optimizers", {})
    
    if not optimizers_config:
        optimizers_config = {
            "muon": {
                "lr": 0.02,
                "momentum": 0.95,
                "ns_depth": 5,
                "use_rms": False,
                "use_orthogonalization": True,
                "weight_decay": 0.0,
                "adamw_lr": 0.001
            },
            "sgd": {
                "lr": 0.01,
                "momentum": 0.0,
                "weight_decay": 0.0
            },
            "adamw": {
                "lr": 0.001,
                "weight_decay": 0.01
            }
        }
    
    print("="*60)
    print("Task 1: Sharpness Tracking Experiment")
    print("="*60)
    print(f"Model: {config['model']['type']}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Optimizers to test: {', '.join(optimizers_config.keys())}")
    print("="*60)
    
    results = {}
    for opt_name, opt_config in optimizers_config.items():
        success = run_training_with_optimizer(config, opt_name, opt_config)
        results[opt_name] = "success" if success else "failed"
    
    print("\n" + "="*60)
    print("Task 1 Summary")
    print("="*60)
    for opt_name, status in results.items():
        print(f"  {opt_name.upper()}: {status}")
    print("\nAll runs logged to wandb project 'muon'")
    print("Filter by: task1_sharpness_* to see all Task 1 runs")
    print("="*60)


if __name__ == "__main__":
    main()
