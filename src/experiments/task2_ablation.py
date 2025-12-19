"""
Task 2: Ablate Muon's components (NS depth, RMS, orthogonalization, weight decay).

This script runs multiple Muon-only trainings (each is a separate run) by
calling basic_training.py internally, just like task1_sharpness.py.

Ablations:
- ns_depth: [0, 1, 3, 5, 7] (0 disables orthogonalization)
- use_rms: [true, false]
- use_orthogonalization: [true, false]
- weight_decay: [0.0, 0.01]
- variant presets: full, no_ortho, no_rms, ortho_only, rms_only, none
"""

import torch
import yaml
from pathlib import Path
import argparse
import subprocess
import sys
from typing import Dict, Any, List, Tuple
import copy


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    

def make_variants(cfg: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Build (suffix, muon_config) variants.

    Expected YAML keys:
      ablation:
        ns_depth: [...]
        use_rms: [...]
        use_orthogonalization: [...]
        weight_decay: [...]
        variants: [...]  # optional presets

      muon_base_config:  # defaults shared across all variants
        lr: ...
        momentum: ...
        adamw_lr: ...
    """
    abl = cfg.get("ablation", {})
    base = cfg.get("muon_base_config", {})

    ns_list = abl.get("ns_depth", [0, 1, 3, 5, 7])
    rms_list = abl.get("use_rms", [True, False])
    ortho_list = abl.get("use_orthogonalization", [True, False])
    wd_list = abl.get("weight_decay", [0.0, 0.01])

    # Optional preset variants. If provided, we *use these* instead of raw ortho/rms toggles.
    preset_variants = abl.get("variants", None)

    def preset_to_flags(name: str) -> Dict[str, Any]:
        # Keys must match basic_training.py -> Muon(...) signature
        if name == "full":
            return {"use_rms": True, "use_orthogonalization": True}
        if name == "no_ortho":
            return {"use_rms": True, "use_orthogonalization": False}
        if name == "no_rms":
            return {"use_rms": False, "use_orthogonalization": True}
        if name == "ortho_only":
            return {"use_rms": False, "use_orthogonalization": True}
        if name == "rms_only":
            return {"use_rms": True, "use_orthogonalization": False}
        if name == "none":
            return {"use_rms": False, "use_orthogonalization": False}
        raise ValueError(f"Unknown preset variant: {name}")

    variants: List[Tuple[str, Dict[str, Any]]] = []

    if preset_variants is not None:
        # Preset mode: variants x ns_depth x weight_decay
        for preset_name in preset_variants:
            flags = preset_to_flags(preset_name)
            for ns in ns_list:
                for wd in wd_list:
                    muon_cfg = dict(base)
                    muon_cfg.update(flags)
                    muon_cfg["ns_depth"] = int(ns)
                    muon_cfg["weight_decay"] = float(wd)

                    # ns_depth == 0 => orthogonalization must be off
                    if int(ns) == 0:
                        muon_cfg["use_orthogonalization"] = False

                    wd_str = str(wd).replace(".", "p")
                    suffix = f"{preset_name}_ns{ns}_wd{wd_str}"
                    variants.append((suffix, muon_cfg))
    else:
        # Grid mode: ns_depth x rms x ortho x weight_decay
        for ns in ns_list:
            for use_rms in rms_list:
                for use_ortho in ortho_list:
                    for wd in wd_list:
                        muon_cfg = dict(base)
                        muon_cfg["ns_depth"] = int(ns)
                        muon_cfg["use_rms"] = bool(use_rms)
                        muon_cfg["use_orthogonalization"] = bool(use_ortho)
                        muon_cfg["weight_decay"] = float(wd)

                        if int(ns) == 0:
                            muon_cfg["use_orthogonalization"] = False

                        suffix = f"ns{ns}_rms{1 if use_rms else 0}_ortho{1 if muon_cfg['use_orthogonalization'] else 0}_wd{wd}"
                        variants.append((suffix, muon_cfg))

    return variants


def run_training_with_muon_variant(base_config: Dict[str, Any], run_suffix: str, muon_config: Dict[str, Any]) -> bool:
    """
    Run basic_training.py with optimizer fixed to Muon and given config.
    """

    config = copy.deepcopy(base_config)

    config["optimizer"] = {
        "type": "muon",
        "config": muon_config
    }

    # Keep names short and stable
    config["experiment"]["name"] = f"t2_{run_suffix}"

    temp_config_path = Path(f"configs/temp_task2_{run_suffix}.yaml")
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\n{'='*80}")
    print(f"Running Task 2 variant: {run_suffix}")
    print(f"Muon config: {muon_config}")
    print(f"{'='*80}\n")

    try:
        subprocess.run(
            [sys.executable, "-m", "src.experiments.basic_training", "--config", str(temp_config_path)],
            check=True
        )
        success = True
    except subprocess.CalledProcessError as e:
        print(f"Error running variant {run_suffix}: {e}")
        success = False
    finally:
        if temp_config_path.exists():
            temp_config_path.unlink()

    return success


def main():
    parser = argparse.ArgumentParser(description="Task 2: Muon Ablation Study")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/task2_ablation.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    print("=" * 80)
    print("Task 2: Muon Ablation Study")
    print("=" * 80)
    print(f"Model: {config['model']['type']}")
    print(f"Dataset: {config['dataset']['name']}")
    print("=" * 80)

    variants = make_variants(config)
    print(f"Total variants to run: {len(variants)}")

    results = {}
    for suffix, muon_cfg in variants:
        ok = run_training_with_muon_variant(config, suffix, muon_cfg)
        results[suffix] = "success" if ok else "failed"

    print("\n" + "=" * 80)
    print("Task 2 Summary")
    print("=" * 80)
    failed = [k for k, v in results.items() if v != "success"]
    print(f"Success: {len(results) - len(failed)}/{len(results)}")
    if failed:
        print("Failed variants:")
        for k in failed:
            print(f"  - {k}")

    print("\nAll runs logged to wandb project 'muon'")
    print("Filter by: task2_ablation_ablate_* (or your prefix) in wandb run names")
    print("=" * 80)


if __name__ == "__main__":
    main()
