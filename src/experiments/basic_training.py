"""
Basic training script used for optimizer sanity checks and curvature tracking.
"""

import torch
import torch.nn as nn
import argparse
import yaml
import math
from pathlib import Path
import wandb
from datetime import datetime
import torch.nn.functional as F  

from src.models import MLP, TinyViT
from src.optimizers import Muon
from src.utils.data import load_mnist, load_cifar10
from src.utils.training import train_full_batch, evaluate_model
from src.utils.visualization import visualize_predictions, plot_training_history, plot_confusion_matrix, plot_lambda_max, plot_task2_lambdas
from src.geometry.hessian import compute_lambda_max
from src.geometry.curvature import compute_lambda_grad, compute_lambda_muon


def get_device(device_str: str = "auto"):
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_str)


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(model_config: dict, device: torch.device):
    model_type = model_config["type"]
    config = model_config.get("config", {})
    
    if model_type == "mlp":
        model = MLP(
            input_size=config.get("input_size", 784),
            hidden_sizes=config.get("hidden_sizes", [128, 64]),
            num_classes=config.get("num_classes", 10),
            dropout=config.get("dropout", 0.0)
        )
    elif model_type == "tiny_vit":
        model = TinyViT(
            img_size=config.get("img_size", 32),
            patch_size=config.get("patch_size", 4),
            num_classes=config.get("num_classes", 10),
            embed_dim=config.get("embed_dim", 128),
            depth=config.get("depth", 4),
            num_heads=config.get("num_heads", 4),
            mlp_ratio=config.get("mlp_ratio", 4.0),
            dropout=config.get("dropout", 0.0)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)
    


def create_optimizer(model: nn.Module, optimizer_config: dict):
    optimizer_type = optimizer_config["type"]
    config = optimizer_config.get("config", {})
    optimizers = []
    
    if optimizer_type == "muon":
        muon_params = []
        adamw_params = []
        
        for name, param in model.named_parameters():
            if (param.ndim >= 2 and 
                "output_layer" not in name and 
                "head" not in name and
                "embed" not in name.lower() and
                "bias" not in name):
                muon_params.append(param)
            else:
                adamw_params.append(param)
        
        if len(muon_params) > 0:
            muon_opt = Muon(
                muon_params,
                lr=config.get("lr", 0.02),
                momentum=config.get("momentum", 0.95),
                ns_depth=config.get("ns_depth", 5),
                use_rms=config.get("use_rms", False),
                use_orthogonalization=config.get("use_orthogonalization", True),
                weight_decay=config.get("weight_decay", 0.0)
            )
            optimizers.append(muon_opt)
        
        if len(adamw_params) > 0:
            adamw_opt = torch.optim.AdamW(
                adamw_params,
                lr=config.get("adamw_lr", 1e-3),
                weight_decay=config.get("weight_decay", 0.0)
            )
            optimizers.append(adamw_opt)
        
        total_params = sum(1 for _ in model.parameters())
        covered_params = len(muon_params) + len(adamw_params)
        if total_params != covered_params:
            print(f"Warning: {total_params - covered_params} parameters not assigned to any optimizer")

    elif optimizer_type == "sgd":
        opt = torch.optim.SGD(
            model.parameters(),
            lr=config.get("lr", 0.01),
            momentum=config.get("momentum", 0.0),
            weight_decay=config.get("weight_decay", 0.0)
        )
        optimizers.append(opt)

    elif optimizer_type == "adamw":
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 0.01)
        )
        optimizers.append(opt)

    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizers


def main():
    parser = argparse.ArgumentParser(
        description="Basic training script with config file support"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    args = parser.parse_args()
    
    config = load_config(args.config)
    exp_config = config.get("experiment", {})
    dataset_config = config.get("dataset", {})
    model_config = config.get("model", {})
    optimizer_config = config.get("optimizer", {})
    training_config = config.get("training", {})
    curvature_config = config.get("curvature", {})
    logging_config = config.get("logging", {})
    
    seed = exp_config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    device_str = exp_config.get("device", "auto")
    device = get_device(device_str)
    print(f"Using device: {device}")
    
    save_dir = Path(logging_config.get("save_dir", "./results/basic_training"))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_name = dataset_config.get("name", "mnist")
    data_root = dataset_config.get("root", "./data")
    full_batch = dataset_config.get("full_batch", True)
    train_subset_size = dataset_config.get("train_subset_size", None)
    
    if dataset_name == "mnist":
        train_data, train_targets = load_mnist(
            root=data_root,
            train=True,
            download=True,
            full_batch=full_batch
        )
        test_data, test_targets = load_mnist(
            root=data_root,
            train=False,
            download=True,
            full_batch=full_batch
        )
    elif dataset_name == "cifar10":
        train_data, train_targets = load_cifar10(
            root=data_root,
            train=True,
            download=True,
            full_batch=full_batch
        )
        test_data, test_targets = load_cifar10(
            root=data_root,
            train=False,
            download=True,
            full_batch=full_batch
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    original_train_size = len(train_data)
    subset_indices = None
    if train_subset_size is not None and train_subset_size > 0:
        if train_subset_size < original_train_size:
            torch.manual_seed(seed)
            indices = torch.randperm(original_train_size)[:train_subset_size]
            indices = torch.sort(indices)[0]
            subset_indices = indices
            train_data = train_data[indices]
            train_targets = train_targets[indices]
    
    config["_original_train_size"] = original_train_size
    
    train_data = train_data.to(device)
    train_targets = train_targets.to(device)
    test_data = test_data.to(device)
    test_targets = test_targets.to(device)
    
    model = create_model(model_config, device)
    optimizers = create_optimizer(model, optimizer_config)
    
    loss_fn_name = training_config.get("loss_fn", "cross_entropy")
    if loss_fn_name == "cross_entropy":
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_fn_name}")
    
    exp_name = exp_config.get("name", "training")
    model_type = model_config.get("type", "unknown")
    dataset_name = dataset_config.get("name", "unknown")
    optimizer_type = optimizer_config.get("type", "unknown")
    timestamp = datetime.now().strftime('%m%d_%H%M')
    
    run_name = f"{exp_name}_{model_type}_{dataset_name}_{optimizer_type}_{timestamp}"
    
    wandb_config = logging_config.get("wandb", {})
    if wandb_config.get("enabled", True):
        wandb.init(
            project="muon",
            name=run_name,
            config=config
        )
    
    track_curvature = curvature_config.get("track", False)
    curvature_fn = None
    if track_curvature:
        def compute_curvature(model, loss_fn, data, targets):
            return compute_lambda_max(
                model,
                loss_fn,
                data,
                targets,
                max_iter=curvature_config.get("max_iter", 50),
                tol=curvature_config.get("tol", 1e-6),
                device=device
            )
        curvature_fn = compute_curvature
    
    num_epochs = training_config.get("num_epochs", 50)
    curvature_frequency = curvature_config.get("frequency", 5)
    save_frequency = logging_config.get("save_frequency", 10)
    
    run_dir = save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "lambda_max": [],
        "lambda_max_epochs": [],

        "lambda_grad": [],
        "lambda_muon": [],
        "lambda_eff_ratio": [],
        "lambda_eff_epochs": [],
    }

    
    for epoch in range(num_epochs):
        model.train()
        for opt in optimizers:
            opt.zero_grad()
        
        outputs = model(train_data)
        loss = loss_fn(outputs, train_targets)
        loss.backward()
        
        for opt in optimizers:
            opt.step()
        
        train_loss = loss.item()
        
        with torch.no_grad():
            train_preds = outputs.argmax(dim=1)
            train_acc = (train_preds == train_targets).float().mean().item()
        
        test_metrics = evaluate_model(model, loss_fn, test_data, test_targets, device)

        model.eval()
        with torch.no_grad():
            test_logits = model(test_data)
            test_loss_dbg = F.cross_entropy(test_logits, test_targets, reduction="mean").item()
            test_acc_dbg = (test_logits.argmax(dim=1) == test_targets).float().mean().item()

        if epoch % 10 == 0:
            print(
                f"[DBG] epoch={epoch} helper_loss={test_metrics['loss']:.6f} "
                f"inline_loss={test_loss_dbg:.6f} "
                f"helper_acc={test_metrics['accuracy']:.4f} "
                f"inline_acc={test_acc_dbg:.4f}"
            )

        if wandb_config.get("enabled", True):
            wandb.log(
                {
                    "test_loss_inline": test_loss_dbg,
                    "test_acc_inline": test_acc_dbg,
                },
                step=epoch
            )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_metrics["loss"])
        history["test_acc"].append(test_metrics["accuracy"])
        

        if (
            track_curvature and curvature_fn is not None and
            (epoch % curvature_frequency == 0 or epoch == num_epochs - 1)
        ):
            try:
                # --- pick subset for curvature (same as before) ---
                if len(train_data) > 10000:
                    curvature_subset_size = min(5000, len(train_data))
                    torch.manual_seed(seed + epoch)
                    indices = torch.randperm(len(train_data))[:curvature_subset_size]
                    curvature_data = train_data[indices]
                    curvature_targets = train_targets[indices]
                else:
                    curvature_data = train_data
                    curvature_targets = train_targets

                # --- (1) lambda_max ---
                lambda_max = curvature_fn(model, loss_fn, curvature_data, curvature_targets)
                history["lambda_max"].append(lambda_max)
                history["lambda_max_epochs"].append(epoch)

                # --- (2) lambda_grad ---
                lambda_grad = compute_lambda_grad(model, loss_fn, curvature_data, curvature_targets)
                history["lambda_grad"].append(lambda_grad)

                # --- (3) lambda_muon (lambda_eff for Muon variants) ---
                if optimizer_type == "muon":
                    lambda_muon = compute_lambda_muon(model, loss_fn, curvature_data, curvature_targets, optimizers)
                else:
                    lambda_muon = float("nan")
                history["lambda_muon"].append(lambda_muon)

                # --- (4) ratio lambda_eff / lambda_grad (only meaningful if both finite) ---
                if (
                    lambda_grad is not None and lambda_muon is not None
                    and isinstance(lambda_grad, float) and isinstance(lambda_muon, float)
                    and (not math.isnan(lambda_grad)) and (not math.isnan(lambda_muon))
                    and abs(lambda_grad) > 1e-20
                ):
                    ratio = float(lambda_muon / lambda_grad)
                else:
                    ratio = float("nan")

                history["lambda_eff_ratio"].append(ratio)
                history["lambda_eff_epochs"].append(epoch)

                print(f"[DBG] epoch={epoch} lam_max={lambda_max:.2f} lam_grad={lambda_grad:.2f} lam_muon={lambda_muon:.2f} ratio={ratio:.4f}")

                if wandb_config.get("enabled", True):
                    wandb.log(
                        {
                            "lambda_max": lambda_max,
                            "lambda_grad": None if math.isnan(lambda_grad) else lambda_grad,
                            "lambda_muon": None if math.isnan(lambda_muon) else lambda_muon,
                            "lambda_eff_ratio": None if math.isnan(ratio) else ratio,
                        },
                        step=epoch
                    )

            except Exception as e:
                print(f"Warning: Could not compute curvature: {e}")

                history["lambda_max"].append(None)
                history["lambda_max_epochs"].append(epoch)

                history["lambda_grad"].append(None)
                history["lambda_muon"].append(None)
                history["lambda_eff_ratio"].append(None)
                history["lambda_eff_epochs"].append(epoch)

        
        if wandb_config.get("enabled", True):
            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_metrics["loss"],
                    "test_acc": test_metrics["accuracy"],
                    "epoch": epoch
                },
                step=epoch
            )
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:4d}/{num_epochs}: "
                f"Train Loss={train_loss:.4f}, "
                f"Train Acc={train_acc:.4f}, "
                f"Test Loss={test_metrics['loss']:.4f}, "
                f"Test Acc={test_metrics['accuracy']:.4f}"
            )
        
        if (epoch + 1) % save_frequency == 0 or epoch == num_epochs - 1:
            checkpoint_path = run_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "history": history,
                    "config": config
                },
                checkpoint_path
            )
    
    results_path = run_dir / f"{run_name}_results.pt"
    save_dict = {
        "history": history,
        "model_state": model.state_dict(),
        "config": config
    }
    if subset_indices is not None:
        save_dict["subset_indices"] = subset_indices.cpu()
    torch.save(save_dict, results_path)
    
    config_path = run_dir / f"{run_name}_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    if subset_indices is not None:
        indices_path = run_dir / f"{run_name}_subset_indices.txt"
        with open(indices_path, 'w') as f:
            f.write(",".join(map(str, subset_indices.cpu().tolist())))
    
    vis_dir = run_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    history_plot_path = vis_dir / f"training_history.png"
    plot_training_history(history, save_path=history_plot_path, show_lambda_max=track_curvature)
    
    if track_curvature and history.get("lambda_max"):
        lambda_max_plot_path = vis_dir / f"lambda_max.png"
        plot_lambda_max(history, save_path=lambda_max_plot_path)
    
    if exp_name.startswith("t2_") and history.get("lambda_eff_epochs"):
        task2_lambdas_path = vis_dir / f"task2_lambdas.png"
        plot_task2_lambdas(history, save_path=task2_lambdas_path)
    
    class_names = None
    if dataset_name == "mnist":
        class_names = [str(i) for i in range(10)]
    elif dataset_name == "cifar10":
        class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    
    predictions_plot_path = vis_dir / f"predictions.png"
    visualize_predictions(
        model,
        test_data[:100],
        test_targets[:100],
        num_samples=16,
        class_names=class_names,
        save_path=predictions_plot_path,
        device=device
    )
    
    confusion_matrix_path = vis_dir / f"confusion_matrix.png"
    plot_confusion_matrix(
        model,
        test_data,
        test_targets,
        class_names=class_names,
        save_path=confusion_matrix_path,
        device=device
    )
    
    if wandb_config.get("enabled", True):
        log_dict = {
            "training_history": wandb.Image(str(history_plot_path)),
            "predictions": wandb.Image(str(predictions_plot_path)),
            "confusion_matrix": wandb.Image(str(confusion_matrix_path))
        }
        if track_curvature and history.get("lambda_max"):
            lambda_max_plot_path = vis_dir / f"lambda_max.png"
            if lambda_max_plot_path.exists():
                log_dict["lambda_max"] = wandb.Image(str(lambda_max_plot_path))
        wandb.log(log_dict)
    
    if wandb_config.get("enabled", True):
        wandb.finish()


if __name__ == "__main__":
    main()
