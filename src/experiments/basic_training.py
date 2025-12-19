"""
Basic training script to verify all implementations work.
Uses wandb for logging and tracks curvature metrics.
All hyperparameters are specified via config YAML files.
"""

import torch
import torch.nn as nn
import argparse
import yaml
from pathlib import Path
import wandb
from datetime import datetime

from src.models import MLP, TinyViT
from src.optimizers import Muon
from src.utils.data import load_mnist, load_cifar10
from src.utils.training import train_full_batch, evaluate_model
from src.utils.visualization import visualize_predictions, plot_training_history, plot_confusion_matrix, plot_lambda_max
from src.geometry.hessian import compute_lambda_max


def get_device(device_str: str = "auto"):
    """
    Get available device.
    
    Args:
        device_str: Device specification ("auto", "cuda", "mps", or "cpu")
        
    Returns:
        torch.device
    """
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
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(model_config: dict, device: torch.device):
    """
    Create model based on config.
    
    Args:
        model_config: Model configuration dictionary
        device: Device to place model on
        
    Returns:
        PyTorch model
    """
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
    """
    Create optimizer based on config.
    
    Args:
        model: PyTorch model
        optimizer_config: Optimizer configuration dictionary
        
    Returns:
        List of optimizers (may contain multiple for Muon + AdamW)
    """
    optimizer_type = optimizer_config["type"]
    config = optimizer_config.get("config", {})
    optimizers = []
    
    if optimizer_type == "muon":
        # Separate parameters for Muon (2D weight matrices) and AdamW (others)
        muon_params = []
        adamw_params = []
        
        for name, param in model.named_parameters():
            # Muon for 2D weight matrices in hidden layers
            # Exclude output layers, embeddings, and biases
            if (param.ndim >= 2 and 
                "output_layer" not in name and 
                "head" not in name and
                "embed" not in name.lower() and
                "bias" not in name):
                muon_params.append(param)
            else:
                adamw_params.append(param)
        
        if len(muon_params) > 0:
            print(f"Muon optimizer: {len(muon_params)} parameter groups")
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
            print(f"AdamW optimizer: {len(adamw_params)} parameter groups")
            adamw_opt = torch.optim.AdamW(
                adamw_params,
                lr=config.get("adamw_lr", 1e-3),
                weight_decay=config.get("weight_decay", 0.0)
            )
            optimizers.append(adamw_opt)
        
        # Verify all parameters are covered
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
    
    # Load configuration
    config = load_config(args.config)
    exp_config = config.get("experiment", {})
    dataset_config = config.get("dataset", {})
    model_config = config.get("model", {})
    optimizer_config = config.get("optimizer", {})
    training_config = config.get("training", {})
    curvature_config = config.get("curvature", {})
    logging_config = config.get("logging", {})
    
    # Set random seed
    seed = exp_config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Get device
    device_str = exp_config.get("device", "auto")
    device = get_device(device_str)
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(logging_config.get("save_dir", "./results/basic_training"))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    dataset_name = dataset_config.get("name", "mnist")
    data_root = dataset_config.get("root", "./data")
    full_batch = dataset_config.get("full_batch", True)
    # Use subset of training data (for memory constraints or faster training)
    train_subset_size = dataset_config.get("train_subset_size", None)
    
    print(f"Loading {dataset_name} dataset...")
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
    
    # Use subset of training data if specified (ensures same subset across runs)
    original_train_size = len(train_data)
    subset_indices = None
    if train_subset_size is not None and train_subset_size > 0:
        if train_subset_size < original_train_size:
            # Use fixed seed to ensure same subset across all runs (for comparability)
            torch.manual_seed(seed)
            indices = torch.randperm(original_train_size)[:train_subset_size]
            # Sort indices to ensure deterministic order (same subset every time)
            indices = torch.sort(indices)[0]
            subset_indices = indices
            train_data = train_data[indices]
            train_targets = train_targets[indices]
            print(f"Using subset of training data: {len(train_data)} samples (out of {original_train_size} total)")
            print(f"Subset indices will be saved for reproducibility (same subset used across all runs)")
        else:
            print(f"train_subset_size ({train_subset_size}) >= dataset size ({original_train_size}), using full dataset")
    
    # Store original_train_size for later use
    config["_original_train_size"] = original_train_size
    
    print(f"Training samples: {len(train_data)}, Test samples: {len(test_data)}")
    
    # Move data to device
    print(f"Moving data to {device}...")
    train_data = train_data.to(device)
    train_targets = train_targets.to(device)
    test_data = test_data.to(device)
    test_targets = test_targets.to(device)
    
    # Create model
    print(f"Creating {model_config.get('type')} model...")
    model = create_model(model_config, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizers = create_optimizer(model, optimizer_config)
    
    # Loss function
    loss_fn_name = training_config.get("loss_fn", "cross_entropy")
    if loss_fn_name == "cross_entropy":
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_fn_name}")
    
    # Create run name (used for both wandb and directory)
    exp_name = exp_config.get("name", "training")
    # Truncated name 
    '''name_parts = exp_name.split("_")
    if len(name_parts) >= 2:
        prefix = "_".join(name_parts[:2])  # e.g., "basic_training", "task_1"
    elif len(name_parts) == 1:
        prefix = name_parts[0]
    else:
        prefix = "basic_training"  # Default prefix '''
    
    model_type = model_config.get("type", "unknown")
    dataset_name = dataset_config.get("name", "unknown")
    optimizer_type = optimizer_config.get("type", "unknown")
    timestamp = datetime.now().strftime('%m%d_%H%M')
    
    # Create run name with format: {prefix}_{model}_{dataset}_{optimizer}_{timestamp}
    run_name = f"{exp_name}_{model_type}_{dataset_name}_{optimizer_type}_{timestamp}"
    
    # Initialize wandb
    wandb_config = logging_config.get("wandb", {})
    if wandb_config.get("enabled", True):
        wandb.init(
            project="muon",  # Consistent project name
            name=run_name,
            config=config
        )
    
    # Curvature tracking function
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
    
    # Training loop setup
    num_epochs = training_config.get("num_epochs", 50)
    curvature_frequency = curvature_config.get("frequency", 5)
    save_frequency = logging_config.get("save_frequency", 10)
    
    # Create run-specific directory early to save checkpoints
    run_dir = save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("Starting training...")
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "lambda_max": [],
        "lambda_max_epochs": []  # Track which epochs lambda_max was computed at
    }
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        for opt in optimizers:
            opt.zero_grad()
        
        # Forward pass (full-batch on subset if specified)
        outputs = model(train_data)
        loss = loss_fn(outputs, train_targets)
        loss.backward()
        
        # Update optimizers
        for opt in optimizers:
            opt.step()
        
        train_loss = loss.item()
        
        # Compute train metrics
        with torch.no_grad():
            train_preds = outputs.argmax(dim=1)
            train_acc = (train_preds == train_targets).float().mean().item()
        
        # Evaluate on test set
        test_metrics = evaluate_model(model, loss_fn, test_data, test_targets, device)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_metrics["loss"])
        history["test_acc"].append(test_metrics["accuracy"])
        
        # Track curvature
        if (track_curvature and curvature_fn is not None and 
            (epoch % curvature_frequency == 0 or epoch == num_epochs - 1)):
            print(f"Computing curvature at epoch {epoch}...")
            try:
                # For very large datasets, use subset for curvature computation (faster, still representative)
                if len(train_data) > 10000:
                    # Use subset of 5000 samples for curvature (faster computation)
                    curvature_subset_size = min(5000, len(train_data))
                    # Use fixed seed for consistent subset
                    torch.manual_seed(seed + epoch)  # Different seed per epoch but deterministic
                    indices = torch.randperm(len(train_data))[:curvature_subset_size]
                    curvature_data = train_data[indices]
                    curvature_targets = train_targets[indices]
                    lambda_max = curvature_fn(model, loss_fn, curvature_data, curvature_targets)
                else:
                    lambda_max = curvature_fn(model, loss_fn, train_data, train_targets)
                history["lambda_max"].append(lambda_max)
                history["lambda_max_epochs"].append(epoch)  # Track epoch number
                if wandb_config.get("enabled", True):
                    wandb.log({"lambda_max": lambda_max}, step=epoch)
                print(f"  λ_max = {lambda_max:.6f}")
            except Exception as e:
                print(f"  Warning: Could not compute curvature: {e}")
                history["lambda_max"].append(None)
                history["lambda_max_epochs"].append(epoch)  # Still track epoch even if computation failed
        
        # Log to wandb
        if wandb_config.get("enabled", True):
            wandb.log({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_metrics["loss"],
                "test_acc": test_metrics["accuracy"],
                "epoch": epoch
            }, step=epoch)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{num_epochs}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Test Loss={test_metrics['loss']:.4f}, Test Acc={test_metrics['accuracy']:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % save_frequency == 0 or epoch == num_epochs - 1:
            checkpoint_path = run_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "history": history,
                "config": config
            }, checkpoint_path)
    
    # Save final results (run_dir already created)
    results_path = run_dir / f"{run_name}_results.pt"
    save_dict = {
        "history": history,
        "model_state": model.state_dict(),
        "config": config
    }
    # Save subset indices if used (for reproducibility)
    if subset_indices is not None:
        save_dict["subset_indices"] = subset_indices.cpu()
    torch.save(save_dict, results_path)
    print(f"Results saved to {results_path}")
    
    # Save config as YAML file for easy inspection
    config_path = run_dir / f"{run_name}_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Config saved to {config_path}")
    
    # Save subset indices as text file if used
    if subset_indices is not None:
        original_train_size = config.get("_original_train_size", "unknown")
        indices_path = run_dir / f"{run_name}_subset_indices.txt"
        with open(indices_path, 'w') as f:
            f.write(f"Subset size: {len(subset_indices)}\n")
            f.write(f"Original dataset size: {original_train_size}\n")
            f.write(f"Seed used: {seed}\n")
            f.write(f"Indices (comma-separated):\n")
            f.write(",".join(map(str, subset_indices.cpu().tolist())))
        print(f"Subset indices saved to {indices_path} (for reproducibility)")
    
    # Save lambda_max to CSV
    if track_curvature and history["lambda_max"]:
        import csv
        csv_path = run_dir / f"{run_name}_lambda_max.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'lambda_max'])
            for epoch, lambda_val in zip(history["lambda_max_epochs"], history["lambda_max"]):
                if lambda_val is not None:
                    writer.writerow([epoch, lambda_val])
        print(f"Lambda_max data saved to {csv_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    vis_dir = run_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    # 1. Plot training history
    history_plot_path = vis_dir / f"{run_name}_training_history.png"
    plot_training_history(history, save_path=history_plot_path, show_lambda_max=track_curvature)
    
    # 2. Plot lambda_max separately (if tracked)
    if track_curvature and history.get("lambda_max"):
        lambda_max_plot_path = vis_dir / f"{run_name}_lambda_max.png"
        plot_lambda_max(history, save_path=lambda_max_plot_path)
    
    # 3. Visualize predictions on test set
    class_names = None
    if dataset_name == "mnist":
        class_names = [str(i) for i in range(10)]
    elif dataset_name == "cifar10":
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
    
    predictions_plot_path = vis_dir / f"{run_name}_predictions.png"
    visualize_predictions(
        model, 
        test_data[:100],  # Use first 100 test samples
        test_targets[:100],
        num_samples=16,
        class_names=class_names,
        save_path=predictions_plot_path,
        device=device
    )
    
    # 4. Confusion matrix
    confusion_matrix_path = vis_dir / f"{run_name}_confusion_matrix.png"
    plot_confusion_matrix(
        model,
        test_data,
        test_targets,
        class_names=class_names,
        save_path=confusion_matrix_path,
        device=device
    )
    
    # Log visualizations to wandb
    if wandb_config.get("enabled", True):
        log_dict = {
            "training_history": wandb.Image(str(history_plot_path)),
            "predictions": wandb.Image(str(predictions_plot_path)),
            "confusion_matrix": wandb.Image(str(confusion_matrix_path))
        }
        # Add lambda_max plot if it exists
        if track_curvature and history.get("lambda_max"):
            lambda_max_plot_path = vis_dir / f"{run_name}_lambda_max.png"
            if lambda_max_plot_path.exists():
                log_dict["lambda_max"] = wandb.Image(str(lambda_max_plot_path))
        wandb.log(log_dict)
        print(f"Visualizations logged to wandb")
    
    print(f"Visualizations saved to {vis_dir}")
    
    if wandb_config.get("enabled", True):
        wandb.finish()
    print("Training completed!")


if __name__ == "__main__":
    main()
