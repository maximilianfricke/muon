"""
Visualization utilities for model predictions and training analysis.
"""

import torch
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path


def visualize_predictions(
    model: torch.nn.Module,
    data: torch.Tensor,
    targets: torch.Tensor,
    num_samples: int = 16,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    device: Optional[torch.device] = None
):
    """
    Visualize model predictions on a grid of samples.
    Args:
        model: Trained model
        data: Input data tensor (N, C, H, W) or (N, features)
        targets: True labels
        num_samples: Number of samples to visualize
        class_names: List of class names (e.g., ['0', '1', ..., '9'] for MNIST)
        save_path: Path to save the figure
        device: Device to run inference on
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    indices = torch.randperm(len(data))[:num_samples]
    sample_data = data[indices].to(device)
    sample_targets = targets[indices]
    
    with torch.no_grad():
        outputs = model(sample_data)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1).cpu()
        confidences = probs.max(dim=1)[0].cpu()
    
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    is_image = len(sample_data.shape) == 4  # (N, C, H, W)
    
    for i, ax in enumerate(axes):
        if i >= num_samples:
            ax.axis('off')
            continue
        
        if is_image:
            img = sample_data[i].cpu()
            if img.shape[0] == 1:  
                img = img.squeeze(0)
            elif img.shape[0] == 3:  
                img = img.permute(1, 2, 0)
            if img.min() < 0:
                img = (img + 1) / 2
            img = torch.clamp(img, 0, 1)
            ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        else:
            ax.bar(range(len(sample_data[i])), sample_data[i].cpu().numpy())
            ax.set_title(f'Sample {i}')
        
        true_label = sample_targets[i].item()
        pred_label = preds[i].item()
        confidence = confidences[i].item()
        
        if class_names:
            true_name = class_names[true_label]
            pred_name = class_names[pred_label]
        else:
            true_name = str(true_label)
            pred_name = str(pred_label)
        
        color = 'green' if true_label == pred_label else 'red'
        title = f'True: {true_name} | Pred: {pred_name}\nConf: {confidence:.2f}'
        ax.set_title(title, color=color, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    return fig


def plot_training_history(
    history: dict,
    save_path: Optional[Path] = None,
    show_lambda_max: bool = True,   
):
    """
    Plot training history (loss, accuracy).
    NOTE: λ_max is NOT plotted here anymore.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    epochs = range(len(history["train_loss"]))

    ax1 = axes[0]
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="blue", linewidth=2)
    ax1.plot(epochs, history["test_loss"], label="Test Loss", color="red", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Test Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(epochs, history["train_acc"], label="Train Accuracy", color="blue", linewidth=2)
    ax2.plot(epochs, history["test_acc"], label="Test Accuracy", color="red", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Test Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training history plot saved to {save_path}")

    return fig


def plot_lambda_max(
    history: dict,
    save_path: Optional[Path] = None
):
    """
    Plot lambda_max vs epoch (standalone plot).
    Style: green, dashed '---', no markers.
    """
    if "lambda_max" not in history or not history["lambda_max"]:
        print("No lambda_max data to plot")
        return None

    if "lambda_max_epochs" in history and history["lambda_max_epochs"]:
        lambda_epochs = [
            epoch for val, epoch in zip(history["lambda_max"], history["lambda_max_epochs"])
            if val is not None
        ]
        lambda_values = [
            val for val, epoch in zip(history["lambda_max"], history["lambda_max_epochs"])
            if val is not None
        ]
    else:
        lambda_epochs = [i for i, val in enumerate(history["lambda_max"]) if val is not None]
        lambda_values = [val for val in history["lambda_max"] if val is not None]

    if not lambda_values:
        print("No valid lambda_max values to plot")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        lambda_epochs,
        lambda_values,
        linewidth=2,
        linestyle="--",
        color="green",
        label="λ_max(H)",
    )
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("λ_max(H) (Largest Hessian Eigenvalue)", fontsize=12)
    ax.set_title("Sharpness Evolution (λ_max(H))", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Lambda_max plot saved to {save_path}")

    return fig


def plot_task2_lambdas(history, save_path):
    epochs = history.get("lambda_eff_epochs", [])
    lam_max = history.get("lambda_max", [])
    lam_g   = history.get("lambda_grad", [])
    lam_m   = history.get("lambda_muon", [])
    ratio   = history.get("lambda_eff_ratio", [])

    def _filter(x, y):
        ox, oy = [], []
        for a, b in zip(x, y):
            if b is None:
                continue
            if isinstance(b, float) and math.isnan(b):
                continue
            ox.append(a); oy.append(b)
        return ox, oy

    x1, y1 = _filter(epochs, lam_max)
    x2, y2 = _filter(epochs, lam_g)
    x3, y3 = _filter(epochs, lam_m)
    xr, yr = _filter(epochs, ratio)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 3]}
    )

    if x1: ax_top.plot(x1, y1, label="lambda_max(H)", linewidth=2)
    if x2: ax_top.plot(x2, y2, label="lambda_grad", linewidth=2)

    ax_top.set_ylabel("curvature (max / grad)")
    ax_top.set_title("Task 2: Curvatures on tracked epochs")

    ax_right = ax_top.twinx()
    if x3:
        ax_right.plot(
            x3, y3,
            linestyle="--",
            linewidth=1.8,
            alpha=0.9,
            color = "green",
            label="lambda_muon"
        )

        ymax = max(y3)
        ax_right.set_ylim(0.0, max(10.0, 2.5 * ymax))

    ax_right.set_ylabel("curvature (muon)")

    lines_l, labels_l = ax_top.get_legend_handles_labels()
    lines_r, labels_r = ax_right.get_legend_handles_labels()
    ax_top.legend(lines_l + lines_r, labels_l + labels_r, loc="upper right")

    ax_top.grid(alpha=0.3)

    if xr:
        ax_bot.plot(xr, yr, linewidth=2, label="lambda_eff_ratio (muon / grad)")

    ax_bot.set_xlabel("epoch")
    ax_bot.set_ylabel("ratio")
    ax_bot.set_title("Task 2: lambda_eff_ratio")
    ax_bot.legend()
    ax_bot.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Task2 combined lambdas + ratio plot saved to {save_path}")


def plot_confusion_matrix(
    model: torch.nn.Module,
    data: torch.Tensor,
    targets: torch.Tensor,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    device: Optional[torch.device] = None
):
    """
    Plot confusion matrix for model predictions.
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    data = data.to(device)
    
    with torch.no_grad():
        outputs = model(data)
        preds = outputs.argmax(dim=1).cpu().numpy()
    
    targets_np = targets.cpu().numpy()
    num_classes = len(np.unique(targets_np))
    
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(targets_np, preds):
        cm[true, pred] += 1
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    
    ax.set(xticks=np.arange(num_classes),
           yticks=np.arange(num_classes),
           xticklabels=class_names,
           yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    thresh = cm.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return fig


