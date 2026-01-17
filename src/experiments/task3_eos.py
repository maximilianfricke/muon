"""
Task 3: Edge-of-Stability (EoS) tracking.

We track two related diagnostics during full-batch training:

(A) Baseline EoS from the true Hessian:
    - For GD / SGD (no preconditioner): EoS(t) = η · λ_max(H_t)
    - For AdamW / Muon (preconditioned): the stability-relevant quantity is
            η · λ_max(H_eff,t),   where  H_eff,t = P_t^{-1/2} H_t P_t^{-1/2}
      We log the baseline EoS(t) = η · λ_max(H_t) for all optimizers and label it as such.

(B) Directional curvature probes (baseline):
    We additionally log Rayleigh quotients vᵀ H_t v for several random probe
    directions v (mean/min/max) as a lightweight curvature summary.

Note: For preconditioned optimizers (AdamW, Muon), the stability-relevant EoS quantity
is η · λ_max(H_eff,t), where H_eff,t = P_t^{-1/2} H_t P_t^{-1/2}.
In this experiment we intentionally track the baseline quantity
η · λ_max(H_t) for all optimizers, which reflects the raw Hessian geometry
independently of preconditioning. Computing λ_max(H_eff,t) is left for a
subsequent extension.
"""


import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
import yaml
import wandb
import math

from src.models import MLP, TinyViT
from src.optimizers import Muon
from src.utils.data import load_mnist, load_cifar10
from src.geometry.hessian import compute_lambda_max
import matplotlib.pyplot as plt


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_device(device_str: str = "auto") -> torch.device:    
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def create_model(model_config: Dict[str, Any], device: torch.device) -> nn.Module:
    model_type = model_config["type"]
    cfg = model_config.get("config", {})

    if model_type == "mlp":
        model = MLP(
            input_size=cfg.get("input_size", 784),
            hidden_sizes=cfg.get("hidden_sizes", [128, 64]),
            num_classes=cfg.get("num_classes", 10),
            dropout=cfg.get("dropout", 0.0),
        )
    elif model_type == "tiny_vit":
        model = TinyViT(
            img_size=cfg.get("img_size", 32),
            patch_size=cfg.get("patch_size", 4),
            num_classes=cfg.get("num_classes", 10),
            embed_dim=cfg.get("embed_dim", 128),
            depth=cfg.get("depth", 4),
            num_heads=cfg.get("num_heads", 4),
            mlp_ratio=cfg.get("mlp_ratio", 4.0),
            dropout=cfg.get("dropout", 0.0),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device)


def load_data(dataset_cfg: Dict[str, Any], seed: int, device: torch.device):
    name = dataset_cfg.get("name", "mnist")
    root = dataset_cfg.get("root", "./data")
    full_batch = dataset_cfg.get("full_batch", True)
    subset = dataset_cfg.get("train_subset_size", None)

    if name == "mnist":
        train_x, train_y = load_mnist(root=root, train=True, download=True, full_batch=full_batch)
        test_x, test_y = load_mnist(root=root, train=False, download=True, full_batch=full_batch)
    elif name == "cifar10":
        train_x, train_y = load_cifar10(root=root, train=True, download=True, full_batch=full_batch)
        test_x, test_y = load_cifar10(root=root, train=False, download=True, full_batch=full_batch)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    if subset is not None and subset > 0 and subset < len(train_x):
        torch.manual_seed(seed)
        idx = torch.randperm(len(train_x))[:subset]
        idx = torch.sort(idx)[0]
        train_x, train_y = train_x[idx], train_y[idx]

    return train_x.to(device), train_y.to(device), test_x.to(device), test_y.to(device)


def create_optimizer(model: nn.Module, optimizer_cfg: Dict[str, Any]):
    opt_type = optimizer_cfg["type"]
    cfg = optimizer_cfg.get("config", {})

    if opt_type in ["gd", "sgd"]:
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.get("lr", 0.01),
            momentum=cfg.get("momentum", 0.0),
            weight_decay=cfg.get("weight_decay", 0.0),
        )

    if opt_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.get("lr", 1e-3),
            weight_decay=cfg.get("weight_decay", 0.01),
        )

    if opt_type == "muon":
        muon_params, adamw_params = [], []
        for name, p in model.named_parameters():
            if (p.ndim >= 2 and
                "output_layer" not in name and
                "head" not in name and
                "embed" not in name.lower() and
                "bias" not in name):
                muon_params.append(p)
            else:
                adamw_params.append(p)

        muon_opt = Muon(
            muon_params,
            lr=cfg.get("lr", 0.02),
            momentum=cfg.get("momentum", 0.95),
            ns_depth=cfg.get("ns_depth", 5),
            use_rms=cfg.get("use_rms", False),
            use_orthogonalization=cfg.get("use_orthogonalization", True),
            weight_decay=cfg.get("weight_decay", 0.0),
        )
        adamw_opt = torch.optim.AdamW(
            adamw_params,
            lr=cfg.get("adamw_lr", 1e-3),
            weight_decay=cfg.get("weight_decay", 0.0),
        )
        return (muon_opt, adamw_opt)

    raise ValueError(f"Unknown optimizer type: {opt_type}")


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def _zero_opt(opt):
    if isinstance(opt, tuple):
        for o in opt:
            o.zero_grad()
    else:
        opt.zero_grad()


def _step_opt(opt):
    if isinstance(opt, tuple):
        for o in opt:
            o.step()
    else:
        opt.step()


def hvp(loss: torch.Tensor, params: List[torch.nn.Parameter], v: torch.Tensor) -> torch.Tensor:
    """
    Hessian-vector product H v using autograd.
    Returns flattened Hv.
    """
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True, allow_unused=True)
    grad_flat = torch.cat([
        (g if g is not None else torch.zeros_like(p)).reshape(-1)
        for g, p in zip(grads, params)
    ])

    gv = (grad_flat * v).sum()
    hv = torch.autograd.grad(gv, params, retain_graph=True, allow_unused=True)
    hv_flat = torch.cat([
        (h if h is not None else torch.zeros_like(p)).reshape(-1)
        for h, p in zip(hv, params)
    ])
    return hv_flat


def directional_curvature_stats(
    model: nn.Module,
    loss_fn: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    num_probe_vecs: int = 8,
) -> Dict[str, float]:
    """
    Part (B) baseline: directional Rayleigh quotients v^T H v / v^T v
    for random probe vectors v.
    """
    model.zero_grad(set_to_none=True)
    logits = model(x)
    loss = loss_fn(logits, y)

    params = [p for p in model.parameters() if p.requires_grad]
    # ensure graph exists
    _ = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

    d = sum(p.numel() for p in params)
    rq_vals = []

    for _ in range(num_probe_vecs):
        v = torch.randn(d, device=x.device, dtype=torch.float32)
        v = v / (v.norm() + 1e-12)

        Hv = hvp(loss, params, v)
        rq = float((v * Hv).sum().detach().cpu().item())  # since ||v||=1, denom=1
        rq_vals.append(rq)

    rq_tensor = torch.tensor(rq_vals, dtype=torch.float32)
    return {
        "dircurv_mean": float(rq_tensor.mean().item()),
        "dircurv_max": float(rq_tensor.max().item()),
        "dircurv_min": float(rq_tensor.min().item()),
    }


def _filter_xy(xs, ys):
    out_x, out_y = [], []
    for x, y in zip(xs, ys):
        if y is None:
            continue
        if isinstance(y, float) and math.isnan(y):
            continue
        out_x.append(x)
        out_y.append(y)
    return out_x, out_y


def _filter_triplet(xs, ys1, ys2, ys3):
    out_x, o1, o2, o3 = [], [], [], []
    for x, a, b, c in zip(xs, ys1, ys2, ys3):
        if a is None or b is None or c is None:
            continue
        if any(isinstance(v, float) and math.isnan(v) for v in (a, b, c)):
            continue
        out_x.append(x); o1.append(a); o2.append(b); o3.append(c)
    return out_x, o1, o2, o3


def plot_task3(history: Dict[str, List[float]], save_path: Path, title: str) -> None:
    epochs = history["epoch"]

    # (A) Sharpness + EoS 
    lam_x, lam_y = _filter_xy(epochs, history.get("lambda_max", []))
    eos_x, eos_y = _filter_xy(epochs, history.get("eos_value", []))

    plt.figure()

    ax_left = plt.gca()
    if lam_x:
        ax_left.plot(lam_x, lam_y, label="λ_max(H)", linewidth=2)

    ax_left.set_xlabel("epoch")
    ax_left.set_title(title)
    ax_left.grid(alpha=0.3)

    # Decide whether EoS needs its own axis
    use_right_axis = False
    if lam_y and eos_y:
        max_lam = max(abs(v) for v in lam_y) if lam_y else 0.0
        max_eos = max(abs(v) for v in eos_y) if eos_y else 0.0
        # if eos is much smaller, separate axis
        if max_lam > 0 and max_eos > 0 and (max_eos / max_lam) < 0.05:
            use_right_axis = True

    if eos_x:
        if use_right_axis:
            ax_right = ax_left.twinx()
            ax_right.plot(eos_x, eos_y, label="EoS = η·λ_max(H)", linewidth=2, color="orange")

            max_eos = max(abs(v) for v in eos_y) if eos_y else 1.0
            ymax = max(1e-12, 5.0 * max_eos)
            ax_right.set_ylim(0.0, ymax)
            ax_right.set_ylabel("EoS", rotation=270, labelpad=15)

            lines_l, labels_l = ax_left.get_legend_handles_labels()
            lines_r, labels_r = ax_right.get_legend_handles_labels()
            ax_left.legend(lines_l + lines_r, labels_l + labels_r, loc="best")
        else:
            ax_left.plot(eos_x, eos_y, label="EoS = η·λ_max(H)", linewidth=2, color="orange")
            ax_left.legend(loc="best")
    else:
        ax_left.legend(loc="best")

    plt.tight_layout()
    sharp_path = save_path.with_name("sharpness_eos.png")
    plt.savefig(sharp_path)
    plt.close()

    # (B) Directional curvature stats 
    if (
        "dircurv_mean" in history
        and "dircurv_min" in history
        and "dircurv_max" in history
        and len(history["dircurv_mean"]) == len(epochs)
    ):
        dc_x, dc_mean, dc_min, dc_max = _filter_triplet(
            epochs, history["dircurv_mean"], history["dircurv_min"], history["dircurv_max"]
        )

        plt.figure()
        plt.plot(dc_x, dc_mean, label="dircurv_mean", linewidth=2)
        plt.plot(dc_x, dc_min,  label="dircurv_min",  linewidth=2)
        plt.plot(dc_x, dc_max,  label="dircurv_max",  linewidth=2)
        plt.xlabel("epoch")
        plt.title(f"{title} (Directional curvature)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        dc_path = save_path.with_name("dircurv.png")
        plt.savefig(dc_path)
        plt.close()


def run_eos_analysis(config: Dict[str, Any], opt_name: str = "single") -> Dict[str, List[float]]:
    """
    Task 3 (A + B):
      - trains full-batch
      - tracks lambda_max(H) + eos at eos_analysis.track_frequency
      - tracks directional curvature stats at curvature_analysis.track_frequency
    """

    log_cfg = config.get("logging", {}).get("wandb", {})
    if not log_cfg.get("enabled", False):
        raise RuntimeError(
            "W&B logging is mandatory for Task 3.\n"
            "Set:\n"
            "logging:\n"
            "  wandb:\n"
            "    enabled: true\n"
        )

    try:
        wandb.ensure_configured()
    except Exception as e:
        raise RuntimeError(
            "W&B is required for Task 3 but you are not logged in.\n"
            "Run: wandb login\n"
            "or set WANDB_API_KEY in your environment.\n"
        ) from e

    exp = config["experiment"]
    seed = int(exp.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = get_device(exp.get("device", "auto"))

    train_x, train_y, test_x, test_y = load_data(config["dataset"], seed, device)
    model = create_model(config["model"], device)

    train_cfg = config.get("training", {})
    num_epochs = int(train_cfg.get("num_epochs", 100))

    loss_fn_name = train_cfg.get("loss_fn", "cross_entropy")
    if loss_fn_name != "cross_entropy":
        raise ValueError("Only cross_entropy supported right now.")
    loss_fn = nn.CrossEntropyLoss()

    optimizer_cfg = config["optimizer"]
    opt_type = optimizer_cfg["type"]
    opt_cfg = optimizer_cfg.get("config", {})
    optimizer = create_optimizer(model, optimizer_cfg)

    eos_cfg = config.get("eos_analysis", {})
    eos_track_freq = int(eos_cfg.get("track_frequency", 1))

    curv_cfg = config.get("curvature_analysis", {})
    curv_enabled = bool(curv_cfg.get("enabled", False))
    curv_track_freq = int(curv_cfg.get("track_frequency", eos_track_freq))
    num_probe_vecs = int(curv_cfg.get("num_probe_vecs", 8))
    probe_subset_size = curv_cfg.get("probe_subset_size", None)

    # Hessian power-iteration params
    h_cfg = config.get("hessian", {})
    max_iter = int(h_cfg.get("max_iter", 50))
    tol = float(h_cfg.get("tol", 1e-6))

    lr = float(opt_cfg.get("lr", 0.01))

    out_cfg = config.get("output", {})
    save_root = Path(out_cfg.get("save_dir", "./results/task3"))
    save_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{exp.get('name','task3_eos')}__{opt_name}__{timestamp}"

    run_dir = save_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = run_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    group_name = exp.get("name", "task3_eos")

    wandb.init(
        project=log_cfg.get("project", "muon"),
        entity=log_cfg.get("entity", None),
        name=run_name,
        group=group_name,
        job_type=opt_name,
        config=config,
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "lambda_max": [],
        "eos_value": [],
        "eos_kind": [],
        "dircurv_mean": [],
        "dircurv_max": [],
        "dircurv_min": [],
    }

    for epoch in range(num_epochs):
        model.train()
        _zero_opt(optimizer)

        logits = model(train_x)
        loss = loss_fn(logits, train_y)
        loss.backward()
        _step_opt(optimizer)

        tr_loss = float(loss.item())
        tr_acc = float(accuracy(logits.detach(), train_y))

        model.eval()
        with torch.no_grad():
            test_logits = model(test_x)
            te_loss = float(loss_fn(test_logits, test_y).item())
            te_acc = float(accuracy(test_logits, test_y))

        lam_max = float("nan")
        eos_val = float("nan")
        eos_kind = ""

        dir_mean = float("nan")
        dir_max = float("nan")
        dir_min = float("nan")


        # (A) EoS tracking
        if (epoch % eos_track_freq == 0) or (epoch == num_epochs - 1):
            lam = compute_lambda_max(
                model, loss_fn, train_x, train_y,
                max_iter=max_iter, tol=tol, device=device
            )
            lam_max = float(lam)

            if opt_type in ["gd", "sgd"]:
                eos_val = lr * lam_max
                eos_kind = "eta*lambda_max(H)"
            else:
                eos_val = lr * lam_max
                eos_kind = "baseline eta*lambda_max(H) (H_eff not computed yet)"

        # (B) Directional curvature tracking
        if curv_enabled and ((epoch % curv_track_freq == 0) or (epoch == num_epochs - 1)):
            # optional: curvature on a subset for speed
            x_probe, y_probe = train_x, train_y
            if probe_subset_size is not None and int(probe_subset_size) > 0 and int(probe_subset_size) < len(train_x):
                idx = torch.randperm(len(train_x), device=train_x.device)[: int(probe_subset_size)]
                idx = torch.sort(idx)[0]
                x_probe, y_probe = train_x[idx], train_y[idx]

            stats = directional_curvature_stats(
                model=model,
                loss_fn=loss_fn,
                x=x_probe,
                y=y_probe,
                num_probe_vecs=num_probe_vecs,
            )
            dir_mean = stats["dircurv_mean"]
            dir_max = stats["dircurv_max"]
            dir_min = stats["dircurv_min"]

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        history["lambda_max"].append(lam_max)
        history["eos_value"].append(eos_val)
        history["eos_kind"].append(eos_kind)
        history["dircurv_mean"].append(dir_mean)
        history["dircurv_max"].append(dir_max)
        history["dircurv_min"].append(dir_min)

        wandb.log(
            {
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "test_loss": te_loss,
                "test_acc": te_acc,
                "lambda_max": None if math.isnan(lam_max) else lam_max,
                "eos_value": None if math.isnan(eos_val) else eos_val,
                "dircurv_mean": None if math.isnan(dir_mean) else dir_mean,
                "dircurv_max": None if math.isnan(dir_max) else dir_max,
                "dircurv_min": None if math.isnan(dir_min) else dir_min,
            },
            step=epoch,
        )

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:4d}/{num_epochs} "
                f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
                f"test_loss={te_loss:.4f} test_acc={te_acc:.4f} "
                f"lambda_max={lam_max:.4f} eos={eos_val:.4f} "
                f"dir_mean={dir_mean:.4f}"
            )

    results_path = run_dir / f"{run_name}_results.pt"
    torch.save({"history": history, "config": config}, results_path)

    cfg_path = run_dir / f"{run_name}_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def plot_path(base: Path, suffix: str) -> Path:
        return base.with_name(base.stem + suffix)

    base_plot_path = vis_dir / f"task3"

    plot_task3(history, base_plot_path, title=f"Task 3: EoS and Sharpness ({opt_type})")

    sharp_img = plot_path(base_plot_path, "_sharpness_eos.png")
    dc_img    = plot_path(base_plot_path, "_dircurv.png")

    final_step = int(history["epoch"][-1]) if history["epoch"] else 0

    media = {}
    if sharp_img.is_file():
        media["plot_eos"] = wandb.Image(str(sharp_img))
    if dc_img.is_file():
        media["plot_dircurv"] = wandb.Image(str(dc_img))

    if media:
        wandb.log(media, step=final_step, commit=True)
        for key, img in media.items():
            wandb.run.summary[key] = img

    wandb.finish()

    print(f"Saved run ({opt_name}) to: {run_dir}")
    return history


def main():
    parser = argparse.ArgumentParser(
        description="Task 3: Edge-of-Stability Analysis (Parts A + B)"
    )
    parser.add_argument("--config", type=str, default="configs/task3_eos.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    optimizers = config.get("optimizers", None)
    if optimizers:
        print("=" * 80)
        print("Task 3(A+B): Running multiple optimizers (one W&B run each)")
        print("Optimizers:", ", ".join(optimizers.keys()))
        print("=" * 80)

        summary = {}
        for opt_name, opt_cfg in optimizers.items():
            run_cfg = dict(config)
            run_cfg["optimizer"] = opt_cfg

            try:
                run_eos_analysis(run_cfg, opt_name=opt_name)
                summary[opt_name] = "success"
            except Exception as e:
                print(f"[FAILED] {opt_name}: {e}")
                summary[opt_name] = "failed"

        print("\n" + "=" * 80)
        print("Task 3(A+B) Summary")
        print("=" * 80)
        for k, v in summary.items():
            print(f"{k}: {v}")
        return

    if "optimizer" not in config:
        raise KeyError(
            "Config must contain either `optimizer:` (single run) "
            "or `optimizers:` (multi run)."
        )

    run_eos_analysis(config, opt_name=config["optimizer"]["type"])


if __name__ == "__main__":
    main()
