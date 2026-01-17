"""
Directional curvature analysis utilities.
"""

import torch
from typing import Callable

from src.geometry.hessian import hessian_vector_product


def _flatten_params(params):
    return torch.cat([p.reshape(-1) for p in params])


def compute_directional_curvature(
    model: torch.nn.Module,
    loss_fn: Callable,
    data: torch.Tensor,
    targets: torch.Tensor,
    direction: torch.Tensor,
) -> float:
    """
    Directional curvature (Rayleigh quotient):
        lambda(dir) = (d^T H d) / (d^T d)

    NOTE: direction can be unnormalized; we divide by d^T d safely.
    direction must be a flattened vector over *all* model parameters in the same
    order as list(model.parameters()).
    """
    device = next(model.parameters()).device
    d = direction.to(device=device)

    denom = float((d @ d).item())
    if denom < 1e-20:
        return float("nan")

    Hv = hessian_vector_product(model, loss_fn, data, targets, d)
    num = float((d @ Hv).item())
    return num / denom


def compute_lambda_grad(
    model: torch.nn.Module,
    loss_fn: Callable,
    data: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    λ_grad = curvature along the (full-batch) gradient direction.
    """
    device = next(model.parameters()).device
    model.zero_grad(set_to_none=True)

    model.train()  # allow grads
    logits = model(data.to(device))
    loss = loss_fn(logits, targets.to(device))

    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=False, retain_graph=False, allow_unused=True)

    g_parts = []
    for g, p in zip(grads, params):
        if g is None:
            g_parts.append(torch.zeros_like(p).reshape(-1))
        else:
            g_parts.append(g.detach().reshape(-1))
    g = torch.cat(g_parts)

    # If grad is (near) zero, curvature is undefined / uninformative.
    if float(g.norm().item()) < 1e-12:
        return float("nan")

    return compute_directional_curvature(model, loss_fn, data, targets, g)


def compute_lambda_muon(
    model: torch.nn.Module,
    loss_fn: Callable,
    data: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    """
    λ_Muon (λ_eff for Task2) = curvature along Muon's preconditioned update direction.

    We construct a full parameter-space direction vector:
      - for params handled by Muon: use Muon preconditioned update u
      - for all other params: set direction component to 0 (we focus on Muon mechanism)
    """
    device = next(model.parameters()).device
    muon_opt = None
    if isinstance(optimizer, (list, tuple)):
        for opt in optimizer:
            # Import here to avoid circular imports if any
            from src.optimizers import Muon as MuonClass
            if isinstance(opt, MuonClass):
                muon_opt = opt
                break
    else:
        from src.optimizers import Muon as MuonClass
        if isinstance(optimizer, MuonClass):
            muon_opt = optimizer

    if muon_opt is None:
        raise ValueError("compute_lambda_muon: could not find a Muon optimizer instance.")

    # Ensure grads exist (Muon uses p.grad + momentum buffers to form its update direction)
    model.zero_grad(set_to_none=True)
    model.train()
    logits = model(data.to(device))
    loss = loss_fn(logits, targets.to(device))
    loss.backward()

    # Get Muon's per-parameter update tensors (already preconditioned, no lr step applied)
    muon_updates = muon_opt.compute_update_direction()

    if isinstance(muon_updates, dict):
        update_map = muon_updates
    else:
        update_map = {p: u for (p, u) in muon_updates}

    # Build a full flattened direction vector aligned with model.parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    d_parts = []
    for p in params:
        if p in update_map:
            d_parts.append(update_map[p].detach().reshape(-1))
        else:
            d_parts.append(torch.zeros_like(p).reshape(-1))

    d = torch.cat(d_parts)

    # DBG
    print("[DBG] muon d norm:", float(d.norm().item()))
    print("[DBG] muon nonzero elems:", int((d != 0).sum().item()))
    print("[DBG] muon update_map size:", len(update_map))
    
    if float(d.norm().item()) < 1e-12:
        return float("nan")

    return compute_directional_curvature(model, loss_fn, data, targets, d)



