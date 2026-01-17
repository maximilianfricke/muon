"""
Utilities for computing Hessian eigenvalues, particularly λ_max.
Uses Curvlinops if available, otherwise pure PyTorch implementation.
"""

import torch
from typing import Optional, Callable

try:
    from curvlinops import HessianLinearOperator
    HAS_CURVLINOPS = True
except ImportError:
    HAS_CURVLINOPS = False


def hessian_vector_product_pytorch(
    model: torch.nn.Module,
    loss_fn: Callable,
    data: torch.Tensor,
    targets: torch.Tensor,
    vector: torch.Tensor
) -> torch.Tensor:
    """
    Compute Hessian-vector product Hv using pure PyTorch autograd.
    Args:
        model: PyTorch model
        loss_fn: Loss function
        data: Input data
        targets: Target labels
        vector: Vector to multiply with Hessian (flattened parameter vector)
    Returns:
        Hessian-vector product Hv (flattened)
    """
    model.eval()
    
    # Get all parameters as a flat vector
    params = list(model.parameters())
    param_shapes = [p.shape for p in params]
    param_sizes = [p.numel() for p in params]
    
    # Reshape vector to match parameter shapes
    vector_parts = torch.split(vector, param_sizes)
    vector_params = [v.reshape(shape) for v, shape in zip(vector_parts, param_shapes)]
    
    # Compute gradient
    loss = loss_fn(model(data), targets)
    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=True,
        retain_graph=True
    )
    
    # Compute Hv = d/dv (grad · vector)
    grad_vector = sum((g * v).sum() for g, v in zip(grads, vector_params))
    Hv = torch.autograd.grad(
        grad_vector,
        params,
        retain_graph=False
    )
    
    # Flatten result
    Hv_flat = torch.cat([h.flatten() for h in Hv])
    
    return Hv_flat


def compute_lambda_max(
    model: torch.nn.Module,
    loss_fn: Callable,
    data: torch.Tensor,
    targets: torch.Tensor,
    max_iter: int = 100,
    tol: float = 1e-6,
    device: Optional[torch.device] = None
) -> float:
    """
    Compute the largest Hessian eigenvalue (λ_max) using power iteration.
    Args:
        model: PyTorch model
        loss_fn: Loss function
        data: Input data
        targets: Target labels
        max_iter: Maximum number of power iteration steps
        tol: Convergence tolerance
        device: Device to perform computation on
    Returns:
        Largest Hessian eigenvalue (λ_max)
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    data = data.to(device)
    targets = targets.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    
    v = torch.randn(n_params, device=device)
    v = v / v.norm()
    
    lambda_max = None
    tol = float(tol)
    
    for i in range(max_iter):
        # Compute Hv
        if HAS_CURVLINOPS:
            hessian_op = HessianLinearOperator(
                model,
                loss_fn,
                (data, targets)
            )
            Hv = hessian_op @ v.cpu().numpy()
            Hv = torch.tensor(Hv, device=device, dtype=v.dtype)
        else:
            Hv = hessian_vector_product_pytorch(model, loss_fn, data, targets, v)
        
        # Estimate eigenvalue: λ ≈ v^T H v
        lambda_est = float((v @ Hv).item())
        
        # Normalize
        v_norm = Hv.norm()
        v_norm_val = v_norm.item()
        if v_norm_val < 1e-10:
            break
        
        v = Hv / v_norm
        
        # Check convergence
        if lambda_max is not None and abs(lambda_est - lambda_max) < tol:
            break
        
        lambda_max = lambda_est
    
    return float(lambda_max if lambda_max is not None else lambda_est)


def hessian_vector_product(
    model: torch.nn.Module,
    loss_fn: Callable,
    data: torch.Tensor,
    targets: torch.Tensor,
    vector: torch.Tensor
) -> torch.Tensor:
    """
    Compute Hessian-vector product Hv efficiently.
    Uses Curvlinops if available, otherwise falls back to pure PyTorch implementation.
    Args:
        model: PyTorch model
        loss_fn: Loss function
        data: Input data
        targets: Target labels
        vector: Vector to multiply with Hessian (flattened parameter vector)  
    Returns:
        Hessian-vector product Hv (flattened)
    """
    device = next(model.parameters()).device
    model.eval()
    data = data.to(device)
    targets = targets.to(device)
    
    if isinstance(vector, torch.Tensor):
        vector = vector.to(device)
    else:
        vector = torch.tensor(vector, device=device, dtype=torch.float32)
    
    if HAS_CURVLINOPS:
        hessian_op = HessianLinearOperator(
            model,
            loss_fn,
            (data, targets)
        )
        if isinstance(vector, torch.Tensor):
            vector_np = vector.cpu().numpy()
        else:
            vector_np = vector
        
        Hv = hessian_op @ vector_np
        Hv = torch.tensor(Hv, device=device, dtype=vector.dtype if isinstance(vector, torch.Tensor) else torch.float32)
    else:
        Hv = hessian_vector_product_pytorch(model, loss_fn, data, targets, vector)
    
    return Hv
