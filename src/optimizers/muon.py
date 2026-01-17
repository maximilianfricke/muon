"""
Muon optimizer implementation.

Muon - MomentUm Orthogonalized by Newton-schulz
https://kellerjordan.github.io/posts/muon/

Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
matrix. For efficient orthogonalization we use a Newton-Schulz iteration.
"""

import torch
from torch.optim import Optimizer
from typing import Optional
from typing import List, Tuple, Dict


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    
    Args:
        G: Matrix to orthogonalize (can be batched, ndim >= 2)
        steps: Number of Newton-Schulz iterations
        
    Returns:
        Orthogonalized matrix
    """
    assert G.ndim >= 2  # batched Muon implementation
    
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    X = G.float()
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A  # quintic computation strategy
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    
    return X.to(G.dtype)

def muon_update(
    grad,
    momentum,
    beta=0.95,
    ns_steps=5,
    nesterov=True,
    use_orthogonalization=True,
    use_rms=False,
    debug=False,
):
    g = grad.detach()
    momentum.lerp_(g, 1 - beta)
    update = g.lerp(momentum, beta) if nesterov else momentum

    # DBG: norms before anything 
    grad_norm = g.norm()
    upd_norm_pre = update.norm()

    if update.ndim == 4:
        update = update.view(len(update), -1)

    upd_norm_pre_ortho = update.norm()

    if use_orthogonalization:
        update = zeropower_via_newtonschulz5(update, steps=ns_steps)

    upd_norm_post_ortho = update.norm()

    # Base Muon scale (make it a tensor so debug always works)
    m, n = update.size(-2), update.size(-1)
    base_scale_float = max(1.0, m / n) ** 0.5
    base_scale = update.new_tensor(base_scale_float)  # tensor on correct device/dtype

    # RMS normalization
    rms = None
    if use_rms:
        rms = update.norm(dim=(-2, -1), keepdim=True) / ((m * n) ** 0.5)
        eff_scale = base_scale / (rms + 1e-7)  # tensor
    else:
        eff_scale = base_scale  # tensor

    update = update * eff_scale

    if debug:
        dbg = {
            "grad_norm": float(grad_norm),
            "upd_norm_pre": float(upd_norm_pre),
            "upd_norm_pre_ortho": float(upd_norm_pre_ortho),
            "upd_norm_post_ortho": float(upd_norm_post_ortho),
            "base_scale": float(base_scale),
            "rms": float(rms.mean()) if rms is not None else float("nan"),
            "eff_scale": float(eff_scale.mean()),
        }
        return update, dbg

    return update




class Muon(Optimizer):
    """
    Muon optimizer - MomentUm Orthogonalized by Newton-schulz.
    
    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.
    Hidden convolutional weights can be trained using Muon by viewing them as 2D and then
    collapsing their last 3 dimensions.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate, in units of spectral norm per update
        momentum: Momentum coefficient (default: 0.95)
        ns_depth: Number of Newton-Schulz iteration steps (default: 5)
        use_rms: Whether to use RMS normalization (for ablation study)
        use_orthogonalization: Whether to use orthogonalization (for ablation study)
        weight_decay: AdamW-style weight decay coefficient
        nesterov: Whether to use Nesterov momentum (default: True)
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_depth: int = 5,
        use_rms: bool = False,
        use_orthogonalization: bool = True,
        weight_decay: float = 0.0,
        nesterov: bool = True
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_depth=ns_depth,
            use_rms=use_rms,
            use_orthogonalization=use_orthogonalization,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
        super().__init__(params, defaults)

        
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
            
        Returns:
            Loss value if closure is provided, None otherwise
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                
                # Compute Muon update
                update, dbg = muon_update(
                    p.grad,
                    state["momentum_buffer"],
                    beta=group["momentum"],
                    ns_steps=group["ns_depth"],
                    nesterov=group["nesterov"],
                    use_orthogonalization=group["use_orthogonalization"],
                    use_rms=group["use_rms"],
                    debug=True,   # DBG
                )

                # DBG 
                dbg_state = self.state.setdefault(
                    "_dbg",
                    {"t": 0, "sum_rms": 0.0, "sum_scale": 0.0, "sum_upd": 0.0, "cnt": 0},
                )

                dbg_state["t"] += 1
                dbg_state["sum_rms"] += dbg["rms"]
                dbg_state["sum_scale"] += dbg["eff_scale"]
                dbg_state["sum_upd"] += dbg["upd_norm_post_ortho"]
                dbg_state["cnt"] += 1

                if dbg_state["t"] % 100 == 0:
                    c = max(1, dbg_state["cnt"])
                    print(
                        f"[MUON DBG] step={dbg_state['t']} | "
                        f"use_rms={group['use_rms']} "
                        f"use_ortho={group['use_orthogonalization']} "
                        f"ns={group['ns_depth']} | "
                        f"avg_rms={dbg_state['sum_rms']/c:.4f} "
                        f"avg_eff_scale={dbg_state['sum_scale']/c:.4f} "
                        f"avg_upd_norm={dbg_state['sum_upd']/c:.4f}"
                    )
                    dbg_state["sum_rms"] = dbg_state["sum_scale"] = dbg_state["sum_upd"] = 0.0
                    dbg_state["cnt"] = 0

                # Apply weight decay (AdamW-style)
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss
    
    @torch.no_grad()
    def _ensure_state(self, p: torch.Tensor):
        state = self.state[p]
        if len(state) == 0:
            state["momentum_buffer"] = torch.zeros_like(p)
        return state


    @torch.no_grad()
    def compute_update_direction(self) -> Dict[torch.nn.Parameter, torch.Tensor]:
        update_map: Dict[torch.nn.Parameter, torch.Tensor] = {}

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self._ensure_state(p)
                g_work = p.grad.detach().clone()
                u = muon_update(
                    g_work,
                    state["momentum_buffer"],
                    beta=group["momentum"],
                    ns_steps=group["ns_depth"],
                    nesterov=group["nesterov"],
                    use_orthogonalization=group["use_orthogonalization"],
                    use_rms=group["use_rms"],
                )
                update_map[p] = u.reshape(p.shape).detach().clone()

        return update_map


