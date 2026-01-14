import torch
from physics.navier_stokes import compute_NS_operator


def pde_residual_loss(w_prev_pred: torch.Tensor, w_t_true: torch.Tensor, nu: float = 1e-3) -> torch.Tensor:
    """Physics residual loss for backward step using vorticity NS operator.

    Discrete PDE: d/dt w + N(w) = 0, approximated as:
      dwdt ≈ (w_t - w_prev_pred)

    Args:
      w_prev_pred: (B,H,W,1) predicted previous field
      w_t_true:    (B,H,W,1) current field from input channel 0
    """
    dwdt = (w_t_true - w_prev_pred)  # (B,H,W,1)

    w_prev_ch = w_prev_pred.permute(0, 3, 1, 2)  # (B,1,H,W)
    Nw = compute_NS_operator(w_prev_ch, nu=nu).permute(0, 2, 3, 1)  # (B,H,W,1)

    R = dwdt + Nw
    return torch.mean(R ** 2)
