import torch
import torch.nn.functional as F


def sobolev_loss(pred: torch.Tensor, target: torch.Tensor, lam: float = 0.1) -> torch.Tensor:
    """MSE + gradient MSE (Sobolev/H1-like) for (B,H,W,1)."""
    mse = F.mse_loss(pred, target)

    # gradients along H (dim=1) and W (dim=2)
    grad_px = pred[:, 1:, :, :] - pred[:, :-1, :, :]
    grad_tx = target[:, 1:, :, :] - target[:, :-1, :, :]

    grad_py = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    grad_ty = target[:, :, 1:, :] - target[:, :, :-1, :]

    g_loss = F.mse_loss(grad_px, grad_tx) + F.mse_loss(grad_py, grad_ty)
    return mse + lam * g_loss
