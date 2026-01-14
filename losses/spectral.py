import torch


def spectral_reg_loss(field: torch.Tensor, cutoff: float = 0.3, weight: float = 1e-4) -> torch.Tensor:
    """Penalize high-frequency energy to suppress noise amplification.

    Args:
      field: (B,H,W,1) or (B,H,W,C) - uses channel 0
      cutoff: fraction of low-frequency block to keep (others penalized)
      weight: scaling factor for the penalty
    """
    x = field[..., 0]
    ft = torch.fft.fft2(x)

    B, H, W = ft.shape
    cx, cy = int(H * cutoff), int(W * cutoff)

    high = ft.clone()
    high[:, :cx, :cy] = 0
    high[:, -cx:, :cy] = 0
    high[:, :cx, -cy:] = 0
    high[:, -cx:, -cy:] = 0

    return weight * torch.mean(torch.abs(high) ** 2)
