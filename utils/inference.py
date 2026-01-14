import numpy as np
import torch
from data.normalize import UnitGaussianNormalizer


def evaluate_backward(
    model: torch.nn.Module,
    traj: np.ndarray,
    normalizer: UnitGaussianNormalizer,
    device: torch.device | None = None,
    sigma: float = 0.05,
    max_k: int = 5,
) -> np.ndarray:
    """Backward reconstruction from a noisy terminal state.

    Args:
      model: maps (B,H,W,3) -> (B,H,W,1)
      traj: (T,H,W) in physical space
      normalizer: UnitGaussianNormalizer built from training data
      sigma: noise std applied to terminal field (physical space)
      max_k: max backward jump per step (matches training setting)

    Returns:
      w0_pred: (H,W) in physical space
    """
    if device is None:
        device = next(model.parameters()).device

    T, H, W = traj.shape

    wT_phys = traj[-1] + sigma * np.random.randn(H, W)
    wT_norm = normalizer.encode(torch.from_numpy(wT_phys).float()).cpu().numpy()

    xs = np.linspace(0, 1, H)
    ys = np.linspace(0, 1, W)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    grid = np.stack([X, Y], axis=-1)  # (H,W,2)

    w_est = wT_norm
    t = T - 1

    model.eval()
    with torch.no_grad():
        while t > 0:
            k = min(max_k, t)
            inp = np.zeros((H, W, 3), dtype=np.float32)
            inp[..., 0] = w_est
            inp[..., 1:] = grid

            x = torch.from_numpy(inp).unsqueeze(0).to(device)
            pred_norm = model(x)[0].cpu().numpy().squeeze()

            w_est = pred_norm
            t -= k

    w_est_flat = torch.from_numpy(w_est).view(1, H, W)
    w0_pred = normalizer.decode(w_est_flat).cpu().numpy()[0]
    return w0_pred
