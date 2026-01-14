import torch
import torch.fft as fft


def compute_NS_operator(w: torch.Tensor, nu: float = 1e-3) -> torch.Tensor:
    """Navier–Stokes vorticity operator N(w) = u·∇w - nu*Δw (periodic BC).

    Args:
      w: (B, 1, H, W)

    Returns:
      Nw: (B, 1, H, W)
    """
    B, C, H, W = w.shape
    device = w.device

    ky = torch.fft.fftfreq(H, d=1.0).to(device).view(1, 1, H, 1)
    kx = torch.fft.fftfreq(W, d=1.0).to(device).view(1, 1, 1, W)
    k2 = kx**2 + ky**2
    k2[..., 0, 0] = 1e-6

    w_hat = fft.fft2(w)

    psi_hat = -w_hat / k2
    # psi = fft.ifft2(psi_hat).real  # not needed explicitly

    dpsi_dx_hat = 1j * kx * psi_hat
    dpsi_dy_hat = 1j * ky * psi_hat

    u_x = fft.ifft2(-dpsi_dy_hat).real
    u_y = fft.ifft2(dpsi_dx_hat).real

    dw_dx = fft.ifft2(1j * kx * w_hat).real
    dw_dy = fft.ifft2(1j * ky * w_hat).real

    adv = u_x * dw_dx + u_y * dw_dy

    lap_hat = -k2 * w_hat
    lap = fft.ifft2(lap_hat).real

    return adv - nu * lap
