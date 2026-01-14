from .sobolev import sobolev_loss
from .spectral import spectral_reg_loss
from .pde import pde_residual_loss

__all__ = ["sobolev_loss", "spectral_reg_loss", "pde_residual_loss"]
