import torch
import torch.nn.functional as F
from .spectral_conv import SpectralConv2d


class StableFNOBlock2D(torch.nn.Module):
    """Stacked spectral conv + 1x1 conv blocks with GELU."""

    def __init__(self, width: int, modes1: int = 12, modes2: int = 12, layers: int = 4):
        super().__init__()
        self.layers = torch.nn.ModuleList([SpectralConv2d(width, width, modes1, modes2) for _ in range(layers)])
        self.ws = torch.nn.ModuleList([torch.nn.Conv2d(width, width, 1) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv, w in zip(self.layers, self.ws):
            x = F.gelu(conv(x) + w(x))
        return x
