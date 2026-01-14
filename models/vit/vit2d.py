import torch
from .vit_layer import ViTLayer2D


class ViT2D(torch.nn.Module):
    def __init__(self, width: int, layers: int = 2):
        super().__init__()
        self.layers = torch.nn.ModuleList([ViTLayer2D(width) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
