import torch
import torch.nn as nn
import torch.nn.functional as F
from .simple_block import SimpleBlock2d


class FNO2d(nn.Module):
    """2D Fourier Neural Operator.

    Input:  (B,H,W,3) = [w_t, x, y]
    Output: (B,H,W,1) = w_{t-k}
    """

    def __init__(self, modes1: int = 12, modes2: int = 12, width: int = 32):
        super().__init__()
        self.block = SimpleBlock2d(modes1, modes2, width)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
