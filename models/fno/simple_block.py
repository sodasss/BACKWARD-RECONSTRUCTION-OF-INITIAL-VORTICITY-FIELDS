import torch
import torch.nn as nn
import torch.nn.functional as F
from .spectral_conv import SpectralConv2d


class SimpleBlock2d(nn.Module):
    """Core FNO block with 4 spectral conv layers and 1x1 skip convs."""

    def __init__(self, modes1: int, modes2: int, width: int):
        super().__init__()
        self.fc0 = nn.Linear(3, width)

        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)

        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,H,W,3)
        x = self.fc0(x).permute(0, 3, 1, 2)  # (B,width,H,W)

        x1 = F.relu(self.conv0(x) + self.w0(x))
        x2 = F.relu(self.conv1(x1) + self.w1(x1))
        x3 = F.relu(self.conv2(x2) + self.w2(x2))
        x4 = self.conv3(x3) + self.w3(x3)

        return x4.permute(0, 2, 3, 1)  # (B,H,W,width)
