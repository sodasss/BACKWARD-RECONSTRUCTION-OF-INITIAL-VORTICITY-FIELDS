import torch
import torch.nn.functional as F
from .vit2d import ViT2D
from .stable_fno import StableFNOBlock2D


class ViTStableFNO(torch.nn.Module):
    """Hybrid model: Linear embed -> ViT -> StableFNO -> MLP head.

    Input:  (B,H,W,3) = [w_t, x, y]
    Output: (B,H,W,1) = predicted w_{t-k}
    """

    def __init__(
        self,
        width: int = 48,
        vit_layers: int = 2,
        fno_layers: int = 4,
        modes1: int = 12,
        modes2: int = 12,
    ):
        super().__init__()
        self.fc0 = torch.nn.Linear(3, width)
        self.vit = ViT2D(width, layers=vit_layers)
        self.fno = StableFNOBlock2D(width, modes1=modes1, modes2=modes2, layers=fno_layers)
        self.fc1 = torch.nn.Linear(width, 128)
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,H,W,3)
        x = self.fc0(x)           # (B,H,W,width)
        x = x.permute(0, 3, 1, 2) # (B,width,H,W)

        x = self.vit(x)
        x = self.fno(x)

        x = x.permute(0, 2, 3, 1) # (B,H,W,width)
        x = F.gelu(self.fc1(x))
        out = self.fc2(x)         # (B,H,W,1)
        return out
