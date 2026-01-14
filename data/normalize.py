import torch


class UnitGaussianNormalizer:
    """Normalize x (N,H,W) to mean=0, std=1 using dataset statistics."""

    def __init__(self, x: torch.Tensor):
        self.mean = x.mean(dim=0, keepdim=True)
        self.std = x.std(dim=0, keepdim=True) + 1e-6

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean
