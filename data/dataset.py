import numpy as np
import torch
from torch.utils.data import Dataset


class BackwardDataset(Dataset):
    """Construct training samples for backward prediction w_t -> w_{t-k}.

    Each item returns:
      inp: (H, W, 3) = [w_t, x, y]
      out: (H, W, 1) = w_{t-k}
    """

    def __init__(self, data_norm: torch.Tensor, max_k: int = 5):
        """data_norm: (N, T, H, W) normalized trajectories."""
        self.data = data_norm
        self.N, self.T, self.H, self.W = data_norm.shape
        self.max_k = max_k

        xs = np.linspace(0, 1, self.H)
        ys = np.linspace(0, 1, self.W)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        self.grid = torch.from_numpy(np.stack([X, Y], axis=-1)).float()  # (H,W,2)

    def __len__(self) -> int:
        return self.N * (self.T - 1)

    def __getitem__(self, idx: int):
        i = idx % self.N
        t = torch.randint(1, self.T, (1,)).item()
        k = torch.randint(1, self.max_k + 1, (1,)).item()
        t_prev = max(0, t - k)

        w_t = self.data[i, t]         # (H,W)
        w_prev = self.data[i, t_prev] # (H,W)

        inp = torch.zeros(self.H, self.W, 3)
        inp[..., 0] = w_t
        inp[..., 1:] = self.grid

        return inp, w_prev.unsqueeze(-1)
