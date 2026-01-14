import h5py
import numpy as np
import torch


def load_h5_data(path: str, key: str = "/data") -> torch.Tensor:
    """Load HDF5 tensor of shape (N, T, H, W) from `key` (default: /data)."""
    with h5py.File(path, "r") as f:
        arr = np.array(f[key])
    return torch.from_numpy(arr).float()
