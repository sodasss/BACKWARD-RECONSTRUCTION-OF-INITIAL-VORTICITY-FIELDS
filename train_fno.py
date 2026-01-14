import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.io import load_h5_data
from data.normalize import UnitGaussianNormalizer
from data.dataset import BackwardDataset
from models.fno.fno2d import FNO2d
from losses.spectral import spectral_reg_loss
from utils.device import get_device


def main():
    p = argparse.ArgumentParser(description="Train baseline FNO for backward prediction")
    p.add_argument("--data_path", type=str, default="w_data.h5")
    p.add_argument("--key", type=str, default="/data")
    p.add_argument("--ntrain", type=int, default=1600)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--max_k", type=int, default=5)

    p.add_argument("--modes1", type=int, default=12)
    p.add_argument("--modes2", type=int, default=12)
    p.add_argument("--width", type=int, default=32)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--spec_cutoff", type=float, default=0.3)
    p.add_argument("--spec_weight", type=float, default=1e-4)

    p.add_argument("--save_dir", type=str, default="checkpoints/fno")
    p.add_argument("--save_name", type=str, default="fno2d.pth")
    p.add_argument("--save_normalizer", type=str, default="normalizer.pt")

    args = p.parse_args()

    device = get_device()
    print("Using device:", device)

    data = load_h5_data(args.data_path, key=args.key)  # (N,T,H,W)
    N, T, H, W = data.shape
    ntrain = min(args.ntrain, N)

    train_data = data[:ntrain]
    flat_train = train_data.reshape(-1, H, W)
    normalizer = UnitGaussianNormalizer(flat_train)

    train_norm = normalizer.encode(flat_train).reshape(ntrain, T, H, W)

    train_loader = DataLoader(
        BackwardDataset(train_norm, max_k=args.max_k),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = FNO2d(modes1=args.modes1, modes2=args.modes2, width=args.width).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)

    for ep in range(args.epochs):
        model.train()
        acc = 0.0
        for inp, out in train_loader:
            inp = inp.to(device)
            out = out.to(device)

            pred = model(inp)
            loss_data = F.mse_loss(pred, out)
            loss_spec = spectral_reg_loss(pred, cutoff=args.spec_cutoff, weight=args.spec_weight)
            loss = loss_data + loss_spec

            opt.zero_grad()
            loss.backward()
            opt.step()

            acc += loss.item()

        print(f"Epoch {ep+1}/{args.epochs} - loss={acc/len(train_loader):.6f}")

    torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name))
    torch.save(normalizer, os.path.join(args.save_dir, args.save_normalizer))
    print("Saved to:", args.save_dir)


if __name__ == "__main__":
    main()
