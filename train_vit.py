import argparse
import os

import torch
from torch.utils.data import DataLoader

from data.io import load_h5_data
from data.normalize import UnitGaussianNormalizer
from data.dataset import BackwardDataset
from models.vit.vit_stablefno import ViTStableFNO
from losses.sobolev import sobolev_loss
from losses.spectral import spectral_reg_loss
from losses.pde import pde_residual_loss
from utils.device import get_device


def main():
    p = argparse.ArgumentParser(description="Train ViT+StableFNO with Sobolev + Spectral + PDE losses")
    p.add_argument("--data_path", type=str, default="w_data.h5")
    p.add_argument("--key", type=str, default="/data")
    p.add_argument("--ntrain", type=int, default=1600)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--max_k", type=int, default=5)

    p.add_argument("--width", type=int, default=48)
    p.add_argument("--vit_layers", type=int, default=2)
    p.add_argument("--fno_layers", type=int, default=4)
    p.add_argument("--modes1", type=int, default=12)
    p.add_argument("--modes2", type=int, default=12)

    p.add_argument("--lr", type=float, default=1e-3)

    # loss weights / params
    p.add_argument("--sobolev_lam", type=float, default=0.1)
    p.add_argument("--spec_cutoff", type=float, default=0.3)
    p.add_argument("--spec_weight", type=float, default=1e-4)
    p.add_argument("--pde_weight", type=float, default=1e-3)
    p.add_argument("--nu", type=float, default=1e-3)

    p.add_argument("--save_dir", type=str, default="checkpoints/vit")
    p.add_argument("--save_latest", type=str, default="vit_latest.pth")
    p.add_argument("--save_best", type=str, default="vit_best.pth")
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

    model = ViTStableFNO(
        width=args.width,
        vit_layers=args.vit_layers,
        fno_layers=args.fno_layers,
        modes1=args.modes1,
        modes2=args.modes2,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)

    best = float("inf")

    for ep in range(args.epochs):
        model.train()
        acc = 0.0

        for inp, out in train_loader:
            inp = inp.to(device)  # (B,H,W,3)
            out = out.to(device)  # (B,H,W,1)

            pred = model(inp)

            loss_data = sobolev_loss(pred, out, lam=args.sobolev_lam)
            loss_spec = spectral_reg_loss(pred, cutoff=args.spec_cutoff, weight=args.spec_weight)

            w_t_true = inp[..., 0:1]
            loss_pde = pde_residual_loss(pred, w_t_true, nu=args.nu)

            loss = loss_data + loss_spec + args.pde_weight * loss_pde

            opt.zero_grad()
            loss.backward()
            opt.step()

            acc += loss.item()

        avg = acc / len(train_loader)
        print(f"Epoch {ep+1}/{args.epochs} - loss={avg:.6f}")

        # save latest
        torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_latest))

        # save best
        if avg < best:
            best = avg
            torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_best))
            print(f"  ↳ New best saved (loss={best:.6f})")

    # save normalizer at end
    torch.save(normalizer, os.path.join(args.save_dir, args.save_normalizer))
    print("Saved to:", args.save_dir)


if __name__ == "__main__":
    main()
