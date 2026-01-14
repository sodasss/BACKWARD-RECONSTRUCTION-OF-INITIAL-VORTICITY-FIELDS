import argparse
import os

import numpy as np
import torch

from data.io import load_h5_data
from utils.device import get_device
from utils.inference import evaluate_backward
from utils.visualization import visualize

from models.fno.fno2d import FNO2d
from models.vit.vit_stablefno import ViTStableFNO


def main():
    p = argparse.ArgumentParser(description="Evaluate backward reconstruction for FNO or ViT model")
    p.add_argument("--model_type", type=str, choices=["fno", "vit"], default="fno")

    p.add_argument("--data_path", type=str, default="w_data.h5")
    p.add_argument("--key", type=str, default="/data")
    p.add_argument("--sample_index", type=int, default=0)

    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--max_k", type=int, default=5)
    p.add_argument("--no_plot", action="store_true")

    # paths
    p.add_argument("--ckpt_path", type=str, default="")
    p.add_argument("--normalizer_path", type=str, default="")

    # FNO params (only used if model_type=fno)
    p.add_argument("--modes1", type=int, default=12)
    p.add_argument("--modes2", type=int, default=12)
    p.add_argument("--width", type=int, default=32)

    # ViT params (only used if model_type=vit)
    p.add_argument("--vit_width", type=int, default=48)
    p.add_argument("--vit_layers", type=int, default=2)
    p.add_argument("--fno_layers", type=int, default=4)

    args = p.parse_args()

    device = get_device()
    print("Using device:", device)

    data = load_h5_data(args.data_path, key=args.key).cpu().numpy()  # (N,T,H,W)
    N, T, H, W = data.shape

    assert 0 <= args.sample_index < N, "sample_index out of range"
    traj = data[args.sample_index]  # (T,H,W)
    real_w0 = traj[0]

    # default paths
    if args.model_type == "fno":
        if not args.ckpt_path:
            args.ckpt_path = os.path.join("checkpoints", "fno", "fno2d.pth")
        if not args.normalizer_path:
            args.normalizer_path = os.path.join("checkpoints", "fno", "normalizer.pt")

        model = FNO2d(modes1=args.modes1, modes2=args.modes2, width=args.width).to(device)
    else:
        if not args.ckpt_path:
            args.ckpt_path = os.path.join("checkpoints", "vit", "vit_best.pth")
        if not args.normalizer_path:
            args.normalizer_path = os.path.join("checkpoints", "vit", "normalizer.pt")

        model = ViTStableFNO(
            width=args.vit_width,
            vit_layers=args.vit_layers,
            fno_layers=args.fno_layers,
            modes1=args.modes1,
            modes2=args.modes2,
        ).to(device)

    state = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("Loaded model:", args.ckpt_path)

    normalizer = torch.load(args.normalizer_path, map_location="cpu")
    print("Loaded normalizer:", args.normalizer_path)

    pred_w0 = evaluate_backward(
        model=model,
        traj=traj,
        normalizer=normalizer,
        device=device,
        sigma=args.sigma,
        max_k=args.max_k,
    )

    if not args.no_plot:
        visualize(real_w0, pred_w0)


if __name__ == "__main__":
    main()
