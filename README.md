# Backward Reconstruction with FNO and ViT-StableFNO

This repository contains two PyTorch models for **backward-time reconstruction** of 2D vorticity fields:

- **FNO (baseline)**: Fourier Neural Operator for mapping \( [w_t, x, y] \to w_{t-k} \)
- **ViT-StableFNO (experiment)**: a hybrid model combining an ViT-style frequency mixer and a Stable-FNO block, trained with:
  - Sobolev (H1-like) loss
  - Spectral high-frequency regularization
  - Navier–Stokes PDE residual loss

Both models share the same dataset format and the same backward inference procedure.

---

## Data Format

The code expects an HDF5 file (default: `w_data.h5`) containing a dataset:

- Key: `/data` (configurable)
- Shape: `(N, T, H, W)` (e.g., `(2000, 50, 64, 64)`)

Place `w_data.h5` in the project root, or pass `--data_path`.

---

## Installation

```bash
pip install -r requirements.txt
```

> For GPU acceleration, install the CUDA-enabled PyTorch build matching your system.

---

## Train

### 1) Train FNO (baseline)

```bash
python train_fno.py \
  --data_path w_data.h5 \
  --epochs 30 \
  --batch_size 8 \
  --max_k 5
```

Outputs:
- `checkpoints/fno/fno2d.pth`
- `checkpoints/fno/normalizer.pt`

---

### 2) Train ViT-StableFNO (experiment)

```bash
python train_vit.py \
  --data_path w_data.h5 \
  --epochs 300 \
  --batch_size 8 \
  --max_k 5
```

Outputs:
- `checkpoints/vit/vit_latest.pth`
- `checkpoints/vit/vit_best.pth`
- `checkpoints/vit/normalizer.pt`

---

## Evaluate / Visualize

### Evaluate FNO

```bash
python evaluate.py --model_type fno --sample_index 0
```

### Evaluate ViT-StableFNO

```bash
python evaluate.py --model_type vit --sample_index 0
```

Useful flags:
- `--sigma 0.05` : noise std added to the terminal state
- `--max_k 5` : backward step size (should match training)
- `--no_plot` : disable matplotlib plotting

---

## Code Organization

- `models/`
  - `models/fno/` baseline FNO
  - `models/vit/` ViT + StableFNO hybrid
- `data/` dataset + normalizer + HDF5 loader
- `physics/` Navier–Stokes operator in Fourier domain
- `losses/` Sobolev / Spectral / PDE residual losses
- `utils/` device, inference, visualization
- `train_fno.py`, `train_vit.py`, `evaluate.py`

---

## Notes

- The ViT experiment is based on your original notebook design:
  - ViT-style FFT real/imag mixing with residuals
  - Stable-FNO spectral conv stack
  - Sobolev + Spectral + PDE losses
- You can easily add ablations by toggling loss terms in `train_vit.py`.

