import torch
import torch.fft as fft


class ViTLayer2D(torch.nn.Module):
    """Single ViT layer:
      - FFT
      - concat real/imag
      - 1x1 conv mixing in frequency domain
      - iFFT
      - spatial 1x1 MLP
      - residual connections
    """

    def __init__(self, width: int):
        super().__init__()
        self.width = width
        self.freq_mlp = torch.nn.Conv2d(2 * width, 2 * width, 1)
        self.spatial_mlp = torch.nn.Sequential(
            torch.nn.Conv2d(width, width, 1),
            torch.nn.GELU(),
            torch.nn.Conv2d(width, width, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        x_ft = fft.fft2(x)
        x_cat = torch.cat([x_ft.real, x_ft.imag], dim=1)

        y_cat = self.freq_mlp(x_cat)
        y_real, y_imag = torch.split(y_cat, C, dim=1)

        y = torch.complex(y_real, y_imag)
        y = fft.ifft2(y).real

        x = x + y  # frequency residual
        z = self.spatial_mlp(x)
        x = x + z  # spatial residual
        return x
