import torch
import torch.fft as fft


class SpectralConv2d(torch.nn.Module):
    """A slightly safer spectral conv used in your ViT experiment."""

    def __init__(self, in_c: int, out_c: int, modes1: int, modes2: int):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_c * out_c)
        self.w1 = torch.nn.Parameter(scale * torch.randn(in_c, out_c, modes1, modes2, dtype=torch.cfloat))
        self.w2 = torch.nn.Parameter(scale * torch.randn(in_c, out_c, modes1, modes2, dtype=torch.cfloat))

    @staticmethod
    def compl_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a: (B,in,H,Wf), b: (in,out,Hm,Wm)
        return torch.einsum("bixy,ioxy->boxy", a, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = fft.rfft2(x)
        out_ft = torch.zeros(B, self.out_c, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)

        m1 = min(self.modes1, H)
        m2 = min(self.modes2, W // 2 + 1)

        out_ft[:, :, :m1, :m2] = self.compl_mul(x_ft[:, :, :m1, :m2], self.w1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self.compl_mul(x_ft[:, :, -m1:, :m2], self.w2[:, :, :m1, :m2])

        return fft.irfft2(out_ft, s=(H, W))
