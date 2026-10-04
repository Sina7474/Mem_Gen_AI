"""Unconditional U-Net denoiser and the DDIM wrapper around it.

The architecture is deliberately small and unconditional: the only conditioning
signal is the diffusion time, so nothing but the training data itself can drive
the transition from generalisation to memorisation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BETAS, N_T

__all__ = ["Unet", "DDIM", "build_ddim", "find_t_eval", "evaluate_denoising_loss"]


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_res: bool = False):
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        if not self.is_res:
            return x2
        out = x + x2 if self.same_channels else x1 + x2
        return out / 1.414


class UnetDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.model = nn.Sequential(
            ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        )

    def forward(self, x, skip):
        return self.model(torch.cat((x, skip), 1))


class EmbedFC(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim), nn.GELU(), nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, x):
        return self.model(x.view(-1, self.input_dim))


class Unet(nn.Module):
    """Unconditional U-Net; the only embedding is the diffusion timestep.

    Args:
        in_channels: 2, for the real and imaginary parts of the channel.
        n_feat: base width ``W``, the knob used in the capacity ablation.
        bottleneck: width of the spatial bottleneck after two down-sampling
            stages. It equals ``cols // 4`` for the supported channel shapes
            (8 for the 4x32 Sionna channels, 2 for the 4x8 measured ones).
    """

    def __init__(self, in_channels: int = 2, n_feat: int = 256,
                 bottleneck: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.to_vec = nn.Sequential(nn.AvgPool2d((1, bottleneck)), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat,
                               kernel_size=(1, bottleneck),
                               stride=(1, bottleneck)),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, in_channels, 3, 1, 1),
        )

    def forward(self, x, t):
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hidden = self.to_vec(down2)

        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hidden)
        up2 = self.up1(up1 + temb1, down2)
        up3 = self.up2(up2 + temb2, down1)
        return self.out(torch.cat((up3, x), 1))


def ddim_schedules(beta1: float, beta2: float, n_t: int) -> dict[str, torch.Tensor]:
    """Linear beta schedule and the derived DDIM coefficients."""
    assert 0.0 < beta1 < beta2 < 1.0
    beta_t = (beta2 - beta1) * torch.arange(0, n_t + 1, dtype=torch.float32) / n_t + beta1
    alpha_t = 1.0 - beta_t
    alphabar_t = torch.cumsum(torch.log(alpha_t), dim=0).exp()
    return {
        "alpha_t": alpha_t,
        "oneover_sqrta": 1.0 / torch.sqrt(alpha_t),
        "sqrt_beta_t": torch.sqrt(beta_t),
        "alphabar_t": alphabar_t,
        "sqrtab": torch.sqrt(alphabar_t),
        "sqrtmab": torch.sqrt(1.0 - alphabar_t),
        "DDIM_coeff": torch.sqrt(1.0 - alphabar_t)
        - torch.sqrt(alpha_t) * torch.sqrt(1.0 - alphabar_t / alpha_t),
    }


class DDIM(nn.Module):
    """Denoising diffusion implicit model with a deterministic sampler."""

    def __init__(self, nn_model: nn.Module, betas=BETAS, n_T: int = N_T,
                 device: torch.device | str = "cpu"):
        super().__init__()
        self.nn_model = nn_model.to(device)
        for name, value in ddim_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(name, value)
        self.n_T = n_T
        self.device = device
        self.loss_mse = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Training objective: predict the noise at a uniformly random step."""
        ts = torch.randint(1, self.n_T + 1, (x.shape[0],), device=x.device)
        noise = torch.randn_like(x)
        x_t = self.sqrtab[ts, None, None, None] * x + self.sqrtmab[ts, None, None, None] * noise
        return self.loss_mse(noise, self.nn_model(x_t, ts / self.n_T))

    def denoising_loss(self, x: torch.Tensor, t_eval: int,
                       noise: torch.Tensor) -> torch.Tensor:
        """Denoising MSE at a *fixed* step with pre-drawn noise."""
        ts = torch.full((x.shape[0],), t_eval, dtype=torch.long, device=x.device)
        x_t = self.sqrtab[ts, None, None, None] * x + self.sqrtmab[ts, None, None, None] * noise
        return F.mse_loss(self.nn_model(x_t, ts.float() / self.n_T), noise)

    @torch.no_grad()
    def sample(self, n_sample: int, size: tuple[int, ...],
               device: torch.device | str) -> torch.Tensor:
        """Deterministic DDIM sampling from pure noise."""
        x_i = torch.randn(n_sample, *size, device=device)
        for i in range(self.n_T, 0, -1):
            t_is = torch.full((n_sample, 1, 1, 1), i / self.n_T, device=device)
            eps = self.nn_model(x_i, t_is)
            x_i = self.oneover_sqrta[i] * (x_i - eps * self.DDIM_coeff[i])
        return x_i


def build_ddim(sample_shape: tuple[int, int, int], width: int,
               device: torch.device | str) -> DDIM:
    """Instantiate a DDIM matching a dataset's channel shape."""
    channels, _, cols = sample_shape
    model = Unet(in_channels=channels, n_feat=width, bottleneck=cols // 4)
    return DDIM(model, betas=BETAS, n_T=N_T, device=device).to(device)


def find_t_eval(alphabar_t: torch.Tensor) -> tuple[int, float, float]:
    """Pick the low-noise step used to probe train/test denoising loss.

    The target ``alpha_bar = exp(-0.02)`` is the discrete analogue of the
    continuous-time evaluation level used in the memorisation literature.
    """
    target = float(np.exp(-2 * 0.01))
    values = alphabar_t[1:].detach().cpu().numpy()
    t_eval = int(np.argmin(np.abs(values - target)) + 1)
    actual = float(alphabar_t[t_eval].item())
    return t_eval, actual, float(np.sqrt(1.0 - actual))


@torch.no_grad()
def evaluate_denoising_loss(ddim: DDIM, data: torch.Tensor, noise: torch.Tensor,
                            t_eval: int, device, batch_size: int = 256) -> float:
    """Mean denoising MSE over a dataset at the fixed step ``t_eval``."""
    ddim.eval()
    total = 0.0
    for start in range(0, len(data), batch_size):
        stop = min(start + batch_size, len(data))
        loss = ddim.denoising_loss(
            data[start:stop].to(device), t_eval, noise[start:stop].to(device)
        )
        total += loss.item() * (stop - start)
    return total / len(data)
