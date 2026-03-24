from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _to_float01(x: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    # assumes x is already roughly within [0, data_range]
    return torch.clamp(x / data_range, 0.0, 1.0)


@torch.no_grad()
def psnr(sr: torch.Tensor, hr: torch.Tensor, data_range: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    # sr/hr: [B,C,H,W]
    mse = torch.mean((sr - hr) ** 2, dim=(1, 2, 3))
    return 10.0 * torch.log10((data_range ** 2) / (mse + eps))


def _ssim_per_channel(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0, k1: float = 0.01, k2: float = 0.03) -> torch.Tensor:
    # Simplified SSIM with 3x3 uniform window (fast, dependency-free)
    # x,y: [B,1,H,W]
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    kernel = torch.ones((1, 1, 3, 3), device=x.device, dtype=x.dtype) / 9.0
    mu_x = F.conv2d(x, kernel, padding=1)
    mu_y = F.conv2d(y, kernel, padding=1)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=1) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=1) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=1) - mu_xy

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-12)
    return ssim_map.mean(dim=(1, 2, 3))


@torch.no_grad()
def ssim(sr: torch.Tensor, hr: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    # average over channels
    b, c, _, _ = sr.shape
    vals = []
    for ch in range(c):
        vals.append(_ssim_per_channel(sr[:, ch:ch+1], hr[:, ch:ch+1], data_range=data_range))
    return torch.stack(vals, dim=1).mean(dim=1)


def sam(sr: torch.Tensor, hr: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Spectral Angle Mapper distance (radians), averaged over pixels and batch
    # sr/hr: [B,C,H,W]
    b, c, h, w = sr.shape
    sr_v = sr.reshape(b, c, -1)
    hr_v = hr.reshape(b, c, -1)
    dot = torch.sum(sr_v * hr_v, dim=1)                          # [B,HW]
    sr_n = torch.sqrt(torch.sum(sr_v ** 2, dim=1) + eps)         # [B,HW]
    hr_n = torch.sqrt(torch.sum(hr_v ** 2, dim=1) + eps)
    cos = torch.clamp(dot / (sr_n * hr_n + eps), -1.0, 1.0)
    ang = torch.acos(cos)                                        # [B,HW]
    return ang.mean(dim=1)                                       # [B]


def ergas(sr: torch.Tensor, hr: torch.Tensor, scale: int, eps: float = 1e-8) -> torch.Tensor:
    # ERGAS: 100/scale * sqrt( mean_k ( RMSE_k^2 / mean_k(hr)^2 ) )
    # sr/hr: [B,C,H,W]
    b, c, _, _ = sr.shape
    rmse_k = torch.sqrt(torch.mean((sr - hr) ** 2, dim=(2, 3)) + eps)     # [B,C]
    mean_k = torch.mean(hr, dim=(2, 3)).abs() + eps                       # [B,C]
    ratio = (rmse_k / mean_k) ** 2                                        # [B,C]
    return (100.0 / float(scale)) * torch.sqrt(torch.mean(ratio, dim=1) + eps)  # [B]
