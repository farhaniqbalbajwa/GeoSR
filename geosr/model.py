from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    # x: [B, H, W, C] -> windows: [num_windows*B, window_size, window_size, C]
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    # windows: [num_windows*B, window_size, window_size, C] -> x: [B, H, W, C]
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*nW, N, C] where N = window_size*window_size
        BnW, N, C = x.shape
        qkv = self.qkv(x).reshape(BnW, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # [3, BnW, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(BnW, N, C)
        out = self.proj(out)
        out = self.drop(out)
        return out


class SwinBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int, shift: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift = shift

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: [B, H*W, C]
        B, L, C = x.shape
        assert L == H * W

        x_img = x.view(B, H, W, C)

        # cyclic shift
        if self.shift > 0:
            x_img = torch.roll(x_img, shifts=(-self.shift, -self.shift), dims=(1, 2))

        # partition windows
        ws = self.window_size
        x_win = window_partition(x_img, ws)                      # [BnW, ws, ws, C]
        x_win = x_win.view(-1, ws * ws, C)                       # [BnW, N, C]

        # attention
        x_win = self.norm1(x_win)
        attn_out = self.attn(x_win)

        # merge windows
        attn_out = attn_out.view(-1, ws, ws, C)
        x_img2 = window_reverse(attn_out, ws, H, W)              # [B,H,W,C]

        # reverse shift
        if self.shift > 0:
            x_img2 = torch.roll(x_img2, shifts=(self.shift, self.shift), dims=(1, 2))

        x = x + x_img2.view(B, H * W, C)

        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class GeoSRFM(nn.Module):
    """
    GeoSR-FM: a multispectral-ready SR "foundation-style" model.

    Modes:
      - forward_sr(lr) : supervised SR
      - forward_mae(x) : masked reconstruction pretraining on HR-like tiles
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: int = 4,
        dim: int = 96,
        depth: int = 8,
        num_heads: int = 6,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        mae_mask_ratio: float = 0.6,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale

        self.mae_mask_ratio = mae_mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))

        # Encoder (shallow feature extraction)
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        )

        # Transformer backbone at fixed resolution (Swin-style window attention)
        blocks = []
        for i in range(depth):
            shift = 0 if (i % 2 == 0) else (window_size // 2)
            blocks.append(SwinBlock(dim, num_heads, window_size, shift, mlp_ratio, dropout))
        self.blocks = nn.ModuleList(blocks)
        self.backbone_norm = nn.LayerNorm(dim)

        # Decoder for SR (pixel shuffle)
        self.sr_head = nn.Sequential(
            nn.Conv2d(dim, dim * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(dim, out_channels, kernel_size=3, padding=1),
        )

        # Decoder for MAE-like masked reconstruction (predict original channels at same resolution)
        self.mae_head = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, out_channels, kernel_size=1),
        )

        nn.init.normal_(self.mask_token, std=0.02)

    def _forward_backbone(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [B, C, H, W]
        B, C, H, W = feat.shape
        x = feat.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)  # [B, HW, C]
        for blk in self.blocks:
            x = blk(x, H, W)
        x = self.backbone_norm(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x

    def forward_sr(self, lr: torch.Tensor) -> torch.Tensor:
        # lr: [B, in_channels, H, W] (H,W divisible by window_size)
        feat = self.enc(lr)
        feat = self._forward_backbone(feat)
        sr = self.sr_head(feat)
        return sr

    def forward_mae(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MAE-like masked reconstruction:
          - Randomly mask a subset of spatial tokens in feature space
          - Predict original channels (same resolution as x)
        Returns: (pred, mask) where mask is [B,1,H,W] with 1=masked
        """
        B, _, H, W = x.shape
        feat = self.enc(x)
        C = feat.shape[1]

        # Create random mask over spatial positions
        num_tokens = H * W
        num_mask = int(self.mae_mask_ratio * num_tokens)

        mask = torch.zeros((B, num_tokens), device=x.device, dtype=torch.bool)
        for b in range(B):
            idx = torch.randperm(num_tokens, device=x.device)[:num_mask]
            mask[b, idx] = True

        feat_tokens = feat.permute(0, 2, 3, 1).contiguous().view(B, num_tokens, C)
        masked_tokens = feat_tokens.clone()
        masked_tokens[mask] = self.mask_token.expand(mask.sum(), C)

        feat_masked = masked_tokens.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        out_feat = self._forward_backbone(feat_masked)
        pred = self.mae_head(out_feat)

        mask_img = mask.view(B, 1, H, W).float()
        return pred, mask_img

    def forward(self, x: torch.Tensor, mode: str = "sr") -> torch.Tensor:
        if mode == "sr":
            return self.forward_sr(x)
        raise ValueError("Use forward_mae for MAE mode.")
