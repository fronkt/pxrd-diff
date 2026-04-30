"""1D ResNet encoder for PXRD intensity vectors.

Maps (B, n_bins) -> (B, d_model). Four residual blocks with progressive
downsampling, followed by global average pool and a linear projection.
The output embedding conditions the diffusion denoiser on the input pattern.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
        )
        self.skip = (nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False)
                     if in_ch != out_ch or stride != 1 else nn.Identity())
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.skip(x))


class PXRDEncoder(nn.Module):
    """1D ResNet: (B, n_bins) -> (B, d_model)."""

    def __init__(self, n_bins: int = 4251, d_model: int = 256,
                 channels: tuple[int, ...] = (64, 128, 256, 256)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels[0], 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
        )
        blocks = []
        in_ch = channels[0]
        for out_ch in channels:
            stride = 2 if out_ch != in_ch else 1
            blocks.append(ResBlock1d(in_ch, out_ch, stride))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(channels[-1], d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)         # (B, 1, n_bins)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.proj(x)         # (B, d_model)