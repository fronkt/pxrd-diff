"""1D ResNet encoder for PXRD intensity vectors.

Returns both a global embedding (B, d_model) for auxiliary tasks and
multi-resolution feature maps (B, L_total, d_model) for cross-attention
conditioning in the denoiser.
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
    """1D ResNet: (B, n_bins) -> global (B, d_model) + features (B, L, d_model)."""

    def __init__(self, n_bins: int = 4251, d_model: int = 256,
                 channels: tuple[int, ...] = (64, 128, 256, 256)):
        super().__init__()
        self.d_model = d_model
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels[0], 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
        )
        blocks, feat_projs = [], []
        in_ch = channels[0]
        for out_ch in channels:
            stride = 2 if out_ch != in_ch else 1
            blocks.append(ResBlock1d(in_ch, out_ch, stride))
            feat_projs.append(nn.Conv1d(out_ch, d_model, 1))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.feat_projs = nn.ModuleList(feat_projs)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(channels[-1], d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
        x = (x - mu) / sigma

        x = x.unsqueeze(1)
        x = self.stem(x)

        features = []
        for block, proj in zip(self.blocks, self.feat_projs):
            x = block(x)
            features.append(proj(x).transpose(1, 2))  # (B, L_i, d_model)

        global_emb = self.proj(self.pool(x).squeeze(-1))  # (B, d_model)
        multi_res = torch.cat(features, dim=1)             # (B, L_total, d_model)

        return global_emb, multi_res
