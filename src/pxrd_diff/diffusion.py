"""Diffusion process for crystal structures.

Handles two continuous channels:
  - Fractional coordinates (N, 3) on the flat torus T^3 = [0, 1)^3
  - Lattice parameters (6,) in R^6 (normalized by dataset statistics)

Uses a cosine noise schedule (Nichol & Dhariwal 2021). The model predicts
the noise epsilon added during the forward process; the loss is simple MSE
weighted by atom masks.

Atom types are held fixed (conditioned on true composition). Predicting types
is a Phase 2+ extension.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def cosine_alpha_bar(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule: alpha_bar(t) = cos^2(pi/2 * (t+s)/(1+s))."""
    return torch.cos(math.pi / 2 * (t + s) / (1 + s)) ** 2


class DiffusionProcess(nn.Module):
    """VP diffusion with cosine schedule on coords (torus) + lattice (R^6)."""

    def __init__(self, T: int = 1000):
        super().__init__()
        self.T = T

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample continuous t in [0, 1]."""
        return torch.rand(batch_size, device=device)

    def forward_q(self, x0: torch.Tensor, t: torch.Tensor,
                  wrap: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t ~ q(x_t | x_0).

        Args:
            x0: clean data, any shape (..., D)
            t:  (B,) or broadcastable time in [0, 1]
            wrap: if True, wrap result to [0, 1) (for fractional coords)

        Returns:
            x_t: noisy data, same shape as x0
            eps: noise that was added
        """
        alpha_bar = cosine_alpha_bar(t)
        while alpha_bar.dim() < x0.dim():
            alpha_bar = alpha_bar.unsqueeze(-1)

        eps = torch.randn_like(x0)
        x_t = alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * eps

        if wrap:
            x_t = x_t % 1.0    # wrap to torus [0, 1)

        return x_t, eps

    def loss(self, eps_pred: torch.Tensor, eps_true: torch.Tensor,
             mask: torch.Tensor | None = None,
             periodic: bool = False) -> torch.Tensor:
        """Masked MSE loss. If periodic, wrap diff to [-0.5, 0.5] (torus)."""
        diff = eps_pred - eps_true
        if periodic:
            diff = (diff + 0.5) % 1.0 - 0.5
        sq = diff ** 2
        if mask is not None:
            mask = mask.unsqueeze(-1).float()
            return (sq * mask).sum() / mask.sum().clamp(min=1) / sq.shape[-1]
        return sq.mean()


class SinusoidalTimestepEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)      # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1) # (B, dim)