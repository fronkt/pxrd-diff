"""Auxiliary lattice head.

Predicts normalized lattice parameters (a, b, c, α, β, γ) directly from the
PXRD encoder's global embedding via a small 2-layer MLP.

Trained alongside the diffusion model in scripts/02_train.py as an MSE
auxiliary loss — the gradient nudges the encoder to capture peak positions
(d-spacings) faithfully, since lattice parameters are determined by Bragg's
law at the peak positions.

Phase 5 promotes this head from "training regularizer" to "inference-time
lattice predictor": the head's deterministic output is used in place of the
diffusion sampler's noisy lattice estimate, which has historically produced
many physically invalid combinations even after the v13 lattice-input fix.
"""
from __future__ import annotations

import torch.nn as nn


class AuxLatHead(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 6),
        )

    def forward(self, emb):
        return self.net(emb)
