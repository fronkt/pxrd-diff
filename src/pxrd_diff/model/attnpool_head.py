"""Position-aware pooling lattice head (JAC HAT5032 revision, referee 2 point 2).

The submitted paper attributes the lattice failure to the encoder's global
average pooling, which discards absolute peak positions. Referee 2 asked for a
controlled ablation in which everything is held fixed and ONLY the pooling
strategy changes. This head is the position-aware arm of that ablation.

It reads the encoder's multi-resolution feature map (B, L, d) instead of the
globally pooled vector g. Every token receives (i) a sinusoidal encoding of its
absolute position on the 2θ axis (the quantity global pooling throws away) and
(ii) a learned embedding of the resolution level it came from. K learned
queries then attend over the sequence and a small MLP maps the K pooled
vectors to lattice parameters, sigmoid-bounded to the same physical ranges as
ConstrainedLatHead and PeakAugmentedLatHead so that the three ablation arms
share one output parameterisation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _sinusoidal(pos: torch.Tensor, d_model: int, max_pos: float = 4251.0) -> torch.Tensor:
    """pos: (L,) in [0, 1] -> (L, d_model) sinusoidal encoding of pos * max_pos."""
    x = pos.unsqueeze(-1) * max_pos                                        # (L, 1)
    i = torch.arange(0, d_model, 2, device=pos.device, dtype=pos.dtype)   # (d/2,)
    div = torch.exp(-math.log(10000.0) * i / d_model)                      # (d/2,)
    pe = torch.zeros(pos.shape[0], d_model, device=pos.device, dtype=pos.dtype)
    pe[:, 0::2] = torch.sin(x * div)
    pe[:, 1::2] = torch.cos(x * div)
    return pe


class AttnPoolLatHead(nn.Module):
    def __init__(self, d_model: int = 256, n_queries: int = 4, n_heads: int = 4,
                 hidden: int = 256, n_levels: int = 8,
                 len_min: float = 2.0, len_max: float = 20.0,
                 ang_min: float = 30.0, ang_max: float = 150.0):
        super().__init__()
        self.d_model = d_model
        self.queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.level_emb = nn.Embedding(n_levels, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Sequential(
            nn.Linear(n_queries * d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 6),
        )
        self.len_min, self.len_max = len_min, len_max
        self.ang_min, self.ang_max = ang_min, ang_max
        self._pe_cache: dict = {}

    def _positions(self, level_lengths: list[int], device, dtype):
        key = (tuple(level_lengths), str(device), str(dtype))
        if key not in self._pe_cache:
            pos = torch.cat([(torch.arange(n, device=device, dtype=dtype) + 0.5) / n
                             for n in level_lengths])                         # (L,)
            lvl = torch.cat([torch.full((n,), i, device=device, dtype=torch.long)
                             for i, n in enumerate(level_lengths)])           # (L,)
            self._pe_cache[key] = (_sinusoidal(pos, self.d_model), lvl)
        return self._pe_cache[key]

    def forward(self, multi_res: torch.Tensor, level_lengths: list[int]) -> torch.Tensor:
        """multi_res: (B, L, d) encoder feature map; level_lengths: [L_1, ..., L_k], sum = L."""
        B, L, d = multi_res.shape
        assert sum(level_lengths) == L, (sum(level_lengths), L)
        pe, lvl = self._positions(level_lengths, multi_res.device, multi_res.dtype)
        x = self.norm(multi_res + pe.unsqueeze(0) + self.level_emb(lvl).unsqueeze(0))
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        pooled, _ = self.attn(q, x, x, need_weights=False)                  # (B, K, d)
        raw = self.out(pooled.reshape(B, -1))
        lengths = self.len_min + (self.len_max - self.len_min) * torch.sigmoid(raw[..., :3])
        angles = self.ang_min + (self.ang_max - self.ang_min) * torch.sigmoid(raw[..., 3:])
        return torch.cat([lengths, angles], dim=-1)
