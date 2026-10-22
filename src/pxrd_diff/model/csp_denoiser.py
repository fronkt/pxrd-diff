"""Periodic denoiser for the JAC R2 redesign (model_family "csp").

Message passing in the style of DiffCSP's CSPNet (Jiao et al. 2023): every pair
of real atoms exchanges a message built from the two atom states, Fourier
features of the fractional difference f_j - f_i, and invariant features of the
cell. sin/cos(2 pi k (f_j - f_i)) is periodic and unchanged by a global
translation, so the output is invariant to the arbitrary CIF origin and to
integer shifts of single coordinates, and equivariant to atom permutations.
There is no absolute-coordinate embedding (the v10-v24 coord_proj was neither).

PXRD conditioning: the LayerNorm-ed global pattern embedding enters the FiLM
conditioning vector with the time and the cell, and every layer cross-attends
from LayerNorm-ed atom states to LayerNorm-ed pattern tokens (the pre-norm block
from denoiser.py). A learned null pattern replaces the pattern for the
dropped-out fraction of training samples, so the same network also runs with
no pattern.

The cell enters through cell_given: 0 = the cell is being denoised jointly
(learned mode; the network sees the current noisy cell only), 1 = a clean cell
is supplied (oracle / indexer modes).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from pxrd_diff.diffusion import SinusoidalTimestepEmb
from pxrd_diff.model.denoiser import MAX_ATOMIC_NUM, PXRDCrossAttention

N_LAT_FEATS = 6


def lattice_features(lat_params: torch.Tensor) -> torch.Tensor:
    """(B, 6) a, b, c (A), alpha, beta, gamma (deg) -> rotation-invariant,
    O(1)-scaled features: log lengths and cosines of the angles."""
    lengths = lat_params[:, :3].clamp(min=0.5)
    angles = torch.deg2rad(lat_params[:, 3:].clamp(10.0, 170.0))
    return torch.cat([lengths.log(), angles.cos()], dim=-1)


class FourierFrac(nn.Module):
    """Periodic features of fractional differences: sin/cos(2 pi k d), k = 1..K."""

    def __init__(self, n_freq: int = 10):
        super().__init__()
        self.register_buffer("k", 2 * math.pi * torch.arange(1, n_freq + 1, dtype=torch.float32))
        self.dim = 6 * n_freq

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        x = d.unsqueeze(-1) * self.k                              # (..., 3, K)
        return torch.cat([x.sin(), x.cos()], dim=-1).flatten(-2)  # (..., 6K)


class CSPLayer(nn.Module):
    def __init__(self, d: int, edge_in: int, n_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * d + edge_in, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU())
        self.node_mlp = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, d))
        self.film = nn.Linear(d, 2 * d)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.cross_attn = PXRDCrossAttention(d, n_heads, prenorm=True)

    def forward(self, h, edge_feats, pair_mask, mask, cond, pxrd_feats):
        B, N, d = h.shape
        hn = self.norm(h)
        hi = hn.unsqueeze(2).expand(B, N, N, d)
        hj = hn.unsqueeze(1).expand(B, N, N, d)
        m = self.edge_mlp(torch.cat([hi, hj, edge_feats], dim=-1))
        pm = pair_mask.unsqueeze(-1).to(m.dtype)
        agg = (m * pm).sum(2) / pm.sum(2).clamp(min=1)
        upd = self.node_mlp(torch.cat([hn, agg], dim=-1))
        scale, shift = self.film(cond).unsqueeze(1).chunk(2, dim=-1)
        h = h + (upd * (1 + scale) + shift) * mask.unsqueeze(-1).to(h.dtype)
        return self.cross_attn(h, pxrd_feats, mask)


class CSPDenoiser(nn.Module):
    def __init__(self, d_model: int = 256, n_layers: int = 6, n_heads: int = 4,
                 n_freq: int = 10):
        super().__init__()
        d = d_model
        self.d_model = d
        self.fourier = FourierFrac(n_freq)
        self.atom_emb = nn.Embedding(MAX_ATOMIC_NUM + 1, d, padding_idx=0)
        self.time_emb = nn.Sequential(
            SinusoidalTimestepEmb(d), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.pxrd_norm = nn.LayerNorm(d)
        self.pxrd_proj = nn.Linear(d, d)
        # cell: invariant features + the normalised parameters + the cell_given flag
        self.lat_proj = nn.Linear(N_LAT_FEATS + 6 + 1, d)
        self.null_global = nn.Parameter(torch.zeros(d))
        self.null_token = nn.Parameter(torch.zeros(1, 1, d))
        self.layers = nn.ModuleList(
            [CSPLayer(d, self.fourier.dim + N_LAT_FEATS, n_heads) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d)
        self.coord_head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 3))
        self.lattice_head = nn.Sequential(
            nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 6))

    def forward(self, frac_t, atom_types, lat_params, lat_norm, t,
                pxrd_global, pxrd_feats, mask, cell_given, pattern_drop=None):
        """
        frac_t:      (B, N, 3) noisy fractional coordinates in [0, 1)
        atom_types:  (B, N) atomic numbers, 0 = padding
        lat_params:  (B, 6) physical cell the network sees (noisy in learned mode)
        lat_norm:    (B, 6) the same cell, normalised by the training statistics
        t:           (B,) diffusion time in [0, 1]
        pxrd_global: (B, d), pxrd_feats: (B, L, d) from PXRDEncoder
        mask:        (B, N) bool, real atoms
        cell_given:  (B,) bool, True = clean supplied cell
        pattern_drop:(B,) bool or None, True = replace the pattern by the null pattern
        Returns (normalised coordinate score (B, N, 3), lattice noise estimate (B, 6)).
        """
        B, N, _ = frac_t.shape
        if pattern_drop is not None:
            pd = pattern_drop.view(B, 1)
            pxrd_global = torch.where(pd, self.null_global.expand(B, -1), pxrd_global)
            pxrd_feats = torch.where(pd.unsqueeze(-1),
                                     self.null_token.expand(B, pxrd_feats.shape[1], -1), pxrd_feats)
        lat_f = lattice_features(lat_params)
        cond = (self.time_emb(t * 1000.0)
                + self.pxrd_proj(self.pxrd_norm(pxrd_global))
                + self.lat_proj(torch.cat([lat_f, lat_norm, cell_given.float().unsqueeze(-1)], dim=-1)))

        h = self.atom_emb(atom_types) + cond.unsqueeze(1)
        diff = frac_t.unsqueeze(1) - frac_t.unsqueeze(2)              # [b, i, j] = f_j - f_i
        edge_feats = torch.cat(
            [self.fourier(diff), lat_f[:, None, None, :].expand(B, N, N, N_LAT_FEATS)], dim=-1)
        eye = torch.eye(N, dtype=torch.bool, device=mask.device)
        pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1) & ~eye

        for layer in self.layers:
            h = layer(h, edge_feats, pair_mask, mask, cond, pxrd_feats)

        h = self.out_norm(h)
        m = mask.unsqueeze(-1).to(h.dtype)
        score = self.coord_head(h) * m
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        eps_lat = self.lattice_head(torch.cat([pooled, cond], dim=-1))
        return score, eps_lat
