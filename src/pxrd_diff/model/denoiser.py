"""Crystal structure denoiser with multi-resolution PXRD cross-attention.

Architecture:
  - Atom embeddings:  learned embedding for atomic number + noisy coord features
  - Pairwise features: RBF-encoded minimum-image periodic distances
  - Message passing:  N layers, each followed by cross-attention to PXRD features
  - FiLM conditioning from timestep at each layer (PXRD signal now via cross-attn)
  - Output heads:     per-atom coord noise (N, 3) + lattice noise (6,)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from pxrd_diff.diffusion import SinusoidalTimestepEmb

MAX_ATOMIC_NUM = 100
N_RBF = 64
RBF_CUTOFF = 12.0
N_WYCKOFF = 27  # 'a'..'z' + fallback


def rbf_expansion(d: torch.Tensor, n_rbf: int = N_RBF,
                  cutoff: float = RBF_CUTOFF) -> torch.Tensor:
    centers = torch.linspace(0.0, cutoff, n_rbf, device=d.device)
    gamma = 1.0 / (2 * (cutoff / n_rbf) ** 2)
    return torch.exp(-gamma * (d.unsqueeze(-1) - centers) ** 2)


# Lattice translations searched by the minimum-image mode: every image within
# +-2 cells of the component-wise wrapped difference (125 shifts).
_IMAGE_SHIFTS = torch.tensor(
    [[i, j, k] for i in range(-2, 3) for j in range(-2, 3) for k in range(-2, 3)],
    dtype=torch.float32,
)


def periodic_distances(frac_coords: torch.Tensor,
                       lattice: torch.Tensor,
                       mask: torch.Tensor,
                       min_image: bool = False) -> torch.Tensor:
    """Pairwise periodic Cartesian distances (B, N, N).

    min_image=False is the featurisation of hat5032 v1/R1 (v10-v22): fractional
    differences are wrapped component-wise to [-0.5, 0.5] and then transformed,
    which is the minimum image only for orthogonal cells (JAC R2 Q2: wrong
    image for 16 % of MP-20 test pairs). min_image=True keeps the shortest of
    the 125 images within +-2 cells of that wrapped difference.
    """
    delta = frac_coords.unsqueeze(2) - frac_coords.unsqueeze(1)
    delta = delta - delta.round()
    cart = torch.einsum("bnmd,bdc->bnmc", delta, lattice)
    if min_image:
        B, N = cart.shape[:2]
        shifts = _IMAGE_SHIFTS.to(cart)
        shift_cart = torch.einsum("kd,bdc->bkc", shifts, lattice)         # (B, K, 3)
        # |cart + s|^2 = |cart|^2 + 2 cart.s + |s|^2, minimised over the shifts
        sq = (2.0 * torch.einsum("bnmc,bkc->bnmk", cart, shift_cart)
              + shift_cart.pow(2).sum(-1)[:, None, None, :])
        best = sq.argmin(dim=-1).reshape(B, N * N, 1).expand(-1, -1, 3)
        cart = cart + shift_cart.gather(1, best).reshape(B, N, N, 3)
    dist = cart.norm(dim=-1)
    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
    dist = dist.masked_fill(~pair_mask, RBF_CUTOFF + 1.0)
    return dist


class PXRDCrossAttention(nn.Module):
    """Atoms attend to multi-resolution PXRD feature maps."""
    def __init__(self, d: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, h: torch.Tensor, pxrd_feats: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        # h: (B, N, d), pxrd_feats: (B, L, d), mask: (B, N)
        key_padding_mask = None  # PXRD features are dense (no padding)
        out, _ = self.attn(h, pxrd_feats, pxrd_feats,
                           key_padding_mask=key_padding_mask)
        out = out * mask.unsqueeze(-1).float()
        return self.norm(h + out)


class MessagePassingLayer(nn.Module):
    def __init__(self, d: int, n_rbf: int = N_RBF, n_heads: int = 4):
        super().__init__()
        self.edge_net = nn.Sequential(
            nn.Linear(n_rbf, d), nn.SiLU(), nn.Linear(d, d),
        )
        self.node_net = nn.Sequential(
            nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, d),
        )
        # FiLM: time conditioning only (PXRD now handled by cross-attention)
        self.film = nn.Linear(d, 2 * d)
        self.norm = nn.LayerNorm(d)
        self.cross_attn = PXRDCrossAttention(d, n_heads)

    def forward(self, h: torch.Tensor, rbf: torch.Tensor,
                mask: torch.Tensor, t_cond: torch.Tensor,
                pxrd_feats: torch.Tensor) -> torch.Tensor:
        edge_w = self.edge_net(rbf)
        pair_mask = mask.unsqueeze(2).unsqueeze(-1).float()
        agg = (edge_w * h.unsqueeze(1) * pair_mask).sum(dim=2)
        out = self.node_net(torch.cat([h, agg], dim=-1))
        scale, shift = self.film(t_cond).unsqueeze(1).chunk(2, dim=-1)
        out = out * (1 + scale) + shift
        h = self.norm(h + out)
        h = self.cross_attn(h, pxrd_feats, mask)
        return h


class CrystalDenoiser(nn.Module):
    def __init__(self, d_model: int = 256, n_layers: int = 3,
                 n_rbf: int = N_RBF, max_atoms: int = 20, n_heads: int = 4,
                 min_image: bool = False):
        super().__init__()
        self.d_model = d_model
        self.min_image = min_image  # see periodic_distances; not a parameter

        self.atom_emb = nn.Embedding(MAX_ATOMIC_NUM + 1, d_model, padding_idx=0)
        self.wyckoff_emb = nn.Embedding(N_WYCKOFF, d_model)
        nn.init.zeros_(self.wyckoff_emb.weight)  # so it has no effect at init
        self.coord_proj = nn.Linear(3, d_model)
        self.time_emb = nn.Sequential(
            SinusoidalTimestepEmb(d_model),
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model),
        )
        self.lat_in_proj = nn.Linear(6, d_model)

        self.layers = nn.ModuleList(
            [MessagePassingLayer(d_model, n_rbf, n_heads) for _ in range(n_layers)]
        )

        self.coord_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 3),
        )
        # Lattice head sees: pooled atom features + PXRD global + noisy lat + time
        self.lattice_head = nn.Sequential(
            nn.Linear(4 * d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, 6),
        )
        # Distance matrix head: predicts periodic Cartesian distances between atom pairs
        self.dist_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, noisy_coords, atom_types, lattice, t,
                pxrd_global, pxrd_feats, mask, noisy_lat_p,
                wyckoff=None, return_dist: bool = False):
        """
        wyckoff: (B, N) Wyckoff site IDs in [0, N_WYCKOFF). None disables.
        noisy_lat_p: (B, 6) — noisy normalized lattice parameters being denoised
        pxrd_global: (B, d) — global PXRD embedding
        pxrd_feats:  (B, L, d) — multi-resolution features for cross-attention
        return_dist: if True, also return predicted pairwise distance matrix (B, N, N)
        """
        B, N, _ = noisy_coords.shape

        h = self.atom_emb(atom_types) + self.coord_proj(noisy_coords)
        if wyckoff is not None:
            h = h + self.wyckoff_emb(wyckoff)
        t_cond = self.time_emb(t)

        dist = periodic_distances(noisy_coords, lattice, mask,
                                  min_image=self.min_image)
        rbf = rbf_expansion(dist)

        for layer in self.layers:
            h = layer(h, rbf, mask, t_cond, pxrd_feats)

        eps_coord = self.coord_head(h)
        h_masked = h * mask.unsqueeze(-1).float()
        h_pool = h_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).float()
        lat_in = self.lat_in_proj(noisy_lat_p)
        lat_features = torch.cat([h_pool, pxrd_global, lat_in, t_cond], dim=-1)
        eps_lattice = self.lattice_head(lat_features)

        if return_dist:
            h_i = h.unsqueeze(2).expand(-1, -1, N, -1)
            h_j = h.unsqueeze(1).expand(-1, N, -1, -1)
            d_pair = self.dist_head(torch.cat([h_i, h_j], dim=-1)).squeeze(-1)
            return eps_coord, eps_lattice, d_pair

        return eps_coord, eps_lattice
