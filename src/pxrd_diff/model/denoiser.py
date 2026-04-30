"""Crystal structure denoiser with PXRD conditioning.

Takes a noisy crystal structure (fractional coords + atom types + lattice)
at diffusion timestep t, conditioned on a PXRD pattern embedding, and
predicts the noise epsilon for coordinates and lattice parameters.

Architecture:
  - Atom embeddings:  learned embedding for atomic number + noisy coord features
  - Pairwise features: RBF-encoded minimum-image periodic distances (all-pairs, <=20 atoms)
  - Message passing:  3 layers of simple equivariant message passing
  - Conditioning:     PXRD embedding + timestep embedding added to global context
  - Output heads:     per-atom coord noise (N, 3) + lattice noise (6,)

This is intentionally simple for the Phase 1 CPU prototype. Phase 2+ will
swap in a MACE-style backbone with higher-order equivariant features.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from pxrd_diff.diffusion import SinusoidalTimestepEmb

MAX_ATOMIC_NUM = 100
N_RBF = 64
RBF_CUTOFF = 12.0  # Angstrom


def rbf_expansion(d: torch.Tensor, n_rbf: int = N_RBF,
                  cutoff: float = RBF_CUTOFF) -> torch.Tensor:
    """Gaussian RBF expansion of distances. (B, N, N) -> (B, N, N, n_rbf)."""
    centers = torch.linspace(0.0, cutoff, n_rbf, device=d.device)
    gamma = 1.0 / (2 * (cutoff / n_rbf) ** 2)
    return torch.exp(-gamma * (d.unsqueeze(-1) - centers) ** 2)


def periodic_distances(frac_coords: torch.Tensor,
                       lattice: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
    """All-pairs minimum-image distance matrix.

    Args:
        frac_coords: (B, N, 3) fractional coordinates
        lattice:     (B, 3, 3) lattice matrix (rows = lattice vectors)
        mask:        (B, N) bool

    Returns:
        dist: (B, N, N) Cartesian distance matrix (masked positions get large distance)
    """
    delta = frac_coords.unsqueeze(2) - frac_coords.unsqueeze(1)   # (B, N, N, 3)
    delta = delta - delta.round()                                  # minimum image
    cart = torch.einsum("bnmd,bdc->bnmc", delta, lattice)          # (B, N, N, 3)
    dist = cart.norm(dim=-1)                                       # (B, N, N)

    # Mask out padded atom pairs with large distance
    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)              # (B, N, N)
    dist = dist.masked_fill(~pair_mask, RBF_CUTOFF + 1.0)
    return dist


class MessagePassingLayer(nn.Module):
    def __init__(self, d: int, n_rbf: int = N_RBF):
        super().__init__()
        self.edge_net = nn.Sequential(
            nn.Linear(n_rbf, d), nn.SiLU(), nn.Linear(d, d),
        )
        self.node_net = nn.Sequential(
            nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, d),
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, h: torch.Tensor, rbf: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        h:    (B, N, d) node features
        rbf:  (B, N, N, n_rbf) RBF-encoded distances
        mask: (B, N) bool
        """
        edge_w = self.edge_net(rbf)                                # (B, N, N, d)
        pair_mask = mask.unsqueeze(2).unsqueeze(-1).float()        # (B, N, 1, 1)
        agg = (edge_w * h.unsqueeze(1) * pair_mask).sum(dim=2)    # (B, N, d)
        out = self.node_net(torch.cat([h, agg], dim=-1))           # (B, N, d)
        return self.norm(h + out)


class CrystalDenoiser(nn.Module):
    """Predicts noise on fractional coords and lattice params.

    Args:
        d_model:   hidden dimension (shared with PXRD encoder)
        n_layers:  number of message-passing layers
        n_rbf:     number of RBF basis functions
        max_atoms: padding size
    """
    def __init__(self, d_model: int = 256, n_layers: int = 3,
                 n_rbf: int = N_RBF, max_atoms: int = 20):
        super().__init__()
        self.d_model = d_model

        # Embeddings
        self.atom_emb = nn.Embedding(MAX_ATOMIC_NUM + 1, d_model, padding_idx=0)
        self.coord_proj = nn.Linear(3, d_model)
        self.time_emb = nn.Sequential(
            SinusoidalTimestepEmb(d_model),
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model),
        )
        self.cond_proj = nn.Linear(d_model, d_model)  # project PXRD embedding

        # Message passing
        self.layers = nn.ModuleList(
            [MessagePassingLayer(d_model, n_rbf) for _ in range(n_layers)]
        )

        # Output heads
        self.coord_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 3),
        )
        self.lattice_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 6),
        )

    def forward(self, noisy_coords: torch.Tensor, atom_types: torch.Tensor,
                lattice: torch.Tensor, t: torch.Tensor,
                pxrd_emb: torch.Tensor, mask: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            noisy_coords: (B, N, 3)  fractional coords at time t
            atom_types:   (B, N)     atomic numbers
            lattice:      (B, 3, 3)  lattice matrix
            t:            (B,)       diffusion time in [0, 1]
            pxrd_emb:     (B, D)     PXRD encoder output
            mask:         (B, N)     True for real atoms

        Returns:
            eps_coord:   (B, N, 3) predicted coord noise
            eps_lattice: (B, 6)   predicted lattice noise
        """
        B, N, _ = noisy_coords.shape

        # Node features = atom_type_emb + coord_proj + time_emb + pxrd_emb
        h = self.atom_emb(atom_types)                              # (B, N, D)
        h = h + self.coord_proj(noisy_coords)
        t_emb = self.time_emb(t)                                   # (B, D)
        c_emb = self.cond_proj(pxrd_emb)                           # (B, D)
        h = h + (t_emb + c_emb).unsqueeze(1)                      # broadcast to atoms

        # Build pairwise features
        dist = periodic_distances(noisy_coords, lattice, mask)     # (B, N, N)
        rbf = rbf_expansion(dist)                                  # (B, N, N, n_rbf)

        # Message passing
        for layer in self.layers:
            h = layer(h, rbf, mask)

        # Predict noise
        eps_coord = self.coord_head(h)                             # (B, N, 3)
        # Lattice noise from masked mean-pool of node features
        h_masked = h * mask.unsqueeze(-1).float()
        h_pool = h_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).float()
        eps_lattice = self.lattice_head(h_pool)                    # (B, 6)

        return eps_coord, eps_lattice