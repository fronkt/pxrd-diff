"""Constrained lattice head + space-group head.

Phase 5B (ConstrainedLatHead) addresses the failure of Phase 5 Path A: the
unconstrained AuxLatHead produced 99.8 % "valid-looking" lattices but with
~1 Å length MAE and ~13° angle MAE — enough to flatten the predicted PXRD's
Pearson against the target to ~0. The constrained head outputs lattice
parameters in physical units (Å for lengths, ° for angles) bounded to a
physically plausible range via sigmoids:
    a, b, c    ∈ [len_min, len_max]    (default 2.0–20.0 Å)
    α, β, γ    ∈ [ang_min, ang_max]    (default 30°–150°)
This both prevents pathological predictions and gives the optimizer a
better-conditioned loss landscape than the unbounded AuxLatHead.

Phase 6.1 (SpaceGroupHead) adds a 230-way classifier on the same encoder
embedding. Trained with cross-entropy on the dataset's spacegroup field.

Phase 6.2 (apply_sg_constraints) enforces crystal-system equalities at
inference (e.g. cubic ⇒ a=b=c, all angles 90°) given a predicted SG number.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstrainedLatHead(nn.Module):
    """Predict (a, b, c, α°, β°, γ°) in physical units, bounded by sigmoids.

    The head outputs values directly in physical units, NOT normalized space.
    Training loss should compute MSE in normalized space against the same
    `lat_mean / lat_std` the diffusion lattice uses, so the two heads see
    consistent gradient magnitudes.
    """

    def __init__(self, d_model: int = 256,
                 len_min: float = 2.0, len_max: float = 20.0,
                 ang_min: float = 30.0, ang_max: float = 150.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 6),
        )
        self.len_min = len_min
        self.len_max = len_max
        self.ang_min = ang_min
        self.ang_max = ang_max

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        raw = self.net(emb)                                   # (B, 6) unbounded
        lengths = self.len_min + (self.len_max - self.len_min) * torch.sigmoid(raw[..., :3])
        angles = self.ang_min + (self.ang_max - self.ang_min) * torch.sigmoid(raw[..., 3:])
        return torch.cat([lengths, angles], dim=-1)           # (B, 6) physical units


class SpaceGroupHead(nn.Module):
    """Predict the 230-way space group from the PXRD encoder embedding."""

    def __init__(self, d_model: int = 256, n_classes: int = 230):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb)                                  # (B, 230) logits


def sg_classification_loss(logits: torch.Tensor, sg_numbers: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss for SG prediction.

    Args:
        logits: (B, 230) raw logits over 0-indexed space group classes
        sg_numbers: (B,) integer space group numbers in [1, 230]

    Returns:
        scalar mean cross-entropy loss
    """
    targets = (sg_numbers - 1).clamp(min=0, max=229).long()
    return F.cross_entropy(logits, targets)


def sg_topk_accuracy(logits: torch.Tensor, sg_numbers: torch.Tensor,
                     k: int = 1) -> torch.Tensor:
    """Top-k accuracy for SG prediction. Returns scalar tensor in [0, 1]."""
    topk = logits.topk(k, dim=-1).indices + 1                 # (B, k) in [1, 230]
    hit = (topk == sg_numbers.unsqueeze(-1)).any(dim=-1)
    return hit.float().mean()


def apply_sg_constraints(lat_params: torch.Tensor, sg_numbers: torch.Tensor,
                         ) -> torch.Tensor:
    """Project lattice parameters onto the crystal-system manifold for each SG.

    The crystal system is determined by SG number using the standard
    crystallographic convention. Free parameters of the predicted lp are kept;
    constrained ones are replaced by their canonical values (e.g. 90° angles
    for orthorhombic+) or set equal to peer values (e.g. a=b for tetragonal).
    For length-equality constraints, the average of the relevant lengths is
    used so the gradient flows through all of them.

    Crystal systems:
        triclinic     (sg ∈ [1, 2])         no constraint
        monoclinic    (sg ∈ [3, 15])        α=γ=90°, β free
        orthorhombic  (sg ∈ [16, 74])       α=β=γ=90°
        tetragonal    (sg ∈ [75, 142])      a=b, α=β=γ=90°
        trigonal      (sg ∈ [143, 167])     a=b, α=β=90°, γ=120°  (hex axes)
        hexagonal     (sg ∈ [168, 194])     a=b, α=β=90°, γ=120°
        cubic         (sg ∈ [195, 230])     a=b=c, α=β=γ=90°

    Args:
        lat_params: (B, 6) [a, b, c, α, β, γ] in physical units (Å, °).
        sg_numbers: (B,) integer space group numbers in [1, 230].

    Returns:
        (B, 6) lattice parameters with crystal-system constraints applied.
    """
    a, b, c = lat_params[..., 0], lat_params[..., 1], lat_params[..., 2]
    al, be, ga = lat_params[..., 3], lat_params[..., 4], lat_params[..., 5]

    sg = sg_numbers.long()
    triclinic = sg <= 2
    monoclinic = (sg >= 3) & (sg <= 15)
    # orthorhombic: 16-74 (handled by the "default 90° + free lengths" path)
    tetragonal = (sg >= 75) & (sg <= 142)
    trigonal = (sg >= 143) & (sg <= 167)
    hexagonal = (sg >= 168) & (sg <= 194)
    cubic = sg >= 195

    # Average pairs / triples used for length-equality constraints.
    avg_ab = (a + b) / 2.0
    avg_abc = (a + b + c) / 3.0

    deg90 = torch.full_like(al, 90.0)
    deg120 = torch.full_like(al, 120.0)

    # Default to "all angles = 90°" (orthorhombic and stricter).
    out_al = deg90.clone()
    out_be = deg90.clone()
    out_ga = deg90.clone()
    out_a, out_b, out_c = a.clone(), b.clone(), c.clone()

    # Triclinic: no constraint — restore originals.
    out_a = torch.where(triclinic, a, out_a)
    out_b = torch.where(triclinic, b, out_b)
    out_c = torch.where(triclinic, c, out_c)
    out_al = torch.where(triclinic, al, out_al)
    out_be = torch.where(triclinic, be, out_be)
    out_ga = torch.where(triclinic, ga, out_ga)

    # Monoclinic: only β is free; α and γ stay at 90°.
    out_be = torch.where(monoclinic, be, out_be)

    # Tetragonal: a = b, all angles 90° (already set).
    out_a = torch.where(tetragonal, avg_ab, out_a)
    out_b = torch.where(tetragonal, avg_ab, out_b)

    # Trigonal (hex setting): a = b, γ = 120°, others 90°.
    out_a = torch.where(trigonal, avg_ab, out_a)
    out_b = torch.where(trigonal, avg_ab, out_b)
    out_ga = torch.where(trigonal, deg120, out_ga)

    # Hexagonal: a = b, γ = 120°, others 90°.
    out_a = torch.where(hexagonal, avg_ab, out_a)
    out_b = torch.where(hexagonal, avg_ab, out_b)
    out_ga = torch.where(hexagonal, deg120, out_ga)

    # Cubic: a = b = c, all angles 90°.
    out_a = torch.where(cubic, avg_abc, out_a)
    out_b = torch.where(cubic, avg_abc, out_b)
    out_c = torch.where(cubic, avg_abc, out_c)

    return torch.stack([out_a, out_b, out_c, out_al, out_be, out_ga], dim=-1)
