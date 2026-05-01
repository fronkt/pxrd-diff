"""Differentiable Debye scattering for PXRD pattern generation.

Computes powder XRD intensity from fractional coordinates + lattice using the
Debye equation, fully differentiable w.r.t. atom positions for use as a
physics-informed training loss.

I(Q) = Σᵢ Σⱼ fᵢ(Q)·fⱼ(Q)·sin(Q·rᵢⱼ)/(Q·rᵢⱼ) · exp(-B·s²)

Atomic form factors use the 4-Gaussian + c parameterization from International
Tables, extracted from pymatgen at init time.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_ff_table(max_z: int = 100) -> torch.Tensor:
    """Extract atomic form factor coefficients from pymatgen.

    Returns (max_z+1, 4, 2) tensor: ff_table[Z] = [[a1,b1],[a2,b2],[a3,b3],[a4,b4]].
    f(s) = Σₖ aₖ exp(-bₖ s²), where s = sin(θ)/λ.
    """
    from pymatgen.analysis.diffraction.xrd import ATOMIC_SCATTERING_PARAMS
    from pymatgen.core.periodic_table import Element

    table = torch.zeros(max_z + 1, 4, 2)
    for z in range(1, max_z + 1):
        try:
            el = Element.from_Z(z)
            if el.symbol in ATOMIC_SCATTERING_PARAMS:
                pairs = ATOMIC_SCATTERING_PARAMS[el.symbol]  # [[a1,b1], ...]
                for k, (a, b) in enumerate(pairs[:4]):
                    table[z, k, 0] = a
                    table[z, k, 1] = b
        except Exception:
            continue
    return table


class DebyePXRD(nn.Module):
    """Differentiable Debye scattering → PXRD pattern.

    Uses a coarsened 2θ grid (n_bins=256) for efficiency as a loss signal.
    Peak positions and relative intensities match the full-resolution pattern
    closely enough for gradient-based optimization.
    """

    def __init__(self, two_theta_min: float = 5.0, two_theta_max: float = 90.0,
                 n_bins: int = 256, wavelength: float = 1.54184,
                 b_iso: float = 0.5, peak_sigma: float = 0.3):
        super().__init__()
        self.n_bins = n_bins
        self.wavelength = wavelength
        self.peak_sigma = peak_sigma

        two_theta = torch.linspace(two_theta_min, two_theta_max, n_bins)
        theta_rad = two_theta * math.pi / 360.0
        Q = 4 * math.pi * torch.sin(theta_rad) / wavelength
        s = torch.sin(theta_rad) / wavelength
        s_sq = s ** 2

        # Lorentz-polarization factor: (1 + cos²(2θ)) / (sin²(θ)·cos(θ))
        two_theta_rad = two_theta * math.pi / 180.0
        lp = (1 + torch.cos(two_theta_rad) ** 2) / (
            torch.sin(theta_rad) ** 2 * torch.cos(theta_rad)
        ).clamp(min=1e-6)

        dw = torch.exp(-b_iso * s_sq)

        self.register_buffer("two_theta", two_theta)
        self.register_buffer("Q", Q)
        self.register_buffer("s_sq", s_sq)
        self.register_buffer("lp", lp)
        self.register_buffer("dw", dw)

        ff_table = _build_ff_table()
        self.register_buffer("ff_table", ff_table)

    def form_factors(self, atom_types: torch.Tensor) -> torch.Tensor:
        """Compute f(s) for each atom at each Q point.

        Args:
            atom_types: (B, N) atomic numbers

        Returns:
            (B, N, n_bins) form factor values, including DW factor
        """
        coeffs = self.ff_table[atom_types]   # (B, N, 4, 2)
        a = coeffs[..., 0]                   # (B, N, 4)
        b = coeffs[..., 1]                   # (B, N, 4)

        # f(s) = Σₖ aₖ exp(-bₖ s²)
        s_sq = self.s_sq.view(1, 1, -1, 1)  # (1, 1, n_bins, 1)
        f = (a.unsqueeze(2) * torch.exp(-b.unsqueeze(2) * s_sq)).sum(-1)
        return f * self.dw.view(1, 1, -1)    # (B, N, n_bins)

    def forward(self, frac_coords: torch.Tensor, atom_types: torch.Tensor,
                lattice: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute differentiable PXRD from crystal structure.

        Args:
            frac_coords: (B, N, 3) fractional coordinates (differentiable)
            atom_types:  (B, N) atomic numbers
            lattice:     (B, 3, 3) lattice matrix (row vectors)
            mask:        (B, N) bool atom mask

        Returns:
            (B, n_bins) normalized PXRD pattern
        """
        B, N, _ = frac_coords.shape
        f = self.form_factors(atom_types)  # (B, N, n_bins)

        # Pairwise minimum-image distances (differentiable w.r.t. frac_coords)
        delta = frac_coords.unsqueeze(2) - frac_coords.unsqueeze(1)  # (B, N, N, 3)
        delta = delta - delta.round()
        cart = torch.einsum("bnmd,bdc->bnmc", delta, lattice)
        r_ij = cart.norm(dim=-1)  # (B, N, N)

        # Mask: real atom pairs, exclude self
        pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
        eye = torch.eye(N, device=mask.device, dtype=torch.bool).unsqueeze(0)
        pair_mask = pair_mask & ~eye

        # Debye equation: sin(Q·r)/(Q·r) using torch.sinc
        # torch.sinc(x) = sin(πx)/(πx), so sinc(Qr/π) = sin(Qr)/Qr
        Qr = self.Q.view(1, 1, 1, -1) * r_ij.unsqueeze(-1)  # (B, N, N, n_bins)
        sinc_Qr = torch.sinc(Qr / math.pi)

        fi_fj = f.unsqueeze(2) * f.unsqueeze(1)  # (B, N, N, n_bins)
        I_pairs = (fi_fj * sinc_Qr * pair_mask.unsqueeze(-1).float()).sum(dim=(1, 2))

        # Self-scattering: Σᵢ fᵢ² (sinc(0) = 1)
        I_self = (f ** 2 * mask.unsqueeze(-1).float()).sum(dim=1)

        I = (I_pairs + I_self) * self.lp.unsqueeze(0)

        # Normalize to [0, 1]
        I_max = I.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        return I / I_max


def debye_pxrd_loss(pred_pattern: torch.Tensor, target_pattern: torch.Tensor,
                    n_bins_debye: int = 256) -> torch.Tensor:
    """Compute loss between Debye-predicted and target PXRD patterns.

    Downsamples the target pattern to match the Debye grid resolution,
    then computes 1 - Pearson correlation (scale/shift invariant).
    """
    if target_pattern.shape[-1] != n_bins_debye:
        target_ds = F.interpolate(
            target_pattern.unsqueeze(1), size=n_bins_debye, mode="linear",
            align_corners=True,
        ).squeeze(1)
    else:
        target_ds = target_pattern

    # Pearson correlation loss (1 - r): invariant to scale/shift
    pred_z = pred_pattern - pred_pattern.mean(dim=-1, keepdim=True)
    targ_z = target_ds - target_ds.mean(dim=-1, keepdim=True)
    pred_n = pred_z / pred_z.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    targ_n = targ_z / targ_z.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    pearson = (pred_n * targ_n).sum(dim=-1)
    return (1 - pearson).mean()
