"""Differentiable PXRD via structure factor computation.

Computes Bragg peak intensities from fractional coordinates + lattice,
fully differentiable w.r.t. atom positions for use as a physics-informed loss.

I(hkl) ∝ |F(hkl)|² × LP(θ)
F(hkl) = Σⱼ fⱼ(s) · exp(2πi(h·xⱼ + k·yⱼ + l·zⱼ))

Peaks are placed on a 2θ grid with Gaussian broadening to produce a
continuous, differentiable PXRD pattern.
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
                pairs = ATOMIC_SCATTERING_PARAMS[el.symbol]
                for k, (a, b) in enumerate(pairs[:4]):
                    table[z, k, 0] = a
                    table[z, k, 1] = b
        except Exception:
            continue
    return table


def _enumerate_hkl(max_idx: int = 10) -> torch.Tensor:
    """Generate all (h,k,l) with |h|,|k|,|l| <= max_idx, excluding (0,0,0).

    Returns (M, 3) integer tensor.
    """
    r = torch.arange(-max_idx, max_idx + 1)
    h, k, l = torch.meshgrid(r, r, r, indexing="ij")
    hkl = torch.stack([h.flatten(), k.flatten(), l.flatten()], dim=-1)
    nonzero = hkl.abs().sum(dim=-1) > 0
    return hkl[nonzero]


class DiffPXRD(nn.Module):
    """Differentiable PXRD via Bragg structure factors.

    Enumerates all (hkl) reflections, computes |F(hkl)|² from fractional
    coordinates, and places peaks on a 2θ grid with Gaussian broadening.
    """

    def __init__(self, two_theta_min: float = 5.0, two_theta_max: float = 90.0,
                 n_bins: int = 512, wavelength: float = 1.54184,
                 hkl_max: int = 10, peak_fwhm: float = 0.1, b_iso: float = 0.5):
        super().__init__()
        self.wavelength = wavelength
        self.n_bins = n_bins
        sigma = peak_fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        self.register_buffer("sigma", torch.tensor(sigma))

        two_theta = torch.linspace(two_theta_min, two_theta_max, n_bins)
        self.register_buffer("two_theta", two_theta)

        hkl = _enumerate_hkl(hkl_max).float()
        self.register_buffer("hkl", hkl)  # (M, 3)

        ff_table = _build_ff_table()
        self.register_buffer("ff_table", ff_table)

        self.b_iso = b_iso

    def _form_factor_at_s(self, atom_types: torch.Tensor,
                          s_vals: torch.Tensor) -> torch.Tensor:
        """Compute f(s) for each atom at each reflection's s value.

        atom_types: (B, N)
        s_vals:     (B, M) sin(θ)/λ for each reflection

        Returns: (B, N, M) form factor values
        """
        coeffs = self.ff_table[atom_types]  # (B, N, 4, 2)
        a = coeffs[..., 0]                  # (B, N, 4)
        b = coeffs[..., 1]                  # (B, N, 4)

        s_sq = (s_vals ** 2).unsqueeze(1).unsqueeze(-1)  # (B, 1, M, 1)
        f = (a.unsqueeze(2) * torch.exp(-b.unsqueeze(2) * s_sq)).sum(-1)  # (B, N, M)

        # Debye-Waller factor
        dw = torch.exp(-self.b_iso * s_vals ** 2).unsqueeze(1)  # (B, 1, M)
        return f * dw

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
        B = frac_coords.shape[0]
        hkl = self.hkl  # (M, 3)
        M = hkl.shape[0]

        # Reciprocal lattice: b = 2π (a^T)^{-1}
        # For lattice matrix A where rows are a,b,c vectors:
        # reciprocal B = 2π inv(A)^T → d*_hkl = h*b1 + k*b2 + l*b3
        recip = 2 * math.pi * torch.linalg.inv(lattice).transpose(-1, -2)  # (B, 3, 3)

        # d*-vectors for each hkl: (B, M, 3) = hkl (M,3) @ recip (B,3,3)
        g_hkl = torch.einsum("md,bdc->bmc", hkl, recip)  # (B, M, 3)
        g_len = g_hkl.norm(dim=-1)  # (B, M) = 2π/d_hkl (physics convention)

        # d-spacings and 2θ positions
        d_hkl = 2 * math.pi / g_len.clamp(min=1e-8)  # (B, M)
        sin_theta = self.wavelength / (2 * d_hkl)  # sin(θ) = λ/(2d)

        # Filter: only reflections where |sin(θ)| < 1 (valid Bragg condition)
        valid = sin_theta.abs() < 1.0

        theta = torch.asin(sin_theta.clamp(-1 + 1e-7, 1 - 1e-7))  # (B, M)
        two_theta_hkl = 2 * theta * 180.0 / math.pi  # (B, M) in degrees

        # Filter: only reflections in our 2θ range
        in_range = (two_theta_hkl >= self.two_theta[0]) & (two_theta_hkl <= self.two_theta[-1])
        valid = valid & in_range

        # s = sin(θ)/λ for form factor evaluation
        s_vals = sin_theta / self.wavelength  # sin(θ)/λ — wait, sin_theta IS λ/(2d)
        # Actually s = sin(θ)/λ, and sin(θ) = λ/(2d), so s = 1/(2d) = g_len/2
        s_vals = g_len / (4 * math.pi)  # Q = 2π·g_len, s = Q/(4π) = g_len/(2) ...
        # Let me recalculate: g_len = 2π/d, so d = 2π/g_len
        # sin(θ) = λ/(2d) = λ·g_len/(4π)
        # s = sin(θ)/λ = g_len/(4π)
        s_vals = g_len / (4 * math.pi)

        # Form factors
        f = self._form_factor_at_s(atom_types, s_vals)  # (B, N, M)

        # Structure factor: F(hkl) = Σⱼ fⱼ exp(2πi h·xⱼ)
        # phase = 2π (hkl · frac_coords^T) → (B, M, N)
        phase = 2 * math.pi * torch.einsum("md,bnd->bmn", hkl, frac_coords)  # (B, M, N)

        # F = Σⱼ fⱼ exp(i·phase), with atom masking
        f_t = f.transpose(1, 2)  # (B, M, N)
        mask_f = mask.unsqueeze(1).float()  # (B, 1, N)

        F_real = (f_t * torch.cos(phase) * mask_f).sum(dim=-1)  # (B, M)
        F_imag = (f_t * torch.sin(phase) * mask_f).sum(dim=-1)  # (B, M)
        I_hkl = F_real ** 2 + F_imag ** 2  # (B, M)

        # Lorentz-polarization factor
        cos_2t = torch.cos(2 * theta)
        sin_t = torch.sin(theta).clamp(min=1e-6)
        cos_t = torch.cos(theta).clamp(min=1e-6)
        lp = (1 + cos_2t ** 2) / (sin_t ** 2 * cos_t)

        I_hkl = I_hkl * lp * valid.float()

        # Place peaks on 2θ grid with Gaussian broadening
        # (B, M, 1) vs (1, 1, n_bins) → (B, M, n_bins)
        diff = two_theta_hkl.unsqueeze(-1) - self.two_theta.view(1, 1, -1)
        kern = torch.exp(-0.5 * (diff / self.sigma) ** 2)
        kern = kern * valid.unsqueeze(-1).float()

        I = (I_hkl.unsqueeze(-1) * kern).sum(dim=1)  # (B, n_bins)

        # Normalize
        I_max = I.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        return I / I_max


def diff_pxrd_loss(pred_pattern: torch.Tensor, target_pattern: torch.Tensor,
                   n_bins_diff: int = 512) -> torch.Tensor:
    """Loss between structure-factor PXRD and target pattern.

    Uses 1 - Pearson correlation (scale/shift invariant).
    """
    if target_pattern.shape[-1] != n_bins_diff:
        target_ds = F.interpolate(
            target_pattern.unsqueeze(1), size=n_bins_diff, mode="linear",
            align_corners=True,
        ).squeeze(1)
    else:
        target_ds = target_pattern

    pred_z = pred_pattern - pred_pattern.mean(dim=-1, keepdim=True)
    targ_z = target_ds - target_ds.mean(dim=-1, keepdim=True)
    pred_n = pred_z / pred_z.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    targ_n = targ_z / targ_z.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    pearson = (pred_n * targ_n).sum(dim=-1)
    return (1 - pearson).mean()
