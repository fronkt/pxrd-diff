"""Powder X-ray diffraction (PXRD) simulator.

Wraps pymatgen's `XRDCalculator` to produce a fixed-grid 1D intensity vector
suitable for ML training. Pymatgen returns sparse (2theta, intensity) peaks;
we broaden them onto a regular grid with a Gaussian peak shape.

Defaults (Cu Kalpha_1, 5-90 deg 2theta, 0.02 deg step, 0.1 deg FWHM) match the
typical laboratory-PXRD setup assumed by DiffractGPT and Crystalyze, so our
simulated patterns are directly comparable to the baselines we plan to beat.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure

# Standard X-ray sources, wavelength in Angstrom
WAVELENGTHS = {
    "CuKa": 1.54184,    # weighted Ka1+Ka2 (pymatgen default)
    "CuKa1": 1.540593,
    "MoKa": 0.71073,
    "AgKa": 0.560885,
}


@dataclass
class PXRDConfig:
    wavelength: str = "CuKa"
    two_theta_min: float = 5.0
    two_theta_max: float = 90.0
    two_theta_step: float = 0.02
    peak_fwhm: float = 0.1
    noise_std: float = 0.0           # Gaussian noise added after normalization
    normalize: bool = True            # peak-max -> 1.0
    background_slope: float = 0.0     # linear background coefficient (artifact sim)
    seed: Optional[int] = None

    @property
    def n_bins(self) -> int:
        return int(round((self.two_theta_max - self.two_theta_min) / self.two_theta_step)) + 1

    @property
    def two_theta_grid(self) -> np.ndarray:
        return np.linspace(self.two_theta_min, self.two_theta_max, self.n_bins, dtype=np.float64)


class PXRDSimulator:
    """Fixed-grid PXRD simulator with Gaussian peak broadening."""

    def __init__(self, config: Optional[PXRDConfig] = None):
        self.cfg = config or PXRDConfig()
        self._xrd = XRDCalculator(wavelength=self.cfg.wavelength)
        self._grid = self.cfg.two_theta_grid                       # (B,)
        self._sigma = self.cfg.peak_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        self._rng = np.random.default_rng(self.cfg.seed)

    def simulate(self, structure: Structure) -> np.ndarray:
        """Return a 1D intensity array on the fixed 2theta grid (float32)."""
        pattern = self._xrd.get_pattern(
            structure,
            two_theta_range=(self.cfg.two_theta_min, self.cfg.two_theta_max),
        )
        peaks_2t = np.asarray(pattern.x, dtype=np.float64)        # (P,)
        peaks_I = np.asarray(pattern.y, dtype=np.float64)         # (P,)

        # Broaden each peak as a Gaussian onto the fixed grid (vectorized).
        # I(2t) = sum_p I_p * exp( -((2t - 2t_p)^2) / (2 sigma^2) )
        if peaks_2t.size == 0:
            intensity = np.zeros_like(self._grid)
        else:
            diff = self._grid[:, None] - peaks_2t[None, :]        # (B, P)
            kern = np.exp(-(diff ** 2) / (2.0 * self._sigma ** 2))
            intensity = (kern * peaks_I[None, :]).sum(axis=1)     # (B,)

        if self.cfg.background_slope != 0.0:
            intensity = intensity + self.cfg.background_slope * (self._grid - self.cfg.two_theta_min)

        if self.cfg.normalize:
            peak = intensity.max()
            if peak > 0:
                intensity = intensity / peak

        if self.cfg.noise_std > 0:
            intensity = intensity + self._rng.normal(0.0, self.cfg.noise_std, size=intensity.shape)

        return intensity.astype(np.float32)

    @property
    def two_theta(self) -> np.ndarray:
        return self._grid.copy()