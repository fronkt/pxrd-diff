"""PyTorch Dataset for PXRD-Diff.

Loads cached PXRD patterns + parses CIF structures into padded tensors.
All structures are parsed once at construction and held in memory (~10 MB
for 27k structures with <=20 atoms).

Each sample contains:
  - pxrd_pattern  (B,)     float32   cached simulated PXRD intensities
  - frac_coords   (M, 3)   float32   fractional coordinates, padded
  - atom_types    (M,)     long      atomic numbers, padded with 0
  - lattice       (3, 3)   float32   lattice matrix (rows = lattice vectors)
  - lattice_params (6,)    float32   [a, b, c, alpha, beta, gamma]
  - num_atoms     ()       long      actual atom count
  - mask          (M,)     bool      True for real atoms
  - spacegroup    ()       long      space group number
  - material_id   str
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from pymatgen.io.cif import CifParser
from tqdm import tqdm

MAX_ATOMS = 20

warnings.filterwarnings("ignore", category=UserWarning,
                        message=".*fractional coordinates rounded.*")


def _wyckoff_letters_to_ids(struct, max_id: int = 27) -> np.ndarray:
    """Compute per-atom Wyckoff site IDs via spglib. Returns (N,) int64.

    Each Wyckoff letter ('a'..'z', plus 'A' fallback) maps to int 0..26.
    Letter 'a' is the highest-symmetry site; 'A' is a fallback for unusual cases.
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    try:
        sga = SpacegroupAnalyzer(struct, symprec=0.01)
        ds = sga.get_symmetry_dataset()
        wlist = ds.wyckoffs if hasattr(ds, "wyckoffs") else ds["wyckoffs"]
    except Exception:
        return np.full(len(struct), max_id - 1, dtype=np.int64)
    out = np.full(len(struct), max_id - 1, dtype=np.int64)
    for i, w in enumerate(wlist):
        if isinstance(w, str) and len(w) == 1 and w.isalpha():
            wl = w.lower()
            idx = ord(wl) - ord("a")
            if 0 <= idx < max_id - 1:
                out[i] = idx
    return out


def _parse_cif(cif_str: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse CIF -> (frac_coords, atomic_numbers, lattice_matrix, lattice_params, wyckoff_ids)."""
    struct = CifParser.from_str(cif_str).parse_structures(primitive=False)[0]
    frac = struct.frac_coords.astype(np.float32)                  # (N, 3)
    z = np.array([s.Z for s in struct.species], dtype=np.int64)    # (N,)
    lat = struct.lattice.matrix.astype(np.float32)                 # (3, 3)
    params = np.array(struct.lattice.parameters, dtype=np.float32) # (6,) a,b,c,α,β,γ
    wyck = _wyckoff_letters_to_ids(struct)
    return frac, z, lat, params, wyck


def augment_pxrd_pattern(pattern: np.ndarray,
                         rng: Optional[np.random.Generator] = None,
                         p: float = 0.8) -> np.ndarray:
    """Apply stochastic experimental-style imperfections to a clean simulated
    PXRD pattern. Used during Phase 7 training to make the model robust to
    real-data artifacts not present in the simulated patterns.

    Three augmentations, applied jointly with probability `p` (otherwise the
    pattern is returned unchanged):
      1. 2θ zero-offset (uniform integer shift in [-5, +5] bins ≈ ±0.1°)
         — simulates instrument calibration error.
      2. Lorentzian peak broadening (FWHM uniform in [2, 15] bins ≈ 0.04–0.3°)
         — simulates crystallite-size / strain broadening (Scherrer).
      3. Additive Gaussian noise (σ uniform in [0.5, 3] % of pattern max)
         — simulates counting-statistics / detector noise.

    The output is renormalized to the original max so downstream normalization
    in the encoder does not see a global intensity drift.
    """
    rng = np.random.default_rng() if rng is None else rng
    if rng.random() > p:
        return pattern.astype(np.float32, copy=False)

    out = pattern.astype(np.float32, copy=True)
    orig_max = float(out.max())

    # 1. 2θ zero-offset
    shift = int(rng.integers(-5, 6))
    if shift != 0:
        out = np.roll(out, shift)
        if shift > 0:
            out[:shift] = 0.0
        else:
            out[shift:] = 0.0

    # 2. Lorentzian peak broadening
    fwhm = float(rng.uniform(2.0, 15.0))
    kernel_half = max(1, int(2 * fwhm))
    x = np.arange(-kernel_half, kernel_half + 1, dtype=np.float32)
    kernel = 1.0 / (1.0 + (2.0 * x / fwhm) ** 2)
    kernel /= kernel.sum()
    out = np.convolve(out, kernel, mode="same").astype(np.float32)

    # 3. Gaussian instrument noise
    if out.max() > 0:
        sigma = float(rng.uniform(0.005, 0.03)) * float(out.max())
        out += rng.normal(0.0, sigma, out.shape).astype(np.float32)

    # Clip negatives, restore original scale
    out = np.maximum(out, 0.0)
    if out.max() > 0 and orig_max > 0:
        out = out * (orig_max / out.max())
    return out.astype(np.float32)


def _extract_peak_features(pattern: np.ndarray, n_peaks: int = 20,
                           height_frac: float = 0.05,
                           min_distance: int = 3) -> np.ndarray:
    """Extract top-N peak (position_norm, intensity_norm) pairs from a PXRD pattern.

    Position is the bin index normalized to [0, 1] (linear in 2θ since the
    simulator uses an evenly-spaced grid). Intensity is normalized to the
    pattern's max value. Output is a flat (n_peaks * 2) vector — interleaved
    [pos_1, int_1, pos_2, int_2, ...] — sorted by intensity descending. Slots
    beyond the actual peak count are zero-padded.

    Args:
        pattern: (L,) raw PXRD intensities (already normalized to max=1 by the
                 simulator)
        n_peaks: max peaks to keep
        height_frac: minimum peak height as a fraction of pattern max
        min_distance: minimum bin separation between adjacent peaks
    """
    from scipy.signal import find_peaks
    L = len(pattern)
    pmax = float(pattern.max())
    if pmax <= 0:
        return np.zeros(n_peaks * 2, dtype=np.float32)
    height = height_frac * pmax
    idx, props = find_peaks(pattern, height=height, distance=min_distance)
    if len(idx) == 0:
        return np.zeros(n_peaks * 2, dtype=np.float32)
    heights = props["peak_heights"]
    order = np.argsort(heights)[::-1][:n_peaks]
    top_idx = idx[order]
    top_h = heights[order]
    out = np.zeros(n_peaks * 2, dtype=np.float32)
    pos_norm = top_idx.astype(np.float32) / float(L)
    int_norm = (top_h / top_h.max()).astype(np.float32) if top_h.size else top_h
    out[0:2 * len(top_idx):2] = pos_norm
    out[1:2 * len(top_idx):2] = int_norm
    return out


class CrystalPXRDDataset(Dataset):
    """Dataset pairing cached PXRD patterns with parsed crystal structures.

    With the default `n_peaks=20`, each sample also carries a precomputed
    peak-feature vector (length 40: 20 positions and 20 intensities,
    interleaved). Set `n_peaks=0` to disable.
    """

    def __init__(self, data_dir: str | Path, split: str = "train",
                 max_atoms: int = MAX_ATOMS, preload: bool = True,
                 limit: int | None = None,
                 n_peaks: int = 20,
                 augment: bool = False,
                 augment_seed: Optional[int] = None):
        data_dir = Path(data_dir)
        cache_dir = data_dir / "cache"
        raw_dir = data_dir / "raw"

        npz = np.load(cache_dir / f"{split}.npz", allow_pickle=True)
        n = limit if limit is not None else len(npz["pattern"])
        self.patterns = npz["pattern"][:n]
        self.material_ids = npz["material_id"][:n]
        self.spacegroups = npz["spacegroup"][:n]

        self.max_atoms = max_atoms
        self.n_peaks = n_peaks
        self.augment = augment
        # Phase 7: an explicit seed makes the augmented eval reproducible while
        # training augmentation stays stochastic (uses the global default RNG).
        self._aug_rng = (np.random.default_rng(augment_seed)
                         if augment and augment_seed is not None else None)
        self._structures: Optional[list] = None

        if n_peaks > 0:
            self.peak_features = np.stack([
                _extract_peak_features(p, n_peaks=n_peaks)
                for p in tqdm(self.patterns, desc=f"Peak features {split}",
                              leave=False)
            ]).astype(np.float32)
        else:
            self.peak_features = None

        if preload:
            import pandas as pd
            df = pd.read_csv(raw_dir / f"{split}.csv", nrows=limit)
            self._structures = []
            for _, row in tqdm(df.iterrows(), total=len(df),
                               desc=f"Parsing {split} CIFs", leave=False):
                self._structures.append(_parse_cif(row["cif"]))

    def __len__(self) -> int:
        return len(self.patterns)

    def __getitem__(self, idx: int) -> dict:
        frac, z, lat, params, wyck = self._structures[idx]
        n = len(z)
        M = self.max_atoms

        frac_padded = np.zeros((M, 3), dtype=np.float32)
        z_padded = np.zeros(M, dtype=np.int64)
        wyck_padded = np.zeros(M, dtype=np.int64)
        mask = np.zeros(M, dtype=bool)

        frac_padded[:n] = frac
        z_padded[:n] = z
        wyck_padded[:n] = wyck
        mask[:n] = True

        # Phase 7 augmentation: apply per-call so each epoch sees fresh noise.
        # When augmenting, peak features must be recomputed on the augmented
        # pattern so they stay consistent with what the encoder/head will see.
        pattern = self.patterns[idx]
        if self.augment:
            pattern = augment_pxrd_pattern(pattern, rng=self._aug_rng)
            if self.n_peaks > 0:
                peak_feat = _extract_peak_features(pattern, n_peaks=self.n_peaks)
            else:
                peak_feat = None
        else:
            peak_feat = (self.peak_features[idx] if self.peak_features is not None
                         else None)

        out = {
            "pxrd_pattern": torch.from_numpy(pattern.astype(np.float32, copy=False)),
            "frac_coords": torch.from_numpy(frac_padded),
            "atom_types": torch.from_numpy(z_padded),
            "wyckoff": torch.from_numpy(wyck_padded),
            "lattice": torch.from_numpy(lat),
            "lattice_params": torch.from_numpy(params),
            "num_atoms": torch.tensor(n, dtype=torch.long),
            "mask": torch.from_numpy(mask),
            "spacegroup": torch.tensor(int(self.spacegroups[idx]), dtype=torch.long),
            "material_id": str(self.material_ids[idx]),
        }
        if peak_feat is not None:
            out["peak_features"] = torch.from_numpy(peak_feat.astype(np.float32, copy=False))
        return out


def lattice_params_stats(dataset: CrystalPXRDDataset) -> dict[str, np.ndarray]:
    """Compute mean/std of lattice params for normalization."""
    # Tuple is (frac, z, lat, params, wyck) — params is index 3
    params = np.stack([s[3] for s in dataset._structures])
    return {"mean": params.mean(axis=0), "std": params.std(axis=0)}