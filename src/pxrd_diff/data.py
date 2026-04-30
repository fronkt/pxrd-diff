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


def _parse_cif(cif_str: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse CIF -> (frac_coords, atomic_numbers, lattice_matrix, lattice_params)."""
    struct = CifParser.from_str(cif_str).parse_structures(primitive=False)[0]
    frac = struct.frac_coords.astype(np.float32)                  # (N, 3)
    z = np.array([s.Z for s in struct.species], dtype=np.int64)    # (N,)
    lat = struct.lattice.matrix.astype(np.float32)                 # (3, 3)
    params = np.array(struct.lattice.parameters, dtype=np.float32) # (6,) a,b,c,α,β,γ
    return frac, z, lat, params


class CrystalPXRDDataset(Dataset):
    """Dataset pairing cached PXRD patterns with parsed crystal structures."""

    def __init__(self, data_dir: str | Path, split: str = "train",
                 max_atoms: int = MAX_ATOMS, preload: bool = True):
        data_dir = Path(data_dir)
        cache_dir = data_dir / "cache"
        raw_dir = data_dir / "raw"

        npz = np.load(cache_dir / f"{split}.npz", allow_pickle=True)
        self.patterns = npz["pattern"]           # (N, B) float32
        self.material_ids = npz["material_id"]   # (N,) str
        self.spacegroups = npz["spacegroup"]     # (N,) int16

        self.max_atoms = max_atoms
        self._structures: Optional[list] = None

        if preload:
            import pandas as pd
            df = pd.read_csv(raw_dir / f"{split}.csv")
            self._structures = []
            for _, row in tqdm(df.iterrows(), total=len(df),
                               desc=f"Parsing {split} CIFs", leave=False):
                self._structures.append(_parse_cif(row["cif"]))

    def __len__(self) -> int:
        return len(self.patterns)

    def __getitem__(self, idx: int) -> dict:
        frac, z, lat, params = self._structures[idx]
        n = len(z)
        M = self.max_atoms

        frac_padded = np.zeros((M, 3), dtype=np.float32)
        z_padded = np.zeros(M, dtype=np.int64)
        mask = np.zeros(M, dtype=bool)

        frac_padded[:n] = frac
        z_padded[:n] = z
        mask[:n] = True

        return {
            "pxrd_pattern": torch.from_numpy(self.patterns[idx]),
            "frac_coords": torch.from_numpy(frac_padded),
            "atom_types": torch.from_numpy(z_padded),
            "lattice": torch.from_numpy(lat),
            "lattice_params": torch.from_numpy(params),
            "num_atoms": torch.tensor(n, dtype=torch.long),
            "mask": torch.from_numpy(mask),
            "spacegroup": torch.tensor(int(self.spacegroups[idx]), dtype=torch.long),
            "material_id": str(self.material_ids[idx]),
        }


def lattice_params_stats(dataset: CrystalPXRDDataset) -> dict[str, np.ndarray]:
    """Compute mean/std of lattice params for normalization."""
    params = np.stack([s[3] for s in dataset._structures])
    return {"mean": params.mean(axis=0), "std": params.std(axis=0)}