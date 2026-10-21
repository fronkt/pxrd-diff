"""Tests for the JAC R2 fixes (hat5032, referee questions 1 and 2).

  - periodic_distances(min_image=True) agrees with pymatgen's minimum-image
    distance matrix, including oblique (monoclinic/triclinic) cells
  - periodic_distances(min_image=False) is unchanged (v10-v22 featurisation)
  - lattice_matrix_to_params inverts lattice_params_to_matrix
  - with dist_lattice="noisy" the learned-head sampler output does not depend on
    lattice_init (the reference cell can no longer reach the denoiser), while
    the given-cell modes still condition on it
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pxrd_diff.model.denoiser import CrystalDenoiser, periodic_distances  # noqa: E402
from pxrd_diff.model.pxrd_encoder import PXRDEncoder                       # noqa: E402
from pxrd_diff.sampler import (                                            # noqa: E402
    DDIMSampler,
    lattice_matrix_to_params,
    lattice_params_to_matrix,
)

pymatgen = pytest.importorskip("pymatgen.core")


def _our_distances(struct, min_image: bool) -> np.ndarray:
    n = len(struct)
    f = torch.tensor(struct.frac_coords, dtype=torch.float32).unsqueeze(0)
    lat = torch.tensor(struct.lattice.matrix, dtype=torch.float32).unsqueeze(0)
    mask = torch.ones(1, n, dtype=torch.bool)
    return periodic_distances(f, lat, mask, min_image=min_image)[0].numpy()


@pytest.mark.parametrize("params", [
    (4.0, 5.0, 6.0, 90.0, 90.0, 90.0),     # orthorhombic: rounding is already exact
    (5.0, 6.0, 7.0, 90.0, 115.0, 90.0),    # monoclinic
    (4.0, 5.0, 6.0, 70.0, 110.0, 60.0),    # triclinic
    (5.0, 5.0, 5.0, 60.0, 60.0, 60.0),     # fcc primitive cell
    (3.0, 3.0, 9.0, 90.0, 90.0, 120.0),    # hexagonal
    (6.0, 6.0, 6.0, 35.0, 35.0, 35.0),     # strongly sheared rhombohedral cell
])
def test_min_image_matches_pymatgen(params):
    rng = np.random.default_rng(0)
    lattice = pymatgen.Lattice.from_parameters(*params)
    struct = pymatgen.Structure(lattice, ["Si"] * 12, rng.random((12, 3)))
    ours = _our_distances(struct, min_image=True)
    off = ~np.eye(len(struct), dtype=bool)
    np.testing.assert_allclose(ours[off], struct.distance_matrix[off], atol=1e-4)


def test_min_image_matches_pymatgen_on_mp20_test():
    csv = ROOT / "data" / "raw" / "test.csv"
    if not csv.exists():
        pytest.skip("MP-20 data not built (data/ is gitignored)")
    import pandas as pd
    from pymatgen.io.cif import CifParser
    df = pd.read_csv(csv).head(1000)            # the evaluation set
    worst, n_checked = 0.0, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for cif in df["cif"]:
            s = CifParser.from_str(cif).parse_structures(primitive=False)[0]
            if len(s) < 2:
                continue
            off = ~np.eye(len(s), dtype=bool)
            worst = max(worst, float(np.abs(_our_distances(s, True) - s.distance_matrix)[off].max()))
            n_checked += 1
    assert n_checked > 900
    assert worst < 1e-3, f"max |ours - pymatgen| = {worst:.4f} Å"


def test_round_mode_is_unchanged():
    torch.manual_seed(0)
    f = torch.rand(3, 7, 3)
    lat = lattice_params_to_matrix(torch.tensor([[4.0, 5.0, 6.0, 70.0, 110.0, 60.0]] * 3))
    mask = torch.ones(3, 7, dtype=torch.bool)
    mask[1, 5:] = False
    delta = f.unsqueeze(2) - f.unsqueeze(1)
    delta = delta - delta.round()
    legacy = torch.einsum("bnmd,bdc->bnmc", delta, lat).norm(dim=-1)
    legacy = legacy.masked_fill(~(mask.unsqueeze(2) & mask.unsqueeze(1)), 13.0)
    assert torch.equal(periodic_distances(f, lat, mask), legacy)
    # and the minimum image is never longer than the rounded one
    assert (periodic_distances(f, lat, mask, min_image=True) <= legacy + 1e-5).all()


def test_lattice_matrix_params_roundtrip():
    lp = torch.tensor([[4.0, 5.0, 6.0, 70.0, 110.0, 60.0],
                       [3.0, 3.0, 9.0, 90.0, 90.0, 120.0],
                       [5.0, 5.0, 5.0, 60.0, 60.0, 60.0]])
    back = lattice_matrix_to_params(lattice_params_to_matrix(lp))
    assert torch.allclose(back, lp, atol=1e-3), back


def _tiny_sampler(dist_lattice: str) -> DDIMSampler:
    torch.manual_seed(0)
    enc = PXRDEncoder(d_model=32).eval()
    den = CrystalDenoiser(d_model=32, n_layers=1, n_heads=2, min_image=True).eval()
    lat_mean = torch.tensor([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
    lat_std = torch.tensor([1.0, 1.0, 1.0, 10.0, 10.0, 10.0])
    return DDIMSampler(enc, den, n_steps=4, lat_mean=lat_mean, lat_std=lat_std,
                       predict_x0=True, dist_lattice=dist_lattice)


def _run(sampler: DDIMSampler, lattice_init: torch.Tensor, fixed: bool):
    torch.manual_seed(1)
    pxrd = torch.rand(2, 4251)
    types = torch.tensor([[14, 8, 8, 0], [26, 26, 8, 8]])
    mask = types > 0
    return sampler.sample(pxrd, types, lattice_init, mask, fixed_lattice=fixed)


_CELL_A = lattice_params_to_matrix(torch.tensor([[4.0, 5.0, 6.0, 80.0, 100.0, 95.0],
                                                 [7.0, 7.0, 7.0, 90.0, 90.0, 90.0]]))
_CELL_B = lattice_params_to_matrix(torch.tensor([[9.0, 3.0, 4.0, 60.0, 120.0, 70.0],
                                                 [3.5, 3.5, 11.0, 90.0, 90.0, 120.0]]))


def test_noisy_learned_mode_ignores_lattice_init():
    s = _tiny_sampler("noisy")
    c_a, l_a = _run(s, _CELL_A, fixed=False)
    c_b, l_b = _run(s, _CELL_B, fixed=False)
    c_nan, l_nan = _run(s, torch.full_like(_CELL_A, float("nan")), fixed=False)
    assert torch.equal(c_a, c_b) and torch.equal(l_a, l_b)
    assert torch.equal(c_a, c_nan) and torch.equal(l_a, l_nan)


def test_clean_learned_mode_reads_lattice_init():
    # Documents the R2 Q1 leak: the legacy path conditions on whatever cell it is handed.
    s = _tiny_sampler("clean")
    c_a, _ = _run(s, _CELL_A, fixed=False)
    c_b, _ = _run(s, _CELL_B, fixed=False)
    assert not torch.allclose(c_a, c_b)


def test_noisy_fixed_mode_conditions_on_given_cell():
    s = _tiny_sampler("noisy")
    c_a, l_a = _run(s, _CELL_A, fixed=True)
    c_b, l_b = _run(s, _CELL_B, fixed=True)
    assert not torch.allclose(c_a, c_b)
    assert torch.allclose(l_a, lattice_matrix_to_params(_CELL_A), atol=1e-3)
    assert torch.allclose(l_b, lattice_matrix_to_params(_CELL_B), atol=1e-3)
