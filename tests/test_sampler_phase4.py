"""Smoke tests for Phase 4 sampler additions.

These run on CPU and do not require any checkpoint. They verify:
  - lattice_params_to_matrix produces correct lattice matrices for cubic/hex
  - pearson_score is 1.0 for identity, -1.0 for negation, per-sample
  - select_best_by_pearson picks the candidate that's closer in pattern space
  - refine_structure reduces (1 - Pearson) loss when given a perturbed start
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pxrd_diff.debye import DiffPXRD                         # noqa: E402
from pxrd_diff.sampler import (                               # noqa: E402
    lattice_params_to_matrix,
    pearson_score,
    refine_structure,
    select_best_by_pearson,
)


def test_lattice_params_to_matrix_cubic():
    # a=b=c=4.0, all angles 90 -> 4*I
    lp = torch.tensor([[4.0, 4.0, 4.0, 90.0, 90.0, 90.0]])
    m = lattice_params_to_matrix(lp)[0]
    expected = 4.0 * torch.eye(3)
    assert torch.allclose(m, expected, atol=1e-5), f"\n{m}\n{expected}"


def test_lattice_params_to_matrix_hexagonal():
    # a=b=3, c=5, alpha=beta=90, gamma=120
    lp = torch.tensor([[3.0, 3.0, 5.0, 90.0, 90.0, 120.0]])
    m = lattice_params_to_matrix(lp)[0]
    # |a| should be 3
    assert torch.isclose(m[0].norm(), torch.tensor(3.0), atol=1e-5)
    # |b| should be 3
    assert torch.isclose(m[1].norm(), torch.tensor(3.0), atol=1e-5)
    # |c| should be 5
    assert torch.isclose(m[2].norm(), torch.tensor(5.0), atol=1e-5)
    # angle between a, b should be 120 deg
    cos_g = (m[0] @ m[1]) / (m[0].norm() * m[1].norm())
    assert torch.isclose(cos_g, torch.tensor(math.cos(math.radians(120.0))), atol=1e-5)


def test_lattice_params_to_matrix_is_differentiable():
    lp = torch.tensor([[4.0, 4.0, 4.0, 90.0, 90.0, 90.0]], requires_grad=True)
    m = lattice_params_to_matrix(lp)
    m.sum().backward()
    assert lp.grad is not None
    assert torch.isfinite(lp.grad).all()


def test_pearson_score_identity_is_one():
    p = torch.randn(3, 100)
    s = pearson_score(p, p)
    assert torch.allclose(s, torch.ones(3), atol=1e-5), s


def test_pearson_score_negation_is_minus_one():
    p = torch.randn(3, 100)
    s = pearson_score(p, -p)
    assert torch.allclose(s, -torch.ones(3), atol=1e-5), s


def test_pearson_score_resamples_target_to_pred_bins():
    pred = torch.randn(2, 64)
    target = torch.randn(2, 256)  # different bin count
    s = pearson_score(pred, target)
    assert s.shape == (2,)
    assert torch.isfinite(s).all()


def _tiny_inputs(B=2, S=3, N=4, n_bins_target=128):
    torch.manual_seed(0)
    coords = torch.rand(B, S, N, 3)
    lat_params = torch.tensor([[4.0, 4.0, 4.0, 90.0, 90.0, 90.0]] * (B * S)).view(B, S, 6)
    atom_types = torch.tensor([[6, 8, 14, 0]] * B)  # C, O, Si, padding
    mask = torch.tensor([[True, True, True, False]] * B)
    target_pxrd = torch.randn(B, n_bins_target).clamp(min=0.0)
    return coords, lat_params, atom_types, mask, target_pxrd


def test_select_best_by_pearson_picks_target_match():
    """Plant the true structure as one of the candidates and verify
    select_best_by_pearson picks it."""
    B, S, N = 2, 4, 4
    coords_true = torch.rand(B, N, 3)
    atom_types = torch.tensor([[6, 8, 14, 0]] * B)
    mask = torch.tensor([[True, True, True, False]] * B)
    lat = torch.eye(3).unsqueeze(0).expand(B, -1, -1) * 4.0

    debye = DiffPXRD(n_bins=128, hkl_max=4)
    debye.eval()

    # Compute the "target" PXRD from the true structures
    with torch.no_grad():
        target_pxrd = debye(coords_true, atom_types, lat, mask)

    # Build S candidates: one is the truth, others are random
    cand_coords = torch.randn(B, S, N, 3) % 1.0
    plant_idx = torch.tensor([2, 0])  # different positions in each batch
    for b in range(B):
        cand_coords[b, plant_idx[b]] = coords_true[b]
    cand_lat_params = torch.tensor([[4.0, 4.0, 4.0, 90.0, 90.0, 90.0]] * (B * S)).view(B, S, 6)

    best_coords, best_lat_params, best_scores = select_best_by_pearson(
        cand_coords, cand_lat_params, atom_types, mask, lat,
        target_pxrd, debye,
    )

    assert best_coords.shape == (B, N, 3)
    assert best_lat_params.shape == (B, 6)
    # The planted candidate should win
    for b in range(B):
        assert torch.allclose(best_coords[b], coords_true[b], atol=1e-5), \
            f"batch {b}: did not pick planted truth (score={best_scores[b]:.3f})"


def test_refine_structure_reduces_loss():
    """Start from a perturbed copy of the truth; refinement should decrease
    1 - Pearson against the target pattern."""
    B, N = 2, 4
    torch.manual_seed(42)
    coords_true = torch.rand(B, N, 3)
    atom_types = torch.tensor([[6, 8, 14, 0]] * B)
    mask = torch.tensor([[True, True, True, False]] * B)
    lat = torch.eye(3).unsqueeze(0).expand(B, -1, -1) * 4.0

    debye = DiffPXRD(n_bins=128, hkl_max=4)
    debye.eval()

    with torch.no_grad():
        target_pxrd = debye(coords_true, atom_types, lat, mask)

    # Perturb truth by 0.05 in fractional coords
    coords_init = (coords_true + 0.05 * torch.randn_like(coords_true)) % 1.0

    coords_refined, lat_refined, hist = refine_structure(
        coords_init, atom_types, lat, mask, target_pxrd, debye,
        steps=100, lr=1e-2,
    )

    assert coords_refined.shape == (B, N, 3)
    assert torch.all(coords_refined >= 0.0) and torch.all(coords_refined < 1.0 + 1e-6), \
        "refined coords must be on the [0, 1) torus"
    # Loss should monotonically-ish decrease
    assert hist[-1] < hist[0], f"refinement did not reduce loss: {hist[0]:.4f} -> {hist[-1]:.4f}"
    # Final loss should be close to zero (we know there's a perfect fit)
    assert hist[-1] < 0.05, f"refinement converged poorly: final loss {hist[-1]:.4f}"
