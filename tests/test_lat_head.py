"""Smoke tests for the Phase 5B + 6 heads (constrained lattice, SG, SG constraints)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pxrd_diff.model.lat_head import (                          # noqa: E402
    ConstrainedLatHead,
    SpaceGroupHead,
    apply_sg_constraints,
    sg_classification_loss,
    sg_topk_accuracy,
)


# ---------- ConstrainedLatHead ------------------------------------------------------

def test_constrained_lat_head_within_bounds():
    head = ConstrainedLatHead(d_model=32, len_min=2.0, len_max=20.0,
                              ang_min=30.0, ang_max=150.0)
    head.eval()
    # Push large absolute inputs; sigmoid should clip the result to bounds anyway.
    emb = torch.randn(64, 32) * 100.0
    out = head(emb)
    assert out.shape == (64, 6)
    a, b, c = out[:, 0], out[:, 1], out[:, 2]
    al, be, ga = out[:, 3], out[:, 4], out[:, 5]
    assert torch.all(a >= 2.0) and torch.all(a <= 20.0)
    assert torch.all(b >= 2.0) and torch.all(b <= 20.0)
    assert torch.all(c >= 2.0) and torch.all(c <= 20.0)
    assert torch.all(al >= 30.0) and torch.all(al <= 150.0)
    assert torch.all(be >= 30.0) and torch.all(be <= 150.0)
    assert torch.all(ga >= 30.0) and torch.all(ga <= 150.0)


def test_constrained_lat_head_is_differentiable():
    head = ConstrainedLatHead(d_model=16)
    emb = torch.randn(4, 16, requires_grad=True)
    out = head(emb)
    out.sum().backward()
    assert emb.grad is not None
    assert torch.isfinite(emb.grad).all()


# ---------- SpaceGroupHead ----------------------------------------------------------

def test_sg_head_shapes():
    head = SpaceGroupHead(d_model=32)
    emb = torch.randn(7, 32)
    logits = head(emb)
    assert logits.shape == (7, 230)


def test_sg_classification_loss_decreases_when_targets_match():
    head = SpaceGroupHead(d_model=32)
    emb = torch.randn(8, 32, requires_grad=True)
    sg_target = torch.randint(low=1, high=231, size=(8,))
    logits = head(emb)
    loss0 = sg_classification_loss(logits, sg_target).item()
    # Train one step toward the target — loss should drop on a matched batch
    optimizer = torch.optim.Adam(list(head.parameters()) + [emb], lr=1e-1)
    for _ in range(50):
        optimizer.zero_grad()
        loss = sg_classification_loss(head(emb), sg_target)
        loss.backward()
        optimizer.step()
    assert loss.item() < loss0, f"SG loss did not decrease: {loss0} -> {loss.item()}"


def test_sg_topk_accuracy_perfect_predictions():
    # Build logits that perfectly point at known targets; accuracy should be 1.0.
    sg_target = torch.tensor([1, 100, 230, 50, 200])
    logits = torch.full((5, 230), -10.0)
    for i, t in enumerate(sg_target):
        logits[i, t.item() - 1] = 10.0
    assert sg_topk_accuracy(logits, sg_target, k=1).item() == 1.0
    assert sg_topk_accuracy(logits, sg_target, k=5).item() == 1.0


# ---------- apply_sg_constraints ----------------------------------------------------

def test_sg_constraints_cubic_equates_lengths():
    lp = torch.tensor([[3.0, 5.0, 7.0, 80.0, 92.0, 95.0]])
    sg = torch.tensor([225])  # cubic (Fm-3m)
    out = apply_sg_constraints(lp, sg)[0]
    expected_a = (3.0 + 5.0 + 7.0) / 3.0
    assert torch.allclose(out[:3], torch.full((3,), expected_a), atol=1e-5)
    assert torch.allclose(out[3:], torch.full((3,), 90.0), atol=1e-5)


def test_sg_constraints_tetragonal_a_equals_b():
    lp = torch.tensor([[3.0, 5.0, 7.0, 80.0, 92.0, 95.0]])
    sg = torch.tensor([139])  # tetragonal I4/mmm
    out = apply_sg_constraints(lp, sg)[0]
    expected_ab = 4.0
    assert torch.isclose(out[0], torch.tensor(expected_ab))
    assert torch.isclose(out[1], torch.tensor(expected_ab))
    assert torch.isclose(out[2], torch.tensor(7.0))
    assert torch.allclose(out[3:], torch.full((3,), 90.0))


def test_sg_constraints_hexagonal():
    lp = torch.tensor([[3.0, 5.0, 7.0, 80.0, 92.0, 95.0]])
    sg = torch.tensor([194])  # hexagonal P6_3/mmc
    out = apply_sg_constraints(lp, sg)[0]
    expected_ab = 4.0
    assert torch.isclose(out[0], torch.tensor(expected_ab))
    assert torch.isclose(out[1], torch.tensor(expected_ab))
    assert torch.isclose(out[2], torch.tensor(7.0))
    assert torch.isclose(out[3], torch.tensor(90.0))
    assert torch.isclose(out[4], torch.tensor(90.0))
    assert torch.isclose(out[5], torch.tensor(120.0))


def test_sg_constraints_orthorhombic_only_angles_changed():
    lp = torch.tensor([[3.0, 5.0, 7.0, 80.0, 92.0, 95.0]])
    sg = torch.tensor([62])  # orthorhombic Pnma
    out = apply_sg_constraints(lp, sg)[0]
    assert torch.isclose(out[0], torch.tensor(3.0))
    assert torch.isclose(out[1], torch.tensor(5.0))
    assert torch.isclose(out[2], torch.tensor(7.0))
    assert torch.allclose(out[3:], torch.full((3,), 90.0))


def test_sg_constraints_monoclinic_only_alpha_gamma_changed():
    lp = torch.tensor([[3.0, 5.0, 7.0, 80.0, 92.0, 95.0]])
    sg = torch.tensor([14])  # monoclinic P2_1/c
    out = apply_sg_constraints(lp, sg)[0]
    assert torch.isclose(out[0], torch.tensor(3.0))
    assert torch.isclose(out[1], torch.tensor(5.0))
    assert torch.isclose(out[2], torch.tensor(7.0))
    assert torch.isclose(out[3], torch.tensor(90.0))
    assert torch.isclose(out[4], torch.tensor(92.0))    # β preserved
    assert torch.isclose(out[5], torch.tensor(90.0))


def test_sg_constraints_triclinic_unchanged():
    lp = torch.tensor([[3.0, 5.0, 7.0, 80.0, 92.0, 95.0]])
    sg = torch.tensor([1])  # triclinic
    out = apply_sg_constraints(lp, sg)[0]
    assert torch.allclose(out, lp[0])


def test_sg_constraints_batched_mixed_systems():
    lp = torch.tensor([
        [3.0, 5.0, 7.0, 80.0, 92.0, 95.0],
        [4.0, 4.0, 6.0, 90.0, 90.0, 90.0],
        [5.0, 5.0, 5.0, 90.0, 90.0, 90.0],
    ])
    sg = torch.tensor([14, 139, 225])  # monoclinic, tetragonal, cubic
    out = apply_sg_constraints(lp, sg)
    # Cubic row should have a=b=c
    assert out[2, 0] == out[2, 1] == out[2, 2]
    # Tetragonal row should have a=b but c different
    assert out[1, 0] == out[1, 1]
    # Monoclinic row should preserve β
    assert torch.isclose(out[0, 4], torch.tensor(92.0))
