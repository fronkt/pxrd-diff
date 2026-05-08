"""Smoke tests for the Phase 5B + 6 heads (constrained lattice, SG, SG constraints)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pxrd_diff.model.lat_head import (                          # noqa: E402
    ConstrainedLatHead,
    PeakAugmentedLatHead,
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


# ---------- PeakAugmentedLatHead -------------------------------------------------

def test_peak_aug_head_within_bounds():
    head = PeakAugmentedLatHead(d_model=32, peak_dim=40)
    head.eval()
    emb = torch.randn(8, 32) * 100.0
    peaks = torch.rand(8, 40)
    out = head(emb, peaks)
    assert out.shape == (8, 6)
    assert torch.all(out[:, :3] >= 2.0) and torch.all(out[:, :3] <= 20.0)
    assert torch.all(out[:, 3:] >= 30.0) and torch.all(out[:, 3:] <= 150.0)


def test_peak_aug_head_uses_peak_features():
    """If two batches differ only in peak features, the output should differ
    (i.e. peak features actually feed into the prediction, not dead weights)."""
    torch.manual_seed(0)
    head = PeakAugmentedLatHead(d_model=32, peak_dim=40)
    head.eval()
    emb = torch.randn(4, 32)
    peaks_a = torch.rand(4, 40)
    peaks_b = torch.rand(4, 40)
    with torch.no_grad():
        out_a = head(emb, peaks_a)
        out_b = head(emb, peaks_b)
    assert not torch.allclose(out_a, out_b, atol=1e-3), \
        "Peak features have no effect — head is ignoring them"


def test_peak_aug_head_is_differentiable_through_peaks():
    head = PeakAugmentedLatHead(d_model=16, peak_dim=20)
    emb = torch.randn(2, 16)
    peaks = torch.rand(2, 20, requires_grad=True)
    out = head(emb, peaks)
    out.sum().backward()
    assert peaks.grad is not None
    assert torch.isfinite(peaks.grad).all()


# ---------- Phase 5D: d-spacing conversion -----------------------------------

def test_d_spacing_conversion_matches_braggs_law():
    """Verify _peaks_to_d_spacing implements d = lambda / (2 sin theta)
    correctly for a few hand-computed positions."""
    head = PeakAugmentedLatHead(
        d_model=4, peak_dim=4, use_d_spacing=True,
        wavelength=1.54184, two_theta_min=5.0, two_theta_max=90.0,
    )
    # One peak at pos=0.3 (=> 2θ = 5 + 0.3 * 85 = 30.5°, θ = 15.25°)
    # d = 1.54184 / (2 * sin(15.25°)) ≈ 2.93 Å, log_d ≈ 1.075
    # Plus one padded slot at pos=0, intensity=0
    peak_features = torch.tensor([[0.3, 1.0, 0.0, 0.0]])
    transformed = head._peaks_to_d_spacing(peak_features)
    log_d_expected = math.log(1.54184 / (2 * math.sin(math.radians(15.25))))
    assert torch.isclose(transformed[0, 0], torch.tensor(log_d_expected), atol=1e-4), \
        f"got {transformed[0,0]:.4f}, expected {log_d_expected:.4f}"
    # Intensity preserved
    assert transformed[0, 1].item() == 1.0
    # Padded slot: log_d should be 0
    assert transformed[0, 2].item() == 0.0
    assert transformed[0, 3].item() == 0.0


def test_d_spacing_head_outputs_within_bounds():
    head = PeakAugmentedLatHead(d_model=32, peak_dim=40, use_d_spacing=True)
    head.eval()
    emb = torch.randn(8, 32)
    # Realistic peak features: random positions in [0, 1] and intensities in [0, 1]
    peaks = torch.rand(8, 40)
    out = head(emb, peaks)
    assert out.shape == (8, 6)
    assert torch.all(out[:, :3] >= 2.0) and torch.all(out[:, :3] <= 20.0)
    assert torch.all(out[:, 3:] >= 30.0) and torch.all(out[:, 3:] <= 150.0)


def test_d_spacing_changes_output_vs_raw_2theta():
    """If we feed the same peak features through use_d_spacing=False vs True,
    outputs differ — i.e. the conversion isn't a no-op."""
    torch.manual_seed(0)
    h_raw = PeakAugmentedLatHead(d_model=16, peak_dim=20, use_d_spacing=False)
    torch.manual_seed(0)  # Identical weight init
    h_d = PeakAugmentedLatHead(d_model=16, peak_dim=20, use_d_spacing=True)
    h_raw.eval(); h_d.eval()

    emb = torch.randn(4, 16)
    peaks = torch.rand(4, 20).abs()  # all positive intensities
    out_raw = h_raw(emb, peaks)
    out_d = h_d(emb, peaks)
    assert not torch.allclose(out_raw, out_d, atol=1e-3), \
        "d-spacing conversion had no effect — feature transform is broken"


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
