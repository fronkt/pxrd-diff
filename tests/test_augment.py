"""Smoke tests for Phase 7 PXRD pattern augmentation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pxrd_diff.data import augment_pxrd_pattern, _extract_peak_features  # noqa: E402


def _synthetic_pxrd(n_bins: int = 4251, peak_positions=(500, 1500, 2500, 3500)):
    """Build a synthetic PXRD pattern with delta peaks for testing."""
    p = np.zeros(n_bins, dtype=np.float32)
    for pos in peak_positions:
        p[pos] = 1.0
    return p


def test_augment_returns_same_shape():
    p = _synthetic_pxrd()
    out = augment_pxrd_pattern(p, rng=np.random.default_rng(0), p=1.0)
    assert out.shape == p.shape
    assert out.dtype == np.float32


def test_augment_can_be_disabled():
    p = _synthetic_pxrd()
    rng = np.random.default_rng(0)
    out = augment_pxrd_pattern(p, rng=rng, p=0.0)
    assert np.array_equal(out, p.astype(np.float32))


def test_augment_changes_pattern_when_applied():
    p = _synthetic_pxrd()
    out = augment_pxrd_pattern(p, rng=np.random.default_rng(7), p=1.0)
    # Delta peaks should now be broadened, so most non-peak bins are nonzero.
    assert (out > 0).sum() > 100, "Augmented pattern should have many nonzero bins (broadening)"


def test_augment_preserves_max_scale_roughly():
    """Renormalization restores the original max within a reasonable factor."""
    p = _synthetic_pxrd()
    out = augment_pxrd_pattern(p, rng=np.random.default_rng(13), p=1.0)
    assert out.max() == 1.0  # exactly restored — implementation rescales to orig max


def test_augment_deterministic_with_seed():
    p = _synthetic_pxrd()
    a = augment_pxrd_pattern(p, rng=np.random.default_rng(42), p=1.0)
    b = augment_pxrd_pattern(p, rng=np.random.default_rng(42), p=1.0)
    assert np.array_equal(a, b)


def test_augment_no_negatives():
    p = _synthetic_pxrd()
    out = augment_pxrd_pattern(p, rng=np.random.default_rng(99), p=1.0)
    assert out.min() >= 0.0


def test_peak_features_change_after_augmentation():
    p = _synthetic_pxrd()
    feat_clean = _extract_peak_features(p, n_peaks=20)
    aug = augment_pxrd_pattern(p, rng=np.random.default_rng(1), p=1.0)
    feat_aug = _extract_peak_features(aug, n_peaks=20)
    # Augmentation should shift peak positions enough to change the features.
    assert not np.array_equal(feat_clean, feat_aug)
