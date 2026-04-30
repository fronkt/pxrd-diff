"""Evaluation metrics for PXRD -> structure inversion.

Three views of "did the model get it right?":

1. Structure-domain
   - space_group_match: predicted vs true space group number (with tolerance sweep)
   - coord_rmsd:        RMSD between matched-and-aligned atomic positions
                        (StructureMatcher; returns NaN if structures don't match)
   - composition_match: same set of element multiplicities

2. Pattern-domain
   - rwp:    weighted profile R-factor between predicted and true PXRD
   - r_pearson: Pearson correlation of patterns (sanity)

A model is "correct" on a sample if and only if (a) compositions match exactly,
(b) space groups agree at SYMPREC=0.1, and (c) coord_rmsd <= 0.1 Angstrom.
This combined "all-of-three" criterion is the headline metric for the paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from pymatgen.core.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Tolerances at which to evaluate space group recovery.
SG_TOLS = (0.01, 0.05, 0.1, 0.2)
DEFAULT_RMSD_THRESHOLD = 0.1   # Angstrom; tunable in ablations.


# ---------- Structure-domain metrics ------------------------------------------------

def space_group_number(struct: Structure, symprec: float = 0.1) -> int:
    try:
        return SpacegroupAnalyzer(struct, symprec=symprec).get_space_group_number()
    except Exception:
        return -1


def composition_match(a: Structure, b: Structure) -> bool:
    return Composition(a.composition).reduced_formula == Composition(b.composition).reduced_formula


def coord_rmsd(pred: Structure, true: Structure,
               ltol: float = 0.2, stol: float = 0.3, angle_tol: float = 5.0) -> float:
    """Return Cartesian RMSD between matched/aligned structures, or NaN if no match.

    Uses pymatgen's StructureMatcher with default-ish CDVAE-style tolerances. The
    matcher handles permutation/translation/rotation; what comes back is a single
    scalar Angstrom value comparable across structures of different sizes.
    """
    matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol,
                               primitive_cell=False, scale=True, attempt_supercell=True)
    rms = matcher.get_rms_dist(pred, true)
    if rms is None:
        return float("nan")
    return float(rms[0])    # (rms, max) tuple -> rms


# ---------- Pattern-domain metrics --------------------------------------------------

def rwp(pred: np.ndarray, true: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """Weighted profile R-factor.  Rwp = sqrt( sum w (y_t - y_p)^2 / sum w y_t^2 )."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if weights is None:
        weights = 1.0 / np.maximum(true, 1e-3)        # standard counting-stat-like weight
    num = np.sum(weights * (true - pred) ** 2)
    den = np.sum(weights * true ** 2)
    return float(np.sqrt(num / max(den, 1e-12)))


def r_pearson(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64).ravel()
    true = np.asarray(true, dtype=np.float64).ravel()
    pm, tm = pred - pred.mean(), true - true.mean()
    den = np.sqrt((pm ** 2).sum() * (tm ** 2).sum())
    if den < 1e-12:
        return 0.0
    return float((pm * tm).sum() / den)


# ---------- Per-sample / aggregate --------------------------------------------------

@dataclass
class SampleMetrics:
    material_id: str
    composition_ok: bool
    sg_match: dict          # {tol: bool}
    rmsd: float             # Angstrom; NaN if no match
    rwp: float
    pearson: float

    @property
    def all_correct(self) -> bool:
        return (self.composition_ok
                and self.sg_match.get(0.1, False)
                and (not np.isnan(self.rmsd))
                and self.rmsd <= DEFAULT_RMSD_THRESHOLD)


def evaluate_one(material_id: str,
                 pred_struct: Structure, true_struct: Structure,
                 pred_pattern: np.ndarray, true_pattern: np.ndarray) -> SampleMetrics:
    sg_match = {}
    for tol in SG_TOLS:
        sg_p = space_group_number(pred_struct, symprec=tol)
        sg_t = space_group_number(true_struct, symprec=tol)
        sg_match[tol] = (sg_p == sg_t and sg_p > 0)
    return SampleMetrics(
        material_id=material_id,
        composition_ok=composition_match(pred_struct, true_struct),
        sg_match=sg_match,
        rmsd=coord_rmsd(pred_struct, true_struct),
        rwp=rwp(pred_pattern, true_pattern),
        pearson=r_pearson(pred_pattern, true_pattern),
    )


def aggregate(metrics: Iterable[SampleMetrics]) -> dict:
    metrics = list(metrics)
    n = len(metrics)
    if n == 0:
        return {"n": 0}
    rmsd = np.array([m.rmsd for m in metrics])
    out = {
        "n": n,
        "composition_match_rate": float(np.mean([m.composition_ok for m in metrics])),
        "rmsd_mean": float(np.nanmean(rmsd)),
        "rmsd_median": float(np.nanmedian(rmsd)),
        "match_rate (StructureMatcher)": float(np.mean(~np.isnan(rmsd))),
        "rwp_mean": float(np.mean([m.rwp for m in metrics])),
        "pearson_mean": float(np.mean([m.pearson for m in metrics])),
        "headline_all_correct": float(np.mean([m.all_correct for m in metrics])),
    }
    for tol in SG_TOLS:
        out[f"sg_match@{tol}"] = float(np.mean([m.sg_match[tol] for m in metrics]))
    return out