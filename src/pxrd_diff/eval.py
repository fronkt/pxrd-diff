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

import multiprocessing as _mp
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from pymatgen.core.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Tolerances at which to evaluate space group recovery.
SG_TOLS = (0.01, 0.05, 0.1, 0.2)
DEFAULT_RMSD_THRESHOLD = 0.1   # Angstrom; tunable in ablations.

# spglib (SpacegroupAnalyzer) and StructureMatcher.get_rms_dist (attempt_supercell)
# are C-extension calls that can hang for hours on pathological predicted cells.
# A Python signal/alarm cannot interrupt a C call, so we run the structure-domain
# comparison in a worker process and HARD-KILL it on overrun. A structure that
# cannot be scored in bounded time is treated as a non-match (NaN rmsd, sg=miss),
# consistent with the "match" semantics used throughout.
STRUCTUREMATCH_TIMEOUT_S = 30


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
    scalar Angstrom value comparable across structures of different sizes. NOTE: in
    the evaluation loop this is called via `structure_domain_metrics` so it runs
    under a hard process timeout; called directly it has no timeout.
    """
    matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol,
                               primitive_cell=False, scale=True, attempt_supercell=True)
    try:
        rms = matcher.get_rms_dist(pred, true)
    except Exception:
        return float("nan")
    if rms is None:
        return float("nan")
    return float(rms[0])    # (rms, max) tuple -> rms


# ---------- Hard-timeout wrapper for the C-heavy structure-domain ops ----------------

def _struct_domain_worker(pred: Structure, true: Structure,
                          sg_tols, ltol, stol, angle_tol):
    """Run in a worker process: space-group match across tols + coord RMSD."""
    sg = {}
    for tol in sg_tols:
        sg_p = space_group_number(pred, symprec=tol)
        sg_t = space_group_number(true, symprec=tol)
        sg[tol] = (sg_p == sg_t and sg_p > 0)
    return sg, coord_rmsd(pred, true, ltol, stol, angle_tol)


# A single reusable "spawn" worker (spawn, not fork: the parent holds a CUDA
# context after sampling, and fork-after-CUDA is unsafe). Recreated after a kill.
_POOL = None


def _get_pool():
    global _POOL
    if _POOL is None:
        _POOL = _mp.get_context("spawn").Pool(processes=1)
    return _POOL


def structure_domain_metrics(pred: Structure, true: Structure,
                             sg_tols=SG_TOLS, ltol=0.2, stol=0.3, angle_tol=5.0,
                             timeout: int = STRUCTUREMATCH_TIMEOUT_S):
    """(sg_match dict, rmsd) computed under a hard process timeout.

    On overrun the worker is terminated and recreated, and we return a clean miss
    (all sg False, NaN rmsd) so one pathological structure cannot stall the run.
    """
    global _POOL
    try:
        res = _get_pool().apply_async(
            _struct_domain_worker, (pred, true, sg_tols, ltol, stol, angle_tol))
        return res.get(timeout=timeout)
    except _mp.TimeoutError:
        try:
            _POOL.terminate(); _POOL.join()
        except Exception:
            pass
        _POOL = None  # force a fresh worker next call
        return ({t: False for t in sg_tols}, float("nan"))
    except Exception:
        return ({t: False for t in sg_tols}, float("nan"))


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
    # spglib + StructureMatcher run under a hard process timeout (see above).
    sg_match, rmsd = structure_domain_metrics(pred_struct, true_struct)
    return SampleMetrics(
        material_id=material_id,
        composition_ok=composition_match(pred_struct, true_struct),
        sg_match=sg_match,
        rmsd=rmsd,
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