"""Smoke tests for the evaluation harness.

We verify the metric implementations behave sensibly against trivial baselines:
    - identity baseline (predict ground truth)         -> 100% correct, 0 RMSD
    - random baseline (predict a different structure)  -> ~0% headline correct

If either of these check fails, the metric code is wrong and we should fix it
before training any model on top.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pymatgen.io.cif import CifParser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pxrd_diff.eval import (  # noqa: E402
    aggregate,
    coord_rmsd,
    composition_match,
    evaluate_one,
    r_pearson,
    rwp,
    space_group_number,
)
from pxrd_diff.simulator import PXRDSimulator  # noqa: E402

N_SAMPLES = 8


def _load_structures(n: int = N_SAMPLES):
    df = pd.read_csv(ROOT / "data" / "raw" / "test.csv", nrows=n)
    out = []
    for _, row in df.iterrows():
        s = CifParser.from_str(row["cif"]).parse_structures(primitive=False)[0]
        out.append((row["material_id"], s))
    return out


@pytest.fixture(scope="module")
def loaded():
    structs = _load_structures()
    sim = PXRDSimulator()
    pats = [sim.simulate(s) for _, s in structs]
    return structs, pats


def test_identity_baseline_is_perfect(loaded):
    structs, pats = loaded
    metrics = [
        evaluate_one(mid, s, s, p, p)
        for (mid, s), p in zip(structs, pats)
    ]
    agg = aggregate(metrics)
    assert agg["composition_match_rate"] == 1.0
    assert agg["sg_match@0.1"] == 1.0
    assert agg["rmsd_mean"] < 1e-6
    assert agg["headline_all_correct"] == 1.0
    assert agg["rwp_mean"] < 1e-6
    assert agg["pearson_mean"] > 0.9999


def test_random_baseline_is_bad(loaded):
    structs, pats = loaded
    # Predict structure i+1 (a different material) for sample i.
    metrics = []
    for i, ((mid, s_true), p_true) in enumerate(zip(structs, pats)):
        j = (i + 1) % len(structs)
        s_pred = structs[j][1]
        p_pred = pats[j]
        metrics.append(evaluate_one(mid, s_pred, s_true, p_pred, p_true))
    agg = aggregate(metrics)
    # No structure should perfectly match a different one.
    assert agg["headline_all_correct"] == 0.0
    # Pearson on a different pattern should be much lower than the identity case.
    assert agg["pearson_mean"] < 0.99


def test_rwp_zero_for_identity():
    a = np.linspace(1.0, 5.0, 100)
    assert rwp(a, a) < 1e-9


def test_rwp_grows_with_error():
    a = np.linspace(1.0, 5.0, 100)
    assert rwp(a + 0.5, a) > rwp(a + 0.1, a) > 0


def test_pearson_self_one():
    a = np.random.default_rng(0).random(200)
    assert r_pearson(a, a) == pytest.approx(1.0, abs=1e-9)


def test_composition_match_basic():
    structs = _load_structures(2)
    s1 = structs[0][1]
    s2 = structs[1][1]
    assert composition_match(s1, s1) is True
    # Different materials shouldn't share reduced formulas (highly probable).
    assert composition_match(s1, s2) is False


def test_space_group_returns_int():
    s = _load_structures(1)[0][1]
    sg = space_group_number(s, symprec=0.1)
    assert isinstance(sg, int) and 1 <= sg <= 230