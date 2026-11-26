"""Phase 19 B1 check: the Niggli-reduced dataset describes the same crystals as the raw one.

For every structure of each split: same atom count, same volume (rel. 1e-4), and the reduced
structure matches the raw one under a tight StructureMatcher (ltol 1e-3, stol 1e-3, angle 0.1,
no primitive reduction, no scaling). For a sample per split, the simulated pattern of the
reduced structure equals the cached pattern (max abs difference after the cache's
normalisation). Reports the fraction of cells the reduction changed and the angle range.

Usage: python check_niggli.py REPO [--splits train val test] [--n-pattern 50]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np


def check_one(args):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Lattice, Structure
    raw, red = args
    s0 = Structure(Lattice(raw[2]), raw[1].tolist(), raw[0])
    s1 = Structure(Lattice(red[2]), red[1].tolist(), red[0])
    ok_n = bool(len(s0) == len(s1) and (raw[1] == red[1]).all())
    ok_v = bool(abs(s1.volume / s0.volume - 1) < 1e-4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm = StructureMatcher(ltol=1e-3, stol=1e-3, angle_tol=0.1, primitive_cell=False, scale=False)
        ok_m = bool(sm.fit(s0, s1))
    changed = bool(np.abs(raw[3] - red[3]).max() > 1e-3)
    return ok_n, ok_v, ok_m, changed, red[3][3:].tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--n-pattern", type=int, default=50)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo / "src"))
    from pymatgen.core import Lattice, Structure
    from pxrd_diff.data import CrystalPXRDDataset
    from pxrd_diff.simulator import PXRDConfig, PXRDSimulator

    out = {}
    for split in args.splits:
        raw = CrystalPXRDDataset(repo / "data", split=split, n_peaks=0)
        red = CrystalPXRDDataset(repo / "data", split=split, n_peaks=0, niggli=True)
        with Pool(args.workers) as p:
            res = p.map(check_one, list(zip(raw._structures, red._structures)), chunksize=64)
        ang = np.array([r[4] for r in res])
        o = {"n": len(res),
             "atoms_species_same": int(sum(r[0] for r in res)),
             "volume_same": int(sum(r[1] for r in res)),
             "tight_match": int(sum(r[2] for r in res)),
             "cell_changed_frac": round(float(sum(r[3] for r in res)) / len(res), 4),
             "reduced_angle_min_max": [float(ang.min()), float(ang.max())]}
        # pattern invariance on a sample, against the cached patterns
        sim = PXRDSimulator(PXRDConfig())
        rng = np.random.default_rng(0)
        idx = rng.choice(len(red), min(args.n_pattern, len(red)), replace=False)
        diffs = []
        for i in idx:
            f, z, lat, _, _ = red._structures[i]
            pat = sim.simulate(Structure(Lattice(lat), z.tolist(), f))
            cached = red.patterns[i]
            diffs.append(float(np.abs(pat - cached).max() / max(np.abs(cached).max(), 1e-12)))
        o["pattern_max_rel_diff"] = max(diffs)
        out[split] = o
        print(split, json.dumps(o), flush=True)
    print(json.dumps(out, indent=1))
    bad = [s for s, o in out.items()
           if not (o["atoms_species_same"] == o["volume_same"] == o["tight_match"] == o["n"]
                   and o["pattern_max_rel_diff"] < 1e-3)]
    if bad:
        sys.exit(f"NIGGLI CHECK FAILED for {bad}")
    print("NIGGLI CHECK OK")


if __name__ == "__main__":
    main()
