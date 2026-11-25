"""G3 diagnosis (2026-11-25): where does the v30 null-pattern arm lose against DiffCSP?

For every test structure of a saved-structures run (03_sample.py --save-structures), compare the
predicted and reference cells after Niggli reduction (volume ratio, reduced lengths and angles),
re-match at DiffCSP's tolerances (stol 0.5, ltol 0.3, angle 10), and record the shortest
interatomic distance. Misses are split into
  lattice_off : the reduced cells differ by more than the matcher's ltol / angle_tol, so no
                coordinates could have matched;
  coords_off  : the cells agree within tolerance but the atoms do not.
Results are broken down by atom count. Read-only on its inputs.

Usage: python null_breakdown.py REPO STRUCTS_JSONL [--n 1000] [--out-json PATH]
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

STOL, LTOL, ANGLE = 0.5, 0.3, 10.0


def niggli_params(s: Structure) -> np.ndarray:
    return np.array(s.lattice.get_niggli_reduced_lattice().parameters)


def one(args):
    mid, pred_cif, ref_cif = args
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = Structure.from_str(ref_cif, fmt="cif")
        out = {"material_id": mid, "n_atoms": len(ref), "have_pred": pred_cif is not None}
        if pred_cif is None:
            return out
        try:
            pred = Structure.from_str(pred_cif, fmt="cif")
        except ValueError:          # flagged miss (index fallback / unbuildable) saved without a cell
            out["have_pred"] = False
            return out
        # StructureMatcher compares primitive cells; reduce both sides the same way.
        rp, pp = ref.get_primitive_structure(), pred.get_primitive_structure()
        a, b = niggli_params(rp), niggli_params(pp)
        out["vol_ratio"] = pp.volume / len(pp) / (rp.volume / len(rp))
        # ltol in StructureMatcher is a fractional length tolerance after scaling volumes
        # (scale=True), so compare lengths after scaling the predicted cell to the reference volume.
        scale = (rp.volume / len(rp) / (pp.volume / len(pp))) ** (1 / 3)
        out["len_err_scaled"] = float(np.max(np.abs(b[:3] * scale / a[:3] - 1)))
        out["ang_err"] = float(np.max(np.abs(b[3:] - a[3:])))
        out["n_prim_pred"], out["n_prim_ref"] = len(pp), len(rp)
        # the matcher's own lattice test: after scaling to the reference volume per atom, is there
        # a choice of lattice vectors within ltol (fractional length) and angle_tol (degrees)?
        lp = pp.lattice.scale(rp.volume / len(rp) * len(pp))
        out["lattice_maps"] = next(iter(rp.lattice.find_all_mappings(lp, LTOL, ANGLE)), None) is not None
        dm = pred.distance_matrix + np.eye(len(pred)) * 1e9
        out["min_dist"] = float(dm.min()) if len(pred) > 1 else None
        sm = StructureMatcher(stol=STOL, ltol=LTOL, angle_tol=ANGLE)
        try:
            r = sm.get_rms_dist(pred, ref)
        except Exception:
            r = None
        out["match"] = r is not None
        out["rms"] = r[0] if r is not None else None
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("structs")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    df = pd.read_csv(Path(args.repo) / "data" / "raw" / "test.csv").head(args.n)
    pred = {}
    for line in open(args.structs):
        r = json.loads(line)
        pred[r["material_id"]] = r["cif"]
    jobs = [(m, pred.get(m), c) for m, c in zip(df["material_id"], df["cif"])]
    with Pool(args.workers) as p:
        rows = p.map(one, jobs, chunksize=8)

    have = [r for r in rows if r["have_pred"]]
    for r in have:
        r["lattice_ok"] = r["lattice_maps"] and r["n_prim_pred"] == r["n_prim_ref"]
    n = len(rows)
    hits = sum(r["match"] for r in have)
    miss = [r for r in have if not r["match"]]
    summ = {
        "n": n, "n_pred": len(have), "match_rate": hits / n,
        "miss_lattice_off": sum(not r["lattice_ok"] for r in miss),
        "miss_coords_off": sum(r["lattice_ok"] for r in miss),
        "miss_prim_count_differs": sum(r["n_prim_pred"] != r["n_prim_ref"] for r in miss),
        "lattice_ok_rate": sum(r["lattice_ok"] for r in have) / n,
        "match_given_lattice_ok": (sum(r["match"] for r in have if r["lattice_ok"])
                                   / max(1, sum(r["lattice_ok"] for r in have))),
        "vol_ratio_median": float(np.median([r["vol_ratio"] for r in have])),
        "vol_ratio_iqr": [float(np.percentile([r["vol_ratio"] for r in have], q)) for q in (25, 75)],
        "min_dist_lt_0.5A": sum((r["min_dist"] or 9) < 0.5 for r in have),
        "min_dist_lt_1.0A": sum((r["min_dist"] or 9) < 1.0 for r in have),
    }
    by_n = defaultdict(lambda: [0, 0, 0])
    for r in have:
        k = "1-4" if r["n_atoms"] <= 4 else "5-8" if r["n_atoms"] <= 8 else "9-12" if r["n_atoms"] <= 12 else "13-20"
        by_n[k][0] += 1; by_n[k][1] += r["match"]; by_n[k][2] += r["lattice_ok"]
    summ["by_n_atoms"] = {k: {"n": v[0], "match": round(v[1] / v[0], 3), "lattice_ok": round(v[2] / v[0], 3)}
                          for k, v in sorted(by_n.items(), key=lambda kv: int(kv[0].split("-")[0]))}
    print(json.dumps(summ, indent=1))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
