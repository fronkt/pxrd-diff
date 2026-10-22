"""Re-match saved predicted structures (03_sample.py --save-structures) against the
reference structures at chosen StructureMatcher tolerances.

Gate G3 uses DiffCSP's tolerances (stol 0.5, ltol 0.3, angle 10) on the null-pattern
run, to compare with DiffCSP's published 1-sample MP-20 match rate (51.5 %). The
denominator is --n: structures without a saved prediction count as misses.

Usage: python rematch.py REPO STRUCTS_JSONL [--n 1000] [--stol 0.5 --ltol 0.3 --angle 10]
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

ap = argparse.ArgumentParser()
ap.add_argument("repo")
ap.add_argument("structs")
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--stol", type=float, default=0.5)
ap.add_argument("--ltol", type=float, default=0.3)
ap.add_argument("--angle", type=float, default=10.0)
ap.add_argument("--out-json", default=None)
args = ap.parse_args()

df = pd.read_csv(Path(args.repo) / "data" / "raw" / "test.csv").head(args.n)
ref = dict(zip(df["material_id"], df["cif"]))
pred = {}
for line in open(args.structs):
    r = json.loads(line)
    pred[r["material_id"]] = r["cif"]

sm = StructureMatcher(stol=args.stol, ltol=args.ltol, angle_tol=args.angle)
hits, rms = 0, []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for mid, cif in ref.items():
        if mid not in pred:
            continue
        a = Structure.from_str(pred[mid], fmt="cif")
        b = Structure.from_str(cif, fmt="cif")
        try:
            r = sm.get_rms_dist(a, b)
        except Exception:
            r = None
        if r is not None:
            hits += 1
            rms.append(r[0])
out = {"n": len(ref), "n_pred": len(pred), "matches": hits, "match_rate": hits / len(ref),
       "rms_mean": sum(rms) / len(rms) if rms else None,
       "tolerances": {"stol": args.stol, "ltol": args.ltol, "angle_tol": args.angle}}
print(json.dumps(out))
if args.out_json:
    Path(args.out_json).write_text(json.dumps(out, indent=1))
