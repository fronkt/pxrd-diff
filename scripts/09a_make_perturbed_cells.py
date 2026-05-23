"""Make a fake `lat-from-index` JSON by perturbing ground-truth cells with
calibrated Gaussian noise. Drives the Phase 9.1.2 sensitivity study:
how does match rate degrade as a function of cell error?

Output schema matches `09_index_benchmark.py` (top-1 mode): a JSON with
`rows: [{mid, pred_params, ...}, ...]` so it plugs straight into
`03_sample.py --lat-from-index`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--sigma-len", type=float, required=True,
                    help="σ for lengths a,b,c in Å (Gaussian, additive)")
    ap.add_argument("--sigma-ang", type=float, default=0.0,
                    help="σ for angles α,β,γ in ° (Gaussian, additive)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", type=str, default=str(ROOT / "data" / "raw" / "test.csv"))
    ap.add_argument("--cache", type=str, default=str(ROOT / "data" / "cache" / "test.npz"))
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    cache = np.load(args.cache, allow_pickle=True)
    mids = list(cache["material_id"])[: args.n]
    df = pd.read_csv(args.csv)
    cif_by_id = dict(zip(df["material_id"], df["cif"]))

    rows = []
    for mid in mids:
        cif = cif_by_id.get(mid)
        if cif is None:
            continue
        try:
            s = Structure.from_str(cif, fmt="cif")
            conv = SpacegroupAnalyzer(s, symprec=0.01).get_conventional_standard_structure()
            true_p = np.array(conv.lattice.parameters, float)
            system = SpacegroupAnalyzer(s, symprec=0.01).get_crystal_system()
        except Exception:
            continue
        noise = np.concatenate([
            rng.normal(0.0, args.sigma_len, size=3),
            rng.normal(0.0, args.sigma_ang, size=3),
        ])
        pert = true_p + noise
        # clamp to physically valid range
        pert[:3] = np.clip(pert[:3], 0.5, 100.0)
        pert[3:] = np.clip(pert[3:], 10.0, 170.0)
        rows.append(dict(
            mid=mid, system=system,
            pred_params=[round(float(x), 4) for x in pert],
            sigma_len=args.sigma_len, sigma_ang=args.sigma_ang,
            len_mae=round(float(np.mean(np.abs(noise[:3]))), 4),
            ang_mae=round(float(np.mean(np.abs(noise[3:]))), 3),
        ))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    overall = dict(n=len(rows), sigma_len=args.sigma_len, sigma_ang=args.sigma_ang,
                   mean_len_mae=round(float(np.mean([r["len_mae"] for r in rows])), 4),
                   mean_ang_mae=round(float(np.mean([r["ang_mae"] for r in rows])), 3))
    out.write_text(json.dumps(dict(overall=overall, rows=rows), indent=2))
    print(f"[perturb] σ_len={args.sigma_len}Å σ_ang={args.sigma_ang}° → {out}")
    print(f"  realised mean MAE: len {overall['mean_len_mae']} Å,"
          f" ang {overall['mean_ang_mae']}°  (n={len(rows)})")


if __name__ == "__main__":
    sys.exit(main())
