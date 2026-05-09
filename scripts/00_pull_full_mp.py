"""Pull full Materials Project structures (Phase 8) and write CDVAE-format CSVs.

Filters to keep the dataset compatible with the model trained on MP-20:
    - num_sites <= 20 (the existing dataset's max_atoms)
    - energy_above_hull <= 0.1 eV/atom (near-stable; dropping highly-unstable
      compounds keeps the structures physically meaningful)
    - deprecated == False

Excludes any material_ids that appear in the MP-20 test split — those stay
the held-out test set so Phase 8's evaluation can be compared apples-to-apples
with v18, v19, v20.

Output: data/raw/full_mp/{train,val,test}.csv with the same column names the
existing data pipeline expects:
    material_id, formation_energy_per_atom, band_gap, pretty_formula,
    e_above_hull, elements, cif, spacegroup.number

The MP-20 test split is also copied verbatim into data/raw/full_mp/test.csv
so we can reuse the existing simulation cache for evaluation comparability.

Usage:
    MP_API_KEY=... python3 scripts/00_pull_full_mp.py [--limit N]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap on structures pulled (debug). None = all matching.")
    ap.add_argument("--out-dir", type=str, default=str(ROOT / "data" / "raw" / "full_mp"))
    ap.add_argument("--max-sites", type=int, default=20)
    ap.add_argument("--max-ehull", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.05)
    args = ap.parse_args()

    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        print("ERROR: MP_API_KEY env var not set", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load MP-20 test split material_ids to exclude ---------------------
    mp20_test_ids: set[str] = set()
    mp20_test_csv = ROOT / "data" / "raw" / "test.csv"
    if mp20_test_csv.exists():
        mp20_test_df = pd.read_csv(mp20_test_csv)
        mp20_test_ids = set(mp20_test_df["material_id"].tolist())
        print(f"[mp20] loaded {len(mp20_test_ids)} test material_ids to exclude from training")
    else:
        print("WARN: data/raw/test.csv not found; cannot exclude MP-20 test split", file=sys.stderr)

    # ---- Pull from MP via mp-api -------------------------------------------
    from mp_api.client import MPRester

    print(f"[mp] querying: num_sites <= {args.max_sites}, "
          f"e_above_hull <= {args.max_ehull}")
    t0 = time.perf_counter()
    fields = [
        "material_id",
        "formula_pretty",
        "elements",
        "formation_energy_per_atom",
        "band_gap",
        "energy_above_hull",
        "symmetry",
        "structure",
        "deprecated",
    ]
    with MPRester(api_key=api_key) as r:
        docs = r.summary.search(
            num_sites=(1, args.max_sites),
            energy_above_hull=(0.0, args.max_ehull),
            deprecated=False,
            fields=fields,
        )
    print(f"[mp] got {len(docs):,} docs in {time.perf_counter() - t0:.0f}s")

    if args.limit is not None:
        docs = docs[: args.limit]
        print(f"[mp] limited to {len(docs):,} docs (--limit)")

    # ---- Build rows -----------------------------------------------------------
    rows = []
    skipped_test = 0
    skipped_err = 0
    for i, d in enumerate(docs):
        mid = str(d.material_id)
        if mid in mp20_test_ids:
            skipped_test += 1
            continue
        try:
            cif = d.structure.to(fmt="cif")
            sg_num = int(d.symmetry.number)
            elements = [str(e) for e in d.elements]
            rows.append({
                "material_id": mid,
                "formation_energy_per_atom": float(d.formation_energy_per_atom),
                "band_gap": float(d.band_gap) if d.band_gap is not None else 0.0,
                "pretty_formula": str(d.formula_pretty),
                "e_above_hull": float(d.energy_above_hull),
                "elements": str(elements),
                "cif": cif,
                "spacegroup.number": sg_num,
            })
        except Exception as e:
            skipped_err += 1
            if skipped_err <= 5:
                print(f"  [skip] {mid}: {e}")
        if (i + 1) % 5000 == 0:
            print(f"  built {len(rows):,} rows so far ({i+1:,}/{len(docs):,})")

    print(f"[mp] built {len(rows):,} rows  "
          f"(excluded {skipped_test} mp20-test, {skipped_err} errors)")

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(rows))
    n_val = int(round(len(rows) * args.val_frac))
    val_idx = set(perm[:n_val].tolist())
    train_rows = [rows[i] for i in range(len(rows)) if i not in val_idx]
    val_rows = [rows[i] for i in range(len(rows)) if i in val_idx]

    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    print(f"[mp] wrote {train_path} ({train_path.stat().st_size / 1e6:.1f} MB, {len(train_df):,} rows)")
    print(f"[mp] wrote {val_path}   ({val_path.stat().st_size / 1e6:.1f} MB, {len(val_df):,} rows)")

    # ---- Copy MP-20 test verbatim so the pipeline finds test.csv ------------
    test_path = out_dir / "test.csv"
    if mp20_test_csv.exists():
        import shutil
        shutil.copyfile(mp20_test_csv, test_path)
        print(f"[mp] copied MP-20 test split -> {test_path} "
              f"({len(mp20_test_df):,} rows)")
    else:
        print("WARN: no MP-20 test.csv to copy", file=sys.stderr)

    print("\nDone. Suggested next steps:")
    print(f"  python scripts/01_simulate_pxrd.py "
          f"--data-root {out_dir.parent.parent}  # add this CLI flag if missing")
    print("  Or symlink: ln -sf data/raw/full_mp data/raw_full_mp_link")
    return 0


if __name__ == "__main__":
    sys.exit(main())
