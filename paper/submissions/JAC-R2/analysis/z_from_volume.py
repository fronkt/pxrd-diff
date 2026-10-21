"""JAC R2 Q3: how well can the number of formula units per cell (Z) be recovered
from the cell volume and the composition alone, as classical structure solution
does after indexing?

Per-element atomic volumes v_el are fitted by non-negative least squares on the
MP-20 TRAIN split (V_cell = sum_el n_el v_el). For each of the 1000 evaluation
structures, Z is then estimated as the feasible integer (Z * atoms per formula
unit <= 20, the MP-20 cap) whose predicted volume Z * V_formula is closest to the
cell volume in log ratio. Two cell volumes are used:
  (i)  the true cell (what the oracle arm is given), and
  (ii) the indexed cell (crystal system given; what the indexer arm is given). The
       indexed cell can be a sub- or super-cell of the reference, so its target is
       the formula-unit count that cell actually holds, N_true * V_idx / V_true,
       which must be a whole number of formula units for the cell to be usable.

Usage: python z_from_volume.py REPO OUT_JSON [--workers 6]
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls


def _parse(cif: str):
    warnings.simplefilter("ignore")
    from pymatgen.io.cif import CifParser
    s = CifParser.from_str(cif).parse_structures(primitive=False)[0]
    counts = {el.symbol: int(round(n)) for el, n in s.composition.items()}
    return counts, float(s.volume), len(s)


def _system(sg: int) -> str:
    return ("triclinic" if sg <= 2 else "monoclinic" if sg <= 15 else
            "orthorhombic" if sg <= 74 else "tetragonal" if sg <= 142 else
            "trigonal" if sg <= 167 else "hexagonal" if sg <= 194 else "cubic")


def _estimate_z(volume: float, v_formula: float, n_formula: int, max_atoms: int = 20) -> int:
    zs = np.arange(1, max(1, max_atoms // n_formula) + 1)
    return int(zs[np.argmin(np.abs(np.log(volume / (zs * v_formula))))])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("out_json")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    repo = Path(args.repo)

    train = pd.read_csv(repo / "data/raw/train.csv")
    test = pd.read_csv(repo / "data/raw/test.csv").head(1000)
    with Pool(args.workers) as pool:
        tr = pool.map(_parse, train["cif"], chunksize=64)
        te = pool.map(_parse, test["cif"], chunksize=16)

    elements = sorted({el for c, _, _ in tr + te for el in c})
    col = {el: i for i, el in enumerate(elements)}
    A = np.zeros((len(tr), len(elements)))
    for r, (c, _, _) in enumerate(tr):
        for el, n in c.items():
            A[r, col[el]] = n
    v_el, _ = nnls(A, np.array([v for _, v, _ in tr]))
    fit_err = np.abs(A @ v_el / np.array([v for _, v, _ in tr]) - 1)
    print(f"fitted {len(elements)} element volumes on {len(tr)} train cells; "
          f"median |V_fit/V - 1| = {np.median(fit_err):.3f}")

    sgs = np.load(repo / "data/cache/test.npz", allow_pickle=True)["spacegroup"]
    idx_rows = {r["mid"]: r for r in json.load(open(
        repo / "paper/phase9_results/index_benchmark_v2_native.json"))["rows"]}

    per = []
    for i, (c, vol, n_atoms) in enumerate(te):
        g = math.gcd(*c.values())
        reduced = {el: n // g for el, n in c.items()}
        n_f = sum(reduced.values())
        z_true = n_atoms // n_f
        v_f = sum(n * v_el[col[el]] for el, n in reduced.items())
        row = dict(mid=test["material_id"].iloc[i], system=_system(int(sgs[i])),
                   n_atoms=n_atoms, n_formula=n_f, z_true=z_true,
                   z_est_true_cell=_estimate_z(vol, v_f, n_f))
        ir = idx_rows.get(row["mid"])
        if ir and ir.get("pred_params") and ir.get("vol_ratio"):
            v_idx = vol / ir["vol_ratio"]
            fu_in_cell = z_true / ir["vol_ratio"]            # formula units the indexed cell holds
            row.update(indexed=True, idx_cell_whole=bool(
                fu_in_cell >= 0.95 and abs(fu_in_cell - round(fu_in_cell)) < 0.05 * fu_in_cell),
                z_target_idx=int(round(fu_in_cell)),
                z_est_idx_cell=_estimate_z(v_idx, v_f, n_f, max_atoms=10 ** 6),
                idx_same_volume=bool(abs(ir["vol_ratio"] - 1) < 0.03))
        else:
            row.update(indexed=False)
        per.append(row)

    def summarise(rows):
        n = len(rows)
        ok_true = sum(r["z_est_true_cell"] == r["z_true"] for r in rows)
        z1 = sum(r["z_true"] == 1 for r in rows)
        idx = [r for r in rows if r["indexed"]]
        whole = [r for r in idx if r["idx_cell_whole"]]
        ok_idx = sum(r["z_est_idx_cell"] == r["z_target_idx"] for r in whole)
        same = [r for r in idx if r["idx_same_volume"]]
        ok_same = sum(r["z_est_idx_cell"] == r["z_true"] for r in same)
        return dict(n=n,
                    z_true_cell_correct=ok_true, z_true_cell_pct=100 * ok_true / max(1, n),
                    baseline_always_z1_pct=100 * z1 / max(1, n),
                    indexed=len(idx), indexed_whole_fu=len(whole),
                    z_idx_cell_correct=ok_idx,
                    z_idx_cell_pct_of_whole=100 * ok_idx / max(1, len(whole)),
                    z_idx_cell_pct_of_all=100 * ok_idx / max(1, n),
                    indexed_same_volume=len(same), z_same_volume_correct=ok_same)

    by = defaultdict(list)
    for r in per:
        by[r["system"]].append(r)
    out = {"ALL": summarise(per)}
    for k in ["triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal",
              "hexagonal", "cubic"]:
        out[k] = summarise(by[k])
    out["method"] = ("NNLS per-element volumes on MP-20 train; Z = feasible integer "
                     "closest in log volume ratio; true cell vs given-system indexed cell")
    out["element_volumes_A3"] = {el: round(float(v_el[col[el]]), 3) for el in elements}
    Path(args.out_json).write_text(json.dumps({"summary": out, "rows": per}, indent=1))
    for k, v in out.items():
        if isinstance(v, dict) and "n" in v:
            print(f"{k:13s} n={v['n']:4d}  Z from true cell {v['z_true_cell_pct']:5.1f}% "
                  f"(always-Z=1 {v['baseline_always_z1_pct']:5.1f}%)  indexed {v['indexed']:4d}, "
                  f"whole-fu {v['indexed_whole_fu']:4d}, Z from indexed cell "
                  f"{v['z_idx_cell_pct_of_whole']:5.1f}% of those ({v['z_idx_cell_pct_of_all']:5.1f}% of all)")


if __name__ == "__main__":
    main()
