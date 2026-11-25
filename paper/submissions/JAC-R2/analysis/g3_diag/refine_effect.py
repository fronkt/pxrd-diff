"""G3 diagnosis (2026-11-25): what does the Debye refinement step do to correct structures?

Takes the UNREFINED v30 null-pattern predictions (1 sample, no refinement), runs the same
refine_structure() call 03_sample.py uses for the paper arms (Adam on fractional coordinates,
1 - Pearson(DiffPXRD, target pattern), cell fixed), and re-matches before and after with both the
paper matcher (ltol 0.2, stol 0.3, angle 5) and DiffCSP's (ltol 0.3, stol 0.5, angle 10).
Also records the shortest interatomic distance and the coordinate displacement.

Usage: python refine_effect.py REPO STRUCTS_JSONL --steps 200 --lr 1e-3 [--n 1000] [--out-json P]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

MATCHERS = {"paper": dict(ltol=0.2, stol=0.3, angle_tol=5.0),
            "diffcsp": dict(ltol=0.3, stol=0.5, angle_tol=10.0)}


def match_pair(args):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    pred_d, ref_cif = args
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = Structure.from_str(ref_cif, fmt="cif")
        pred = Structure.from_dict(pred_d)
        for name, tol in MATCHERS.items():
            try:
                r = StructureMatcher(**tol).get_rms_dist(pred, ref)
            except Exception:
                r = None
            out[name] = r is not None
        dm = pred.distance_matrix + np.eye(len(pred)) * 1e9
        out["min_dist"] = float(dm.min())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("structs")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=50)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo / "src"))
    import pandas as pd
    from pymatgen.core import Structure
    from pxrd_diff.data import CrystalPXRDDataset
    from pxrd_diff.debye import DiffPXRD
    from pxrd_diff.sampler import lattice_params_to_matrix, refine_structure

    torch.manual_seed(0)
    torch.set_num_threads(6)
    ref = pd.read_csv(repo / "data" / "raw" / "test.csv").head(args.n)
    ref_cif = dict(zip(ref["material_id"], ref["cif"]))
    ds = CrystalPXRDDataset(repo / "data", split="test")
    pattern = {}
    for i in range(min(args.n, len(ds))):
        b = ds[i]
        pattern[b["material_id"]] = b["pxrd_pattern"]
    preds = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for line in open(args.structs):
            r = json.loads(line)
            if r["material_id"] in ref_cif:
                preds.append((r["material_id"], Structure.from_str(r["cif"], fmt="cif")))
    debye = DiffPXRD(n_bins=256, hkl_max=5).eval()   # 03_sample.py defaults

    before, after, disp, loss0, loss1 = [], [], [], [], []
    for lo in range(0, len(preds), args.bs):
        chunk = preds[lo:lo + args.bs]
        N = max(len(s) for _, s in chunk)
        B = len(chunk)
        fc = torch.zeros(B, N, 3); at = torch.zeros(B, N, dtype=torch.long)
        mk = torch.zeros(B, N, dtype=torch.bool); lp = torch.zeros(B, 6)
        for i, (_, s) in enumerate(chunk):
            n = len(s)
            fc[i, :n] = torch.tensor(s.frac_coords % 1.0, dtype=torch.float32)
            at[i, :n] = torch.tensor([sp.Z for sp in s.species])
            mk[i, :n] = True
            lp[i] = torch.tensor(s.lattice.parameters, dtype=torch.float32)
        lat = lattice_params_to_matrix(lp)
        tgt = torch.stack([pattern[m] for m, _ in chunk])
        fc2, _, hist = refine_structure(fc, at, lat, mk, tgt, debye, steps=args.steps, lr=args.lr)
        loss0.append(hist[0]); loss1.append(hist[-1])
        d = (fc2 - fc); d = d - torch.round(d)
        for i, (m, s) in enumerate(chunk):
            n = len(s)
            disp.append(float((d[i, :n] * torch.tensor(s.lattice.abc)).norm(dim=-1).max()))
            s2 = Structure(s.lattice, s.species, fc2[i, :n].numpy())
            before.append((s.as_dict(), ref_cif[m]))
            after.append((s2.as_dict(), ref_cif[m]))
        print(f"refined {lo + B}/{len(preds)}  loss {hist[0]:.3f} -> {hist[-1]:.3f}", flush=True)

    with Pool(6) as p:
        mb = p.map(match_pair, before, chunksize=8)
        ma = p.map(match_pair, after, chunksize=8)
    n = args.n
    summ = {"steps": args.steps, "lr": args.lr, "n": n,
            "refine_loss_mean_first": float(np.mean(loss0)), "refine_loss_mean_last": float(np.mean(loss1)),
            "max_atom_displacement_A_median": float(np.median(disp)),
            "max_atom_displacement_A_p90": float(np.percentile(disp, 90))}
    for name in MATCHERS:
        b = [x[name] for x in mb]; a = [x[name] for x in ma]
        summ[name] = {"before": sum(b) / n, "after": sum(a) / n,
                      "lost": sum(x and not y for x, y in zip(b, a)),
                      "gained": sum(y and not x for x, y in zip(b, a))}
    summ["min_dist_lt_0.5A"] = {"before": sum(x["min_dist"] < 0.5 for x in mb),
                                "after": sum(x["min_dist"] < 0.5 for x in ma)}
    print(json.dumps(summ, indent=1))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
