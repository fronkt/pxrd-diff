"""Re-score deCIFer-generated CIFs through OUR StructureMatcher harness, so the
match% is apples-to-apples with PXRD-Diff / DiffractGPT / PXRDnet (paper Table 3)."""
import argparse, glob, gzip, pickle, json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from pymatgen.core import Structure

ROOT = Path("/workspace/pxrd-diff")
sys.path.insert(0, str(ROOT / "src"))
from pxrd_diff.eval import aggregate, evaluate_one          # noqa: E402
from pxrd_diff.simulator import PXRDConfig, PXRDSimulator    # noqa: E402
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--eval-folder", required=True)
ap.add_argument("--cache", default=str(ROOT / "data/cache/test.npz"))
ap.add_argument("--grid", default=str(ROOT / "data/cache/two_theta.npy"))
ap.add_argument("--csv", default=str(ROOT / "data/raw/test.csv"))
ap.add_argument("--out-json", required=True)
args = ap.parse_args()

tt = np.load(args.grid)
cache = np.load(args.cache, allow_pickle=True)
pat_by_id = dict(zip(list(cache["material_id"]), cache["pattern"]))
df = pd.read_csv(args.csv)
cif_by_id = dict(zip(df["material_id"], df["cif"]))
sim = PXRDSimulator(PXRDConfig(two_theta_min=float(tt[0]),
                               two_theta_max=float(tt[-1]),
                               two_theta_step=float(tt[1] - tt[0])))

files = sorted(glob.glob(args.eval_folder + "/**/*.pkl.gz", recursive=True))
metrics, parse_fail, miss = [], 0, 0
for f in files:
    d = pickle.load(gzip.open(f, "rb"))
    mid = "_".join(Path(f).stem.split("_")[:-1])   # strip trailing _<rep>
    true_cif = cif_by_id.get(mid); true_pat = pat_by_id.get(mid)
    if true_cif is None or true_pat is None:
        miss += 1; continue
    try:
        pred = Structure.from_str(d.get("cif_string_gen"), fmt="cif")
        true = Structure.from_str(true_cif, fmt="cif")
        pred_pat = sim.simulate(pred).astype(np.float32)
    except Exception:
        parse_fail += 1; continue
    metrics.append(evaluate_one(mid, pred, true, pred_pat, true_pat.astype(np.float32)))

summary = aggregate(metrics)
summary.update(n_files=len(files), parse_fail=parse_fail, miss=miss,
               config={"model": "deCIFer_v1 (Johansen et al. 2025, arXiv:2502.02189)",
                       "ckpt": "decifer_v1_ckpt.pt", "cond": "XRD + composition",
                       "scoring": "pxrd_diff.eval StructureMatcher ltol0.2 stol0.3 angle_tol5"})
json.dump(summary, open(args.out_json, "w"), indent=2)
print(json.dumps(summary, indent=2))
