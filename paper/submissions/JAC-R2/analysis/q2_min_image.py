"""Referee R2-Q2: does periodic_distances() (component-wise rounding) return the
minimum-image distance? Compare against an explicit search over neighbouring
images on the MP-20 test structures the evaluation uses (first n of the split)."""
import sys, json, itertools
from collections import defaultdict
from pathlib import Path
import numpy as np, torch

REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO / "src"))
from pxrd_diff.data import CrystalPXRDDataset                 # noqa: E402
from pxrd_diff.model.denoiser import periodic_distances, RBF_CUTOFF  # noqa: E402

def system(sg):
    return ("triclinic" if sg <= 2 else "monoclinic" if sg <= 15 else
            "orthorhombic" if sg <= 74 else "tetragonal" if sg <= 142 else
            "trigonal" if sg <= 167 else "hexagonal" if sg <= 194 else "cubic")

ds = CrystalPXRDDataset(REPO / "data", split="test")
sgs = np.load(REPO / "data/cache/test.npz", allow_pickle=True)["spacegroup"]
n = min(int(sys.argv[2]) if len(sys.argv) > 2 else 1000, len(ds))
shifts = torch.tensor(list(itertools.product(range(-2, 3), repeat=3)), dtype=torch.float32)

agg = defaultdict(lambda: dict(structs=0, structs_wrong=0, pairs=0, pairs_wrong=0,
                               nn_wrong=0, atoms=0, err=[], nn_err=[]))
for i in range(n):
    b = ds[i]
    na = int(b["num_atoms"])
    f = b["frac_coords"][:na].unsqueeze(0).float()
    L = b["lattice"].unsqueeze(0).float()
    m = torch.ones(1, na, dtype=torch.bool)
    d_round = periodic_distances(f, L, m)[0]                       # (na, na)
    delta = f[0].unsqueeze(1) - f[0].unsqueeze(0)
    delta = delta - delta.round()
    cand = (delta.unsqueeze(2) + shifts.view(1, 1, -1, 3)) @ L[0]  # (na, na, 125, 3)
    d_min = cand.norm(dim=-1).min(dim=-1).values
    off = ~torch.eye(na, dtype=torch.bool)
    e = (d_round - d_min)[off]
    wrong = e > 1e-3
    # nearest-neighbour distance per atom (what bonding features hinge on)
    big = torch.full_like(d_round, 1e9)
    nn_r = torch.where(off, d_round, big).min(dim=1).values
    nn_m = torch.where(off, d_min, big).min(dim=1).values
    s = system(int(sgs[i]))
    for key in (s, "ALL"):
        a = agg[key]
        a["structs"] += 1
        a["structs_wrong"] += int(wrong.any())
        a["pairs"] += int(off.sum())
        a["pairs_wrong"] += int(wrong.sum())
        a["atoms"] += na
        a["nn_wrong"] += int(((nn_r - nn_m) > 1e-3).sum()) if na > 1 else 0
        a["err"].extend(e[wrong].tolist())
        a["nn_err"].extend((nn_r - nn_m)[(nn_r - nn_m) > 1e-3].tolist())

out = {}
order = ["triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal",
         "hexagonal", "cubic", "ALL"]
print(f"{'system':13s} {'n':>5s} {'struct%':>8s} {'pair%':>7s} {'NN%':>6s} "
      f"{'err_med':>8s} {'err_max':>8s} {'nnerr_med':>9s}")
for k in order:
    if k not in agg:
        continue
    a = agg[k]
    med = float(np.median(a["err"])) if a["err"] else 0.0
    mx = float(np.max(a["err"])) if a["err"] else 0.0
    nnm = float(np.median(a["nn_err"])) if a["nn_err"] else 0.0
    row = dict(n=a["structs"], structs_wrong_pct=100 * a["structs_wrong"] / a["structs"],
               pairs_wrong_pct=100 * a["pairs_wrong"] / max(1, a["pairs"]),
               nn_wrong_pct=100 * a["nn_wrong"] / max(1, a["atoms"]),
               err_median_A=med, err_max_A=mx, nn_err_median_A=nnm)
    out[k] = row
    print(f"{k:13s} {row['n']:5d} {row['structs_wrong_pct']:8.1f} {row['pairs_wrong_pct']:7.2f} "
          f"{row['nn_wrong_pct']:6.2f} {med:8.3f} {mx:8.3f} {nnm:9.3f}")
print("RBF cutoff", RBF_CUTOFF)
json.dump(out, open(sys.argv[3] if len(sys.argv) > 3 else "q2_min_image.json", "w"), indent=2)
