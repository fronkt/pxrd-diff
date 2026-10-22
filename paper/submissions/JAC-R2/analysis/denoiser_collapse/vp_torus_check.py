"""How much of the VP-on-torus x0 target is a structure-free contraction?

Compares, on the training objective (periodic MSE of x0_pred vs x0, t ~ U(0,1)):
  identity:   x0_pred = x_t
  uncontract: x0_pred = x_t / sqrt(alpha_bar)   (knows nothing about the structure)
and the same two after a random global translation of each structure's origin,
to show the target depends on the arbitrary CIF origin.
"""
import sys
from pathlib import Path

import torch

repo = Path(sys.argv[1])
sys.path.insert(0, str(repo / "src"))
from pxrd_diff.data import CrystalPXRDDataset            # noqa: E402
from pxrd_diff.diffusion import DiffusionProcess, cosine_alpha_bar  # noqa: E402

ds = CrystalPXRDDataset(repo / "data", split="train")
n = 4096
idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[:n]
items = [ds[int(i)] for i in idx]
coords = torch.stack([b["frac_coords"] for b in items])
mask = torch.stack([b["mask"] for b in items])
dp = DiffusionProcess()
wrap = lambda x: (x + 0.5) % 1.0 - 0.5


def run(c0, label):
    torch.manual_seed(1)
    t = torch.rand(n)
    xt, _ = dp.forward_q(c0, t, wrap=True)
    a = cosine_alpha_bar(t).sqrt().view(-1, 1, 1)
    out = {}
    for name, pred in (("identity", xt), ("uncontract", xt / a)):
        out[name] = dp.loss(pred, c0, mask, periodic=True).item()
    # per noise band
    bands = {}
    for lo, hi in ((0, .25), (.25, .5), (.5, .75), (.75, 1.0)):
        sel = (t >= lo) & (t < hi)
        bands[f"{lo}-{hi}"] = tuple(round(dp.loss(p[sel], c0[sel], mask[sel], periodic=True).item(), 4)
                                   for p in (xt, xt / a))
    print(f"{label:22s} identity {out['identity']:.4f}  uncontract {out['uncontract']:.4f}  "
          f"bands (identity, uncontract): {bands}")


run(coords, "CIF origin")
shift = torch.rand(n, 1, 3, generator=torch.Generator().manual_seed(2))
run((coords + shift) % 1.0, "random origin")
