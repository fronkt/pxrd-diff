"""JAC R2 gate G2 for the periodic model (model_family "csp"), fixed before the run.

PASS needs all of:
  (a) coordinate loss on the fixed validation batch (val.jsonl from 14_train_csp.py,
      the zero predictor scores 1.0) averaged over sigma in {0.01, 0.02, 0.05, 0.1, 0.2}
      < 0.7, in the learned-cell and in the given-cell mode;
  (b) at mid noise (t = 0.5, sigma = 0.05), relative L2 change of the coordinate
      output over real atoms, in both modes: re-noised coordinates >= 1e-2,
      within-structure spread >= 1e-2, random species >= 1e-3, another
      structure's pattern >= 1e-3. (The collapsed v22 denoiser gives ~1e-7.)
Exit code 0 = pass, 3 = fail.

Usage: python csp_gate.py REPO CKPT VAL_JSONL OUT_JSON [--n 128]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("repo")
ap.add_argument("ckpt")
ap.add_argument("val_jsonl")
ap.add_argument("out_json")
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--split", default="val")
args = ap.parse_args()
repo = Path(args.repo)
sys.path.insert(0, str(repo / "src"))
from pxrd_diff.data import CrystalPXRDDataset              # noqa: E402
from pxrd_diff.diffusion import cosine_alpha_bar            # noqa: E402
from pxrd_diff.model.csp_denoiser import CSPDenoiser        # noqa: E402
from pxrd_diff.model.pxrd_encoder import PXRDEncoder        # noqa: E402
from pxrd_diff.torus import TorusVE                         # noqa: E402

LOSS_SIGMAS = (0.01, 0.02, 0.05, 0.1, 0.2)

ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
ta = ck["args"]
step = ck["step"]
rows = [json.loads(line) for line in open(args.val_jsonl)]
row = next(r for r in rows if r["step"] == step)
loss = {mode: sum(row[f"{mode}@{s}"] for s in LOSS_SIGMAS) / len(LOSS_SIGMAS)
        for mode in ("learned", "given")}

enc = PXRDEncoder(d_model=ta["d_model"]).eval()
den = CSPDenoiser(d_model=ta["d_model"], n_layers=ta["n_layers"], n_heads=ta["n_heads"],
                  n_freq=ta["n_freq"]).eval()
enc.load_state_dict(ck["encoder"])
den.load_state_dict(ck["denoiser"])
torus = TorusVE(ta["sigma_min"], ta["sigma_max"])
lat_mean, lat_std = ck["lat_mean"], ck["lat_std"]

ds = CrystalPXRDDataset(repo / "data", split=args.split)
items = [ds[i] for i in range(args.n)]
st = lambda k: torch.stack([b[k] for b in items])
pxrd, types, f0, lp, mask = (st(k) for k in ("pxrd_pattern", "atom_types", "frac_coords",
                                             "lattice_params", "mask"))
m = mask.unsqueeze(-1).float()
l0 = (lp - lat_mean) / lat_std


def rel(a, b):
    return float(((a - b) * m).norm() / (b * m).norm())


out = {"ckpt": args.ckpt, "step": step, "val_loss": loss, "sens": {}}
with torch.no_grad():
    g, feats = enc(pxrd)
    t = torch.full((args.n,), 0.5)
    ab = cosine_alpha_bar(torch.tensor(0.5)).item()
    for mode, given in (("learned", False), ("given", True)):
        torch.manual_seed(0)
        f_t, _, _ = torus.forward_q(f0, t)
        l_in = l0 if given else ab ** 0.5 * l0 + (1 - ab) ** 0.5 * torch.randn_like(l0)
        gv = torch.full((args.n,), given)
        run = lambda f, ty, gg, ff: den(f, ty, l_in * lat_std + lat_mean, l_in, t, gg, ff, mask, gv)[0]
        c0 = run(f_t, types, g, feats)
        torch.manual_seed(1)
        f_t2, _, _ = torus.forward_q(f0, t)
        torch.manual_seed(2)
        rand_types = torch.where(mask, torch.randint(1, 84, types.shape), types)
        cm = (c0 * m).sum(1, keepdim=True) / m.sum(1, keepdim=True)
        out["sens"][mode] = dict(
            renoised_coords=rel(run(f_t2, types, g, feats), c0),
            random_species=rel(run(f_t, rand_types, g, feats), c0),
            other_pxrd=rel(run(f_t, types, g.roll(1, 0), feats.roll(1, 0)), c0),
            within_structure_spread=float(((c0 - cm) * m).norm() / (c0 * m).norm()),
        )

s = out["sens"]
out["gate_pass"] = bool(
    all(loss[k] < 0.7 for k in loss)
    and all(s[k]["renoised_coords"] >= 1e-2 and s[k]["within_structure_spread"] >= 1e-2
            and s[k]["random_species"] >= 1e-3 and s[k]["other_pxrd"] >= 1e-3 for k in s))
Path(args.out_json).write_text(json.dumps(out, indent=1))
print(f"step {step}: val loss learned {loss['learned']:.3f} given {loss['given']:.3f}  "
      + "  ".join(f"{k}: coords {v['renoised_coords']:.2e} species {v['random_species']:.2e} "
                  f"pxrd {v['other_pxrd']:.2e} spread {v['within_structure_spread']:.2e}"
                  for k, v in s.items())
      + f"  GATE {'PASS' if out['gate_pass'] else 'FAIL'}")
sys.exit(0 if out["gate_pass"] else 3)
