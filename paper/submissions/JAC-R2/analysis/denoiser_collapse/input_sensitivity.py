"""JAC R2: does a trained denoiser's coordinate output depend on its per-atom inputs?

For n test structures at three noise levels, the coordinate output is compared
against itself after (i) re-drawing the noisy coordinates, (ii) replacing every
species with a random one, (iii) giving it another structure's PXRD pattern, and
its spread across the atoms of a structure is measured. All values are relative
L2 changes over real (unpadded) atoms. A collapsed denoiser (v21, v22, v23 at 10 k
steps) gives ~1e-7 for (i), (ii) and the spread.

GATE (fixed before the R2 architecture-fix run): at t = 0.5 the re-noised-coords
change and the within-structure spread must both be >= 1e-2, and the
random-species change >= 1e-3. Exit code 0 = pass, 3 = fail.

Usage: python input_sensitivity.py REPO CKPT OUT_JSON [--n 128]
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
ap.add_argument("out_json")
ap.add_argument("--n", type=int, default=128)
args = ap.parse_args()
repo = Path(args.repo)
sys.path.insert(0, str(repo / "src"))
from pxrd_diff.data import CrystalPXRDDataset          # noqa: E402
from pxrd_diff.diffusion import DiffusionProcess        # noqa: E402
from pxrd_diff.model.denoiser import CrystalDenoiser    # noqa: E402
from pxrd_diff.model.pxrd_encoder import PXRDEncoder    # noqa: E402

ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
ta = ck["args"]
enc = PXRDEncoder(d_model=ta.get("d_model", 256)).eval()
den = CrystalDenoiser(d_model=ta.get("d_model", 256), n_layers=ta.get("n_layers", 3),
                      min_image=ta.get("periodic_dist", "round") == "min_image",
                      arch=ta.get("denoiser_arch", "legacy")).eval()
enc.load_state_dict(ck["encoder"])
den.load_state_dict(ck["denoiser"])

ds = CrystalPXRDDataset(repo / "data", split="test")
items = [ds[i] for i in range(args.n)]
st = lambda k: torch.stack([b[k] for b in items])
pxrd, types, coords, lat, lat_p, mask = (st("pxrd_pattern"), st("atom_types"), st("frac_coords"),
                                         st("lattice"), st("lattice_params"), st("mask"))
nl0 = (lat_p - ck["lat_mean"]) / ck["lat_std"]
dp = DiffusionProcess()
m = mask.unsqueeze(-1).float()


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a - b) * m).norm() / (b * m).norm())


out = {"ckpt": args.ckpt, "step": ck.get("step"), "arch": ta.get("denoiser_arch", "legacy"),
       "n": args.n, "levels": {}}
with torch.no_grad():
    g, feats = enc(pxrd)
    for tv in (0.1, 0.5, 0.9):
        t = torch.full((args.n,), tv)
        torch.manual_seed(0)
        nc, _ = dp.forward_q(coords, t, wrap=True)
        nl, _ = dp.forward_q(nl0, t)
        c0, _ = den(nc, types, lat, t, g, feats, mask, nl)
        torch.manual_seed(1)
        nc2, _ = dp.forward_q(coords, t, wrap=True)
        c_coords, _ = den(nc2, types, lat, t, g, feats, mask, nl)
        torch.manual_seed(2)
        rand_types = torch.where(mask, torch.randint(1, 84, types.shape), types)
        c_types, _ = den(nc, rand_types, lat, t, g, feats, mask, nl)
        c_pxrd, _ = den(nc, types, lat, t, g.roll(1, 0), feats.roll(1, 0), mask, nl)
        cm = (c0 * m).sum(1, keepdim=True) / m.sum(1, keepdim=True)
        spread = float(((c0 - cm) * m).norm() / (c0 * m).norm())
        out["levels"][str(tv)] = dict(renoised_coords=rel(c_coords, c0), random_species=rel(c_types, c0),
                                      other_pxrd=rel(c_pxrd, c0), within_structure_spread=spread)
lv = out["levels"]["0.5"]
out["gate_pass"] = bool(lv["renoised_coords"] >= 1e-2 and lv["within_structure_spread"] >= 1e-2
                        and lv["random_species"] >= 1e-3)
Path(args.out_json).write_text(json.dumps(out, indent=1))
print(f"step {out['step']} arch {out['arch']}: " + "  ".join(
    f"t={t}: coords {v['renoised_coords']:.2e} species {v['random_species']:.2e} "
    f"pxrd {v['other_pxrd']:.2e} spread {v['within_structure_spread']:.2e}" for t, v in out["levels"].items())
    + f"  GATE {'PASS' if out['gate_pass'] else 'FAIL'}")
sys.exit(0 if out["gate_pass"] else 3)
