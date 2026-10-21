"""JAC R2 Q1: how much does a trained denoiser's output depend on the lattice
behind its distance features?

For n test structures and several noise levels, the denoiser is evaluated on the
same noisy coordinates and noisy lattice parameters with its distance features
computed in (a) the reference cell (training-time behaviour of v10-v22), (b) the
noisy cell l_t (v23 behaviour), (c) the reference cell scaled by 1.5 and (d) a
far cell (all distances above the RBF cutoff, i.e. no edge information). Reported:
relative change of the coordinate and lattice outputs against (a).

Usage: python lattice_sensitivity.py REPO CKPT OUT_JSON [--n 256]
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
ap.add_argument("--n", type=int, default=256)
args = ap.parse_args()
repo = Path(args.repo)
sys.path.insert(0, str(repo / "src"))
from pxrd_diff.data import CrystalPXRDDataset                     # noqa: E402
from pxrd_diff.diffusion import DiffusionProcess, cosine_alpha_bar  # noqa: E402,F401
from pxrd_diff.model.denoiser import CrystalDenoiser               # noqa: E402
from pxrd_diff.model.pxrd_encoder import PXRDEncoder               # noqa: E402
from pxrd_diff.sampler import noisy_lattice_matrix                 # noqa: E402

torch.manual_seed(0)
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
ta = ck["args"]
enc = PXRDEncoder(d_model=ta.get("d_model", 256)).eval()
den = CrystalDenoiser(d_model=ta.get("d_model", 256), n_layers=ta.get("n_layers", 3),
                      min_image=ta.get("periodic_dist", "round") == "min_image",
                      arch=ta.get("denoiser_arch", "legacy")).eval()
enc.load_state_dict(ck["encoder"])
den.load_state_dict(ck["denoiser"])
lat_mean, lat_std = ck["lat_mean"], ck["lat_std"]

ds = CrystalPXRDDataset(repo / "data", split="test")
items = [ds[i] for i in range(args.n)]
st = lambda k: torch.stack([b[k] for b in items])
pxrd, types, coords, lat, lat_p, mask = (st("pxrd_pattern"), st("atom_types"), st("frac_coords"),
                                         st("lattice"), st("lattice_params"), st("mask"))
diffusion = DiffusionProcess()
far = torch.eye(3).expand(args.n, 3, 3) * 1000.0

out = {"ckpt": args.ckpt, "n": args.n, "trained_dist_lattice": ta.get("dist_lattice", "clean"),
       "periodic_dist": ta.get("periodic_dist", "round"), "levels": {}}
with torch.no_grad():
    g, feats = enc(pxrd)
    for tval in (0.1, 0.3, 0.5, 0.7, 0.9):
        t = torch.full((args.n,), tval)
        nc, _ = diffusion.forward_q(coords, t, wrap=True)
        nl, _ = diffusion.forward_q((lat_p - lat_mean) / lat_std, t)
        cells = {"reference": lat, "noisy": noisy_lattice_matrix(nl, lat_mean, lat_std),
                 "reference_x1.5": lat * 1.5, "far (no edges)": far}
        res = {k: den(nc, types, L, t, g, feats, mask, nl) for k, L in cells.items()}
        c0, l0 = res["reference"]
        m = mask.unsqueeze(-1).float()
        row = {}
        for k, (c, l) in res.items():
            if k == "reference":
                continue
            row[k] = {"coord_rel_change": float(((c - c0) * m).norm() / (c0 * m).norm()),
                      "lattice_rel_change": float((l - l0).norm() / l0.norm())}
        out["levels"][str(tval)] = row
        print(f"t={tval}: " + "  ".join(f"{k}: coord {v['coord_rel_change']:.4f} lat {v['lattice_rel_change']:.4f}"
                                        for k, v in row.items()))
Path(args.out_json).write_text(json.dumps(out, indent=1))
