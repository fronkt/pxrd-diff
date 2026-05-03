"""Compare ablation runs: λ_debye ∈ {0, 0.1, 1, 10}.

Loads gpu_v6/v7/v8/v9 final checkpoints, samples N test structures from each,
runs the eval harness, and prints a side-by-side table.

Usage:
  python scripts/05_compare_ablation.py [--n 64]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pxrd_diff.data import CrystalPXRDDataset
from pxrd_diff.eval import aggregate, evaluate_one
from pxrd_diff.model.denoiser import CrystalDenoiser
from pxrd_diff.model.pxrd_encoder import PXRDEncoder
from pxrd_diff.sampler import DDIMSampler
from pxrd_diff.simulator import PXRDSimulator

from pymatgen.core import Lattice, Structure


RUNS = [
    ("gpu_v10", 0.0),
    ("gpu_v11", 1.0),
    ("gpu_v13", 1.0),  # x0 residual prediction
]


def is_valid_lattice(lp: np.ndarray) -> bool:
    a, b, c, al, be, ga = lp
    if any(x < 0.5 or x > 100 for x in [a, b, c]):
        return False
    if any(x < 10 or x > 170 for x in [al, be, ga]):
        return False
    return True


def tensor_to_structure(frac_coords, atom_types, lattice_params, num_atoms):
    coords = (frac_coords[:num_atoms].cpu().numpy()) % 1.0
    species = atom_types[:num_atoms].cpu().tolist()
    lp = lattice_params.cpu().numpy()
    lat = Lattice.from_parameters(
        a=max(float(lp[0]), 0.5), b=max(float(lp[1]), 0.5), c=max(float(lp[2]), 0.5),
        alpha=np.clip(float(lp[3]), 10.0, 170.0),
        beta=np.clip(float(lp[4]), 10.0, 170.0),
        gamma=np.clip(float(lp[5]), 10.0, 170.0),
    )
    return Structure(lat, species, coords)


def evaluate_checkpoint(ckpt_path, batch_items, device, sim, ddim_steps,
                        use_true_lattice=False):
    """Evaluate a checkpoint. If use_true_lattice, swap predicted lattice with
    ground truth (since lat loss never converged to better than random baseline,
    isolating coord quality for the ablation comparison)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = ckpt["args"]
    d_model = train_args.get("d_model", 256)
    n_layers = train_args.get("n_layers", 3)
    n_heads = train_args.get("n_heads", 4)

    encoder = PXRDEncoder(d_model=d_model).to(device).eval()
    denoiser = CrystalDenoiser(d_model=d_model, n_layers=n_layers, n_heads=n_heads).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])
    denoiser.load_state_dict(ckpt["denoiser"])

    pxrd = torch.stack([b["pxrd_pattern"] for b in batch_items]).to(device)
    atom_types = torch.stack([b["atom_types"] for b in batch_items]).to(device)
    lattice = torch.stack([b["lattice"] for b in batch_items]).to(device)
    true_lat_params = torch.stack([b["lattice_params"] for b in batch_items]).to(device)
    mask = torch.stack([b["mask"] for b in batch_items]).to(device)
    wyckoff = None
    if train_args.get("use_wyckoff", False):
        wyckoff = torch.stack([b["wyckoff"] for b in batch_items]).to(device)

    sampler = DDIMSampler(encoder, denoiser, n_steps=ddim_steps, eta=0.0,
                          lat_mean=ckpt.get("lat_mean"), lat_std=ckpt.get("lat_std"),
                          predict_x0=train_args.get("predict_x0", False))
    pred_coords, pred_lat_params = sampler.sample(pxrd, atom_types, lattice, mask, wyckoff=wyckoff)
    if use_true_lattice:
        pred_lat_params = true_lat_params

    metrics_list = []
    skipped = 0
    for i, b in enumerate(batch_items):
        na = b["num_atoms"].item()
        lp = pred_lat_params[i].cpu().numpy()
        if not is_valid_lattice(lp):
            skipped += 1
            continue
        try:
            pred_struct = tensor_to_structure(pred_coords[i], atom_types[i], pred_lat_params[i], na)
            true_struct = tensor_to_structure(b["frac_coords"], b["atom_types"], b["lattice_params"], na)
            try:
                pred_pattern = sim.simulate(pred_struct)
            except Exception:
                pred_pattern = np.zeros(sim.cfg.n_bins, dtype=np.float32)
            metrics_list.append(evaluate_one(
                b["material_id"], pred_struct, true_struct,
                pred_pattern, b["pxrd_pattern"].numpy(),
            ))
        except Exception:
            skipped += 1
            continue

    agg = aggregate(metrics_list) if metrics_list else {"n": 0}
    agg["skipped"] = skipped
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--ddim-steps", type=int, default=50)
    ap.add_argument("--use-true-lattice", action="store_true",
                    help="Use ground-truth lattice (legacy hack, no longer needed)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = CrystalPXRDDataset(ROOT / "data", split="test")
    indices = list(range(min(args.n, len(ds))))
    batch_items = [ds[i] for i in indices]
    sim = PXRDSimulator()

    print(f"Evaluating {len(batch_items)} test structures per run, {args.ddim_steps} DDIM steps\n")

    results = {}
    for name, lam in RUNS:
        ckpt_path = ROOT / "runs" / name / "ckpt_final.pt"
        if not ckpt_path.exists():
            print(f"[{name} λ={lam}] checkpoint not found, skipping")
            continue
        print(f"=== {name} (λ_debye={lam}) ===")
        t0 = time.perf_counter()
        agg = evaluate_checkpoint(ckpt_path, batch_items, device, sim,
                                  args.ddim_steps, use_true_lattice=args.use_true_lattice)
        results[(name, lam)] = agg
        print(f"  elapsed: {time.perf_counter()-t0:.1f}s")
        for k, v in agg.items():
            if isinstance(v, float):
                print(f"    {k:>30s}: {v:.4f}")
            else:
                print(f"    {k:>30s}: {v}")
        print()

    print("\n=== Side-by-side summary ===")
    print(f"{'metric':>30s}  " + "  ".join(f"λ={lam:>5}" for _, lam in RUNS))
    keys = ["n", "rmsd_mean", "rmsd_median", "match_rate (StructureMatcher)",
            "rwp_mean", "pearson_mean", "headline_all_correct",
            "sg_match@0.1", "composition_match_rate"]
    for k in keys:
        row = [results.get((name, lam), {}).get(k, "—") for name, lam in RUNS]
        row_str = "  ".join(f"{v:>7.4f}" if isinstance(v, float) else f"{v:>7}" for v in row)
        print(f"{k:>30s}  {row_str}")


if __name__ == "__main__":
    main()
