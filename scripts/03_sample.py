"""Sample crystal structures from a trained PXRD-Diff model and evaluate.

Loads a checkpoint, samples structures for test-set PXRD patterns using DDIM,
reconstructs pymatgen Structures, runs the eval harness, and prints metrics.

Usage:
  python scripts/03_sample.py --ckpt runs/smoke/ckpt_final.pt --n 16
  python scripts/03_sample.py --ckpt runs/gpu_v16/ckpt_final.pt --n 1000 --true-lattice
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pxrd_diff.data import CrystalPXRDDataset                        # noqa: E402
from pxrd_diff.eval import aggregate, evaluate_one                    # noqa: E402
from pxrd_diff.model.denoiser import CrystalDenoiser                 # noqa: E402
from pxrd_diff.model.pxrd_encoder import PXRDEncoder                 # noqa: E402
from pxrd_diff.sampler import DDIMSampler                             # noqa: E402
from pxrd_diff.simulator import PXRDSimulator                        # noqa: E402

from pymatgen.core import Lattice, Structure                          # noqa: E402


def is_valid_lattice(lp: np.ndarray) -> bool:
    """Reject obviously degenerate lattice parameters before hitting spglib."""
    a, b, c, al, be, ga = lp
    if any(x < 0.5 or x > 100 for x in [a, b, c]):
        return False
    if any(x < 10 or x > 170 for x in [al, be, ga]):
        return False
    return True


def tensor_to_structure(frac_coords: torch.Tensor, atom_types: torch.Tensor,
                        lattice_params: torch.Tensor, num_atoms: int,
                        ) -> Structure:
    """Convert model output tensors to a pymatgen Structure."""
    coords = frac_coords[:num_atoms].cpu().numpy()
    species = atom_types[:num_atoms].cpu().tolist()
    lp = lattice_params.cpu().numpy()
    lat = Lattice.from_parameters(
        a=max(float(lp[0]), 0.5),
        b=max(float(lp[1]), 0.5),
        c=max(float(lp[2]), 0.5),
        alpha=np.clip(float(lp[3]), 10.0, 170.0),
        beta=np.clip(float(lp[4]), 10.0, 170.0),
        gamma=np.clip(float(lp[5]), 10.0, 170.0),
    )
    # Clamp coords to [0, 1)
    coords = coords % 1.0
    return Structure(lat, species, coords)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint .pt")
    ap.add_argument("--n", type=int, default=16, help="Number of test samples")
    ap.add_argument("--ddim-steps", type=int, default=50)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--split", default="test")
    ap.add_argument("--true-lattice", action="store_true",
                    help="Use ground-truth lattice params for coord-only eval")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    train_args = ckpt["args"]
    d_model = train_args.get("d_model", 256)
    n_layers = train_args.get("n_layers", 3)

    encoder = PXRDEncoder(d_model=d_model).to(device)
    denoiser = CrystalDenoiser(d_model=d_model, n_layers=n_layers).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    denoiser.load_state_dict(ckpt["denoiser"])
    encoder.eval()
    denoiser.eval()

    lat_mean = ckpt.get("lat_mean")
    lat_std = ckpt.get("lat_std")

    print(f"Loaded checkpoint: step={ckpt['step']}, d_model={d_model}")
    if args.true_lattice:
        print("Mode: coord-only eval with ground-truth lattice")

    # Dataset
    ds = CrystalPXRDDataset(ROOT / "data", split=args.split)

    # Collate a batch
    indices = list(range(min(args.n, len(ds))))
    batch_items = [ds[i] for i in indices]

    pxrd = torch.stack([b["pxrd_pattern"] for b in batch_items]).to(device)
    atom_types = torch.stack([b["atom_types"] for b in batch_items]).to(device)
    lattice = torch.stack([b["lattice"] for b in batch_items]).to(device)
    lattice_params_true = torch.stack([b["lattice_params"] for b in batch_items]).to(device)
    mask = torch.stack([b["mask"] for b in batch_items]).to(device)
    num_atoms = [b["num_atoms"].item() for b in batch_items]
    material_ids = [b["material_id"] for b in batch_items]

    # Sample
    sampler = DDIMSampler(encoder, denoiser, n_steps=args.ddim_steps,
                          eta=args.eta, lat_mean=lat_mean, lat_std=lat_std)

    print(f"Sampling {len(indices)} structures ({args.ddim_steps} DDIM steps)...")
    t0 = time.perf_counter()
    pred_coords, pred_lat_params = sampler.sample(pxrd, atom_types, lattice, mask)
    elapsed = time.perf_counter() - t0
    print(f"Sampling done in {elapsed:.1f}s ({elapsed/len(indices):.2f}s/sample)")

    # Simulate PXRD for predicted structures and evaluate
    sim = PXRDSimulator()
    metrics_list = []

    for i, idx in enumerate(indices):
        na = num_atoms[i]
        mid = material_ids[i]

        # Choose lattice for building predicted structure
        eval_lat_params = lattice_params_true[i] if args.true_lattice else pred_lat_params[i]

        # Validate lattice (skip if using ground truth)
        if not args.true_lattice:
            lp = eval_lat_params.cpu().numpy()
            if not is_valid_lattice(lp):
                print(f"  [{mid}] invalid lattice: {lp.round(1).tolist()}")
                continue

        # Build predicted structure
        try:
            pred_struct = tensor_to_structure(
                pred_coords[i], atom_types[i], eval_lat_params, na
            )
        except Exception as e:
            print(f"  [{mid}] failed to build structure: {e}")
            continue

        # Build true structure
        true_struct = tensor_to_structure(
            batch_items[i]["frac_coords"], batch_items[i]["atom_types"],
            batch_items[i]["lattice_params"], na
        )

        # Simulate PXRD for predicted structure
        try:
            pred_pattern = sim.simulate(pred_struct)
        except Exception:
            pred_pattern = np.zeros(sim.cfg.n_bins, dtype=np.float32)

        true_pattern = batch_items[i]["pxrd_pattern"].numpy()

        m = evaluate_one(mid, pred_struct, true_struct,
                         pred_pattern, true_pattern)
        metrics_list.append(m)

        sg_ok = "Y" if m.sg_match.get(0.1, False) else "N"
        rmsd_str = f"{m.rmsd:.3f}" if not np.isnan(m.rmsd) else "  NaN"
        print(f"  {mid:>14s}  comp={'Y' if m.composition_ok else 'N'}  "
              f"sg@0.1={sg_ok}  rmsd={rmsd_str}  rwp={m.rwp:.3f}  "
              f"pearson={m.pearson:.3f}  all={'Y' if m.all_correct else 'N'}")

    # Aggregate
    if metrics_list:
        agg = aggregate(metrics_list)
        print("\n--- Aggregate Metrics ---")
        for k, v in agg.items():
            if isinstance(v, float):
                print(f"  {k:>30s}: {v:.4f}")
            else:
                print(f"  {k:>30s}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
