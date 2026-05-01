"""Verify differentiable Debye scattering against pymatgen XRDCalculator.

Computes Pearson correlation and Rwp between Debye-predicted and pymatgen-simulated
PXRD patterns on 50 test structures. Also verifies gradients flow through coords.

Usage:
  python scripts/04_verify_debye.py [--n 50]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pxrd_diff.data import CrystalPXRDDataset
from pxrd_diff.debye import DebyePXRD, debye_pxrd_loss
from pxrd_diff.eval import r_pearson


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = CrystalPXRDDataset(ROOT / "data", split="test")
    debye = DebyePXRD(n_bins=256).to(device)

    pearsons, losses = [], []

    for i in range(min(args.n, len(ds))):
        sample = ds[i]
        coords = sample["frac_coords"].unsqueeze(0).to(device).requires_grad_(True)
        types = sample["atom_types"].unsqueeze(0).to(device)
        lat = sample["lattice"].unsqueeze(0).to(device)
        mask = sample["mask"].unsqueeze(0).to(device)
        pxrd_target = sample["pxrd_pattern"].unsqueeze(0).to(device)
        mid = sample["material_id"]

        pred = debye(coords, types, lat, mask)  # (1, 256)

        loss = debye_pxrd_loss(pred, pxrd_target, n_bins_debye=256)
        loss.backward()

        grad_ok = coords.grad is not None and not coords.grad.isnan().any()

        pred_np = pred[0].detach().cpu().numpy()
        # Downsample target to 256 bins for comparison
        target_ds = torch.nn.functional.interpolate(
            pxrd_target.unsqueeze(1), size=256, mode="linear", align_corners=True
        ).squeeze(1)
        target_np = target_ds[0].cpu().numpy()

        p = r_pearson(pred_np, target_np)
        pearsons.append(p)
        losses.append(loss.item())

        if i < 10 or (i + 1) % 10 == 0:
            print(f"  [{i+1:3d}] {mid:>14s}  pearson={p:.4f}  loss={loss.item():.4f}  grad={'OK' if grad_ok else 'FAIL'}")

        coords.grad = None

    pearsons = np.array(pearsons)
    losses = np.array(losses)

    print(f"\n--- Debye vs pymatgen XRDCalculator ({len(pearsons)} structures) ---")
    print(f"  Pearson: mean={pearsons.mean():.4f}  std={pearsons.std():.4f}  "
          f"min={pearsons.min():.4f}  max={pearsons.max():.4f}")
    print(f"  Loss:    mean={losses.mean():.4f}  std={losses.std():.4f}")
    print(f"  Pearson > 0.5: {(pearsons > 0.5).sum()}/{len(pearsons)}")
    print(f"  Pearson > 0.7: {(pearsons > 0.7).sum()}/{len(pearsons)}")


if __name__ == "__main__":
    main()
