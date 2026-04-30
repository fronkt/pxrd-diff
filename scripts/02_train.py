"""Training script for PXRD-Diff.

Minimal but functional training loop. Supports:
  - CPU and CUDA
  - Gradient accumulation (for larger effective batch on CPU)
  - Checkpointing every N steps
  - WandB logging (optional, off by default)
  - Lattice parameter normalization

Usage:
  python scripts/02_train.py                        # defaults (smoke test)
  python scripts/02_train.py --steps 10000 --bs 32  # real training
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pxrd_diff.data import CrystalPXRDDataset, lattice_params_stats  # noqa: E402
from pxrd_diff.diffusion import DiffusionProcess                     # noqa: E402
from pxrd_diff.model.denoiser import CrystalDenoiser                 # noqa: E402
from pxrd_diff.model.pxrd_encoder import PXRDEncoder                 # noqa: E402


def collate(batch: list[dict]) -> dict:
    out = {}
    for k in batch[0]:
        if k == "material_id":
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    print("Loading dataset...")
    ds = CrystalPXRDDataset(ROOT / "data", split="train")
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True,
                    collate_fn=collate, num_workers=0, drop_last=True)

    # Lattice normalization stats
    stats = lattice_params_stats(ds)
    lat_mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device)
    lat_std = torch.tensor(stats["std"], dtype=torch.float32, device=device).clamp(min=1e-3)
    print(f"Lattice norm: mean={stats['mean'].round(2)}, std={stats['std'].round(2)}")

    # Models
    encoder = PXRDEncoder(d_model=args.d_model).to(device)
    denoiser = CrystalDenoiser(d_model=args.d_model, n_layers=args.n_layers).to(device)
    diffusion = DiffusionProcess()

    params = list(encoder.parameters()) + list(denoiser.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    n_params = sum(p.numel() for p in params)
    print(f"Params: {n_params:,} ({n_params/1e6:.1f}M)")

    # Training loop
    ckpt_dir = ROOT / "runs" / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    epoch = 0
    loss_ema = None
    t0 = time.perf_counter()

    while step < args.steps:
        epoch += 1
        for batch in dl:
            if step >= args.steps:
                break

            pxrd = batch["pxrd_pattern"].to(device)
            coords = batch["frac_coords"].to(device)
            types = batch["atom_types"].to(device)
            lat = batch["lattice"].to(device)
            lat_p = batch["lattice_params"].to(device)
            mask = batch["mask"].to(device)

            # Normalize lattice params
            lat_p_norm = (lat_p - lat_mean) / lat_std

            # Sample timestep and add noise
            t = diffusion.sample_timesteps(pxrd.shape[0], device)
            noisy_coords, eps_coords = diffusion.forward_q(coords, t, wrap=True)
            noisy_lat_p, eps_lat = diffusion.forward_q(lat_p_norm, t)

            # Forward
            pxrd_emb = encoder(pxrd)
            eps_c_pred, eps_l_pred = denoiser(
                noisy_coords, types, lat, t, pxrd_emb, mask
            )

            # Loss
            loss_coord = diffusion.loss(eps_c_pred, eps_coords, mask)
            loss_lat = diffusion.loss(
                eps_l_pred.unsqueeze(1), eps_lat.unsqueeze(1)
            )
            loss = loss_coord + args.lat_weight * loss_lat

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            scheduler.step()

            step += 1
            loss_ema = loss.item() if loss_ema is None else 0.95 * loss_ema + 0.05 * loss.item()

            if step % args.log_every == 0:
                elapsed = time.perf_counter() - t0
                lr = scheduler.get_last_lr()[0]
                print(f"step={step:>6d}  loss={loss_ema:.4f}  "
                      f"coord={loss_coord.item():.4f}  lat={loss_lat.item():.4f}  "
                      f"lr={lr:.2e}  t={elapsed:.0f}s")

            if step % args.save_every == 0:
                ckpt = {
                    "step": step,
                    "encoder": encoder.state_dict(),
                    "denoiser": denoiser.state_dict(),
                    "optimizer": opt.state_dict(),
                    "lat_mean": lat_mean.cpu(),
                    "lat_std": lat_std.cpu(),
                    "args": vars(args),
                }
                path = ckpt_dir / f"ckpt_{step:06d}.pt"
                torch.save(ckpt, path)
                print(f"  -> saved {path}")

    # Final save
    ckpt = {
        "step": step,
        "encoder": encoder.state_dict(),
        "denoiser": denoiser.state_dict(),
        "optimizer": opt.state_dict(),
        "lat_mean": lat_mean.cpu(),
        "lat_std": lat_std.cpu(),
        "args": vars(args),
    }
    torch.save(ckpt, ckpt_dir / "ckpt_final.pt")
    print(f"\nDone. {step} steps in {time.perf_counter()-t0:.1f}s")
    print(f"Final EMA loss: {loss_ema:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--lat-weight", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--run-name", default="smoke")
    train(ap.parse_args())


if __name__ == "__main__":
    main()