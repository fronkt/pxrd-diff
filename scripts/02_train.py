"""Training script for PXRD-Diff with multi-resolution cross-attention.

Usage:
  python scripts/02_train.py --steps 100000 --bs 64 --run-name gpu_v5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pxrd_diff.data import CrystalPXRDDataset, lattice_params_stats
from pxrd_diff.debye import DiffPXRD, diff_pxrd_loss
from pxrd_diff.diffusion import DiffusionProcess, cosine_alpha_bar
from pxrd_diff.model.denoiser import CrystalDenoiser, periodic_distances
from pxrd_diff.model.pxrd_encoder import PXRDEncoder


class AuxLatHead(nn.Module):
    """Predict normalized lattice params from PXRD embedding (auxiliary task)."""
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 6),
        )
    def forward(self, emb):
        return self.net(emb)


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

    print("Loading dataset...")
    ds = CrystalPXRDDataset(ROOT / "data", split="train")
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True,
                    collate_fn=collate, num_workers=0, drop_last=True)

    stats = lattice_params_stats(ds)
    lat_mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device)
    lat_std = torch.tensor(stats["std"], dtype=torch.float32, device=device).clamp(min=1e-3)
    print(f"Lattice norm: mean={stats['mean'].round(2)}, std={stats['std'].round(2)}")

    encoder = PXRDEncoder(d_model=args.d_model).to(device)
    denoiser = CrystalDenoiser(d_model=args.d_model, n_layers=args.n_layers,
                                n_heads=args.n_heads).to(device)
    aux_head = AuxLatHead(args.d_model).to(device)
    diffusion = DiffusionProcess()

    diff_pxrd = None
    if args.debye_weight > 0:
        diff_pxrd = DiffPXRD(n_bins=256, hkl_max=5).to(device)
        diff_pxrd.eval()

    params = list(encoder.parameters()) + list(denoiser.parameters()) + list(aux_head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    n_params = sum(p.numel() for p in params)
    print(f"Params: {n_params:,} ({n_params/1e6:.1f}M)")

    ckpt_dir = ROOT / "runs" / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    epoch = 0
    loss_ema = None
    t0 = time.perf_counter()

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = ROOT / resume_path
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        denoiser.load_state_dict(ckpt["denoiser"])
        aux_head.load_state_dict(ckpt["aux_head"])
        opt.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        for _ in range(step):
            scheduler.step()
        print(f"  -> resumed at step {step}")

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

            lat_p_norm = (lat_p - lat_mean) / lat_std

            t = diffusion.sample_timesteps(pxrd.shape[0], device)
            noisy_coords, eps_coords = diffusion.forward_q(coords, t, wrap=True)
            noisy_lat_p, eps_lat = diffusion.forward_q(lat_p_norm, t)

            pxrd_global, pxrd_feats = encoder(pxrd)
            use_dist = args.dist_weight > 0
            denoiser_out = denoiser(
                noisy_coords, types, lat, t,
                pxrd_global, pxrd_feats, mask, noisy_lat_p,
                return_dist=use_dist,
            )
            if use_dist:
                pred_c, pred_l, d_pred = denoiser_out
            else:
                pred_c, pred_l = denoiser_out

            if args.predict_x0:
                # Residual: pred_c is correction from noisy_coords toward x0.
                # MLP head outputs ~0 at init, so x0_pred starts as noisy_coords.
                x0_pred_c = noisy_coords + pred_c
                x0_pred_l = noisy_lat_p + pred_l
                loss_coord = diffusion.loss(x0_pred_c, coords, mask, periodic=True)
                loss_lat = diffusion.loss(
                    x0_pred_l.unsqueeze(1), lat_p_norm.unsqueeze(1)
                )
                eps_c_pred = x0_pred_c  # for downstream Debye loss
            else:
                loss_coord = diffusion.loss(pred_c, eps_coords, mask)
                loss_lat = diffusion.loss(
                    pred_l.unsqueeze(1), eps_lat.unsqueeze(1)
                )
                eps_c_pred = pred_c

            lat_pred = aux_head(pxrd_global)
            loss_aux = ((lat_pred - lat_p_norm) ** 2).mean()

            loss = loss_coord + args.lat_weight * loss_lat + args.aux_weight * loss_aux

            loss_dist = torch.tensor(0.0, device=device)
            if use_dist:
                # True periodic Cartesian distances from clean coords + true lattice
                d_true = periodic_distances(coords, lat, mask)
                pair_mask = (mask.unsqueeze(2) & mask.unsqueeze(1)).float()
                # Mask diagonal (self-distances)
                eye = torch.eye(d_true.shape[-1], device=device).unsqueeze(0)
                pair_mask = pair_mask * (1 - eye)
                # Clamp distances to RBF cutoff for stable training
                d_true_clamped = d_true.clamp(max=12.0)
                sq = (d_pred - d_true_clamped) ** 2
                loss_dist = (sq * pair_mask).sum() / pair_mask.sum().clamp(min=1)
                loss = loss + args.dist_weight * loss_dist

            loss_debye = torch.tensor(0.0, device=device)
            if diff_pxrd is not None:
                if args.predict_x0:
                    x0_pred = pred_c % 1.0
                else:
                    alpha_bar = cosine_alpha_bar(t)
                    while alpha_bar.dim() < noisy_coords.dim():
                        alpha_bar = alpha_bar.unsqueeze(-1)
                    x0_pred = ((noisy_coords - (1 - alpha_bar).sqrt() * eps_c_pred)
                               / alpha_bar.sqrt().clamp(min=1e-6)) % 1.0
                pred_pxrd = diff_pxrd(x0_pred, types, lat, mask)
                loss_debye = diff_pxrd_loss(pred_pxrd, pxrd, n_bins_diff=256)
                loss = loss + args.debye_weight * loss_debye

            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            scheduler.step()

            step += 1
            loss_ema = loss.item() if loss_ema is None else 0.95 * loss_ema + 0.05 * loss.item()

            if step % args.log_every == 0:
                elapsed = time.perf_counter() - t0
                lr = scheduler.get_last_lr()[0]
                emb_std = pxrd_global.std().item()
                emb_abs = pxrd_global.abs().mean().item()
                enc_gnorm = sum(p.grad.norm().item()**2 for p in encoder.parameters() if p.grad is not None)**0.5
                debye_str = f"  debye={loss_debye.item():.4f}" if diff_pxrd is not None else ""
                dist_str = f"  dist={loss_dist.item():.4f}" if use_dist else ""
                print(f"step={step:>6d}  loss={loss_ema:.4f}  "
                      f"coord={loss_coord.item():.4f}  lat={loss_lat.item():.4f}  "
                      f"aux={loss_aux.item():.4f}{debye_str}{dist_str}  "
                      f"lr={lr:.2e}  gnorm={grad_norm:.2f}  enc_gn={enc_gnorm:.3f}  "
                      f"emb_std={emb_std:.3f}  emb_abs={emb_abs:.3f}  "
                      f"t={elapsed:.0f}s")

            if step % args.save_every == 0:
                ckpt = {
                    "step": step,
                    "encoder": encoder.state_dict(),
                    "denoiser": denoiser.state_dict(),
                    "aux_head": aux_head.state_dict(),
                    "optimizer": opt.state_dict(),
                    "lat_mean": lat_mean.cpu(),
                    "lat_std": lat_std.cpu(),
                    "args": vars(args),
                }
                path = ckpt_dir / f"ckpt_{step:06d}.pt"
                torch.save(ckpt, path)
                print(f"  -> saved {path}")

    ckpt = {
        "step": step,
        "encoder": encoder.state_dict(),
        "denoiser": denoiser.state_dict(),
        "aux_head": aux_head.state_dict(),
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
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--lat-weight", type=float, default=0.1)
    ap.add_argument("--aux-weight", type=float, default=0.5)
    ap.add_argument("--debye-weight", type=float, default=0.0)
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to ckpt to resume from")
    ap.add_argument("--predict-x0", action="store_true",
                    help="Predict x0 directly instead of eps")
    ap.add_argument("--dist-weight", type=float, default=0.0,
                    help="Weight for pairwise distance auxiliary loss")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--run-name", default="smoke")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
