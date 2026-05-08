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
from pxrd_diff.model.aux_head import AuxLatHead
from pxrd_diff.model.denoiser import CrystalDenoiser, periodic_distances
from pxrd_diff.model.lat_head import (
    ConstrainedLatHead,
    PeakAugmentedLatHead,
    SpaceGroupHead,
    sg_classification_loss,
    sg_topk_accuracy,
)
from pxrd_diff.model.pxrd_encoder import PXRDEncoder


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
    ds = CrystalPXRDDataset(
        ROOT / "data", split="train",
        n_peaks=args.n_peaks if args.peak_aug_lat_head else 0,
        augment=args.noise_aug,
    )
    if args.noise_aug:
        print("Phase 7: noise augmentation enabled "
              "(zero-offset + Lorentzian broadening + Gaussian noise, p=0.8)")
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True,
                    collate_fn=collate, num_workers=0, drop_last=True)

    stats = lattice_params_stats(ds)
    lat_mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device)
    lat_std = torch.tensor(stats["std"], dtype=torch.float32, device=device).clamp(min=1e-3)
    print(f"Lattice norm: mean={stats['mean'].round(2)}, std={stats['std'].round(2)}")

    encoder = PXRDEncoder(d_model=args.d_model).to(device)
    denoiser = CrystalDenoiser(d_model=args.d_model, n_layers=args.n_layers,
                                n_heads=args.n_heads).to(device)
    if args.peak_aug_lat_head:
        aux_head = PeakAugmentedLatHead(
            d_model=args.d_model, peak_dim=2 * args.n_peaks,
            use_d_spacing=args.use_d_spacing,
        ).to(device)
        mode = "Phase 5D, d-spacing" if args.use_d_spacing else "Phase 5C, raw 2θ"
        print(f"Using PeakAugmentedLatHead ({mode}, n_peaks={args.n_peaks})")
    elif args.constrained_lat_head:
        aux_head = ConstrainedLatHead(args.d_model).to(device)
        print("Using ConstrainedLatHead (Phase 5B)")
    else:
        aux_head = AuxLatHead(args.d_model).to(device)
    sg_head = None
    if args.sg_weight > 0:
        sg_head = SpaceGroupHead(args.d_model).to(device)
        print(f"Using SpaceGroupHead (Phase 6.1) sg_weight={args.sg_weight}")
    diffusion = DiffusionProcess()

    diff_pxrd = None
    if args.debye_weight > 0:
        diff_pxrd = DiffPXRD(n_bins=256, hkl_max=5).to(device)
        diff_pxrd.eval()

    params = (list(encoder.parameters()) + list(denoiser.parameters())
              + list(aux_head.parameters()))
    if sg_head is not None:
        params = params + list(sg_head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    if args.const_lr:
        # Fine-tune mode: no LR annealing, lr stays at args.lr for the
        # whole run. Avoids the case where a cosine endpoint at the resume
        # ckpt's step makes the resumed lr ~0 even after fast-forwarding.
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda _: 1.0)
        print(f"Constant LR mode: lr={args.lr:.2e} for the whole run")
    else:
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
        denoiser.load_state_dict(ckpt["denoiser"], strict=False)
        # Match the checkpoint's lat-head architecture to the configured one.
        ckpt_args = ckpt.get("args", {})
        ckpt_constrained = bool(ckpt_args.get("constrained_lat_head", False))
        ckpt_peak_aug = bool(ckpt_args.get("peak_aug_lat_head", False))
        same_head = (ckpt_peak_aug == args.peak_aug_lat_head
                     and ckpt_constrained == args.constrained_lat_head)
        if same_head and "aux_head" in ckpt:
            aux_head.load_state_dict(ckpt["aux_head"])
            print("  -> resumed lat head from checkpoint")
        else:
            print("  -> NOT resuming lat head (architecture mismatch — fresh init)")
        if sg_head is not None and "sg_head" in ckpt:
            sg_head.load_state_dict(ckpt["sg_head"])
            print("  -> resumed sg_head from checkpoint")
        elif sg_head is not None:
            print("  -> NOT resuming sg_head (new in this run — fresh init)")
        # Optimizer state is keyed by parameter id, so it only matches if every
        # head had its weights loaded. Skip when any new head was just init'd.
        load_opt = (same_head
                    and (sg_head is None or "sg_head" in ckpt))
        if load_opt and "optimizer" in ckpt:
            try:
                opt.load_state_dict(ckpt["optimizer"])
                print("  -> resumed optimizer state")
            except (ValueError, KeyError) as e:
                print(f"  -> NOT resuming optimizer ({e!s})")
        else:
            print("  -> NOT resuming optimizer (new heads added — starting fresh)")
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
            wyckoff = batch["wyckoff"].to(device) if args.use_wyckoff else None
            peak_features = (batch["peak_features"].to(device)
                             if args.peak_aug_lat_head else None)

            lat_p_norm = (lat_p - lat_mean) / lat_std

            t = diffusion.sample_timesteps(pxrd.shape[0], device)
            noisy_coords, eps_coords = diffusion.forward_q(coords, t, wrap=True)
            noisy_lat_p, eps_lat = diffusion.forward_q(lat_p_norm, t)

            pxrd_global, pxrd_feats = encoder(pxrd)
            use_dist = args.dist_weight > 0
            denoiser_out = denoiser(
                noisy_coords, types, lat, t,
                pxrd_global, pxrd_feats, mask, noisy_lat_p,
                wyckoff=wyckoff,
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

            if args.peak_aug_lat_head:
                lat_pred_raw = aux_head(pxrd_global, peak_features)
            else:
                lat_pred_raw = aux_head(pxrd_global)
            if args.constrained_lat_head or args.peak_aug_lat_head:
                # Heads with sigmoid-bounded outputs emit physical units;
                # normalize for consistent gradient magnitudes vs lat_p_norm.
                lat_pred_norm = (lat_pred_raw - lat_mean) / lat_std
            else:
                lat_pred_norm = lat_pred_raw
            loss_aux = ((lat_pred_norm - lat_p_norm) ** 2).mean()

            loss_sg = torch.tensor(0.0, device=device)
            sg_top1 = torch.tensor(0.0, device=device)
            if sg_head is not None:
                sg_logits = sg_head(pxrd_global)
                sg_target = batch["spacegroup"].to(device)
                loss_sg = sg_classification_loss(sg_logits, sg_target)
                with torch.no_grad():
                    sg_top1 = sg_topk_accuracy(sg_logits, sg_target, k=1)

            loss = (loss_coord
                    + args.lat_weight * loss_lat
                    + args.aux_weight * loss_aux
                    + args.sg_weight * loss_sg)

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
                sg_str = (f"  sg={loss_sg.item():.4f}/top1={sg_top1.item():.3f}"
                          if sg_head is not None else "")
                print(f"step={step:>6d}  loss={loss_ema:.4f}  "
                      f"coord={loss_coord.item():.4f}  lat={loss_lat.item():.4f}  "
                      f"aux={loss_aux.item():.4f}{sg_str}{debye_str}{dist_str}  "
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
                if sg_head is not None:
                    ckpt["sg_head"] = sg_head.state_dict()
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
    if sg_head is not None:
        ckpt["sg_head"] = sg_head.state_dict()
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
    ap.add_argument("--use-wyckoff", action="store_true",
                    help="Use Wyckoff position tokens as additional atom feature")
    # ---- Phase 5B + 6 -------------------------------------------------------
    ap.add_argument("--constrained-lat-head", action="store_true",
                    help="Phase 5B: replace AuxLatHead with ConstrainedLatHead "
                         "(physical units, sigmoid-bounded). Recommended with "
                         "a higher --aux-weight than the v13 default of 0.5.")
    ap.add_argument("--sg-weight", type=float, default=0.0,
                    help="Phase 6.1: weight for the SpaceGroupHead "
                         "cross-entropy loss. 0 disables the head.")
    # ---- Phase 5C -----------------------------------------------------------
    ap.add_argument("--peak-aug-lat-head", action="store_true",
                    help="Phase 5C: replace the lattice head with "
                         "PeakAugmentedLatHead, which receives explicit "
                         "top-N peak position+intensity features alongside "
                         "the encoder embedding. Overrides "
                         "--constrained-lat-head if both are set.")
    ap.add_argument("--n-peaks", type=int, default=20,
                    help="Number of top peaks to extract per pattern for "
                         "Phase 5C. Must match the dataset's n_peaks.")
    ap.add_argument("--use-d-spacing", action="store_true",
                    help="Phase 5D: convert each peak's normalized 2θ position "
                         "to log d-spacing via Bragg's law before feeding it "
                         "to the PeakAugmentedLatHead. d-spacing is the "
                         "physics quantity that determines the lattice; the "
                         "MLP no longer has to learn the trigonometric inverse "
                         "implicitly.")
    # ---- Phase 7 ------------------------------------------------------------
    ap.add_argument("--noise-aug", action="store_true",
                    help="Phase 7: apply experimental-style augmentation to "
                         "PXRD patterns during training "
                         "(zero-offset + Lorentzian broadening + Gaussian "
                         "noise, p=0.8 per call). Makes the model robust to "
                         "real-data artifacts not present in simulated patterns.")
    ap.add_argument("--const-lr", action="store_true",
                    help="Disable the cosine LR schedule and use a constant "
                         "lr=args.lr for the whole run. Recommended for "
                         "fine-tune resumes where the resume step is near "
                         "the cosine endpoint of the previous run.")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--run-name", default="smoke")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
