"""Train the periodic PXRD-conditioned model (model_family "csp", JAC R2 redesign).

Coordinates: wrapped-normal VE score matching (pxrd_diff.torus). Cell: VP DDPM
on the normalised lattice parameters, noise prediction. For a fraction
--p-cell-given of each batch the clean cell is supplied instead (cell_given = 1,
no lattice loss), so one network serves the learned, oracle and indexer modes.
For --p-pattern-drop of each batch the pattern is replaced by the learned null
pattern. AuxLatHead regresses the cell from the pattern embedding as before.

The validation pass logs the coordinate loss at fixed noise levels on a fixed
batch with fixed noise (zero prediction = 1.0); it is the training-side gate.

Usage:
  python scripts/14_train_csp.py --run-name gpu_v30_csp --steps 100000 --bs 256
  python scripts/14_train_csp.py --run-name cpu_g1 --subset 256 --steps 3000 --bs 32 \
      --d-model 128 --n-layers 4 --val-split train      # G1 overfit check
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pxrd_diff.data import CrystalPXRDDataset, lattice_params_stats  # noqa: E402
from pxrd_diff.diffusion import DiffusionProcess, cosine_alpha_bar   # noqa: E402
from pxrd_diff.model.aux_head import AuxLatHead                      # noqa: E402
from pxrd_diff.model.csp_denoiser import CSPDenoiser                 # noqa: E402
from pxrd_diff.model.pxrd_encoder import PXRDEncoder                 # noqa: E402
from pxrd_diff.torus import TorusVE, wrapped_normal_score            # noqa: E402

VAL_SIGMAS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)


def collate(batch: list[dict]) -> dict:
    return {k: ([b[k] for b in batch] if k == "material_id" else torch.stack([b[k] for b in batch]))
            for k in batch[0]}


def coord_loss(pred, target, mask):
    m = mask.unsqueeze(-1).to(pred.dtype)
    return (((pred - target) ** 2) * m).sum() / (m.sum() * 3).clamp(min=1)


@torch.no_grad()
def validate(nets, torus, val, lat_mean, lat_std, device):
    """Coordinate loss per fixed sigma, learned and given-cell modes, fixed noise."""
    encoder, denoiser = nets
    encoder.eval(); denoiser.eval()
    gen = torch.Generator(device="cpu").manual_seed(1234)
    pxrd, f0, types, mask, lp = (val[k].to(device) for k in
                                 ("pxrd_pattern", "frac_coords", "atom_types", "mask", "lattice_params"))
    B = f0.shape[0]
    g, feats = encoder(pxrd)
    l0 = (lp - lat_mean) / lat_std
    out = {}
    for sig in VAL_SIGMAS:
        t_val = math.log(sig / torus.sigma_min) / math.log(torus.sigma_max / torus.sigma_min)
        t = torch.full((B,), t_val, device=device)
        s = torus.sigma(t).view(-1, 1, 1)
        eps = torch.randn(f0.shape, generator=gen).to(device)
        f_t = (f0 + s * eps) % 1.0
        target = wrapped_normal_score(f_t - f0, s) / torus.score_rms(torus.sigma(t)).view(-1, 1, 1)
        ab = cosine_alpha_bar(torch.tensor(t_val)).item()
        l_t = ab ** 0.5 * l0 + (1 - ab) ** 0.5 * torch.randn(l0.shape, generator=gen).to(device)
        for mode, given in (("learned", False), ("given", True)):
            l_in = l0 if given else l_t
            gv = torch.full((B,), given, dtype=torch.bool, device=device)
            pred, _ = denoiser(f_t, types, l_in * lat_std + lat_mean, l_in, t, g, feats, mask, gv)
            out[f"{mode}@{sig}"] = round(coord_loss(pred, target, mask).item(), 4)
    encoder.train(); denoiser.train()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-freq", type=int, default=10)
    ap.add_argument("--sigma-min", type=float, default=0.005)
    ap.add_argument("--sigma-max", type=float, default=0.5)
    ap.add_argument("--p-cell-given", type=float, default=0.5)
    ap.add_argument("--p-pattern-drop", type=float, default=0.1)
    ap.add_argument("--lat-weight", type=float, default=1.0)
    ap.add_argument("--aux-weight", type=float, default=1.0)
    ap.add_argument("--subset", type=int, default=0, help="train on the first N structures only")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--n-val", type=int, default=512)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--amp", choices=["bf16", "none"], default="bf16")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="checkpoint to continue from (same args)")
    ap.add_argument("--stop-at", type=int, default=0,
                    help="stop (and save) at this step, keeping the --steps LR schedule")
    ap.add_argument("--max-minutes", type=float, default=0,
                    help="save a checkpoint and exit after this much wall time")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    use_amp = args.amp == "bf16" and device.type == "cuda"
    print(f"Device: {device}, amp: {use_amp}, args: {vars(args)}", flush=True)

    ds = CrystalPXRDDataset(ROOT / "data", split="train")
    stats = lattice_params_stats(ds)
    lat_mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device)
    lat_std = torch.tensor(stats["std"], dtype=torch.float32, device=device).clamp(min=1e-3)
    if args.subset:
        ds = Subset(ds, range(args.subset))
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, collate_fn=collate, drop_last=True,
                    num_workers=args.workers if device.type == "cuda" else 0,
                    persistent_workers=device.type == "cuda" and args.workers > 0)
    vds = ds if args.val_split == "train" else CrystalPXRDDataset(ROOT / "data", split=args.val_split)
    val = collate([vds[i] for i in range(min(args.n_val, len(vds)))])

    encoder = PXRDEncoder(d_model=args.d_model).to(device)
    denoiser = CSPDenoiser(d_model=args.d_model, n_layers=args.n_layers,
                           n_heads=args.n_heads, n_freq=args.n_freq).to(device)
    aux_head = AuxLatHead(args.d_model).to(device)
    torus = TorusVE(args.sigma_min, args.sigma_max).to(device)
    lat_diff = DiffusionProcess()
    params = list(encoder.parameters()) + list(denoiser.parameters()) + list(aux_head.parameters())
    print(f"Params: {sum(p.numel() for p in params) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup)
                                              * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))

    ckpt_dir = ROOT / "runs" / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    val_log = open(ckpt_dir / "val.jsonl", "a")

    def save(step: int, name: str) -> None:
        torch.save({"step": step, "model_family": "csp", "encoder": encoder.state_dict(),
                    "denoiser": denoiser.state_dict(), "aux_head": aux_head.state_dict(),
                    "optimizer": opt.state_dict(), "lat_mean": lat_mean.cpu(), "lat_std": lat_std.cpu(),
                    "args": {**vars(args), "model_family": "csp"}}, ckpt_dir / name)

    step, t0, ema = 0, time.perf_counter(), {}
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        encoder.load_state_dict(ck["encoder"])
        denoiser.load_state_dict(ck["denoiser"])
        aux_head.load_state_dict(ck["aux_head"])
        opt.load_state_dict(ck["optimizer"])
        step = ck["step"]
        for _ in range(step):
            sched.step()
        print(f"Resumed from {args.resume} at step {step}", flush=True)
    while step < args.steps:
        for batch in dl:
            if step >= args.steps:
                break
            pxrd = batch["pxrd_pattern"].to(device)
            f0 = batch["frac_coords"].to(device)
            types = batch["atom_types"].to(device)
            mask = batch["mask"].to(device)
            l0 = (batch["lattice_params"].to(device) - lat_mean) / lat_std
            B = f0.shape[0]

            t = torch.rand(B, device=device)
            f_t, target, _ = torus.forward_q(f0, t)
            l_t, eps_l = lat_diff.forward_q(l0, t)
            given = torch.rand(B, device=device) < args.p_cell_given
            l_in = torch.where(given.unsqueeze(-1), l0, l_t)
            drop = torch.rand(B, device=device) < args.p_pattern_drop

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                g, feats = encoder(pxrd)
                pred, eps_l_pred = denoiser(f_t, types, l_in * lat_std + lat_mean, l_in, t,
                                            g, feats, mask, given, drop)
                lat_aux = aux_head(g)
            pred, eps_l_pred, lat_aux = pred.float(), eps_l_pred.float(), lat_aux.float()
            loss_c = coord_loss(pred, target, mask)
            learned = (~given).float()
            loss_l = (((eps_l_pred - eps_l) ** 2).mean(-1) * learned).sum() / learned.sum().clamp(min=1)
            loss_a = ((lat_aux - l0) ** 2).mean()
            loss = loss_c + args.lat_weight * loss_l + args.aux_weight * loss_a

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1

            for k, v in (("coord", loss_c), ("lat", loss_l), ("aux", loss_a), ("gnorm", gnorm)):
                ema[k] = 0.98 * ema.get(k, v.item()) + 0.02 * v.item()
            if step % args.log_every == 0 or step == 1:
                print(f"step {step:6d}  coord {ema['coord']:.4f}  lat {ema['lat']:.4f}  "
                      f"aux {ema['aux']:.4f}  gnorm {ema['gnorm']:.2f}  lr {sched.get_last_lr()[0]:.2e}  "
                      f"t={time.perf_counter() - t0:.0f}s", flush=True)
            if step % args.val_every == 0:
                v = validate((encoder, denoiser), torus, val, lat_mean, lat_std, device)
                val_log.write(json.dumps({"step": step, **v}) + "\n"); val_log.flush()
                print(f"  val step {step}: {v}", flush=True)
            if step % args.save_every == 0:
                save(step, f"ckpt_{step:06d}.pt")
            out_of_time = args.max_minutes and time.perf_counter() - t0 > 60 * args.max_minutes
            if (args.stop_at and step >= args.stop_at) or out_of_time:
                if step % args.save_every:
                    save(step, f"ckpt_{step:06d}.pt")
                print(f"Stopped at step {step} ({'time limit' if out_of_time else '--stop-at'})", flush=True)
                return
    v = validate((encoder, denoiser), torus, val, lat_mean, lat_std, device)
    val_log.write(json.dumps({"step": step, "final": True, **v}) + "\n"); val_log.close()
    print(f"  val final: {v}", flush=True)
    save(step, "ckpt_final.pt")
    print(f"Done: {step} steps in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
