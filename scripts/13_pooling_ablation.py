"""Phase 15 (JAC HAT5032 revision), referee 2 point 2: controlled pooling ablation.

Question: does the encoder's multi-resolution feature map still contain the
absolute d-spacing information that the globally pooled vector g has lost?

Design: the trained PXRD encoder is FROZEN (so the representation is held
fixed) and three lattice heads are trained on it under identical data,
optimiser, loss and output parameterisation. The ONLY difference between arms
is what part of the encoder output the head may read:

    gpool     ConstrainedLatHead(g)                    global average pool (as in the paper)
    attnpool  AttnPoolLatHead(multi_res)               position-aware attention pooling
    peaks     PeakAugmentedLatHead(g, peak_features)   explicit peak positions (d-spacing)

Each arm reports test-set lattice MAE (Å, °) and writes its predicted cells in
the `rows: [{mid, pred_params}]` schema so `03_sample.py --lat-from-index`
can measure the end-to-end match rate with that head's lattice on a GPU box.
The checkpoint's own AuxLatHead is evaluated as the reference row.

Usage (CPU is fine for the head training; ~minutes per epoch):
  python scripts/13_pooling_ablation.py --ckpt E:/gpu_v21_p9/ckpt_final.pt \
      --epochs 3 --bs 64 --n-test 1000 --out paper/phase15_results/pooling_ablation
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pxrd_diff.data import CrystalPXRDDataset                           # noqa: E402
from pxrd_diff.model.attnpool_head import AttnPoolLatHead              # noqa: E402
from pxrd_diff.model.aux_head import AuxLatHead                        # noqa: E402
from pxrd_diff.model.lat_head import ConstrainedLatHead, PeakAugmentedLatHead  # noqa: E402
from pxrd_diff.model.pxrd_encoder import PXRDEncoder                   # noqa: E402

ARMS = ("gpool", "attnpool", "peaks")


def collate(batch: list[dict]) -> dict:
    """Same collate as scripts/02_train.py (material_id stays a list)."""
    out = {}
    for k in batch[0]:
        if k == "material_id":
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


def build_head(arm: str, d_model: int, n_peaks: int) -> nn.Module:
    if arm == "gpool":
        return ConstrainedLatHead(d_model)
    if arm == "attnpool":
        return AttnPoolLatHead(d_model)
    if arm == "peaks":
        return PeakAugmentedLatHead(d_model, peak_dim=2 * n_peaks, use_d_spacing=True)
    raise ValueError(arm)


def head_forward(arm: str, head: nn.Module, g, feats, level_lengths, peaks):
    if arm == "gpool":
        return head(g)
    if arm == "attnpool":
        return head(feats, level_lengths)
    if arm == "peaks":
        return head(g, peaks)
    if arm == "ckpt_aux":
        return head(g)
    raise ValueError(arm)


@torch.no_grad()
def evaluate(arm, head, encoder, dl, device, lat_mean, lat_std, physical_output):
    head.eval()
    abs_err, rows = [], []
    for batch in dl:
        pxrd = batch["pxrd_pattern"].to(device)
        lat_p = batch["lattice_params"].to(device)
        peaks = batch["peak_features"].to(device) if "peak_features" in batch else None
        g, feats = encoder(pxrd)
        pred = head_forward(arm, head, g, feats, encoder.level_lengths, peaks)
        if not physical_output:
            pred = pred * lat_std + lat_mean
        abs_err.append((pred - lat_p).abs().cpu())
        for mid, p in zip(batch["material_id"], pred.cpu().tolist()):
            rows.append({"mid": str(mid), "pred_params": [round(x, 4) for x in p]})
    e = torch.cat(abs_err)                       # (N, 6)
    mae = e.mean(0).tolist()
    return {
        "n": int(e.shape[0]),
        "mae_abc_A": [round(x, 4) for x in mae[:3]],
        "mae_len_A": round(float(np.mean(mae[:3])), 4),
        "mae_angles_deg": [round(x, 3) for x in mae[3:]],
        "mae_ang_deg": round(float(np.mean(mae[3:])), 3),
        "median_len_err_A": round(float(e[:, :3].mean(1).median()), 4),
        "frac_len_err_below_0p5A": round(float((e[:, :3].mean(1) < 0.5).float().mean()), 4),
    }, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-train", type=int, default=None, help="subsample train (smoke tests)")
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--n-peaks", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "paper" / "phase15_results" / "pooling_ablation"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    targs = ckpt.get("args", {})
    d_model = int(targs.get("d_model", 256))
    lat_mean = torch.as_tensor(ckpt["lat_mean"], dtype=torch.float32, device=device)
    lat_std = torch.as_tensor(ckpt["lat_std"], dtype=torch.float32, device=device).clamp(min=1e-3)

    encoder = PXRDEncoder(d_model=d_model).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    print(f"Frozen encoder from {args.ckpt} (run {targs.get('run_name')}, step {ckpt.get('step')})")

    ds_tr = CrystalPXRDDataset(ROOT / "data", split="train", n_peaks=args.n_peaks,
                               limit=args.n_train, preload=True)
    ds_te = CrystalPXRDDataset(ROOT / "data", split="test", n_peaks=args.n_peaks,
                               limit=args.n_test, preload=True)
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, collate_fn=collate,
                       num_workers=0, drop_last=True)
    dl_te = DataLoader(ds_te, batch_size=args.bs, shuffle=False, collate_fn=collate, num_workers=0)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"ckpt": args.ckpt, "run_name": targs.get("run_name"), "seed": args.seed,
               "epochs": args.epochs, "bs": args.bs, "lr": args.lr,
               "n_train": len(ds_tr), "n_test": len(ds_te), "arms": {}}

    # Reference row: the checkpoint's own (unconstrained, normalised-output) aux head.
    if "aux_head" in ckpt and not targs.get("peak_aug_lat_head") and not targs.get("constrained_lat_head"):
        ref = AuxLatHead(d_model).to(device)
        ref.load_state_dict(ckpt["aux_head"])
        m, rows = evaluate("ckpt_aux", ref, encoder, dl_te, device, lat_mean, lat_std,
                           physical_output=False)
        results["arms"]["ckpt_aux"] = {"test": m, "note": "checkpoint AuxLatHead, no retraining"}
        (out_dir / "ckpt_aux_cells.json").write_text(json.dumps({"rows": rows}))
        print(f"[ckpt_aux] test MAE len {m['mae_len_A']} A, ang {m['mae_ang_deg']} deg")

    for arm in args.arms:
        torch.manual_seed(args.seed)
        head = build_head(arm, d_model, args.n_peaks).to(device)
        n_par = sum(p.numel() for p in head.parameters())
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
        steps = args.epochs * len(dl_tr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
        print(f"\n=== arm {arm}: {n_par:,} head params, {steps} steps ===")
        hist = []
        t0 = time.time()
        step = 0
        for ep in range(args.epochs):
            head.train()
            run = 0.0
            for batch in dl_tr:
                pxrd = batch["pxrd_pattern"].to(device)
                lat_p = batch["lattice_params"].to(device)
                peaks = batch["peak_features"].to(device) if "peak_features" in batch else None
                with torch.no_grad():
                    g, feats = encoder(pxrd)
                pred = head_forward(arm, head, g, feats, encoder.level_lengths, peaks)
                # all three arms emit physical units; loss on the normalised scale,
                # exactly as 02_train.py does for the bounded heads
                loss = (((pred - lat_mean) / lat_std - (lat_p - lat_mean) / lat_std) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                opt.step()
                sched.step()
                run += loss.item()
                step += 1
                if step % 50 == 0:
                    print(f"  ep {ep} step {step}/{steps} loss {run / 50:.4f} "
                          f"({time.time() - t0:.0f}s)", flush=True)
                    run = 0.0
            m, rows = evaluate(arm, head, encoder, dl_te, device, lat_mean, lat_std,
                               physical_output=True)
            hist.append({"epoch": ep, **m})
            print(f"  [{arm}] epoch {ep}: test MAE len {m['mae_len_A']} A "
                  f"(abc {m['mae_abc_A']}), ang {m['mae_ang_deg']} deg, "
                  f"frac<0.5A {m['frac_len_err_below_0p5A']}", flush=True)
        results["arms"][arm] = {"n_params": n_par, "test": hist[-1], "history": hist,
                                "train_seconds": round(time.time() - t0)}
        (out_dir / f"{arm}_cells.json").write_text(json.dumps({"rows": rows}))
        torch.save(head.state_dict(), out_dir / f"{arm}_head.pt")
        (out_dir / "pooling_ablation.json").write_text(json.dumps(results, indent=1))

    print("\n=== summary (test, final epoch) ===")
    for arm, r in results["arms"].items():
        t = r["test"]
        print(f"{arm:10s} len MAE {t['mae_len_A']:.3f} A  ang MAE {t['mae_ang_deg']:.2f} deg  "
              f"frac<0.5A {t['frac_len_err_below_0p5A']:.3f}")
    print(f"wrote {out_dir / 'pooling_ablation.json'}")


if __name__ == "__main__":
    main()
