"""Sample crystal structures from a trained PXRD-Diff model and evaluate.

Loads a checkpoint, samples structures for test-set PXRD patterns using DDIM,
reconstructs pymatgen Structures, runs the eval harness, and prints metrics.

Phase 4 inference improvements (no retrain required):
  --n-samples N        Generate N candidates per pattern, pick the one whose
                       DiffPXRD-pearson against the target is highest. N=1
                       reproduces the original single-sample behavior.
  --ensemble-eta E     DDIM stochasticity to use during ensemble sampling.
                       0.0 is fully deterministic given init noise (different
                       noise per candidate already gives diversity).
  --refine-steps K     After picking a candidate, run K Adam steps minimizing
                       1 - Pearson(DiffPXRD(structure), target) to refine
                       fractional coords. K=0 disables.
  --refine-lr LR       Adam learning rate for refinement (default 1e-3).
  --refine-lattice     Also refine the 3x3 lattice matrix (default: coords only).

Usage:
  python scripts/03_sample.py --ckpt runs/smoke/ckpt_final.pt --n 16
  python scripts/03_sample.py --ckpt runs/gpu_v13/ckpt_final.pt --n 1000 --true-lattice
  python scripts/03_sample.py --ckpt runs/gpu_v13/ckpt_final.pt --n 1000 --true-lattice \
                              --n-samples 20 --refine-steps 200
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
from pxrd_diff.debye import DiffPXRD                                  # noqa: E402
from pxrd_diff.eval import aggregate, evaluate_one                    # noqa: E402
from pxrd_diff.model.aux_head import AuxLatHead                       # noqa: E402
from pxrd_diff.model.denoiser import CrystalDenoiser                 # noqa: E402
from pxrd_diff.model.lat_head import (                                # noqa: E402
    ConstrainedLatHead,
    PeakAugmentedLatHead,
    SpaceGroupHead,
    apply_sg_constraints,
)
from pxrd_diff.model.pxrd_encoder import PXRDEncoder                 # noqa: E402
from pxrd_diff.sampler import (                                       # noqa: E402
    DDIMSampler,
    lattice_params_to_matrix,
    refine_structure,
    select_best_by_pearson,
)
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
    # ---- Phase 4 -------------------------------------------------------------
    ap.add_argument("--n-samples", type=int, default=1,
                    help="Candidates per pattern (Phase 4.1 ensemble; 1 = single)")
    ap.add_argument("--ensemble-eta", type=float, default=0.0,
                    help="DDIM eta for ensemble sampling (0 = deterministic)")
    ap.add_argument("--refine-steps", type=int, default=0,
                    help="Phase 4.2 Rietveld refinement steps (0 = disabled)")
    ap.add_argument("--refine-lr", type=float, default=1e-3,
                    help="Adam LR for refinement")
    ap.add_argument("--refine-lattice", action="store_true",
                    help="Refine lattice matrix as well as coords")
    ap.add_argument("--debye-n-bins", type=int, default=256,
                    help="DiffPXRD n_bins for scoring/refinement (matches training)")
    ap.add_argument("--debye-hkl-max", type=int, default=5,
                    help="DiffPXRD hkl_max for scoring/refinement (matches training)")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Patterns processed per chunk (avoids OOM at large n × n_samples)")
    # ---- Phase 5 -------------------------------------------------------------
    ap.add_argument("--lat-from-aux", action="store_true",
                    help="Phase 5: substitute aux-head lattice prediction for the "
                         "diffusion sampler's lattice. Ignored when --true-lattice "
                         "is also set (true lattice wins). The aux-vs-true MAE "
                         "diagnostic is always printed regardless of this flag.")
    # ---- Phase 6 -------------------------------------------------------------
    ap.add_argument("--sg-constrain-lat", action="store_true",
                    help="Phase 6.2: project the aux/constrained-head lattice "
                         "prediction onto the crystal-system manifold of the "
                         "predicted space group (e.g. cubic ⇒ a=b=c, all "
                         "angles 90°). Requires a checkpoint with sg_head.")
    # ---- Phase 7 -------------------------------------------------------------
    ap.add_argument("--noise-aug-eval", action="store_true",
                    help="Phase 7: apply experimental-style augmentation to "
                         "test patterns (zero-offset + Lorentzian broadening "
                         "+ Gaussian noise, deterministic seed=42). Use this "
                         "to measure robustness of a model trained with or "
                         "without --noise-aug.")
    ap.add_argument("--out-json", default=None,
                    help="If set, write aggregate metrics JSON here")
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

    # Lattice prediction head: pick by what the checkpoint was trained with.
    # Priority order: PeakAugmentedLatHead (Phase 5C) > ConstrainedLatHead
    # (Phase 5B) > AuxLatHead (legacy).
    constrained_lat = bool(train_args.get("constrained_lat_head", False))
    peak_aug_lat = bool(train_args.get("peak_aug_lat_head", False))
    n_peaks_train = int(train_args.get("n_peaks", 20))
    aux_head = None
    if "aux_head" in ckpt:
        if peak_aug_lat:
            aux_head = PeakAugmentedLatHead(
                d_model=d_model, peak_dim=2 * n_peaks_train,
            ).to(device)
        elif constrained_lat:
            aux_head = ConstrainedLatHead(d_model=d_model).to(device)
        else:
            aux_head = AuxLatHead(d_model=d_model).to(device)
        aux_head.load_state_dict(ckpt["aux_head"])
        aux_head.eval()

    sg_head = None
    if "sg_head" in ckpt:
        sg_head = SpaceGroupHead(d_model=d_model).to(device)
        sg_head.load_state_dict(ckpt["sg_head"])
        sg_head.eval()

    lat_mean = ckpt.get("lat_mean")
    lat_std = ckpt.get("lat_std")
    predict_x0 = bool(train_args.get("predict_x0", False))

    head_kind_short = (
        "PeakAugmented" if peak_aug_lat
        else "Constrained" if constrained_lat
        else "Aux" if aux_head is not None
        else "none"
    )
    print(f"Loaded checkpoint: step={ckpt['step']}, d_model={d_model}, "
          f"predict_x0={predict_x0}, "
          f"lat_head={head_kind_short}, "
          f"sg_head={sg_head is not None}")
    if args.true_lattice:
        print("Mode: coord-only eval with ground-truth lattice")
    elif args.lat_from_aux:
        if aux_head is None:
            sys.exit("--lat-from-aux requested but checkpoint has no aux_head")
        if peak_aug_lat:
            print(f"Phase 5C: lattice from PeakAugmentedLatHead "
                  f"(n_peaks={n_peaks_train})")
        elif constrained_lat:
            print("Phase 5B: lattice from ConstrainedLatHead (physical units, sigmoid-bounded)")
        else:
            print("Phase 5: lattice from AuxLatHead (Path A — known to underperform)")
    if args.sg_constrain_lat:
        if sg_head is None:
            sys.exit("--sg-constrain-lat requested but checkpoint has no sg_head")
        print("Phase 6.2: applying SG-based crystal-system constraints to predicted lattice")
    if args.n_samples > 1:
        print(f"Phase 4.1: ensemble n_samples={args.n_samples}, eta={args.ensemble_eta}")
    if args.refine_steps > 0:
        print(f"Phase 4.2: Rietveld refinement steps={args.refine_steps}, "
              f"lr={args.refine_lr}, refine_lattice={args.refine_lattice}")

    # Dataset (precompute peak features iff the checkpoint's head needs them)
    ds = CrystalPXRDDataset(
        ROOT / "data", split=args.split,
        n_peaks=n_peaks_train if peak_aug_lat else 0,
        augment=args.noise_aug_eval,
        augment_seed=42,
    )
    if args.noise_aug_eval:
        print("Phase 7 eval: applying noise augmentation to test patterns "
              "(deterministic seed=42)")

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
    wyckoff = None
    if "wyckoff" in batch_items[0]:
        wyckoff = torch.stack([b["wyckoff"] for b in batch_items]).to(device)
    peak_features = None
    if peak_aug_lat and "peak_features" in batch_items[0]:
        peak_features = torch.stack(
            [b["peak_features"] for b in batch_items]
        ).to(device)

    # DiffPXRD module (used iff Phase 4 features enabled)
    debye = None
    if args.n_samples > 1 or args.refine_steps > 0:
        debye = DiffPXRD(
            n_bins=args.debye_n_bins,
            hkl_max=args.debye_hkl_max,
        ).to(device).eval()

    # Sampler
    sampler = DDIMSampler(
        encoder, denoiser, n_steps=args.ddim_steps,
        eta=args.eta, lat_mean=lat_mean, lat_std=lat_std,
        predict_x0=predict_x0,
    )

    B = len(indices)
    bs = max(1, min(args.batch_size, B))
    n_chunks = (B + bs - 1) // bs

    # ---- Phase 5/5B: lattice from auxiliary head (always computed for diagnostic)
    aux_lat_params: torch.Tensor | None = None
    aux_lat_matrix: torch.Tensor | None = None
    sg_pred_top1: torch.Tensor | None = None
    if aux_head is not None:
        with torch.no_grad():
            pxrd_global_full, _ = encoder(pxrd)                    # (B, d)
            if peak_aug_lat:
                head_out = aux_head(pxrd_global_full, peak_features)  # (B, 6)
            else:
                head_out = aux_head(pxrd_global_full)              # (B, 6)
            if constrained_lat or peak_aug_lat:
                # Heads with sigmoid-bounded outputs emit physical units already.
                aux_lat_params = head_out
            elif lat_mean is not None and lat_std is not None:
                aux_lat_params = (head_out * lat_std.to(device)
                                  + lat_mean.to(device))
            else:
                aux_lat_params = head_out

            # Optional Phase 6.1 / 6.2: predict SG and project lattice onto
            # the corresponding crystal-system manifold.
            if sg_head is not None:
                sg_logits = sg_head(pxrd_global_full)              # (B, 230)
                sg_pred_top1 = sg_logits.argmax(dim=-1) + 1        # (B,) in 1-230
                # Top-1 / top-5 SG accuracy diagnostic
                sg_topk = sg_logits.topk(5, dim=-1).indices + 1
                sg_true = torch.tensor(
                    [int(b["spacegroup"].item()) for b in batch_items],
                    device=device,
                )
                top1_acc = (sg_pred_top1 == sg_true).float().mean().item()
                top5_acc = (sg_topk == sg_true.unsqueeze(-1)).any(-1).float().mean().item()
                print(f"SG-head accuracy (n={B}): top1={top1_acc:.3f}  top5={top5_acc:.3f}")

                if args.sg_constrain_lat:
                    aux_lat_params = apply_sg_constraints(aux_lat_params, sg_pred_top1)

        # Aux/constrained-head MAE per dim vs. ground truth (cheap, always shown).
        mae = (aux_lat_params - lattice_params_true).abs().mean(dim=0)
        head_kind = (
            "PeakAug-lat" if peak_aug_lat
            else "Constrained-lat" if constrained_lat
            else "Aux-head"
        )
        sg_constr_str = " (after SG constraint)" if (sg_head is not None and args.sg_constrain_lat) else ""
        print(f"{head_kind} lattice MAE vs true{sg_constr_str} (a,b,c in Å, α,β,γ in °):")
        print(f"  a={mae[0]:.3f}  b={mae[1]:.3f}  c={mae[2]:.3f}  "
              f"α={mae[3]:.2f}  β={mae[4]:.2f}  γ={mae[5]:.2f}")
        # How many aux lattice predictions are physically valid?
        aux_lat_np = aux_lat_params.cpu().numpy()
        n_valid = sum(is_valid_lattice(lp) for lp in aux_lat_np)
        print(f"{head_kind} lattice validity: {n_valid}/{B} "
              f"({100.0 * n_valid / B:.1f}%)")
        # Build the corresponding 3×3 matrices (clamp to a sane physical range
        # before the matrix construction so we don't take sqrt of a negative
        # number for pathological predictions). The clamp matches is_valid_lattice.
        aux_lat_clamped = aux_lat_params.clone()
        aux_lat_clamped[:, :3] = aux_lat_clamped[:, :3].clamp(min=0.5, max=100.0)
        aux_lat_clamped[:, 3:] = aux_lat_clamped[:, 3:].clamp(min=10.0, max=170.0)
        aux_lat_matrix = lattice_params_to_matrix(aux_lat_clamped)

    # When Phase 5 is requested (and we're not in --true-lattice mode), the
    # sampler should use aux-head lattice both as lattice_init and as the
    # final predicted lat_params for downstream eval/refinement.
    use_aux_lat = (args.lat_from_aux and not args.true_lattice
                   and aux_head is not None)
    sampler_lattice = aux_lat_matrix if use_aux_lat else lattice

    # Allocate output tensors filled in by chunked predict pass
    N = atom_types.shape[1]
    pred_coords = torch.empty(B, N, 3, device=device)
    pred_lat_params = torch.empty(B, 6, device=device)

    if args.n_samples > 1:
        print(f"Ensemble: {B} patterns x {args.n_samples} candidates "
              f"in chunks of {bs} ({n_chunks} chunk{'s' if n_chunks > 1 else ''})")
    else:
        print(f"Sampling {B} structures in chunks of {bs} "
              f"({n_chunks} chunk{'s' if n_chunks > 1 else ''}, "
              f"{args.ddim_steps} DDIM steps)")

    if args.refine_steps > 0:
        print(f"Refinement: {args.refine_steps} Adam steps @ lr={args.refine_lr}, "
              f"refine_lattice={args.refine_lattice}")

    t_sample = 0.0
    t_select = 0.0
    t_refine = 0.0
    pearson_scores_all = []
    refine_loss_first = []
    refine_loss_last = []

    # When using a fixed lattice (true or aux), the ensemble selector and the
    # refinement loop should both consume that lattice rather than the
    # sampler's diffusion-predicted one.
    use_fixed_lat = args.true_lattice or use_aux_lat

    for ci, lo in enumerate(range(0, B, bs)):
        hi = min(lo + bs, B)
        cb = hi - lo

        pxrd_c = pxrd[lo:hi]
        at_c = atom_types[lo:hi]
        lat_c = sampler_lattice[lo:hi]
        mask_c = mask[lo:hi]
        wyck_c = wyckoff[lo:hi] if wyckoff is not None else None

        # ---- Sample ---------------------------------------------------------
        t0 = time.perf_counter()
        if args.n_samples > 1:
            cand_coords_c, cand_lat_c = sampler.sample_ensemble(
                pxrd_c, at_c, lat_c, mask_c,
                n_samples=args.n_samples, eta=args.ensemble_eta, wyckoff=wyck_c,
            )
            t_sample += time.perf_counter() - t0

            # Build the lattice tensor used for scoring
            if use_fixed_lat:
                lattice_for_score_c = lat_c  # (cb, 3, 3) — broadcasts inside
            else:
                lp_flat = cand_lat_c.reshape(cb * args.n_samples, 6)
                lat_mat_flat = lattice_params_to_matrix(lp_flat)
                lattice_for_score_c = lat_mat_flat.view(cb, args.n_samples, 3, 3)

            t0 = time.perf_counter()
            pc_c, pl_c, scores_c = select_best_by_pearson(
                cand_coords_c, cand_lat_c, at_c, mask_c,
                lattice_for_score_c, pxrd_c, debye,
            )
            t_select += time.perf_counter() - t0
            pearson_scores_all.append(scores_c.detach().cpu())
        else:
            pc_c, pl_c = sampler.sample(pxrd_c, at_c, lat_c, mask_c, wyckoff=wyck_c)
            t_sample += time.perf_counter() - t0

        # ---- Phase 5: substitute aux-head lattice for the diffusion lattice
        if use_aux_lat:
            pl_c = aux_lat_params[lo:hi]

        # ---- Refine ---------------------------------------------------------
        if args.refine_steps > 0:
            if use_fixed_lat:
                lat_for_refine_c = lat_c
            else:
                lat_for_refine_c = lattice_params_to_matrix(pl_c)

            t0 = time.perf_counter()
            pc_c, lat_refined_c, hist = refine_structure(
                pc_c, at_c, lat_for_refine_c, mask_c, pxrd_c, debye,
                steps=args.refine_steps, lr=args.refine_lr,
                refine_lattice=args.refine_lattice,
            )
            t_refine += time.perf_counter() - t0
            refine_loss_first.append(hist[0])
            refine_loss_last.append(hist[-1])

            if args.refine_lattice:
                from pymatgen.core import Lattice as PmgLat
                new_params = []
                for i in range(cb):
                    m = lat_refined_c[i].cpu().numpy()
                    lp = PmgLat(m).parameters
                    new_params.append(torch.tensor(lp, device=device))
                pl_c = torch.stack(new_params)

        pred_coords[lo:hi] = pc_c
        pred_lat_params[lo:hi] = pl_c

        if (ci + 1) % max(1, n_chunks // 10) == 0 or ci == n_chunks - 1:
            print(f"  chunk {ci + 1}/{n_chunks} done "
                  f"(sample={t_sample:.1f}s  select={t_select:.1f}s  refine={t_refine:.1f}s)")

    print(f"Total: sample={t_sample:.1f}s  select={t_select:.1f}s  "
          f"refine={t_refine:.1f}s")
    if pearson_scores_all:
        scores_all = torch.cat(pearson_scores_all)
        print(f"Selected-candidate pearson: mean={scores_all.mean().item():.3f} "
              f"median={scores_all.median().item():.3f}")
    if refine_loss_first:
        print(f"Refinement loss avg first->last: "
              f"{sum(refine_loss_first) / len(refine_loss_first):.4f} -> "
              f"{sum(refine_loss_last) / len(refine_loss_last):.4f}")

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
        if i < 30 or args.n <= 50:
            # Avoid drowning the terminal on n=1000 runs
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

        if args.out_json:
            agg["config"] = {
                "ckpt": args.ckpt,
                "n": args.n,
                "true_lattice": args.true_lattice,
                "lat_from_aux": args.lat_from_aux,
                "sg_constrain_lat": args.sg_constrain_lat,
                "noise_aug_eval": args.noise_aug_eval,
                "n_samples": args.n_samples,
                "ensemble_eta": args.ensemble_eta,
                "refine_steps": args.refine_steps,
                "refine_lr": args.refine_lr,
                "refine_lattice": args.refine_lattice,
                "ddim_steps": args.ddim_steps,
                "eta": args.eta,
            }
            Path(args.out_json).write_text(json.dumps(agg, indent=2))
            print(f"\nWrote aggregate metrics to {args.out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
