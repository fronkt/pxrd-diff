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
    # ---- Phase 9 -------------------------------------------------------------
    ap.add_argument("--lat-from-index", default=None,
                    help="Phase 9.1: use classically-indexed unit cells as the "
                         "fixed lattice. Takes a JSON from 09_index_benchmark.py "
                         "(its `rows`, each with material_id + pred_params). "
                         "Patterns whose pattern was not indexed fall back to the "
                         "true lattice; coverage is reported. Ignored when "
                         "--true-lattice is also set.")
    ap.add_argument("--lat-from-index-topk", default=None,
                    help="Phase 9.1.4: like --lat-from-index but uses the top-K "
                         "ranked candidate cells per structure (from indexer run "
                         "with --topk K). For each pattern, sample "
                         "--samples-per-cell candidates per cell, score all "
                         "K*M candidates by Debye-Pearson against target, pick "
                         "the global best. The picked cell becomes the final "
                         "predicted lattice. Mutually exclusive with --lat-from-index.")
    ap.add_argument("--samples-per-cell", type=int, default=4,
                    help="Phase 9.1.4: samples per candidate cell. With "
                         "--lat-from-index-topk this overrides --n-samples; "
                         "total candidates per pattern = K * samples-per-cell.")
    # ---- Phase 9.2 -----------------------------------------------------------
    ap.add_argument("--guide-scale", type=float, default=0.0,
                    help="Phase 9.2: Debye-gradient guidance strength. 0 = "
                         "current behaviour. ~0.1-1.0 typical. Multiplied per "
                         "step by (1-t) so late steps get more guidance.")
    ap.add_argument("--guide-start-t", type=float, default=0.5,
                    help="Phase 9.2: only apply guidance when t <= this value "
                         "(skip the noisy early steps). Default 0.5.")
    ap.add_argument("--guide-clip", type=float, default=0.05,
                    help="Phase 9.2: per-sample L2-norm clip on the guidance "
                         "nudge to keep samples on the data manifold.")
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
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for torch/numpy (Phase 12: multi-seed CIs)")
    ap.add_argument("--per-sample-json", default=None,
                    help="If set, dump per-structure flags as JSONL (one row per "
                         "material) for paired McNemar / bootstrap re-analysis")
    args = ap.parse_args()

    # Seed everything so multi-seed reruns are reproducible and so a paired
    # McNemar test across lattice sources sees the same material order/noise.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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
    use_d_spacing = bool(train_args.get("use_d_spacing", False))
    n_peaks_train = int(train_args.get("n_peaks", 20))
    aux_head = None
    if "aux_head" in ckpt:
        if peak_aug_lat:
            aux_head = PeakAugmentedLatHead(
                d_model=d_model, peak_dim=2 * n_peaks_train,
                use_d_spacing=use_d_spacing,
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
            mode = "Phase 5D d-spacing" if use_d_spacing else "Phase 5C raw 2θ"
            print(f"{mode}: lattice from PeakAugmentedLatHead "
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

    # DiffPXRD module (used iff Phase 4/9 features enabled)
    debye = None
    if (args.n_samples > 1 or args.refine_steps > 0
            or args.lat_from_index_topk or args.guide_scale > 0):
        debye = DiffPXRD(
            n_bins=args.debye_n_bins,
            hkl_max=args.debye_hkl_max,
        ).to(device).eval()
    use_guide = args.guide_scale > 0.0 and debye is not None
    if use_guide:
        print(f"Phase 9.2: Debye-gradient guidance — scale={args.guide_scale}, "
              f"start_t={args.guide_start_t}, clip={args.guide_clip}")

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

    # ---- Phase 9.1: lattice from classical indexing -------------------------
    if args.lat_from_index and args.lat_from_index_topk:
        sys.exit("--lat-from-index and --lat-from-index-topk are mutually exclusive")
    index_lat_params = None
    index_lat_matrix = None
    use_index_lat = bool(args.lat_from_index) and not args.true_lattice
    if use_index_lat:
        with open(args.lat_from_index) as f:
            idx_rows = json.load(f).get("rows", [])
        idx_by_id = {r["mid"]: r["pred_params"]
                     for r in idx_rows if r.get("pred_params")}
        index_lat_params = lattice_params_true.clone()        # fall back to true
        n_cov = 0
        for bi, mid in enumerate(material_ids):
            if mid in idx_by_id:
                index_lat_params[bi] = torch.tensor(
                    idx_by_id[mid], dtype=torch.float32, device=device)
                n_cov += 1
        print(f"Phase 9.1: indexed-cell lattice — {n_cov}/{B} covered "
              f"({100.0 * n_cov / B:.1f}%); uncovered fall back to true lattice")
        mae = (index_lat_params - lattice_params_true).abs().mean(dim=0)
        print(f"  indexed-cell MAE vs true: a={mae[0]:.3f} b={mae[1]:.3f} "
              f"c={mae[2]:.3f}  α={mae[3]:.2f} β={mae[4]:.2f} γ={mae[5]:.2f}")
        idx_clamped = index_lat_params.clone()
        idx_clamped[:, :3] = idx_clamped[:, :3].clamp(min=0.5, max=100.0)
        idx_clamped[:, 3:] = idx_clamped[:, 3:].clamp(min=10.0, max=170.0)
        index_lat_matrix = lattice_params_to_matrix(idx_clamped)

    # ---- Phase 9.1.4: top-K candidate cells, picked by Debye-Pearson rerank --
    index_topk_lat_params = None      # (B, K, 6)
    index_topk_lat_matrix = None      # (B, K, 3, 3)
    use_index_topk = bool(args.lat_from_index_topk) and not args.true_lattice
    if use_index_topk:
        with open(args.lat_from_index_topk) as f:
            topk_rows = json.load(f).get("rows", [])
        topk_by_id = {r["mid"]: r.get("candidates") for r in topk_rows
                      if r.get("candidates")}
        # Determine K from data; pad shorter cell lists by repeating last cell.
        Ks = [len(c) for c in topk_by_id.values()]
        if not Ks:
            sys.exit("--lat-from-index-topk JSON has no candidates")
        K = max(Ks)
        print(f"Phase 9.1.4: top-K rerank — K_max={K}, "
              f"K_observed=[{min(Ks)}, {max(Ks)}], M={args.samples_per_cell} "
              f"samples per cell → {K * args.samples_per_cell} candidates per pattern")
        # Fall back: for patterns with no candidates use true lattice (repeated K).
        index_topk_lat_params = lattice_params_true.unsqueeze(1).expand(-1, K, -1).clone()
        n_cov_topk = 0
        for bi, mid in enumerate(material_ids):
            cands = topk_by_id.get(mid)
            if not cands:
                continue
            n_cov_topk += 1
            for ki in range(K):
                # repeat last cell if fewer than K candidates
                ci = min(ki, len(cands) - 1)
                index_topk_lat_params[bi, ki] = torch.tensor(
                    cands[ci]["params"], dtype=torch.float32, device=device)
        print(f"  coverage: {n_cov_topk}/{B} patterns "
              f"({100.0 * n_cov_topk / B:.1f}%); uncovered patterns use true cell × K")
        flat = index_topk_lat_params.reshape(B * K, 6).clone()
        flat[:, :3] = flat[:, :3].clamp(min=0.5, max=100.0)
        flat[:, 3:] = flat[:, 3:].clamp(min=10.0, max=170.0)
        index_topk_lat_matrix = lattice_params_to_matrix(flat).view(B, K, 3, 3)

    # When Phase 5 is requested (and we're not in --true-lattice mode), the
    # sampler should use aux-head lattice both as lattice_init and as the
    # final predicted lat_params for downstream eval/refinement.
    use_aux_lat = (args.lat_from_aux and not args.true_lattice
                   and aux_head is not None and not use_index_lat
                   and not use_index_topk)
    # In topk mode the lattice fed to sampler is per-cell (handled inside loop);
    # use the lattice tensor here as a placeholder.
    sampler_lattice = (index_lat_matrix if use_index_lat
                       else aux_lat_matrix if use_aux_lat else lattice)

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
    use_fixed_lat = (args.true_lattice or use_aux_lat or use_index_lat
                     or use_index_topk)
    picked_cell_ranks: list[int] = []   # 9.1.4 diagnostic

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
        if use_index_topk:
            # 9.1.4: K candidate cells × M samples per cell.
            K = index_topk_lat_matrix.shape[1]
            M = args.samples_per_cell
            lat_topk_c = index_topk_lat_matrix[lo:hi]              # (cb, K, 3, 3)
            lat_topk_params_c = index_topk_lat_params[lo:hi]       # (cb, K, 6)
            # Tile chunk by K: each pattern repeated for each cell.
            pxrd_kc = pxrd_c.unsqueeze(1).expand(-1, K, *([-1] * (pxrd_c.dim() - 1))) \
                            .reshape(cb * K, *pxrd_c.shape[1:])
            at_kc = at_c.unsqueeze(1).expand(-1, K, -1).reshape(cb * K, N)
            mask_kc = mask_c.unsqueeze(1).expand(-1, K, -1).reshape(cb * K, N)
            wyck_kc = (wyck_c.unsqueeze(1).expand(-1, K, -1).reshape(cb * K, N)
                       if wyck_c is not None else None)
            lat_kc = lat_topk_c.reshape(cb * K, 3, 3)
            # Sample M candidates per (pattern, cell)
            cand_coords_kc, cand_lat_kc = sampler.sample_ensemble(
                pxrd_kc, at_kc, lat_kc, mask_kc,
                n_samples=M, eta=args.ensemble_eta, wyckoff=wyck_kc,
                guide_debye=debye if use_guide else None,
                guide_target=pxrd_kc if use_guide else None,
                guide_scale=args.guide_scale,
                guide_start_t=args.guide_start_t,
                guide_clip=args.guide_clip,
            )                                                       # (cb*K, M, N, 3)
            t_sample += time.perf_counter() - t0
            # Reshape to (cb, K*M, ...)
            cand_coords_c = cand_coords_kc.view(cb, K, M, N, 3).reshape(cb, K * M, N, 3)
            cand_lat_c = cand_lat_kc.view(cb, K, M, 6).reshape(cb, K * M, 6)
            # Lattice for Debye scoring: per-cell, broadcast across M
            lat_for_score_c = lat_topk_c.unsqueeze(2).expand(-1, K, M, -1, -1) \
                                        .reshape(cb, K * M, 3, 3)
            t0 = time.perf_counter()
            pc_c, _ignored_lat, scores_c = select_best_by_pearson(
                cand_coords_c, cand_lat_c, at_c, mask_c,
                lat_for_score_c, pxrd_c, debye,
            )
            t_select += time.perf_counter() - t0
            pearson_scores_all.append(scores_c.detach().cpu())
            # Recover which cell (0..K-1) was picked per pattern by re-doing argmax
            # on the per-candidate scores; cheap.
            with torch.no_grad():
                coords_flat = cand_coords_c.reshape(cb * K * M, N, 3)
                at_rep_full = at_c.unsqueeze(1).expand(-1, K * M, -1).reshape(cb * K * M, N)
                mask_rep_full = mask_c.unsqueeze(1).expand(-1, K * M, -1).reshape(cb * K * M, N)
                lat_rep_full = lat_for_score_c.reshape(cb * K * M, 3, 3)
                pred_p = debye(coords_flat, at_rep_full, lat_rep_full, mask_rep_full)
                from pxrd_diff.sampler import pearson_score                       # noqa: E402
                targ_p = pxrd_c.unsqueeze(1).expand(-1, K * M, -1).reshape(cb * K * M, -1)
                scores_full = pearson_score(pred_p, targ_p).view(cb, K, M)
                cell_idx_per_pattern = scores_full.max(dim=2).values.argmax(dim=1)  # (cb,)
            for bi in range(cb):
                picked_cell_ranks.append(int(cell_idx_per_pattern[bi].item()))
            # The picked lattice (per pattern) becomes the predicted lat_params
            pl_c = lat_topk_params_c[torch.arange(cb, device=device),
                                     cell_idx_per_pattern]               # (cb, 6)
            # Override lat_c for refinement to the per-pattern picked cell
            lat_c = lat_topk_c[torch.arange(cb, device=device),
                               cell_idx_per_pattern]                     # (cb, 3, 3)
        elif args.n_samples > 1:
            cand_coords_c, cand_lat_c = sampler.sample_ensemble(
                pxrd_c, at_c, lat_c, mask_c,
                n_samples=args.n_samples, eta=args.ensemble_eta, wyckoff=wyck_c,
                guide_debye=debye if use_guide else None,
                guide_target=pxrd_c if use_guide else None,
                guide_scale=args.guide_scale,
                guide_start_t=args.guide_start_t,
                guide_clip=args.guide_clip,
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
            pc_c, pl_c = sampler.sample(
                pxrd_c, at_c, lat_c, mask_c, wyckoff=wyck_c,
                guide_debye=debye if use_guide else None,
                guide_target=pxrd_c if use_guide else None,
                guide_scale=args.guide_scale,
                guide_start_t=args.guide_start_t,
                guide_clip=args.guide_clip,
            )
            t_sample += time.perf_counter() - t0

        # ---- Phase 5/9: substitute external lattice for the diffusion lattice
        if use_aux_lat:
            pl_c = aux_lat_params[lo:hi]
        elif use_index_lat:
            pl_c = index_lat_params[lo:hi]
        # (use_index_topk: pl_c already set above to the picked cell.)

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
    if use_index_topk and picked_cell_ranks:
        from collections import Counter
        ctr = Counter(picked_cell_ranks)
        K = index_topk_lat_matrix.shape[1]
        total = sum(ctr.values())
        dist = "  ".join(f"rank{k}: {ctr.get(k, 0)} ({100*ctr.get(k, 0)/total:.1f}%)"
                         for k in range(K))
        print(f"Phase 9.1.4 picked-cell distribution: {dist}")
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

    # Per-structure flag dump (Phase 12): enables a paired McNemar test across
    # lattice sources and bootstrap CIs that the aggregate JSON cannot support.
    if args.per_sample_json and metrics_list:
        with open(args.per_sample_json, "w") as f:
            for m in metrics_list:
                f.write(json.dumps({
                    "material_id": m.material_id,
                    "seed": args.seed,
                    "composition_ok": bool(m.composition_ok),
                    "match": bool(not np.isnan(m.rmsd)),
                    "sg_match@0.1": bool(m.sg_match.get(0.1, False)),
                    "rmsd": None if np.isnan(m.rmsd) else float(m.rmsd),
                    "all_correct": bool(m.all_correct),
                }) + "\n")
        print(f"Wrote per-sample flags to {args.per_sample_json}")

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
                "lat_from_index": args.lat_from_index,
                "lat_from_index_topk": args.lat_from_index_topk,
                "samples_per_cell": args.samples_per_cell,
                "sg_constrain_lat": args.sg_constrain_lat,
                "noise_aug_eval": args.noise_aug_eval,
                "n_samples": args.n_samples,
                "ensemble_eta": args.ensemble_eta,
                "refine_steps": args.refine_steps,
                "refine_lr": args.refine_lr,
                "refine_lattice": args.refine_lattice,
                "ddim_steps": args.ddim_steps,
                "eta": args.eta,
                "seed": args.seed,
            }
            if use_index_topk and picked_cell_ranks:
                from collections import Counter
                ctr = Counter(picked_cell_ranks)
                K = index_topk_lat_matrix.shape[1]
                agg["picked_cell_distribution"] = {
                    f"rank{k}": ctr.get(k, 0) for k in range(K)
                }
            Path(args.out_json).write_text(json.dumps(agg, indent=2))
            print(f"\nWrote aggregate metrics to {args.out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
