"""Phase 9.4a' — PXRDnet (Guo et al., arXiv 2406.10796) external-baseline reproduction.

Why this script exists. Crystalyze's checkpoint is still gated by an inactive
Google-Drive link (README "not yet active" as of 2026-06-01). PXRDnet is the
closest published substitute: CDVAE+XRD architecture, public HF checkpoint
(`therealgabeguo/cdvae_xrd_sinc10`/`_sinc100`), runs on MP-20.

What it does. Loads the sinc100 PXRDnet checkpoint, runs gradient-guided latent
optimization (the headline PXRDnet inference) on the first N materials in
their test pickle (which is a 1130-row subset of the CDVAE MP-20 test split),
parses each best candidate to a pymatgen.Structure, scores against the same
ground-truth CIFs we use for the DGpt baseline (via our `src/pxrd_diff/eval`
StructureMatcher harness — ltol=0.2 stol=0.3 angle_tol=5). Outputs a JSON in
the same schema as `baseline_diffractgpt_n1000.json` so the head-to-head table
in the paper has matching columns.

Why their pickle and not our test.npz. Their CrystDataset baked-in pre-computes
xrd at 4096-bin Q range (0..4pi/lambda) and applies a sinc^2 broadening from
training. Re-deriving xrd in their format from our 2theta patterns would
introduce a domain shift; using their pickled xrd is what PXRDnet was
trained on. Structure scoring is still apples-to-apples (we compare predicted
vs CDVAE-test ground-truth, same as DGpt).

Cost note. Each material runs ~3.5M decoder forward+backward passes
(`num_starting_points` x `num_gradient_steps` x cosine-warm-restart factor 7).
At ~1-5 min/material on RTX 5090 this caps practical n at ~20 per
overnight rental — single-rental ML workshop budget.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

# Suppress noisy warnings from pyg / hydra / pymatgen.
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pxrd_diff.eval import SampleMetrics, aggregate, evaluate_one  # noqa: E402
from pxrd_diff.simulator import PXRDConfig, PXRDSimulator               # noqa: E402

# Paths assumed when running on the vast.ai box (override via CLI).
PXRDNET_REPO = Path("/workspace/cdvae_xrd")
PXRDNET_CKPT_ROOT = Path("/workspace/pxrdnet_ckpt")
DEFAULT_PATTERNS = ROOT / "data" / "cache" / "test.npz"
DEFAULT_GRID = ROOT / "data" / "cache" / "two_theta.npy"
DEFAULT_TRUTH_CSV = ROOT / "data" / "raw" / "test.csv"


# ---------------------------------------------------------------------------
# Env / monkey-patch boilerplate. Must run BEFORE any cdvae import:
#  - PROJECT_ROOT for hydra ${oc.env:PROJECT_ROOT} interpolations in cfg
#  - torch.load defaults to weights_only=True since pt 2.6; their ckpt has
#    DictConfig in metadata so it errors. We trust this ckpt (HF).
# ---------------------------------------------------------------------------
def _bootstrap_env(pxrdnet_repo: Path):
    os.environ.setdefault("PROJECT_ROOT", str(pxrdnet_repo))
    os.environ.setdefault("HYDRA_JOBS", "/tmp/hydra")
    os.environ.setdefault("WABDB_DIR", "/tmp/wandb")
    sys.path.insert(0, str(pxrdnet_repo / "scripts"))
    sys.path.insert(0, str(pxrdnet_repo))   # for `visualization`, `compute_metrics`
    _orig_load = torch.load
    def _safe_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig_load(*args, **kwargs)
    torch.load = _safe_load


# ---------------------------------------------------------------------------
# Subset their test pickle into a small temp pickle so CrystDataset only
# preprocesses n materials, not all 1130. Their preprocessing is multiprocess
# CIF -> graph_arrays; on 128 cores it'd still take ~2 min for the full 1130
# but ~3s for n=20.
# ---------------------------------------------------------------------------
def make_subset_pickle(full_pickle: Path, n: int, out_pickle: Path,
                       material_ids: list[str] | None = None) -> list[str]:
    df = pd.read_pickle(full_pickle)
    if material_ids:
        df_subset = df[df["material_id"].isin(material_ids)].reset_index(drop=True)
        # Preserve user-specified order.
        df_subset = df_subset.set_index("material_id").loc[material_ids].reset_index()
    else:
        df_subset = df.iloc[:n].reset_index(drop=True)
    out_pickle.parent.mkdir(parents=True, exist_ok=True)
    df_subset.to_pickle(out_pickle)
    return df_subset["material_id"].tolist()


# ---------------------------------------------------------------------------
# Per-material inference. Mirrors conditional_generation.optimization() but
# strips file IO / plotting / wandb / smooth-vs-sinc viz so we can iterate
# fast. Returns (best_pred_struct, target_xrd_512) for downstream scoring.
# ---------------------------------------------------------------------------
def run_one(args, model, ld_kwargs, batch, dataset):
    """Run PXRDnet inference on one batch (batch_size=1) and return the
    top-ranked predicted pymatgen.Structure.
    """
    from conditional_generation import optimize_latent_code, smooth_xrds   # type: ignore
    from visualization.visualize_materials import create_materials  # type: ignore
    from pymatgen.core import Lattice, Structure
    from types import SimpleNamespace as _NS
    import wandb

    # optimize_latent_code calls wandb.log internally; init disabled per material.
    wandb.init(mode="disabled", reinit=True)
    batch = batch.to(model.device)

    target_noisy_xrd = batch.y.reshape(1, 512)

    # Step 1: latent-space gradient guidance against target XRD.
    z = optimize_latent_code(args=args, model=model, batch=batch,
                             target_noisy_xrd=target_noisy_xrd, z_init=None)

    # Step 2: annealed Langevin dynamics decodes z -> crystal candidates.
    # We pass gt_num_atoms=None and gt_atom_types=None so PXRDnet predicts
    # them from the latent (apples-to-apples with their reported "no-truth"
    # numbers); their default `--num_atom_lambda` and `--composition_lambda`
    # are still nonzero so the gradient loss soft-constrains both.
    crystals = model.langevin_dynamics(z, ld_kwargs,
                                       gt_num_atoms=None, gt_atom_types=None)
    crystals = {k: crystals[k] for k in
                ["frac_coords", "atom_types", "num_atoms", "lengths", "angles"]}

    # Step 3: assemble per-candidate pymatgen structures via their helper.
    # Mirrors `conditional_generation.create_xrd_args` exactly.
    xrd_args = SimpleNamespace(
        wave_source="CuKa",
        num_materials=args.num_starting_points,
        xrd_vector_dim=4096,
        max_theta=180, min_theta=0,
    )
    (_, _, opt_xrds, gen_crystals_list) = create_materials(
        xrd_args, crystals["frac_coords"], crystals["num_atoms"],
        crystals["atom_types"], crystals["lengths"], crystals["angles"],
        create_xrd=True, symprec=0.01,
    )

    # Step 4: rank by SIMULATED XRD of each candidate vs target (their method).
    # smooth_xrds applies the dataset's sinc + downsample to 512-bin so the
    # comparison is on the same scale as `batch.y`.
    with torch.no_grad():
        opt_smoothed, _ = smooth_xrds(opt_generated_xrds=opt_xrds,
                                      data_loader=_NS(dataset=dataset))
        opt_smoothed = opt_smoothed.to(target_noisy_xrd.device)
        target = target_noisy_xrd.expand_as(opt_smoothed)
        l1 = torch.abs(opt_smoothed - target).mean(dim=1)
        best_idx = int(l1.argmin().item())

    # Don't wrap in compute_metrics.Crystal — that triggers smact_validity ->
    # smact.neutral_ratios which broke in newer SMACT (returns 0-tuple where
    # cdvae expects 2). We just need a pymatgen Structure for our matcher.
    cd = gen_crystals_list[best_idx]
    lat = Lattice.from_parameters(*(list(cd["lengths"]) + list(cd["angles"])))
    pred_struct = Structure(lat, cd["atom_types"], cd["frac_coords"],
                            coords_are_cartesian=False)
    return pred_struct, target_noisy_xrd.squeeze().cpu().numpy()


# ---------------------------------------------------------------------------
# Main loop. Iterates n batches (batch_size=1) through the data loader and
# scores each predicted structure against ground truth.
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20,
                    help="Number of test structures to evaluate (first N from their test pickle).")
    ap.add_argument("--material-ids", default=None,
                    help="Comma-separated material_ids to subset to (overrides --n).")
    ap.add_argument("--pxrdnet-repo", default=str(PXRDNET_REPO))
    ap.add_argument("--ckpt-dir", default=str(PXRDNET_CKPT_ROOT / "mp_20_sinc100"))
    ap.add_argument("--test-pickle", default=str(PXRDNET_REPO / "data" / "mp_20" / "test.csv"))
    ap.add_argument("--truth-csv", default=str(DEFAULT_TRUTH_CSV))
    ap.add_argument("--patterns",  default=str(DEFAULT_PATTERNS))
    ap.add_argument("--grid",      default=str(DEFAULT_GRID))
    ap.add_argument("--out", default=str(
        ROOT / "paper" / "phase9_results" / "baseline_pxrdnet_sinc100_n20.json"))
    ap.add_argument("--subset-pickle-dir", default="/tmp/pxrdnet_subset")
    # PXRDnet hyperparams — match `conditional_generation_sinc100.sh` exactly.
    ap.add_argument("--num_starting_points", type=int, default=100)
    ap.add_argument("--num_candidates",      type=int, default=10)
    ap.add_argument("--num_gradient_steps",  type=int, default=5000)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--min_lr", type=float, default=1e-4)
    ap.add_argument("--l2_penalty", type=float, default=2e-4)
    ap.add_argument("--num_atom_lambda", type=float, default=0.1)
    ap.add_argument("--composition_lambda", type=float, default=0.1)
    ap.add_argument("--n_step_each", type=int, default=100)
    ap.add_argument("--step_lr", type=float, default=1e-4)
    ap.add_argument("--min_sigma", type=float, default=0.0)
    ap.add_argument("--r_min", type=float, default=0.0)
    ap.add_argument("--r_max", type=float, default=30.0)
    ap.add_argument("--l1_loss", action="store_true", default=True)
    ap.add_argument("--xrd_filter", default="both")
    # Glue args that conditional_generation reads even when we drive it ourselves.
    ap.add_argument("--save_traj", action="store_true", default=False)
    ap.add_argument("--disable_bar", action="store_true", default=False)
    ap.add_argument("--lattice_lambda", type=float, default=1e-3)
    ap.add_argument("--start_from", default="data")
    ap.add_argument("--start_from_init", default=None)
    ap.add_argument("--first_idx", type=int, default=0)
    args = ap.parse_args()

    # Bootstrap PXRDnet's import environment (env vars + sys.path + monkey-patch).
    _bootstrap_env(Path(args.pxrdnet_repo))

    # Import here, after _bootstrap_env, so cdvae sees the right env.
    from eval_utils import load_model                            # type: ignore
    from cdvae.pl_data.dataset import CrystDataset                # type: ignore
    from torch_geometric.loader import DataLoader                 # type: ignore

    print(f"[pxrdnet] loading model {args.ckpt_dir}", flush=True)
    t0 = time.time()
    model, _, cfg = load_model(Path(args.ckpt_dir), load_data=False)
    model = model.cuda().eval()
    print(f"[pxrdnet] model ready ({time.time()-t0:.0f}s)", flush=True)

    # Build subset pickle and a CrystDataset on it.
    if args.material_ids:
        mids = [m.strip() for m in args.material_ids.split(",") if m.strip()]
        subset_pkl = Path(args.subset_pickle_dir) / f"test_custom{len(mids)}.csv"
        subset_ids = make_subset_pickle(Path(args.test_pickle), len(mids), subset_pkl,
                                        material_ids=mids)
        args.n = len(subset_ids)
        print(f"[pxrdnet] subset = {len(subset_ids)} custom material_ids "
              f"from {args.test_pickle}", flush=True)
    else:
        subset_pkl = Path(args.subset_pickle_dir) / f"test_first{args.n}.csv"
        subset_ids = make_subset_pickle(Path(args.test_pickle), args.n, subset_pkl)
        print(f"[pxrdnet] subset = first {args.n} from {args.test_pickle}", flush=True)

    print(f"[pxrdnet] building CrystDataset (preprocessing {args.n} CIFs)...", flush=True)
    dataset = CrystDataset(
        name="pxrdnet-eval", path=str(subset_pkl),
        prop=cfg.data.prop, niggli=cfg.data.niggli, primitive=cfg.data.primitive,
        graph_method=cfg.data.graph_method, preprocess_workers=8,
        lattice_scale_method=cfg.data.lattice_scale_method,
        xrd_filter=cfg.data.xrd_filter,
        nanomaterial_size_angstrom=cfg.data.nanomaterial_size_angstrom,
        n_presubsample=cfg.data.n_presubsample,
        n_postsubsample=cfg.data.n_postsubsample,
        wavesource=cfg.data.wavesource,
    )
    # Hook the dataset's scaler from the loaded model.
    dataset.lattice_scaler = model.lattice_scaler
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    ld_kwargs = SimpleNamespace(
        n_step_each=args.n_step_each, step_lr=args.step_lr,
        min_sigma=args.min_sigma, save_traj=args.save_traj,
        disable_bar=True,                          # quiet the inner LD progress bar
    )

    # Build a true-structure lookup for scoring.
    truth_df = pd.read_csv(args.truth_csv)
    truth_by_id = dict(zip(truth_df["material_id"], truth_df["cif"]))
    # Pattern-domain scoring uses our PXRDSimulator + cached test pattern.
    two_theta_grid = np.load(args.grid)
    cache = np.load(args.patterns, allow_pickle=True)
    pattern_by_id = dict(zip(cache["material_id"], cache["pattern"]))
    sim = PXRDSimulator(PXRDConfig(
        two_theta_min=float(two_theta_grid[0]),
        two_theta_max=float(two_theta_grid[-1]),
        two_theta_step=float(two_theta_grid[1] - two_theta_grid[0]),
    ))

    print(f"[pxrdnet] running n={args.n} inferences "
          f"(num_starts={args.num_starting_points}, grad_steps={args.num_gradient_steps})",
          flush=True)
    metrics = []
    parse_fail = 0
    skip_fail = 0
    t1 = time.time()
    for i, batch in enumerate(loader):
        if i >= args.n:
            break
        mid = batch.mpid[0] if hasattr(batch, "mpid") else subset_ids[i]
        true_cif = truth_by_id.get(mid)
        if true_cif is None:
            skip_fail += 1
            continue
        try:
            from pymatgen.core import Structure
            true_struct = Structure.from_str(true_cif, fmt="cif")
        except Exception:
            skip_fail += 1
            continue

        t_one = time.time()
        try:
            pred_struct, _target = run_one(args, model, ld_kwargs, batch, dataset)
        except Exception as e:
            import traceback
            print(f"  [{i+1}/{args.n}] {mid}: inference error: {type(e).__name__}: {e}",
                  flush=True)
            print("    " + traceback.format_exc().replace("\n", "\n    "), flush=True)
            parse_fail += 1
            continue

        # Pattern-domain: re-simulate predicted struct on OUR test grid.
        try:
            pred_pattern = sim.simulate(pred_struct)
        except Exception:
            parse_fail += 1
            continue
        true_pattern = pattern_by_id.get(mid)
        if true_pattern is None:
            skip_fail += 1
            continue
        true_pattern = true_pattern.astype(np.float32)

        m = evaluate_one(mid, pred_struct, true_struct, pred_pattern, true_pattern)
        metrics.append(m)

        elapsed = time.time() - t1
        per = elapsed / (i + 1)
        eta = per * (args.n - i - 1)
        ok = sum(1 for x in metrics if not np.isnan(x.rmsd))
        print(f"  [{i+1}/{args.n}] {mid}  ({time.time()-t_one:.0f}s)  "
              f"matched={ok}/{len(metrics)}  ETA={eta/60:.1f}min", flush=True)

        # Intermediate dump — multi-day run, don't lose work on crash/SSH drop.
        try:
            partial = aggregate(metrics)
            partial["n_partial"] = len(metrics)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out + ".partial").write_text(json.dumps(partial, indent=2))
        except Exception:
            pass

    summary = aggregate(metrics)
    summary["n_requested"] = args.n
    summary["n_skipped_input"] = skip_fail
    summary["n_parse_fail"] = parse_fail
    summary["config"] = dict(
        model_name="therealgabeguo/cdvae_xrd_sinc100 (PXRDnet, Guo et al. 2406.10796)",
        ckpt=str(args.ckpt_dir),
        num_starting_points=args.num_starting_points,
        num_gradient_steps=args.num_gradient_steps,
        num_candidates=args.num_candidates,
        nanomaterial_size_angstrom=100,
        pattern_bins=512, pattern_domain="Q (CuKa, 0..4pi/lambda)",
        scoring="pxrd_diff.eval (StructureMatcher ltol=0.2 stol=0.3 angle_tol=5)",
        note=("PXRDnet test pickle (their pre-computed xrd) used as input; "
              "predictions scored against CDVAE MP-20 true CIFs via our matcher."),
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    elapsed = time.time() - t1
    print(f"\n[pxrdnet] wrote {out}  ({elapsed/60:.1f}min total)")
    print(f"[pxrdnet] match% = {summary.get('match_rate (StructureMatcher)', 0)*100:.2f}  "
          f"all-correct = {summary.get('headline_all_correct', 0)*100:.2f}  "
          f"n_scored = {summary.get('n', 0)}")


if __name__ == "__main__":
    main()
