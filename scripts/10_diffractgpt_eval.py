"""Phase 9.4b — DiffractGPT external-baseline reproduction.

Runs `knc6/diffractgpt_mistral_chemical_formula` (Choudhary et al., J. Phys.
Chem. Lett. 2024) on the same n=1000 CDVAE MP-20 test split we use for
Phase 9.1.3 / 9.0.7, scored by the SAME StructureMatcher harness from
`src/pxrd_diff/eval.py`. The point is an apples-to-apples comparison: any
difference in match% reflects the model, not the eval pipeline.

DiffractGPT input format (per `atomgpt.inverse_models.utils.smooth_xrd`):
    - 2theta range [0, 90], 0.3 deg bin -> 300 intensity values
    - normalized to max=1
    - rendered as ";"-separated string with 2-decimal precision
    - alpaca prompt: "The chemical formula is {f} The  XRD is {y_str} ...
       Generate atomic structure description ..."

Their output format (per `text2atoms`):
    line 0: header
    line 1: a b c (lattice lengths, space-separated)
    line 2: alpha beta gamma (lattice angles)
    line 3+: element x y z (fractional coords)

Output: JSON in our `paper/phase9_results/baseline_diffractgpt_n1000.json`,
schema matching `evaluate_one + aggregate` in src/pxrd_diff/eval.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pxrd_diff.eval import SampleMetrics, aggregate, evaluate_one
from pxrd_diff.simulator import PXRDConfig, PXRDSimulator


warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# DiffractGPT pattern format — translate our 4251-bin 5-90 deg pattern to
# their 300-bin 0-90 deg @ 0.3 deg pattern.
# ---------------------------------------------------------------------------
def to_dgpt_pattern(intensity: np.ndarray, two_theta_grid: np.ndarray,
                    bin_step: float = 0.3, theta_max: float = 90.0) -> str:
    """Render our pattern in DiffractGPT's ;-separated string format.

    Their `smooth_xrd` produces 300 values for thetas=[0, 90], intvl=0.3:
        np.arange(0, 90, 0.3) -> 300 bins centered at 0.00, 0.30, ..., 89.70.
    We linearly interpolate from our 5-90/0.02 grid onto that grid, zero-pad
    the 0-5 deg region, and renormalize so the max is 1.
    """
    x_new = np.arange(0.0, theta_max, bin_step)               # (300,)
    y_new = np.zeros_like(x_new)

    # Interpolate only where x_new is in-range of our grid; outside -> 0.
    in_range = (x_new >= two_theta_grid[0]) & (x_new <= two_theta_grid[-1])
    y_new[in_range] = np.interp(x_new[in_range], two_theta_grid, intensity)

    peak = float(np.max(y_new))
    if peak > 0:
        y_new = y_new / peak

    return ";".join(f"{x:.2f}" for x in y_new)


# ---------------------------------------------------------------------------
# Output parsing — jarvis Atoms -> pymatgen Structure.
# ---------------------------------------------------------------------------
def jarvis_atoms_to_pmg(atoms) -> Structure:
    """Convert a jarvis-tools Atoms to a pymatgen Structure."""
    lattice = Lattice(np.array(atoms.lattice.matrix))
    species = list(atoms.elements)
    coords = np.array(atoms.frac_coords)
    return Structure(lattice, species, coords, coords_are_cartesian=False)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000,
                    help="Number of test structures to evaluate.")
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache" / "test.npz"))
    ap.add_argument("--grid",  default=str(ROOT / "data" / "cache" / "two_theta.npy"))
    ap.add_argument("--csv",   default=str(ROOT / "data" / "raw" / "test.csv"))
    ap.add_argument("--out",   default=str(ROOT / "paper" / "phase9_results"
                                          / "baseline_diffractgpt_n1000.json"))
    ap.add_argument("--model-name", default="knc6/diffractgpt_mistral_chemical_formula")
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # Vanilla HF load. We do NOT use unsloth `FastLanguageModel.for_inference`
    # because that path requires xformers, which doesn't have a prebuilt wheel
    # for torch 2.11+cu130. Vanilla HF generate with bnb 4-bit quantization
    # gives identical outputs at ~10s/pattern on RTX 5090.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from atomgpt.inverse_models.utils import text2atoms

    alpaca_prompt = ("### Instruction:\n{}\n### Input:\n{}\n### Output:\n{}")
    instruction = "Below is a description of a material."

    print(f"[dgpt] loading {args.model_name} (vanilla HF, 4-bit)", flush=True)
    t0 = time.time()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, quantization_config=bnb, device_map="auto",
    )
    model.eval()
    print(f"[dgpt] model ready ({time.time()-t0:.0f}s)", flush=True)

    two_theta_grid = np.load(args.grid)
    cache = np.load(args.cache, allow_pickle=True)
    patterns = cache["pattern"]
    mat_ids = list(cache["material_id"])
    df = pd.read_csv(args.csv)
    cif_by_id = dict(zip(df["material_id"], df["cif"]))
    formula_by_id = dict(zip(df["material_id"], df["pretty_formula"]))

    # We score predicted structures against truth using our own evaluator,
    # which needs both structures and both PATTERNS. The truth pattern is
    # the cached one. The predicted-structure pattern is simulated with our
    # PXRDSimulator on the same 2theta grid (so rwp/pearson are comparable).
    sim = PXRDSimulator(PXRDConfig(
        two_theta_min=float(two_theta_grid[0]),
        two_theta_max=float(two_theta_grid[-1]),
        two_theta_step=float(two_theta_grid[1] - two_theta_grid[0]),
    ))

    n = min(args.n, len(patterns))
    print(f"[dgpt] running n={n} inferences", flush=True)
    metrics = []
    fail_count = 0
    parse_fail = 0
    t1 = time.time()
    for i in range(n):
        mid = mat_ids[i]
        formula = formula_by_id.get(mid)
        true_cif = cif_by_id.get(mid)
        if formula is None or true_cif is None:
            fail_count += 1
            continue
        try:
            true_struct = Structure.from_str(true_cif, fmt="cif")
        except Exception:
            fail_count += 1
            continue

        y_str = to_dgpt_pattern(patterns[i], two_theta_grid)
        prompt = (
            f"The chemical formula is {formula} The  XRD is {y_str}."
            f" Generate atomic structure description with lattice lengths, "
            f"angles, coordinates and atom types."
        )

        # Generation via vanilla HF. We re-implement the small slice of
        # `gen_atoms` we need (alpaca format -> generate -> extract output
        # section -> text2atoms parse) inline so we can keep the unsloth-
        # free path.
        full_prompt = alpaca_prompt.format(instruction, prompt, "")
        try:
            inputs = tokenizer([full_prompt], return_tensors="pt").to(args.device)
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens,
                    do_sample=False, use_cache=True,
                )
            decoded = tokenizer.batch_decode(out_ids)[0]
            # Match atomgpt.gen_atoms exactly: split on "# Output:" (matches
            # both "# Output:" and "### Output:"), strip only the eos token,
            # NOT trailing/leading whitespace. text2atoms reads lattice
            # lengths from index [1] of split("\n"), so the leading newline
            # after "# Output:" is load-bearing.
            response = decoded.split("# Output:")[1].strip("</s>")
            atoms = text2atoms(response)
        except Exception as e:
            print(f"  [{i+1}/{n}] {mid}: generation error: {e}", flush=True)
            parse_fail += 1
            continue
        if atoms is None:
            parse_fail += 1
            continue

        try:
            pred_struct = jarvis_atoms_to_pmg(atoms)
        except Exception as e:
            print(f"  [{i+1}/{n}] {mid}: structure parse error: {e}", flush=True)
            parse_fail += 1
            continue

        # Patterns for scoring: predicted structure -> simulated pattern;
        # true structure -> cached pattern.
        try:
            pred_pattern = sim.simulate(pred_struct)
        except Exception:
            parse_fail += 1
            continue
        true_pattern = patterns[i].astype(np.float32)

        m = evaluate_one(mid, pred_struct, true_struct, pred_pattern, true_pattern)
        metrics.append(m)

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t1
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            ok = sum(1 for x in metrics if not np.isnan(x.rmsd))
            print(f"  {i+1}/{n}  ({elapsed:.0f}s, ~{rate:.2f}/s, ETA {eta:.0f}s) "
                  f"matched={ok}/{len(metrics)}  parse_fail={parse_fail}", flush=True)

    summary = aggregate(metrics)
    summary["n_requested"] = n
    summary["n_skipped_input"] = fail_count
    summary["n_parse_fail"] = parse_fail
    summary["config"] = dict(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        pattern_bins=300,
        theta_range=[0.0, 90.0],
        bin_step=0.3,
        scoring="pxrd_diff.eval (StructureMatcher ltol=0.2 stol=0.3 angle_tol=5)",
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    elapsed = time.time() - t1
    print(f"\n[dgpt] wrote {out}  ({elapsed:.0f}s total)")
    print(f"[dgpt] match% = {summary.get('match_rate (StructureMatcher)', 0)*100:.2f}  "
          f"all-correct = {summary.get('headline_all_correct', 0)*100:.2f}  "
          f"n_scored = {summary.get('n', 0)}")


if __name__ == "__main__":
    main()
