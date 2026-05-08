#!/usr/bin/env bash
# Phase 7 robustness evaluation: 2x2 design — {clean-trained v18, aug-trained v19}
# x {clean-test, noisy-test}. All four configs use the strongest predicted-lattice
# pipeline (Phase 5C peak-aug head + Phase 6.2 SG constraint + Phase 4 ensemble +
# Rietveld refinement) so the comparison isolates the noise-augmentation effect.
#
# Usage:
#   bash scripts/run_phase7_eval.sh [n] [n_samples] [refine_steps]
#
# Defaults: n=1000, n_samples=20, refine_steps=200
set -euo pipefail

N="${1:-1000}"
NSAMPLES="${2:-20}"
REFINE="${3:-200}"

CKPT_CLEAN="runs/gpu_v18_p5c/ckpt_final.pt"
CKPT_AUG="runs/gpu_v19_p7/ckpt_final.pt"

OUT_DIR="runs/phase7"
mkdir -p "$OUT_DIR"

run_one() {
    local label="$1"; shift
    local out_json="$OUT_DIR/${label}.json"
    if [ -f "$out_json" ]; then
        echo "================================================================"
        echo "  $label  (skipping — $out_json exists)"
        echo "================================================================"
        return 0
    fi
    echo "================================================================"
    echo "  $label"
    echo "================================================================"
    python3 scripts/03_sample.py \
        --n "$N" \
        --out-json "$out_json" \
        --lat-from-aux --sg-constrain-lat \
        --n-samples "$NSAMPLES" --ensemble-eta 0.0 \
        --refine-steps "$REFINE" --refine-lr 1e-3 \
        "$@" \
        2>&1 | tee "$OUT_DIR/${label}.log" | tail -50
    echo
}

# Clean-trained model on both clean and noisy test patterns.
run_one "v18_on_clean"   --ckpt "$CKPT_CLEAN"
run_one "v18_on_noisy"   --ckpt "$CKPT_CLEAN" --noise-aug-eval

# Aug-trained model on both clean and noisy test patterns.
run_one "v19_on_clean"   --ckpt "$CKPT_AUG"
run_one "v19_on_noisy"   --ckpt "$CKPT_AUG" --noise-aug-eval

echo "================================================================"
echo "  Summary"
echo "================================================================"
python3 - <<EOF
import json
from pathlib import Path

out_dir = Path("$OUT_DIR")
labels = ["v18_on_clean", "v18_on_noisy", "v19_on_clean", "v19_on_noisy"]
keys = [
    "headline_all_correct",
    "match_rate (StructureMatcher)",
    "rmsd_mean",
    "rmsd_median",
    "pearson_mean",
    "rwp_mean",
    "sg_match@0.1",
]

data = {}
for l in labels:
    p = out_dir / f"{l}.json"
    if p.exists():
        data[l] = json.loads(p.read_text())

w = 18
print(f"{'metric':<32s}" + "".join(f"{l:>{w}s}" for l in labels))
print("-" * (32 + w * len(labels)))
for k in keys:
    row = f"{k:<32s}"
    for l in labels:
        v = data.get(l, {}).get(k, float("nan"))
        row += f"{v:>{w}.4f}"
    print(row)

# Robustness summary: how much performance drops on noisy data
print()
print("Robustness (clean -> noisy, % degradation in match_rate):")
for trained in ["v18", "v19"]:
    c = data.get(f"{trained}_on_clean", {}).get("match_rate (StructureMatcher)", float("nan"))
    n = data.get(f"{trained}_on_noisy", {}).get("match_rate (StructureMatcher)", float("nan"))
    if c > 0:
        print(f"  {trained}: {c:.4f} -> {n:.4f}  ({100*(c-n)/c:.1f}% drop)")
    else:
        print(f"  {trained}: {c:.4f} -> {n:.4f}  (clean is 0)")
EOF
