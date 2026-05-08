#!/usr/bin/env bash
# Phase 5C evaluation: PeakAugmentedLatHead with explicit peak-position features,
# with and without SG-based crystal-system constraints, with and without Phase 4
# ensemble + Rietveld refinement. All four configs are without --true-lattice.
#
# Usage:
#   bash scripts/run_phase5c_eval.sh [ckpt_path] [n] [n_samples] [refine_steps]
#
# Defaults: runs/gpu_v18_p5c/ckpt_final.pt, n=1000, n_samples=20, refine_steps=200
set -euo pipefail

CKPT="${1:-runs/gpu_v18_p5c/ckpt_final.pt}"
N="${2:-1000}"
NSAMPLES="${3:-20}"
REFINE="${4:-200}"

OUT_DIR="runs/phase5c"
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
        --ckpt "$CKPT" \
        --n "$N" \
        --out-json "$out_json" \
        "$@" \
        2>&1 | tee "$OUT_DIR/${label}.log" | tail -70
    echo
}

run_one "p5c_baseline" \
    --lat-from-aux --n-samples 1 --refine-steps 0

run_one "p5c_phase4" \
    --lat-from-aux \
    --n-samples "$NSAMPLES" --ensemble-eta 0.0 \
    --refine-steps "$REFINE" --refine-lr 1e-3

run_one "p5c_sg_baseline" \
    --lat-from-aux --sg-constrain-lat --n-samples 1 --refine-steps 0

run_one "p5c_sg_phase4" \
    --lat-from-aux --sg-constrain-lat \
    --n-samples "$NSAMPLES" --ensemble-eta 0.0 \
    --refine-steps "$REFINE" --refine-lr 1e-3

echo "================================================================"
echo "  Summary"
echo "================================================================"
python3 - <<EOF
import json
from pathlib import Path

out_dir = Path("$OUT_DIR")
labels = ["p5c_baseline", "p5c_phase4", "p5c_sg_baseline", "p5c_sg_phase4"]
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

w = 22
print(f"{'metric':<32s}" + "".join(f"{l:>{w}s}" for l in labels))
print("-" * (32 + w * len(labels)))
for k in keys:
    row = f"{k:<32s}"
    for l in labels:
        v = data.get(l, {}).get(k, float("nan"))
        row += f"{v:>{w}.4f}"
    print(row)
EOF
