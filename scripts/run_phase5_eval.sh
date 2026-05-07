#!/usr/bin/env bash
# Phase 5 evaluation: predicted lattice quality and end-to-end match rate.
#
# Compares two lattice sources (diffusion vs aux head), each in two regimes
# (single-sample baseline vs Phase 4.1+4.2 ensemble + Rietveld). All four
# runs are without --true-lattice — i.e. the model must produce its own
# lattice estimate from the PXRD pattern.
#
# Usage:
#   bash scripts/run_phase5_eval.sh [ckpt_path] [n] [n_samples] [refine_steps]
#
# Defaults: runs/gpu_v13/ckpt_079500.pt, n=1000, n_samples=20, refine_steps=200
set -euo pipefail

CKPT="${1:-runs/gpu_v13/ckpt_079500.pt}"
N="${2:-1000}"
NSAMPLES="${3:-20}"
REFINE="${4:-200}"

OUT_DIR="runs/phase5"
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
        2>&1 | tee "$OUT_DIR/${label}.log" | tail -60
    echo
}

# Predicted-lattice mode (no --true-lattice). Diffusion vs aux head.
run_one "diffusion_lat_baseline" \
    --n-samples 1 --refine-steps 0

run_one "aux_lat_baseline" \
    --lat-from-aux \
    --n-samples 1 --refine-steps 0

run_one "diffusion_lat_phase4" \
    --n-samples "$NSAMPLES" --ensemble-eta 0.0 \
    --refine-steps "$REFINE" --refine-lr 1e-3

run_one "aux_lat_phase4" \
    --lat-from-aux \
    --n-samples "$NSAMPLES" --ensemble-eta 0.0 \
    --refine-steps "$REFINE" --refine-lr 1e-3

echo "================================================================"
echo "  Summary"
echo "================================================================"
python3 - <<EOF
import json
from pathlib import Path

out_dir = Path("$OUT_DIR")
labels = [
    "diffusion_lat_baseline",
    "aux_lat_baseline",
    "diffusion_lat_phase4",
    "aux_lat_phase4",
]
keys = [
    "headline_all_correct",
    "match_rate (StructureMatcher)",
    "rmsd_mean",
    "rmsd_median",
    "pearson_mean",
    "rwp_mean",
    "sg_match@0.1",
]

data = {l: json.loads((out_dir / f"{l}.json").read_text()) for l in labels}

w = 26
print(f"{'metric':<32s}" + "".join(f"{l:>{w}s}" for l in labels))
print("-" * (32 + w * len(labels)))
for k in keys:
    row = f"{k:<32s}"
    for l in labels:
        v = data[l].get(k, float("nan"))
        row += f"{v:>{w}.4f}"
    print(row)
EOF
