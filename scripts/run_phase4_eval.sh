#!/usr/bin/env bash
# Phase 4 evaluation: compare single-sample / ensemble / ensemble+refine
# on n=1000 test patterns using the v13 best checkpoint with true lattice.
#
# Usage:
#   bash scripts/run_phase4_eval.sh [ckpt_path] [n] [n_samples] [refine_steps]
#
# Defaults: runs/gpu_v13/ckpt_final.pt, n=1000, n_samples=20, refine_steps=200
set -euo pipefail

CKPT="${1:-runs/gpu_v13/ckpt_final.pt}"
N="${2:-1000}"
NSAMPLES="${3:-20}"
REFINE="${4:-200}"

OUT_DIR="runs/phase4"
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
        --true-lattice \
        --out-json "$out_json" \
        "$@" \
        2>&1 | tee "$OUT_DIR/${label}.log" | tail -60
    echo
}

# Baseline: single sample, no refinement (matches existing v13 reported numbers)
run_one "baseline_single" \
    --n-samples 1 --refine-steps 0

# Phase 4.1: ensemble + Pearson selection
run_one "phase4_1_ensemble" \
    --n-samples "$NSAMPLES" --ensemble-eta 0.0 --refine-steps 0

# Phase 4.1 + 4.2: ensemble + Rietveld refinement
run_one "phase4_2_ensemble_refine" \
    --n-samples "$NSAMPLES" --ensemble-eta 0.0 \
    --refine-steps "$REFINE" --refine-lr 1e-3

echo "================================================================"
echo "  Summary"
echo "================================================================"
python3 - <<EOF
import json, sys
from pathlib import Path

out_dir = Path("$OUT_DIR")
labels = ["baseline_single", "phase4_1_ensemble", "phase4_2_ensemble_refine"]
keys = ["headline_all_correct", "rmsd_mean", "rmsd_median",
        "match_rate (StructureMatcher)", "pearson_mean", "rwp_mean", "sg_match@0.1"]

print(f"{'metric':<32s}" + "".join(f"{l:>26s}" for l in labels))
print("-" * (32 + 26 * len(labels)))
data = {l: json.loads((out_dir / f"{l}.json").read_text()) for l in labels}
for k in keys:
    row = f"{k:<32s}"
    for l in labels:
        v = data[l].get(k, float("nan"))
        row += f"{v:>26.4f}"
    print(row)
EOF
