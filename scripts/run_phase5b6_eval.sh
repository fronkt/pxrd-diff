#!/usr/bin/env bash
# Phase 5B + 6 evaluation: ConstrainedLatHead lattice, optional SG-based
# crystal-system constraints, with and without Phase 4 ensemble + refine.
# All four configs are without --true-lattice (lattice must be predicted).
#
# Usage:
#   bash scripts/run_phase5b6_eval.sh [ckpt_path] [n] [n_samples] [refine_steps]
#
# Defaults: runs/gpu_v17_p5b6/ckpt_final.pt, n=1000, n_samples=20, refine_steps=200
set -euo pipefail

CKPT="${1:-runs/gpu_v17_p5b6/ckpt_final.pt}"
N="${2:-1000}"
NSAMPLES="${3:-20}"
REFINE="${4:-200}"

OUT_DIR="runs/phase5b6"
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

# Phase 5B alone — constrained lattice head, single sample, no Phase 4.
run_one "p5b_baseline" \
    --lat-from-aux --n-samples 1 --refine-steps 0

# Phase 5B + Phase 4 — adds 20-candidate ensemble + 200-step refinement.
run_one "p5b_phase4" \
    --lat-from-aux \
    --n-samples "$NSAMPLES" --ensemble-eta 0.0 \
    --refine-steps "$REFINE" --refine-lr 1e-3

# Phase 5B + 6 — adds SG-based crystal-system constraints to the predicted lattice.
run_one "p5b_sg_baseline" \
    --lat-from-aux --sg-constrain-lat --n-samples 1 --refine-steps 0

# Full stack: Phase 5B + 6 + Phase 4.
run_one "p5b_sg_phase4" \
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
labels = ["p5b_baseline", "p5b_phase4", "p5b_sg_baseline", "p5b_sg_phase4"]
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
