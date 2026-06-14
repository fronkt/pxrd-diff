#!/usr/bin/env bash
# Phase 12 — multi-seed reruns for the Table 2 lattice-source comparison, so the
# learned-head -> indexer lift can be tested with a PAIRED McNemar test instead
# of the underpowered single-seed / unpaired view in the workshop draft.
#
# It runs two lattice sources x several seeds, each emitting per-structure flags:
#   A = learned head only   (full pipeline, no lattice override)
#   B = classical indexer   (--lat-from-index, the §3.5 drop-in)
# then calls scripts/stats_ci.py for Wilson CIs + the paired McNemar test.
#
# Flags mirror the Phase 11.4 production run recorded in tasks/todo.md
# (n=1000, n_samples=20, refine 200, ddim 50). If your Table 2 numbers were
# produced with a different flag set, edit COMMON below to match — the point of
# this script is apples-to-apples reruns, so the flags MUST equal production.
#
# Usage:
#   bash scripts/run_phase12_multiseed.sh [CKPT] [N] ["seed0 seed1 ..."]
# Example (vast.ai, v21 checkpoint on the E:/local box):
#   bash scripts/run_phase12_multiseed.sh runs/gpu_v21_p9/ckpt_final.pt 1000 "0 1 2"
set -euo pipefail

CKPT="${1:-runs/gpu_v21_p9/ckpt_final.pt}"     # v21 final; gitignored, set to wherever it lives
N="${2:-1000}"
SEEDS="${3:-0 1 2}"
INDEX_JSON="paper/phase9_results/index_benchmark_v2_native.json"

OUT_DIR="paper/phase9_results/phase12_multiseed"
mkdir -p "$OUT_DIR"

# Keep these identical between A and B so the only difference is the lattice source.
COMMON=(--ckpt "$CKPT" --n "$N" --n-samples 20 --ensemble-eta 0.0 \
        --refine-steps 200 --refine-lr 1e-3 --ddim-steps 50)

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found at $CKPT — pass the v21 path as arg 1." >&2
    exit 1
fi

run_one() {
    local mode="$1" seed="$2"; shift 2
    local tag="${mode}_s${seed}"
    local out="$OUT_DIR/${tag}.json"
    local per="$OUT_DIR/${tag}.per_sample.jsonl"
    if [ -f "$per" ]; then
        echo "  $tag  (skip — $per exists)"; return 0
    fi
    echo "================  $tag  ================"
    python3 scripts/03_sample.py "${COMMON[@]}" \
        --seed "$seed" --out-json "$out" --per-sample-json "$per" \
        "$@" 2>&1 | tee "$OUT_DIR/${tag}.log" | tail -20
}

LEARNED=()
INDEXED=()
for s in $SEEDS; do
    run_one "learned" "$s"                                    # A: learned lattice head
    run_one "indexer" "$s" --lat-from-index "$INDEX_JSON"     # B: classical indexer drop-in
    LEARNED+=("$OUT_DIR/learned_s${s}.per_sample.jsonl")
    INDEXED+=("$OUT_DIR/indexer_s${s}.per_sample.jsonl")
done

# Join arrays with commas for stats_ci.py.
A=$(IFS=,; echo "${LEARNED[*]}")
B=$(IFS=,; echo "${INDEXED[*]}")

echo
echo "================  Wilson CIs (match)  ================"
python3 scripts/stats_ci.py ci "${LEARNED[@]}" --flag match
python3 scripts/stats_ci.py ci "${INDEXED[@]}" --flag match
echo
echo "================  Paired McNemar (match)  ================"
python3 scripts/stats_ci.py mcnemar --a "$A" --b "$B" --flag match
echo
echo "================  Paired McNemar (all_correct)  ================"
python3 scripts/stats_ci.py mcnemar --a "$A" --b "$B" --flag all_correct
echo
echo "Done. Per-structure flags + aggregates in $OUT_DIR/"
echo "Fold the pooled CIs and McNemar p into Table 2 and §5.2 of paper.md."
