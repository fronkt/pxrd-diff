#!/usr/bin/env bash
# P1.5 robustness row: how does the classical indexer degrade under realistic
# 2theta miscalibration (zero-point shift, sample displacement)? n=300 test.
set -uo pipefail
cd /workspace/pxrd-diff
source /venv/main/bin/activate
RES=/workspace/results; mkdir -p "$RES"
N=300
run () { # label  extra-args
  local label="$1"; shift
  [ -f "$RES/robust_${label}.json" ] && { echo "skip $label"; return; }
  echo "[$(date +%H:%M:%S)] robust $label :: $*"
  python3 scripts/09b_index_robustness.py --n "$N" --out "$RES/robust_${label}.json" "$@" \
    > /workspace/robust_${label}.log 2>&1 || echo "[WARN] robust $label failed"
}
run base
run zshift0.05  --zero-shift 0.05
run zshift0.10  --zero-shift 0.10
run zshift0.20  --zero-shift 0.20
run disp0.10    --displacement 0.10
run disp0.20    --displacement 0.20
echo "[$(date +%H:%M:%S)] ===== ROBUSTNESS DONE ====="
