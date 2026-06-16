#!/usr/bin/env bash
# Waits for the training matrix to finish, then runs deCIFer full generation
# (n=300, condition on XRD + composition) and re-scores through our harness.
set -uo pipefail
cd /workspace/deCIFer
source /venv/main/bin/activate
export PYTHONPATH=/workspace/deCIFer
RES=/workspace/results

echo "[$(date +%H:%M:%S)] finalize: waiting for MATRIX DONE ..."
while ! grep -q "MATRIX DONE" /workspace/matrix.log 2>/dev/null; do sleep 120; done
echo "[$(date +%H:%M:%S)] matrix done — starting deCIFer n=300 generation"

if [ ! -f "$RES/decifer_n300.json" ]; then
  python bin/evaluate.py --model-ckpt decifer_v1_ckpt.pt \
    --dataset-path data/mp20test/mp20test/serialized/test.h5 \
    --condition --add-composition --num-reps 1 \
    --out-folder /workspace/decifer_eval_full \
    --dataset-name mp20test --model-name decifer_v1 --override \
    > /workspace/decifer_gen_full.log 2>&1 || echo "[WARN] decifer gen failed"
  echo "[$(date +%H:%M:%S)] generation done — re-scoring"
  python /workspace/score_decifer.py --eval-folder /workspace/decifer_eval_full \
    --out-json "$RES/decifer_n300.json" > /workspace/decifer_score_full.log 2>&1 \
    || echo "[WARN] decifer score failed"
fi
echo "[$(date +%H:%M:%S)] ===== FINALIZE DONE ====="
