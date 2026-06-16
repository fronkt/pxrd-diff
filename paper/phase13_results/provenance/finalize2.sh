#!/usr/bin/env bash
# Stage 2: after deCIFer (FINALIZE DONE), run the oracle multi-seed (P1.6) with
# the v21 checkpoint at the paper's 5.6% protocol (true-lattice, n_samples=20,
# refine 200) across 3 sampling seeds, to firm up the single-seed oracle.
set -uo pipefail
cd /workspace/pxrd-diff
source /venv/main/bin/activate
RES=/workspace/results
echo "[$(date +%H:%M:%S)] finalize2: waiting for FINALIZE DONE (deCIFer) ..."
while ! grep -q "FINALIZE DONE" /workspace/finalize.log 2>/dev/null; do sleep 120; done
echo "[$(date +%H:%M:%S)] deCIFer done — oracle 3-seed multiseed"
for s in 0 1 2; do
  out="$RES/oracle_s${s}.json"
  [ -f "$out" ] && { echo "skip oracle_s$s"; continue; }
  echo "[$(date +%H:%M:%S)] oracle seed $s"
  python scripts/03_sample.py --ckpt runs/gpu_v21_p9/ckpt_final.pt \
    --n 1000 --true-lattice --n-samples 20 --refine-steps 200 --refine-lr 1e-3 \
    --ddim-steps 50 --seed "$s" --out-json "$out" \
    > /workspace/oracle_s${s}.log 2>&1 || echo "[WARN] oracle s$s failed"
done
echo "[$(date +%H:%M:%S)] ===== ALL DONE ====="
