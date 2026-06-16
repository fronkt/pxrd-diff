#!/usr/bin/env bash
# 3-seed Phase-4 ablation matrix: v11 (eps,lambda=1), v13 (x0,lambda=1), v13 (x0,lambda=0).
# Train from scratch (100k steps, save only ckpt_final to spare disk), then single-shot
# true-lattice eval at n=1000. Resumable: skips any run whose ckpt/json already exists.
set -uo pipefail
cd /workspace/pxrd-diff
source /venv/main/bin/activate
RES=/workspace/results; mkdir -p "$RES"
TRAIN="--lr 5e-4 --steps 100000 --bs 64 --save-every 200000"
EVAL="--n 1000 --true-lattice --n-samples 1 --refine-steps 0 --ddim-steps 50 --seed 42"

flags_for () {
  case "$1" in
    v13_l1_*) echo "--predict-x0 --debye-weight 1.0" ;;
    v13_l0_*) echo "--predict-x0 --debye-weight 0.0" ;;
    v11_*)    echo "--debye-weight 1.0" ;;            # eps-prediction (no --predict-x0)
  esac
}
seed_for () { echo "${1##*_s}"; }

ORDER="v13_l1_s0 v13_l0_s0 v11_s0 v13_l1_s1 v13_l0_s1 v11_s1 v13_l1_s2 v13_l0_s2 v11_s2"

for name in $ORDER; do
  ck="runs/${name}/ckpt_final.pt"
  fl="$(flags_for "$name")"; sd="$(seed_for "$name")"
  if [ ! -f "$ck" ]; then
    echo "[$(date +%H:%M:%S)] TRAIN $name :: $fl --seed $sd"
    python scripts/02_train.py $fl --seed "$sd" $TRAIN --run-name "$name" \
      > /workspace/${name}_train.log 2>&1 || echo "[WARN] train failed: $name"
  fi
  if [ -f "$ck" ] && [ ! -f "$RES/${name}.json" ]; then
    echo "[$(date +%H:%M:%S)] EVAL  $name"
    python scripts/03_sample.py --ckpt "$ck" $EVAL --out-json "$RES/${name}.json" \
      > /workspace/${name}_eval.log 2>&1 || echo "[WARN] eval failed: $name"
  fi
done
echo "[$(date +%H:%M:%S)] ===== MATRIX DONE ====="
