#!/usr/bin/env bash
# Checkpoint-step sweep to settle the eps-vs-x0 demotion: retrain v11 (eps) and
# v13_l1 (x0-residual) at seeds 0,1 saving every 20k; eval each checkpoint at the
# ablation protocol (true-lattice n=1000, 1 sample, no refine). Does x0 ever hit
# the original single-seed 2.5% at any checkpoint?
set -uo pipefail
cd /workspace/pxrd-diff
source /venv/main/bin/activate
RES=/workspace/results/ckpt_sweep; mkdir -p "$RES"
TRAIN="--lr 5e-4 --steps 100000 --bs 64 --save-every 20000"
EVAL="--n 1000 --true-lattice --n-samples 1 --refine-steps 0 --ddim-steps 50 --seed 42"
run () {
  name="$1"; flags="$2"; seed="$3"; rn="sweep_${name}_s${seed}"
  if [ ! -f "runs/${rn}/ckpt_final.pt" ]; then
    echo "[$(date +%H:%M:%S)] TRAIN $rn :: ${flags} seed ${seed}"
    python scripts/02_train.py ${flags} --debye-weight 1.0 --seed "$seed" ${TRAIN} --run-name "$rn" \
      > /workspace/${rn}_train.log 2>&1 || echo "[WARN] train failed $rn"
  fi
  for ck in runs/${rn}/ckpt_[0-9]*.pt; do
    [ -f "$ck" ] || continue
    step=$(basename "$ck" .pt | sed "s/ckpt_//")
    out="$RES/${rn}_${step}.json"
    [ -f "$out" ] && continue
    echo "[$(date +%H:%M:%S)] EVAL $rn $step"
    python scripts/03_sample.py --ckpt "$ck" ${EVAL} --out-json "$out" \
      > /workspace/${rn}_${step}_eval.log 2>&1 || echo "[WARN] eval failed $rn $step"
  done
}
for s in 0 1; do
  run v11    ""             "$s"
  run v13_l1 "--predict-x0" "$s"
done
echo "[$(date +%H:%M:%S)] ===== SWEEP DONE ====="
