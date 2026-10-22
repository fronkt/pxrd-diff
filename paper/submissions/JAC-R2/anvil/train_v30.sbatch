#!/bin/bash
#SBATCH -A che260157-gpu
#SBATCH -p gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus-per-node=1
#SBATCH -t 10:00:00
#SBATCH -J pxrd-v30-train
#SBATCH -o /anvil/scratch/x-fcai3/pxrd_ops/%x_%j.out
# JAC R2 redesign: simulate patterns (if missing), train v30, gate G2 at 10k steps
# (analysis/denoiser_collapse/csp_gate.py, criteria fixed in tasks/todo.md Phase 18).
# G2 FAIL -> kill training, exit 3 (the dependent eval array then never starts).
set -u
P=/anvil/projects/x-che260157
export PATH=$P/pxrd_env/bin:$PATH
OPS=/anvil/scratch/x-fcai3/pxrd_ops
RUN=${RUN:-gpu_v30_csp}
cd "$P/pxrd-diff"
echo "commit $(git log --oneline -1)  host $(hostname)  start $(date -u +%FT%TZ)"
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))" || exit 2
[ -f data/cache/val.npz ] || python scripts/01_simulate_pxrd.py --splits test val train --workers 30 || exit 2
OUT=paper/phase18_results
GATE=paper/submissions/JAC-R2/analysis/denoiser_collapse/csp_gate.py
mkdir -p "$OUT/gate"
RES=""
LAST=$(ls runs/$RUN/ckpt_[0-9]*.pt 2>/dev/null | sort | tail -1)
[ -n "$LAST" ] && RES="--resume $LAST"
python scripts/14_train_csp.py --run-name $RUN --steps 100000 --bs 256 --workers 8 $RES \
    >> "$OPS/train_$RUN.log" 2>&1 &
TP=$!
if [ ! -f "$OUT/gate/${RUN}_010000.json" ]; then
    until [ -f runs/$RUN/ckpt_010000.pt ] || ! kill -0 $TP 2>/dev/null; do sleep 30; done
    sleep 60
    kill -0 $TP 2>/dev/null || { echo "training died before 10k"; tail -40 "$OPS/train_$RUN.log"; exit 4; }
    python "$GATE" . runs/$RUN/ckpt_010000.pt runs/$RUN/val.jsonl "$OUT/gate/${RUN}_010000.json"
    rc=$?
    echo "G2 rc=$rc $(date -u +%FT%TZ)"
    if [ $rc -ne 0 ]; then kill $TP; echo "G2 FAIL: training stopped"; exit 3; fi
fi
wait $TP
rc=$?
echo "train rc=$rc $(date -u +%FT%TZ)"
[ -f runs/$RUN/ckpt_final.pt ] || exit 5
python "$GATE" . runs/$RUN/ckpt_final.pt runs/$RUN/val.jsonl "$OUT/gate/${RUN}_final.json"
echo "done $(date -u +%FT%TZ)"
