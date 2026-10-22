#!/bin/bash
#SBATCH -A che260157-gpu
#SBATCH -p gpu-debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus-per-node=1
#SBATCH -t 00:30:00
#SBATCH -J pxrd-v30-g2
#SBATCH -o /anvil/scratch/x-fcai3/pxrd_ops/%x_%j.out
# Gate G2 on gpu-debug (30 min cap per job). Simulates and caches the data if missing,
# trains v30 with the production schedule (--steps 100000) up to step 10000 across as
# many 30-min jobs as needed (each one submits the next), then runs csp_gate.py at 10k.
# Own run name (gpu_v30_g2), so it cannot collide with the queued production job.
set -u
P=/anvil/projects/x-che260157
export PATH=$P/pxrd_env/bin:$PATH
OPS=/anvil/scratch/x-fcai3/pxrd_ops
RUN=gpu_v30_g2
START=$(date +%s)
cd "$P/pxrd-diff"
echo "commit $(git log --oneline -1)  host $(hostname)  start $(date -u +%FT%TZ)"
OUT=paper/phase18_results
GATE=paper/submissions/JAC-R2/analysis/denoiser_collapse/csp_gate.py
mkdir -p "$OUT/gate"
if [ -f "$OUT/gate/${RUN}_010000.json" ]; then echo "gate already run"; exit 0; fi
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))" || exit 2
[ -f data/cache/val.npz ] || python scripts/01_simulate_pxrd.py --splits test val train --workers 30 || exit 2
python -c "
import sys; sys.path.insert(0, 'src')
from pxrd_diff.data import CrystalPXRDDataset
for s in ('train', 'val', 'test'):
    print(s, len(CrystalPXRDDataset('data', split=s)))
" 2>&1 | grep -v -i "warn\|electronegativity\|other.X" | tail -3
if [ ! -f runs/$RUN/ckpt_010000.pt ]; then
    BUDGET=$(( 22 - ($(date +%s) - START) / 60 ))
    if [ $BUDGET -ge 4 ]; then
        LAST=$(ls runs/$RUN/ckpt_[0-9]*.pt 2>/dev/null | sort | tail -1)
        RES=""; [ -n "$LAST" ] && RES="--resume $LAST"
        python scripts/14_train_csp.py --run-name $RUN --steps 100000 --bs 256 --workers 8 \
            --stop-at 10000 --max-minutes $BUDGET $RES >> "$OPS/train_$RUN.log" 2>&1
        echo "train slice rc=$? $(grep -a '^step' "$OPS/train_$RUN.log" | tail -1)"
    fi
fi
if [ -f runs/$RUN/ckpt_010000.pt ]; then
    python "$GATE" . runs/$RUN/ckpt_010000.pt runs/$RUN/val.jsonl "$OUT/gate/${RUN}_010000.json"
    rc=$?
    echo "G2 rc=$rc $(date -u +%FT%TZ)"
    exit $rc
fi
echo "not at 10k yet: chaining the next slice"
sbatch paper/submissions/JAC-R2/anvil/g2_debug.sbatch
