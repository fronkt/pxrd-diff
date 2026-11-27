#!/bin/bash
# JAC R2 Phase 19 box setup (1 x RTX 5090, pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime image).
# Usage (once, over ssh): bash p19_setup.sh COMMIT CAP_HOURS
# Clone at COMMIT, deps, MP-20 from CDVAE + simulated patterns (workers sized to the cgroup quota),
# parsed + Niggli caches, G0 tests on the GPU box, then launch p19_pipeline.sh detached.
# A4 (AWS EC2, Deep Learning OSS Nvidia Driver AMI GPU PyTorch, user ubuntu): /workspace is created by
# the launcher; the data and Anvil's checkpoints are uploaded before this runs, so the MP-20 download
# and the simulation are skipped; the AMI's PyTorch venv is activated here and inherited by the pipeline.
if [ -f /opt/pytorch/bin/activate ]; then source /opt/pytorch/bin/activate; fi
set -euo pipefail
COMMIT=${1:?commit}; CAP_H=${2:?cap hours}
mkdir -p /workspace/logs
cd /workspace
[ -d pxrd-diff ] || git clone -q https://github.com/fronkt/pxrd-diff.git
cd pxrd-diff
git fetch -q origin && git checkout -q "$COMMIT"
git log --oneline -1
pip install -q -r requirements.txt
pip install -q -e .
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
Q=$(awk '{ if ($1 == "max") print 24; else printf "%d\n", $1 / $2 }' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 8)
W=$(( Q < 24 ? Q : 24 )); [ "$W" -ge 2 ] || W=2
echo "cgroup cpu quota ~$Q, simulate workers $W"
[ -f data/raw/val.csv ] || python scripts/00_pull_mp20.py
[ -f data/cache/train.npz ] || python scripts/01_simulate_pxrd.py --splits test val train --workers "$W"
md5sum data/raw/*.csv
python - <<'PY'
import hashlib, numpy as np, sys
sys.path.insert(0, "src")
for s in ("train", "val", "test"):
    p = np.load(f"data/cache/{s}.npz", allow_pickle=True)["pattern"]
    print(s, p.shape, "pattern-array md5", hashlib.md5(np.ascontiguousarray(p).tobytes()).hexdigest())
from pxrd_diff.data import CrystalPXRDDataset
for s in ("train", "val", "test"):
    CrystalPXRDDataset("data", split=s, n_peaks=0, niggli=True)   # builds parsed + Niggli caches
print("caches ready")
PY
python -m pytest -q -p no:cacheprovider tests/test_csp_model.py tests/test_train_micro_batch.py
echo "$CAP_H" > /workspace/CAP_H
cp paper/submissions/JAC-R2/box/p19_pipeline.sh paper/submissions/JAC-R2/box/p19_onstart.sh /workspace/
sed -i 's/\r$//' /workspace/p19_pipeline.sh /workspace/p19_onstart.sh
bash -n /workspace/p19_pipeline.sh && bash -n /workspace/p19_onstart.sh
cd /workspace
setsid nohup bash p19_pipeline.sh "$CAP_H" > /workspace/logs/p19.out 2>&1 < /dev/null &
echo "pipeline launched, cap $CAP_H h"
