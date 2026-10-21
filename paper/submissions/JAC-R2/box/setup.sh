#!/bin/bash
# JAC R2 box setup (RTX 5090, pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime image).
# Clone, deps, MP-20 + simulated patterns, v22 checkpoint from Zenodo (md5-checked),
# R2 tests on the box, then launch pipeline.sh detached.
set -euo pipefail
mkdir -p /workspace/logs
cd /workspace
[ -d pxrd-diff ] || git clone -q https://github.com/fronkt/pxrd-diff.git
cd pxrd-diff
git pull -q
git log --oneline -1
pip install -q -r requirements.txt
pip install -q -e .
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
[ -f data/raw/train.csv ] || python scripts/00_pull_mp20.py
[ -f data/cache/train.npz ] || python scripts/01_simulate_pxrd.py --splits test val train --workers 24
mkdir -p runs/gpu_v22_jac
if [ ! -f runs/gpu_v22_jac/ckpt_final.pt ]; then
    curl -sL -o runs/gpu_v22_jac/ckpt_final.pt \
        "https://zenodo.org/records/22973311/files/gpu_v22_jac_ckpt_final.pt?download=1"
fi
echo "89d407d337647429115c1829dddb95e6  runs/gpu_v22_jac/ckpt_final.pt" | md5sum -c -
python -m pytest -q -p no:cacheprovider tests/test_r2_fixes.py
cp paper/submissions/JAC-R2/box/pipeline.sh /workspace/pipeline.sh
sed -i 's/\r$//' /workspace/pipeline.sh
bash -n /workspace/pipeline.sh
cd /workspace
setsid nohup bash pipeline.sh > /workspace/logs/pipeline.out 2>&1 < /dev/null &
echo "pipeline launched"
