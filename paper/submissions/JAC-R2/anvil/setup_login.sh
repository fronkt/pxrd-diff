#!/bin/bash
# JAC R2 redesign (v30, model_family csp): one-time setup on the Anvil login node.
# Light work only: clone, env, MP-20 csv download, CPU import + unit tests.
# The pattern simulation runs inside the GPU job (train_v30.sbatch).
set -euo pipefail
P=/anvil/projects/x-che260157
cd "$P"
[ -d pxrd-diff ] || git clone -q https://github.com/fronkt/pxrd-diff.git
cd pxrd-diff
git pull -q
git log --oneline -1
ENV=$P/pxrd_env
[ -x "$ENV/bin/python" ] || "$P/n3_conda/bin/conda" create -y -q -p "$ENV" python=3.11
export PIP_CACHE_DIR=/anvil/scratch/x-fcai3/pipcache
"$ENV/bin/python" -c "import torch" 2>/dev/null || \
    "$ENV/bin/pip" install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
"$ENV/bin/pip" install -q -r requirements.txt
"$ENV/bin/pip" install -q -e .
"$ENV/bin/python" -c "import torch, pymatgen, pxrd_diff; print('torch', torch.__version__, 'cuda build', torch.version.cuda)"
[ -f data/raw/train.csv ] || "$ENV/bin/python" scripts/00_pull_mp20.py
"$ENV/bin/python" -m pytest -q -p no:cacheprovider tests/test_csp_model.py
mkdir -p /anvil/scratch/x-fcai3/pxrd_ops
echo SETUP_OK
