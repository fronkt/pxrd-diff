#!/bin/bash
# JAC R2 Phase 19 on Anvil (amendment A1, tasks/todo.md): one-time login-node setup. Light work only.
# Usage: bash p19_setup_login.sh COMMIT
# Clones the repo at COMMIT into SCRATCH (the project fileset refuses new files since 2026-11-18),
# copies the MP-20 csv files, simulated patterns and parsed-CIF caches from the v30 checkout on the
# project fileset (md5-checked), checks that the v30 env imports this checkout, runs the G0 model
# tests on CPU. The Niggli caches and the Niggli check run in the first GPU job (32 cores).
set -euo pipefail
COMMIT=${1:?usage: p19_setup_login.sh COMMIT}
P=/anvil/projects/x-che260157
SRC=$P/pxrd-diff
R=/anvil/scratch/x-fcai3/pxrd-diff-p19
OPS=/anvil/scratch/x-fcai3/pxrd_ops/p19
export PATH=$P/pxrd_env/bin:$PATH PYTHONPATH=$R/src
mkdir -p "$OPS"
[ -d "$R" ] || git clone -q https://github.com/fronkt/pxrd-diff.git "$R"
cd "$R"
git fetch -q origin && git checkout -q "$COMMIT"
git log --oneline -1
mkdir -p data/raw data/cache
FILES="raw/train.csv raw/val.csv raw/test.csv cache/train.npz cache/val.npz cache/test.npz
       cache/two_theta.npy cache/train_parsed.pkl cache/val_parsed.pkl cache/test_parsed.pkl"
for f in $FILES; do [ -f "data/$f" ] || cp -p "$SRC/data/$f" "data/$f"; done
(cd "$SRC" && for f in $FILES; do md5sum "data/$f"; done) > "$OPS/data_src.md5"
md5sum -c --quiet "$OPS/data_src.md5"
echo "data copied, md5 identical to $SRC"
python - "$R" <<'PY'
import sys, torch, pymatgen.core, pxrd_diff
assert pxrd_diff.__file__.startswith(sys.argv[1]), f"pxrd_diff imported from {pxrd_diff.__file__}"
print("torch", torch.__version__, "cuda build", torch.version.cuda, "pxrd_diff from", pxrd_diff.__file__)
PY
python -m pytest -q -p no:cacheprovider tests/test_csp_model.py
for f in p19_train.sbatch p19_arms.sbatch; do bash -n "paper/submissions/JAC-R2/anvil/$f"; done
! grep -l $'\r' paper/submissions/JAC-R2/anvil/p19_*.sbatch
echo SETUP_OK
