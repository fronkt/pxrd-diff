export PYTHONIOENCODING=utf-8
S=/c/Users/frank/AppData/Local/Temp/claude/C--Users-frank/4cecb8ad-5260-46ef-a6e3-8da63c126f63/scratchpad
CK=C:/Users/frank/pxrd-diff-pub/runs/gpu_v22_jac/ckpt_final.pt
for mode in ${MODES:-real null}; do
  python -X utf8 -I $S/nulltest/run_null.py $mode --ckpt $CK --n 40 --n-samples 20 --ensemble-eta 0.0 --refine-steps 200 \
    --refine-lr 1e-3 --ddim-steps 50 --true-lattice --seed 0 --batch-size 8 \
    --out-json $S/nulltest/$mode.json --per-sample-json $S/nulltest/$mode.jsonl > $S/nulltest/$mode.log 2>&1
  echo "$mode rc=$?" >> $S/nulltest/status
done
