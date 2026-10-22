#!/bin/bash
#SBATCH -A che260157-gpu
#SBATCH -p gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus-per-node=1
#SBATCH -t 02:00:00
#SBATCH -J pxrd-v30-eval
#SBATCH --array=0-12
#SBATCH -o /anvil/scratch/x-fcai3/pxrd_ops/%x_%A_%a.out
# JAC R2 redesign: gate G3 (null-pattern sanity anchor) + G4 paper arms, one per array task.
# Submit with --dependency=afterok:<train job>. Idempotent: an arm with output is skipped.
set -u
P=/anvil/projects/x-che260157
export PATH=$P/pxrd_env/bin:$PATH
cd "$P/pxrd-diff"
echo "commit $(git log --oneline -1)  host $(hostname)  task $SLURM_ARRAY_TASK_ID  start $(date -u +%FT%TZ)"
CK=runs/${RUN:-gpu_v30_csp}/ckpt_final.pt
OUT=paper/phase18_results
mkdir -p "$OUT"
IDX=paper/phase9_results/index_benchmark_v2_native.json
UNK=paper/phase15_results/index_cells_test1000_unknown.json
C=(--ckpt $CK --n 1000 --n-samples 20 --refine-steps 200 --refine-lr 1e-3 --pc-steps 1000)
case $SLURM_ARRAY_TASK_ID in
  0)  TAG=v30_null_1s;        ARGS=(--ckpt $CK --n 1000 --n-samples 1 --pc-steps 1000 --null-pattern --seed 0) ;;
  1|2|3)  s=$((SLURM_ARRAY_TASK_ID-1)); TAG=v30_learned_s$s; ARGS=("${C[@]}" --seed $s) ;;
  4|5|6)  s=$((SLURM_ARRAY_TASK_ID-4)); TAG=v30_oracle_s$s;  ARGS=("${C[@]}" --seed $s --true-lattice) ;;
  7|8|9)  s=$((SLURM_ARRAY_TASK_ID-7)); TAG=v30_indexer_s$s; ARGS=("${C[@]}" --seed $s --lat-from-index $IDX) ;;
  10|11|12) s=$((SLURM_ARRAY_TASK_ID-10)); TAG=v30_indexer_unknown_s$s; ARGS=("${C[@]}" --seed $s --lat-from-index $UNK) ;;
esac
if [ -f "$OUT/$TAG.per_sample.jsonl" ]; then echo "$TAG exists, skip"; exit 0; fi
python scripts/03_sample.py "${ARGS[@]}" --out-json "$OUT/$TAG.json" \
    --per-sample-json "$OUT/$TAG.per_sample.jsonl" --save-structures "$OUT/$TAG.structs.jsonl"
rc=$?
echo "$TAG rc=$rc $(date -u +%FT%TZ)"
if [ $SLURM_ARRAY_TASK_ID -eq 0 ] && [ $rc -eq 0 ]; then
    python paper/submissions/JAC-R2/analysis/rematch.py . "$OUT/$TAG.structs.jsonl" \
        --out-json "$OUT/$TAG.diffcsp_match.json"
fi
exit $rc
