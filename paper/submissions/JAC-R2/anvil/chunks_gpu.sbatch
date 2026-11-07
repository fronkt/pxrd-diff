#!/bin/bash
#SBATCH -A che260157-gpu
#SBATCH -p gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus-per-node=1
#SBATCH -t 04:00:00
#SBATCH -J pxrd-v30-ch
#SBATCH --array=0-119%8
#SBATCH -o /anvil/scratch/x-fcai3/pxrd_ops/%x_%A_%a.out
# JAC R2: the 12 remaining eval arms of chain_debug.sbatch (part C), as one gpu-partition array task
# per (arm, 100-structure chunk), because a learned-type chunk (20 samples x 1000 PC steps + 200
# refine steps) needs more than the 26.5 min a gpu-debug slice leaves: v30_learned_s0 c0/c100 timed
# out at ~1590 s in three slices (2026-11-05/06) and the chain stopped itself (noprogress=2).
# The command per chunk is chain_debug.sbatch's, unchanged (same args_for, --n/--start, outputs and
# .done markers in paper/phase18_results/chunks/), except python -u (unbuffered log) and no
# slice timeout (Slurm's -t applies). The null arm is already merged. Task id -> arm ARMS[1 + id/10],
# chunk start (id % 10) * 100; seed-0 arms first. When the last chunk of an arm finishes, that task
# merges the arm (merge_chunks.py, under a mkdir lock), and the task that completes all 13 arms
# writes ALL_DONE, as part D of the chain does. DRY=1 prints the task's command and exits.
set -u
P=/anvil/projects/x-che260157
export PATH=$P/pxrd_env/bin:$PATH
export OMP_NUM_THREADS=16
RUN=gpu_v30_csp
CHUNK=100; N=1000
cd "$P/pxrd-diff"
OUT=paper/phase18_results; CH=$OUT/chunks
MERGE=paper/submissions/JAC-R2/anvil/merge_chunks.py
CK=runs/$RUN/ckpt_final.pt
IDX=paper/phase9_results/index_benchmark_v2_native.json
UNK=paper/phase15_results/index_cells_test1000_unknown.json
NULLM=$OUT/v30_null_1s.diffcsp_match.json
ARMS=(v30_null_1s v30_learned_s0 v30_oracle_s0 v30_indexer_s0 v30_indexer_unknown_s0
      v30_learned_s1 v30_oracle_s1 v30_indexer_s1 v30_indexer_unknown_s1
      v30_learned_s2 v30_oracle_s2 v30_indexer_s2 v30_indexer_unknown_s2)
args_for() {   # identical to chain_debug.sbatch
    local C=(--ckpt $CK --n-samples 20 --refine-steps 200 --refine-lr 1e-3 --pc-steps 1000)
    case $1 in
        v30_null_1s) echo --ckpt $CK --n-samples 1 --pc-steps 1000 --null-pattern --seed 0 ;;
        v30_learned_s*) echo "${C[@]}" --seed ${1##*_s} ;;
        v30_oracle_s*) echo "${C[@]}" --seed ${1##*_s} --true-lattice ;;
        v30_indexer_unknown_s*) echo "${C[@]}" --seed ${1##*_s} --lat-from-index $UNK ;;
        v30_indexer_s*) echo "${C[@]}" --seed ${1##*_s} --lat-from-index $IDX ;;
    esac
}
id=${SLURM_ARRAY_TASK_ID:?array task id}
arm=${ARMS[$(( 1 + id / 10 ))]}; st=$(( (id % 10) * CHUNK ))
base=$CH/$arm.c$(printf %04d $st)
cmd=(python -u scripts/03_sample.py $(args_for $arm) --n $CHUNK --start $st --out-json $base.json
     --per-sample-json $base.per_sample.jsonl --save-structures $base.structs.jsonl)
echo "commit $(git log --oneline -1 | cut -c1-60)  host $(hostname)  job ${SLURM_ARRAY_JOB_ID:-?}_$id  $arm c$st  start $(date -u +%FT%TZ)"
if [ "${DRY:-0}" = 1 ]; then echo "DRY: ${cmd[*]}"; exit 0; fi
[ -f $CK ] || { echo "no $CK"; exit 1; }
if [ -f $OUT/$arm.per_sample.jsonl ] || [ -f $base.done ]; then
    echo "$arm c$st already done"
else
    t1=$(date +%s)
    CUDA_VISIBLE_DEVICES=0 "${cmd[@]}" > $base.log 2>&1
    rc=$?; dt=$(( $(date +%s) - t1 ))
    if [ $rc -eq 0 ] && [ -s $base.per_sample.jsonl ]; then
        touch $base.done; echo "$arm $st $dt" >> "$CH/durations.txt"; echo "$arm c$st ok ${dt}s"
    else
        echo "$arm $st rc=$rc ${dt}s job ${SLURM_ARRAY_JOB_ID:-?}_$id" >> "$CH/failures.txt"
        echo "$arm c$st FAILED rc=$rc after ${dt}s (see $base.log)"; exit 1
    fi
fi
# merge this arm once all its chunks are done (one merger at a time)
if [ ! -f $OUT/$arm.per_sample.jsonl ] && [ $(ls $CH/$arm.c*.done 2>/dev/null | wc -l) -eq $(( N / CHUNK )) ] \
        && mkdir $CH/$arm.merge.lock 2>/dev/null; then
    [ -f $OUT/$arm.per_sample.jsonl ] || python "$MERGE" $OUT $arm $N $CHUNK
    echo "merge $arm rc=$?"; rmdir $CH/$arm.merge.lock
fi
if [ $(ls $OUT/v30_*.per_sample.jsonl 2>/dev/null | wc -l) -eq ${#ARMS[@]} ] && [ -f $NULLM ] && [ ! -f $OUT/ALL_DONE ]; then
    date -u +%FT%TZ > $OUT/ALL_DONE; echo "ALL_DONE"
fi
echo "end $(date -u +%FT%TZ)"
