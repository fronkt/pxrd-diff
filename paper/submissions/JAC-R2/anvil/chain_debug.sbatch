#!/bin/bash
#SBATCH -A che260157-gpu
#SBATCH -p gpu-debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus-per-node=2
#SBATCH -t 00:30:00
#SBATCH -J pxrd-v30-chain
#SBATCH -o /anvil/scratch/x-fcai3/pxrd_ops/%x_%j.out
# JAC R2 redesign, run on gpu-debug because on 2026-11-02 the gpu queue estimated a start of
# Nov 24 (deadline Nov 18; MATH outage Nov 6-9). gpu-debug QoS: 30 min, <= 2 GPUs, 1 running
# and 2 submitted jobs per user. Each job of this self-chaining script does, in order:
#   A. continue training gpu_v30_csp from its latest checkpoint (GPU 0, --max-minutes slice;
#      same arguments as train_v30.sbatch, which timed out at step 95,400 on 2026-10-23);
#   B. once ckpt_final.pt exists: csp_gate.py on it (informational, as in train_v30.sbatch);
#   C. the 13 eval_v30.sbatch arms with the same arguments, split into CHUNK-structure chunks
#      (03_sample.py --start/--n), two workers (one per GPU), each chunk an idempotent item
#      marked .done; arms ordered null, then seed 0 of every arm, then seeds 1 and 2;
#   D. merge each finished arm into the eval_v30 file names (merge_chunks.py) and run the
#      task-0 DiffCSP rematch, so analysis/r2_stats.py and G3 read the usual files.
# It queues its successor at START. Chaining stops when everything is merged ($OUT/ALL_DONE),
# when $OPS/CHAIN_STOP exists, after MAX_JOBS jobs, or after 2 jobs in a row without progress.
# A chunk that fails twice is skipped (listed in chunks/failures.txt) and its arm is not merged.
set -u
P=/anvil/projects/x-che260157
export PATH=$P/pxrd_env/bin:$PATH
export OMP_NUM_THREADS=16
OPS=/anvil/scratch/x-fcai3/pxrd_ops
RUN=gpu_v30_csp
CHUNK=100; N=1000; MAX_JOBS=80
T0=$(date +%s); END=$((T0 + 27 * 60))
cd "$P/pxrd-diff"
OUT=paper/phase18_results; CH=$OUT/chunks
mkdir -p "$CH" "$OUT/gate"
GATE=paper/submissions/JAC-R2/analysis/denoiser_collapse/csp_gate.py
ME=paper/submissions/JAC-R2/anvil/chain_debug.sbatch
MERGE=paper/submissions/JAC-R2/anvil/merge_chunks.py
CK=runs/$RUN/ckpt_final.pt
IDX=paper/phase9_results/index_benchmark_v2_native.json
UNK=paper/phase15_results/index_cells_test1000_unknown.json
echo "commit $(git log --oneline -1)  host $(hostname)  job $SLURM_JOB_ID  start $(date -u +%FT%TZ)"

[ -f "$OUT/ALL_DONE" ] && { echo "ALL_DONE exists: nothing to do"; exit 0; }
if squeue -h -u "$USER" -n pxrd-v30-chain -t R -o %i | grep -vqx "$SLURM_JOB_ID"; then
    echo "another chain job is running: exiting"; exit 0
fi
njobs=$(( $(cat "$OPS/chain_njobs" 2>/dev/null || echo 0) + 1 )); echo $njobs > "$OPS/chain_njobs"
nop=$(cat "$OPS/chain_noprogress" 2>/dev/null || echo 0)
if [ -f "$OPS/CHAIN_STOP" ] || [ $njobs -gt $MAX_JOBS ] || [ $nop -ge 2 ]; then
    echo "chain stopped: stopfile=$([ -f "$OPS/CHAIN_STOP" ] && echo y || echo n) jobs=$njobs noprogress=$nop"
    exit 0
fi
sbatch "$ME"                                  # successor; runs after this one (1 running/user)
rm -rf "$CH"/*.lock                           # only one chain job runs, so any lock is stale

ARMS=(v30_null_1s v30_learned_s0 v30_oracle_s0 v30_indexer_s0 v30_indexer_unknown_s0
      v30_learned_s1 v30_oracle_s1 v30_indexer_s1 v30_indexer_unknown_s1
      v30_learned_s2 v30_oracle_s2 v30_indexer_s2 v30_indexer_unknown_s2)
args_for() {   # eval_v30.sbatch arguments minus --n (chunking supplies --n/--start)
    local C=(--ckpt $CK --n-samples 20 --refine-steps 200 --refine-lr 1e-3 --pc-steps 1000)
    case $1 in
        v30_null_1s) echo --ckpt $CK --n-samples 1 --pc-steps 1000 --null-pattern --seed 0 ;;
        v30_learned_s*) echo "${C[@]}" --seed ${1##*_s} ;;
        v30_oracle_s*) echo "${C[@]}" --seed ${1##*_s} --true-lattice ;;
        v30_indexer_unknown_s*) echo "${C[@]}" --seed ${1##*_s} --lat-from-index $UNK ;;
        v30_indexer_s*) echo "${C[@]}" --seed ${1##*_s} --lat-from-index $IDX ;;
    esac
}
progress() {
    echo "$(ls "$CH"/*.done 2>/dev/null | wc -l) $(ls $OUT/v30_*.per_sample.jsonl 2>/dev/null | wc -l)" \
         "$(ls runs/$RUN/ckpt_*.pt 2>/dev/null | wc -l) $([ -f $OUT/gate/${RUN}_final.json ] && echo g)" \
         "$([ -f $OUT/v30_null_1s.diffcsp_match.json ] && echo r)"
}
P0=$(progress)

# ---- A. training
if [ ! -f $CK ]; then
    LAST=$(ls runs/$RUN/ckpt_[0-9]*.pt 2>/dev/null | sort | tail -1)
    BUDGET=$(( (END - $(date +%s)) / 60 - 4 ))   # data loading ~2 min precedes the timer
    if [ $BUDGET -ge 5 ]; then
        CUDA_VISIBLE_DEVICES=0 python scripts/14_train_csp.py --run-name $RUN --steps 100000 --bs 256 \
            --workers 8 --max-minutes $BUDGET ${LAST:+--resume $LAST} >> "$OPS/train_$RUN.log" 2>&1
        echo "train slice rc=$? $(grep -a '^step' "$OPS/train_$RUN.log" | tail -1)"
    fi
fi

# ---- B. gate on the final checkpoint
if [ -f $CK ] && [ ! -f $OUT/gate/${RUN}_final.json ] && [ $(( END - $(date +%s) )) -ge 300 ]; then
    timeout $(( END - $(date +%s) - 60 ))s python "$GATE" . $CK runs/$RUN/val.jsonl $OUT/gate/${RUN}_final.json
    echo "final gate rc=$? $(date -u +%FT%TZ)"
fi

# ---- C. eval chunks
worker() {
    local g=$1 arm st base left est nf t1 rc dt
    for arm in "${ARMS[@]}"; do
        [ -f $OUT/$arm.per_sample.jsonl ] && continue
        for st in $(seq 0 $CHUNK $((N - 1))); do
            base=$CH/$arm.c$(printf %04d $st)
            [ -f $base.done ] && continue
            nf=$(grep -c "^$arm $st " "$CH/failures.txt" 2>/dev/null); [ "${nf:-0}" -ge 2 ] && continue
            mkdir $base.lock 2>/dev/null || continue
            left=$(( END - $(date +%s) ))
            est=$(awk -v k="${arm%%_s*}" '$1 ~ "^"k {if ($3 > m) m = $3} END {print m + 0}' \
                  "$CH/durations.txt" 2>/dev/null); est=${est:-0}
            if [ $left -lt 240 ] || [ $left -lt $(( est + est / 5 + 60 )) ]; then rmdir $base.lock; continue; fi
            t1=$(date +%s)
            CUDA_VISIBLE_DEVICES=$g timeout $(( left - 30 ))s python scripts/03_sample.py $(args_for $arm) \
                --n $CHUNK --start $st --out-json $base.json --per-sample-json $base.per_sample.jsonl \
                --save-structures $base.structs.jsonl > $base.log 2>&1
            rc=$?; dt=$(( $(date +%s) - t1 ))
            if [ $rc -eq 0 ] && [ -s $base.per_sample.jsonl ]; then
                touch $base.done; echo "$arm $st $dt" >> "$CH/durations.txt"
                echo "gpu$g $arm c$st ok ${dt}s"
            elif [ $rc -eq 124 ]; then             # cut off by the slice: retried, never skipped
                echo "$arm $st timeout ${dt}s job $SLURM_JOB_ID" >> "$CH/timeouts.txt"
                echo "gpu$g $arm c$st TIMEOUT after ${dt}s"
            else
                echo "$arm $st rc=$rc ${dt}s job $SLURM_JOB_ID" >> "$CH/failures.txt"
                echo "gpu$g $arm c$st FAILED rc=$rc after ${dt}s (see $base.log)"
            fi
            rmdir $base.lock
        done
    done
}
# ---- D. merge finished arms; G3 rematch on the null arm (CPU, in the background while the
#      GPU workers run). Called before and after the workers.
NULLM=$OUT/v30_null_1s.diffcsp_match.json
merge_ready() {
    for arm in "${ARMS[@]}"; do
        [ -f $OUT/$arm.per_sample.jsonl ] && continue
        [ $(ls $CH/$arm.c*.done 2>/dev/null | wc -l) -eq $(( N / CHUNK )) ] || continue
        python "$MERGE" $OUT $arm $N $CHUNK
    done
}
g3_rematch() {
    [ -f $OUT/v30_null_1s.structs.jsonl ] && [ ! -f $NULLM ] && [ $(( END - $(date +%s) )) -ge 180 ] || return 0
    timeout $(( END - $(date +%s) - 30 ))s python paper/submissions/JAC-R2/analysis/rematch.py . \
        $OUT/v30_null_1s.structs.jsonl --out-json $NULLM > $OUT/v30_null_1s.diffcsp_match.log 2>&1
    echo "G3 rematch rc=$? $(date -u +%FT%TZ)"
}
merge_ready
if [ -f $CK ]; then
    g3_rematch & worker 0 & worker 1 & wait
fi
merge_ready
g3_rematch
if [ $(ls $OUT/v30_*.per_sample.jsonl 2>/dev/null | wc -l) -eq ${#ARMS[@]} ] && [ -f $NULLM ]; then
    date -u +%FT%TZ > $OUT/ALL_DONE
    scancel -u "$USER" -n pxrd-v30-chain -t PD
    echo "ALL_DONE"
fi

P1=$(progress)
if [ "$P1" = "$P0" ]; then echo $((nop + 1)) > "$OPS/chain_noprogress"; else echo 0 > "$OPS/chain_noprogress"; fi
echo "progress [$P0] -> [$P1]  end $(date -u +%FT%TZ)"
