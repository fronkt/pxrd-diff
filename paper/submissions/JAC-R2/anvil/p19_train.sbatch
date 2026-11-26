#!/bin/bash
#SBATCH -A che260157-gpu
#SBATCH -p gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus-per-node=1
#SBATCH -t 01:50:00
#SBATCH -J p19-train
#SBATCH -o /anvil/scratch/x-fcai3/pxrd_ops/p19/%x_%j.out
# JAC R2 Phase 19 on Anvil (amendment A1 in tasks/todo.md; otherwise the Phase 19 spec, unchanged).
# Self-chaining 1-GPU jobs of <= 1:50, the size Slurm backfill started 33 times on 11-18..11-23
# (sbatch --test-only says 2027-01-29 for any gpu job). Each job does, skipping finished steps:
#   0. Niggli caches + the B1 Niggli check on Anvil's data copies (check_niggli.py exits non-zero on
#      any mismatch; a failure stops the chain);
#   1. width rule (Phase 19): 300-step benchmark at d 512, else 384, else 256; the first whose
#      100k-step projection on this A100 is <= 16 h wins, else 256. Then a sampling benchmark at that
#      width (100 structures x 20 samples, 50 PC steps) that sizes the 20-sample G4 chunks
#      (chunk20.txt, <= 80 min each) and projects the G4 GPU-hours;
#   2. trains gpu_p19 for the job's remaining wall time (--max-minutes) from the latest checkpoint;
#   3. at ckpt_final.pt: cancels its successor and submits the G3 array (p19_arms.sbatch G3).
# It queues its successor (afterany) at start. The chain stops at ckpt_final, at $OPS/STOP, after
# MAX_JOBS jobs, or after 2 jobs in a row without progress.
set -u
P=/anvil/projects/x-che260157
R=/anvil/scratch/x-fcai3/pxrd-diff-p19
OPS=/anvil/scratch/x-fcai3/pxrd_ops/p19
export PATH=$P/pxrd_env/bin:$PATH PYTHONPATH=$R/src OMP_NUM_THREADS=16
cd "$R" || exit 2
OUT=paper/phase19_results
RUN=gpu_p19
CK=runs/$RUN/ckpt_final.pt
STATUS=$OPS/p19.status
ME=paper/submissions/JAC-R2/anvil/p19_train.sbatch
ARMS_SB=paper/submissions/JAC-R2/anvil/p19_arms.sbatch
MAX_JOBS=40
END=$(( $(date +%s) + 106 * 60 ))
mkdir -p "$OUT" "$OPS"
log() { echo "$(date -u +%FT%TZ) [train $SLURM_JOB_ID] $*" | tee -a "$STATUS" >&2; }   # not stdout: $(bench_hours)
left_min() { echo $(( (END - $(date +%s)) / 60 )); }

log "start host $(hostname) commit $(git log --oneline -1 | cut -c1-12)"
[ -f "$CK" ] && { log "ckpt_final exists: nothing to do"; exit 0; }
[ -f "$OPS/STOP" ] && { log "STOP file: chain stopped"; exit 0; }
njobs=$(( $(cat "$OPS/train_njobs" 2>/dev/null || echo 0) + 1 )); echo "$njobs" > "$OPS/train_njobs"
nop=$(cat "$OPS/train_noprogress" 2>/dev/null || echo 0)
if [ "$njobs" -gt $MAX_JOBS ] || [ "$nop" -ge 2 ]; then
    log "chain stopped: jobs=$njobs noprogress=$nop"; exit 0
fi
SUCC=$(sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" "$ME") || SUCC=
log "successor ${SUCC:-NOT QUEUED}"
progress() {
    echo "$([ -f $OUT/check_niggli_anvil.ok ] && echo n)$(cat $OUT/width.txt 2>/dev/null)$(cat $OUT/chunk20.txt 2>/dev/null)" \
         "$(ls runs/$RUN/ckpt_[0-9]*.pt 2>/dev/null | sort | tail -1)$([ -f $CK ] && echo F)"
}
P0=$(progress)
on_exit() {
    local p1; p1=$(progress)
    if [ "$p1" = "$P0" ]; then echo $((nop + 1)) > "$OPS/train_noprogress"; else echo 0 > "$OPS/train_noprogress"; fi
    log "end, progress [$P0] -> [$p1]"
}
trap on_exit EXIT
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))" \
    || { log "no usable GPU"; exit 2; }

# ---- 0. Niggli caches + check -----------------------------------------------------------------
if [ ! -f "$OUT/check_niggli_anvil.ok" ]; then
    python -u paper/submissions/JAC-R2/analysis/g3_diag/check_niggli.py . --splits train val test \
        --workers 30 > "$OUT/check_niggli_anvil.log" 2>&1
    rc=$?
    grep -E '^(train|val|test) ' "$OUT/check_niggli_anvil.log" >> "$STATUS"
    if [ $rc -ne 0 ]; then
        log "NIGGLI CHECK FAILED rc=$rc (see $OUT/check_niggli_anvil.log): chain stopped"
        touch "$OPS/STOP"; [ -n "$SUCC" ] && scancel "$SUCC"; exit 3
    fi
    touch "$OUT/check_niggli_anvil.ok"; log "Niggli check ok"
fi

# ---- 1. width rule + sampling benchmark --------------------------------------------------------
TRAIN_COMMON=(--steps 100000 --bs 256 --n-layers 6 --n-heads 4 --n-freq 128 --p-pattern-drop 0.3
              --niggli --workers 8)
bench_hours() {   # $1 = d_model; prints the projected hours for 100k steps (steps 100 -> 300)
    local d=$1 lg=$OPS/bench_d$1.log
    if [ ! -f "$OUT/bench_d$d.hours" ]; then
        rm -rf "runs/bench_d$d"
        python -u scripts/14_train_csp.py --run-name "bench_d$d" "${TRAIN_COMMON[@]}" --d-model "$d" \
            --stop-at 300 --log-every 50 --val-every 1000000 --save-every 1000000 > "$lg" 2>&1
        grep -q "^Stopped at step 300" "$lg" && [ -f "runs/bench_d$d/ckpt_000300.pt" ] || return 1
        awk '/^step +100 /{for(i=1;i<=NF;i++) if($i ~ /^t=/){sub("t=","",$i); sub("s","",$i); a=$i+0}}
             /^step +300 /{for(i=1;i<=NF;i++) if($i ~ /^t=/){sub("t=","",$i); sub("s","",$i); b=$i+0}}
             END{ if (b > a) printf "%.2f\n", (b - a) / 200 * 100000 / 3600; else print "999" }' \
            "$lg" > "$OUT/bench_d$d.hours"
        log "bench d$d: $(cat "$OUT/bench_d$d.hours") h per 100k steps"
    fi
    cat "$OUT/bench_d$d.hours"
}
fits() { awk -v h="$1" 'BEGIN{exit !(h <= 16.0)}'; }
if [ ! -f "$OUT/width.txt" ]; then
    D=
    for d in 512 384 256; do
        h=$(bench_hours $d) || { log "bench d$d FAILED (see $OPS/bench_d$d.log)"; exit 4; }
        if [ $d = 256 ] || fits "$h"; then D=$d; break; fi
    done
    echo "$D" > "$OUT/width.txt"
    log "width rule -> d_model $D"
fi
D=$(cat "$OUT/width.txt")
if [ ! -f "$OUT/chunk20.txt" ]; then
    python -u scripts/03_sample.py --ckpt "runs/bench_d$D/ckpt_000300.pt" --n 100 --start 0 \
        --n-samples 20 --pc-steps 50 --refine-steps 0 --amp bf16 --bucket --batch-size 50 --seed 0 \
        --out-json "$OUT/bench_sample_d$D.json" > "$OPS/bench_sample_d$D.log" 2>&1 \
        || { log "sampling bench FAILED (see $OPS/bench_sample_d$D.log)"; exit 5; }
    tot=$(grep -h '^Total:' "$OPS/bench_sample_d$D.log")
    python - "$tot" "$OUT/chunk20.txt" <<'PY' | while read -r l; do log "$l"; done
import re, sys
tot, path = sys.argv[1], sys.argv[2]
s = float(re.search(r"sample=([\d.]+)s", tot).group(1))
sel = float(re.search(r"select=([\d.]+)s", tot).group(1))
t100 = 20 * s + sel + 120        # s per 100-structure x 20-sample chunk at 1000 PC steps (+ load, match)
c = next((c for c in (100, 50, 25, 20, 10) if t100 * c / 100 <= 80 * 60), 10)
open(path, "w").write(f"{c}\n")
print(f"bench sample ({tot}) -> {t100 / 60:.0f} min per 100 x 20 chunk, chunk20 {c}; "
      f"12 twenty-sample G4 arms ~{12 * 10 * t100 / 3600:.0f} GPU-h (+ 3 null arms)")
PY
fi

# ---- 2. training slice -------------------------------------------------------------------------
if [ ! -f "$CK" ]; then
    LAST=$(ls runs/$RUN/ckpt_[0-9]*.pt 2>/dev/null | sort | tail -1)
    BUDGET=$(( $(left_min) - 8 ))    # data loading precedes the timer; the last save follows it
    if [ $BUDGET -ge 10 ]; then
        log "train d$D budget $BUDGET min ${LAST:+resume $LAST}"
        python -u scripts/14_train_csp.py --run-name $RUN "${TRAIN_COMMON[@]}" --d-model "$D" \
            --save-every 5000 --val-every 2000 --max-minutes "$BUDGET" ${LAST:+--resume "$LAST"} \
            >> "$OPS/train_$RUN.log" 2>&1
        log "train slice rc=$? $(grep -a '^step' "$OPS/train_$RUN.log" | tail -1)"
    fi
fi

# ---- 3. done: G3 -------------------------------------------------------------------------------
if [ -f "$CK" ]; then
    md5sum "$CK" > "$OUT/ckpt_final.md5"
    [ -n "$SUCC" ] && scancel "$SUCC"
    log "TRAIN DONE, final val: $(tail -1 runs/$RUN/val.jsonl)"
    G3=$(sbatch --parsable --array=0-9 "$ARMS_SB" G3) && log "G3 array $G3 submitted" \
        || log "G3 SUBMIT FAILED"
fi
