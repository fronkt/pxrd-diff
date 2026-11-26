#!/bin/bash
#SBATCH -A che260157-gpu
#SBATCH -p gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus-per-node=1
#SBATCH -t 01:50:00
#SBATCH -J p19-work
#SBATCH -o /anvil/scratch/x-fcai3/pxrd_ops/p19/%x_%A_%a.out
# JAC R2 Phase 19 on Anvil (amendments A1 + A2 in tasks/todo.md; otherwise the Phase 19 spec, unchanged).
# A2: Anvil ranks pending jobs by time waited (PriorityWeightAge 20000, max age 7 days, fair-share weight
# 0), so a self-chaining job restarted at zero priority with every link (the A1 chain's first job was
# estimated to start 12-02). Instead this is one array of identical 1-GPU <= 1:50 workers submitted
# together, so all of them age; each worker does the next piece of work, skipping finished steps:
#   0. the B1 Niggli check on Anvil's data copies (any mismatch stops the run);
#   1. width rule (300-step benchmark at d 512, else 384, else 256; the first whose 100k-step projection
#      on this A100 is <= 16 h wins, else 256), then the sampling benchmark at that width (100 structures
#      x 20 samples, 50 PC steps) that sizes the 20-sample G4 chunks (chunk20.txt, <= 80 min each),
#      estimates chunk minutes (chunk_min.txt) and projects the G4 GPU-hours;
#   2. a training slice of gpu_p19 from the latest checkpoint for the time left (--max-minutes);
#   3. once ckpt_final.pt exists: unclaimed G3 chunks (p19_arms.sbatch G3 ID) while the time left covers
#      one; G4 chunks the same way only if G3 passed and $OPS/G4_ON_ANVIL exists.
# Steps 0-2 run under one lock ($OPS/train.lock); a worker that finds it held exits at once. Locks and
# chunk claims are mkdir directories holding the holder's job id (flock is not cluster-wide on Anvil's
# GPFS: tested 11-26); a lock whose holder is no longer RUNNING (sacct) is taken over.
# BENCH_ONLY=1 stops after step 1 (for gpu-debug: --export=ALL,BENCH_ONLY=1,WALL_MIN=30 -p gpu-debug
# -t 00:30:00). WALL_MIN = the job's time limit in minutes (default 110).
# Stops: $OPS/STOP (a Niggli mismatch, a failed chunk, or 2 training slices in a row without progress)
# or no work left; the worker then cancels its array's pending tasks.
set -u
P=${P19_P:-/anvil/projects/x-che260157}   # P19_P/R/OPS: overrides for the mock test only
R=${P19_R:-/anvil/scratch/x-fcai3/pxrd-diff-p19}
OPS=${P19_OPS:-/anvil/scratch/x-fcai3/pxrd_ops/p19}
export PATH=$P/pxrd_env/bin:$PATH PYTHONPATH=$R/src OMP_NUM_THREADS=16
cd "$R" || exit 2
OUT=paper/phase19_results; CH=$OUT/chunks; CLAIMS=$CH/claims
RUN=gpu_p19
CK=runs/$RUN/ckpt_final.pt
STATUS=$OPS/p19.status
ARMS_SB=paper/submissions/JAC-R2/anvil/p19_arms.sbatch
END=$(( $(date +%s) + (${WALL_MIN:-110} - 4) * 60 ))
JOB=${SLURM_ARRAY_JOB_ID:+${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}; JOB=${JOB:-${SLURM_JOB_ID:-local$$}}
mkdir -p "$OUT" "$OPS" "$CLAIMS"
log() { echo "$(date -u +%FT%TZ) [work $JOB] $*" | tee -a "$STATUS" >&2; }   # not stdout: $(bench_hours)
left_min() { echo $(( (END - $(date +%s)) / 60 )); }
stop_pending() {   # no work left for this array: cancel its pending tasks
    [ -n "${SLURM_ARRAY_JOB_ID:-}" ] && scancel -t PENDING "$SLURM_ARRAY_JOB_ID" \
        && log "cancelled the pending tasks of $SLURM_ARRAY_JOB_ID ($*)"
    return 0
}
halt() { log "$1: STOP"; touch "$OPS/STOP"; stop_pending "$1"; exit "${2:-1}"; }
take() {   # $1 = lock dir; 0 = this job holds it now
    local L=$1 j s
    mkdir "$L" 2>/dev/null && { echo "$JOB" > "$L/job"; return 0; }
    j=$(cat "$L/job" 2>/dev/null); [ -n "$j" ] || return 1
    s=$(sacct -n -P -X -j "$j" -o State 2>/dev/null | head -1)
    [ -n "$s" ] && [ "$s" != RUNNING ] || return 1          # held by a running job, or sacct unsure
    find "$L.steal" -maxdepth 0 -mmin +10 -exec rmdir {} \; 2>/dev/null
    mkdir "$L.steal" 2>/dev/null || return 1                 # one taker at a time
    if [ "$(cat "$L/job" 2>/dev/null)" = "$j" ]; then
        rm -rf "$L"; mkdir "$L" 2>/dev/null && echo "$JOB" > "$L/job"
        log "took over $L from $j ($s)"
    fi
    rmdir "$L.steal"
    [ "$(cat "$L/job" 2>/dev/null)" = "$JOB" ]
}
release() { [ "$(cat "$1/job" 2>/dev/null)" = "$JOB" ] && rm -rf "$1"; return 0; }

log "start host $(hostname) commit $(git log --oneline -1 | cut -c1-12) wall ${WALL_MIN:-110} min"
[ -f "$OPS/STOP" ] && { log "STOP file: exit"; stop_pending "STOP file"; exit 0; }
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))" \
    || { log "no usable GPU"; exit 2; }

# ===== steps 0-2 under the training lock ==========================================================
if [ ! -f "$CK" ]; then
    take "$OPS/train.lock" || { log "another worker holds the training lock: exit"; exit 0; }
    nop=$(cat "$OPS/train_noprogress" 2>/dev/null || echo 0)
    [ "$nop" -ge 2 ] && { release "$OPS/train.lock"; halt "2 lock-holding workers in a row made no progress" 0; }
    progress() {
        echo "$([ -f $OUT/check_niggli_anvil.ok ] && echo n)$(ls $OUT/bench_d*.hours 2>/dev/null | wc -l)" \
             "$(cat $OUT/width.txt 2>/dev/null)$(cat $OUT/chunk20.txt 2>/dev/null)" \
             "$(ls runs/$RUN/ckpt_[0-9]*.pt 2>/dev/null | sort | tail -1)$([ -f $CK ] && echo F)"
    }
    P0=$(progress)
    on_exit() {
        local p1; p1=$(progress)
        if [ "$p1" = "$P0" ]; then echo $((nop + 1)) > "$OPS/train_noprogress"; else echo 0 > "$OPS/train_noprogress"; fi
        log "lock released, progress [$P0] -> [$p1]"; release "$OPS/train.lock"
    }
    trap on_exit EXIT

    # ---- 0. Niggli check ---------------------------------------------------------------------
    if [ ! -f "$OUT/check_niggli_anvil.ok" ]; then
        python -u paper/submissions/JAC-R2/analysis/g3_diag/check_niggli.py . --splits train val test \
            --workers 30 > "$OUT/check_niggli_anvil.log" 2>&1
        rc=$?
        grep -E '^(train|val|test) ' "$OUT/check_niggli_anvil.log" >> "$STATUS"
        [ $rc -eq 0 ] || halt "NIGGLI CHECK FAILED rc=$rc (see $OUT/check_niggli_anvil.log)" 3
        touch "$OUT/check_niggli_anvil.ok"; log "Niggli check ok"
    fi

    # ---- 1. width rule + sampling benchmark ---------------------------------------------------
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
        python - "$tot" "$OUT/chunk20.txt" "$OUT/chunk_min.txt" <<'PY' | while read -r l; do log "$l"; done
import re, sys
tot, path, mins = sys.argv[1], sys.argv[2], sys.argv[3]
s = float(re.search(r"sample=([\d.]+)s", tot).group(1))
sel = float(re.search(r"select=([\d.]+)s", tot).group(1))
t100 = 20 * s + sel + 120        # s per 100-structure x 20-sample chunk at 1000 PC steps (+ load, match)
c = next((c for c in (100, 50, 25, 20, 10) if t100 * c / 100 <= 80 * 60), 10)
null100 = 2 * s + 120            # 100 x 1 sample at 1000 steps: same model calls, smaller batches (x2)
open(path, "w").write(f"{c}\n")
open(mins, "w").write(f"null {null100 / 60:.1f}\ntwenty {t100 * c / 100 / 60:.1f}\n")
print(f"bench sample ({tot}) -> {t100 / 60:.0f} min per 100 x 20 chunk, chunk20 {c}; "
      f"12 twenty-sample G4 arms ~{12 * 10 * t100 / 3600:.0f} GPU-h (+ 3 null arms ~{3 * 10 * null100 / 3600:.0f})")
PY
    fi
    [ "${BENCH_ONLY:-0}" = 1 ] && { log "bench only: done"; exit 0; }

    # ---- 2. training slice ----------------------------------------------------------------------
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
    on_exit; trap - EXIT
    [ -f "$CK" ] || exit 0
fi

# ===== 3. sampling chunks once training is done ====================================================
if [ ! -f "$OUT/ckpt_final.md5" ] && take "$OUT/md5.lock"; then
    md5sum "$CK" > "$OUT/ckpt_final.md5"
    log "TRAIN DONE, final val: $(tail -1 runs/$RUN/val.jsonl)"
fi
arm_of() { env -u SLURM_ARRAY_TASK_ID -u OFFSET DRY=1 bash "$ARMS_SB" "$1" "$2" | awk '{print $2}'; }
need_min() {   # $1 = arm: minutes a chunk of it needs (bench estimate or 1.25 x the slowest so far) + 15
    local k=twenty; case $1 in p19_null_*) k=null ;; esac
    awk -v k=$k -v f="$CH/durations.txt" '
        $1 == k { b = $2 }
        END { m = 0
              while ((getline l < f) > 0) { split(l, a, " ")
                  if ((k == "null") == (a[1] ~ /^p19_null_/) && a[4] / 60 * 1.25 > m) m = a[4] / 60 * 1.25 }
              print int((m > b ? m : b) + 15.999) }' "$OUT/chunk_min.txt"
}
run_stage() {   # $1 = G3|G4: claim and run unfinished chunks; 0 = none left to claim, 2 = out of time
    local st=$1 nt id arm need
    nt=$(env -u SLURM_ARRAY_TASK_ID -u OFFSET COUNT=1 bash "$ARMS_SB" "$st" 0) || halt "$st task count failed"
    for id in $(seq 0 $((nt - 1))); do
        [ -f "$CLAIMS/$st.$id.done" ] && continue
        take "$CLAIMS/$st.$id" || continue
        arm=$(arm_of "$st" "$id"); need=$(need_min "$arm")
        if [ "$(left_min)" -lt "$need" ]; then
            release "$CLAIMS/$st.$id"; log "$(left_min) min left < $need for $st task $id: exit"; return 2
        fi
        if env -u SLURM_ARRAY_TASK_ID -u OFFSET WORKER="$JOB" bash "$ARMS_SB" "$st" "$id"; then
            touch "$CLAIMS/$st.$id.done"; release "$CLAIMS/$st.$id"
        else
            halt "$st task $id ($arm) FAILED"
        fi
    done
    return 0
}
g3_pass() { python -c "import json, sys; sys.exit(0 if json.load(open('$OUT/G3.json'))['pass'] else 1)"; }

run_stage G3; [ $? -eq 2 ] && exit 0
if [ ! -f "$OUT/G3.json" ]; then
    n_done=$(ls "$CLAIMS"/G3.*.done 2>/dev/null | wc -l)
    [ "$n_done" -eq "$(env -u SLURM_ARRAY_TASK_ID -u OFFSET COUNT=1 bash "$ARMS_SB" G3 0)" ] \
        && halt "all G3 chunks done but no G3.json (merge stuck? see $CH)"
    log "G3 chunks still running in other workers: exit"; exit 0
fi
if g3_pass && [ -f "$OPS/G4_ON_ANVIL" ]; then
    run_stage G4; [ $? -eq 2 ] && exit 0
    [ -f "$OUT/ALL_DONE" ] && stop_pending "ALL_DONE"
else
    stop_pending "G3 $(g3_pass && echo passed, G4 not on Anvil || echo failed its bar)"
fi
exit 0
