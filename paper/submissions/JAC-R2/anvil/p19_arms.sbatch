#!/bin/bash
#SBATCH -A che260157-gpu
#SBATCH -p gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus-per-node=1
#SBATCH -t 01:50:00
#SBATCH -J p19-arms
#SBATCH -o /anvil/scratch/x-fcai3/pxrd_ops/p19/%x_%A_%a.out
# JAC R2 Phase 19 on Anvil (amendment A1 in tasks/todo.md): the G3 gate arm and the G4 arms, one
# array task per (arm, chunk).
#   sbatch --array=0-9 p19_arms.sbatch G3                 (submitted by p19_train.sbatch at ckpt_final)
#   sbatch --export=ALL,OFFSET=o --array=0-$((k - 1))%12 p19_arms.sbatch G4   (blocks of <= 1000 tasks,
#                    submitted by the task that passes G3 if $OPS/G4_ON_ANVIL exists; else by hand)
#   COUNT=1 bash p19_arms.sbatch STAGE 0   prints NT;   DRY=1 bash p19_arms.sbatch STAGE ID   prints a command
# Command per chunk = box/p19_pipeline.sh's SAMPLE + arm arguments (Phase 19 spec: 1000 PC steps, no
# refinement, bf16, atom-count buckets) with --start/--n; 03_sample seeds a chunk with
# seed + 100003 * start. Chunks: null arms 100 structures, 20-sample arms chunk20.txt (sized by the
# sampling benchmark in p19_train.sbatch). The task that finishes an arm's last chunk merges it
# (merge_chunks.py) and rematches it at the DiffCSP tolerances (rematch.py). The merge of p19_null_s0
# writes G3.json with the Phase 19 bar (0.45): pass -> G4, fail -> stop (spec: no G4, ask Frank).
set -u
P=/anvil/projects/x-che260157
R=/anvil/scratch/x-fcai3/pxrd-diff-p19
OPS=/anvil/scratch/x-fcai3/pxrd_ops/p19
export PATH=$P/pxrd_env/bin:$PATH PYTHONPATH=$R/src OMP_NUM_THREADS=16
cd "$R" || exit 2
STAGE=${1:?usage: p19_arms.sbatch G3|G4 [task id]}
id=$(( ${SLURM_ARRAY_TASK_ID:-${2:?task id}} + ${OFFSET:-0} ))   # OFFSET: arrays are <= 1001 tasks
OUT=paper/phase19_results; CH=$OUT/chunks
CK=runs/gpu_p19/ckpt_final.pt
STATUS=$OPS/p19.status
ME=paper/submissions/JAC-R2/anvil/p19_arms.sbatch
MERGE=paper/submissions/JAC-R2/anvil/merge_chunks.py
REMATCH=paper/submissions/JAC-R2/analysis/rematch.py
IDX=paper/phase9_results/index_benchmark_v2_native.json
UNK=paper/phase15_results/index_cells_test1000_unknown.json
N=1000; G3_BAR=0.45
mkdir -p "$CH"
log() { echo "$(date -u +%FT%TZ) [$STAGE ${SLURM_ARRAY_JOB_ID:-dry}_$id] $*" | tee -a "$STATUS" >&2; }

arms_for() {
    case $1 in
        G3) echo p19_null_s0 ;;
        G4) echo p19_null_s1 p19_null_s2
            for s in 0 1 2; do echo p19_learned_s$s p19_oracle_s$s p19_indexer_s$s p19_indexer_unknown_s$s; done ;;
        *) echo "unknown stage $1" >&2; exit 2 ;;
    esac
}
chunk_for() { case $1 in p19_null_*) echo 100 ;; *) cat "$OUT/chunk20.txt" ;; esac; }
args_for() {
    local s=${1##*_s}
    case $1 in
        p19_null_s*) echo --n-samples 1 --null-pattern --seed "$s" ;;
        p19_learned_s*) echo --n-samples 20 --seed "$s" ;;
        p19_oracle_s*) echo --n-samples 20 --seed "$s" --true-lattice ;;
        p19_indexer_unknown_s*) echo --n-samples 20 --seed "$s" --lat-from-index $UNK ;;
        p19_indexer_s*) echo --n-samples 20 --seed "$s" --lat-from-index $IDX ;;
    esac
}
TASKS=()
for arm in $(arms_for "$STAGE"); do
    c=$(chunk_for "$arm"); [ -n "$c" ] || { echo "no chunk size for $arm" >&2; exit 2; }
    for st in $(seq 0 "$c" $((N - 1))); do TASKS+=("$arm $st $c"); done
done
[ "${COUNT:-0}" = 1 ] && { echo ${#TASKS[@]}; exit 0; }
[ "$id" -lt ${#TASKS[@]} ] || { log "task id $id >= ${#TASKS[@]}: nothing to do"; exit 0; }
read -r arm st c <<< "${TASKS[$id]}"
base=$CH/$arm.c$(printf %04d "$st")
cmd=(python -u scripts/03_sample.py --ckpt "$CK" --pc-steps 1000 --refine-steps 0 --amp bf16 --bucket
     --batch-size 50 $(args_for "$arm") --n "$c" --start "$st" --out-json "$base.json"
     --per-sample-json "$base.per_sample.jsonl" --save-structures "$base.structs.jsonl")
[ "${DRY:-0}" = 1 ] && { echo "DRY $arm c$st: ${cmd[*]}"; exit 0; }
[ -f "$CK" ] || { log "no $CK"; exit 1; }

if [ -f "$OUT/$arm.per_sample.jsonl" ] || [ -f "$base.done" ]; then
    log "$arm c$st already done"
else
    log "$arm c$st start host $(hostname)"
    t1=$(date +%s)
    "${cmd[@]}" > "$base.log" 2>&1
    rc=$?; dt=$(( $(date +%s) - t1 ))
    if [ $rc -eq 0 ] && [ -s "$base.per_sample.jsonl" ]; then
        touch "$base.done"; echo "$arm $st $c $dt" >> "$CH/durations.txt"; log "$arm c$st ok ${dt}s"
    else
        echo "$arm $st rc=$rc ${dt}s ${SLURM_ARRAY_JOB_ID:-}_$id" >> "$CH/failures.txt"
        log "$arm c$st FAILED rc=$rc after ${dt}s (see $base.log)"; exit 1
    fi
fi

# ---- merge + DiffCSP rematch once all of the arm's chunks are done (one merger per arm) ----------
nch=$(( (N + c - 1) / c ))
if [ ! -f "$OUT/$arm.per_sample.jsonl" ] && [ "$(ls "$CH/$arm".c*.done 2>/dev/null | wc -l)" -eq $nch ] \
        && mkdir "$CH/$arm.merge.lock" 2>/dev/null; then
    if python "$MERGE" "$OUT" "$arm" $N "$c" >> "$STATUS" 2>&1 \
            && python "$REMATCH" . "$OUT/$arm.structs.jsonl" --n $N \
               --out-json "$OUT/$arm.diffcsp_match.json" > /dev/null 2>> "$STATUS"; then
        log "merged $arm: paper match $(python -c "import json; print(json.load(open('$OUT/$arm.json'))['rates_from_per_sample']['match'])")  diffcsp $(python -c "import json; print(json.load(open('$OUT/$arm.diffcsp_match.json'))['match_rate'])")"
    else
        log "MERGE/REMATCH FAILED for $arm (lock kept: $CH/$arm.merge.lock)"; exit 1
    fi
    rmdir "$CH/$arm.merge.lock"
fi

# ---- G3 gate ---------------------------------------------------------------------------------
if [ "$STAGE" = G3 ] && [ -f "$OUT/p19_null_s0.diffcsp_match.json" ] && [ ! -f "$OUT/G3.json" ] \
        && mkdir "$OUT/G3.lock" 2>/dev/null; then
    python - "$OUT" "$G3_BAR" <<'EOF'
import json, sys
out, bar = sys.argv[1], float(sys.argv[2])
r = json.load(open(f"{out}/p19_null_s0.diffcsp_match.json"))["match_rate"]
json.dump({"match_rate_diffcsp": r, "bar": bar, "pass": r >= bar,
           "published_diffcsp_1sample": 0.5149}, open(f"{out}/G3.json", "w"), indent=1)
EOF
    log "G3: $(tr -d '\n ' < "$OUT/G3.json")"
    if python -c "import json, sys; sys.exit(0 if json.load(open('$OUT/G3.json'))['pass'] else 1)"; then
        if [ -f "$OPS/G4_ON_ANVIL" ]; then
            NT=$(COUNT=1 bash "$ME" G4 0)
            for o in $(seq 0 1000 $((NT - 1))); do
                k=$(( NT - o < 1000 ? NT - o : 1000 ))
                J=$(sbatch --parsable --export=ALL,OFFSET=$o --array=0-$((k - 1))%12 "$ME" G4) \
                    && log "G3 PASS -> G4 array $J (tasks $o-$((o + k - 1)) of $NT)" || log "G3 PASS, G4 SUBMIT FAILED"
            done
        else
            log "G3 PASS; G4 waits for the compute decision (touch $OPS/G4_ON_ANVIL and submit, or Vast)"
        fi
    else
        log "G3 FAILED its bar: G4 not run (Phase 19 spec: stop and ask Frank)"
    fi
fi

# ---- G4 complete -------------------------------------------------------------------------------
if [ "$STAGE" = G4 ] && [ ! -f "$OUT/ALL_DONE" ]; then
    n_done=0; n_arms=0
    for a in $(arms_for G4); do n_arms=$((n_arms + 1)); [ -f "$OUT/$a.diffcsp_match.json" ] && n_done=$((n_done + 1)); done
    [ $n_done -eq $n_arms ] && { date -u +%FT%TZ > "$OUT/ALL_DONE"; log "ALL_DONE: G3 + $n_arms G4 arms merged"; }
fi
exit 0
