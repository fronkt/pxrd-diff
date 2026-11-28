#!/bin/bash
# JAC R2 Phase 19 (option B, Frank 2026-11-25) box run, one GPU. Spec: tasks/todo.md Phase 19.
#   bench (width rule) -> train gpu_p19 -> G3 gate (null, 1 sample, DiffCSP matcher >= 0.45)
#   -> G4 arms (no refinement) -> pack -> RUN_DONE -> wait <= 3 h for PULLED -> self-stop.
# Launched detached by p19_setup.sh: setsid nohup bash p19_pipeline.sh CAP_H > logs/p19.out &
# A4 (AWS): the same script on EC2 g6e boxes; self_stop shuts the box down (shutdown behaviour stop),
# the train box resumes Anvil's checkpoint, and /workspace/ARMS splits the G4 arms across boxes.
# CAP_H = hard cap in hours from launch (set from the credit; the watchdog packs and stops).
# Every step is idempotent (markers / outputs in $OUT), so a relaunch resumes: training resumes from
# its latest checkpoint, finished arms are skipped. After each stage the results are packed to
# /workspace/p19_partial.tgz for the local pull loop (p19_pull_loop.sh).
set -u
CAP_H=${1:?usage: p19_pipeline.sh CAP_HOURS}
cd /workspace/pxrd-diff
OUT=paper/phase19_results
LOG=/workspace/logs
STATUS=$LOG/p19.status
RUN=gpu_p19
G3_BAR=0.45
mkdir -p "$OUT" "$LOG"
log() { echo "$(date -u +%FT%TZ) $*" >> "$STATUS"; }

self_stop() {   # Vast: stop the container through its API; AWS (A4): shut down (the box stops)
    log "self-stop ($1)"
    if tr '\0' '\n' < /proc/1/environ 2>/dev/null | grep -q '^CONTAINER_API_KEY='; then
        eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^(CONTAINER_API_KEY|CONTAINER_ID)=' | sed 's/^/export /')"
        curl -s -X PUT -H "Authorization: Bearer $CONTAINER_API_KEY" -d '{"state":"stopped"}' \
            "https://console.vast.ai/api/v0/instances/$CONTAINER_ID/" >> "$STATUS"
    else
        sudo shutdown -h now
    fi
}
# A4: /workspace/ARMS (one arm per line) limits this box's G4 arms; no file = all arms
want() { [ ! -f /workspace/ARMS ] || grep -qx "$1" /workspace/ARMS; }

pack() {   # $1 = output tgz; the checkpoint goes only into the final pack (pulled once on its own)
    local extra=()
    [ "$1" = /workspace/p19_out.tgz ] && [ -f "runs/$RUN/ckpt_final.pt" ]         && extra=(-C /workspace/pxrd-diff "runs/$RUN/ckpt_final.pt")
    [ -f "runs/$RUN/val.jsonl" ] && extra+=(-C /workspace/pxrd-diff "runs/$RUN/val.jsonl")
    tar -czf "$1.tmp" -C /workspace/pxrd-diff "$OUT" -C /workspace logs "${extra[@]}" 2>> "$STATUS" \
        && mv "$1.tmp" "$1" && md5sum "$1" > "$1.md5"
}

finish() {
    log "RUN_DONE ($1)"
    pack /workspace/p19_out.tgz
    touch /workspace/RUN_DONE
    for _ in $(seq 1 36); do [ -f /workspace/PULLED ] && break; sleep 300; done
    self_stop "$1; pulled=$([ -f /workspace/PULLED ] && echo yes || echo no)"
}

# Watchdog: stop 30 min after the runner dies; at the hard cap kill the runner, pack, stop.
RUNNER_PID=$$
(
    dead=0; elapsed=0
    while true; do
        sleep 300; elapsed=$((elapsed + 5))
        if kill -0 "$RUNNER_PID" 2>/dev/null; then dead=0; else dead=$((dead + 5)); fi
        [ -f /workspace/RUN_DONE ] && dead=0          # finish() owns the stop after RUN_DONE
        if [ "$dead" -ge 30 ]; then self_stop "watchdog: runner gone 30 min"; exit 0; fi
        if [ "$elapsed" -ge $((CAP_H * 60)) ] && [ ! -f /workspace/RUN_DONE ]; then
            kill "$RUNNER_PID" 2>/dev/null; pkill -x python; sleep 5
            finish "watchdog: ${CAP_H} h cap"; exit 0
        fi
    done
) &

log "start commit $(git log --oneline -1 | cut -c1-12) cap ${CAP_H} h gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
TRAIN_COMMON=(--steps 100000 --bs 256 --n-layers 6 --n-heads 4 --n-freq 128 --p-pattern-drop 0.3
              --niggli --workers 8)

# ---- 1. width rule: d 512 if 100k steps fit in <= 16 h, else 384, else 256 --------------------
bench_hours() {   # $1 = d_model; prints projected hours for 100k steps (steps 100 -> 300)
    local d=$1 lg=$LOG/bench_d$1.log
    if [ ! -f "$OUT/bench_d$d.hours" ]; then
        rm -rf "runs/bench_d$d"
        python -u scripts/14_train_csp.py --run-name "bench_d$d" "${TRAIN_COMMON[@]}" --d-model "$d" \
            --stop-at 300 --log-every 50 --val-every 1000000 --save-every 1000000 > "$lg" 2>&1
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
    D=256
    if fits "$(bench_hours 512)"; then D=512; elif fits "$(bench_hours 384)"; then D=384
    else bench_hours 256 > /dev/null; fi          # d256 timing + checkpoint for the sampling bench
    echo "$D" > "$OUT/width.txt"
    log "width rule -> d_model $D"
    # sampling speed for the cost estimate: 100 structures x 20 samples, 50 PC steps (x20 = 1000)
    python -u scripts/03_sample.py --ckpt "runs/bench_d$D/ckpt_000300.pt" --n 100 --start 0 \
        --n-samples 20 --pc-steps 50 --refine-steps 0 --amp bf16 --bucket --batch-size 50 --seed 0 \
        --out-json "$OUT/bench_sample_d$D.json" > "$LOG/bench_sample_d$D.log" 2>&1
    log "bench sample d$D (50 PC steps, 100 x 20): $(grep -h '^Total:' "$LOG/bench_sample_d$D.log")"
    pack /workspace/p19_partial.tgz
fi
D=$(cat "$OUT/width.txt")
# A4: the width came from Anvil's A100 bench; time the sampling bench on this box's GPU for the G4 cost
if [ ! -f "$OUT/bench_sample_box.log" ] && [ -f "runs/bench_d$D/ckpt_000300.pt" ]; then
    python -u scripts/03_sample.py --ckpt "runs/bench_d$D/ckpt_000300.pt" --n 100 --start 0 \
        --n-samples 20 --pc-steps 50 --refine-steps 0 --amp bf16 --bucket --batch-size 50 --seed 0 \
        --out-json "$LOG/bench_sample_box.json" > "$OUT/bench_sample_box.log" 2>&1
    log "bench sample on this box (50 PC steps, 100 x 20): $(grep -h '^Total:' "$OUT/bench_sample_box.log")"
fi

# ---- 2. training -----------------------------------------------------------------------------
if [ ! -f "runs/$RUN/ckpt_final.pt" ]; then
    latest=$(ls runs/$RUN/ckpt_[0-9]*.pt 2>/dev/null | sort | tail -1)
    log "train $RUN d$D ${latest:+resume $latest}"
    python -u scripts/14_train_csp.py --run-name "$RUN" "${TRAIN_COMMON[@]}" --d-model "$D" \
        --save-every 5000 --val-every 2000 ${latest:+--resume "$latest"} >> "$LOG/train_$RUN.log" 2>&1
    [ -f "runs/$RUN/ckpt_final.pt" ] || { log "TRAIN FAILED (see train_$RUN.log)"; finish "train failed"; exit 1; }
    log "train done"
    md5sum "runs/$RUN/ckpt_final.pt" > /workspace/ckpt_final.md5
    pack /workspace/p19_partial.tgz
fi
CK=runs/$RUN/ckpt_final.pt

SAMPLE=(--ckpt "$CK" --n 1000 --pc-steps 1000 --refine-steps 0 --amp bf16 --bucket --batch-size 50)
run_arm() {   # tag, extra args...
    local tag=$1; shift
    [ "$tag" = p19_null_s0 ] || want "$tag" || return 0
    if [ -f "$OUT/$tag.per_sample.jsonl" ]; then return 0; fi
    log "$tag start"
    python -u scripts/03_sample.py "${SAMPLE[@]}" "$@" --out-json "$OUT/$tag.json" \
        --per-sample-json "$OUT/$tag.per_sample.jsonl.tmp" --save-structures "$OUT/$tag.structs.jsonl" \
        > "$LOG/$tag.log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ] || [ ! -s "$OUT/$tag.per_sample.jsonl.tmp" ]; then log "$tag FAILED rc=$rc"; return 1; fi
    mv "$OUT/$tag.per_sample.jsonl.tmp" "$OUT/$tag.per_sample.jsonl"
    python paper/submissions/JAC-R2/analysis/rematch.py . "$OUT/$tag.structs.jsonl" --n 1000 \
        --out-json "$OUT/$tag.diffcsp_match.json" > /dev/null 2>> "$STATUS"
    log "$tag done $(grep -h 'Total:' "$LOG/$tag.log")  diffcsp $(python -c "import json;print(json.load(open('$OUT/$tag.diffcsp_match.json'))['match_rate'])")"
    pack /workspace/p19_partial.tgz
}

# ---- 3. G3 gate ------------------------------------------------------------------------------
if [ ! -f "$OUT/G3.json" ]; then
    run_arm p19_null_s0 --n-samples 1 --null-pattern --seed 0 || { finish "G3 arm failed"; exit 1; }
    python - "$OUT" "$G3_BAR" <<'EOF'
import json, sys
out, bar = sys.argv[1], float(sys.argv[2])
r = json.load(open(f"{out}/p19_null_s0.diffcsp_match.json"))["match_rate"]
json.dump({"match_rate_diffcsp": r, "bar": bar, "pass": r >= bar,
           "published_diffcsp_1sample": 0.5149}, open(f"{out}/G3.json", "w"), indent=1)
EOF
    log "G3: $(cat "$OUT/G3.json" | tr -d '\n ')"
fi
if ! python -c "import json,sys; sys.exit(0 if json.load(open('$OUT/G3.json'))['pass'] else 1)"; then
    # A5 (Frank 2026-11-28): G4 runs on this checkpoint although G3 missed its bar; the launcher writes the
    # marker, G3.json stays as measured and the paper reports G3 as failed. No marker = the old stop.
    if [ -s /workspace/G4_DESPITE_G3 ]; then
        log "A5: G3 below its bar ($(python -c "import json;print(json.load(open('$OUT/G3.json'))['match_rate_diffcsp'])") < $G3_BAR); G4 runs anyway: $(head -1 /workspace/G4_DESPITE_G3)"
    else
        finish "G3 FAILED its bar; G4 not run (spec: stop and ask Frank)"; exit 0
    fi
fi

# ---- 4. G4 arms (no refinement) --------------------------------------------------------------
IDX=paper/phase9_results/index_benchmark_v2_native.json
UNK=paper/phase15_results/index_cells_test1000_unknown.json
fails=0
for s in 1 2; do run_arm "p19_null_s$s" --n-samples 1 --null-pattern --seed "$s" || fails=$((fails+1)); done
for s in 0 1 2; do
    run_arm "p19_learned_s$s" --n-samples 20 --seed "$s" || fails=$((fails+1))
    run_arm "p19_oracle_s$s" --n-samples 20 --seed "$s" --true-lattice || fails=$((fails+1))
    run_arm "p19_indexer_s$s" --n-samples 20 --seed "$s" --lat-from-index "$IDX" || fails=$((fails+1))
    run_arm "p19_indexer_unknown_s$s" --n-samples 20 --seed "$s" --lat-from-index "$UNK" || fails=$((fails+1))
done
finish "G4 done, $fails failed arms"
