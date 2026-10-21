#!/bin/bash
# JAC R2 (hat5032) box run 2: v24 = v22 configuration + distance fixes + the "r2"
# denoiser architecture (pre-norm cross-attention on normalised PXRD features,
# sender masking; 02_train.py defaults). Training starts at full length; the
# step-10k checkpoint is gated with analysis/denoiser_collapse/input_sensitivity.py
# while training continues. Gate FAIL -> stop training, pack, self-stop.
# Gate PASS -> finish training, then learned / oracle / indexer x 3 seeds and the
# unknown-system indexer x 3, pack, self-stop. Idempotent on relaunch for the evals.
set -u
cd /workspace/pxrd-diff
OUT=paper/phase17_results
LOG=/workspace/logs
STATUS=$LOG/pipeline_r2fix.status
RUN=gpu_v24_r2
GATE=paper/submissions/JAC-R2/analysis/denoiser_collapse/input_sensitivity.py
mkdir -p "$OUT/gate" "$LOG"

self_stop() {
    eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^(CONTAINER_API_KEY|CONTAINER_ID)=' | sed 's/^/export /')"
    echo "self-stop $(date -u +%FT%TZ) ($1)" >> "$STATUS"
    curl -s -X PUT -H "Authorization: Bearer $CONTAINER_API_KEY" -d '{"state":"stopped"}' \
        "https://console.vast.ai/api/v0/instances/$CONTAINER_ID/" >> "$STATUS"
}

pack_and_stop() {
    echo "RUN_DONE $(date -u +%FT%TZ) ($1)" >> "$STATUS"
    rm -f /workspace/RUN_DONE /workspace/PULLED
    CKS=""
    [ -f runs/$RUN/ckpt_final.pt ] && CKS="runs/$RUN/ckpt_final.pt"
    [ -f runs/$RUN/ckpt_010000.pt ] && CKS="$CKS runs/$RUN/ckpt_010000.pt"
    tar -cf /workspace/out.tar "$OUT" $CKS -C /workspace logs 2>> "$STATUS"
    md5sum /workspace/out.tar > /workspace/out.tar.md5
    touch /workspace/RUN_DONE
    for _ in $(seq 1 9); do
        [ -f /workspace/PULLED ] && break
        sleep 300
    done
    self_stop "$1; pulled=$([ -f /workspace/PULLED ] && echo yes || echo no)"
}

# Watchdog: runner gone 30 min, or the 4 h cap.
RUNNER_PID=$$
(
    dead=0; elapsed=0
    while true; do
        sleep 300
        elapsed=$((elapsed + 5))
        if kill -0 "$RUNNER_PID" 2>/dev/null; then dead=0; else dead=$((dead + 5)); fi
        if [ "$dead" -ge 30 ]; then self_stop "watchdog: runner gone 30 min"; exit 0; fi
        if [ "$elapsed" -ge 240 ]; then pkill -x python; pack_and_stop "watchdog: 4 h cap"; exit 0; fi
    done
) &

gate() {   # gate CKPT_FILE -> appends a line, returns the script's exit code
    local ck=$1 tag; tag=$(basename "$ck" .pt)
    python "$GATE" /workspace/pxrd-diff "$ck" "$OUT/gate/${RUN}_${tag}.json" > "$LOG/gate_${tag}.log" 2>&1
    local rc=$?
    echo "gate $tag rc=$rc $(grep -h '^step' "$LOG/gate_${tag}.log" | tail -1)" >> "$STATUS"
    return $rc
}

CK=runs/$RUN/ckpt_final.pt
if [ ! -f "$CK" ]; then
    echo "train start $(date -u +%FT%TZ)" >> "$STATUS"
    python scripts/02_train.py --run-name $RUN --steps 100000 --bs 64 --lr 1e-3 \
        --predict-x0 --debye-weight 1.0 --debye-form-factor pymatgen --save-every 2000 \
        > "$LOG/train_$RUN.log" 2>&1 &
    TRAIN_PID=$!
    until [ -f runs/$RUN/ckpt_010000.pt ] || ! kill -0 $TRAIN_PID 2>/dev/null; do sleep 20; done
    sleep 30   # let torch.save finish writing
    for s in 002000 004000 006000 008000; do [ -f runs/$RUN/ckpt_$s.pt ] && gate runs/$RUN/ckpt_$s.pt; done
    if ! gate runs/$RUN/ckpt_010000.pt; then
        kill $TRAIN_PID 2>/dev/null
        pack_and_stop "gate FAIL at 10k"
        exit 0
    fi
    echo "gate PASS at 10k; training continues $(date -u +%FT%TZ)" >> "$STATUS"
    wait $TRAIN_PID
    echo "train rc=$? $(date -u +%FT%TZ)" >> "$STATUS"
    [ -f "$CK" ] && gate "$CK"
fi

COMMON=(--n 1000 --n-samples 20 --ensemble-eta 0.0 --refine-steps 200 --refine-lr 1e-3 --ddim-steps 50)
run() {
    local tag=$1; shift
    if [ -f "$OUT/$tag.per_sample.jsonl" ]; then echo "$tag skip" >> "$STATUS"; return 0; fi
    echo "$tag start $(date -u +%FT%TZ)" >> "$STATUS"
    python scripts/03_sample.py "$@" --out-json "$OUT/$tag.json" \
        --per-sample-json "$OUT/$tag.per_sample.jsonl" > "$LOG/$tag.log" 2>&1
    echo "$tag rc=$? $(date -u +%FT%TZ)" >> "$STATUS"
}
if [ -f "$CK" ]; then
    for s in 0 1 2; do
        run v24_learned_s$s --ckpt $CK "${COMMON[@]}" --seed $s
        run v24_oracle_s$s  --ckpt $CK "${COMMON[@]}" --seed $s --true-lattice
        run v24_indexer_s$s --ckpt $CK "${COMMON[@]}" --seed $s --lat-from-index paper/phase9_results/index_benchmark_v2_native.json
    done
    for s in 0 1 2; do
        run v24_indexer_unknown_s$s --ckpt $CK "${COMMON[@]}" --seed $s \
            --lat-from-index paper/phase15_results/index_cells_test1000_unknown.json
    done
fi
pack_and_stop "pipeline finished"
