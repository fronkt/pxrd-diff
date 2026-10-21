#!/bin/bash
# JAC R2 (hat5032) box run: D1 diagnostic on v22, then v23 retrain + evals.
# Launched once with `setsid nohup bash pipeline.sh > /workspace/logs/pipeline.out 2>&1 &`.
# Idempotent: a finished eval (its per-sample file exists) or a finished training
# run (ckpt_final.pt exists) is skipped on a relaunch. Ends by packing
# /workspace/out.tar, touching /workspace/RUN_DONE, waiting up to 45 min for the
# local pull to touch /workspace/PULLED, then stopping the instance itself.
set -u
cd /workspace/pxrd-diff
OUT=paper/phase17_results
LOG=/workspace/logs
STATUS=$LOG/pipeline.status
mkdir -p "$OUT" "$LOG"

self_stop() {
    eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^(CONTAINER_API_KEY|CONTAINER_ID)=' | sed 's/^/export /')"
    echo "self-stop $(date -u +%FT%TZ) ($1)" >> "$STATUS"
    curl -s -X PUT -H "Authorization: Bearer $CONTAINER_API_KEY" -d '{"state":"stopped"}' \
        "https://console.vast.ai/api/v0/instances/$CONTAINER_ID/" >> "$STATUS"
}

# Pack whatever exists, signal the local pull loop (it polls for RUN_DONE), wait
# up to 45 min for it to touch PULLED, then stop.
pack_and_stop() {
    echo "RUN_DONE $(date -u +%FT%TZ) ($1)" >> "$STATUS"
    tar -cf /workspace/out.tar -C /workspace/pxrd-diff "$OUT" -C /workspace logs         $( [ -f /workspace/pxrd-diff/runs/gpu_v23_r2/ckpt_final.pt ] && echo -C /workspace/pxrd-diff runs/gpu_v23_r2/ckpt_final.pt )         2>> "$STATUS"
    md5sum /workspace/out.tar > /workspace/out.tar.md5
    touch /workspace/RUN_DONE
    for _ in $(seq 1 9); do
        [ -f /workspace/PULLED ] && break
        sleep 300
    done
    self_stop "$1; pulled=$([ -f /workspace/PULLED ] && echo yes || echo no)"
}

# Watchdog (a forked subshell keeps this script's argv, so it tracks the runner by
# PID): stop the box 30 min after the runner dies, or at the 4.4 h hard cap
# (credit USD 4.27 at USD 0.884/h runs out at ~4.8 h).
RUNNER_PID=$$
(
    dead=0; elapsed=0
    while true; do
        sleep 300
        elapsed=$((elapsed + 5))
        if kill -0 "$RUNNER_PID" 2>/dev/null; then dead=0; else dead=$((dead + 5)); fi
        if [ "$dead" -ge 30 ]; then self_stop "watchdog: runner gone 30 min"; exit 0; fi
        if [ "$elapsed" -ge 265 ]; then pkill -x python; pack_and_stop "watchdog: 4.4 h cap"; exit 0; fi
    done
) &

COMMON=(--n 1000 --n-samples 20 --ensemble-eta 0.0 --refine-steps 200 --refine-lr 1e-3 --ddim-steps 50)
IDX_GIVEN=paper/phase9_results/index_benchmark_v2_native.json
IDX_UNKNOWN=paper/phase15_results/index_cells_test1000_unknown.json

run() {
    local tag=$1; shift
    if [ -f "$OUT/$tag.per_sample.jsonl" ]; then echo "$tag skip" >> "$STATUS"; return 0; fi
    echo "$tag start $(date -u +%FT%TZ)" >> "$STATUS"
    python scripts/03_sample.py "$@" --out-json "$OUT/$tag.json" \
        --per-sample-json "$OUT/$tag.per_sample.jsonl" > "$LOG/$tag.log" 2>&1
    echo "$tag rc=$? $(date -u +%FT%TZ)" >> "$STATUS"
}

# D1: the published v22 checkpoint with the reference lattice fully excluded from
# sampling (distances from the cell being sampled). v22 was trained with the
# clean cell, so this is out of distribution for it: a diagnostic, not a fix.
V22=runs/gpu_v22_jac/ckpt_final.pt
for s in 0 1 2; do
    run v22_learned_noref_s$s --ckpt $V22 "${COMMON[@]}" --seed $s --dist-lattice noisy
done

# R1: v23 = the v22 configuration with both R2 fixes (train defaults:
# --dist-lattice noisy --periodic-dist min_image).
CK=runs/gpu_v23_r2/ckpt_final.pt
if [ ! -f "$CK" ]; then
    echo "train start $(date -u +%FT%TZ)" >> "$STATUS"
    python scripts/02_train.py --run-name gpu_v23_r2 --steps 100000 --bs 64 --lr 1e-3 \
        --predict-x0 --debye-weight 1.0 --debye-form-factor pymatgen --save-every 10000 \
        > "$LOG/train.log" 2>&1
    echo "train rc=$? $(date -u +%FT%TZ)" >> "$STATUS"
fi
if [ -f "$CK" ]; then
    # core arms first; the unknown-system indexer last (first to drop if credit runs short)
    for s in 0 1 2; do
        run v23_learned_s$s         --ckpt $CK "${COMMON[@]}" --seed $s
        run v23_oracle_s$s          --ckpt $CK "${COMMON[@]}" --seed $s --true-lattice
        run v23_indexer_s$s         --ckpt $CK "${COMMON[@]}" --seed $s --lat-from-index $IDX_GIVEN
    done
    for s in 0 1 2; do
        run v23_indexer_unknown_s$s --ckpt $CK "${COMMON[@]}" --seed $s --lat-from-index $IDX_UNKNOWN
    done
fi

pack_and_stop "pipeline finished"
