#!/bin/bash
# Local side of the JAC R2 box run: poll for /workspace/RUN_DONE (one ssh every
# 4 min, bounded to ~7 h), pull out.tar, check its md5, unpack into the repo,
# then touch /workspace/PULLED so the box stops itself.
# Usage: bash pull_loop.sh HOST PORT
set -u
HOST=$1; PORT=$2
REPO=/c/Users/frank/pxrd-diff-pub
STAGE=$REPO/paper/submissions/JAC-R2/box/pull_stage
SSH=(ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30 "root@$HOST")
mkdir -p "$STAGE"
for i in $(seq 1 105); do
    if timeout 60 "${SSH[@]}" 'test -f /workspace/RUN_DONE' 2>/dev/null; then
        echo "RUN_DONE seen $(date -u +%FT%TZ)"
        for attempt in 1 2 3 4 5; do
            timeout 1200 scp -P "$PORT" -o StrictHostKeyChecking=no \
                "root@$HOST:/workspace/out.tar" "root@$HOST:/workspace/out.tar.md5" "$STAGE/" \
                && (cd "$STAGE" && sed 's|/workspace/||' out.tar.md5 | md5sum -c -) && break
            echo "pull attempt $attempt failed"; sleep 30
        done
        (cd "$STAGE" && sed 's|/workspace/||' out.tar.md5 | md5sum -c -) || { echo "PULL FAILED"; exit 1; }
        tar -xf "$STAGE/out.tar" -C "$STAGE"
        mkdir -p "$REPO/paper/phase17_results" "$REPO/paper/submissions/JAC-R2/analysis/box_logs" "$REPO/runs/gpu_v23_r2"
        cp -r "$STAGE/paper/phase17_results/." "$REPO/paper/phase17_results/"
        cp -r "$STAGE/logs/." "$REPO/paper/submissions/JAC-R2/analysis/box_logs/"
        [ -f "$STAGE/runs/gpu_v23_r2/ckpt_final.pt" ] && cp "$STAGE/runs/gpu_v23_r2/ckpt_final.pt" "$REPO/runs/gpu_v23_r2/"
        timeout 60 "${SSH[@]}" 'touch /workspace/PULLED' && echo "PULLED $(date -u +%FT%TZ)"
        exit 0
    fi
    sleep 240
done
echo "gave up waiting for RUN_DONE"
exit 2
