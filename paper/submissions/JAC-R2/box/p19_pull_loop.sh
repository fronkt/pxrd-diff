#!/bin/bash
# Local side of the Phase 19 box run. Every 30 min (one ssh per poll): pull the partial pack if it
# changed, pull ckpt_final.pt once when training is done (md5-checked); on RUN_DONE pull the final
# pack, check md5, unpack into the repo and touch /workspace/PULLED so the box stops itself.
# Bounded to ~96 h. Usage: bash p19_pull_loop.sh HOST PORT
set -u
HOST=$1; PORT=$2
REPO=/c/Users/frank/pxrd-diff-pub
STAGE=$REPO/paper/submissions/JAC-R2/box/p19_stage
SSH=(ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30 "root@$HOST")
SCP=(scp -P "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=20)
mkdir -p "$STAGE" "$REPO/runs/gpu_p19"
get_checked() {   # remote file, local dir -> pulls file + .md5, verifies
    local f=$1 d=$2
    for a in 1 2 3; do
        timeout 1800 "${SCP[@]}" "root@$HOST:$f" "root@$HOST:$f.md5" "$d/" 2>/dev/null \
            && (cd "$d" && sed 's|/workspace/||; s|runs/gpu_p19/||' "$(basename "$f").md5" | md5sum -c - >/dev/null) \
            && return 0
        sleep 60
    done
    return 1
}
last=""
for i in $(seq 1 192); do
    st=$(timeout 60 "${SSH[@]}" 'cat /workspace/p19_partial.tgz.md5 2>/dev/null; ls /workspace/RUN_DONE /workspace/ckpt_final.md5 2>/dev/null; tail -2 /workspace/logs/p19.status' 2>/dev/null)
    echo "$(date -u +%FT%TZ) $(echo "$st" | tail -2 | tr '\n' ' ')"
    if echo "$st" | grep -q RUN_DONE; then
        get_checked /workspace/p19_out.tgz "$STAGE" || { echo "FINAL PULL FAILED"; exit 1; }
        tar -xzf "$STAGE/p19_out.tgz" -C "$STAGE"
        mkdir -p "$REPO/paper/phase19_results" "$REPO/paper/submissions/JAC-R2/analysis/box_logs_p19"
        cp -r "$STAGE/paper/phase19_results/." "$REPO/paper/phase19_results/"
        cp -r "$STAGE/logs/." "$REPO/paper/submissions/JAC-R2/analysis/box_logs_p19/"
        [ -d "$STAGE/runs" ] && cp -r "$STAGE/runs/." "$REPO/runs/"
        timeout 60 "${SSH[@]}" 'touch /workspace/PULLED' && echo "PULLED $(date -u +%FT%TZ)"
        exit 0
    fi
    if echo "$st" | grep -q ckpt_final.md5 && [ ! -f "$REPO/runs/gpu_p19/ckpt_final.pt.ok" ]; then
        timeout 1800 "${SCP[@]}" "root@$HOST:/workspace/pxrd-diff/runs/gpu_p19/ckpt_final.pt" \
            "root@$HOST:/workspace/ckpt_final.md5" "$REPO/runs/gpu_p19/" 2>/dev/null \
            && (cd "$REPO/runs/gpu_p19" && sed 's|runs/gpu_p19/||' ckpt_final.md5 | md5sum -c -) \
            && touch "$REPO/runs/gpu_p19/ckpt_final.pt.ok" && echo "ckpt pulled"
    fi
    cur=$(echo "$st" | grep p19_partial | head -1 | cut -c1-32)
    if [ -n "$cur" ] && [ "$cur" != "$last" ] && get_checked /workspace/p19_partial.tgz "$STAGE"; then
        tar -xzf "$STAGE/p19_partial.tgz" -C "$STAGE" && last=$cur && echo "partial pulled $cur"
    fi
    sleep 1800
done
echo "gave up after ~96 h"
exit 2
