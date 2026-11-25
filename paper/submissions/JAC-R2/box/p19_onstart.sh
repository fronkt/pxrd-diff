#!/bin/bash
# Vast onstart hook: after a restart (e.g. a credit top-up restarts a stopped box), relaunch the
# Phase 19 pipeline unless the run already finished. The pipeline resumes from its markers.
[ -f /workspace/p19_pipeline.sh ] || exit 0
[ -f /workspace/RUN_DONE ] && exit 0
cd /workspace
echo "$(date -u +%FT%TZ) onstart relaunch" >> /workspace/logs/p19.status
setsid nohup bash p19_pipeline.sh "$(cat /workspace/CAP_H)" >> /workspace/logs/p19.out 2>&1 < /dev/null &
