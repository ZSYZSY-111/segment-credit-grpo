#!/usr/bin/env bash
# Restart the Stage 5 orchestrator without touching running jobs, e.g. after a code update, or to replace an old
# static queue with the dynamic one:
#   bash launch/stage5_restart_queue.sh [new-plan-file]
# Only the orchestrator processes are stopped. Training and evaluation processes keep running; the new queue sees them
# by their output directory, waits for them, and runs their remaining step (the evaluation) when they finish. A plan
# change alone needs no restart: the dynamic queue re-reads $OUT/stage5.env every minute.
set -euo pipefail
source "$(dirname "$0")/common.sh"
if [ -n "${1:-}" ]; then
  [ -f "$1" ] || { echo "no such plan file: $1" >&2; exit 1; }
  [ "$(readlink -f "$1")" = "$(readlink -f "$OUT/stage5.env")" ] || cp "$1" "$OUT/stage5.env"
fi
pkill -f "[l]aunch/stage5_segment_credit_grpo.sh" || true
for _ in $(seq 1 15); do pgrep -f "[l]aunch/stage5_segment_credit_grpo.sh" > /dev/null || break; sleep 1; done
QUEUE_LOG="logs/stage5_queue_$(date +%Y%m%d_%H%M%S).log"
nohup bash launch/stage5_segment_credit_grpo.sh > "$QUEUE_LOG" 2>&1 < /dev/null &
echo "[$(date '+%F %T')] queue restarted (pid $!), log: $QUEUE_LOG"
