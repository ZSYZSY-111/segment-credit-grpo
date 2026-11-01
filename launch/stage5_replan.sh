#!/usr/bin/env bash
# Restart a Stage 5 queue from scratch with a new plan, stopping its training jobs. Rarely needed: the dynamic queue
# re-reads $OUT/stage5.env every minute, and launch/stage5_restart_queue.sh restarts it without stopping any job.
# Replace the plan of a Stage 5 queue that is already running:
#   bash launch/stage5_replan.sh <new-plan-file>
# Stops the orchestrators so nothing new starts, lets a running evaluation finish (short, and its output is the
# deliverable), stops the training jobs (each resumes from its latest complete checkpoint, losing at most 10 steps),
# removes a checkpoint that was cut mid-write, installs the new plan as $OUT/stage5.env and restarts the queue in the
# background, logging to logs/stage5_queue_<timestamp>.log. Runs that already have an adapter and an evaluation are
# skipped by the queue, so finished work is never repeated.
set -euo pipefail
source "$(dirname "$0")/common.sh"
NEW_ENV=${1:?usage: stage5_replan.sh <new-plan-file>}
[ -f "$NEW_ENV" ] || { echo "no such plan file: $NEW_ENV" >&2; exit 1; }
log() { echo "[$(date '+%F %T')] $*"; }

log "stopping the orchestrators"
pkill -f "launch/autodl_bootstrap.sh" || true
pkill -f "launch/stage5_segment_credit_grpo.sh" || true
sleep 2
while pgrep -f "scripts/eval_validation.py" >/dev/null; do log "a evaluation is running; waiting for it to finish"; sleep 60; done
log "stopping the training jobs (they resume from their checkpoints)"
pkill -f "scripts/train_grpo.py" || true
for _ in $(seq 1 45); do pgrep -f "scripts/train_grpo.py" >/dev/null || break; sleep 2; done
pkill -9 -f "scripts/train_grpo.py" 2>/dev/null || true
sleep 3
for ckpt in "$OUT"/stage5_*_seed*/checkpoint-*; do
  [ -d "$ckpt" ] || continue
  if [ ! -f "$ckpt/trainer_state.json" ] || [ ! -f "$ckpt/optimizer.pt" ] || ! ls "$ckpt"/adapter_model.safetensors >/dev/null 2>&1; then
    log "removing incomplete checkpoint $ckpt"; rm -rf "$ckpt"
  fi
done
[ "$(readlink -f "$NEW_ENV")" = "$(readlink -f "$OUT/stage5.env")" ] || cp "$NEW_ENV" "$OUT/stage5.env"
log "new plan installed:"; cat "$OUT/stage5.env"
START=${START:-$OUT/stage4_weighted_seed42/final_adapter}
QUEUE_LOG="logs/stage5_queue_$(date +%Y%m%d_%H%M%S).log"
nohup env START="$START" bash launch/stage5_segment_credit_grpo.sh > "$QUEUE_LOG" 2>&1 < /dev/null &
log "queue restarted (pid $!), log: $QUEUE_LOG"
