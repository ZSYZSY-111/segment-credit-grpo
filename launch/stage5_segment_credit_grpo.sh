#!/usr/bin/env bash
# Stage 5: GRPO arms from one start adapter, run as a dynamic job queue (docs/05_segment_credit_grpo.md).
#
# Jobs, in priority order:
#   name:lambda:mode:seed[:extra]  train an arm into $OUT/stage5_<name>_seed<seed>, then evaluate it on the CMExam
#                                  validation split. extra is a +-separated list of train_grpo.py arguments appended
#                                  after the shared ones, e.g. g16:2.0:segment:42:--num-generations+16+--prompts-per-step+8.
#                                  The control modes uniform, shuffled and sequence are in scripts/grpo_segment_credit.py.
#   cmbtest:<run>                  evaluate $OUT/<run>/final_adapter on the official CMB-Exam test split in rationale mode
#                                  (launch/evaluate_cmb_rationale.sh); it waits until that adapter exists.
# JOBS gives the list; without it, ARMS (name:lambda:mode) × SEEDS builds one.
#
# The queue is dynamic: every QUEUE_POLL seconds it re-reads $OUT/stage5.env, so jobs can be added, removed or
# reordered and PARALLEL changed while it runs, with no restart. A job is done when its evaluation report exists, and
# running while its lock in $OUT/locks is held or a process for its output directory is alive, so a job never runs
# twice, not even beside an orphaned process from an earlier queue. Training resumes from the latest checkpoint and is
# skipped once the final adapter exists. A line "name:seed" or "cmbtest:<run>" in $OUT/stage5.skip skips a job;
# $OUT/stage5.stop stops new starts. When nothing is left, every arm is compared with standard GRPO per seed, on both
# benchmarks, and aggregate_seeds.py summarises the seeds.
# Per job on one RTX 5090: standard ~2 h, probed arms ~3 h, evaluation ~20 min; PARALLEL=3 fits on 96 GB.
#   bash launch/stage5_segment_credit_grpo.sh
#   SEEDS="42 7" PARALLEL=3 ARMS="standard:0:segment segment_credit:2.0:segment" bash launch/stage5_segment_credit_grpo.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"
STAGE5_ENV=${STAGE5_ENV:-$OUT/stage5.env}
LOCKS=$OUT/locks
mkdir -p "$LOCKS" logs
SEEDS=${SEEDS:-42}
ARMS=${ARMS:-"standard:0:segment segment_credit:2.0:segment"}
PARALLEL=${PARALLEL:-1}
START=${START:-$OUT/stage4_weighted_seed42/final_adapter}
POOL=$DATA/rl_cot_pool
STEPS=${STEPS:-150}
GEN_CHUNK=${GEN_CHUNK:-32}        # completions per generate call; raise on cards with more memory
PROBE_BATCH=${PROBE_BATCH:-64}    # probe forward passes per batch
QUEUE_POLL=${QUEUE_POLL:-60}
STAGGER=${STAGGER:-20}            # seconds between two starts, so model loading does not overlap
MAX_FAILS=${MAX_FAILS:-2}         # a job that failed this many times is left out (training already retries 6 times)
stamp() { date '+%F %T'; }

# one queue at a time; job subshells close this descriptor so their processes never hold it
exec 7> "$LOCKS/queue.lock"
flock -n 7 || { echo "[$(stamp)] another stage-5 queue is running; edit $STAGE5_ENV to change its plan"; exit 1; }

load_plan() {
  if [ -f "$STAGE5_ENV" ]; then source "$STAGE5_ENV"; fi
  PLAN=${JOBS:-}
  if [ -z "$PLAN" ]; then
    for seed in $SEEDS; do for arm in $ARMS; do PLAN="$PLAN $arm:$seed"; done; done
  fi
}

parse() {  # parse <job> → sets name lam mode seed extra kind id
  IFS=: read -r name lam mode seed extra <<< "$1"
  extra=${extra:-}
  if [ "$name" = cmbtest ]; then kind=cmb; id="cmb_${lam}"; else kind=train; id="stage5_${name}_seed${seed}"; fi
}

job_done() {
  if [ "$kind" = cmb ]; then [ -f "$OUT/$lam/eval_cmb_test/headroom_report.json" ]
  else [ -f "$OUT/$id/eval_cmexam_val/headroom_report.json" ]; fi
}

job_ready() {
  if [ "$kind" = cmb ]; then [ -f "$OUT/$lam/final_adapter/adapter_config.json" ]
  else [ -f "$START/adapter_config.json" ]; fi
}

job_running() {
  local pattern
  if ! flock -n "$LOCKS/$id.lock" true 2>/dev/null; then return 0; fi
  if [ "$kind" = cmb ]; then pattern="scripts/eval_validation\.py .*--output-dir $OUT/$lam/eval_cmb_test( |\$)"
  else pattern="scripts/(train_grpo|eval_validation)\.py .*--output-dir $OUT/$id(/eval_cmexam_val)?( |\$)"; fi
  pgrep -f "python[^ ]* -u $pattern" > /dev/null 2>&1
}

job_failed_out() {
  [ -f "$LOCKS/$id.failed" ] && [ "$(wc -l < "$LOCKS/$id.failed")" -ge "$MAX_FAILS" ]
}

job_skipped() {
  [ -f "$OUT/stage5.skip" ] || return 1
  if [ "$kind" = cmb ]; then grep -qx "cmbtest:$lam" "$OUT/stage5.skip"; else grep -qx "$name:$seed" "$OUT/stage5.skip"; fi
}

records() {  # records <name> <seed> [benchmark] → the per-question record file of that run
  echo "$OUT/stage5_${1}_seed${2}/${3:-eval_cmexam_val}/headroom_records.jsonl"
}

accuracy() {  # accuracy <report.json>
  $PY -c "import json,sys; print(json.load(open(sys.argv[1]))['overall'].get('pass1_greedy'))" "$1" 2>/dev/null || echo "?"
}

train_arm() {  # train_arm <name> <lambda> <mode> <seed> [extra]
  local name=$1 lam=$2 mode=$3 seed=$4 extra=${5:-} run attempt latest resume extra_args common
  run="stage5_${name}_seed${seed}"
  [ -f "$OUT/$run/final_adapter/adapter_config.json" ] && [ -f "$OUT/$run/train_results.json" ] && return 0
  extra_args=(${extra//+/ })
  common=(--model "$BASE" --adapter "$START" --train-file "$POOL/train.jsonl" --validation-file "$POOL/validation.jsonl"
          --reward-mode cot --prompts-per-step 16 --num-generations 8 --micro-batch 4 --generation-chunk "$GEN_CHUNK"
          --segment-probe-batch "$PROBE_BATCH" --max-completion-length 384 --temperature 1.0 --learning-rate 1e-5
          --scale-rewards none --loss-type dapo --epsilon 0.2 --epsilon-high 0.28 --eval-steps 50 --save-steps 10
          --save-total-limit 3 --max-eval-samples 64 --log-completions)
  for attempt in 1 2 3 4 5 6; do
    resume=()
    latest=$(ls -d "$OUT/$run"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
    [ -n "$latest" ] && resume=(--resume-from-checkpoint "$latest")
    if $PY -u scripts/train_grpo.py "${common[@]}" --seed "$seed" --output-dir "$OUT/$run" --max-steps "$STEPS" \
        --segment-credit-lambda "$lam" --credit-mode "$mode" ${extra_args[@]+"${extra_args[@]}"} "${resume[@]}" \
        > "logs/${run}_try$attempt.log" 2>&1; then
      return 0
    fi
    echo "[$(stamp)] attempt $attempt of $run failed; resuming from ${latest:-scratch}" >&2
  done
  return 1
}

evaluate_arm() {  # evaluate_arm <run>
  [ -f "$OUT/$1/eval_cmexam_val/headroom_report.json" ] && return 0
  $PY -u scripts/eval_validation.py --model "$BASE" --adapter "$OUT/$1/final_adapter" \
    --input-file "$DATA/cmexam_sft_validation.jsonl" --output-dir "$OUT/$1/eval_cmexam_val" \
    --mode cot --prompt-style legacy_explain --limit 0 --batch-size 64 --max-new-tokens 768 \
    --skip-sampling --skip-shuffle --seed 42 --overwrite > "logs/${1}_eval.log" 2>&1
}

start_job() {  # start_job <job>: runs in the background while holding the job's lock
  local job=$1
  parse "$job"
  (
    exec 7>&-
    exec 9> "$LOCKS/$id.lock"
    flock -n 9 || exit 0
    parse "$job"
    if [ "$kind" = cmb ]; then
      if bash launch/evaluate_cmb_rationale.sh "$OUT/$lam/final_adapter" "$lam" > "logs/${lam}_cmb_test_eval.log" 2>&1; then
        echo "[$(stamp)] done: cmbtest $lam → greedy accuracy $(accuracy "$OUT/$lam/eval_cmb_test/headroom_report.json")"
      else
        echo "[$(stamp)] FAILED: cmbtest $lam (see logs/${lam}_cmb_test_eval.log)" >&2
        echo "$(stamp)" >> "$LOCKS/$id.failed"
      fi
    elif train_arm "$name" "$lam" "$mode" "$seed" "$extra" && evaluate_arm "$id"; then
      echo "[$(stamp)] done: $name seed $seed → greedy accuracy $(accuracy "$OUT/$id/eval_cmexam_val/headroom_report.json")"
    else
      echo "[$(stamp)] FAILED: $name seed $seed (see logs/${id}_*.log)" >&2
      echo "$(stamp)" >> "$LOCKS/$id.failed"
    fi
  ) &
}

echo "[$(stamp)] queue started; plan: $STAGE5_ENV (re-read every ${QUEUE_POLL}s); start adapter $START"
last_state=""; failed_logged=""
while true; do
  load_plan
  running=0; waiting=0; startable=()
  for job in $PLAN; do
    parse "$job"
    job_done && continue
    if job_running; then running=$((running + 1)); continue; fi
    job_skipped && continue
    if job_failed_out; then
      case " $failed_logged " in *" $id "*) ;; *) echo "[$(stamp)] leaving out $job after $MAX_FAILS failures ($LOCKS/$id.failed)"; failed_logged="$failed_logged $id";; esac
      continue
    fi
    if job_ready; then startable+=("$job"); else waiting=$((waiting + 1)); fi
  done
  state="running $running, startable ${#startable[@]}, waiting $waiting, parallel $PARALLEL"
  [ "$state" != "$last_state" ] && echo "[$(stamp)] $state" && last_state=$state
  if [ -f "$OUT/stage5.stop" ]; then
    [ "$running" -eq 0 ] && { echo "[$(stamp)] $OUT/stage5.stop present and nothing running: stopping"; break; }
    startable=()
  fi
  if [ "${#startable[@]}" -eq 0 ] && [ "$running" -eq 0 ]; then
    [ "$waiting" -gt 0 ] && echo "[$(stamp)] $waiting job(s) wait for adapters no queued job will produce"
    break
  fi
  for job in ${startable[@]+"${startable[@]}"}; do
    [ "$running" -ge "$PARALLEL" ] && break
    echo "[$(stamp)] starting $job"
    start_job "$job"
    running=$((running + 1))
    sleep "$STAGGER"
  done
  sleep "$QUEUE_POLL"
done
wait || true

# paired comparisons with standard GRPO per seed, on each benchmark, then the cross-seed summaries
names=""; seeds=""
for job in $PLAN; do
  parse "$job"
  [ "$kind" = train ] || continue
  names="$names $name"; seeds="$seeds $seed"
done
names=$(echo $names | tr ' ' '\n' | awk 'NF && !seen[$0]++')
seeds=$(echo $seeds | tr ' ' '\n' | awk 'NF && !seen[$0]++')
for bench in eval_cmexam_val eval_cmb_test; do
  tag=""; [ "$bench" = eval_cmb_test ] && tag="_cmb"
  for seed in $seeds; do
    for name in $names; do
      [ "$name" = standard ] && continue
      if [ -f "$(records standard "$seed" "$bench")" ] && [ -f "$(records "$name" "$seed" "$bench")" ]; then
        $PY scripts/compare_headroom_records.py --a "$(records standard "$seed" "$bench")" --b "$(records "$name" "$seed" "$bench")" \
          --label-a "standard GRPO" --label-b "$name" --json "$OUT/stage5${tag}_standard_vs_${name}_seed${seed}.json" > /dev/null || true
      fi
    done
  done
  present=""; compares=()
  for name in $names; do
    for seed in $seeds; do
      if [ -f "$(records "$name" "$seed" "$bench")" ]; then
        present="$present $name"
        [ "$name" != standard ] && compares+=(--compare "$name:standard")
        break
      fi
    done
  done
  if [ -n "$present" ]; then
    $PY scripts/aggregate_seeds.py --pattern "$OUT/stage5_{arm}_seed{seed}/$bench/headroom_records.jsonl" \
      --arms "$(echo $present | tr ' ' ',')" --seeds "$(echo $seeds)" ${compares[@]+"${compares[@]}"} \
      --json "$OUT/stage5${tag}_seeds.json" > /dev/null || true
  fi
done
echo "[$(stamp)] queue finished"
