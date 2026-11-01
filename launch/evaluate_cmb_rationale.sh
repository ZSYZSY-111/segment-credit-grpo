#!/usr/bin/env bash
# Evaluate a rationale-mode adapter on the official CMB-Exam test split, with the same per-question records as the
# CMExam validation runs, so RL arms can be compared on a second benchmark:
#   bash launch/evaluate_cmb_rationale.sh <adapter-dir> <run-name> > logs/<run-name>_cmb_test_eval.log 2>&1
# Results go to $OUT/<run-name>/eval_cmb_test; a finished evaluation is not repeated. The messages file is built once
# from the official question and answer JSON (scripts/make_cmb_test_messages.py), under a lock so that evaluations
# started together do not race. About 35 minutes per adapter on one RTX 5090 (11,200 questions).
set -euo pipefail
source "$(dirname "$0")/common.sh"
ADAPTER=$1; NAME=$2
CMB_MSG=${CMB_MSG:-$DATA/cmb_test_messages.jsonl}
[ -f "$OUT/$NAME/eval_cmb_test/headroom_report.json" ] && { echo "already evaluated: $OUT/$NAME/eval_cmb_test"; exit 0; }
mkdir -p "$(dirname "$CMB_MSG")"
(
  flock 8
  if [ ! -f "$CMB_MSG.ok" ]; then
    $PY scripts/make_cmb_test_messages.py --questions "$DATA/CMB/CMB-Exam/CMB-test/CMB-test-choice-question-merge.json" \
      --answers "$DATA/CMB/CMB-test-choice-answer.json" --output "$CMB_MSG" && touch "$CMB_MSG.ok"
  fi
) 8> "$CMB_MSG.lock"
[ -f "$CMB_MSG.ok" ] || { echo "could not build $CMB_MSG" >&2; exit 1; }
$PY -u scripts/eval_validation.py --model "$BASE" --adapter "$ADAPTER" --input-file "$CMB_MSG" \
  --output-dir "$OUT/$NAME/eval_cmb_test" --mode cot --prompt-style legacy_explain --limit 0 --batch-size "${EVAL_BATCH:-64}" \
  --max-new-tokens 768 --skip-sampling --skip-shuffle --seed 42 --overwrite
