#!/usr/bin/env bash
# Stage 4: segment-level supervision for rationale SFT (docs/06_segment_weighted_sft.md).
# Three arms on the same 9,484 scored questions, LoRA r=16, 2 epochs; only the loss differs:
#   control   rationale LM loss only
#   predict   + auxiliary head regressing the teacher's segment score (weight 1)
#   weighted  segment scores as per-token loss weights (the RL starting point)
# SEEDS lists the training seeds (default 42, the recorded run); every arm and seed writes to stage4_<arm>_seed<seed>,
# and aggregate_seeds.py compares weighted and predict against control per seed and across seeds at the end.
# Each arm trains in about 1 hour on one RTX 5090 and is then evaluated in rationale mode on the CMExam validation split.
# Inputs: data/segment_scores/sft/{train,dev}.jsonl (see data/README.md, "Segment scores").
#   bash launch/stage4_segment_weighted_sft.sh                     # seed 42 only
#   SEEDS="42 7 2024" bash launch/stage4_segment_weighted_sft.sh   # three seeds
set -euo pipefail
source "$(dirname "$0")/common.sh"
SFT=$DATA/segment_scores/sft
SEEDS=${SEEDS:-42}
ARMS=${ARMS:-"control:0: predict:1.0: weighted:0:score"}
for seed in $SEEDS; do
  for arm in $ARMS; do
    IFS=: read -r name weight weighting <<< "$arm"
    run="stage4_${name}_seed${seed}"
    extra=()
    [ -n "${weighting:-}" ] && extra=(--sentence-weighting "$weighting")
    $PY -u scripts/train_sft_score_head.py --model "$BASE" --train-file "$SFT/train.jsonl" --dev-file "$SFT/dev.jsonl" \
      --output-dir "$OUT/$run" --score-loss-weight "$weight" "${extra[@]}" \
      --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --learning-rate 1e-5 --head-learning-rate 1e-3 --warmup-ratio 0.05 \
      --num-train-epochs 2 --per-device-train-batch-size 8 --gradient-accumulation-steps 4 --seed "$seed" 2>&1 | tee "logs/$run.log"
    # rationale-mode accuracy on the validation split, rendered with the prompt the rationale data was written in
    $PY -u scripts/eval_validation.py --model "$BASE" --adapter "$OUT/$run/final_adapter" \
      --input-file "$DATA/cmexam_sft_validation.jsonl" --output-dir "$OUT/$run/eval_cmexam_val" \
      --mode cot --prompt-style legacy_explain --limit 0 --batch-size 64 --max-new-tokens 768 \
      --skip-sampling --skip-shuffle --seed 42 --overwrite 2>&1 | tee "logs/${run}_eval.log"
  done
done
# paired comparisons against the control arm (when it ran), per seed and across seeds
names=(); for arm in $ARMS; do IFS=: read -r name _ <<< "$arm"; names+=("$name"); done
compares=()
if printf '%s\n' "${names[@]}" | grep -qx control; then
  for name in "${names[@]}"; do [ "$name" != control ] && compares+=(--compare "$name:control"); done
fi
$PY scripts/aggregate_seeds.py --pattern "$OUT/stage4_{arm}_seed{seed}/eval_cmexam_val/headroom_records.jsonl" \
  --arms "$(IFS=,; echo "${names[*]}")" --seeds "$SEEDS" ${compares[@]+"${compares[@]}"} --json "$OUT/stage4_seeds.json"
