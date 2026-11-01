#!/usr/bin/env bash
# One-shot setup and run on a fresh AutoDL instance: packages, base model, datasets, data files, the Stage 4 start
# adapter, then Stage 5 (standard vs segment-credit GRPO) for every seed in SEEDS. Every step is skipped when its
# output already exists, so after an interruption just run it again and it continues (Stage 5 resumes from checkpoints).
#
#   cd /root/autodl-tmp && git clone -b claude/vibrant-planck-emxeh5 https://github.com/ZSYZSY-111/segment-credit-grpo.git
#   cd segment-credit-grpo && mkdir -p logs && SEEDS="42 7" nohup bash launch/autodl_bootstrap.sh > logs/bootstrap.log 2>&1 &
#   tail -f logs/bootstrap.log
#
# Everything (model, data, caches, outputs) lives under the repository directory, so clone it on the data disk.
# Override any of SEEDS, BASE, DATA, OUT, GEN_CHUNK, PROBE_BATCH from the environment.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
export SEEDS=${SEEDS:-"42 7"}
export DATA=${DATA:-data}
export OUT=${OUT:-training_outputs}
MODELS=${MODELS:-models}
export BASE=${BASE:-$MODELS/Qwen2.5-7B-Instruct}
export HF_HOME=${HF_HOME:-$REPO/hf_home}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export GEN_CHUNK=${GEN_CHUNK:-128}      # 96 GB: generate a whole step at once (32 on the 32 GB card)
export PROBE_BATCH=${PROBE_BATCH:-128}  # probe forward passes per batch (64 on the 32 GB card)
mkdir -p logs "$MODELS" "$DATA" "$HF_HOME"
log() { echo "[$(date '+%F %T')] $*"; }
exec 9> logs/.bootstrap.lock
flock -n 9 || { log "another bootstrap is already running; follow it with: tail -f logs/bootstrap.log"; exit 1; }
# a non-interactive ssh command skips ~/.bashrc, so conda's python can be missing from PATH: add the usual locations
for d in /root/miniconda3/bin /opt/conda/bin /root/anaconda3/bin /usr/local/miniconda3/bin /usr/local/conda/bin; do
  if [ -x "$d/python" ]; then case ":$PATH:" in *":$d:"*) ;; *) export PATH="$d:$PATH";; esac; break; fi
done
# a bare image may have python3 but no python, no pip, or no git
PY=${PY:-$(command -v python || command -v python3 || true)}
[ -n "$PY" ] || { log "no python found: apt-get update && apt-get install -y python3 python3-pip git, or pick a Miniconda image"; exit 1; }
export PY
"$PY" -c 'import sys; v = sys.version_info; assert v >= (3, 10), f"python {v.major}.{v.minor} is too old: conda create -n sc python=3.12 && conda activate sc, then re-run"'
"$PY" -m pip --version >/dev/null 2>&1 || "$PY" -m ensurepip --upgrade >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq python3-pip; }
command -v git >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq git; }
PIP="$PY -m pip"
log "python: $PY ($("$PY" --version 2>&1))"
turbo_on()  { [ -f /etc/network_turbo ] && source /etc/network_turbo || true; }   # AutoDL academic acceleration (GitHub, HF)
turbo_off() { unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true; }
CMB_TEST=$DATA/CMB/CMB-Exam/CMB-test/CMB-test-choice-question-merge.json

log "repo $REPO; seeds: $SEEDS"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
df -h "$REPO" | tail -1

# ---------------------------------------------------------------- 1. python packages
log "step 1: python packages"
turbo_off
if ! $PY - <<'EOF' 2>/dev/null
import sys, torch
ok = torch.cuda.is_available() and any(a.endswith("120") for a in torch.cuda.get_arch_list())
sys.exit(0 if ok else 1)
EOF
then
  log "installing torch 2.8.0 (Blackwell needs sm_120 kernels; the PyPI Linux wheel is the CUDA 12.8 build)"
  $PIP install -q torch==2.8.0 || $PIP install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
fi
$PIP install -q "transformers==5.13.1" "peft==0.19.1" "trl==1.8.0" "datasets==5.0.0" "accelerate>=1.0.0,<2.0.0" \
    safetensors datasketch scipy tqdm huggingface_hub modelscope \
  || { log "exact pins failed; falling back to requirements.txt ranges"; $PIP install -q -r requirements.txt modelscope; }
$PY -c "import torch, transformers, peft, trl, datasets; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0)); print('transformers', transformers.__version__, 'peft', peft.__version__, 'trl', trl.__version__, 'datasets', datasets.__version__)"
if $PY -m unittest discover -s tests -t . > logs/unittest.log 2>&1; then
  log "unit tests: $(grep -E '^(OK|FAILED)' logs/unittest.log | tail -1)"
else
  log "WARNING: unit tests failed, continuing; full output in logs/unittest.log"
  grep -E '^(FAIL|ERROR):|^(OK|FAILED)' logs/unittest.log || true
fi

# ---------------------------------------------------------------- 2. base model
log "step 2: base model -> $BASE"
model_complete() {
  $PY - "$BASE" <<'EOF'
import json, os, sys
d = sys.argv[1]
index = os.path.join(d, "model.safetensors.index.json")
if not os.path.isfile(os.path.join(d, "config.json")):
    sys.exit(1)
if os.path.isfile(index):
    shards = set(json.load(open(index))["weight_map"].values())
    sys.exit(0 if all(os.path.isfile(os.path.join(d, f)) for f in shards) else 1)
sys.exit(0 if os.path.isfile(os.path.join(d, "model.safetensors")) else 1)
EOF
}
if ! model_complete; then
  modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir "$BASE" \
    || $PY -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='$BASE')"
fi
model_complete || { log "model download incomplete: re-run this script to resume"; exit 1; }
du -sh "$BASE"
# machine-local defaults for every launcher (explicit environment variables still win); offline from here on
cat > launch/local.env <<EOF
PY=\${PY:-$PY}
BASE=\${BASE:-$BASE}
DATA=\${DATA:-$DATA}
OUT=\${OUT:-$OUT}
GEN_CHUNK=\${GEN_CHUNK:-$GEN_CHUNK}
PROBE_BATCH=\${PROBE_BATCH:-$PROBE_BATCH}
export HF_HOME=\${HF_HOME:-$HF_HOME}
export HF_ENDPOINT=\${HF_ENDPOINT:-$HF_ENDPOINT}
export HF_HUB_OFFLINE=\${HF_HUB_OFFLINE:-1}
EOF
log "wrote launch/local.env"

# ---------------------------------------------------------------- 3. datasets
log "step 3: datasets"
if [ ! -f "$DATA/CMExam/data/train.csv" ]; then
  turbo_on; git clone --depth 1 https://github.com/williamliujl/CMExam "$DATA/CMExam"; turbo_off
fi
if [ ! -f "$CMB_TEST" ]; then
  [ -d "$DATA/CMB_repo" ] || { turbo_on; git clone --depth 1 https://github.com/FreedomIntelligence/CMB "$DATA/CMB_repo"; turbo_off; }
  mkdir -p "$DATA/CMB"
  $PY -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$DATA/CMB_repo/data/CMB.zip" "$DATA/CMB"
  [ -d "$DATA/CMB/CMB/CMB-Exam" ] && mv "$DATA/CMB/CMB/"* "$DATA/CMB/"   # zip with a top-level folder
  cp "$DATA/CMB_repo/data/CMB-test-choice-answer.json" "$DATA/CMB/"
fi
for f in "$DATA/CMExam/data/train.csv" "$DATA/CMExam/data/val.csv" "$DATA/CMExam/data/test_with_annotations.csv" \
         "$CMB_TEST" "$DATA/CMB/CMB-Exam/CMB-val/CMB-val-merge.json" "$DATA/CMB/CMB-test-choice-answer.json"; do
  [ -f "$f" ] || { log "missing $f; see data/README.md for the expected layout (a 1 KB file means Git LFS pointers: apt install git-lfs, then git lfs pull in that clone)"; exit 1; }
done

# ---------------------------------------------------------------- 4. data files
log "step 4: data files"
decontaminated() { $PY -c "import json, sys; sys.exit(0 if 'cmb_decontamination' in json.load(open(sys.argv[1])) else 1)" "$1" 2>/dev/null; }
if [ ! -f "$DATA/cmexam_sft_validation.jsonl" ] || ! decontaminated "$DATA/cmexam_sft_report.json"; then
  $PY scripts/build_cmexam_direct_sft.py --input-dir "$DATA/CMExam/data" --output-dir "$DATA"
  $PY scripts/decontaminate_cmexam_against_cmb.py --cmb-test-file "$CMB_TEST" --train-file "$DATA/cmexam_sft_train.jsonl" \
    --validation-file "$DATA/cmexam_sft_validation.jsonl" --report-file "$DATA/cmexam_sft_report.json"
fi
sha=$(sha256sum "$DATA/cmexam_sft_train.jsonl" | cut -c1-64)
case "$sha" in
  e15ff4a5c7d6*050a830) log "cmexam_sft_train.jsonl matches the recorded build ($sha)";;
  *) log "WARNING: cmexam_sft_train.jsonl sha256 $sha differs from the recorded e15ff4a5c7d6…050a830 (upstream CSVs changed?); continuing";;
esac
wc -l "$DATA/cmexam_sft_train.jsonl" "$DATA/cmexam_sft_validation.jsonl"
RAT=$DATA/cmexam_rationale
if [ ! -f "$RAT/cmexam_rationale_sft_train.jsonl" ] || ! decontaminated "$RAT/cmexam_rationale_sft_report.json"; then
  $PY scripts/build_cmexam_rationale_sft.py --input-dir "$DATA/CMExam/data" --output-dir "$RAT"
  $PY scripts/decontaminate_cmexam_against_cmb.py --cmb-test-file "$CMB_TEST" --train-file "$RAT/cmexam_rationale_sft_train.jsonl" \
    --validation-file "$RAT/cmexam_rationale_sft_validation.jsonl" --report-file "$RAT/cmexam_rationale_sft_report.json"
fi
SFT=$DATA/segment_scores/sft
if [ ! -f "$SFT/train.jsonl" ]; then   # scored rationale rows from the released segment scores (data/README.md)
  mkdir -p "$SFT"
  $PY - "$REPO/data/segment_scores/cmexam_rationale_segment_scores.jsonl.gz" "$RAT/cmexam_rationale_sft_train.jsonl" "$SFT" <<'EOF'
import gzip, hashlib, json, sys
scores_path, rows_path, out_dir = sys.argv[1:4]
scores = {}
for line in gzip.open(scores_path, "rt", encoding="utf-8"):
    row = json.loads(line); scores[row["sample_id"]] = row
out = {"train": open(f"{out_dir}/train.jsonl", "w", encoding="utf-8"), "dev": open(f"{out_dir}/dev.jsonl", "w", encoding="utf-8")}
written = {"train": 0, "dev": 0}
for line in open(rows_path, encoding="utf-8"):
    row = json.loads(line)
    sid = hashlib.sha256("\n".join(m["content"] for m in row["messages"] if m["role"] == "user").encode("utf-8")).hexdigest()
    if sid not in scores:
        continue
    content = row["messages"][-1]["content"]
    explanation = content[len("解析："):content.rindex("\n答案：")]
    assert len(explanation) == scores[sid]["rationale_chars"], sid
    split = scores[sid]["split"]
    out[split].write(json.dumps({"sample_id": sid, "task": "sentence_score", "messages": row["messages"], "explanation": explanation,
        "gold": content.rsplit("答案：", 1)[-1].strip(), "sentences": scores[sid]["segments"]}, ensure_ascii=False) + "\n")
    written[split] += 1
print("segment-scored rows:", written, "(recorded: 9484 train / 500 dev)")
EOF
fi
POOL=$DATA/rl_cot_pool
if [ ! -f "$POOL/validation.jsonl" ]; then
  mkdir -p "$POOL"
  $PY scripts/build_cot_rl_pool.py --input "$RAT/cmexam_rationale_sft_train.jsonl" --exclude "$SFT/train.jsonl" "$SFT/dev.jsonl" \
    --limit 4000 --max-user-chars 450 --output "$POOL/train.jsonl"
  $PY scripts/build_cot_rl_pool.py --input "$RAT/cmexam_rationale_sft_validation.jsonl" --limit 256 --seed 7 --output "$POOL/validation.jsonl"
fi
wc -l "$SFT/train.jsonl" "$SFT/dev.jsonl" "$POOL/train.jsonl" "$POOL/validation.jsonl"

# ---------------------------------------------------------------- 5. stage 4: the weighted SFT adapter the RL arms start from
log "step 5: stage 4 weighted adapter (seed 42)"
START=$OUT/stage4_weighted_seed42/final_adapter
if [ ! -f "$START/adapter_config.json" ]; then
  ARMS="weighted:0:score" SEEDS=42 bash launch/stage4_segment_weighted_sft.sh
elif [ ! -f "$OUT/stage4_weighted_seed42/eval_cmexam_val/headroom_report.json" ]; then
  $PY -u scripts/eval_validation.py --model "$BASE" --adapter "$START" --input-file "$DATA/cmexam_sft_validation.jsonl" \
    --output-dir "$OUT/stage4_weighted_seed42/eval_cmexam_val" --mode cot --prompt-style legacy_explain --limit 0 --batch-size 64 \
    --max-new-tokens 768 --skip-sampling --skip-shuffle --seed 42 --overwrite 2>&1 | tee logs/stage4_weighted_seed42_eval.log
fi

# ---------------------------------------------------------------- 6. stage 5: both GRPO arms per seed, then the cross-seed summary
if [ -z "${PARALLEL:-}" ]; then   # jobs sharing the GPU at once, from its memory (~25 GB per job)
  mem_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
  if [ "${mem_mib:-0}" -ge 90000 ]; then PARALLEL=3; elif [ "${mem_mib:-0}" -ge 45000 ]; then PARALLEL=2; else PARALLEL=1; fi
fi
export PARALLEL
log "step 6: stage 5 for seeds: $SEEDS (parallel $PARALLEL; $OUT/stage5.env overrides the plan if present)"
START=$START bash launch/stage5_segment_credit_grpo.sh
log "done. results: $OUT/stage5_seeds.json and $OUT/stage5_standard_vs_segment_credit_seed*.json"
