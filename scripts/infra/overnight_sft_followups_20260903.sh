#!/usr/bin/env bash

set -u
set -o pipefail

ROOT=/workspace/mtpatcher
PROJECT="$ROOT/repo/MT-Patcher-Reproduction-Ascend"
VERL="$ROOT/repo/verl-v0.9.0"

VERL_ENV="$ROOT/envs/verl-v0.9.0-a3"
LEGACY_PY="$ROOT/envs/mtpatcher-npu-py311/bin/python"

BASE_CFG="$PROJECT/configs/sft/human6565_qwen3_06b_legacy_like_v2.yaml"

EVAL="$PROJECT/scripts/pilot_v2/eval_qwen3_06b_mt_ascend.py"
SCORE="$PROJECT/scripts/pilot_v2/score_mt_jsonl.py"

TOKENIZER="$ROOT/models/Qwen3-0.6B"
DATA="$ROOT/data/pilot_v2_qwen3_06b"

BASE_V2_RUN="$ROOT/runs/sft/human6565_qwen3_06b_legacy_like_v2_20260902_162558"
BASE_V2_EVAL="$ROOT/runs/eval/legacy_like_v2_human6565_20260902"

STAMP=$(date +%Y%m%d_%H%M%S)
OVERNIGHT="$ROOT/runs/overnight/sft_followups_$STAMP"
EVALROOT="$ROOT/runs/eval/sft_followups_$STAMP"

mkdir -p \
  "$OVERNIGHT" \
  "$OVERNIGHT/configs" \
  "$EVALROOT"

source "$VERL_ENV/bin/activate"

echo "$OVERNIGHT" > /tmp/mtpatcher_overnight_sft_followups
echo "$EVALROOT" > /tmp/mtpatcher_overnight_sft_evalroot


###############################################################################
# MANIFEST
###############################################################################

{
  echo "timestamp=$(date -Is)"
  echo "project=$PROJECT"
  echo "base_cfg=$BASE_CFG"

  echo -n "project_commit="
  git -C "$PROJECT" rev-parse HEAD 2>/dev/null || true

  echo -n "verl_commit="
  git -C "$VERL" rev-parse HEAD 2>/dev/null || true

  echo
  echo "=== BASE CONFIG SHA ==="
  sha256sum "$BASE_CFG" 2>/dev/null || true

  echo
  echo "=== PROJECT STATUS ==="
  git -C "$PROJECT" status --short || true

  echo
  echo "=== NPU ==="
  python - <<'PY'
import torch
print("npu_available =", torch.npu.is_available())
print("npu_count =", torch.npu.device_count())
PY
} > "$OVERNIGHT/manifest.txt" 2>&1


###############################################################################
# SAFETY GATE
###############################################################################

if [[ ! -f "$BASE_CFG" ]]; then
  echo "FATAL: missing base config: $BASE_CFG"
  exit 2
fi

COUNT=$(
python - <<'PY'
import torch
print(torch.npu.device_count())
PY
)

if [[ "$COUNT" -ne 16 ]]; then
  echo "FATAL: expected 16 NPUs, got $COUNT"
  exit 2
fi

echo "OVERNIGHT_PRECHECK_PASS"


###############################################################################
# CONFIG BUILDER
#
# This only edits Hydra config.
# It contains ZERO training implementation.
###############################################################################

make_cfg() {
  local OUTCFG="$1"
  local NAME="$2"
  local NPROC="$3"
  local MICRO="$4"
  local SCHED="$5"
  shift 5

  python - \
    "$BASE_CFG" \
    "$OUTCFG" \
    "$NAME" \
    "$NPROC" \
    "$MICRO" \
    "$SCHED" \
    "$@" <<'PY'

import sys
import yaml

src = sys.argv[1]
dst = sys.argv[2]
name = sys.argv[3]
nproc = int(sys.argv[4])
micro = int(sys.argv[5])
sched = sys.argv[6]
extras = sys.argv[7:]

cfg = yaml.safe_load(
    open(src, encoding="utf-8")
)

cfg["name"] = name
cfg["nproc_per_node"] = nproc

ovs = list(cfg["overrides"])


def norm_key(item):
    k = item.split("=", 1)[0]
    return k.lstrip("+")


def set_override(item):
    global ovs

    target = norm_key(item)

    ovs = [
        x for x in ovs
        if norm_key(x) != target
    ]

    ovs.append(item)


set_override(
    f"trainer.experiment_name={name}"
)

set_override(
    f"trainer.n_gpus_per_node={nproc}"
)

set_override(
    f"data.micro_batch_size_per_gpu={micro}"
)

set_override(
    f"optim.lr_scheduler_type={sched}"
)

# Freeze the already-probed setting.
set_override(
    "engine.use_torch_compile=false"
)

for x in extras:
    set_override(x)

cfg["overrides"] = ovs

with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(
        cfg,
        f,
        allow_unicode=True,
        sort_keys=False,
    )

print("CONFIG_WRITTEN", dst)
PY
}


###############################################################################
# EVALUATION
###############################################################################

evaluate_model() {
  local MODEL="$1"
  local NAME="$2"

  local OUT="$EVALROOT/$NAME"

  mkdir -p "$OUT/metrics"

  echo
  echo "================================================================"
  echo "EVALUATE: $NAME"
  echo "MODEL=$MODEL"
  echo "================================================================"

  if [[ ! -s "$MODEL/model.safetensors" ]]; then
    echo "ERROR: missing merged model: $MODEL/model.safetensors"
    return 1
  fi

  for SPEC in \
    "wmt24:$DATA/wmt24_zh_en998.jsonl" \
    "flores:$DATA/flores_zh_en1012.jsonl" \
    "challenge:$DATA/challenge_zh_en197.jsonl"
  do
    local SET="${SPEC%%:*}"
    local INPUT="${SPEC#*:}"

    python "$EVAL" \
      --model "$MODEL" \
      --tokenizer "$TOKENIZER" \
      --input "$INPUT" \
      --output "$OUT/${SET}.jsonl" \
      --method "$NAME" \
      --batch-size 16 \
      --max-new-tokens 256 \
      --attn-implementation sdpa \
    || return 1
  done

  "$LEGACY_PY" - <<'PY'
import sacrebleu
assert sacrebleu.__version__ == "2.5.1"
print("SACREBLEU =", sacrebleu.__version__)
PY

  for SET in wmt24 flores challenge
  do
    "$LEGACY_PY" "$SCORE" \
      --input "$OUT/${SET}.jsonl" \
      --output "$OUT/metrics/${SET}.json" \
    || return 1
  done

  echo "EVALUATION_PASS: $NAME"

  return 0
}


###############################################################################
# MERGE
###############################################################################

merge_step() {
  local RUN="$1"
  local STEP="$2"

  local CKPT="$RUN/checkpoints/global_step_$STEP"
  local MODEL="$RUN/export_hf/global_step_$STEP"

  if [[ -s "$MODEL/model.safetensors" ]]; then
    echo "MERGE_ALREADY_DONE: $MODEL"
    echo "$MODEL"
    return 0
  fi

  if [[ ! -d "$CKPT" ]]; then
    echo "ERROR: checkpoint missing: $CKPT"
    return 1
  fi

  mkdir -p "$MODEL"

  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$CKPT" \
    --target_dir "$MODEL" \
  || return 1

  if [[ ! -s "$MODEL/model.safetensors" ]]; then
    echo "ERROR: merger completed but model missing: $MODEL"
    return 1
  fi

  echo "MERGE_PASS: $MODEL"

  return 0
}


###############################################################################
# RUN A VERL ARM
###############################################################################

run_arm() {
  local NAME="$1"
  local NPROC="$2"
  local MICRO="$3"
  local SCHED="$4"
  local DO_EVAL="$5"

  shift 5

  local CFG="$OVERNIGHT/configs/${NAME}.yaml"
  local RUN="$OVERNIGHT/$NAME"

  mkdir -p "$RUN/checkpoints"

  make_cfg \
    "$CFG" \
    "$NAME" \
    "$NPROC" \
    "$MICRO" \
    "$SCHED" \
    "$@" \
  || return 1

  readarray -t OVS < <(
    python - "$CFG" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(
    open(sys.argv[1], encoding="utf-8")
)

for x in cfg["overrides"]:
    print(x)
PY
  )

  OVS+=(
    "trainer.default_local_dir=$RUN/checkpoints"
  )

  {
    echo "name=$NAME"
    echo "nproc=$NPROC"
    echo "micro=$MICRO"
    echo "scheduler=$SCHED"
    printf 'override=%s\n' "${OVS[@]}"
  } > "$RUN/run_contract.txt"

  echo
  echo "################################################################"
  echo "TRAIN: $NAME"
  echo "nproc=$NPROC micro=$MICRO scheduler=$SCHED"
  echo "################################################################"

  set +e

  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NPROC" \
    -m verl.trainer.sft_trainer \
    "${OVS[@]}" \
    2>&1 | tee "$RUN/train.log"

  STATUS=${PIPESTATUS[0]}

  set -e 2>/dev/null || true
  set +e

  echo "$STATUS" > "$RUN/train_exit_status.txt"

  if [[ "$STATUS" -ne 0 ]]; then
    echo "TRAIN_FAIL: $NAME status=$STATUS"
    return "$STATUS"
  fi

  if [[ ! -d "$RUN/checkpoints/global_step_1230" ]]; then
    echo "TRAIN_FAIL: global_step_1230 missing for $NAME"
    return 1
  fi

  echo "TRAIN_PASS: $NAME"

  if [[ "$DO_EVAL" != "yes" ]]; then
    echo "TRAIN_ONLY_ARM_COMPLETE: $NAME"
    return 0
  fi

  merge_step "$RUN" 1230 || return 1

  local MODEL="$RUN/export_hf/global_step_1230"

  evaluate_model "$MODEL" "$NAME" || return 1

  return 0
}


###############################################################################
# EXISTING V2 EPOCH DIAGNOSTIC
###############################################################################

eval_existing_v2_epoch() {
  local STEP="$1"
  local NAME="$2"

  local CKPT="$BASE_V2_RUN/checkpoints/global_step_$STEP"

  if [[ ! -d "$CKPT" ]]; then
    echo "SKIP: existing v2 checkpoint absent: $CKPT"
    return 0
  fi

  merge_step "$BASE_V2_RUN" "$STEP" || return 1

  evaluate_model \
    "$BASE_V2_RUN/export_hf/global_step_$STEP" \
    "$NAME" \
  || return 1

  return 0
}


###############################################################################
# MASTER EXECUTION
###############################################################################

RESULTS="$OVERNIGHT/master_results.txt"

touch "$RESULTS"

mark() {
  echo "$(date -Is) $*" | tee -a "$RESULTS"
}


###############################################################################
# 0. Snapshot existing V2 E3
###############################################################################

mark "START snapshot_existing_v2_e3"

mkdir -p "$EVALROOT/v16_const_existing/metrics"

for SET in wmt24 flores challenge
do
  SRC="$BASE_V2_EVAL/metrics/${SET}.json"

  if [[ -s "$SRC" ]]; then
    cp "$SRC" \
      "$EVALROOT/v16_const_existing/metrics/${SET}.json"
  fi
done

mark "DONE snapshot_existing_v2_e3"


###############################################################################
# 1. V16 cosine
#
# Main scheduler-shape probe.
# Still 100% Verl-native.
###############################################################################

mark "START v16_cosine"

if run_arm \
  v16_cosine \
  16 \
  1 \
  cosine \
  yes
then
  mark "PASS v16_cosine"
else
  mark "FAIL v16_cosine"
fi


###############################################################################
# 2. V16 one-factor TRAIN-ONLY probes
#
# Deliberately NO test-set evaluation tonight.
###############################################################################

mark "START v16_fp32master_trainonly"

if run_arm \
  v16_fp32master_trainonly \
  16 \
  1 \
  constant \
  no \
  "engine.model_dtype=fp32"
then
  mark "PASS v16_fp32master_trainonly"
else
  mark "FAIL v16_fp32master_trainonly"
fi


mark "START v16_remove_padding_on_trainonly"

if run_arm \
  v16_remove_padding_on_trainonly \
  16 \
  1 \
  constant \
  no \
  "model.use_remove_padding=true"
then
  mark "PASS v16_remove_padding_on_trainonly"
else
  mark "FAIL v16_remove_padding_on_trainonly"
fi


mark "START v16_gradckpt_on_trainonly"

if run_arm \
  v16_gradckpt_on_trainonly \
  16 \
  1 \
  constant \
  no \
  "model.enable_gradient_checkpointing=true"
then
  mark "PASS v16_gradckpt_on_trainonly"
else
  mark "FAIL v16_gradckpt_on_trainonly"
fi


###############################################################################
# 3. Existing V2 E1 / E2 diagnostic
#
# Diagnostic only. Do NOT pick best checkpoint from test.
###############################################################################

mark "START v2_epoch1_diagnostic"

if eval_existing_v2_epoch \
  410 \
  v16_const_existing_e1
then
  mark "PASS v2_epoch1_diagnostic"
else
  mark "FAIL v2_epoch1_diagnostic"
fi


mark "START v2_epoch2_diagnostic"

if eval_existing_v2_epoch \
  820 \
  v16_const_existing_e2
then
  mark "PASS v2_epoch2_diagnostic"
else
  mark "FAIL v2_epoch2_diagnostic"
fi


###############################################################################
# 4. V1-full long-run
#
# Same Verl trainer, same global batch 16.
# world=1, micro=16.
#
# Isolates:
#   V1-full vs V16
###############################################################################

mark "START v1_full_const"

if run_arm \
  v1_full_const \
  1 \
  16 \
  constant \
  yes
then
  mark "PASS v1_full_const"
else
  mark "FAIL v1_full_const"
fi


###############################################################################
# 5. V1-micro long-run
#
# world=1, micro=1.
#
# Isolates:
#   V1-full vs V1-micro
###############################################################################

mark "START v1_micro_const"

if run_arm \
  v1_micro_const \
  1 \
  1 \
  constant \
  yes
then
  mark "PASS v1_micro_const"
else
  mark "FAIL v1_micro_const"
fi


###############################################################################
# FINAL SUMMARY
###############################################################################

mark "START final_summary"

python - "$EVALROOT" <<'PY' \
  | tee "$OVERNIGHT/final_metrics_summary.txt"

import json
import sys
from pathlib import Path

root = Path(sys.argv[1])

base_root = Path(
    "/workspace/mtpatcher/runs/eval/"
    "canonical_human6565_20260902/metrics"
)

sets = [
    "wmt24",
    "flores",
    "challenge",
]


def load_metric(path):
    d = json.loads(
        path.read_text(encoding="utf-8")
    )

    return (
        float(d["BLEU"]),
        float(d["chrF"]),
    )


base = {}

for s in sets:
    p = base_root / f"base_{s}.json"

    if p.exists():
        base[s] = load_metric(p)


print()
print("============================================================")
print("OVERNIGHT DOWNSTREAM SUMMARY")
print("============================================================")

if len(base) == 3:
    bb = sum(base[s][0] for s in sets) / 3
    bc = sum(base[s][1] for s in sets) / 3

    print(
        f"{'BASE':32s} "
        f"BLEU={bb:.6f} "
        f"chrF={bc:.6f}"
    )

else:
    bb = None
    bc = None

for arm in sorted(root.iterdir()):
    metrics = arm / "metrics"

    if not metrics.is_dir():
        continue

    vals = {}

    ok = True

    for s in sets:
        p = metrics / f"{s}.json"

        if not p.exists():
            ok = False
            break

        vals[s] = load_metric(p)

    if not ok:
        continue

    avg_bleu = sum(
        vals[s][0] for s in sets
    ) / 3

    avg_chrf = sum(
        vals[s][1] for s in sets
    ) / 3

    if bb is None:
        print(
            f"{arm.name:32s} "
            f"BLEU={avg_bleu:.6f} "
            f"chrF={avg_chrf:.6f}"
        )

    else:
        print(
            f"{arm.name:32s} "
            f"BLEU={avg_bleu:.6f} "
            f"dBLEU={avg_bleu-bb:+.6f} "
            f"chrF={avg_chrf:.6f} "
            f"dchrF={avg_chrf-bc:+.6f}"
        )

    for s in sets:
        b, c = vals[s]

        print(
            f"    {s:10s} "
            f"BLEU={b:.6f} "
            f"chrF={c:.6f}"
        )

print()
print("SUMMARY_PASS")
PY

mark "DONE final_summary"

echo
echo "################################################################"
echo "OVERNIGHT MASTER COMPLETE"
echo "################################################################"
echo "OVERNIGHT=$OVERNIGHT"
echo "EVALROOT=$EVALROOT"
echo "RESULTS=$RESULTS"
echo "SUMMARY=$OVERNIGHT/final_metrics_summary.txt"
