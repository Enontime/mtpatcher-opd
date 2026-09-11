#!/usr/bin/env bash

# Intentionally NOT fail-fast.
# Failures are recorded in status files and printed,
# but this wrapper never calls exit and never intentionally
# terminates the caller's SSH shell.
set +e
set +u
set +o pipefail 2>/dev/null

PROJECT=/workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend

CONFIG=$PROJECT/configs/sft/canonical_seqkd_broad20k_qwen3_06b_8b.yaml
COMPOSER=$PROJECT/scripts/infra/compose_verl_sft_config.py

RUN_ROOT=/workspace/mtpatcher/runs/sft
STAMP=$(date +%Y%m%d_%H%M%S)

FALLBACK_NAME=canonical_seqkd_broad20k_qwen3_06b_8b


show_infra_state() {
  local label="$1"

  echo "=== INFRA $label ==="

  echo -n "zombies="
  ps -eo stat= |
    awk '$1 ~ /^Z/ {n++} END {print n+0}'

  echo -n "processes="
  ps -e -o pid= | wc -l

  local f=/sys/fs/cgroup/pids/pids.current

  if [[ -r "$f" ]]; then
    echo "pids_file=$f"
    echo "pids.current=$(cat "$f")"
  else
    echo "pids.current=UNAVAILABLE"
  fi
}


# ============================================================
# 1. READ EXPERIMENT METADATA
# ============================================================

META_OUTPUT="$(
python - "$CONFIG" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(
    open(
        sys.argv[1],
        encoding="utf-8",
    )
)

print(cfg["name"])
print(cfg["nproc_per_node"])
PY
)"

META_STATUS=$?

if [[ "$META_STATUS" -eq 0 ]]; then
  NAME="$(
    printf '%s\n' "$META_OUTPUT" |
      sed -n '1p'
  )"

  NPROC="$(
    printf '%s\n' "$META_OUTPUT" |
      sed -n '2p'
  )"

  echo "METADATA_READ_PASS"
else
  NAME="$FALLBACK_NAME"
  NPROC=16

  echo "METADATA_READ_FAIL status=$META_STATUS"
fi


RUN="$RUN_ROOT/${NAME}_${STAMP}"
mkdir -p "$RUN"

echo "$RUN" \
  > /tmp/mtpatcher_canonical_sft_run_dir

echo
echo "run=$RUN"


# ============================================================
# 2. LOAD STATIC OVERRIDES
# ============================================================

python - "$CONFIG" \
  > "$RUN/overrides.txt" \
<<'PY'
import sys
import yaml

cfg = yaml.safe_load(
    open(
        sys.argv[1],
        encoding="utf-8",
    )
)

for x in cfg["overrides"]:
    print(x)
PY

OVERRIDE_STATUS=$?

if [[ "$OVERRIDE_STATUS" -eq 0 ]]; then
  mapfile -t OVERRIDES \
    < "$RUN/overrides.txt"

  echo "OVERRIDES_LOAD_PASS"
else
  OVERRIDES=()

  echo \
    "OVERRIDES_LOAD_FAIL status=$OVERRIDE_STATUS"
fi


# Runtime output path stays launcher-owned.
OVERRIDES+=(
  "trainer.default_local_dir=$RUN/checkpoints"
)


# ============================================================
# 3. INITIAL INFRA STATE
# ============================================================

show_infra_state BEFORE |
  tee "$RUN/infra_before.txt"

cp "$CONFIG" \
  "$RUN/experiment_config.yaml" \
  2>/dev/null


# ============================================================
# 4. COMPOSE EXACT VERL CONFIG
# ============================================================

READY=1
FAIL_STATUS=1

if [[ "$META_STATUS" -ne 0 ]]; then
  READY=0
  FAIL_STATUS="$META_STATUS"
fi

if [[ "$OVERRIDE_STATUS" -ne 0 ]]; then
  READY=0
  FAIL_STATUS="$OVERRIDE_STATUS"
fi


if [[ "$READY" -eq 1 ]]; then

  python "$COMPOSER" \
    --config-dir \
    "/workspace/mtpatcher/repo/verl-v0.9.0/verl/trainer/config" \
    --output \
    "$RUN/resolved_config.yaml" \
    -- \
    "${OVERRIDES[@]}"

  COMPOSE_STATUS=$?

else
  COMPOSE_STATUS=99
fi


echo "$COMPOSE_STATUS" \
  > "$RUN/compose_exit_status.txt"


if [[ "$COMPOSE_STATUS" -eq 0 ]]; then
  echo "HYDRA_COMPOSE_CONFIG_PASS"
else
  echo \
    "HYDRA_COMPOSE_CONFIG_FAIL status=$COMPOSE_STATUS"

  READY=0
  FAIL_STATUS="$COMPOSE_STATUS"
fi


# ============================================================
# 5. CANONICAL BROAD20K SEQKD CONTRACT
# ============================================================

if [[ "$READY" -eq 1 ]]; then

  RUN="$RUN" python - <<'PY'
import os
from pathlib import Path

from omegaconf import (
    OmegaConf,
)


run = Path(
    os.environ["RUN"]
)

cfg = OmegaConf.load(
    run / "resolved_config.yaml"
)


def req(x, msg):
    if not x:
        raise AssertionError(msg)


EXPECTED_DATA = (
    "/workspace/mtpatcher/data/"
    "verl_science_broad20k/"
    "seqkd_broad20k.parquet"
)

EXPECTED_MODEL = (
    "/workspace/mtpatcher/models/"
    "Qwen3-0.6B"
)

EXPECTED_ADAPTER = (
    "/workspace/mtpatcher/repo/"
    "MT-Patcher-Reproduction-Ascend/"
    "scripts/data/"
    "verl_mt_response_sft_dataset.py"
)


# ------------------------------------------------------------
# Dataset identity / response-only semantics
# ------------------------------------------------------------

req(
    cfg.data.train_files == EXPECTED_DATA,
    "wrong training data",
)

req(
    cfg.data.val_files is None,
    "val_files must be null",
)

req(
    cfg.data.train_batch_size == 16,
    "wrong global batch",
)

req(
    cfg.data.micro_batch_size_per_gpu == 1,
    "wrong micro batch",
)

req(
    cfg.data.use_dynamic_bsz is False,
    "dynamic batch must be false",
)

req(
    cfg.data.messages_key == "messages",
    "wrong messages_key",
)

req(
    cfg.data.max_length == 1024,
    "wrong max_length",
)

req(
    cfg.data.truncation == "error",
    "truncation must fail closed",
)

req(
    cfg.data.custom_cls.path
    == EXPECTED_ADAPTER,
    "wrong dataset adapter path",
)

req(
    cfg.data.custom_cls.name
    == "MTResponseOnlySFTDataset",
    "wrong dataset adapter class",
)


# ------------------------------------------------------------
# Student / execution substrate
# ------------------------------------------------------------

req(
    cfg.model.path == EXPECTED_MODEL,
    "wrong student model",
)

req(
    cfg.model.lora_rank == 0,
    "formal SeqKD must be full fine-tuning",
)

req(
    cfg.model.use_remove_padding is False,
    "remove padding drift",
)

req(
    cfg.model.enable_gradient_checkpointing
    is False,
    "gradient checkpointing drift",
)

req(
    cfg.engine.ulysses_sequence_parallel_size
    == 1,
    "ulysses drift",
)

req(
    cfg.engine.model_dtype == "bfloat16",
    "wrong model dtype",
)

req(
    cfg.engine.dtype == "bfloat16",
    "wrong compute dtype",
)

req(
    cfg.engine.seed == 20260820,
    "wrong engine seed",
)

req(
    cfg.engine.use_torch_compile is False,
    "compile drift",
)

req(
    OmegaConf.select(
        cfg,
        "model.override_config."
        "attn_implementation",
    )
    == "sdpa",
    "attention implementation drift",
)


# ------------------------------------------------------------
# Optimizer / scheduler
# ------------------------------------------------------------

req(
    cfg.optim.optimizer == "AdamW",
    "wrong optimizer",
)

req(
    cfg.optim.optimizer_impl
    == "torch.optim",
    "wrong optimizer implementation",
)

req(
    abs(
        float(cfg.optim.lr)
        - 2e-5
    )
    < 1e-12,
    "wrong lr",
)

req(
    abs(
        float(cfg.optim.weight_decay)
        - 0.01
    )
    < 1e-12,
    "wrong weight decay",
)

req(
    abs(
        float(cfg.optim.clip_grad)
        - 1.0
    )
    < 1e-12,
    "wrong grad clip",
)

req(
    cfg.optim.lr_scheduler_type
    == "cosine",
    "wrong scheduler",
)

req(
    abs(
        float(
            cfg.optim.lr_warmup_steps_ratio
        )
        - 0.03
    )
    < 1e-12,
    "wrong warmup ratio",
)


# ------------------------------------------------------------
# Frozen 20k × 3 budget
# ------------------------------------------------------------

req(
    cfg.trainer.seed == 20260820,
    "wrong trainer seed",
)

req(
    cfg.trainer.nnodes == 1,
    "wrong nnodes",
)

req(
    cfg.trainer.n_gpus_per_node == 16,
    "wrong world size",
)

req(
    cfg.trainer.total_epochs == 3,
    "wrong source-pass count",
)

req(
    cfg.trainer.total_training_steps is None,
    "total steps must remain epoch-derived",
)

req(
    cfg.trainer.save_freq == 1250,
    "wrong checkpoint frequency",
)

req(
    cfg.trainer.test_freq == -1,
    "trainer-side test must remain disabled",
)

req(
    cfg.trainer.resume_mode == "disable",
    "formal run must start from frozen Base",
)

req(
    list(cfg.trainer.logger)
    == ["console", "tensorboard"],
    "wrong logger",
)


sources = 20_000
global_batch = 16
passes = 3

req(
    sources % global_batch == 0,
    "Broad20k does not divide batch",
)

updates_per_pass = (
    sources // global_batch
)

source_exposures = (
    sources * passes
)

total_updates = (
    updates_per_pass * passes
)

req(
    updates_per_pass == 1250,
    "wrong updates/pass",
)

req(
    source_exposures == 60000,
    "wrong source exposures",
)

req(
    total_updates == 3750,
    "wrong total updates",
)


print(
    "CANONICAL_SEQKD_BROAD20K_CONFIG_CONTRACT_PASS"
)

print(
    "EXPECTED_NATIVE_VERL:"
)

print(
    "  population_rows = 20000"
)

print(
    "  global_batch = 16"
)

print(
    "  native_drop_last = true"
)

print(
    "  consumed_rows_per_epoch = 20000"
)

print(
    "  expected_steps_per_epoch = 1250"
)

print(
    "  expected_total_steps = 3750"
)
PY

  CONTRACT_STATUS=$?

else
  CONTRACT_STATUS=99
fi


echo "$CONTRACT_STATUS" \
  > "$RUN/contract_exit_status.txt"


if [[ "$CONTRACT_STATUS" -eq 0 ]]; then
  echo \
    "CANONICAL_SEQKD_BROAD20K_CONTRACT_PASS"
else
  echo \
    "CANONICAL_SEQKD_BROAD20K_CONTRACT_FAIL status=$CONTRACT_STATUS"

  READY=0
  FAIL_STATUS="$CONTRACT_STATUS"
fi


# ============================================================
# 6. CONFIG-ONLY MODE
# ============================================================

if [[ "$READY" -eq 1 ]] &&
   [[ "${CONFIG_ONLY:-0}" == "1" ]]; then

  echo \
    "CANONICAL_SEQKD_BROAD20K_CONFIG_ONLY_PASS"

  echo "0" \
    > "$RUN/final_exit_status.txt"

  show_infra_state AFTER_CONFIG_ONLY \
    > "$RUN/infra_after.txt" 2>&1

  echo \
    "CONFIG_ONLY_FINISHED_NO_TRAINING"

fi


# ============================================================
# 7. REAL TRAINING
#
# Only reached when:
#   READY=1
#   CONFIG_ONLY != 1
# ============================================================

if [[ "$READY" -eq 1 ]] &&
   [[ "${CONFIG_ONLY:-0}" != "1" ]]; then

  cd "$RUN"

  echo
  echo \
    "=== START CANONICAL BROAD20K SEQKD ==="

  echo "run=$RUN"
  echo "nproc=$NPROC"

  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NPROC" \
    -m verl.trainer.sft_trainer \
    "${OVERRIDES[@]}" \
    2>&1 |
    tee "$RUN/train.log"

  TORCH_STATUS=${PIPESTATUS[0]}

  echo "$TORCH_STATUS" \
    > "$RUN/torchrun_exit_status.txt"

  echo "$TORCH_STATUS" \
    > "$RUN/final_exit_status.txt"

  if [[ "$TORCH_STATUS" -eq 0 ]]; then
    echo \
      "CANONICAL_SEQKD_BROAD20K_TRAIN_PASS"
  else
    echo \
      "CANONICAL_SEQKD_BROAD20K_TRAIN_FAIL status=$TORCH_STATUS"
  fi

  show_infra_state AFTER_TRAIN \
    > "$RUN/infra_after.txt" 2>&1

fi


# ============================================================
# 8. PRE-TRAIN FAILURE
# ============================================================

if [[ "$READY" -ne 1 ]]; then

  echo "$FAIL_STATUS" \
    > "$RUN/final_exit_status.txt"

  show_infra_state AFTER_PRETRAIN_FAIL \
    > "$RUN/infra_after.txt" 2>&1

  echo \
    "CANONICAL_SEQKD_BROAD20K_PRETRAIN_FAIL status=$FAIL_STATUS"

  echo \
    "TRAINING_SKIPPED"

fi


echo
echo "final_status_file=$RUN/final_exit_status.txt"

if [[ -f "$RUN/final_exit_status.txt" ]]; then
  echo -n "recorded_status="
  cat "$RUN/final_exit_status.txt"
fi

echo \
  "CANONICAL_SEQKD_WRAPPER_FINISHED"
