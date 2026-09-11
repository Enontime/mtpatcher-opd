#!/usr/bin/env bash
set -euo pipefail

PROJECT=/workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend
MODEL=/workspace/mtpatcher/models/Qwen3-0.6B
DATA=/workspace/mtpatcher/data/verl_sft_qwen3_06b/human64.parquet
DATASET=$PROJECT/scripts/data/verl_mt_response_sft_dataset.py

RUN_ROOT=/workspace/mtpatcher/runs/sft
STAMP=$(date +%Y%m%d_%H%M%S)
RUN=$RUN_ROOT/gate4_qwen3_06b_$STAMP

mkdir -p "$RUN"

echo "$RUN" > /tmp/mtpatcher_gate4_run_dir

# ------------------------------------------------------------------
# One source of truth for BOTH config preflight and actual execution.
# ------------------------------------------------------------------

OVERRIDES=(
  "data.train_files=$DATA"
  "data.val_files=null"

  "data.train_batch_size=${GLOBAL_BATCH:-8}"
  "data.micro_batch_size_per_gpu=${MICRO_BATCH:-1}"
  "data.use_dynamic_bsz=false"

  "data.messages_key=messages"
  "data.enable_thinking_default=false"
  "data.pad_mode=no_padding"
  "data.max_length=1024"
  "data.truncation=error"
  "data.num_workers=0"
  "data.ignore_input_ids_mismatch=false"

  "+data.apply_chat_template_kwargs.enable_thinking=false"

  "data.custom_cls.path=$DATASET"
  "data.custom_cls.name=MTResponseOnlySFTDataset"

  "model.path=$MODEL"
  "model.lora_rank=0"
  "model.use_remove_padding=true"
  "model.enable_gradient_checkpointing=true"

  "engine=fsdp"
  "engine.ulysses_sequence_parallel_size=1"

  "optim.optimizer=AdamW"
  "optim.optimizer_impl=torch.optim"
  "optim.lr=2e-5"
  "optim.weight_decay=0.01"
  "optim.clip_grad=1.0"

  # Smoke only: use the simplest framework-native scheduler.
  "optim.lr_scheduler_type=constant"
  "optim.lr_warmup_steps_ratio=0.0"

  "trainer.project_name=mtpatcher-sft"
  "trainer.experiment_name=gate4-qwen3-06b"

  "trainer.default_local_dir=$RUN/checkpoints"

  'trainer.logger=["console","tensorboard"]'

  "trainer.total_epochs=${TOTAL_EPOCHS:-1}"
  "trainer.total_training_steps=${TOTAL_STEPS:-2}"

  "trainer.save_freq=${SAVE_FREQ:-2}"
  "trainer.test_freq=-1"

  "trainer.resume_mode=${RESUME_MODE:-disable}"
)

if [[ "${RESUME_MODE:-disable}" == "resume_path" ]]; then
  if [[ -z "${RESUME_FROM_PATH:-}" ]]; then
    echo "RESUME_FROM_PATH is required for resume_path mode" >&2
    exit 2
  fi

  OVERRIDES+=(
    "trainer.resume_from_path=$RESUME_FROM_PATH"
  )
fi

# ------------------------------------------------------------------
# Infra state
# ------------------------------------------------------------------

show_infra_state() {
  local label="$1"

  echo "=== INFRA $label ==="

  local zombies
  local processes

  zombies=$(
    ps -eo stat= |
      awk '$1 ~ /^Z/ {n++} END {print n+0}'
  )

  processes=$(
    ps -e -o pid= | wc -l
  )

  echo "zombies=$zombies"
  echo "processes=$processes"

  local pids_file=""

  for f in \
    /sys/fs/cgroup/pids.current \
    /sys/fs/cgroup/pids/pids.current
  do
    if [[ -r "$f" ]]; then
      pids_file="$f"
      break
    fi
  done

  if [[ -z "$pids_file" ]]; then
    pids_file=$(
      find /sys/fs/cgroup \
        -maxdepth 5 \
        -type f \
        -name pids.current \
        -readable \
        -print \
        -quit \
        2>/dev/null || true
    )
  fi

  if [[ -n "$pids_file" ]]; then
    echo "pids_file=$pids_file"
    echo "pids.current=$(cat "$pids_file")"
  else
    echo "pids.current=UNAVAILABLE"
    echo "self_cgroup=$(tr '\n' ';' </proc/self/cgroup)"
  fi
}

show_infra_state BEFORE | tee "$RUN/infra_before.txt"

# Always preserve final infrastructure state, including preflight failures.
record_exit_state() {
  local status=$?

  echo "$status" > "$RUN/final_exit_status.txt"

  show_infra_state AFTER_EXIT     > "$RUN/infra_after.txt" 2>&1 || true
}

trap record_exit_state EXIT

touch "$RUN/run_started.marker"

# ------------------------------------------------------------------
# Resolve EXACT config that will be used.
# ------------------------------------------------------------------

python "$PROJECT/scripts/infra/compose_verl_sft_config.py" \
  --config-dir "/workspace/mtpatcher/repo/verl-v0.9.0/verl/trainer/config" \
  --output "$RUN/resolved_config.yaml" \
  -- \
  "${OVERRIDES[@]}"

# ------------------------------------------------------------------
# Machine-readable config contract.
# Fail BEFORE touching the model/NPU if anything drifted.
# ------------------------------------------------------------------

RUN="$RUN" \
PROJECT="$PROJECT" \
MODEL="$MODEL" \
DATA="$DATA" \
DATASET="$DATASET" \
python - <<'PY'
import os
from pathlib import Path

from omegaconf import OmegaConf


run = Path(os.environ["RUN"])
project = Path(os.environ["PROJECT"])
model = os.environ["MODEL"]
data = os.environ["DATA"]
dataset = os.environ["DATASET"]

cfg = OmegaConf.load(run / "resolved_config.yaml")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


# ---------------- data ----------------

require(
    str(cfg.data.train_files) == data,
    f"train_files drifted: {cfg.data.train_files}",
)

require(
    cfg.data.val_files is None,
    f"val_files must be null: {cfg.data.val_files}",
)

expected_global_batch = int(
    os.environ.get("GLOBAL_BATCH", "8")
)
expected_micro_batch = int(
    os.environ.get("MICRO_BATCH", "1")
)

require(
    cfg.data.train_batch_size == expected_global_batch,
    f"train_batch_size: {cfg.data.train_batch_size} "
    f"!= {expected_global_batch}",
)

require(
    cfg.data.micro_batch_size_per_gpu == expected_micro_batch,
    f"micro_batch_size_per_gpu: "
    f"{cfg.data.micro_batch_size_per_gpu} "
    f"!= {expected_micro_batch}",
)
require(cfg.data.use_dynamic_bsz is False, "dynamic_bsz")

require(cfg.data.messages_key == "messages", "messages_key")
require(cfg.data.enable_thinking_default is False, "thinking default")

require(cfg.data.pad_mode == "no_padding", "pad_mode")
require(cfg.data.max_length == 1024, "max_length")
require(cfg.data.truncation == "error", "truncation")
require(cfg.data.num_workers == 0, "num_workers")

require(
    cfg.data.ignore_input_ids_mismatch is False,
    "ignore_input_ids_mismatch",
)

require(
    cfg.data.apply_chat_template_kwargs.enable_thinking is False,
    "chat-template thinking must be false",
)

require(
    str(cfg.data.custom_cls.path) == dataset,
    f"custom_cls.path drifted: {cfg.data.custom_cls.path}",
)

require(
    cfg.data.custom_cls.name == "MTResponseOnlySFTDataset",
    f"custom_cls.name drifted: {cfg.data.custom_cls.name}",
)


# ---------------- model ----------------

require(str(cfg.model.path) == model, "model.path")
require(cfg.model.lora_rank == 0, "LoRA must be disabled")
require(cfg.model.use_remove_padding is True, "remove padding")
require(
    cfg.model.enable_gradient_checkpointing is True,
    "gradient checkpointing",
)


# ---------------- engine ----------------

require(
    cfg.engine.ulysses_sequence_parallel_size == 1,
    "Ulysses must be disabled for 1-NPU smoke",
)


# ---------------- optimizer ----------------

require(cfg.optim.optimizer == "AdamW", "optimizer")
require(cfg.optim.optimizer_impl == "torch.optim", "optimizer impl")

require(abs(float(cfg.optim.lr) - 2e-5) < 1e-12, "lr")
require(abs(float(cfg.optim.weight_decay) - 0.01) < 1e-12, "wd")
require(abs(float(cfg.optim.clip_grad) - 1.0) < 1e-12, "clip")

require(
    cfg.optim.lr_scheduler_type == "constant",
    "scheduler",
)

require(
    float(cfg.optim.lr_warmup_steps_ratio) == 0.0,
    "warmup",
)


# Important:
# exact Verl v0.9 patches optimizer_config.total_training_steps
# from trainer.total_training_steps immediately before
# training_client.reset(). Therefore -1 here is expected.
require(
    cfg.optim.total_training_steps == -1,
    "unexpected static optim.total_training_steps",
)


# ---------------- trainer ----------------

require(cfg.trainer.project_name == "mtpatcher-sft", "project")
require(
    cfg.trainer.experiment_name == "gate4-qwen3-06b",
    "experiment",
)

require(
    str(cfg.trainer.default_local_dir)
    == str(run / "checkpoints"),
    "checkpoint path",
)

require(
    list(cfg.trainer.logger) == ["console", "tensorboard"],
    f"logger mismatch: {cfg.trainer.logger}",
)

expected_epochs = int(os.environ.get("TOTAL_EPOCHS", "1"))
expected_steps = int(os.environ.get("TOTAL_STEPS", "2"))
expected_save_freq = int(os.environ.get("SAVE_FREQ", "2"))
expected_resume_mode = os.environ.get("RESUME_MODE", "disable")
expected_resume_from = os.environ.get("RESUME_FROM_PATH")

require(
    cfg.trainer.total_epochs == expected_epochs,
    f"epochs: {cfg.trainer.total_epochs} != {expected_epochs}",
)

require(
    cfg.trainer.total_training_steps == expected_steps,
    f"steps: {cfg.trainer.total_training_steps} != {expected_steps}",
)

require(
    cfg.trainer.save_freq == expected_save_freq,
    f"save_freq: {cfg.trainer.save_freq} != {expected_save_freq}",
)

require(cfg.trainer.test_freq == -1, "test_freq")

require(
    cfg.trainer.resume_mode == expected_resume_mode,
    f"resume_mode: {cfg.trainer.resume_mode} != {expected_resume_mode}",
)

if expected_resume_mode == "resume_path":
    require(
        expected_resume_from is not None,
        "RESUME_FROM_PATH must be set",
    )
    require(
        str(cfg.trainer.resume_from_path) == expected_resume_from,
        f"resume_from_path: {cfg.trainer.resume_from_path} "
        f"!= {expected_resume_from}",
    )
else:
    require(
        cfg.trainer.resume_from_path is None,
        f"unexpected resume_from_path: {cfg.trainer.resume_from_path}",
    )

require(
    list(cfg.checkpoint.save_contents)
    == ["model", "optimizer", "extra"],
    f"checkpoint contents: {cfg.checkpoint.save_contents}",
)

print("GATE4_CONFIG_CONTRACT_PASS")
print("resolved_config =", run / "resolved_config.yaml")
PY

# ------------------------------------------------------------------
# Real framework smoke.
#
# No custom optimizer loop.
# No custom FSDP.
# No custom checkpointing.
# ------------------------------------------------------------------

if [[ "${CONFIG_ONLY:-0}" == "1" ]]; then
  echo "GATE4_CONFIG_ONLY_PASS"
  exit 0
fi

cd "$RUN"

echo
echo "=== START VERL 2-STEP SFT ==="
echo "run_dir=$RUN"

set +e

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE:-1}" \
  -m verl.trainer.sft_trainer \
  "${OVERRIDES[@]}" \
  2>&1 | tee "$RUN/train.log"

STATUS=${PIPESTATUS[0]}

set -e

echo
echo "torchrun_exit_status=$STATUS" |
  tee "$RUN/exit_status.txt"

show_infra_state AFTER |
  tee "$RUN/infra_after.txt"

# ------------------------------------------------------------------
# Artifact inventory only.
# Do NOT decide PASS merely from file existence.
# ------------------------------------------------------------------

echo
echo "=== CHECKPOINT INVENTORY ==="

if [[ -d "$RUN/checkpoints" ]]; then
  find "$RUN/checkpoints" \
    -maxdepth 5 \
    -type f \
    -printf '%p %s bytes\n' \
    | sort \
    | tee "$RUN/checkpoint_inventory.txt"
else
  echo "NO_CHECKPOINT_DIRECTORY" |
    tee "$RUN/checkpoint_inventory.txt"
fi

echo
echo "=== TENSORBOARD INVENTORY ==="

find "$RUN" \
  -type f \
  -name 'events.out.tfevents.*' \
  -printf '%p %s bytes\n' \
  2>/dev/null \
  | sort \
  | tee "$RUN/tensorboard_inventory.txt"

echo
echo "=== IMPORTANT TRAIN LOG LINES ==="

grep -Ei \
  'Total training steps|train/loss|train/grad_norm|train/lr|loss[^a-z]|grad_norm|checkpoint|Traceback|ERROR|RuntimeError|OutOfMemory|OOM' \
  "$RUN/train.log" \
  | tail -120 \
  | tee "$RUN/important_train_lines.txt" \
  || true

echo
echo "RUN_DIR=$RUN"

exit "$STATUS"
