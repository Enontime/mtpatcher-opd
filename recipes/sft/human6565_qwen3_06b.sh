#!/usr/bin/env bash
set -euo pipefail

PROJECT=/workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend
CONFIG=$PROJECT/configs/sft/human6565_qwen3_06b.yaml
COMPOSER=$PROJECT/scripts/infra/compose_verl_sft_config.py

RUN_ROOT=/workspace/mtpatcher/runs/sft
STAMP=$(date +%Y%m%d_%H%M%S)

readarray -t META < <(
python - "$CONFIG" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))

print(cfg["name"])
print(cfg["nproc_per_node"])
PY
)

NAME="${META[0]}"
NPROC="${META[1]}"

RUN="$RUN_ROOT/${NAME}_${STAMP}"
mkdir -p "$RUN"

echo "$RUN" > /tmp/mtpatcher_canonical_sft_run_dir

readarray -t OVERRIDES < <(
python - "$CONFIG" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))

for x in cfg["overrides"]:
    print(x)
PY
)

# Runtime provenance/output location is intentionally supplied by launcher,
# not duplicated as an experiment hyperparameter in YAML.
OVERRIDES+=(
  "trainer.default_local_dir=$RUN/checkpoints"
)

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

record_exit_state() {
  local status=$?

  echo "$status" > "$RUN/final_exit_status.txt"

  show_infra_state AFTER_EXIT \
    > "$RUN/infra_after.txt" 2>&1 || true
}

show_infra_state BEFORE |
  tee "$RUN/infra_before.txt"

trap record_exit_state EXIT

cp "$CONFIG" "$RUN/experiment_config.yaml"

# ------------------------------------------------------------
# Compose exactly the same overrides that real training will use.
# ------------------------------------------------------------

python "$COMPOSER" \
  --config-dir "/workspace/mtpatcher/repo/verl-v0.9.0/verl/trainer/config" \
  --output "$RUN/resolved_config.yaml" \
  -- \
  "${OVERRIDES[@]}"

# ------------------------------------------------------------
# Canonical contract.
# ------------------------------------------------------------

RUN="$RUN" python - <<'PY'
import os
from pathlib import Path

from omegaconf import OmegaConf

run = Path(os.environ["RUN"])
cfg = OmegaConf.load(run / "resolved_config.yaml")


def req(x, msg):
    if not x:
        raise AssertionError(msg)


req(
    str(cfg.data.train_files).endswith(
        "/verl_sft_qwen3_06b/human6565.parquet"
    ),
    "wrong training data",
)

req(cfg.data.val_files is None, "val must be null")

req(cfg.data.train_batch_size == 16, "global batch")
req(cfg.data.micro_batch_size_per_gpu == 1, "micro batch")
req(cfg.data.use_dynamic_bsz is False, "dynamic batch")

req(cfg.data.max_length == 1024, "max length")
req(cfg.data.truncation == "error", "truncation")

req(
    cfg.data.custom_cls.name == "MTResponseOnlySFTDataset",
    "dataset adapter",
)

req(cfg.model.lora_rank == 0, "full finetune")

req(
    cfg.engine.ulysses_sequence_parallel_size == 1,
    "ulysses",
)

req(cfg.engine.model_dtype == "fp32", "model dtype")
req(cfg.engine.dtype == "bfloat16", "compute dtype")
req(cfg.engine.seed == 20260820, "engine seed")

req(cfg.optim.optimizer == "AdamW", "optimizer")
req(cfg.optim.optimizer_impl == "torch.optim", "optimizer impl")
req(abs(float(cfg.optim.lr) - 2e-5) < 1e-12, "lr")
req(abs(float(cfg.optim.weight_decay) - 0.01) < 1e-12, "wd")
req(abs(float(cfg.optim.clip_grad) - 1.0) < 1e-12, "clip")

req(
    cfg.optim.lr_scheduler_type == "constant",
    "scheduler",
)

req(
    abs(float(cfg.optim.lr_warmup_steps_ratio) - 0.03) < 1e-12,
    "warmup ratio",
)

req(cfg.trainer.seed == 20260820, "trainer seed")

req(cfg.trainer.nnodes == 1, "nnodes")
req(cfg.trainer.n_gpus_per_node == 16, "n_gpus_per_node")

req(cfg.trainer.total_epochs == 3, "epochs")

req(
    cfg.trainer.total_training_steps is None,
    "total steps must be auto-derived",
)

req(
    cfg.trainer.save_freq == "after_each_epoch",
    "save frequency",
)

req(cfg.trainer.test_freq == -1, "test freq")
req(cfg.trainer.resume_mode == "disable", "resume mode")

req(
    list(cfg.trainer.logger)
    == ["console", "tensorboard"],
    "logger",
)

print("CANONICAL_HUMAN6565_CONFIG_CONTRACT_PASS")

print("EXPECTED_NATIVE_VERL:")
print("  population_rows = 6565")
print("  global_batch = 16")
print("  native_drop_last = true")
print("  consumed_rows_per_epoch = 6560")
print("  expected_steps_per_epoch = 410")
print("  expected_total_steps = 1230")
PY

if [[ "${CONFIG_ONLY:-0}" == "1" ]]; then
  echo "CANONICAL_HUMAN6565_CONFIG_ONLY_PASS"
  exit 0
fi

cd "$RUN"

echo
echo "=== START CANONICAL HUMAN6565 SFT ==="
echo "run=$RUN"
echo "nproc=$NPROC"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NPROC" \
  -m verl.trainer.sft_trainer \
  "${OVERRIDES[@]}" \
  2>&1 | tee "$RUN/train.log"

STATUS=${PIPESTATUS[0]}

echo "$STATUS" > "$RUN/torchrun_exit_status.txt"

exit "$STATUS"
