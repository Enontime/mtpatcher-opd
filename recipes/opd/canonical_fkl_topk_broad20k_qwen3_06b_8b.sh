#!/usr/bin/env bash

set +e
set +u
set +o pipefail 2>/dev/null

ROOT=/workspace/mtpatcher
PROJECT=$ROOT/repo/MT-Patcher-Reproduction-Ascend

CONFIG=$PROJECT/configs/opd/canonical_fkl_topk_broad20k_qwen3_06b_8b.yaml

RUN_ROOT=$ROOT/runs/opd
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}


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
# Read static experiment specification and emit exact Hydra
# overrides consumed by Verl.
# ============================================================

SPEC_OUT="$(
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
PY
)"

SPEC_READ_STATUS=$?

if [[ "$SPEC_READ_STATUS" -eq 0 ]]; then
  NAME="$SPEC_OUT"
else
  NAME=canonical_fkl_topk_broad20k_qwen3_06b_8b
fi


RUN=${RUN:-"$RUN_ROOT/${NAME}_${STAMP}"}

mkdir -p \
  "$RUN/checkpoints"

echo "$RUN" \
  > /tmp/mtpatcher_canonical_opd_run_dir

echo
echo "run=$RUN"


python - \
  "$CONFIG" \
  "$RUN" \
  > "$RUN/overrides.txt" \
<<'PY'
import sys
import yaml


config_path = sys.argv[1]
run = sys.argv[2]

with open(
    config_path,
    encoding="utf-8",
) as f:
    c = yaml.safe_load(f)


student = c["student"]
teacher = c["teacher"]
population = c["population"]
budget = c["budget"]
rollout = c["rollout"]
objective = c["objective"]
optimization = c["optimization"]
execution = c["execution"]
trainer = c["trainer"]
provenance = c["provenance"]


max_model_len = (
    rollout["max_prompt_length"]
    + rollout["max_response_length"]
    + 1
)


ovs = [
    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------
    f"data.train_files=['{population['train_file']}']",
    f"data.val_files=['{population['train_file']}']",

    f"data.train_batch_size={budget['global_train_batch']}",

    (
        "data.max_prompt_length="
        f"{rollout['max_prompt_length']}"
    ),
    (
        "data.max_response_length="
        f"{rollout['max_response_length']}"
    ),

    "data.filter_overlong_prompts=False",
    "data.truncation=error",
    "data.shuffle=False",

    "+data.apply_chat_template_kwargs.enable_thinking=False",

    # --------------------------------------------------------
    # Generic main_ppo shell
    # --------------------------------------------------------
    "algorithm.adv_estimator=grpo",

    (
        "algorithm.use_kl_in_reward="
        f"{str(objective['use_kl_in_reward'])}"
    ),

    (
        "reward.custom_reward_function.path="
        f"{provenance['zero_reward_adapter']}"
    ),
    "reward.custom_reward_function.name=compute_score",

    (
        "actor_rollout_ref.actor.use_kl_loss="
        f"{str(objective['actor_use_kl_loss'])}"
    ),

    # --------------------------------------------------------
    # Student / Actor
    # --------------------------------------------------------
    (
        "actor_rollout_ref.model.path="
        f"{student['model']}"
    ),

    (
        "actor_rollout_ref.model.use_remove_padding="
        f"{str(execution['actor_use_remove_padding'])}"
    ),

    (
        "actor_rollout_ref.model.enable_gradient_checkpointing="
        f"{str(execution['actor_gradient_checkpointing'])}"
    ),

    (
        "actor_rollout_ref.actor.use_torch_compile="
        f"{str(execution['actor_torch_compile'])}"
    ),

    (
        "actor_rollout_ref.actor.optim.lr="
        f"{optimization['actor_lr']}"
    ),

    (
        "actor_rollout_ref.actor.ppo_mini_batch_size="
        f"{budget['global_train_batch']}"
    ),

    (
        "actor_rollout_ref.actor.ppo_epochs="
        f"{optimization['ppo_epochs']}"
    ),

    (
        "actor_rollout_ref.actor.loss_agg_mode="
        f"{objective['loss_agg_mode']}"
    ),

    (
        "actor_rollout_ref.actor.use_dynamic_bsz="
        f"{str(execution['actor_dynamic_bsz'])}"
    ),

    (
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu="
        f"{execution['actor_max_token_len_per_gpu']}"
    ),

    "actor_rollout_ref.actor.fsdp_config.param_offload=False",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",

    # --------------------------------------------------------
    # Student rollout
    # --------------------------------------------------------
    "actor_rollout_ref.rollout.name=vllm",

    (
        "actor_rollout_ref.rollout."
        "tensor_model_parallel_size="
        f"{execution['rollout_tp']}"
    ),

    (
        "actor_rollout_ref.rollout.n="
        f"{rollout['n']}"
    ),

    (
        "actor_rollout_ref.rollout.temperature="
        f"{rollout['temperature']}"
    ),

    (
        "actor_rollout_ref.rollout.top_p="
        f"{rollout['top_p']}"
    ),

    (
        "actor_rollout_ref.rollout.top_k="
        f"{rollout['top_k']}"
    ),

    (
        "actor_rollout_ref.rollout.dtype="
        f"{execution['dtype']}"
    ),

    (
        "actor_rollout_ref.rollout."
        "gpu_memory_utilization="
        f"{execution['rollout_gpu_memory_utilization']}"
    ),

    (
        "actor_rollout_ref.rollout.max_model_len="
        f"{max_model_len}"
    ),

    (
        "actor_rollout_ref.rollout.enforce_eager="
        f"{str(execution['rollout_enforce_eager'])}"
    ),

    (
        "actor_rollout_ref.rollout."
        "log_prob_use_dynamic_bsz="
        f"{str(execution['rollout_log_prob_dynamic_bsz'])}"
    ),

    (
        "actor_rollout_ref.rollout."
        "log_prob_micro_batch_size_per_gpu="
        f"{execution['rollout_log_prob_micro_batch_size_per_gpu']}"
    ),

    (
        "actor_rollout_ref.rollout."
        "log_prob_max_token_len_per_gpu="
        f"{execution['rollout_log_prob_max_token_len_per_gpu']}"
    ),

    # --------------------------------------------------------
    # Native Verl Teacher-topK direct FKL
    # --------------------------------------------------------
    "distillation.enabled=True",

    (
        "distillation.distillation_loss.loss_mode="
        f"{objective['loss_mode']}"
    ),

    (
        "distillation.distillation_loss.topk="
        f"{teacher['topk']}"
    ),

    (
        "distillation.distillation_loss.use_task_rewards="
        f"{str(objective['use_task_rewards'])}"
    ),

    (
        "distillation.distillation_loss.use_policy_gradient="
        f"{str(objective['use_policy_gradient'])}"
    ),

    "distillation.distillation_loss.loss_max_clamp=null",
    "distillation.distillation_loss.log_prob_min_clamp=null",

    # --------------------------------------------------------
    # Teacher resource pool
    # --------------------------------------------------------
    (
        "distillation.n_gpus_per_node="
        f"{execution['teacher_gpus']}"
    ),
    "distillation.nnodes=1",

    (
        "distillation.teacher_models."
        "teacher_model.model_path="
        f"{teacher['model']}"
    ),

    (
        "distillation.teacher_models."
        "teacher_model.inference.name=vllm"
    ),

    (
        "distillation.teacher_models."
        "teacher_model.inference."
        "tensor_model_parallel_size="
        f"{execution['teacher_tp']}"
    ),

    (
        "distillation.teacher_models."
        "teacher_model.inference."
        "data_parallel_size="
        f"{execution['teacher_dp']}"
    ),

    (
        "distillation.teacher_models."
        "teacher_model.inference.dtype="
        f"{execution['dtype']}"
    ),

    (
        "distillation.teacher_models."
        "teacher_model.inference."
        "gpu_memory_utilization="
        f"{execution['teacher_gpu_memory_utilization']}"
    ),

    (
        "distillation.teacher_models."
        "teacher_model.inference."
        "enforce_eager="
        f"{str(execution['teacher_enforce_eager'])}"
    ),

    (
        "distillation.teacher_models."
        "teacher_model.inference.max_model_len="
        f"{max_model_len}"
    ),

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------
    "trainer.nnodes=1",

    (
        "trainer.n_gpus_per_node="
        f"{execution['actor_gpus']}"
    ),

    (
        "trainer.project_name="
        f"{trainer['project_name']}"
    ),

    (
        "trainer.experiment_name="
        f"{trainer['experiment_name']}"
    ),

    (
        "trainer.logger="
        "['console','tensorboard']"
    ),

    (
        "trainer.val_before_train="
        f"{str(trainer['val_before_train'])}"
    ),

    (
        "trainer.test_freq="
        f"{trainer['test_freq']}"
    ),

    (
        "trainer.total_epochs="
        f"{budget['source_passes']}"
    ),

    (
        "trainer.save_freq="
        f"{trainer['save_freq']}"
    ),

    (
        "trainer.default_local_dir="
        f"{run}/checkpoints"
    ),

    (
        "trainer.resume_mode="
        f"{trainer['resume_mode']}"
    ),
]


for x in ovs:
    print(x)
PY

OVERRIDE_STATUS=$?

if [[ "$OVERRIDE_STATUS" -eq 0 ]]; then
  mapfile -t OVERRIDES \
    < "$RUN/overrides.txt"

  echo "FORMAL_OPD_OVERRIDES_BUILD_PASS"
else
  OVERRIDES=()

  echo \
    "FORMAL_OPD_OVERRIDES_BUILD_FAIL status=$OVERRIDE_STATUS"
fi


cp "$CONFIG" \
  "$RUN/experiment_config.yaml" \
  2>/dev/null


show_infra_state BEFORE |
  tee "$RUN/infra_before.txt"


# ============================================================
# Static identity contract.
# ============================================================

python - \
  "$CONFIG" \
<<'PY'
import hashlib
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml


cfg = yaml.safe_load(
    open(
        sys.argv[1],
        encoding="utf-8",
    )
)


def req(x, msg):
    if not x:
        raise AssertionError(msg)


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(
                8 * 1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


population = cfg["population"]
budget = cfg["budget"]
rollout = cfg["rollout"]
objective = cfg["objective"]
teacher = cfg["teacher"]
execution = cfg["execution"]
trainer = cfg["trainer"]
provenance = cfg["provenance"]


data = Path(
    population["train_file"]
)

smoke = Path(
    provenance[
        "validated_smoke_recipe"
    ]
)

zero_reward = Path(
    provenance[
        "zero_reward_adapter"
    ]
)


req(
    data.is_file(),
    "missing frozen Broad20k OPD parquet",
)

req(
    sha256(data)
    == population["sha256"],
    "Broad20k OPD parquet SHA drift",
)

req(
    pq.ParquetFile(
        data
    ).metadata.num_rows
    == 20000,
    "Broad20k OPD row-count drift",
)

req(
    smoke.is_file(),
    "missing validated smoke recipe",
)

req(
    sha256(smoke)
    == provenance[
        "validated_smoke_sha256"
    ],
    "validated smoke recipe drift",
)

req(
    zero_reward.is_file(),
    "missing zero-reward adapter",
)


req(
    budget["source_passes"] == 3,
    "wrong source passes",
)

req(
    budget["source_exposures"]
    == 60000,
    "wrong source exposures",
)

req(
    budget["global_train_batch"]
    == 16,
    "wrong global batch",
)

req(
    budget["updates_per_pass"]
    == 1250,
    "wrong updates/pass",
)

req(
    budget["total_updates"]
    == 3750,
    "wrong total updates",
)


req(
    rollout["n"] == 1,
    "rollout n drift",
)

req(
    rollout["temperature"]
    == 1.0,
    "canonical Verl temperature drift",
)

req(
    rollout["top_p"]
    == 1.0,
    "canonical Verl top-p drift",
)

req(
    rollout["top_k"]
    == -1,
    "canonical Verl top-k drift",
)


req(
    objective["loss_mode"]
    == "forward_kl_topk",
    "wrong distillation objective",
)

req(
    teacher["topk"] == 32,
    "wrong teacher top-k",
)

req(
    objective["use_task_rewards"]
    is False,
    "task reward must be disabled",
)

req(
    objective["use_policy_gradient"]
    is False,
    "policy gradient must be disabled",
)

req(
    objective["actor_use_kl_loss"]
    is False,
    "actor KL must be disabled",
)

req(
    objective["use_kl_in_reward"]
    is False,
    "reward KL must be disabled",
)

req(
    objective["loss_agg_mode"]
    == "token-mean",
    "wrong loss aggregation",
)


req(
    execution["actor_gpus"] == 8,
    "actor pool drift",
)

req(
    execution["teacher_gpus"] == 4,
    "teacher pool drift",
)

req(
    execution["rollout_tp"] == 2,
    "rollout TP drift",
)

req(
    execution["teacher_tp"] == 2,
    "teacher TP drift",
)


req(
    trainer["save_freq"] == 1250,
    "wrong save frequency",
)

req(
    trainer["resume_mode"]
    == "disable",
    "formal run must start from Base",
)


print(
    "CANONICAL_OPD_STATIC_CONTRACT_PASS"
)

print(
    "population_rows = 20000"
)

print(
    "source_exposures = 60000"
)

print(
    "expected_updates_per_pass = 1250"
)

print(
    "expected_total_updates = 3750"
)

print(
    "rollout_sampling = "
    "temperature1.0/top_p1.0/top_k-1"
)

print(
    "validated_smoke_sha = PASS"
)

print(
    "frozen_opd_asset_sha = PASS"
)
PY

STATIC_STATUS=$?

echo "$STATIC_STATUS" \
  > "$RUN/static_contract_status.txt"


# ============================================================
# Hydra compose only.
# This does not enter Ray training.
# ============================================================

READY=1

if [[ "$SPEC_READ_STATUS" -ne 0 ]]; then
  READY=0
fi

if [[ "$OVERRIDE_STATUS" -ne 0 ]]; then
  READY=0
fi

if [[ "$STATIC_STATUS" -ne 0 ]]; then
  READY=0
fi


if [[ "$READY" -eq 1 ]]; then

  python -m verl.trainer.main_ppo \
    --cfg job \
    "${OVERRIDES[@]}" \
    > "$RUN/resolved_config.yaml"

  COMPOSE_STATUS=$?

else
  COMPOSE_STATUS=99
fi


echo "$COMPOSE_STATUS" \
  > "$RUN/compose_status.txt"


if [[ "$COMPOSE_STATUS" -eq 0 ]]; then
  echo \
    "CANONICAL_OPD_HYDRA_COMPOSE_PASS"
else
  echo \
    "CANONICAL_OPD_HYDRA_COMPOSE_FAIL status=$COMPOSE_STATUS"

  READY=0
fi


# ============================================================
# Exact resolved-config contract.
# ============================================================

if [[ "$READY" -eq 1 ]]; then

  python - \
    "$RUN/resolved_config.yaml" \
  <<'PY'
import sys

from omegaconf import OmegaConf

from verl.utils.config import (
    omega_conf_to_dataclass,
)


cfg = OmegaConf.load(
    sys.argv[1]
)


def req(x, msg):
    if not x:
        raise AssertionError(msg)


DATA = (
    "/workspace/mtpatcher/data/"
    "verl_science_broad20k/"
    "opd_broad20k.parquet"
)


req(
    list(cfg.data.train_files)
    == [DATA],
    "resolved training population drift",
)

req(
    cfg.data.train_batch_size
    == 16,
    "resolved batch drift",
)

req(
    cfg.data.max_prompt_length
    == 1024,
    "resolved prompt length drift",
)

req(
    cfg.data.max_response_length
    == 256,
    "resolved response length drift",
)

req(
    cfg.data.shuffle is False,
    "resolved data shuffle drift",
)


r = cfg.actor_rollout_ref.rollout
a = cfg.actor_rollout_ref.actor


req(
    r.n == 1,
    "resolved rollout n drift",
)

req(
    float(r.temperature)
    == 1.0,
    "resolved temperature drift",
)

req(
    float(r.top_p)
    == 1.0,
    "resolved top-p drift",
)

req(
    int(r.top_k)
    == -1,
    "resolved top-k drift",
)

req(
    r.tensor_model_parallel_size
    == 2,
    "resolved rollout TP drift",
)

req(
    r.max_model_len
    == 1281,
    "resolved rollout max_model_len drift",
)


req(
    a.ppo_mini_batch_size
    == 16,
    "resolved PPO minibatch drift",
)

req(
    a.ppo_epochs == 1,
    "resolved PPO epoch drift",
)

req(
    a.loss_agg_mode
    == "token-mean",
    "resolved loss aggregation drift",
)

req(
    a.use_dynamic_bsz is True,
    "resolved actor dynamic batch drift",
)

req(
    a.ppo_micro_batch_size_per_gpu
    is None,
    "actor fixed microbatch unexpectedly set",
)

req(
    a.ppo_max_token_len_per_gpu
    == 4096,
    "actor token budget drift",
)

req(
    a.use_kl_loss is False,
    "actor KL unexpectedly enabled",
)


req(
    cfg.algorithm.use_kl_in_reward
    is False,
    "reward KL unexpectedly enabled",
)


d = omega_conf_to_dataclass(
    cfg.distillation
)


req(
    d.enabled is True,
    "distillation disabled",
)

req(
    d.distillation_loss.loss_mode
    == "forward_kl_topk",
    "resolved distillation loss drift",
)

req(
    d.distillation_loss.topk
    == 32,
    "resolved Teacher top-k drift",
)

req(
    d.distillation_loss.use_task_rewards
    is False,
    "resolved task reward drift",
)

req(
    d.distillation_loss.use_policy_gradient
    is False,
    "resolved policy gradient drift",
)

req(
    d.n_gpus_per_node
    * d.nnodes
    == 4,
    "resolved teacher pool drift",
)


teacher_items = list(
    d.teacher_models.items()
)

req(
    len(teacher_items) == 1,
    "resolved teacher count drift",
)

teacher_dict_key, t = (
    teacher_items[0]
)

req(
    teacher_dict_key == "default",
    "resolved teacher dict-key drift",
)

req(
    getattr(
        t,
        "key",
        None,
    ) == "default",
    "resolved teacher runtime-key drift",
)

req(
    getattr(
        t,
        "num_replicas",
        None,
    ) == 2,
    "resolved teacher replica-count drift",
)

req(
    t.model_path
    ==
    "/workspace/mtpatcher/models/Qwen3-8B",
    "resolved teacher model drift",
)

req(
    t.inference.tensor_model_parallel_size
    == 2,
    "resolved teacher TP drift",
)

req(
    t.inference.data_parallel_size
    == 1,
    "resolved teacher DP drift",
)

req(
    t.inference.max_model_len
    == 1281,
    "resolved teacher max_model_len drift",
)


req(
    cfg.trainer.n_gpus_per_node
    == 8,
    "resolved actor pool drift",
)

req(
    cfg.trainer.nnodes == 1,
    "resolved node count drift",
)

req(
    cfg.trainer.total_epochs
    == 3,
    "resolved source-pass drift",
)

req(
    cfg.trainer.save_freq
    == 1250,
    "resolved save frequency drift",
)

req(
    cfg.trainer.test_freq
    == -1,
    "resolved trainer test drift",
)

req(
    cfg.trainer.resume_mode
    == "disable",
    "resolved resume drift",
)


print(
    "CANONICAL_OPD_RESOLVED_CONTRACT_PASS"
)

print(
    "distillation_type =",
    type(d),
)

print(
    "loss_mode =",
    d.distillation_loss.loss_mode,
)

print(
    "teacher_topk =",
    d.distillation_loss.topk,
)

print(
    "actor_pool =",
    cfg.trainer.n_gpus_per_node,
)

print(
    "teacher_pool =",
    d.n_gpus_per_node * d.nnodes,
)

print(
    "rollout_tp =",
    r.tensor_model_parallel_size,
)

print(
    "teacher_tp =",
    t.inference.tensor_model_parallel_size,
)

print(
    "sampling =",
    r.temperature,
    r.top_p,
    r.top_k,
)

print(
    "max_model_len =",
    r.max_model_len,
)

print(
    "epochs =",
    cfg.trainer.total_epochs,
)

print(
    "save_freq =",
    cfg.trainer.save_freq,
)
PY

  CONTRACT_STATUS=$?

else
  CONTRACT_STATUS=99
fi


echo "$CONTRACT_STATUS" \
  > "$RUN/resolved_contract_status.txt"


if [[ "$CONTRACT_STATUS" -eq 0 ]]; then
  echo \
    "CANONICAL_OPD_CONFIG_CONTRACT_PASS"
else
  echo \
    "CANONICAL_OPD_CONFIG_CONTRACT_FAIL status=$CONTRACT_STATUS"

  READY=0
fi


# ============================================================
# CONFIG_ONLY branch.
#
# No Ray training is entered.
# ============================================================

if [[ "$READY" -eq 1 ]] &&
   [[ "${CONFIG_ONLY:-0}" == "1" ]]; then

  echo "0" \
    > "$RUN/final_status.txt"

  show_infra_state AFTER_CONFIG_ONLY \
    > "$RUN/infra_after.txt" 2>&1

  echo \
    "CANONICAL_OPD_CONFIG_ONLY_PASS"

  echo \
    "CONFIG_ONLY_FINISHED_NO_RAY_TRAINING"

fi


# ============================================================
# Formal Verl execution.
# ============================================================

if [[ "$READY" -eq 1 ]] &&
   [[ "${CONFIG_ONLY:-0}" != "1" ]]; then

  echo
  echo \
    "=== START CANONICAL VERL TEACHER-TOPK FKL OPD ==="

  echo "run=$RUN"

  python -m verl.trainer.main_ppo \
    "${OVERRIDES[@]}" \
    2>&1 |
    tee "$RUN/train.log"

  MAIN_STATUS=${PIPESTATUS[0]}

  echo "$MAIN_STATUS" \
    > "$RUN/main_ppo_status.txt"

  echo "$MAIN_STATUS" \
    > "$RUN/final_status.txt"

  if [[ "$MAIN_STATUS" -eq 0 ]]; then
    echo \
      "CANONICAL_OPD_MAIN_PPO_PASS"
  else
    echo \
      "CANONICAL_OPD_MAIN_PPO_FAIL status=$MAIN_STATUS"
  fi

  show_infra_state AFTER_TRAIN \
    > "$RUN/infra_after.txt" 2>&1

fi


# ============================================================
# Pre-run failure.
# ============================================================

if [[ "$READY" -ne 1 ]]; then

  echo "1" \
    > "$RUN/final_status.txt"

  show_infra_state AFTER_PRETRAIN_FAIL \
    > "$RUN/infra_after.txt" 2>&1

  echo \
    "CANONICAL_OPD_PRETRAIN_CONTRACT_FAIL"

  echo \
    "FORMAL_TRAINING_SKIPPED"

fi


echo
echo "run=$RUN"
echo "final_status_file=$RUN/final_status.txt"

if [[ -f "$RUN/final_status.txt" ]]; then
  echo -n "recorded_status="
  cat "$RUN/final_status.txt"
fi

echo \
  "CANONICAL_OPD_WRAPPER_FINISHED"
