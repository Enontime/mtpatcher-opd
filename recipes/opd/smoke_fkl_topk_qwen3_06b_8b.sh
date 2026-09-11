#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/mtpatcher

DATA="$ROOT/data/verl_opd_smoke/mt_human16_opd.parquet"

STUDENT="$ROOT/models/Qwen3-0.6B"
TEACHER="$ROOT/models/Qwen3-8B"

STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}

RUN=${RUN:-\
"$ROOT/runs/opd/smoke_fkl_topk_qwen3_06b_8b_$STAMP"}

mkdir -p "$RUN/checkpoints"

echo "$RUN" \
  > /tmp/mtpatcher_opd_smoke_run


# ------------------------------------------------------------
# Smoke-only parameters.
# These are NOT canonical scientific OPD hyperparameters.
# ------------------------------------------------------------

TRAIN_BATCH=16

MAX_PROMPT=384
MAX_RESPONSE=128
MAX_MODEL_LEN=$((MAX_PROMPT + MAX_RESPONSE + 1))

ACTOR_GPUS=8

TEACHER_GPUS=4
ROLLOUT_TP=2
TEACHER_TP=2

DISTILL_TOPK=32


OVERRIDES=(

  # ----------------------------------------------------------
  # Data / MT rollout semantics
  # ----------------------------------------------------------

  "data.train_files=['$DATA']"
  "data.val_files=['$DATA']"

  "data.train_batch_size=$TRAIN_BATCH"

  "data.max_prompt_length=$MAX_PROMPT"
  "data.max_response_length=$MAX_RESPONSE"

  "data.filter_overlong_prompts=False"
  "data.truncation=error"
  "data.shuffle=False"

  "+data.apply_chat_template_kwargs.enable_thinking=False"


  # ----------------------------------------------------------
  # Algorithm shell required by main_ppo
  # ----------------------------------------------------------

  "algorithm.adv_estimator=grpo"

  # No reference-policy KL.
  "algorithm.use_kl_in_reward=False"
  "reward.custom_reward_function.path=/workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend/scripts/opd/constant_zero_reward.py"
  "reward.custom_reward_function.name=compute_score"
  "actor_rollout_ref.actor.use_kl_loss=False"


  # ----------------------------------------------------------
  # Student model / actor
  # ----------------------------------------------------------

  "actor_rollout_ref.model.path=$STUDENT"

  # Use the native OPD no-padding path.
  "actor_rollout_ref.model.use_remove_padding=True"

  # Smoke: remove unnecessary execution features.
  "actor_rollout_ref.model.enable_gradient_checkpointing=False"
  "actor_rollout_ref.actor.use_torch_compile=False"

  "actor_rollout_ref.actor.optim.lr=1e-6"

  "actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH"
  "actor_rollout_ref.actor.ppo_epochs=1"

  # Explicit direct-FKL aggregation semantics.
  "actor_rollout_ref.actor.loss_agg_mode=token-mean"

  # One local microbatch per rank:
  # global 16 / DP 8 = 2 examples per rank.
  "actor_rollout_ref.actor.use_dynamic_bsz=True"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096"

  "actor_rollout_ref.actor.fsdp_config.param_offload=False"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"


  # ----------------------------------------------------------
  # Student rollout: fully Verl/vLLM
  # ----------------------------------------------------------

  "actor_rollout_ref.rollout.name=vllm"

  "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP"

  "actor_rollout_ref.rollout.n=1"

  "actor_rollout_ref.rollout.dtype=bfloat16"

  "actor_rollout_ref.rollout.gpu_memory_utilization=0.35"

  "actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN"

  "actor_rollout_ref.rollout.enforce_eager=True"

  "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2"
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096"


  # ----------------------------------------------------------
  # Canonical algorithm identity:
  # Teacher-topK direct FKL
  # ----------------------------------------------------------

  "distillation.enabled=True"

  "distillation.distillation_loss.loss_mode=forward_kl_topk"
  "distillation.distillation_loss.topk=$DISTILL_TOPK"

  "distillation.distillation_loss.use_task_rewards=False"
  "distillation.distillation_loss.use_policy_gradient=False"

  # Keep exact YAML no-clamp semantics for smoke visibility.
  "distillation.distillation_loss.loss_max_clamp=null"
  "distillation.distillation_loss.log_prob_min_clamp=null"


  # ----------------------------------------------------------
  # Dedicated Teacher resource pool
  # ----------------------------------------------------------

  "distillation.n_gpus_per_node=$TEACHER_GPUS"
  "distillation.nnodes=1"

  "distillation.teacher_models.teacher_model.model_path=$TEACHER"

  "distillation.teacher_models.teacher_model.inference.name=vllm"

  "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=$TEACHER_TP"
  "distillation.teacher_models.teacher_model.inference.data_parallel_size=1"

  "distillation.teacher_models.teacher_model.inference.dtype=bfloat16"

  "distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.40"

  "distillation.teacher_models.teacher_model.inference.enforce_eager=True"

  "distillation.teacher_models.teacher_model.inference.max_model_len=$MAX_MODEL_LEN"


  # ----------------------------------------------------------
  # Resource topology
  # ----------------------------------------------------------

  "trainer.nnodes=1"
  "trainer.n_gpus_per_node=$ACTOR_GPUS"

  "trainer.project_name=mtpatcher-opd"
  "trainer.experiment_name=smoke-fkl-topk-qwen3-06b-8b"

  "trainer.logger=['console','tensorboard']"

  "trainer.val_before_train=False"
  "trainer.test_freq=-1"

  "trainer.total_epochs=1"

  # With 16 rows / batch16 / epoch1:
  # exactly one rollout/training iteration.
  "trainer.save_freq=1"

  "trainer.default_local_dir=$RUN/checkpoints"
  "trainer.resume_mode=disable"
)


printf '%s\n' "${OVERRIDES[@]}" \
  > "$RUN/overrides.txt"


if [[ "${CONFIG_ONLY:-0}" == "1" ]]; then

  python -m verl.trainer.main_ppo \
    --cfg job \
    "${OVERRIDES[@]}" \
    > "$RUN/resolved_config.yaml"

  echo "OPD_SMOKE_CONFIG_ONLY_PASS"
  echo "RUN=$RUN"

  exit 0
fi


python -m verl.trainer.main_ppo \
  "${OVERRIDES[@]}"

echo "OPD_SMOKE_MAIN_PPO_EXIT_PASS"
echo "RUN=$RUN"
