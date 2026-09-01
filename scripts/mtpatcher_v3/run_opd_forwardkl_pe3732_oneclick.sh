#!/usr/bin/env bash
set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

EXP="mtpatcher_v3_full6565_20260823"
RUN_NAME="opd_forwardkl_pe3732_qwen3_06b_from_8b_v1"

EXP_DATA="$DATA_ROOT/$EXP"
EXP_RUN="$RUN_ROOT/$EXP"

STUDENT="$MODEL_ROOT/Qwen3-0.6B"
TEACHER="$MODEL_ROOT/Qwen3-8B"

PE_JSONL="$EXP_DATA/pe_k1_clean3732.jsonl"
TRAIN_PARQUET="$EXP_DATA/opd_pe3732_sources.parquet"

OUT="$EXP_RUN/$RUN_NAME"
EVAL_ROOT="$OUT/eval_epoch3"

mkdir -p "$OUT" "$EVAL_ROOT"

echo "======================================================================"
echo "MT-PATCHER V3 — PE-SOURCE-MATCHED FORWARD-KL OPD"
date
echo "======================================================================"

echo
echo "Student       = $STUDENT"
echo "Teacher       = $TEACHER"
echo "PE source set = $PE_JSONL"
echo "Output        = $OUT"
echo

###############################################################################
# STAGE 1 — PREFLIGHT
###############################################################################

echo "===== STAGE 1/6: PREFLIGHT ====="

export STUDENT TEACHER PE_JSONL

python - <<'PY'
from pathlib import Path
import importlib

student = Path(__import__("os").environ["STUDENT"])
teacher = Path(__import__("os").environ["TEACHER"])
pe = Path(__import__("os").environ["PE_JSONL"])

for p in (student, teacher, pe):
    if not p.exists():
        raise RuntimeError(f"missing required path: {p}")

mods = [
    "torch",
    "torch_npu",
    "ray",
    "vllm",
    "verl",
    "pandas",
    "pyarrow",
]

for mod in mods:
    m = importlib.import_module(mod)
    print(f"{mod:12s} OK  {getattr(m, '__version__', '')}")

import torch
import torch_npu

n = torch.npu.device_count()
print("NPU_COUNT =", n)

if n < 16:
    raise RuntimeError(f"expected >=16 NPUs, got {n}")

# Critically verify this verl actually contains the current OPD implementation.
from verl.workers.config.distillation import (
    DistillationConfig,
    DistillationLossConfig,
)

from verl.trainer.distillation.fsdp.losses import (
    compute_forward_kl_topk,
)

from transformers import AutoConfig

s_cfg = AutoConfig.from_pretrained(
    student,
    local_files_only=True,
)

t_cfg = AutoConfig.from_pretrained(
    teacher,
    local_files_only=True,
)

print("STUDENT_MODEL_TYPE =", s_cfg.model_type)
print("TEACHER_MODEL_TYPE =", t_cfg.model_type)
print("STUDENT_VOCAB =", s_cfg.vocab_size)
print("TEACHER_VOCAB =", t_cfg.vocab_size)

if s_cfg.vocab_size != t_cfg.vocab_size:
    raise RuntimeError(
        "student/teacher vocab mismatch; OPD requires compatible tokenizer/vocab"
    )

print("VERL_FORWARD_KL_OPD_IMPORT_PASS")
print("OPD_PREFLIGHT_PASS")
PY

###############################################################################
# STAGE 2 — BUILD SOURCE-ONLY OPD DATA
###############################################################################

echo
echo "===== STAGE 2/6: BUILD OPD DATA ====="

export TRAIN_PARQUET

python - <<'PY'
import json
import os
from pathlib import Path

import pandas as pd

src = Path(os.environ["PE_JSONL"])
out = Path(os.environ["TRAIN_PARQUET"])

rows = []

with src.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        x = json.loads(line)

        if "reference" in x:
            raise RuntimeError(
                f"unexpected reference field in PE row index={x.get('index')}"
            )

        messages = x.get("messages")

        if not isinstance(messages, list) or not messages:
            raise RuntimeError(
                f"invalid messages index={x.get('index')}"
            )

        rows.append({
            "data_source": "mtpatcher_pe3732",
            "prompt": messages,
            "ability": "translation",
            "enable_thinking": False,
            "extra_info": {
                "index": int(x["index"]),
                "source": x["source"],
            },
        })

if len(rows) != 3732:
    raise RuntimeError(
        f"expected 3732 OPD prompts, got {len(rows)}"
    )

df = pd.DataFrame(rows)

df.to_parquet(
    out,
    index=False,
)

check = pd.read_parquet(out)

print("OPD_ROWS =", len(check))
print("COLUMNS =", list(check.columns))
print("REFERENCE_USED =", False)
print("OPD_DATA =", out)

print("MTPATCHER_V3_OPD_DATA_PASS")
PY

###############################################################################
# STAGE 3 — RUN FORWARD-KL OPD
###############################################################################

echo
echo "===== STAGE 3/6: START VERL FORWARD-KL OPD ====="

# Scientific controls:
#
# Same 3732 sources as PE-k1.
# Student = Qwen3-0.6B Base.
# Teacher = Qwen3-8B.
#
# 3 epochs:
#   match the data-pass budget used in our SFT controls.
#
# forward_kl_topk / topk=128:
#   current verl GKD OPD implementation.
#
# use_task_rewards=False:
#   isolate distillation; no GRPO/task reward mixed in.
#
# use_policy_gradient=False:
#   direct GKD forward-KL backprop.
#
# 12 actor NPUs + 4 teacher NPUs = all 16 NPUs.
#
# MT response is short:
#   max_prompt=512
#   max_response=256

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="['$TRAIN_PARQUET']" \
    data.val_files="['$TRAIN_PARQUET']" \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="$STUDENT" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.8 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.rollout.max_model_len=769 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
    trainer.logger='["console"]' \
    trainer.project_name="mtpatcher_v3_opd" \
    trainer.experiment_name="$RUN_NAME" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=12 \
    trainer.device=npu \
    trainer.balance_batch=True \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=59 \
    trainer.total_epochs=3 \
    trainer.resume_mode=disable \
    trainer.default_local_dir="$OUT" \
    distillation.enabled=True \
    distillation.n_gpus_per_node=4 \
    distillation.nnodes=1 \
    distillation.teacher_models.teacher_model.model_path="$TEACHER" \
    distillation.teacher_models.teacher_model.num_replicas=4 \
    distillation.teacher_models.teacher_model.inference.name=vllm \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.55 \
    distillation.teacher_models.teacher_model.inference.max_model_len=769 \
    distillation.distillation_loss.loss_mode=forward_kl_topk \
    distillation.distillation_loss.topk=128 \
    distillation.distillation_loss.use_task_rewards=False \
    distillation.distillation_loss.use_policy_gradient=False \
    distillation.distillation_loss.loss_max_clamp=10.0 \
    distillation.distillation_loss.log_prob_min_clamp=-10.0 \
    ray_kwargs.ray_init.runtime_env.py_executable=null

echo
echo "VERL_OPD_TRAINING_FINISHED"

###############################################################################
# STAGE 4 — LOCATE + MERGE LAST FSDP CHECKPOINT
###############################################################################

echo
echo "===== STAGE 4/6: MERGE LATEST CHECKPOINT ====="

LATEST="$(find "$OUT" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)"

echo "LATEST_CHECKPOINT=$LATEST"

export LATEST

python - <<'PY'
import os
from pathlib import Path

p = Path(os.environ["LATEST"])

if not str(p) or not p.exists():
    raise RuntimeError(
        f"cannot locate verl checkpoint: {p}"
    )

actor = p / "actor"

if not actor.exists():
    raise RuntimeError(
        f"missing actor checkpoint: {actor}"
    )

print("OPD_CHECKPOINT_FOUND =", actor)
PY

MERGED="$OUT/merged_hf"

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$LATEST/actor" \
    --target_dir "$MERGED"

echo "OPD_MERGE_PASS"
echo "MERGED_MODEL=$MERGED"

###############################################################################
# STAGE 5 — EVALUATE WMT24 / FLORES / CHALLENGE
###############################################################################

echo
echo "===== STAGE 5/6: EVALUATE ====="

EVAL_SCRIPT="$ROOT/scripts/pilot_v2/eval_qwen3_06b_mt_ascend.py"
SCORE_SCRIPT="$ROOT/scripts/pilot_v2/score_mt_jsonl.py"

WMT="$DATA_ROOT/pilot_v2_qwen3_06b/wmt24_zh_en998.jsonl"
FLORES="$DATA_ROOT/pilot_v2_qwen3_06b/flores_zh_en1012.jsonl"
CHALLENGE="$DATA_ROOT/pilot_v2_qwen3_06b/challenge_zh_en197.jsonl"

run_eval () {
    NAME="$1"
    CARD="$2"
    DATA="$3"

    DIR="$EVAL_ROOT/$NAME"
    mkdir -p "$DIR"

    env ASCEND_RT_VISIBLE_DEVICES="$CARD" \
        python "$EVAL_SCRIPT" \
            --model "$MERGED" \
            --tokenizer "$STUDENT" \
            --input "$DATA" \
            --output "$DIR/predictions.jsonl" \
            --method "opd_forwardkl_pe3732_${NAME}" \
            --batch-size 16 \
            --max-new-tokens 256 \
            --attn-implementation sdpa \
        > "$DIR/generation.log" 2>&1

    python "$SCORE_SCRIPT" \
        --input "$DIR/predictions.jsonl" \
        --output "$DIR/metrics.json" \
        >> "$DIR/generation.log" 2>&1
}

run_eval wmt24 0 "$WMT" &
PID_WMT=$!

run_eval flores 1 "$FLORES" &
PID_FLORES=$!

run_eval challenge 2 "$CHALLENGE" &
PID_CHALLENGE=$!

wait "$PID_WMT"
wait "$PID_FLORES"
wait "$PID_CHALLENGE"

echo "OPD_EVAL_PASS"

###############################################################################
# STAGE 6 — FINAL SUMMARY
###############################################################################

echo
echo "===== STAGE 6/6: FINAL SUMMARY ====="

export EVAL_ROOT

python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["EVAL_ROOT"])

base = {
    "wmt24": 15.536214,
    "flores": 19.971480,
    "challenge": 16.537871,
}

# Already frozen comparison systems.
pe3732 = {
    "wmt24": 15.857393,
    "flores": 20.248076,
    "challenge": 17.108629,
}

seqkd_selected3732 = {
    "wmt24": 16.838339,
    "flores": 21.113730,
    "challenge": 17.638367,
}

seqkd_full6565 = {
    "wmt24": 17.108236,
    "flores": 21.177450,
    "challenge": 18.155411,
}

opd = {}

for ds in (
    "wmt24",
    "flores",
    "challenge",
):
    p = root / ds / "metrics.json"

    with p.open(
        "r",
        encoding="utf-8",
    ) as f:
        m = json.load(f)

    opd[ds] = float(m["BLEU"])


def avg_delta(system):
    return sum(
        system[x] - base[x]
        for x in base
    ) / 3


print()
print("=" * 82)
print("MT-PATCHER V3 — FORWARD-KL OPD RESULT")
print("=" * 82)

print(
    f"{'SYSTEM':26s}"
    f"{'WMT24':>10s}"
    f"{'FLORES':>10s}"
    f"{'CHALL':>10s}"
    f"{'AVGΔ':>10s}"
)

print("-" * 66)

systems = {
    "Base": base,
    "PE3732": pe3732,
    "SeqKD-Selected3732": seqkd_selected3732,
    "SeqKD-Full6565": seqkd_full6565,
    "OPD-FKL-PE3732": opd,
}

for name, values in systems.items():

    delta = avg_delta(values)

    print(
        f"{name:26s}"
        f"{values['wmt24']:10.3f}"
        f"{values['flores']:10.3f}"
        f"{values['challenge']:10.3f}"
        f"{delta:+10.3f}"
    )


def gap(a, b):
    return sum(
        a[x] - b[x]
        for x in base
    ) / 3


print()
print("===== KEY COMPARISONS =====")

print(
    "OPD - PE3732 =",
    f"{gap(opd, pe3732):+.4f}",
)

print(
    "OPD - SeqKD-Selected3732 =",
    f"{gap(opd, seqkd_selected3732):+.4f}",
)

print(
    "OPD - SeqKD-Full6565 =",
    f"{gap(opd, seqkd_full6565):+.4f}",
)

summary = {
    "base": base,
    "pe3732": pe3732,
    "seqkd_selected3732": seqkd_selected3732,
    "seqkd_full6565": seqkd_full6565,
    "opd_forwardkl_pe3732": opd,
}

out = root.parent / "opd_forwardkl_final_summary.json"

out.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print()
print("SUMMARY =", out)
print("MTPATCHER_V3_OPD_FORWARDKL_ALL_PASS")
PY

echo
echo "======================================================================"
echo "FORWARD-KL OPD PIPELINE FINISHED"
date
echo "======================================================================"
