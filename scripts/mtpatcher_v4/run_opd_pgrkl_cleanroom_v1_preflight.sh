#!/usr/bin/env bash
set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

EXP="mtpatcher_v3_full6565_20260823"
NAME="opd_torchnpu_pgrkl_cleanroom_preflight208_v1"

TRAIN="$DATA_ROOT/$EXP/pe_k1_clean3732.jsonl"

STUDENT="$MODEL_ROOT/Qwen3-0.6B"
TEACHER="$MODEL_ROOT/Qwen3-8B"

TRAINER="$ROOT/scripts/mtpatcher_v4/preflight_opd_pgrkl_cleanroom_v1_torchnpu.py"

OUT="$RUN_ROOT/$EXP/$NAME"
EVAL_ROOT="$OUT/eval_epoch3"

mkdir -p \
    "$OUT" \
    "$EVAL_ROOT"

echo "======================================================================"
echo "MT-PATCHER V3 — TORCH-NPU ON-POLICY PG-RKL-K1"
date
echo "======================================================================"

echo
echo "===== STAGE 1/4: PREFLIGHT ====="

export STUDENT TEACHER TRAIN

python - <<'PY'
import json
import os
from pathlib import Path

import torch
import torch_npu
import transformers

student = Path(
    os.environ["STUDENT"]
)

teacher = Path(
    os.environ["TEACHER"]
)

train = Path(
    os.environ["TRAIN"]
)

for p in (
    student,
    teacher,
    train,
):
    if not p.exists():
        raise RuntimeError(
            f"missing required path: {p}"
        )

rows = []

with train.open(
    "r",
    encoding="utf-8",
) as f:
    for line in f:
        if line.strip():
            rows.append(
                json.loads(line)
            )

if len(rows) != 3732:
    raise RuntimeError(
        f"expected 3732 rows, got {len(rows)}"
    )

for x in rows:
    if "reference" in x:
        raise RuntimeError(
            "reference leakage "
            f"index={x.get('index')}"
        )

    messages = x.get("messages")

    if not isinstance(
        messages,
        list,
    ):
        raise RuntimeError(
            "messages missing "
            f"index={x.get('index')}"
        )

    if any(
        m.get("role")
        == "assistant"
        for m in messages
    ):
        raise RuntimeError(
            "assistant leakage "
            f"index={x.get('index')}"
        )

print(
    "torch =",
    torch.__version__,
)

print(
    "torch_npu =",
    torch_npu.__version__,
)

print(
    "transformers =",
    transformers.__version__,
)

print(
    "NPU_COUNT =",
    torch.npu.device_count(),
)

if torch.npu.device_count() < 16:
    raise RuntimeError(
        "fewer than 16 visible NPUs"
    )

print(
    "TRAIN_ROWS =",
    len(rows),
)

print(
    "REFERENCE_USED = False"
)

print(
    "OPD_TORCHNPU_PREFLIGHT_PASS"
)
PY


echo
echo "===== STAGE 2/4: 16-NPU OPD TRAINING ====="

setsid python -m torch.distributed.run \
    --nproc_per_node=16 \
    --master_port=29648 \
    "$TRAINER" \
    --student "$STUDENT" \
    --teacher "$TEACHER" \
    --train "$TRAIN" \
    --output-dir "$OUT" \
    --epochs 3 \
    --lr 1e-6 \
    --max-prompt-length 512 \
    --max-new-tokens 256 \
    --warmup-ratio 0.03 \
    --weight-decay 0.01 \
    --max-grad-norm 1.0 \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --seed 20260824

echo
echo "CLEAN_PG_PREFLIGHT_LAUNCHER_PASS"

