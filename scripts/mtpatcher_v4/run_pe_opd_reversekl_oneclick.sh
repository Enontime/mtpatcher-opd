#!/usr/bin/env bash

set -euo pipefail


source /workspace/mtpatcher/project_env.sh


export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1


export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15



EXP="mtpatcher_v3_full6565_20260823"

DATA="$DATA_ROOT/$EXP/pe_k1_clean3732.jsonl"


STUDENT="$MODEL_ROOT/Qwen3-0.6B"

TEACHER="$MODEL_ROOT/Qwen3-8B"



OUT="$RUN_ROOT/$EXP/pe_opd_reversekl_qwen3_06b"



TRAINER="$ROOT/scripts/mtpatcher_v4/train_pe_opd_reversekl.py"



mkdir -p "$OUT"



echo "PE OPD REVERSE KL START"



torchrun \
--nproc_per_node=16 \
--master_port=29633 \
"$TRAINER" \
--student "$STUDENT" \
--teacher "$TEACHER" \
--data "$DATA" \
--output "$OUT" \
--epochs 3 \
--lr 1e-6



echo "PE_OPD_REVERSEKL_ALL_DONE"

