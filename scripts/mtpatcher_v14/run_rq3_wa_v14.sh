#!/usr/bin/env bash
set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

SCRIPT_DIR="$ROOT/scripts/mtpatcher_v14"
EXP_DATA="$DATA_ROOT/$EXP"
EXP_LOG="$LOG_ROOT/$EXP"

MODEL="$MODEL_ROOT/Qwen3-8B"

PE="$EXP_DATA/pe_k1_clean3732.jsonl"

PE_PDS="$EXP_DATA/rq3_pe_plus_pds_v13_paperbudget.jsonl"

ANCHOR_JOBS="$EXP_DATA/rq3_wa_anchor_jobs_v14.jsonl"
ANCHOR_AUDIT="$EXP_DATA/rq3_wa_anchor_jobs_v14_audit.json"

ANALOG_DIR="$EXP_DATA/rq3_wa_analogs_v14"
ANALOG_MERGED="$EXP_DATA/rq3_wa_analogs_merged_v14.jsonl"
ANALOG_AUDIT="$EXP_DATA/rq3_wa_analogs_audit_v14.json"

CONTEXT_JOBS="$EXP_DATA/rq3_wa_context_jobs_v14.jsonl"
CONTEXT_DIR="$EXP_DATA/rq3_wa_contexts_v14"

WA_VALID="$EXP_DATA/rq3_wa_valid_v14.jsonl"
WA_AUDIT="$EXP_DATA/rq3_wa_audit_v14.json"

FULL="$EXP_DATA/rq3_pe_pds_wa_v14.jsonl"
FULL_AUDIT="$EXP_DATA/rq3_pe_pds_wa_v14_audit.json"

echo "======================================================================"
echo "RQ3 WA V14 — PAPER-BUDGET KNOWLEDGE EXTENSION"
echo "======================================================================"

###############################################################################
# BUILD 3732 ANALOG ANCHORS
###############################################################################

python \
"$SCRIPT_DIR/build_wa_anchor_jobs_v14.py" \
    --pe "$PE" \
    --output "$ANCHOR_JOBS" \
    --audit "$ANCHOR_AUDIT"

###############################################################################
# ANALOG GENERATION — 16 NPU
###############################################################################

mkdir -p "$ANALOG_DIR"

PIDS=()

for DEVICE in $(seq 0 15); do
    LOG="$EXP_LOG/rq3_wa_v14/analog_device_${DEVICE}.log"

    python \
    "$SCRIPT_DIR/generate_wa_analogs_qwen3_8b_v14.py" \
        --jobs "$ANCHOR_JOBS" \
        --output "$ANALOG_DIR/device_${DEVICE}.jsonl" \
        --model "$MODEL" \
        --device-id "$DEVICE" \
        --world-size 16 \
        --batch-size 8 \
        --max-new-tokens 256 \
        --temperature 1.0 \
        --seed 20260825 \
        > "$LOG" 2>&1 &

    PIDS+=("$!")
done

FAIL=0

for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        FAIL=1
    fi
done

if [ "$FAIL" -ne 0 ]; then
    echo "WA_ANALOG_WORKER_FAILURE"
    false
fi

echo "WA_ALL_16_ANALOG_WORKERS_COMPLETE"

###############################################################################
# MERGE ANALOGS AND REQUIRE EXACT 4×3732
###############################################################################

python \
"$SCRIPT_DIR/merge_wa_analogs_build_context_jobs_v14.py" \
    --shard-dir "$ANALOG_DIR" \
    --merged "$ANALOG_MERGED" \
    --audit "$ANALOG_AUDIT" \
    --context-jobs "$CONTEXT_JOBS"

###############################################################################
# WA → PDS CONTEXT GENERATION — 16 NPU
###############################################################################

mkdir -p "$CONTEXT_DIR"

PIDS=()

for DEVICE in $(seq 0 15); do
    LOG="$EXP_LOG/rq3_wa_v14/context_device_${DEVICE}.log"

    python \
    "$SCRIPT_DIR/generate_wa_contexts_qwen3_8b_v14.py" \
        --jobs "$CONTEXT_JOBS" \
        --output "$CONTEXT_DIR/device_${DEVICE}.jsonl" \
        --model "$MODEL" \
        --device-id "$DEVICE" \
        --world-size 16 \
        --batch-size 8 \
        --max-new-tokens 192 \
        --temperature 1.5 \
        --seed 20260825 \
        > "$LOG" 2>&1 &

    PIDS+=("$!")
done

FAIL=0

for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        FAIL=1
    fi
done

if [ "$FAIL" -ne 0 ]; then
    echo "WA_CONTEXT_WORKER_FAILURE"
    false
fi

echo "WA_ALL_16_CONTEXT_WORKERS_COMPLETE"

###############################################################################
# PAPER-STYLE MERGE + BUILD FULL BASELINE
###############################################################################

python \
"$SCRIPT_DIR/merge_wa_contexts_build_full_v14.py" \
    --shard-dir "$CONTEXT_DIR" \
    --existing-pe-pds "$PE_PDS" \
    --wa-output "$WA_VALID" \
    --wa-audit "$WA_AUDIT" \
    --combined "$FULL" \
    --combined-audit "$FULL_AUDIT"

echo
echo "======================================================================"
echo "FINAL FILES"
echo "======================================================================"

wc -l \
    "$PE" \
    "$PE_PDS" \
    "$WA_VALID" \
    "$FULL"

echo

sha256sum \
    "$WA_VALID" \
    "$FULL"

echo
echo "RQ3_WA_V14_ALL_PASS"
