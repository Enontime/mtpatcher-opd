#!/usr/bin/env bash
set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

SCRIPT_DIR="/workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend/scripts/mtpatcher_v11"
EXP_DATA="/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823"
EXP_LOG="/workspace/mtpatcher/logs/mtpatcher_v3_full6565_20260823"

PE="/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/pe_k1_clean3732.jsonl"
JOBS="/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/rq3_pds_jobs_v11.jsonl"

PDS_DIR="/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/rq3_pds_v11"
PDS_VALID="/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/rq3_pds_valid_v11.jsonl"
PDS_AUDIT="/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/rq3_pds_audit_v11.json"

PE_PDS="/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/rq3_pe_plus_pds_v11.jsonl"
PE_PDS_AUDIT="/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/rq3_pe_plus_pds_audit_v11.json"

MODEL="/workspace/mtpatcher/models/Qwen3-8B"

mkdir -p "$PDS_DIR" "$EXP_LOG"

echo "======================================================================"
echo "RQ3 PDS V11 — BUILD JOBS"
echo "======================================================================"

python "$SCRIPT_DIR/build_pds_jobs_v11.py"     --pe "$PE"     --output "$JOBS"     --repeat 4

echo
echo "======================================================================"
echo "RQ3 PDS V11 — 16 NPU GENERATION"
echo "======================================================================"

PIDS=()

for DEVICE_ID in $(seq 0 15); do
    DEVICE_LOG="$EXP_LOG/rq3_pds_v11_device_${DEVICE_ID}.log"
    DEVICE_OUT="$PDS_DIR/device_${DEVICE_ID}.jsonl"

    python "$SCRIPT_DIR/generate_pds_qwen3_8b_v11.py"         --jobs "$JOBS"         --output "$DEVICE_OUT"         --model "$MODEL"         --device-id "$DEVICE_ID"         --world-size 16         --batch-size 8         --max-new-tokens 192         --seed 20260825         > "$DEVICE_LOG" 2>&1 &

    PIDS+=("$!")
done

FAIL=0

for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        FAIL=1
    fi
done

if [ "$FAIL" -ne 0 ]; then
    echo "PDS_GENERATION_WORKER_FAILURE"
    false
fi

echo "PDS_ALL_16_WORKERS_COMPLETE"

echo
echo "======================================================================"
echo "RQ3 PDS V11 — MERGE + SCIENTIFIC AUDIT"
echo "======================================================================"

python "$SCRIPT_DIR/merge_and_audit_pds_v11.py"     --jobs "$JOBS"     --shard-dir "$PDS_DIR"     --output "$PDS_VALID"     --audit "$PDS_AUDIT"     --world-size 16

echo
echo "======================================================================"
echo "RQ3 PDS V11 — BUILD PE + PDS SFT DATA"
echo "======================================================================"

python "$SCRIPT_DIR/build_pe_plus_pds_v11.py"     --pe "$PE"     --pds "$PDS_VALID"     --output "$PE_PDS"     --audit "$PE_PDS_AUDIT"     --seed 20260825

echo
echo "======================================================================"
echo "RQ3 PE+PDS DATA READY"
echo "======================================================================"

cat "$PDS_AUDIT"
echo
cat "$PE_PDS_AUDIT"

echo
echo "PDS_GENERATION_ALL_PASS"
echo "PE_PLUS_PDS_DATA_ALL_PASS"
echo "RQ3_PDS_BASELINE_DATA_READY"
