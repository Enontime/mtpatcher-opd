#!/usr/bin/env bash
set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

PIPE="$ROOT/scripts/mtpatcher_paper_faithful_v2/run_demo_pipeline_v2.sh"
TRAIN="$ROOT/scripts/mtpatcher_paper_faithful_v2/train_patcher_fullft_v2.py"

SMOKE_BASE="$DATA_ROOT/$EXP/paperfaith_smoke64_v2"
FULL_BASE="$DATA_ROOT/$EXP/paperfaith_paper20k_v2"

SMOKE_LOG="$LOG_ROOT/$EXP/paperfaith_smoke64_v2.log"
FULL_LOG="$LOG_ROOT/$EXP/paperfaith_paper20k_v2.log"

TRAIN_SMOKE_OUT="$RUN_ROOT/$EXP/patcher_qwen3_8b_fullft_smoke_v2"
TRAIN_SMOKE_LOG="$LOG_ROOT/$EXP/patcher_qwen3_8b_fullft_smoke_v2.log"

TRAIN_OUT="$RUN_ROOT/$EXP/patcher_qwen3_8b_paperfaith_fullft_v2"
TRAIN_LOG="$LOG_ROOT/$EXP/patcher_qwen3_8b_paperfaith_fullft_v2.log"

mkdir -p \
    "$LOG_ROOT/$EXP" \
    "$RUN_ROOT/$EXP"


###############################################################################
# FAILURE DIAGNOSTICS
###############################################################################

show_failure_logs () {
    RC="$?"

    echo
    echo "============================================================"
    echo "CHAIN_COMMAND_FAILED rc=$RC"
    echo "============================================================"

    for F in \
        "$SMOKE_LOG" \
        "$TRAIN_SMOKE_LOG" \
        "$FULL_LOG" \
        "$TRAIN_LOG"
    do
        if [ -f "$F" ]; then
            echo
            echo "---------- $F ----------"
            tail -n 180 "$F" || true
        fi
    done
}

trap show_failure_logs ERR

echo "============================================================"
echo "MT-PATCHER PAPER-FAITHFUL AUTO CHAIN V2"
echo "============================================================"

###############################################################################
# PREFLIGHT
###############################################################################

echo
echo "===== PREFLIGHT ====="

for F in \
    "$PIPE" \
    "$TRAIN" \
    "$ROOT/scripts/mtpatcher_paper_faithful_v2/paper_repro_v2.py" \
    "$MODEL_ROOT/Qwen3-0.6B" \
    "$MODEL_ROOT/Qwen3-8B" \
    "$DATA_ROOT/$EXP/rq0_newscrawl50k_sources_v1.jsonl" \
    "$ROOT/vendor/MT-Patcher-official"
do
    if [ ! -e "$F" ]; then
        echo "MISSING_DEPENDENCY=$F"
        false
    fi
done

python -m py_compile \
    "$ROOT/scripts/mtpatcher_paper_faithful_v2/paper_repro_v2.py" \
    "$TRAIN"

echo "CHAIN_PREFLIGHT_PASS"

###############################################################################
# STAGE 1 — demo smoke64
###############################################################################

echo
echo "===== STAGE 1: DEMO SMOKE64 ====="

if grep -q \
    "PAPER_FAITHFUL_DEMO_PIPELINE_V2_PASS" \
    "$SMOKE_LOG" \
    2>/dev/null
then
    echo "DEMO_SMOKE_ALREADY_PASS"
else
    N=64 \
    TAG=smoke64_v2 \
    bash "$PIPE" \
        > "$SMOKE_LOG" 2>&1
fi

if ! grep -q \
    "PAPER_FAITHFUL_DEMO_PIPELINE_V2_PASS" \
    "$SMOKE_LOG"
then
    echo "DEMO_SMOKE_FAILURE"
    tail -n 160 "$SMOKE_LOG"
    false
fi

echo "CHAIN_DEMO_SMOKE_PASS"

###############################################################################
# STAGE 2 — 8B full-FT environment smoke
###############################################################################

echo
echo "===== STAGE 2: PATCHER FULL-FT SMOKE ====="

rm -rf "$TRAIN_SMOKE_OUT"

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
torchrun \
    --standalone \
    --nproc_per_node=16 \
    "$TRAIN" \
    --model "$MODEL_ROOT/Qwen3-8B" \
    --train "$SMOKE_BASE/patcher_sft.jsonl" \
    --output "$TRAIN_SMOKE_OUT" \
    --max-rows 128 \
    --max-steps 2 \
    --smoke \
    > "$TRAIN_SMOKE_LOG" 2>&1

if ! grep -q \
    "PATCHER_FULLFT_SMOKE_PASS" \
    "$TRAIN_SMOKE_LOG"
then
    echo "PATCHER_FULLFT_SMOKE_FAILURE"
    tail -n 200 "$TRAIN_SMOKE_LOG"
    false
fi

echo "CHAIN_PATCHER_FULLFT_SMOKE_PASS"

###############################################################################
# STAGE 3 — formal paper Section 3.3 random20k
###############################################################################

echo
echo "===== STAGE 3: FORMAL RANDOM20K DEMONSTRATIONS ====="

if grep -q \
    "PAPER_FAITHFUL_DEMO_PIPELINE_V2_PASS" \
    "$FULL_LOG" \
    2>/dev/null
then
    echo "FORMAL_20K_ALREADY_PASS"
else
    N=20000 \
    TAG=paper20k_v2 \
    bash "$PIPE" \
        > "$FULL_LOG" 2>&1
fi

if ! grep -q \
    "PAPER_FAITHFUL_DEMO_PIPELINE_V2_PASS" \
    "$FULL_LOG"
then
    echo "FORMAL_20K_FAILURE"
    tail -n 220 "$FULL_LOG"
    false
fi

echo "CHAIN_FORMAL_20K_PASS"

echo
echo "FORMAL_SFT_ROWS=$(wc -l < "$FULL_BASE/patcher_sft.jsonl")"
echo "FORMAL_SFT_SHA256=$(sha256sum "$FULL_BASE/patcher_sft.jsonl" | awk '{print $1}')"

###############################################################################
# STAGE 4 — Appendix B full parameter FT
###############################################################################

echo
echo "===== STAGE 4: FORMAL QWEN3-8B PATCHER FULL FT ====="

if \
    [ -f "$TRAIN_OUT/config.json" ] \
    && \
    grep -q \
        "PATCHER_FULLFT_PAPER_V2_PASS" \
        "$TRAIN_LOG" \
        2>/dev/null
then
    echo "FORMAL_PATCHER_ALREADY_PASS"
else
    ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
    torchrun \
        --standalone \
        --nproc_per_node=16 \
        "$TRAIN" \
        --model "$MODEL_ROOT/Qwen3-8B" \
        --train "$FULL_BASE/patcher_sft.jsonl" \
        --output "$TRAIN_OUT" \
        > "$TRAIN_LOG" 2>&1
fi

if ! grep -q \
    "PATCHER_FULLFT_PAPER_V2_PASS" \
    "$TRAIN_LOG"
then
    echo "FORMAL_PATCHER_TRAIN_FAILURE"
    tail -n 220 "$TRAIN_LOG"
    false
fi

echo "CHAIN_FORMAL_PATCHER_FULLFT_PASS"

###############################################################################
# FINAL
###############################################################################

echo
echo "============================================================"
echo "FINAL"
echo "============================================================"

cat \
    "$TRAIN_OUT/paper_fidelity_manifest.json"

echo

echo \
    "OFFICIAL_COMMIT=$(git -C "$ROOT/vendor/MT-Patcher-official" rev-parse HEAD)"

echo \
    "PATCHER_DATA_SHA256=$(sha256sum "$FULL_BASE/patcher_sft.jsonl" | awk '{print $1}')"

echo

echo "MT_PATCHER_PAPER_FAITHFUL_V2_ALL_PASS"
