#!/usr/bin/env bash

source /workspace/mtpatcher/project_env.sh

export PYTHONPATH="$ROOT/vendor/trl-v1.0.0:$ROOT/vendor/peft0200${PYTHONPATH:+:$PYTHONPATH}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MODEL="$MODEL_ROOT/Qwen3-0.6B"
TRAIN="$DATA_ROOT/pilot_v2_grpo/train1024_seed20260821.jsonl"

PE_SCRIPT="$ROOT/scripts/pilot_v2/train_pegrl_inspired_v2_matchedtemp_ascend.py"
VANILLA_SCRIPT="$ROOT/scripts/pilot_v2/train_grpo_mt_bleu_g8_ascend.py"

DATE_TAG="20260823"

RUN_BASE="$RUN_ROOT/overnight_pegrl_m_sweep_${DATE_TAG}"
LOG_BASE="$LOG_ROOT/overnight_pegrl_m_sweep_${DATE_TAG}"

mkdir -p "$RUN_BASE" "$LOG_BASE"

echo "============================================================"
echo "OVERNIGHT PEGRL SWEEP"
echo "============================================================"
echo "start=$(date)"
echo "model=$MODEL"
echo "train=$TRAIN"
echo "PE=$PE_SCRIPT"
echo

# ============================================================
# Stage 0: M=8 technical acceptance
# ============================================================

ACCEPT_OUT="$RUN_BASE/accept_m8_step3"
ACCEPT_LOG="$LOG_BASE/accept_m8_step3.log"

mkdir -p "$ACCEPT_OUT"

echo "[PRECHECK] M=8 / 3 steps"

(
    export ASCEND_RT_VISIBLE_DEVICES=14

    python "$PE_SCRIPT" \
        --model "$MODEL" \
        --train "$TRAIN" \
        --output-dir "$ACCEPT_OUT" \
        --seed 20260821 \
        --max-steps 3 \
        --pe-children 8 \
        --pe-max-new-tokens 128
) > "$ACCEPT_LOG" 2>&1

if ! grep -q "PEGRL_INSPIRED_V2_MATCHEDTEMP_PASS" "$ACCEPT_LOG"; then
    echo "OVERNIGHT_PRECHECK_FAILED"
    tail -120 "$ACCEPT_LOG"
    return 1 2>/dev/null || true
fi

if ! grep -q "pe_children_per_parent = 8" "$ACCEPT_LOG"; then
    echo "OVERNIGHT_PRECHECK_BAD_M"
    tail -120 "$ACCEPT_LOG"
    return 1 2>/dev/null || true
fi

if ! grep -q "PE_AUX_LOSS_CALL" "$ACCEPT_LOG"; then
    echo "OVERNIGHT_PRECHECK_NO_PE_AUX"
    tail -120 "$ACCEPT_LOG"
    return 1 2>/dev/null || true
fi

echo "OVERNIGHT_M8_PRECHECK_PASS"

# ============================================================
# Helper
# ============================================================

launch_pe () {
    CARD="$1"
    M="$2"
    SEED="$3"

    NAME="pegrl_v2_n8_m${M}_seed${SEED}_step1536"

    OUT="$RUN_BASE/$NAME"
    LOG="$LOG_BASE/$NAME.log"

    mkdir -p "$OUT"

    (
        export ASCEND_RT_VISIBLE_DEVICES="$CARD"

        python "$PE_SCRIPT" \
            --model "$MODEL" \
            --train "$TRAIN" \
            --output-dir "$OUT" \
            --seed "$SEED" \
            --max-steps 1536 \
            --pe-children "$M" \
            --pe-max-new-tokens 128
    ) > "$LOG" 2>&1 &

    echo \
        "START PE card=$CARD M=$M seed=$SEED PID=$!"
}


launch_vanilla () {
    CARD="$1"
    SEED="$2"

    NAME="vanilla_g8_seed${SEED}_step1536"

    OUT="$RUN_BASE/$NAME"
    LOG="$LOG_BASE/$NAME.log"

    mkdir -p "$OUT"

    (
        export ASCEND_RT_VISIBLE_DEVICES="$CARD"

        python "$VANILLA_SCRIPT" \
            --model "$MODEL" \
            --train "$TRAIN" \
            --output-dir "$OUT" \
            --seed "$SEED" \
            --max-steps 1536
    ) > "$LOG" 2>&1 &

    echo \
        "START VANILLA card=$CARD seed=$SEED PID=$!"
}

# ============================================================
# 0-7: M=8, eight independent seeds
# ============================================================

launch_pe 0 8 20260821
launch_pe 1 8 20260822
launch_pe 2 8 20260823
launch_pe 3 8 20260824
launch_pe 4 8 20260825
launch_pe 5 8 20260826
launch_pe 6 8 20260827
launch_pe 7 8 20260828

# ============================================================
# 8-11: M=4, matched first four seeds
# ============================================================

launch_pe 8  4 20260821
launch_pe 9  4 20260822
launch_pe 10 4 20260823
launch_pe 11 4 20260824

# ============================================================
# 12-13: vanilla matched seeds
# ============================================================

launch_vanilla 12 20260821
launch_vanilla 13 20260822

echo
echo "============================================================"
echo "14 JOBS STARTED"
echo "cards used = 0..13"
echo "cards spare = 14,15"
echo "============================================================"

wait

echo
echo "============================================================"
echo "OVERNIGHT TRAINING FINISHED"
echo "end=$(date)"
echo "============================================================"
