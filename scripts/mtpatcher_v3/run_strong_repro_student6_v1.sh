#!/usr/bin/env bash
set -Eeuo pipefail

source /workspace/mtpatcher/project_env.sh

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

EXP="mtpatcher_v3_full6565_20260823"

D="$DATA_ROOT/$EXP"

DATA_DIR="$D/strong_repro_student_arms_freeze_v1"

RUN_DIR="$RUN_ROOT/$EXP/strong_repro_student6_v1"
LOG_DIR="$LOG_ROOT/$EXP/strong_repro_student6_v1"

STUDENT="$MODEL_ROOT/Qwen3-0.6B"

TRAINER="$ROOT/scripts/pilot_v2/train_qwen3_06b_full_sft_ascend.py"
EVAL="$ROOT/scripts/pilot_v2/eval_qwen3_06b_mt_ascend.py"
SCORE="$ROOT/scripts/pilot_v2/score_mt_jsonl.py"

WMT="$DATA_ROOT/pilot_v2_qwen3_06b/wmt24_zh_en998.jsonl"
FLORES="$DATA_ROOT/pilot_v2_qwen3_06b/flores_zh_en1012.jsonl"
CHALLENGE="$DATA_ROOT/pilot_v2_qwen3_06b/challenge_zh_en197.jsonl"

PASS="$RUN_DIR/STRONG_REPRO_STUDENT6_V1.PASS"
FAIL="$RUN_DIR/STRONG_REPRO_STUDENT6_V1.FAIL"

START_META="$RUN_DIR/start_meta_v1.txt"
RESULT_INDEX="$RUN_DIR/result_index_v1.txt"

mkdir -p "$RUN_DIR" "$LOG_DIR"

rm -f "$PASS" "$FAIL"

###############################################################################
# Frozen six datasets + hashes
###############################################################################

ARMS=(
    "K1_ALL_PE11792"
    "K2_CONSISTENT_PE4960"
    "RANDOM_K1_PE4960"
    "RANDOM_SEQKD4960"
    "SAME_SOURCE_SEQKD4960"
    "FULL_SEQKD20000"
)

FILES=(
    "$DATA_DIR/k1_all_pe11792_v1.jsonl"
    "$DATA_DIR/k2_consistent_pe4960_v1.jsonl"
    "$DATA_DIR/random_k1_pe4960_seed20260831_v1.jsonl"
    "$DATA_DIR/random_seqkd4960_seed20260831_v1.jsonl"
    "$DATA_DIR/same_source_seqkd4960_v1.jsonl"
    "$DATA_DIR/full_seqkd20000_v1.jsonl"
)

EXPECTED_ROWS=(
    11792
    4960
    4960
    4960
    4960
    20000
)

EXPECTED_SHA=(
    "7d8c5c1de8249db19e7e881174ef103c9ab6ebfa590623b6ef3190283684ed18"
    "7b666ab189e5d82a9623abceec0440bdcfef8fa210fb99c49b63bce511405dc7"
    "27a1bb424791e477e049655ac010f9da042014139bdcd1c07d987dbcbeedb27e"
    "1999e42830358f8636f07f75f129b8e006f0f9613e80d80bc9a8ac7a5dcd9f41"
    "ef758883aa49b932b535ef0a86e04ef7a3853394d3c919956faad634585d9253"
    "ee2e8edf4e17962e728aa46ae996086fe8eac57e10574ef9b46b0b31063f9dbe"
)

DEVICES=(
    0
    1
    2
    3
    4
    5
)

###############################################################################
# ETA
###############################################################################

START_EPOCH="$(date +%s)"

# LOW–MEDIUM confidence range.
ETA_MIN_SECONDS=5400
ETA_MAX_SECONDS=10800

{
    echo "START_EPOCH=$START_EPOCH"
    echo "START_CST=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "START_UTC=$(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"

    echo "ETA_MIN_CST=$(TZ=Asia/Shanghai date -d "@$((START_EPOCH + ETA_MIN_SECONDS))" '+%Y-%m-%d %H:%M:%S %Z')"
    echo "ETA_MAX_CST=$(TZ=Asia/Shanghai date -d "@$((START_EPOCH + ETA_MAX_SECONDS))" '+%Y-%m-%d %H:%M:%S %Z')"

    echo "ETA_MIN_UTC=$(TZ=UTC date -d "@$((START_EPOCH + ETA_MIN_SECONDS))" '+%Y-%m-%d %H:%M:%S %Z')"
    echo "ETA_MAX_UTC=$(TZ=UTC date -d "@$((START_EPOCH + ETA_MAX_SECONDS))" '+%Y-%m-%d %H:%M:%S %Z')"

    echo "ETA_CONFIDENCE=LOW-MEDIUM"
} > "$START_META"

echo "======================================================================"
echo "STRONG REPRO STUDENT-LEVEL SIX-ARM V1"
echo "======================================================================"
echo
cat "$START_META"
echo
echo "预计总运行时间：约 1.5–3 小时"
echo "包括：3 epoch Student SFT + WMT/FLORES/Challenge evaluation"
echo


trap '
rc=$?
echo
echo "======================================================================"
echo "STRONG REPRO STUDENT6 V1 FAILED"
echo "return_code=$rc"
echo "time=$(date -Iseconds)"
echo "======================================================================"
touch "'"$FAIL"'"
' ERR


###############################################################################
# STAGE 1/5 — frozen input preflight
###############################################################################

echo "======================================================================"
echo "STAGE 1/5 — FROZEN INPUT PREFLIGHT"
echo "======================================================================"

for F in \
    "$TRAINER" \
    "$EVAL" \
    "$SCORE" \
    "$WMT" \
    "$FLORES" \
    "$CHALLENGE" \
    "$STUDENT/config.json"
do
    if [ ! -f "$F" ]; then
        echo "MISSING REQUIRED FILE: $F"
        false
    fi
done

python -m py_compile \
    "$TRAINER" \
    "$EVAL" \
    "$SCORE"

for I in "${!ARMS[@]}"; do

    ARM="${ARMS[$I]}"
    DATA="${FILES[$I]}"
    WANT_ROWS="${EXPECTED_ROWS[$I]}"
    WANT_SHA="${EXPECTED_SHA[$I]}"

    if [ ! -f "$DATA" ]; then
        echo "MISSING DATASET arm=$ARM path=$DATA"
        false
    fi

    GOT_ROWS="$(wc -l < "$DATA")"
    GOT_SHA="$(sha256sum "$DATA" | awk '{print $1}')"

    echo
    echo "ARM=$ARM"
    echo "  DATA=$DATA"
    echo "  rows=$GOT_ROWS"
    echo "  expected_rows=$WANT_ROWS"
    echo "  sha256=$GOT_SHA"
    echo "  expected_sha=$WANT_SHA"

    if [ "$GOT_ROWS" -ne "$WANT_ROWS" ]; then
        echo "ROW_COUNT_MISMATCH arm=$ARM"
        false
    fi

    if [ "$GOT_SHA" != "$WANT_SHA" ]; then
        echo "SHA_MISMATCH arm=$ARM"
        false
    fi

done

echo
echo "===== TRAINING RECIPE ====="
echo "model=$STUDENT"
echo "lr=2e-5"
echo "epochs=3"
echo "batch_size=4"
echo "grad_accum=4"
echo "effective_batch=16"
echo "max_length=1024"
echo "warmup_ratio=0.03"
echo "weight_decay=0.01"
echo "max_grad_norm=1.0"
echo "seed=20260820"
echo "num_workers=2"

echo
echo "===== EVALUATION RECIPE ====="
echo "WMT=$WMT"
echo "FLORES=$FLORES"
echo "CHALLENGE=$CHALLENGE"
echo "batch_size=16"
echo "max_new_tokens=256"
echo "attn_implementation=sdpa"

echo
echo "FROZEN_INPUT_PREFLIGHT_PASS"


###############################################################################
# STAGE 2/5 — run each arm independently.
#
# Each ARM pipeline:
#   train epoch1..3
#   -> WMT eval + score
#   -> FLORES eval + score
#   -> Challenge eval + score
#
# Six pipelines run concurrently on devices 0..5.
###############################################################################

echo
echo "======================================================================"
echo "STAGE 2/5 — SIX PARALLEL STUDENT PIPELINES"
echo "======================================================================"

declare -A PIDS


for I in "${!ARMS[@]}"; do

    ARM="${ARMS[$I]}"
    DATA="${FILES[$I]}"
    DEVICE="${DEVICES[$I]}"

    ARM_RUN="$RUN_DIR/$ARM"
    ARM_LOG_DIR="$LOG_DIR/$ARM"

    ARM_LOG="$ARM_LOG_DIR/pipeline.log"
    ARM_PASS="$ARM_RUN/ARM_PIPELINE.PASS"

    mkdir -p \
        "$ARM_RUN" \
        "$ARM_LOG_DIR" \
        "$ARM_RUN/eval"

    if [ -f "$ARM_PASS" ]; then

        echo "ARM_ALREADY_COMPLETE arm=$ARM"

        continue
    fi


    # Do not silently overwrite a suspicious partial Student run.
    if \
        [ -d "$ARM_RUN/epoch1" ] \
        || [ -d "$ARM_RUN/epoch2" ] \
        || [ -d "$ARM_RUN/epoch3" ]
    then

        if \
            [ ! -f "$ARM_RUN/training_manifest.json" ] \
            || [ ! -f "$ARM_RUN/epoch3/config.json" ]
        then

            echo
            echo "PARTIAL_TRAINING_OUTPUT_DETECTED arm=$ARM"
            echo "path=$ARM_RUN"
            echo "Refusing to silently overwrite it."
            false
        fi
    fi


    echo \
        "ARM_PIPELINE_LAUNCH " \
        "arm=$ARM " \
        "device=$DEVICE"


    (
        set -Eeuo pipefail

        source /workspace/mtpatcher/project_env.sh

        export TOKENIZERS_PARALLELISM=false
        export PYTORCH_ALLOC_CONF=expandable_segments:True

        export ASCEND_RT_VISIBLE_DEVICES="$DEVICE"

        ARM_START="$(date +%s)"

        echo "======================================================================"
        echo "ARM PIPELINE START"
        echo "ARM=$ARM"
        echo "DEVICE=$DEVICE"
        echo "DATA=$DATA"
        echo "START_CST=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
        echo "START_UTC=$(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
        echo "======================================================================"


        #######################################################################
        # TRAIN
        #######################################################################

        if \
            [ -f "$ARM_RUN/training_manifest.json" ] \
            && \
            [ -f "$ARM_RUN/epoch3/config.json" ]
        then

            echo "TRAIN_ALREADY_COMPLETE arm=$ARM"

        else

            echo
            echo "===== TRAIN START arm=$ARM ====="

            python -u "$TRAINER" \
                --model "$STUDENT" \
                --train "$DATA" \
                --output-dir "$ARM_RUN" \
                --lr 2e-5 \
                --epochs 3 \
                --batch-size 4 \
                --grad-accum 4 \
                --max-length 1024 \
                --warmup-ratio 0.03 \
                --weight-decay 0.01 \
                --max-grad-norm 1.0 \
                --seed 20260820 \
                --num-workers 2

            echo "===== TRAIN PROCESS RETURNED arm=$ARM ====="
        fi


        if [ ! -f "$ARM_RUN/training_manifest.json" ]; then
            echo "MISSING training_manifest.json arm=$ARM"
            false
        fi

        if [ ! -f "$ARM_RUN/epoch3/config.json" ]; then
            echo "MISSING epoch3/config.json arm=$ARM"
            false
        fi

        echo "TRAIN_COMPLETE arm=$ARM"


        #######################################################################
        # EVALUATION
        #######################################################################

        MODEL="$ARM_RUN/epoch3"

        for SPEC in \
            "wmt24:$WMT" \
            "flores:$FLORES" \
            "challenge:$CHALLENGE"
        do

            NAME="${SPEC%%:*}"
            EVAL_DATA="${SPEC#*:}"

            PRED="$ARM_RUN/eval/${NAME}.pred.jsonl"
            METRIC="$ARM_RUN/eval/${NAME}.metrics.json"

            if [ -f "$METRIC" ]; then

                echo \
                    "EVAL_ALREADY_COMPLETE " \
                    "arm=$ARM dataset=$NAME"

                continue
            fi


            PRED_TMP="$ARM_RUN/eval/${NAME}.pred.tmp.$$.jsonl"
            METRIC_TMP="$ARM_RUN/eval/${NAME}.metrics.tmp.$$.json"

            rm -f \
                "$PRED_TMP" \
                "$METRIC_TMP"

            echo
            echo \
                "===== EVAL START " \
                "arm=$ARM dataset=$NAME ====="

            python -u "$EVAL" \
                --model "$MODEL" \
                --tokenizer "$MODEL" \
                --input "$EVAL_DATA" \
                --output "$PRED_TMP" \
                --method "strong_repro_${ARM}_${NAME}" \
                --batch-size 16 \
                --max-new-tokens 256 \
                --attn-implementation sdpa

            python "$SCORE" \
                --input "$PRED_TMP" \
                --output "$METRIC_TMP"

            mv "$PRED_TMP" "$PRED"
            mv "$METRIC_TMP" "$METRIC"

            echo \
                "EVAL_COMPLETE " \
                "arm=$ARM dataset=$NAME"

            echo \
                "METRIC_FILE=$METRIC"

            cat "$METRIC"
        done


        #######################################################################
        # ARM COMPLETE
        #######################################################################

        for NAME in \
            wmt24 \
            flores \
            challenge
        do

            test -f \
                "$ARM_RUN/eval/${NAME}.metrics.json"
        done

        ARM_END="$(date +%s)"
        ARM_ELAPSED=$((ARM_END - ARM_START))

        touch "$ARM_PASS"

        echo
        echo "======================================================================"
        echo "ARM_PIPELINE_PASS"
        echo "ARM=$ARM"
        echo "DEVICE=$DEVICE"
        echo "ELAPSED_SECONDS=$ARM_ELAPSED"
        echo "FINISH_CST=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
        echo "FINISH_UTC=$(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
        echo "======================================================================"

    ) > "$ARM_LOG" 2>&1 &

    PIDS[$ARM]=$!

    echo \
        "ARM_PIPELINE_PID " \
        "arm=$ARM " \
        "pid=${PIDS[$ARM]}"

done


###############################################################################
# Wait for all non-skipped pipelines.
###############################################################################

FAILED_ARMS=()

for ARM in "${ARMS[@]}"; do

    if [ -z "${PIDS[$ARM]+x}" ]; then
        continue
    fi

    PID="${PIDS[$ARM]}"

    if wait "$PID"; then

        echo \
            "ARM_PROCESS_FINISH " \
            "arm=$ARM status=0"

    else

        STATUS=$?

        echo \
            "ARM_PROCESS_FINISH " \
            "arm=$ARM status=$STATUS"

        FAILED_ARMS+=("$ARM")
    fi

done


if [ "${#FAILED_ARMS[@]}" -ne 0 ]; then

    echo
    echo "FAILED_ARMS=${FAILED_ARMS[*]}"

    for ARM in "${FAILED_ARMS[@]}"; do

        echo
        echo "===== FAILED ARM LOG: $ARM ====="

        tail -100 \
            "$LOG_DIR/$ARM/pipeline.log" \
            2>/dev/null || true
    done

    false
fi


###############################################################################
# STAGE 3/5 — verify all training/evaluation outputs.
###############################################################################

echo
echo "======================================================================"
echo "STAGE 3/5 — GLOBAL COMPLETION AUDIT"
echo "======================================================================"

for ARM in "${ARMS[@]}"; do

    ARM_RUN="$RUN_DIR/$ARM"

    test -f \
        "$ARM_RUN/ARM_PIPELINE.PASS"

    test -f \
        "$ARM_RUN/training_manifest.json"

    test -f \
        "$ARM_RUN/epoch3/config.json"

    for NAME in \
        wmt24 \
        flores \
        challenge
    do

        test -f \
            "$ARM_RUN/eval/${NAME}.pred.jsonl"

        test -f \
            "$ARM_RUN/eval/${NAME}.metrics.json"
    done

    echo "GLOBAL_ARM_PASS arm=$ARM"

done

echo "GLOBAL_COMPLETION_AUDIT_PASS"


###############################################################################
# STAGE 4/5 — freeze result index + provenance
###############################################################################

echo
echo "======================================================================"
echo "STAGE 4/5 — RESULT INDEX"
echo "======================================================================"

{
    echo "PROTOCOL=STRONG_REPRO_STUDENT6_V1"
    echo "DATE=$(date -Iseconds)"

    echo
    echo "===== TRAINING RECIPE ====="
    echo "student=$STUDENT"
    echo "lr=2e-5"
    echo "epochs=3"
    echo "batch_size=4"
    echo "grad_accum=4"
    echo "effective_batch_size=16"
    echo "max_length=1024"
    echo "warmup_ratio=0.03"
    echo "weight_decay=0.01"
    echo "max_grad_norm=1.0"
    echo "seed=20260820"
    echo "num_workers=2"

    echo
    echo "===== CODE SHA256 ====="

    sha256sum \
        "$TRAINER" \
        "$EVAL" \
        "$SCORE"

    echo
    echo "===== STUDENT MODEL CONFIG SHA256 ====="

    for F in \
        "$STUDENT/config.json" \
        "$STUDENT/generation_config.json" \
        "$STUDENT/tokenizer_config.json" \
        "$STUDENT/tokenizer.json"
    do
        if [ -f "$F" ]; then
            sha256sum "$F"
        fi
    done

    echo
    echo "===== EVAL DATA SHA256 ====="

    sha256sum \
        "$WMT" \
        "$FLORES" \
        "$CHALLENGE"

    echo
    echo "===== RESULTS ====="

    for ARM in "${ARMS[@]}"; do

        ARM_RUN="$RUN_DIR/$ARM"

        echo
        echo "### ARM=$ARM"

        echo "CHECKPOINT=$ARM_RUN/epoch3"

        for NAME in \
            wmt24 \
            flores \
            challenge
        do

            METRIC="$ARM_RUN/eval/${NAME}.metrics.json"

            echo
            echo "DATASET=$NAME"
            echo "METRIC=$METRIC"
            sha256sum "$METRIC"
            cat "$METRIC"
        done
    done

} > "$RESULT_INDEX"

cat "$RESULT_INDEX"


###############################################################################
# STAGE 5/5 — final sentinel
###############################################################################

echo
echo "======================================================================"
echo "STAGE 5/5 — FINALIZE"
echo "======================================================================"

END_EPOCH="$(date +%s)"
ELAPSED=$((END_EPOCH - START_EPOCH))

touch "$PASS"
rm -f "$FAIL"

echo
echo "======================================================================"
echo "STRONG REPRO STUDENT6 V1 PASS"
echo "======================================================================"
echo "TOTAL_ELAPSED_SECONDS=$ELAPSED"
echo "FINISH_CST=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
echo "FINISH_UTC=$(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
echo
echo "RESULT_INDEX=$RESULT_INDEX"
echo "PASS=$PASS"
echo "======================================================================"

