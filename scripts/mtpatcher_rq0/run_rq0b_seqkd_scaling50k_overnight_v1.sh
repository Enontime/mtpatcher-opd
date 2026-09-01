#!/usr/bin/env bash
set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

SCRIPT_DIR="$ROOT/scripts/mtpatcher_rq0"
EXP_DATA="$DATA_ROOT/$EXP"
EXP_RUN="$RUN_ROOT/$EXP"
EXP_LOG="$LOG_ROOT/$EXP"

SOURCE="$EXP_DATA/rq0_newscrawl50k_sources_v1.jsonl"

EXPECTED_SOURCE_SHA="57a1a5b0b757499b1849be4d3ea8acb76ba2f97718fd89253ede396a1ac1b2fe"

TEACHER="$MODEL_ROOT/Qwen3-8B"
STUDENT="$MODEL_ROOT/Qwen3-0.6B"

GEN="$SCRIPT_DIR/generate_seqkd50k_teacher_v1.py"
MERGE="$SCRIPT_DIR/merge_slice_seqkd50k_v1.py"
PREFLIGHT="$SCRIPT_DIR/preflight_seqkd50k_tokens_v1.py"

TRAINER="$ROOT/scripts/pilot_v2/train_qwen3_06b_full_sft_ascend.py"
EVAL="$ROOT/scripts/pilot_v2/eval_qwen3_06b_mt_ascend.py"
SCORE="$ROOT/scripts/pilot_v2/score_mt_jsonl.py"

WMT="$DATA_ROOT/pilot_v2_qwen3_06b/wmt24_zh_en998.jsonl"
FLORES="$DATA_ROOT/pilot_v2_qwen3_06b/flores_zh_en1012.jsonl"
CHALLENGE="$DATA_ROOT/pilot_v2_qwen3_06b/challenge_zh_en197.jsonl"

SHARD_DIR="$EXP_DATA/rq0_seqkd50k_teacher_shards_v1"
SEQKD_DIR="$EXP_DATA/rq0_seqkd_scaling50k_v1"

RUN_ROOT_RQ0="$EXP_RUN/rq0_seqkd_scaling50k_v1"
LOG_ROOT_RQ0="$EXP_LOG/rq0_seqkd_scaling50k_v1"

mkdir -p \
    "$SHARD_DIR" \
    "$SEQKD_DIR" \
    "$RUN_ROOT_RQ0" \
    "$LOG_ROOT_RQ0"


echo "======================================================================"
echo "RQ0-B — SEQKD DATA SCALING 6.5K / 10K / 20K / 50K"
echo "======================================================================"


###############################################################################
# STAGE 1 — STATIC PREFLIGHT
###############################################################################

echo
echo "======================================================================"
echo "STAGE 1/7 — STATIC PREFLIGHT"
echo "======================================================================"

for F in \
    "$SOURCE" \
    "$TEACHER" \
    "$STUDENT" \
    "$GEN" \
    "$MERGE" \
    "$PREFLIGHT" \
    "$TRAINER" \
    "$EVAL" \
    "$SCORE" \
    "$WMT" \
    "$FLORES" \
    "$CHALLENGE"
do
    if [ ! -e "$F" ]; then
        echo "MISSING=$F"
        false
    fi
done

ROWS="$(wc -l < "$SOURCE")"
SHA="$(sha256sum "$SOURCE" | awk '{print $1}')"

echo "SOURCE_ROWS=$ROWS"
echo "SOURCE_SHA256=$SHA"

if [ "$ROWS" -ne 50000 ]; then
    echo "BAD_SOURCE_ROWS"
    false
fi

if [ "$SHA" != "$EXPECTED_SOURCE_SHA" ]; then
    echo "BAD_SOURCE_SHA"
    false
fi

python -m py_compile \
    "$GEN" \
    "$MERGE" \
    "$PREFLIGHT"

echo "RQ0_B_STATIC_PREFLIGHT_PASS"


###############################################################################
# STAGE 2 — QWEN3-8B TEACHER GENERATION, 16 NPUs
###############################################################################

echo
echo "======================================================================"
echo "STAGE 2/7 — 50K TEACHER GENERATION"
echo "======================================================================"

PIDS=()

for DEVICE in $(seq 0 15); do

    OUT="$SHARD_DIR/device_${DEVICE}.jsonl"
    LOG="$LOG_ROOT_RQ0/teacher_device_${DEVICE}.log"

    python -u "$GEN" \
        --input "$SOURCE" \
        --output "$OUT" \
        --model "$TEACHER" \
        --device-id "$DEVICE" \
        --world-size 16 \
        --batch-size 16 \
        --max-new-tokens 512 \
        > "$LOG" 2>&1 &

    PIDS+=("$!")

    echo \
        "TEACHER_WORKER_LAUNCHED " \
        "device=$DEVICE pid=$!"
done

FAIL=0

for PID in "${PIDS[@]}"; do

    if ! wait "$PID"; then
        FAIL=1
    fi
done

if [ "$FAIL" -ne 0 ]; then

    echo "RQ0_B_TEACHER_WORKER_FAILURE"

    for DEVICE in $(seq 0 15); do
        echo "----- DEVICE $DEVICE -----"

        tail -n 40 \
            "$LOG_ROOT_RQ0/teacher_device_${DEVICE}.log" \
            2>/dev/null || true
    done

    false
fi

TOTAL=0

for DEVICE in $(seq 0 15); do

    F="$SHARD_DIR/device_${DEVICE}.jsonl"

    N="$(wc -l < "$F")"

    echo \
        "TEACHER_DEVICE_ROWS " \
        "device=$DEVICE rows=$N"

    TOTAL=$((TOTAL + N))
done

echo "TOTAL_TEACHER_ROWS=$TOTAL"

if [ "$TOTAL" -ne 50000 ]; then
    echo "TEACHER_TOTAL_ROW_FAILURE"
    false
fi

echo "RQ0_B_TEACHER_50K_PASS"


###############################################################################
# STAGE 3 — MERGE / SLICE / TOKEN AUDIT
###############################################################################

echo
echo "======================================================================"
echo "STAGE 3/7 — MERGE / SLICE / TOKEN AUDIT"
echo "======================================================================"

python "$MERGE" \
    --sources "$SOURCE" \
    --shard-dir "$SHARD_DIR" \
    --output-dir "$SEQKD_DIR"

FULL="$SEQKD_DIR/seqkd_newscrawl50000_qwen3_8b_v1.jsonl"

python "$PREFLIGHT" \
    --model "$STUDENT" \
    --data "$FULL" \
    --max-length 1024

for N in 6565 10000 20000 50000; do

    DATA="$SEQKD_DIR/seqkd_newscrawl${N}_qwen3_8b_v1.jsonl"

    ACTUAL="$(wc -l < "$DATA")"

    if [ "$ACTUAL" -ne "$N" ]; then
        echo \
            "SLICE_CARDINALITY_FAILURE " \
            "n=$N actual=$ACTUAL"

        false
    fi

    echo \
        "SLICE_READY " \
        "n=$N " \
        "sha=$(sha256sum "$DATA" | awk '{print $1}')"
done

echo "RQ0_B_DATA_READY_PASS"


###############################################################################
# STAGE 4 — FOUR PARALLEL STUDENT TRAINING RUNS
###############################################################################

echo
echo "======================================================================"
echo "STAGE 4/7 — PARALLEL STUDENT TRAINING"
echo "======================================================================"

TRAIN_PIDS=()

SIZES=(6565 10000 20000 50000)
DEVICES=(0 1 2 3)

for POS in "${!SIZES[@]}"; do

    N="${SIZES[$POS]}"
    DEVICE="${DEVICES[$POS]}"

    DATA="$SEQKD_DIR/seqkd_newscrawl${N}_qwen3_8b_v1.jsonl"

    OUT="$RUN_ROOT_RQ0/seqkd_newscrawl${N}_b4ga4"

    LOG="$LOG_ROOT_RQ0/train_${N}.log"

    if \
        [ -f "$OUT/training_manifest.json" ] \
        && \
        [ -f "$OUT/epoch3/config.json" ]
    then
        echo \
            "TRAIN_ALREADY_COMPLETE " \
            "n=$N"

        continue
    fi

    (
        export ASCEND_RT_VISIBLE_DEVICES="$DEVICE"

        python -u "$TRAINER" \
            --model "$STUDENT" \
            --train "$DATA" \
            --output-dir "$OUT" \
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

    ) > "$LOG" 2>&1 &

    TRAIN_PIDS+=("$!")

    echo \
        "TRAIN_LAUNCHED " \
        "n=$N device=$DEVICE pid=$!"
done

TRAIN_FAIL=0

for PID in "${TRAIN_PIDS[@]}"; do

    if ! wait "$PID"; then
        TRAIN_FAIL=1
    fi
done

if [ "$TRAIN_FAIL" -ne 0 ]; then

    echo "RQ0_B_TRAIN_FAILURE"

    for N in "${SIZES[@]}"; do
        echo "----- TRAIN $N -----"

        tail -n 50 \
            "$LOG_ROOT_RQ0/train_${N}.log" \
            2>/dev/null || true
    done

    false
fi

for N in "${SIZES[@]}"; do

    OUT="$RUN_ROOT_RQ0/seqkd_newscrawl${N}_b4ga4"

    test -f \
        "$OUT/training_manifest.json"

    test -f \
        "$OUT/epoch3/config.json"

    echo \
        "TRAIN_COMPLETE n=$N"
done

echo "RQ0_B_ALL_TRAIN_PASS"


###############################################################################
# STAGE 5 — 12 FIXED EVALUATIONS IN PARALLEL
###############################################################################

echo
echo "======================================================================"
echo "STAGE 5/7 — FIXED EPOCH3 EVALUATIONS"
echo "======================================================================"

EVAL_PIDS=()

DEVICE_COUNTER=0

for N in "${SIZES[@]}"; do

    MODEL="$RUN_ROOT_RQ0/seqkd_newscrawl${N}_b4ga4/epoch3"

    for SPEC in \
        "wmt24:$WMT" \
        "flores:$FLORES" \
        "challenge:$CHALLENGE"
    do

        NAME="${SPEC%%:*}"
        DATA="${SPEC#*:}"

        DEVICE="$DEVICE_COUNTER"
        DEVICE_COUNTER=$((DEVICE_COUNTER + 1))

        OUT_DIR="$RUN_ROOT_RQ0/eval_${N}"
        mkdir -p "$OUT_DIR"

        PRED="$OUT_DIR/${NAME}.jsonl"
        METRIC="$OUT_DIR/${NAME}_metrics.json"

        LOG="$LOG_ROOT_RQ0/eval_${N}_${NAME}.log"

        if [ -f "$METRIC" ]; then

            echo \
                "EVAL_ALREADY_COMPLETE " \
                "n=$N name=$NAME"

            continue
        fi

        (
            export ASCEND_RT_VISIBLE_DEVICES="$DEVICE"

            python -u "$EVAL" \
                --model "$MODEL" \
                --tokenizer "$MODEL" \
                --input "$DATA" \
                --output "$PRED" \
                --method "rq0_seqkd_newscrawl${N}_${NAME}" \
                --batch-size 16 \
                --max-new-tokens 256 \
                --attn-implementation sdpa

            python "$SCORE" \
                --input "$PRED" \
                --output "$METRIC"

        ) > "$LOG" 2>&1 &

        EVAL_PIDS+=("$!")

        echo \
            "EVAL_LAUNCHED " \
            "n=$N name=$NAME " \
            "device=$DEVICE pid=$!"
    done
done

EVAL_FAIL=0

for PID in "${EVAL_PIDS[@]}"; do

    if ! wait "$PID"; then
        EVAL_FAIL=1
    fi
done

if [ "$EVAL_FAIL" -ne 0 ]; then

    echo "RQ0_B_EVAL_FAILURE"

    for N in "${SIZES[@]}"; do

        for NAME in \
            wmt24 \
            flores \
            challenge
        do

            echo \
                "----- EVAL $N $NAME -----"

            tail -n 40 \
                "$LOG_ROOT_RQ0/eval_${N}_${NAME}.log" \
                2>/dev/null || true
        done
    done

    false
fi

for N in "${SIZES[@]}"; do

    for NAME in \
        wmt24 \
        flores \
        challenge
    do

        test -f \
            "$RUN_ROOT_RQ0/eval_${N}/${NAME}_metrics.json"
    done
done

echo "RQ0_B_ALL_EVAL_PASS"


###############################################################################
# STAGE 6 — SCALING SUMMARY
###############################################################################

echo
echo "======================================================================"
echo "STAGE 6/7 — SCALING SUMMARY"
echo "======================================================================"

export RUN_ROOT_RQ0
export SEQKD_DIR

python - <<'PY'
import json
import os
from pathlib import Path


root = Path(
    os.environ["RUN_ROOT_RQ0"]
)

data_root = Path(
    os.environ["SEQKD_DIR"]
)

sizes = [
    6565,
    10000,
    20000,
    50000,
]

sets = [
    "wmt24",
    "flores",
    "challenge",
]


base = {
    "wmt24": {
        "BLEU": 15.536214,
        "chrF": 45.537530,
    },

    "flores": {
        "BLEU": 19.971480,
        "chrF": 50.860857,
    },

    "challenge": {
        "BLEU": 16.537871,
        "chrF": 45.758287,
    },
}


results = {}


for n in sizes:

    metrics = {}

    for name in sets:

        p = (
            root
            / f"eval_{n}"
            / f"{name}_metrics.json"
        )

        obj = json.loads(
            p.read_text(
                encoding="utf-8"
            )
        )

        metrics[name] = {
            "BLEU":
                float(obj["BLEU"]),

            "chrF":
                float(obj["chrF"]),
        }

    avg_bleu_delta = sum(
        metrics[name]["BLEU"]
        - base[name]["BLEU"]
        for name in sets
    ) / len(sets)

    avg_chrf_delta = sum(
        metrics[name]["chrF"]
        - base[name]["chrF"]
        for name in sets
    ) / len(sets)

    results[str(n)] = {
        "metrics":
            metrics,

        "avg_delta_bleu":
            avg_bleu_delta,

        "avg_delta_chrf":
            avg_chrf_delta,
    }


threshold = 2.3

first_reach = None

for n in sizes:

    if (
        results[str(n)]
        ["avg_delta_bleu"]
        >= threshold
    ):
        first_reach = n
        break


summary = {
    "research_question":
        "RQ0-B: Is limited SeqKD gain primarily caused by insufficient absolute distillation data scale?",

    "student":
        "Qwen3-0.6B",

    "teacher":
        "Qwen3-8B",

    "data":
        "WMT News Crawl 2023 zh",

    "nested_sizes":
        sizes,

    "protocol":
        (
            "Each natural dataset size is trained "
            "for 3 epochs with the frozen b4ga4 recipe. "
            "This is a natural data-scaling comparison "
            "and is not compute-matched."
        ),

    "base":
        base,

    "results":
        results,

    "historical_old_seqkd_full6565_avg_delta_bleu":
        1.465,

    "teacher_student_headroom_avg_bleu":
        9.601661995391678,

    "half_paper_target_bleu":
        threshold,

    "first_size_reaching_target":
        first_reach,
}


if first_reach is not None:

    diagnosis = (
        f"SCALE_SIGNAL_POSITIVE: "
        f"{first_reach} rows reaches "
        f"the +{threshold:.1f} BLEU target. "
        f"Absolute distillation data scale is "
        f"a major contributor to the previous "
        f"limited-gain regime."
    )

elif (
    results["50000"]
    ["avg_delta_bleu"]
    >= 2.0
):

    diagnosis = (
        "SCALE_SIGNAL_MODERATE: 50k gives a "
        "clear scaling gain but remains below "
        "the +2.3 BLEU half-paper target. "
        "Scale matters, while additional "
        "transfer/optimization limitations remain."
    )

else:

    diagnosis = (
        "SCALE_SIGNAL_WEAK: even 50k remains "
        "below +2 BLEU. Data quantity alone "
        "does not explain the transfer gap; "
        "target quality/training recipe/knowledge "
        "transfer should become the next focus."
    )


summary["diagnosis"] = diagnosis


summary_path = (
    root
    / "rq0b_seqkd_scaling_summary.json"
)

summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


print()
print(
    f"{'ROWS':>8s} "
    f"{'WMT':>10s} "
    f"{'FLORES':>10s} "
    f"{'CHALL':>10s} "
    f"{'AVG Δ':>10s} "
    f"{'chrF Δ':>10s}"
)

print("-" * 66)


for n in sizes:

    r = results[str(n)]

    w = (
        r["metrics"]["wmt24"]["BLEU"]
        - base["wmt24"]["BLEU"]
    )

    f = (
        r["metrics"]["flores"]["BLEU"]
        - base["flores"]["BLEU"]
    )

    c = (
        r["metrics"]["challenge"]["BLEU"]
        - base["challenge"]["BLEU"]
    )

    print(
        f"{n:8d} "
        f"{w:+10.4f} "
        f"{f:+10.4f} "
        f"{c:+10.4f} "
        f"{r['avg_delta_bleu']:+10.4f} "
        f"{r['avg_delta_chrf']:+10.4f}"
    )


print()

print(
    "OLD_SEQKD_FULL6565_AVG_DELTA_BLEU = "
    "+1.465000"
)

for n in sizes:

    print(
        f"NEW_SEQKD_{n}_AVG_DELTA_BLEU = "
        f"{results[str(n)]['avg_delta_bleu']:+.6f}"
    )


print(
    "HALF_PAPER_TARGET = "
    "+2.300000"
)

print(
    "FIRST_SIZE_REACHING_TARGET =",
    first_reach,
)

print(
    "DIAGNOSIS =",
    diagnosis,
)

print(
    "SUMMARY =",
    summary_path,
)

print(
    "RQ0_B_SCALING_SUMMARY_PASS"
)
PY


###############################################################################
# STAGE 7 — FINAL
###############################################################################

echo
echo "======================================================================"
echo "STAGE 7/7 — FINAL"
echo "======================================================================"

cat \
    "$RUN_ROOT_RQ0/rq0b_seqkd_scaling_summary.json"

echo
echo "RQ0_B_SEQKD_SCALING50K_ALL_PASS"

