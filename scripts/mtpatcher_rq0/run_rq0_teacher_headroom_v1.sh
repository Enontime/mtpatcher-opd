#!/usr/bin/env bash
set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

MODEL="$MODEL_ROOT/Qwen3-8B"
TOKENIZER="$MODEL_ROOT/Qwen3-8B"

EVAL="$ROOT/scripts/pilot_v2/eval_qwen3_06b_mt_ascend.py"
SCORE="$ROOT/scripts/pilot_v2/score_mt_jsonl.py"

WMT="$DATA_ROOT/pilot_v2_qwen3_06b/wmt24_zh_en998.jsonl"
FLORES="$DATA_ROOT/pilot_v2_qwen3_06b/flores_zh_en1012.jsonl"
CHALLENGE="$DATA_ROOT/pilot_v2_qwen3_06b/challenge_zh_en197.jsonl"

OUT="$RUN_ROOT/$EXP/rq0_teacher_qwen3_8b_eval_v1"
LOG_DIR="$LOG_ROOT/$EXP/rq0_teacher_qwen3_8b_eval_v1"

SUMMARY="$OUT/rq0_teacher_headroom_summary.json"

mkdir -p \
    "$OUT" \
    "$LOG_DIR"

echo "======================================================================"
echo "RQ0-A — QWEN3-8B TEACHER HEADROOM"
echo "======================================================================"

###############################################################################
# 1. INPUT AUDIT
###############################################################################

for F in \
    "$MODEL" \
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

echo "MODEL=$MODEL"
echo "WMT=$WMT"
echo "FLORES=$FLORES"
echo "CHALLENGE=$CHALLENGE"

echo "RQ0_A_INPUT_AUDIT_PASS"

###############################################################################
# 2. FIXED TEACHER EVALUATION
###############################################################################

PIDS=()

run_eval () {
    local NAME="$1"
    local DEVICE="$2"
    local DATA="$3"

    local DEST="$OUT/$NAME"
    local PRED="$DEST/predictions.jsonl"
    local METRIC="$DEST/metrics.json"
    local LOG="$LOG_DIR/${NAME}.log"

    mkdir -p "$DEST"

    if [ -f "$METRIC" ]; then
        echo "TEACHER_EVAL_ALREADY_COMPLETE name=$NAME"
        return
    fi

    (
        export ASCEND_RT_VISIBLE_DEVICES="$DEVICE"

        echo "==================================================" 
        echo "START name=$NAME device=$DEVICE"
        echo "=================================================="

        python "$EVAL" \
            --model "$MODEL" \
            --tokenizer "$TOKENIZER" \
            --input "$DATA" \
            --output "$PRED" \
            --method "rq0_qwen3_8b_teacher_${NAME}" \
            --batch-size 16 \
            --max-new-tokens 256 \
            --attn-implementation sdpa

        python "$SCORE" \
            --input "$PRED" \
            --output "$METRIC"

        echo "RQ0_TEACHER_DATASET_COMPLETE name=$NAME"

    ) > "$LOG" 2>&1 &

    PIDS+=("$!")

    echo "LAUNCHED name=$NAME pid=$! device=$DEVICE"
}

run_eval \
    "wmt24" \
    0 \
    "$WMT"

run_eval \
    "flores" \
    1 \
    "$FLORES"

run_eval \
    "challenge" \
    2 \
    "$CHALLENGE"

FAIL=0

for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        FAIL=1
    fi
done

if [ "$FAIL" -ne 0 ]; then
    echo "RQ0_A_TEACHER_EVAL_WORKER_FAILURE"
    false
fi

for NAME in wmt24 flores challenge; do
    if [ ! -f "$OUT/$NAME/metrics.json" ]; then
        echo "MISSING_METRIC=$NAME"
        false
    fi
done

echo "RQ0_A_TEACHER_EVAL_PASS"

###############################################################################
# 3. HEADROOM SUMMARY
###############################################################################

export OUT SUMMARY

python - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["OUT"])
summary_path = Path(os.environ["SUMMARY"])

names = (
    "wmt24",
    "flores",
    "challenge",
)

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

teacher = {}

for name in names:
    p = out / name / "metrics.json"

    obj = json.loads(
        p.read_text(
            encoding="utf-8"
        )
    )

    if "BLEU" not in obj or "chrF" not in obj:
        raise RuntimeError(
            f"Invalid metrics file: {p}"
        )

    teacher[name] = {
        "BLEU": float(obj["BLEU"]),
        "chrF": float(obj["chrF"]),
    }


def mean_metric(table, metric):
    return sum(
        table[name][metric]
        for name in names
    ) / len(names)


def mean_gap(metric):
    return sum(
        teacher[name][metric]
        - base[name][metric]
        for name in names
    ) / len(names)


base_avg_bleu = mean_metric(
    base,
    "BLEU",
)

teacher_avg_bleu = mean_metric(
    teacher,
    "BLEU",
)

gap_bleu = mean_gap(
    "BLEU"
)

gap_chrf = mean_gap(
    "chrF"
)

# Existing frozen anchors.
seqkd_full_gain = 1.465
mtpatcher_full_gain = 0.9363376197756678

if gap_bleu > 0:
    seqkd_transfer_ratio = (
        seqkd_full_gain / gap_bleu
    )

    mtpatcher_transfer_ratio = (
        mtpatcher_full_gain / gap_bleu
    )
else:
    seqkd_transfer_ratio = None
    mtpatcher_transfer_ratio = None


result = {
    "research_question":
        "RQ0-A: How much translation headroom exists between Qwen3-8B Teacher and Qwen3-0.6B Student?",

    "student":
        "Qwen3-0.6B",

    "teacher":
        "Qwen3-8B",

    "protocol":
        "same fixed greedy WMT24/FLORES/Challenge evaluator",

    "base":
        base,

    "teacher_metrics":
        teacher,

    "base_avg_bleu":
        base_avg_bleu,

    "teacher_avg_bleu":
        teacher_avg_bleu,

    "avg_teacher_student_gap_bleu":
        gap_bleu,

    "avg_teacher_student_gap_chrf":
        gap_chrf,

    "frozen_seqkd_full6565_gain_bleu":
        seqkd_full_gain,

    "frozen_mtpatcher_full_gain_bleu":
        mtpatcher_full_gain,

    "seqkd_fraction_of_teacher_gap":
        seqkd_transfer_ratio,

    "mtpatcher_fraction_of_teacher_gap":
        mtpatcher_transfer_ratio,
}


if gap_bleu < 2.0:
    diagnosis = (
        "LIMITED_HEADROOM: current teacher-student pair leaves "
        "little room for a +2.3~2.8 BLEU distillation target."
    )

elif gap_bleu < 3.0:
    diagnosis = (
        "MODERATE_HEADROOM: some improvement is possible, "
        "but reaching half of MT-PATCHER paper gains may be difficult."
    )

elif gap_bleu < 5.0:
    diagnosis = (
        "GOOD_HEADROOM: current setup has enough room; "
        "data scale and knowledge-transfer efficiency become primary suspects."
    )

else:
    diagnosis = (
        "LARGE_HEADROOM: current setup has substantial room; "
        "the existing +1.465 SeqKD gain transfers only a fraction of available teacher capability."
    )


result["diagnosis"] = diagnosis


summary_path.write_text(
    json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


print()
print(
    f"{'DATASET':12s}"
    f"{'STUDENT':>12s}"
    f"{'TEACHER':>12s}"
    f"{'GAP':>12s}"
)

print("-" * 48)

for name in names:
    s = base[name]["BLEU"]
    t = teacher[name]["BLEU"]

    print(
        f"{name:12s}"
        f"{s:12.4f}"
        f"{t:12.4f}"
        f"{t-s:+12.4f}"
    )


print()
print(
    "BASE_AVG_BLEU =",
    f"{base_avg_bleu:.6f}",
)

print(
    "TEACHER_AVG_BLEU =",
    f"{teacher_avg_bleu:.6f}",
)

print(
    "AVG_TEACHER_STUDENT_GAP_BLEU =",
    f"{gap_bleu:+.6f}",
)

print(
    "AVG_TEACHER_STUDENT_GAP_CHRF =",
    f"{gap_chrf:+.6f}",
)

print(
    "SEQKD_FULL6565_GAIN =",
    f"{seqkd_full_gain:+.6f}",
)

print(
    "MTPATCHER_FULL_GAIN =",
    f"{mtpatcher_full_gain:+.6f}",
)

if seqkd_transfer_ratio is not None:
    print(
        "SEQKD_TRANSFER_FRACTION =",
        f"{seqkd_transfer_ratio:.4f}",
    )

if mtpatcher_transfer_ratio is not None:
    print(
        "MTPATCHER_TRANSFER_FRACTION =",
        f"{mtpatcher_transfer_ratio:.4f}",
    )

print()
print(
    "DIAGNOSIS =",
    diagnosis,
)

print(
    "SUMMARY =",
    summary_path,
)

print("RQ0_A_TEACHER_HEADROOM_SUMMARY_PASS")
PY

echo
echo "======================================================================"
echo "FINAL SUMMARY"
echo "======================================================================"

cat "$SUMMARY"

echo
echo "RQ0_A_TEACHER_HEADROOM_ALL_PASS"
