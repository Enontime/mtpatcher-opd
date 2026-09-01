#!/usr/bin/env bash
set -Eeuo pipefail

source /workspace/mtpatcher/project_env.sh

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

###############################################################################
# FROZEN QUESTION
#
# Is the seed-1 WA near-null / mixed BLEU result robust to a paired new
# Student-training seed?
###############################################################################

EXP="mtpatcher_v3_full6565_20260823"
D="$DATA_ROOT/$EXP"

SEED=20260901

STUDENT="$MODEL_ROOT/Qwen3-0.6B"

TRAINER="$ROOT/scripts/pilot_v2/train_qwen3_06b_full_sft_ascend.py"
EVAL="$ROOT/scripts/pilot_v2/eval_qwen3_06b_mt_ascend.py"
SCORE="$ROOT/scripts/pilot_v2/score_mt_jsonl.py"

###############################################################################
# Exact frozen datasets from completed productions
###############################################################################

PDS_DATA="$D/strong_repro_pds_full_to_student_v2/k1all_plus_pds_v2.jsonl"

PDS_WA_DATA="$D/strong_repro_wa_full_v1/pe_pds_wa_train_full_v1.jsonl"

SEED1_PDS_RESULT="$D/strong_repro_pds_full_to_student_v2/final_student_bleu_v2.json"

SEED1_WA_RESULT="$D/strong_repro_wa_full_v1/final_pe_pds_wa_bleu_full_v1.json"

###############################################################################
# Seed-2 outputs
###############################################################################

OUT="$D/wa_paired_seed2_v1"

RUN_ROOT2="$RUN_ROOT/$EXP/wa_paired_seed2_v1"

PDS_RUN="$RUN_ROOT2/PE_PDS"
WA_RUN="$RUN_ROOT2/PE_PDS_WA"

PDS_LOG="$OUT/PE_PDS"
WA_LOG="$OUT/PE_PDS_WA"

FINAL="$OUT/wa_paired_seed2_final_v1.json"

PASS="$OUT/WA_PAIRED_SEED2_V1.PASS"
FAIL="$OUT/WA_PAIRED_SEED2_V1.FAIL"

mkdir -p \
    "$OUT" \
    "$PDS_RUN/eval" \
    "$WA_RUN/eval" \
    "$PDS_LOG" \
    "$WA_LOG"

rm -f "$PASS" "$FAIL"

START="$(date +%s)"

echo "$START" > "$OUT/start_epoch.txt"


on_error () {
    rc=$?

    echo
    echo "======================================================================"
    echo "WA PAIRED SEED2 FAILED"
    echo "RETURN_CODE=$rc"
    echo "FAIL_CST=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "FAIL_UTC=$(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "======================================================================"

    touch "$FAIL"
}

trap on_error ERR


echo "======================================================================"
echo "WA PAIRED TRAINING SEED — CONFIRMATION V1"
echo "======================================================================"
echo
echo "NEW_PAIRED_SEED=$SEED"
echo
echo "PRIMARY:"
echo "  Delta_WA(seed2) = M(PDS+WA, seed2) - M(PDS, seed2)"
echo
echo "IMPORTANT:"
echo "  both arms start from same Base"
echo "  same training seed"
echo "  same training recipe"
echo "  same evaluator"
echo "  no data regeneration"
echo "  no corpus filtering"
echo "  no WA rescue"
echo
echo "预计运行时长：1.5–2.5 小时"
echo "ETA confidence=MEDIUM-HIGH"
echo
echo "START_CST=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
echo "START_UTC=$(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
echo "ETA_MIN_CST=$(TZ=Asia/Shanghai date -d "@$((START + 90*60))" '+%Y-%m-%d %H:%M:%S %Z')"
echo "ETA_MAX_CST=$(TZ=Asia/Shanghai date -d "@$((START + 150*60))" '+%Y-%m-%d %H:%M:%S %Z')"


###############################################################################
# STAGE 1 — PREFLIGHT
###############################################################################

echo
echo "======================================================================"
echo "STAGE 1/4 — PREFLIGHT"
echo "======================================================================"

BAD=0

for F in \
    "$TRAINER" \
    "$EVAL" \
    "$SCORE" \
    "$PDS_DATA" \
    "$PDS_WA_DATA" \
    "$SEED1_PDS_RESULT" \
    "$SEED1_WA_RESULT"
do
    if [ ! -f "$F" ]; then
        echo "MISSING=$F"
        BAD=1
    fi
done

if [ ! -d "$STUDENT" ]; then
    echo "MISSING_MODEL=$STUDENT"
    BAD=1
fi

if [ "$BAD" -ne 0 ]; then
    false
fi


PDS_ROWS="$(wc -l < "$PDS_DATA")"
WA_ROWS="$(wc -l < "$PDS_WA_DATA")"

echo "PDS_ROWS=$PDS_ROWS"
echo "PDS_WA_ROWS=$WA_ROWS"

[ "$PDS_ROWS" -eq 68917 ]
[ "$WA_ROWS" -eq 103876 ]


echo
echo "===== FROZEN DATA HASHES ====="

sha256sum \
    "$PDS_DATA" \
    "$PDS_WA_DATA" \
    | tee "$OUT/frozen_data_sha256.txt"


python - \
    "$SEED1_PDS_RESULT" \
    "$SEED1_WA_RESULT" <<'PY'

import json
import sys

pds = json.load(
    open(
        sys.argv[1],
        encoding="utf-8",
    )
)

wa = json.load(
    open(
        sys.argv[2],
        encoding="utf-8",
    )
)

pds_macro = float(
    pds[
        "K1ALL_PDS"
    ][
        "macro_BLEU"
    ]
)

wa_macro = float(
    wa[
        "PE_PDS_WA"
    ][
        "macro_BLEU"
    ]
)

seed1_delta = (
    wa_macro
    -
    pds_macro
)

print(
    "SEED1_PDS_MACRO_BLEU =",
    pds_macro,
)

print(
    "SEED1_PDS_WA_MACRO_BLEU =",
    wa_macro,
)

print(
    "SEED1_DELTA_WA_MACRO_BLEU =",
    seed1_delta,
)

if abs(
    pds_macro
    -
    18.56962769521186
) > 1e-9:
    raise RuntimeError(
        "seed1 PDS frozen anchor mismatch"
    )

if abs(
    seed1_delta
    -
    (-0.01268185553034229)
) > 1e-9:
    raise RuntimeError(
        "seed1 WA contrast mismatch"
    )

print(
    "SEED1_ANCHORS_PASS"
)
PY


echo "PREFLIGHT_PASS"


###############################################################################
# Shared arm
###############################################################################

run_arm () {
    local ARM="$1"
    local CARD="$2"
    local DATA="$3"
    local RUN="$4"
    local LOG="$5"

    echo
    echo "======================================================================"
    echo "ARM_START=$ARM CARD=$CARD"
    echo "======================================================================"

    mkdir -p \
        "$RUN" \
        "$RUN/eval" \
        "$LOG"

    date +%s \
        > "$LOG/start_epoch.txt"

    ###########################################################################
    # TRAIN
    ###########################################################################

    ASCEND_RT_VISIBLE_DEVICES="$CARD" \
    python -u "$TRAINER" \
        --model "$STUDENT" \
        --train "$DATA" \
        --output-dir "$RUN" \
        --lr 2e-5 \
        --epochs 3 \
        --batch-size 4 \
        --grad-accum 4 \
        --max-length 1024 \
        --warmup-ratio 0.03 \
        --weight-decay 0.01 \
        --max-grad-norm 1.0 \
        --seed "$SEED" \
        --num-workers 2 \
        > "$LOG/train.log" \
        2>&1

    [ -d "$RUN/epoch3" ]

    echo "TRAIN_PASS=$ARM"


    ###########################################################################
    # EVAL
    ###########################################################################

    WMT="$DATA_ROOT/pilot_v2_qwen3_06b/wmt24_zh_en998.jsonl"
    FLORES="$DATA_ROOT/pilot_v2_qwen3_06b/flores_zh_en1012.jsonl"
    CHALLENGE="$DATA_ROOT/pilot_v2_qwen3_06b/challenge_zh_en197.jsonl"

    for NAME in \
        wmt24 \
        flores \
        challenge
    do
        case "$NAME" in
            wmt24)
                INPUT="$WMT"
                ;;

            flores)
                INPUT="$FLORES"
                ;;

            challenge)
                INPUT="$CHALLENGE"
                ;;
        esac

        PRED="$RUN/eval/${NAME}.pred.jsonl"
        METRIC="$RUN/eval/${NAME}.metrics.json"

        ASCEND_RT_VISIBLE_DEVICES="$CARD" \
        python -u "$EVAL" \
            --model "$RUN/epoch3" \
            --tokenizer "$RUN/epoch3" \
            --input "$INPUT" \
            --output "$PRED" \
            --method "wa_paired_seed2_${ARM}_${NAME}" \
            --batch-size 16 \
            --max-new-tokens 256 \
            --attn-implementation sdpa \
            > "$LOG/eval_${NAME}.log" \
            2>&1

        python "$SCORE" \
            --input "$PRED" \
            --output "$METRIC"

        echo
        echo "===== $ARM / $NAME ====="
        cat "$METRIC"
        echo
    done


    date +%s \
        > "$LOG/finish_epoch.txt"

    echo "ARM_COMPLETE=$ARM"
}


###############################################################################
# STAGE 2 — PAIRED PARALLEL TRAIN + EVAL
###############################################################################

echo
echo "======================================================================"
echo "STAGE 2/4 — PAIRED TRAIN + EVAL"
echo "======================================================================"

run_arm \
    "PE_PDS" \
    0 \
    "$PDS_DATA" \
    "$PDS_RUN" \
    "$PDS_LOG" \
    > "$PDS_LOG/arm_master.log" \
    2>&1 &

PID_PDS=$!

echo "ARM_LAUNCHED=PE_PDS CARD=0 PID=$PID_PDS"


run_arm \
    "PE_PDS_WA" \
    1 \
    "$PDS_WA_DATA" \
    "$WA_RUN" \
    "$WA_LOG" \
    > "$WA_LOG/arm_master.log" \
    2>&1 &

PID_WA=$!

echo "ARM_LAUNCHED=PE_PDS_WA CARD=1 PID=$PID_WA"


STATUS_PDS=0
STATUS_WA=0

wait "$PID_PDS" || STATUS_PDS=$?
wait "$PID_WA" || STATUS_WA=$?


echo "ARM_FINISHED=PE_PDS STATUS=$STATUS_PDS"
echo "ARM_FINISHED=PE_PDS_WA STATUS=$STATUS_WA"


[ "$STATUS_PDS" -eq 0 ]
[ "$STATUS_WA" -eq 0 ]


###############################################################################
# STAGE 3 — PAIRED SCIENTIFIC SUMMARY
###############################################################################

echo
echo "======================================================================"
echo "STAGE 3/4 — PAIRED SEED SCIENTIFIC SUMMARY"
echo "======================================================================"

python - \
    "$PDS_RUN" \
    "$WA_RUN" \
    "$SEED1_PDS_RESULT" \
    "$SEED1_WA_RESULT" \
    "$FINAL" <<'PY'

import json
import sys
from pathlib import Path


PDS_RUN = Path(sys.argv[1])
WA_RUN = Path(sys.argv[2])
SEED1_PDS_FILE = Path(sys.argv[3])
SEED1_WA_FILE = Path(sys.argv[4])
OUT = Path(sys.argv[5])


SETS = (
    "wmt24",
    "flores",
    "challenge",
)


def load_arm(run):
    result = {}

    for name in SETS:
        x = json.loads(
            (
                run
                /
                "eval"
                /
                f"{name}.metrics.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        result[name] = {
            "BLEU":
                float(
                    x["BLEU"]
                ),

            "chrF":
                float(
                    x["chrF"]
                ),
        }

    result[
        "macro_BLEU"
    ] = sum(
        result[name][
            "BLEU"
        ]
        for name
        in SETS
    ) / 3

    result[
        "macro_chrF"
    ] = sum(
        result[name][
            "chrF"
        ]
        for name
        in SETS
    ) / 3

    return result


pds2 = load_arm(
    PDS_RUN
)

wa2 = load_arm(
    WA_RUN
)


delta2_bleu = {
    name:
        wa2[name]["BLEU"]
        -
        pds2[name]["BLEU"]
    for name
    in SETS
}


delta2_chrf = {
    name:
        wa2[name]["chrF"]
        -
        pds2[name]["chrF"]
    for name
    in SETS
}


delta2_macro_bleu = (
    wa2[
        "macro_BLEU"
    ]
    -
    pds2[
        "macro_BLEU"
    ]
)


delta2_macro_chrf = (
    wa2[
        "macro_chrF"
    ]
    -
    pds2[
        "macro_chrF"
    ]
)


positive_bleu2 = sum(
    x > 0
    for x
    in delta2_bleu.values()
)

negative_bleu2 = sum(
    x < 0
    for x
    in delta2_bleu.values()
)


###############################################################################
# Seed 1
###############################################################################

seed1_pds_raw = json.loads(
    SEED1_PDS_FILE.read_text(
        encoding="utf-8"
    )
)

seed1_wa_raw = json.loads(
    SEED1_WA_FILE.read_text(
        encoding="utf-8"
    )
)

pds1 = seed1_pds_raw[
    "K1ALL_PDS"
]

wa1 = seed1_wa_raw[
    "PE_PDS_WA"
]


delta1 = (
    float(
        wa1[
            "macro_BLEU"
        ]
    )
    -
    float(
        pds1[
            "macro_BLEU"
        ]
    )
)


###############################################################################
# PRE-REGISTERED BRANCH.
#
# This is NOT an automatic paper verdict.
###############################################################################

if abs(
    delta2_macro_bleu
) <= 0.2:

    branch = (
        "CLOSE_WA_NO_ROBUST_BLEU_BENEFIT"
    )

elif (
    delta2_macro_bleu > 0.3
    and
    positive_bleu2 == 3
):

    branch = (
        "STRONG_SEED_INTERACTION__THIRD_PAIRED_SEED_REQUIRED"
    )

elif (
    delta2_macro_bleu < -0.2
    and
    negative_bleu2 >= 2
):

    branch = (
        "CLOSE_WA_NEGATIVE_UNDER_ADAPTATION"
    )

else:

    branch = (
        "MIXED_BOUNDARY_CASE__REVIEW_BEFORE_NEXT_ACTION"
    )


result = {
    "protocol":
        "WA_PAIRED_SEED2_V1",

    "seed2":
        20260901,

    "PE_PDS_seed2":
        pds2,

    "PE_PDS_WA_seed2":
        wa2,

    "delta_WA_seed2": {
        "BLEU":
            delta2_bleu,

        "macro_BLEU":
            delta2_macro_bleu,

        "chrF":
            delta2_chrf,

        "macro_chrF":
            delta2_macro_chrf,

        "positive_BLEU_benchmarks":
            positive_bleu2,

        "negative_BLEU_benchmarks":
            negative_bleu2,
    },

    "seed1_reference": {
        "macro_delta_WA_BLEU":
            delta1,

        "macro_delta_WA_chrF":
            float(
                seed1_wa_raw[
                    "delta_WA"
                ][
                    "macro_chrF"
                ]
            ),
    },

    "cross_seed": {
        "seed1_delta_WA_macro_BLEU":
            delta1,

        "seed2_delta_WA_macro_BLEU":
            delta2_macro_bleu,

        "mean_delta_WA_macro_BLEU_two_seeds":
            (
                delta1
                +
                delta2_macro_bleu
            )
            /
            2,
    },

    "pre_registered_branch":
        branch,

    "important_note":
        (
            "The branch is a preregistered decision aid, "
            "not an automatic final reviewer verdict."
        ),
}


OUT.write_text(
    json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print(
    "=" * 100
)

print(
    "WA PAIRED SEED2 RESULT"
)

print(
    "=" * 100
)

print(
    f"{'SET':12s}"
    f"{'PDS(seed2)':>14s}"
    f"{'PDS+WA(seed2)':>16s}"
    f"{'DELTA_WA':>12s}"
)


for name in SETS:

    print(
        f"{name:12s}"
        f"{pds2[name]['BLEU']:14.4f}"
        f"{wa2[name]['BLEU']:16.4f}"
        f"{delta2_bleu[name]:+12.4f}"
    )


print()

print(
    "SEED1_DELTA_WA_MACRO_BLEU =",
    delta1,
)

print(
    "SEED2_DELTA_WA_MACRO_BLEU =",
    delta2_macro_bleu,
)

print(
    "SEED2_DELTA_WA_MACRO_CHRF =",
    delta2_macro_chrf,
)

print(
    "SEED2_POSITIVE_BLEU_BENCHMARKS =",
    positive_bleu2,
)

print(
    "TWO_SEED_MEAN_DELTA_WA_MACRO_BLEU =",
    (
        delta1
        +
        delta2_macro_bleu
    )
    /
    2,
)

print(
    "PRE_REGISTERED_BRANCH =",
    branch,
)

print(
    "PAIRED_SEED2_SUMMARY_PASS"
)
PY


###############################################################################
# STAGE 4 — FREEZE
###############################################################################

echo
echo "======================================================================"
echo "STAGE 4/4 — FREEZE"
echo "======================================================================"

sha256sum \
    "$PDS_DATA" \
    "$PDS_WA_DATA" \
    "$PDS_RUN/eval/wmt24.metrics.json" \
    "$PDS_RUN/eval/flores.metrics.json" \
    "$PDS_RUN/eval/challenge.metrics.json" \
    "$WA_RUN/eval/wmt24.metrics.json" \
    "$WA_RUN/eval/flores.metrics.json" \
    "$WA_RUN/eval/challenge.metrics.json" \
    "$FINAL" \
    > "$OUT/final_sha256_v1.txt"


touch "$PASS"

rm -f "$FAIL"

END="$(date +%s)"

echo
echo "======================================================================"
echo "WA PAIRED SEED2 V1 PASS"
echo "======================================================================"
echo "TOTAL_ELAPSED_SECONDS=$((END - START))"
echo "FINISH_CST=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
echo "FINISH_UTC=$(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
echo "FINAL=$FINAL"
echo "PASS=$PASS"
echo "======================================================================"
