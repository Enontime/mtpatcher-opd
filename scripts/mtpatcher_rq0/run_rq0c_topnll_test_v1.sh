#!/usr/bin/env bash
set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

BASE="$MODEL_ROOT/Qwen3-0.6B"

SEQKD50="$DATA_ROOT/$EXP/rq0_seqkd_scaling50k_v1/seqkd_newscrawl50000_qwen3_8b_v1.jsonl"

SCORE="$DATA_ROOT/$EXP/rq0_newscrawl50k_student_nll_v1.jsonl"

TOPDIR="$DATA_ROOT/$EXP/rq0_topnll_v1"

RUN="$RUN_ROOT/$EXP/rq0_topnll_v1"
LOGDIR="$LOG_ROOT/$EXP/rq0_topnll_v1"

SCORER="$ROOT/scripts/mtpatcher_rq0/score_student_nll_v1.py"
BUILDER="$ROOT/scripts/mtpatcher_rq0/build_topnll_sets_v1.py"

TRAINER="$ROOT/scripts/pilot_v2/train_qwen3_06b_full_sft_ascend.py"
EVAL="$ROOT/scripts/pilot_v2/eval_qwen3_06b_mt_ascend.py"
METRIC="$ROOT/scripts/pilot_v2/score_mt_jsonl.py"

WMT="$DATA_ROOT/pilot_v2_qwen3_06b/wmt24_zh_en998.jsonl"
FLORES="$DATA_ROOT/pilot_v2_qwen3_06b/flores_zh_en1012.jsonl"
CHALLENGE="$DATA_ROOT/pilot_v2_qwen3_06b/challenge_zh_en197.jsonl"

mkdir -p \
    "$TOPDIR" \
    "$RUN" \
    "$LOGDIR"


echo "============================================================"
echo "RQ0-C — STUDENT NLL SELECTION"
echo "============================================================"


###############################################################################
# STAGE 1 — SCORE
###############################################################################

if [ ! -f "$SCORE" ]; then

    python -u "$SCORER" \
        --input "$SEQKD50" \
        --output "$SCORE" \
        --model "$BASE" \
        --device 0 \
        --batch-size 16 \
        --max-length 1024 \
        > "$LOGDIR/student_nll_50k.log" \
        2>&1

else
    echo "NLL_SCORE_ALREADY_EXISTS"
fi

test "$(wc -l < "$SCORE")" -eq 50000

echo "RQ0_C_50K_NLL_PASS"


###############################################################################
# STAGE 2 — BUILD TOP SUBSETS
###############################################################################

python "$BUILDER" \
    --teacher-data "$SEQKD50" \
    --scores "$SCORE" \
    --out-dir "$TOPDIR"

echo "RQ0_C_TOPSETS_PASS"


###############################################################################
# STAGE 3 — TRAIN TOP 6565 + TOP 10000 IN PARALLEL
###############################################################################

PIDS=()

for SPEC in \
    "6565:0" \
    "10000:1"
do

    N="${SPEC%%:*}"
    DEV="${SPEC#*:}"

    DATA="$TOPDIR/seqkd_newscrawl_topnll${N}_v1.jsonl"
    OUT="$RUN/topnll${N}_b4ga4"
    LOG="$LOGDIR/train_topnll${N}.log"

    (
        export ASCEND_RT_VISIBLE_DEVICES="$DEV"

        python -u "$TRAINER" \
            --model "$BASE" \
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

    PIDS+=("$!")

    echo \
        "TRAIN_LAUNCHED n=$N device=$DEV pid=$!"
done

FAIL=0

for P in "${PIDS[@]}"; do
    if ! wait "$P"; then
        FAIL=1
    fi
done

if [ "$FAIL" -ne 0 ]; then
    echo "RQ0_C_TRAIN_FAILURE"
    false
fi

echo "RQ0_C_TRAIN_PASS"


###############################################################################
# STAGE 4 — SIX EVALS
###############################################################################

PIDS=()
DEV=0

for N in 6565 10000; do

    MODEL="$RUN/topnll${N}_b4ga4/epoch3"

    for SPEC in \
        "wmt24:$WMT" \
        "flores:$FLORES" \
        "challenge:$CHALLENGE"
    do

        NAME="${SPEC%%:*}"
        DATA="${SPEC#*:}"

        OUTDIR="$RUN/eval_${N}"
        mkdir -p "$OUTDIR"

        PRED="$OUTDIR/${NAME}.jsonl"
        MET="$OUTDIR/${NAME}_metrics.json"

        (
            export ASCEND_RT_VISIBLE_DEVICES="$DEV"

            python -u "$EVAL" \
                --model "$MODEL" \
                --tokenizer "$MODEL" \
                --input "$DATA" \
                --output "$PRED" \
                --method "rq0_topnll_${N}_${NAME}" \
                --batch-size 16 \
                --max-new-tokens 256 \
                --attn-implementation sdpa

            python "$METRIC" \
                --input "$PRED" \
                --output "$MET"

        ) > "$LOGDIR/eval_${N}_${NAME}.log" \
        2>&1 &

        PIDS+=("$!")

        DEV=$((DEV + 1))
    done
done

FAIL=0

for P in "${PIDS[@]}"; do
    if ! wait "$P"; then
        FAIL=1
    fi
done

if [ "$FAIL" -ne 0 ]; then
    echo "RQ0_C_EVAL_FAILURE"
    false
fi

echo "RQ0_C_EVAL_PASS"


###############################################################################
# STAGE 5 — SUMMARY
###############################################################################

export RUN

python - <<'PY'
import json
import os
from pathlib import Path

root = Path(
    os.environ["RUN"]
)

base = {
    "wmt24": 15.536214,
    "flores": 19.971480,
    "challenge": 16.537871,
}

random_baselines = {
    6565: 0.8727155348410675,
    10000: 1.0190617541910398,
    20000: 1.449641655422992,
    50000: 2.250423706625344,
}

curated6565 = 1.465

results = {}

for n in [6565, 10000]:

    vals = {}

    for name in [
        "wmt24",
        "flores",
        "challenge",
    ]:

        p = (
            root
            / f"eval_{n}"
            / f"{name}_metrics.json"
        )

        x = json.loads(
            p.read_text(
                encoding="utf-8"
            )
        )

        vals[name] = float(
            x["BLEU"]
        )

    delta = sum(
        vals[k] - base[k]
        for k in base
    ) / 3

    results[n] = {
        "metrics":
            vals,

        "avg_delta_bleu":
            delta,

        "advantage_over_random_same_n":
            delta
            - random_baselines[n],
    }


print()
print(
    f"{'METHOD':26s}"
    f"{'AVG ΔBLEU':>14s}"
)

print("-" * 40)

print(
    f"{'News Random 6565':26s}"
    f"{random_baselines[6565]:+14.4f}"
)

print(
    f"{'News TopNLL 6565':26s}"
    f"{results[6565]['avg_delta_bleu']:+14.4f}"
)

print(
    f"{'Curated 6565':26s}"
    f"{curated6565:+14.4f}"
)

print(
    f"{'News Random 10000':26s}"
    f"{random_baselines[10000]:+14.4f}"
)

print(
    f"{'News TopNLL 10000':26s}"
    f"{results[10000]['avg_delta_bleu']:+14.4f}"
)

print(
    f"{'News Random 20000':26s}"
    f"{random_baselines[20000]:+14.4f}"
)

print(
    f"{'News Random 50000':26s}"
    f"{random_baselines[50000]:+14.4f}"
)

print()

print(
    "TOP6565_SELECTION_ADVANTAGE = "
    f"{results[6565]['advantage_over_random_same_n']:+.6f}"
)

print(
    "TOP10000_SELECTION_ADVANTAGE = "
    f"{results[10000]['advantage_over_random_same_n']:+.6f}"
)

print(
    "TOP6565_GAP_TO_CURATED6565 = "
    f"{results[6565]['avg_delta_bleu'] - curated6565:+.6f}"
)

summary = {
    "random":
        random_baselines,

    "curated6565":
        curated6565,

    "topnll":
        results,
}

out = (
    root
    / "rq0c_topnll_summary.json"
)

out.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print(
    "SUMMARY=",
    out,
)

print(
    "RQ0_C_TOPNLL_ALL_PASS"
)
PY

