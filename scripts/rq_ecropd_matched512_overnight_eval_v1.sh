#!/usr/bin/env bash
set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

TRAIN_LOG="/workspace/mtpatcher/logs/rq_ecropd_matched512_ecropd_v1r1.log"

VANILLA="$RUN_ROOT/$EXP/rq_ecropd_matched512_vanilla_fkl_v1"
ECROPD="$RUN_ROOT/$EXP/rq_ecropd_matched512_ecropd_v1r1"

EVAL="$ROOT/scripts/pilot_v2/eval_qwen3_06b_mt_ascend.py"
SCORE="$ROOT/scripts/pilot_v2/score_mt_jsonl.py"

DATA="$DATA_ROOT/pilot_v2_qwen3_06b"

WMT="$DATA/wmt24_zh_en998.jsonl"
FLORES="$DATA/flores_zh_en1012.jsonl"
CHALLENGE="$DATA/challenge_zh_en197.jsonl"

OUT="$RUN_ROOT/$EXP/rq_ecropd_matched512_overnight_eval_v1"

mkdir -p "$OUT"

echo "============================================================"
echo "WAIT FOR EC-ROPD TRAINING"
echo "============================================================"

while pgrep -f \
  '[m]tpatcher_ecropd_matched512_train_v1.py' \
  >/dev/null
do
    echo "$(date '+%F %T') EC-ROPD still running"
    tail -n 6 "$TRAIN_LOG" || true
    sleep 60
done

echo
echo "============================================================"
echo "TRAINING PROCESS FINISHED"
echo "============================================================"

tail -n 120 "$TRAIN_LOG"

grep -q \
  'MTPATCHER_EC_ROPD_MATCHED512_V1_PASS' \
  "$TRAIN_LOG"

test -f "$ECROPD/epoch3/model.safetensors"

echo "EC_ROPD_TRAINING_PASS_CONFIRMED"

echo
echo "============================================================"
echo "START FROZEN GREEDY EVALUATION"
echo "PRIMARY CHECKPOINT = EPOCH3"
echo "EPOCH1/2 = DIAGNOSTIC ONLY"
echo "============================================================"

run_eval () {
    LABEL="$1"
    MODEL="$2"

    test -f "$MODEL/model.safetensors"

    DIR="$OUT/$LABEL"
    mkdir -p "$DIR"

    for SPEC in \
        "wmt24:$WMT" \
        "flores:$FLORES" \
        "challenge:$CHALLENGE"
    do
        NAME="${SPEC%%:*}"
        INPUT="${SPEC#*:}"

        PRED="$DIR/${NAME}.jsonl"
        METRIC="$DIR/${NAME}_metrics.json"

        echo
        echo "========================================================"
        echo "MODEL=$LABEL"
        echo "DATASET=$NAME"
        echo "========================================================"

        python "$EVAL" \
          --model "$MODEL" \
          --input "$INPUT" \
          --output "$PRED" \
          --method "rq_ecropd_${LABEL}_${NAME}" \
          --batch-size 16 \
          --max-new-tokens 256 \
          --attn-implementation sdpa

        python "$SCORE" \
          --input "$PRED" \
          --output "$METRIC"
    done

    echo "${LABEL}_EVAL_PASS"
}

for EPOCH in 1 2 3
do
    run_eval \
      "vanilla_e${EPOCH}" \
      "$VANILLA/epoch${EPOCH}"
done

for EPOCH in 1 2 3
do
    run_eval \
      "ecropd_e${EPOCH}" \
      "$ECROPD/epoch${EPOCH}"
done

echo
echo "============================================================"
echo "BUILD SUMMARY"
echo "============================================================"

python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])

methods = [
    "vanilla_e1",
    "vanilla_e2",
    "vanilla_e3",
    "ecropd_e1",
    "ecropd_e2",
    "ecropd_e3",
]

datasets = [
    "wmt24",
    "flores",
    "challenge",
]

summary = {}

print()
print(
    f"{'method':14s} "
    f"{'WMT':>9s} "
    f"{'FLORES':>9s} "
    f"{'CHALL':>9s} "
    f"{'AVG BLEU':>10s} "
    f"{'AVG chrF':>10s}"
)

print("-" * 70)

for method in methods:
    summary[method] = {}

    bleus = []
    chrfs = []

    vals = []

    for ds in datasets:
        p = root / method / f"{ds}_metrics.json"

        obj = json.loads(
            p.read_text(
                encoding="utf-8"
            )
        )

        bleu = float(obj["BLEU"])
        chrf = float(obj["chrF"])

        summary[method][ds] = {
            "BLEU": bleu,
            "chrF": chrf,
        }

        bleus.append(bleu)
        chrfs.append(chrf)
        vals.append(bleu)

    avg_bleu = sum(bleus) / len(bleus)
    avg_chrf = sum(chrfs) / len(chrfs)

    summary[method]["avg"] = {
        "BLEU": avg_bleu,
        "chrF": avg_chrf,
    }

    print(
        f"{method:14s} "
        f"{vals[0]:9.4f} "
        f"{vals[1]:9.4f} "
        f"{vals[2]:9.4f} "
        f"{avg_bleu:10.4f} "
        f"{avg_chrf:10.4f}"
    )

v = summary["vanilla_e3"]["avg"]
e = summary["ecropd_e3"]["avg"]

summary["primary_epoch3_comparison"] = {
    "ECROPD_minus_Vanilla": {
        "BLEU":
            e["BLEU"] - v["BLEU"],
        "chrF":
            e["chrF"] - v["chrF"],
    }
}

print()
print("===== PRIMARY EPOCH3 COMPARISON =====")
print(
    "ECROPD - VANILLA AVG BLEU = "
    f"{e['BLEU'] - v['BLEU']:+.6f}"
)
print(
    "ECROPD - VANILLA AVG chrF = "
    f"{e['chrF'] - v['chrF']:+.6f}"
)

out = root / "summary.json"

out.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print("SUMMARY =", out)
print("RQ_ECROPD_MATCHED512_OVERNIGHT_EVAL_PASS")
PY
