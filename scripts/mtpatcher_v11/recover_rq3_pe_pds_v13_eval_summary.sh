#!/usr/bin/env bash
set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

EXP_RUN="$RUN_ROOT/$EXP"
EXP_LOG="$LOG_ROOT/$EXP"

RUN_NAME="rq3_pe_pds_v13_paperbudget_b4ga4"

MODEL_E3="$EXP_RUN/$RUN_NAME/epoch3"
TOKENIZER="$MODEL_ROOT/Qwen3-0.6B"

EVAL_ROOT="$EXP_RUN/${RUN_NAME}_eval_epoch3"
PE_EVAL_ROOT="$EXP_RUN/pe_k1_eval_epoch3"

LOG_DIR="$EXP_LOG/${RUN_NAME}_eval_recovery"
SUMMARY="$EVAL_ROOT/rq3_pe_pds_v13_summary.json"

EVAL="$ROOT/scripts/pilot_v2/eval_qwen3_06b_mt_ascend.py"
SCORE="$ROOT/scripts/pilot_v2/score_mt_jsonl.py"

WMT="$DATA_ROOT/pilot_v2_qwen3_06b/wmt24_zh_en998.jsonl"
FLORES="$DATA_ROOT/pilot_v2_qwen3_06b/flores_zh_en1012.jsonl"
CHALLENGE="$DATA_ROOT/pilot_v2_qwen3_06b/challenge_zh_en197.jsonl"

mkdir -p "$EVAL_ROOT" "$LOG_DIR"

echo "======================================================================"
echo "RQ3 PE+PDS V13 — EVAL-ONLY RECOVERY"
echo "======================================================================"

if [ ! -f "$MODEL_E3/model.safetensors" ]; then
    echo "MISSING_EPOCH3_MODEL=$MODEL_E3/model.safetensors"
    false
fi

if [ ! -f "$EXP_RUN/$RUN_NAME/training_manifest.json" ]; then
    echo "MISSING_TRAINING_MANIFEST"
    false
fi

echo "EPOCH3_MODEL_FOUND"
echo "TRAINING_MANIFEST_FOUND"

python - <<'PY'
import json
import os
from pathlib import Path

exp = "mtpatcher_v3_full6565_20260823"
p = (
    Path(os.environ["RUN_ROOT"])
    / exp
    / "rq3_pe_pds_v13_paperbudget_b4ga4"
    / "training_manifest.json"
)

m = json.loads(p.read_text(encoding="utf-8"))

print("TRAINING_COMPLETE =", m.get("training_complete"))
print("ROWS =", m.get("rows"))
print("EPOCHS =", m.get("epochs"))

for x in m.get("epoch_metrics", []):
    print(
        "EPOCH_METRIC",
        x.get("epoch"),
        "LOSS=",
        x.get("token_mean_response_loss"),
        "UPDATES=",
        x.get("optimizer_updates_total"),
    )

if not m.get("training_complete"):
    raise RuntimeError("Training manifest is not complete")

if m.get("rows") != 18610:
    raise RuntimeError(
        f"Unexpected training rows: {m.get('rows')}"
    )

print("RQ3_V13_FROZEN_TRAINING_AUDIT_PASS")
PY

declare -A DATA
declare -A EXPECTED
declare -A CARD
declare -A ATTEMPTS

DATA[wmt24]="$WMT"
DATA[flores]="$FLORES"
DATA[challenge]="$CHALLENGE"

EXPECTED[wmt24]=998
EXPECTED[flores]=1012
EXPECTED[challenge]=197

CARD[wmt24]=0
CARD[flores]=1
CARD[challenge]=2

ATTEMPTS[wmt24]=0
ATTEMPTS[flores]=0
ATTEMPTS[challenge]=0

metric_valid () {
    local metric="$1"
    local expected="$2"

    python - "$metric" "$expected" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
expected = int(sys.argv[2])

if not p.exists():
    raise SystemExit(1)

try:
    x = json.loads(p.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)

if int(x.get("rows", -1)) != expected:
    raise SystemExit(1)

for key in ("BLEU", "chrF"):
    if key not in x:
        raise SystemExit(1)

raise SystemExit(0)
PY
}

job_active () {
    local pred="$1"

    ps -eo args= | \
        grep -F -- "$pred" | \
        grep -E \
        'eval_qwen3_06b_mt_ascend.py|score_mt_jsonl.py' | \
        grep -v grep \
        >/dev/null 2>&1
}

score_existing_prediction () {
    local name="$1"

    local dest="$EVAL_ROOT/$name"
    local pred="$dest/predictions.jsonl"
    local metric="$dest/metrics.json"
    local log="$LOG_DIR/${name}.log"

    echo "SCORING_EXISTING_PREDICTION name=$name"

    python "$SCORE" \
        --input "$pred" \
        --output "$metric" \
        >> "$log" 2>&1
}

launch_eval () {
    local name="$1"

    local data="${DATA[$name]}"
    local card="${CARD[$name]}"

    local dest="$EVAL_ROOT/$name"
    local pred="$dest/predictions.jsonl"
    local metric="$dest/metrics.json"
    local log="$LOG_DIR/${name}.log"

    mkdir -p "$dest"

    ATTEMPTS[$name]=$((ATTEMPTS[$name] + 1))

    echo \
        "LAUNCH_EVAL name=$name card=$card attempt=${ATTEMPTS[$name]}"

    (
        export ASCEND_RT_VISIBLE_DEVICES="$card"

        python "$EVAL" \
            --model "$MODEL_E3" \
            --tokenizer "$TOKENIZER" \
            --input "$data" \
            --output "$pred" \
            --method "rq3_pe_pds_v13_epoch3_${name}" \
            --batch-size 16 \
            --max-new-tokens 256 \
            --attn-implementation sdpa

        python "$SCORE" \
            --input "$pred" \
            --output "$metric"

        echo "EVAL_CHAIN_COMPLETE name=$name"
    ) >> "$log" 2>&1 &

    echo "LAUNCHED_PID name=$name pid=$!"
}

echo
echo "======================================================================"
echo "RECOVER / COMPLETE THREE FIXED EVALUATIONS"
echo "======================================================================"

while true; do
    COMPLETE=0

    for NAME in wmt24 flores challenge
    do
        DEST="$EVAL_ROOT/$NAME"
        PRED="$DEST/predictions.jsonl"
        METRIC="$DEST/metrics.json"
        EXPECT="${EXPECTED[$NAME]}"

        mkdir -p "$DEST"

        if metric_valid "$METRIC" "$EXPECT"; then
            echo "METRIC_READY name=$NAME"
            COMPLETE=$((COMPLETE + 1))
            continue
        fi

        if job_active "$PRED"; then
            ROWS=0

            if [ -f "$PRED" ]; then
                ROWS=$(wc -l < "$PRED")
            fi

            echo \
                "EVAL_STILL_ACTIVE name=$NAME rows=$ROWS/$EXPECT"

            continue
        fi

        ROWS=0

        if [ -f "$PRED" ]; then
            ROWS=$(wc -l < "$PRED")
        fi

        if [ "$ROWS" -eq "$EXPECT" ]; then
            score_existing_prediction "$NAME"

            if metric_valid "$METRIC" "$EXPECT"; then
                echo "METRIC_RECOVERED_FROM_COMPLETE_PRED name=$NAME"
                COMPLETE=$((COMPLETE + 1))
                continue
            fi
        fi

        if [ "${ATTEMPTS[$NAME]}" -ge 2 ]; then
            echo \
                "EVAL_RECOVERY_EXHAUSTED name=$NAME rows=$ROWS/$EXPECT"
            false
        fi

        if [ -f "$PRED" ] && [ "$ROWS" -ne 0 ]; then
            STAMP=$(date +%Y%m%d_%H%M%S)

            mv \
                "$PRED" \
                "${PRED}.partial_${ROWS}_${STAMP}"

            echo \
                "PARTIAL_PRED_PRESERVED name=$NAME rows=$ROWS"
        fi

        if [ -f "$METRIC" ]; then
            STAMP=$(date +%Y%m%d_%H%M%S)

            mv \
                "$METRIC" \
                "${METRIC}.invalid_${STAMP}"
        fi

        launch_eval "$NAME"
    done

    if [ "$COMPLETE" -eq 3 ]; then
        break
    fi

    sleep 10
done

echo
echo "RQ3_PE_PDS_V13_EVAL_PASS"

echo
echo "======================================================================"
echo "FINAL METRICS"
echo "======================================================================"

for NAME in wmt24 flores challenge
do
    echo "----- $NAME -----"
    cat "$EVAL_ROOT/$NAME/metrics.json"
    echo
done

echo
echo "======================================================================"
echo "BUILD RQ3 PDS SUMMARY"
echo "======================================================================"

export EVAL_ROOT PE_EVAL_ROOT SUMMARY

python - <<'PY'
import json
import os
from pathlib import Path


eval_root = Path(os.environ["EVAL_ROOT"])
pe_root = Path(os.environ["PE_EVAL_ROOT"])
summary_path = Path(os.environ["SUMMARY"])

names = ("wmt24", "flores", "challenge")


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


def find_pe_metric(name):
    candidates = [
        pe_root / name / "metrics.json",
        pe_root / f"{name}_metrics.json",
    ]

    for p in candidates:
        if p.exists():
            return p

    hits = []

    if pe_root.exists():
        for p in pe_root.rglob("*.json"):
            if name.lower() in str(p).lower():
                try:
                    x = json.loads(
                        p.read_text(encoding="utf-8")
                    )
                except Exception:
                    continue

                if (
                    "BLEU" in x
                    and "chrF" in x
                ):
                    hits.append(p)

    if len(hits) == 1:
        return hits[0]

    raise RuntimeError(
        f"Could not uniquely resolve PE metric for {name}: "
        f"{hits}"
    )


pe = {}
pds = {}
pe_paths = {}


for name in names:
    p = find_pe_metric(name)

    pe_paths[name] = str(p)

    pe[name] = json.loads(
        p.read_text(encoding="utf-8")
    )

    pds_path = (
        eval_root
        / name
        / "metrics.json"
    )

    pds[name] = json.loads(
        pds_path.read_text(encoding="utf-8")
    )


def mean_delta(a, b, metric):
    return sum(
        float(a[name][metric])
        - float(b[name][metric])
        for name in names
    ) / len(names)


result = {
    "protocol":
        "rq3_fixed_epoch3",

    "system":
        "PE+PDS-v13-paperbudget",

    "training_rows":
        18610,

    "pe_metric_paths":
        pe_paths,

    "base":
        base,

    "PE3732": {
        name: {
            "BLEU": float(pe[name]["BLEU"]),
            "chrF": float(pe[name]["chrF"]),
        }
        for name in names
    },

    "PE_plus_PDS_v13": {
        name: {
            "BLEU": float(pds[name]["BLEU"]),
            "chrF": float(pds[name]["chrF"]),
        }
        for name in names
    },
}


result["avg_delta_bleu_vs_base"] = (
    mean_delta(
        pds,
        base,
        "BLEU",
    )
)

result["avg_delta_chrf_vs_base"] = (
    mean_delta(
        pds,
        base,
        "chrF",
    )
)

result["avg_delta_bleu_vs_PE"] = (
    mean_delta(
        pds,
        pe,
        "BLEU",
    )
)

result["avg_delta_chrf_vs_PE"] = (
    mean_delta(
        pds,
        pe,
        "chrF",
    )
)


summary_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

summary_path.write_text(
    json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


print(
    f"{'DATASET':12s}"
    f"{'BASE':>11s}"
    f"{'PE':>11s}"
    f"{'PE+PDS':>11s}"
    f"{'PDS-PE':>11s}"
)

print("-" * 56)

for name in names:
    b = base[name]["BLEU"]
    p = float(pe[name]["BLEU"])
    q = float(pds[name]["BLEU"])

    print(
        f"{name:12s}"
        f"{b:11.4f}"
        f"{p:11.4f}"
        f"{q:11.4f}"
        f"{q-p:+11.4f}"
    )


print()
print(
    "AVG_DELTA_BLEU_VS_BASE =",
    f"{result['avg_delta_bleu_vs_base']:+.6f}",
)

print(
    "AVG_DELTA_BLEU_VS_PE =",
    f"{result['avg_delta_bleu_vs_PE']:+.6f}",
)

print(
    "AVG_DELTA_CHRF_VS_BASE =",
    f"{result['avg_delta_chrf_vs_base']:+.6f}",
)

print(
    "AVG_DELTA_CHRF_VS_PE =",
    f"{result['avg_delta_chrf_vs_PE']:+.6f}",
)

print("SUMMARY =", summary_path)

print("RQ3_PE_PDS_V13_SUMMARY_PASS")
PY

echo
echo "RQ3_PE_PDS_V13_EVAL_RECOVERY_ALL_PASS"
