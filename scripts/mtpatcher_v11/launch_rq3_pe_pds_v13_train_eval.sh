#!/usr/bin/env bash

source /workspace/mtpatcher/project_env.sh

SCRIPT="$ROOT/scripts/mtpatcher_v11/run_rq3_pe_pds_v13_train_eval.sh"

cat > "$SCRIPT" <<'RUNNER'
#!/usr/bin/env bash
set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

EXP_DATA="$DATA_ROOT/$EXP"
EXP_RUN="$RUN_ROOT/$EXP"
EXP_LOG="$LOG_ROOT/$EXP"

MODEL="$MODEL_ROOT/Qwen3-0.6B"

TRAIN="$EXP_DATA/rq3_pe_plus_pds_v13_paperbudget.jsonl"

EXPECTED_ROWS=18610
EXPECTED_SHA="e03f13ef18d8a5c0135524e8c310b160e8061f92c7c2ffa98e11f0e36968b46b"

TRAINER="$ROOT/scripts/pilot_v2/train_qwen3_06b_full_sft_ascend.py"

EVAL="$ROOT/scripts/pilot_v2/eval_qwen3_06b_mt_ascend.py"
SCORE="$ROOT/scripts/pilot_v2/score_mt_jsonl.py"

RUN_NAME="rq3_pe_pds_v13_paperbudget_b4ga4"

OUT="$EXP_RUN/$RUN_NAME"
EVAL_ROOT="$EXP_RUN/${RUN_NAME}_eval_epoch3"

RUN_LOG_ROOT="$EXP_LOG/$RUN_NAME"

WMT="$DATA_ROOT/pilot_v2_qwen3_06b/wmt24_zh_en998.jsonl"
FLORES="$DATA_ROOT/pilot_v2_qwen3_06b/flores_zh_en1012.jsonl"
CHALLENGE="$DATA_ROOT/pilot_v2_qwen3_06b/challenge_zh_en197.jsonl"

PE_EVAL_ROOT="$EXP_RUN/pe_k1_eval_epoch3"

SUMMARY="$EVAL_ROOT/rq3_pe_pds_v13_summary.json"

mkdir -p \
    "$EXP_RUN" \
    "$EXP_LOG" \
    "$RUN_LOG_ROOT" \
    "$EVAL_ROOT"

echo "======================================================================"
echo "RQ3 PE+PDS V13 — FIXED PROTOCOL"
echo "======================================================================"

echo "TRAIN=$TRAIN"
echo "MODEL=$MODEL"
echo "OUT=$OUT"

###############################################################################
# 1. STATIC INPUT AUDIT
###############################################################################

echo
echo "======================================================================"
echo "STAGE 1/6 — STATIC INPUT AUDIT"
echo "======================================================================"

for F in \
    "$TRAIN" \
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

ROWS=$(wc -l < "$TRAIN")

echo "TRAIN_ROWS=$ROWS"

if [ "$ROWS" -ne "$EXPECTED_ROWS" ]; then
    echo "ROW_CARDINALITY_MISMATCH"
    false
fi

ACTUAL_SHA=$(sha256sum "$TRAIN" | awk '{print $1}')

echo "TRAIN_SHA256=$ACTUAL_SHA"

if [ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]; then
    echo "TRAIN_SHA256_MISMATCH"
    false
fi

python - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

exp = "mtpatcher_v3_full6565_20260823"

path = (
    Path(os.environ["DATA_ROOT"])
    / exp
    / "rq3_pe_plus_pds_v13_paperbudget.jsonl"
)

rows = []

with path.open(encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        if not line.strip():
            continue

        row = json.loads(line)

        if not isinstance(row.get("messages"), list):
            raise RuntimeError(
                f"line={line_no} invalid messages"
            )

        if not row["messages"]:
            raise RuntimeError(
                f"line={line_no} empty messages"
            )

        target = row.get("target_translation")

        if not isinstance(target, str) or not target.strip():
            raise RuntimeError(
                f"line={line_no} invalid target"
            )

        if "reference" in row:
            raise RuntimeError(
                f"line={line_no} reference leakage"
            )

        rows.append(row)

counts = Counter(
    row.get("rq3_data_component")
    for row in rows
)

print("TOTAL =", len(rows))
print("COMPONENT_COUNTS =", dict(counts))

if len(rows) != 18610:
    raise RuntimeError(
        f"Expected 18610 rows, got {len(rows)}"
    )

if counts["PE"] != 3732:
    raise RuntimeError(
        f"Expected PE=3732, got {counts['PE']}"
    )

if counts["PDS"] != 14878:
    raise RuntimeError(
        f"Expected PDS=14878, got {counts['PDS']}"
    )

print("RQ3_V13_SCHEMA_PASS")
PY

echo "RQ3_V13_STATIC_INPUT_PASS"

###############################################################################
# 2. TOKEN LENGTH PREFLIGHT
###############################################################################

echo
echo "======================================================================"
echo "STAGE 2/6 — TOKEN LENGTH PREFLIGHT"
echo "======================================================================"

python - <<'PY'
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

exp = "mtpatcher_v3_full6565_20260823"

train = (
    Path(os.environ["DATA_ROOT"])
    / exp
    / "rq3_pe_plus_pds_v13_paperbudget.jsonl"
)

model = (
    Path(os.environ["MODEL_ROOT"])
    / "Qwen3-0.6B"
)

tok = AutoTokenizer.from_pretrained(
    model,
    local_files_only=True,
)

lengths = []
response_lengths = []
overlong = []

with train.open(encoding="utf-8") as f:
    for i, line in enumerate(f):
        if not line.strip():
            continue

        row = json.loads(line)

        prompt = tok.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        prompt_ids = tok(
            prompt,
            add_special_tokens=False,
        )["input_ids"]

        target_ids = tok(
            row["target_translation"].strip(),
            add_special_tokens=False,
        )["input_ids"]

        n = len(prompt_ids) + len(target_ids) + 1

        lengths.append(n)
        response_lengths.append(
            len(target_ids) + 1
        )

        if n > 1024:
            overlong.append(
                (
                    i,
                    row.get("index"),
                    n,
                )
            )

s = sorted(lengths)

def pct(p):
    k = min(
        len(s) - 1,
        int(round((len(s) - 1) * p)),
    )
    return s[k]

print("TOKEN_ROWS =", len(lengths))
print("SEQ_LEN_MIN =", min(lengths))
print("SEQ_LEN_MEAN =", sum(lengths) / len(lengths))
print("SEQ_LEN_P50 =", pct(0.50))
print("SEQ_LEN_P95 =", pct(0.95))
print("SEQ_LEN_P99 =", pct(0.99))
print("SEQ_LEN_MAX =", max(lengths))

print(
    "RESPONSE_LEN_MEAN =",
    sum(response_lengths) / len(response_lengths),
)

print("OVERLONG_1024 =", len(overlong))

if overlong:
    print(
        "OVERLONG_FIRST20 =",
        overlong[:20],
    )

    raise RuntimeError(
        "Training data exceeds frozen max_length=1024"
    )

print("RQ3_V13_TOKEN_LENGTH_PASS")
PY

###############################################################################
# 3. TRAIN PE + PDS
###############################################################################

echo
echo "======================================================================"
echo "STAGE 3/6 — QWEN3-0.6B PE+PDS FULL SFT"
echo "======================================================================"

if [ -f "$OUT/epoch3/model.safetensors" ] \
   && [ -f "$OUT/training_manifest.json" ]; then

    echo "RQ3_PE_PDS_V13_TRAIN_ALREADY_COMPLETE"

else

    if [ -d "$OUT" ] \
       && [ "$(find "$OUT" -mindepth 1 -maxdepth 1 | wc -l)" -gt 0 ]; then

        STAMP=$(date +%Y%m%d_%H%M%S)

        mv \
            "$OUT" \
            "${OUT}.partial_${STAMP}"

        echo "PARTIAL_RUN_PRESERVED=${OUT}.partial_${STAMP}"
    fi

    mkdir -p "$OUT"

    export ASCEND_RT_VISIBLE_DEVICES=0

    python "$TRAINER" \
        --model "$MODEL" \
        --train "$TRAIN" \
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
fi

if [ ! -f "$OUT/epoch3/model.safetensors" ]; then
    echo "EPOCH3_MODEL_MISSING"
    false
fi

if [ ! -f "$OUT/training_manifest.json" ]; then
    echo "TRAINING_MANIFEST_MISSING"
    false
fi

echo "RQ3_PE_PDS_V13_TRAIN_PASS"

###############################################################################
# 4. FIXED EPOCH3 EVALUATION
###############################################################################

echo
echo "======================================================================"
echo "STAGE 4/6 — FIXED EPOCH3 EVALUATION"
echo "======================================================================"

MODEL_E3="$OUT/epoch3"
TOKENIZER="$MODEL_ROOT/Qwen3-0.6B"

run_eval () {
    NAME="$1"
    CARD="$2"
    DATA="$3"

    DEST="$EVAL_ROOT/$NAME"
    LOG="$RUN_LOG_ROOT/eval_${NAME}.log"

    mkdir -p "$DEST"

    if [ -f "$DEST/metrics.json" ]; then
        echo "EVAL_ALREADY_COMPLETE name=$NAME"
        return
    fi

    (
        export ASCEND_RT_VISIBLE_DEVICES="$CARD"

        python "$EVAL" \
            --model "$MODEL_E3" \
            --tokenizer "$TOKENIZER" \
            --input "$DATA" \
            --output "$DEST/predictions.jsonl" \
            --method "rq3_pe_pds_v13_epoch3_${NAME}" \
            --batch-size 16 \
            --max-new-tokens 256 \
            --attn-implementation sdpa

        python "$SCORE" \
            --input "$DEST/predictions.jsonl" \
            --output "$DEST/metrics.json"
    ) > "$LOG" 2>&1 &

    echo "$!"
}

P1=$(run_eval "wmt24" 0 "$WMT")
P2=$(run_eval "flores" 1 "$FLORES")
P3=$(run_eval "challenge" 2 "$CHALLENGE")

FAIL=0

for PID in "$P1" "$P2" "$P3"
do
    if [[ "$PID" =~ ^[0-9]+$ ]]; then
        if ! wait "$PID"; then
            FAIL=1
        fi
    fi
done

if [ "$FAIL" -ne 0 ]; then
    echo "RQ3_PE_PDS_V13_EVAL_WORKER_FAILURE"
    false
fi

for NAME in wmt24 flores challenge
do
    if [ ! -f "$EVAL_ROOT/$NAME/metrics.json" ]; then
        echo "MISSING_METRICS=$NAME"
        false
    fi
done

echo "RQ3_PE_PDS_V13_EVAL_PASS"

###############################################################################
# 5. SUMMARY
###############################################################################

echo
echo "======================================================================"
echo "STAGE 5/6 — RQ3 CAUSAL SUMMARY"
echo "======================================================================"

export EVAL_ROOT PE_EVAL_ROOT SUMMARY

python - <<'PY'
import json
import os
from pathlib import Path

eval_root = Path(os.environ["EVAL_ROOT"])
pe_root = Path(os.environ["PE_EVAL_ROOT"])
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

pe = {}
pds = {}

for name in names:
    pe_path = (
        pe_root
        / name
        / "metrics.json"
    )

    if not pe_path.exists():
        raise RuntimeError(
            f"Missing frozen PE metric: {pe_path}"
        )

    pe[name] = json.loads(
        pe_path.read_text(
            encoding="utf-8"
        )
    )

    pds[name] = json.loads(
        (
            eval_root
            / name
            / "metrics.json"
        ).read_text(
            encoding="utf-8"
        )
    )


def mean_delta(a, b, metric):
    return sum(
        a[name][metric] - b[name][metric]
        for name in names
    ) / len(names)


result = {
    "protocol":
        "fixed_epoch3",

    "system":
        "PE+PDS-v13-paperbudget",

    "training_rows":
        18610,

    "training": {
        "student":
            "Qwen3-0.6B",

        "epochs":
            3,

        "lr":
            2e-5,

        "batch_size":
            4,

        "grad_accum":
            4,

        "effective_batch_size":
            16,

        "max_length":
            1024,

        "warmup_ratio":
            0.03,

        "weight_decay":
            0.01,

        "seed":
            20260820,
    },

    "base":
        base,

    "PE3732":
        {
            name: {
                "BLEU":
                    pe[name]["BLEU"],

                "chrF":
                    pe[name]["chrF"],
            }
            for name in names
        },

    "PE_plus_PDS_v13":
        {
            name: {
                "BLEU":
                    pds[name]["BLEU"],

                "chrF":
                    pds[name]["chrF"],
            }
            for name in names
        },
}


result[
    "avg_delta_bleu_vs_base"
] = mean_delta(
    pds,
    base,
    "BLEU",
)


result[
    "avg_delta_chrf_vs_base"
] = mean_delta(
    pds,
    base,
    "chrF",
)


result[
    "avg_delta_bleu_vs_PE"
] = mean_delta(
    pds,
    pe,
    "BLEU",
)


result[
    "avg_delta_chrf_vs_PE"
] = mean_delta(
    pds,
    pe,
    "chrF",
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


print()
print(
    f"{'DATASET':12s} "
    f"{'BASE':>10s} "
    f"{'PE':>10s} "
    f"{'PE+PDS':>10s} "
    f"{'PDS-PE':>10s}"
)

print("-" * 58)

for name in names:
    print(
        f"{name:12s} "
        f"{base[name]['BLEU']:10.4f} "
        f"{pe[name]['BLEU']:10.4f} "
        f"{pds[name]['BLEU']:10.4f} "
        f"{pds[name]['BLEU']-pe[name]['BLEU']:+10.4f}"
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

print(
    "SUMMARY =",
    summary_path,
)

print("RQ3_PE_PDS_V13_SUMMARY_PASS")
PY

###############################################################################
# 6. FINAL
###############################################################################

echo
echo "======================================================================"
echo "STAGE 6/6 — FINAL STATUS"
echo "======================================================================"

cat "$SUMMARY"

echo
echo "RQ3_PE_PDS_V13_ALL_PASS"
RUNNER

chmod +x "$SCRIPT"

MASTER_LOG="$LOG_ROOT/mtpatcher_v3_full6565_20260823/rq3_pe_pds_v13_train_eval_master.log"

if pgrep -af "run_rq3_pe_pds_v13_train_eval.sh" >/dev/null 2>&1; then
    echo "RQ3_PE_PDS_V13_ALREADY_RUNNING"
    pgrep -af "run_rq3_pe_pds_v13_train_eval.sh" || true
else
    nohup setsid bash "$SCRIPT" \
        > "$MASTER_LOG" 2>&1 < /dev/null &

    PID="$!"

    echo "RQ3_PE_PDS_V13_STARTED"
    echo "PID=$PID"
    echo "LOG=$MASTER_LOG"
fi

sleep 15

echo
echo "========== PROCESS =========="

pgrep -af \
'run_rq3_pe_pds_v13_train_eval.sh|train_qwen3_06b_full_sft_ascend.py' \
|| true

echo
echo "========== MASTER LOG =========="

tail -n 100 "$MASTER_LOG" 2>/dev/null || true

echo
echo "RQ3_PE_PDS_V13_DETACHED_SAFE"
