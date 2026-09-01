#!/usr/bin/env bash
set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

EXP_DATA="$DATA_ROOT/$EXP"
EXP_RUN="$RUN_ROOT/$EXP"
EXP_LOG="$LOG_ROOT/$EXP"

MODEL="$MODEL_ROOT/Qwen3-0.6B"

TRAIN="$EXP_DATA/rq3_pe_pds_wa_v14.jsonl"

EXPECTED_ROWS=32009
EXPECTED_SHA="299b016a29dd8828a36fe98f2e4a8fa683a895bccacde78b58edaeff35f44498"

FROZEN_PE_PDS="$EXP_DATA/rq3_pe_plus_pds_v13_paperbudget.jsonl"
EXPECTED_PE_PDS_SHA="e03f13ef18d8a5c0135524e8c310b160e8061f92c7c2ffa98e11f0e36968b46b"

TRAINER="$ROOT/scripts/pilot_v2/train_qwen3_06b_full_sft_ascend.py"

EVAL="$ROOT/scripts/pilot_v2/eval_qwen3_06b_mt_ascend.py"
SCORE="$ROOT/scripts/pilot_v2/score_mt_jsonl.py"

RUN_NAME="rq3_pe_pds_wa_v14_b4ga4"

OUT="$EXP_RUN/$RUN_NAME"
EVAL_ROOT="$EXP_RUN/${RUN_NAME}_eval_epoch3"

PE_PDS_EVAL_ROOT="$EXP_RUN/rq3_pe_pds_v13_paperbudget_b4ga4_eval_epoch3"
PE_EVAL_ROOT="$EXP_RUN/pe_k1_eval_epoch3"

LOG_DIR="$EXP_LOG/$RUN_NAME"
SUMMARY="$EVAL_ROOT/rq3_pe_pds_wa_v14_summary.json"

WMT="$DATA_ROOT/pilot_v2_qwen3_06b/wmt24_zh_en998.jsonl"
FLORES="$DATA_ROOT/pilot_v2_qwen3_06b/flores_zh_en1012.jsonl"
CHALLENGE="$DATA_ROOT/pilot_v2_qwen3_06b/challenge_zh_en197.jsonl"

mkdir -p \
    "$OUT" \
    "$EVAL_ROOT" \
    "$LOG_DIR"

echo "======================================================================"
echo "RQ3-C — PE + PDS + WA V14"
echo "======================================================================"

###############################################################################
# 1. FROZEN DATA AUDIT
###############################################################################

echo
echo "======================================================================"
echo "STAGE 1/6 — FROZEN DATA AUDIT"
echo "======================================================================"

for F in \
    "$TRAIN" \
    "$FROZEN_PE_PDS" \
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
SHA=$(sha256sum "$TRAIN" | awk '{print $1}')

PE_PDS_SHA=$(
    sha256sum "$FROZEN_PE_PDS" |
    awk '{print $1}'
)

echo "TRAIN_ROWS=$ROWS"
echo "TRAIN_SHA256=$SHA"
echo "PE_PDS_SHA256=$PE_PDS_SHA"

if [ "$ROWS" -ne "$EXPECTED_ROWS" ]; then
    echo "TRAIN_ROW_MISMATCH"
    false
fi

if [ "$SHA" != "$EXPECTED_SHA" ]; then
    echo "TRAIN_SHA_MISMATCH"
    false
fi

if [ "$PE_PDS_SHA" != "$EXPECTED_PE_PDS_SHA" ]; then
    echo "FROZEN_PE_PDS_CHANGED"
    false
fi

python - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

exp = "mtpatcher_v3_full6565_20260823"

p = (
    Path(os.environ["DATA_ROOT"])
    / exp
    / "rq3_pe_pds_wa_v14.jsonl"
)

rows = []

with p.open(encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        if not line.strip():
            continue

        row = json.loads(line)

        source = row.get("source")
        target = row.get("target_translation")
        messages = row.get("messages")

        if not isinstance(source, str) or not source.strip():
            raise RuntimeError(
                f"invalid source line={line_no}"
            )

        if not isinstance(target, str) or not target.strip():
            raise RuntimeError(
                f"invalid target line={line_no}"
            )

        if not isinstance(messages, list) or not messages:
            raise RuntimeError(
                f"invalid messages line={line_no}"
            )

        if "reference" in row:
            raise RuntimeError(
                f"reference leakage line={line_no}"
            )

        rows.append(row)

counts = Counter(
    row.get("rq3_data_component")
    for row in rows
)

print("TOTAL =", len(rows))
print("COMPONENT_COUNTS =", dict(counts))

expected = {
    "PE": 3732,
    "PDS": 14878,
    "WA": 13399,
}

if len(rows) != 32009:
    raise RuntimeError(
        f"expected 32009 rows, got {len(rows)}"
    )

for key, value in expected.items():
    if counts[key] != value:
        raise RuntimeError(
            f"{key}: expected {value}, got {counts[key]}"
        )

print("RQ3_C_COMPONENT_AUDIT_PASS")
PY

echo "RQ3_C_FROZEN_DATA_AUDIT_PASS"

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
    / "rq3_pe_pds_wa_v14.jsonl"
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
    for row_pos, line in enumerate(f):
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

        seq_len = (
            len(prompt_ids)
            + len(target_ids)
            + 1
        )

        lengths.append(seq_len)

        response_lengths.append(
            len(target_ids) + 1
        )

        if seq_len > 1024:
            overlong.append(
                {
                    "row_pos":
                        row_pos,

                    "index":
                        row.get("index"),

                    "component":
                        row.get(
                            "rq3_data_component"
                        ),

                    "seq_len":
                        seq_len,
                }
            )

s = sorted(lengths)

def percentile(p):
    i = int(
        round(
            (len(s) - 1) * p
        )
    )

    return s[
        min(
            len(s) - 1,
            max(0, i),
        )
    ]

print("ROWS =", len(lengths))
print("SEQ_MIN =", min(lengths))
print(
    "SEQ_MEAN =",
    sum(lengths) / len(lengths),
)
print("SEQ_P50 =", percentile(0.50))
print("SEQ_P95 =", percentile(0.95))
print("SEQ_P99 =", percentile(0.99))
print("SEQ_MAX =", max(lengths))

print(
    "RESPONSE_MEAN =",
    sum(response_lengths)
    / len(response_lengths),
)

print(
    "OVERLONG_1024 =",
    len(overlong),
)

if overlong:
    print(
        "OVERLONG_FIRST20 =",
        overlong[:20],
    )

    raise RuntimeError(
        "Frozen max_length=1024 violated"
    )

print("RQ3_C_TOKEN_LENGTH_PASS")
PY

###############################################################################
# 3. TRAIN FROM BASE
###############################################################################

echo
echo "======================================================================"
echo "STAGE 3/6 — TRAIN PE+PDS+WA FROM BASE"
echo "======================================================================"

if [ -f "$OUT/epoch3/model.safetensors" ] \
   && [ -f "$OUT/training_manifest.json" ]; then

    echo "RQ3_C_TRAIN_ALREADY_COMPLETE"

else

    if [ -d "$OUT" ] \
       && [ "$(find "$OUT" -mindepth 1 -maxdepth 1 | wc -l)" -gt 0 ]; then

        STAMP=$(date +%Y%m%d_%H%M%S)

        mv \
            "$OUT" \
            "${OUT}.partial_${STAMP}"

        mkdir -p "$OUT"

        echo \
            "PRESERVED_PARTIAL=${OUT}.partial_${STAMP}"
    fi

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
    echo "RQ3_C_EPOCH3_MODEL_MISSING"
    false
fi

if [ ! -f "$OUT/training_manifest.json" ]; then
    echo "RQ3_C_MANIFEST_MISSING"
    false
fi

echo "RQ3_C_TRAIN_PASS"

###############################################################################
# 4. FIXED EPOCH3 EVAL
###############################################################################

echo
echo "======================================================================"
echo "STAGE 4/6 — FIXED EPOCH3 EVALUATION"
echo "======================================================================"

MODEL_E3="$OUT/epoch3"
TOKENIZER="$MODEL_ROOT/Qwen3-0.6B"

PIDS=()

run_eval () {
    local NAME="$1"
    local CARD="$2"
    local DATA="$3"

    local DEST="$EVAL_ROOT/$NAME"
    local LOG="$LOG_DIR/eval_${NAME}.log"

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
            --method "rq3_pe_pds_wa_v14_epoch3_${NAME}" \
            --batch-size 16 \
            --max-new-tokens 256 \
            --attn-implementation sdpa

        python "$SCORE" \
            --input "$DEST/predictions.jsonl" \
            --output "$DEST/metrics.json"

        echo \
            "RQ3_C_EVAL_DATASET_COMPLETE name=$NAME"

    ) > "$LOG" 2>&1 &

    PIDS+=("$!")
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
    echo "RQ3_C_EVAL_WORKER_FAILURE"
    false
fi

for NAME in wmt24 flores challenge; do

    METRIC="$EVAL_ROOT/$NAME/metrics.json"

    if [ ! -f "$METRIC" ]; then
        echo \
            "RQ3_C_MISSING_METRIC=$NAME"
        false
    fi

done

echo "RQ3_C_EVAL_PASS"

###############################################################################
# 5. CAUSAL SUMMARY
###############################################################################

echo
echo "======================================================================"
echo "STAGE 5/6 — RQ3 CAUSAL SUMMARY"
echo "======================================================================"

export \
    EVAL_ROOT \
    PE_PDS_EVAL_ROOT \
    PE_EVAL_ROOT \
    SUMMARY

python - <<'PY'
import json
import os
from pathlib import Path


names = (
    "wmt24",
    "flores",
    "challenge",
)

eval_root = Path(
    os.environ["EVAL_ROOT"]
)

pe_pds_root = Path(
    os.environ["PE_PDS_EVAL_ROOT"]
)

pe_root = Path(
    os.environ["PE_EVAL_ROOT"]
)

summary = Path(
    os.environ["SUMMARY"]
)


base = {
    "wmt24": {
        "BLEU":
            15.536214,

        "chrF":
            45.537530,
    },

    "flores": {
        "BLEU":
            19.971480,

        "chrF":
            50.860857,
    },

    "challenge": {
        "BLEU":
            16.537871,

        "chrF":
            45.758287,
    },
}


def read_metric(path):
    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


def resolve_metric(
    root,
    name,
):
    direct = (
        root
        / name
        / "metrics.json"
    )

    if direct.exists():
        return direct

    candidates = []

    if root.exists():
        for p in root.rglob(
            "*.json"
        ):
            if (
                name.lower()
                not in str(p).lower()
            ):
                continue

            try:
                obj = read_metric(p)
            except Exception:
                continue

            if (
                "BLEU" in obj
                and "chrF" in obj
            ):
                candidates.append(p)

    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(
        f"Could not resolve metric "
        f"name={name} root={root} "
        f"candidates={candidates}"
    )


pe = {}

pe_pds = {}

full = {}


for name in names:

    pe[name] = read_metric(
        resolve_metric(
            pe_root,
            name,
        )
    )

    pe_pds[name] = read_metric(
        resolve_metric(
            pe_pds_root,
            name,
        )
    )

    full[name] = read_metric(
        eval_root
        / name
        / "metrics.json"
    )


def avg_delta(
    a,
    b,
    metric,
):
    return sum(
        float(a[name][metric])
        -
        float(b[name][metric])
        for name in names
    ) / len(names)


result = {
    "research_question":
        "RQ3: Can OPD replace MT-PATCHER Knowledge Extension?",

    "anchor":
        "C = PE + PDS + WA",

    "protocol":
        "fixed_epoch3",

    "student":
        "Qwen3-0.6B",

    "training_rows":
        32009,

    "component_rows": {
        "PE":
            3732,

        "PDS":
            14878,

        "WA":
            13399,
    },

    "natural_baseline_note":
        (
            "Three epochs over each natural dataset size. "
            "This comparison is not compute-matched."
        ),

    "base":
        base,

    "PE":
        {
            name: {
                "BLEU":
                    float(
                        pe[name]["BLEU"]
                    ),

                "chrF":
                    float(
                        pe[name]["chrF"]
                    ),
            }
            for name in names
        },

    "PE_plus_PDS":
        {
            name: {
                "BLEU":
                    float(
                        pe_pds[name]["BLEU"]
                    ),

                "chrF":
                    float(
                        pe_pds[name]["chrF"]
                    ),
            }
            for name in names
        },

    "PE_plus_PDS_plus_WA":
        {
            name: {
                "BLEU":
                    float(
                        full[name]["BLEU"]
                    ),

                "chrF":
                    float(
                        full[name]["chrF"]
                    ),
            }
            for name in names
        },
}


result[
    "avg_delta_bleu_vs_base"
] = avg_delta(
    full,
    base,
    "BLEU",
)


result[
    "avg_delta_chrf_vs_base"
] = avg_delta(
    full,
    base,
    "chrF",
)


result[
    "avg_delta_bleu_vs_PE"
] = avg_delta(
    full,
    pe,
    "BLEU",
)


result[
    "avg_delta_chrf_vs_PE"
] = avg_delta(
    full,
    pe,
    "chrF",
)


result[
    "avg_delta_bleu_WA_increment"
] = avg_delta(
    full,
    pe_pds,
    "BLEU",
)


result[
    "avg_delta_chrf_WA_increment"
] = avg_delta(
    full,
    pe_pds,
    "chrF",
)


result[
    "avg_delta_bleu_PDS_increment"
] = avg_delta(
    pe_pds,
    pe,
    "BLEU",
)


result[
    "avg_delta_chrf_PDS_increment"
] = avg_delta(
    pe_pds,
    pe,
    "chrF",
)


summary.parent.mkdir(
    parents=True,
    exist_ok=True,
)

summary.write_text(
    json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


print(
    f"{'DATASET':12s}"
    f"{'BASE':>10s}"
    f"{'PE':>10s}"
    f"{'PE+PDS':>10s}"
    f"{'FULL':>10s}"
    f"{'WA INC':>10s}"
)

print("-" * 62)


for name in names:

    b = float(
        base[name]["BLEU"]
    )

    p = float(
        pe[name]["BLEU"]
    )

    q = float(
        pe_pds[name]["BLEU"]
    )

    r = float(
        full[name]["BLEU"]
    )

    print(
        f"{name:12s}"
        f"{b:10.4f}"
        f"{p:10.4f}"
        f"{q:10.4f}"
        f"{r:10.4f}"
        f"{r-q:+10.4f}"
    )


print()

print(
    "AVG_PDS_INCREMENT_BLEU =",
    f"{result['avg_delta_bleu_PDS_increment']:+.6f}",
)

print(
    "AVG_WA_INCREMENT_BLEU =",
    f"{result['avg_delta_bleu_WA_increment']:+.6f}",
)

print(
    "AVG_FULL_DELTA_BLEU_VS_BASE =",
    f"{result['avg_delta_bleu_vs_base']:+.6f}",
)

print(
    "AVG_WA_INCREMENT_CHRF =",
    f"{result['avg_delta_chrf_WA_increment']:+.6f}",
)

print(
    "SUMMARY =",
    summary,
)

print(
    "RQ3_C_CAUSAL_SUMMARY_PASS"
)
PY

###############################################################################
# 6. FINAL
###############################################################################

echo
echo "======================================================================"
echo "STAGE 6/6 — FINAL"
echo "======================================================================"

cat "$SUMMARY"

echo

echo "RQ3_PE_PDS_WA_V14_ALL_PASS"
