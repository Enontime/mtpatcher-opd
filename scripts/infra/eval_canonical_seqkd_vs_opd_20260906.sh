#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/workspace/mtpatcher
PROJECT=$ROOT/repo/MT-Patcher-Reproduction-Ascend

source "$ROOT/project_env.sh"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

EVAL=$PROJECT/scripts/pilot_v2/eval_qwen3_06b_mt_ascend.py
SCORE=$PROJECT/scripts/pilot_v2/score_mt_jsonl.py

BASE=$MODEL_ROOT/Qwen3-0.6B
MODEL_ROOT_EVAL=$MODEL_ROOT/eval_canonical_20260906

WMT=$DATA_ROOT/pilot_v2_qwen3_06b/wmt24_zh_en998.jsonl
FLORES=$DATA_ROOT/pilot_v2_qwen3_06b/flores_zh_en1012.jsonl
CHALLENGE=$DATA_ROOT/pilot_v2_qwen3_06b/challenge_zh_en197.jsonl

RUN=$RUN_ROOT/science/canonical_seqkd_vs_opd_eval_20260906
BASE_RUN=$RUN/base

mkdir -p "$RUN"

LABELS=(
    seqkd_pass1
    seqkd_pass2
    seqkd_pass3
    opd_pass1
    opd_pass2
    opd_pass3
)

MODELS=(
    "$MODEL_ROOT_EVAL/seqkd_pass1_step1250"
    "$MODEL_ROOT_EVAL/seqkd_pass2_step2500"
    "$MODEL_ROOT_EVAL/seqkd_pass3_step3750"
    "$MODEL_ROOT_EVAL/opd_pass1_step1250"
    "$MODEL_ROOT_EVAL/opd_pass2_step2500"
    "$MODEL_ROOT_EVAL/opd_pass3_step3750"
)

DEVICES=(
    0
    1
    2
    3
    4
    5
)

echo "======================================================================"
echo "CANONICAL SEQKD VS OPD — FORMAL EVALUATION"
echo "======================================================================"

###############################################################################
# 1. Framework / evaluator preflight
###############################################################################

echo
echo "=== FRAMEWORK PREFLIGHT ==="

test -f "$EVAL"
test -f "$SCORE"

test -f "$WMT"
test -f "$FLORES"
test -f "$CHALLENGE"

python - <<'PY'
import sacrebleu

print("sacrebleu_version =", sacrebleu.__version__)
assert sacrebleu.__version__ == "2.5.1"

print("SCORER_ENV_PASS")
PY

###############################################################################
# 2. Base parity gate
###############################################################################

echo
echo "=== BASE PARITY GATE ==="

python - "$BASE_RUN" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])

historical = {
    "wmt24": 15.536214,
    "flores": 19.971480,
    "challenge": 16.537871,
}

values = {}

for name in ("wmt24", "flores", "challenge"):
    path = root / f"{name}.metrics.json"
    assert path.is_file(), path

    metric = json.loads(path.read_text(encoding="utf-8"))
    values[name] = float(metric["BLEU"])

    delta = values[name] - historical[name]

    print(
        name,
        "current=",
        f"{values[name]:.9f}",
        "historical=",
        f"{historical[name]:.9f}",
        "delta=",
        f"{delta:+.9f}",
    )

    assert abs(delta) < 1e-5

macro = sum(values.values()) / 3
historical_macro = 17.3485217

print("macro_BLEU =", f"{macro:.9f}")
print(
    "macro_delta =",
    f"{macro - historical_macro:+.9f}",
)

assert abs(macro - historical_macro) < 1e-5

print("BASE_PARITY_GATE_PASS")
PY

###############################################################################
# 3. Six-model artifact preflight
###############################################################################

echo
echo "=== MODEL ARTIFACT PREFLIGHT ==="

for i in "${!LABELS[@]}"; do
    label="${LABELS[$i]}"
    model="${MODELS[$i]}"

    test -f "$model/config.json"
    test -f "$model/model.safetensors"
    test -f "$model/tokenizer.json"
    test -f "$model/tokenizer_config.json"

    echo "$label READY $model"
done

###############################################################################
###############################################################################
# 4. Frozen Base tokenizer contract
###############################################################################

echo
echo "=== FROZEN BASE TOKENIZER CONTRACT ==="

python - \
    "$BASE" \
    "$WMT" \
    "${MODELS[@]}" <<'PY'
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

base = Path(sys.argv[1])
data = Path(sys.argv[2])
models = [Path(x) for x in sys.argv[3:]]

base_config = json.loads(
    (base / "config.json").read_text(encoding="utf-8")
)

base_vocab_size = int(base_config["vocab_size"])
base_model_type = base_config["model_type"]

tok = AutoTokenizer.from_pretrained(
    base,
    local_files_only=True,
)

with data.open("r", encoding="utf-8-sig") as f:
    sample = json.loads(
        next(line for line in f if line.strip())
    )

rendered = tok.apply_chat_template(
    sample["messages"],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

ids = tok(
    rendered,
    add_special_tokens=False,
)["input_ids"]

assert len(ids) > 0

print(
    "FROZEN_BASE_TOKENIZER_LOAD_PASS",
    "tokenizer_len=",
    len(tok),
    "base_vocab_size=",
    base_vocab_size,
    "sample_tokens=",
    len(ids),
)

for model in models:
    cfg = json.loads(
        (model / "config.json").read_text(
            encoding="utf-8"
        )
    )

    vocab_size = int(cfg["vocab_size"])
    model_type = cfg["model_type"]

    assert vocab_size == base_vocab_size, (
        model,
        vocab_size,
        base_vocab_size,
    )

    assert model_type == base_model_type, (
        model,
        model_type,
        base_model_type,
    )

    print(
        "MODEL_TOKENIZER_COMPAT_PASS",
        model.name,
        "vocab_size=",
        vocab_size,
        "model_type=",
        model_type,
    )

print("FROZEN_BASE_TOKENIZER_CONTRACT_PASS")
PY

###############################################################################
# 5. Frozen provenance
###############################################################################

echo
echo "=== FROZEN PROVENANCE ==="

MANIFEST=$RUN/evaluation_manifest.tsv
PROVENANCE=$RUN/evaluation_provenance.sha256

printf "label\tdevice\tmodel\n" > "$MANIFEST"

for i in "${!LABELS[@]}"; do
    printf "%s\t%s\t%s\n" \
        "${LABELS[$i]}" \
        "${DEVICES[$i]}" \
        "${MODELS[$i]}" \
        >> "$MANIFEST"
done

cat "$MANIFEST"

{
    sha256sum \
        "$EVAL" \
        "$SCORE" \
        "$WMT" \
        "$FLORES" \
        "$CHALLENGE"

    for model in "${MODELS[@]}"; do
        sha256sum \
            "$model/config.json" \
            "$model/generation_config.json" \
            "$model/model.safetensors" \
            "$model/tokenizer.json" \
            "$model/tokenizer_config.json"

        if test -f "$model/chat_template.jinja"; then
            sha256sum "$model/chat_template.jinja"
        fi
    done
} > "$PROVENANCE"

echo "MANIFEST=$MANIFEST"
echo "PROVENANCE=$PROVENANCE"

###############################################################################
# 6. Formal evaluator worker
###############################################################################

eval_model () {
    local LABEL="$1"
    local MODEL="$2"
    local DEVICE="$3"

    local OUT="$RUN/$LABEL"

    mkdir -p "$OUT"

    echo "MODEL_START label=$LABEL device=$DEVICE"

    for NAME in wmt24 flores challenge; do

        local INPUT

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

        local PRED="$OUT/${NAME}.pred.jsonl"
        local METRIC="$OUT/${NAME}.metrics.json"

        local PRED_PART="$OUT/${NAME}.pred.partial.jsonl"
        local METRIC_PART="$OUT/${NAME}.metrics.partial.json"

        local LOG="$OUT/${NAME}.eval.log"

        rm -f "$METRIC_PART"

        if test -f "$PRED" && test -f "$METRIC"; then
            echo \
                "DATASET_REUSE label=$LABEL dataset=$NAME"
            continue
        fi

        echo \
            "DATASET_START label=$LABEL dataset=$NAME device=$DEVICE"

        ASCEND_RT_VISIBLE_DEVICES="$DEVICE" \
        python -u "$EVAL" \
            --model "$MODEL" \
            --tokenizer "$BASE" \
            --input "$INPUT" \
            --output "$PRED_PART" \
            --method "canonical_${LABEL}_${NAME}" \
            --batch-size 16 \
            --max-new-tokens 256 \
            --attn-implementation sdpa \
            > "$LOG" \
            2>&1

        python "$SCORE" \
            --input "$PRED_PART" \
            --output "$METRIC_PART" \
            >> "$LOG" \
            2>&1

        mv "$PRED_PART" "$PRED"
        mv "$METRIC_PART" "$METRIC"

        echo \
            "DATASET_COMPLETE label=$LABEL dataset=$NAME"

        cat "$METRIC"
    done

    echo "MODEL_COMPLETE label=$LABEL device=$DEVICE"
}

###############################################################################
# 7. Six models in parallel, one NPU each
###############################################################################

echo
echo "=== SIX-MODEL EVALUATION ==="

PIDS=()

for i in "${!LABELS[@]}"; do
    eval_model \
        "${LABELS[$i]}" \
        "${MODELS[$i]}" \
        "${DEVICES[$i]}" \
        &

    PIDS+=("$!")
done

BAD=0

set +e

for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"
    RC=$?

    echo \
        "WORKER_STATUS label=${LABELS[$i]} rc=$RC"

    if test "$RC" -ne 0; then
        BAD=1
    fi
done

set -e

test "$BAD" -eq 0

echo "ALL_SIX_MODEL_EVALUATIONS_PASS"

###############################################################################
# 8. Formal summary
###############################################################################

echo
echo "======================================================================"
echo "CANONICAL COMPARISON SUMMARY"
echo "======================================================================"

python - "$RUN" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])

labels = [
    "base",
    "seqkd_pass1",
    "seqkd_pass2",
    "seqkd_pass3",
    "opd_pass1",
    "opd_pass2",
    "opd_pass3",
]

sets = [
    "wmt24",
    "flores",
    "challenge",
]

result = {}

for label in labels:
    result[label] = {}

    for name in sets:
        metric = json.loads(
            (
                root
                / label
                / f"{name}.metrics.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        result[label][name] = {
            "BLEU": float(metric["BLEU"]),
            "chrF": float(metric["chrF"]),
            "sacrebleu_version":
                metric["sacrebleu_version"],
        }

    result[label]["macro_BLEU"] = (
        sum(
            result[label][name]["BLEU"]
            for name in sets
        )
        / 3
    )

    result[label]["macro_chrF"] = (
        sum(
            result[label][name]["chrF"]
            for name in sets
        )
        / 3
    )

base = result["base"]["macro_BLEU"]

for label in labels:
    result[label]["delta_vs_base"] = (
        result[label]["macro_BLEU"]
        - base
    )

contrasts = {}

for p in (1, 2, 3):
    seq = result[f"seqkd_pass{p}"]["macro_BLEU"]
    opd = result[f"opd_pass{p}"]["macro_BLEU"]

    contrasts[f"pass{p}"] = {
        "OPD_minus_SeqKD": opd - seq,
        "SeqKD_minus_Base": seq - base,
        "OPD_minus_Base": opd - base,
    }

seq3_gain = contrasts["pass3"]["SeqKD_minus_Base"]
opd3_gain = contrasts["pass3"]["OPD_minus_Base"]

gain_recovery = (
    opd3_gain / seq3_gain
    if seq3_gain > 0
    else None
)

historical_fullseqkd = 19.1652

summary = {
    "models": result,
    "contrasts": contrasts,
    "primary_pass3": {
        "SeqKD_macro_BLEU":
            result["seqkd_pass3"]["macro_BLEU"],
        "OPD_macro_BLEU":
            result["opd_pass3"]["macro_BLEU"],
        "OPD_minus_SeqKD":
            contrasts["pass3"]["OPD_minus_SeqKD"],
        "SeqKD_minus_Base":
            seq3_gain,
        "OPD_minus_Base":
            opd3_gain,
        "gain_recovery_ratio_descriptive":
            gain_recovery,
    },
    "historical_reference": {
        "FullSeqKD20k_macro_BLEU":
            historical_fullseqkd,
        "canonical_SeqKD_pass3_minus_historical":
            (
                result["seqkd_pass3"]["macro_BLEU"]
                - historical_fullseqkd
            ),
    },
}

out = root / "canonical_comparison_summary.json"

out.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print(
    f"{'MODEL':16s}"
    f"{'WMT24':>11s}"
    f"{'FLORES':>11s}"
    f"{'CHALL':>11s}"
    f"{'MACRO':>11s}"
    f"{'chrF':>11s}"
    f"{'ΔBASE':>11s}"
)

for label in labels:
    x = result[label]

    print(
        f"{label:16s}"
        f"{x['wmt24']['BLEU']:11.4f}"
        f"{x['flores']['BLEU']:11.4f}"
        f"{x['challenge']['BLEU']:11.4f}"
        f"{x['macro_BLEU']:11.4f}"
        f"{x['macro_chrF']:11.4f}"
        f"{x['delta_vs_base']:11.4f}"
    )

print()

for p in (1, 2, 3):
    x = contrasts[f"pass{p}"]

    print(
        f"PASS{p}: "
        f"SeqKD-Base={x['SeqKD_minus_Base']:+.6f} "
        f"OPD-Base={x['OPD_minus_Base']:+.6f} "
        f"OPD-SeqKD={x['OPD_minus_SeqKD']:+.6f}"
    )

print()
print(
    "PRIMARY_PASS3_OPD_MINUS_SEQKD =",
    f"{contrasts['pass3']['OPD_minus_SeqKD']:+.9f}",
)

print(
    "PRIMARY_PASS3_SEQKD_MINUS_BASE =",
    f"{seq3_gain:+.9f}",
)

print(
    "PRIMARY_PASS3_OPD_MINUS_BASE =",
    f"{opd3_gain:+.9f}",
)

print(
    "PASS3_GAIN_RECOVERY_RATIO_DESCRIPTIVE =",
    gain_recovery,
)

print(
    "CANONICAL_SEQKD_PASS3_MINUS_"
    "HISTORICAL_FULLSEQKD20K =",
    f"{result['seqkd_pass3']['macro_BLEU'] - historical_fullseqkd:+.9f}",
)

print()
print("SUMMARY_JSON =", out)
print("CANONICAL_COMPARISON_COMPLETE")
PY
