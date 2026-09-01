#!/usr/bin/env bash
set -Eeuo pipefail

source /workspace/mtpatcher/project_env.sh

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

EXP="mtpatcher_v3_full6565_20260823"
D="$DATA_ROOT/$EXP"

SCRIPT_DIR="$ROOT/scripts/mtpatcher_v3"

HELPER="$SCRIPT_DIR/strong_repro_wa_core_v1.py"

WORKER="$SCRIPT_DIR/strong_repro_pds_qwen_worker_v1.py"

PARSER="$SCRIPT_DIR/build_pds_structural_acceptance_v1.py"

OFFICIAL="$ROOT/vendor/MT-Patcher-official"

TEACHER="$MODEL_ROOT/Qwen3-8B"
STUDENT="$MODEL_ROOT/Qwen3-0.6B"

K1_FROZEN="/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/strong_repro_student_arms_freeze_v1/k1_all_pe11792_v1.jsonl"
K1="/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/strong_repro_wa_smoke128_v2/k1_wa_anchor_reconstructed11792_v2.jsonl"

ANALYSIS="$D/strong_repro_pds_full_to_student_v2/analysis_topic_domain_style_v2.jsonl"

OUT="$D/strong_repro_wa_smoke128_v2"

ANALOG_JOBS="$OUT/wa_analog_jobs128_v1.jsonl"
ANALOG_SHARDS="$OUT/analog_shards16"
ANALOG_PAIRS="$OUT/wa_analog_pairs_v1.jsonl"

ANCHOR_REPORT="$OUT/wa_anchor_report_v1.json"
ANALOG_REPORT="$OUT/wa_analog_report_v1.json"

CONTEXT_JOBS="$OUT/wa_context_jobs_v1.jsonl"
CONTEXT_SHARDS="$OUT/context_shards16"

ACCEPTED="$OUT/wa_accepted_v1.jsonl"
SFT="$OUT/wa_sft_rows_v1.jsonl"
CONTEXT_REPORT="$OUT/wa_context_report_v1.json"

SMOKE_REPORT="$OUT/wa_smoke128_final_v1.json"

PASS="$OUT/STRONG_REPRO_WA_SMOKE128_V2.PASS"
FAIL="$OUT/STRONG_REPRO_WA_SMOKE128_V2.FAIL"

mkdir -p \
    "$OUT" \
    "$ANALOG_SHARDS" \
    "$CONTEXT_SHARDS"

rm -f "$PASS" "$FAIL"

START="$(date +%s)"

echo "$START" > "$OUT/start_epoch.txt"

on_error () {
    rc=$?

    echo
    echo "======================================================================"
    echo "WA SMOKE128 FAILED"
    echo "RETURN_CODE=$rc"
    echo "FAIL_CST=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "======================================================================"

    touch "$FAIL"
}

trap on_error ERR


echo "======================================================================"
echo "STRONG REPRO — WA END-TO-END IMPLEMENTATION SMOKE128 V2 — CANONICAL LOCAL-PAIR ANCHORS"
echo "======================================================================"
echo
echo "NO STUDENT TRAINING"
echo "NO BLEU"
echo "NO SEMANTIC AUDIT"
echo "NO PROMPT TUNING"
echo
echo "Purpose = cheap implementation falsification before Full WA"
echo
echo "预计运行时长：5–12 分钟"
echo "ETA confidence=MEDIUM"
echo
echo "START_CST=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
echo "START_UTC=$(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
echo
echo "ETA_MIN_CST=$(TZ=Asia/Shanghai date -d "@$((START + 5*60))" '+%Y-%m-%d %H:%M:%S %Z')"
echo "ETA_MAX_CST=$(TZ=Asia/Shanghai date -d "@$((START + 12*60))" '+%Y-%m-%d %H:%M:%S %Z')"
echo "ETA_MIN_UTC=$(TZ=UTC date -d "@$((START + 5*60))" '+%Y-%m-%d %H:%M:%S %Z')"
echo "ETA_MAX_UTC=$(TZ=UTC date -d "@$((START + 12*60))" '+%Y-%m-%d %H:%M:%S %Z')"


###############################################################################
# PRE-FLIGHT
###############################################################################

echo
echo "======================================================================"
echo "STAGE 1/6 — PREFLIGHT"
echo "======================================================================"

BAD=0

for F in \
    "$HELPER" \
    "$WORKER" \
    "$PARSER" \
    "$K1_FROZEN" \
    "$K1" \
    "$ANALYSIS"
do
    if [ ! -f "$F" ]; then
        echo "MISSING=$F"
        BAD=1
    fi
done

for M in \
    "$TEACHER" \
    "$STUDENT" \
    "$OFFICIAL"
do
    if [ ! -d "$M" ]; then
        echo "MISSING_DIR=$M"
        BAD=1
    fi
done

if [ "$BAD" -ne 0 ]; then
    false
fi

[ "$(wc -l < "$K1")" -eq 11792 ]

[ "$(wc -l < "$ANALYSIS")" -eq 11669 ]

K1_SHA="$(sha256sum "$K1_FROZEN" | awk '{print $1}')"

echo "K1_SHA=$K1_SHA"

[ "$K1_SHA" = \
"7d8c5c1de8249db19e7e881174ef103c9ab6ebfa590623b6ef3190283684ed18" ]

python -m py_compile "$HELPER"

echo "PREFLIGHT_PASS"


###############################################################################
# Shared 16-NPU launcher
###############################################################################

run_16way () {
    local INPUT="$1"
    local SHARD_DIR="$2"
    local LOG_DIR="$3"

    mkdir -p \
        "$SHARD_DIR" \
        "$LOG_DIR"

    rm -f \
        "$SHARD_DIR"/device_*.jsonl \
        "$LOG_DIR"/device_*.log

    local PIDS=()
    local DEVICE

    for DEVICE in $(seq 0 15)
    do
        python -u "$WORKER" \
            --input "$INPUT" \
            --output "$SHARD_DIR/device_${DEVICE}.jsonl" \
            --model "$TEACHER" \
            --device-id "$DEVICE" \
            --world-size 16 \
            --batch-size 8 \
            --max-new-tokens 256 \
            --mode case \
            --seed 20260831 \
            > "$LOG_DIR/device_${DEVICE}.log" \
            2>&1 &

        PIDS+=("$!")

        echo \
            "WORKER_START device=$DEVICE pid=$!"
    done

    local BAD_WORKER=0
    local PID

    for PID in "${PIDS[@]}"
    do
        if wait "$PID"; then
            :
        else
            echo "WORKER_FAILED pid=$PID"
            BAD_WORKER=1
        fi
    done

    [ "$BAD_WORKER" -eq 0 ]
}


###############################################################################
# STAGE 2 — 128 ANALOG JOBS
###############################################################################

echo
echo "======================================================================"
echo "STAGE 2/6 — WORD ANALOGY GENERATION"
echo "======================================================================"

python -u "$HELPER" \
    build-analog-jobs \
    --k1 "$K1" \
    --analysis "$ANALYSIS" \
    --official "$OFFICIAL" \
    --output "$ANALOG_JOBS" \
    --report "$ANCHOR_REPORT" \
    --sample-size 128 \
    --seed 20260831

[ "$(wc -l < "$ANALOG_JOBS")" -eq 128 ]

echo 128 \
    > "$OUT/analog_expected_rows.txt"

date +%s \
    > "$OUT/analog_start_epoch.txt"

run_16way \
    "$ANALOG_JOBS" \
    "$ANALOG_SHARDS" \
    "$OUT/log_analog16"

date +%s \
    > "$OUT/analog_finish_epoch.txt"


###############################################################################
# STAGE 3 — PARSE 2 + 2
###############################################################################

echo
echo "======================================================================"
echo "STAGE 3/6 — ANALOG STRUCTURAL PARSE"
echo "======================================================================"

python -u "$HELPER" \
    parse-analogs \
    --jobs "$ANALOG_JOBS" \
    --shards "$ANALOG_SHARDS" \
    --output "$ANALOG_PAIRS" \
    --report "$ANALOG_REPORT"

python - "$ANALOG_REPORT" <<'PY'
import json
import sys

r = json.load(
    open(
        sys.argv[1],
        encoding="utf-8",
    )
)

json_rate = float(
    r["json_object_rate"]
)

both_rate = float(
    r["both_aspects_rate"]
)

pairs = int(
    r["valid_analog_pairs"]
)

print(
    "ENGINEERING_GATE_JSON_RATE =",
    json_rate,
)

print(
    "ENGINEERING_GATE_BOTH_ASPECT_RATE =",
    both_rate,
)

print(
    "ENGINEERING_GATE_VALID_ANALOG_PAIRS =",
    pairs,
)

# Loose catastrophic-failure gates only.
# They are NOT semantic-quality criteria.
if json_rate < 0.70:
    raise RuntimeError(
        "SMOKE_FAIL: WA JSON interface catastrophically low"
    )

if both_rate < 0.60:
    raise RuntimeError(
        "SMOKE_FAIL: both WA aspects structurally absent too often"
    )

if pairs < 256:
    raise RuntimeError(
        "SMOKE_FAIL: fewer than 2 valid analog pairs per anchor on average"
    )

print(
    "ANALOG_ENGINEERING_GATES_PASS"
)
PY


###############################################################################
# STAGE 4 — ONE CONTEXT PER VALID ANALOG
###############################################################################

echo
echo "======================================================================"
echo "STAGE 4/6 — WA CONTEXT GENERATION"
echo "======================================================================"

python -u "$HELPER" \
    build-context-jobs \
    --analogs "$ANALOG_PAIRS" \
    --official "$OFFICIAL" \
    --output "$CONTEXT_JOBS"

CONTEXT_N="$(wc -l < "$CONTEXT_JOBS")"

echo "$CONTEXT_N" \
    > "$OUT/context_expected_rows.txt"

date +%s \
    > "$OUT/context_start_epoch.txt"

run_16way \
    "$CONTEXT_JOBS" \
    "$CONTEXT_SHARDS" \
    "$OUT/log_context16"

date +%s \
    > "$OUT/context_finish_epoch.txt"


###############################################################################
# STAGE 5 — FROZEN STRUCTURAL ACCEPTANCE
###############################################################################

echo
echo "======================================================================"
echo "STAGE 5/6 — BILINGUAL STRUCTURAL ACCEPTANCE"
echo "======================================================================"

python -u "$HELPER" \
    accept-contexts \
    --jobs "$CONTEXT_JOBS" \
    --shards "$CONTEXT_SHARDS" \
    --parser "$PARSER" \
    --accepted "$ACCEPTED" \
    --sft "$SFT" \
    --report "$CONTEXT_REPORT"

python - "$CONTEXT_REPORT" <<'PY'
import json
import sys

r = json.load(
    open(
        sys.argv[1],
        encoding="utf-8",
    )
)

rate = float(
    r["accept_rate"]
)

accepted = int(
    r["accepted"]
)

print(
    "ENGINEERING_GATE_CONTEXT_ACCEPT_RATE =",
    rate,
)

print(
    "ENGINEERING_GATE_CONTEXT_ACCEPTED =",
    accepted,
)

# Intentionally loose.
# This catches broken parsing / schema / prompt interfaces,
# not ordinary model-quality variation.
if rate < 0.20:
    raise RuntimeError(
        "SMOKE_FAIL: context structural acceptance below 20%"
    )

if accepted < 64:
    raise RuntimeError(
        "SMOKE_FAIL: fewer than 64 accepted WA contexts"
    )

print(
    "CONTEXT_ENGINEERING_GATES_PASS"
)
PY


###############################################################################
# STAGE 6 — STUDENT INPUT SCHEMA, NO TRAINING
###############################################################################

echo
echo "======================================================================"
echo "STAGE 6/6 — STUDENT SFT SCHEMA / TOKENIZATION"
echo "======================================================================"

python -u "$HELPER" \
    validate-sft \
    --sft "$SFT" \
    --model "$STUDENT"

python - \
    "$ANCHOR_REPORT" \
    "$ANALOG_REPORT" \
    "$CONTEXT_REPORT" \
    "$SMOKE_REPORT" <<'PY'

import json
import sys
from pathlib import Path

anchor = json.load(
    open(
        sys.argv[1],
        encoding="utf-8",
    )
)

analog = json.load(
    open(
        sys.argv[2],
        encoding="utf-8",
    )
)

context = json.load(
    open(
        sys.argv[3],
        encoding="utf-8",
    )
)

final = {
    "protocol":
        "STRONG_REPRO_WA_SMOKE128_V2",

    "purpose":
        "implementation smoke only; no scientific WA-effect claim",

    "anchor":
        anchor,

    "analog":
        analog,

    "context":
        context,

    "decision":
        "IMPLEMENTATION_PASS",

    "full_WA_authorized":
        True,

    "frozen_next_step":
        (
            "Immediately run Full WA with the same WA core; "
            "no semantic audit or tuning inserted."
        ),
}

Path(
    sys.argv[4]
).write_text(
    json.dumps(
        final,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print(
    json.dumps(
        final,
        ensure_ascii=False,
        indent=2,
    )
)

print(
    "WA_SMOKE128_IMPLEMENTATION_PASS"
)
PY

touch "$PASS"

rm -f "$FAIL"

END="$(date +%s)"

echo
echo "======================================================================"
echo "WA SMOKE128 PASS"
echo "======================================================================"
echo "TOTAL_ELAPSED_SECONDS=$((END - START))"
echo "FINISH_CST=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
echo "FINISH_UTC=$(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
echo "REPORT=$SMOKE_REPORT"
echo "PASS=$PASS"
echo
echo "NEXT = FULL WA immediately; no further corpus audit."
echo "======================================================================"
