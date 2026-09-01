#!/usr/bin/env bash
set -Eeuo pipefail

source /workspace/mtpatcher/project_env.sh

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

###############################################################################
# FROZEN PROVENANCE
###############################################################################

EXP="mtpatcher_v3_full6565_20260823"
D="$DATA_ROOT/$EXP"

SCRIPT_DIR="$ROOT/scripts/mtpatcher_v3"

HELPER="$SCRIPT_DIR/strong_repro_wa_core_v1.py"
WORKER="$SCRIPT_DIR/strong_repro_pds_qwen_worker_v1.py"
PARSER="$SCRIPT_DIR/build_pds_structural_acceptance_v1.py"

TRAINER="$ROOT/scripts/pilot_v2/train_qwen3_06b_full_sft_ascend.py"
EVAL="$ROOT/scripts/pilot_v2/eval_qwen3_06b_mt_ascend.py"
SCORE="$ROOT/scripts/pilot_v2/score_mt_jsonl.py"

OFFICIAL="$ROOT/vendor/MT-Patcher-official"

TEACHER="$MODEL_ROOT/Qwen3-8B"
STUDENT="$MODEL_ROOT/Qwen3-0.6B"

###############################################################################
# Frozen Strong-Repro inputs
###############################################################################

K1_FROZEN="$D/strong_repro_student_arms_freeze_v1/k1_all_pe11792_v1.jsonl"

SMOKE="$D/strong_repro_wa_smoke128_v2"

K1_WA="$SMOKE/k1_wa_anchor_reconstructed11792_v2.jsonl"

SMOKE_PASS="$SMOKE/STRONG_REPRO_WA_SMOKE128_V2.PASS"

SMOKE_REPORT="$SMOKE/wa_smoke128_final_v1.json"

ANALYSIS="$D/strong_repro_pds_full_to_student_v2/analysis_topic_domain_style_v2.jsonl"

BASE_PE_PDS="$D/strong_repro_pds_full_to_student_v2/k1all_plus_pds_v2.jsonl"

PDS_FINAL="$D/strong_repro_pds_full_to_student_v2/final_student_bleu_v2.json"

###############################################################################
# Full WA outputs
###############################################################################

OUT="$D/strong_repro_wa_full_v1"

ANALOG_JOBS="$OUT/wa_analog_jobs_full_v1.jsonl"
ANALOG_SHARDS="$OUT/analog_shards16"
ANALOG_REPORT="$OUT/wa_analog_report_full_v1.json"
ANCHOR_REPORT="$OUT/wa_anchor_report_full_v1.json"

ANALOG_PAIRS="$OUT/wa_analog_pairs_full_v1.jsonl"

CONTEXT_JOBS="$OUT/wa_context_jobs_full_v1.jsonl"
CONTEXT_SHARDS="$OUT/context_shards16"

WA_ACCEPTED="$OUT/wa_accepted_full_v1.jsonl"
WA_SFT_RAW="$OUT/wa_sft_rows_raw_full_v1.jsonl"
WA_CONTEXT_REPORT="$OUT/wa_context_report_full_v1.json"

WA_SFT="$OUT/wa_sft_rows_full_v1.jsonl"
TRAIN_DATA="$OUT/pe_pds_wa_train_full_v1.jsonl"

DATA_MANIFEST="$OUT/pe_pds_wa_data_manifest_full_v1.json"

RUN_DIR="$RUN_ROOT/$EXP/strong_repro_pe_pds_wa_full_v1"

FINAL="$OUT/final_pe_pds_wa_bleu_full_v1.json"

PASS="$OUT/STRONG_REPRO_WA_FULL_V1.PASS"
FAIL="$OUT/STRONG_REPRO_WA_FULL_V1.FAIL"

mkdir -p \
    "$OUT" \
    "$ANALOG_SHARDS" \
    "$CONTEXT_SHARDS"

rm -f "$PASS" "$FAIL"

START_EPOCH="$(date +%s)"

echo "$START_EPOCH" \
    > "$OUT/start_epoch.txt"

on_error () {
    rc=$?

    echo
    echo "======================================================================"
    echo "STRONG REPRO FULL WA FAILED"
    echo "RETURN_CODE=$rc"
    echo "FAIL_CST=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "FAIL_UTC=$(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "======================================================================"

    touch "$FAIL"
}

trap on_error ERR


echo "======================================================================"
echo "STRONG REPRO — FULL WA -> STUDENT BLEU V1"
echo "======================================================================"
echo
echo "FROZEN SCIENTIFIC QUESTION:"
echo "  Does WA add downstream utility beyond frozen PE+PDS?"
echo
echo "PRIMARY:"
echo "  ΔWA = BLEU(PE+PDS+WA) - BLEU(PE+PDS)"
echo
echo "FROZEN TREATMENT:"
echo "  one first canonical error anchor / PE parent"
echo "  2 Category analogs"
echo "  2 Semantic/co-occurrence analogs"
echo "  1 new context / analog"
echo
echo "FROZEN CAVEAT:"
echo "  Feedback-derived P<->Q is not always a clean analogy-compatible unit."
echo "  Some topic/phrase-level drift is retained intentionally."
echo
echo "NO MORE WA AUDIT"
echo "NO SEMANTIC CLEANER"
echo "NO LENGTH FILTER"
echo "NO MATCHED WA CONTROL"
echo "NO PROMPT TUNING"
echo "NO K2"
echo
echo "预计总运行时长：3–5 小时"
echo "ETA confidence=MEDIUM"
echo
echo "START_CST=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
echo "START_UTC=$(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
echo
echo "预计完成窗口："
echo "  CST_MIN=$(TZ=Asia/Shanghai date -d "@$((START_EPOCH + 3*3600))" '+%Y-%m-%d %H:%M:%S %Z')"
echo "  CST_MAX=$(TZ=Asia/Shanghai date -d "@$((START_EPOCH + 5*3600))" '+%Y-%m-%d %H:%M:%S %Z')"
echo "  UTC_MIN=$(TZ=UTC date -d "@$((START_EPOCH + 3*3600))" '+%Y-%m-%d %H:%M:%S %Z')"
echo "  UTC_MAX=$(TZ=UTC date -d "@$((START_EPOCH + 5*3600))" '+%Y-%m-%d %H:%M:%S %Z')"


###############################################################################
# Shared 16-NPU generation launcher
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
# STAGE 1 — PREFLIGHT / FREEZE
###############################################################################

echo
echo "======================================================================"
echo "STAGE 1/9 — PREFLIGHT"
echo "======================================================================"

BAD=0

for F in \
    "$HELPER" \
    "$WORKER" \
    "$PARSER" \
    "$TRAINER" \
    "$EVAL" \
    "$SCORE" \
    "$K1_FROZEN" \
    "$K1_WA" \
    "$ANALYSIS" \
    "$BASE_PE_PDS" \
    "$PDS_FINAL" \
    "$SMOKE_PASS" \
    "$SMOKE_REPORT"
do
    if [ ! -f "$F" ]; then
        echo "MISSING=$F"
        BAD=1
    fi
done

for DIR in \
    "$OFFICIAL" \
    "$TEACHER" \
    "$STUDENT"
do
    if [ ! -d "$DIR" ]; then
        echo "MISSING_DIR=$DIR"
        BAD=1
    fi
done

if [ "$BAD" -ne 0 ]; then
    false
fi

[ "$(wc -l < "$K1_FROZEN")" -eq 11792 ]

[ "$(wc -l < "$K1_WA")" -eq 11792 ]

[ "$(wc -l < "$ANALYSIS")" -eq 11669 ]

[ "$(wc -l < "$BASE_PE_PDS")" -eq 68917 ]


K1_SHA="$(
    sha256sum "$K1_FROZEN" \
    | awk '{print $1}'
)"

echo "FROZEN_K1_SHA=$K1_SHA"

[ "$K1_SHA" = \
"7d8c5c1de8249db19e7e881174ef103c9ab6ebfa590623b6ef3190283684ed18" ]


python - "$PDS_FINAL" "$SMOKE_REPORT" <<'PY'
import json
import sys

pds = json.load(
    open(
        sys.argv[1],
        encoding="utf-8",
    )
)

smoke = json.load(
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

print(
    "FROZEN_PE_PDS_MACRO_BLEU =",
    pds_macro,
)

if abs(
    pds_macro
    -
    18.56962769521186
) > 1e-9:
    raise RuntimeError(
        "PE+PDS frozen baseline mismatch"
    )

if (
    smoke.get(
        "decision"
    )
    !=
    "IMPLEMENTATION_PASS"
):
    raise RuntimeError(
        "WA smoke did not authorize Full WA"
    )

print(
    "WA_SMOKE_AUTHORIZATION = PASS"
)

print(
    "FROZEN_BASELINES_PASS"
)
PY


python -m py_compile "$HELPER"

echo "PREFLIGHT_PASS"


###############################################################################
# STAGE 2 — FULL WORD ANALOGY GENERATION
###############################################################################

echo
echo "======================================================================"
echo "STAGE 2/9 — FULL WORD ANALOGY GENERATION"
echo "======================================================================"

python -u "$HELPER" \
    build-analog-jobs \
    --k1 "$K1_WA" \
    --analysis "$ANALYSIS" \
    --official "$OFFICIAL" \
    --output "$ANALOG_JOBS" \
    --report "$ANCHOR_REPORT" \
    --sample-size 0 \
    --seed 20260831

ANALOG_EXPECTED="$(
    wc -l < "$ANALOG_JOBS"
)"

echo "FULL_WA_ANCHOR_JOBS=$ANALOG_EXPECTED"

# Frozen eligible universe established by Smoke128 V2.
[ "$ANALOG_EXPECTED" -eq 11669 ]

echo "$ANALOG_EXPECTED" \
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
# STAGE 3 — PARSE 2 CATEGORY + 2 SEMANTIC
###############################################################################

echo
echo "======================================================================"
echo "STAGE 3/9 — PARSE FULL WORD ANALOGIES"
echo "======================================================================"

python -u "$HELPER" \
    parse-analogs \
    --jobs "$ANALOG_JOBS" \
    --shards "$ANALOG_SHARDS" \
    --output "$ANALOG_PAIRS" \
    --report "$ANALOG_REPORT"

ANALOG_PAIR_N="$(
    wc -l < "$ANALOG_PAIRS"
)"

echo "FULL_WA_VALID_ANALOG_PAIRS=$ANALOG_PAIR_N"

[ "$ANALOG_PAIR_N" -gt 0 ]


###############################################################################
# STAGE 4 — ONE CONTEXT PER ANALOG
###############################################################################

echo
echo "======================================================================"
echo "STAGE 4/9 — FULL WA CONTEXT GENERATION"
echo "======================================================================"

python -u "$HELPER" \
    build-context-jobs \
    --analogs "$ANALOG_PAIRS" \
    --official "$OFFICIAL" \
    --output "$CONTEXT_JOBS"

CONTEXT_EXPECTED="$(
    wc -l < "$CONTEXT_JOBS"
)"

echo "FULL_WA_CONTEXT_JOBS=$CONTEXT_EXPECTED"

[ "$CONTEXT_EXPECTED" -gt 0 ]

echo "$CONTEXT_EXPECTED" \
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
# STAGE 5 — SAME FROZEN STRUCTURAL ACCEPTANCE
###############################################################################

echo
echo "======================================================================"
echo "STAGE 5/9 — FULL WA STRUCTURAL ACCEPTANCE"
echo "======================================================================"

python -u "$HELPER" \
    accept-contexts \
    --jobs "$CONTEXT_JOBS" \
    --shards "$CONTEXT_SHARDS" \
    --parser "$PARSER" \
    --accepted "$WA_ACCEPTED" \
    --sft "$WA_SFT_RAW" \
    --report "$WA_CONTEXT_REPORT"

WA_ACCEPTED_N="$(
    wc -l < "$WA_SFT_RAW"
)"

echo "FULL_WA_ACCEPTED_ROWS=$WA_ACCEPTED_N"

[ "$WA_ACCEPTED_N" -gt 0 ]


python -u "$HELPER" \
    validate-sft \
    --sft "$WA_SFT_RAW" \
    --model "$STUDENT"


###############################################################################
# STAGE 6 — BUILD PE + PDS + WA
###############################################################################

echo
echo "======================================================================"
echo "STAGE 6/9 — BUILD FINAL PE+PDS+WA TRAIN DATA"
echo "======================================================================"

python - \
    "$BASE_PE_PDS" \
    "$WA_SFT_RAW" \
    "$WA_SFT" \
    "$TRAIN_DATA" \
    "$DATA_MANIFEST" <<'PY'

import copy
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path


BASE = Path(sys.argv[1])
WA_RAW = Path(sys.argv[2])
WA_OUT = Path(sys.argv[3])
TRAIN = Path(sys.argv[4])
MANIFEST = Path(sys.argv[5])


def load(path):
    rows = []

    with path.open(
        encoding="utf-8-sig",
        errors="replace",
    ) as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def dump(path, rows):
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                +
                "\n"
            )


def sha(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            b = f.read(
                1024 * 1024
            )

            if not b:
                break

            h.update(b)

    return h.hexdigest()


base = load(BASE)
wa = load(WA_RAW)


if len(base) != 68917:
    raise RuntimeError(
        f"frozen PE+PDS expected68917 got={len(base)}"
    )


###############################################################################
# Metadata-only normalization.
# NO scientific row filtering.
###############################################################################

normalized_wa = []

for i, row in enumerate(wa):

    new = copy.deepcopy(
        row
    )

    new[
        "index"
    ] = (
        f"strong_wa_full_v1_{i}"
    )

    new[
        "construction_method"
    ] = (
        "STRONG_REPRO_WA_FULL_V1"
    )

    new[
        "rq3_data_component"
    ] = "WA"

    normalized_wa.append(
        new
    )


dump(
    WA_OUT,
    normalized_wa,
)


###############################################################################
# Diagnostics ONLY.
# We deliberately DO NOT deduplicate or filter based on them.
###############################################################################

base_pairs = Counter(
    (
        str(x["source"]).strip(),
        str(
            x[
                "target_translation"
            ]
        ).strip(),
    )
    for x in base
)

wa_pairs = Counter(
    (
        str(x["source"]).strip(),
        str(
            x[
                "target_translation"
            ]
        ).strip(),
    )
    for x in normalized_wa
)


exact_overlap_base_occurrences = sum(
    count
    for pair, count
    in wa_pairs.items()
    if pair in base_pairs
)


exact_duplicate_wa_occurrences = sum(
    count - 1
    for count
    in wa_pairs.values()
    if count > 1
)


combined = (
    copy.deepcopy(base)
    +
    copy.deepcopy(
        normalized_wa
    )
)


rng = random.Random(
    20260831
)

rng.shuffle(
    combined
)


dump(
    TRAIN,
    combined,
)


manifest = {
    "protocol":
        "STRONG_REPRO_PE_PDS_WA_FULL_V1",

    "primary_estimand":
        (
            "BLEU(PE+PDS+WA) "
            "- BLEU(frozen PE+PDS)"
        ),

    "frozen_PE_PDS_rows":
        len(base),

    "WA_rows":
        len(
            normalized_wa
        ),

    "PE_PDS_WA_total_rows":
        len(combined),

    "WA_specification":
        {
            "anchor":
                (
                    "first canonical local error "
                    "per PE parent"
                ),

            "category_analogs":
                2,

            "semantic_analogs":
                2,

            "contexts_per_analog":
                1,
        },

    "construct_caveat":
        (
            "Feedback-derived P<->Q is not always "
            "a clean analogy-compatible unit; "
            "ordinary phrase/topic-level drift is retained."
        ),

    "filtering":
        (
            "same structural acceptance as Smoke128; "
            "no semantic cleaner, no length filter, "
            "no post-audit treatment modification"
        ),

    "diagnostic_only": {
        "exact_WA_pair_overlap_with_PE_PDS_occurrences":
            exact_overlap_base_occurrences,

        "exact_duplicate_WA_occurrences":
            exact_duplicate_wa_occurrences,
    },

    "row_efficiency_claim_supported":
        False,

    "sha256": {
        "frozen_PE_PDS":
            sha(BASE),

        "WA":
            sha(WA_OUT),

        "PE_PDS_WA":
            sha(TRAIN),
    },
}


MANIFEST.write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print(
    "FROZEN_PE_PDS_ROWS =",
    len(base),
)

print(
    "FULL_WA_ROWS =",
    len(
        normalized_wa
    ),
)

print(
    "PE_PDS_WA_TOTAL_ROWS =",
    len(combined),
)

print(
    "WA_EXACT_OVERLAP_BASE_DIAGNOSTIC =",
    exact_overlap_base_occurrences,
)

print(
    "WA_EXACT_DUPLICATES_DIAGNOSTIC =",
    exact_duplicate_wa_occurrences,
)

print(
    "ROW_EFFICIENCY_CLAIM_SUPPORTED = False"
)

print(
    "BUILD_FULL_PE_PDS_WA_PASS"
)
PY


TRAIN_ROWS="$(
    wc -l < "$TRAIN_DATA"
)"

echo "FINAL_TRAIN_ROWS=$TRAIN_ROWS"

[ "$TRAIN_ROWS" -gt 68917 ]


###############################################################################
# STAGE 7 — STUDENT FROM SAME BASE
###############################################################################

echo
echo "======================================================================"
echo "STAGE 7/9 — TRAIN PE+PDS+WA STUDENT FROM BASE"
echo "======================================================================"

mkdir -p \
    "$RUN_DIR" \
    "$RUN_DIR/eval"

date +%s \
    > "$OUT/student_start_epoch.txt"

export ASCEND_RT_VISIBLE_DEVICES=0

python -u "$TRAINER" \
    --model "$STUDENT" \
    --train "$TRAIN_DATA" \
    --output-dir "$RUN_DIR" \
    --lr 2e-5 \
    --epochs 3 \
    --batch-size 4 \
    --grad-accum 4 \
    --max-length 1024 \
    --warmup-ratio 0.03 \
    --weight-decay 0.01 \
    --max-grad-norm 1.0 \
    --seed 20260820 \
    --num-workers 2 \
    > "$OUT/student_train.log" \
    2>&1

[ -d "$RUN_DIR/epoch3" ]

date +%s \
    > "$OUT/student_finish_epoch.txt"

echo "STUDENT_TRAIN_PASS"


###############################################################################
# STAGE 8 — EVAL + SCIENTIFIC RESULT
###############################################################################

echo
echo "======================================================================"
echo "STAGE 8/9 — WMT24 / FLORES / CHALLENGE"
echo "======================================================================"

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


    PRED="$RUN_DIR/eval/${NAME}.pred.jsonl"
    METRIC="$RUN_DIR/eval/${NAME}.metrics.json"


    python -u "$EVAL" \
        --model "$RUN_DIR/epoch3" \
        --tokenizer "$RUN_DIR/epoch3" \
        --input "$INPUT" \
        --output "$PRED" \
        --method "strong_repro_pe_pds_wa_full_v1_${NAME}" \
        --batch-size 16 \
        --max-new-tokens 256 \
        --attn-implementation sdpa \
        > "$OUT/eval_${NAME}.log" \
        2>&1


    python "$SCORE" \
        --input "$PRED" \
        --output "$METRIC"


    echo
    echo "===== $NAME ====="
    cat "$METRIC"
    echo

done


###############################################################################
# Summary uses EXACT frozen PE+PDS result file.
###############################################################################

python - \
    "$RUN_DIR" \
    "$PDS_FINAL" \
    "$DATA_MANIFEST" \
    "$FINAL" <<'PY'

import json
import sys
from pathlib import Path


RUN = Path(sys.argv[1])
PDS_FILE = Path(sys.argv[2])
MANIFEST_FILE = Path(sys.argv[3])
OUT = Path(sys.argv[4])


SETS = (
    "wmt24",
    "flores",
    "challenge",
)


current = {}

for name in SETS:

    metric = json.loads(
        (
            RUN
            /
            "eval"
            /
            f"{name}.metrics.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    current[name] = {
        "BLEU":
            float(
                metric["BLEU"]
            ),

        "chrF":
            float(
                metric["chrF"]
            ),
    }


current[
    "macro_BLEU"
] = sum(
    current[x]["BLEU"]
    for x in SETS
) / 3


current[
    "macro_chrF"
] = sum(
    current[x]["chrF"]
    for x in SETS
) / 3


pds_file = json.loads(
    PDS_FILE.read_text(
        encoding="utf-8"
    )
)

pds = pds_file[
    "K1ALL_PDS"
]


delta_bleu = {
    name:
        current[name]["BLEU"]
        -
        float(
            pds[name]["BLEU"]
        )
    for name in SETS
}


delta_chrf = {
    name:
        current[name]["chrF"]
        -
        float(
            pds[name]["chrF"]
        )
    for name in SETS
}


delta_macro_bleu = (
    current[
        "macro_BLEU"
    ]
    -
    float(
        pds[
            "macro_BLEU"
        ]
    )
)


delta_macro_chrf = (
    current[
        "macro_chrF"
    ]
    -
    float(
        pds[
            "macro_chrF"
        ]
    )
)


BASE = 17.3485217
FULL_SEQKD = 19.1652


full_gap = (
    current[
        "macro_BLEU"
    ]
    -
    FULL_SEQKD
)


recovery = (
    current[
        "macro_BLEU"
    ]
    -
    BASE
) / (
    FULL_SEQKD
    -
    BASE
)


positive_count = sum(
    1
    for x
    in delta_bleu.values()
    if x > 0
)


manifest = json.loads(
    MANIFEST_FILE.read_text(
        encoding="utf-8"
    )
)


result = {
    "PE_PDS_WA": current,

    "frozen_PE_PDS": {
        name: pds[name]
        for name in (
            "wmt24",
            "flores",
            "challenge",
            "macro_BLEU",
            "macro_chrF",
        )
    },

    "delta_WA": {
        "BLEU":
            delta_bleu,

        "macro_BLEU":
            delta_macro_bleu,

        "chrF":
            delta_chrf,

        "macro_chrF":
            delta_macro_chrf,
    },

    "full_seqkd_comparison": {
        "FullSeqKD20k_macro_BLEU":
            FULL_SEQKD,

        "PE_PDS_WA_minus_FullSeqKD20k":
            full_gap,

        "gain_recovery_fraction":
            recovery,
    },

    "directionality": {
        "positive_BLEU_benchmarks":
            positive_count,

        "all_BLEU_nonnegative":
            all(
                x >= 0
                for x
                in delta_bleu.values()
            ),

        "macro_chrF_positive":
            delta_macro_chrf > 0,
    },

    "training_rows": {
        "PE_PDS":
            manifest[
                "frozen_PE_PDS_rows"
            ],

        "WA":
            manifest[
                "WA_rows"
            ],

        "PE_PDS_WA":
            manifest[
                "PE_PDS_WA_total_rows"
            ],
    },

    "reviewer_note":
        (
            "Do not infer final verdict from macro alone. "
            "Read WMT24, FLORES, Challenge, macro BLEU, "
            "macro chrF, FullSeqKD gap and recovery jointly."
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
    "=" * 92
)

print(
    "FINAL FULL WA SCIENTIFIC RESULT"
)

print(
    "=" * 92
)

print(
    f"{'SET':12s}"
    f"{'PDS':>12s}"
    f"{'PDS+WA':>12s}"
    f"{'DELTA':>12s}"
)


for name in SETS:

    print(
        f"{name:12s}"
        f"{float(pds[name]['BLEU']):12.4f}"
        f"{current[name]['BLEU']:12.4f}"
        f"{delta_bleu[name]:+12.4f}"
    )


print()

print(
    "PDS_MACRO_BLEU =",
    float(
        pds[
            "macro_BLEU"
        ]
    ),
)

print(
    "PE_PDS_WA_MACRO_BLEU =",
    current[
        "macro_BLEU"
    ],
)

print(
    "PRIMARY_DELTA_WA_MACRO_BLEU =",
    delta_macro_bleu,
)

print(
    "PRIMARY_DELTA_WA_MACRO_CHRF =",
    delta_macro_chrf,
)

print(
    "POSITIVE_BLEU_BENCHMARKS =",
    positive_count,
)

print(
    "GAP_TO_FULLSEQKD20K =",
    full_gap,
)

print(
    "FULLSEQKD_GAIN_RECOVERY =",
    recovery,
)

print(
    "FINAL_FULL_WA_RESULT_PASS"
)
PY


###############################################################################
# STAGE 9 — FREEZE
###############################################################################

echo
echo "======================================================================"
echo "STAGE 9/9 — FREEZE"
echo "======================================================================"

sha256sum \
    "$K1_FROZEN" \
    "$K1_WA" \
    "$ANALYSIS" \
    "$BASE_PE_PDS" \
    "$ANALOG_JOBS" \
    "$ANALOG_PAIRS" \
    "$CONTEXT_JOBS" \
    "$WA_ACCEPTED" \
    "$WA_SFT" \
    "$TRAIN_DATA" \
    "$DATA_MANIFEST" \
    "$FINAL" \
    > "$OUT/final_sha256_full_v1.txt"


touch "$PASS"

rm -f "$FAIL"

END_EPOCH="$(date +%s)"


echo
echo "======================================================================"
echo "STRONG REPRO FULL WA V1 PASS"
echo "======================================================================"
echo "TOTAL_ELAPSED_SECONDS=$((END_EPOCH - START_EPOCH))"
echo "FINISH_CST=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
echo "FINISH_UTC=$(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
echo "FINAL=$FINAL"
echo "PASS=$PASS"
echo "======================================================================"
