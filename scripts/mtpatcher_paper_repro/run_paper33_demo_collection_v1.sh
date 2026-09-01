#!/usr/bin/env bash
set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

DEMO_N="${DEMO_N:-20000}"
TAG="${TAG:-paper20k}"

STUDENT="$MODEL_ROOT/Qwen3-0.6B"

# ---------------------------------------------------------
# ADAPTATION:
# Paper: GPT-4 demonstration annotator
# Ours: Qwen3-8B because this is the strongest local model.
# This is written into manifest and must not be described
# later as an exact GPT-4 reproduction.
# ---------------------------------------------------------
ANNOTATOR="$MODEL_ROOT/Qwen3-8B"

SRC="$DATA_ROOT/$EXP/rq0_newscrawl50k_sources_v1.jsonl"

SCRIPT="$ROOT/scripts/mtpatcher_paper_repro/paper33_data.py"
GEN="$ROOT/scripts/mtpatcher_paper_repro/generate_jobs_npu_v1.py"

BASE="$DATA_ROOT/$EXP/paper33_${TAG}"
SHARDS="$BASE/shards"
LOGDIR="$LOG_ROOT/$EXP/paper33_${TAG}"

mkdir -p \
    "$BASE" \
    "$SHARDS" \
    "$LOGDIR"


run16 () {

    local JOBS="$1"
    local MODEL="$2"
    local PREFIX="$3"
    local BS="$4"
    local MAXNEW="$5"
    local SAMPLE="$6"

    PIDS=()

    for D in $(seq 0 15); do

        EXTRA=()

        if [ "$SAMPLE" = "1" ]; then
            EXTRA+=(
                --do-sample
                --temperature 0.7
            )
        fi

        python -u "$GEN" \
            --jobs "$JOBS" \
            --output "$SHARDS/${PREFIX}_${D}.jsonl" \
            --model "$MODEL" \
            --device "$D" \
            --world-size 16 \
            --batch-size "$BS" \
            --max-new-tokens "$MAXNEW" \
            "${EXTRA[@]}" \
            > "$LOGDIR/${PREFIX}_${D}.log" \
            2>&1 &

        PIDS+=("$!")

    done


    FAIL=0

    for P in "${PIDS[@]}"; do

        if ! wait "$P"; then
            FAIL=1
        fi

    done


    if [ "$FAIL" -ne 0 ]; then

        echo "GENERATION_FAILURE prefix=$PREFIX"

        for D in $(seq 0 15); do

            echo "----- device $D -----"

            tail -n 20 \
                "$LOGDIR/${PREFIX}_${D}.log" \
                2>/dev/null || true

        done

        false
    fi
}


echo "============================================================"
echo "MT-PATCHER PAPER §3.3 DEMONSTRATION CONSTRUCTION"
echo "N=$DEMO_N"
echo "TAG=$TAG"
echo "============================================================"


##############################################################################
# PAPER §3.3:
# "20,000 monolingual sentences randomly selected..."
##############################################################################

python "$SCRIPT" sample \
    --input "$SRC" \
    --output "$BASE/demo_pool.jsonl" \
    --n "$DEMO_N" \
    --seed 20260826


##############################################################################
# PAPER §3.3:
# "...use [student] to generate its translation..."
##############################################################################

python "$SCRIPT" student-jobs \
    --pool "$BASE/demo_pool.jsonl" \
    --output "$BASE/student_jobs.jsonl"

run16 \
    "$BASE/student_jobs.jsonl" \
    "$STUDENT" \
    "student" \
    32 \
    256 \
    0

python "$SCRIPT" merge \
    --shard-dir "$SHARDS" \
    --prefix student \
    --output "$BASE/student_translations.jsonl"


##############################################################################
# PAPER §3.3 item (1):
# feedback f given source X and Student translation Y.
#
# PAPER §3.1:
# f = c, {(s_i,e_i,t_i)}, p
##############################################################################

python "$SCRIPT" feedback-jobs \
    --student "$BASE/student_translations.jsonl" \
    --output "$BASE/feedback_jobs.jsonl"

run16 \
    "$BASE/feedback_jobs.jsonl" \
    "$ANNOTATOR" \
    "feedback" \
    8 \
    768 \
    0

python "$SCRIPT" merge \
    --shard-dir "$SHARDS" \
    --prefix feedback \
    --output "$BASE/feedback.jsonl"


##############################################################################
# PAPER §3.3 item (2):
# analyze domain/topic/style.
#
# PAPER §3.2:
# this is the information bottleneck before PDS.
##############################################################################

python "$SCRIPT" analysis-jobs \
    --student "$BASE/student_translations.jsonl" \
    --output "$BASE/analysis_jobs.jsonl"

run16 \
    "$BASE/analysis_jobs.jsonl" \
    "$ANNOTATOR" \
    "analysis" \
    16 \
    128 \
    0

python "$SCRIPT" merge \
    --shard-dir "$SHARDS" \
    --prefix analysis \
    --output "$BASE/analysis.jsonl"


##############################################################################
# PAPER §3.3 item (3):
# word analogy given X and erroneous source word s.
#
# PAPER §3.2:
# category + semantics/co-occurrence
# rare + challenging analogous words.
##############################################################################

python "$SCRIPT" analogy-jobs \
    --feedback "$BASE/feedback.jsonl" \
    --output "$BASE/analogy_jobs.jsonl"

run16 \
    "$BASE/analogy_jobs.jsonl" \
    "$ANNOTATOR" \
    "analogy" \
    8 \
    384 \
    0

python "$SCRIPT" merge \
    --shard-dir "$SHARDS" \
    --prefix analogy \
    --output "$BASE/analogy.jsonl"


##############################################################################
# PAPER §3.3 item (4):
# synthesize parallel sentence containing (s,c)
# with the same d/t/style.
##############################################################################

python "$SCRIPT" pds-jobs \
    --feedback "$BASE/feedback.jsonl" \
    --analysis "$BASE/analysis.jsonl" \
    --output "$BASE/pds_jobs.jsonl"

# Sampling is used only for synthesis diversity.
# Decoding parameters are not specified by the paper,
# so this is explicitly an implementation choice.

run16 \
    "$BASE/pds_jobs.jsonl" \
    "$ANNOTATOR" \
    "pds" \
    8 \
    384 \
    1

python "$SCRIPT" merge \
    --shard-dir "$SHARDS" \
    --prefix pds \
    --output "$BASE/pds.jsonl"


##############################################################################
# PAPER §3.3:
# collect four demonstration task types and use them
# to finetune the LLM into MT-PATCHER.
##############################################################################

python "$SCRIPT" build-sft \
    --feedback "$BASE/feedback.jsonl" \
    --analysis "$BASE/analysis.jsonl" \
    --analogy "$BASE/analogy.jsonl" \
    --pds "$BASE/pds.jsonl" \
    --output "$BASE/patcher_sft.jsonl"


echo
echo "============================================================"
echo "FINAL AUDIT"
echo "============================================================"

wc -l \
    "$BASE/demo_pool.jsonl" \
    "$BASE/student_translations.jsonl" \
    "$BASE/feedback.jsonl" \
    "$BASE/analysis.jsonl" \
    "$BASE/analogy.jsonl" \
    "$BASE/pds.jsonl" \
    "$BASE/patcher_sft.jsonl"

sha256sum \
    "$BASE/patcher_sft.jsonl"

cat \
    "$BASE/patcher_sft.jsonl.manifest.json"

echo
echo "PAPER33_DEMONSTRATION_COLLECTION_ALL_PASS"
