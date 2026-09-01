#!/usr/bin/env bash
set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

N="${N:-20000}"
TAG="${TAG:-paper20k_v2}"

SCRIPT="$ROOT/scripts/mtpatcher_paper_faithful_v2/paper_repro_v2.py"

OFFICIAL="$ROOT/vendor/MT-Patcher-official"

# Required by the frozen official release:
# translation.py / analyzer / WA / PDS import pipeline.data_utils.
export PYTHONPATH="$OFFICIAL${PYTHONPATH:+:$PYTHONPATH}"

SOURCE="$DATA_ROOT/$EXP/rq0_newscrawl50k_sources_v1.jsonl"

STUDENT="$MODEL_ROOT/Qwen3-0.6B"
ANNOTATOR="$MODEL_ROOT/Qwen3-8B"

BASE="$DATA_ROOT/$EXP/paperfaith_${TAG}"
SHARDS="$BASE/shards"

LOGDIR="$LOG_ROOT/$EXP/paperfaith_${TAG}"

mkdir -p \
    "$BASE" \
    "$SHARDS" \
    "$LOGDIR"


run16 () {

    JOBS="$1"
    MODEL="$2"
    PREFIX="$3"
    BS="$4"

    COUNT="$(wc -l < "$JOBS")"

    echo \
        "RUN16 prefix=$PREFIX " \
        "jobs=$COUNT model=$MODEL"

    if [ "$COUNT" -eq 0 ]; then
        : > "$BASE/${PREFIX}.jsonl"
        echo "NO_JOBS prefix=$PREFIX"
        return
    fi

    PIDS=()

    for D in $(seq 0 15); do

        python -u "$SCRIPT" generate \
            --jobs "$JOBS" \
            --output "$SHARDS/${PREFIX}_${D}.jsonl" \
            --model "$MODEL" \
            --device "$D" \
            --world-size 16 \
            --batch-size "$BS" \
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

        echo \
            "GENERATION_FAILURE prefix=$PREFIX"

        for D in $(seq 0 15); do

            echo \
                "----- $PREFIX device=$D -----"

            tail -n 30 \
                "$LOGDIR/${PREFIX}_${D}.log" \
                2>/dev/null || true

        done

        false
    fi

    python "$SCRIPT" merge \
        --jobs "$JOBS" \
        --shard-dir "$SHARDS" \
        --prefix "$PREFIX" \
        --output "$BASE/${PREFIX}.jsonl"

}


echo "============================================================"
echo "MT-PATCHER PAPER-FAITHFUL DEMO PIPELINE V2"
echo "N=$N"
echo "TAG=$TAG"
echo "============================================================"

echo \
    "OFFICIAL_COMMIT="\
"$(git -C "$OFFICIAL" rev-parse HEAD)"


###############################################################################
# §3.3 — random monolingual sample
###############################################################################

python "$SCRIPT" sample \
    --input "$SOURCE" \
    --output "$BASE/demo_pool.jsonl" \
    --n "$N" \
    --seed 20260826


###############################################################################
# §3.3 — Student first translates X
# Official translation prompt imported verbatim.
###############################################################################

python "$SCRIPT" translation-jobs \
    --official "$OFFICIAL" \
    --pool "$BASE/demo_pool.jsonl" \
    --output "$BASE/translation_jobs.jsonl"

run16 \
    "$BASE/translation_jobs.jsonl" \
    "$STUDENT" \
    student \
    32


###############################################################################
# §3.3 item 1 / §3.1 — Feedback
###############################################################################

python "$SCRIPT" feedback-jobs \
    --official "$OFFICIAL" \
    --student "$BASE/student.jsonl" \
    --output "$BASE/feedback_jobs.jsonl"

run16 \
    "$BASE/feedback_jobs.jsonl" \
    "$ANNOTATOR" \
    feedback \
    8


###############################################################################
# Engineering adapter:
# raw Feedback -> model_assessment_parsed
#
# Raw feedback remains the actual demonstration target.
###############################################################################

python "$SCRIPT" parser-jobs \
    --feedback "$BASE/feedback.jsonl" \
    --output "$BASE/parser_jobs.jsonl"

run16 \
    "$BASE/parser_jobs.jsonl" \
    "$ANNOTATOR" \
    parser \
    16

python "$SCRIPT" validate-parser \
    --parser "$BASE/parser.jsonl" \
    --output "$BASE/feedback_parsed.jsonl"


###############################################################################
# §3.3 item 2 — Sentence Analyzer
###############################################################################

python "$SCRIPT" analysis-jobs \
    --official "$OFFICIAL" \
    --student "$BASE/student.jsonl" \
    --output "$BASE/analysis_jobs.jsonl"

run16 \
    "$BASE/analysis_jobs.jsonl" \
    "$ANNOTATOR" \
    analysis \
    16


###############################################################################
# §3.3 item 3 — Word Analogy
###############################################################################

python "$SCRIPT" analogy-jobs \
    --official "$OFFICIAL" \
    --parsed "$BASE/feedback_parsed.jsonl" \
    --output "$BASE/analogy_jobs.jsonl"

run16 \
    "$BASE/analogy_jobs.jsonl" \
    "$ANNOTATOR" \
    analogy \
    8


###############################################################################
# §3.3 item 4 — Parallel Data Synthesis
#
# Patcher demonstration construction:
# every parsed error × 1 demonstration pair.
#
# The release Step-4 patch-generation pipeline later uses
# num_case=4 contexts per error.
###############################################################################

python "$SCRIPT" pds-jobs \
    --official "$OFFICIAL" \
    --parsed "$BASE/feedback_parsed.jsonl" \
    --analysis "$BASE/analysis.jsonl" \
    --output "$BASE/pds_jobs.jsonl"

run16 \
    "$BASE/pds_jobs.jsonl" \
    "$ANNOTATOR" \
    pds \
    8


###############################################################################
# §3.3 — collect four demonstration types
###############################################################################

python "$SCRIPT" build-sft \
    --feedback "$BASE/feedback.jsonl" \
    --analysis "$BASE/analysis.jsonl" \
    --analogy "$BASE/analogy.jsonl" \
    --pds "$BASE/pds.jsonl" \
    --output "$BASE/patcher_sft.jsonl" \
    --seed 42


###############################################################################
# Frozen provenance manifest
###############################################################################

OFFICIAL_COMMIT="$(
    git -C "$OFFICIAL" rev-parse HEAD
)"

cat > "$BASE/provenance.txt" <<EOF
protocol=MT_PATCHER_PAPER_FAITHFUL_ADAPTED_V2
official_repo=https://github.com/NJUNLP/MT-Patcher
official_commit=$OFFICIAL_COMMIT

source_corpus_adaptation=WMT NewsCrawl 2023 zh
student_adaptation=Qwen3-0.6B
annotator_adaptation=Qwen3-8B replacing GPT-4
patcher_backbone_adaptation=Qwen3-8B replacing Baichuan2-13B
qwen_chat_template_adaptation=true

paper_demo_sample_n=$N
paper_demo_seed=20260826

pds_demo_policy=all_parsed_errors_x1
pds_demo_basis=paper_appendix_single_parallel_pair
pds_downstream_patch_policy=current_release_all_errors_x4
legacy_first_error_conflict=recorded_unresolved
EOF


echo
echo "============================================================"
echo "FINAL DEMO AUDIT"
echo "============================================================"

wc -l \
    "$BASE/demo_pool.jsonl" \
    "$BASE/student.jsonl" \
    "$BASE/feedback.jsonl" \
    "$BASE/feedback_parsed.jsonl" \
    "$BASE/analysis.jsonl" \
    "$BASE/analogy.jsonl" \
    "$BASE/pds.jsonl" \
    "$BASE/patcher_sft.jsonl"

echo

sha256sum \
    "$BASE/patcher_sft.jsonl"

echo

cat \
    "$BASE/patcher_sft.jsonl.manifest.json"

echo

cat \
    "$BASE/feedback_parsed.jsonl.audit.json"

echo

echo \
    "PAPER_FAITHFUL_DEMO_PIPELINE_V2_PASS"
