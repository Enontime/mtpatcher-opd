#!/usr/bin/env bash
set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

SCRIPT_DIR="$ROOT/scripts/mtpatcher_v11"
EXP_DATA="$DATA_ROOT/$EXP"
EXP_LOG="$LOG_ROOT/$EXP"

PE="$EXP_DATA/pe_k1_clean3732.jsonl"

BUILDER="$SCRIPT_DIR/build_pds_jobs_v11.py"
RUNNER="$SCRIPT_DIR/run_rq3_pds_generation_v11.sh"

JOBS="$EXP_DATA/rq3_pds_jobs_v11.jsonl"
JOB_AUDIT="$EXP_DATA/rq3_pds_jobs_audit_v11.json"

PDS_DIR="$EXP_DATA/rq3_pds_v11"
PDS_VALID="$EXP_DATA/rq3_pds_valid_v11.jsonl"
PDS_AUDIT="$EXP_DATA/rq3_pds_audit_v11.json"

PE_PDS="$EXP_DATA/rq3_pe_plus_pds_v11.jsonl"
PE_PDS_AUDIT="$EXP_DATA/rq3_pe_plus_pds_audit_v11.json"

MASTER_LOG="$EXP_LOG/rq3_pds_generation_v11.log"

STAMP="$(date +%Y%m%d_%H%M%S)"

echo "======================================================================"
echo "STAGE 1/6 — VERIFY FAILED RUN IS NOT ACTIVE"
echo "======================================================================"

if pgrep -f 'generate_pds_qwen3_8b_v11.py' >/dev/null 2>&1; then
    echo "ACTIVE_PDS_WORKER_FOUND"
    echo "Refusing duplicate launch."
    pgrep -af 'generate_pds_qwen3_8b_v11.py' || true
    false
fi

echo "NO_ACTIVE_PDS_WORKERS"

echo
echo "======================================================================"
echo "STAGE 2/6 — PRESERVE FAILED STATE"
echo "======================================================================"

if [ -f "$BUILDER" ]; then
    cp -a \
        "$BUILDER" \
        "${BUILDER}.before_guardfix_${STAMP}"
    echo "BUILDER_BACKUP=${BUILDER}.before_guardfix_${STAMP}"
fi

if [ -f "$MASTER_LOG" ]; then
    mv \
        "$MASTER_LOG" \
        "${MASTER_LOG}.failed_guard_${STAMP}"
    echo "FAILED_LOG_PRESERVED=${MASTER_LOG}.failed_guard_${STAMP}"
fi

if [ -d "$PDS_DIR" ]; then
    mv \
        "$PDS_DIR" \
        "${PDS_DIR}.before_guardfix_${STAMP}"
    echo "PDS_DIR_PRESERVED=${PDS_DIR}.before_guardfix_${STAMP}"
fi

for F in \
    "$JOBS" \
    "$JOB_AUDIT" \
    "$PDS_VALID" \
    "$PDS_AUDIT" \
    "$PE_PDS" \
    "$PE_PDS_AUDIT"
do
    if [ -e "$F" ]; then
        mv "$F" "${F}.before_guardfix_${STAMP}"
        echo "PRESERVED=$F"
    fi
done

echo
echo "======================================================================"
echo "STAGE 3/6 — INSTALL ROBUST PDS JOB BUILDER"
echo "======================================================================"

cat > "$BUILDER" <<'PY'
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def clean_text(x):
    if not isinstance(x, str):
        return ""
    return x.strip()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--pe",
        required=True
    )

    ap.add_argument(
        "--output",
        required=True
    )

    ap.add_argument(
        "--audit",
        default=None
    )

    ap.add_argument(
        "--repeat",
        type=int,
        default=4
    )

    args = ap.parse_args()

    pe_path = Path(args.pe)
    output_path = Path(args.output)

    if args.audit:
        audit_path = Path(args.audit)
    else:
        audit_path = (
            output_path.parent
            / "rq3_pds_jobs_audit_v11.json"
        )

    rows = []

    with pe_path.open(
        encoding="utf-8"
    ) as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    if len(rows) != 3732:
        raise RuntimeError(
            f"Expected frozen PE3732 input, "
            f"got {len(rows)}"
        )

    jobs = []

    total_error_records = 0
    valid_error_pairs = 0

    malformed_non_dict = 0
    missing_source_span = 0
    missing_correction = 0

    span_exact = 0
    span_not_exact = 0

    multi_error_rows = 0

    error_type_counts = Counter()

    span_mismatch_examples = []
    malformed_examples = []

    job_id = 0

    for row_pos, row in enumerate(rows):
        index = row.get("index")

        source = clean_text(
            row.get("source")
        )

        student_translation = clean_text(
            row.get("student_translation")
        )

        errors = row.get(
            "feedback_errors"
        )

        if not isinstance(errors, list):
            raise RuntimeError(
                "feedback_errors is not a list: "
                f"row_pos={row_pos}, "
                f"index={index}"
            )

        if not errors:
            raise RuntimeError(
                "PE row has empty feedback_errors: "
                f"row_pos={row_pos}, "
                f"index={index}"
            )

        if len(errors) > 1:
            multi_error_rows += 1

        for error_index, err in enumerate(errors):
            total_error_records += 1

            if not isinstance(err, dict):
                malformed_non_dict += 1

                if len(malformed_examples) < 20:
                    malformed_examples.append(
                        {
                            "reason":
                                "non_dict_error",
                            "row_pos":
                                row_pos,
                            "index":
                                index,
                            "error_index":
                                error_index,
                            "value":
                                repr(err)[:1000],
                        }
                    )

                continue

            source_span = clean_text(
                err.get("source_span")
            )

            correction = clean_text(
                err.get("correction")
            )

            translation_span = clean_text(
                err.get("translation_span")
            )

            error_type = clean_text(
                err.get("error_type")
            )

            explanation = clean_text(
                err.get("explanation")
            )

            if not source_span:
                missing_source_span += 1

                if len(malformed_examples) < 20:
                    malformed_examples.append(
                        {
                            "reason":
                                "missing_source_span",
                            "row_pos":
                                row_pos,
                            "index":
                                index,
                            "error_index":
                                error_index,
                            "error":
                                err,
                        }
                    )

                continue

            if not correction:
                missing_correction += 1

                if len(malformed_examples) < 20:
                    malformed_examples.append(
                        {
                            "reason":
                                "missing_correction",
                            "row_pos":
                                row_pos,
                            "index":
                                index,
                            "error_index":
                                error_index,
                            "error":
                                err,
                        }
                    )

                continue

            exact_in_parent = (
                source_span in source
            )

            if exact_in_parent:
                span_exact += 1
            else:
                span_not_exact += 1

                if len(span_mismatch_examples) < 30:
                    span_mismatch_examples.append(
                        {
                            "row_pos":
                                row_pos,
                            "index":
                                index,
                            "error_index":
                                error_index,
                            "source":
                                source,
                            "source_span":
                                source_span,
                            "correction":
                                correction,
                            "error_type":
                                error_type,
                        }
                    )

            valid_error_pairs += 1

            error_type_counts[
                error_type
            ] += 1

            for slot in range(args.repeat):
                jobs.append(
                    {
                        "job_id":
                            job_id,

                        "parent_row_pos":
                            row_pos,

                        "parent_index":
                            index,

                        "error_index":
                            error_index,

                        "pds_slot":
                            slot,

                        "source":
                            source,

                        "student_translation":
                            student_translation,

                        "source_span":
                            source_span,

                        "source_span_exact_in_parent":
                            exact_in_parent,

                        "translation_span":
                            translation_span,

                        "correction":
                            correction,

                        "error_type":
                            error_type,

                        "explanation":
                            explanation,

                        "construction_method":
                            "MT_PATCHER_PDS_QWEN3_8B_V11",
                    }
                )

                job_id += 1

    malformed_total = (
        malformed_non_dict
        + missing_source_span
        + missing_correction
    )

    if total_error_records == 0:
        raise RuntimeError(
            "No feedback errors found"
        )

    malformed_ratio = (
        malformed_total
        / total_error_records
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    audit_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as f:
        for job in jobs:
            f.write(
                json.dumps(
                    job,
                    ensure_ascii=False
                )
                + "\n"
            )

    expected_jobs = (
        valid_error_pairs
        * args.repeat
    )

    if len(jobs) != expected_jobs:
        raise RuntimeError(
            "PDS cardinality invariant failed: "
            f"jobs={len(jobs)}, "
            f"expected={expected_jobs}"
        )

    audit = {
        "pe_rows":
            len(rows),

        "total_error_records":
            total_error_records,

        "valid_error_pairs":
            valid_error_pairs,

        "malformed_total":
            malformed_total,

        "malformed_ratio":
            malformed_ratio,

        "malformed_non_dict":
            malformed_non_dict,

        "missing_source_span":
            missing_source_span,

        "missing_correction":
            missing_correction,

        "source_span_exact_in_parent":
            span_exact,

        "source_span_not_exact_in_parent":
            span_not_exact,

        "source_span_exact_ratio":
            (
                span_exact
                / valid_error_pairs
                if valid_error_pairs
                else 0.0
            ),

        "multi_error_rows":
            multi_error_rows,

        "pds_repeat":
            args.repeat,

        "pds_total_jobs":
            len(jobs),

        "error_type_counts":
            dict(error_type_counts),

        "span_mismatch_examples":
            span_mismatch_examples,

        "malformed_examples":
            malformed_examples,

        "pe_sha256":
            sha256(pe_path),

        "jobs_sha256":
            sha256(output_path),

        "method":
            "MT_PATCHER_PDS_QWEN3_8B_V11",

        "guard_policy":
            {
                "source_span_exact_match":
                    "audited_warning_only",

                "missing_source_span":
                    "skip_and_audit",

                "missing_correction":
                    "skip_and_audit",

                "non_dict_error":
                    "skip_and_audit",
            },
    }

    with audit_path.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            audit,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        "PE_ROWS =",
        len(rows)
    )

    print(
        "TOTAL_ERROR_RECORDS =",
        total_error_records
    )

    print(
        "VALID_ERROR_PAIRS =",
        valid_error_pairs
    )

    print(
        "MALFORMED_TOTAL =",
        malformed_total
    )

    print(
        "MALFORMED_RATIO =",
        f"{malformed_ratio:.8f}"
    )

    print(
        "SOURCE_SPAN_EXACT_IN_PARENT =",
        span_exact
    )

    print(
        "SOURCE_SPAN_NOT_EXACT_IN_PARENT =",
        span_not_exact
    )

    print(
        "MULTI_ERROR_ROWS =",
        multi_error_rows
    )

    print(
        "PDS_REPEAT =",
        args.repeat
    )

    print(
        "PDS_TOTAL_JOBS =",
        len(jobs)
    )

    print(
        "ERROR_TYPE_COUNTS =",
        dict(error_type_counts)
    )

    print(
        "PE_SHA256 =",
        sha256(pe_path)
    )

    print(
        "JOBS_SHA256 =",
        sha256(output_path)
    )

    print(
        "JOB_AUDIT =",
        str(audit_path)
    )

    if malformed_ratio > 0.01:
        raise RuntimeError(
            "More than 1% of feedback errors "
            "lack a usable bilingual pair. "
            "Scientific audit required before "
            "PDS generation."
        )

    if valid_error_pairs == 0:
        raise RuntimeError(
            "No valid bilingual error pairs"
        )

    print(
        "PDS_JOB_SCHEMA_AUDIT_PASS"
    )

    if span_not_exact:
        print(
            "PDS_SOURCE_SPAN_MISMATCH_WARNING "
            f"count={span_not_exact}"
        )

    print(
        "PDS_JOB_BUILD_PASS"
    )


if __name__ == "__main__":
    main()
PY

python -m py_compile "$BUILDER"

echo "PDS_BUILDER_COMPILE_PASS"

echo
echo "======================================================================"
echo "STAGE 4/6 — RUN JOB PRECHECK"
echo "======================================================================"

python "$BUILDER" \
    --pe "$PE" \
    --output "$JOBS" \
    --audit "$JOB_AUDIT" \
    --repeat 4

echo
echo "===== JOB AUDIT ====="
cat "$JOB_AUDIT"

echo
echo "PDS_JOB_PREFLIGHT_PASS"

echo
echo "======================================================================"
echo "STAGE 5/6 — VERIFY DOWNSTREAM PIPELINE"
echo "======================================================================"

if [ ! -f "$RUNNER" ]; then
    echo "MISSING_RUNNER=$RUNNER"
    false
fi

python -m py_compile \
    "$SCRIPT_DIR/generate_pds_qwen3_8b_v11.py" \
    "$SCRIPT_DIR/merge_and_audit_pds_v11.py" \
    "$SCRIPT_DIR/build_pe_plus_pds_v11.py"

echo "PDS_DOWNSTREAM_COMPILE_PASS"

echo
echo "======================================================================"
echo "STAGE 6/6 — RELAUNCH DETACHED PIPELINE"
echo "======================================================================"

nohup setsid bash "$RUNNER" \
    > "$MASTER_LOG" 2>&1 < /dev/null &

PID="$!"

echo "RQ3_PDS_V11_RESTARTED"
echo "PID=$PID"
echo "LOG=$MASTER_LOG"

sleep 12

echo
echo "========== PROCESS =========="
pgrep -af \
'rq3_pds_generation_v11|generate_pds_qwen3_8b_v11' \
|| true

echo
echo "========== MASTER LOG =========="
tail -n 120 "$MASTER_LOG" || true

echo
echo "RQ3_PDS_V11_RELAUNCH_HEALTHCHECK_DONE"
