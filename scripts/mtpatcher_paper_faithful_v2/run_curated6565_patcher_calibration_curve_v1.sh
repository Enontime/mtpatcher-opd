#!/usr/bin/env bash
set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export PYTHONPATH="/workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend/vendor/MT-Patcher-official:${PYTHONPATH:-}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

PIPE="$ROOT/scripts/mtpatcher_paper_faithful_v2/paper_repro_v2.py"
OFFICIAL="$ROOT/vendor/MT-Patcher-official"

STUDENT_PRED="$RUN_ROOT/$EXP/student_base_full6565/predictions.jsonl"

BASE8="$MODEL_ROOT/Qwen3-8B"
PATCHER_ROOT="$RUN_ROOT/$EXP/patcher_qwen3_8b_paperfaith_fullft_v2"

E1="$PATCHER_ROOT/checkpoint-1584"
E2="$PATCHER_ROOT/checkpoint-3168"
E3="$PATCHER_ROOT/checkpoint-4752"

OUT="$DATA_ROOT/$EXP/curated6565_patcher_calibration_curve_v1"
LOGDIR="$LOG_ROOT/$EXP/curated6565_patcher_calibration_curve_v1"

COMPAT="$OUT/student_compat6565.jsonl"
FEEDBACK_JOBS="$OUT/feedback_jobs6565.jsonl"
SUMMARY="$OUT/summary.json"

mkdir -p "$OUT" "$LOGDIR"

echo "========================================================================"
echo "CURATED6565 PATCHER CALIBRATION CURVE V1"
echo "========================================================================"
echo "SCIENTIFIC_QUESTION:"
echo "Does paper-faithful Patcher specialization calibrate error selection"
echo "and improve corrections on the actual curated6565 Student drafts?"
echo
echo "POOL=6565"
echo "Student=Qwen3-0.6B base drafts"
echo "arms=zero,e1,e2,e3"
echo "Feedback max_new_tokens=256"
echo "Feedback temperature=official 0.1"
echo "Structured parser=ENGINEERING_ADAPTATION"
echo "No Student training will be launched."
echo

# ----------------------------------------------------------------------
# PRECHECK
# ----------------------------------------------------------------------

for F in \
    "$PIPE" \
    "$STUDENT_PRED"
do
    if [ ! -f "$F" ]; then
        echo "MISSING_FILE=$F"
        false
    fi
done

for D in \
    "$OFFICIAL" \
    "$BASE8" \
    "$E1" \
    "$E2" \
    "$E3"
do
    if [ ! -d "$D" ]; then
        echo "MISSING_DIR=$D"
        false
    fi
done

echo "PIPE_SHA256=$(sha256sum "$PIPE" | awk '{print $1}')"
echo "STUDENT_PRED_SHA256=$(sha256sum "$STUDENT_PRED" | awk '{print $1}')"
echo "PRECHECK_FILES_PASS"

# ----------------------------------------------------------------------
# MATERIALIZE LEAKAGE-SAFE STUDENT INPUT
#
# Important:
# reference is deliberately NOT copied into COMPAT.
# Feedbacker only sees source + Student translation.
# ----------------------------------------------------------------------

export STUDENT_PRED COMPAT

python - <<'PY'
import json
import os
from pathlib import Path

src = Path(os.environ["STUDENT_PRED"])
out = Path(os.environ["COMPAT"])

rows = [
    json.loads(x)
    for x in src.read_text(encoding="utf-8").splitlines()
    if x.strip()
]

rows.sort(key=lambda x: int(x["index"]))

if len(rows) != 6565:
    raise RuntimeError(f"expected 6565 Student predictions, got {len(rows)}")

indices = [int(x["index"]) for x in rows]
if indices != list(range(6565)):
    raise RuntimeError("Student indices are not exactly 0..6564")

compat = []

for x in rows:
    source = str(x["source"]).strip()
    response = str(x["student_translation"]).strip()

    if not source:
        raise RuntimeError(f"empty source index={x['index']}")
    if not response:
        raise RuntimeError(f"empty student translation index={x['index']}")

    compat.append(
        {
            "demo_id": int(x["index"]),
            "source": source,
            "response": response,
        }
    )

with out.open("w", encoding="utf-8") as f:
    for x in compat:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

print("COMPAT_ROWS =", len(compat))
print("COMPAT_KEYS =", sorted(compat[0]))
print("REFERENCE_PRESENT =", any("reference" in x for x in compat))

if any("reference" in x for x in compat):
    raise RuntimeError("reference leakage into Feedbacker input")

print("CURATED6565_COMPAT_PASS")
PY

# ----------------------------------------------------------------------
# BUILD OFFICIAL-PROMPT FEEDBACK JOBS
# ----------------------------------------------------------------------

python "$PIPE" feedback-jobs \
    --official "$OFFICIAL" \
    --student "$COMPAT" \
    --output "$FEEDBACK_JOBS"

python - "$FEEDBACK_JOBS" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
rows = [
    json.loads(x)
    for x in p.read_text(encoding="utf-8").splitlines()
    if x.strip()
]

assert len(rows) == 6565, len(rows)
assert [int(x["job_id"]) for x in rows] == list(range(6565))

for x in rows:
    assert int(x["max_new_tokens"]) == 256
    assert abs(float(x["temperature"]) - 0.1) < 1e-9

print("FEEDBACK_JOB_AUDIT_PASS")
print("rows =", len(rows))
print("temperature =", rows[0]["temperature"])
print("max_new_tokens =", rows[0]["max_new_tokens"])
PY

# ----------------------------------------------------------------------
# 16-NPU GENERATION
# ----------------------------------------------------------------------

run16 () {
    local PREFIX="$1"
    local JOBS="$2"
    local MODEL="$3"
    local BATCH="$4"

    local SD="$OUT/shards/$PREFIX"
    mkdir -p "$SD"

    echo
    echo "========================================================================"
    echo "RUN16_START arm=$PREFIX"
    echo "model=$MODEL"
    echo "jobs=$JOBS"
    echo "========================================================================"

    local pids=()

    for D in $(seq 0 15); do
        python -u "$PIPE" generate \
            --jobs "$JOBS" \
            --output "$SD/device_${D}.jsonl" \
            --model "$MODEL" \
            --device "$D" \
            --world-size 16 \
            --batch-size "$BATCH" \
            > "$LOGDIR/${PREFIX}_device${D}.log" 2>&1 &

        pids+=("$!")
    done

    local failed=0

    for P in "${pids[@]}"; do
        if ! wait "$P"; then
            failed=1
        fi
    done

    if [ "$failed" -ne 0 ]; then
        echo "RUN16_FAILED arm=$PREFIX"
        for F in "$LOGDIR/${PREFIX}_device"*.log; do
            echo "-------- $F --------"
            tail -n 30 "$F" || true
        done
        false
    fi

    echo "RUN16_PASS arm=$PREFIX"
}

merge16 () {
    local JOBS="$1"
    local PREFIX="$2"
    local OUTPUT="$3"

    export MERGE_JOBS="$JOBS"
    export MERGE_DIR="$OUT/shards/$PREFIX"
    export MERGE_OUTPUT="$OUTPUT"

    python - <<'PY'
import json
import os
from pathlib import Path

jobs_path = Path(os.environ["MERGE_JOBS"])
shard_dir = Path(os.environ["MERGE_DIR"])
out_path = Path(os.environ["MERGE_OUTPUT"])

jobs = [
    json.loads(x)
    for x in jobs_path.read_text(encoding="utf-8").splitlines()
    if x.strip()
]

expected = {int(x["job_id"]) for x in jobs}

merged = {}

shards = sorted(shard_dir.glob("device_*.jsonl"))

if len(shards) != 16:
    raise RuntimeError(f"expected 16 shard files, got {len(shards)}")

for p in shards:
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        row = json.loads(line)
        jid = int(row["job_id"])

        if jid in merged:
            raise RuntimeError(f"duplicate job_id={jid}")

        merged[jid] = row

if set(merged) != expected:
    missing = sorted(expected - set(merged))
    extra = sorted(set(merged) - expected)

    raise RuntimeError(
        f"merge coverage mismatch "
        f"missing={missing[:20]} extra={extra[:20]}"
    )

ordered = [merged[j] for j in sorted(expected)]

with out_path.open("w", encoding="utf-8") as f:
    for row in ordered:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("MERGED_ROWS =", len(ordered))
print("MERGED_OUTPUT =", out_path)
print("MERGE16_PASS")
PY
}

# ----------------------------------------------------------------------
# ONE ARM = Feedback generation -> structured parsing -> validation
# ----------------------------------------------------------------------

run_arm () {
    local ARM="$1"
    local MODEL="$2"

    local RAW="$OUT/${ARM}_feedback.jsonl"
    local PJOBS="$OUT/${ARM}_parser_jobs.jsonl"
    local PRAW="$OUT/${ARM}_parser_raw.jsonl"
    local PARSED="$OUT/${ARM}_parsed.jsonl"

    run16 "${ARM}_feedback" "$FEEDBACK_JOBS" "$MODEL" 8
    merge16 "$FEEDBACK_JOBS" "${ARM}_feedback" "$RAW"

    python "$PIPE" parser-jobs \
        --feedback "$RAW" \
        --output "$PJOBS"

    run16 "${ARM}_parser" "$PJOBS" "$BASE8" 8
    merge16 "$PJOBS" "${ARM}_parser" "$PRAW"

    python "$PIPE" validate-parser \
        --parser "$PRAW" \
        --output "$PARSED"

    echo "ARM_COMPLETE=$ARM"
    echo "PARSED=$PARSED"
    echo
}

# ----------------------------------------------------------------------
# PRE-REGISTERED CALIBRATION CURVE
#
# Same 6565 Student drafts, same prompt, same decoder.
# Only Patcher checkpoint changes.
# ----------------------------------------------------------------------

run_arm "zero" "$BASE8"
run_arm "epoch1" "$E1"
run_arm "epoch2" "$E2"
run_arm "epoch3" "$E3"

# ----------------------------------------------------------------------
# SCORE CALIBRATION + CORRECTION PROXIES
#
# Human references enter ONLY here, after generation is frozen.
# chrF/BLEU here are diagnostics/proxies, not semantic ground truth.
# ----------------------------------------------------------------------

export OUT STUDENT_PRED SUMMARY BASE8 E1 E2 E3

python - <<'PY'
import json
import os
import statistics
from pathlib import Path

import sacrebleu

out = Path(os.environ["OUT"])
pred_path = Path(os.environ["STUDENT_PRED"])
summary_path = Path(os.environ["SUMMARY"])

pred_rows = [
    json.loads(x)
    for x in pred_path.read_text(encoding="utf-8").splitlines()
    if x.strip()
]

pred = {
    int(x["index"]): x
    for x in pred_rows
}

if len(pred) != 6565:
    raise RuntimeError(f"prediction map rows={len(pred)}")

arms = {
    "zero": os.environ["BASE8"],
    "epoch1": os.environ["E1"],
    "epoch2": os.environ["E2"],
    "epoch3": os.environ["E3"],
}

parsed = {}

def load_jsonl(path):
    return [
        json.loads(x)
        for x in Path(path).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]

def sent_chrf(hyp, ref):
    return sacrebleu.sentence_chrf(
        hyp,
        [ref],
    ).score

def corpus_metrics(hyps, refs):
    if not hyps:
        return {
            "rows": 0,
            "bleu": None,
            "chrf": None,
        }

    return {
        "rows": len(hyps),
        "bleu": sacrebleu.corpus_bleu(
            hyps,
            [refs],
        ).score,
        "chrf": sacrebleu.corpus_chrf(
            hyps,
            [refs],
        ).score,
    }

summary = {
    "protocol": "CURATED6565_PATCHER_CALIBRATION_CURVE_V1",
    "scientific_question": (
        "Does paper-faithful Patcher specialization calibrate error "
        "selection and improve corrections on the actual curated6565 "
        "Student drafts?"
    ),
    "pool_rows": 6565,
    "student_predictions": str(pred_path),
    "reference_usage": (
        "Hidden from Feedbacker/Patcher; used only in frozen post-generation "
        "diagnostic scoring."
    ),
    "primary_metrics": [
        "structured_parser_valid_rate",
        "selection_rate_total",
        "selection_rate_among_valid",
        "zero_vs_checkpoint_selection_overlap",
    ],
    "secondary_proxy_metrics": (
        "BLEU/chrF and sentence-chrF change of parsed post-edits vs hidden "
        "human references; diagnostic proxy only."
    ),
    "arms": {},
    "pairwise_vs_zero": {},
}

for arm, model in arms.items():
    p = out / f"{arm}_parsed.jsonl"
    audit_path = Path(str(p) + ".audit.json")

    rows = load_jsonl(p)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    by_id = {
        int(x["demo_id"]): x
        for x in rows
    }

    if len(by_id) != len(rows):
        raise RuntimeError(f"{arm}: duplicate parsed demo_id")

    parsed[arm] = by_id

    selected = {
        i
        for i, x in by_id.items()
        if bool(x["has_error"])
    }

    no_error = {
        i
        for i, x in by_id.items()
        if not bool(x["has_error"])
    }

    correction_ids = []

    for i in sorted(selected):
        x = by_id[i]
        post = str(x.get("post_edit", "")).strip()

        if post and i in pred:
            correction_ids.append(i)

    student_h = []
    post_h = []
    refs = []
    sent_delta = []

    for i in correction_ids:
        s = pred[i]["student_translation"]
        pedit = parsed[arm][i]["post_edit"]
        ref = pred[i]["reference"]

        student_h.append(s)
        post_h.append(pedit)
        refs.append(ref)

        sent_delta.append(
            sent_chrf(pedit, ref)
            - sent_chrf(s, ref)
        )

    student_metrics = corpus_metrics(
        student_h,
        refs,
    )

    post_metrics = corpus_metrics(
        post_h,
        refs,
    )

    proxy = {
        "rows_with_nonempty_post_edit":
            len(correction_ids),

        "student_on_same_rows":
            student_metrics,

        "post_edit_on_same_rows":
            post_metrics,

        "delta_bleu":
            (
                post_metrics["bleu"]
                - student_metrics["bleu"]
                if correction_ids
                else None
            ),

        "delta_chrf":
            (
                post_metrics["chrf"]
                - student_metrics["chrf"]
                if correction_ids
                else None
            ),

        "sentence_chrf_delta_mean":
            (
                sum(sent_delta) / len(sent_delta)
                if sent_delta
                else None
            ),

        "sentence_chrf_delta_median":
            (
                statistics.median(sent_delta)
                if sent_delta
                else None
            ),

        "sentence_chrf_positive_fraction":
            (
                sum(x > 0 for x in sent_delta) / len(sent_delta)
                if sent_delta
                else None
            ),
    }

    summary["arms"][arm] = {
        "model": model,

        "parser_audit": audit,

        "valid_rows": len(by_id),

        "selected_rows": len(selected),

        "no_error_rows": len(no_error),

        # Conservative wrt parser failures: invalid rows are not silently
        # treated as selected.
        "selection_rate_total":
            len(selected) / 6565,

        "selection_rate_among_valid":
            len(selected) / max(1, len(by_id)),

        "correction_quality_proxy":
            proxy,
    }

zero_valid = set(parsed["zero"])
zero_sel = {
    i
    for i, x in parsed["zero"].items()
    if bool(x["has_error"])
}

for arm in ("epoch1", "epoch2", "epoch3"):
    arm_valid = set(parsed[arm])
    arm_sel = {
        i
        for i, x in parsed[arm].items()
        if bool(x["has_error"])
    }

    common_valid = zero_valid & arm_valid

    label_agree = sum(
        bool(parsed["zero"][i]["has_error"])
        ==
        bool(parsed[arm][i]["has_error"])
        for i in common_valid
    )

    union_sel = zero_sel | arm_sel
    inter_sel = zero_sel & arm_sel

    summary["pairwise_vs_zero"][arm] = {
        "common_valid_rows":
            len(common_valid),

        "has_error_label_agreement":
            (
                label_agree / len(common_valid)
                if common_valid
                else None
            ),

        "zero_selected_rows":
            len(zero_sel),

        "arm_selected_rows":
            len(arm_sel),

        "selection_rate_total_shift":
            (
                len(arm_sel) - len(zero_sel)
            ) / 6565,

        "selection_intersection":
            len(inter_sel),

        "selection_jaccard":
            (
                len(inter_sel) / len(union_sel)
                if union_sel
                else 1.0
            ),

        "zero_error_to_arm_no_error":
            len(
                zero_sel
                & common_valid
                - arm_sel
            ),

        "zero_no_error_to_arm_error":
            len(
                arm_sel
                & common_valid
                - zero_sel
            ),
    }

summary_path.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print()
print("=" * 100)
print("CURATED6565 PATCHER CALIBRATION CURVE — FINAL")
print("=" * 100)

print(
    f"{'ARM':10s} "
    f"{'VALID':>7s} "
    f"{'SEL':>7s} "
    f"{'SEL/TOTAL':>11s} "
    f"{'SEL/VALID':>11s} "
    f"{'POST_N':>8s} "
    f"{'ΔBLEU':>10s} "
    f"{'ΔchrF':>10s} "
    f"{'sent+':>9s}"
)

for arm in ("zero", "epoch1", "epoch2", "epoch3"):
    x = summary["arms"][arm]
    p = x["correction_quality_proxy"]

    db = p["delta_bleu"]
    dc = p["delta_chrf"]
    pf = p["sentence_chrf_positive_fraction"]

    print(
        f"{arm:10s} "
        f"{x['valid_rows']:7d} "
        f"{x['selected_rows']:7d} "
        f"{x['selection_rate_total']:11.4%} "
        f"{x['selection_rate_among_valid']:11.4%} "
        f"{p['rows_with_nonempty_post_edit']:8d} "
        f"{db if db is not None else float('nan'):10.4f} "
        f"{dc if dc is not None else float('nan'):10.4f} "
        f"{pf if pf is not None else float('nan'):9.4f}"
    )

print()
print("PAIRWISE VS ZERO")

for arm in ("epoch1", "epoch2", "epoch3"):
    x = summary["pairwise_vs_zero"][arm]

    print(
        arm,
        "selection_shift=",
        f"{x['selection_rate_total_shift']:+.4%}",
        "jaccard=",
        f"{x['selection_jaccard']:.4f}",
        "label_agreement=",
        f"{x['has_error_label_agreement']:.4f}",
        "error->noerror=",
        x["zero_error_to_arm_no_error"],
        "noerror->error=",
        x["zero_no_error_to_arm_error"],
    )

print()
print("SUMMARY =", summary_path)
print("IMPORTANT: BLEU/chrF correction scores above are diagnostic proxies.")
print("No downstream Student training has been run.")
print("CURATED6565_PATCHER_CALIBRATION_CURVE_V1_PASS")
PY

echo
echo "========================================================================"
echo "OVERNIGHT EXPERIMENT FINISHED"
echo "========================================================================"
echo "SUMMARY=$SUMMARY"
echo "NO_FOLLOWUP_TRAINING_LAUNCHED"
echo "CURATED6565_PATCHER_CALIBRATION_CURVE_V1_ALL_DONE"
