#!/usr/bin/env bash

set -u

source /workspace/mtpatcher/project_env.sh

EXP="mtpatcher_v3_full6565_20260823"

DIR="$ROOT/scripts/mtpatcher_paper_faithful_v2"
CORE="$DIR/paper_repro_v2.py"
EVALCORE="$DIR/paper_repro_eval512_v1.py"

OFFICIAL="$ROOT/vendor/MT-Patcher-official"

OUT="$DATA_ROOT/$EXP/patcher_core_overnight_v1"
LOGDIR="$LOG_ROOT/$EXP/patcher_core_overnight_v1"

HELD="$DATA_ROOT/$EXP/patcher_quality_gate_v1"

FORMAL="$DATA_ROOT/$EXP/paperfaith_paper20k_v2"

CKPTROOT="$RUN_ROOT/$EXP/patcher_qwen3_8b_paperfaith_fullft_v2"

ZERO="$MODEL_ROOT/Qwen3-8B"
E1="$CKPTROOT/checkpoint-1584"
E2="$CKPTROOT/checkpoint-3168"
E3="$CKPTROOT/checkpoint-4752"

mkdir -p "$OUT" "$LOGDIR" "$OUT/shards"

export PYTHONPATH="$OFFICIAL:${PYTHONPATH:-}"

echo "============================================================"
echo "MT-PATCHER CORE OVERNIGHT V1"
echo "============================================================"
echo "PURPOSE=core-mechanism diagnostic"
echo "MAX_NEW_TOKENS=512"
echo "FIDELITY=ADAPTATION: evaluation-only output-length diagnostic"
echo "ZERO=$ZERO"
echo "E1=$E1"
echo "E2=$E2"
echo "E3=$E3"
echo

###############################################################################
# Helpers
###############################################################################

line_count () {
    local F="$1"

    if [ -f "$F" ]; then
        wc -l < "$F"
    else
        echo 0
    fi
}


rewrite512 () {
    local SRC="$1"
    local DST="$2"

    export SRC DST

    python - <<'PY'
import json
import os
from pathlib import Path

src = Path(os.environ["SRC"])
dst = Path(os.environ["DST"])

rows = []

with src.open(
    encoding="utf-8",
) as f:
    for line in f:
        if not line.strip():
            continue

        x = json.loads(line)

        x["max_new_tokens"] = 512

        rows.append(x)

with dst.open(
    "w",
    encoding="utf-8",
) as f:
    for x in rows:
        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )

print(
    "REWRITE512_ROWS=",
    len(rows),
)

print(
    "REWRITE512_OUTPUT=",
    dst,
)
PY
}


merge16 () {
    local PREFIX="$1"
    local MERGED="$2"
    local EXPECT="$3"

    export PREFIX MERGED EXPECT OUT

    python - <<'PY'
import json
import os
from pathlib import Path

prefix = os.environ["PREFIX"]
merged = Path(os.environ["MERGED"])
expect = int(os.environ["EXPECT"])
out = Path(os.environ["OUT"])

rows = []

for d in range(16):
    p = (
        out
        / "shards"
        / f"{prefix}_{d}.jsonl"
    )

    if not p.exists():
        raise RuntimeError(
            f"missing shard {p}"
        )

    with p.open(
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )


def jid(x):
    return int(
        x.get(
            "job_id",
            x.get(
                "demo_id",
                x.get("id"),
            ),
        )
    )


rows.sort(
    key=jid
)

ids = [
    jid(x)
    for x in rows
]

if len(rows) != expect:
    raise RuntimeError(
        f"{prefix}: "
        f"expected={expect} "
        f"got={len(rows)}"
    )

if len(ids) != len(set(ids)):
    raise RuntimeError(
        f"{prefix}: duplicate ids"
    )

with merged.open(
    "w",
    encoding="utf-8",
) as f:
    for x in rows:
        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )

print(
    f"MERGED prefix={prefix} "
    f"rows={len(rows)}"
)

print(
    f"MERGED_FILE={merged}"
)
PY
}


run16 () {
    local PREFIX="$1"
    local JOBS="$2"
    local MODEL="$3"
    local EXPECT="$4"
    local BATCH="$5"

    local MERGED="$OUT/${PREFIX}.jsonl"

    if [ -f "$MERGED" ] \
        && [ "$(line_count "$MERGED")" -eq "$EXPECT" ]
    then
        echo
        echo "ALREADY_PASS prefix=$PREFIX rows=$EXPECT"
        return 0
    fi

    echo
    echo "============================================================"
    echo "RUN16 prefix=$PREFIX"
    echo "jobs=$EXPECT"
    echo "model=$MODEL"
    echo "============================================================"

    local PIDS=()
    local BAD=0

    for D in $(seq 0 15); do
        python -u "$EVALCORE" generate \
            --jobs "$JOBS" \
            --output "$OUT/shards/${PREFIX}_${D}.jsonl" \
            --model "$MODEL" \
            --device "$D" \
            --world-size 16 \
            --batch-size "$BATCH" \
            > "$LOGDIR/${PREFIX}_${D}.log" 2>&1 &

        PIDS+=("$!")
    done

    for PID in "${PIDS[@]}"; do
        if ! wait "$PID"; then
            BAD=1
        fi
    done

    if [ "$BAD" -ne 0 ]; then
        echo "RUN16_FAIL prefix=$PREFIX"

        for D in $(seq 0 15); do
            echo "---------- device $D ----------"

            tail -n 15 \
                "$LOGDIR/${PREFIX}_${D}.log" \
                2>/dev/null || true
        done

        return 1
    fi

    if ! merge16 \
        "$PREFIX" \
        "$MERGED" \
        "$EXPECT"
    then
        echo "MERGE_FAIL prefix=$PREFIX"
        return 1
    fi

    echo "RUN16_PASS prefix=$PREFIX"
    return 0
}


safe_run16 () {
    local PREFIX="$1"
    local JOBS="$2"
    local MODEL="$3"
    local EXPECT="$4"
    local BATCH="$5"

    if run16 \
        "$PREFIX" \
        "$JOBS" \
        "$MODEL" \
        "$EXPECT" \
        "$BATCH"
    then
        touch "$OUT/${PREFIX}.PASS"
    else
        touch "$OUT/${PREFIX}.FAIL"

        echo
        echo "CONTINUE_AFTER_FAILURE prefix=$PREFIX"
    fi
}


###############################################################################
# PREFLIGHT
###############################################################################

echo
echo "========== PREFLIGHT =========="

for P in \
    "$EVALCORE" \
    "$OFFICIAL" \
    "$ZERO" \
    "$E1" \
    "$E2" \
    "$E3" \
    "$HELD/feedback_jobs.jsonl" \
    "$HELD/student.jsonl" \
    "$FORMAL/feedback_jobs.jsonl" \
    "$FORMAL/feedback.jsonl"
do
    if [ ! -e "$P" ]; then
        echo "MISSING=$P"
    else
        echo "FOUND=$P"
    fi
done

echo "OVERNIGHT_PREFLIGHT_DONE"


###############################################################################
# A. HELD-OUT 2207 — highest priority
###############################################################################

echo
echo "============================================================"
echo "STAGE A: HELDOUT2207 512-TOKEN CORE GATE"
echo "============================================================"

rewrite512 \
    "$HELD/feedback_jobs.jsonl" \
    "$OUT/held2207_feedback_jobs_512.jsonl"

safe_run16 \
    "held2207_zero512" \
    "$OUT/held2207_feedback_jobs_512.jsonl" \
    "$ZERO" \
    2207 \
    8

safe_run16 \
    "held2207_e1_512" \
    "$OUT/held2207_feedback_jobs_512.jsonl" \
    "$E1" \
    2207 \
    8

safe_run16 \
    "held2207_e2_512" \
    "$OUT/held2207_feedback_jobs_512.jsonl" \
    "$E2" \
    2207 \
    8

safe_run16 \
    "held2207_e3_512" \
    "$OUT/held2207_feedback_jobs_512.jsonl" \
    "$E3" \
    2207 \
    8

touch "$OUT/STAGE_A_DONE"


###############################################################################
# B. CURATED 6565 — apples-to-apples old branch diagnostic
###############################################################################

echo
echo "============================================================"
echo "STAGE B: CURATED6565 CORE GATE"
echo "============================================================"

CURATED_SRC="$DATA_ROOT/$EXP/patch_pool6565_generation.jsonl"
CURATED_COMPAT="$OUT/curated6565_student_compat.jsonl"
CURATED_JOBS="$OUT/curated6565_feedback_jobs.jsonl"
CURATED_JOBS512="$OUT/curated6565_feedback_jobs_512.jsonl"

export CURATED_SRC CURATED_COMPAT

python - <<'PY'
import json
import os
from pathlib import Path

src = Path(
    os.environ["CURATED_SRC"]
)

dst = Path(
    os.environ["CURATED_COMPAT"]
)

if not src.exists():
    raise RuntimeError(
        f"missing curated generation: {src}"
    )

rows = []

with src.open(
    encoding="utf-8",
) as f:
    for line in f:
        if not line.strip():
            continue

        x = json.loads(line)

        source = None

        for k in [
            "source",
            "src",
            "zh",
            "input",
        ]:
            if (
                k in x
                and str(x[k]).strip()
            ):
                source = str(
                    x[k]
                ).strip()
                break

        response = None

        for k in [
            "response",
            "student_translation",
            "prediction",
            "output",
            "translation",
        ]:
            if (
                k in x
                and str(x[k]).strip()
            ):
                response = str(
                    x[k]
                ).strip()
                break

        if source is None:
            raise RuntimeError(
                f"source missing: "
                f"keys={list(x.keys())}"
            )

        if response is None:
            raise RuntimeError(
                f"student response missing: "
                f"keys={list(x.keys())}"
            )

        y = dict(x)

        y["source"] = source
        y["response"] = response

        rows.append(y)

if len(rows) != 6565:
    raise RuntimeError(
        f"expected curated6565, "
        f"got {len(rows)}"
    )

with dst.open(
    "w",
    encoding="utf-8",
) as f:
    for x in rows:
        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )

print(
    "CURATED_COMPAT_ROWS=",
    len(rows),
)

print(
    "CURATED_COMPAT_PASS"
)
PY

if python "$CORE" feedback-jobs \
    --official "$OFFICIAL" \
    --student "$CURATED_COMPAT" \
    --output "$CURATED_JOBS"
then
    echo \
    "CURATED_FEEDBACK_JOBS=$(wc -l < "$CURATED_JOBS")"

    rewrite512 \
        "$CURATED_JOBS" \
        "$CURATED_JOBS512"

    safe_run16 \
        "curated6565_zero512" \
        "$CURATED_JOBS512" \
        "$ZERO" \
        6565 \
        8

    safe_run16 \
        "curated6565_e1_512" \
        "$CURATED_JOBS512" \
        "$E1" \
        6565 \
        8

    safe_run16 \
        "curated6565_e2_512" \
        "$CURATED_JOBS512" \
        "$E2" \
        6565 \
        8

    safe_run16 \
        "curated6565_e3_512" \
        "$CURATED_JOBS512" \
        "$E3" \
        6565 \
        8

    touch "$OUT/STAGE_B_DONE"
else
    echo "CURATED_FEEDBACK_JOB_BUILD_FAIL"
    touch "$OUT/STAGE_B.FAIL"
fi


###############################################################################
# C. TRAIN-DEMO REPLAY — filler with real scientific value
#
# This distinguishes:
#   train imitation success
#       vs
#   held-out generalization/calibration failure
###############################################################################

echo
echo "============================================================"
echo "STAGE C: FORMAL20K TRAIN-DEMO REPLAY"
echo "============================================================"

rewrite512 \
    "$FORMAL/feedback_jobs.jsonl" \
    "$OUT/train20k_feedback_jobs_512.jsonl"

safe_run16 \
    "train20k_e1_512" \
    "$OUT/train20k_feedback_jobs_512.jsonl" \
    "$E1" \
    20000 \
    8

safe_run16 \
    "train20k_e2_512" \
    "$OUT/train20k_feedback_jobs_512.jsonl" \
    "$E2" \
    20000 \
    8

safe_run16 \
    "train20k_e3_512" \
    "$OUT/train20k_feedback_jobs_512.jsonl" \
    "$E3" \
    20000 \
    8

touch "$OUT/STAGE_C_DONE"


###############################################################################
# D. OFFLINE SUMMARY
###############################################################################

echo
echo "============================================================"
echo "STAGE D: OFFLINE SUMMARY"
echo "============================================================"

export OUT HELD FORMAL EXP

python - <<'PY'
import json
import os
import re
from pathlib import Path

import sacrebleu

out = Path(
    os.environ["OUT"]
)

held = Path(
    os.environ["HELD"]
)

formal = Path(
    os.environ["FORMAL"]
)


def load(path):
    with path.open(
        encoding="utf-8"
    ) as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def jid(x):
    return int(
        x.get(
            "job_id",
            x.get(
                "demo_id",
                x.get("id"),
            ),
        )
    )


def response(x):
    for k in [
        "response",
        "generated_text",
        "output",
        "prediction",
    ]:
        if k in x:
            return str(
                x[k]
            ).strip()

    return ""


def student_text(x):
    for k in [
        "student_translation",
        "response",
        "prediction",
        "output",
    ]:
        if (
            k in x
            and str(x[k]).strip()
        ):
            return str(
                x[k]
            ).strip()

    return ""


def raw_feedback(raw, draft):
    text = raw.strip()

    has_no = bool(
        re.search(
            r"(?i)\bNo\s+error\.?\b",
            text,
        )
    )

    explicit_error = bool(
        re.search(
            r"(?i)"
            r"(Error\s*(?:Type|Details|\d+)"
            r"|Chinese Segment"
            r"|Correct Translation)",
            text,
        )
    )

    if (
        has_no
        and not explicit_error
    ):
        return {
            "has_error": False,
            "post_edit": draft,
            "extract_ok": True,
            "route": "no_error",
        }

    patterns = [
        r"(?is)"
        r"\*\*Good Translation\*\*\s*:?\s*(.+?)\s*$",

        r"(?is)"
        r"Good Translation\s*:?\s*(.+?)\s*$",

        r"(?is)"
        r"Good translation\s*:?\s*(.+?)\s*$",

        r"(?is)"
        r"\*\*Final Translation\*\*\s*:?\s*(.+?)\s*$",

        r"(?is)"
        r"Final Translation\s*:?\s*(.+?)\s*$",
    ]

    candidate = None

    for pattern in patterns:
        m = re.search(
            pattern,
            text,
        )

        if m:
            candidate = (
                m.group(1).strip()
            )
            break

    if candidate:
        candidate = re.sub(
            r"^\s*[-*]\s*",
            "",
            candidate,
        ).strip()

        return {
            "has_error": True,
            "post_edit": candidate,
            "extract_ok": True,
            "route": "post_edit",
        }

    return {
        "has_error": True,
        "post_edit": draft,
        "extract_ok": False,
        "route": "fallback_draft",
    }


def mt_metrics(rows, key):
    hyp = [
        x[key]
        for x in rows
    ]

    ref = [
        x["reference"]
        for x in rows
    ]

    return {
        "bleu":
            sacrebleu.corpus_bleu(
                hyp,
                [ref],
            ).score,

        "chrf":
            sacrebleu.corpus_chrf(
                hyp,
                [ref],
            ).score,
    }


summary = {
    "protocol":
        "PATCHER_CORE_OVERNIGHT_V1",

    "max_new_tokens":
        512,

    "fidelity":
        "ADAPTATION: evaluation-only output-length diagnostic",

    "held2207":
        {},

    "curated6565":
        {},

    "train20k_replay":
        {},
}


###############################################################################
# HELD2207
###############################################################################

held_rows = load(
    held / "student.jsonl"
)

held_by_id = {
    int(x["job_id"]): x
    for x in held_rows
}

echo_ids = set()

echo_file = (
    held
    / "student_prompt_echo_ids.json"
)

if echo_file.exists():
    obj = json.loads(
        echo_file.read_text(
            encoding="utf-8"
        )
    )

    echo_ids = {
        int(x["job_id"])
        for x in obj.get(
            "rows",
            [],
        )
    }


held_systems = {
    "zero":
        out / "held2207_zero512.jsonl",

    "epoch1":
        out / "held2207_e1_512.jsonl",

    "epoch2":
        out / "held2207_e2_512.jsonl",

    "epoch3":
        out / "held2207_e3_512.jsonl",
}


for name, path in held_systems.items():
    if not path.exists():
        summary[
            "held2207"
        ][name] = {
            "status": "missing"
        }
        continue

    raw = {
        jid(x): x
        for x in load(path)
    }

    rows = []

    for i in sorted(
        held_by_id
    ):
        if i not in raw:
            continue

        base = held_by_id[i]

        draft = (
            base[
                "student_translation"
            ]
        )

        parsed = raw_feedback(
            response(raw[i]),
            draft,
        )

        rows.append({
            **base,

            "post_edit":
                parsed[
                    "post_edit"
                ],

            "has_error":
                parsed[
                    "has_error"
                ],

            "extract_ok":
                parsed[
                    "extract_ok"
                ],
        })

    metrics_all = mt_metrics(
        rows,
        "post_edit",
    )

    clean = [
        x for x in rows
        if int(
            x["job_id"]
        ) not in echo_ids
    ]

    metrics_clean = (
        mt_metrics(
            clean,
            "post_edit",
        )
        if clean
        else None
    )

    summary[
        "held2207"
    ][name] = {
        "rows":
            len(rows),

        "has_error_rate":
            sum(
                x["has_error"]
                for x in rows
            )
            / len(rows),

        "extract_rate":
            sum(
                x["extract_ok"]
                for x in rows
            )
            / len(rows),

        "metrics_all":
            metrics_all,

        "prompt_echo_excluded":
            len(rows)
            - len(clean),

        "metrics_no_prompt_echo":
            metrics_clean,
    }


###############################################################################
# CURATED6565
###############################################################################

curated_student = (
    out
    / "curated6565_student_compat.jsonl"
)

human_candidates = list(
    Path(
        os.environ[
            "DATA_ROOT"
        ]
    ).rglob(
        "human_train6565.jsonl"
    )
)

if (
    curated_student.exists()
    and human_candidates
):
    student_rows = load(
        curated_student
    )

    human_rows = load(
        human_candidates[0]
    )


    def pick_source(x):
        for k in [
            "source",
            "src",
            "zh",
            "input",
        ]:
            if (
                k in x
                and str(
                    x[k]
                ).strip()
            ):
                return str(
                    x[k]
                ).strip()

        return None


    def pick_ref(x):
        for k in [
            "reference",
            "target",
            "target_translation",
            "translation",
            "tgt",
            "english",
            "en",
        ]:
            if (
                k in x
                and str(
                    x[k]
                ).strip()
            ):
                return str(
                    x[k]
                ).strip()

        return None


    ref_map = {}

    for x in human_rows:
        s = pick_source(x)
        r = pick_ref(x)

        if (
            s is not None
            and r is not None
            and s not in ref_map
        ):
            ref_map[s] = r

    student_by_id = {
        jid(x): x
        for x in student_rows
    }

    systems = {
        "zero":
            out
            / "curated6565_zero512.jsonl",

        "epoch1":
            out
            / "curated6565_e1_512.jsonl",

        "epoch2":
            out
            / "curated6565_e2_512.jsonl",

        "epoch3":
            out
            / "curated6565_e3_512.jsonl",
    }

    for name, path in systems.items():
        if not path.exists():
            summary[
                "curated6565"
            ][name] = {
                "status": "missing"
            }
            continue

        raw = {
            jid(x): x
            for x in load(path)
        }

        scored = []

        all_extract = []
        all_error = []

        for i, base in student_by_id.items():
            if i not in raw:
                continue

            draft = student_text(
                base
            )

            parsed = raw_feedback(
                response(raw[i]),
                draft,
            )

            all_extract.append(
                parsed[
                    "extract_ok"
                ]
            )

            all_error.append(
                parsed[
                    "has_error"
                ]
            )

            src = pick_source(
                base
            )

            ref = ref_map.get(
                src
            )

            if ref is None:
                continue

            scored.append({
                "reference":
                    ref,

                "post_edit":
                    parsed[
                        "post_edit"
                    ],

                "student":
                    draft,
            })

        d = {
            "generated_rows":
                len(raw),

            "reference_matched_rows":
                len(scored),

            "extract_rate":
                (
                    sum(
                        all_extract
                    )
                    / len(
                        all_extract
                    )
                ),

            "has_error_rate":
                (
                    sum(
                        all_error
                    )
                    / len(
                        all_error
                    )
                ),
        }

        if scored:
            d[
                "post_edit"
            ] = mt_metrics(
                scored,
                "post_edit",
            )

            d[
                "student"
            ] = mt_metrics(
                scored,
                "student",
            )

        summary[
            "curated6565"
        ][name] = d


###############################################################################
# TRAIN20K REPLAY
###############################################################################

target_rows = load(
    formal / "feedback.jsonl"
)

targets = {
    jid(x): x
    for x in target_rows
}


def is_error_label(raw):
    parsed = raw_feedback(
        raw,
        "",
    )

    return (
        parsed[
            "has_error"
        ]
    )


train_systems = {
    "epoch1":
        out / "train20k_e1_512.jsonl",

    "epoch2":
        out / "train20k_e2_512.jsonl",

    "epoch3":
        out / "train20k_e3_512.jsonl",
}


for name, path in train_systems.items():
    if not path.exists():
        summary[
            "train20k_replay"
        ][name] = {
            "status": "missing"
        }
        continue

    pred_rows = load(
        path
    )

    pred = {
        jid(x): x
        for x in pred_rows
    }

    ids = sorted(
        set(targets)
        & set(pred)
    )

    target_text = [
        response(
            targets[i]
        )
        for i in ids
    ]

    pred_text = [
        response(
            pred[i]
        )
        for i in ids
    ]

    raw_chrf = (
        sacrebleu.corpus_chrf(
            pred_text,
            [target_text],
        ).score
    )

    label_agree = sum(
        is_error_label(
            response(
                pred[i]
            )
        )
        ==
        is_error_label(
            response(
                targets[i]
            )
        )
        for i in ids
    )

    target_extract = []
    pred_extract = []

    for i in ids:
        target_extract.append(
            raw_feedback(
                response(
                    targets[i]
                ),
                "",
            )[
                "extract_ok"
            ]
        )

        pred_extract.append(
            raw_feedback(
                response(
                    pred[i]
                ),
                "",
            )[
                "extract_ok"
            ]
        )

    summary[
        "train20k_replay"
    ][name] = {
        "rows":
            len(ids),

        "raw_feedback_chrf_to_training_target":
            raw_chrf,

        "has_error_label_agreement":
            label_agree
            / len(ids),

        "training_target_extract_rate":
            sum(
                target_extract
            )
            / len(
                target_extract
            ),

        "pred_extract_rate":
            sum(
                pred_extract
            )
            / len(
                pred_extract
            ),
    }


###############################################################################
# SAVE + PRINT
###############################################################################

summary_path = (
    out
    / "overnight_summary.json"
)

summary_path.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print()
print(
    "============================================================"
)
print(
    "MT-PATCHER CORE OVERNIGHT SUMMARY"
)
print(
    "============================================================"
)

print()
print("HELD2207")

for name, d in summary[
    "held2207"
].items():
    print()
    print(name)
    print(
        json.dumps(
            d,
            ensure_ascii=False,
            indent=2,
        )
    )

print()
print("CURATED6565")

for name, d in summary[
    "curated6565"
].items():
    print()
    print(name)
    print(
        json.dumps(
            d,
            ensure_ascii=False,
            indent=2,
        )
    )

print()
print("TRAIN20K_REPLAY")

for name, d in summary[
    "train20k_replay"
].items():
    print()
    print(name)
    print(
        json.dumps(
            d,
            ensure_ascii=False,
            indent=2,
        )
    )

print()
print(
    "SUMMARY_FILE=",
    summary_path,
)

print(
    "PATCHER_CORE_OVERNIGHT_V1_SUMMARY_PASS"
)
PY

touch "$OUT/STAGE_D_DONE"

echo
echo "============================================================"
echo "FINAL"
echo "============================================================"

echo "PATCHER_CORE_OVERNIGHT_V1_ALL_DONE"
