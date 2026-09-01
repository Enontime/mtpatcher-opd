#!/usr/bin/env bash
set -euo pipefail

source /workspace/mtpatcher/project_env.sh

EXP="mtpatcher_v3_full6565_20260823"

DIR="$ROOT/scripts/mtpatcher_paper_faithful_v2"
CORE="$DIR/paper_repro_v2.py"
OFFICIAL="$ROOT/vendor/MT-Patcher-official"

PATCHER="$RUN_ROOT/$EXP/patcher_qwen3_8b_paperfaith_fullft_v2/checkpoint-4752"
STUDENT="$MODEL_ROOT/Qwen3-0.6B"
TEACHER="$MODEL_ROOT/Qwen3-8B"

WORK="$DATA_ROOT/$EXP/patcher_quality_gate_v1"
LOGDIR="$LOG_ROOT/$EXP/patcher_quality_gate_v1"

mkdir -p "$WORK" "$LOGDIR" "$WORK/shards"

# Official MT-PATCHER modules import `pipeline.*` as a top-level package.
export PYTHONPATH="$OFFICIAL:${PYTHONPATH:-}"

echo "PYTHONPATH_OFFICIAL=$OFFICIAL"

###############################################################################
# PRECHECK
###############################################################################

echo "============================================================"
echo "SPECIALIZED MT-PATCHER QUALITY GATE V1"
echo "============================================================"

test -d "$PATCHER"
test -d "$STUDENT"
test -d "$TEACHER"
test -f "$CORE"
test -d "$OFFICIAL"

TEMPLATE="$DATA_ROOT/$EXP/paperfaith_paper20k_v2/feedback_jobs.jsonl"
test -f "$TEMPLATE"

find_eval () {
    local NAME="$1"

    find "$DATA_ROOT" \
        -type f \
        -name "$NAME" \
        -print \
        2>/dev/null \
        | head -n 1
}

WMT="$(find_eval wmt24_zh_en998.jsonl)"
FLORES="$(find_eval flores_zh_en1012.jsonl)"
CHALLENGE="$(find_eval challenge_zh_en197.jsonl)"

test -n "$WMT"
test -n "$FLORES"
test -n "$CHALLENGE"

test -f "$WMT"
test -f "$FLORES"
test -f "$CHALLENGE"

echo "WMT=$WMT"
echo "FLORES=$FLORES"
echo "CHALLENGE=$CHALLENGE"

python - <<'PY'
import sacrebleu
print("sacrebleu =", sacrebleu.__version__)
PY

echo "QUALITY_GATE_PREFLIGHT_PASS"


###############################################################################
# BUILD COMBINED HELD-OUT EVAL
###############################################################################

export WMT FLORES CHALLENGE WORK TEMPLATE

python - <<'PY'
import copy
import json
import os
from pathlib import Path

work = Path(os.environ["WORK"])

datasets = [
    ("wmt24", Path(os.environ["WMT"])),
    ("flores", Path(os.environ["FLORES"])),
    ("challenge", Path(os.environ["CHALLENGE"])),
]

source_keys = [
    "source",
    "src",
    "zh",
    "chinese",
    "input",
]

reference_keys = [
    "reference",
    "target",
    "target_translation",
    "translation",
    "tgt",
    "english",
    "en",
]


def load_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pick(row, keys, kind):
    for k in keys:
        if k in row and str(row[k]).strip():
            return str(row[k]).strip()

    raise RuntimeError(
        f"Cannot detect {kind}. keys={list(row.keys())}"
    )


combined = []

for dataset, path in datasets:
    rows = load_jsonl(path)

    print(
        f"DATASET {dataset} rows={len(rows)} "
        f"keys={list(rows[0].keys())}"
    )

    for local_id, row in enumerate(rows):
        combined.append({
            "job_id": len(combined),
            "demo_id": len(combined),
            "dataset": dataset,
            "local_id": local_id,
            "source": pick(
                row,
                source_keys,
                "source",
            ),
            "reference": pick(
                row,
                reference_keys,
                "reference",
            ),
        })

expected = {
    "wmt24": 998,
    "flores": 1012,
    "challenge": 197,
}

for name, n in expected.items():
    got = sum(
        x["dataset"] == name
        for x in combined
    )
    if got != n:
        raise RuntimeError(
            f"{name}: expected={n}, got={got}"
        )

if len(combined) != 2207:
    raise RuntimeError(
        f"combined rows != 2207: {len(combined)}"
    )

out = work / "eval2207.jsonl"

with out.open("w", encoding="utf-8") as f:
    for x in combined:
        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )

print("EVAL2207_ROWS =", len(combined))
print("EVAL2207_BUILD_PASS")
PY


###############################################################################
# BUILD STUDENT TRANSLATION JOBS
#
# Reuse an existing paper_repro_v2 job as schema template so generation
# stays compatible with the already-tested Ascend generator.
###############################################################################

python - <<'PY'
import copy
import json
import os
from pathlib import Path

work = Path(os.environ["WORK"])
template_path = Path(os.environ["TEMPLATE"])


def load_jsonl(p):
    with p.open(encoding="utf-8") as f:
        return [
            json.loads(x)
            for x in f
            if x.strip()
        ]


eval_rows = load_jsonl(
    work / "eval2207.jsonl"
)

template = load_jsonl(
    template_path
)[0]

out = work / "student_jobs.jsonl"

with out.open("w", encoding="utf-8") as f:
    for x in eval_rows:
        j = copy.deepcopy(template)

        j["job_id"] = x["job_id"]
        j["demo_id"] = x["demo_id"]
        j["source"] = x["source"]

        j["prompt"] = (
            "Translate the following sentences "
            "from Chinese to English.\n"
            f"Input: {x['source']}\n"
            "Output:"
        )

        if "temperature" in j:
            j["temperature"] = 0.0

        if "max_new_tokens" in j:
            j["max_new_tokens"] = 256

        j["gate_kind"] = "student_translation"
        j["dataset"] = x["dataset"]

        f.write(
            json.dumps(
                j,
                ensure_ascii=False,
            )
            + "\n"
        )

print(
    "STUDENT_JOBS=",
    len(eval_rows),
)
PY


###############################################################################
# GENERATION HELPERS
###############################################################################

run16 () {
    local PREFIX="$1"
    local JOBS="$2"
    local MODEL="$3"
    local BATCH="$4"

    echo
    echo "RUN16 prefix=$PREFIX model=$MODEL"

    local PIDS=()

    for D in $(seq 0 15); do
        python -u "$CORE" generate \
            --jobs "$JOBS" \
            --output "$WORK/shards/${PREFIX}_${D}.jsonl" \
            --model "$MODEL" \
            --device "$D" \
            --world-size 16 \
            --batch-size "$BATCH" \
            > "$LOGDIR/${PREFIX}_${D}.log" 2>&1 &

        PIDS+=("$!")
    done

    local BAD=0

    for PID in "${PIDS[@]}"; do
        if ! wait "$PID"; then
            BAD=1
        fi
    done

    if [ "$BAD" -ne 0 ]; then
        echo "RUN16_FAILED prefix=$PREFIX"

        for D in $(seq 0 15); do
            echo "---------- $PREFIX device $D ----------"
            tail -n 30 \
                "$LOGDIR/${PREFIX}_${D}.log" \
                2>/dev/null || true
        done

        return 1
    fi

    echo "RUN16_PASS prefix=$PREFIX"
}


merge16 () {
    local PREFIX="$1"
    local OUT="$2"

    export PREFIX OUT WORK

    python - <<'PY'
import json
import os
from pathlib import Path

prefix = os.environ["PREFIX"]
out = Path(os.environ["OUT"])
work = Path(os.environ["WORK"])

rows = []

for d in range(16):
    p = (
        work
        / "shards"
        / f"{prefix}_{d}.jsonl"
    )

    if not p.exists():
        raise RuntimeError(
            f"missing shard: {p}"
        )

    with p.open(encoding="utf-8") as f:
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


rows.sort(key=jid)

ids = [jid(x) for x in rows]

if len(ids) != len(set(ids)):
    raise RuntimeError(
        "duplicate job ids"
    )

with out.open(
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
    f"MERGED {prefix} rows={len(rows)} "
    f"output={out}"
)
PY
}


###############################################################################
# STUDENT
###############################################################################

if [ -f "$WORK/student_raw.jsonl" ] \
    && [ "$(wc -l < "$WORK/student_raw.jsonl")" -eq 2207 ]
then
    echo "STUDENT_RAW_ALREADY_PASS rows=2207"
else
    run16 \
        "student" \
        "$WORK/student_jobs.jsonl" \
        "$STUDENT" \
        32

    merge16 \
        "student" \
        "$WORK/student_raw.jsonl"
fi


###############################################################################
# MATERIALIZE STUDENT ROWS
###############################################################################

python - <<'PY'
import json
import os
from pathlib import Path

work = Path(os.environ["WORK"])


def load(p):
    with p.open(encoding="utf-8") as f:
        return [
            json.loads(x)
            for x in f
            if x.strip()
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
            return str(x[k]).strip()

    raise RuntimeError(
        f"cannot find generated response keys={list(x.keys())}"
    )


def clean_translation(s):
    s = s.strip()

    if s.lower().startswith("output:"):
        s = s.split(":", 1)[1].strip()

    if (
        len(s) >= 2
        and s[0] == s[-1]
        and s[0] in "\"'"
    ):
        s = s[1:-1].strip()

    return s


eval_rows = {
    int(x["job_id"]): x
    for x in load(
        work / "eval2207.jsonl"
    )
}

raw_rows = {
    jid(x): x
    for x in load(
        work / "student_raw.jsonl"
    )
}

if set(eval_rows) != set(raw_rows):
    raise RuntimeError(
        "student/eval id coverage mismatch"
    )

out_rows = []

for i in sorted(eval_rows):
    e = eval_rows[i]

    out_rows.append({
        "job_id": i,
        "demo_id": i,
        "dataset": e["dataset"],
        "local_id": e["local_id"],
        "source": e["source"],
        "reference": e["reference"],
        "student_translation":
            clean_translation(
                response(raw_rows[i])
            ),
    })

out = work / "student.jsonl"

with out.open(
    "w",
    encoding="utf-8",
) as f:
    for x in out_rows:
        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )

print(
    "STUDENT_MATERIALIZED_ROWS=",
    len(out_rows),
)
print("STUDENT_MATERIALIZE_PASS")
PY


###############################################################################
# EXACT OFFICIAL FEEDBACK JOBS
###############################################################################

python "$CORE" feedback-jobs \
    --official "$OFFICIAL" \
    --student "$WORK/student_for_feedback_jobs.jsonl" \
    --output "$WORK/feedback_jobs.jsonl"

echo \
"FEEDBACK_JOBS=$(wc -l < "$WORK/feedback_jobs.jsonl")"


###############################################################################
# SPECIALIZED PATCHER
###############################################################################

run16 \
    "specialized_feedback" \
    "$WORK/feedback_jobs.jsonl" \
    "$PATCHER" \
    8

merge16 \
    "specialized_feedback" \
    "$WORK/specialized_feedback_raw.jsonl"


###############################################################################
# BUILD ONE BASE-8B COMBO:
#   first 2207 = zero-shot Feedback
#   second 2207 = direct translation
#
# This saves an extra Qwen3-8B model-loading pass.
###############################################################################

python - <<'PY'
import copy
import json
import os
from pathlib import Path

work = Path(os.environ["WORK"])


def load(p):
    with p.open(encoding="utf-8") as f:
        return [
            json.loads(x)
            for x in f
            if x.strip()
        ]


fb = load(
    work / "feedback_jobs.jsonl"
)

student = load(
    work / "student.jsonl"
)

if len(fb) != 2207:
    raise RuntimeError(
        f"feedback jobs != 2207: {len(fb)}"
    )

student_by_id = {
    int(x["job_id"]): x
    for x in student
}

combo = []

# zero-shot Feedback jobs
for x in fb:
    j = copy.deepcopy(x)

    old = int(
        j.get(
            "job_id",
            j.get("demo_id"),
        )
    )

    j["job_id"] = old
    j["gate_kind"] = "zero_feedback"

    combo.append(j)

# direct Teacher translation jobs
offset = len(fb)

for x in fb:
    j = copy.deepcopy(x)

    old = int(
        j.get(
            "job_id",
            j.get("demo_id"),
        )
    )

    s = student_by_id[old]

    j["job_id"] = offset + old
    j["demo_id"] = offset + old

    j["source"] = s["source"]

    j["prompt"] = (
        "Translate the following sentences "
        "from Chinese to English.\n"
        f"Input: {s['source']}\n"
        "Output:"
    )

    if "temperature" in j:
        j["temperature"] = 0.0

    if "max_new_tokens" in j:
        j["max_new_tokens"] = 256

    j["gate_kind"] = "direct_teacher"
    j["original_job_id"] = old

    combo.append(j)

out = work / "base8b_combo_jobs.jsonl"

with out.open(
    "w",
    encoding="utf-8",
) as f:
    for x in combo:
        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )

print(
    "BASE8B_COMBO_JOBS=",
    len(combo),
)
PY


###############################################################################
# ZERO-SHOT FEEDBACK + DIRECT TEACHER
###############################################################################

run16 \
    "base8b_combo" \
    "$WORK/base8b_combo_jobs.jsonl" \
    "$TEACHER" \
    8

merge16 \
    "base8b_combo" \
    "$WORK/base8b_combo_raw.jsonl"


###############################################################################
# SCORE
###############################################################################

python - <<'PY'
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import sacrebleu

work = Path(os.environ["WORK"])


def load(name):
    p = work / name

    with p.open(encoding="utf-8") as f:
        return [
            json.loads(x)
            for x in f
            if x.strip()
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
            return str(x[k]).strip()

    raise RuntimeError(
        f"response field missing: {list(x.keys())}"
    )


def clean_translation(s):
    s = s.strip()

    if s.lower().startswith("output:"):
        s = s.split(":", 1)[1].strip()

    s = re.sub(
        r"^\s*(?:Translation|English)\s*:\s*",
        "",
        s,
        flags=re.I,
    )

    if (
        len(s) >= 2
        and s[0] == s[-1]
        and s[0] in "\"'"
    ):
        s = s[1:-1].strip()

    return s


def raw_feedback_to_postedit(raw, draft):
    text = raw.strip()

    # Conservative No-Error detection.
    no_error = bool(
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

    if no_error and not explicit_error:
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
    ]

    candidate = None

    for pattern in patterns:
        m = re.search(
            pattern,
            text,
        )

        if m:
            candidate = m.group(1).strip()
            break

    if candidate:
        candidate = re.sub(
            r"^\s*[-*]\s*",
            "",
            candidate,
        ).strip()

        if (
            len(candidate) >= 2
            and candidate[0] == candidate[-1]
            and candidate[0] in "\"'"
        ):
            candidate = (
                candidate[1:-1].strip()
            )

        return {
            "has_error": True,
            "post_edit": candidate,
            "extract_ok": True,
            "route": "error_good_translation",
        }

    return {
        "has_error": True,
        "post_edit": draft,
        "extract_ok": False,
        "route": "error_extract_fail_fallback_draft",
    }


student = load("student.jsonl")
special = load(
    "specialized_feedback_raw.jsonl"
)
combo = load(
    "base8b_combo_raw.jsonl"
)

N = len(student)

if N != 2207:
    raise RuntimeError(
        f"N != 2207: {N}"
    )

student_by_id = {
    int(x["job_id"]): x
    for x in student
}

special_by_id = {
    jid(x): x
    for x in special
}

zero_raw = {}
teacher_raw = {}

for x in combo:
    i = jid(x)

    if i < N:
        zero_raw[i] = x
    else:
        teacher_raw[i - N] = x

expected = set(range(N))

for name, obj in [
    ("special", special_by_id),
    ("zero", zero_raw),
    ("teacher", teacher_raw),
]:
    if set(obj) != expected:
        raise RuntimeError(
            f"{name} id coverage mismatch: "
            f"{len(obj)}/{N}"
        )


materialized = []

for i in range(N):
    s = student_by_id[i]

    draft = s["student_translation"]

    sp_raw = response(
        special_by_id[i]
    )

    z_raw = response(
        zero_raw[i]
    )

    t_raw = clean_translation(
        response(
            teacher_raw[i]
        )
    )

    sp = raw_feedback_to_postedit(
        sp_raw,
        draft,
    )

    z = raw_feedback_to_postedit(
        z_raw,
        draft,
    )

    materialized.append({
        **s,

        "zero_raw_feedback":
            z_raw,

        "zero_has_error":
            z["has_error"],

        "zero_extract_ok":
            z["extract_ok"],

        "zero_post_edit":
            z["post_edit"],

        "specialized_raw_feedback":
            sp_raw,

        "specialized_has_error":
            sp["has_error"],

        "specialized_extract_ok":
            sp["extract_ok"],

        "specialized_post_edit":
            sp["post_edit"],

        "direct_teacher":
            t_raw,
    })


out = work / "quality_gate_materialized.jsonl"

with out.open(
    "w",
    encoding="utf-8",
) as f:
    for x in materialized:
        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )


def metrics(rows, key):
    hyp = [
        x[key]
        for x in rows
    ]

    ref = [
        x["reference"]
        for x in rows
    ]

    bleu = sacrebleu.corpus_bleu(
        hyp,
        [ref],
    ).score

    chrf = sacrebleu.corpus_chrf(
        hyp,
        [ref],
    ).score

    return {
        "bleu": bleu,
        "chrf": chrf,
    }


summary = {
    "rows": N,
    "checkpoint":
        str(
            Path(os.environ["RUN_ROOT"])
            / os.environ["EXP"]
            / "patcher_qwen3_8b_paperfaith_fullft_v2"
            / "checkpoint-4752"
        ),

    "datasets": {},
}

for dataset in [
    "wmt24",
    "flores",
    "challenge",
]:
    rows = [
        x for x in materialized
        if x["dataset"] == dataset
    ]

    d = {
        "rows": len(rows),

        "student":
            metrics(
                rows,
                "student_translation",
            ),

        "zero_shot_patcher":
            metrics(
                rows,
                "zero_post_edit",
            ),

        "specialized_patcher":
            metrics(
                rows,
                "specialized_post_edit",
            ),

        "direct_teacher":
            metrics(
                rows,
                "direct_teacher",
            ),

        "zero_has_error_rate":
            sum(
                x["zero_has_error"]
                for x in rows
            )
            / len(rows),

        "specialized_has_error_rate":
            sum(
                x["specialized_has_error"]
                for x in rows
            )
            / len(rows),

        "zero_extract_rate":
            sum(
                x["zero_extract_ok"]
                for x in rows
            )
            / len(rows),

        "specialized_extract_rate":
            sum(
                x["specialized_extract_ok"]
                for x in rows
            )
            / len(rows),
    }

    d["specialized_minus_zero_bleu"] = (
        d["specialized_patcher"]["bleu"]
        - d["zero_shot_patcher"]["bleu"]
    )

    d["specialized_minus_student_bleu"] = (
        d["specialized_patcher"]["bleu"]
        - d["student"]["bleu"]
    )

    d["teacher_minus_specialized_bleu"] = (
        d["direct_teacher"]["bleu"]
        - d["specialized_patcher"]["bleu"]
    )

    summary["datasets"][dataset] = d


all_rows = materialized

summary["overall_micro"] = {
    "student":
        metrics(
            all_rows,
            "student_translation",
        ),

    "zero_shot_patcher":
        metrics(
            all_rows,
            "zero_post_edit",
        ),

    "specialized_patcher":
        metrics(
            all_rows,
            "specialized_post_edit",
        ),

    "direct_teacher":
        metrics(
            all_rows,
            "direct_teacher",
        ),

    "zero_has_error_rate":
        sum(
            x["zero_has_error"]
            for x in all_rows
        )
        / N,

    "specialized_has_error_rate":
        sum(
            x["specialized_has_error"]
            for x in all_rows
        )
        / N,

    "zero_extract_rate":
        sum(
            x["zero_extract_ok"]
            for x in all_rows
        )
        / N,

    "specialized_extract_rate":
        sum(
            x["specialized_extract_ok"]
            for x in all_rows
        )
        / N,
}

o = summary["overall_micro"]

o["specialized_minus_zero_bleu"] = (
    o["specialized_patcher"]["bleu"]
    - o["zero_shot_patcher"]["bleu"]
)

o["specialized_minus_student_bleu"] = (
    o["specialized_patcher"]["bleu"]
    - o["student"]["bleu"]
)

o["teacher_minus_specialized_bleu"] = (
    o["direct_teacher"]["bleu"]
    - o["specialized_patcher"]["bleu"]
)

summary_path = (
    work
    / "quality_gate_summary.json"
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
print("============================================================")
print("SPECIALIZED PATCHER QUALITY GATE RESULTS")
print("============================================================")

for dataset, d in summary["datasets"].items():
    print()
    print(dataset)

    for k in [
        "student",
        "zero_shot_patcher",
        "specialized_patcher",
        "direct_teacher",
    ]:
        print(
            f"  {k:24s} "
            f"BLEU={d[k]['bleu']:.4f} "
            f"chrF={d[k]['chrf']:.4f}"
        )

    print(
        "  specialized-zero BLEU = "
        f"{d['specialized_minus_zero_bleu']:+.4f}"
    )

    print(
        "  specialized-student BLEU = "
        f"{d['specialized_minus_student_bleu']:+.4f}"
    )

    print(
        "  teacher-specialized BLEU = "
        f"{d['teacher_minus_specialized_bleu']:+.4f}"
    )

    print(
        "  zero has_error = "
        f"{d['zero_has_error_rate']:.4%}"
    )

    print(
        "  specialized has_error = "
        f"{d['specialized_has_error_rate']:.4%}"
    )

    print(
        "  zero extract = "
        f"{d['zero_extract_rate']:.4%}"
    )

    print(
        "  specialized extract = "
        f"{d['specialized_extract_rate']:.4%}"
    )


print()
print("OVERALL MICRO")

for k in [
    "student",
    "zero_shot_patcher",
    "specialized_patcher",
    "direct_teacher",
]:
    print(
        f"  {k:24s} "
        f"BLEU={o[k]['bleu']:.4f} "
        f"chrF={o[k]['chrf']:.4f}"
    )

print(
    "  specialized-zero BLEU = "
    f"{o['specialized_minus_zero_bleu']:+.4f}"
)

print(
    "  specialized-student BLEU = "
    f"{o['specialized_minus_student_bleu']:+.4f}"
)

print(
    "  teacher-specialized BLEU = "
    f"{o['teacher_minus_specialized_bleu']:+.4f}"
)

print(
    "  zero has_error = "
    f"{o['zero_has_error_rate']:.4%}"
)

print(
    "  specialized has_error = "
    f"{o['specialized_has_error_rate']:.4%}"
)

print(
    "  zero extract = "
    f"{o['zero_extract_rate']:.4%}"
)

print(
    "  specialized extract = "
    f"{o['specialized_extract_rate']:.4%}"
)

print()
print(
    "SUMMARY_FILE=",
    summary_path,
)

print(
    "MATERIALIZED_FILE=",
    out,
)

print(
    "PATCHER_QUALITY_GATE_V1_PASS"
)
PY
