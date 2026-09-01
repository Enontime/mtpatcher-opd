#!/usr/bin/env bash
set -Eeuo pipefail

source /workspace/mtpatcher/project_env.sh

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

###############################################################################
# FROZEN PATHS
###############################################################################

EXP="mtpatcher_v3_full6565_20260823"
D="$DATA_ROOT/$EXP"

WORKER="$ROOT/scripts/mtpatcher_v3/strong_repro_pds_qwen_worker_v1.py"
STRUCTURAL_PARSER="$ROOT/scripts/mtpatcher_v3/build_pds_structural_acceptance_v1.py"

TRAINER="$ROOT/scripts/pilot_v2/train_qwen3_06b_full_sft_ascend.py"
EVAL="$ROOT/scripts/pilot_v2/eval_qwen3_06b_mt_ascend.py"
SCORE="$ROOT/scripts/pilot_v2/score_mt_jsonl.py"

OFFICIAL="$ROOT/vendor/MT-Patcher-official"

TEACHER="$MODEL_ROOT/Qwen3-8B"
STUDENT="$MODEL_ROOT/Qwen3-0.6B"

POP="$D/strong_repro_pds_population_v1"

PARENTS="$POP/pds_parent11792_v1.jsonl"
PAIRS="$POP/pds_local_pairs_v1.jsonl"

K1="$D/strong_repro_student_arms_freeze_v1/k1_all_pe11792_v1.jsonl"

###############################################################################
# OUTPUT
###############################################################################

OUT="$D/strong_repro_pds_full_to_student_v2"

ANALYSIS_JOBS="$OUT/analysis_jobs11792_v2.jsonl"
ANALYSIS_SHARDS="$OUT/analysis_shards16"
ANALYSIS_CLEAN="$OUT/analysis_topic_domain_style_v2.jsonl"
ANALYSIS_FAIL="$OUT/analysis_canonicalization_failures_v2.jsonl"

CASE_JOBS="$OUT/pds_case_jobs_v2.jsonl"
CASE_SHARDS="$OUT/case_shards16"

PDS_RAW="$OUT/pds_case_raw_v2.jsonl"
PDS_ACCEPTED="$OUT/pds_structural_accepted_v2.jsonl"
PDS_REJECTED="$OUT/pds_structural_rejected_v2.jsonl"
PDS_STUDENT="$OUT/pds_student_rows_v2.jsonl"

ARM_PDS="$OUT/k1all_plus_pds_v2.jsonl"
ARM_REPEAT="$OUT/k1_parent_matched_repeat_v2.jsonl"

MANIFEST="$OUT/student_arm_manifest_v2.json"
FINAL="$OUT/final_student_bleu_v2.json"

RUN_OUT="$RUN_ROOT/$EXP/strong_repro_pds_student2_v2"

PASS="$OUT/STRONG_REPRO_PDS_TO_STUDENT_BLEU_V2.PASS"
FAIL="$OUT/STRONG_REPRO_PDS_TO_STUDENT_BLEU_V2.FAIL"

mkdir -p \
    "$OUT" \
    "$ANALYSIS_SHARDS" \
    "$CASE_SHARDS" \
    "$RUN_OUT"

rm -f "$PASS" "$FAIL"

START_EPOCH="$(date +%s)"
echo "$START_EPOCH" > "$OUT/start_epoch.txt"

on_error () {
    rc=$?

    echo
    echo "======================================================================"
    echo "PDS -> STUDENT BLEU V2 FAILED"
    echo "RETURN_CODE=$rc"
    echo "FAIL_CST=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "FAIL_UTC=$(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "======================================================================"

    touch "$FAIL"
}

trap on_error ERR

echo "======================================================================"
echo "STRONG REPRO — FULL PDS -> STUDENT BLEU V2"
echo "======================================================================"
echo
echo "PRIMARY CONTRAST:"
echo "  K1ALL_PDS - K1_PARENT_MATCHED_REPEAT"
echo
echo "GENERATOR / CORPUS DIAGNOSTICS = CLOSED"
echo
echo "预计总运行时长：5–8 小时"
echo "ETA confidence = MEDIUM"
echo
echo "START_CST=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
echo "START_UTC=$(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
echo
echo "预计完成窗口："
echo "  CST_MIN=$(TZ=Asia/Shanghai date -d "@$((START_EPOCH + 5*3600))" '+%Y-%m-%d %H:%M:%S %Z')"
echo "  CST_MAX=$(TZ=Asia/Shanghai date -d "@$((START_EPOCH + 8*3600))" '+%Y-%m-%d %H:%M:%S %Z')"
echo "  UTC_MIN=$(TZ=UTC date -d "@$((START_EPOCH + 5*3600))" '+%Y-%m-%d %H:%M:%S %Z')"
echo "  UTC_MAX=$(TZ=UTC date -d "@$((START_EPOCH + 8*3600))" '+%Y-%m-%d %H:%M:%S %Z')"

###############################################################################
# STAGE 1 — PREFLIGHT
###############################################################################

echo
echo "======================================================================"
echo "STAGE 1/9 — PREFLIGHT"
echo "======================================================================"

BAD=0

for F in \
    "$WORKER" \
    "$STRUCTURAL_PARSER" \
    "$TRAINER" \
    "$EVAL" \
    "$SCORE" \
    "$PARENTS" \
    "$PAIRS" \
    "$K1"
do
    if [ ! -f "$F" ]; then
        echo "MISSING=$F"
        BAD=1
    fi
done

if [ ! -d "$TEACHER" ]; then
    echo "MISSING_MODEL=$TEACHER"
    BAD=1
fi

if [ ! -d "$STUDENT" ]; then
    echo "MISSING_MODEL=$STUDENT"
    BAD=1
fi

if [ "$BAD" -ne 0 ]; then
    false
fi

[ "$(wc -l < "$PARENTS")" -eq 11792 ]
[ "$(wc -l < "$PAIRS")" -eq 25000 ]
[ "$(wc -l < "$K1")" -eq 11792 ]

PAIR_SHA="$(sha256sum "$PAIRS" | awk '{print $1}')"
K1_SHA="$(sha256sum "$K1" | awk '{print $1}')"

echo "PAIR_SHA=$PAIR_SHA"
echo "K1_SHA=$K1_SHA"

[ "$PAIR_SHA" = \
"8d91211bcf733d6833d133dbe777e06810d026d0835d72726b8aacdc1f122016" ]

[ "$K1_SHA" = \
"7d8c5c1de8249db19e7e881174ef103c9ab6ebfa590623b6ef3190283684ed18" ]

###############################################################################
# Recover the EXISTING worker's analysis mode from source.
###############################################################################

ANALYSIS_MODE="$(
python - "$WORKER" <<'PY'
import ast
import re
import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text(
    encoding="utf-8"
)

candidates = []

for m in re.finditer(
    r"choices\s*=\s*(\[[^\]]+\]|\([^\)]+\))",
    src,
    re.S,
):
    try:
        values = ast.literal_eval(
            m.group(1)
        )
    except Exception:
        continue

    for x in values:
        if isinstance(x, str):
            candidates.append(x)

candidates += re.findall(
    r"args\.mode\s*==\s*[\"']([^\"']+)[\"']",
    src,
)

unique = []

for x in candidates:
    if x not in unique:
        unique.append(x)

for preferred in (
    "analysis",
    "sentence_analysis",
    "analyzer",
    "sentence",
):
    if preferred in unique:
        print(preferred)
        raise SystemExit

noncase = [
    x
    for x in unique
    if x != "case"
]

if len(noncase) == 1:
    print(noncase[0])
    raise SystemExit

raise RuntimeError(
    f"cannot determine analysis mode: {unique}"
)
PY
)"

echo "ANALYSIS_MODE=$ANALYSIS_MODE"

echo "PREFLIGHT_PASS"

###############################################################################
# 16-NPU worker launcher
###############################################################################

run_16way () {
    local INPUT="$1"
    local SHARD_DIR="$2"
    local MODE="$3"
    local LOG_DIR="$4"

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
            --mode "$MODE" \
            --seed 20260831 \
            > "$LOG_DIR/device_${DEVICE}.log" 2>&1 &

        PIDS+=("$!")

        echo \
            "WORKER_START mode=$MODE device=$DEVICE pid=$!"
    done

    local BAD_WORKER=0
    local PID

    for PID in "${PIDS[@]}"
    do
        if wait "$PID"; then
            :
        else
            echo "WORKER_FAILED pid=$PID mode=$MODE"
            BAD_WORKER=1
        fi
    done

    [ "$BAD_WORKER" -eq 0 ]
}

###############################################################################
# STAGE 2 — BUILD 11792 SENTENCE-ANALYZER JOBS
###############################################################################

echo
echo "======================================================================"
echo "STAGE 2/9 — BUILD + RUN SENTENCE ANALYZER"
echo "======================================================================"

python - "$PARENTS" "$OFFICIAL" "$ANALYSIS_JOBS" <<'PY'
import json
import sys
from pathlib import Path

parents_path = Path(sys.argv[1])
official_root = Path(sys.argv[2])
out = Path(sys.argv[3])

sys.path.insert(
    0,
    str(official_root),
)

from pipeline.data_manager.llama_sentence_analyzer import (
    SentenceAnalyzerDataManager,
)

prompt_template = SentenceAnalyzerDataManager.prompt

parents = []

with parents_path.open(
    encoding="utf-8",
) as f:
    for line in f:
        if line.strip():
            parents.append(
                json.loads(line)
            )

jobs = []

for jid, row in enumerate(
    parents
):
    source = str(
        row["source"]
    ).strip()

    prompt = (
        prompt_template
        .replace(
            "<src_text>",
            source,
        )
        .replace(
            "<srclang>",
            "Chinese",
        )
        .replace(
            "<tgtlang>",
            "English",
        )
    )

    jobs.append({
        "job_id":
            jid,

        "parent_index":
            int(
                row["index"]
            ),

        "source":
            source,

        "prompt":
            prompt,
    })

out.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with out.open(
    "w",
    encoding="utf-8",
) as f:
    for x in jobs:
        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )

print(
    "ANALYSIS_JOBS =",
    len(jobs),
)

if len(jobs) != 11792:
    raise RuntimeError(
        f"expected11792 got={len(jobs)}"
    )

print(
    "ANALYSIS_JOB_BUILD_PASS"
)
PY

date +%s > "$OUT/analysis_start_epoch.txt"

run_16way \
    "$ANALYSIS_JOBS" \
    "$ANALYSIS_SHARDS" \
    "$ANALYSIS_MODE" \
    "$OUT/log_analysis16"

date +%s > "$OUT/analysis_finish_epoch.txt"

###############################################################################
# STAGE 3 — TOPIC / DOMAIN / STYLE CANONICALIZATION
###############################################################################

echo
echo "======================================================================"
echo "STAGE 3/9 — ANALYZER CANONICALIZATION"
echo "======================================================================"

python - \
    "$ANALYSIS_JOBS" \
    "$ANALYSIS_SHARDS" \
    "$ANALYSIS_CLEAN" \
    "$ANALYSIS_FAIL" <<'PY'

import json
import re
import sys
from pathlib import Path


JOBS = Path(sys.argv[1])
SHARDS = Path(sys.argv[2])
CLEAN = Path(sys.argv[3])
FAIL = Path(sys.argv[4])


def load(path):
    rows = []

    with path.open(
        encoding="utf-8-sig",
        errors="replace",
    ) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue

            rows.append(
                json.loads(line)
            )

    return rows


def clean_md(s):
    s = str(s).strip()

    s = re.sub(
        r"^\s*#{1,6}\s*",
        "",
        s,
    )

    s = re.sub(
        r"^\s*[-•]\s*",
        "",
        s,
    )

    s = s.replace(
        "**",
        "",
    )

    s = s.replace(
        "__",
        "",
    )

    return s.strip()


patterns = {
    "topic":
        re.compile(
            r"^(?:\d+\.\s*)?"
            r"(?:"
            r"Topic(?:\s*[\(（]主题[\)）])?"
            r"|主题(?:\s*[\(（]Topic[\)）])?"
            r")"
            r"\s*[:：]\s*(.*)$",
            re.I,
        ),

    "domain":
        re.compile(
            r"^(?:\d+\.\s*)?"
            r"(?:"
            r"Domain(?:\s*[\(（]领域[\)）])?"
            r"|领域(?:\s*[\(（]Domain[\)）])?"
            r")"
            r"\s*[:：]\s*(.*)$",
            re.I,
        ),

    "style":
        re.compile(
            r"^(?:\d+\.\s*)?"
            r"(?:"
            r"Style(?:\s*[\(（]风格[\)）])?"
            r"|风格(?:\s*[\(（]Style[\)）])?"
            r")"
            r"\s*[:：]\s*(.*)$",
            re.I,
        ),
}


def get_raw(row):
    for k in (
        "raw_generation",
        "response",
        "generation",
        "output",
        "sentence_analysis",
    ):
        v = row.get(k)

        if (
            isinstance(v, str)
            and v.strip()
        ):
            return v

    return ""


def canonical(raw):
    lines = [
        clean_md(x)
        for x in str(raw).splitlines()
    ]

    found = {}

    for i, line in enumerate(lines):

        if not line:
            continue

        for name, pat in patterns.items():

            if name in found:
                continue

            m = pat.match(line)

            if not m:
                continue

            value = clean_md(
                m.group(1)
            )

            if not value:
                for j in range(
                    i + 1,
                    min(
                        len(lines),
                        i + 8,
                    ),
                ):
                    candidate = clean_md(
                        lines[j]
                    )

                    if not candidate:
                        continue

                    if candidate == "---":
                        continue

                    if any(
                        p.match(candidate)
                        for p in patterns.values()
                    ):
                        break

                    value = candidate
                    break

            if value:
                found[name] = value

    if set(found) != {
        "topic",
        "domain",
        "style",
    }:
        return None

    return (
        f"Topic: {found['topic']}\n"
        f"Domain: {found['domain']}\n"
        f"Style: {found['style']}"
    )


jobs = {
    int(x["job_id"]): x
    for x in load(JOBS)
}

results = {}
physical = 0
duplicates = 0

paths = sorted(
    SHARDS.glob(
        "device_*.jsonl"
    )
)

if len(paths) != 16:
    raise RuntimeError(
        f"expected16 analysis shards got={len(paths)}"
    )

for p in paths:
    for x in load(p):
        physical += 1

        jid = int(
            x["job_id"]
        )

        if jid in results:
            duplicates += 1
            continue

        results[jid] = x


missing = set(jobs) - set(results)

if missing:
    raise RuntimeError(
        f"analysis missing={len(missing)} "
        f"first={sorted(missing)[:20]}"
    )


clean_rows = []
fail_rows = []
leak_after = 0


for jid in sorted(jobs):

    job = jobs[jid]

    raw = get_raw(
        results[jid]
    )

    c = canonical(
        raw
    )

    if c is None:
        fail_rows.append({
            "job_id":
                jid,

            "parent_index":
                job[
                    "parent_index"
                ],

            "raw":
                raw,
        })

        continue

    source = str(
        job["source"]
    ).strip()

    if (
        source
        and
        source in c
    ):
        leak_after += 1

        fail_rows.append({
            "job_id":
                jid,

            "parent_index":
                job[
                    "parent_index"
                ],

            "reason":
                "FULL_PARENT_LEAK_AFTER",

            "raw":
                raw,

            "canonical":
                c,
        })

        continue

    clean_rows.append({
        "job_id":
            jid,

        "parent_index":
            int(
                job[
                    "parent_index"
                ]
            ),

        "source":
            source,

        "sentence_analysis":
            c,
    })


with CLEAN.open(
    "w",
    encoding="utf-8",
) as f:
    for x in clean_rows:
        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )


with FAIL.open(
    "w",
    encoding="utf-8",
) as f:
    for x in fail_rows:
        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )


print(
    "ANALYSIS_PHYSICAL_ROWS =",
    physical,
)

print(
    "ANALYSIS_DUPLICATE_ROWS =",
    duplicates,
)

print(
    "ANALYSIS_CANONICAL_OK =",
    len(clean_rows),
)

print(
    "ANALYSIS_CANONICAL_FAIL =",
    len(fail_rows),
)

print(
    "ANALYSIS_FULL_PARENT_LEAK_AFTER =",
    leak_after,
)

if not clean_rows:
    raise RuntimeError(
        "zero usable sentence analyses"
    )

print(
    "ANALYZER_CANONICALIZATION_PASS"
)
PY

###############################################################################
# STAGE 4 — RELEASED-REPO MECHANICAL GATE + ×4 CASE JOBS
###############################################################################

echo
echo "======================================================================"
echo "STAGE 4/9 — BUILD PDS CASE JOBS"
echo "======================================================================"

python - \
    "$PAIRS" \
    "$ANALYSIS_CLEAN" \
    "$OFFICIAL" \
    "$CASE_JOBS" <<'PY'

import json
import sys
from collections import Counter
from pathlib import Path


PAIRS = Path(sys.argv[1])
ANALYSIS = Path(sys.argv[2])
OFFICIAL = Path(sys.argv[3])
OUT = Path(sys.argv[4])


def load(path):
    rows = []

    with path.open(
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


sys.path.insert(
    0,
    str(OFFICIAL),
)

from pipeline.data_manager.llama_case_generation import (
    CaseGenerationDataManager,
)


pairs = load(PAIRS)

analyses = {
    int(x["parent_index"]):
        x["sentence_analysis"]
    for x in load(ANALYSIS)
}


num_case = int(
    CaseGenerationDataManager.num_case
)

prompt_template = (
    CaseGenerationDataManager.prompt
)

if num_case != 4:
    raise RuntimeError(
        f"repo num_case expected4 got={num_case}"
    )


reject = Counter()
passed_pairs = 0
jobs = []


for row in pairs:

    P = str(
        row["source_span"]
    ).strip()

    Q = str(
        row["correction"]
    ).strip()

    student = str(
        row["student_translation"]
    )

    post_edit = str(
        row["post_edit"]
    )

    parent_index = int(
        row["parent_index"]
    )


    # Released repo correction-consistency clauses.
    if not Q:
        reject[
            "EMPTY_Q"
        ] += 1
        continue

    if Q in student:
        reject[
            "Q_ALREADY_IN_STUDENT"
        ] += 1
        continue

    if Q not in post_edit:
        reject[
            "Q_NOT_IN_POST_EDIT"
        ] += 1
        continue

    if parent_index not in analyses:
        reject[
            "NO_CANONICAL_ANALYSIS"
        ] += 1
        continue


    analysis = analyses[
        parent_index
    ]


    prompt = (
        prompt_template
        .replace(
            "<domain_topic_style>",
            analysis,
        )
        .replace(
            "<word_pair>",
            f"{P}({Q})",
        )
        .replace(
            "<srclang>",
            "Chinese",
        )
        .replace(
            "<tgtlang>",
            "English",
        )
    )


    passed_pairs += 1


    for slot in range(
        num_case
    ):

        jobs.append({
            "job_id":
                len(jobs),

            "pair_id":
                int(
                    row["pair_id"]
                ),

            "parent_index":
                parent_index,

            "error_index":
                int(
                    row["error_index"]
                ),

            "pds_slot":
                slot,

            "source_span":
                P,

            "correction":
                Q,

            "error_type":
                row.get(
                    "error_type",
                    "",
                ),

            "original_source":
                row["source"],

            "student_translation":
                row[
                    "student_translation"
                ],

            "post_edit":
                post_edit,

            "sentence_analysis":
                analysis,

            "prompt":
                prompt,
        })


with OUT.open(
    "w",
    encoding="utf-8",
) as f:
    for x in jobs:
        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )


print(
    "INPUT_LOCAL_PAIRS =",
    len(pairs),
)

print(
    "REPO_MECHANICAL_PASS_PAIRS =",
    passed_pairs,
)

print(
    "CASE_REJECT_COUNTS =",
    dict(reject),
)

print(
    "REPO_NUM_CASE =",
    num_case,
)

print(
    "CASE_JOBS =",
    len(jobs),
)

if not jobs:
    raise RuntimeError(
        "zero PDS case jobs"
    )

print(
    "CASE_JOB_BUILD_PASS"
)
PY

CASE_EXPECTED="$(wc -l < "$CASE_JOBS")"

echo "$CASE_EXPECTED" \
    > "$OUT/case_expected_rows.txt"

echo "CASE_EXPECTED_ROWS=$CASE_EXPECTED"

###############################################################################
# STAGE 5 — FULL CASE GENERATION
###############################################################################

echo
echo "======================================================================"
echo "STAGE 5/9 — FULL CASE GENERATION — 16 NPUs"
echo "======================================================================"

date +%s \
    > "$OUT/case_start_epoch.txt"

run_16way \
    "$CASE_JOBS" \
    "$CASE_SHARDS" \
    "case" \
    "$OUT/log_case16"

date +%s \
    > "$OUT/case_finish_epoch.txt"

###############################################################################
# STAGE 6 — STRUCTURAL ACCEPTANCE + PARENT-MATCHED CONTROL
###############################################################################

echo
echo "======================================================================"
echo "STAGE 6/9 — BUILD STUDENT ARMS"
echo "======================================================================"

python - \
    "$CASE_JOBS" \
    "$CASE_SHARDS" \
    "$STRUCTURAL_PARSER" \
    "$K1" \
    "$PDS_RAW" \
    "$PDS_ACCEPTED" \
    "$PDS_REJECTED" \
    "$PDS_STUDENT" \
    "$ARM_PDS" \
    "$ARM_REPEAT" \
    "$MANIFEST" <<'PY'

import ast
import copy
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path


(
    CASE_JOBS,
    CASE_SHARDS,
    PARSER_PATH,
    K1_PATH,
    RAW_OUT,
    ACCEPT_OUT,
    REJECT_OUT,
    PDS_STUDENT_OUT,
    ARM_PDS_OUT,
    ARM_REPEAT_OUT,
    MANIFEST_OUT,
) = map(
    Path,
    sys.argv[1:],
)


PROMPT_PREFIX = (
    "Translate the following text into English "
    "without additional explanations:\n\n"
)


def load(path):
    rows = []

    with path.open(
        encoding="utf-8-sig",
        errors="replace",
    ) as f:
        for n, line in enumerate(f, 1):

            if not line.strip():
                continue

            rows.append(
                json.loads(line)
            )

    return rows


def dump(path, rows):

    with path.open(
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


###############################################################################
# Load ONLY deterministic parser definitions from frozen structural parser.
###############################################################################

src = PARSER_PATH.read_text(
    encoding="utf-8"
)

tree = ast.parse(
    src
)

body = []

wanted_assign = {
    "ZH_LABEL",
    "EN_LABEL",
    "META_HEAD",
}

for node in tree.body:

    if isinstance(
        node,
        (
            ast.Import,
            ast.ImportFrom,
            ast.FunctionDef,
        ),
    ):
        body.append(
            node
        )

    elif isinstance(
        node,
        ast.Assign,
    ):
        names = {
            t.id
            for t in node.targets
            if isinstance(
                t,
                ast.Name,
            )
        }

        if names & wanted_assign:
            body.append(
                node
            )


module = ast.Module(
    body=body,
    type_ignores=[],
)

ns = {}

exec(
    compile(
        module,
        str(PARSER_PATH),
        "exec",
    ),
    ns,
    ns,
)

parse_pair = ns[
    "parse_pair"
]

norm = ns[
    "norm"
]


###############################################################################
# Merge frozen generation shards.
###############################################################################

jobs = {
    int(x["job_id"]): x
    for x in load(
        CASE_JOBS
    )
}

results = {}
physical = 0
duplicates = 0

paths = sorted(
    CASE_SHARDS.glob(
        "device_*.jsonl"
    )
)

if len(paths) != 16:
    raise RuntimeError(
        f"expected16 case shards got={len(paths)}"
    )


for p in paths:
    for x in load(p):

        physical += 1

        jid = int(
            x["job_id"]
        )

        if jid in results:
            duplicates += 1
            continue

        results[jid] = x


missing = set(jobs) - set(results)

if missing:
    raise RuntimeError(
        f"missing case jobs={len(missing)} "
        f"first={sorted(missing)[:20]}"
    )


def get_raw(row):

    for key in (
        "raw_generation",
        "response",
        "generation",
        "output",
    ):
        value = row.get(
            key
        )

        if (
            isinstance(
                value,
                str,
            )
            and value.strip()
        ):
            return value

    return ""


raw_rows = [
    results[j]
    for j in sorted(
        results
    )
]

dump(
    RAW_OUT,
    raw_rows,
)


###############################################################################
# Frozen structural acceptance:
#   parse unique bilingual X',Y'
#   reject parent copy
#   require P in X'
#   Q exact is diagnostic only
###############################################################################

accepted = []
rejected = []

reject_counts = Counter()
parse_methods = Counter()

xy_seen = set()
duplicate_xy = 0


for jid in sorted(
    jobs
):

    job = jobs[jid]

    raw = get_raw(
        results[jid]
    )

    P = str(
        job["source_span"]
    ).strip()

    Q = str(
        job["correction"]
    ).strip()

    parent = str(
        job["original_source"]
    ).strip()


    parsed = parse_pair(
        raw,
        P,
    )


    if (
        parsed["status"]
        !=
        "PARSEABLE"
    ):

        reason = parsed[
            "status"
        ]

        reject_counts[
            reason
        ] += 1

        rejected.append({
            "job_id":
                jid,

            "pair_id":
                job[
                    "pair_id"
                ],

            "reason":
                reason,
        })

        continue


    X = parsed[
        "X_prime"
    ]

    Y = parsed[
        "Y_prime"
    ]


    parse_methods[
        parsed["method"]
    ] += 1


    P_in_X = (
        bool(
            norm(P)
        )
        and
        norm(P)
        in
        norm(X)
    )


    parent_copy = (
        bool(
            norm(parent)
        )
        and
        norm(parent)
        in
        norm(X)
    )


    Q_in_Y = (
        bool(
            norm(Q)
        )
        and
        norm(Q)
        in
        norm(Y)
    )


    if parent_copy:

        reject_counts[
            "PARENT_COPY_X"
        ] += 1

        rejected.append({
            "job_id":
                jid,

            "pair_id":
                job[
                    "pair_id"
                ],

            "reason":
                "PARENT_COPY_X",
        })

        continue


    if not P_in_X:

        reject_counts[
            "P_MISSING_X"
        ] += 1

        rejected.append({
            "job_id":
                jid,

            "pair_id":
                job[
                    "pair_id"
                ],

            "reason":
                "P_MISSING_X",
        })

        continue


    xy = (
        norm(X),
        norm(Y),
    )

    if xy in xy_seen:
        duplicate_xy += 1
    else:
        xy_seen.add(
            xy
        )


    accepted.append({
        "job_id":
            jid,

        "pair_id":
            int(
                job[
                    "pair_id"
                ]
            ),

        "parent_index":
            int(
                job[
                    "parent_index"
                ]
            ),

        "error_index":
            int(
                job[
                    "error_index"
                ]
            ),

        "pds_slot":
            int(
                job[
                    "pds_slot"
                ]
            ),

        "source":
            X,

        "target_translation":
            Y,

        "source_span":
            P,

        "correction":
            Q,

        "error_type":
            job.get(
                "error_type",
                "",
            ),

        "original_source":
            parent,

        "parse_method":
            parsed[
                "method"
            ],

        "Q_exact_in_Y_DIAGNOSTIC_ONLY":
            Q_in_Y,
    })


dump(
    ACCEPT_OUT,
    accepted,
)

dump(
    REJECT_OUT,
    rejected,
)


if not accepted:
    raise RuntimeError(
        "zero structurally accepted PDS"
    )


###############################################################################
# Convert accepted PDS to Student SFT schema.
###############################################################################

pds_student = []

for row in accepted:

    src = row[
        "source"
    ]

    tgt = row[
        "target_translation"
    ]


    pds_student.append({
        "index":
            f"strong_pds_v2_{row['job_id']}",

        "source":
            src,

        "messages":
            [
                {
                    "role":
                        "user",

                    "content":
                        PROMPT_PREFIX
                        +
                        src
                        +
                        "\n\n",
                }
            ],

        "target_translation":
            tgt,

        "student_translation":
            "",

        "feedback_errors":
            [
                {
                    "source_span":
                        row[
                            "source_span"
                        ],

                    "translation_span":
                        "",

                    "error_type":
                        row.get(
                            "error_type",
                            "",
                        ),

                    "explanation":
                        "PDS synthesized context",

                    "correction":
                        row[
                            "correction"
                        ],
                }
            ],

        "construction_method":
            "STRONG_REPRO_PDS_V2",

        "parent_index":
            row[
                "parent_index"
            ],

        "error_index":
            row[
                "error_index"
            ],

        "pds_slot":
            row[
                "pds_slot"
            ],
    })


dump(
    PDS_STUDENT_OUT,
    pds_student,
)


###############################################################################
# Frozen K1.
###############################################################################

k1 = load(
    K1_PATH
)

if len(k1) != 11792:
    raise RuntimeError(
        f"K1 rows={len(k1)}"
    )


k1_by_index = {}

for row in k1:

    idx = int(
        row["index"]
    )

    if idx in k1_by_index:
        raise RuntimeError(
            f"duplicate K1 index={idx}"
        )

    k1_by_index[
        idx
    ] = row


###############################################################################
# Parent contribution c_i.
###############################################################################

parent_pds_counts = Counter(
    int(
        row[
            "parent_index"
        ]
    )
    for row in accepted
)


unknown_parents = (
    set(
        parent_pds_counts
    )
    -
    set(
        k1_by_index
    )
)

if unknown_parents:
    raise RuntimeError(
        "PDS references parent outside K1: "
        f"{sorted(unknown_parents)[:20]}"
    )


###############################################################################
# ARM A
#
#   1 x K1_i + c_i x PDS_i
###############################################################################

arm_pds = (
    copy.deepcopy(
        k1
    )
    +
    copy.deepcopy(
        pds_student
    )
)

rng_a = random.Random(
    20260831
)

rng_a.shuffle(
    arm_pds
)


###############################################################################
# ARM B
#
#   (1 + c_i) x K1_i
#
# Exact parent-level row exposure match.
###############################################################################

arm_repeat = []

for row in k1:

    idx = int(
        row["index"]
    )

    copies = (
        1
        +
        parent_pds_counts.get(
            idx,
            0,
        )
    )

    for _ in range(
        copies
    ):

        arm_repeat.append(
            copy.deepcopy(
                row
            )
        )


if len(
    arm_repeat
) != len(
    arm_pds
):

    raise RuntimeError(
        "parent-matched row count mismatch: "
        f"PDS={len(arm_pds)} "
        f"Repeat={len(arm_repeat)}"
    )


rng_b = random.Random(
    20260831
)

rng_b.shuffle(
    arm_repeat
)


dump(
    ARM_PDS_OUT,
    arm_pds,
)

dump(
    ARM_REPEAT_OUT,
    arm_repeat,
)


###############################################################################
# Explicit parent exposure invariant.
###############################################################################

max_parent_diff = 0

for idx in k1_by_index:

    pds_exposure = (
        1
        +
        parent_pds_counts.get(
            idx,
            0,
        )
    )

    repeat_exposure = (
        1
        +
        parent_pds_counts.get(
            idx,
            0,
        )
    )

    max_parent_diff = max(
        max_parent_diff,
        abs(
            pds_exposure
            -
            repeat_exposure
        ),
    )


if max_parent_diff != 0:
    raise RuntimeError(
        "parent exposure invariant failed"
    )


hist = Counter(
    parent_pds_counts.values()
)

hist[0] = (
    len(k1)
    -
    len(parent_pds_counts)
)


manifest = {
    "protocol":
        "STRONG_REPRO_PDS_TO_STUDENT_V2_PARENT_MATCHED",

    "interpretation":
        (
            "released-repo-faithful PDS adaptation; "
            "not Table-1 cardinality reproduction"
        ),

    "primary_contrast":
        (
            "K1ALL_PDS minus K1_PARENT_MATCHED_REPEAT"
        ),

    "claim_scope":
        (
            "PDS treatment utility beyond parent-matched "
            "repetition of PE supervision; "
            "not context-alone effect; "
            "not row/compute efficiency"
        ),

    "case_jobs":
        len(jobs),

    "case_physical_rows":
        physical,

    "case_duplicate_physical_rows":
        duplicates,

    "PDS_structural_accepted":
        len(accepted),

    "PDS_structural_accept_rate":
        len(accepted)
        /
        len(jobs),

    "structural_reject_counts":
        dict(
            reject_counts
        ),

    "parse_method_counts":
        dict(
            parse_methods
        ),

    "exact_XY_duplicate_occurrences_diagnostic":
        duplicate_xy,

    "Q_exact_is_gate":
        False,

    "K1_rows":
        len(k1),

    "PDS_rows":
        len(pds_student),

    "arm_rows_each":
        len(arm_pds),

    "parent_exposure_exact_match":
        True,

    "parent_exposure_max_absolute_difference":
        max_parent_diff,

    "parents_with_accepted_PDS":
        len(
            parent_pds_counts
        ),

    "parents_without_accepted_PDS":
        len(k1)
        -
        len(
            parent_pds_counts
        ),

    "max_PDS_rows_from_one_parent":
        max(
            parent_pds_counts.values(),
            default=0,
        ),

    "PDS_rows_per_parent_histogram":
        {
            str(k):
                int(v)
            for k, v
            in sorted(
                hist.items()
            )
        },

    "sha256": {
        "PDS_student":
            sha(
                PDS_STUDENT_OUT
            ),

        "K1ALL_PDS":
            sha(
                ARM_PDS_OUT
            ),

        "K1_PARENT_MATCHED_REPEAT":
            sha(
                ARM_REPEAT_OUT
            ),
    },
}


MANIFEST_OUT.write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print(
    "CASE_PHYSICAL_ROWS =",
    physical,
)

print(
    "CASE_DUPLICATE_PHYSICAL_ROWS =",
    duplicates,
)

print(
    "STRUCTURAL_ACCEPTED_PDS =",
    len(accepted),
)

print(
    "STRUCTURAL_ACCEPT_RATE =",
    f"{len(accepted)/len(jobs):.6f}",
)

print(
    "STRUCTURAL_REJECTS =",
    dict(reject_counts),
)

print(
    "EXACT_XY_DUPLICATES_DIAGNOSTIC =",
    duplicate_xy,
)

print(
    "PARENTS_WITH_ACCEPTED_PDS =",
    len(parent_pds_counts),
)

print(
    "MAX_PDS_ROWS_ONE_PARENT =",
    max(
        parent_pds_counts.values(),
        default=0,
    ),
)

print(
    "K1_ROWS =",
    len(k1),
)

print(
    "PDS_ROWS =",
    len(pds_student),
)

print(
    "MATCHED_ARM_TOTAL_ROWS =",
    len(arm_pds),
)

print(
    "PARENT_EXPOSURE_MAX_DIFF =",
    max_parent_diff,
)

print(
    "PARENT_MATCHED_CONTROL_PASS"
)
PY

###############################################################################
# NON-GATING TOKEN DIAGNOSTIC
###############################################################################

echo
echo "======================================================================"
echo "TOKEN EXPOSURE DIAGNOSTIC — NON-GATING"
echo "======================================================================"

if python - \
    "$STUDENT" \
    "$ARM_PDS" \
    "$ARM_REPEAT" \
    "$MANIFEST" <<'PY'

import json
import sys
from pathlib import Path

from transformers import (
    AutoTokenizer,
)


MODEL = sys.argv[1]
PDS = Path(sys.argv[2])
REPEAT = Path(sys.argv[3])
MANIFEST = Path(sys.argv[4])


tok = AutoTokenizer.from_pretrained(
    MODEL,
    local_files_only=True,
    trust_remote_code=True,
)


def count(path):

    rows = 0
    src_n = 0
    tgt_n = 0

    src_batch = []
    tgt_batch = []


    def flush():

        nonlocal src_n
        nonlocal tgt_n
        nonlocal src_batch
        nonlocal tgt_batch

        if not src_batch:
            return

        a = tok(
            src_batch,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )[
            "input_ids"
        ]

        b = tok(
            tgt_batch,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )[
            "input_ids"
        ]

        src_n += sum(
            len(x)
            for x in a
        )

        tgt_n += sum(
            len(x)
            for x in b
        )

        src_batch = []
        tgt_batch = []


    with path.open(
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            x = json.loads(
                line
            )

            src_batch.append(
                str(
                    x["source"]
                )
            )

            tgt_batch.append(
                str(
                    x[
                        "target_translation"
                    ]
                )
            )

            rows += 1

            if len(
                src_batch
            ) >= 512:

                flush()


    flush()


    return {
        "rows":
            rows,

        "raw_source_tokens":
            src_n,

        "raw_target_tokens":
            tgt_n,

        "raw_source_plus_target_tokens":
            src_n
            +
            tgt_n,
    }


diagnostic = {
    "note":
        (
            "untruncated raw source/target token counts; "
            "diagnostic only; not matched compute"
        ),

    "K1ALL_PDS":
        count(PDS),

    "K1_PARENT_MATCHED_REPEAT":
        count(REPEAT),
}


manifest = json.loads(
    MANIFEST.read_text(
        encoding="utf-8"
    )
)

manifest[
    "token_exposure_diagnostic"
] = diagnostic


MANIFEST.write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print(
    json.dumps(
        diagnostic,
        ensure_ascii=False,
        indent=2,
    )
)

print(
    "TOKEN_DIAGNOSTIC_PASS"
)
PY
then
    echo "TOKEN_DIAGNOSTIC_STATUS=PASS"
else
    echo "TOKEN_DIAGNOSTIC_STATUS=WARNING_NONBLOCKING"
fi

###############################################################################
# STAGE 7 — TWO STUDENT ARMS IN PARALLEL
###############################################################################

echo
echo "======================================================================"
echo "STAGE 7/9 — STUDENT TRAIN + EVAL"
echo "======================================================================"

WMT="$DATA_ROOT/pilot_v2_qwen3_06b/wmt24_zh_en998.jsonl"
FLORES="$DATA_ROOT/pilot_v2_qwen3_06b/flores_zh_en1012.jsonl"
CHALLENGE="$DATA_ROOT/pilot_v2_qwen3_06b/challenge_zh_en197.jsonl"

date +%s \
    > "$OUT/student_start_epoch.txt"


run_student () {

    local ARM="$1"
    local CARD="$2"
    local TRAIN_DATA="$3"

    local ARM_DIR="$RUN_OUT/$ARM"
    local LOG="$OUT/${ARM}.train_eval.log"

    mkdir -p \
        "$ARM_DIR" \
        "$ARM_DIR/eval"

    (
        export ASCEND_RT_VISIBLE_DEVICES="$CARD"

        echo "ARM=$ARM"
        echo "CARD=$CARD"
        echo "ROWS=$(wc -l < "$TRAIN_DATA")"
        echo "START_CST=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"

        python -u "$TRAINER" \
            --model "$STUDENT" \
            --train "$TRAIN_DATA" \
            --output-dir "$ARM_DIR" \
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

        [ -d "$ARM_DIR/epoch3" ]


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


            PRED="$ARM_DIR/eval/${NAME}.pred.jsonl"
            METRIC="$ARM_DIR/eval/${NAME}.metrics.json"


            python -u "$EVAL" \
                --model "$ARM_DIR/epoch3" \
                --tokenizer "$ARM_DIR/epoch3" \
                --input "$INPUT" \
                --output "$PRED" \
                --method "strong_repro_v2_${ARM}_${NAME}" \
                --batch-size 16 \
                --max-new-tokens 256 \
                --attn-implementation sdpa


            python "$SCORE" \
                --input "$PRED" \
                --output "$METRIC"


            echo
            echo "===== $ARM / $NAME ====="
            cat "$METRIC"
            echo

        done


        echo "ARM_COMPLETE=$ARM"
        echo "FINISH_CST=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"

    ) > "$LOG" 2>&1
}


run_student \
    "K1ALL_PDS" \
    0 \
    "$ARM_PDS" &

PID_PDS=$!

echo \
    "STUDENT_ARM_LAUNCHED arm=K1ALL_PDS card=0 pid=$PID_PDS"


run_student \
    "K1_PARENT_MATCHED_REPEAT" \
    1 \
    "$ARM_REPEAT" &

PID_REPEAT=$!

echo \
    "STUDENT_ARM_LAUNCHED arm=K1_PARENT_MATCHED_REPEAT card=1 pid=$PID_REPEAT"


STUDENT_BAD=0

if wait "$PID_PDS"; then
    echo \
        "STUDENT_ARM_FINISHED arm=K1ALL_PDS status=0"
else
    echo \
        "STUDENT_ARM_FAILED arm=K1ALL_PDS"
    STUDENT_BAD=1
fi


if wait "$PID_REPEAT"; then
    echo \
        "STUDENT_ARM_FINISHED arm=K1_PARENT_MATCHED_REPEAT status=0"
else
    echo \
        "STUDENT_ARM_FAILED arm=K1_PARENT_MATCHED_REPEAT"
    STUDENT_BAD=1
fi


[ "$STUDENT_BAD" -eq 0 ]

date +%s \
    > "$OUT/student_finish_epoch.txt"

###############################################################################
# STAGE 8 — FINAL BLEU
###############################################################################

echo
echo "======================================================================"
echo "STAGE 8/9 — FINAL STUDENT BLEU"
echo "======================================================================"

python - "$RUN_OUT" "$FINAL" <<'PY'

import json
import sys
from pathlib import Path


ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])


arms = [
    "K1ALL_PDS",
    "K1_PARENT_MATCHED_REPEAT",
]

sets = [
    "wmt24",
    "flores",
    "challenge",
]


result = {}


for arm in arms:

    result[
        arm
    ] = {}

    for s in sets:

        p = (
            ROOT
            /
            arm
            /
            "eval"
            /
            f"{s}.metrics.json"
        )

        x = json.loads(
            p.read_text(
                encoding="utf-8"
            )
        )

        result[
            arm
        ][
            s
        ] = {
            "BLEU":
                float(
                    x["BLEU"]
                ),

            "chrF":
                float(
                    x["chrF"]
                ),
        }


    result[
        arm
    ][
        "macro_BLEU"
    ] = sum(
        result[arm][s][
            "BLEU"
        ]
        for s in sets
    ) / 3


    result[
        arm
    ][
        "macro_chrF"
    ] = sum(
        result[arm][s][
            "chrF"
        ]
        for s in sets
    ) / 3


pds = result[
    "K1ALL_PDS"
][
    "macro_BLEU"
]

repeat = result[
    "K1_PARENT_MATCHED_REPEAT"
][
    "macro_BLEU"
]


BASE = 17.3485217
K1_OLD = 17.5139
FULL_SEQKD = 19.1652


result[
    "contrasts"
] = {
    "PRIMARY_PDS_MINUS_PARENT_MATCHED_REPEAT":
        pds
        -
        repeat,

    "PDS_MINUS_FROZEN_K1ALL":
        pds
        -
        K1_OLD,

    "PDS_MINUS_FULLSEQKD20K":
        pds
        -
        FULL_SEQKD,

    "PDS_GAIN_OVER_BASE":
        pds
        -
        BASE,

    "PDS_RECOVERY_FRACTION_OF_FULLSEQKD_GAIN":
        (
            pds
            -
            BASE
        )
        /
        (
            FULL_SEQKD
            -
            BASE
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
    "=" * 88
)

print(
    "FINAL STUDENT BLEU"
)

print(
    "=" * 88
)

print(
    f"{'ARM':30s}"
    f"{'WMT24':>10s}"
    f"{'FLORES':>10s}"
    f"{'CHALL':>10s}"
    f"{'MACRO':>10s}"
    f"{'CHRFMAC':>10s}"
)


for arm in arms:

    print(
        f"{arm:30s}"
        f"{result[arm]['wmt24']['BLEU']:10.4f}"
        f"{result[arm]['flores']['BLEU']:10.4f}"
        f"{result[arm]['challenge']['BLEU']:10.4f}"
        f"{result[arm]['macro_BLEU']:10.4f}"
        f"{result[arm]['macro_chrF']:10.4f}"
    )


print()

print(
    "PRIMARY_PDS_MINUS_PARENT_MATCHED_MACRO_BLEU =",
    result[
        "contrasts"
    ][
        "PRIMARY_PDS_MINUS_PARENT_MATCHED_REPEAT"
    ],
)

print(
    "PDS_MINUS_FROZEN_K1ALL_MACRO_BLEU =",
    result[
        "contrasts"
    ][
        "PDS_MINUS_FROZEN_K1ALL"
    ],
)

print(
    "PDS_RECOVERY_FRACTION_FULLSEQKD_GAIN =",
    result[
        "contrasts"
    ][
        "PDS_RECOVERY_FRACTION_OF_FULLSEQKD_GAIN"
    ],
)

print(
    "FINAL_STUDENT_BLEU_PASS"
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
    "$PARENTS" \
    "$PAIRS" \
    "$K1" \
    "$ANALYSIS_JOBS" \
    "$ANALYSIS_CLEAN" \
    "$CASE_JOBS" \
    "$PDS_ACCEPTED" \
    "$PDS_STUDENT" \
    "$ARM_PDS" \
    "$ARM_REPEAT" \
    "$MANIFEST" \
    "$FINAL" \
    > "$OUT/final_sha256_manifest_v2.txt"


touch "$PASS"
rm -f "$FAIL"


END_EPOCH="$(date +%s)"


echo
echo "======================================================================"
echo "STRONG REPRO PDS -> STUDENT BLEU V2 PASS"
echo "======================================================================"
echo "TOTAL_ELAPSED_SECONDS=$((END_EPOCH - START_EPOCH))"
echo "FINISH_CST=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
echo "FINISH_UTC=$(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
echo "FINAL=$FINAL"
echo "PASS=$PASS"
echo "======================================================================"

