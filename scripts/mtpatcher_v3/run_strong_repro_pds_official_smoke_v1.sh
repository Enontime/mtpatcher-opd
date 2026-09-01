#!/usr/bin/env bash
set -Eeuo pipefail

source /workspace/mtpatcher/project_env.sh

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

EXP="mtpatcher_v3_full6565_20260823"
D="$DATA_ROOT/$EXP"

MODEL="$MODEL_ROOT/Qwen3-8B"

WORKER="$ROOT/scripts/mtpatcher_v3/strong_repro_pds_qwen_worker_v1.py"

###############################################################################
# Frozen population
###############################################################################

POP="$D/strong_repro_pds_population_v1"

PARENTS="$POP/pds_parent11792_v1.jsonl"
PAIRS="$POP/pds_local_pairs_v1.jsonl"

POP_PASS="$POP/STRONG_REPRO_PDS_POPULATION_V1.PASS"

###############################################################################
# Official repo
###############################################################################

OFFICIAL="$ROOT/vendor/MT-Patcher-official"

OFFICIAL_ANALYZER="$OFFICIAL/pipeline/data_manager/llama_sentence_analyzer.py"

OFFICIAL_CASE="$OFFICIAL/pipeline/data_manager/llama_case_generation.py"

###############################################################################
# Smoke
###############################################################################

OUT="$D/strong_repro_pds_official_smoke_v1"

LOG_DIR="$LOG_ROOT/$EXP/strong_repro_pds_official_smoke_v1"

mkdir -p \
    "$OUT" \
    "$LOG_DIR"

SPEC="$OUT/official_pds_spec_v1.json"

SAMPLE_IDS="$OUT/smoke_parent_ids128_seed20260831.txt"

ANALYSIS_JOBS="$OUT/sentence_analysis_jobs128_v1.jsonl"

ANALYSIS_SHARDS="$OUT/sentence_analysis_shards16"

ANALYSIS_MERGED="$OUT/sentence_analysis128_v1.jsonl"

PDS_JOBS="$OUT/pds_smoke_jobs_v1.jsonl"

PDS_SHARDS="$OUT/pds_smoke_shards16"

PDS_RAW="$OUT/pds_smoke_raw_v1.jsonl"

PDS_VALID="$OUT/pds_smoke_valid_v1.jsonl"

AUDIT="$OUT/pds_smoke_audit_v1.json"

META="$OUT/start_meta_v1.txt"

PASS="$OUT/STRONG_REPRO_PDS_OFFICIAL_SMOKE_V1.PASS"

FAIL="$OUT/STRONG_REPRO_PDS_OFFICIAL_SMOKE_V1.FAIL"

SAMPLE_N=128
SEED=20260831
WORLD_SIZE=16

rm -f \
    "$PASS" \
    "$FAIL"

mkdir -p \
    "$ANALYSIS_SHARDS" \
    "$PDS_SHARDS"

START_EPOCH="$(date +%s)"

ETA_MIN=180
ETA_MAX=420

{
    echo "START_EPOCH=$START_EPOCH"
    echo "START_CST=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "START_UTC=$(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"

    echo "ETA_MIN_CST=$(TZ=Asia/Shanghai date -d "@$((START_EPOCH + ETA_MIN))" '+%Y-%m-%d %H:%M:%S %Z')"

    echo "ETA_MAX_CST=$(TZ=Asia/Shanghai date -d "@$((START_EPOCH + ETA_MAX))" '+%Y-%m-%d %H:%M:%S %Z')"

    echo "ETA_MIN_UTC=$(TZ=UTC date -d "@$((START_EPOCH + ETA_MIN))" '+%Y-%m-%d %H:%M:%S %Z')"

    echo "ETA_MAX_UTC=$(TZ=UTC date -d "@$((START_EPOCH + ETA_MAX))" '+%Y-%m-%d %H:%M:%S %Z')"

    echo "ETA_CONFIDENCE=MEDIUM"

} > "$META"


echo "======================================================================"
echo "STRONG REPRO — OFFICIAL-STYLE PDS SMOKE V1"
echo "======================================================================"
echo
echo "预计运行时长：3–7 分钟"
echo "ETA confidence = MEDIUM"
echo
cat "$META"
echo


trap '
rc=$?
echo
echo "======================================================================"
echo "PDS OFFICIAL SMOKE FAILED"
echo "return_code=$rc"
echo "time=$(date -Iseconds)"
echo "======================================================================"
touch "'"$FAIL"'"
' ERR


###############################################################################
# STAGE 1/8 — frozen preflight + exact official config extraction
###############################################################################

echo "======================================================================"
echo "STAGE 1/8 — PREFLIGHT / OFFICIAL SPEC LOCK"
echo "======================================================================"

for F in \
    "$PARENTS" \
    "$PAIRS" \
    "$POP_PASS" \
    "$OFFICIAL_ANALYZER" \
    "$OFFICIAL_CASE" \
    "$MODEL/config.json" \
    "$WORKER"
do

    if [ ! -e "$F" ]; then
        echo "MISSING=$F"
        false
    fi

done


if [ "$(wc -l < "$PARENTS")" -ne 11792 ]; then
    echo "PARENT_COUNT_MISMATCH"
    false
fi


if [ "$(wc -l < "$PAIRS")" -ne 25000 ]; then
    echo "PAIR_COUNT_MISMATCH"
    false
fi


python - \
    "$OFFICIAL_ANALYZER" \
    "$OFFICIAL_CASE" \
    "$SPEC" <<'PY'

import ast
import json
import sys
from pathlib import Path

analyzer_path = Path(
    sys.argv[1]
)

case_path = Path(
    sys.argv[2]
)

out = Path(
    sys.argv[3]
)


def class_attrs(
    path,
    class_name,
):

    tree = ast.parse(
        path.read_text(
            encoding="utf-8",
        )
    )

    attrs = {}

    for node in tree.body:

        if (
            isinstance(node, ast.ClassDef)
            and
            node.name == class_name
        ):

            for stmt in node.body:

                if not isinstance(
                    stmt,
                    ast.Assign,
                ):
                    continue

                if len(stmt.targets) != 1:
                    continue

                target = stmt.targets[0]

                if not isinstance(
                    target,
                    ast.Name,
                ):
                    continue

                try:
                    attrs[
                        target.id
                    ] = ast.literal_eval(
                        stmt.value
                    )

                except Exception:
                    pass

    return attrs


analyzer = class_attrs(
    analyzer_path,
    "SentenceAnalyzerDataManager",
)

case = class_attrs(
    case_path,
    "CaseGenerationDataManager",
)


if not isinstance(
    analyzer.get("prompt"),
    str,
):
    raise RuntimeError(
        "official analyzer prompt not recovered"
    )


if not isinstance(
    case.get("prompt"),
    str,
):
    raise RuntimeError(
        "official case prompt not recovered"
    )


if case.get("num_case") != 4:
    raise RuntimeError(
        f"official num_case changed: "
        f"{case.get('num_case')}"
    )


if case.get("beam_size") != 1:
    raise RuntimeError(
        f"official beam_size changed: "
        f"{case.get('beam_size')}"
    )


if float(
    case.get("temperature")
) != 1.0:
    raise RuntimeError(
        f"official temperature changed: "
        f"{case.get('temperature')}"
    )


spec = {
    "official_sentence_analyzer": {
        "path":
            str(analyzer_path),

        "prompt":
            analyzer["prompt"],
    },

    "official_case_generation": {
        "path":
            str(case_path),

        "prompt":
            case["prompt"],

        "num_case":
            case["num_case"],

        "beam_size":
            case["beam_size"],

        "temperature":
            case["temperature"],
    },

    "strong_repro_adaptation": {
        "model":
            "Qwen3-8B",

        "language_pair":
            "Chinese -> English",

        "thinking":
            False,

        "analysis_decode":
            "greedy",

        "case_decode":
            (
                "sampling, temperature=1.0, "
                "num_beams=1"
            ),

        "case_max_new_tokens":
            256,

        "analysis_max_new_tokens":
            256,
    },
}


out.write_text(
    json.dumps(
        spec,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print(
    "OFFICIAL_NUM_CASE =",
    case["num_case"],
)

print(
    "OFFICIAL_BEAM_SIZE =",
    case["beam_size"],
)

print(
    "OFFICIAL_TEMPERATURE =",
    case["temperature"],
)

print(
    "OFFICIAL_SPEC_EXTRACTION_PASS"
)
PY


echo
echo "===== FROZEN UPSTREAM HASHES ====="

sha256sum \
    "$PARENTS" \
    "$PAIRS" \
    "$OFFICIAL_ANALYZER" \
    "$OFFICIAL_CASE" \
    "$SPEC"


###############################################################################
# STAGE 2/8 — deterministic 128-parent sample
###############################################################################

echo
echo "======================================================================"
echo "STAGE 2/8 — BUILD 128-PARENT SMOKE SAMPLE"
echo "======================================================================"

python - \
    "$PARENTS" \
    "$SPEC" \
    "$SAMPLE_IDS" \
    "$ANALYSIS_JOBS" \
    "$SAMPLE_N" \
    "$SEED" <<'PY'

import json
import random
import sys
from pathlib import Path

parents_path = Path(
    sys.argv[1]
)

spec_path = Path(
    sys.argv[2]
)

ids_path = Path(
    sys.argv[3]
)

jobs_path = Path(
    sys.argv[4]
)

sample_n = int(
    sys.argv[5]
)

seed = int(
    sys.argv[6]
)


parents = [
    json.loads(x)
    for x in parents_path.read_text(
        encoding="utf-8"
    ).splitlines()
    if x.strip()
]

spec = json.loads(
    spec_path.read_text(
        encoding="utf-8"
    )
)

prompt_template = (
    spec[
        "official_sentence_analyzer"
    ][
        "prompt"
    ]
)


rng = random.Random(
    seed
)

selected = sorted(
    rng.sample(
        parents,
        sample_n,
    ),
    key=lambda x: int(
        x["index"]
    ),
)


ids_path.write_text(
    "".join(
        f"{int(x['index'])}\n"
        for x in selected
    ),
    encoding="utf-8",
)


jobs = []

for job_id, row in enumerate(
    selected
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
            job_id,

        "parent_index":
            int(
                row["index"]
            ),

        "source":
            source,

        "prompt":
            prompt,

        "stage":
            "sentence_analysis",
    })


with jobs_path.open(
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
    "SMOKE_PARENT_ROWS =",
    len(selected),
)

print(
    "ANALYSIS_JOBS =",
    len(jobs),
)

print(
    "SMOKE_SAMPLE_BUILD_PASS"
)
PY


###############################################################################
# STAGE 3/8 — sentence analysis
###############################################################################

echo
echo "======================================================================"
echo "STAGE 3/8 — QWEN3-8B SENTENCE ANALYSIS"
echo "======================================================================"

AN_PIDS=()

for DEVICE in $(seq 0 15); do

    LOG="$LOG_DIR/analysis_${DEVICE}.log"

    python -u "$WORKER" \
        --input "$ANALYSIS_JOBS" \
        --output "$ANALYSIS_SHARDS/device_${DEVICE}.jsonl" \
        --model "$MODEL" \
        --device-id "$DEVICE" \
        --world-size 16 \
        --batch-size 8 \
        --max-new-tokens 256 \
        --mode analysis \
        --seed "$SEED" \
        > "$LOG" 2>&1 &

    AN_PIDS+=("$!")

done


AN_FAIL=0

for PID in "${AN_PIDS[@]}"; do

    if wait "$PID"; then
        :
    else
        AN_FAIL=1
    fi

done


if [ "$AN_FAIL" -ne 0 ]; then

    echo "SENTENCE_ANALYSIS_WORKER_FAILURE"

    for F in "$LOG_DIR"/analysis_*.log; do
        echo "===== $F ====="
        tail -60 "$F" || true
    done

    false
fi


###############################################################################
# STAGE 4/8 — merge analysis and build repo-exact ×4 PDS jobs
###############################################################################

echo
echo "======================================================================"
echo "STAGE 4/8 — MERGE ANALYSIS / BUILD ×4 CASE JOBS"
echo "======================================================================"

python - \
    "$ANALYSIS_JOBS" \
    "$ANALYSIS_SHARDS" \
    "$ANALYSIS_MERGED" \
    "$SAMPLE_IDS" \
    "$PAIRS" \
    "$SPEC" \
    "$PDS_JOBS" <<'PY'

import json
import sys
from pathlib import Path

analysis_jobs_path = Path(
    sys.argv[1]
)

shard_dir = Path(
    sys.argv[2]
)

merged_path = Path(
    sys.argv[3]
)

ids_path = Path(
    sys.argv[4]
)

pairs_path = Path(
    sys.argv[5]
)

spec_path = Path(
    sys.argv[6]
)

pds_jobs_path = Path(
    sys.argv[7]
)


expected = {
    int(
        json.loads(line)[
            "job_id"
        ]
    )
    for line in analysis_jobs_path.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
}


results = {}

for device in range(16):

    p = (
        shard_dir
        / f"device_{device}.jsonl"
    )

    if not p.exists():
        raise RuntimeError(
            f"missing analysis shard {p}"
        )

    for line in p.read_text(
        encoding="utf-8"
    ).splitlines():

        if not line.strip():
            continue

        row = json.loads(line)

        results[
            int(row["job_id"])
        ] = row


if set(results) != expected:

    raise RuntimeError(
        "analysis result IDs mismatch: "
        f"expected={len(expected)} "
        f"got={len(results)}"
    )


ordered = [
    results[x]
    for x in sorted(results)
]


with merged_path.open(
    "w",
    encoding="utf-8",
) as f:

    for row in ordered:

        if not str(
            row.get(
                "raw_generation",
                "",
            )
        ).strip():

            raise RuntimeError(
                "empty sentence analysis "
                f"parent={row['parent_index']}"
            )

        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


analysis_by_parent = {
    int(x["parent_index"]):
        str(
            x["raw_generation"]
        ).strip()

    for x in ordered
}


selected_ids = {
    int(x)
    for x in ids_path.read_text(
        encoding="utf-8"
    ).splitlines()
    if x.strip()
}


pairs = [
    json.loads(x)
    for x in pairs_path.read_text(
        encoding="utf-8"
    ).splitlines()
    if x.strip()
]


pairs = [
    x
    for x in pairs
    if int(
        x["parent_index"]
    )
    in selected_ids
]


spec = json.loads(
    spec_path.read_text(
        encoding="utf-8"
    )
)

cg = spec[
    "official_case_generation"
]

num_case = int(
    cg["num_case"]
)

if num_case != 4:
    raise RuntimeError(
        f"expected repo num_case=4, "
        f"got {num_case}"
    )

prompt_template = cg[
    "prompt"
]


jobs = []
job_id = 0

for pair in pairs:

    parent_index = int(
        pair["parent_index"]
    )

    analysis = (
        analysis_by_parent[
            parent_index
        ]
    )

    P = str(
        pair["source_span"]
    ).strip()

    Q = str(
        pair["correction"]
    ).strip()

    word_pair = (
        f"{P}({Q})"
    )

    prompt = (
        prompt_template
        .replace(
            "<domain_topic_style>",
            analysis,
        )
        .replace(
            "<word_pair>",
            word_pair,
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

    for slot in range(
        num_case
    ):

        jobs.append({
            "job_id":
                job_id,

            "parent_index":
                parent_index,

            "pair_id":
                int(
                    pair["pair_id"]
                ),

            "error_index":
                int(
                    pair["error_index"]
                ),

            "pds_slot":
                slot,

            "source_span":
                P,

            "correction":
                Q,

            "original_source":
                str(
                    pair["source"]
                ).strip(),

            "parent_post_edit":
                str(
                    pair["post_edit"]
                ).strip(),

            "sentence_analysis":
                analysis,

            "word_pair":
                word_pair,

            "prompt":
                prompt,

            "construction_method":
                (
                    "STRONG_REPRO_OFFICIAL_"
                    "PDS_SMOKE_V1"
                ),

            "repo_num_case":
                num_case,
        })

        job_id += 1


if len(jobs) != (
    len(pairs)
    *
    num_case
):

    raise RuntimeError(
        "repo ×4 cardinality failure"
    )


with pds_jobs_path.open(
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
    "SMOKE_LOCAL_PAIRS =",
    len(pairs),
)

print(
    "REPO_NUM_CASE =",
    num_case,
)

print(
    "PDS_SMOKE_RAW_JOBS =",
    len(jobs),
)

print(
    "EXPECTED = LOCAL_PAIRS * 4"
)

print(
    "PDS_JOB_BUILD_PASS"
)
PY


###############################################################################
# STAGE 5/8 — CaseGeneration
###############################################################################

echo
echo "======================================================================"
echo "STAGE 5/8 — QWEN3-8B CASE GENERATION"
echo "======================================================================"

PDS_PIDS=()

for DEVICE in $(seq 0 15); do

    LOG="$LOG_DIR/pds_${DEVICE}.log"

    python -u "$WORKER" \
        --input "$PDS_JOBS" \
        --output "$PDS_SHARDS/device_${DEVICE}.jsonl" \
        --model "$MODEL" \
        --device-id "$DEVICE" \
        --world-size 16 \
        --batch-size 8 \
        --max-new-tokens 256 \
        --mode case \
        --seed "$SEED" \
        > "$LOG" 2>&1 &

    PDS_PIDS+=("$!")

done


PDS_FAIL=0

for PID in "${PDS_PIDS[@]}"; do

    if wait "$PID"; then
        :
    else
        PDS_FAIL=1
    fi

done


if [ "$PDS_FAIL" -ne 0 ]; then

    echo "PDS_WORKER_FAILURE"

    for F in "$LOG_DIR"/pds_*.log; do
        echo "===== $F ====="
        tail -60 "$F" || true
    done

    false
fi


###############################################################################
# STAGE 6/8 — merge / parse / mechanical validity gate
###############################################################################

echo
echo "======================================================================"
echo "STAGE 6/8 — PARSE + P/Q + NEW-CONTEXT + DEDUP"
echo "======================================================================"

python - \
    "$PDS_JOBS" \
    "$PDS_SHARDS" \
    "$PDS_RAW" \
    "$PDS_VALID" \
    "$AUDIT" <<'PY'

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

jobs_path = Path(
    sys.argv[1]
)

shard_dir = Path(
    sys.argv[2]
)

raw_path = Path(
    sys.argv[3]
)

valid_path = Path(
    sys.argv[4]
)

audit_path = Path(
    sys.argv[5]
)


def norm(s):

    s = unicodedata.normalize(
        "NFKC",
        str(s),
    ).lower()

    return "".join(
        ch
        for ch in s
        if ch.isalnum()
    )


def strip_md(s):

    s = str(s).strip()

    s = re.sub(
        r"^[\-\*\s]+",
        "",
        s,
    )

    s = s.strip("* ")

    return s.strip()


def parse_pair(text):

    text = str(text).strip()

    lines = [
        x.strip()
        for x in text.splitlines()
        if x.strip()
    ]

    cn = ""
    en = ""

    cn_patterns = [
        r"^(?:[-*]\s*)?"
        r"(?:\*\*)?"
        r"(?:Chinese Sentence|Chinese|"
        r"Sentence X\s*\(中文\)|"
        r"中文句子|中文)"
        r"(?:\*\*)?"
        r"\s*[:：]\s*(.*)$",
    ]

    en_patterns = [
        r"^(?:[-*]\s*)?"
        r"(?:\*\*)?"
        r"(?:English Sentence|English|"
        r"Sentence Y\s*\(英文\)|"
        r"英文句子|英文)"
        r"(?:\*\*)?"
        r"\s*[:：]\s*(.*)$",
    ]


    for i, line in enumerate(lines):

        for pat in cn_patterns:

            m = re.match(
                pat,
                line,
                flags=re.I,
            )

            if m:

                cn = strip_md(
                    m.group(1)
                )

                if (
                    not cn
                    and
                    i + 1 < len(lines)
                ):
                    cn = strip_md(
                        lines[i + 1]
                    )

                break


        for pat in en_patterns:

            m = re.match(
                pat,
                line,
                flags=re.I,
            )

            if m:

                en = strip_md(
                    m.group(1)
                )

                if (
                    not en
                    and
                    i + 1 < len(lines)
                ):
                    en = strip_md(
                        lines[i + 1]
                    )

                break


    # Fallback for two plain sentence lines.
    if not cn or not en:

        candidates = [
            strip_md(x)
            for x in lines
            if not x.startswith(
                "#"
            )
            and x != "---"
        ]

        if not cn:

            for x in candidates:

                cjk = sum(
                    "\u4e00"
                    <= ch
                    <= "\u9fff"
                    for ch in x
                )

                if cjk >= 4:
                    cn = x
                    break


        if not en:

            for x in candidates:

                latin = sum(
                    ch.isascii()
                    and
                    ch.isalpha()
                    for ch in x
                )

                if latin >= 15:
                    en = x
                    break


    return (
        cn.strip(),
        en.strip(),
    )


jobs = {}

for line in jobs_path.read_text(
    encoding="utf-8"
).splitlines():

    if not line.strip():
        continue

    x = json.loads(line)

    jobs[
        int(x["job_id"])
    ] = x


results = {}

for device in range(16):

    p = (
        shard_dir
        / f"device_{device}.jsonl"
    )

    if not p.exists():
        raise RuntimeError(
            f"missing PDS shard {p}"
        )

    for line in p.read_text(
        encoding="utf-8"
    ).splitlines():

        if not line.strip():
            continue

        x = json.loads(line)

        results[
            int(x["job_id"])
        ] = x


if set(jobs) != set(results):

    raise RuntimeError(
        "PDS result ID mismatch: "
        f"jobs={len(jobs)} "
        f"results={len(results)}"
    )


ordered = [
    results[i]
    for i in sorted(results)
]


with raw_path.open(
    "w",
    encoding="utf-8",
) as f:

    for x in ordered:

        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )


counts = Counter()

valid = []

seen_pairs = set()

examples = {
    "parse_fail": [],
    "missing_P": [],
    "missing_Q": [],
    "same_as_parent": [],
    "duplicate_pair": [],
}


for result in ordered:

    jid = int(
        result["job_id"]
    )

    job = jobs[jid]

    raw = str(
        result.get(
            "raw_generation",
            "",
        )
    )

    src, tgt = (
        parse_pair(raw)
    )

    counts[
        "total"
    ] += 1


    if not src or not tgt:

        counts[
            "parse_fail"
        ] += 1

        if len(
            examples["parse_fail"]
        ) < 10:

            examples[
                "parse_fail"
            ].append({
                "job_id":
                    jid,

                "raw":
                    raw[:1500],
            })

        continue


    counts[
        "parsed"
    ] += 1


    P = str(
        job["source_span"]
    ).strip()

    Q = str(
        job["correction"]
    ).strip()

    original = str(
        job["original_source"]
    ).strip()


    p_ok = (
        norm(P)
        in
        norm(src)
    )

    q_ok = (
        norm(Q)
        in
        norm(tgt)
    )

    new_ok = (
        norm(src)
        !=
        norm(original)
    )


    if p_ok:
        counts["P_ok"] += 1

    else:

        if len(
            examples["missing_P"]
        ) < 10:

            examples[
                "missing_P"
            ].append({
                "job_id":
                    jid,

                "P":
                    P,

                "src":
                    src,

                "raw":
                    raw[:1200],
            })


    if q_ok:
        counts["Q_ok"] += 1

    else:

        if len(
            examples["missing_Q"]
        ) < 10:

            examples[
                "missing_Q"
            ].append({
                "job_id":
                    jid,

                "Q":
                    Q,

                "tgt":
                    tgt,

                "raw":
                    raw[:1200],
            })


    if new_ok:
        counts[
            "new_context"
        ] += 1

    else:

        if len(
            examples[
                "same_as_parent"
            ]
        ) < 10:

            examples[
                "same_as_parent"
            ].append({
                "job_id":
                    jid,

                "src":
                    src,

                "parent":
                    original,
            })


    if not (
        p_ok
        and
        q_ok
        and
        new_ok
    ):
        continue


    pair_key = (
        norm(src),
        norm(tgt),
    )

    if pair_key in seen_pairs:

        counts[
            "duplicate_pair"
        ] += 1

        if len(
            examples[
                "duplicate_pair"
            ]
        ) < 10:

            examples[
                "duplicate_pair"
            ].append({
                "job_id":
                    jid,

                "src":
                    src,

                "tgt":
                    tgt,
            })

        continue


    seen_pairs.add(
        pair_key
    )


    counts[
        "valid"
    ] += 1


    valid.append({
        "job_id":
            jid,

        "parent_index":
            job["parent_index"],

        "pair_id":
            job["pair_id"],

        "error_index":
            job["error_index"],

        "pds_slot":
            job["pds_slot"],

        "source":
            src,

        "messages": [
            {
                "role":
                    "user",

                "content":
                    (
                        "Translate the following "
                        "text into English without "
                        "additional explanations:"
                        "\n\n"
                        + src
                        + "\n\n"
                    ),
            }
        ],

        "target_translation":
            tgt,

        "source_span":
            P,

        "correction":
            Q,

        "original_source":
            original,

        "sentence_analysis":
            job[
                "sentence_analysis"
            ],

        "construction_method":
            (
                "STRONG_REPRO_OFFICIAL_"
                "PDS_SMOKE_VALID_V1"
            ),
    })


with valid_path.open(
    "w",
    encoding="utf-8",
) as f:

    for x in valid:

        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )


total = counts["total"]
parsed = counts["parsed"]

parse_rate = (
    parsed / total
    if total
    else 0.0
)

p_rate = (
    counts["P_ok"]
    / parsed
    if parsed
    else 0.0
)

q_rate = (
    counts["Q_ok"]
    / parsed
    if parsed
    else 0.0
)

new_rate = (
    counts["new_context"]
    / parsed
    if parsed
    else 0.0
)

valid_rate = (
    counts["valid"]
    / total
    if total
    else 0.0
)


audit = {
    "protocol":
        "STRONG_REPRO_PDS_OFFICIAL_SMOKE_V1",

    "generation_only":
        True,

    "student_training_started":
        False,

    "counts":
        dict(counts),

    "rates": {
        "parse_rate":
            parse_rate,

        "P_containment_given_parse":
            p_rate,

        "Q_containment_given_parse":
            q_rate,

        "new_context_given_parse":
            new_rate,

        "final_valid_rate":
            valid_rate,
    },

    "pass_gates": {
        "parse_rate":
            0.90,

        "P_containment_given_parse":
            0.85,

        "Q_containment_given_parse":
            0.80,

        "new_context_given_parse":
            0.80,

        "final_valid_rate":
            0.50,
    },

    "examples":
        examples,
}


audit_path.write_text(
    json.dumps(
        audit,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print(
    json.dumps(
        audit,
        ensure_ascii=False,
        indent=2,
    )
)


print()
print(
    "PARSE_RATE =",
    parse_rate,
)

print(
    "P_CONTAINMENT =",
    p_rate,
)

print(
    "Q_CONTAINMENT =",
    q_rate,
)

print(
    "NEW_CONTEXT_RATE =",
    new_rate,
)

print(
    "FINAL_VALID_RATE =",
    valid_rate,
)

print(
    "FINAL_VALID_ROWS =",
    len(valid),
)


if parse_rate < 0.90:
    raise RuntimeError(
        "SMOKE_FAIL parse_rate"
    )

if p_rate < 0.85:
    raise RuntimeError(
        "SMOKE_FAIL P containment"
    )

if q_rate < 0.80:
    raise RuntimeError(
        "SMOKE_FAIL Q containment"
    )

if new_rate < 0.80:
    raise RuntimeError(
        "SMOKE_FAIL new-context rate"
    )

if valid_rate < 0.50:
    raise RuntimeError(
        "SMOKE_FAIL final-valid rate"
    )


print(
    "PDS_OFFICIAL_SMOKE_QUALITY_GATE_PASS"
)
PY


###############################################################################
# STAGE 7/8 — provenance freeze
###############################################################################

echo
echo "======================================================================"
echo "STAGE 7/8 — FREEZE SMOKE PROVENANCE"
echo "======================================================================"

{
    echo "DATE=$(date -Iseconds)"

    echo
    echo "===== INPUTS ====="

    sha256sum \
        "$PARENTS" \
        "$PAIRS" \
        "$OFFICIAL_ANALYZER" \
        "$OFFICIAL_CASE"

    echo
    echo "===== SPEC / SAMPLE ====="

    sha256sum \
        "$SPEC" \
        "$SAMPLE_IDS" \
        "$ANALYSIS_JOBS" \
        "$ANALYSIS_MERGED" \
        "$PDS_JOBS"

    echo
    echo "===== OUTPUT ====="

    sha256sum \
        "$PDS_RAW" \
        "$PDS_VALID" \
        "$AUDIT"

} > "$OUT/frozen_sha256_manifest_v1.txt"

cat "$OUT/frozen_sha256_manifest_v1.txt"


###############################################################################
# STAGE 8/8 — finalize
###############################################################################

echo
echo "======================================================================"
echo "STAGE 8/8 — FINALIZE"
echo "======================================================================"

touch "$PASS"
rm -f "$FAIL"

END_EPOCH="$(date +%s)"

ELAPSED=$(
    (
        END_EPOCH
        -
        START_EPOCH
    )
)

echo
echo "======================================================================"
echo "STRONG REPRO PDS OFFICIAL SMOKE V1 PASS"
echo "======================================================================"
echo "TOTAL_ELAPSED_SECONDS=$ELAPSED"
echo "FINISH_CST=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
echo "FINISH_UTC=$(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
echo
echo "PDS_JOBS=$PDS_JOBS"
echo "PDS_VALID=$PDS_VALID"
echo "AUDIT=$AUDIT"
echo "PASS=$PASS"
echo "======================================================================"
