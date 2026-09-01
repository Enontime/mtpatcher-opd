#!/usr/bin/env bash
set -Eeuo pipefail

source /workspace/mtpatcher/project_env.sh

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

EXP="mtpatcher_v3_full6565_20260823"
D="$DATA_ROOT/$EXP"

BASE="$D/strong_repro_pds_official_smoke_v1"

ANALYSIS="$BASE/sentence_analysis128_v1.jsonl"
OLD_JOBS="$BASE/pds_smoke_jobs_v1.jsonl"
OLD_RAW="$BASE/pds_smoke_raw_v1.jsonl"
SPEC="$BASE/official_pds_spec_v1.json"

WORKER="$ROOT/scripts/mtpatcher_v3/strong_repro_pds_qwen_worker_v1.py"
MODEL="$MODEL_ROOT/Qwen3-8B"

OUT="$D/pds_smoke_analysis_clean_ablation_v3"
LOG_DIR="$LOG_ROOT/$EXP/pds_smoke_analysis_clean_ablation_v3"

CANON="$OUT/sentence_analysis_topic_domain_style128_v2.jsonl"
NEW_JOBS="$OUT/pds_jobs_analysis_clean1100_v2.jsonl"
SHARDS="$OUT/pds_shards16"
NEW_RAW="$OUT/pds_raw_analysis_clean1100_v2.jsonl"

REPORT="$OUT/analysis_clean_ablation_report_v2.json"
HASHES="$OUT/frozen_sha256_manifest_v2.txt"
META="$OUT/start_meta_v2.txt"

PASS="$OUT/PDS_SMOKE_ANALYSIS_CLEAN_ABLATION_V3.PASS"
FAIL="$OUT/PDS_SMOKE_ANALYSIS_CLEAN_ABLATION_V3.FAIL"

mkdir -p "$OUT" "$LOG_DIR" "$SHARDS"
rm -f "$PASS" "$FAIL"

START_EPOCH="$(date +%s)"
ETA_MIN=150
ETA_MAX=270

{
    echo "START_EPOCH=$START_EPOCH"
    echo "START_CST=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "START_UTC=$(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "ETA_MIN_CST=$(TZ=Asia/Shanghai date -d "@$((START_EPOCH + ETA_MIN))" '+%Y-%m-%d %H:%M:%S %Z')"
    echo "ETA_MAX_CST=$(TZ=Asia/Shanghai date -d "@$((START_EPOCH + ETA_MAX))" '+%Y-%m-%d %H:%M:%S %Z')"
    echo "ETA_MIN_UTC=$(TZ=UTC date -d "@$((START_EPOCH + ETA_MIN))" '+%Y-%m-%d %H:%M:%S %Z')"
    echo "ETA_MAX_UTC=$(TZ=UTC date -d "@$((START_EPOCH + ETA_MAX))" '+%Y-%m-%d %H:%M:%S %Z')"
} > "$META"

echo "======================================================================"
echo "PDS ANALYZER-CLEAN MATCHED ABLATION V2"
echo "======================================================================"
echo "预计运行时长：2.5–4.5 分钟"
cat "$META"
echo

trap '
rc=$?
echo
echo "======================================================================"
echo "ANALYZER CLEAN ABLATION V2 FAILED"
echo "return_code=$rc"
echo "time=$(date -Iseconds)"
echo "======================================================================"
touch "'"$FAIL"'"
' ERR


###############################################################################
# STAGE 1 — PREFLIGHT
###############################################################################

echo "======================================================================"
echo "STAGE 1/5 — PREFLIGHT"
echo "======================================================================"

for F in \
    "$ANALYSIS" \
    "$OLD_JOBS" \
    "$OLD_RAW" \
    "$SPEC" \
    "$WORKER" \
    "$MODEL/config.json"
do
    if [ ! -f "$F" ]; then
        echo "MISSING=$F"
        false
    fi
done

[ "$(wc -l < "$ANALYSIS")" -eq 128 ] || false
[ "$(wc -l < "$OLD_JOBS")" -eq 1100 ] || false
[ "$(wc -l < "$OLD_RAW")" -eq 1100 ] || false

python -m py_compile "$WORKER"

echo "PREFLIGHT_PASS"


###############################################################################
# STAGE 2 — ROBUST ANALYZER CANONICALIZATION
###############################################################################

echo
echo "======================================================================"
echo "STAGE 2/5 — CANONICALIZE TOPIC / DOMAIN / STYLE"
echo "======================================================================"

python - \
    "$ANALYSIS" \
    "$OLD_JOBS" \
    "$SPEC" \
    "$CANON" \
    "$NEW_JOBS" <<'PY'

import json
import re
import sys
import unicodedata
from pathlib import Path

ANALYSIS = Path(sys.argv[1])
OLD_JOBS = Path(sys.argv[2])
SPEC = Path(sys.argv[3])
CANON = Path(sys.argv[4])
NEW_JOBS = Path(sys.argv[5])


def load(path):
    rows = []

    with path.open(
        encoding="utf-8-sig",
        errors="replace",
    ) as f:

        for n, line in enumerate(f, 1):

            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))

            except Exception as e:
                raise RuntimeError(
                    f"{path}:{n}: {e}"
                )

    return rows


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


def strip_markup(s):

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

    s = s.replace("**", "")
    s = s.replace("__", "")

    return s.strip()


def heading_form(s):

    s = strip_markup(s)

    s = re.sub(
        r"^\s*\d+\s*"
        r"(?:[.)、:：-])?\s*",
        "",
        s,
    )

    return strip_markup(s)


LABELS = {
    "topic": re.compile(
        r"^(?:Topic(?:\s*[\(（]主题[\)）])?|主题)"
        r"\s*[:：]\s*(.*)$",
        re.I,
    ),

    "domain": re.compile(
        r"^(?:Domain(?:\s*[\(（]领域[\)）])?|领域)"
        r"\s*[:：]\s*(.*)$",
        re.I,
    ),

    "style": re.compile(
        r"^(?:Style(?:\s*[\(（]风格[\)）])?|风格)"
        r"\s*[:：]\s*(.*)$",
        re.I,
    ),
}


def is_heading(line):

    s = heading_form(line)

    return any(
        rx.match(s)
        for rx in LABELS.values()
    )


def next_value(lines, start):

    for j in range(
        start,
        min(len(lines), start + 8),
    ):

        raw = lines[j]

        if not raw.strip():
            continue

        if is_heading(raw):
            return ""

        value = strip_markup(raw)

        if not value:
            continue

        if value == "---":
            continue

        if re.match(
            r"^(?:Explanation|说明|解释|Summary)"
            r"\s*[:：]?",
            value,
            flags=re.I,
        ):
            return ""

        return value

    return ""


def extract_tds(raw):

    lines = str(raw).splitlines()

    found = {}

    for i, raw_line in enumerate(lines):

        line = heading_form(
            raw_line
        )

        for key, rx in LABELS.items():

            if key in found:
                continue

            m = rx.match(line)

            if not m:
                continue

            value = strip_markup(
                m.group(1)
            )

            if not value:
                value = next_value(
                    lines,
                    i + 1,
                )

            if value:
                found[key] = value

    return found


analysis_rows = load(
    ANALYSIS
)

old_jobs = load(
    OLD_JOBS
)

spec = json.loads(
    SPEC.read_text(
        encoding="utf-8"
    )
)

prompt_template = (
    spec[
        "official_case_generation"
    ][
        "prompt"
    ]
)


parent_source = {}

for row in old_jobs:

    pid = int(
        row["parent_index"]
    )

    src = str(
        row["original_source"]
    ).strip()

    if (
        pid in parent_source
        and
        parent_source[pid] != src
    ):
        raise RuntimeError(
            f"parent source mismatch {pid}"
        )

    parent_source[pid] = src


canon = {}
failures = []

leak_before = 0
leak_after = 0


for row in analysis_rows:

    pid = int(
        row["parent_index"]
    )

    raw = str(
        row["raw_generation"]
    ).strip()

    parent = parent_source[pid]

    if (
        norm(parent)
        and
        norm(parent) in norm(raw)
    ):
        leak_before += 1


    tds = extract_tds(
        raw
    )

    missing = [
        k
        for k in (
            "topic",
            "domain",
            "style",
        )
        if not tds.get(k)
    ]


    if missing:

        failures.append({
            "parent_index":
                pid,

            "missing":
                missing,

            "raw":
                raw[:2200],
        })

        continue


    canonical = (
        f"Topic: {tds['topic']}\n"
        f"Domain: {tds['domain']}\n"
        f"Style: {tds['style']}"
    )


    if (
        norm(parent)
        and
        norm(parent)
        in norm(canonical)
    ):
        leak_after += 1


    canon[pid] = {
        "parent_index":
            pid,

        "source":
            parent,

        "raw_sentence_analysis":
            raw,

        "topic":
            tds["topic"],

        "domain":
            tds["domain"],

        "style":
            tds["style"],

        "canonical_sentence_analysis":
            canonical,
    }


print(
    "ANALYSIS_ROWS =",
    len(analysis_rows),
)

print(
    "STRUCTURED_PARSE_SUCCESS =",
    len(canon),
)

print(
    "STRUCTURED_PARSE_FAILURE =",
    len(failures),
)

print(
    "FULL_PARENT_LEAK_BEFORE =",
    leak_before,
)

print(
    "FULL_PARENT_LEAK_AFTER =",
    leak_after,
)


if failures:

    print()
    print("FIRST_FAILURES =")

    for x in failures[:20]:

        print(
            json.dumps(
                x,
                ensure_ascii=False,
            )
        )

    raise RuntimeError(
        "canonicalization incomplete; "
        "NO generation authorized"
    )


if len(canon) != 128:

    raise RuntimeError(
        f"expected128 analyses "
        f"got={len(canon)}"
    )


if leak_after != 0:

    raise RuntimeError(
        f"canonical analysis still leaks "
        f"full parent: {leak_after}"
    )


with CANON.open(
    "w",
    encoding="utf-8",
) as f:

    for pid in sorted(canon):

        f.write(
            json.dumps(
                canon[pid],
                ensure_ascii=False,
            )
            + "\n"
        )


new_jobs = []

for old in sorted(
    old_jobs,
    key=lambda x:
        int(x["job_id"]),
):

    pid = int(
        old["parent_index"]
    )

    analysis = (
        canon[pid][
            "canonical_sentence_analysis"
        ]
    )

    P = str(
        old["source_span"]
    ).strip()

    Q = str(
        old["correction"]
    ).strip()


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


    new = dict(
        old
    )

    new[
        "original_sentence_analysis"
    ] = old[
        "sentence_analysis"
    ]

    new[
        "sentence_analysis"
    ] = analysis

    new[
        "prompt"
    ] = prompt

    new[
        "analysis_treatment"
    ] = (
        "TOPIC_DOMAIN_STYLE_ONLY_V3"
    )

    new_jobs.append(
        new
    )


if len(new_jobs) != 1100:

    raise RuntimeError(
        f"expected1100 jobs "
        f"got={len(new_jobs)}"
    )


old_sorted = sorted(
    old_jobs,
    key=lambda x:
        int(x["job_id"]),
)


for old, new in zip(
    old_sorted,
    new_jobs,
):

    for key in (
        "job_id",
        "parent_index",
        "pair_id",
        "error_index",
        "pds_slot",
        "source_span",
        "correction",
        "original_source",
        "repo_num_case",
    ):

        if (
            old.get(key)
            !=
            new.get(key)
        ):

            raise RuntimeError(
                "causal invariant changed: "
                f"job={old['job_id']} "
                f"field={key}"
            )


with NEW_JOBS.open(
    "w",
    encoding="utf-8",
) as f:

    for row in new_jobs:

        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


print(
    "NEW_JOB_ROWS =",
    len(new_jobs),
)

print(
    "JOB_IDENTITY_INVARIANTS_PASS"
)

print(
    "ANALYZER_CANONICALIZATION_128_PASS"
)
PY


###############################################################################
# STAGE 3 — MATCHED CASE GENERATION
###############################################################################

echo
echo "======================================================================"
echo "STAGE 3/5 — MATCHED CASE GENERATION"
echo "======================================================================"

PIDS=()

for DEVICE in $(seq 0 15); do

    LOG="$LOG_DIR/device_${DEVICE}.log"

    python -u "$WORKER" \
        --input "$NEW_JOBS" \
        --output "$SHARDS/device_${DEVICE}.jsonl" \
        --model "$MODEL" \
        --device-id "$DEVICE" \
        --world-size 16 \
        --batch-size 8 \
        --max-new-tokens 256 \
        --mode case \
        --seed 20260831 \
        > "$LOG" 2>&1 &

    PIDS+=("$!")
done


WORKER_FAIL=0

for PID in "${PIDS[@]}"; do

    if wait "$PID"; then
        :
    else
        WORKER_FAIL=1
    fi
done


if [ "$WORKER_FAIL" -ne 0 ]; then

    echo "CASE_GENERATION_WORKER_FAILURE"

    for F in "$LOG_DIR"/device_*.log; do
        echo "===== $F ====="
        tail -60 "$F" || true
    done

    false
fi


###############################################################################
# STAGE 4 — MATCHED RAW-LEVEL EVALUATION
###############################################################################

echo
echo "======================================================================"
echo "STAGE 4/5 — MATCHED RAW-LEVEL EVALUATION"
echo "======================================================================"

python - \
    "$OLD_JOBS" \
    "$OLD_RAW" \
    "$NEW_JOBS" \
    "$SHARDS" \
    "$NEW_RAW" \
    "$REPORT" <<'PY'

import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

OLD_JOBS = Path(sys.argv[1])
OLD_RAW = Path(sys.argv[2])
NEW_JOBS = Path(sys.argv[3])
SHARDS = Path(sys.argv[4])
NEW_RAW = Path(sys.argv[5])
REPORT = Path(sys.argv[6])


def load(path):

    rows = []

    with path.open(
        encoding="utf-8-sig",
        errors="replace",
    ) as f:

        for n, line in enumerate(f, 1):

            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))

            except Exception as e:
                raise RuntimeError(
                    f"{path}:{n}: {e}"
                )

    return rows


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


def q_words(s):

    return len(
        re.findall(
            r"[A-Za-z0-9]+"
            r"(?:['’-][A-Za-z0-9]+)?",
            str(s),
        )
    )


def qbin(q):

    n = q_words(q)

    if n <= 2:
        return "01_1-2"

    if n <= 5:
        return "02_3-5"

    if n <= 10:
        return "03_6-10"

    if n <= 20:
        return "04_11-20"

    return "05_21plus"


old_jobs_rows = load(
    OLD_JOBS
)

new_jobs_rows = load(
    NEW_JOBS
)

old_raw_rows = load(
    OLD_RAW
)


new_physical = []

for device in range(16):

    path = (
        SHARDS
        / f"device_{device}.jsonl"
    )

    if not path.exists():

        raise RuntimeError(
            f"missing shard={path}"
        )

    new_physical.extend(
        load(path)
    )


old_jobs = {
    int(x["job_id"]): x
    for x in old_jobs_rows
}

new_jobs = {
    int(x["job_id"]): x
    for x in new_jobs_rows
}

old_raw = {
    int(x["job_id"]): x
    for x in old_raw_rows
}


new_buckets = defaultdict(list)

for x in new_physical:

    new_buckets[
        int(x["job_id"])
    ].append(x)


duplicate_physical = sum(
    max(
        0,
        len(v) - 1,
    )
    for v in new_buckets.values()
)


new_raw = {
    jid:
        rows[-1]

    for jid, rows
    in new_buckets.items()
}


expected = set(
    old_jobs
)


for name, obj in (
    ("new_jobs", new_jobs),
    ("old_raw", old_raw),
    ("new_raw", new_raw),
):

    if set(obj) != expected:

        raise RuntimeError(
            f"{name} ID mismatch "
            f"expected={len(expected)} "
            f"got={len(obj)}"
        )


def evaluate(
    jobs,
    raw,
):

    counts = Counter()
    pair_jobs = defaultdict(list)
    qgroups = defaultdict(Counter)

    for jid in sorted(jobs):

        job = jobs[jid]

        text = str(
            raw[jid].get(
                "raw_generation",
                "",
            )
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


        p_ok = (
            bool(norm(P))
            and
            norm(P) in norm(text)
        )

        q_ok = (
            bool(norm(Q))
            and
            norm(Q) in norm(text)
        )

        parent_copy = (
            bool(norm(parent))
            and
            norm(parent) in norm(text)
        )


        counts["n"] += 1
        counts["P"] += int(p_ok)
        counts["Q"] += int(q_ok)
        counts["both"] += int(
            p_ok and q_ok
        )
        counts["copy"] += int(
            parent_copy
        )


        g = qgroups[
            qbin(Q)
        ]

        g["n"] += 1
        g["Q"] += int(q_ok)
        g["both"] += int(
            p_ok and q_ok
        )
        g["copy"] += int(
            parent_copy
        )


        pair_jobs[
            int(
                job["pair_id"]
            )
        ].append(jid)


    n = counts["n"]

    overall = {
        "n":
            n,

        "P_rate":
            counts["P"] / n,

        "Q_rate":
            counts["Q"] / n,

        "both_rate":
            counts["both"] / n,

        "parent_copy_rate":
            counts["copy"] / n,

        "parent_copy_count":
            counts["copy"],
    }


    q_strat = {}

    for key, g in sorted(
        qgroups.items()
    ):

        q_strat[key] = {
            "n":
                g["n"],

            "Q_rate":
                g["Q"] / g["n"],

            "both_rate":
                g["both"] / g["n"],

            "copy_rate":
                g["copy"] / g["n"],
        }


    pair_counts = Counter()

    for pair_id, jids in pair_jobs.items():

        if len(jids) != 4:

            raise RuntimeError(
                f"pair={pair_id} "
                f"jobs={len(jids)}"
            )


        qn = 0
        bn = 0

        for jid in jids:

            job = jobs[jid]

            text = str(
                raw[jid].get(
                    "raw_generation",
                    "",
                )
            )

            P = str(
                job["source_span"]
            ).strip()

            Q = str(
                job["correction"]
            ).strip()

            p_ok = (
                norm(P)
                in
                norm(text)
            )

            q_ok = (
                norm(Q)
                in
                norm(text)
            )

            qn += int(q_ok)
            bn += int(
                p_ok and q_ok
            )


        pair_counts[
            f"q_{qn}"
        ] += 1

        pair_counts[
            f"b_{bn}"
        ] += 1

        pair_counts[
            "any_q"
        ] += int(
            qn > 0
        )

        pair_counts[
            "any_b"
        ] += int(
            bn > 0
        )


    npairs = len(
        pair_jobs
    )


    pair = {
        "pairs":
            npairs,

        "any_Q_of4":
            pair_counts["any_q"]
            / npairs,

        "any_both_of4":
            pair_counts["any_b"]
            / npairs,

        "Q_slot_hist": {
            str(i):
                pair_counts[
                    f"q_{i}"
                ]
            for i in range(5)
        },

        "both_slot_hist": {
            str(i):
                pair_counts[
                    f"b_{i}"
                ]
            for i in range(5)
        },
    }


    return (
        overall,
        q_strat,
        pair,
    )


old_overall, old_q, old_pair = (
    evaluate(
        old_jobs,
        old_raw,
    )
)

new_overall, new_q, new_pair = (
    evaluate(
        new_jobs,
        new_raw,
    )
)


with NEW_RAW.open(
    "w",
    encoding="utf-8",
) as f:

    for jid in sorted(
        new_raw
    ):

        f.write(
            json.dumps(
                new_raw[jid],
                ensure_ascii=False,
            )
            + "\n"
        )


delta = {
    key:
        (
            new_overall[key]
            -
            old_overall[key]
        )

    for key in (
        "P_rate",
        "Q_rate",
        "both_rate",
        "parent_copy_rate",
    )
}


report = {
    "protocol":
        "PDS_SMOKE_ANALYSIS_CLEAN_MATCHED_ABLATION_V3",

    "primary_question":
        (
            "Does full-parent leakage through "
            "sentence_analysis cause parent copying?"
        ),

    "old":
        {
            "overall":
                old_overall,

            "Q_length":
                old_q,

            "pair":
                old_pair,
        },

    "analysis_clean":
        {
            "overall":
                new_overall,

            "Q_length":
                new_q,

            "pair":
                new_pair,
        },

    "delta_new_minus_old":
        delta,

    "duplicate_physical_new_rows":
        duplicate_physical,

    "student_training_started":
        False,
}


REPORT.write_text(
    json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print("=" * 78)
print(
    "ANALYZER CLEAN MATCHED ABLATION V2 RESULT"
)
print("=" * 78)

print()
print("===== OLD =====")

for k, v in old_overall.items():
    print(k, "=", v)

print()
print("===== ANALYSIS CLEAN =====")

for k, v in new_overall.items():
    print(k, "=", v)

print()
print("===== DELTA NEW - OLD =====")

for k, v in delta.items():
    print(k, "=", v)

print()
print("===== PAIR LEVEL =====")

print(
    "OLD_ANY_Q_OF4 =",
    old_pair["any_Q_of4"],
)

print(
    "NEW_ANY_Q_OF4 =",
    new_pair["any_Q_of4"],
)

print(
    "OLD_ANY_BOTH_OF4 =",
    old_pair["any_both_of4"],
)

print(
    "NEW_ANY_BOTH_OF4 =",
    new_pair["any_both_of4"],
)

print(
    "OLD_Q_SLOT_HIST =",
    old_pair["Q_slot_hist"],
)

print(
    "NEW_Q_SLOT_HIST =",
    new_pair["Q_slot_hist"],
)


print()
print("===== NEW Q LENGTH =====")

for k, x in new_q.items():

    print(
        k,
        "n=",
        x["n"],
        "Q_rate=",
        f"{x['Q_rate']:.4f}",
        "both_rate=",
        f"{x['both_rate']:.4f}",
        "copy_rate=",
        f"{x['copy_rate']:.4f}",
    )


print()
print(
    "DUPLICATE_PHYSICAL_NEW_ROWS =",
    duplicate_physical,
)

print(
    "REPORT =",
    REPORT,
)

print()
print(
    "ANALYZER_CLEAN_MATCHED_ABLATION_V2_COMPLETE"
)
PY


###############################################################################
# STAGE 5 — FREEZE
###############################################################################

echo
echo "======================================================================"
echo "STAGE 5/5 — FREEZE"
echo "======================================================================"

{
    echo "DATE=$(date -Iseconds)"

    echo
    echo "===== FROZEN OLD ====="

    sha256sum \
        "$ANALYSIS" \
        "$OLD_JOBS" \
        "$OLD_RAW" \
        "$SPEC" \
        "$WORKER"

    echo
    echo "===== ANALYSIS CLEAN V2 ====="

    sha256sum \
        "$CANON" \
        "$NEW_JOBS" \
        "$NEW_RAW" \
        "$REPORT"

} > "$HASHES"

cat "$HASHES"

touch "$PASS"
rm -f "$FAIL"

END_EPOCH="$(date +%s)"
ELAPSED=$((END_EPOCH - START_EPOCH))

echo
echo "======================================================================"
echo "PDS ANALYZER CLEAN ABLATION V2 PASS"
echo "======================================================================"
echo "TOTAL_ELAPSED_SECONDS=$ELAPSED"
echo "FINISH_CST=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
echo "FINISH_UTC=$(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
echo
echo "PASS=$PASS"
echo "REPORT=$REPORT"
echo "======================================================================"
