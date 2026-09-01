#!/usr/bin/env bash
set -Eeuo pipefail

source /workspace/mtpatcher/project_env.sh

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

EXP="mtpatcher_v3_full6565_20260823"
D="$DATA_ROOT/$EXP"

###############################################################################
# Exact previously-established assets
###############################################################################

BASE="$D/strong_repro_pds_official_smoke_v1"

ANALYSIS="$BASE/sentence_analysis128_v1.jsonl"
OLD_JOBS="$BASE/pds_smoke_jobs_v1.jsonl"
OLD_RAW="$BASE/pds_smoke_raw_v1.jsonl"
SPEC="$BASE/official_pds_spec_v1.json"

WORKER="$ROOT/scripts/mtpatcher_v3/strong_repro_pds_qwen_worker_v1.py"

MODEL="$MODEL_ROOT/Qwen3-8B"

###############################################################################
# New matched ablation
###############################################################################

OUT="$D/pds_smoke_analysis_clean_ablation_v1"
LOG_DIR="$LOG_ROOT/$EXP/pds_smoke_analysis_clean_ablation_v1"

CANON_ANALYSIS="$OUT/sentence_analysis_canonical128_v1.jsonl"
NEW_JOBS="$OUT/pds_jobs_analysis_clean1100_v1.jsonl"

SHARDS="$OUT/pds_shards16"
NEW_RAW="$OUT/pds_raw_analysis_clean1100_v1.jsonl"

REPORT="$OUT/analysis_clean_ablation_report_v1.json"
HASHES="$OUT/frozen_sha256_manifest_v1.txt"

META="$OUT/start_meta_v1.txt"

PASS="$OUT/PDS_SMOKE_ANALYSIS_CLEAN_ABLATION_V1.PASS"
FAIL="$OUT/PDS_SMOKE_ANALYSIS_CLEAN_ABLATION_V1.FAIL"

mkdir -p \
    "$OUT" \
    "$LOG_DIR" \
    "$SHARDS"

rm -f "$PASS" "$FAIL"

START_EPOCH="$(date +%s)"

ETA_MIN=300
ETA_MAX=480

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
echo "PDS SMOKE — ANALYZER CLEAN MATCHED ABLATION V1"
echo "======================================================================"
echo
echo "SCIENTIFIC QUESTION:"
echo "Does full-parent leakage through sentence_analysis cause parent copying?"
echo
echo "UNCHANGED:"
echo "  same 128 parents"
echo "  same 275 local pairs"
echo "  same 4 slots / pair"
echo "  same official CaseGeneration prompt"
echo "  same Qwen3-8B"
echo "  same temperature=1.0"
echo "  same num_beams=1"
echo "  same seed=20260831"
echo
echo "ONLY TREATMENT CHANGE:"
echo "  raw sentence_analysis"
echo "      -> canonical Topic / Domain / Style only"
echo
echo "预计运行时长：5–8 分钟"
cat "$META"
echo


trap '
rc=$?
echo
echo "======================================================================"
echo "ANALYZER CLEAN ABLATION FAILED"
echo "return_code=$rc"
echo "time=$(date -Iseconds)"
echo "======================================================================"
touch "'"$FAIL"'"
' ERR


###############################################################################
# STAGE 1/5 — preflight
###############################################################################

echo
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

python -m py_compile "$WORKER"

if [ "$(wc -l < "$ANALYSIS")" -ne 128 ]; then
    echo "ANALYSIS_COUNT_MISMATCH"
    false
fi

if [ "$(wc -l < "$OLD_JOBS")" -ne 1100 ]; then
    echo "OLD_JOB_COUNT_MISMATCH"
    false
fi

if [ "$(wc -l < "$OLD_RAW")" -ne 1100 ]; then
    echo "OLD_RAW_COUNT_MISMATCH"
    false
fi

echo "PREFLIGHT_PASS"


###############################################################################
# STAGE 2/5 — canonicalize Topic / Domain / Style only
###############################################################################

echo
echo "======================================================================"
echo "STAGE 2/5 — CANONICALIZE SENTENCE ANALYSIS"
echo "======================================================================"

python - \
    "$ANALYSIS" \
    "$OLD_JOBS" \
    "$SPEC" \
    "$CANON_ANALYSIS" \
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


def plain_line(s):
    s = str(s).strip()

    # Remove bullet + markdown emphasis for label parsing.
    s = re.sub(
        r"^\s*[-*•]\s*",
        "",
        s,
    )

    s = s.replace("**", "")
    s = s.replace("__", "")

    return s.strip()


LABELS = {
    "topic": re.compile(
        r"^(?:Topic(?:\s*\(主题\))?|主题)\s*[:：]\s*(.*)$",
        re.I,
    ),

    "domain": re.compile(
        r"^(?:Domain(?:\s*\(领域\))?|领域)\s*[:：]\s*(.*)$",
        re.I,
    ),

    "style": re.compile(
        r"^(?:Style(?:\s*\(风格\))?|风格)\s*[:：]\s*(.*)$",
        re.I,
    ),
}


def next_value(lines, i):

    for j in range(
        i + 1,
        min(
            len(lines),
            i + 4,
        ),
    ):

        x = plain_line(
            lines[j]
        )

        if not x:
            continue

        if x == "---":
            return ""

        # Do not steal another heading.
        if any(
            rx.match(x)
            for rx in LABELS.values()
        ):
            return ""

        if re.match(
            r"^(?:Explanation|解释)\s*[:：]?",
            x,
            flags=re.I,
        ):
            return ""

        return x

    return ""


def extract_tds(raw):

    lines = str(
        raw
    ).splitlines()

    out = {}

    for i, raw_line in enumerate(lines):

        line = plain_line(
            raw_line
        )

        for key, rx in LABELS.items():

            if key in out:
                continue

            m = rx.match(line)

            if not m:
                continue

            value = (
                m.group(1)
                .strip()
            )

            if not value:
                value = next_value(
                    lines,
                    i,
                )

            value = plain_line(
                value
            )

            if value:
                out[key] = value

    return out


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


# Parent source lineage from old jobs.
parent_source = {}

for x in old_jobs:

    pid = int(
        x["parent_index"]
    )

    src = str(
        x["original_source"]
    ).strip()

    if (
        pid in parent_source
        and
        parent_source[pid]
        != src
    ):
        raise RuntimeError(
            f"parent source mismatch {pid}"
        )

    parent_source[
        pid
    ] = src


analysis_by_parent = {}

failures = []

leak_before = 0
leak_after = 0


for x in analysis_rows:

    pid = int(
        x["parent_index"]
    )

    raw = str(
        x["raw_generation"]
    ).strip()

    parent = parent_source[
        pid
    ]

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
                raw[:2500],
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


    analysis_by_parent[
        pid
    ] = {
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
    len(analysis_by_parent),
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


# This must succeed BEFORE any model load.
if failures:

    print()
    print(
        "FIRST_PARSE_FAILURES ="
    )

    for x in failures[:20]:
        print(
            json.dumps(
                x,
                ensure_ascii=False,
            )
        )

    raise RuntimeError(
        "Sentence-analysis Topic/Domain/Style "
        "canonicalization incomplete. "
        "No generation authorized."
    )


if leak_after != 0:

    raise RuntimeError(
        "canonical analysis still contains "
        "complete parent source"
    )


with CANON.open(
    "w",
    encoding="utf-8",
) as f:

    for pid in sorted(
        analysis_by_parent
    ):

        f.write(
            json.dumps(
                analysis_by_parent[
                    pid
                ],
                ensure_ascii=False,
            )
            + "\n"
        )


###############################################################################
# Rebuild EXACT same 1100 scientific jobs.
# Only sentence_analysis + derived prompt change.
###############################################################################

new_jobs = []

for old in sorted(
    old_jobs,
    key=lambda x:
        int(x["job_id"]),
):

    pid = int(
        old["parent_index"]
    )

    canonical = (
        analysis_by_parent[
            pid
        ][
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
            canonical,
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


    new = dict(old)

    new[
        "original_sentence_analysis"
    ] = old[
        "sentence_analysis"
    ]

    new[
        "sentence_analysis"
    ] = canonical

    new[
        "prompt"
    ] = prompt

    new[
        "analysis_treatment"
    ] = (
        "TOPIC_DOMAIN_STYLE_ONLY_V1"
    )

    new_jobs.append(
        new
    )


if len(new_jobs) != 1100:
    raise RuntimeError(
        f"new jobs !=1100: "
        f"{len(new_jobs)}"
    )


# Exact causal identity checks.
for old, new in zip(
    sorted(
        old_jobs,
        key=lambda x:
            int(x["job_id"]),
    ),
    new_jobs,
):

    invariant_fields = [
        "job_id",
        "parent_index",
        "pair_id",
        "error_index",
        "pds_slot",
        "source_span",
        "correction",
        "original_source",
        "repo_num_case",
    ]

    for key in invariant_fields:

        if old.get(key) != new.get(key):

            raise RuntimeError(
                f"causal invariant changed: "
                f"job={old['job_id']} "
                f"field={key}"
            )


with NEW_JOBS.open(
    "w",
    encoding="utf-8",
) as f:

    for x in new_jobs:

        f.write(
            json.dumps(
                x,
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
    "ANALYZER_CANONICALIZATION_PASS"
)
PY


###############################################################################
# STAGE 3/5 — matched CaseGeneration
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


FAIL_WORKER=0

for PID in "${PIDS[@]}"; do

    if wait "$PID"; then
        :
    else
        FAIL_WORKER=1
    fi

done


if [ "$FAIL_WORKER" -ne 0 ]; then

    echo "CASE_GENERATION_WORKER_FAILURE"

    for F in "$LOG_DIR"/device_*.log; do

        echo "===== $F ====="
        tail -60 "$F" || true

    done

    false
fi


###############################################################################
# STAGE 4/5 — merge + matched raw-level causal audit
###############################################################################

echo
echo "======================================================================"
echo "STAGE 4/5 — MATCHED RAW-LEVEL COMPARISON"
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
                rows.append(
                    json.loads(line)
                )

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
            r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?",
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

new_raw_rows = []

for device in range(16):

    p = (
        SHARDS
        / f"device_{device}.jsonl"
    )

    if not p.exists():

        raise RuntimeError(
            f"missing new shard: {p}"
        )

    new_raw_rows.extend(
        load(p)
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

for x in new_raw_rows:

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
    jid: rows[-1]
    for jid, rows
    in new_buckets.items()
}


ids = set(
    old_jobs
)

for name, obj in [
    ("new_jobs", new_jobs),
    ("old_raw", old_raw),
    ("new_raw", new_raw),
]:

    if set(obj) != ids:

        raise RuntimeError(
            f"ID mismatch {name}: "
            f"expected={len(ids)} "
            f"got={len(obj)}"
        )


def metrics(
    jobs,
    raw,
):

    c = Counter()
    qgroups = defaultdict(
        lambda: Counter()
    )

    for jid in sorted(jobs):

        j = jobs[jid]

        text = str(
            raw[jid].get(
                "raw_generation",
                "",
            )
        )

        P = str(
            j["source_span"]
        ).strip()

        Q = str(
            j["correction"]
        ).strip()

        parent = str(
            j["original_source"]
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


        c["n"] += 1
        c["P"] += int(p_ok)
        c["Q"] += int(q_ok)
        c["both"] += int(
            p_ok and q_ok
        )
        c["parent_copy"] += int(
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


    n = c["n"]

    overall = {
        "n":
            n,

        "P_rate":
            c["P"] / n,

        "Q_rate":
            c["Q"] / n,

        "both_rate":
            c["both"] / n,

        "parent_copy_rate":
            c["parent_copy"] / n,

        "parent_copy_count":
            c["parent_copy"],
    }


    strat = {}

    for key, g in sorted(
        qgroups.items()
    ):

        strat[key] = {
            "n":
                g["n"],

            "Q_rate":
                g["Q"]
                / g["n"],

            "both_rate":
                g["both"]
                / g["n"],

            "parent_copy_rate":
                g["copy"]
                / g["n"],
        }


    # Pair-level ×4
    pair_jobs = defaultdict(
        list
    )

    for jid in sorted(jobs):

        pair_jobs[
            int(
                jobs[jid]["pair_id"]
            )
        ].append(jid)


    pair_counts = Counter()

    for pair_id, jids in pair_jobs.items():

        if len(jids) != 4:

            raise RuntimeError(
                f"pair={pair_id} "
                f"jobs={len(jids)}"
            )

        q_success = 0
        both_success = 0

        for jid in jids:

            j = jobs[jid]

            text = str(
                raw[jid].get(
                    "raw_generation",
                    "",
                )
            )

            P = str(
                j["source_span"]
            ).strip()

            Q = str(
                j["correction"]
            ).strip()

            p_ok = (
                norm(P)
                in norm(text)
            )

            q_ok = (
                norm(Q)
                in norm(text)
            )

            q_success += int(
                q_ok
            )

            both_success += int(
                p_ok and q_ok
            )


        pair_counts[
            f"Qslots_{q_success}"
        ] += 1

        pair_counts[
            f"bothslots_{both_success}"
        ] += 1

        pair_counts[
            "any_Q"
        ] += int(
            q_success > 0
        )

        pair_counts[
            "any_both"
        ] += int(
            both_success > 0
        )


    npairs = len(
        pair_jobs
    )

    pair_summary = {
        "pairs":
            npairs,

        "any_Q_of4":
            pair_counts["any_Q"]
            / npairs,

        "any_both_of4":
            pair_counts["any_both"]
            / npairs,

        "Q_slot_hist":
            {
                str(k):
                    pair_counts[
                        f"Qslots_{k}"
                    ]
                for k in range(5)
            },

        "both_slot_hist":
            {
                str(k):
                    pair_counts[
                        f"bothslots_{k}"
                    ]
                for k in range(5)
            },
    }


    return (
        overall,
        strat,
        pair_summary,
    )


old_overall, old_strat, old_pair = (
    metrics(
        old_jobs,
        old_raw,
    )
)

new_overall, new_strat, new_pair = (
    metrics(
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
    k:
        (
            new_overall[k]
            -
            old_overall[k]
        )
    for k in (
        "P_rate",
        "Q_rate",
        "both_rate",
        "parent_copy_rate",
    )
}


report = {
    "protocol":
        "PDS_SMOKE_ANALYSIS_CLEAN_MATCHED_ABLATION_V1",

    "scientific_question":
        (
            "Does parent-source leakage "
            "through sentence_analysis "
            "cause PDS parent copying?"
        ),

    "treatment_changed":
        (
            "sentence_analysis only: "
            "raw zero-shot output -> "
            "Topic/Domain/Style canonical form"
        ),

    "unchanged": [
        "same job IDs",
        "same parent IDs",
        "same pair IDs",
        "same P/Q",
        "same pds slots",
        "same num_case=4",
        "same model",
        "same case prompt template",
        "same temperature=1.0",
        "same num_beams=1",
        "same seed",
    ],

    "duplicate_physical_new_rows":
        duplicate_physical,

    "old":
        {
            "overall":
                old_overall,

            "Q_length":
                old_strat,

            "pair_level":
                old_pair,
        },

    "analysis_clean":
        {
            "overall":
                new_overall,

            "Q_length":
                new_strat,

            "pair_level":
                new_pair,
        },

    "delta_new_minus_old":
        delta,

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
print("ANALYZER CLEAN MATCHED ABLATION RESULT")
print("=" * 78)

print()
print("===== ORIGINAL =====")

for k, v in old_overall.items():
    print(k, "=", v)


print()
print("===== ANALYSIS-CLEAN =====")

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
print("===== NEW Q-LENGTH STRATIFICATION =====")

for key, x in new_strat.items():

    print(
        key,
        "n=",
        x["n"],
        "Q_rate=",
        f"{x['Q_rate']:.4f}",
        "both_rate=",
        f"{x['both_rate']:.4f}",
        "copy_rate=",
        f"{x['parent_copy_rate']:.4f}",
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
    "ANALYZER_CLEAN_MATCHED_ABLATION_COMPLETE"
)
PY


###############################################################################
# STAGE 5/5 — provenance / completion
###############################################################################

echo
echo "======================================================================"
echo "STAGE 5/5 — FREEZE"
echo "======================================================================"

{
    echo "DATE=$(date -Iseconds)"

    echo
    echo "===== FROZEN OLD INPUT ====="

    sha256sum \
        "$ANALYSIS" \
        "$OLD_JOBS" \
        "$OLD_RAW" \
        "$SPEC" \
        "$WORKER"

    echo
    echo "===== NEW TREATMENT ====="

    sha256sum \
        "$CANON_ANALYSIS" \
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
echo "PDS SMOKE ANALYZER CLEAN ABLATION V1 PASS"
echo "======================================================================"
echo "TOTAL_ELAPSED_SECONDS=$ELAPSED"
echo "FINISH_CST=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
echo "FINISH_UTC=$(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
echo
echo "PASS=$PASS"
echo "REPORT=$REPORT"
echo "======================================================================"
