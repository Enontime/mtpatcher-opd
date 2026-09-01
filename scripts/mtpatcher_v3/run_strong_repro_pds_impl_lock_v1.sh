#!/usr/bin/env bash
set -Eeuo pipefail

source /workspace/mtpatcher/project_env.sh

EXP="mtpatcher_v3_full6565_20260823"
D="$DATA_ROOT/$EXP"

OUT="$D/strong_repro_pds_implementation_lock_v1"

REPORT="$OUT/pds_implementation_lock_report_v1.txt"
JSON="$OUT/pds_implementation_lock_report_v1.json"

SCRIPT_INDEX="$OUT/current_repo_pds_candidates_v1.txt"
DATA_INDEX="$OUT/data_pds_candidates_v1.txt"
LOG_INDEX="$OUT/log_pds_candidates_v1.txt"
OFFICIAL_INDEX="$OUT/official_repo_pds_candidates_v1.txt"

SCRIPT_EVIDENCE="$OUT/current_repo_pds_source_evidence_v1.txt"
OFFICIAL_EVIDENCE="$OUT/official_repo_pds_source_evidence_v1.txt"
DATA_EVIDENCE="$OUT/pds_asset_schema_evidence_v1.txt"
LOG_EVIDENCE="$OUT/pds_invocation_evidence_v1.txt"

PASS="$OUT/STRONG_REPRO_PDS_IMPL_LOCK_V1.PASS"
FAIL="$OUT/STRONG_REPRO_PDS_IMPL_LOCK_V1.FAIL"

OFFICIAL="$ROOT/vendor/MT-Patcher-official"

mkdir -p "$OUT"
rm -f "$PASS" "$FAIL"

START_EPOCH="$(date +%s)"
EST_SECONDS=120

echo "======================================================================"
echo "STRONG REPRO PDS IMPLEMENTATION READ-ONLY LOCK V1"
echo "======================================================================"
echo
echo "MODE = READ-ONLY SPECIFICATION AUDIT"
echo "NO GENERATION"
echo "NO STUDENT TRAINING"
echo "NO NPU MODEL LOAD"
echo
echo "预计运行时长：20–90 秒；保守上限约 2 分钟"
echo "预计最晚完成："
echo "  北京时间: $(TZ=Asia/Shanghai date -d "@$((START_EPOCH + EST_SECONDS))" '+%Y-%m-%d %H:%M:%S %Z')"
echo "  Server UTC: $(TZ=UTC date -d "@$((START_EPOCH + EST_SECONDS))" '+%Y-%m-%d %H:%M:%S %Z')"
echo

trap '
rc=$?
echo
echo "======================================================================"
echo "PDS IMPLEMENTATION LOCK FAILED"
echo "return_code=$rc"
echo "time=$(date -Iseconds)"
echo "======================================================================"
touch "'"$FAIL"'"
' ERR


###############################################################################
# STAGE 1/6
# Locate actual PDS-related code/assets/logs.
#
# Important:
# We deliberately DISCOVER paths from the server.
# No guessed historical PDS generator path is hard-coded here.
###############################################################################

echo "======================================================================"
echo "STAGE 1/6 — DISCOVER ACTUAL PDS IMPLEMENTATION"
echo "======================================================================"

find "$ROOT/scripts" \
    -type f \
    \( \
        -iname '*pds*' \
        -o -iname '*knowledge*extension*' \
        -o -iname '*sentence*analy*' \
        -o -iname '*context*generat*' \
    \) \
    2>/dev/null \
    | sort \
    > "$SCRIPT_INDEX"

echo "CURRENT_REPO_PDS_CANDIDATES=$(wc -l < "$SCRIPT_INDEX")"
cat "$SCRIPT_INDEX"


find "$D" \
    -maxdepth 7 \
    -type f \
    \( \
        -iname '*pds*' \
        -o -iname '*sentence*analy*' \
        -o -iname '*domain*topic*style*' \
        -o -iname '*context*' \
    \) \
    2>/dev/null \
    | sort \
    > "$DATA_INDEX"

echo
echo "DATA_PDS_CANDIDATES=$(wc -l < "$DATA_INDEX")"
cat "$DATA_INDEX"


if [ -d "$LOG_ROOT/$EXP" ]; then

    find "$LOG_ROOT/$EXP" \
        -type f \
        \( \
            -iname '*pds*' \
            -o -iname '*knowledge*extension*' \
            -o -iname '*sentence*analy*' \
        \) \
        2>/dev/null \
        | sort \
        > "$LOG_INDEX"

else

    : > "$LOG_INDEX"

fi

echo
echo "LOG_PDS_CANDIDATES=$(wc -l < "$LOG_INDEX")"
cat "$LOG_INDEX"


if [ -d "$OFFICIAL" ]; then

    find "$OFFICIAL" \
        -type f \
        \( \
            -iname '*pds*' \
            -o -iname '*knowledge*extension*' \
            -o -iname '*sentence*analy*' \
            -o -iname '*synth*' \
        \) \
        2>/dev/null \
        | sort \
        > "$OFFICIAL_INDEX"

else

    : > "$OFFICIAL_INDEX"

fi

echo
echo "OFFICIAL_REPO_PDS_CANDIDATES=$(wc -l < "$OFFICIAL_INDEX")"
cat "$OFFICIAL_INDEX"

echo
echo "DISCOVERY_PASS"


###############################################################################
# STAGE 2/6
# Extract implementation semantics from every discovered current script.
###############################################################################

echo
echo "======================================================================"
echo "STAGE 2/6 — CURRENT IMPLEMENTATION SOURCE EVIDENCE"
echo "======================================================================"

python - \
    "$SCRIPT_INDEX" \
    "$SCRIPT_EVIDENCE" <<'PY'

import re
import sys
from pathlib import Path

index = Path(sys.argv[1])
out = Path(sys.argv[2])

paths = [
    Path(x.strip())
    for x in index.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()
    if x.strip()
]

patterns = [
    # local correction / feedback structure
    r"errors?",
    r"error[_ -]?span",
    r"source[_ -]?span",
    r"incorrect",
    r"correct",
    r"correction",
    r"post[_ -]?edit",
    r"student_translation",

    # knowledge pair
    r"word[_ -]?pair",
    r"phrase[_ -]?pair",
    r"\(s,\s*c\)",
    r"source_phrase",
    r"target_phrase",

    # analyzer attributes
    r"domain",
    r"topic",
    r"style",
    r"analy[sz]",

    # multiplicity / multi-error behavior
    r"for .*error",
    r"for .*pair",
    r"enumerate\(.*error",
    r"num[_ -]?(samples|contexts|augment)",
    r"n[_ -]?(samples|contexts|augment)",
    r"4",
    r"range\(",

    # generation treatment
    r"prompt",
    r"messages",
    r"apply_chat_template",
    r"enable_thinking",
    r"do_sample",
    r"temperature",
    r"top_p",
    r"top_k",
    r"max_new_tokens",
    r"batch[_ -]?size",

    # parser/filter/dedup
    r"json",
    r"parse",
    r"dedup",
    r"duplicate",
    r"unique",
    r"set\(",

    # CLI / lineage
    r"argparse",
    r"add_argument",
    r"--input",
    r"--output",
]

regexes = [
    re.compile(p, re.I)
    for p in patterns
]

parts = []

for path in paths:

    try:
        lines = path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()

    except Exception as e:

        parts.append(
            f"\n===== FILE {path} =====\n"
            f"READ_ERROR={e}\n"
        )

        continue

    hit = set()

    for i, line in enumerate(lines):

        if any(r.search(line) for r in regexes):

            for j in range(
                max(0, i - 5),
                min(len(lines), i + 6),
            ):
                hit.add(j)

    parts.append(
        "\n"
        + "=" * 100
        + f"\nFILE={path}\n"
        + "=" * 100
        + "\n"
    )

    if not hit:

        parts.append(
            "NO_RELEVANT_LINES_FOUND\n"
        )

        continue

    last = None

    for j in sorted(hit):

        if (
            last is not None
            and j != last + 1
        ):
            parts.append("\n---\n")

        parts.append(
            f"{j+1:05d}: {lines[j]}\n"
        )

        last = j


out.write_text(
    "".join(parts),
    encoding="utf-8",
)

print(
    "CURRENT_SCRIPT_FILES_SCANNED =",
    len(paths),
)

print(
    "CURRENT_SCRIPT_EVIDENCE =",
    out,
)

print(
    "CURRENT_IMPLEMENTATION_SOURCE_EXTRACTION_PASS"
)
PY

cat "$SCRIPT_EVIDENCE"


###############################################################################
# STAGE 3/6
# Inspect historical PDS-related data schemas.
#
# We do NOT modify any asset.
# For JSONL we read a tiny prefix and count rows.
###############################################################################

echo
echo "======================================================================"
echo "STAGE 3/6 — HISTORICAL PDS ASSET SCHEMA EVIDENCE"
echo "======================================================================"

python - \
    "$DATA_INDEX" \
    "$DATA_EVIDENCE" <<'PY'

import json
import os
import sys
from pathlib import Path

index = Path(sys.argv[1])
out = Path(sys.argv[2])

paths = [
    Path(x.strip())
    for x in index.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()
    if x.strip()
]

MAX_FILES = 100
MAX_SAMPLE_ROWS = 3

parts = []

for path in paths[:MAX_FILES]:

    parts.append(
        "\n"
        + "=" * 100
        + f"\nFILE={path}\n"
        + "=" * 100
        + "\n"
    )

    try:
        st = path.stat()

        parts.append(
            f"SIZE_BYTES={st.st_size}\n"
        )

    except Exception as e:

        parts.append(
            f"STAT_ERROR={e}\n"
        )

        continue

    suffix = path.suffix.lower()

    if suffix == ".jsonl":

        count = 0
        samples = []

        try:

            with path.open(
                encoding="utf-8-sig",
                errors="replace",
            ) as f:

                for line_no, line in enumerate(f, 1):

                    if not line.strip():
                        continue

                    count += 1

                    if len(samples) < MAX_SAMPLE_ROWS:

                        try:
                            x = json.loads(line)

                            samples.append({
                                "line_no": line_no,
                                "keys": sorted(x.keys())
                                    if isinstance(x, dict)
                                    else None,
                                "row": x,
                            })

                        except Exception as e:

                            samples.append({
                                "line_no": line_no,
                                "parse_error": str(e),
                                "raw_prefix": line[:1000],
                            })

        except Exception as e:

            parts.append(
                f"READ_ERROR={e}\n"
            )

            continue

        parts.append(
            f"ROWS={count}\n"
        )

        for sample in samples:

            parts.append(
                "SAMPLE="
                + json.dumps(
                    sample,
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )

    elif suffix == ".json":

        try:

            obj = json.loads(
                path.read_text(
                    encoding="utf-8-sig",
                    errors="replace",
                )
            )

            if isinstance(obj, dict):
                parts.append(
                    "TOP_LEVEL_KEYS="
                    + repr(sorted(obj.keys()))
                    + "\n"
                )

            parts.append(
                "JSON_PREFIX="
                + json.dumps(
                    obj,
                    ensure_ascii=False,
                    indent=2,
                )[:6000]
                + "\n"
            )

        except Exception as e:

            parts.append(
                f"JSON_READ_ERROR={e}\n"
            )

    else:

        try:

            text = path.read_text(
                encoding="utf-8",
                errors="replace",
            )

            parts.append(
                "TEXT_PREFIX=\n"
                + text[:5000]
                + "\n"
            )

        except Exception as e:

            parts.append(
                f"NON_TEXT_OR_READ_ERROR={e}\n"
            )


out.write_text(
    "".join(parts),
    encoding="utf-8",
)

print(
    "DATA_CANDIDATES_SCANNED =",
    min(len(paths), MAX_FILES),
)

print(
    "TOTAL_DATA_CANDIDATES =",
    len(paths),
)

print(
    "PDS_ASSET_SCHEMA_EVIDENCE =",
    out,
)

print(
    "PDS_ASSET_SCHEMA_EXTRACTION_PASS"
)
PY

cat "$DATA_EVIDENCE"


###############################################################################
# STAGE 4/6
# Recover actual historical invocations from logs.
###############################################################################

echo
echo "======================================================================"
echo "STAGE 4/6 — HISTORICAL PDS INVOCATION EVIDENCE"
echo "======================================================================"

{
    echo "===== DIRECT PDS LOG FILES ====="

    while IFS= read -r F; do

        [ -n "$F" ] || continue

        echo
        echo "===================================================================================================="
        echo "FILE=$F"
        echo "===================================================================================================="

        grep -nEi \
            'python|pds|domain|topic|style|pair|error|post_edit|post-edit|batch|temperature|top_p|do_sample|max_new_tokens|parse|dedup|duplicate|input|output' \
            "$F" \
            2>/dev/null \
            | head -500 \
            || true

    done < "$LOG_INDEX"


    echo
    echo "===== BROADER EXP LOG REFERENCES TO PDS ====="

    grep -RInEi \
        'generate[_-]?pds|pds[_-]?generation|knowledge[_ -]?extension|sentence[_ -]?analy|domain.*topic.*style' \
        "$LOG_ROOT/$EXP" \
        2>/dev/null \
        | head -2000 \
        || true

} > "$LOG_EVIDENCE"

cat "$LOG_EVIDENCE"

echo
echo "HISTORICAL_INVOCATION_EXTRACTION_PASS"


###############################################################################
# STAGE 5/6
# Inspect the visible frozen official repo separately.
#
# This is evidence about the released repository only.
# Absence here must NOT be interpreted as absence from the paper method.
###############################################################################

echo
echo "======================================================================"
echo "STAGE 5/6 — FROZEN OFFICIAL REPO PDS EVIDENCE"
echo "======================================================================"

if [ -d "$OFFICIAL" ]; then

    {
        echo "OFFICIAL_ROOT=$OFFICIAL"

        if git -C "$OFFICIAL" rev-parse HEAD >/dev/null 2>&1; then
            echo "OFFICIAL_HEAD=$(git -C "$OFFICIAL" rev-parse HEAD)"
        fi

        echo
        echo "===== PDS-NAMED FILES ====="
        cat "$OFFICIAL_INDEX"

        echo
        echo "===== REPO-WIDE RELEVANT REFERENCES ====="

        grep -RInEi \
            'PDS|knowledge extension|parallel sentence|domain|topic|style|word pair|phrase pair|post-edit|post_edit' \
            "$OFFICIAL" \
            2>/dev/null \
            | head -3000 \
            || true

    } > "$OFFICIAL_EVIDENCE"

else

    echo "OFFICIAL_REPO_NOT_FOUND=$OFFICIAL" \
        > "$OFFICIAL_EVIDENCE"

fi

cat "$OFFICIAL_EVIDENCE"

echo
echo "OFFICIAL_REPO_EVIDENCE_EXTRACTION_PASS"


###############################################################################
# STAGE 6/6
# Build a compact evidence manifest.
#
# IMPORTANT:
# The script does not auto-decide ambiguous scientific specs.
# It records what evidence exists and leaves REVIEW_REQUIRED fields explicit.
###############################################################################

echo
echo "======================================================================"
echo "STAGE 6/6 — FREEZE SPECIFICATION EVIDENCE"
echo "======================================================================"

python - \
    "$SCRIPT_INDEX" \
    "$DATA_INDEX" \
    "$LOG_INDEX" \
    "$OFFICIAL_INDEX" \
    "$SCRIPT_EVIDENCE" \
    "$DATA_EVIDENCE" \
    "$LOG_EVIDENCE" \
    "$OFFICIAL_EVIDENCE" \
    "$REPORT" \
    "$JSON" <<'PY'

import hashlib
import json
import sys
from pathlib import Path

(
    SCRIPT_INDEX,
    DATA_INDEX,
    LOG_INDEX,
    OFFICIAL_INDEX,
    SCRIPT_EVIDENCE,
    DATA_EVIDENCE,
    LOG_EVIDENCE,
    OFFICIAL_EVIDENCE,
    REPORT,
    JSON_OUT,
) = map(Path, sys.argv[1:])


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def lines(path):
    if not path.exists():
        return []

    return [
        x.strip()
        for x in path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
        if x.strip()
    ]


manifest = {
    "protocol":
        "STRONG_REPRO_PDS_IMPLEMENTATION_LOCK_V1",

    "mode":
        "READ_ONLY_SPECIFICATION_AUDIT",

    "generation_started":
        False,

    "student_training_started":
        False,

    "canonical_population_predecision":
        "K1_ALL_11792",

    "k2_used_as_canonical_filter":
        False,

    "scientific_questions_to_lock": {
        "local_pair_extraction":
            "REVIEW_REQUIRED",

        "multi_error_handling":
            "REVIEW_REQUIRED",

        "domain_topic_style_asset":
            "REVIEW_REQUIRED",

        "pds_multiplicity_unit":
            "REVIEW_REQUIRED",

        "generation_prompt":
            "REVIEW_REQUIRED",

        "generation_decode":
            "REVIEW_REQUIRED",

        "parser_filter_dedup":
            "REVIEW_REQUIRED",

        "current_vs_official_fidelity":
            "REVIEW_REQUIRED",
    },

    "candidate_counts": {
        "current_repo_scripts":
            len(lines(SCRIPT_INDEX)),

        "data_assets":
            len(lines(DATA_INDEX)),

        "logs":
            len(lines(LOG_INDEX)),

        "official_repo_files":
            len(lines(OFFICIAL_INDEX)),
    },

    "evidence_files": {
        "current_repo_candidate_index": {
            "path": str(SCRIPT_INDEX),
            "sha256": sha256(SCRIPT_INDEX),
        },

        "data_candidate_index": {
            "path": str(DATA_INDEX),
            "sha256": sha256(DATA_INDEX),
        },

        "log_candidate_index": {
            "path": str(LOG_INDEX),
            "sha256": sha256(LOG_INDEX),
        },

        "official_candidate_index": {
            "path": str(OFFICIAL_INDEX),
            "sha256": sha256(OFFICIAL_INDEX),
        },

        "current_source_evidence": {
            "path": str(SCRIPT_EVIDENCE),
            "sha256": sha256(SCRIPT_EVIDENCE),
        },

        "historical_asset_evidence": {
            "path": str(DATA_EVIDENCE),
            "sha256": sha256(DATA_EVIDENCE),
        },

        "historical_invocation_evidence": {
            "path": str(LOG_EVIDENCE),
            "sha256": sha256(LOG_EVIDENCE),
        },

        "official_repo_evidence": {
            "path": str(OFFICIAL_EVIDENCE),
            "sha256": sha256(OFFICIAL_EVIDENCE),
        },
    },

    "decision": {
        "status":
            "REVIEW_REQUIRED",

        "note":
            (
                "Audit completed only. "
                "No unresolved PDS treatment field "
                "is auto-filled by this script."
            ),
    },
}


JSON_OUT.write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


report = f"""
================================================================================
STRONG REPRO PDS IMPLEMENTATION LOCK V1
================================================================================

MODE
----
READ-ONLY specification audit.
No generation.
No Student training.

CANONICAL SCIENTIFIC POPULATION ALREADY FROZEN
----------------------------------------------
K1-All11792

K2 is NOT the canonical PDS pre-filter.

FIELDS REQUIRING EVIDENCE-BASED LOCK
------------------------------------
1. K1 structured-feedback -> local (s,c) extraction
2. multi-error handling
3. domain/topic/style analyzer asset and reuse path
4. multiplicity unit:
      four per PE sentence?
      four per local pair?
      another historical adaptation?
5. synthesis prompt
6. generation decode
7. parser / filtering / dedup
8. current implementation vs frozen official repo fidelity

IMPORTANT
---------
47168 PDS rows is NOT yet frozen.
It assumes exactly 4 outputs per K1 PE sentence.
That assumption remains unresolved until the implementation evidence is reviewed.

Likewise, 58960 total K1+PDS rows is only provisional.

The future matched-exposure control must be constructed from the ACTUAL
final K1+PDS Student-training cardinality/update budget after PDS generation
and filtering, not from an assumed 58960.

CANDIDATE COUNTS
----------------
current_repo_scripts = {len(lines(SCRIPT_INDEX))}
data_assets          = {len(lines(DATA_INDEX))}
log_files            = {len(lines(LOG_INDEX))}
official_repo_files  = {len(lines(OFFICIAL_INDEX))}

EVIDENCE
--------
CURRENT_SOURCE_EVIDENCE={SCRIPT_EVIDENCE}
DATA_EVIDENCE={DATA_EVIDENCE}
LOG_EVIDENCE={LOG_EVIDENCE}
OFFICIAL_EVIDENCE={OFFICIAL_EVIDENCE}

DECISION
--------
STATUS = REVIEW_REQUIRED

No PDS generation has started.
No Student training has started.
"""


REPORT.write_text(
    report.lstrip(),
    encoding="utf-8",
)

print(report)

print(
    "PDS_IMPLEMENTATION_LOCK_EVIDENCE_FREEZE_PASS"
)
PY


{
    echo "DATE=$(date -Iseconds)"

    echo
    echo "===== REPORTS ====="

    sha256sum \
        "$REPORT" \
        "$JSON"

    echo
    echo "===== EVIDENCE ====="

    sha256sum \
        "$SCRIPT_INDEX" \
        "$DATA_INDEX" \
        "$LOG_INDEX" \
        "$OFFICIAL_INDEX" \
        "$SCRIPT_EVIDENCE" \
        "$DATA_EVIDENCE" \
        "$LOG_EVIDENCE" \
        "$OFFICIAL_EVIDENCE"

} > "$OUT/frozen_sha256_manifest_v1.txt"

cat "$OUT/frozen_sha256_manifest_v1.txt"

touch "$PASS"
rm -f "$FAIL"

END_EPOCH="$(date +%s)"
ELAPSED=$((END_EPOCH - START_EPOCH))

echo
echo "======================================================================"
echo "STRONG REPRO PDS IMPLEMENTATION LOCK V1 PASS"
echo "======================================================================"
echo "实际运行时长: ${ELAPSED} 秒"
echo "完成时间:"
echo "  北京时间: $(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
echo "  Server UTC: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
echo
echo "PASS=$PASS"
echo "REPORT=$REPORT"
echo "JSON=$JSON"
echo
echo "IMPORTANT:"
echo "PASS = audit/extraction completed."
echo "PASS != PDS treatment scientifically frozen."
echo "No generation/training started."
echo "======================================================================"
