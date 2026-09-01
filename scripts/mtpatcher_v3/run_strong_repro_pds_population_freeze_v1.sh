#!/usr/bin/env bash
set -Eeuo pipefail

source /workspace/mtpatcher/project_env.sh

EXP="mtpatcher_v3_full6565_20260823"
D="$DATA_ROOT/$EXP"

###############################################################################
# Frozen upstream
###############################################################################

K1_SHARDS="$D/strong_repro_broad20k_conservative_k2_v1/k1_output_shards16"

K1_ALL="$D/strong_repro_student_arms_freeze_v1/k1_all_pe11792_v1.jsonl"

EXPECTED_K1_ALL_SHA="7d8c5c1de8249db19e7e881174ef103c9ab6ebfa590623b6ef3190283684ed18"

###############################################################################
# Output
###############################################################################

OUT="$D/strong_repro_pds_population_v1"

PAIRS="$OUT/pds_local_pairs_v1.jsonl"
JOBS="$OUT/pds_jobs_repeat4_v1.jsonl"
PARENTS="$OUT/pds_parent11792_v1.jsonl"

AUDIT="$OUT/pds_population_audit_v1.json"
HASHES="$OUT/frozen_sha256_manifest_v1.txt"

PASS="$OUT/STRONG_REPRO_PDS_POPULATION_V1.PASS"
FAIL="$OUT/STRONG_REPRO_PDS_POPULATION_V1.FAIL"

REPEAT=4

mkdir -p "$OUT"
rm -f "$PASS" "$FAIL"

START_EPOCH="$(date +%s)"
EST_SECONDS=60

echo "======================================================================"
echo "STRONG REPRO PDS POPULATION FREEZE V1"
echo "======================================================================"
echo
echo "预计运行时长：5–20 秒；保守上限 1 分钟"
echo "预计最晚完成："
echo "  北京时间: $(TZ=Asia/Shanghai date -d "@$((START_EPOCH + EST_SECONDS))" '+%Y-%m-%d %H:%M:%S %Z')"
echo "  Server UTC: $(TZ=UTC date -d "@$((START_EPOCH + EST_SECONDS))" '+%Y-%m-%d %H:%M:%S %Z')"
echo
echo "PDS_REPEAT_PER_LOCAL_PAIR=$REPEAT"
echo

trap '
rc=$?
echo
echo "======================================================================"
echo "STRONG REPRO PDS POPULATION FREEZE FAILED"
echo "return_code=$rc"
echo "time=$(date -Iseconds)"
echo "======================================================================"
touch "'"$FAIL"'"
' ERR


###############################################################################
# STAGE 1/3 — frozen lineage
###############################################################################

echo "======================================================================"
echo "STAGE 1/3 — FROZEN K1 LINEAGE"
echo "======================================================================"

if [ ! -d "$K1_SHARDS" ]; then
    echo "MISSING K1 SHARDS: $K1_SHARDS"
    false
fi

if [ ! -f "$K1_ALL" ]; then
    echo "MISSING K1-ALL: $K1_ALL"
    false
fi

ACTUAL_SHA="$(sha256sum "$K1_ALL" | awk '{print $1}')"

echo "K1_ALL=$K1_ALL"
echo "EXPECTED_SHA=$EXPECTED_K1_ALL_SHA"
echo "ACTUAL_SHA=$ACTUAL_SHA"

if [ "$ACTUAL_SHA" != "$EXPECTED_K1_ALL_SHA" ]; then
    echo "K1_ALL_SHA_MISMATCH"
    false
fi

echo "K1_ALL_HASH_LOCK_PASS"


###############################################################################
# STAGE 2/3 — enumerate every local error pair
###############################################################################

python - \
    "$K1_SHARDS" \
    "$K1_ALL" \
    "$PARENTS" \
    "$PAIRS" \
    "$JOBS" \
    "$AUDIT" \
    "$REPEAT" <<'PY'

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

K1_SHARDS = Path(sys.argv[1])
K1_ALL = Path(sys.argv[2])
PARENTS = Path(sys.argv[3])
PAIRS = Path(sys.argv[4])
JOBS = Path(sys.argv[5])
AUDIT = Path(sys.argv[6])
REPEAT = int(sys.argv[7])


def load(path):
    rows = []

    with path.open(
        encoding="utf-8-sig"
    ) as f:
        for line_no, line in enumerate(f, 1):

            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))

            except Exception as e:
                raise RuntimeError(
                    f"JSON parse error "
                    f"{path}:{line_no}: {e}"
                )

    return rows


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def clean(x):
    if not isinstance(x, str):
        return ""

    return x.strip()


def idx_of(x):
    return int(x["index"])


###############################################################################
# Frozen K1-All training population
###############################################################################

k1_all_rows = load(K1_ALL)

if len(k1_all_rows) != 11792:
    raise RuntimeError(
        f"K1-All expected11792 got={len(k1_all_rows)}"
    )

k1_all = {
    idx_of(x): x
    for x in k1_all_rows
}

if len(k1_all) != 11792:
    raise RuntimeError(
        "duplicate IDs in frozen K1-All"
    )

frozen_ids = set(k1_all)


###############################################################################
# Full K1 structured feedback universe
###############################################################################

k1_rows = []

files = sorted(
    K1_SHARDS.glob("shard_*.jsonl")
)

if not files:
    raise RuntimeError(
        f"no shard_*.jsonl under {K1_SHARDS}"
    )

for path in files:
    k1_rows.extend(load(path))

if len(k1_rows) != 20000:
    raise RuntimeError(
        f"K1 raw expected20000 got={len(k1_rows)}"
    )

k1 = {
    idx_of(x): x
    for x in k1_rows
}

if len(k1) != 20000:
    raise RuntimeError(
        "duplicate K1 structured-feedback IDs"
    )


###############################################################################
# Reconstruct exact K1 eligibility and require ID equality with frozen arm.
###############################################################################

def eligible(x):

    errors = x.get("errors")

    post = clean(
        x.get("post_edit")
    )

    student = clean(
        x.get("student_translation")
    )

    return (
        x.get("parse_ok") is True
        and
        x.get("has_error") is True
        and
        isinstance(errors, list)
        and
        len(errors) > 0
        and
        bool(post)
        and
        bool(student)
        and
        post != student
    )


eligible_ids = {
    idx
    for idx, row in k1.items()
    if eligible(row)
}

print(
    "RECONSTRUCTED_K1_ELIGIBLE =",
    len(eligible_ids),
)

if len(eligible_ids) != 11792:
    raise RuntimeError(
        "reconstructed K1 eligible !=11792"
    )

if eligible_ids != frozen_ids:

    missing = sorted(
        frozen_ids - eligible_ids
    )[:30]

    extra = sorted(
        eligible_ids - frozen_ids
    )[:30]

    raise RuntimeError(
        "K1-All lineage mismatch "
        f"missing={missing} extra={extra}"
    )

print(
    "FROZEN_K1_ALL_IDENTITY_PASS"
)


###############################################################################
# Verify frozen PE targets are exactly K1 post-edits.
###############################################################################

for idx in sorted(frozen_ids):

    src_a = clean(
        k1[idx].get("source")
    )

    src_b = clean(
        k1_all[idx].get("source")
    )

    tgt_a = clean(
        k1[idx].get("post_edit")
    )

    tgt_b = clean(
        k1_all[idx].get("target_translation")
    )

    if src_a != src_b:
        raise RuntimeError(
            f"source mismatch index={idx}"
        )

    if tgt_a != tgt_b:
        raise RuntimeError(
            f"post_edit mismatch index={idx}"
        )

print(
    "FROZEN_K1_ALL_TARGET_LINEAGE_PASS"
)


###############################################################################
# Enumerate every structured error occurrence.
#
# Scientific decision:
# - local knowledge pair = source_span -> correction
# - ALL errors are enumerated
# - nonempty pair required
# - source-span exactness in parent is DIAGNOSTIC, not an exclusion gate
#   (matches the guard-fixed historical adaptation)
###############################################################################

parents = []
pairs = []
jobs = []

stats = Counter()
error_type_counts = Counter()

pair_id = 0
job_id = 0

mismatch_examples = []
malformed_examples = []


for idx in sorted(frozen_ids):

    row = k1[idx]

    source = clean(
        row.get("source")
    )

    student = clean(
        row.get("student_translation")
    )

    post_edit = clean(
        row.get("post_edit")
    )

    errors = row.get("errors")

    if not isinstance(errors, list):
        raise RuntimeError(
            f"errors non-list index={idx}"
        )


    parents.append({
        "index": idx,
        "source": source,
        "student_translation": student,
        "post_edit": post_edit,
        "error_count": len(errors),
    })


    if len(errors) > 1:
        stats["multi_error_rows"] += 1


    for error_index, err in enumerate(errors):

        stats["total_error_records"] += 1

        if not isinstance(err, dict):

            stats["malformed_non_dict"] += 1

            if len(malformed_examples) < 30:
                malformed_examples.append({
                    "index": idx,
                    "error_index": error_index,
                    "reason": "non_dict",
                    "value": repr(err),
                })

            continue


        source_span = clean(
            err.get("source_span")
        )

        correction = clean(
            err.get("correction")
        )

        translation_span = clean(
            err.get("translation_span")
        )

        error_type = clean(
            err.get("error_type")
        )

        explanation = clean(
            err.get("explanation")
        )


        if not source_span:

            stats["missing_source_span"] += 1

            if len(malformed_examples) < 30:
                malformed_examples.append({
                    "index": idx,
                    "error_index": error_index,
                    "reason": "missing_source_span",
                })

            continue


        if not correction:

            stats["missing_correction"] += 1

            if len(malformed_examples) < 30:
                malformed_examples.append({
                    "index": idx,
                    "error_index": error_index,
                    "reason": "missing_correction",
                })

            continue


        exact_in_parent = (
            source_span in source
        )

        if exact_in_parent:
            stats["source_span_exact"] += 1
        else:
            stats[
                "source_span_not_exact_in_parent"
            ] += 1

            if len(mismatch_examples) < 30:
                mismatch_examples.append({
                    "index": idx,
                    "error_index": error_index,
                    "source_span": source_span,
                    "source": source,
                })


        stats["valid_local_pairs"] += 1

        if error_type:
            error_type_counts[
                error_type
            ] += 1


        pair = {
            "pair_id": pair_id,
            "parent_index": idx,
            "error_index": error_index,

            "source": source,
            "student_translation": student,
            "post_edit": post_edit,

            "source_span": source_span,
            "correction": correction,

            "translation_span":
                translation_span,

            "error_type":
                error_type,

            "explanation":
                explanation,

            "source_span_exact_in_parent":
                exact_in_parent,

            "construction_method":
                (
                    "STRONG_REPRO_PDS_"
                    "LOCAL_PAIR_V1"
                ),
        }

        pairs.append(pair)


        for slot in range(REPEAT):

            jobs.append({
                **pair,

                "job_id":
                    job_id,

                "pds_slot":
                    slot,

                "pds_repeat_per_local_pair":
                    REPEAT,

                "sentence_analysis":
                    None,

                "pds_stage":
                    "WAITING_FOR_SENTENCE_ANALYSIS",
            })

            job_id += 1


        pair_id += 1


###############################################################################
# Invariants
###############################################################################

if len(parents) != 11792:
    raise RuntimeError(
        f"parent rows !=11792: {len(parents)}"
    )

if len(pairs) != stats["valid_local_pairs"]:
    raise RuntimeError(
        "pair cardinality mismatch"
    )

if len(jobs) != len(pairs) * REPEAT:
    raise RuntimeError(
        "PDS repeat cardinality mismatch"
    )


###############################################################################
# Freeze files
###############################################################################

def write(path, rows):

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as f:

        for x in rows:

            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    tmp.replace(path)


write(PARENTS, parents)
write(PAIRS, pairs)
write(JOBS, jobs)


audit = {
    "protocol":
        "STRONG_REPRO_PDS_POPULATION_V1",

    "status":
        "POPULATION_FREEZE_ONLY",

    "generation_started":
        False,

    "student_training_started":
        False,

    "canonical_population":
        "K1_ALL_11792",

    "k2_prefilter_used":
        False,

    "local_pair_definition": {
        "P":
            "source_span",

        "Q":
            "correction",

        "multi_error_policy":
            "enumerate_all_error_records",

        "missing_pair_policy":
            (
                "skip error records with empty "
                "source_span or correction"
            ),

        "source_span_parent_mismatch_policy":
            (
                "retain as audited warning; "
                "not a selection gate at population stage"
            ),
    },

    "multiplicity": {
        "unit":
            "valid_local_error_pair",

        "repeat":
            REPEAT,

        "fidelity":
            (
                "HISTORICAL_ADAPTATION; "
                "not claimed PAPER-EXACT"
            ),
    },

    "counts": {
        "parents":
            len(parents),

        "total_error_records":
            stats["total_error_records"],

        "valid_local_pairs":
            len(pairs),

        "multi_error_rows":
            stats["multi_error_rows"],

        "source_span_exact":
            stats["source_span_exact"],

        "source_span_not_exact_in_parent":
            stats[
                "source_span_not_exact_in_parent"
            ],

        "malformed_non_dict":
            stats["malformed_non_dict"],

        "missing_source_span":
            stats["missing_source_span"],

        "missing_correction":
            stats["missing_correction"],

        "raw_pds_jobs":
            len(jobs),
    },

    "error_type_counts":
        dict(error_type_counts),

    "source_span_mismatch_examples":
        mismatch_examples,

    "malformed_examples":
        malformed_examples,

    "files": {
        "parents": {
            "path": str(PARENTS),
        },

        "pairs": {
            "path": str(PAIRS),
        },

        "jobs": {
            "path": str(JOBS),
        },
    },
}


AUDIT.write_text(
    json.dumps(
        audit,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("=" * 78)
print("STRONG REPRO PDS POPULATION RESULT")
print("=" * 78)

print(
    "PARENT_K1_ROWS =",
    len(parents),
)

print(
    "TOTAL_ERROR_RECORDS =",
    stats["total_error_records"],
)

print(
    "VALID_LOCAL_ERROR_PAIRS =",
    len(pairs),
)

print(
    "MULTI_ERROR_ROWS =",
    stats["multi_error_rows"],
)

print(
    "SOURCE_SPAN_EXACT =",
    stats["source_span_exact"],
)

print(
    "SOURCE_SPAN_NOT_EXACT =",
    stats[
        "source_span_not_exact_in_parent"
    ],
)

print(
    "MALFORMED_NON_DICT =",
    stats["malformed_non_dict"],
)

print(
    "MISSING_SOURCE_SPAN =",
    stats["missing_source_span"],
)

print(
    "MISSING_CORRECTION =",
    stats["missing_correction"],
)

print(
    "PDS_REPEAT_PER_LOCAL_PAIR =",
    REPEAT,
)

print(
    "RAW_PDS_JOBS =",
    len(jobs),
)

print()
print(
    "NOTE: RAW_PDS_JOBS = "
    "VALID_LOCAL_ERROR_PAIRS * 4"
)

print(
    "NOTE: it is NOT K1_ROWS * 4"
)

print()
print(
    "PAIRS_SHA256 =",
    sha256(PAIRS),
)

print(
    "JOBS_SHA256 =",
    sha256(JOBS),
)

print(
    "PDS_POPULATION_CONSTRUCTION_PASS"
)
PY


###############################################################################
# STAGE 3/3 — freeze hashes
###############################################################################

echo
echo "======================================================================"
echo "STAGE 3/3 — FINAL FREEZE"
echo "======================================================================"

{
    echo "DATE=$(date -Iseconds)"
    echo "REPEAT=$REPEAT"

    echo
    sha256sum \
        "$K1_ALL" \
        "$PARENTS" \
        "$PAIRS" \
        "$JOBS" \
        "$AUDIT"

} > "$HASHES"

cat "$HASHES"

touch "$PASS"
rm -f "$FAIL"

END_EPOCH="$(date +%s)"
ELAPSED=$((END_EPOCH - START_EPOCH))

echo
echo "======================================================================"
echo "STRONG REPRO PDS POPULATION V1 PASS"
echo "======================================================================"
echo "实际运行时长: ${ELAPSED} 秒"
echo "完成时间:"
echo "  北京时间: $(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
echo "  Server UTC: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
echo
echo "AUDIT=$AUDIT"
echo "PAIRS=$PAIRS"
echo "JOBS=$JOBS"
echo "PASS=$PASS"
echo "======================================================================"
