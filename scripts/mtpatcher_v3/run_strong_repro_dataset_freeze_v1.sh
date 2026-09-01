#!/usr/bin/env bash
set -Eeuo pipefail

source /workspace/mtpatcher/project_env.sh

EXP="mtpatcher_v3_full6565_20260823"
D="$DATA_ROOT/$EXP"

###############################################################################
# Frozen upstream assets
###############################################################################

K2DIR="$D/strong_repro_broad20k_conservative_k2_v1"

K1_SHARDS="$K2DIR/k1_output_shards16"
K2_ACCEPT_SOURCE="$K2DIR/k2_verified_clean.jsonl"

TEACHER_DIR="$D/strong_repro_broad20k_seqkd_same_source_v1"
TEACHER="$TEACHER_DIR/seqkd_broad20000_qwen3_8b_fresh_v1.jsonl"

BROAD="$D/paperfaith_paper20k_v2/demo_pool.jsonl"
JOBS="$D/paperfaith_paper20k_v2/feedback_jobs.jsonl"

###############################################################################
# New frozen Student-arm datasets
###############################################################################

OUT="$D/strong_repro_student_arms_freeze_v1"

K1_ALL="$OUT/k1_all_pe11792_v1.jsonl"
K2_PE="$OUT/k2_consistent_pe4960_v1.jsonl"
RAND_K1="$OUT/random_k1_pe4960_seed20260831_v1.jsonl"
RAND_SEQKD="$OUT/random_seqkd4960_seed20260831_v1.jsonl"
SAME_SEQKD="$OUT/same_source_seqkd4960_v1.jsonl"
FULL_SEQKD="$OUT/full_seqkd20000_v1.jsonl"

K2_IDS="$OUT/k2_consistent_ids4960_v1.txt"
RAND_K1_IDS="$OUT/random_k1_ids4960_seed20260831_v1.txt"
RAND_SEQKD_IDS="$OUT/random_seqkd_ids4960_seed20260831_v1.txt"

MANIFEST="$OUT/strong_repro_student_arms_manifest_v1.json"

PASS="$OUT/STRONG_REPRO_STUDENT_ARMS_FREEZE_V1.PASS"
FAIL="$OUT/STRONG_REPRO_STUDENT_ARMS_FREEZE_V1.FAIL"

SAMPLE_SEED=20260831

###############################################################################
# Frozen upstream hashes
###############################################################################

EXPECTED_TEACHER_SHA="f80d30412232bd689cb6382dfbf4b9025d4cc0e043aed509af3021466a5de923"

EXPECTED_K2_ACCEPT_SHA="cf216d51c2dc67c892b7c968a57d3832ee8ae705b62d60f0218e19929993130e"

EXPECTED_BROAD_SHA="2ab671159dd79cce181f4a38f951ad3d93d15575dd1c74e398b05bbc10e5b59e"

EXPECTED_JOBS_SHA="1ac060f2cdb03ef3011bd1e0f5f4b28de831e2f921ccf2a89f1f4d6cfb5edaaa"

mkdir -p "$OUT"

rm -f "$FAIL"

###############################################################################
# Estimated completion time
###############################################################################

START_EPOCH="$(date +%s)"
EST_SECONDS=120

echo "======================================================================"
echo "STRONG REPRO STUDENT ARM DATASET FREEZE V1"
echo "======================================================================"
echo
echo "预计运行时长：20–90 秒；保守上限约 2 分钟"
echo "预计最晚完成时间："
echo "  北京时间: $(TZ=Asia/Shanghai date -d "@$((START_EPOCH + EST_SECONDS))" '+%Y-%m-%d %H:%M:%S %Z')"
echo "  Server UTC: $(TZ=UTC date -d "@$((START_EPOCH + EST_SECONDS))" '+%Y-%m-%d %H:%M:%S %Z')"
echo
echo "SAMPLE_SEED=$SAMPLE_SEED"
echo "OUT=$OUT"
echo

trap '
rc=$?
echo
echo "======================================================================"
echo "STRONG REPRO STUDENT ARM DATASET FREEZE FAILED"
echo "return_code=$rc"
echo "time=$(date -Iseconds)"
echo "======================================================================"
touch "'"$FAIL"'"
' ERR


###############################################################################
# If already frozen successfully, do not silently mutate it.
###############################################################################

if [ -f "$PASS" ]; then
    echo "ALREADY_FROZEN = YES"
    echo "PASS=$PASS"

    if [ -f "$MANIFEST" ]; then
        cat "$MANIFEST"
    fi

    echo
    echo "No dataset was modified."
    echo "STRONG_REPRO_STUDENT_ARMS_ALREADY_FROZEN"
else

###############################################################################
# STAGE 1/5 — upstream provenance
###############################################################################

echo "======================================================================"
echo "STAGE 1/5 — UPSTREAM PROVENANCE LOCK"
echo "======================================================================"

for F in \
    "$TEACHER" \
    "$K2_ACCEPT_SOURCE" \
    "$BROAD" \
    "$JOBS"
do
    if [ ! -f "$F" ]; then
        echo "MISSING REQUIRED FILE: $F"
        false
    fi
done

if [ ! -d "$K1_SHARDS" ]; then
    echo "MISSING K1 SHARD DIRECTORY: $K1_SHARDS"
    false
fi


check_sha() {
    local FILE="$1"
    local EXPECTED="$2"
    local NAME="$3"

    local ACTUAL
    ACTUAL="$(sha256sum "$FILE" | awk '{print $1}')"

    echo "$NAME"
    echo "  path=$FILE"
    echo "  expected=$EXPECTED"
    echo "  actual=$ACTUAL"

    if [ "$ACTUAL" != "$EXPECTED" ]; then
        echo "SHA_MISMATCH=$NAME"
        false
    fi
}


check_sha \
    "$TEACHER" \
    "$EXPECTED_TEACHER_SHA" \
    "FRESH_TEACHER20K"

check_sha \
    "$K2_ACCEPT_SOURCE" \
    "$EXPECTED_K2_ACCEPT_SHA" \
    "HISTORICAL_K2_ACCEPT_ARTIFACT"

check_sha \
    "$BROAD" \
    "$EXPECTED_BROAD_SHA" \
    "BROAD20K"

check_sha \
    "$JOBS" \
    "$EXPECTED_JOBS_SHA" \
    "FEEDBACK_JOBS20K"


echo
echo "===== K1 SHARDS ====="

K1_RAW_ROWS=0

for F in "$K1_SHARDS"/shard_*.jsonl; do
    if [ -f "$F" ]; then
        N="$(wc -l < "$F")"
        printf "%-22s %5d\n" "$(basename "$F")" "$N"
        K1_RAW_ROWS=$((K1_RAW_ROWS + N))
    fi
done

echo "K1_RAW_ROWS=$K1_RAW_ROWS"

if [ "$K1_RAW_ROWS" -ne 20000 ]; then
    echo "K1_RAW_ROWS_EXPECTED_20000"
    false
fi

echo
echo "UPSTREAM_PROVENANCE_LOCK_PASS"


###############################################################################
# STAGE 2/5 — construct six arms
###############################################################################

echo
echo "======================================================================"
echo "STAGE 2/5 — CONSTRUCT SIX FROZEN ARMS"
echo "======================================================================"

python - \
    "$K1_SHARDS" \
    "$K2_ACCEPT_SOURCE" \
    "$TEACHER" \
    "$BROAD" \
    "$JOBS" \
    "$OUT" \
    "$SAMPLE_SEED" <<'PY'

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

K1_SHARDS = Path(sys.argv[1])
K2_FILE = Path(sys.argv[2])
TEACHER_FILE = Path(sys.argv[3])
BROAD_FILE = Path(sys.argv[4])
JOBS_FILE = Path(sys.argv[5])
OUT = Path(sys.argv[6])
SEED = int(sys.argv[7])

PROMPT = (
    "Translate the following text into English "
    "without additional explanations:\n\n"
    "{source}\n\n"
)

K1_ALL = OUT / "k1_all_pe11792_v1.jsonl"
K2_PE = OUT / "k2_consistent_pe4960_v1.jsonl"
RAND_K1 = OUT / "random_k1_pe4960_seed20260831_v1.jsonl"
RAND_SEQKD = OUT / "random_seqkd4960_seed20260831_v1.jsonl"
SAME_SEQKD = OUT / "same_source_seqkd4960_v1.jsonl"
FULL_SEQKD = OUT / "full_seqkd20000_v1.jsonl"

K2_IDS_FILE = OUT / "k2_consistent_ids4960_v1.txt"
RAND_K1_IDS_FILE = OUT / "random_k1_ids4960_seed20260831_v1.txt"
RAND_SEQKD_IDS_FILE = OUT / "random_seqkd_ids4960_seed20260831_v1.txt"

MANIFEST = OUT / "strong_repro_student_arms_manifest_v1.json"


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def load_jsonl(path):
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
                    f"JSON parse failure "
                    f"{path}:{line_no}: {e}"
                )

    return rows


def idx_of(x):
    if x.get("index") is not None:
        return int(x["index"])

    if x.get("demo_id") is not None:
        return int(x["demo_id"])

    raise RuntimeError(
        f"missing index/demo_id keys={sorted(x.keys())}"
    )


def source_of(x):
    s = x.get("source")

    if not isinstance(s, str):
        raise RuntimeError(
            f"source missing/non-string "
            f"index={idx_of(x)}"
        )

    s = s.strip()

    if not s:
        raise RuntimeError(
            f"empty source index={idx_of(x)}"
        )

    return s


def write_jsonl_atomic(path, rows):
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


def write_ids_atomic(path, ids):
    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        "".join(
            f"{x}\n"
            for x in ids
        ),
        encoding="utf-8",
    )

    tmp.replace(path)


def training_row(
    idx,
    source,
    target,
    arm,
    target_provenance,
):
    if not isinstance(target, str):
        raise RuntimeError(
            f"non-string target index={idx} arm={arm}"
        )

    target = target.strip()

    if not target:
        raise RuntimeError(
            f"empty target index={idx} arm={arm}"
        )

    return {
        "index": int(idx),
        "source": source,
        "messages": [
            {
                "role": "user",
                "content": PROMPT.format(
                    source=source
                ),
            }
        ],
        "target_translation": target,
        "strong_repro_arm": arm,
        "target_provenance": target_provenance,
    }


###############################################################################
# Load Broad20k canonical source universe.
###############################################################################

broad = load_jsonl(BROAD_FILE)
jobs = load_jsonl(JOBS_FILE)

if len(broad) != 20000:
    raise RuntimeError(
        f"Broad rows expected20000 got={len(broad)}"
    )

if len(jobs) != 20000:
    raise RuntimeError(
        f"Jobs rows expected20000 got={len(jobs)}"
    )


jobs_map = {
    idx_of(x): source_of(x)
    for x in jobs
}

if len(jobs_map) != 20000:
    raise RuntimeError(
        "feedback_jobs duplicate IDs"
    )

if set(jobs_map) != set(range(20000)):
    raise RuntimeError(
        "feedback_jobs IDs != 0..19999"
    )


# Broad and jobs were previously checked by sequence.
# Check again here so this freeze is self-contained.
for pos, (b, j) in enumerate(
    zip(broad, jobs)
):
    if source_of(b) != source_of(j):
        raise RuntimeError(
            f"Broad/jobs source mismatch position={pos}"
        )

print("BROAD20K_CANONICAL_UNIVERSE_PASS")


###############################################################################
# Load fresh Teacher20k.
###############################################################################

teacher_rows = load_jsonl(TEACHER_FILE)

if len(teacher_rows) != 20000:
    raise RuntimeError(
        f"Teacher rows expected20000 "
        f"got={len(teacher_rows)}"
    )

teacher = {}

for x in teacher_rows:
    idx = idx_of(x)

    if idx in teacher:
        raise RuntimeError(
            f"duplicate Teacher index={idx}"
        )

    if source_of(x) != jobs_map[idx]:
        raise RuntimeError(
            f"Teacher source mismatch index={idx}"
        )

    target = x.get("target_translation")

    if (
        not isinstance(target, str)
        or
        not target.strip()
    ):
        raise RuntimeError(
            f"Teacher empty target index={idx}"
        )

    teacher[idx] = x

if set(teacher) != set(range(20000)):
    raise RuntimeError(
        "Teacher ID universe != 0..19999"
    )

print("TEACHER20K_LINEAGE_PASS")


###############################################################################
# Load all frozen K1 feedback outputs.
###############################################################################

k1_rows = []

for p in sorted(
    K1_SHARDS.glob("shard_*.jsonl")
):
    k1_rows.extend(
        load_jsonl(p)
    )

if len(k1_rows) != 20000:
    raise RuntimeError(
        f"K1 rows expected20000 got={len(k1_rows)}"
    )

k1 = {}

for x in k1_rows:
    idx = idx_of(x)

    if idx in k1:
        raise RuntimeError(
            f"duplicate K1 index={idx}"
        )

    if source_of(x) != jobs_map[idx]:
        raise RuntimeError(
            f"K1 source mismatch index={idx}"
        )

    k1[idx] = x

if set(k1) != set(range(20000)):
    raise RuntimeError(
        "K1 ID universe != 0..19999"
    )

print("K1_FULL20K_LINEAGE_PASS")


###############################################################################
# Reconstruct exact K1 PE eligibility.
###############################################################################

def k1_is_eligible(x):

    errors = x.get("errors")
    post = x.get("post_edit")
    student = x.get("student_translation")

    return (
        x.get("parse_ok") is True
        and
        x.get("has_error") is True
        and
        isinstance(errors, list)
        and
        len(errors) > 0
        and
        isinstance(post, str)
        and
        bool(post.strip())
        and
        isinstance(student, str)
        and
        post.strip() != student.strip()
    )


k1_eligible_ids = sorted(
    idx
    for idx, x in k1.items()
    if k1_is_eligible(x)
)

print(
    "K1_ELIGIBLE_ROWS =",
    len(k1_eligible_ids),
)

if len(k1_eligible_ids) != 11792:
    raise RuntimeError(
        "K1 eligibility cardinality mismatch: "
        f"expected11792 got={len(k1_eligible_ids)}"
    )

k1_eligible_set = set(
    k1_eligible_ids
)


###############################################################################
# Load historical artifact named k2_verified_clean.jsonl.
#
# IMPORTANT:
# We do NOT interpret the historical filename as semantic cleanliness.
# Here it is used only to recover K2-consistent IDs.
###############################################################################

k2_rows = load_jsonl(K2_FILE)

if len(k2_rows) != 4960:
    raise RuntimeError(
        f"K2 accepted rows expected4960 "
        f"got={len(k2_rows)}"
    )

k2_ids = []

for x in k2_rows:
    idx = idx_of(x)

    if idx in k2_ids:
        raise RuntimeError(
            f"duplicate K2 index={idx}"
        )

    if idx not in k1_eligible_set:
        raise RuntimeError(
            f"K2 index not K1-eligible: {idx}"
        )

    if source_of(x) != jobs_map[idx]:
        raise RuntimeError(
            f"K2 source mismatch index={idx}"
        )

    # K2 verification input was the K1 post-edit.
    # Verify that linkage when the field is present.
    k1_target = str(
        k1[idx]["post_edit"]
    ).strip()

    k2_draft = x.get(
        "student_translation"
    )

    if (
        isinstance(k2_draft, str)
        and
        k2_draft.strip() != k1_target
    ):
        raise RuntimeError(
            f"K2 recursion mismatch index={idx}"
        )

    k2_ids.append(idx)


k2_ids = sorted(k2_ids)

if len(set(k2_ids)) != 4960:
    raise RuntimeError(
        "K2 unique cardinality !=4960"
    )

print(
    "K2_CONSISTENT_IDS =",
    len(k2_ids),
)

print(
    "K2_IS_SUBSET_OF_K1_ELIGIBLE =",
    set(k2_ids).issubset(
        k1_eligible_set
    ),
)


###############################################################################
# Pre-register random controls.
#
# Random-K1:
# uniform sample from K1-eligible universe.
#
# Random-SeqKD:
# uniform sample from complete frozen Broad20k universe.
#
# Separate RNG streams prevent one universe's implementation details
# from changing the other sample.
###############################################################################

rng_k1 = random.Random(
    f"{SEED}:RANDOM_K1_PE"
)

rng_seqkd = random.Random(
    f"{SEED}:RANDOM_SEQKD"
)

random_k1_ids = sorted(
    rng_k1.sample(
        k1_eligible_ids,
        4960,
    )
)

random_seqkd_ids = sorted(
    rng_seqkd.sample(
        list(range(20000)),
        4960,
    )
)


if len(set(random_k1_ids)) != 4960:
    raise RuntimeError(
        "Random-K1 duplicate sample"
    )

if len(set(random_seqkd_ids)) != 4960:
    raise RuntimeError(
        "Random-SeqKD duplicate sample"
    )

if not set(random_k1_ids).issubset(
    k1_eligible_set
):
    raise RuntimeError(
        "Random-K1 not subset of K1 eligible"
    )


###############################################################################
# Construct all six datasets in a COMMON training schema.
###############################################################################

k1_all_rows = [
    training_row(
        idx=idx,
        source=jobs_map[idx],
        target=k1[idx]["post_edit"],
        arm="K1_ALL_PE11792",
        target_provenance="K1_CONSERVATIVE_POST_EDIT",
    )
    for idx in k1_eligible_ids
]


k2_pe_rows = [
    training_row(
        idx=idx,
        source=jobs_map[idx],
        target=k1[idx]["post_edit"],
        arm="K2_CONSISTENT_PE4960",
        target_provenance=(
            "K1_CONSERVATIVE_POST_EDIT_"
            "SURVIVED_K2_CONSISTENCY"
        ),
    )
    for idx in k2_ids
]


random_k1_rows = [
    training_row(
        idx=idx,
        source=jobs_map[idx],
        target=k1[idx]["post_edit"],
        arm="RANDOM_K1_PE4960",
        target_provenance=(
            "UNIFORM_RANDOM_K1_ELIGIBLE_"
            "K1_CONSERVATIVE_POST_EDIT"
        ),
    )
    for idx in random_k1_ids
]


random_seqkd_rows = [
    training_row(
        idx=idx,
        source=jobs_map[idx],
        target=teacher[idx]["target_translation"],
        arm="RANDOM_SEQKD4960",
        target_provenance=(
            "UNIFORM_RANDOM_BROAD20K_"
            "FRESH_QWEN3_8B_TEACHER"
        ),
    )
    for idx in random_seqkd_ids
]


same_source_seqkd_rows = [
    training_row(
        idx=idx,
        source=jobs_map[idx],
        target=teacher[idx]["target_translation"],
        arm="SAME_SOURCE_SEQKD4960",
        target_provenance=(
            "K2_EXACT_SOURCE_IDS_"
            "FRESH_QWEN3_8B_TEACHER"
        ),
    )
    for idx in k2_ids
]


full_seqkd_rows = [
    training_row(
        idx=idx,
        source=jobs_map[idx],
        target=teacher[idx]["target_translation"],
        arm="FULL_SEQKD20000",
        target_provenance=(
            "FULL_BROAD20K_"
            "FRESH_QWEN3_8B_TEACHER"
        ),
    )
    for idx in range(20000)
]


###############################################################################
# Critical causal matching checks.
###############################################################################

def ids(rows):
    return [
        int(x["index"])
        for x in rows
    ]


assert len(k1_all_rows) == 11792
assert len(k2_pe_rows) == 4960
assert len(random_k1_rows) == 4960
assert len(random_seqkd_rows) == 4960
assert len(same_source_seqkd_rows) == 4960
assert len(full_seqkd_rows) == 20000


# Most important exact-source treatment contrast.
if ids(k2_pe_rows) != ids(
    same_source_seqkd_rows
):
    raise RuntimeError(
        "K2PE / SameSourceSeqKD "
        "index/order mismatch"
    )


for a, b in zip(
    k2_pe_rows,
    same_source_seqkd_rows,
):
    if a["source"] != b["source"]:
        raise RuntimeError(
            "K2PE / SameSourceSeqKD "
            f"source mismatch index={a['index']}"
        )

print(
    "K2PE_SAMESOURCE_SEQKD_EXACT_SOURCE_MATCH = "
    "4960 / 4960"
)


# K2 PE must be exact K1 targets on those sources.
for row in k2_pe_rows:
    idx = row["index"]

    if (
        row["target_translation"]
        !=
        str(
            k1[idx]["post_edit"]
        ).strip()
    ):
        raise RuntimeError(
            f"K2PE target lineage failure index={idx}"
        )


# Random K1 must use exact same K1 target treatment.
for row in random_k1_rows:
    idx = row["index"]

    if (
        row["target_translation"]
        !=
        str(
            k1[idx]["post_edit"]
        ).strip()
    ):
        raise RuntimeError(
            f"RandomK1 target lineage failure index={idx}"
        )


# All SeqKD targets must come from one fresh Teacher20k asset.
for rows in (
    random_seqkd_rows,
    same_source_seqkd_rows,
    full_seqkd_rows,
):
    for row in rows:
        idx = row["index"]

        if (
            row["target_translation"]
            !=
            teacher[idx]["target_translation"].strip()
        ):
            raise RuntimeError(
                f"SeqKD target lineage failure index={idx}"
            )


###############################################################################
# Freeze files.
###############################################################################

write_jsonl_atomic(
    K1_ALL,
    k1_all_rows,
)

write_jsonl_atomic(
    K2_PE,
    k2_pe_rows,
)

write_jsonl_atomic(
    RAND_K1,
    random_k1_rows,
)

write_jsonl_atomic(
    RAND_SEQKD,
    random_seqkd_rows,
)

write_jsonl_atomic(
    SAME_SEQKD,
    same_source_seqkd_rows,
)

write_jsonl_atomic(
    FULL_SEQKD,
    full_seqkd_rows,
)


write_ids_atomic(
    K2_IDS_FILE,
    k2_ids,
)

write_ids_atomic(
    RAND_K1_IDS_FILE,
    random_k1_ids,
)

write_ids_atomic(
    RAND_SEQKD_IDS_FILE,
    random_seqkd_ids,
)


###############################################################################
# Diagnostics that are informative but NOT constraints.
###############################################################################

k2_random_k1_overlap = len(
    set(k2_ids)
    &
    set(random_k1_ids)
)

k2_random_seqkd_overlap = len(
    set(k2_ids)
    &
    set(random_seqkd_ids)
)

random_control_overlap = len(
    set(random_k1_ids)
    &
    set(random_seqkd_ids)
)


###############################################################################
# Freeze manifest.
###############################################################################

dataset_specs = {
    "K1_ALL_PE11792": {
        "path": str(K1_ALL),
        "rows": len(k1_all_rows),
        "sha256": sha256(K1_ALL),
        "index_universe":
            "all frozen K1 PE-eligible rows",
        "target":
            "K1 conservative post_edit",
    },

    "K2_CONSISTENT_PE4960": {
        "path": str(K2_PE),
        "rows": len(k2_pe_rows),
        "sha256": sha256(K2_PE),
        "index_universe":
            "K2-consistent subset of K1-eligible",
        "target":
            "K1 conservative post_edit",
    },

    "RANDOM_K1_PE4960": {
        "path": str(RAND_K1),
        "rows": len(random_k1_rows),
        "sha256": sha256(RAND_K1),
        "sampling_seed": SEED,
        "rng_stream":
            f"{SEED}:RANDOM_K1_PE",
        "index_universe":
            "uniform sample from 11792 K1-eligible rows",
        "target":
            "K1 conservative post_edit",
    },

    "RANDOM_SEQKD4960": {
        "path": str(RAND_SEQKD),
        "rows": len(random_seqkd_rows),
        "sha256": sha256(RAND_SEQKD),
        "sampling_seed": SEED,
        "rng_stream":
            f"{SEED}:RANDOM_SEQKD",
        "index_universe":
            "uniform sample from frozen Broad20k",
        "target":
            "fresh Qwen3-8B Teacher translation",
    },

    "SAME_SOURCE_SEQKD4960": {
        "path": str(SAME_SEQKD),
        "rows": len(same_source_seqkd_rows),
        "sha256": sha256(SAME_SEQKD),
        "index_universe":
            "exact same 4960 IDs/order as K2-consistent PE",
        "target":
            "fresh Qwen3-8B Teacher translation",
    },

    "FULL_SEQKD20000": {
        "path": str(FULL_SEQKD),
        "rows": len(full_seqkd_rows),
        "sha256": sha256(FULL_SEQKD),
        "index_universe":
            "all frozen Broad20k rows",
        "target":
            "fresh Qwen3-8B Teacher translation",
    },
}


manifest = {
    "protocol":
        "STRONG_REPRO_STUDENT_ARMS_FREEZE_V1",

    "status":
        "DATASET_FREEZE_ONLY",

    "student_training_started":
        False,

    "sampling": {
        "seed": SEED,
        "policy": (
            "Random-K1 is uniform without replacement "
            "from frozen K1-eligible11792. "
            "Random-SeqKD is uniform without replacement "
            "from frozen Broad20k. "
            "The two use independent deterministic RNG streams."
        ),
    },

    "upstream": {
        "broad20k": {
            "path": str(BROAD_FILE),
            "sha256": sha256(BROAD_FILE),
        },

        "feedback_jobs20k": {
            "path": str(JOBS_FILE),
            "sha256": sha256(JOBS_FILE),
        },

        "fresh_teacher20k": {
            "path": str(TEACHER_FILE),
            "sha256": sha256(TEACHER_FILE),
        },

        "historical_k2_accept_artifact": {
            "path": str(K2_FILE),
            "sha256": sha256(K2_FILE),

            "terminology_note": (
                "The upstream filename contains "
                "'verified_clean', but downstream "
                "scientific terminology is "
                "'K2-consistent', because semantic "
                "construct validation showed only "
                "moderate enrichment, not semantic cleanliness."
            ),
        },

        "k1_shard_dir":
            str(K1_SHARDS),
    },

    "cardinality": {
        "broad20k": 20000,
        "k1_eligible": 11792,
        "k2_consistent": 4960,
    },

    "causal_checks": {
        "k2_pe_same_source_seqkd_exact_ids":
            True,

        "k2_pe_same_source_seqkd_exact_sources":
            True,

        "all_pe_targets_from_frozen_k1_post_edit":
            True,

        "all_seqkd_targets_from_single_fresh_teacher20k":
            True,

        "common_input_prompt_schema":
            True,
    },

    "informative_overlap_counts": {
        "k2_vs_random_k1":
            k2_random_k1_overlap,

        "k2_vs_random_seqkd":
            k2_random_seqkd_overlap,

        "random_k1_vs_random_seqkd":
            random_control_overlap,
    },

    "id_files": {
        "k2_consistent":
            str(K2_IDS_FILE),

        "random_k1":
            str(RAND_K1_IDS_FILE),

        "random_seqkd":
            str(RAND_SEQKD_IDS_FILE),
    },

    "datasets":
        dataset_specs,

    "primary_pre_registered_contrasts": {
        "K2_minus_RandomK1":
            (
                "Does K2 target-quality enrichment "
                "translate into Student-level utility?"
            ),

        "SameSourceSeqKD_minus_K2":
            (
                "At exactly matched K2-selected sources, "
                "what target-treatment gap remains between "
                "full Teacher SeqKD and PE?"
            ),

        "K2_vs_RandomSeqKD":
            (
                "Paper-facing equal-budget comparison; "
                "not a standalone verdict on the complete "
                "PE->PDS->WA reproduction."
            ),

        "K1All_vs_K2":
            (
                "Quality-versus-quantity/diversity trade-off "
                "induced by K2 filtering."
            ),

        "FullSeqKD":
            (
                "Fresh Broad20k direct-KD anchor for this "
                "Strong-Reproduction run."
            ),
    },
}


tmp_manifest = MANIFEST.with_suffix(
    ".json.tmp"
)

tmp_manifest.write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

tmp_manifest.replace(
    MANIFEST
)


###############################################################################
# Final concise audit output.
###############################################################################

print()
print("=" * 78)
print("STRONG REPRO STUDENT ARM DATASET FREEZE RESULT")
print("=" * 78)

for name, x in dataset_specs.items():
    print(
        f"{name:28s} "
        f"rows={x['rows']:5d} "
        f"sha256={x['sha256']}"
    )

print()
print(
    "K2PE_SAMESOURCE_SEQKD_IDS_EXACT =",
    ids(k2_pe_rows)
    ==
    ids(same_source_seqkd_rows),
)

print(
    "K2PE_SAMESOURCE_SEQKD_SOURCE_EXACT =",
    all(
        a["source"] == b["source"]
        for a, b in zip(
            k2_pe_rows,
            same_source_seqkd_rows,
        )
    ),
)

print()
print(
    "K2_vs_RANDOM_K1_OVERLAP =",
    k2_random_k1_overlap,
)

print(
    "K2_vs_RANDOM_SEQKD_OVERLAP =",
    k2_random_seqkd_overlap,
)

print(
    "RANDOM_K1_vs_RANDOM_SEQKD_OVERLAP =",
    random_control_overlap,
)

print()
print(
    "MANIFEST =",
    MANIFEST,
)

print(
    "DATASET_FREEZE_CONSTRUCTION_PASS"
)
PY


###############################################################################
# STAGE 3/5 — cardinality and format audit
###############################################################################

echo
echo "======================================================================"
echo "STAGE 3/5 — CARDINALITY / FORMAT AUDIT"
echo "======================================================================"

declare -A EXPECTED

EXPECTED["$K1_ALL"]=11792
EXPECTED["$K2_PE"]=4960
EXPECTED["$RAND_K1"]=4960
EXPECTED["$RAND_SEQKD"]=4960
EXPECTED["$SAME_SEQKD"]=4960
EXPECTED["$FULL_SEQKD"]=20000

for F in \
    "$K1_ALL" \
    "$K2_PE" \
    "$RAND_K1" \
    "$RAND_SEQKD" \
    "$SAME_SEQKD" \
    "$FULL_SEQKD"
do
    if [ ! -f "$F" ]; then
        echo "MISSING OUTPUT: $F"
        false
    fi

    ACTUAL="$(wc -l < "$F")"
    WANT="${EXPECTED[$F]}"

    printf "%-45s actual=%5d expected=%5d\n" \
        "$(basename "$F")" \
        "$ACTUAL" \
        "$WANT"

    if [ "$ACTUAL" -ne "$WANT" ]; then
        echo "CARDINALITY_FAILURE=$F"
        false
    fi
done

echo
echo "CARDINALITY_AUDIT_PASS"


###############################################################################
# STAGE 4/5 — final independent causal audit
###############################################################################

echo
echo "======================================================================"
echo "STAGE 4/5 — INDEPENDENT CAUSAL RELATION AUDIT"
echo "======================================================================"

python - \
    "$K1_ALL" \
    "$K2_PE" \
    "$RAND_K1" \
    "$RAND_SEQKD" \
    "$SAME_SEQKD" \
    "$FULL_SEQKD" <<'PY'

import json
import sys
from pathlib import Path

paths = [
    Path(x)
    for x in sys.argv[1:]
]


def load(path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


(
    k1,
    k2,
    rk1,
    rs,
    ss,
    full,
) = [
    load(p)
    for p in paths
]


def ids(rows):
    return [
        int(x["index"])
        for x in rows
    ]


def unique(rows):
    x = ids(rows)
    return len(x) == len(set(x))


for name, rows in [
    ("K1_ALL", k1),
    ("K2_PE", k2),
    ("RANDOM_K1", rk1),
    ("RANDOM_SEQKD", rs),
    ("SAME_SOURCE_SEQKD", ss),
    ("FULL_SEQKD", full),
]:
    if not unique(rows):
        raise RuntimeError(
            f"duplicate IDs in {name}"
        )


if ids(k2) != ids(ss):
    raise RuntimeError(
        "critical failure: K2 PE and "
        "SameSourceSeqKD IDs/order differ"
    )


if any(
    a["source"] != b["source"]
    for a, b in zip(k2, ss)
):
    raise RuntimeError(
        "critical failure: K2 PE and "
        "SameSourceSeqKD sources differ"
    )


if set(ids(k2)) - set(ids(k1)):
    raise RuntimeError(
        "K2 is not subset of K1-All"
    )


if set(ids(rk1)) - set(ids(k1)):
    raise RuntimeError(
        "Random-K1 is not subset of K1-All"
    )


if set(ids(rs)) - set(ids(full)):
    raise RuntimeError(
        "Random-SeqKD is not subset of FullSeqKD"
    )


if set(ids(ss)) - set(ids(full)):
    raise RuntimeError(
        "SameSourceSeqKD is not subset of FullSeqKD"
    )


# Common message format across all arms.
for name, rows in [
    ("K1_ALL", k1),
    ("K2_PE", k2),
    ("RANDOM_K1", rk1),
    ("RANDOM_SEQKD", rs),
    ("SAME_SOURCE_SEQKD", ss),
    ("FULL_SEQKD", full),
]:
    for x in rows:
        msgs = x.get("messages")

        if (
            not isinstance(msgs, list)
            or len(msgs) != 1
            or msgs[0].get("role") != "user"
            or not isinstance(
                msgs[0].get("content"),
                str,
            )
            or not isinstance(
                x.get("target_translation"),
                str,
            )
            or not x["target_translation"].strip()
        ):
            raise RuntimeError(
                f"training schema failure "
                f"arm={name} index={x.get('index')}"
            )


print(
    "K2_PE ⊂ K1_ALL =",
    len(k2),
    "/",
    len(k1),
)

print(
    "RANDOM_K1 ⊂ K1_ALL =",
    len(rk1),
    "/",
    len(k1),
)

print(
    "RANDOM_SEQKD ⊂ FULL_SEQKD =",
    len(rs),
    "/",
    len(full),
)

print(
    "SAME_SOURCE_SEQKD ⊂ FULL_SEQKD =",
    len(ss),
    "/",
    len(full),
)

print(
    "K2_PE ↔ SAME_SOURCE_SEQKD "
    "exact matched sources = 4960 / 4960"
)

print(
    "INDEPENDENT_CAUSAL_RELATION_AUDIT_PASS"
)
PY


###############################################################################
# STAGE 5/5 — freeze hashes and sentinel
###############################################################################

echo
echo "======================================================================"
echo "STAGE 5/5 — FINAL FREEZE"
echo "======================================================================"

HASHES="$OUT/frozen_sha256_manifest_v1.txt"

{
    echo "DATE=$(date -Iseconds)"
    echo "SAMPLE_SEED=$SAMPLE_SEED"

    echo
    echo "===== SIX STUDENT ARMS ====="

    sha256sum \
        "$K1_ALL" \
        "$K2_PE" \
        "$RAND_K1" \
        "$RAND_SEQKD" \
        "$SAME_SEQKD" \
        "$FULL_SEQKD"

    echo
    echo "===== ID SETS ====="

    sha256sum \
        "$K2_IDS" \
        "$RAND_K1_IDS" \
        "$RAND_SEQKD_IDS"

    echo
    echo "===== MANIFEST ====="

    sha256sum "$MANIFEST"

} > "$HASHES"

cat "$HASHES"

touch "$PASS"
rm -f "$FAIL"

END_EPOCH="$(date +%s)"
ELAPSED=$((END_EPOCH - START_EPOCH))

echo
echo "======================================================================"
echo "STRONG REPRO STUDENT ARMS FREEZE V1 PASS"
echo "======================================================================"
echo "实际运行时长: ${ELAPSED} 秒"
echo "完成时间:"
echo "  北京时间: $(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
echo "  Server UTC: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
echo
echo "PASS=$PASS"
echo "MANIFEST=$MANIFEST"
echo "======================================================================"

cat "$MANIFEST"

fi
