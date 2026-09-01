#!/usr/bin/env bash
set -Eeuo pipefail

source /workspace/mtpatcher/project_env.sh

EXP="mtpatcher_v3_full6565_20260823"
D="$DATA_ROOT/$EXP"

###############################################################################
# Frozen inputs
###############################################################################

BROAD="$D/paperfaith_paper20k_v2/demo_pool.jsonl"

JOBS="$D/paperfaith_paper20k_v2/feedback_jobs.jsonl"

GEN="$ROOT/scripts/mtpatcher_rq0/generate_seqkd50k_teacher_v1.py"

MODEL="$MODEL_ROOT/Qwen3-8B"

###############################################################################
# New fresh Teacher asset
###############################################################################

OUT="$D/strong_repro_broad20k_seqkd_same_source_v1"

INPUT="$OUT/teacher_input_broad20k_v1.jsonl"

SHARD_DIR="$OUT/teacher_shards16"
WORKER_LOG_DIR="$OUT/worker_logs"

MERGED="$OUT/seqkd_broad20000_qwen3_8b_fresh_v1.jsonl"

MANIFEST="$OUT/teacher_broad20k_manifest_v1.json"

PASS="$OUT/STRONG_REPRO_BROAD20K_TEACHER_SEQKD_V1.PASS"
FAIL="$OUT/STRONG_REPRO_BROAD20K_TEACHER_SEQKD_V1.FAIL"

###############################################################################
# Frozen hashes already established by provenance audit
###############################################################################

EXPECTED_BROAD_SHA="2ab671159dd79cce181f4a38f951ad3d93d15575dd1c74e398b05bbc10e5b59e"

EXPECTED_JOBS_SHA="1ac060f2cdb03ef3011bd1e0f5f4b28de831e2f921ccf2a89f1f4d6cfb5edaaa"

EXPECTED_GEN_SHA="5113d4a8a323200cd80033f427e8dac2884e20dfc656ea82ca5aa99e26f665e8"

EXPECTED_MODEL_CONFIG_SHA="f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30"

EXPECTED_GENERATION_CONFIG_SHA="2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2"

EXPECTED_TOKENIZER_CONFIG_SHA="d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"

EXPECTED_TOKENIZER_SHA="aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"

mkdir -p \
    "$OUT" \
    "$SHARD_DIR" \
    "$WORKER_LOG_DIR"

rm -f "$PASS" "$FAIL"

trap '
rc=$?
echo
echo "======================================================================"
echo "STRONG REPRO BROAD20K TEACHER SEQKD V1 FAILED"
echo "return_code=$rc"
date
echo "======================================================================"
touch "'"$FAIL"'"
' ERR


echo "======================================================================"
echo "STRONG REPRO — FRESH EXACT-SOURCE BROAD20K TEACHER SEQKD V1"
date
echo "======================================================================"

echo
echo "BROAD=$BROAD"
echo "JOBS=$JOBS"
echo "GEN=$GEN"
echo "MODEL=$MODEL"
echo "OUT=$OUT"
echo


###############################################################################
# STAGE 1/6
# Static provenance lock.
###############################################################################

echo "======================================================================"
echo "STAGE 1/6 — STATIC PROVENANCE LOCK"
echo "======================================================================"

for F in \
    "$BROAD" \
    "$JOBS" \
    "$GEN" \
    "$MODEL/config.json" \
    "$MODEL/generation_config.json" \
    "$MODEL/tokenizer_config.json" \
    "$MODEL/tokenizer.json"
do
    if [ ! -f "$F" ]; then
        echo "MISSING REQUIRED FILE: $F"
        false
    fi
done


check_sha() {
    local FILE="$1"
    local EXPECTED="$2"
    local NAME="$3"

    local ACTUAL
    ACTUAL="$(sha256sum "$FILE" | awk '{print $1}')"

    echo "$NAME"
    echo "  file=$FILE"
    echo "  expected=$EXPECTED"
    echo "  actual=$ACTUAL"

    if [ "$ACTUAL" != "$EXPECTED" ]; then
        echo "SHA_MISMATCH: $NAME"
        false
    fi
}


check_sha \
    "$BROAD" \
    "$EXPECTED_BROAD_SHA" \
    "BROAD20K"

check_sha \
    "$JOBS" \
    "$EXPECTED_JOBS_SHA" \
    "FEEDBACK_JOBS20K"

check_sha \
    "$GEN" \
    "$EXPECTED_GEN_SHA" \
    "TEACHER_GENERATOR"

check_sha \
    "$MODEL/config.json" \
    "$EXPECTED_MODEL_CONFIG_SHA" \
    "MODEL_CONFIG"

check_sha \
    "$MODEL/generation_config.json" \
    "$EXPECTED_GENERATION_CONFIG_SHA" \
    "GENERATION_CONFIG"

check_sha \
    "$MODEL/tokenizer_config.json" \
    "$EXPECTED_TOKENIZER_CONFIG_SHA" \
    "TOKENIZER_CONFIG"

check_sha \
    "$MODEL/tokenizer.json" \
    "$EXPECTED_TOKENIZER_SHA" \
    "TOKENIZER_JSON"

echo
echo "STATIC_PROVENANCE_LOCK_PASS"


###############################################################################
# STAGE 2/6
# Build exact Broad20k Teacher input.
#
# Canonical IDs come from feedback_jobs because K1/K2 already use that lineage.
# demo_pool and feedback_jobs must agree source-by-source.
###############################################################################

echo
echo "======================================================================"
echo "STAGE 2/6 — BUILD EXACT BROAD20K TEACHER INPUT"
echo "======================================================================"

python - \
    "$BROAD" \
    "$JOBS" \
    "$INPUT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

BROAD = Path(sys.argv[1])
JOBS = Path(sys.argv[2])
OUT = Path(sys.argv[3])


def load(path):
    rows = []

    with path.open(
        encoding="utf-8-sig"
    ) as f:

        for line_no, line in enumerate(f, 1):

            line = line.strip()

            if not line:
                continue

            x = json.loads(line)

            rows.append(x)

    return rows


def get_idx(x):

    if x.get("index") is not None:
        return int(x["index"])

    if x.get("demo_id") is not None:
        return int(x["demo_id"])

    raise RuntimeError(
        "row has neither index nor demo_id; "
        f"keys={sorted(x.keys())}"
    )


def get_source(x):

    source = x.get("source")

    if not isinstance(source, str):
        raise RuntimeError(
            "source is missing/non-string; "
            f"keys={sorted(x.keys())}"
        )

    source = source.strip()

    if not source:
        raise RuntimeError("empty source")

    return source


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for b in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(b)

    return h.hexdigest()


broad = load(BROAD)
jobs = load(JOBS)

print("BROAD_ROWS =", len(broad))
print("JOBS_ROWS =", len(jobs))

if len(broad) != 20000:
    raise RuntimeError(
        f"Broad expected 20000, got {len(broad)}"
    )

if len(jobs) != 20000:
    raise RuntimeError(
        f"feedback_jobs expected 20000, got {len(jobs)}"
    )


###############################################################################
# Broad pool ↔ feedback_jobs exact source sequence.
###############################################################################

source_mismatch = []

for pos, (b, j) in enumerate(
    zip(broad, jobs)
):

    bs = get_source(b)
    js = get_source(j)

    if bs != js:

        source_mismatch.append({
            "position": pos,
            "broad_source": bs,
            "jobs_source": js,
        })

        if len(source_mismatch) >= 20:
            break


print(
    "BROAD_JOBS_SOURCE_SEQUENCE_MISMATCH =",
    len(source_mismatch),
)

if source_mismatch:
    raise RuntimeError(
        "Broad20k and feedback_jobs source "
        f"sequence mismatch: {source_mismatch}"
    )


###############################################################################
# Use K1/K2 canonical IDs from feedback_jobs.
###############################################################################

prepared = []

for row in jobs:

    idx = get_idx(row)
    source = get_source(row)

    prepared.append({
        "index": idx,
        "source": source,
    })


ids = [
    x["index"]
    for x in prepared
]

if len(ids) != len(set(ids)):
    raise RuntimeError(
        "duplicate canonical index in feedback_jobs"
    )


expected = set(range(20000))

actual = set(ids)

if actual != expected:

    missing = sorted(
        expected - actual
    )[:30]

    extra = sorted(
        actual - expected
    )[:30]

    raise RuntimeError(
        "canonical Broad20k ID set is not 0..19999; "
        f"missing={missing}, extra={extra}"
    )


prepared.sort(
    key=lambda x: x["index"]
)


###############################################################################
# Confirm sorted ID→source mapping is still exactly the same Broad20k set.
###############################################################################

jobs_by_idx = {
    get_idx(x): get_source(x)
    for x in jobs
}

for x in prepared:

    if (
        jobs_by_idx[x["index"]]
        != x["source"]
    ):
        raise RuntimeError(
            f"internal mapping error index={x['index']}"
        )


with OUT.open(
    "w",
    encoding="utf-8",
) as f:

    for x in prepared:

        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )


print("TEACHER_INPUT_ROWS =", len(prepared))
print("FIRST_INDEX =", prepared[0]["index"])
print("LAST_INDEX =", prepared[-1]["index"])
print("INPUT_SHA256 =", sha256(OUT))
print("BROAD20K_TEACHER_INPUT_BUILD_PASS")
PY


echo
wc -l "$INPUT"
sha256sum "$INPUT"


###############################################################################
# STAGE 3/6
# Fresh Teacher generation — exact historical treatment.
#
# Important: output shards are NOT deleted.
# Historical generator is resume-aware.
###############################################################################

echo
echo "======================================================================"
echo "STAGE 3/6 — FRESH QWEN3-8B TEACHER GENERATION"
echo "======================================================================"

declare -A PIDS

for DEVICE in $(seq 0 15); do

    OUTPUT="$SHARD_DIR/device_${DEVICE}.jsonl"
    LOG="$WORKER_LOG_DIR/device_${DEVICE}.log"

    echo \
        "TEACHER_WORKER_START " \
        "device=$DEVICE " \
        "output=$OUTPUT"

    python -u "$GEN" \
        --input "$INPUT" \
        --output "$OUTPUT" \
        --model "$MODEL" \
        --device-id "$DEVICE" \
        --world-size 16 \
        --batch-size 16 \
        --max-new-tokens 512 \
        > "$LOG" 2>&1 &

    PIDS[$DEVICE]=$!

    echo \
        "TEACHER_WORKER_PID " \
        "device=$DEVICE " \
        "pid=${PIDS[$DEVICE]}"
done


FAILED_DEVICES=()

for DEVICE in $(seq 0 15); do

    PID="${PIDS[$DEVICE]}"

    if wait "$PID"; then

        echo \
            "TEACHER_WORKER_FINISH " \
            "device=$DEVICE status=0"

    else

        STATUS=$?

        echo \
            "TEACHER_WORKER_FINISH " \
            "device=$DEVICE status=$STATUS"

        FAILED_DEVICES+=("$DEVICE")
    fi
done


echo
echo "FIRST_PASS_FAILED_DEVICES=${FAILED_DEVICES[*]:-none}"


###############################################################################
# STAGE 4/6
# Global integrity audit + one automatic resume retry.
###############################################################################

echo
echo "======================================================================"
echo "STAGE 4/6 — GLOBAL INTEGRITY / OPTIONAL RETRY"
echo "======================================================================"

python - \
    "$INPUT" \
    "$SHARD_DIR" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

INPUT = Path(sys.argv[1])
SHARDS = Path(sys.argv[2])


def load(path):

    if not path.exists():
        return []

    rows = []

    with path.open(
        encoding="utf-8-sig"
    ) as f:

        for line in f:

            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


inp = load(INPUT)

expected = {
    int(x["index"])
    for x in inp
}

all_rows = []

for d in range(16):

    p = SHARDS / f"device_{d}.jsonl"

    rows = load(p)

    all_rows.extend(rows)

    ids = [
        int(x["index"])
        for x in rows
    ]

    print(
        f"device={d:02d} "
        f"rows={len(rows)} "
        f"unique={len(set(ids))}"
    )


ids = [
    int(x["index"])
    for x in all_rows
]

counts = Counter(ids)

duplicates = sorted(
    idx
    for idx, n in counts.items()
    if n > 1
)

got = set(ids)

missing = sorted(
    expected - got
)

extra = sorted(
    got - expected
)

print()
print("RAW_ROWS =", len(all_rows))
print("UNIQUE_IDS =", len(got))
print("DUPLICATE_IDS =", len(duplicates))
print("MISSING_IDS =", len(missing))
print("EXTRA_IDS =", len(extra))

if duplicates:
    print(
        "DUPLICATE_EXAMPLES =",
        duplicates[:30],
    )

if missing:
    print(
        "MISSING_EXAMPLES =",
        missing[:30],
    )

if extra:
    print(
        "EXTRA_EXAMPLES =",
        extra[:30],
    )

if (
    len(got) == 20000
    and
    not duplicates
    and
    not missing
    and
    not extra
):
    print(
        "GLOBAL_GENERATION_INTEGRITY_PASS"
    )
else:
    print(
        "GLOBAL_GENERATION_INTEGRITY_INCOMPLETE"
    )
PY


NEED_RETRY="$(
python - "$INPUT" "$SHARD_DIR" <<'PY'
import json
import sys
from pathlib import Path

INPUT = Path(sys.argv[1])
SHARDS = Path(sys.argv[2])


def load(path):

    if not path.exists():
        return []

    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]


expected = {
    int(x["index"])
    for x in load(INPUT)
}

got = set()
duplicate = False

for d in range(16):

    rows = load(
        SHARDS
        / f"device_{d}.jsonl"
    )

    ids = [
        int(x["index"])
        for x in rows
    ]

    if len(ids) != len(set(ids)):
        duplicate = True

    for idx in ids:

        if idx in got:
            duplicate = True

        got.add(idx)


if (
    got == expected
    and
    not duplicate
):
    print("0")
else:
    print("1")
PY
)"


if [ "$NEED_RETRY" = "1" ]; then

    echo
    echo "GLOBAL OUTPUT INCOMPLETE — ONE RESUME RETRY"

    declare -A RETRY_PIDS

    for DEVICE in $(seq 0 15); do

        OUTPUT="$SHARD_DIR/device_${DEVICE}.jsonl"
        LOG="$WORKER_LOG_DIR/device_${DEVICE}.log"

        python -u "$GEN" \
            --input "$INPUT" \
            --output "$OUTPUT" \
            --model "$MODEL" \
            --device-id "$DEVICE" \
            --world-size 16 \
            --batch-size 16 \
            --max-new-tokens 512 \
            >> "$LOG" 2>&1 &

        RETRY_PIDS[$DEVICE]=$!

        echo \
            "RETRY_WORKER_START " \
            "device=$DEVICE " \
            "pid=${RETRY_PIDS[$DEVICE]}"
    done


    for DEVICE in $(seq 0 15); do

        PID="${RETRY_PIDS[$DEVICE]}"

        if wait "$PID"; then

            echo \
                "RETRY_WORKER_FINISH " \
                "device=$DEVICE status=0"

        else

            STATUS=$?

            echo \
                "RETRY_WORKER_FINISH " \
                "device=$DEVICE status=$STATUS"
        fi
    done

else

    echo "NO_RETRY_REQUIRED"
fi


###############################################################################
# STAGE 5/6
# Merge and perform final semantic-free asset integrity QA.
###############################################################################

echo
echo "======================================================================"
echo "STAGE 5/6 — MERGE + FINAL ASSET QA"
echo "======================================================================"

python - \
    "$INPUT" \
    "$SHARD_DIR" \
    "$MERGED" \
    "$MANIFEST" \
    "$BROAD" \
    "$JOBS" \
    "$GEN" \
    "$MODEL" <<'PY'

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

INPUT = Path(sys.argv[1])
SHARDS = Path(sys.argv[2])
MERGED = Path(sys.argv[3])
MANIFEST = Path(sys.argv[4])
BROAD = Path(sys.argv[5])
JOBS = Path(sys.argv[6])
GEN = Path(sys.argv[7])
MODEL = Path(sys.argv[8])


PROMPT = (
    "Translate the following text into English "
    "without additional explanations:\n\n"
    "{source}\n\n"
)


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
                x = json.loads(line)

            except Exception as e:
                raise RuntimeError(
                    f"JSON parse failure "
                    f"{path}:{line_no}: {e}"
                )

            rows.append(x)

    return rows


def sha256(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for b in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(b)

    return h.hexdigest()


inp_rows = load(INPUT)

if len(inp_rows) != 20000:
    raise RuntimeError(
        f"teacher input rows !=20000: "
        f"{len(inp_rows)}"
    )

inp = {
    int(x["index"]): x
    for x in inp_rows
}

if len(inp) != 20000:
    raise RuntimeError(
        "duplicate teacher input indices"
    )


all_rows = []

for d in range(16):

    p = (
        SHARDS
        / f"device_{d}.jsonl"
    )

    if not p.exists():
        raise RuntimeError(
            f"missing shard {p}"
        )

    rows = load(p)

    print(
        f"MERGE_SHARD "
        f"device={d} rows={len(rows)}"
    )

    all_rows.extend(rows)


if len(all_rows) != 20000:
    raise RuntimeError(
        "fresh Teacher raw row count "
        f"expected20000 got={len(all_rows)}"
    )


out = {
    int(x["index"]): x
    for x in all_rows
}

if len(out) != 20000:
    raise RuntimeError(
        "duplicate Teacher output indices"
    )


expected_ids = set(range(20000))

if set(inp) != expected_ids:
    raise RuntimeError(
        "teacher input index set !=0..19999"
    )

if set(out) != expected_ids:
    missing = sorted(
        expected_ids - set(out)
    )[:30]

    extra = sorted(
        set(out) - expected_ids
    )[:30]

    raise RuntimeError(
        f"Teacher output index mismatch "
        f"missing={missing} extra={extra}"
    )


counts = Counter()

bad_examples = []


for idx in range(20000):

    src = inp[idx]["source"]
    row = out[idx]

    source_exact = (
        row.get("source") == src
    )

    expected_messages = [
        {
            "role": "user",
            "content":
                PROMPT.format(
                    source=src
                ),
        }
    ]

    messages_exact = (
        row.get("messages")
        == expected_messages
    )

    target = row.get(
        "target_translation"
    )

    target_nonempty = (
        isinstance(target, str)
        and bool(target.strip())
    )

    teacher_exact = (
        row.get("teacher_model")
        == str(MODEL)
    )

    thinking_exact = (
        row.get("teacher_thinking")
        is False
    )

    sample_exact = (
        row.get("do_sample")
        is False
    )

    max_tokens_exact = (
        row.get("max_new_tokens")
        == 512
    )

    construction_exact = (
        row.get("construction_method")
        ==
        "RQ0_SEQKD_NEWCRAWL_QWEN3_8B"
    )

    new_tokens = row.get(
        "new_tokens"
    )

    new_tokens_valid = (
        isinstance(new_tokens, int)
        and
        0 < new_tokens <= 512
    )

    hit_max = (
        new_tokens == 512
    )


    checks = {
        "source_exact":
            source_exact,

        "messages_exact":
            messages_exact,

        "target_nonempty":
            target_nonempty,

        "teacher_exact":
            teacher_exact,

        "thinking_exact":
            thinking_exact,

        "do_sample_exact":
            sample_exact,

        "max_tokens_exact":
            max_tokens_exact,

        "construction_exact":
            construction_exact,

        "new_tokens_valid":
            new_tokens_valid,
    }


    for key, value in checks.items():

        if value:
            counts[key] += 1


    if hit_max:
        counts["hit_max_new_tokens"] += 1


    if not all(checks.values()):

        if len(bad_examples) < 30:

            bad_examples.append({
                "index": idx,
                "checks": checks,
                "row": row,
            })


required = [
    "source_exact",
    "messages_exact",
    "target_nonempty",
    "teacher_exact",
    "thinking_exact",
    "do_sample_exact",
    "max_tokens_exact",
    "construction_exact",
    "new_tokens_valid",
]


for key in required:

    if counts[key] != 20000:

        raise RuntimeError(
            f"final QA failed "
            f"{key}={counts[key]}/20000; "
            f"examples={bad_examples[:3]}"
        )


###############################################################################
# Freeze merged asset in canonical index order.
###############################################################################

with MERGED.open(
    "w",
    encoding="utf-8",
) as f:

    for idx in range(20000):

        f.write(
            json.dumps(
                out[idx],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )


manifest = {
    "protocol":
        "STRONG_REPRO_BROAD20K_FRESH_TEACHER_SEQKD_V1",

    "scientific_role":
        (
            "Single frozen exact-source Teacher pool "
            "for all Broad20k Strong-Reproduction "
            "SeqKD control arms."
        ),

    "status":
        "GENERATION_ONLY",

    "rows":
        20000,

    "source_pool": {
        "demo_pool_path":
            str(BROAD),

        "demo_pool_sha256":
            sha256(BROAD),

        "feedback_jobs_path":
            str(JOBS),

        "feedback_jobs_sha256":
            sha256(JOBS),

        "teacher_input_path":
            str(INPUT),

        "teacher_input_sha256":
            sha256(INPUT),
    },

    "teacher": {
        "model":
            str(MODEL),

        "generator":
            str(GEN),

        "generator_sha256":
            sha256(GEN),

        "prompt":
            PROMPT,

        "enable_thinking":
            False,

        "do_sample":
            False,

        "max_new_tokens":
            512,

        "batch_size":
            16,

        "world_size":
            16,

        "dtype":
            "torch.bfloat16",

        "attn_implementation":
            "sdpa",
    },

    "qa": {
        key: counts[key]
        for key in required
    },

    "hit_max_new_tokens_count":
        counts["hit_max_new_tokens"],

    "merged_asset": {
        "path":
            str(MERGED),

        "sha256":
            sha256(MERGED),
    },

    "student_training_started":
        False,

    "fidelity_note": (
        "Fresh generation on the frozen Broad20k "
        "source pool using the already provenance-"
        "locked historical SeqKD Teacher treatment. "
        "No historical Teacher targets are mixed "
        "into this asset."
    ),
}


MANIFEST.write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("============================================================")
print("FRESH BROAD20K TEACHER FINAL QA")
print("============================================================")

print("ROWS = 20000")

for key in required:

    print(
        f"{key} = "
        f"{counts[key]} / 20000"
    )

print(
    "HIT_MAX_NEW_TOKENS =",
    counts["hit_max_new_tokens"],
)

print(
    "MERGED =",
    MERGED,
)

print(
    "MERGED_SHA256 =",
    sha256(MERGED),
)

print(
    "MANIFEST =",
    MANIFEST,
)

print(
    "FRESH_BROAD20K_TEACHER_ASSET_QA_PASS"
)
PY


###############################################################################
# STAGE 6/6
# Freeze hashes + PASS sentinel.
###############################################################################

echo
echo "======================================================================"
echo "STAGE 6/6 — FREEZE"
echo "======================================================================"

{
    echo "DATE=$(date -Iseconds)"

    echo
    echo "===== INPUTS ====="

    sha256sum \
        "$BROAD" \
        "$JOBS" \
        "$INPUT" \
        "$GEN"

    echo
    echo "===== MODEL CONFIG ====="

    sha256sum \
        "$MODEL/config.json" \
        "$MODEL/generation_config.json" \
        "$MODEL/tokenizer_config.json" \
        "$MODEL/tokenizer.json"

    echo
    echo "===== FRESH TEACHER ASSET ====="

    sha256sum \
        "$MERGED" \
        "$MANIFEST"

} > "$OUT/frozen_sha256_manifest_v1.txt"


cat "$OUT/frozen_sha256_manifest_v1.txt"

touch "$PASS"
rm -f "$FAIL"

echo
echo "======================================================================"
echo "STRONG REPRO BROAD20K FRESH TEACHER SEQKD V1 PASS"
date
echo
echo "PASS=$PASS"
echo "MERGED=$MERGED"
echo "MANIFEST=$MANIFEST"
echo "======================================================================"

cat "$MANIFEST"

