#!/usr/bin/env bash
set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Keep official repo importable if paper_repro_v2 transitively imports it.
export PYTHONPATH="$ROOT/vendor/MT-Patcher-official:${PYTHONPATH:-}"

EXP="mtpatcher_v3_full6565_20260823"

DIR="$ROOT/scripts/mtpatcher_paper_faithful_v2"
PIPE="$DIR/paper_repro_v2.py"

PATCHER="$RUN_ROOT/$EXP/patcher_qwen3_8b_paperfaith_fullft_v2/checkpoint-4752"

JOBS="$DATA_ROOT/$EXP/patcher_epoch3_curated6565_raw_v1/feedback_jobs6565.jsonl"

WORK="$DATA_ROOT/$EXP/patcher_epoch3_curated6565_raw_v2"
SHARDS="$WORK/shards"
RAW="$WORK/epoch3_feedback_raw6565.jsonl"
MANIFEST="$WORK/manifest.json"

LOGDIR="${LOG_ROOT:-/workspace/mtpatcher/logs}/$EXP/patcher_epoch3_curated6565_raw_v2"

mkdir -p "$WORK" "$SHARDS" "$LOGDIR"

echo "======================================================================"
echo "EPOCH3 PATCHER CURATED6565 — GENERATION CONTINUATION V2"
echo "======================================================================"
echo "MODEL=$PATCHER"
echo "JOBS=$JOBS"
echo "NO_PARSER=1"
echo "NO_SCORING=1"
echo "NO_STUDENT_TRAINING=1"
echo "NO_PDS_WA_OPD=1"

test -f "$PIPE"
test -f "$JOBS"
test -d "$PATCHER"

# ----------------------------------------------------------------------
# Re-audit the already frozen jobs. No regeneration.
# ----------------------------------------------------------------------

python - "$JOBS" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])

rows = [
    json.loads(x)
    for x in p.read_text(encoding="utf-8").splitlines()
    if x.strip()
]

print("JOBS_ROWS =", len(rows))

assert len(rows) == 6565

ids = [int(r["job_id"]) for r in rows]

assert len(ids) == len(set(ids))
assert set(ids) == set(range(6565))

temps = sorted(set(float(r["temperature"]) for r in rows))
max_tokens = sorted(set(int(r["max_new_tokens"]) for r in rows))

print("TEMPERATURE_VALUES =", temps)
print("MAX_NEW_TOKENS_VALUES =", max_tokens)

assert temps == [0.1]
assert max_tokens == [256]

print("FROZEN_JOB_AUDIT_PASS")
PY

# ----------------------------------------------------------------------
# 16-NPU final checkpoint generation.
# ----------------------------------------------------------------------

echo
echo "======================================================================"
echo "LAUNCHING 16 GENERATION WORKERS"
echo "======================================================================"

pids=()

for D in $(seq 0 15); do
    python -u "$PIPE" generate \
        --jobs "$JOBS" \
        --output "$SHARDS/device_${D}.jsonl" \
        --model "$PATCHER" \
        --device "$D" \
        --world-size 16 \
        --batch-size 8 \
        > "$LOGDIR/device_${D}.log" 2>&1 &

    pids+=("$!")
done

failed=0

for P in "${pids[@]}"; do
    if ! wait "$P"; then
        failed=1
    fi
done

if [ "$failed" -ne 0 ]; then
    echo "GENERATION_WORKER_FAILURE"

    for F in "$LOGDIR"/device_*.log; do
        echo
        echo "---------------- $F ----------------"
        tail -n 40 "$F" || true
    done

    test "$failed" -eq 0
fi

echo "ALL_16_GENERATION_WORKERS_PASS"

# ----------------------------------------------------------------------
# Exact shard merge.
# ----------------------------------------------------------------------

export JOBS SHARDS RAW

python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

jobs_path = Path(os.environ["JOBS"])
shard_dir = Path(os.environ["SHARDS"])
raw_path = Path(os.environ["RAW"])

jobs = [
    json.loads(x)
    for x in jobs_path.read_text(encoding="utf-8").splitlines()
    if x.strip()
]

expected = {int(x["job_id"]) for x in jobs}

files = sorted(shard_dir.glob("device_*.jsonl"))

print("SHARD_FILES =", len(files))
assert len(files) == 16

merged = {}

for p in files:
    local_n = 0

    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        r = json.loads(line)
        jid = int(r["job_id"])

        if jid in merged:
            raise RuntimeError(
                f"duplicate job_id={jid}, file={p}"
            )

        merged[jid] = r
        local_n += 1

    print(p.name, "ROWS =", local_n)

missing = expected - set(merged)
extra = set(merged) - expected

print("MERGED_ROWS =", len(merged))
print("MISSING =", len(missing))
print("EXTRA =", len(extra))

assert not missing, sorted(missing)[:20]
assert not extra, sorted(extra)[:20]
assert len(merged) == 6565

ordered = [merged[i] for i in sorted(expected)]

with raw_path.open("w", encoding="utf-8") as f:
    for r in ordered:
        f.write(
            json.dumps(r, ensure_ascii=False)
            + "\n"
        )

blob = raw_path.read_bytes()

print("RAW =", raw_path)
print("RAW_ROWS =", len(ordered))
print("RAW_BYTES =", len(blob))
print("RAW_SHA256 =", hashlib.sha256(blob).hexdigest())
print("MERGE_COVERAGE_PASS")
PY

# ----------------------------------------------------------------------
# Structural audit only.
# No semantic interpretation tonight.
# ----------------------------------------------------------------------

export RAW MANIFEST PATCHER JOBS

python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

raw = Path(os.environ["RAW"])
manifest = Path(os.environ["MANIFEST"])

rows = [
    json.loads(x)
    for x in raw.read_text(encoding="utf-8").splitlines()
    if x.strip()
]

assert len(rows) == 6565

print("FIRST_ROW_KEYS =", sorted(rows[0].keys()))

candidate_fields = [
    "response",
    "output",
    "generated_text",
    "text",
]

observed_field = None

for k in candidate_fields:
    if k in rows[0]:
        observed_field = k
        break

empty = None
length_mean = None

if observed_field is not None:
    vals = [
        str(r.get(observed_field, ""))
        for r in rows
    ]

    empty = sum(not x.strip() for x in vals)

    length_mean = (
        sum(len(x) for x in vals)
        / len(vals)
    )

data = {
    "protocol":
        "PATCHER_EPOCH3_CURATED6565_RAW_V2",

    "rows":
        len(rows),

    "model":
        os.environ["PATCHER"],

    "jobs":
        os.environ["JOBS"],

    "raw_file":
        str(raw),

    "raw_sha256":
        hashlib.sha256(
            raw.read_bytes()
        ).hexdigest(),

    "first_row_keys":
        sorted(rows[0].keys()),

    "observed_text_field":
        observed_field,

    "empty_text_rows":
        empty,

    "mean_text_chars":
        length_mean,

    "interpretation_boundary": (
        "Raw final-checkpoint asset only. "
        "No parser-derived selection rate; "
        "no calibration claim; "
        "no semantic correction-quality claim."
    ),
}

manifest.write_text(
    json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print()
print(json.dumps(
    data,
    ensure_ascii=False,
    indent=2,
))

print()
print("MANIFEST =", manifest)
print("PATCHER_EPOCH3_CURATED6565_RAW_V2_PASS")
PY

echo
echo "======================================================================"
echo "OVERNIGHT TASK COMPLETE"
echo "======================================================================"

echo "RAW=$RAW"
echo "MANIFEST=$MANIFEST"

echo "NO_PARSER_RUN"
echo "NO_SCORING_RUN"
echo "NO_FOLLOWUP_TRAINING_LAUNCHED"

echo "PATCHER_EPOCH3_CURATED6565_RAW_V2_ALL_DONE"
