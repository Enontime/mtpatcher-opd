#!/usr/bin/env bash
set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

DIR="$ROOT/scripts/mtpatcher_paper_faithful_v2"
PIPE="$DIR/paper_repro_v2.py"
OFFICIAL="$ROOT/vendor/MT-Patcher-official"

PATCHER="$RUN_ROOT/$EXP/patcher_qwen3_8b_paperfaith_fullft_v2/checkpoint-4752"
STUDENT_PRED="$RUN_ROOT/$EXP/student_base_full6565/predictions.jsonl"

WORK="$DATA_ROOT/$EXP/patcher_epoch3_curated6565_raw_v1"
LOGDIR="${LOG_ROOT:-/workspace/mtpatcher/logs}/$EXP/patcher_epoch3_curated6565_raw_v1"

COMPAT="$WORK/student_compat6565.jsonl"
JOBS="$WORK/feedback_jobs6565.jsonl"
SHARDS="$WORK/shards"
RAW="$WORK/epoch3_feedback_raw6565.jsonl"
MANIFEST="$WORK/manifest.json"

mkdir -p "$WORK" "$LOGDIR" "$SHARDS"

# paper_repro_v2 imports the frozen official pipeline.
export PYTHONPATH="$OFFICIAL:${PYTHONPATH:-}"

echo "======================================================================"
echo "FINAL EPOCH3 PATCHER ON CURATED6565 — RAW ASSET V1"
echo "======================================================================"
echo "PURPOSE=asset_generation_for_later_paper_grounded_feedback_audit"
echo "MODEL=$PATCHER"
echo "STUDENT_PRED=$STUDENT_PRED"
echo "NO_PARSER=1"
echo "NO_SCORING=1"
echo "NO_STUDENT_TRAINING=1"
echo "NO_PDS_WA_OPD=1"

test -f "$PIPE"
test -f "$STUDENT_PRED"
test -d "$OFFICIAL"
test -d "$PATCHER"

echo "PIPE_SHA256=$(sha256sum "$PIPE" | awk '{print $1}')"
echo "STUDENT_PRED_SHA256=$(sha256sum "$STUDENT_PRED" | awk '{print $1}')"

# ----------------------------------------------------------------------
# Materialize only source + Student draft.
# Human reference is deliberately hidden from the Patcher.
# ----------------------------------------------------------------------

export STUDENT_PRED COMPAT

python - <<'PY'
import json
import os
from pathlib import Path

src = Path(os.environ["STUDENT_PRED"])
out = Path(os.environ["COMPAT"])

rows = [
    json.loads(x)
    for x in src.read_text(encoding="utf-8").splitlines()
    if x.strip()
]

rows.sort(key=lambda x: int(x["index"]))

assert len(rows) == 6565, len(rows)
assert [int(x["index"]) for x in rows] == list(range(6565))

compat = []

for x in rows:
    source = str(x["source"]).strip()
    response = str(x["student_translation"]).strip()

    assert source
    assert response

    compat.append({
        "demo_id": int(x["index"]),
        "source": source,
        "response": response,
    })

with out.open("w", encoding="utf-8") as f:
    for x in compat:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

assert all("reference" not in x for x in compat)

print("COMPAT_ROWS =", len(compat))
print("COMPAT_KEYS =", sorted(compat[0]))
print("REFERENCE_LEAKAGE =", False)
print("COMPAT_PASS")
PY

# ----------------------------------------------------------------------
# Build paper-faithful Feedback jobs.
# Formal reproduction settings are carried by these jobs:
# feedback temperature 0.1, max_new_tokens 256.
# ----------------------------------------------------------------------

python "$PIPE" feedback-jobs \
    --official "$OFFICIAL" \
    --student "$COMPAT" \
    --output "$JOBS"

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

assert len(rows) == 6565, len(rows)

ids = [int(x["job_id"]) for x in rows]
assert len(ids) == len(set(ids))

temps = sorted(set(float(x["temperature"]) for x in rows))
max_tokens = sorted(set(int(x["max_new_tokens"]) for x in rows))

print("FEEDBACK_JOBS =", len(rows))
print("TEMPERATURE_VALUES =", temps)
print("MAX_NEW_TOKENS_VALUES =", max_tokens)

assert temps == [0.1], temps
assert max_tokens == [256], max_tokens

print("FEEDBACK_JOB_AUDIT_PASS")
PY

# ----------------------------------------------------------------------
# 16-device generation.
# One treatment only: final specialized checkpoint.
# No cross-arm RNG claim is made.
# ----------------------------------------------------------------------

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
    echo "GENERATION_FAILED"

    for F in "$LOGDIR"/device_*.log; do
        echo "---------------- $F ----------------"
        tail -n 30 "$F" || true
    done

    false
fi

echo "ALL_16_GENERATION_WORKERS_PASS"

# ----------------------------------------------------------------------
# Exact merge / coverage audit.
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

shards = sorted(shard_dir.glob("device_*.jsonl"))
assert len(shards) == 16, len(shards)

merged = {}

for p in shards:
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        r = json.loads(line)
        jid = int(r["job_id"])

        if jid in merged:
            raise RuntimeError(f"duplicate job_id={jid}")

        merged[jid] = r

missing = expected - set(merged)
extra = set(merged) - expected

assert not missing, sorted(missing)[:20]
assert not extra, sorted(extra)[:20]

ordered = [merged[i] for i in sorted(expected)]

with raw_path.open("w", encoding="utf-8") as f:
    for r in ordered:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

blob = raw_path.read_bytes()

print("RAW_ROWS =", len(ordered))
print("RAW_SHA256 =", hashlib.sha256(blob).hexdigest())
print("RAW_BYTES =", len(blob))
print("MERGE_COVERAGE_PASS")
PY

# ----------------------------------------------------------------------
# Structural audit only.
# Deliberately no has_error parsing and no semantic scoring tonight.
# ----------------------------------------------------------------------

export RAW MANIFEST PATCHER PIPE OFFICIAL

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

keys = sorted(rows[0].keys())

empty_like = 0
lengths = []

for r in rows:
    # Generator output field is inspected structurally only.
    value = None

    for k in ("response", "output", "generated_text", "text"):
        if k in r:
            value = r[k]
            break

    if value is not None:
        s = str(value)
        lengths.append(len(s))

        if not s.strip():
            empty_like += 1

data = {
    "protocol": "PATCHER_EPOCH3_CURATED6565_RAW_V1",
    "rows": len(rows),
    "model": os.environ["PATCHER"],
    "pipeline": os.environ["PIPE"],
    "official_repo": os.environ["OFFICIAL"],
    "raw_file": str(raw),
    "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
    "raw_first_keys": keys,
    "structural_output_rows_observed": len(lengths),
    "empty_structural_outputs": empty_like,
    "interpretation_boundary": (
        "Raw asset only. No parser-derived selection rate, no correction "
        "quality conclusion, no calibration conclusion."
    ),
}

manifest.write_text(
    json.dumps(data, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(json.dumps(data, ensure_ascii=False, indent=2))
print("MANIFEST =", manifest)
print("PATCHER_EPOCH3_CURATED6565_RAW_V1_PASS")
PY

echo
echo "======================================================================"
echo "OVERNIGHT TASK FINISHED"
echo "======================================================================"
echo "RAW=$RAW"
echo "MANIFEST=$MANIFEST"
echo "NO_FOLLOWUP_EXPERIMENT_LAUNCHED"
echo "PATCHER_EPOCH3_CURATED6565_RAW_V1_ALL_DONE"
