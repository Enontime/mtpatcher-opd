#!/usr/bin/env bash
set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"
SCRIPT_DIR="$ROOT/scripts/mtpatcher_v11"
EXP_DATA="$DATA_ROOT/$EXP"
EXP_LOG="$LOG_ROOT/$EXP"

GENERATOR="$SCRIPT_DIR/generate_pds_qwen3_8b_v11.py"
JOBS="$EXP_DATA/rq3_pds_jobs_v11.jsonl"
PDS_DIR="$EXP_DATA/rq3_pds_v11"
MASTER_LOG="$EXP_LOG/rq3_pds_generation_v11.log"

echo "======================================================================"
echo "STAGE 1/4 — CURRENT GENERATOR PROCESS AUDIT"
echo "======================================================================"

python - <<'PY'
import os
import re
import signal
import subprocess
import time
from collections import defaultdict

needle = "generate_pds_qwen3_8b_v11.py"

text = subprocess.check_output(
    ["ps", "-eo", "pid=,ppid=,etimes=,args="],
    text=True,
)

groups = defaultdict(list)

for line in text.splitlines():
    if needle not in line:
        continue

    parts = line.strip().split(None, 3)

    if len(parts) != 4:
        continue

    pid = int(parts[0])
    ppid = int(parts[1])
    etimes = int(parts[2])
    cmd = parts[3]

    m_dev = re.search(
        r"--device-id\s+(\d+)",
        cmd,
    )

    m_out = re.search(
        r"--output\s+(\S+)",
        cmd,
    )

    if not m_dev or not m_out:
        continue

    device = int(m_dev.group(1))
    output = m_out.group(1)

    groups[(device, output)].append(
        {
            "pid": pid,
            "ppid": ppid,
            "etimes": etimes,
            "cmd": cmd,
        }
    )

print("ACTIVE_GENERATOR_GROUPS =", len(groups))

duplicates = []

for key in sorted(groups):
    procs = sorted(
        groups[key],
        key=lambda x: (
            -x["etimes"],
            x["pid"],
        ),
    )

    device, output = key

    print(
        f"DEVICE={device:02d} "
        f"OUTPUT={output} "
        f"COUNT={len(procs)}"
    )

    for p in procs:
        print(
            "  "
            f"PID={p['pid']} "
            f"PPID={p['ppid']} "
            f"ELAPSED={p['etimes']}s"
        )

    if len(procs) > 1:
        duplicates.append(
            (key, procs)
        )

if not duplicates:
    print("NO_ACTIVE_DUPLICATE_WORKERS")
else:
    print(
        "ACTIVE_DUPLICATE_GROUPS =",
        len(duplicates)
    )

    for (device, output), procs in duplicates:
        keep = procs[0]
        extras = procs[1:]

        print(
            f"KEEP device={device} "
            f"pid={keep['pid']} "
            f"output={output}"
        )

        for p in extras:
            print(
                f"TERM_DUPLICATE device={device} "
                f"pid={p['pid']}"
            )

            try:
                os.kill(
                    p["pid"],
                    signal.SIGTERM,
                )
            except ProcessLookupError:
                pass

    time.sleep(3)

    for (device, output), procs in duplicates:
        for p in procs[1:]:
            try:
                os.kill(
                    p["pid"],
                    0,
                )
            except ProcessLookupError:
                continue

            print(
                f"KILL_STILL_ALIVE_DUPLICATE "
                f"device={device} "
                f"pid={p['pid']}"
            )

            try:
                os.kill(
                    p["pid"],
                    signal.SIGKILL,
                )
            except ProcessLookupError:
                pass

    print("DUPLICATE_WORKER_REPAIR_PASS")
PY

echo
echo "======================================================================"
echo "STAGE 2/4 — JOB CARDINALITY"
echo "======================================================================"

python - <<'PY'
import json
import os
from pathlib import Path

exp = "mtpatcher_v3_full6565_20260823"

jobs = (
    Path(os.environ["DATA_ROOT"])
    / exp
    / "rq3_pds_jobs_v11.jsonl"
)

rows = []

with jobs.open(
    encoding="utf-8"
) as f:
    for line in f:
        if line.strip():
            rows.append(
                json.loads(line)
            )

ids = [
    int(x["job_id"])
    for x in rows
]

print("JOB_ROWS =", len(rows))
print("UNIQUE_JOB_IDS =", len(set(ids)))

if len(rows) != 26672:
    raise RuntimeError(
        f"Expected 26672 jobs, got {len(rows)}"
    )

if len(set(ids)) != len(rows):
    raise RuntimeError(
        "Duplicate job IDs in job file"
    )

print("JOB_CARDINALITY_PASS")
PY

echo
echo "======================================================================"
echo "STAGE 3/4 — CURRENT SHARD INTEGRITY"
echo "======================================================================"

python - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

exp = "mtpatcher_v3_full6565_20260823"

root = (
    Path(os.environ["DATA_ROOT"])
    / exp
    / "rq3_pds_v11"
)

grand_lines = 0
grand_unique = set()
grand_duplicates = 0
grand_bad_json = 0
wrong_device = 0

for device in range(16):
    path = root / f"device_{device}.jsonl"

    if not path.exists():
        print(
            f"device={device:02d} "
            "lines=0 unique=0 dup=0 bad_json=0"
        )
        continue

    ids = []
    bad_json = 0
    bad_assignment = 0

    with path.open(
        encoding="utf-8",
        errors="replace",
    ) as f:
        for lineno, line in enumerate(
            f,
            1,
        ):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
                jid = int(row["job_id"])
            except Exception:
                bad_json += 1
                continue

            ids.append(jid)

            if jid % 16 != device:
                bad_assignment += 1

    counts = Counter(ids)

    duplicates = sum(
        n - 1
        for n in counts.values()
        if n > 1
    )

    unique_ids = set(ids)

    print(
        f"device={device:02d} "
        f"lines={len(ids)} "
        f"unique={len(unique_ids)} "
        f"dup={duplicates} "
        f"bad_json={bad_json} "
        f"wrong_device={bad_assignment}"
    )

    grand_lines += len(ids)
    grand_unique.update(unique_ids)
    grand_duplicates += duplicates
    grand_bad_json += bad_json
    wrong_device += bad_assignment

print("TOTAL_PARSED_LINES =", grand_lines)
print("TOTAL_UNIQUE_JOB_IDS =", len(grand_unique))
print("TOTAL_DUPLICATE_LINES =", grand_duplicates)
print("TOTAL_BAD_JSON =", grand_bad_json)
print("TOTAL_WRONG_DEVICE =", wrong_device)

if grand_bad_json:
    raise RuntimeError(
        f"Malformed JSON already present: "
        f"{grand_bad_json}"
    )

if wrong_device:
    raise RuntimeError(
        f"Wrong-device job IDs: {wrong_device}"
    )

print("CURRENT_SHARD_JSON_INTEGRITY_PASS")

if grand_duplicates:
    print(
        "SHARD_DUPLICATE_JOB_WARNING "
        f"count={grand_duplicates}"
    )
else:
    print("NO_SHARD_DUPLICATE_JOB_IDS")
PY

echo
echo "======================================================================"
echo "STAGE 4/4 — PIPELINE HEALTH"
echo "======================================================================"

echo "----- ACTIVE PROCESSES -----"
pgrep -af \
'rq3_pds_generation_v11|generate_pds_qwen3_8b_v11' \
|| true

echo
echo "----- WORKER KEY STATUS -----"

grep -hE \
'ASSIGNED=|PDS_MODEL_READY|GENERATED=|PDS_DEVICE_.*COMPLETE|Traceback|RuntimeError|ERROR' \
"$EXP_LOG"/rq3_pds_v11_device_*.log \
2>/dev/null | tail -n 220 || true

echo
echo "----- MASTER KEY STATUS -----"

grep -E \
'PDS_JOB_BUILD_PASS|PDS_ALL_16_WORKERS_COMPLETE|PDS_MERGE_AUDIT_PASS|PE_PLUS_PDS_BUILD_PASS|PDS_GENERATION_ALL_PASS|RQ3_PDS_BASELINE_DATA_READY|Traceback|RuntimeError|FAILURE' \
"$MASTER_LOG" \
2>/dev/null | tail -n 120 || true

echo
echo "RQ3_PDS_HEALTH_REPAIR_V3_DONE"
