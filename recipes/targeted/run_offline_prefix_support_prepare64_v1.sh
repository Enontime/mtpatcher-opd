#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/mtpatcher
REPO="$ROOT/repo/MT-Patcher-Reproduction-Ascend"

RUN=${RUN:-"$ROOT/runs/targeted/offline_prefix_support_replay_prepare64_v1_20260913"}
PACK=${PACK:-"/tmp/offline_prefix_support_verl_patch_context_20260913.txt"}

PREP="$REPO/scripts/targeted/offline_prefix_support_prepare_v1.py"
COLLECT="$REPO/scripts/targeted/collect_offline_prefix_support_verl_context_v1.py"

mkdir -p "$RUN"

if [[ -f "$ROOT/project_env.sh" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$ROOT/project_env.sh"
  set -u
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}

python - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path

p = Path("$RUN/state.json")
tmp = Path(str(p) + ".tmp")
obj = {
    "status": "RUNNING",
    "phase": "prepare64_no_parameter_update",
    "parameter_updates": False,
    "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
}
tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
tmp.replace(p)
PY

{
  echo "============================================================"
  echo "OFFLINE PREFIX-SUPPORT PREPARE64"
  echo "============================================================"
  date -u '+UTC=%Y-%m-%dT%H:%M:%SZ'
  echo "run=$RUN"
  echo "repo_head=$(git -C "$REPO" rev-parse HEAD)"
  echo "PARAMETER_UPDATES=FALSE"
  echo
  echo "=== SCRIPT SHA256 ==="
  sha256sum "$PREP" "$COLLECT" "$0"
  echo
  echo "=== NPU SNAPSHOT ==="
  npu-smi info || true
} | tee "$RUN/preflight.log"

python -u "$PREP" \
  --run "$RUN" \
  --n 64 \
  --student-device npu:0 \
  --teacher-device npu:1 \
  > "$RUN/prepare64.log" 2>&1 &
PREP_PID=$!

echo "$PREP_PID" > "$RUN/prepare64.pid"

while kill -0 "$PREP_PID" 2>/dev/null; do
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "$RUN/heartbeat.txt.tmp"
  mv "$RUN/heartbeat.txt.tmp" "$RUN/heartbeat.txt"
  sleep 30
done

set +e
wait "$PREP_PID"
PREP_RC=$?
set -e

echo "$PREP_RC" > "$RUN/prepare64_status.txt"

set +e
python -u "$COLLECT" \
  --run "$RUN" \
  --out "$PACK" \
  > "$RUN/collect_context.log" 2>&1
COLLECT_RC=$?
set -e

echo "$COLLECT_RC" > "$RUN/collect_context_status.txt"

python - "$RUN" "$PACK" "$PREP_RC" "$COLLECT_RC" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

run = Path(sys.argv[1])
pack = Path(sys.argv[2])
prep_rc = int(sys.argv[3])
collect_rc = int(sys.argv[4])

status = (
    "PASS_NO_PARAMETER_UPDATE"
    if prep_rc == 0 and collect_rc == 0
    else "FAIL"
)

obj = {
    "status": status,
    "phase": "prepare64_no_parameter_update",
    "parameter_updates": False,
    "prepare64_returncode": prep_rc,
    "collect_context_returncode": collect_rc,
    "context_pack": str(pack),
    "context_pack_exists": pack.is_file(),
    "context_pack_bytes": pack.stat().st_size if pack.is_file() else None,
    "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
}

p = run / "state.json"
tmp = Path(str(p) + ".tmp")
tmp.write_text(
    json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
tmp.replace(p)

(run / ("PASS" if status.startswith("PASS") else "FAIL")).touch()
print(json.dumps(obj, ensure_ascii=False, indent=2))
PY

echo
echo "============================================================"
echo "FINAL"
echo "============================================================"
cat "$RUN/state.json"

if [[ -f "$PACK" ]]; then
  wc -l -c "$PACK"
  sha256sum "$PACK"
fi

echo
echo "UPLOAD_AFTER_COMPLETION:"
echo "$PACK"

test "$PREP_RC" -eq 0
test "$COLLECT_RC" -eq 0
