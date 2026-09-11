#!/usr/bin/env bash
source /workspace/mtpatcher/project_env.sh
set +e

ROOT=/workspace/mtpatcher
REPO=$ROOT/repo/MT-Patcher-Reproduction-Ascend

SRC=$ROOT/data/mtpatcher_v3_full6565_20260823/targeted_section43_contexts_qwen3_8b_v2r4_20260910
PARENT=$ROOT/data/mtpatcher_v3_full6565_20260823/targeted_section43_contexts_qwen3_8b_v2r3_20260910
OUT=$ROOT/data/mtpatcher_v3_full6565_20260823/targeted_section43_contexts_qwen3_8b_v2r5_20260910
MODEL=$ROOT/models/Qwen3-8B
SCRIPT=$REPO/scripts/targeted/finalize_targeted_contexts_v2r5.py
LOCK=$ROOT/logs/targeted_contexts_v2r5.lock

echo "============================================================"
echo "V2R5 FINAL 9-ROW CONSTRUCTION REPAIR"
echo "started_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "started_+08=$(TZ=Asia/Shanghai date '+%Y-%m-%dT%H:%M:%S%z')"
echo "============================================================"

python -m py_compile "$SCRIPT"
COMPILE_RC=$?
echo "COMPILE_RC=$COMPILE_RC"

if [[ "$COMPILE_RC" -ne 0 ]]; then
  echo "V2R5_RESULT=FAIL_COMPILE"
else
  if [[ ! -d "$OUT" ]]; then
    cp -a "$SRC" "$OUT"
    echo "CREATED_FROM=$SRC"
  else
    echo "RESUME_EXISTING_OUT=$OUT"
  fi

  flock -n "$LOCK" \
    env PYTHONUNBUFFERED=1 python -u "$SCRIPT" \
      --out "$OUT" \
      --parent-v2r3 "$PARENT" \
      --model "$MODEL" \
      --device 0 \
      --num-shards 16 \
      2>&1 | tee -a "$OUT/v2r5_finalize.log"

  RC=${PIPESTATUS[0]}
  echo "V2R5_RC=$RC"

  if [[ "$RC" -eq 0 ]]; then
    echo "V2R5_RESULT=CONTEXT_FREEZE_PASS"
  elif [[ "$RC" -eq 1 ]]; then
    echo "V2R5_RESULT=STILL_BLOCKED"
  else
    echo "V2R5_RESULT=FAILED_RC_$RC"
  fi
fi

echo "finished_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "finished_+08=$(TZ=Asia/Shanghai date '+%Y-%m-%dT%H:%M:%S%z')"
echo "OUT=$OUT"
echo "============================================================"
