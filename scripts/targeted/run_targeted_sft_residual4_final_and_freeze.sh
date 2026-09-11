#!/usr/bin/env bash
source /workspace/mtpatcher/project_env.sh
set +e

ROOT=/workspace/mtpatcher
REPO=$ROOT/repo/MT-Patcher-Reproduction-Ascend

SCRIPT=$REPO/scripts/targeted/repair_targeted_sft_residual4_final_and_freeze.py
PARENT=$ROOT/data/mtpatcher_v3_full6565_20260823/targeted_sft_positive_control_targets_qwen3_8b_v1_20260910
TEACHER=$ROOT/models/Qwen3-8B
STUDENT=$ROOT/models/Qwen3-0.6B
LOCK=$ROOT/logs/targeted_sft_residual4_final_repair_v1.lock
LOG=$PARENT/residual4_final_repair_v1/launcher.log

mkdir -p "$PARENT/residual4_final_repair_v1" "$ROOT/logs"

echo "============================================================"
echo "TARGETED SFT FINAL RESIDUAL4 REPAIR + FREEZE"
echo "============================================================"
echo "Question: Can the last 4 deterministic residuals be closed without touching the 10,951 original successes or 45 successful prior repairs?"
echo "Competing explanations: Teacher-only source substitution closes the exact-format residuals vs some row remains invalid."
echo "Falsifiable prediction: 4/4 PASS; final 11k passes all hard checks."
echo "Decision after result: PASS freezes target artifact; FAIL inspects only any still-missing final row."
echo
echo "SCIENTIFIC_CLASS=LAB_ADAPTATION_LOCAL_RESIDUAL_REPAIR"
echo "ORIGINAL_PRESERVED=10951"
echo "RESIDUAL49_SUCCESS_PRESERVED=45"
echo "FINAL_REPAIR_TARGETS=4"
echo "WORKERS=4"
echo

python -m py_compile "$SCRIPT"
COMPILE_RC=$?
echo "COMPILE_RC=$COMPILE_RC"

if [[ "$COMPILE_RC" -ne 0 ]]; then
  echo "FINAL_RESULT=COMPILE_FAIL"
else
  flock -n -E 75 "$LOCK" \
    env PYTHONUNBUFFERED=1 python -u "$SCRIPT" \
      --parent "$PARENT" \
      --teacher "$TEACHER" \
      --student-tokenizer "$STUDENT" \
      --num-workers 4 \
      2>&1 | tee -a "$LOG"

  RC=${PIPESTATUS[0]}
  echo "RUN_RC=$RC"

  if [[ "$RC" -eq 0 ]]; then
    echo "FINAL_RESULT=TARGETED_SFT_POSITIVE_CONTROL_TARGETS_FROZEN"
  elif [[ "$RC" -eq 75 ]]; then
    echo "FINAL_RESULT=ALREADY_RUNNING"
  else
    echo "FINAL_RESULT=FINAL4_REPAIR_INCOMPLETE_OR_FAIL_RC_$RC"
  fi
fi

echo "PARENT=$PARENT"
echo "finished_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "finished_+08=$(TZ=Asia/Shanghai date '+%Y-%m-%dT%H:%M:%S%z')"
echo "============================================================"
