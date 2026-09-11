#!/usr/bin/env bash
source /workspace/mtpatcher/project_env.sh
set +e

ROOT=/workspace/mtpatcher
REPO=$ROOT/repo/MT-Patcher-Reproduction-Ascend

SCRIPT=$REPO/scripts/targeted/repair_targeted_sft_residual49_and_freeze.py
PARENT=$ROOT/data/mtpatcher_v3_full6565_20260823/targeted_sft_positive_control_targets_qwen3_8b_v1_20260910
TEACHER=$ROOT/models/Qwen3-8B
STUDENT=$ROOT/models/Qwen3-0.6B
LOCK=$ROOT/logs/targeted_sft_residual49_repair_v1.lock
LOG=$PARENT/residual49_repair_v1/launcher.log

mkdir -p "$PARENT/residual49_repair_v1" "$ROOT/logs"

echo "============================================================"
echo "TARGETED SFT RESIDUAL49 LOCAL REPAIR + FINAL FREEZE"
echo "============================================================"
echo "Question: Can the 49 deterministic formatting/lexical residuals be repaired without touching the 10,951 valid parent targets?"
echo "Competing explanations: stricter local repair prompts close all residuals vs some rows still violate frozen target invariants."
echo "Falsifiable prediction: 49/49 repaired; 10,951 parent targets remain byte-identical; final 11k passes terminology/language/max-length checks."
echo "Decision after result: PASS freezes SFT target artifact; FAIL inspects only remaining residual rows."
echo
echo "SCIENTIFIC_CLASS=LAB_ADAPTATION_LOCAL_RESIDUAL_REPAIR"
echo "PARENT_VALID_TARGETS=10951"
echo "REPAIR_TARGETS=49"
echo "CHEMISTRY_REPAIR=19"
echo "IDIOM_REPAIR=30"
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
    echo "FINAL_RESULT=RESIDUAL_REPAIR_INCOMPLETE_OR_FAIL_RC_$RC"
  fi
fi

echo "PARENT=$PARENT"
echo "finished_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "finished_+08=$(TZ=Asia/Shanghai date '+%Y-%m-%dT%H:%M:%S%z')"
echo "============================================================"
