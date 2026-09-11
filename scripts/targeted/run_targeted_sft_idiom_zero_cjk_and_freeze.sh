#!/usr/bin/env bash
source /workspace/mtpatcher/project_env.sh
set +e

ROOT=/workspace/mtpatcher
REPO=$ROOT/repo/MT-Patcher-Reproduction-Ascend

SCRIPT=$REPO/scripts/targeted/repair_targeted_sft_idiom_zero_cjk_and_freeze.py
PARENT=$ROOT/data/mtpatcher_v3_full6565_20260823/targeted_sft_positive_control_targets_qwen3_8b_v1_20260910
TEACHER=$ROOT/models/Qwen3-8B
STUDENT=$ROOT/models/Qwen3-0.6B
LOCK=$ROOT/logs/targeted_sft_idiom_zero_cjk_repair_v1.lock
LOG=$PARENT/idiom_zero_cjk_repair_v1/launcher.log

mkdir -p "$PARENT/idiom_zero_cjk_repair_v1" "$ROOT/logs"

echo "============================================================"
echo "TARGETED SFT IDIOM ZERO-CJK REPAIR + FINAL FREEZE"
echo "============================================================"
echo "Question: Can the 86 previously accepted Idiom targets containing 1-4 residual CJK characters be repaired locally?"
echo "Competing explanations: English-only semantic rewrite closes contamination vs some rows remain invalid."
echo "Falsifiable prediction: exactly 86 selected; 86/86 repaired; final 5500 Idiom targets contain zero CJK."
echo "Decision after result: PASS freezes target artifact; FAIL inspects only unfinished selected rows."
echo
echo "SCIENTIFIC_CLASS=LAB_ADAPTATION_LOCAL_RESIDUAL_REPAIR"
echo "EXPECTED_REPAIR_ROWS=86"
echo "WORKERS=8"
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
      --num-workers 8 \
      2>&1 | tee -a "$LOG"

  RC=${PIPESTATUS[0]}
  echo "RUN_RC=$RC"

  if [[ "$RC" -eq 0 ]]; then
    echo "FINAL_RESULT=TARGETED_SFT_POSITIVE_CONTROL_TARGETS_FROZEN"
  elif [[ "$RC" -eq 75 ]]; then
    echo "FINAL_RESULT=ALREADY_RUNNING"
  else
    echo "FINAL_RESULT=ZERO_CJK_REPAIR_INCOMPLETE_OR_FAIL_RC_$RC"
  fi
fi

echo "PARENT=$PARENT"
echo "finished_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "finished_+08=$(TZ=Asia/Shanghai date '+%Y-%m-%dT%H:%M:%S%z')"
echo "============================================================"
