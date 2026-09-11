#!/usr/bin/env bash
source /workspace/mtpatcher/project_env.sh
set +e

ROOT=/workspace/mtpatcher
REPO=$ROOT/repo/MT-Patcher-Reproduction-Ascend

SCRIPT=$REPO/scripts/targeted/materialize_targeted_sft_positive_control_targets.py

DATA=$ROOT/data/mtpatcher_v3_full6565_20260823/targeted_section43_contexts_qwen3_8b_final_20260910
TEACHER=$ROOT/models/Qwen3-8B
STUDENT=$ROOT/models/Qwen3-0.6B

OUT=$ROOT/data/mtpatcher_v3_full6565_20260823/targeted_sft_positive_control_targets_qwen3_8b_v1_20260910
LOG=$OUT/launcher.log
LOCK=$ROOT/logs/targeted_sft_positive_control_targets_qwen3_8b_v1.lock

mkdir -p "$OUT" "$ROOT/logs"

echo "============================================================"
echo "TARGETED SFT POSITIVE-CONTROL TARGET SYNTHESIS"
echo "============================================================"
echo "Question: Can C0 learn targeted Seen knowledge when given explicit high-quality offline sequence targets?"
echo "Competing explanations: targeted data is learnable vs even direct sequence supervision cannot transfer it."
echo "Falsifiable prediction: later C1/C2 should improve their matching targeted diagnostics when trained on these frozen targets."
echo "Decision after result: if SFT succeeds, proceed to same-source targeted OPD; if it fails, inspect train target/reference/scoring before blaming OPD."
echo
echo "SCIENTIFIC_CLASS=LAB_ADAPTATION_POSITIVE_CONTROL_TARGET_SYNTHESIS"
echo "TEACHER=$TEACHER"
echo "STUDENT_LEXICAL_HINT=FALSE"
echo "TRAIN_ONLY=TRUE"
echo "UC_ACCESS=FALSE"
echo "UW_ACCESS=FALSE"
echo "WORKERS=16"
echo

python -m py_compile "$SCRIPT"
COMPILE_RC=$?
echo "COMPILE_RC=$COMPILE_RC"

if [[ "$COMPILE_RC" -ne 0 ]]; then
  echo "FINAL_RESULT=COMPILE_FAIL"
else
  flock -n -E 75 "$LOCK" \
    env PYTHONUNBUFFERED=1 python -u "$SCRIPT" \
      --data "$DATA" \
      --teacher "$TEACHER" \
      --student-tokenizer "$STUDENT" \
      --out "$OUT" \
      --num-workers 16 \
      2>&1 | tee -a "$LOG"

  RC=${PIPESTATUS[0]}
  echo "RUN_RC=$RC"

  if [[ "$RC" -eq 0 ]]; then
    echo "FINAL_RESULT=TARGETED_SFT_POSITIVE_CONTROL_TARGETS_FROZEN"
  elif [[ "$RC" -eq 75 ]]; then
    echo "FINAL_RESULT=ALREADY_RUNNING"
  else
    echo "FINAL_RESULT=INCOMPLETE_OR_FAIL_RC_$RC"
  fi
fi

echo "OUT=$OUT"
echo "finished_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "finished_+08=$(TZ=Asia/Shanghai date '+%Y-%m-%dT%H:%M:%S%z')"
echo "============================================================"
