#!/usr/bin/env bash
source /workspace/mtpatcher/project_env.sh
set +e

ROOT=/workspace/mtpatcher
REPO=$ROOT/repo/MT-Patcher-Reproduction-Ascend

PARENT=$ROOT/data/mtpatcher_v3_full6565_20260823/targeted_section43_contexts_qwen3_8b_v2r5_20260910
OUT=$ROOT/data/mtpatcher_v3_full6565_20260823/targeted_section43_contexts_qwen3_8b_final_20260910
MODEL=$ROOT/models/Qwen3-8B
SCRIPT=$REPO/scripts/targeted/finalize_targeted_context_near_duplicates.py
LOCK=$ROOT/logs/targeted_context_near_dup_final.lock

echo "============================================================"
echo "TARGETED CONTEXT — FINAL NEAR-DUPLICATE CLOSURE"
echo "started_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "started_+08=$(TZ=Asia/Shanghai date '+%Y-%m-%dT%H:%M:%S%z')"
echo "============================================================"
echo "Question: Do same-entity train/UC contexts remain near-copies after term masking?"
echo "Competing explanations: independent context diversity vs deterministic paraphrastic copies."
echo "Falsifiable prediction: repairing UC rows with context-only similarity >=0.95 closes the near-duplicate gate without changing lexical identity/split."
echo "Decision after result: PASS freezes Chemistry+Idiom contexts and proceeds to C0; FAIL repairs only reported residuals."
echo
echo "SCIENTIFIC_CLASS=LAB_ADAPTATION"
echo "NEAR_DUP_THRESHOLD=0.95"
echo "METRIC=term-masked context-only SequenceMatcher after NFKC/punctuation-whitespace normalization"

python -m py_compile "$SCRIPT"
COMPILE_RC=$?
echo "COMPILE_RC=$COMPILE_RC"

if [[ "$COMPILE_RC" -eq 0 ]]; then
  if [[ ! -d "$OUT" ]]; then
    cp -a "$PARENT" "$OUT"
    echo "CREATED_FROM=$PARENT"
  else
    echo "RESUME_EXISTING_OUT=$OUT"
  fi

  flock -n -E 75 "$LOCK" \
    env PYTHONUNBUFFERED=1 python -u "$SCRIPT" \
      --parent "$PARENT" \
      --out "$OUT" \
      --model "$MODEL" \
      --device 0 \
      --num-shards 16 \
      2>&1 | tee -a "$OUT/final_near_dup_closure.log"

  RC=${PIPESTATUS[0]}
  echo "RUN_RC=$RC"

  if [[ "$RC" -eq 0 ]]; then
    echo "FINAL_RESULT=TARGETED_CONTEXT_DATA_FROZEN_CLOSED"
  elif [[ "$RC" -eq 75 ]]; then
    echo "FINAL_RESULT=ALREADY_RUNNING"
  else
    echo "FINAL_RESULT=STILL_BLOCKED_RC_$RC"
  fi
else
  echo "FINAL_RESULT=COMPILE_FAIL"
fi

echo "finished_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "finished_+08=$(TZ=Asia/Shanghai date '+%Y-%m-%dT%H:%M:%S%z')"
echo "OUT=$OUT"
echo "============================================================"
