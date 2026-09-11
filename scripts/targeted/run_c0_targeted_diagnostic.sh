#!/usr/bin/env bash
source /workspace/mtpatcher/project_env.sh
set +e

ROOT=/workspace/mtpatcher
REPO=$ROOT/repo/MT-Patcher-Reproduction-Ascend

SCRIPT=$REPO/scripts/targeted/run_c0_targeted_diagnostic.py
MODEL=$ROOT/models/Qwen3-0.6B
DATA=$ROOT/data/mtpatcher_v3_full6565_20260823/targeted_section43_contexts_qwen3_8b_final_20260910
DIAG=$ROOT/data/mtpatcher_v3_full6565_20260823/targeted_diagnostic1000_v1_20260910
OUT=$ROOT/runs/targeted/c0_targeted_diagnostic1000_20260910
LOCK=$ROOT/logs/c0_targeted_diagnostic1000_20260910.lock

mkdir -p "$OUT" "$ROOT/logs"

echo "============================================================"
echo "C0 TARGETED DIAGNOSTIC1000 / DOMAIN"
echo "started_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "started_+08=$(TZ=Asia/Shanghai date '+%Y-%m-%dT%H:%M:%S%z')"
echo "============================================================"
echo "Question: Does frozen C0 show measurable targeted Chemistry/Idiom headroom?"
echo "Competing explanations: substantial targeted knowledge gap vs already-saturated C0."
echo "Falsifiable prediction: the frozen diagnostic1000 exposes non-ceiling Chemistry/Idiom performance."
echo "Decision after result: reuse byte-identical diagnostic1000 for C1/C2/C3; full 6000 is deferred until targeted effect is established."
echo "SCIENTIFIC_CLASS=DIAGNOSTIC_ONLY_LAB_ADAPTATION"
echo "WORKERS=16"

python -m py_compile "$SCRIPT"
COMPILE_RC=$?
echo "COMPILE_RC=$COMPILE_RC"

if [[ "$COMPILE_RC" -eq 0 ]]; then
  flock -n -E 75 "$LOCK" \
    env PYTHONUNBUFFERED=1 python -u "$SCRIPT" \
      --model "$MODEL" \
      --data "$DATA" \
      --diagnostic-dir "$DIAG" \
      --out "$OUT" \
      --num-workers 16 \
      2>&1 | tee -a "$OUT/launcher.log"

  RC=${PIPESTATUS[0]}
  echo "RUN_RC=$RC"

  if [[ "$RC" -eq 0 ]]; then
    echo "FINAL_RESULT=C0_TARGETED_DIAGNOSTIC_TRANSLATIONS_PASS"
  elif [[ "$RC" -eq 75 ]]; then
    echo "FINAL_RESULT=ALREADY_RUNNING"
  else
    echo "FINAL_RESULT=FAIL_RC_$RC"
  fi
else
  echo "FINAL_RESULT=COMPILE_FAIL"
fi

echo "finished_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "finished_+08=$(TZ=Asia/Shanghai date '+%Y-%m-%dT%H:%M:%S%z')"
echo "OUT=$OUT"
echo "============================================================"
