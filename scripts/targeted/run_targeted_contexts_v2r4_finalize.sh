#!/usr/bin/env bash
source /workspace/mtpatcher/project_env.sh
set +e

ROOT=/workspace/mtpatcher
REPO=$ROOT/repo/MT-Patcher-Reproduction-Ascend

SRC=$ROOT/data/mtpatcher_v3_full6565_20260823/targeted_section43_contexts_qwen3_8b_v2r3_20260910
OUT=$ROOT/data/mtpatcher_v3_full6565_20260823/targeted_section43_contexts_qwen3_8b_v2r4_20260910
MODEL=$ROOT/models/Qwen3-8B
SCRIPT=$REPO/scripts/targeted/finalize_targeted_contexts_v2r4.py

echo "============================================================"
echo "TARGETED CONTEXT V2R4 FINALIZATION"
echo "============================================================"
echo "Question: Can the frozen lexical split yield independent train/UC contexts with zero exact leakage?"
echo "Competing explanations: deterministic generation caused duplicate contexts; remaining issues may be exact unseen-term contamination."
echo "Falsifiable prediction: repairing protocol-violating rows only removes duplicates/contamination without changing lexical identity or split."
echo "Decision after result: hard-check PASS closes context construction; FAIL reports the remaining protocol class only."

rm -rf "$OUT"
cp -a "$SRC" "$OUT"

python -m py_compile "$SCRIPT"
COMPILE_RC=$?
echo "COMPILE_RC=$COMPILE_RC"

if [[ "$COMPILE_RC" -eq 0 ]]; then
  python "$SCRIPT" \
    --out "$OUT" \
    --model "$MODEL" \
    --device 0 \
    --num-shards 16 \
    >"$OUT/v2r4_finalize.log" 2>&1
  FINALIZE_RC=$?
else
  FINALIZE_RC=99
fi

echo "FINALIZE_RC=$FINALIZE_RC"
cat "$OUT/v2r4_finalize.log" 2>/dev/null || true

echo
echo "=== FINAL STATUS ==="
if [[ "$FINALIZE_RC" -eq 0 ]]; then
  echo "V2R4_RESULT=CONTEXT_FREEZE_PASS"
  echo "--- manifest ---"
  cat "$OUT/context_freeze_manifest.json"
else
  echo "V2R4_RESULT=STILL_BLOCKED"
  echo "--- audit artifacts ---"
  find "$OUT" -maxdepth 2 -type f \
    \( -name 'audit_*.jsonl' -o -name '*near_duplicate*' -o -name 'v2r4_finalize.log' \) \
    -printf '%p %s bytes\n' 2>/dev/null | sort
fi

echo "OUT=$OUT"
echo "============================================================"
