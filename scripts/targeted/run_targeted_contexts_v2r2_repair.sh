#!/usr/bin/env bash
source /workspace/mtpatcher/project_env.sh
set +e

ROOT=/workspace/mtpatcher
REPO=$ROOT/repo/MT-Patcher-Reproduction-Ascend

SRC=$ROOT/data/mtpatcher_v3_full6565_20260823/targeted_section43_contexts_qwen3_8b_v2_20260910
OUT=$ROOT/data/mtpatcher_v3_full6565_20260823/targeted_section43_contexts_qwen3_8b_v2r2_20260910

MODEL=$ROOT/models/Qwen3-8B
REPAIR=$REPO/scripts/targeted/repair_targeted_contexts_qwen3_8b_v2r2.py
MERGER=$REPO/scripts/targeted/merge_targeted_contexts_v2r2_strict.py

echo "============================================================"
echo "TARGETED CONTEXT V2R2 — REPAIR V2 REJECTED ROWS ONLY"
echo "============================================================"
echo "Question: Can V2 construction rejects be repaired without touching frozen lexical identity/split?"
echo "Competing explanations: validator false negatives vs malformed/meta/normalized generations."
echo "Falsifiable prediction: safe direct rows canonicalize; malformed rows alone regenerate; strict merge advances to duplicate/leakage gates."
echo "Decision after result: PASS freezes contexts; later-gate failure repairs only that gate. No 23k rerun."

rm -rf "$OUT"
cp -a "$SRC" "$OUT"

python -m py_compile "$REPAIR" "$MERGER"
COMPILE_RC=$?
echo "COMPILE_RC=$COMPILE_RC"

if [[ "$COMPILE_RC" -eq 0 ]]; then
  python "$REPAIR" \
    --out "$OUT" \
    --model "$MODEL" \
    --device 0 \
    --num-shards 16 \
    --max-new-tokens 96 \
    --max-attempts 8 \
    >"$OUT/v2r2_repair.log" 2>&1
  REPAIR_RC=$?
else
  REPAIR_RC=99
fi

echo "REPAIR_RC=$REPAIR_RC"
cat "$OUT/v2r2_repair.log" 2>/dev/null || true

if [[ "$REPAIR_RC" -eq 0 ]]; then
  mv "$OUT/audit_bad_exact_term.jsonl" \
     "$OUT/repairs/audit_bad_exact_term.pre_v2r2.jsonl" 2>/dev/null || true

  python "$MERGER" \
    --out "$OUT" \
    --model "$MODEL" \
    --num-shards 16 \
    >"$OUT/v2r2_merge.log" 2>&1
  MERGE_RC=$?
else
  MERGE_RC=98
fi

echo "MERGE_RC=$MERGE_RC"
cat "$OUT/v2r2_merge.log" 2>/dev/null || true

echo
echo "=== FINAL AUDIT FILES ==="
find "$OUT" -maxdepth 1 -type f -name 'audit_*.jsonl' \
  -printf '%f %s bytes\n' 2>/dev/null | sort

echo
echo "=== FINAL STATUS ==="
if [[ "$MERGE_RC" -eq 0 ]]; then
  echo "V2R2_RESULT=CONTEXT_FREEZE_PASS"
  cat "$OUT/context_freeze_manifest.json"
else
  echo "V2R2_RESULT=STILL_BLOCKED"
fi

echo "OUT=$OUT"
echo "============================================================"
