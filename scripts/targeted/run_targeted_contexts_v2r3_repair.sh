#!/usr/bin/env bash
source /workspace/mtpatcher/project_env.sh
set +e

ROOT=/workspace/mtpatcher
REPO=$ROOT/repo/MT-Patcher-Reproduction-Ascend
SRC=$ROOT/data/mtpatcher_v3_full6565_20260823/targeted_section43_contexts_qwen3_8b_v2_20260910
OUT=$ROOT/data/mtpatcher_v3_full6565_20260823/targeted_section43_contexts_qwen3_8b_v2r3_20260910
MODEL=$ROOT/models/Qwen3-8B

REPAIR=$REPO/scripts/targeted/repair_targeted_contexts_qwen3_8b_v2r3.py
MERGER=$REPO/scripts/targeted/merge_targeted_contexts_v2r2_strict.py

echo "============================================================"
echo "V2R3 TARGETED CONTEXT RESIDUAL REPAIR"
echo "============================================================"

rm -rf "$OUT"
cp -a "$SRC" "$OUT"

python -m py_compile "$REPAIR" "$MERGER"
echo "COMPILE_RC=$?"

python "$REPAIR" --out "$OUT" --model "$MODEL" --device 0 --num-shards 16 \
  >"$OUT/v2r3_repair.log" 2>&1
REPAIR_RC=$?

echo "REPAIR_RC=$REPAIR_RC"
cat "$OUT/v2r3_repair.log"

if [[ "$REPAIR_RC" -eq 0 ]]; then
  mkdir -p "$OUT/repairs"
  mv "$OUT/audit_bad_exact_term.jsonl" \
     "$OUT/repairs/audit_bad_exact_term.pre_v2r3.jsonl" 2>/dev/null || true

  python "$MERGER" --out "$OUT" --model "$MODEL" --num-shards 16 \
    >"$OUT/v2r3_merge.log" 2>&1
  MERGE_RC=$?
else
  MERGE_RC=98
fi

echo "MERGE_RC=$MERGE_RC"
cat "$OUT/v2r3_merge.log" 2>/dev/null || true

echo "=== FINAL ==="
if [[ "$MERGE_RC" -eq 0 ]]; then
  echo "V2R3_RESULT=CONTEXT_FREEZE_PASS"
else
  echo "V2R3_RESULT=STILL_BLOCKED"
  find "$OUT" -maxdepth 2 -type f \
    \( -name 'audit_*.jsonl' -o -name 'v2r3_unrepaired.jsonl' \) \
    -printf '%p %s bytes\n' 2>/dev/null | sort
fi
echo "OUT=$OUT"
