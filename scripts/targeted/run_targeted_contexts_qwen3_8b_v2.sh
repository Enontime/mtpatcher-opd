#!/usr/bin/env bash
# Uniform V2 rebuild for targeted Chemistry/Idiom train/UC/UW contexts.
# V1 is preserved as INVALID_RUN; V2 uses a [[TERM]] slot and deterministic exact-term insertion.
source /workspace/mtpatcher/project_env.sh
set +e

ROOT=/workspace/mtpatcher
REPO=$ROOT/repo/MT-Patcher-Reproduction-Ascend
SCRIPT=$REPO/scripts/targeted/generate_targeted_contexts_qwen3_8b_v2.py

MODEL=$ROOT/models/Qwen3-8B
CHEM=$ROOT/data/mtpatcher_v3_full6565_20260823/wa_section43_chemistry_source_v1/chemistry_section43_6000_20260910_v1
IDIOM=$ROOT/data/mtpatcher_v3_full6565_20260823/wa_section43_idiom_source_v1/idiom_gate1_split_v1_20260910

OUT=$ROOT/data/mtpatcher_v3_full6565_20260823/targeted_section43_contexts_qwen3_8b_v2_20260910
LOG=$ROOT/logs/targeted_section43_contexts_qwen3_8b_v2_20260910

mkdir -p "$OUT/shards" "$LOG"

echo "============================================================"
echo "TARGETED CONTEXT GENERATION V2"
echo "============================================================"
echo "QUESTION=Can frozen Chemistry/Idiom lexical items be placed into independent natural train/UC/UW contexts without leakage?"
echo "COMPETING_EXPLANATIONS=V1 exact-term failures were copy/normalization failures rather than lexical-split failures."
echo "FALSIFIABLE_PREDICTION=V2 produces 23000 rows with zero exact-term failures; merge then tests train/UC duplicates and unseen-term contamination."
echo "DECISION_AFTER_RESULT=PASS freezes contexts; FAIL preserves artifacts and repairs only the newly exposed construction defect."
echo "SCRIPT=$SCRIPT"
echo "MODEL=$MODEL"
echo "CHEM=$CHEM"
echo "IDIOM=$IDIOM"
echo "OUT=$OUT"
echo "LOG=$LOG"

python -m py_compile "$SCRIPT"
COMPILE_RC=$?
echo "COMPILE_RC=$COMPILE_RC"

if [[ "$COMPILE_RC" -eq 0 ]]; then
  python "$SCRIPT" build \
    --chem "$CHEM" \
    --idiom "$IDIOM" \
    --out "$OUT" \
    --model "$MODEL"
  BUILD_RC=$?
else
  BUILD_RC=99
fi
echo "BUILD_RC=$BUILD_RC"

if [[ "$BUILD_RC" -eq 0 ]]; then
  rm -f "$LOG"/status_*.txt "$LOG"/final_status.txt "$LOG"/merge.log

  for i in $(seq 0 15); do
    (
      sleep $((i * 6))
      python "$SCRIPT" generate \
        --out "$OUT" \
        --model "$MODEL" \
        --shard-id "$i" \
        --num-shards 16 \
        --device "$i" \
        --batch-size 16 \
        --max-new-tokens 96 \
        >"$LOG/shard_${i}.log" 2>&1
      printf '%s\n' "$?" >"$LOG/status_${i}.txt"
    ) &
  done

  wait

  FAIL=0
  for i in $(seq 0 15); do
    if [[ ! -f "$LOG/status_${i}.txt" ]]; then
      echo "SHARD_STATUS_MISSING=$i"
      FAIL=1
      continue
    fi
    RC=$(cat "$LOG/status_${i}.txt")
    echo "SHARD_$i RC=$RC"
    if [[ "$RC" != "0" ]]; then
      FAIL=1
    fi
  done

  if [[ "$FAIL" -eq 0 ]]; then
    python "$SCRIPT" merge \
      --out "$OUT" \
      --model "$MODEL" \
      --num-shards 16 \
      >"$LOG/merge.log" 2>&1
    MERGE_RC=$?
  else
    MERGE_RC=98
  fi
else
  MERGE_RC=97
fi

echo "MERGE_RC=$MERGE_RC"

if [[ "$MERGE_RC" -eq 0 ]]; then
  echo "PASS" >"$LOG/final_status.txt"
  echo "Problem -> Result -> Interpretation -> Next step"
  echo "V1 exact-term copy failures -> V2 context freeze PASS -> placeholder-slot construction closes copy-normalization defect -> proceed to C0 targeted baseline/evaluator sanity."
  cat "$OUT/context_freeze_manifest.json"
else
  echo "FAIL" >"$LOG/final_status.txt"
  echo "Problem -> Result -> Interpretation -> Next step"
  echo "V2 merge did not freeze -> inspect the single emitted audit class -> repair that construction defect only; do not change lexical split."
  echo "--- merge tail ---"
  tail -n 160 "$LOG/merge.log" 2>/dev/null || true
  echo "--- audit files ---"
  find "$OUT" -maxdepth 1 -type f -name 'audit_*.jsonl' -printf '%f %s bytes\n' 2>/dev/null | sort
fi

echo "============================================================"
echo "FINAL_STATUS=$(cat "$LOG/final_status.txt" 2>/dev/null || echo UNKNOWN)"
echo "MASTER_OUT=$OUT"
echo "MASTER_LOG=$LOG"
echo "============================================================"
