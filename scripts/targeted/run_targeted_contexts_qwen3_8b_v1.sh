#!/usr/bin/env bash
# Generate and freeze Chemistry/Idiom train/UC/UW source contexts with Qwen3-8B.
source /workspace/mtpatcher/project_env.sh
set +e

ROOT=/workspace/mtpatcher
REPO=$ROOT/repo/MT-Patcher-Reproduction-Ascend
SCRIPT=$REPO/scripts/targeted/generate_targeted_contexts_qwen3_8b_v1.py

MODEL=$ROOT/models/Qwen3-8B
CHEM=$ROOT/data/mtpatcher_v3_full6565_20260823/wa_section43_chemistry_source_v1/chemistry_section43_6000_20260910_v1
IDIOM=$ROOT/data/mtpatcher_v3_full6565_20260823/wa_section43_idiom_source_v1/idiom_gate1_split_v1_20260910

OUT=$ROOT/data/mtpatcher_v3_full6565_20260823/targeted_section43_contexts_qwen3_8b_v1_20260910
LOG=$ROOT/logs/targeted_section43_contexts_qwen3_8b_v1_20260910

mkdir -p "$OUT/shards" "$LOG"

echo "============================================================"
echo "TARGETED CONTEXT GENERATION V1"
echo "============================================================"
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
  rm -f "$LOG"/status_*.txt

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
  cat "$OUT/context_freeze_manifest.json"
else
  echo "FAIL" >"$LOG/final_status.txt"
  echo "--- failing shard tails ---"
  for i in $(seq 0 15); do
    if [[ -f "$LOG/status_${i}.txt" ]] && [[ "$(cat "$LOG/status_${i}.txt")" != "0" ]]; then
      echo "### SHARD $i"
      tail -n 80 "$LOG/shard_${i}.log"
    fi
  done
  echo "--- merge tail ---"
  tail -n 120 "$LOG/merge.log" 2>/dev/null || true
fi

echo "============================================================"
echo "FINAL_STATUS=$(cat "$LOG/final_status.txt" 2>/dev/null || echo UNKNOWN)"
echo "MASTER_OUT=$OUT"
echo "MASTER_LOG=$LOG"
echo "============================================================"
