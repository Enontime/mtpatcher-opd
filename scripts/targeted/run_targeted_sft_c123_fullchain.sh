#!/usr/bin/env bash
source /workspace/mtpatcher/project_env.sh
set +e

ROOT=/workspace/mtpatcher
REPO=$ROOT/repo/MT-Patcher-Reproduction-Ascend
PY=$REPO/scripts/targeted/targeted_sft_c123_pipeline.py

C0=$ROOT/models/Qwen3-0.6B
TARGETS=$ROOT/data/mtpatcher_v3_full6565_20260823/targeted_sft_positive_control_targets_qwen3_8b_v1_20260910
DIAG=$ROOT/data/mtpatcher_v3_full6565_20260823/targeted_diagnostic1000_v1_20260910
C0RUN=$ROOT/runs/targeted/c0_targeted_diagnostic1000_20260910

RUN=$ROOT/runs/targeted/wa_sft_positive_control_c123_20260910
MASTER_LOG=$RUN/master.log
STATE=$RUN/state.json
LOCK=$ROOT/logs/wa_sft_positive_control_c123_20260910.lock

mkdir -p "$RUN" "$ROOT/logs"

write_state () {
  python - "$STATE" "$1" "$2" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
p=Path(sys.argv[1])
obj={
    "status":sys.argv[2],
    "phase":sys.argv[3],
    "updated_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
}
tmp=Path(str(p)+".tmp")
tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n",encoding="utf-8")
os.replace(tmp,p)
PY
}

show_progress () {
  python - "$RUN" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
parts=[]
for arm in ("C1","C2","C3"):
    phase="?"
    text=""
    tp=root/arm/"train"/"progress.json"
    ep=root/arm/"eval"/"progress.json"
    if ep.exists():
        p=ep
        phase="eval"
    elif tp.exists():
        p=tp
        phase="train"
    else:
        parts.append(f"{arm}=not_started")
        continue
    try:
        d=json.loads(p.read_text())
        pct=d.get("percentage","?")
        status=d.get("status","?")
        if phase=="train":
            text=f"{status}:{pct}% u={d.get('completed_updates','?')}/{d.get('total_updates','?')}"
        else:
            text=f"{status}:{pct}% n={d.get('done','?')}/{d.get('total','?')}"
        parts.append(f"{arm}[{phase}]={text}")
    except Exception as e:
        parts.append(f"{arm}[{phase}]=state_read_error")
print(" | ".join(parts), flush=True)
PY
}

run_locked () {
  echo "============================================================"
  echo "TARGETED WA-SFT POSITIVE CONTROL C1/C2/C3 FULL SERVER CHAIN"
  echo "============================================================"
  echo "Question: Does explicit targeted WA sequence supervision transfer Chemistry/Idiom knowledge into C0?"
  echo "Competing explanations: frozen targeted supervision is learnable vs it does not produce held-out targeted gain."
  echo "Falsifiable prediction: C1 improves Idiom, C2 improves Chemistry, C3 improves both relative to C0."
  echo "Decision after result: clear positive control -> run WA-OPD on the same frozen source contexts; weak/negative control -> inspect supervision/training before attributing failure to OPD."
  echo
  echo "SCIENTIFIC_CLASS=LAB_ADAPTATION_TARGETED_WA_SFT_POSITIVE_CONTROL"
  echo "C0=$C0"
  echo "TARGETS=$TARGETS"
  echo "DIAG=$DIAG"
  echo "RUN=$RUN"
  echo "EXPOSURE=C1 idiom5500x3 | C2 chemistry5500x3 | C3 combined11000x3"
  echo "TRAINING=batch4 ga4 lr2e-5 wd0.01 warmup0.03 linear bf16 seed20260820 maxlen1024"
  echo "EVAL=frozen diagnostic2000 greedy thinking_off max_new_tokens512"
  echo

  python -m py_compile "$PY"
  COMPILE_RC=$?
  echo "PY_COMPILE_RC=$COMPILE_RC"
  if [[ "$COMPILE_RC" -ne 0 ]]; then
    write_state "FAIL" "compile"
    echo "FINAL_RESULT=COMPILE_FAIL"
    return 1
  fi

  write_state "RUNNING" "train_c123"

  echo
  echo "=== TRAIN C1/C2/C3 CONCURRENTLY ON NPU 0/1/2 ==="

  ASCEND_RT_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    python -u "$PY" train \
      --arm C1 \
      --c0 "$C0" \
      --targets-dir "$TARGETS" \
      --diagnostic-dir "$DIAG" \
      --train "$TARGETS/idiom_train5500_sft.jsonl" \
      --run-dir "$RUN/C1/train" \
      > "$RUN/C1_train.log" 2>&1 &
  P1=$!

  ASCEND_RT_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
    python -u "$PY" train \
      --arm C2 \
      --c0 "$C0" \
      --targets-dir "$TARGETS" \
      --diagnostic-dir "$DIAG" \
      --train "$TARGETS/chemistry_train5500_sft.jsonl" \
      --run-dir "$RUN/C2/train" \
      > "$RUN/C2_train.log" 2>&1 &
  P2=$!

  ASCEND_RT_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 \
    python -u "$PY" train \
      --arm C3 \
      --c0 "$C0" \
      --targets-dir "$TARGETS" \
      --diagnostic-dir "$DIAG" \
      --train "$TARGETS/combined_train11000_sft.jsonl" \
      --run-dir "$RUN/C3/train" \
      > "$RUN/C3_train.log" 2>&1 &
  P3=$!

  echo "TRAIN_PIDS C1=$P1 C2=$P2 C3=$P3"

  while kill -0 "$P1" 2>/dev/null || kill -0 "$P2" 2>/dev/null || kill -0 "$P3" 2>/dev/null
  do
    printf '%s phase=train_c123 ' "$(TZ=Asia/Shanghai date '+%Y-%m-%dT%H:%M:%S%z')"
    show_progress
    sleep 20
  done

  wait "$P1"; RC1=$?
  wait "$P2"; RC2=$?
  wait "$P3"; RC3=$?

  echo "TRAIN_RC C1=$RC1 C2=$RC2 C3=$RC3"

  if [[ "$RC1" -ne 0 || "$RC2" -ne 0 || "$RC3" -ne 0 ]]; then
    write_state "FAIL" "train_c123"
    echo "=== TRAIN TAILS ==="
    echo "--- C1 ---"; tail -n 80 "$RUN/C1_train.log"
    echo "--- C2 ---"; tail -n 80 "$RUN/C2_train.log"
    echo "--- C3 ---"; tail -n 80 "$RUN/C3_train.log"
    echo "FINAL_RESULT=C123_TRAIN_FAIL"
    return 2
  fi

  echo "C1_SFT=PASS"
  echo "C2_SFT=PASS"
  echo "C3_SFT=PASS"

  write_state "RUNNING" "eval_c123"

  echo
  echo "=== EVAL C1/C2/C3 CONCURRENTLY, 4 NPUS PER ARM ==="

  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 PYTHONUNBUFFERED=1 \
    python -u "$PY" eval-master \
      --arm C1 \
      --model "$RUN/C1/train/final_hf" \
      --c0 "$C0" \
      --targets-dir "$TARGETS" \
      --diagnostic-dir "$DIAG" \
      --eval-dir "$RUN/C1/eval" \
      --num-workers 4 \
      > "$RUN/C1_eval.log" 2>&1 &
  E1=$!

  ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 PYTHONUNBUFFERED=1 \
    python -u "$PY" eval-master \
      --arm C2 \
      --model "$RUN/C2/train/final_hf" \
      --c0 "$C0" \
      --targets-dir "$TARGETS" \
      --diagnostic-dir "$DIAG" \
      --eval-dir "$RUN/C2/eval" \
      --num-workers 4 \
      > "$RUN/C2_eval.log" 2>&1 &
  E2=$!

  ASCEND_RT_VISIBLE_DEVICES=8,9,10,11 PYTHONUNBUFFERED=1 \
    python -u "$PY" eval-master \
      --arm C3 \
      --model "$RUN/C3/train/final_hf" \
      --c0 "$C0" \
      --targets-dir "$TARGETS" \
      --diagnostic-dir "$DIAG" \
      --eval-dir "$RUN/C3/eval" \
      --num-workers 4 \
      > "$RUN/C3_eval.log" 2>&1 &
  E3=$!

  echo "EVAL_PIDS C1=$E1 C2=$E2 C3=$E3"

  while kill -0 "$E1" 2>/dev/null || kill -0 "$E2" 2>/dev/null || kill -0 "$E3" 2>/dev/null
  do
    printf '%s phase=eval_c123 ' "$(TZ=Asia/Shanghai date '+%Y-%m-%dT%H:%M:%S%z')"
    show_progress
    sleep 20
  done

  wait "$E1"; ER1=$?
  wait "$E2"; ER2=$?
  wait "$E3"; ER3=$?

  echo "EVAL_RC C1=$ER1 C2=$ER2 C3=$ER3"

  if [[ "$ER1" -ne 0 || "$ER2" -ne 0 || "$ER3" -ne 0 ]]; then
    write_state "FAIL" "eval_c123"
    echo "=== EVAL TAILS ==="
    echo "--- C1 ---"; tail -n 80 "$RUN/C1_eval.log"
    echo "--- C2 ---"; tail -n 80 "$RUN/C2_eval.log"
    echo "--- C3 ---"; tail -n 80 "$RUN/C3_eval.log"
    echo "FINAL_RESULT=C123_EVAL_FAIL"
    return 3
  fi

  write_state "RUNNING" "combine"

  python -u "$PY" combine \
    --run-root "$RUN" \
    --c0-chemistry-summary "$C0RUN/c0_chemistry_summary.json"
  COMBINE_RC=$?
  echo "COMBINE_RC=$COMBINE_RC"

  if [[ "$COMBINE_RC" -ne 0 ]]; then
    write_state "FAIL" "combine"
    echo "FINAL_RESULT=COMBINE_FAIL"
    return 4
  fi

  write_state "PASS" "server_complete"

  echo
  echo "=== FINAL TRAIN TAILS ==="
  echo "--- C1 ---"; tail -n 25 "$RUN/C1_train.log"
  echo "--- C2 ---"; tail -n 25 "$RUN/C2_train.log"
  echo "--- C3 ---"; tail -n 25 "$RUN/C3_train.log"

  echo
  echo "=== FINAL EVAL TAILS ==="
  echo "--- C1 ---"; tail -n 25 "$RUN/C1_eval.log"
  echo "--- C2 ---"; tail -n 25 "$RUN/C2_eval.log"
  echo "--- C3 ---"; tail -n 25 "$RUN/C3_eval.log"

  echo
  echo "SERVER_C123_PIPELINE=PASS"
  echo "IDIOM_C123_JUDGE_INPUT=$RUN/idiom_c123_diagnostic3000_for_judge.jsonl"
  echo "CHEMISTRY_COMPARISON=$RUN/chemistry_c0_c123_comparison.json"
  echo "NEXT=RUN_WINDOWS_DEEPSEEK_JUDGE"
  echo "FINAL_RESULT=SERVER_C123_SFT_AND_DIAGNOSTIC_PASS"
  return 0
}

(
  flock -n -E 75 9
  LOCK_RC=$?
  if [[ "$LOCK_RC" -ne 0 ]]; then
    echo "FINAL_RESULT=ALREADY_RUNNING"
  else
    run_locked 2>&1 | tee -a "$MASTER_LOG"
    PIPE_RC=${PIPESTATUS[0]}
    echo "FULLCHAIN_RC=$PIPE_RC" | tee -a "$MASTER_LOG"
  fi
) 9>"$LOCK"

echo "RUN=$RUN"
echo "MASTER_LOG=$MASTER_LOG"
echo "STATE=$STATE"
