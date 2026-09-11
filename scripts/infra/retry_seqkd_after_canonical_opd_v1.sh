#!/usr/bin/env bash

set +e
set +u
set +o pipefail 2>/dev/null

ROOT=/workspace/mtpatcher
PROJECT=$ROOT/repo/MT-Patcher-Reproduction-Ascend

SFT_RECIPE=$PROJECT/recipes/sft/canonical_seqkd_broad20k_qwen3_06b_8b.sh

STAMP=$(date +%Y%m%d_%H%M%S)

RUN=$ROOT/runs/science/seqkd_retry_after_opd_$STAMP

mkdir -p "$RUN"

echo "$RUN" \
  > /tmp/mtpatcher_seqkd_retry_run

echo "watcher_start=$(date -Is)" \
  > "$RUN/status.txt"


# ============================================================
# Discover the currently-running canonical OPD run.
# ============================================================

OPD_RUN=$(
  cat /tmp/mtpatcher_canonical_opd_run_dir \
    2>/dev/null
)

echo "opd_run=$OPD_RUN" \
  >> "$RUN/status.txt"


if [[ -z "$OPD_RUN" ]]; then

  echo \
    "OPD_RUN_POINTER_MISSING" \
    >> "$RUN/status.txt"

  CAN_RUN=0

else

  CAN_RUN=1

fi


# ============================================================
# Wait for OPD wrapper to finish.
#
# We do not care whether OPD scientific status is 0 or nonzero
# for purposes of running SeqKD: the two arms are independent.
# ============================================================

WAIT_MINUTES=0

while [[ "$CAN_RUN" -eq 1 ]] &&
      [[ ! -f "$OPD_RUN/final_status.txt" ]]
do

  sleep 60

  WAIT_MINUTES=$((WAIT_MINUTES + 1))

  if (( WAIT_MINUTES % 30 == 0 )); then

    echo \
      "still_waiting_opd minutes=$WAIT_MINUTES time=$(date -Is)" \
      >> "$RUN/status.txt"

  fi

  # 48h safety ceiling.
  if (( WAIT_MINUTES >= 2880 )); then

    echo \
      "OPD_WAIT_TIMEOUT" \
      >> "$RUN/status.txt"

    CAN_RUN=0

  fi

done


if [[ "$CAN_RUN" -eq 1 ]]; then

  OPD_STATUS=$(
    cat "$OPD_RUN/final_status.txt" \
      2>/dev/null
  )

  echo \
    "opd_finished=$(date -Is) opd_status=$OPD_STATUS" \
    >> "$RUN/status.txt"

fi


# ============================================================
# Give Ray / vLLM time to release resources.
# ============================================================

if [[ "$CAN_RUN" -eq 1 ]]; then

  sleep 120

  ACTIVE=1

  while [[ "$ACTIVE" -gt 0 ]]
  do

    ACTIVE=$(
      ps -eo args= |
        grep -E \
          'verl\.trainer\.main_ppo|verl\.trainer\.sft_trainer|torchrun.*sft_trainer' |
        grep -v grep |
        wc -l
    )

    if [[ "$ACTIVE" -gt 0 ]]; then

      echo \
        "waiting_for_training_cleanup active=$ACTIVE time=$(date -Is)" \
        >> "$RUN/status.txt"

      sleep 60

    fi

  done

fi


# ============================================================
# PID headroom gate.
# ============================================================

if [[ "$CAN_RUN" -eq 1 ]] &&
   [[ -r /sys/fs/cgroup/pids/pids.current ]] &&
   [[ -r /sys/fs/cgroup/pids/pids.max ]]
then

  CUR=$(
    cat /sys/fs/cgroup/pids/pids.current
  )

  MAX=$(
    cat /sys/fs/cgroup/pids/pids.max
  )

  if [[ "$MAX" != "max" ]] &&
     [[ "$CUR" =~ ^[0-9]+$ ]] &&
     [[ "$MAX" =~ ^[0-9]+$ ]]
  then

    REM=$((MAX - CUR))

    echo \
      "pid_headroom=$REM" \
      >> "$RUN/status.txt"

    if (( REM < 3000 )); then

      CAN_RUN=0

      echo \
        "SEQKD_RETRY_BLOCKED_PID_HEADROOM" \
        >> "$RUN/status.txt"

    fi

  fi

fi


# ============================================================
# Retry canonical SeqKD FROM BASE.
#
# This watcher itself is started with setsid, so torchrun has
# no controlling SSH terminal from which to receive SIGHUP.
# ============================================================

if [[ "$CAN_RUN" -eq 1 ]]; then

  echo \
    "seqkd_retry_start=$(date -Is)" \
    >> "$RUN/status.txt"

  bash "$SFT_RECIPE" \
    > "$RUN/seqkd_retry.log" \
    2>&1

  WRAPPER_STATUS=$?

  echo \
    "seqkd_wrapper_shell_status=$WRAPPER_STATUS" \
    >> "$RUN/status.txt"


  SFT_RUN=$(
    cat /tmp/mtpatcher_canonical_sft_run_dir \
      2>/dev/null
  )

  echo \
    "seqkd_run=$SFT_RUN" \
    >> "$RUN/status.txt"


  if [[ -n "$SFT_RUN" ]] &&
     [[ -f "$SFT_RUN/final_exit_status.txt" ]]
  then

    SCI_STATUS=$(
      cat "$SFT_RUN/final_exit_status.txt"
    )

  else

    SCI_STATUS=999

  fi


  echo \
    "seqkd_scientific_status=$SCI_STATUS" \
    >> "$RUN/status.txt"

  echo \
    "seqkd_retry_end=$(date -Is)" \
    >> "$RUN/status.txt"


  if [[ "$SCI_STATUS" == "0" ]]; then

    echo \
      "CANONICAL_SEQKD_RETRY_PASS" \
      >> "$RUN/status.txt"

  else

    echo \
      "CANONICAL_SEQKD_RETRY_FAIL" \
      >> "$RUN/status.txt"

  fi

fi


echo \
  "WATCHER_FINISHED time=$(date -Is)" \
  >> "$RUN/status.txt"
