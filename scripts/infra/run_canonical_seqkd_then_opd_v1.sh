#!/usr/bin/env bash

set +e
set +u
set +o pipefail 2>/dev/null

ROOT=/workspace/mtpatcher
PROJECT=$ROOT/repo/MT-Patcher-Reproduction-Ascend

SFT_RECIPE=$PROJECT/recipes/sft/canonical_seqkd_broad20k_qwen3_06b_8b.sh
OPD_RECIPE=$PROJECT/recipes/opd/canonical_fkl_topk_broad20k_qwen3_06b_8b.sh

STAMP=$(date +%Y%m%d_%H%M%S)

PAIR_RUN=$ROOT/runs/science/canonical_seqkd_vs_opd_$STAMP

mkdir -p "$PAIR_RUN"

echo "$PAIR_RUN" \
  > /tmp/mtpatcher_canonical_pair_run


state() {
  local label="$1"

  echo
  echo "============================================================"
  echo "STATE $label"
  echo "============================================================"

  echo -n "time="
  date -Is

  echo -n "zombies="
  ps -eo stat= |
    awk '$1 ~ /^Z/ {n++} END {print n+0}'

  echo -n "processes="
  ps -e -o pid= | wc -l

  local current=/sys/fs/cgroup/pids/pids.current
  local maximum=/sys/fs/cgroup/pids/pids.max

  if [[ -r "$current" ]]; then
    echo "pids.current=$(cat "$current")"
  else
    echo "pids.current=UNAVAILABLE"
  fi

  if [[ -r "$maximum" ]]; then
    echo "pids.max=$(cat "$maximum")"
  else
    echo "pids.max=UNAVAILABLE"
  fi
}


pid_headroom_ok() {

  local current=/sys/fs/cgroup/pids/pids.current
  local maximum=/sys/fs/cgroup/pids/pids.max

  if [[ ! -r "$current" ]] ||
     [[ ! -r "$maximum" ]]; then

    echo \
      "PID_HEADROOM_UNKNOWN_ALLOW_WITH_LOGGING"

    return 0
  fi

  local cur
  local max

  cur=$(cat "$current")
  max=$(cat "$maximum")

  if [[ "$max" == "max" ]]; then
    echo \
      "PID_HEADROOM_PASS pids.max=max"

    return 0
  fi

  if ! [[ "$cur" =~ ^[0-9]+$ ]] ||
     ! [[ "$max" =~ ^[0-9]+$ ]]; then

    echo \
      "PID_HEADROOM_PARSE_UNKNOWN_ALLOW_WITH_LOGGING"

    return 0
  fi

  local remaining=$((max - cur))

  echo \
    "PID_HEADROOM current=$cur max=$max remaining=$remaining"

  # Conservative guard:
  # do not start another distributed arm with < 3000 PID slots.
  if (( remaining < 3000 )); then

    echo \
      "PID_HEADROOM_FAIL remaining=$remaining"

    return 1
  fi

  echo \
    "PID_HEADROOM_PASS remaining=$remaining"

  return 0
}


echo "============================================================"
echo "CANONICAL SCIENCE PAIR"
echo "============================================================"

echo "pair_run=$PAIR_RUN"

echo
echo "Arm S:"
echo "  Canonical Verl SeqKD20k"

echo
echo "Arm O:"
echo "  Canonical Verl Teacher-TopK FKL OPD20k"

echo
echo \
"compute_matched_claim=false"


# ============================================================
# Exact provenance snapshot
# ============================================================

{
  echo "timestamp=$(date -Is)"

  echo
  echo "=== PROJECT GIT ==="

  git -C "$PROJECT" \
    rev-parse HEAD

  git -C "$PROJECT" \
    status --short

  echo
  echo "=== VERL GIT ==="

  git -C "$ROOT/repo/verl-v0.9.0" \
    rev-parse HEAD

  git -C "$ROOT/repo/verl-v0.9.0" \
    status --short

  echo
  echo "=== SCIENCE FILE SHA256 ==="

  sha256sum \
    "$PROJECT/configs/sft/canonical_seqkd_broad20k_qwen3_06b_8b.yaml" \
    "$SFT_RECIPE" \
    "$PROJECT/configs/opd/canonical_fkl_topk_broad20k_qwen3_06b_8b.yaml" \
    "$OPD_RECIPE" \
    "$PROJECT/recipes/opd/smoke_fkl_topk_qwen3_06b_8b.sh" \
    "$PROJECT/scripts/data/verl_mt_response_sft_dataset.py" \
    "$PROJECT/scripts/opd/constant_zero_reward.py" \
    "$ROOT/data/verl_science_broad20k/seqkd_broad20k.parquet" \
    "$ROOT/data/verl_science_broad20k/opd_broad20k.parquet"

} > "$PAIR_RUN/provenance.txt" 2>&1


state BEFORE_ALL \
  | tee "$PAIR_RUN/state_before_all.txt"


echo
echo "=== NPU STATE BEFORE ALL ==="

npu-smi info \
  > "$PAIR_RUN/npu_before.txt" 2>&1

cat "$PAIR_RUN/npu_before.txt"


# ============================================================
# Prevent accidental overlap with another training job.
# ============================================================

ACTIVE=$(
  ps -eo args= |
    grep -E \
      'verl\.trainer\.main_ppo|verl\.trainer\.sft_trainer|torchrun.*sft_trainer' |
    grep -v grep |
    wc -l
)

echo
echo "active_training_process_matches=$ACTIVE"


ALLOW_START=1

if [[ "$ACTIVE" -gt 0 ]]; then

  ALLOW_START=0

  echo \
    "CANONICAL_PAIR_START_BLOCKED_ACTIVE_TRAINING"

fi


pid_headroom_ok

HEADROOM_STATUS=$?

if [[ "$HEADROOM_STATUS" -ne 0 ]]; then

  ALLOW_START=0

  echo \
    "CANONICAL_PAIR_START_BLOCKED_PID_HEADROOM"

fi


# ============================================================
# ARM S — SeqKD20k
# ============================================================

if [[ "$ALLOW_START" -eq 1 ]]; then

  echo
  echo "============================================================"
  echo "START ARM S — CANONICAL VERL SEQKD20K"
  echo "============================================================"

  date -Is \
    > "$PAIR_RUN/seqkd_start_time.txt"

  bash "$SFT_RECIPE" \
    2>&1 |
    tee "$PAIR_RUN/seqkd_master.log"

  SEQKD_SHELL_STATUS=${PIPESTATUS[0]}

  echo "$SEQKD_SHELL_STATUS" \
    > "$PAIR_RUN/seqkd_shell_status.txt"

  SFT_RUN=$(
    cat /tmp/mtpatcher_canonical_sft_run_dir \
      2>/dev/null
  )

  echo "$SFT_RUN" \
    > "$PAIR_RUN/seqkd_run_dir.txt"

  if [[ -n "$SFT_RUN" ]] &&
     [[ -f "$SFT_RUN/final_exit_status.txt" ]]; then

    SEQKD_STATUS=$(
      cat "$SFT_RUN/final_exit_status.txt"
    )

  else

    SEQKD_STATUS=999

  fi

  echo "$SEQKD_STATUS" \
    > "$PAIR_RUN/seqkd_scientific_status.txt"

  date -Is \
    > "$PAIR_RUN/seqkd_end_time.txt"

  if [[ "$SEQKD_STATUS" == "0" ]]; then

    echo \
      "CANONICAL_SEQKD20K_FORMAL_RUN_PASS"

  else

    echo \
      "CANONICAL_SEQKD20K_FORMAL_RUN_FAIL status=$SEQKD_STATUS"

  fi

  state AFTER_SEQKD \
    | tee "$PAIR_RUN/state_after_seqkd.txt"

else

  echo "999" \
    > "$PAIR_RUN/seqkd_scientific_status.txt"

  echo \
    "SEQKD_FORMAL_RUN_SKIPPED"

fi


# ============================================================
# ARM O — OPD20k
#
# Scientifically independent of SeqKD.
# We start it even if SeqKD itself failed, provided the machine
# still has enough PID headroom.
# ============================================================

pid_headroom_ok

OPD_HEADROOM_STATUS=$?


if [[ "$ALLOW_START" -eq 1 ]] &&
   [[ "$OPD_HEADROOM_STATUS" -eq 0 ]]; then

  echo
  echo "============================================================"
  echo "START ARM O — CANONICAL VERL TOPK-FKL OPD20K"
  echo "============================================================"

  date -Is \
    > "$PAIR_RUN/opd_start_time.txt"

  bash "$OPD_RECIPE" \
    2>&1 |
    tee "$PAIR_RUN/opd_master.log"

  OPD_SHELL_STATUS=${PIPESTATUS[0]}

  echo "$OPD_SHELL_STATUS" \
    > "$PAIR_RUN/opd_shell_status.txt"

  OPD_RUN=$(
    cat /tmp/mtpatcher_canonical_opd_run_dir \
      2>/dev/null
  )

  echo "$OPD_RUN" \
    > "$PAIR_RUN/opd_run_dir.txt"

  if [[ -n "$OPD_RUN" ]] &&
     [[ -f "$OPD_RUN/final_status.txt" ]]; then

    OPD_STATUS=$(
      cat "$OPD_RUN/final_status.txt"
    )

  else

    OPD_STATUS=999

  fi

  echo "$OPD_STATUS" \
    > "$PAIR_RUN/opd_scientific_status.txt"

  date -Is \
    > "$PAIR_RUN/opd_end_time.txt"

  if [[ "$OPD_STATUS" == "0" ]]; then

    echo \
      "CANONICAL_OPD20K_FORMAL_RUN_PASS"

  else

    echo \
      "CANONICAL_OPD20K_FORMAL_RUN_FAIL status=$OPD_STATUS"

  fi

  state AFTER_OPD \
    | tee "$PAIR_RUN/state_after_opd.txt"

else

  echo "999" \
    > "$PAIR_RUN/opd_scientific_status.txt"

  echo \
    "OPD_FORMAL_RUN_SKIPPED_PID_OR_PREFLIGHT"

fi


state AFTER_ALL \
  | tee "$PAIR_RUN/state_after_all.txt"


echo
echo "============================================================"
echo "CANONICAL SCIENCE PAIR FINISHED"
echo "============================================================"

echo "pair_run=$PAIR_RUN"

echo -n "seqkd_status="
cat "$PAIR_RUN/seqkd_scientific_status.txt" \
  2>/dev/null ||
  echo MISSING

echo -n "opd_status="
cat "$PAIR_RUN/opd_scientific_status.txt" \
  2>/dev/null ||
  echo MISSING

echo \
  "CANONICAL_PAIR_WRAPPER_FINISHED"
