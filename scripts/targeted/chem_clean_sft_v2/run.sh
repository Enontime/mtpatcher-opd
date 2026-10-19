#!/usr/bin/env bash
# Native Verl Chemistry Raw SeqKD launcher. Strictly no formal auto-run.
set -euo pipefail
source /workspace/mtpatcher/project_env.sh
ROOT=/workspace/mtpatcher
PROJECT=$ROOT/repo/MT-Patcher-Reproduction-Ascend
VERL=$ROOT/repo/verl-v0.9.0
PACKAGE=$PROJECT/scripts/targeted/chem_clean_sft_v2
STAGE=$ROOT/native_chem12h_v2/clean_sft_stage1
DATA=$ROOT/data/chem_clean_sft_matched_v1/chem_clean_sft_5500.parquet
PYPREP=$ROOT/envs/mtpatcher-npu-py311/bin/python
PY=$ROOT/envs/verl-v0.9.0-a3/bin/python
TORCHRUN=$ROOT/envs/verl-v0.9.0-a3/bin/torchrun
export PYTHONPATH="$PROJECT/scripts/opd:$PROJECT/scripts/targeted:$PROJECT:$VERL:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

usage() {
  cat <<'EOF'
Usage:
  bash run.sh data
  bash run.sh patch-inspect
  bash run.sh patch-apply
  bash run.sh smoke-a /absolute/Stage1_generated_config.yaml
  bash run.sh smoke-b /workspace/mtpatcher/smoke/chem_clean_sft_v2/<UTC>/
  bash run.sh formal /absolute/Stage1_generated_config.yaml /smoke/run_root YES_START_FORMAL
No command automatically kills processes, re-generates Teacher targets, or cleans artifacts.
EOF
}

ensure_idle() {
  if pgrep -af '[v]erl.trainer.sft_trainer|[m]ain_ppo_sync_wrapper_v1.py|[c]hem_clean_sft_data_v1.py generate'; then
    echo 'BLOCKED: other native trainer / teacher generation is present. Inspect before running.' >&2
    return 1
  fi
}

ensure_patched() {
  "$PYPREP" "$PACKAGE/patch_verl_terminal.py" inspect >/dev/null
  grep -q 'CHEM_SFT_V2_EPOCH_FINAL_SAVE' "$VERL/verl/trainer/sft_trainer.py" || { echo 'Run patch-apply only after reviewing git diff' >&2; return 1; }
}

record_provenance() {
  local run=$1
  {
    date -u '+UTC=%FT%TZ'
    git -C "$PROJECT" rev-parse HEAD
    git -C "$VERL" rev-parse HEAD
    echo '=== PROJECT WORKTREE ==='
    git -C "$PROJECT" status --short
    echo '=== NATIVE VERL WORKTREE ==='
    git -C "$VERL" status --short
    echo '=== INPUT SHA256 ==='
    sha256sum "$ROOT/data/chem_clean_sft_matched_v1/teacher_targets.jsonl" "$DATA" \
      "$VERL/verl/trainer/sft_trainer.py" "$PACKAGE/make_config.py"
  } > "$run/provenance.txt"
}

make_config() {
  local template=$1 run=$2 data=$3 epochs=$4 horizon=$5 freq=$6 resume=${7:-}
  local extra=()
  if [[ -n "$resume" ]]; then extra=(--resume "$resume"); fi
  "$PYPREP" "$PACKAGE/make_config.py" \
    --template "$template" --output "$run/config.yaml" \
    --data "$data" --ckpt-dir "$run/checkpoints" \
    --run-name "$(basename "$run")" \
    --epochs "$epochs" --horizon "$horizon" --save-freq "$freq" \
    "${extra[@]}"
  mapfile -t OVERRIDES < <("$PYPREP" - "$run/config.yaml" <<'PY'
import sys,yaml
obj=yaml.safe_load(open(sys.argv[1],encoding='utf-8'))
for o in obj['overrides']:
    print(o)
PY
)
  "$PY" "$PROJECT/scripts/infra/compose_verl_sft_config.py" \
    --config-dir "$VERL/verl/trainer/config" --output "$run/resolved_config.yaml" \
    -- "${OVERRIDES[@]}" > "$run/compose.log" 2>&1 || {
       cat "$run/compose.log" >&2; return 1;
    }
  "$PY" "$PACKAGE/probe_dataset.py" --resolved "$run/resolved_config.yaml" \
    --data "$data" > "$run/dataset_probe.log" 2>&1 || {
       cat "$run/dataset_probe.log" >&2; return 1;
    }
  cat "$run/dataset_probe.log"
}

train_foreground() {
  local run=$1
  mapfile -t OVERRIDES < <("$PYPREP" - "$run/config.yaml" <<'PY'
import sys,yaml
for o in yaml.safe_load(open(sys.argv[1],encoding='utf-8'))['overrides']:
    print(o)
PY
)
  export TENSORBOARD_DIR="$run/tensorboard"
  mkdir -p "$TENSORBOARD_DIR"
  echo "Starting 8 NPU smoke; unmodified original stdout/stderr: $run/train.log"
  (cd "$PROJECT" && "$TORCHRUN" --standalone --nnodes=1 --nproc_per_node=8 \
    -m verl.trainer.sft_trainer "${OVERRIDES[@]}") > "$run/train.log" 2>&1
}

cmd=${1:-}
case "$cmd" in
  data)
    "$PYPREP" "$PACKAGE/audit_prepare.py" build
    ;;
  patch-inspect)
    "$PYPREP" "$PACKAGE/patch_verl_terminal.py" inspect
    ;;
  patch-apply)
    "$PYPREP" "$PACKAGE/patch_verl_terminal.py" apply
    ;;
  smoke-a)
    test $# -eq 2 || { usage; exit 2; }
    ensure_idle
    ensure_patched
    "$PYPREP" "$PACKAGE/audit_prepare.py" verify-built
    id=$(date -u +%Y%m%dT%H%M%SZ)
    run="$ROOT/smoke/chem_clean_sft_v2/$id"
    test ! -e "$run" || { echo "SMOKE_ID_EXISTS $run" >&2; exit 3; }
    mkdir -p "$run"
    printf '%s\n' "$2" > "$run/template_path.txt"
    "$PYPREP" - "$DATA" "$run/smoke80.parquet" <<'PY'
import sys,pyarrow.parquet as pq
p,q=sys.argv[1:]
a=pq.read_table(p)
assert a.num_rows==5500
pq.write_table(a.slice(0,80),q)
print('SMOKE_DATA_80_READY',q)
PY
    mkdir -p "$run/phaseA"
    make_config "$2" "$run/phaseA" "$run/smoke80.parquet" 1 6 2
    record_provenance "$run"
    train_foreground "$run/phaseA" || {
      echo "SMOKE_A_FAILED see raw $run/phaseA/train.log" >&2; exit 4;
    }
    "$PY" "$PACKAGE/check_tb.py" "$run/phaseA/tensorboard" --last 5
    "$PYPREP" "$PACKAGE/check_ckpt.py" "$run/phaseA/checkpoints" --compare 2 5
    touch "$run/SMOKE_A_PASS"
    echo "SMOKE_A_PASS $run"
    ;;
  smoke-b)
    test $# -eq 2 || { usage; exit 2; }
    ensure_idle
    ensure_patched
    run=${2%/}
    test -f "$run/SMOKE_A_PASS" || { echo 'SMOKE_A_NOT_PASSED' >&2; exit 3; }
    test ! -e "$run/phaseB" || { echo 'PHASE_B_ALREADY_EXISTS' >&2; exit 4; }
    mkdir -p "$run/phaseB"
    template=$(cat "$run/template_path.txt")
    make_config "$template" "$run/phaseB" "$run/smoke80.parquet" 2 6 2 "$run/phaseA/checkpoints/global_step_4"
    train_foreground "$run/phaseB" || {
      echo "SMOKE_B_FAILED see raw $run/phaseB/train.log" >&2; exit 5;
    }
    test -d "$run/phaseB/checkpoints/global_step_6" || {
      echo "STEP6_CHECKPOINT_MISSING" >&2
      exit 6
    }
    "$PY" "$PACKAGE/check_tb.py" "$run/phaseB/tensorboard" --start 5 --last 6
    # A resumed run must preserve global-step (step6 rather than restarting at1).
    "$PYPREP" - "$run/phaseB/train.log" <<'PY'
import sys
s=open(sys.argv[1],errors='replace').read().lower()
assert any(x in s for x in ('resume','restore','load checkpoint','loading checkpoint')), 'NO_RESUME_EVIDENCE_IN_LOG'
print('RESUME_LOG_MARKER_PRESENT')
PY
    touch "$run/SMOKE_V2_PASS"
    echo "SMOKE_V2_PASS $run"
    ;;
  formal)
    test $# -eq 4 && test "$4" = YES_START_FORMAL || { usage; exit 2; }
    ensure_idle
    ensure_patched
    test -f "$3/SMOKE_V2_PASS" || { echo 'REQUIRED_SMOKE_V2_PASS_MISSING' >&2; exit 3; }
    "$PYPREP" "$PACKAGE/audit_prepare.py" verify-built
    run="$ROOT/runs/formal/chem_clean_sft_matched_v2"
    test ! -e "$run" || { echo 'FORMAL_ROOT_EXISTS; refusing overwrite' >&2; exit 4; }
    mkdir -p "$run"
    make_config "$2" "$run" "$DATA" 3 1032 160
    record_provenance "$run"
    mapfile -t OVERRIDES < <("$PYPREP" - "$run/config.yaml" <<'PY'
import sys,yaml
for o in yaml.safe_load(open(sys.argv[1],encoding='utf-8'))['overrides']:
 print(o)
PY
)
    export TENSORBOARD_DIR="$run/tensorboard"
    mkdir -p "$TENSORBOARD_DIR"
    printf '%s\n' "$(printf '%q ' "$TORCHRUN" --standalone --nnodes=1 --nproc_per_node=8 -m verl.trainer.sft_trainer "${OVERRIDES[@]}")" > "$run/command.sh.txt"
    echo 'FORMAL_LAUNCHING; output is raw/unfiltered train.log'
    (cd "$PROJECT" && nohup setsid "$TORCHRUN" --standalone --nnodes=1 --nproc_per_node=8 \
      -m verl.trainer.sft_trainer "${OVERRIDES[@]}" \
      > "$run/train.log" 2>&1 < /dev/null & echo "FORMAL_BG_SHELL_PID=$!")
    echo "Raw live log: tail -n 120 -F '$run/train.log'"
    ;;
  *) usage; exit 2;;
esac
