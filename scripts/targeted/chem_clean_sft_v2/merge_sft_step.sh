#!/usr/bin/env bash
set -euo pipefail
source /workspace/mtpatcher/project_env.sh
ROOT=/workspace/mtpatcher
PROJECT="$ROOT/repo/MT-Patcher-Reproduction-Ascend"
VERL="$ROOT/repo/verl-v0.9.0"
PY="$ROOT/envs/verl-v0.9.0-a3/bin/python"
export PYTHONPATH="$PROJECT:$VERL:${PYTHONPATH:-}"
step=${1:-}
case "$step" in 160|320|480|640|800|960|1029) ;; *) echo 'Allowed checkpoint steps: 160 320 480 640 800 960 1029' >&2; exit 2;; esac
run="$ROOT/runs/formal/chem_clean_sft_matched_v2"
ckpt="$run/checkpoints/global_step_$step"
target="$run/evaluation/merged_step_$step"
test -d "$ckpt" || { echo "CHECKPOINT_NOT_FOUND: $ckpt" >&2; exit 3; }
test ! -e "$target" || { echo "TARGET_EXISTS: $target" >&2; exit 4; }
mkdir -p "$run/evaluation"
"$PY" -m verl.model_merger merge --backend fsdp --local_dir "$ckpt" --target_dir "$target"
echo "MERGED_TO=$target"
echo 'Next: call the frozen chemistry1000_native_parity_eval_v1.py using its ACTUAL --help flags; do not invent an evaluator contract.'
