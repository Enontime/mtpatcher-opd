#!/usr/bin/env bash
set -euo pipefail
source /workspace/mtpatcher/project_env.sh
ROOT=/workspace/mtpatcher
PROJECT="$ROOT/repo/MT-Patcher-Reproduction-Ascend"
VERL="$ROOT/repo/verl-v0.9.0"
PKG="$PROJECT/scripts/targeted/chem_clean_sft_v2"
PY="$ROOT/envs/verl-v0.9.0-a3/bin/python"
PYPREP="$ROOT/envs/mtpatcher-npu-py311/bin/python"
run="$ROOT/runs/formal/chem_clean_sft_matched_v2"
test -f "$run/train.log" || { echo 'FORMAL_LOG_MISSING' >&2; exit 2; }
# Do not verify an active run as completed.
if pgrep -af '[v]erl.trainer.sft_trainer'; then echo 'TRAINING_STILL_ACTIVE'; exit 3; fi
"$PY" "$PKG/check_tb.py" "$run/tensorboard" --last 1029
"$PYPREP" "$PKG/check_ckpt.py" "$run/checkpoints" --compare 160 960
"$PYPREP" - "$run/checkpoints/global_step_1029" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]);assert p.is_dir() and any(p.rglob('*')),'FINAL_STEP_1029_CHECKPOINT_MISSING'
print('FINAL_STEP_1029_CHECKPOINT_PRESENT',p)
PY
echo 'ENGINEERING_FINAL_PASS: review resumed optimizer-state evidence, evaluator outputs, and sample-order claims separately.'
