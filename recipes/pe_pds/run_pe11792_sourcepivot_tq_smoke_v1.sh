#!/usr/bin/env bash

set +e
set +u
set +o pipefail 2>/dev/null

source /workspace/mtpatcher/project_env.sh

ROOT=/workspace/mtpatcher
PROJECT="$ROOT/repo/MT-Patcher-Reproduction-Ascend"
FORMAL_VERL="$ROOT/repo/verl-v0.9.0-matched20k-v2-patchaware-v1"
A3="$ROOT/envs/verl-v0.9.0-a3"
PY="$A3/bin/python"

CFG="$PROJECT/configs/opd/pe11792_sourcepivot_tq_smoke_v1.yaml"
PROTOCOL="$PROJECT/manifests/pe_pds/pe11792_matched_formal_protocol_v1.json"
ASSET_MANIFEST="$ROOT/data/verl_science_pe_pds/pe11792_matched_v1/asset_manifest.json"

RUN="$ROOT/runs/diagnostics/pe_pds_v1/sourcepivot_tq_smoke_v1/formal_run"
PROV="$RUN/provenance"
TB="$RUN/tensorboard"

OLD_TB="$ROOT/runs/diagnostics/matched20k_v2/opd_formal_v2_budget_extension8_v1/validation_7600_10000_v1/tensorboard"

cd "$PROJECT"
export PYTHONPATH="$PROJECT:$FORMAL_VERL:${PYTHONPATH:-}"

if [[ -e "$RUN" ]]; then
    echo "FORMAL_RUN_ROOT_EXISTS=BLOCKED"
    echo "$RUN"
    exit 30
fi

ACTIVE_RAY="$(
    ps -eo stat,pid,cmd |
    awk '$1 !~ /^Z/' |
    grep -E 'raylet|gcs_server|ray/dashboard|TaskRunnerV1|vLLMHttpServer|GlobalRequestLoadBalancer' |
    grep -v grep || true
)"

if [[ -n "$ACTIVE_RAY" ]]; then
    echo "ACTIVE_RAY_FOUND=BLOCKED"
    echo "$ACTIVE_RAY"
    exit 31
fi

mkdir -p "$PROV" "$TB"

mapfile -t OVERRIDES < <(
    "$PY" - "$CFG" <<'PY'
import sys, yaml
x = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
for v in x["overrides"]:
    print(v)
PY
)

{
    echo "timestamp_utc=$(date -u +%FT%TZ)"
    echo "project_head=$(git rev-parse HEAD)"
    echo "formal_verl_head=$(git -C "$FORMAL_VERL" rev-parse HEAD)"
    echo
    echo "===== WORKTREE ====="
    git status --short
} > "$PROV/source_state.txt"

sha256sum \
    "$CFG" \
    "$PROTOCOL" \
    "$ASSET_MANIFEST" \
    "$ROOT/data/verl_science_pe_pds/pe11792_matched_v1/source_order_manifest.jsonl" \
    "$ROOT/data/verl_science_pe_pds/pe11792_matched_v1/pe_opd_matched70752.parquet" \
    > "$PROV/input_hashes.sha256"

cp "$CFG" "$PROV/formal_config.yaml"
cp "$PROTOCOL" "$PROV/formal_protocol.json"
cp "$ASSET_MANIFEST" "$PROV/asset_manifest.json"

"$PY" \
    -m verl.trainer.main_ppo \
    --cfg job \
    --resolve \
    "${OVERRIDES[@]}" \
    > "$PROV/resolved_config.yaml" \
    2> "$PROV/compose.stderr"

if [[ "$?" -ne 0 ]]; then
    echo "OPD_FORMAL_COMPOSE=FAIL"
    exit 32
fi

find "$OLD_TB" \
    -type f \
    -name 'events.out.tfevents.*' \
    -print0 \
    2>/dev/null |
sort -z |
xargs -0 -r sha256sum \
    > "$PROV/old_diag_tb_before.sha256"

unset VERL_CUSTOM_TRAINER_MODULE
unset MATCHED20K_VALIDATION_JSONL
unset MATCHED20K_REPLAY_STEPS
unset MATCHED20K_REPLAY_CHECKPOINT_ROOT
unset RAY_ADDRESS

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TENSORBOARD_DIR="$TB"

"$PY" \
    -m verl.trainer.main_ppo \
    "${OVERRIDES[@]}" \
    > "$RUN/train.log" 2>&1

TRAIN_RC=$?
echo "$TRAIN_RC" > "$RUN/final_status.txt"

"$PY" - "$TB" "$RUN/checkpoints/global_step_4422/actor" <<'PY'
from pathlib import Path
import math
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

tb = Path(sys.argv[1])
ckpt = Path(sys.argv[2])

assert ckpt.is_dir(), ckpt

events = list(tb.rglob("events.out.tfevents.*"))
assert events, "missing OPD TensorBoard"

ea = EventAccumulator(str(tb))
ea.Reload()

required = (
    "actor/distillation/loss",
    "actor/loss",
    "actor/grad_norm",
    "actor/lr",
    "training/global_step",
)

for tag in required:
    vals = ea.Scalars(tag)
    assert vals, tag
    assert vals[-1].step == 4422, (tag, vals[-1].step)
    assert all(math.isfinite(x.value) for x in vals), tag

assert ea.Scalars("training/global_step")[-1].value == 4422.0

print("PE_OPD_FORMAL_POST_GATE=PASS")
PY

POST_RC=$?

find "$OLD_TB" \
    -type f \
    -name 'events.out.tfevents.*' \
    -print0 \
    2>/dev/null |
sort -z |
xargs -0 -r sha256sum \
    > "$PROV/old_diag_tb_after.sha256"

diff -u \
    "$PROV/old_diag_tb_before.sha256" \
    "$PROV/old_diag_tb_after.sha256" \
    > "$PROV/old_diag_tb.diff"

OLD_TB_RC=$?

"$A3/bin/ray" stop --force \
    > "$PROV/ray_stop.log" 2>&1 || true

sleep 5

ACTIVE_AFTER="$(
    ps -eo stat,pid,cmd |
    awk '$1 !~ /^Z/' |
    grep -E 'raylet|gcs_server|ray/dashboard|TaskRunnerV1|vLLMHttpServer|GlobalRequestLoadBalancer' |
    grep -v grep || true
)"

if [[ -z "$ACTIVE_AFTER" ]]; then
    RAY_RC=0
else
    RAY_RC=1
    printf '%s\n' "$ACTIVE_AFTER" > "$PROV/active_ray_after.txt"
fi

echo "TRAIN_RC=$TRAIN_RC"
echo "POST_RC=$POST_RC"
echo "OLD_TB_RC=$OLD_TB_RC"
echo "RAY_RC=$RAY_RC"

if [[ \
    "$TRAIN_RC" -eq 0 && \
    "$POST_RC" -eq 0 && \
    "$OLD_TB_RC" -eq 0 && \
    "$RAY_RC" -eq 0 \
]]; then
    echo "PE11792_OPD_FORMAL=PASS"
    exit 0
fi

echo "PE11792_OPD_FORMAL=FAIL"
exit 1
