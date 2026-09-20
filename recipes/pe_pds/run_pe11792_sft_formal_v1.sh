#!/usr/bin/env bash

set +e
set +u
set +o pipefail 2>/dev/null

source /workspace/mtpatcher/project_env.sh

ROOT=/workspace/mtpatcher
PROJECT="$ROOT/repo/MT-Patcher-Reproduction-Ascend"
FORMAL_VERL="$ROOT/repo/verl-v0.9.0-matched20k-v2-formal"
A3="$ROOT/envs/verl-v0.9.0-a3"
PY="$A3/bin/python"
TORCHRUN="$A3/bin/torchrun"

CFG="$PROJECT/configs/sft/pe11792_sft_formal_v1.yaml"
PROTOCOL="$PROJECT/manifests/pe_pds/pe11792_matched_formal_protocol_v1.json"
ASSET_MANIFEST="$ROOT/data/verl_science_pe_pds/pe11792_matched_v1/asset_manifest.json"

RUN="$ROOT/runs/science/pe_pds_v1/pe_sft_formal_v1"
PROV="$RUN/provenance"
TB="$RUN/tensorboard"

cd "$PROJECT"
export PYTHONPATH="$PROJECT:$FORMAL_VERL:${PYTHONPATH:-}"

if [[ -e "$RUN" ]]; then
    echo "FORMAL_RUN_ROOT_EXISTS=BLOCKED"
    echo "$RUN"
    exit 20
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
    "$ROOT/data/verl_science_pe_pds/pe11792_matched_v1/pe_sft_matched70752.parquet" \
    > "$PROV/input_hashes.sha256"

cp "$CFG" "$PROV/formal_config.yaml"
cp "$PROTOCOL" "$PROV/formal_protocol.json"
cp "$ASSET_MANIFEST" "$PROV/asset_manifest.json"

"$PY" \
    scripts/infra/compose_verl_sft_config.py \
    --config-dir "$FORMAL_VERL/verl/trainer/config" \
    --output "$PROV/resolved_config.yaml" \
    -- \
    "${OVERRIDES[@]}" \
    > "$PROV/compose.log" 2>&1

if [[ "$?" -ne 0 ]]; then
    echo "SFT_FORMAL_COMPOSE=FAIL"
    exit 21
fi

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TENSORBOARD_DIR="$TB"

"$TORCHRUN" \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=16 \
    -m verl.trainer.sft_trainer \
    "${OVERRIDES[@]}" \
    > "$RUN/train.log" 2>&1

TRAIN_RC=$?
echo "$TRAIN_RC" > "$RUN/final_status.txt"

"$PY" - "$TB" "$RUN/checkpoints/global_step_4422" <<'PY'
from pathlib import Path
import math
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

tb = Path(sys.argv[1])
ckpt = Path(sys.argv[2])

assert ckpt.is_dir(), ckpt

events = list(tb.rglob("events.out.tfevents.*"))
assert events, "missing SFT TensorBoard"

ea = EventAccumulator(str(tb))
ea.Reload()

for tag in ("train/loss", "train/grad_norm", "train/lr"):
    vals = ea.Scalars(tag)
    assert vals, tag
    assert vals[-1].step == 4422, (tag, vals[-1].step)
    assert all(math.isfinite(x.value) for x in vals), tag

print("PE_SFT_FORMAL_POST_GATE=PASS")
PY

POST_RC=$?

echo "TRAIN_RC=$TRAIN_RC"
echo "POST_RC=$POST_RC"

if [[ "$TRAIN_RC" -eq 0 && "$POST_RC" -eq 0 ]]; then
    echo "PE11792_SFT_FORMAL=PASS"
    exit 0
fi

echo "PE11792_SFT_FORMAL=FAIL"
exit 1
