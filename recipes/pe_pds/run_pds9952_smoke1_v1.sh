#!/usr/bin/env bash
# PDS9952 v1 — one-update engineering smoke for matched SeqKD and OPD.
# This script does NOT authorize or launch formal 3732-step training.

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

SEQCFG="$PROJECT/configs/sft/pds9952_seqkd_formal_v1.yaml"
OPDCFG="$PROJECT/configs/opd/pds9952_opd_formal_v1.yaml"
PROTOCOL="$PROJECT/manifests/pe_pds/pds9952_matched_formal_protocol_v1.json"
DATA="$ROOT/data/verl_science_pe_pds/pds9952_matched_v1"
ASSET_MANIFEST="$DATA/asset_manifest.json"
SCHEDULE="$DATA/source_order_manifest.jsonl"

STAGE="$ROOT/runs/science/pe_pds_v1/pds9952_stage_entry_v1/pe_opd_step4422_merged_hf"
STAGE_HASHES="$ROOT/runs/science/pe_pds_v1/pds9952_stage_entry_v1/merged_hf_files.sha256"

SMOKE="$ROOT/smoke/pe_pds_v1/pds9952_smoke1_v1"
SEQ="$SMOKE/seqkd"
OPD="$SMOKE/opd"
PROV="$SMOKE/provenance"

cd "$PROJECT" || {
    echo "PDS9952_SMOKE=FAIL_CD_PROJECT"
    return 2 2>/dev/null || true
}

export PYTHONPATH="$PROJECT:$FORMAL_VERL:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

fail() {
    echo "PDS9952_SMOKE=FAIL"
    echo "reason=$1"
    return "${2:-1}"
}

main() {
echo "============================================================"
echo "PDS9952 ONE-UPDATE ENGINEERING SMOKE"
date -u
echo "============================================================"

if [[ -e "$SMOKE" ]]; then
    echo "SMOKE_ROOT_EXISTS=BLOCKED"
    echo "$SMOKE"
    echo "Do not overwrite. Inspect or version the smoke root."
    return 20 2>/dev/null || true
fi

for f in "$SEQCFG" "$OPDCFG" "$PROTOCOL" "$ASSET_MANIFEST" "$SCHEDULE"; do
    if [[ ! -f "$f" ]]; then
        echo "MISSING=$f"
        return 21 2>/dev/null || true
    fi
done
if [[ ! -d "$STAGE" ]]; then
    echo "MISSING_STAGE_ENTRY=$STAGE"
    return 22 2>/dev/null || true
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
    return 23 2>/dev/null || true
fi

mkdir -p "$SEQ/tensorboard" "$OPD/tensorboard" "$PROV"

{
    echo "timestamp_utc=$(date -u +%FT%TZ)"
    echo "project_head=$(git rev-parse HEAD)"
    echo "formal_verl_head=$(git -C "$FORMAL_VERL" rev-parse HEAD 2>/dev/null)"
    echo
    echo "===== WORKTREE ====="
    git status --short
} > "$PROV/source_state.txt"

sha256sum \
    "$SEQCFG" \
    "$OPDCFG" \
    "$PROTOCOL" \
    "$ASSET_MANIFEST" \
    "$SCHEDULE" \
    "$DATA/pds9952_seqkd_matched59712.parquet" \
    "$DATA/pds9952_opd_matched59712.parquet" \
    > "$PROV/input_hashes.sha256"

if [[ -f "$STAGE_HASHES" ]]; then
    cp "$STAGE_HASHES" "$PROV/stage_entry_files.sha256"
fi

# Pre-smoke contract gate and the exact first matched global batch.
"$PY" - "$SEQCFG" "$OPDCFG" "$PROTOCOL" "$ASSET_MANIFEST" "$SCHEDULE" "$STAGE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
import yaml

seq=Path(sys.argv[1]); opd=Path(sys.argv[2]); protocol=Path(sys.argv[3])
asset=Path(sys.argv[4]); schedule=Path(sys.argv[5]); stage=Path(sys.argv[6])

s=yaml.safe_load(seq.read_text(encoding="utf-8"))
o=yaml.safe_load(opd.read_text(encoding="utf-8"))
p=json.loads(protocol.read_text(encoding="utf-8"))
a=json.loads(asset.read_text(encoding="utf-8"))

assert p["budget"]["population"] == 9952
assert p["budget"]["source_exposures"] == 59712
assert p["budget"]["optimizer_steps"] == 3732
assert p["primary_endpoint_step"] == 3732
assert a["matched_gate"]["status"] == "PASS"
assert a["matched_gate"]["per_exposure_source_order_equal"] is True
assert a["matched_gate"]["per_exposure_prompt_equal"] is True
assert a["matched_gate"]["terminal_global_step"] == 3732

stage_s=str(stage)
assert f"model.path={stage_s}" in s["overrides"]
assert f"actor_rollout_ref.model.path={stage_s}" in o["overrides"]

rows=[]
with schedule.open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))
            if len(rows) == 16:
                break
assert len(rows) == 16
assert all(int(x["global_step"]) == 1 for x in rows)
assert [int(x["position_in_batch"]) for x in rows] == list(range(16))
source_ids=[int(x["source_id"]) for x in rows]
payload=(",".join(map(str, source_ids))+"\n").encode()
print("FIRST_MATCHED_BATCH_SOURCE_IDS =", source_ids)
print("FIRST_MATCHED_BATCH_SHA256 =", hashlib.sha256(payload).hexdigest())
print("PDS9952_SMOKE_PREFLIGHT=PASS")
PY
PRE_RC=$?
if [[ "$PRE_RC" -ne 0 ]]; then
    echo "PREFLIGHT_RC=$PRE_RC"
    return 24 2>/dev/null || true
fi

# -------------------------
# Arm B: one SeqKD update
# -------------------------
mapfile -t SEQ_OVERRIDES < <(
    "$PY" - "$SEQCFG" <<'PY'
import sys, yaml
x=yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
for v in x["overrides"]:
    print(v)
PY
)

SEQ_OVERRIDES+=(
    "trainer.experiment_name=pds9952-seqkd-smoke1-v1"
    "trainer.total_epochs=1"
    "trainer.total_training_steps=1"
    "trainer.save_freq=1"
    "trainer.default_local_dir=$SEQ/checkpoints"
    "trainer.resume_mode=disable"
)

echo
echo "===== SEQKD ONE-UPDATE SMOKE ====="
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export TENSORBOARD_DIR="$SEQ/tensorboard"

(
    cd "$SEQ" || exit 1
    "$TORCHRUN" \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=16 \
        -m verl.trainer.sft_trainer \
        "${SEQ_OVERRIDES[@]}"
) > "$SEQ/train.log" 2>&1
SEQ_RC=$?
echo "$SEQ_RC" > "$SEQ/train_status.txt"
echo "SEQKD_RC=$SEQ_RC"

if [[ "$SEQ_RC" -ne 0 ]]; then
    tail -n 220 "$SEQ/train.log" || true
    return 30 2>/dev/null || true
fi

if [[ ! -d "$SEQ/checkpoints/global_step_1/huggingface" ]]; then
    echo "SEQKD_STEP1_HF_MISSING"
    find "$SEQ/checkpoints" -maxdepth 3 -type d -print 2>/dev/null | sort
    return 31 2>/dev/null || true
fi

# -------------------------
# Ensure clean Ray boundary
# -------------------------
"$A3/bin/ray" stop --force > "$PROV/ray_stop_before_opd.log" 2>&1 || true
sleep 5

ACTIVE_RAY="$(
    ps -eo stat,pid,cmd |
    awk '$1 !~ /^Z/' |
    grep -E 'raylet|gcs_server|ray/dashboard|TaskRunnerV1|vLLMHttpServer|GlobalRequestLoadBalancer' |
    grep -v grep || true
)"
if [[ -n "$ACTIVE_RAY" ]]; then
    echo "$ACTIVE_RAY" > "$PROV/active_ray_before_opd.txt"
    echo "ACTIVE_RAY_BEFORE_OPD=BLOCKED"
    return 32 2>/dev/null || true
fi

# -------------------------
# Arm C: one OPD update
# -------------------------
mapfile -t OPD_OVERRIDES < <(
    "$PY" - "$OPDCFG" <<'PY'
import sys, yaml
x=yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
for v in x["overrides"]:
    print(v)
PY
)

OPD_OVERRIDES+=(
    "trainer.experiment_name=pds9952-opd-smoke1-v1"
    "trainer.total_epochs=1"
    "trainer.total_training_steps=1"
    "trainer.save_freq=1"
    "trainer.default_local_dir=$OPD/checkpoints"
    "trainer.resume_mode=disable"
)

unset VERL_CUSTOM_TRAINER_MODULE
unset MATCHED20K_VALIDATION_JSONL
unset MATCHED20K_REPLAY_STEPS
unset MATCHED20K_REPLAY_CHECKPOINT_ROOT
unset RAY_ADDRESS

echo
echo "===== OPD ONE-UPDATE SMOKE ====="
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export TENSORBOARD_DIR="$OPD/tensorboard"

(
    cd "$OPD" || exit 1
    "$PY" -m verl.trainer.main_ppo "${OPD_OVERRIDES[@]}"
) > "$OPD/train.log" 2>&1
OPD_RC=$?
echo "$OPD_RC" > "$OPD/train_status.txt"
echo "OPD_RC=$OPD_RC"

if [[ "$OPD_RC" -ne 0 ]]; then
    tail -n 260 "$OPD/train.log" || true
    "$A3/bin/ray" stop --force >/dev/null 2>&1 || true
    return 40 2>/dev/null || true
fi

if [[ ! -d "$OPD/checkpoints/global_step_1/actor" ]]; then
    echo "OPD_STEP1_ACTOR_MISSING"
    find "$OPD/checkpoints" -maxdepth 3 -type d -print 2>/dev/null | sort
    "$A3/bin/ray" stop --force >/dev/null 2>&1 || true
    return 41 2>/dev/null || true
fi

"$A3/bin/ray" stop --force > "$PROV/ray_stop_after_opd.log" 2>&1 || true
sleep 5

ACTIVE_AFTER="$(
    ps -eo stat,pid,cmd |
    awk '$1 !~ /^Z/' |
    grep -E 'raylet|gcs_server|ray/dashboard|TaskRunnerV1|vLLMHttpServer|GlobalRequestLoadBalancer' |
    grep -v grep || true
)"
if [[ -n "$ACTIVE_AFTER" ]]; then
    echo "$ACTIVE_AFTER" > "$PROV/active_ray_after.txt"
    echo "ACTIVE_RAY_AFTER=FAIL"
    RAY_RC=1
else
    RAY_RC=0
fi

# Merge OPD step1 for a direct "weights changed from common init" engineering check.
OPD_MERGED="$OPD/step1_merged_hf"
"$PY" -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$OPD/checkpoints/global_step_1/actor" \
    --target_dir "$OPD_MERGED" \
    > "$OPD/merge.log" 2>&1
MERGE_RC=$?
echo "OPD_STEP1_MERGE_RC=$MERGE_RC"
if [[ "$MERGE_RC" -ne 0 ]]; then
    tail -n 160 "$OPD/merge.log" || true
    return 42 2>/dev/null || true
fi

# -------------------------
# Post-smoke engineering gate
# -------------------------
"$PY" - "$STAGE" "$SEQ/checkpoints/global_step_1/huggingface" "$OPD_MERGED" "$SEQ/tensorboard" "$OPD/tensorboard" "$SEQ/train.log" "$OPD/train.log" "$SMOKE/smoke_summary.json" <<'PY'
import glob
import hashlib
import json
import math
import re
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

stage=Path(sys.argv[1])
seq=Path(sys.argv[2])
opd=Path(sys.argv[3])
seq_tb=Path(sys.argv[4])
opd_tb=Path(sys.argv[5])
seq_log=Path(sys.argv[6])
opd_log=Path(sys.argv[7])
out=Path(sys.argv[8])

def aggregate_model_hash(root: Path) -> str:
    files=sorted(
        p for p in root.rglob("*.safetensors")
        if p.is_file()
    )
    if not files:
        raise RuntimeError(f"no safetensors under {root}")
    h=hashlib.sha256()
    for p in files:
        rel=p.relative_to(root).as_posix().encode()
        h.update(len(rel).to_bytes(8,"big"))
        h.update(rel)
        with p.open("rb") as f:
            for chunk in iter(lambda:f.read(8<<20), b""):
                h.update(chunk)
    return h.hexdigest()

def tb_scalars(root: Path):
    events=[
        Path(p) for p in glob.glob(
            str(root/"**/events.out.tfevents.*"),
            recursive=True,
        )
    ]
    events=[p for p in events if p.is_file() and p.stat().st_size>0]
    if not events:
        raise RuntimeError(f"no TensorBoard events under {root}")
    scalars={}
    tags=set()
    for d in sorted({p.parent for p in events}):
        ea=EventAccumulator(str(d), size_guidance={"scalars":0})
        ea.Reload()
        for tag in ea.Tags().get("scalars",[]):
            tags.add(tag)
            vals=ea.Scalars(tag)
            if vals:
                scalars.setdefault(tag,[]).extend(
                    (int(v.step), float(v.value)) for v in vals
                )
    return events,tags,scalars

def last_of(scalars, candidates):
    for c in candidates:
        vals=scalars.get(c,[])
        if vals:
            return c, vals[-1]
    return None,None

stage_hash=aggregate_model_hash(stage)
seq_hash=aggregate_model_hash(seq)
opd_hash=aggregate_model_hash(opd)

if seq_hash == stage_hash:
    raise RuntimeError("SeqKD step1 model hash equals common stage-entry")
if opd_hash == stage_hash:
    raise RuntimeError("OPD step1 model hash equals common stage-entry")

seq_events,seq_tags,seq_scalars=tb_scalars(seq_tb)
opd_events,opd_tags,opd_scalars=tb_scalars(opd_tb)

seq_loss_tag,seq_loss=last_of(seq_scalars,["train/loss","loss"])
seq_grad_tag,seq_grad=last_of(seq_scalars,["train/grad_norm","grad_norm"])
seq_lr_tag,seq_lr=last_of(seq_scalars,["train/lr","lr"])

opd_loss_tag,opd_loss=last_of(opd_scalars,[
    "actor/distillation/loss",
    "distillation/loss",
])
opd_grad_tag,opd_grad=last_of(opd_scalars,["actor/grad_norm","grad_norm"])
opd_lr_tag,opd_lr=last_of(opd_scalars,["actor/lr","lr"])
opd_step_tag,opd_step=last_of(opd_scalars,["training/global_step"])

for name, val in [
    ("seq_loss",seq_loss),
    ("seq_grad",seq_grad),
    ("seq_lr",seq_lr),
    ("opd_loss",opd_loss),
    ("opd_grad",opd_grad),
    ("opd_lr",opd_lr),
]:
    if val is None:
        raise RuntimeError(
            f"missing metric {name}; "
            f"seq_tags={sorted(seq_tags)} opd_tags={sorted(opd_tags)}"
        )
    if not math.isfinite(float(val[1])):
        raise RuntimeError(f"nonfinite metric {name}={val}")
if float(seq_grad[1]) <= 0:
    raise RuntimeError(f"SeqKD grad norm must be positive: {seq_grad}")
if float(opd_grad[1]) <= 0:
    raise RuntimeError(f"OPD grad norm must be positive: {opd_grad}")

seq_text=seq_log.read_text(encoding="utf-8",errors="replace")
opd_text=opd_log.read_text(encoding="utf-8",errors="replace")

summary={
    "status":"PASS_PDS9952_ONE_UPDATE_ENGINEERING_SMOKE",
    "scientific_effect_claim":False,
    "formal_training_authorized_by_this_file":False,
    "common_stage_entry_model_sha256":stage_hash,
    "seqkd_step1_model_sha256":seq_hash,
    "opd_step1_model_sha256":opd_hash,
    "seqkd":{
        "loss_tag":seq_loss_tag,
        "loss":seq_loss,
        "grad_norm_tag":seq_grad_tag,
        "grad_norm":seq_grad,
        "lr_tag":seq_lr_tag,
        "lr":seq_lr,
        "tensorboard_events":[str(p) for p in seq_events],
        "checkpoint":str(seq),
    },
    "opd":{
        "distillation_loss_tag":opd_loss_tag,
        "distillation_loss":opd_loss,
        "grad_norm_tag":opd_grad_tag,
        "grad_norm":opd_grad,
        "lr_tag":opd_lr_tag,
        "lr":opd_lr,
        "global_step_tag":opd_step_tag,
        "global_step":opd_step,
        "tensorboard_events":[str(p) for p in opd_events],
        "checkpoint":str(opd),
    },
}
tmp=out.with_suffix(".json.tmp")
tmp.write_text(
    json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True)+"\n",
    encoding="utf-8",
)
tmp.replace(out)
print(json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True))
print("PDS9952_ONE_UPDATE_SMOKE_POST_GATE=PASS")
PY
POST_RC=$?

echo "SEQKD_RC=$SEQ_RC"
echo "OPD_RC=$OPD_RC"
echo "MERGE_RC=$MERGE_RC"
echo "RAY_RC=$RAY_RC"
echo "POST_RC=$POST_RC"

if [[ "$SEQ_RC" -eq 0 && "$OPD_RC" -eq 0 && "$MERGE_RC" -eq 0 && "$RAY_RC" -eq 0 && "$POST_RC" -eq 0 ]]; then
    touch "$SMOKE/PASS"
    echo "PDS9952_ONE_UPDATE_ENGINEERING_SMOKE=PASS"
    FINAL_RC=0
else
    touch "$SMOKE/FAIL"
    echo "PDS9952_ONE_UPDATE_ENGINEERING_SMOKE=FAIL"
    FINAL_RC=50
fi

echo "SMOKE_ROOT=$SMOKE"
echo "SUMMARY=$SMOKE/smoke_summary.json"
date -u
return "$FINAL_RC"
}

main "$@"
SCRIPT_RC=$?
echo "PDS9952_SMOKE_SCRIPT_RC=$SCRIPT_RC"
test "$SCRIPT_RC" -eq 0
