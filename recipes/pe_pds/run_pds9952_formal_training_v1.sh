#!/usr/bin/env bash
# PDS9952 v1 formal training launcher.
# Frozen comparison:
#   B = canonical fixed-target SeqKD
#   C = canonical online FKL OPD
# Both start from the same merged PE-OPD step4422 HF checkpoint and consume the
# same 59,712-row frozen source schedule. This launcher does not run validation
# replay; validation is post-hoc on frozen checkpoints.

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
ASSET="$DATA/asset_manifest.json"
STAGE="$ROOT/runs/science/pe_pds_v1/pds9952_stage_entry_v1/pe_opd_step4422_merged_hf"
SMOKE="$ROOT/smoke/pe_pds_v1/pds9952_smoke1_v1"

MODE="${1:---preflight}"


EXPECTED_POP="1491babf8cf2625fcf275cfee1d4a812a0f8989aad1bd6f0d024f5240b86b937"
EXPECTED_TEACHER="7c5295563d4db9f201f223a4da1c309a28a59e6197dd081ce319e5f999f6635d"
EXPECTED_ORDER="eb6473ac6fd09db7171107f2fcd4d89b65aa909bd5a1312a2c6dfb93e9d3dfa2"
EXPECTED_SEQ="76bdfe71c0125ab4401564a349cedf06e54078a115b590afc91003636df94dca"
EXPECTED_OPD="6986a1b1da8738c73d7fa555db9bafe075d67d3db340f51168705b99c39ed9e6"


get_override() {
    local cfg="$1"
    local key="$2"
    "$PY" - "$cfg" "$key" <<'PY'
import sys, yaml
cfg=yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
key=sys.argv[2]
hits=[]
for x in cfg["overrides"]:
    s=str(x)
    if s.startswith(key+"="):
        hits.append(s.split("=",1)[1])
if len(hits)!=1:
    raise SystemExit(f"expected exactly one {key}= override; got {hits}")
print(hits[0].strip("'\""))
PY
}

preflight() {
    echo
    echo "===== PREFLIGHT ====="

    for f in "$SEQCFG" "$OPDCFG" "$PROTOCOL" "$ASSET"; do
        if [[ ! -s "$f" ]]; then
            echo "MISSING=$f"
            return 10
        fi
    done
    if [[ ! -d "$STAGE" ]]; then
        echo "MISSING_STAGE=$STAGE"
        return 11
    fi
    if [[ ! -f "$SMOKE/PASS" ]]; then
        echo "MISSING_SMOKE_PASS=$SMOKE/PASS"
        return 12
    fi

    local got
    got="$(sha256sum "$DATA/pds9952_population_v1.jsonl" | awk '{print $1}')"
    [[ "$got" == "$EXPECTED_POP" ]] || { echo "POP_SHA_MISMATCH=$got"; return 13; }

    got="$(sha256sum "$DATA/teacher_targets_pds9952_qwen3_8b_v1.jsonl" | awk '{print $1}')"
    [[ "$got" == "$EXPECTED_TEACHER" ]] || { echo "TEACHER_SHA_MISMATCH=$got"; return 14; }

    got="$(sha256sum "$DATA/source_order_manifest.jsonl" | awk '{print $1}')"
    [[ "$got" == "$EXPECTED_ORDER" ]] || { echo "ORDER_SHA_MISMATCH=$got"; return 15; }

    got="$(sha256sum "$DATA/pds9952_seqkd_matched59712.parquet" | awk '{print $1}')"
    [[ "$got" == "$EXPECTED_SEQ" ]] || { echo "SEQ_SHA_MISMATCH=$got"; return 16; }

    got="$(sha256sum "$DATA/pds9952_opd_matched59712.parquet" | awk '{print $1}')"
    [[ "$got" == "$EXPECTED_OPD" ]] || { echo "OPD_SHA_MISMATCH=$got"; return 17; }

    "$PY" - "$SEQCFG" "$OPDCFG" "$PROTOCOL" "$ASSET" "$STAGE" <<'PY'
import json, sys, yaml
from pathlib import Path

seq=Path(sys.argv[1]); opd=Path(sys.argv[2])
protocol=Path(sys.argv[3]); asset=Path(sys.argv[4]); stage=str(Path(sys.argv[5]))

s=yaml.safe_load(seq.read_text(encoding="utf-8"))
o=yaml.safe_load(opd.read_text(encoding="utf-8"))
p=json.loads(protocol.read_text(encoding="utf-8"))
a=json.loads(asset.read_text(encoding="utf-8"))

def ov(cfg, key):
    hits=[str(x).split("=",1)[1] for x in cfg["overrides"] if str(x).startswith(key+"=")]
    assert len(hits)==1, (key,hits)
    return hits[0].strip("'\"")

assert int(ov(s,"trainer.total_training_steps")) == 3732
assert int(ov(o,"trainer.total_training_steps")) == 3732
assert int(ov(s,"trainer.save_freq")) == 100
assert int(ov(o,"trainer.save_freq")) == 100
assert ov(s,"trainer.resume_mode") == "disable"
assert ov(o,"trainer.resume_mode") == "disable"

assert ov(s,"model.path") == stage
assert ov(o,"actor_rollout_ref.model.path") == stage

assert ov(s,"data.train_files").endswith("pds9952_seqkd_matched59712.parquet")
assert "pds9952_opd_matched59712.parquet" in ov(o,"data.train_files")
assert ov(s,"data.train_batch_size") == "16"
assert ov(o,"data.train_batch_size") == "16"

assert ov(s,"optim.lr") == "2e-5"
assert ov(s,"optim.lr_scheduler_type") == "cosine"
assert ov(s,"optim.lr_warmup_steps_ratio") == "0.03"
assert ov(o,"actor_rollout_ref.actor.optim.lr") == "1e-6"

assert a["matched_gate"]["status"] == "PASS"
assert a["matched_gate"]["per_exposure_source_order_equal"] is True
assert a["matched_gate"]["per_exposure_prompt_equal"] is True
assert a["matched_gate"]["terminal_global_step"] == 3732
assert a["schedule"]["total_rows"] == 59712
assert a["schedule"]["total_optimizer_steps"] == 3732

assert p["primary_endpoint_step"] == 3732
assert p["budget"]["optimizer_steps"] == 3732
assert p["budget"]["source_exposures"] == 59712

print("PDS9952_FORMAL_SPEC_GATE=PASS")
print("SeqKD warmup steps =", int(0.03*3732))
PY
    local rc=$?
    [[ "$rc" -eq 0 ]] || return 18

    local active
    active="$(
        ps -eo stat,pid,cmd |
        awk '$1 !~ /^Z/' |
        grep -E 'raylet|gcs_server|ray/dashboard|TaskRunnerV1|vLLMHttpServer|GlobalRequestLoadBalancer|verl.trainer.sft_trainer' |
        grep -v grep || true
    )"
    if [[ -n "$active" ]]; then
        echo "ACTIVE_TRAINING_OR_RAY=BLOCKED"
        echo "$active"
        return 19
    fi

    echo "PDS9952_FORMAL_PREFLIGHT=PASS"
    return 0
}

prepare_run_provenance() {
    local run="$1"
    local arm="$2"
    mkdir -p "$run/provenance" "$run/tensorboard"
    {
        echo "arm=$arm"
        echo "timestamp_utc=$(date -u +%FT%TZ)"
        echo "project_head=$(git rev-parse HEAD)"
        echo "formal_verl_head=$(git -C "$FORMAL_VERL" rev-parse HEAD 2>/dev/null)"
        echo "stage=$STAGE"
        echo
        echo "===== WORKTREE ====="
        git status --short
    } > "$run/provenance/source_state.txt"

    sha256sum \
        "$SEQCFG" \
        "$OPDCFG" \
        "$PROTOCOL" \
        "$ASSET" \
        "$DATA/pds9952_population_v1.jsonl" \
        "$DATA/teacher_targets_pds9952_qwen3_8b_v1.jsonl" \
        "$DATA/source_order_manifest.jsonl" \
        "$DATA/pds9952_seqkd_matched59712.parquet" \
        "$DATA/pds9952_opd_matched59712.parquet" \
        > "$run/provenance/input_hashes.sha256"
}

block_existing_run() {
    local run="$1"
    if [[ -f "$run/train.log" ]] || \
       [[ -f "$run/formal_status.txt" ]] || \
       find "$run/checkpoints" -maxdepth 1 -type d -name 'global_step_*' -print -quit 2>/dev/null | grep -q .
    then
        echo "EXISTING_FORMAL_RUN=BLOCKED"
        echo "$run"
        return 1
    fi
    return 0
}

run_seqkd() {
    echo
    echo "============================================================"
    echo "ARM B — PDS9952 SEQKD FORMAL"
    echo "============================================================"

    block_existing_run "$SEQ_RUN" || return 30
    prepare_run_provenance "$SEQ_RUN" "PDS9952-SeqKD"

    mapfile -t OVERRIDES < <(
        "$PY" - "$SEQCFG" <<'PY'
import sys,yaml
cfg=yaml.safe_load(open(sys.argv[1],encoding="utf-8"))
for x in cfg["overrides"]:
    print(x)
PY
    )

    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    export TENSORBOARD_DIR="$SEQ_RUN/tensorboard"

    (
        cd "$SEQ_RUN" || exit 1
        "$TORCHRUN" \
            --standalone \
            --nnodes=1 \
            --nproc_per_node=16 \
            -m verl.trainer.sft_trainer \
            "${OVERRIDES[@]}"
    ) > "$SEQ_RUN/train.log" 2>&1
    local rc=$?

    echo "$rc" > "$SEQ_RUN/formal_status.txt"
    echo "SEQKD_TRAIN_RC=$rc"

    if [[ "$rc" -ne 0 ]]; then
        tail -n 240 "$SEQ_RUN/train.log" || true
        return 31
    fi

    "$PY" - "$SEQ_RUN" <<'PY'
import glob, math, sys
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run=Path(sys.argv[1])
final=run/"checkpoints/global_step_3732/huggingface"
assert final.is_dir(), final
assert list(final.glob("model*.safetensors")), final

expected=list(range(100,3701,100))+[3732]
actual=set()
for p in (run/"checkpoints").glob("global_step_*"):
    try: actual.add(int(p.name.rsplit("_",1)[1]))
    except: pass
missing=[x for x in expected if x not in actual]
assert not missing, f"missing SeqKD checkpoints {missing}"

events=[Path(p) for p in glob.glob(str(run/"tensorboard/**/events.out.tfevents.*"),recursive=True)]
events=[p for p in events if p.is_file() and p.stat().st_size>0]
assert events, "missing SeqKD TensorBoard"

scalars={}
for d in sorted({p.parent for p in events}):
    ea=EventAccumulator(str(d),size_guidance={"scalars":0}); ea.Reload()
    for tag in ea.Tags().get("scalars",[]):
        vals=ea.Scalars(tag)
        if vals: scalars.setdefault(tag,[]).extend(vals)

for tag in ("train/loss","train/grad_norm","train/lr"):
    vals=scalars.get(tag,[])
    assert vals, (tag, sorted(scalars))
    assert any(int(x.step)==3732 for x in vals), (tag, vals[-3:])
    assert all(math.isfinite(float(x.value)) for x in vals), tag

print("PDS9952_SEQKD_FORMAL_POST_GATE=PASS")
PY
    local post=$?
    echo "SEQKD_POST_RC=$post"
    [[ "$post" -eq 0 ]] || return 32

    touch "$SEQ_RUN/PASS"
    return 0
}

clean_ray() {
    "$A3/bin/ray" stop --force > /tmp/pds9952_formal_ray_stop.log 2>&1 || true
    sleep 5
    local active
    active="$(
        ps -eo stat,pid,cmd |
        awk '$1 !~ /^Z/' |
        grep -E 'raylet|gcs_server|ray/dashboard|TaskRunnerV1|vLLMHttpServer|GlobalRequestLoadBalancer' |
        grep -v grep || true
    )"
    if [[ -n "$active" ]]; then
        echo "RAY_CLEANUP_FAIL"
        echo "$active"
        return 1
    fi
    return 0
}

run_opd() {
    echo
    echo "============================================================"
    echo "ARM C — PDS9952 OPD FORMAL"
    echo "============================================================"

    clean_ray || return 40
    block_existing_run "$OPD_RUN" || return 41
    prepare_run_provenance "$OPD_RUN" "PDS9952-OPD"

    mapfile -t OVERRIDES < <(
        "$PY" - "$OPDCFG" <<'PY'
import sys,yaml
cfg=yaml.safe_load(open(sys.argv[1],encoding="utf-8"))
for x in cfg["overrides"]:
    print(x)
PY
    )

    unset VERL_CUSTOM_TRAINER_MODULE
    unset MATCHED20K_VALIDATION_JSONL
    unset MATCHED20K_REPLAY_STEPS
    unset MATCHED20K_REPLAY_CHECKPOINT_ROOT
    unset RAY_ADDRESS

    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    export TENSORBOARD_DIR="$OPD_RUN/tensorboard"

    (
        cd "$OPD_RUN" || exit 1
        "$PY" -m verl.trainer.main_ppo "${OVERRIDES[@]}"
    ) > "$OPD_RUN/train.log" 2>&1
    local rc=$?

    echo "$rc" > "$OPD_RUN/formal_status.txt"
    echo "OPD_TRAIN_RC=$rc"

    clean_ray
    local rayrc=$?
    echo "OPD_RAY_CLEAN_RC=$rayrc"

    if [[ "$rc" -ne 0 ]]; then
        tail -n 300 "$OPD_RUN/train.log" || true
        return 42
    fi
    [[ "$rayrc" -eq 0 ]] || return 43

    "$PY" - "$OPD_RUN" <<'PY'
import glob, math, sys
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run=Path(sys.argv[1])
final=run/"checkpoints/global_step_3732/actor"
assert final.is_dir(), final
assert list(final.glob("model_world_size_*_rank_*.pt")), final

expected=list(range(100,3701,100))+[3732]
actual=set()
for p in (run/"checkpoints").glob("global_step_*"):
    try: actual.add(int(p.name.rsplit("_",1)[1]))
    except: pass
missing=[x for x in expected if x not in actual]
assert not missing, f"missing OPD checkpoints {missing}"

events=[Path(p) for p in glob.glob(str(run/"tensorboard/**/events.out.tfevents.*"),recursive=True)]
events=[p for p in events if p.is_file() and p.stat().st_size>0]
assert events, "missing OPD TensorBoard"

scalars={}
for d in sorted({p.parent for p in events}):
    ea=EventAccumulator(str(d),size_guidance={"scalars":0}); ea.Reload()
    for tag in ea.Tags().get("scalars",[]):
        vals=ea.Scalars(tag)
        if vals: scalars.setdefault(tag,[]).extend(vals)

for tag in ("actor/distillation/loss","actor/grad_norm","actor/lr","training/global_step"):
    vals=scalars.get(tag,[])
    assert vals, (tag, sorted(scalars))
    assert any(int(x.step)==3732 for x in vals), (tag, vals[-3:])
    assert all(math.isfinite(float(x.value)) for x in vals), tag

g=scalars["training/global_step"]
v=[x for x in g if int(x.step)==3732]
assert v and float(v[-1].value)==3732.0, v[-3:]

print("PDS9952_OPD_FORMAL_POST_GATE=PASS")
PY
    local post=$?
    echo "OPD_POST_RC=$post"
    [[ "$post" -eq 0 ]] || return 44

    touch "$OPD_RUN/PASS"
    return 0
}


main() {
    cd "$PROJECT" || {
        echo "PDS9952_FORMAL=FAIL_CD_PROJECT"
        return 2
    }

    export PYTHONPATH="$PROJECT:$FORMAL_VERL:${PYTHONPATH:-}"
    export TOKENIZERS_PARALLELISM=false
    export PYTORCH_ALLOC_CONF=expandable_segments:True

    SEQ_CKPT="$(get_override "$SEQCFG" trainer.default_local_dir)" || return 3
    OPD_CKPT="$(get_override "$OPDCFG" trainer.default_local_dir)" || return 4
    SEQ_RUN="$(dirname "$SEQ_CKPT")"
    OPD_RUN="$(dirname "$OPD_CKPT")"

    echo "============================================================"
    echo "PDS9952 FORMAL TRAINING"
    echo "mode=$MODE"
    echo "seq_run=$SEQ_RUN"
    echo "opd_run=$OPD_RUN"
    date -u
    echo "============================================================"

    preflight
    PREFLIGHT_RC=$?
    echo "PREFLIGHT_RC=$PREFLIGHT_RC"
    if [[ "$PREFLIGHT_RC" -ne 0 ]]; then
        echo "PDS9952_FORMAL=BLOCKED_PREFLIGHT"
        return "$PREFLIGHT_RC"
    fi

    case "$MODE" in
        --preflight)
            echo "PDS9952_FORMAL_PREFLIGHT_ONLY=PASS"
            return 0
            ;;
        --seqkd)
            run_seqkd
            RC=$?
            echo "PDS9952_SEQKD_FORMAL_RC=$RC"
            return "$RC"
            ;;
        --opd)
            run_opd
            RC=$?
            echo "PDS9952_OPD_FORMAL_RC=$RC"
            return "$RC"
            ;;
        --chain)
            run_seqkd
            SEQ_RC=$?
            echo "PDS9952_SEQKD_FORMAL_RC=$SEQ_RC"

            if [[ "$SEQ_RC" -ne 0 ]]; then
                echo "PDS9952_FORMAL_CHAIN=STOP_AFTER_SEQKD_FAILURE"
                return "$SEQ_RC"
            fi

            run_opd
            OPD_RC=$?
            echo "PDS9952_OPD_FORMAL_RC=$OPD_RC"

            if [[ "$OPD_RC" -eq 0 ]]; then
                echo "PDS9952_FORMAL_CHAIN=TRAINING_PASS"
                return 0
            fi

            echo "PDS9952_FORMAL_CHAIN=STOP_AFTER_OPD_FAILURE"
            return "$OPD_RC"
            ;;
        *)
            echo "usage: $0 [--preflight|--seqkd|--opd|--chain]"
            return 64
            ;;
    esac
}

main "$@"
RC=$?
echo "PDS9952_FORMAL_SCRIPT_RC=$RC"
date -u
