#!/usr/bin/env bash
# PDS9952 v1 scientific contract — formal training launcher v2.
#
# Arm B: PDS9952 + fresh-Qwen3-8B fixed target + canonical response-only SeqKD
# Arm C: exact same PDS9952 source schedule + current Student rollout +
#        frozen Qwen3-8B top-k distribution + canonical forward-KL OPD
#
# This launcher performs training only. Validation replay is post-hoc over the
# frozen checkpoint cadence 0,100,...,3700,3732.

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

EXPECTED_POP="1491babf8cf2625fcf275cfee1d4a812a0f8989aad1bd6f0d024f5240b86b937"
EXPECTED_TEACHER="7c5295563d4db9f201f223a4da1c309a28a59e6197dd081ce319e5f999f6635d"
EXPECTED_ORDER="eb6473ac6fd09db7171107f2fcd4d89b65aa909bd5a1312a2c6dfb93e9d3dfa2"
EXPECTED_SEQ="76bdfe71c0125ab4401564a349cedf06e54078a115b590afc91003636df94dca"
EXPECTED_OPD="6986a1b1da8738c73d7fa555db9bafe075d67d3db340f51168705b99c39ed9e6"

MODE="${1:---preflight}"
SCRIPT_SELF="$(readlink -f "$0" 2>/dev/null || printf '%s' "$0")"

get_override() {
    local cfg="$1"
    local key="$2"

    "$PY" - "$cfg" "$key" <<'PY_GET_OVERRIDE'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
key = sys.argv[2]

hits = []
for item in cfg["overrides"]:
    text = str(item)
    if text.startswith(key + "="):
        hits.append(text.split("=", 1)[1])

if len(hits) != 1:
    raise SystemExit(
        f"expected exactly one {key}= override; got {hits}"
    )

print(hits[0].strip("'\""))
PY_GET_OVERRIDE
}

preflight() {
    echo
    echo "===== PREFLIGHT ====="

    local v
    for v in \
        EXPECTED_POP \
        EXPECTED_TEACHER \
        EXPECTED_ORDER \
        EXPECTED_SEQ \
        EXPECTED_OPD
    do
        if [[ -z "${!v:-}" ]]; then
            echo "MISSING_FROZEN_CONSTANT=$v"
            return 9
        fi
    done

    local f
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
    echo "POP_SHA actual=$got expected=$EXPECTED_POP"
    [[ "$got" == "$EXPECTED_POP" ]] || return 13

    got="$(sha256sum "$DATA/teacher_targets_pds9952_qwen3_8b_v1.jsonl" | awk '{print $1}')"
    echo "TEACHER_SHA actual=$got expected=$EXPECTED_TEACHER"
    [[ "$got" == "$EXPECTED_TEACHER" ]] || return 14

    got="$(sha256sum "$DATA/source_order_manifest.jsonl" | awk '{print $1}')"
    echo "ORDER_SHA actual=$got expected=$EXPECTED_ORDER"
    [[ "$got" == "$EXPECTED_ORDER" ]] || return 15

    got="$(sha256sum "$DATA/pds9952_seqkd_matched59712.parquet" | awk '{print $1}')"
    echo "SEQ_PARQUET_SHA actual=$got expected=$EXPECTED_SEQ"
    [[ "$got" == "$EXPECTED_SEQ" ]] || return 16

    got="$(sha256sum "$DATA/pds9952_opd_matched59712.parquet" | awk '{print $1}')"
    echo "OPD_PARQUET_SHA actual=$got expected=$EXPECTED_OPD"
    [[ "$got" == "$EXPECTED_OPD" ]] || return 17

    "$PY" - \
        "$SEQCFG" \
        "$OPDCFG" \
        "$PROTOCOL" \
        "$ASSET" \
        "$STAGE" <<'PY_PREFLIGHT'
import json
import math
import sys
from pathlib import Path

import yaml

seq_path = Path(sys.argv[1])
opd_path = Path(sys.argv[2])
protocol_path = Path(sys.argv[3])
asset_path = Path(sys.argv[4])
stage = str(Path(sys.argv[5]))

seq = yaml.safe_load(seq_path.read_text(encoding="utf-8"))
opd = yaml.safe_load(opd_path.read_text(encoding="utf-8"))
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
asset = json.loads(asset_path.read_text(encoding="utf-8"))

failures = []


def override(cfg, key):
    hits = [
        str(item).split("=", 1)[1]
        for item in cfg["overrides"]
        if str(item).startswith(key + "=")
    ]
    if len(hits) != 1:
        failures.append(
            {
                "field": key,
                "actual": hits,
                "expected": "exactly one override",
            }
        )
        return None
    return hits[0].strip("'\"")


def check(label, actual, expected):
    ok = actual == expected
    print(
        f"{label}: actual={actual!r} "
        f"expected={expected!r} "
        f"{'PASS' if ok else 'FAIL'}"
    )
    if not ok:
        failures.append(
            {
                "field": label,
                "actual": actual,
                "expected": expected,
            }
        )


def check_int(label, actual, expected):
    try:
        parsed = int(actual)
    except Exception:
        parsed = actual
    check(label, parsed, expected)


def check_float(label, actual, expected):
    try:
        parsed = float(actual)
        ok = math.isclose(
            parsed,
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    except Exception:
        parsed = actual
        ok = False

    print(
        f"{label}: actual={actual!r} "
        f"parsed={parsed!r} "
        f"expected={expected!r} "
        f"{'PASS' if ok else 'FAIL'}"
    )

    if not ok:
        failures.append(
            {
                "field": label,
                "actual": actual,
                "parsed": parsed,
                "expected": expected,
            }
        )


check_int(
    "seq.total_training_steps",
    override(seq, "trainer.total_training_steps"),
    3732,
)
check_int(
    "opd.total_training_steps",
    override(opd, "trainer.total_training_steps"),
    3732,
)

check_int(
    "seq.save_freq",
    override(seq, "trainer.save_freq"),
    100,
)
check_int(
    "opd.save_freq",
    override(opd, "trainer.save_freq"),
    100,
)

check(
    "seq.resume_mode",
    override(seq, "trainer.resume_mode"),
    "disable",
)
check(
    "opd.resume_mode",
    override(opd, "trainer.resume_mode"),
    "disable",
)

check(
    "seq.model.path",
    override(seq, "model.path"),
    stage,
)
check(
    "opd.model.path",
    override(opd, "actor_rollout_ref.model.path"),
    stage,
)

seq_train = override(seq, "data.train_files")
opd_train = override(opd, "data.train_files")

seq_train_ok = (
    seq_train is not None
    and seq_train.endswith(
        "pds9952_seqkd_matched59712.parquet"
    )
)
print(
    f"seq.train_files: actual={seq_train!r} "
    f"{'PASS' if seq_train_ok else 'FAIL'}"
)
if not seq_train_ok:
    failures.append(
        {
            "field": "seq.train_files",
            "actual": seq_train,
        }
    )

opd_train_ok = (
    opd_train is not None
    and "pds9952_opd_matched59712.parquet"
    in opd_train
)
print(
    f"opd.train_files: actual={opd_train!r} "
    f"{'PASS' if opd_train_ok else 'FAIL'}"
)
if not opd_train_ok:
    failures.append(
        {
            "field": "opd.train_files",
            "actual": opd_train,
        }
    )

check_int(
    "seq.train_batch_size",
    override(seq, "data.train_batch_size"),
    16,
)
check_int(
    "opd.train_batch_size",
    override(opd, "data.train_batch_size"),
    16,
)

check_float(
    "seq.lr",
    override(seq, "optim.lr"),
    2e-5,
)
check(
    "seq.lr_scheduler_type",
    override(seq, "optim.lr_scheduler_type"),
    "cosine",
)
check_float(
    "seq.lr_warmup_steps_ratio",
    override(seq, "optim.lr_warmup_steps_ratio"),
    0.03,
)
check_float(
    "opd.lr",
    override(opd, "actor_rollout_ref.actor.optim.lr"),
    1e-6,
)

check(
    "asset.matched_gate.status",
    asset["matched_gate"]["status"],
    "PASS",
)
check(
    "asset.source_order_equal",
    asset["matched_gate"][
        "per_exposure_source_order_equal"
    ],
    True,
)
check(
    "asset.prompt_equal",
    asset["matched_gate"][
        "per_exposure_prompt_equal"
    ],
    True,
)
check_int(
    "asset.terminal_global_step",
    asset["matched_gate"]["terminal_global_step"],
    3732,
)
check_int(
    "asset.total_rows",
    asset["schedule"]["total_rows"],
    59712,
)
check_int(
    "asset.total_optimizer_steps",
    asset["schedule"]["total_optimizer_steps"],
    3732,
)

check_int(
    "protocol.primary_endpoint_step",
    protocol["primary_endpoint_step"],
    3732,
)
check_int(
    "protocol.optimizer_steps",
    protocol["budget"]["optimizer_steps"],
    3732,
)
check_int(
    "protocol.source_exposures",
    protocol["budget"]["source_exposures"],
    59712,
)

if failures:
    print()
    print("PDS9952_FORMAL_SPEC_GATE=FAIL")
    print(
        json.dumps(
            failures,
            ensure_ascii=False,
            indent=2,
        )
    )
    raise SystemExit(1)

print()
print("PDS9952_FORMAL_SPEC_GATE=PASS")
print(
    "SeqKD warmup steps (floor informational) =",
    int(0.03 * 3732),
)
PY_PREFLIGHT
    local rc=$?
    [[ "$rc" -eq 0 ]] || return 18

    local active
    active="$(
        ps -eo stat,pid,cmd |
        awk '$1 !~ /^Z/' |
        grep -E \
'raylet|gcs_server|ray/dashboard|TaskRunnerV1|vLLMHttpServer|GlobalRequestLoadBalancer|verl.trainer.sft_trainer|verl.trainer.main_ppo' |
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
        echo "launcher=$SCRIPT_SELF"
        echo
        echo "===== WORKTREE ====="
        git status --short
    } > "$run/provenance/source_state.txt"

    sha256sum \
        "$SCRIPT_SELF" \
        > "$run/provenance/launcher.sha256"

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
       find "$run/checkpoints" \
           -maxdepth 1 \
           -type d \
           -name 'global_step_*' \
           -print \
           -quit 2>/dev/null |
           grep -q .
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
        "$PY" - "$SEQCFG" <<'PY_SEQ_OVERRIDES'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
for item in cfg["overrides"]:
    print(item)
PY_SEQ_OVERRIDES
    )

    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    export TENSORBOARD_DIR="$SEQ_RUN/tensorboard"

    (
        cd "$SEQ_RUN" || return 1
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

    "$PY" - "$SEQ_RUN" <<'PY_SEQ_POST'
import glob
import math
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = Path(sys.argv[1])

final = run / "checkpoints/global_step_3732/huggingface"
assert final.is_dir(), final
assert list(final.glob("model*.safetensors")), final

expected = list(range(100, 3701, 100)) + [3732]
actual = set()

for path in (run / "checkpoints").glob("global_step_*"):
    try:
        actual.add(int(path.name.rsplit("_", 1)[1]))
    except Exception:
        pass

missing = [step for step in expected if step not in actual]
assert not missing, f"missing SeqKD checkpoints {missing}"

events = [
    Path(p)
    for p in glob.glob(
        str(run / "tensorboard/**/events.out.tfevents.*"),
        recursive=True,
    )
]
events = [
    p for p in events
    if p.is_file() and p.stat().st_size > 0
]
assert events, "missing SeqKD TensorBoard"

scalars = {}
for directory in sorted({p.parent for p in events}):
    ea = EventAccumulator(
        str(directory),
        size_guidance={"scalars": 0},
    )
    ea.Reload()

    for tag in ea.Tags().get("scalars", []):
        vals = ea.Scalars(tag)
        if vals:
            scalars.setdefault(tag, []).extend(vals)

for tag in ("train/loss", "train/grad_norm", "train/lr"):
    vals = scalars.get(tag, [])
    assert vals, (tag, sorted(scalars))
    assert any(
        int(item.step) == 3732
        for item in vals
    ), (tag, vals[-3:])
    assert all(
        math.isfinite(float(item.value))
        for item in vals
    ), tag

print("PDS9952_SEQKD_FORMAL_POST_GATE=PASS")
PY_SEQ_POST
    local post=$?

    echo "SEQKD_POST_RC=$post"
    [[ "$post" -eq 0 ]] || return 32

    touch "$SEQ_RUN/PASS"
    return 0
}

clean_ray() {
    "$A3/bin/ray" stop --force \
        > /tmp/pds9952_formal_ray_stop.log \
        2>&1 || true

    sleep 5

    local active
    active="$(
        ps -eo stat,pid,cmd |
        awk '$1 !~ /^Z/' |
        grep -E \
'raylet|gcs_server|ray/dashboard|TaskRunnerV1|vLLMHttpServer|GlobalRequestLoadBalancer' |
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
        "$PY" - "$OPDCFG" <<'PY_OPD_OVERRIDES'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
for item in cfg["overrides"]:
    print(item)
PY_OPD_OVERRIDES
    )

    unset VERL_CUSTOM_TRAINER_MODULE
    unset MATCHED20K_VALIDATION_JSONL
    unset MATCHED20K_REPLAY_STEPS
    unset MATCHED20K_REPLAY_CHECKPOINT_ROOT
    unset RAY_ADDRESS

    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    export TENSORBOARD_DIR="$OPD_RUN/tensorboard"

    (
        cd "$OPD_RUN" || return 1
        "$PY" \
            -m verl.trainer.main_ppo \
            "${OVERRIDES[@]}"
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

    "$PY" - "$OPD_RUN" <<'PY_OPD_POST'
import glob
import math
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = Path(sys.argv[1])

final = run / "checkpoints/global_step_3732/actor"
assert final.is_dir(), final
assert list(
    final.glob("model_world_size_*_rank_*.pt")
), final

expected = list(range(100, 3701, 100)) + [3732]
actual = set()

for path in (run / "checkpoints").glob("global_step_*"):
    try:
        actual.add(int(path.name.rsplit("_", 1)[1]))
    except Exception:
        pass

missing = [step for step in expected if step not in actual]
assert not missing, f"missing OPD checkpoints {missing}"

events = [
    Path(p)
    for p in glob.glob(
        str(run / "tensorboard/**/events.out.tfevents.*"),
        recursive=True,
    )
]
events = [
    p for p in events
    if p.is_file() and p.stat().st_size > 0
]
assert events, "missing OPD TensorBoard"

scalars = {}
for directory in sorted({p.parent for p in events}):
    ea = EventAccumulator(
        str(directory),
        size_guidance={"scalars": 0},
    )
    ea.Reload()

    for tag in ea.Tags().get("scalars", []):
        vals = ea.Scalars(tag)
        if vals:
            scalars.setdefault(tag, []).extend(vals)

for tag in (
    "actor/distillation/loss",
    "actor/grad_norm",
    "actor/lr",
    "training/global_step",
):
    vals = scalars.get(tag, [])
    assert vals, (tag, sorted(scalars))
    assert any(
        int(item.step) == 3732
        for item in vals
    ), (tag, vals[-3:])
    assert all(
        math.isfinite(float(item.value))
        for item in vals
    ), tag

global_steps = scalars["training/global_step"]
endpoint = [
    item
    for item in global_steps
    if int(item.step) == 3732
]
assert endpoint
assert float(endpoint[-1].value) == 3732.0

print("PDS9952_OPD_FORMAL_POST_GATE=PASS")
PY_OPD_POST
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

    SEQ_CKPT="$(
        get_override \
            "$SEQCFG" \
            trainer.default_local_dir
    )" || return 3

    OPD_CKPT="$(
        get_override \
            "$OPDCFG" \
            trainer.default_local_dir
    )" || return 4

    SEQ_RUN="$(dirname "$SEQ_CKPT")"
    OPD_RUN="$(dirname "$OPD_CKPT")"

    echo "============================================================"
    echo "PDS9952 FORMAL TRAINING"
    echo "launcher_version=v2"
    echo "mode=$MODE"
    echo "seq_run=$SEQ_RUN"
    echo "opd_run=$OPD_RUN"
    date -u
    echo "============================================================"

    preflight
    local preflight_rc=$?

    echo "PREFLIGHT_RC=$preflight_rc"

    if [[ "$preflight_rc" -ne 0 ]]; then
        echo "PDS9952_FORMAL=BLOCKED_PREFLIGHT"
        return "$preflight_rc"
    fi

    case "$MODE" in
        --preflight)
            echo "PDS9952_FORMAL_PREFLIGHT_ONLY=PASS"
            return 0
            ;;

        --seqkd)
            run_seqkd
            local seq_rc=$?
            echo "PDS9952_SEQKD_FORMAL_RC=$seq_rc"
            return "$seq_rc"
            ;;

        --opd)
            run_opd
            local opd_rc=$?
            echo "PDS9952_OPD_FORMAL_RC=$opd_rc"
            return "$opd_rc"
            ;;

        --chain)
            run_seqkd
            local seq_rc=$?
            echo "PDS9952_SEQKD_FORMAL_RC=$seq_rc"

            if [[ "$seq_rc" -ne 0 ]]; then
                echo "PDS9952_FORMAL_CHAIN=STOP_AFTER_SEQKD_FAILURE"
                return "$seq_rc"
            fi

            run_opd
            local opd_rc=$?
            echo "PDS9952_OPD_FORMAL_RC=$opd_rc"

            if [[ "$opd_rc" -eq 0 ]]; then
                echo "PDS9952_FORMAL_CHAIN=TRAINING_PASS"
                return 0
            fi

            echo "PDS9952_FORMAL_CHAIN=STOP_AFTER_OPD_FAILURE"
            return "$opd_rc"
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

if [[ "$RC" -eq 0 ]]; then
    true
else
    false
fi
