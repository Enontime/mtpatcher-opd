#!/usr/bin/env bash
# PDS9952 endpoint validation v4.
#
# Scientific scope:
#   - shared step0 = common PE-OPD step4422 stage-entry model
#   - Arm B primary endpoint = SeqKD step3732
#   - Arm C primary endpoint = OPD step3732
#   - exact frozen validation3231 / shared matched20k validator
#   - deterministic validation path (val_kwargs: do_sample=false, temperature=0)
#
# This script does NOT select a checkpoint and does NOT alter training artifacts.

set +e
set +u
set +o pipefail 2>/dev/null

source /workspace/mtpatcher/project_env.sh

ROOT=/workspace/mtpatcher
PROJECT="$ROOT/repo/MT-Patcher-Reproduction-Ascend"
OVERLAY="$ROOT/envs/matched20k-v2-overlay"
FORMAL_VERL="$ROOT/repo/verl-v0.9.0-matched20k-v2-formal"
A3="$ROOT/envs/verl-v0.9.0-a3"
PY="$A3/bin/python"
RAY="$A3/bin/ray"

TEMPLATE="$PROJECT/configs/eval/matched20k_v2_common_validation_v1.yaml"
RUNNER="$PROJECT/scripts/matched20k_v2/run_opd.py"
TRAINER="$PROJECT/scripts/matched20k_v2/opd_trainer.py"
SCORER="$PROJECT/scripts/matched20k_v2/shared_mt_validation.py"

VAL_DATA="$ROOT/data/verl_science_broad20k/matched20k_v2"
VAL_JSONL="$VAL_DATA/validation3231.jsonl"
VAL_PARQUET="$VAL_DATA/validation3231.parquet"

STAGE="$ROOT/runs/science/pe_pds_v1/pds9952_stage_entry_v1/pe_opd_step4422_merged_hf"
SEQ_RUN="$ROOT/runs/science/pe_pds_v1/pds9952_seqkd_formal_v1"
OPD_RUN="$ROOT/runs/science/pe_pds_v1/pds9952_opd_formal_v1"

SEQ_ENDPOINT="$SEQ_RUN/checkpoints/global_step_3732/huggingface"
OPD_ACTOR="$OPD_RUN/checkpoints/global_step_3732/actor"

EVAL_ROOT="$ROOT/runs/science/pe_pds_v1/pds9952_endpoint_validation_v4"
MERGED_OPD="$EVAL_ROOT/model_cache/opd_step3732_merged_hf"
PROV="$EVAL_ROOT/provenance"

SCRIPT_SELF="$(readlink -f "$0" 2>/dev/null || printf '%s' "$0")"

clean_ray() {
    "$RAY" stop --force > /tmp/pds9952_endpoint_validation_ray_stop.log 2>&1 || true
    sleep 4

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

start_frozen_ray() {
    echo
    echo "===== START FROZEN RAY ====="

    clean_ray || return 1

    # Match the historical successful replay environment.
    source "$A3/bin/activate"
    hash -r

    local full_pp

    full_pp="$OVERLAY:$PROJECT:$FORMAL_VERL:/usr/local/Ascend/cann-9.1.0-beta.3/python/site-packages:/usr/local/Ascend/cann-9.1.0-beta.3/opp/built-in/op_impl/ai_core/tbe:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe"

    export PYTHONPATH="$full_pp"

    PYTHONPATH="$full_pp" \
    "$RAY" start \
        --head \
        --include-dashboard=false \
        --disable-usage-stats \
        > /tmp/pds9952_endpoint_validation_ray_head.log \
        2>&1

    local start_rc=$?

    echo "RAY_HEAD_START_RC=$start_rc"

    if [[ "$start_rc" -ne 0 ]]; then
        cat /tmp/pds9952_endpoint_validation_ray_head.log || true
        return 1
    fi

    sleep 4

    local raylet_pid

    raylet_pid="$(
        pgrep -x raylet |
        head -n1
    )"

    if [[ -z "$raylet_pid" ]]; then
        echo "RAYLET_MISSING"
        cat /tmp/pds9952_endpoint_validation_ray_head.log || true
        return 2
    fi

    echo "RAYLET_PID=$raylet_pid"

    echo "===== RAYLET ENV ====="

    tr '\0' '\n' \
        < "/proc/$raylet_pid/environ" |
        grep -E \
        '^(PATH|PYTHONPATH|VIRTUAL_ENV)=' \
        || true

    "$PY" - <<'PY_PRESTARTED_RAY_GATE'
import inspect

import ray

ray.init(
    address="auto",
)


@ray.remote
def probe():
    import inspect
    import sacrebleu
    import verl
    import scripts.matched20k_v2.opd_trainer as opd

    return {
        "sacrebleu_version":
            sacrebleu.__version__,

        "sacrebleu_file":
            inspect.getfile(sacrebleu),

        "verl_file":
            inspect.getfile(verl),

        "opd_file":
            inspect.getfile(opd),
    }


result = ray.get(
    probe.remote()
)

for key, value in result.items():
    print(
        f"PRESTARTED_RAY_{key.upper()} = {value}"
    )

assert (
    "/workspace/mtpatcher/envs/"
    "matched20k-v2-overlay/"
    in result["sacrebleu_file"]
)

assert (
    "/workspace/mtpatcher/repo/"
    "verl-v0.9.0-matched20k-v2-formal/"
    in result["verl_file"]
)

assert (
    "/workspace/mtpatcher/repo/"
    "MT-Patcher-Reproduction-Ascend/"
    in result["opd_file"]
)

ray.shutdown()

print(
    "PDS9952_PRESTARTED_RAY_IMPORT_GATE=PASS"
)
PY_PRESTARTED_RAY_GATE

    local probe_rc=$?

    if [[ "$probe_rc" -ne 0 ]]; then
        echo "PRESTARTED_RAY_IMPORT_GATE=FAIL"
        clean_ray
        return 3
    fi

    echo "PDS9952_FROZEN_RAY_HEAD=PASS"

    return 0
}

model_has_weights() {
    local model="$1"

    [[ -s "$model/config.json" ]] || return 1
    [[ -s "$model/tokenizer.json" ]] || return 1

    find "$model" \
        -maxdepth 1 \
        -type f \
        -name 'model*.safetensors' \
        -size +0c \
        -print \
        -quit 2>/dev/null |
        grep -q .
}

preflight() {
    echo "===== PREFLIGHT ====="

    local f
    for f in \
        "$TEMPLATE" \
        "$RUNNER" \
        "$TRAINER" \
        "$SCORER" \
        "$VAL_JSONL" \
        "$VAL_PARQUET"
    do
        if [[ ! -s "$f" ]]; then
            echo "MISSING=$f"
            return 10
        fi
    done

    if [[ ! -f "$SEQ_RUN/PASS" ]]; then
        echo "MISSING_SEQKD_PASS=$SEQ_RUN/PASS"
        return 11
    fi

    if [[ ! -f "$OPD_RUN/PASS" ]]; then
        echo "MISSING_OPD_PASS=$OPD_RUN/PASS"
        return 12
    fi

    model_has_weights "$STAGE" || {
        echo "BAD_STAGE_MODEL=$STAGE"
        return 13
    }

    model_has_weights "$SEQ_ENDPOINT" || {
        echo "BAD_SEQ_ENDPOINT=$SEQ_ENDPOINT"
        return 14
    }

    if ! find "$OPD_ACTOR" \
        -maxdepth 1 \
        -type f \
        -name 'model_world_size_*_rank_*.pt' \
        -size +0c \
        -print \
        -quit 2>/dev/null |
        grep -q .
    then
        echo "BAD_OPD_ACTOR=$OPD_ACTOR"
        return 15
    fi

    "$PY" - "$VAL_JSONL" "$VAL_PARQUET" "$TEMPLATE" "$STAGE" <<'PY_PREFLIGHT'
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml

jsonl = Path(sys.argv[1])
parquet = Path(sys.argv[2])
template = Path(sys.argv[3])
stage = str(Path(sys.argv[4]))

rows = [
    json.loads(line)
    for line in jsonl.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
assert len(rows) == 3231, len(rows)

counts = {}
for row in rows:
    key = row["eval_group"]
    counts[key] = counts.get(key, 0) + 1

expected = {
    "train_probe": 1024,
    "wmt24": 998,
    "flores": 1012,
    "challenge": 197,
}
assert counts == expected, (counts, expected)

table = pq.read_table(parquet)
assert table.num_rows == 3231, table.num_rows

cfg = yaml.safe_load(template.read_text(encoding="utf-8"))

assert cfg["data"]["val_files"] == [str(parquet)]
assert cfg["data"]["validation_shuffle"] is False
assert cfg["data"]["apply_chat_template_kwargs"]["enable_thinking"] is False
assert cfg["data"]["max_prompt_length"] == 1024
assert cfg["data"]["max_response_length"] == 256

val = cfg["actor_rollout_ref"]["rollout"]["val_kwargs"]
assert val["do_sample"] is False
assert val["temperature"] == 0
assert val["top_p"] == 1.0
assert val["top_k"] == -1
assert val["n"] == 1

assert cfg["actor_rollout_ref"]["rollout"]["full_determinism"] is True

assert cfg["trainer"]["val_only"] is True
assert cfg["trainer"]["val_before_train"] is True
assert cfg["trainer"]["resume_mode"] == "disable"
assert cfg["distillation"]["enabled"] is False

print("VALIDATION3231_COUNTS =", counts)
print("PDS9952_ENDPOINT_VALIDATION_SPEC_GATE=PASS")
PY_PREFLIGHT
    local rc=$?
    [[ "$rc" -eq 0 ]] || return 16

    local active
    active="$(
        ps -eo stat,pid,cmd |
        awk '$1 !~ /^Z/' |
        grep -E 'raylet|gcs_server|ray/dashboard|TaskRunnerV1|vLLMHttpServer|GlobalRequestLoadBalancer|scripts/matched20k_v2/run_opd.py' |
        grep -v grep || true
    )"

    if [[ -n "$active" ]]; then
        echo "ACTIVE_VALIDATOR_OR_RAY=BLOCKED"
        echo "$active"
        return 17
    fi

    echo "PDS9952_ENDPOINT_VALIDATION_PREFLIGHT=PASS"
    return 0
}

write_provenance() {
    mkdir -p "$PROV"

    {
        echo "timestamp_utc=$(date -u +%FT%TZ)"
        echo "project_head=$(git -C "$PROJECT" rev-parse HEAD)"
        echo "formal_verl_head=$(git -C "$FORMAL_VERL" rev-parse HEAD 2>/dev/null)"
        echo "stage=$STAGE"
        echo "seq_endpoint=$SEQ_ENDPOINT"
        echo "opd_actor=$OPD_ACTOR"
        echo "validation_jsonl=$VAL_JSONL"
        echo "validation_parquet=$VAL_PARQUET"
        echo
        echo "===== WORKTREE ====="
        git -C "$PROJECT" status --short
    } > "$PROV/source_state.txt"

    sha256sum \
        "$SCRIPT_SELF" \
        "$TEMPLATE" \
        "$RUNNER" \
        "$TRAINER" \
        "$SCORER" \
        "$VAL_JSONL" \
        "$VAL_PARQUET" \
        > "$PROV/evaluator_inputs.sha256"

    (
        cd "$STAGE" || return 1
        find . -maxdepth 1 -type f -name 'model*.safetensors' -print0 |
        sort -z |
        xargs -0 sha256sum
    ) > "$PROV/stage_model_files.sha256"

    (
        cd "$SEQ_ENDPOINT" || return 1
        find . -maxdepth 1 -type f -name 'model*.safetensors' -print0 |
        sort -z |
        xargs -0 sha256sum
    ) > "$PROV/seqkd_endpoint_model_files.sha256"

    (
        cd "$OPD_ACTOR" || return 1
        find . -maxdepth 1 -type f -name 'model_world_size_*_rank_*.pt' -print0 |
        sort -z |
        xargs -0 sha256sum
    ) > "$PROV/opd_endpoint_actor_shards.sha256"
}

merge_opd_endpoint() {
    echo
    echo "===== MERGE OPD ENDPOINT ====="

    if model_has_weights "$MERGED_OPD"; then
        echo "OPD_ENDPOINT_MERGE_ALREADY_COMPLETE=SKIP"
        return 0
    fi

    if [[ -e "$MERGED_OPD" ]]; then
        echo "PARTIAL_MERGED_OPD_EXISTS=BLOCKED"
        echo "$MERGED_OPD"
        return 20
    fi

    mkdir -p "$(dirname "$MERGED_OPD")"

    (
        cd "$FORMAL_VERL" || return 1
        "$PY" -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "$OPD_ACTOR" \
            --target_dir "$MERGED_OPD"
    ) > "$EVAL_ROOT/opd_endpoint_merge.log" 2>&1
    local rc=$?

    echo "OPD_ENDPOINT_MERGE_RC=$rc"

    if [[ "$rc" -ne 0 ]]; then
        tail -n 160 "$EVAL_ROOT/opd_endpoint_merge.log" || true
        return 21
    fi

    model_has_weights "$MERGED_OPD" || {
        echo "OPD_ENDPOINT_MERGE_OUTPUT_GATE=FAIL"
        return 22
    }

    (
        cd "$MERGED_OPD" || return 1
        find . -maxdepth 1 -type f -name 'model*.safetensors' -print0 |
        sort -z |
        xargs -0 sha256sum
    ) > "$PROV/opd_endpoint_merged_model_files.sha256"

    echo "PDS9952_OPD_ENDPOINT_MERGE=PASS"
    return 0
}

make_eval_config() {
    local arm="$1"
    local step="$2"
    local model="$3"
    local out="$4"
    local cfg="$5"

    "$PY" - \
        "$TEMPLATE" \
        "$cfg" \
        "$model" \
        "$step" \
        "$arm" \
        "$out" <<'PY_CONFIG'
import copy
import sys
from pathlib import Path

import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
model = str(Path(sys.argv[3]))
step = int(sys.argv[4])
arm = sys.argv[5]
out = Path(sys.argv[6])

cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
cfg = copy.deepcopy(cfg)

cfg["actor_rollout_ref"]["model"]["path"] = model
cfg["trainer"]["validation_step_override"] = step
cfg["trainer"]["project_name"] = "mtpatcher-pds9952-endpoint-validation"
cfg["trainer"]["experiment_name"] = f"pds9952-{arm}-endpoint-validation-v1"
cfg["trainer"]["validation_data_dir"] = str(out / "generations")
cfg["trainer"]["default_local_dir"] = str(out / "validator_checkpoints")

assert cfg["trainer"]["val_only"] is True
assert cfg["trainer"]["val_before_train"] is True
assert cfg["trainer"]["resume_mode"] == "disable"
assert cfg["distillation"]["enabled"] is False
assert cfg["data"]["validation_shuffle"] is False
assert cfg["data"]["apply_chat_template_kwargs"]["enable_thinking"] is False

val = cfg["actor_rollout_ref"]["rollout"]["val_kwargs"]
assert val["do_sample"] is False
assert val["temperature"] == 0
assert val["top_p"] == 1.0
assert val["top_k"] == -1
assert val["n"] == 1
assert cfg["actor_rollout_ref"]["rollout"]["full_determinism"] is True

dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(
    yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
PY_CONFIG
}

post_gate() {
    local arm="$1"
    local step="$2"
    local out="$3"

    "$PY" - "$arm" "$step" "$out" <<'PY_POST'
import glob
import json
import math
import re
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

arm = sys.argv[1]
step = int(sys.argv[2])
out = Path(sys.argv[3])

gen = out / "generations" / f"{step}.jsonl"
assert gen.is_file(), gen

rows = [
    json.loads(line)
    for line in gen.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
assert len(rows) == 3231, (arm, step, len(rows))

pattern = re.compile(
    r"<think>|</think>|<analysis>|</analysis>",
    flags=re.I,
)
marker_rows = [
    i
    for i, row in enumerate(rows)
    if pattern.search(str(row.get("output", "")))
]

events = [
    Path(p)
    for p in glob.glob(
        str(out / "tensorboard/**/events.out.tfevents.*"),
        recursive=True,
    )
]
events = [p for p in events if p.is_file() and p.stat().st_size > 0]
assert events, f"missing tensorboard events: {out}"

scalars = {}
for directory in sorted({p.parent for p in events}):
    ea = EventAccumulator(
        str(directory),
        size_guidance={"scalars": 0},
    )
    ea.Reload()

    for tag in ea.Tags().get("scalars", []):
        if not tag.startswith("compare/"):
            continue
        vals = ea.Scalars(tag)
        for item in vals:
            if int(item.step) == step:
                scalars[tag] = float(item.value)

assert len(scalars) == 10, (arm, step, sorted(scalars))
assert all(math.isfinite(x) for x in scalars.values())

payload = {
    "arm": arm,
    "step": step,
    "generation_rows": len(rows),
    "reasoning_marker_rows_observed": len(marker_rows),
    "reasoning_marker_first_indices": marker_rows[:20],
    "thinking_contract": (
        "enable_thinking=False is a chat-template request; "
        "structural marker occurrence is recorded, not used as a metric gate"
    ),
    "metrics": scalars,
}

status = out / "status.json"
status.write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print(f"{arm}_STEP_{step}_VALIDATION=PASS")
print("reasoning_marker_rows_observed =", len(marker_rows))
print(
    "macro_bleu =",
    scalars["compare/benchmark/macro_bleu"],
)
print(
    "macro_chrf =",
    scalars["compare/benchmark/macro_chrf"],
)
PY_POST
}

run_eval() {
    local arm="$1"
    local step="$2"
    local model="$3"

    local out="$EVAL_ROOT/$arm"
    local cfg="$out/config/step_${step}.yaml"
    local log="$out/validator.log"

    echo
    echo "============================================================"
    echo "VALIDATE arm=$arm step=$step"
    echo "model=$model"
    echo "============================================================"

    if [[ -s "$out/status.json" ]]; then
        echo "${arm}_STEP_${step}_STATUS_EXISTS=SKIP"
        cat "$out/status.json"
        return 0
    fi

    mkdir -p \
        "$out/generations" \
        "$out/tensorboard" \
        "$out/config" \
        "$out/validator_checkpoints"

    make_eval_config "$arm" "$step" "$model" "$out" "$cfg" || return 30

    export TENSORBOARD_DIR="$out/tensorboard"

    start_frozen_ray || return 34

    (
        cd "$PROJECT" || return 1
        "$PY" \
            scripts/matched20k_v2/run_opd.py \
            --config-path="$out/config" \
            --config-name="step_${step}" \
            "++ray_kwargs.ray_init.address=auto"
    ) > "$log" 2>&1
    local rc=$?

    echo "${arm}_STEP_${step}_VALIDATOR_RC=$rc"

    clean_ray
    local ray_rc=$?
    echo "${arm}_STEP_${step}_RAY_CLEAN_RC=$ray_rc"

    if [[ "$rc" -ne 0 ]]; then
        tail -n 200 "$log" || true
        return 31
    fi

    [[ "$ray_rc" -eq 0 ]] || return 32

    post_gate "$arm" "$step" "$out"
    local post_rc=$?

    echo "${arm}_STEP_${step}_POST_RC=$post_rc"
    [[ "$post_rc" -eq 0 ]] || {
        tail -n 200 "$log" || true
        return 33
    }

    return 0
}

write_summary() {
    "$PY" - "$EVAL_ROOT" <<'PY_SUMMARY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])

common = json.loads((root / "common_step0/status.json").read_text(encoding="utf-8"))
seq = json.loads((root / "seqkd_step3732/status.json").read_text(encoding="utf-8"))
opd = json.loads((root / "opd_step3732/status.json").read_text(encoding="utf-8"))

def metric(obj, key):
    return float(obj["metrics"][key])

metric_keys = [
    "compare/benchmark/wmt24_bleu",
    "compare/benchmark/wmt24_chrf",
    "compare/benchmark/flores_bleu",
    "compare/benchmark/flores_chrf",
    "compare/benchmark/challenge_bleu",
    "compare/benchmark/challenge_chrf",
    "compare/benchmark/macro_bleu",
    "compare/benchmark/macro_chrf",
]

delta = {
    key: metric(opd, key) - metric(seq, key)
    for key in metric_keys
}

summary = {
    "classification": "ADAPTATION",
    "experiment": "PDS9952 Parent-Diverse Low-Budget Adaptation",
    "comparison": (
        "local PDS-stage comparison after shared PE-OPD stage entry: "
        "fixed-target SeqKD vs online forward-KL OPD"
    ),
    "single_seed_descriptive_only": True,
    "primary_endpoint_step": 3732,
    "shared_step0": common,
    "seqkd_step3732": seq,
    "opd_step3732": opd,
    "opd_minus_seqkd": delta,
}

path = root / "endpoint_summary.json"
path.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print("============================================================")
print("PDS9952 PRIMARY ENDPOINT — DESCRIPTIVE SINGLE SEED")
print("============================================================")
print(
    "STEP0 macro BLEU/chrF =",
    metric(common, "compare/benchmark/macro_bleu"),
    metric(common, "compare/benchmark/macro_chrf"),
)
print(
    "SEQKD3732 macro BLEU/chrF =",
    metric(seq, "compare/benchmark/macro_bleu"),
    metric(seq, "compare/benchmark/macro_chrf"),
)
print(
    "OPD3732 macro BLEU/chrF =",
    metric(opd, "compare/benchmark/macro_bleu"),
    metric(opd, "compare/benchmark/macro_chrf"),
)
print(
    "OPD-SEQKD macro BLEU/chrF =",
    delta["compare/benchmark/macro_bleu"],
    delta["compare/benchmark/macro_chrf"],
)
print("SUMMARY =", path)
print("PDS9952_ENDPOINT_VALIDATION=PASS")
PY_SUMMARY
}

main() {
    cd "$PROJECT" || return 2

    export PYTHONPATH="$OVERLAY:$PROJECT:$FORMAL_VERL:${PYTHONPATH:-}"
    export VERL_CUSTOM_TRAINER_MODULE="scripts.matched20k_v2.opd_trainer"
    export MATCHED20K_VALIDATION_JSONL="$VAL_JSONL"
    export TOKENIZERS_PARALLELISM=false
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

    mkdir -p "$EVAL_ROOT"

    preflight || return $?
    write_provenance || return 40

    merge_opd_endpoint || return $?

    clean_ray || return 41

    run_eval "common_step0" 0 "$STAGE" || return $?
    run_eval "seqkd_step3732" 3732 "$SEQ_ENDPOINT" || return $?
    run_eval "opd_step3732" 3732 "$MERGED_OPD" || return $?

    write_summary || return 50

    touch "$EVAL_ROOT/PASS"

    echo "PDS9952_ENDPOINT_VALIDATION_CHAIN=PASS"
    return 0
}

main "$@"
RC=$?

echo "PDS9952_ENDPOINT_VALIDATION_SCRIPT_RC=$RC"
date -u

if [[ "$RC" -eq 0 ]]; then
    true
else
    false
fi
