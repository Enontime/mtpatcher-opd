#!/usr/bin/env bash

set +e
set +u
set +o pipefail 2>/dev/null

ROOT=/workspace/mtpatcher

PROJECT="$ROOT/repo/MT-Patcher-Reproduction-Ascend"
FORMAL_VERL="$ROOT/repo/verl-v0.9.0-matched20k-v2-formal"

VERL_PY="$ROOT/envs/verl-v0.9.0-a3/bin/python"
OVERLAY="$ROOT/envs/matched20k-v2-overlay"

RUN="$ROOT/runs/science/matched20k_v2/opd_formal_v1"

TEMPLATE="$PROJECT/configs/eval/matched20k_v2_common_validation_v1.yaml"

VALIDATION_JSONL="$ROOT/data/verl_science_broad20k/matched20k_v2/validation3231.jsonl"

VAL_ROOT="$RUN/validation"
GEN_DIR="$VAL_ROOT/generations"
LOG_DIR="$VAL_ROOT/logs"
CFG_DIR="$VAL_ROOT/configs"
STATUS_DIR="$VAL_ROOT/status"

TB_DIR="$RUN/tensorboard"

mkdir -p \
  "$GEN_DIR" \
  "$LOG_DIR" \
  "$CFG_DIR" \
  "$STATUS_DIR"

cd "$PROJECT" || return 1

export PYTHONPATH="$OVERLAY:$PROJECT:$FORMAL_VERL:${PYTHONPATH:-}"

export VERL_CUSTOM_TRAINER_MODULE="scripts.matched20k_v2.opd_trainer"

export MATCHED20K_VALIDATION_JSONL="$VALIDATION_JSONL"

export TENSORBOARD_DIR="$TB_DIR"


check_step() {
    STEP="$1"

    GEN="$GEN_DIR/$STEP.jsonl"

    MARKER=$(printf \
      "%s/step_%04d.json" \
      "$STATUS_DIR" \
      "$STEP"
    )

    "$VERL_PY" - \
      "$GEN" \
      "$TB_DIR" \
      "$STEP" \
      "$MARKER" <<'PY'
from pathlib import Path
import json
import re
import sys

from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

gen = Path(sys.argv[1])
tb = Path(sys.argv[2])
step = int(sys.argv[3])
marker = Path(sys.argv[4])

if not gen.is_file():
    raise RuntimeError(
        f"missing generations: {gen}"
    )

rows = [
    json.loads(line)
    for line in gen.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

assert len(rows) == 3231, (
    step,
    len(rows),
)

pattern = re.compile(
    r"<think>|</think>|<analysis>|</analysis>",
    flags=re.I,
)

bad = [
    i
    for i, row in enumerate(rows)
    if pattern.search(
        str(row.get("output", ""))
    )
]

assert not bad, (
    step,
    bad[:20],
)

ea = EventAccumulator(
    str(tb)
)

ea.Reload()

tags = sorted(
    tag
    for tag in ea.Tags()["scalars"]
    if tag.startswith("compare/")
)

assert len(tags) == 10, (
    step,
    tags,
)

metrics = {}

for tag in tags:
    values = [
        x
        for x in ea.Scalars(tag)
        if x.step == step
    ]

    assert values, (
        step,
        tag,
    )

    metrics[tag] = values[-1].value

payload = {
    "step": step,
    "generation_rows": 3231,
    "reasoning_marker_rows": 0,
    "metrics": metrics,
}

marker.write_text(
    json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)

print(
    f"VALIDATION_STEP_{step}=PASS"
)

print(
    "MACRO_BLEU =",
    metrics[
        "compare/benchmark/macro_bleu"
    ],
)

print(
    "MACRO_CHRF =",
    metrics[
        "compare/benchmark/macro_chrf"
    ],
)
PY
}


FAILED=0

for STEP in $(seq 0 100 7500)
do
    echo
    echo "============================================================"
    echo "VALIDATION STEP $STEP"
    echo "============================================================"

    if check_step "$STEP" \
        > "/tmp/matched20k_opd_check_${STEP}.log" \
        2>&1
    then
        echo "STEP_${STEP}_ALREADY_COMPLETE=SKIP"
        cat "/tmp/matched20k_opd_check_${STEP}.log"
        continue
    fi

    if [[ "$STEP" -eq 0 ]]; then
        MODEL="$ROOT/models/Qwen3-0.6B"
    else
        MODEL="$RUN/checkpoints/global_step_${STEP}/actor/huggingface"
    fi

    if [[ ! -s "$MODEL/config.json" ]] || \
       [[ ! -s "$MODEL/tokenizer.json" ]]; then
        echo "STEP_${STEP}_MODEL_MISSING=FAIL"
        FAILED=1
        break
    fi

    if [[ "$STEP" -ne 0 ]]; then
        if ! find "$MODEL" \
            -maxdepth 1 \
            -type f \
            -name 'model*.safetensors' \
            -size +0c \
            -print \
            -quit \
            | grep -q .
        then
            echo "STEP_${STEP}_WEIGHTS_MISSING=FAIL"
            FAILED=1
            break
        fi
    fi

    CFG_NAME=$(printf \
      "step_%04d" \
      "$STEP"
    )

    CFG="$CFG_DIR/${CFG_NAME}.yaml"

    "$VERL_PY" - \
      "$TEMPLATE" \
      "$CFG" \
      "$MODEL" \
      "$STEP" \
      "$RUN" <<'PY'
from pathlib import Path
import copy
import sys
import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
model = str(Path(sys.argv[3]))
step = int(sys.argv[4])
run = Path(sys.argv[5])

a = yaml.safe_load(
    src.read_text(
        encoding="utf-8"
    )
)

b = copy.deepcopy(a)

b["actor_rollout_ref"]["model"]["path"] = model

b["trainer"]["validation_step_override"] = step

b["trainer"]["experiment_name"] = (
    "matched20k-v2-opd-validation-replay-v1"
)

b["trainer"]["validation_data_dir"] = str(
    run / "validation/generations"
)

b["trainer"]["default_local_dir"] = str(
    run / "validation/validator_checkpoints"
)

assert b["trainer"]["val_only"] is True
assert b["trainer"]["val_before_train"] is True
assert b["trainer"]["resume_mode"] == "disable"

assert b["distillation"]["enabled"] is False
assert b["data"]["validation_shuffle"] is False

assert (
    b["actor_rollout_ref"]["rollout"][
        "full_determinism"
    ]
    is True
)

dst.write_text(
    yaml.safe_dump(
        b,
        sort_keys=False,
        allow_unicode=True,
    ),
    encoding="utf-8",
)
PY

    LOG=$(printf \
      "%s/step_%04d.log" \
      "$LOG_DIR" \
      "$STEP"
    )

    "$VERL_PY" \
      scripts/matched20k_v2/run_opd.py \
      --config-path="$CFG_DIR" \
      --config-name="$CFG_NAME" \
      > "$LOG" 2>&1

    RC=$?

    echo "STEP_${STEP}_VALIDATOR_RC=$RC"

    if [[ "$RC" -ne 0 ]]; then
        tail -n 160 "$LOG"
        FAILED=1
        break
    fi

    check_step "$STEP"

    RC=$?

    if [[ "$RC" -ne 0 ]]; then
        echo "STEP_${STEP}_OUTPUT_GATE=FAIL"
        FAILED=1
        break
    fi

    echo "STEP_${STEP}_COMPLETE=PASS"
done

echo

if [[ "$FAILED" -eq 0 ]]; then
    echo "OPD_VALIDATION_REPLAY_V1=PASS"
else
    echo "OPD_VALIDATION_REPLAY_V1=FAIL"
fi
