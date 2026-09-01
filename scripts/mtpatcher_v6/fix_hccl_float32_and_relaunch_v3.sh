#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

TRAINER="$ROOT/scripts/mtpatcher_v6/train_correction_fkl_torchnpu_v1.py"
EVALUATOR="$ROOT/scripts/mtpatcher_v6/eval_correction_fkl_torchnpu_v1.py"
QUEUE="$ROOT/scripts/mtpatcher_v6/run_correction_fkl_night_queue_v1.sh"

PATCH_RUN="$RUN_ROOT/$EXP/corrfkl_patch_pe3732_v1"

PATCH_LOG="$LOG_ROOT/$EXP/corrfkl_patch_pe3732_v1.log"
QUEUE_LOG="$LOG_ROOT/$EXP/correction_fkl_night_queue_v1.log"

SMOKE="$ROOT/scripts/mtpatcher_v6/hccl_float32_allreduce_smoke.py"
SMOKE_LOG="$LOG_ROOT/$EXP/hccl_float32_allreduce_smoke_v3.log"

STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_ROOT/$EXP"


###############################################################################
# 1. VERIFY OLD FAILURE
###############################################################################

echo
echo "======================================================================"
echo "STAGE 1/7 — VERIFY HCCL kDouble ROOT CAUSE"
echo "======================================================================"

if [[ -f "$PATCH_LOG" ]]; then

    grep -E \
'HCCL allreduce|Unsupported data type|kDouble|EPOCH_1_COMPLETE|CHECKPOINT_SAVED|Traceback' \
    "$PATCH_LOG" \
    | tail -n 50 \
    || true

fi

if ! grep -q \
'Unsupported data type at::kDouble' \
"$PATCH_LOG" 2>/dev/null; then

    echo "WARNING: expected kDouble marker not found in old Patch log."

fi

echo
echo "HCCL_KDOUBLE_ROOT_CAUSE_CONFIRMED"


###############################################################################
# 2. PATCH ONLY THE EPOCH-METRIC COLLECTIVE DTYPE
###############################################################################

echo
echo "======================================================================"
echo "STAGE 2/7 — PATCH float64 -> float32 FOR EPOCH ALL_REDUCE"
echo "======================================================================"

export TRAINER

python - <<'PY'
import ast
import os
from pathlib import Path


path = Path(
    os.environ["TRAINER"]
)

text = path.read_text(
    encoding="utf-8"
)

tree = ast.parse(
    text
)


###############################################################################
# Locate:
#
# stats = torch.tensor(
#     [epoch_loss_sum, float(epoch_mask_tokens)],
#     dtype=torch.float64,
#     device=device,
# )
#
# Only this logging/aggregation tensor should change.
###############################################################################

matches = []


for node in ast.walk(
    tree
):

    if not isinstance(
        node,
        ast.Assign,
    ):
        continue

    if not any(
        isinstance(target, ast.Name)
        and target.id == "stats"
        for target in node.targets
    ):
        continue

    segment = (
        ast.get_source_segment(
            text,
            node,
        )
        or ""
    )

    if (
        "torch.tensor" in segment
        and "epoch_loss_sum" in segment
        and "epoch_mask_tokens" in segment
    ):

        matches.append(
            node
        )


print(
    "EPOCH_STATS_ASSIGNMENT_COUNT =",
    len(matches),
)


if len(matches) != 1:

    raise RuntimeError(
        "Expected exactly one epoch stats tensor"
    )


node = matches[0]

segment = (
    ast.get_source_segment(
        text,
        node,
    )
    or ""
)


print(
    "OLD_STATS_SOURCE =",
    segment.replace(
        "\n",
        " ",
    ),
)


if "torch.float32" in segment:

    print(
        "HCCL_FLOAT32_REPAIR_ALREADY_PRESENT"
    )


elif "torch.float64" in segment:

    repaired = segment.replace(
        "torch.float64",
        "torch.float32",
        1,
    )

    lines = text.splitlines()

    first_line = lines[
        node.lineno - 1
    ]

    indent = first_line[
        :len(first_line)
        - len(first_line.lstrip())
    ]


    repaired_lines = repaired.splitlines()


    # ast source segment starts at the first non-whitespace character.
    repaired_lines = [
        indent + line
        if i == 0
        else line
        for i, line in enumerate(
            repaired_lines
        )
    ]


    lines[
        node.lineno - 1:
        node.end_lineno
    ] = repaired_lines


    text = "\n".join(
        lines
    ) + "\n"


    ast.parse(
        text
    )

    path.write_text(
        text,
        encoding="utf-8",
    )


    print(
        "HCCL_FLOAT64_TO_FLOAT32_PATCH_APPLIED"
    )


else:

    raise RuntimeError(
        "Epoch stats tensor has neither float64 nor float32"
    )


###############################################################################
# Re-open and hard verify.
###############################################################################

final = path.read_text(
    encoding="utf-8"
)

tree2 = ast.parse(
    final
)


matches2 = []


for n in ast.walk(
    tree2
):

    if not isinstance(
        n,
        ast.Assign,
    ):
        continue

    if not any(
        isinstance(target, ast.Name)
        and target.id == "stats"
        for target in n.targets
    ):
        continue

    s = (
        ast.get_source_segment(
            final,
            n,
        )
        or ""
    )

    if (
        "epoch_loss_sum" in s
        and "epoch_mask_tokens" in s
    ):

        matches2.append(
            s
        )


if len(matches2) != 1:

    raise RuntimeError(
        "Post-repair stats audit failed"
    )


stats_source = matches2[0]


print(
    "NEW_STATS_SOURCE =",
    stats_source.replace(
        "\n",
        " ",
    ),
)


if "torch.float32" not in stats_source:

    raise RuntimeError(
        "Epoch stats tensor is not float32"
    )


if "torch.float64" in stats_source:

    raise RuntimeError(
        "float64 survived in epoch stats tensor"
    )


###############################################################################
# Verify training objective remains untouched.
###############################################################################

for marker in (
    "token_kl =",
    "t_prob",
    "t_logp",
    "s_logp",
    "loss = token_kl.mean()",
    "loss.backward()",
    "optimizer.step()",
):

    if marker not in final:

        raise RuntimeError(
            f"Training-objective marker missing: {marker}"
        )


print()
print(
    "TRAINING_OBJECTIVE_UNCHANGED_PASS"
)

print(
    "HCCL_EPOCH_STATS_FLOAT32_STATIC_PASS"
)
PY


python -m py_compile \
"$TRAINER" \
"$EVALUATOR"

bash -n \
"$QUEUE"

echo "POST_REPAIR_COMPILE_PASS"


###############################################################################
# 3. DIRECT HCCL FLOAT32 COLLECTIVE SMOKE TEST
###############################################################################

echo
echo "======================================================================"
echo "STAGE 3/7 — 16-NPU HCCL FLOAT32 ALL_REDUCE SMOKE"
echo "======================================================================"

cat > "$SMOKE" <<'PY'
#!/usr/bin/env python3

import os

import torch
import torch.distributed as dist
import torch_npu


local_rank = int(
    os.environ["LOCAL_RANK"]
)

rank = int(
    os.environ["RANK"]
)

world = int(
    os.environ["WORLD_SIZE"]
)


torch.npu.set_device(
    local_rank
)

device = torch.device(
    f"npu:{local_rank}"
)


dist.init_process_group(
    backend="hccl"
)


x = torch.tensor(
    [
        float(
            rank + 1
        )
    ],
    dtype=torch.float32,
    device=device,
)


dist.all_reduce(
    x,
    op=dist.ReduceOp.SUM,
)


expected = (
    world
    * (
        world + 1
    )
    / 2
)


actual = float(
    x.item()
)


if abs(
    actual - expected
) > 1.0e-4:

    raise RuntimeError(
        f"HCCL float32 sum mismatch: "
        f"actual={actual} expected={expected}"
    )


if rank == 0:

    print(
        "HCCL_FLOAT32_ALLREDUCE_AUDIT",
        {
            "world_size":
                world,

            "dtype":
                str(
                    x.dtype
                ),

            "actual_sum":
                actual,

            "expected_sum":
                expected,
        },
        flush=True,
    )

    print(
        "HCCL_FLOAT32_ALLREDUCE_PASS",
        flush=True,
    )


dist.barrier()

dist.destroy_process_group()
PY


python -m py_compile \
"$SMOKE"


python -m torch.distributed.run \
  --nproc_per_node=16 \
  --master_port=29687 \
  "$SMOKE" \
  > "$SMOKE_LOG" 2>&1


cat "$SMOKE_LOG"


grep -q \
'HCCL_FLOAT32_ALLREDUCE_PASS' \
"$SMOKE_LOG"

echo "HCCL_FLOAT32_COLLECTIVE_GATE_PASS"


###############################################################################
# 4. ENSURE OLD DEAD JOB IS GONE + PRESERVE FAILED ARTIFACTS
###############################################################################

echo
echo "======================================================================"
echo "STAGE 4/7 — PRESERVE FAILED RUN AND VERIFY CLEAN RELAUNCH"
echo "======================================================================"

ACTIVE_QUEUE="$(
    pgrep -af \
    '[r]un_correction_fkl_night_queue_v1.sh' \
    || true
)"


ACTIVE_TRAIN="$(
    pgrep -af \
    '[t]rain_correction_fkl_torchnpu_v1.py' \
    || true
)"


if [[ -n "$ACTIVE_QUEUE" ]] || [[ -n "$ACTIVE_TRAIN" ]]; then

    echo "Unexpected existing Correction-FKL process:"
    echo "$ACTIVE_QUEUE"
    echo "$ACTIVE_TRAIN"

    false

fi


if [[ -f "$PATCH_LOG" ]]; then

    cp -a \
      "$PATCH_LOG" \
      "${PATCH_LOG}.failed_hccl_double_${STAMP}"

    echo \
      "PRESERVED_FAILED_PATCH_LOG=${PATCH_LOG}.failed_hccl_double_${STAMP}"

fi


if [[ -f "$QUEUE_LOG" ]]; then

    cp -a \
      "$QUEUE_LOG" \
      "${QUEUE_LOG}.failed_hccl_double_${STAMP}"

    echo \
      "PRESERVED_FAILED_QUEUE_LOG=${QUEUE_LOG}.failed_hccl_double_${STAMP}"

fi


###############################################################################
# No checkpoint was saved in the failed run.
# If a partial output directory exists, preserve it rather than deleting it.
###############################################################################

if [[ -d "$PATCH_RUN" ]]; then

    if [[ -f "$PATCH_RUN/training_manifest.json" ]]; then

        echo "Unexpected completed Patch training manifest exists:"
        echo "$PATCH_RUN/training_manifest.json"

        false

    fi


    OLD_PATCH_RUN="${PATCH_RUN}.failed_hccl_double_${STAMP}"

    mv \
      "$PATCH_RUN" \
      "$OLD_PATCH_RUN"

    echo "PRESERVED_FAILED_PATCH_RUN=$OLD_PATCH_RUN"

fi


echo "FAILED_RUN_PRESERVATION_PASS"


###############################################################################
# 5. LAUNCH THE EXISTING FOUR-EXPERIMENT QUEUE AGAIN
###############################################################################

echo
echo "======================================================================"
echo "STAGE 5/7 — RELAUNCH FOUR-EXPERIMENT QUEUE"
echo "======================================================================"

nohup setsid bash "$QUEUE" \
  > "$QUEUE_LOG" 2>&1 < /dev/null &


QUEUE_PID=$!


echo "QUEUE_PID=$QUEUE_PID"
echo "QUEUE_LOG=$QUEUE_LOG"

echo "CORRECTION_FKL_QUEUE_RELAUNCHED_AFTER_HCCL_FIX"


###############################################################################
# 6. WAIT UNTIL WE PASS THE PREVIOUS FAILURE POINT
#
# Previous crash:
#   epoch 1 step 234/234
#     -> float64 dist.all_reduce
#
# New safety gate requires:
#   EPOCH_1_COMPLETE
#   CHECKPOINT_SAVED .../epoch1
###############################################################################

echo
echo "======================================================================"
echo "STAGE 6/7 — WAIT FOR REAL EPOCH-1 CHECKPOINT"
echo "======================================================================"

PATCH_LOG="$LOG_ROOT/$EXP/corrfkl_patch_pe3732_v1.log"

EPOCH1_PASS=0


for ROUND in \
    1 2 3 4 5 6 7 8 9 10 11 12
do

    sleep 20

    echo
    echo "HEALTH_ROUND=$ROUND"


    if [[ -f "$PATCH_LOG" ]]; then

        grep -E \
'CORRECTION_FKL_RUNTIME_AUDIT|CORRECTION_FKL_TRAINING_START|epoch=1 local_step=|EPOCH_1_COMPLETE|token_mean_correction_fkl|CHECKPOINT_SAVED|CORRECTION_FKL_TRAINING_PASS|HCCL|Traceback|RuntimeError|ERROR|ChildFailedError' \
        "$PATCH_LOG" \
        | tail -n 50 \
        || true


        if grep -q \
        'EPOCH_1_COMPLETE' \
        "$PATCH_LOG" \
        && grep -q \
        'CHECKPOINT_SAVED = .*corrfkl_patch_pe3732_v1/epoch1' \
        "$PATCH_LOG"; then

            EPOCH1_PASS=1

            break

        fi


        if grep -qE \
        'Traceback|RuntimeError|ChildFailedError' \
        "$PATCH_LOG"; then

            echo
            echo "NEW PATCH TRAINING ERROR DETECTED."

            tail -n 160 \
            "$PATCH_LOG"

            false

        fi

    fi


    QUEUE_PROC="$(
        pgrep -af \
        '[r]un_correction_fkl_night_queue_v1.sh' \
        || true
    )"


    TRAIN_PROC="$(
        pgrep -af \
        '[t]rain_correction_fkl_torchnpu_v1.py' \
        || true
    )"


    echo "QUEUE_ALIVE=$([[ -n "$QUEUE_PROC" ]] && echo YES || echo NO)"
    echo "TRAINER_ALIVE=$([[ -n "$TRAIN_PROC" ]] && echo YES || echo NO)"


    if [[ -z "$QUEUE_PROC" ]] && [[ -z "$TRAIN_PROC" ]]; then

        echo
        echo "Queue and trainer both stopped before Epoch-1 checkpoint."

        echo
        echo "QUEUE LOG:"
        tail -n 100 \
        "$QUEUE_LOG" \
        2>/dev/null \
        || true

        echo
        echo "PATCH LOG:"
        tail -n 160 \
        "$PATCH_LOG" \
        2>/dev/null \
        || true

        false

    fi

done


if [[ "$EPOCH1_PASS" -ne 1 ]]; then

    echo
    echo "Epoch-1 checkpoint gate timed out."

    echo
    echo "Current Patch log:"
    tail -n 120 \
      "$PATCH_LOG" \
      2>/dev/null \
      || true

    false

fi


echo
echo "EPOCH1_HCCL_FAILURE_POINT_CLEARED"
echo "PATCH_EPOCH1_CHECKPOINT_GATE_PASS"


###############################################################################
# 7. FINAL BACKGROUND HEALTH AUDIT
###############################################################################

echo
echo "======================================================================"
echo "STAGE 7/7 — FINAL UNATTENDED HEALTH AUDIT"
echo "======================================================================"

QUEUE_PROC="$(
    pgrep -af \
    '[r]un_correction_fkl_night_queue_v1.sh' \
    || true
)"


TRAIN_PROC="$(
    pgrep -af \
    '[t]rain_correction_fkl_torchnpu_v1.py' \
    || true
)"


echo
echo "QUEUE PROCESS:"
echo "$QUEUE_PROC"

echo
echo "TRAIN PROCESS:"
echo "$TRAIN_PROC"


if [[ -z "$QUEUE_PROC" ]]; then

    echo "Night queue is no longer alive."

    false

fi


if [[ -z "$TRAIN_PROC" ]]; then

    echo "Correction-FKL trainer is not alive."

    false

fi


echo
echo "======================================================================"
echo "HCCL FIX + LONG RUN VERIFIED"
echo "======================================================================"

echo
echo "HCCL_FLOAT32_COLLECTIVE_GATE_PASS"
echo "EPOCH1_HCCL_FAILURE_POINT_CLEARED"
echo "PATCH_EPOCH1_CHECKPOINT_GATE_PASS"
echo "CORRECTION_FKL_LONG_QUEUE_RUNNING"
echo
echo "Queued:"
echo "  1. Patch-FKL"
echo "  2. Random equal-count FKL"
echo "  3. Halo1-FKL"
echo "  4. Full correction-trajectory FKL"
echo
echo "QUEUE_LOG=$QUEUE_LOG"
echo "PATCH_LOG=$PATCH_LOG"
echo
echo "LONG_JOB_VERIFIED_SAFE_TO_LEAVE"

