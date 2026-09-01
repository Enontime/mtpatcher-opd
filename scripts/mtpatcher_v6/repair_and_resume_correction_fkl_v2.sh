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

DATA="$DATA_ROOT/$EXP/patch_aware_k1_clean3732_v2.jsonl"

STUDENT="$MODEL_ROOT/Qwen3-0.6B"
TEACHER="$MODEL_ROOT/Qwen3-8B"

PATCH_PREFLIGHT_LOG="$LOG_ROOT/$EXP/corrfkl_patch_preflight_v2.log"
FULL_PREFLIGHT_LOG="$LOG_ROOT/$EXP/corrfkl_full_preflight_v2.log"

QUEUE_LOG="$LOG_ROOT/$EXP/correction_fkl_night_queue_v1.log"
PATCH_TRAIN_LOG="$LOG_ROOT/$EXP/corrfkl_patch_pe3732_v1.log"

mkdir -p "$LOG_ROOT/$EXP"


###############################################################################
# STAGE 1 — SHOW WHY V1 STOPPED
###############################################################################

echo
echo "======================================================================"
echo "STAGE 1/7 — OLD PREFLIGHT DIAGNOSIS"
echo "======================================================================"

OLD="$LOG_ROOT/$EXP/corrfkl_patch_preflight_v1.log"

if [[ -f "$OLD" ]]; then

    echo "OLD_PREFLIGHT_LOG=$OLD"

    grep -E \
'vocab|tokenizer|Traceback|RuntimeError|Error|ERROR|CORRECTION_FKL' \
    "$OLD" \
    | tail -n 80 \
    || true

else

    echo "Old preflight log absent; continuing with source-level repair."

fi


###############################################################################
# STAGE 2 — REMOVE INVALID tokenizer-length ASSERTION
###############################################################################

echo
echo "======================================================================"
echo "STAGE 2/7 — REPAIR QWEN3 PADDED VOCAB CHECK"
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
# Find:
#
# if int(student_raw.config.vocab_size) != len(tokenizer):
#     raise RuntimeError("tokenizer/model vocab mismatch")
#
# This check is invalid for Qwen3 because its LM-head/embedding vocabulary is
# padded beyond len(tokenizer).
###############################################################################

bad_nodes = []


for node in ast.walk(
    tree
):

    if not isinstance(
        node,
        ast.If,
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
        "student_raw.config.vocab_size" in segment
        and
        "len(" in segment
        and
        "tokenizer" in segment
        and
        "tokenizer/model vocab mismatch" in segment
    ):

        bad_nodes.append(
            node
        )


if len(
    bad_nodes
) > 1:

    raise RuntimeError(
        f"Found multiple invalid vocab assertions: "
        f"{len(bad_nodes)}"
    )


if len(
    bad_nodes
) == 1:

    node = bad_nodes[
        0
    ]

    lines = text.splitlines()

    original_first = lines[
        node.lineno - 1
    ]

    indent = original_first[
        :len(original_first)
        - len(original_first.lstrip())
    ]


    replacement = f'''
{indent}# Qwen3 uses a padded model vocabulary.  The LM-head vocabulary
{indent}# can legitimately exceed len(tokenizer).  Scientific compatibility
{indent}# requires Student and Teacher output vocabularies to match; that
{indent}# model-model check remains above.
{indent}if rank == 0:
{indent}    print(
{indent}        "VOCAB_PADDING_AUDIT",
{indent}        {{
{indent}            "student_model_vocab":
{indent}                int(student_raw.config.vocab_size),
{indent}            "teacher_model_vocab":
{indent}                int(teacher.config.vocab_size),
{indent}            "tokenizer_len":
{indent}                len(tokenizer),
{indent}            "tokenizer_vocab_size":
{indent}                int(tokenizer.vocab_size),
{indent}            "padding_rows":
{indent}                int(student_raw.config.vocab_size)
{indent}                - len(tokenizer),
{indent}        }},
{indent}        flush=True,
{indent}    )
'''


    lines[
        node.lineno - 1:
        node.end_lineno
    ] = replacement.splitlines()


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
        "INVALID_TOKENIZER_LENGTH_ASSERTION_REMOVED"
    )


else:

    if "VOCAB_PADDING_AUDIT" in text:

        print(
            "VOCAB_REPAIR_ALREADY_PRESENT"
        )

    else:

        raise RuntimeError(
            "Could not identify old vocab assertion "
            "and repair marker is absent"
        )


###############################################################################
# Verify essential model-model compatibility check remains.
###############################################################################

check_text = path.read_text(
    encoding="utf-8"
)


if (
    "student_raw.config.vocab_size"
    not in check_text
    or
    "teacher.config.vocab_size"
    not in check_text
):

    raise RuntimeError(
        "Student/Teacher model-vocab audit was lost"
    )


if "tokenizer/model vocab mismatch" in check_text:

    raise RuntimeError(
        "Invalid tokenizer/model equality assertion survived repair"
    )


if "VOCAB_PADDING_AUDIT" not in check_text:

    raise RuntimeError(
        "VOCAB_PADDING_AUDIT missing"
    )


print(
    "QWEN3_VOCAB_REPAIR_STATIC_PASS"
)
PY


python -m py_compile \
"$TRAINER" \
"$EVALUATOR"

bash -n \
"$QUEUE"

echo "TRAINER_COMPILE_AFTER_REPAIR_PASS"


###############################################################################
# STAGE 3 — DATA / PROMPT / LOCAL VOCAB AUDIT
###############################################################################

echo
echo "======================================================================"
echo "STAGE 3/7 — PROMPT + LOCAL MODEL AUDIT"
echo "======================================================================"

python - "$DATA" "$STUDENT" "$TEACHER" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

from transformers import (
    AutoConfig,
    AutoTokenizer,
)


data = Path(
    sys.argv[1]
)

student_path = sys.argv[
    2
]

teacher_path = sys.argv[
    3
]


rows = [
    json.loads(
        line
    )
    for line in data.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]


if len(
    rows
) != 3732:

    raise RuntimeError(
        f"rows={len(rows)}"
    )


###############################################################################
# These messages originate from the parquet prompt and must describe only
# the conditioning prompt.  target_translation is appended separately.
###############################################################################

patterns = Counter()


for i, row in enumerate(
    rows
):

    messages = row[
        "messages"
    ]


    if (
        not isinstance(
            messages,
            list,
        )
        or not messages
    ):

        raise RuntimeError(
            f"invalid messages row={i}"
        )


    roles = tuple(
        str(
            m.get(
                "role",
                "",
            )
        )
        for m in messages
    )


    patterns[
        roles
    ] += 1


    # Critical leakage gate.
    #
    # The target must not already be present as a final assistant turn,
    # because Correction-FKL appends target_translation after this prompt.
    if roles[
        -1
    ] == "assistant":

        raise RuntimeError(
            f"Prompt already ends in assistant response row={i}; "
            f"would duplicate correction target"
        )


print(
    "PROMPT_ROLE_PATTERNS =",
    dict(
        patterns
    ),
)

print(
    "PROMPT_TARGET_LEAKAGE_GATE_PASS"
)


###############################################################################
# Local Qwen padded-vocab audit.
###############################################################################

tokenizer = AutoTokenizer.from_pretrained(
    student_path,
    local_files_only=True,
    trust_remote_code=True,
)

student_cfg = AutoConfig.from_pretrained(
    student_path,
    local_files_only=True,
    trust_remote_code=True,
)

teacher_cfg = AutoConfig.from_pretrained(
    teacher_path,
    local_files_only=True,
    trust_remote_code=True,
)


print(
    "LOCAL_VOCAB_AUDIT =",
    {
        "student_model_vocab":
            int(
                student_cfg.vocab_size
            ),

        "teacher_model_vocab":
            int(
                teacher_cfg.vocab_size
            ),

        "tokenizer_len":
            len(
                tokenizer
            ),

        "tokenizer_vocab_size":
            int(
                tokenizer.vocab_size
            ),

        "model_minus_tokenizer":
            int(
                student_cfg.vocab_size
            )
            - len(
                tokenizer
            ),
    },
)


if int(
    student_cfg.vocab_size
) != int(
    teacher_cfg.vocab_size
):

    raise RuntimeError(
        "Student and Teacher output vocabularies differ"
    )


if max(
    tokenizer.all_special_ids
) >= int(
    student_cfg.vocab_size
):

    raise RuntimeError(
        "Tokenizer emits a special-token id outside model vocabulary"
    )


###############################################################################
# Prepared mask invariants again.
###############################################################################

patch_n = 0
random_n = 0
halo_n = 0
full_n = 0


for row in rows:

    m = row[
        "_patch_aware_v2"
    ]

    p = m[
        "patch_mask"
    ]

    r = m[
        "random_equal_count_mask"
    ]

    h = m[
        "patch_halo1_mask"
    ]

    f = m[
        "full_correction_mask"
    ]


    if len(
        p
    ) != len(
        r
    ):

        raise RuntimeError(
            "rowwise patch/random cardinality mismatch"
        )


    if not set(
        p
    ).issubset(
        h
    ):

        raise RuntimeError(
            "patch not subset of halo"
        )


    if not set(
        h
    ).issubset(
        f
    ):

        raise RuntimeError(
            "halo not subset of full"
        )


    patch_n += len(
        p
    )

    random_n += len(
        r
    )

    halo_n += len(
        h
    )

    full_n += len(
        f
    )


print(
    "MASK_TOTALS =",
    {
        "patch":
            patch_n,

        "random":
            random_n,

        "halo1":
            halo_n,

        "full":
            full_n,
    },
)


if patch_n != 26665:

    raise RuntimeError(
        f"Patch-token total changed: {patch_n}"
    )


if patch_n != random_n:

    raise RuntimeError(
        "Patch/random totals differ"
    )


print(
    "LOCAL_DATA_AND_MODEL_AUDIT_ALL_PASS"
)
PY


###############################################################################
# Ensure no previous correction trainer is still alive.
###############################################################################

LEFTOVER="$(
    pgrep -af \
    'train_correction_fkl_torchnpu_v1.py' \
    || true
)"


if [[ -n "$LEFTOVER" ]]; then

    echo
    echo "LEFTOVER CORRECTION-FKL PROCESS DETECTED:"
    echo "$LEFTOVER"

    echo
    echo "Refusing duplicate launch."

    false

fi


###############################################################################
# STAGE 4 — PATCH PREFLIGHT V2
###############################################################################

echo
echo "======================================================================"
echo "STAGE 4/7 — 16-NPU PATCH PREFLIGHT V2"
echo "======================================================================"

python -m torch.distributed.run \
  --nproc_per_node=16 \
  --master_port=29681 \
  "$TRAINER" \
  --student-model "$STUDENT" \
  --teacher-model "$TEACHER" \
  --train "$DATA" \
  --output-dir "$RUN_ROOT/$EXP/_corrfkl_patch_preflight_v2" \
  --mask-mode patch \
  --lr 1e-6 \
  --epochs 1 \
  --max-length 1024 \
  --warmup-ratio 0.03 \
  --weight-decay 0.01 \
  --max-grad-norm 1.0 \
  --seed 20260825 \
  --preflight \
  > "$PATCH_PREFLIGHT_LOG" 2>&1


grep -E \
'VOCAB_PADDING_AUDIT|CORRECTION_FKL_RUNTIME_AUDIT|CORRECTION_FKL_PREFLIGHT|CORRECTION_FKL_PREFLIGHT_PASS|Traceback|RuntimeError|ERROR' \
"$PATCH_PREFLIGHT_LOG" \
| tail -n 80


grep -q \
'CORRECTION_FKL_PREFLIGHT_PASS patch' \
"$PATCH_PREFLIGHT_LOG"

echo "PATCH_PREFLIGHT_V2_GATE_PASS"


###############################################################################
# STAGE 5 — FULL-CORRECTION PREFLIGHT V2
###############################################################################

echo
echo "======================================================================"
echo "STAGE 5/7 — 16-NPU FULL-CORRECTION PREFLIGHT V2"
echo "======================================================================"

python -m torch.distributed.run \
  --nproc_per_node=16 \
  --master_port=29683 \
  "$TRAINER" \
  --student-model "$STUDENT" \
  --teacher-model "$TEACHER" \
  --train "$DATA" \
  --output-dir "$RUN_ROOT/$EXP/_corrfkl_full_preflight_v2" \
  --mask-mode full \
  --lr 1e-6 \
  --epochs 1 \
  --max-length 1024 \
  --warmup-ratio 0.03 \
  --weight-decay 0.01 \
  --max-grad-norm 1.0 \
  --seed 20260825 \
  --preflight \
  > "$FULL_PREFLIGHT_LOG" 2>&1


grep -E \
'VOCAB_PADDING_AUDIT|CORRECTION_FKL_RUNTIME_AUDIT|CORRECTION_FKL_PREFLIGHT|CORRECTION_FKL_PREFLIGHT_PASS|Traceback|RuntimeError|ERROR' \
"$FULL_PREFLIGHT_LOG" \
| tail -n 80


grep -q \
'CORRECTION_FKL_PREFLIGHT_PASS full' \
"$FULL_PREFLIGHT_LOG"

echo "FULL_PREFLIGHT_V2_GATE_PASS"


###############################################################################
# STAGE 6 — START FOUR-EXPERIMENT LONG QUEUE
###############################################################################

echo
echo "======================================================================"
echo "STAGE 6/7 — START LONG NIGHT QUEUE"
echo "======================================================================"

RUNNING_QUEUE="$(
    pgrep -af \
    'run_correction_fkl_night_queue_v1.sh' \
    || true
)"


if [[ -n "$RUNNING_QUEUE" ]]; then

    echo "QUEUE ALREADY RUNNING:"
    echo "$RUNNING_QUEUE"

else

    nohup setsid bash "$QUEUE" \
      > "$QUEUE_LOG" 2>&1 < /dev/null &

    QUEUE_PID=$!

    echo "QUEUE_PID=$QUEUE_PID"
    echo "QUEUE_LOG=$QUEUE_LOG"
    echo "CORRECTION_FKL_NIGHT_QUEUE_STARTED"

fi


###############################################################################
# STAGE 7 — CONFIRM REAL LONG TRAINING HAS STARTED
###############################################################################

echo
echo "======================================================================"
echo "STAGE 7/7 — WAIT FOR FIRST REAL TRAINING STEP"
echo "======================================================================"

FOUND_STEP=0


for WAIT_ROUND in 1 2 3 4 5 6 7 8; do

    sleep 30

    echo
    echo "HEALTH_ROUND=$WAIT_ROUND"


    if [[ -f "$QUEUE_LOG" ]]; then

        echo "--- queue ---"

        tail -n 25 \
        "$QUEUE_LOG"

    fi


    if [[ -f "$PATCH_TRAIN_LOG" ]]; then

        echo "--- patch training ---"

        grep -E \
'VOCAB_PADDING_AUDIT|CORRECTION_FKL_RUNTIME_AUDIT|CORRECTION_FKL_TRAINING_START|epoch=1 local_step=|EPOCH_|CHECKPOINT|TRAINING_PASS|Traceback|RuntimeError|ERROR' \
        "$PATCH_TRAIN_LOG" \
        | tail -n 40


        if grep -q \
        'epoch=1 local_step=' \
        "$PATCH_TRAIN_LOG"; then

            FOUND_STEP=1

            break

        fi

    fi


    QUEUE_PROC="$(
        pgrep -af \
        'run_correction_fkl_night_queue_v1.sh' \
        || true
    )"


    TRAIN_PROC="$(
        pgrep -af \
        'train_correction_fkl_torchnpu_v1.py' \
        || true
    )"


    echo "QUEUE_PROCESS:"
    echo "$QUEUE_PROC"

    echo "TRAIN_PROCESS:"
    echo "$TRAIN_PROC"


    if (
        [[ -z "$QUEUE_PROC" ]]
        &&
        [[ -z "$TRAIN_PROC" ]]
    ); then

        echo
        echo "QUEUE/TRAINER DIED BEFORE FIRST STEP."

        echo
        echo "QUEUE LOG:"
        tail -n 100 \
        "$QUEUE_LOG" \
        2>/dev/null \
        || true

        echo
        echo "PATCH TRAIN LOG:"
        tail -n 160 \
        "$PATCH_TRAIN_LOG" \
        2>/dev/null \
        || true

        false

    fi

done


if [[ "$FOUND_STEP" -ne 1 ]]; then

    echo
    echo "No first-step marker after health window."

    echo "Queue may still be loading models; current process audit:"

    pgrep -af \
    'run_correction_fkl_night_queue_v1.sh|train_correction_fkl_torchnpu_v1.py' \
    || true

    false

fi


echo
echo "======================================================================"
echo "LONG JOB CONFIRMED"
echo "======================================================================"

echo
echo "PATCH_PREFLIGHT_V2_GATE_PASS"
echo "FULL_PREFLIGHT_V2_GATE_PASS"
echo "CORRECTION_FKL_NIGHT_QUEUE_STARTED"
echo "FIRST_LONG_TRAINING_STEP_PASS"
echo
echo "Four serial experiments are queued:"
echo "  1. Patch-FKL"
echo "  2. Random equal-count FKL"
echo "  3. Halo1-FKL"
echo "  4. Full correction-trajectory FKL"
echo
echo "Each starts independently from Qwen3-0.6B Base."
echo "Each trains 3 epochs on 16 NPUs and is then evaluated."
echo
echo "QUEUE_LOG=$QUEUE_LOG"
echo "PATCH_TRAIN_LOG=$PATCH_TRAIN_LOG"
echo
echo "LONG_JOB_CONFIRMED_SAFE_TO_LEAVE"

