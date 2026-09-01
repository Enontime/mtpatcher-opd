#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

###############################################################################
# Frozen scientific components
###############################################################################

ORIGINAL_TRAINER="$ROOT/scripts/mtpatcher_v4/train_opd_pgrkl_cleanroom_v2_torchnpu.py"

FULL_TRAINER="$ROOT/scripts/mtpatcher_v10/train_opd_pgrkl_cleanroom_full6565_v4_torchnpu.py"

EVALUATOR="$ROOT/scripts/mtpatcher_v6/eval_correction_fkl_torchnpu_v1.py"

STUDENT="$MODEL_ROOT/Qwen3-0.6B"

TEACHER="$MODEL_ROOT/Qwen3-8B"

###############################################################################
# Frozen RQ2 datasets / runs
###############################################################################

FULL_DATA="$DATA_ROOT/$EXP/opd_full_sources6565_v1.jsonl"

SELECTED_RUN="$RUN_ROOT/$EXP/opd_torchnpu_pgrkl_cleanroom_pe3732_v2"

RANDOM_RUN="$RUN_ROOT/$EXP/opd_torchnpu_pgrkl_random3732_v1"

FULL_RUN="$RUN_ROOT/$EXP/opd_torchnpu_pgrkl_full6565_v1"

###############################################################################
# Eval
###############################################################################

WMT="$DATA_ROOT/pilot_v2_qwen3_06b/wmt24_zh_en998.jsonl"

FLORES="$DATA_ROOT/pilot_v2_qwen3_06b/flores_zh_en1012.jsonl"

CHALLENGE="$DATA_ROOT/pilot_v2_qwen3_06b/challenge_zh_en197.jsonl"

###############################################################################
# Runner / log
###############################################################################

RUNNER="$ROOT/scripts/mtpatcher_v10/run_full6565_direct_and_finish_rq2_v4.sh"

LOG="$LOG_ROOT/$EXP/rq2_full6565_direct_v4.log"

SUMMARY="$RUN_ROOT/$EXP/rq2_selection_opd_summary_v4.json"

STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p \
  "$ROOT/scripts/mtpatcher_v10" \
  "$LOG_ROOT/$EXP"


###############################################################################
# STAGE 1 — Freeze completed Selected + Random
###############################################################################

echo
echo "======================================================================"
echo "STAGE 1/6 — VERIFY COMPLETED RQ2 SYSTEMS"
echo "======================================================================"

test -f "$ORIGINAL_TRAINER"
test -f "$EVALUATOR"
test -d "$STUDENT"
test -d "$TEACHER"
test -f "$FULL_DATA"

test -f "$SELECTED_RUN/epoch3/config.json"
test -f "$RANDOM_RUN/epoch3/config.json"


for SPLIT in \
  wmt24 \
  flores \
  challenge
do

    test -f \
      "$SELECTED_RUN/rq2_eval_epoch3/$SPLIT/metrics.json"

    test -f \
      "$RANDOM_RUN/rq2_eval_epoch3/$SPLIT/metrics.json"

done


echo "SELECTED3732_FROZEN_COMPLETE"
echo "RANDOM3732_FROZEN_COMPLETE"


###############################################################################
# STAGE 2 — Audit Full6565
###############################################################################

echo
echo "======================================================================"
echo "STAGE 2/6 — FULL6565 DATA AUDIT"
echo "======================================================================"

python - "$FULL_DATA" <<'PY'
import json
import sys
from pathlib import Path


path = Path(sys.argv[1])

rows = []


with path.open(
    "r",
    encoding="utf-8-sig",
) as f:

    for line in f:

        if line.strip():

            rows.append(
                json.loads(line)
            )


if len(rows) != 6565:

    raise RuntimeError(
        f"Expected 6565 rows, got {len(rows)}"
    )


indices = []


for i, row in enumerate(rows):

    source = str(
        row.get(
            "source",
            "",
        )
    )

    messages = row.get(
        "messages"
    )


    if not source:

        raise RuntimeError(
            f"Empty source row={i}"
        )


    if (
        not isinstance(
            messages,
            list,
        )
        or not messages
    ):

        raise RuntimeError(
            f"Invalid messages row={i}"
        )


    if (
        str(
            messages[-1].get(
                "role",
                "",
            )
        )
        == "assistant"
    ):

        raise RuntimeError(
            f"Target leakage row={i}"
        )


    if row.get(
        "index"
    ) is not None:

        indices.append(
            int(
                row[
                    "index"
                ]
            )
        )


if indices:

    if len(
        set(
            indices
        )
    ) != 6565:

        raise RuntimeError(
            "Full source indices are not unique"
        )


print(
    "FULL6565_DATA_AUDIT =",
    {
        "rows":
            len(rows),

        "unique_indices":
            len(
                set(indices)
            )
            if indices
            else None,

        "reference_used":
            False,
    },
)

print(
    "FULL6565_DATA_AUDIT_PASS"
)
PY


###############################################################################
# STAGE 3 — Patch the ACTUAL trainer guard found by traceback
#
# Exact intended change:
#
#     Expected PE source set size 3732
#
# and its immediately associated conditional:
#
#     ... != 3732
#
# become 6565.
#
# NOTHING ELSE in the trainer is modified.
###############################################################################

echo
echo "======================================================================"
echo "STAGE 3/6 — PATCH TRAINER'S ACTUAL 3732 GUARD"
echo "======================================================================"

echo
echo "===== ORIGINAL TRACEBACK AREA ====="

nl -ba "$ORIGINAL_TRAINER" \
  | sed -n '345,375p'


export \
  ORIGINAL_TRAINER \
  FULL_TRAINER


python - <<'PY'
import difflib
import os
import re
from pathlib import Path


src = Path(
    os.environ[
        "ORIGINAL_TRAINER"
    ]
)

dst = Path(
    os.environ[
        "FULL_TRAINER"
    ]
)


original_lines = src.read_text(
    encoding="utf-8"
).splitlines(
    keepends=True
)


lines = list(
    original_lines
)


###############################################################################
# Find the exact error exposed by the traceback.
###############################################################################

message_hits = [
    i
    for i, line in enumerate(lines)
    if "Expected PE source set size 3732" in line
]


print(
    "EXACT_ERROR_MESSAGE_HITS =",
    [
        x + 1
        for x in message_hits
    ],
)


if len(message_hits) != 1:

    raise RuntimeError(
        "Expected exactly one "
        "'Expected PE source set size 3732' "
        f"message, found {len(message_hits)}"
    )


msg_i = message_hits[0]


###############################################################################
# Find the associated preceding conditional.
###############################################################################

condition_candidates = []


for j in range(
    max(
        0,
        msg_i - 20,
    ),
    msg_i,
):

    line = lines[j]

    if (
        "3732" in line
        and "if" in line
        and (
            "!=" in line
            or "==" in line
        )
    ):

        condition_candidates.append(
            j
        )


print(
    "LOCAL_CONDITION_CANDIDATES =",
    [
        {
            "line":
                j + 1,

            "text":
                lines[j].rstrip(),
        }
        for j in condition_candidates
    ],
)


if len(condition_candidates) != 1:

    raise RuntimeError(
        "Could not uniquely identify the "
        "3732 trainer condition associated "
        "with the traceback"
    )


cond_i = condition_candidates[0]


###############################################################################
# Patch exactly two lines.
###############################################################################

old_cond = lines[
    cond_i
]

old_msg = lines[
    msg_i
]


new_cond = old_cond.replace(
    "3732",
    "6565",
    1,
)

new_msg = old_msg.replace(
    "Expected PE source set size 3732",
    "Expected PE source set size 6565",
    1,
)


if new_cond == old_cond:

    raise RuntimeError(
        "Condition replacement failed"
    )


if new_msg == old_msg:

    raise RuntimeError(
        "Message replacement failed"
    )


lines[
    cond_i
] = new_cond

lines[
    msg_i
] = new_msg


###############################################################################
# Diff audit: exactly two source lines may change.
###############################################################################

diff = list(
    difflib.unified_diff(
        original_lines,
        lines,
        fromfile=str(src),
        tofile=str(dst),
    )
)


print(
    "===== TRAINER PATCH DIFF ====="
)

print(
    "".join(
        diff
    )
)


removed_code = [
    line
    for line in diff
    if (
        line.startswith("-")
        and not line.startswith("---")
    )
]


added_code = [
    line
    for line in diff
    if (
        line.startswith("+")
        and not line.startswith("+++")
    )
]


if len(
    removed_code
) != 2:

    raise RuntimeError(
        f"Expected 2 removed lines, "
        f"got {len(removed_code)}"
    )


if len(
    added_code
) != 2:

    raise RuntimeError(
        f"Expected 2 added lines, "
        f"got {len(added_code)}"
    )


for line in removed_code:

    if "3732" not in line:

        raise RuntimeError(
            "Unexpected removed line"
        )


for line in added_code:

    if "6565" not in line:

        raise RuntimeError(
            "Unexpected added line"
        )


###############################################################################
# Core clean-PG implementation markers must remain identical.
###############################################################################

new_text = "".join(
    lines
)

old_text = "".join(
    original_lines
)


required_markers = [
    "CLEAN_PG_RUNTIME_AUDIT_PASS",
    "exact_tail",
]


for marker in required_markers:

    if old_text.count(
        marker
    ) != new_text.count(
        marker
    ):

        raise RuntimeError(
            f"Runtime/scientific marker changed: "
            f"{marker}"
        )


###############################################################################
# CLI structure unchanged.
###############################################################################

for marker in [
    "--student",
    "--teacher",
    "--train",
    "--output-dir",
    "--epochs",
    "--lr",
    "--temperature",
    "--top-p",
    "--top-k",
]:

    if old_text.count(
        marker
    ) != new_text.count(
        marker
    ):

        raise RuntimeError(
            f"CLI structure changed: "
            f"{marker}"
        )


dst.write_text(
    new_text,
    encoding="utf-8",
)


print(
    "FULL6565_TRAINER_EXACT_TWO_LINE_PATCH_PASS"
)
PY


python -m py_compile \
  "$FULL_TRAINER"


echo
echo "===== FIXED TRACEBACK AREA ====="

nl -ba "$FULL_TRAINER" \
  | sed -n '345,375p'


grep -q \
  'Expected PE source set size 6565' \
  "$FULL_TRAINER"


echo
echo "FULL6565_TRAINER_COMPILE_PASS"


###############################################################################
# STAGE 4 — Remove/preserve only failed Full attempts
###############################################################################

echo
echo "======================================================================"
echo "STAGE 4/6 — FULL RUN STATE"
echo "======================================================================"

RUNNING="$(
    pgrep -af \
    '[t]rain_opd_pgrkl_cleanroom_full6565_v4_torchnpu.py|[r]un_full6565_direct_and_finish_rq2_v4.sh' \
    || true
)"


if [[ -n "$RUNNING" ]]; then

    echo "A V4 Full6565 job is already running:"
    echo "$RUNNING"

    false

fi


if [[ -e "$FULL_RUN" ]] \
   && [[ ! -f "$FULL_RUN/epoch3/config.json" ]]; then

    PRESERVED="${FULL_RUN}.failed_before_real_training_${STAMP}"

    mv \
      "$FULL_RUN" \
      "$PRESERVED"

    echo \
      "PRESERVED_FAILED_FULL_RUN=$PRESERVED"

fi


if [[ -f "$FULL_RUN/epoch3/config.json" ]]; then

    echo "FULL6565_EPOCH3_ALREADY_COMPLETE"

else

    echo "FULL6565_FRESH_OUTPUT_READY"

fi


###############################################################################
# STAGE 5 — Build direct runner
#
# These are the SAME clean PG-RKL settings used by Random3732:
#
#   epochs              3
#   lr                  1e-6
#   max_prompt_length   512
#   max_new_tokens      256
#   warmup_ratio        .03
#   weight_decay        .01
#   max_grad_norm       1.0
#   temperature         .7
#   top_p               .8
#   top_k               20
#   seed                20260824
#
# Only:
#   TRAIN 3732 -> Full6565
#   OUTPUT -> Full run
###############################################################################

echo
echo "======================================================================"
echo "STAGE 5/6 — BUILD DIRECT FULL6565 RUNNER"
echo "======================================================================"

cat > "$RUNNER" <<BASHRUN
#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="$EXP"

TRAINER="$FULL_TRAINER"

STUDENT="$STUDENT"
TEACHER="$TEACHER"

TRAIN="$FULL_DATA"

SELECTED_RUN="$SELECTED_RUN"
RANDOM_RUN="$RANDOM_RUN"
FULL_RUN="$FULL_RUN"

EVALUATOR="$EVALUATOR"

WMT="$WMT"
FLORES="$FLORES"
CHALLENGE="$CHALLENGE"

SUMMARY="$SUMMARY"


echo
echo "======================================================================"
echo "RQ2 FULL6565 — DIRECT CLEAN PG-RKL"
echo "======================================================================"

echo "STUDENT=\$STUDENT"
echo "TEACHER=\$TEACHER"
echo "TRAIN=\$TRAIN"
echo "OUTPUT=\$FULL_RUN"

echo "EPOCHS=3"
echo "LR=1e-6"
echo "MAX_PROMPT_LENGTH=512"
echo "MAX_NEW_TOKENS=256"
echo "WARMUP_RATIO=0.03"
echo "WEIGHT_DECAY=0.01"
echo "MAX_GRAD_NORM=1.0"
echo "TEMPERATURE=0.7"
echo "TOP_P=0.8"
echo "TOP_K=20"
echo "SEED=20260824"


###############################################################################
# Direct training.
###############################################################################

if [[ ! -f "\$FULL_RUN/epoch3/config.json" ]]; then

    python -m torch.distributed.run \
      --nproc_per_node=16 \
      --master_port=29781 \
      "\$TRAINER" \
      --student "\$STUDENT" \
      --teacher "\$TEACHER" \
      --train "\$TRAIN" \
      --output-dir "\$FULL_RUN" \
      --epochs 3 \
      --lr 1e-6 \
      --max-prompt-length 512 \
      --max-new-tokens 256 \
      --warmup-ratio 0.03 \
      --weight-decay 0.01 \
      --max-grad-norm 1.0 \
      --temperature 0.7 \
      --top-p 0.8 \
      --top-k 20 \
      --seed 20260824

fi


test -f \
  "\$FULL_RUN/epoch3/config.json"


echo
echo "RQ2_FULL6565_TRAINING_COMPLETE"


###############################################################################
# Fixed epoch3 evaluation.
###############################################################################

EVAL="\$FULL_RUN/rq2_eval_epoch3"

mkdir -p \
  "\$EVAL"


for SPEC in \
  "wmt24:\$WMT" \
  "flores:\$FLORES" \
  "challenge:\$CHALLENGE"
do

    SPLIT="\${SPEC%%:*}"

    INPUT="\${SPEC#*:}"


    mkdir -p \
      "\$EVAL/\$SPLIT"


    python "\$EVALUATOR" \
      --model "\$FULL_RUN/epoch3" \
      --input "\$INPUT" \
      --output "\$EVAL/\$SPLIT/predictions.jsonl" \
      --metrics "\$EVAL/\$SPLIT/metrics.json" \
      --method "Full-OPD6565_rq2_epoch3_\${SPLIT}" \
      --batch-size 16 \
      --max-new-tokens 256

done


echo
echo "RQ2_EVAL_COMPLETE Full-OPD6565"


###############################################################################
# Final RQ2 comparison.
###############################################################################

python - \
  "\$SELECTED_RUN" \
  "\$RANDOM_RUN" \
  "\$FULL_RUN" \
  "\$SUMMARY" <<'PY'
import json
import sys
from pathlib import Path


selected_run = Path(
    sys.argv[1]
)

random_run = Path(
    sys.argv[2]
)

full_run = Path(
    sys.argv[3]
)

summary_path = Path(
    sys.argv[4]
)


###############################################################################
# Frozen Base results.
###############################################################################

base = {
    "wmt24": {
        "BLEU":
            15.5362135559,

        "chrF":
            45.537530,
    },

    "flores": {
        "BLEU":
            19.9714797904,

        "chrF":
            50.860857,
    },

    "challenge": {
        "BLEU":
            16.5378710573,

        "chrF":
            45.758287,
    },
}


splits = (
    "wmt24",
    "flores",
    "challenge",
)


systems = {
    "Selected-OPD3732":
        selected_run,

    "Random-OPD3732":
        random_run,

    "Full-OPD6565":
        full_run,
}


sizes = {
    "Selected-OPD3732":
        3732,

    "Random-OPD3732":
        3732,

    "Full-OPD6565":
        6565,
}


results = {}


print(
    "=" * 130
)

print(
    "RQ2 — DOES MT-PATCHER SOURCE SELECTION HELP CLEAN OPD?"
)

print(
    "=" * 130
)

print(
    "PRIMARY PROTOCOL: FIXED EPOCH3"
)

print()

print(
    f"{'SYSTEM':22s} "
    f"{'N':>7s} "
    f"{'WMT':>10s} "
    f"{'FLORES':>10s} "
    f"{'CHALL':>10s} "
    f"{'AVG ΔBLEU':>12s} "
    f"{'AVG ΔchrF':>12s}"
)


for system, run in systems.items():

    bleus = []

    delta_bleu = []
    delta_chrf = []


    for split in splits:

        metric_path = (
            run
            / "rq2_eval_epoch3"
            / split
            / "metrics.json"
        )


        if not metric_path.exists():

            raise RuntimeError(
                f"Missing metrics: "
                f"{metric_path}"
            )


        m = json.loads(
            metric_path.read_text(
                encoding="utf-8"
            )
        )


        bleu = float(
            m[
                "BLEU"
            ]
        )

        chrf = float(
            m[
                "chrF"
            ]
        )


        bleus.append(
            bleu
        )


        delta_bleu.append(
            bleu
            - base[
                split
            ][
                "BLEU"
            ]
        )


        delta_chrf.append(
            chrf
            - base[
                split
            ][
                "chrF"
            ]
        )


    avg_db = (
        sum(
            delta_bleu
        )
        / 3
    )

    avg_dc = (
        sum(
            delta_chrf
        )
        / 3
    )


    results[
        system
    ] = {
        "N":
            sizes[
                system
            ],

        "WMT_BLEU":
            bleus[0],

        "FLORES_BLEU":
            bleus[1],

        "CHALLENGE_BLEU":
            bleus[2],

        "avg_delta_bleu":
            avg_db,

        "avg_delta_chrf":
            avg_dc,
    }


    print(
        f"{system:22s} "
        f"{sizes[system]:7d} "
        f"{bleus[0]:10.6f} "
        f"{bleus[1]:10.6f} "
        f"{bleus[2]:10.6f} "
        f"{avg_db:+12.6f} "
        f"{avg_dc:+12.6f}"
    )


selected = results[
    "Selected-OPD3732"
][
    "avg_delta_bleu"
]

random_v = results[
    "Random-OPD3732"
][
    "avg_delta_bleu"
]

full_v = results[
    "Full-OPD6565"
][
    "avg_delta_bleu"
]


sr = (
    selected
    - random_v
)

fs = (
    full_v
    - selected
)

fr = (
    full_v
    - random_v
)


print()

print(
    "=" * 130
)

print(
    "RQ2 PRIMARY COMPARISONS"
)

print(
    "=" * 130
)


print(
    f"Selected3732 - Random3732 = "
    f"{sr:+.6f} BLEU"
)

print(
    f"Full6565 - Selected3732   = "
    f"{fs:+.6f} BLEU"
)

print(
    f"Full6565 - Random3732     = "
    f"{fr:+.6f} BLEU"
)


print()

print(
    "Selected vs Random is the primary same-N selection comparison."
)

print(
    "Full6565 has more examples and more updates, "
    "so it is a coverage/reference system."
)


summary = {
    "protocol":
        "fixed_epoch3",

    "clean_pgrkl_settings": {
        "epochs":
            3,

        "lr":
            1e-6,

        "max_prompt_length":
            512,

        "max_new_tokens":
            256,

        "warmup_ratio":
            0.03,

        "weight_decay":
            0.01,

        "max_grad_norm":
            1.0,

        "temperature":
            0.7,

        "top_p":
            0.8,

        "top_k":
            20,

        "seed":
            20260824,
    },

    "selected_random_same_n":
        True,

    "full_compute_matched":
        False,

    "systems":
        results,

    "selected_minus_random_bleu":
        sr,

    "full_minus_selected_bleu":
        fs,

    "full_minus_random_bleu":
        fr,
}


summary_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)


summary_path.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print()

print(
    "SUMMARY_JSON =",
    summary_path,
)

print()

print(
    "RQ2_SELECTION_OPD_ALL_PASS"
)
PY

BASHRUN


chmod +x \
  "$RUNNER"

bash -n \
  "$RUNNER"


echo
echo "FULL6565_DIRECT_RUNNER_STATIC_PASS"


###############################################################################
# STAGE 6 — Detached launch
###############################################################################

echo
echo "======================================================================"
echo "STAGE 6/6 — LAUNCH DIRECT FULL6565"
echo "======================================================================"

RUNNING="$(
    pgrep -af \
    '[r]un_full6565_direct_and_finish_rq2_v4.sh|[t]rain_opd_pgrkl_cleanroom_full6565_v4_torchnpu.py' \
    || true
)"


if [[ -n "$RUNNING" ]]; then

    echo "A Full6565 V4 job is already running:"
    echo "$RUNNING"

    false

fi


nohup setsid bash "$RUNNER" \
  > "$LOG" 2>&1 < /dev/null &


PID=$!


echo "PID=$PID"

echo "LOG=$LOG"

echo "RQ2_FULL6565_DIRECT_STARTED"


###############################################################################
# Health gate: now trainer itself must accept 6565 and perform a real update.
###############################################################################

PASS=0


for ROUND in $(seq 1 30)
do

    sleep 30

    echo
    echo "HEALTH_ROUND=$ROUND"


    grep -E \
'TRAIN_ROWS|OPD_TRAINING_START|CLEAN_PG_RUNTIME_AUDIT_PASS|epoch=1 local_step=1/|Expected PE source set size|Traceback|RuntimeError|ChildFailedError' \
    "$LOG" \
    2>/dev/null \
    | tail -n 80 \
    || true


    if grep -q \
      'CLEAN_PG_RUNTIME_AUDIT_PASS' \
      "$LOG" \
      2>/dev/null \
      && grep -q \
      'epoch=1 local_step=1/' \
      "$LOG" \
      2>/dev/null; then

        PASS=1

        break

    fi


    if grep -qE \
      'Traceback|RuntimeError|ChildFailedError' \
      "$LOG" \
      2>/dev/null; then

        echo
        echo "FULL6565 DIRECT TRAINING FAILED"

        tail -n 260 \
          "$LOG"

        false

    fi

done


if [[ "$PASS" -ne 1 ]]; then

    echo
    echo "Health window ended before first audited update."

    echo
    echo "PROCESS STATE:"

    pgrep -af \
      'run_full6565_direct_and_finish_rq2_v4.sh|train_opd_pgrkl_cleanroom_full6565_v4_torchnpu.py' \
      || true

    echo
    echo "LATEST LOG:"

    tail -n 260 \
      "$LOG" \
      2>/dev/null \
      || true

    false

fi


echo
echo "======================================================================"
echo "FULL6565 REAL TRAINER UPDATE VERIFIED"
echo "======================================================================"

echo "SELECTED3732_FROZEN_COMPLETE"
echo "RANDOM3732_FROZEN_COMPLETE"
echo "FULL6565_DATA_AUDIT_PASS"
echo "FULL6565_TRAINER_EXACT_TWO_LINE_PATCH_PASS"
echo "FULL6565_TRAINER_COMPILE_PASS"
echo "CLEAN_PG_RUNTIME_AUDIT_PASS"
echo "FULL6565_FIRST_REAL_UPDATE_PASS"

echo
echo "Detached pipeline continues automatically:"
echo "  Full6565 epoch1"
echo "  Full6565 epoch2"
echo "  Full6565 epoch3"
echo "  WMT24 evaluation"
echo "  FLORES evaluation"
echo "  Challenge evaluation"
echo "  RQ2 Selected/Random/Full summary"

echo
echo "LOG=$LOG"

echo
echo "RQ2_FULL6565_LONG_RUN_SAFE"

