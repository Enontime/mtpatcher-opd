#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

V4_TRAINER="$ROOT/scripts/mtpatcher_v4/train_opd_pgrkl_cleanroom_v2_torchnpu.py"

OLD_FULL_MASTER="$ROOT/scripts/mtpatcher_v10/run_pgrkl_full6565_v1.sh"

FIXED_TRAINER="$ROOT/scripts/mtpatcher_v10/train_opd_pgrkl_cleanroom_v2_full6565_torchnpu.py"

FIXED_MASTER="$ROOT/scripts/mtpatcher_v10/run_pgrkl_full6565_fixed_v2.sh"

EVALUATOR="$ROOT/scripts/mtpatcher_v6/eval_correction_fkl_torchnpu_v1.py"

SELECTED_RUN="$RUN_ROOT/$EXP/opd_torchnpu_pgrkl_cleanroom_pe3732_v2"

RANDOM_RUN="$RUN_ROOT/$EXP/opd_torchnpu_pgrkl_random3732_v1"

FULL_RUN="$RUN_ROOT/$EXP/opd_torchnpu_pgrkl_full6565_v1"

FULL_DATA="$DATA_ROOT/$EXP/opd_full_sources6565_v1.jsonl"

BASE_RUN="$RUN_ROOT/$EXP/_verified_base_eval_v3"

WMT="$DATA_ROOT/pilot_v2_qwen3_06b/wmt24_zh_en998.jsonl"

FLORES="$DATA_ROOT/pilot_v2_qwen3_06b/flores_zh_en1012.jsonl"

CHALLENGE="$DATA_ROOT/pilot_v2_qwen3_06b/challenge_zh_en197.jsonl"

OLD_QUEUE_LOG="$LOG_ROOT/$EXP/rq2_selection_opd_queue_v1.log"

FULL_LOG="$LOG_ROOT/$EXP/rq2_full6565_repair_v2.log"

SUMMARY="$RUN_ROOT/$EXP/rq2_selection_opd_summary_v2.json"

STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p \
  "$ROOT/scripts/mtpatcher_v10" \
  "$LOG_ROOT/$EXP"


###############################################################################
# STAGE 1 — PROVE SELECTED / RANDOM ARE ALREADY COMPLETE
###############################################################################

echo
echo "======================================================================"
echo "STAGE 1/7 — PRESERVE COMPLETED RQ2 WORK"
echo "======================================================================"

test -f \
  "$SELECTED_RUN/epoch3/config.json"

test -f \
  "$RANDOM_RUN/epoch3/config.json"


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


echo "SELECTED3732_ALREADY_COMPLETE_PASS"
echo "RANDOM3732_ALREADY_COMPLETE_PASS"


if [[ -f "$OLD_QUEUE_LOG" ]]; then

    cp \
      "$OLD_QUEUE_LOG" \
      "${OLD_QUEUE_LOG}.before_full6565_fix_${STAMP}"

    echo \
      "PRESERVED_OLD_QUEUE_LOG=${OLD_QUEUE_LOG}.before_full6565_fix_${STAMP}"

fi


###############################################################################
# STAGE 2 — VERIFY FULL DATA REALLY IS 6565 SOURCE-ONLY ROWS
###############################################################################

echo
echo "======================================================================"
echo "STAGE 2/7 — FULL6565 DATA AUDIT"
echo "======================================================================"

python - "$FULL_DATA" <<'PY'
import json
import sys
from pathlib import Path


path = Path(
    sys.argv[1]
)


rows = [
    json.loads(line)
    for line in path.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]


if len(rows) != 6565:

    raise RuntimeError(
        f"Expected 6565 rows, got {len(rows)}"
    )


bad_source = 0
bad_messages = 0
target_leak = 0


for row in rows:

    if not str(
        row.get(
            "source",
            "",
        )
    ):

        bad_source += 1


    messages = row.get(
        "messages"
    )


    if (
        not isinstance(
            messages,
            list,
        )
        or not messages
    ):

        bad_messages += 1

    elif str(
        messages[-1].get(
            "role",
            "",
        )
    ) == "assistant":

        target_leak += 1


print(
    "FULL6565_AUDIT =",
    {
        "rows":
            len(rows),

        "bad_source":
            bad_source,

        "bad_messages":
            bad_messages,

        "assistant_target_leak":
            target_leak,
    },
)


if bad_source:

    raise RuntimeError(
        "Empty source found"
    )


if bad_messages:

    raise RuntimeError(
        "Bad messages found"
    )


if target_leak:

    raise RuntimeError(
        "Assistant target leakage found"
    )


print(
    "FULL6565_SOURCE_ONLY_DATA_PASS"
)
PY


###############################################################################
# STAGE 3 — CLONE TRAINER AND PATCH ONLY THE ROW-CARDINALITY GUARD
###############################################################################

echo
echo "======================================================================"
echo "STAGE 3/7 — BUILD FULL6565-SAFE CLEAN PG-RKL TRAINER"
echo "======================================================================"

export \
  V4_TRAINER \
  FIXED_TRAINER


python - <<'PY'
import os
import re
from pathlib import Path


src_path = Path(
    os.environ[
        "V4_TRAINER"
    ]
)

dst_path = Path(
    os.environ[
        "FIXED_TRAINER"
    ]
)


text = src_path.read_text(
    encoding="utf-8"
)


original = text


###############################################################################
# Only patch row-count guards / their diagnostic strings.
###############################################################################

patterns = [
    (
        r'len\(\s*rows\s*\)\s*!=\s*3732',
        'len(rows) != 6565',
    ),

    (
        r'len\(\s*rows\s*\)\s*==\s*3732',
        'len(rows) == 6565',
    ),

    (
        r'assert\s+len\(\s*rows\s*\)\s*==\s*3732',
        'assert len(rows) == 6565',
    ),

    (
        r'expected 3732 rows',
        'expected 6565 rows',
    ),

    (
        r'Expected 3732 rows',
        'Expected 6565 rows',
    ),
]


changes = 0


for pattern, replacement in patterns:

    text, n = re.subn(
        pattern,
        replacement,
        text,
    )

    changes += n


print(
    "TRAINER_ROW_GUARD_REPLACEMENTS =",
    changes,
)


###############################################################################
# A hard guard: do not silently accept zero relevant changes if trainer
# itself contains the cardinality assertion.
###############################################################################

if (
    "3732 rows" in original
    or
    "len(rows) != 3732" in original
    or
    "len(rows)==3732" in original.replace(" ", "")
):

    if changes == 0:

        raise RuntimeError(
            "Trainer contains a 3732 cardinality marker "
            "but no targeted replacement was applied"
        )


###############################################################################
# Scientific invariants must survive unchanged.
###############################################################################

required_markers = [
    "Qwen",
    "CLEAN_PG_RUNTIME_AUDIT_PASS",
]


for marker in required_markers:

    if marker not in text:

        raise RuntimeError(
            f"Scientific/runtime invariant marker missing: {marker}"
        )


###############################################################################
# These objective-related strings are copied, not rewritten.
###############################################################################

for forbidden_change in [
    "--student",
    "--teacher",
]:

    if original.count(
        forbidden_change
    ) != text.count(
        forbidden_change
    ):

        raise RuntimeError(
            f"Unexpected CLI structural change: {forbidden_change}"
        )


dst_path.write_text(
    text,
    encoding="utf-8",
)


print(
    "FULL6565_TRAINER_CARDINALITY_PATCH_PASS"
)
PY


python -m py_compile \
  "$FIXED_TRAINER"


###############################################################################
# Show precisely what changed around cardinality.
###############################################################################

echo
echo "===== FULL6565 TRAINER CARDINALITY LINES ====="

grep -nE \
'6565|expected .* rows|Expected .* rows|len\(rows\)' \
"$FIXED_TRAINER" \
| head -n 80 \
|| true


###############################################################################
# STAGE 4 — CLONE THE ALREADY-GENERATED FULL MASTER
#
# It already has:
#   Full6565 data
#   Qwen3-0.6B Student
#   Qwen3-8B Teacher
#   clean PG-RKL hyperparameters
#
# We only:
#   point to fixed trainer
#   patch any embedded preflight 3732 guard
###############################################################################

echo
echo "======================================================================"
echo "STAGE 4/7 — BUILD FULL6565-SAFE MASTER"
echo "======================================================================"

export \
  OLD_FULL_MASTER \
  FIXED_MASTER \
  FIXED_TRAINER


python - <<'PY'
import os
import re
from pathlib import Path


src = Path(
    os.environ[
        "OLD_FULL_MASTER"
    ]
)

dst = Path(
    os.environ[
        "FIXED_MASTER"
    ]
)

trainer = os.environ[
    "FIXED_TRAINER"
]


text = src.read_text(
    encoding="utf-8"
)


###############################################################################
# Replace trainer path exactly.
###############################################################################

old_trainer = (
    "$ROOT/scripts/mtpatcher_v4/"
    "train_opd_pgrkl_cleanroom_v2_torchnpu.py"
)


if old_trainer not in text:

    raise RuntimeError(
        "Original validated trainer reference "
        "not found in Full master"
    )


text = text.replace(
    old_trainer,
    trainer,
    1,
)


###############################################################################
# Patch only embedded row-count sanity checks.
###############################################################################

patterns = [
    (
        r'len\(\s*rows\s*\)\s*!=\s*3732',
        'len(rows) != 6565',
    ),

    (
        r'len\(\s*rows\s*\)\s*==\s*3732',
        'len(rows) == 6565',
    ),

    (
        r'assert\s+len\(\s*rows\s*\)\s*==\s*3732',
        'assert len(rows) == 6565',
    ),

    (
        r'expected 3732 rows',
        'expected 6565 rows',
    ),

    (
        r'Expected 3732 rows',
        'Expected 6565 rows',
    ),
]


row_changes = 0


for pattern, repl in patterns:

    text, n = re.subn(
        pattern,
        repl,
        text,
    )

    row_changes += n


print(
    "MASTER_ROW_GUARD_REPLACEMENTS =",
    row_changes,
)


###############################################################################
# Required experimental invariants.
###############################################################################

required = [
    'NAME="opd_torchnpu_pgrkl_full6565_v1"',
    "opd_full_sources6565_v1.jsonl",
    'STUDENT="$MODEL_ROOT/Qwen3-0.6B"',
    'TEACHER="$MODEL_ROOT/Qwen3-8B"',
    "train_opd_pgrkl_cleanroom_v2_full6565_torchnpu.py",
]


for marker in required:

    if marker not in text:

        raise RuntimeError(
            f"Full master invariant missing: {marker}"
        )


dst.write_text(
    text,
    encoding="utf-8",
)


print(
    "FULL6565_MASTER_CARDINALITY_PATCH_PASS"
)
PY


chmod +x \
  "$FIXED_MASTER"

bash -n \
  "$FIXED_MASTER"


echo
echo "===== FIXED MASTER KEY LINES ====="

grep -nE \
'NAME=|TRAIN=|STUDENT=|TEACHER=|TRAINER=|6565|master_port|MASTER_PORT' \
"$FIXED_MASTER" \
| head -n 100


echo "FULL6565_MASTER_STATIC_PASS"


###############################################################################
# STAGE 5 — PRESERVE ANY PARTIAL FULL RUN
###############################################################################

echo
echo "======================================================================"
echo "STAGE 5/7 — FRESH FULL6565 OUTPUT GATE"
echo "======================================================================"

if [[ -e "$FULL_RUN" ]] \
   && [[ ! -f "$FULL_RUN/epoch3/config.json" ]]; then

    PRESERVED="${FULL_RUN}.failed_3732_guard_${STAMP}"

    mv \
      "$FULL_RUN" \
      "$PRESERVED"

    echo \
      "PRESERVED_PARTIAL_FULL_RUN=$PRESERVED"

fi


if [[ -f "$FULL_RUN/epoch3/config.json" ]]; then

    echo "FULL6565 epoch3 already exists; training will be reused."

else

    echo "FULL6565 fresh training required."

fi


###############################################################################
# STAGE 6 — BUILD FULL-ONLY RESUME + FINAL RQ2 SUMMARY
###############################################################################

echo
echo "======================================================================"
echo "STAGE 6/7 — BUILD FULL-ONLY RQ2 RESUME"
echo "======================================================================"

RESUME="$ROOT/scripts/mtpatcher_v10/run_full6565_and_finish_rq2_v2.sh"


cat > "$RESUME" <<BASHRUN
#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="$EXP"

MASTER="$FIXED_MASTER"

EVALUATOR="$EVALUATOR"

SELECTED_RUN="$SELECTED_RUN"
RANDOM_RUN="$RANDOM_RUN"
FULL_RUN="$FULL_RUN"
BASE_RUN="$BASE_RUN"

WMT="$WMT"
FLORES="$FLORES"
CHALLENGE="$CHALLENGE"

SUMMARY="$SUMMARY"


echo
echo "======================================================================"
echo "RQ2 RESUME — FULL6565 ONLY"
echo "======================================================================"


###############################################################################
# Train Full6565.
###############################################################################

if [[ ! -f "\$FULL_RUN/epoch3/config.json" ]]; then

    bash "\$MASTER"

fi


test -f \
  "\$FULL_RUN/epoch3/config.json"


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


echo "RQ2_EVAL_COMPLETE Full-OPD6565"


###############################################################################
# Final Selected / Random / Full summary.
###############################################################################

python - \
  "\$SELECTED_RUN" \
  "\$RANDOM_RUN" \
  "\$FULL_RUN" \
  "\$BASE_RUN" \
  "\$SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

import sacrebleu


selected_run = Path(sys.argv[1])
random_run = Path(sys.argv[2])
full_run = Path(sys.argv[3])
base_run = Path(sys.argv[4])
summary_path = Path(sys.argv[5])


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


def load_jsonl(path):

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

    rows.sort(
        key=lambda row:
            int(
                row[
                    "index"
                ]
            )
    )

    return rows


###############################################################################
# Reconstruct frozen Base with the same metric code.
###############################################################################

base_scores = {}


for split in splits:

    rows = load_jsonl(
        base_run
        / split
        / "predictions.jsonl"
    )

    refs = [
        row[
            "reference"
        ]
        for row in rows
    ]

    hyps = [
        row[
            "student_translation"
        ]
        for row in rows
    ]


    base_scores[
        split
    ] = {
        "BLEU":
            sacrebleu.corpus_bleu(
                hyps,
                [refs],
            ).score,

        "chrF":
            sacrebleu.corpus_chrf(
                hyps,
                [refs],
            ).score,
    }


print(
    "=" * 126
)

print(
    "RQ2 — DOES MT-PATCHER SOURCE SELECTION HELP CLEAN ON-POLICY DISTILLATION?"
)

print(
    "=" * 126
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


results = {}


for system, run in systems.items():

    bleus = []
    db = []
    dc = []


    for split in splits:

        metrics_path = (
            run
            / "rq2_eval_epoch3"
            / split
            / "metrics.json"
        )


        if not metrics_path.exists():

            raise RuntimeError(
                f"Missing metrics: {metrics_path}"
            )


        metric = json.loads(
            metrics_path.read_text(
                encoding="utf-8"
            )
        )


        bleu = float(
            metric[
                "BLEU"
            ]
        )

        chrf = float(
            metric[
                "chrF"
            ]
        )


        bleus.append(
            bleu
        )

        db.append(
            bleu
            - base_scores[
                split
            ][
                "BLEU"
            ]
        )

        dc.append(
            chrf
            - base_scores[
                split
            ][
                "chrF"
            ]
        )


    avg_db = sum(db) / 3
    avg_dc = sum(dc) / 3


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


selected_minus_random = (
    selected
    - random_v
)

full_minus_selected = (
    full_v
    - selected
)

full_minus_random = (
    full_v
    - random_v
)


print()

print(
    "=" * 126
)

print(
    "RQ2 PRIMARY COMPARISONS"
)

print(
    "=" * 126
)


print(
    f"Selected3732 - Random3732 = "
    f"{selected_minus_random:+.6f} BLEU"
)

print(
    f"Full6565 - Selected3732   = "
    f"{full_minus_selected:+.6f} BLEU"
)

print(
    f"Full6565 - Random3732     = "
    f"{full_minus_random:+.6f} BLEU"
)


print()

print(
    "INTERPRETATION"
)

print(
    "Selected > Random materially: "
    "MT-Patcher source selection helps current OPD."
)

print(
    "Selected ~= Random: "
    "source selection contributes little."
)

print(
    "Full >> both: "
    "coverage/quantity dominates selection."
)

print(
    "All ~= Base: "
    "current OPD mechanism itself remains the bottleneck."
)


summary = {
    "protocol":
        "fixed_epoch3",

    "selected_random_same_N":
        True,

    "full_is_reference_not_compute_matched":
        True,

    "systems":
        results,

    "selected_minus_random_bleu":
        selected_minus_random,

    "full_minus_selected_bleu":
        full_minus_selected,

    "full_minus_random_bleu":
        full_minus_random,
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
    summary_path
)

print()

print(
    "RQ2_SELECTION_OPD_ALL_PASS"
)
PY

BASHRUN


chmod +x \
  "$RESUME"

bash -n \
  "$RESUME"

echo "RQ2_FULL_RESUME_STATIC_PASS"


###############################################################################
# STAGE 7 — BACKGROUND LAUNCH
###############################################################################

echo
echo "======================================================================"
echo "STAGE 7/7 — LAUNCH FULL6565 RESUME"
echo "======================================================================"

RUNNING="$(
    pgrep -af \
    '[r]un_full6565_and_finish_rq2_v2.sh|[r]un_pgrkl_full6565_fixed_v2.sh' \
    || true
)"


if [[ -n "$RUNNING" ]]; then

    echo "Existing Full6565 repair process:"
    echo "$RUNNING"

    false

fi


nohup setsid bash "$RESUME" \
  > "$FULL_LOG" 2>&1 < /dev/null &


PID=$!


echo "PID=$PID"

echo "LOG=$FULL_LOG"

echo "RQ2_FULL6565_REPAIR_STARTED"


###############################################################################
# Verify we get past the exact old failure and into a real update.
###############################################################################

PASS=0


for ROUND in \
    1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do

    sleep 30

    echo
    echo "HEALTH_ROUND=$ROUND"


    grep -E \
'TRAIN_ROWS|OPD_TORCHNPU_PREFLIGHT_PASS|OPD_TRAINING_START|CLEAN_PG_RUNTIME_AUDIT_PASS|epoch=1 local_step=1/|expected 3732 rows|expected 6565 rows|Traceback|RuntimeError|ChildFailedError' \
    "$FULL_LOG" \
    2>/dev/null \
    | tail -n 60 \
    || true


    if grep -q \
    'CLEAN_PG_RUNTIME_AUDIT_PASS' \
    "$FULL_LOG" \
    2>/dev/null \
    && grep -q \
    'epoch=1 local_step=1/' \
    "$FULL_LOG" \
    2>/dev/null; then

        PASS=1

        break

    fi


    if grep -qE \
    'expected 3732 rows|Traceback|RuntimeError|ChildFailedError' \
    "$FULL_LOG" \
    2>/dev/null; then

        echo
        echo "FULL6565 REPAIR FAILED"

        tail -n 220 \
          "$FULL_LOG"

        false

    fi

done


if [[ "$PASS" -ne 1 ]]; then

    echo
    echo "Full6565 did not reach audited first update in health window."

    echo
    echo "This does not automatically mean the detached job died."

    echo
    echo "Current process state:"

    pgrep -af \
      'run_full6565_and_finish_rq2_v2.sh|run_pgrkl_full6565_fixed_v2.sh|train_opd_pgrkl_cleanroom_v2_full6565_torchnpu.py' \
      || true

    echo
    echo "Latest log:"

    tail -n 220 \
      "$FULL_LOG" \
      2>/dev/null \
      || true

else

    echo
    echo "======================================================================"
    echo "FULL6565 CARDINALITY FAILURE CLEARED"
    echo "======================================================================"

    echo "SELECTED3732_ALREADY_COMPLETE_PASS"
    echo "RANDOM3732_ALREADY_COMPLETE_PASS"
    echo "FULL6565_SOURCE_ONLY_DATA_PASS"
    echo "FULL6565_TRAINER_CARDINALITY_PATCH_PASS"
    echo "FULL6565_MASTER_CARDINALITY_PATCH_PASS"
    echo "CLEAN_PG_RUNTIME_AUDIT_PASS"
    echo "FULL6565_REAL_TRAINING_STARTED"

    echo
    echo "After Full6565 training:"
    echo "  epoch3 evaluation"
    echo "  Selected vs Random vs Full summary"
    echo "  RQ2_SELECTION_OPD_ALL_PASS"

    echo
    echo "LOG=$FULL_LOG"

    echo
    echo "RQ2_FULL6565_LONG_RUN_SAFE"

fi

