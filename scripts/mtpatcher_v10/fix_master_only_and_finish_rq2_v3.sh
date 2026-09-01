#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

###############################################################################
# Frozen components
###############################################################################

TRAINER="$ROOT/scripts/mtpatcher_v4/train_opd_pgrkl_cleanroom_v2_torchnpu.py"

OLD_FULL_MASTER="$ROOT/scripts/mtpatcher_v10/run_pgrkl_full6565_v1.sh"

FIXED_FULL_MASTER="$ROOT/scripts/mtpatcher_v10/run_pgrkl_full6565_masterfix_v3.sh"

EVALUATOR="$ROOT/scripts/mtpatcher_v6/eval_correction_fkl_torchnpu_v1.py"

###############################################################################
# Runs / data
###############################################################################

SELECTED_RUN="$RUN_ROOT/$EXP/opd_torchnpu_pgrkl_cleanroom_pe3732_v2"

RANDOM_RUN="$RUN_ROOT/$EXP/opd_torchnpu_pgrkl_random3732_v1"

FULL_RUN="$RUN_ROOT/$EXP/opd_torchnpu_pgrkl_full6565_v1"

FULL_DATA="$DATA_ROOT/$EXP/opd_full_sources6565_v1.jsonl"

BASE_RUN="$RUN_ROOT/$EXP/_verified_base_eval_v3"

WMT="$DATA_ROOT/pilot_v2_qwen3_06b/wmt24_zh_en998.jsonl"

FLORES="$DATA_ROOT/pilot_v2_qwen3_06b/flores_zh_en1012.jsonl"

CHALLENGE="$DATA_ROOT/pilot_v2_qwen3_06b/challenge_zh_en197.jsonl"

###############################################################################
# Resume / logs
###############################################################################

RESUME="$ROOT/scripts/mtpatcher_v10/run_full6565_finish_rq2_v3.sh"

LOG="$LOG_ROOT/$EXP/rq2_full6565_masterfix_v3.log"

SUMMARY="$RUN_ROOT/$EXP/rq2_selection_opd_summary_v3.json"

STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p \
  "$ROOT/scripts/mtpatcher_v10" \
  "$LOG_ROOT/$EXP"


###############################################################################
# STAGE 1 — Existing completed work must remain untouched
###############################################################################

echo
echo "======================================================================"
echo "STAGE 1/6 — VERIFY COMPLETED SELECTED / RANDOM"
echo "======================================================================"

test -f "$TRAINER"
test -f "$OLD_FULL_MASTER"
test -f "$EVALUATOR"
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

echo "SELECTED3732_FROZEN_PASS"
echo "RANDOM3732_FROZEN_PASS"


###############################################################################
# STAGE 2 — Verify Full dataset
###############################################################################

echo
echo "======================================================================"
echo "STAGE 2/6 — VERIFY FULL6565 DATA"
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
            f"Assistant-target leakage row={i}"
        )

    if row.get("index") is not None:

        indices.append(
            int(
                row["index"]
            )
        )


if indices:

    if len(
        set(indices)
    ) != len(indices):

        raise RuntimeError(
            "Duplicate original indices"
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
    },
)

print(
    "FULL6565_DATA_PASS"
)
PY


###############################################################################
# STAGE 3 — Inspect old master and patch ONLY row-count preflight
###############################################################################

echo
echo "======================================================================"
echo "STAGE 3/6 — PATCH MASTER CARDINALITY ONLY"
echo "======================================================================"

echo
echo "===== OLD MASTER 3732 OCCURRENCES ====="

grep -n \
  '3732' \
  "$OLD_FULL_MASTER" \
  || true


export \
  OLD_FULL_MASTER \
  FIXED_FULL_MASTER


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
        "FIXED_FULL_MASTER"
    ]
)


original = src.read_text(
    encoding="utf-8"
)

text = original


###############################################################################
# These are the ONLY classes of replacements allowed.
###############################################################################

rules = [
    (
        r'len\(\s*rows\s*\)\s*!=\s*3732',
        'len(rows) != 6565',
        'len_rows_ne',
    ),

    (
        r'len\(\s*rows\s*\)\s*==\s*3732',
        'len(rows) == 6565',
        'len_rows_eq',
    ),

    (
        r'assert\s+len\(\s*rows\s*\)\s*==\s*3732',
        'assert len(rows) == 6565',
        'assert_rows_eq',
    ),

    (
        r'expected\s+3732\s+rows',
        'expected 6565 rows',
        'error_text_lower',
    ),

    (
        r'Expected\s+3732\s+rows',
        'Expected 6565 rows',
        'error_text_upper',
    ),
]


counts = {}


for pattern, replacement, name in rules:

    text, n = re.subn(
        pattern,
        replacement,
        text,
    )

    counts[
        name
    ] = n


total = sum(
    counts.values()
)


print(
    "MASTER_CARDINALITY_PATCH_COUNTS =",
    counts,
)

print(
    "MASTER_CARDINALITY_TOTAL_REPLACEMENTS =",
    total,
)


if total < 1:

    raise RuntimeError(
        "No 3732 row-cardinality guard found "
        "in Full one-click master"
    )


###############################################################################
# Important invariants copied verbatim from old Full master.
###############################################################################

required_exact = [
    'NAME="opd_torchnpu_pgrkl_full6565_v1"',
    'STUDENT="$MODEL_ROOT/Qwen3-0.6B"',
    'TEACHER="$MODEL_ROOT/Qwen3-8B"',
    "opd_full_sources6565_v1.jsonl",
    "train_opd_pgrkl_cleanroom_v2_torchnpu.py",
]


for marker in required_exact:

    if marker not in text:

        raise RuntimeError(
            f"Missing experimental invariant: "
            f"{marker}"
        )


###############################################################################
# Student / teacher / trainer must be unchanged.
###############################################################################

for prefix in (
    "STUDENT=",
    "TEACHER=",
    "TRAINER=",
):

    old_lines = [
        line
        for line in original.splitlines()
        if line.strip().startswith(
            prefix
        )
    ]

    new_lines = [
        line
        for line in text.splitlines()
        if line.strip().startswith(
            prefix
        )
    ]

    if old_lines != new_lines:

        raise RuntimeError(
            f"Unexpected change to {prefix}"
        )


###############################################################################
# No stale row-cardinality assertion may remain.
#
# We deliberately do NOT globally forbid string "3732":
# filenames/comments may legitimately contain it.
###############################################################################

bad_patterns = [
    r'len\(\s*rows\s*\)\s*!=\s*3732',
    r'len\(\s*rows\s*\)\s*==\s*3732',
    r'expected\s+3732\s+rows',
    r'Expected\s+3732\s+rows',
]


for pattern in bad_patterns:

    if re.search(
        pattern,
        text,
    ):

        raise RuntimeError(
            f"Stale cardinality guard remains: "
            f"{pattern}"
        )


dst.write_text(
    text,
    encoding="utf-8",
)


print(
    "FULL6565_MASTER_ONLY_PATCH_PASS"
)
PY


chmod +x \
  "$FIXED_FULL_MASTER"

bash -n \
  "$FIXED_FULL_MASTER"


echo
echo "===== FIXED MASTER KEY LINES ====="

grep -nE \
'NAME=|TRAIN=|STUDENT=|TEACHER=|TRAINER=|len\(rows\)|expected .*rows|Expected .*rows|TRAIN_ROWS|master_port' \
"$FIXED_FULL_MASTER" \
| head -n 120


echo
echo "FULL6565_MASTER_STATIC_PASS"


###############################################################################
# STAGE 4 — Deal only with failed partial Full run
###############################################################################

echo
echo "======================================================================"
echo "STAGE 4/6 — FULL OUTPUT STATE"
echo "======================================================================"

if [[ -e "$FULL_RUN" ]] \
   && [[ ! -f "$FULL_RUN/epoch3/config.json" ]]; then

    PRESERVED="${FULL_RUN}.preflight_failed_${STAMP}"

    mv \
      "$FULL_RUN" \
      "$PRESERVED"

    echo \
      "PRESERVED_FAILED_FULL_RUN=$PRESERVED"

fi


if [[ -f "$FULL_RUN/epoch3/config.json" ]]; then

    echo "FULL6565_EPOCH3_ALREADY_EXISTS"

else

    echo "FULL6565_REQUIRES_TRAINING"

fi


###############################################################################
# STAGE 5 — Build Full-only runner + final RQ2 summary
###############################################################################

echo
echo "======================================================================"
echo "STAGE 5/6 — BUILD FULL-ONLY RESUME"
echo "======================================================================"

cat > "$RESUME" <<BASHRUN
#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="$EXP"

MASTER="$FIXED_FULL_MASTER"

EVALUATOR="$EVALUATOR"

SELECTED_RUN="$SELECTED_RUN"
RANDOM_RUN="$RANDOM_RUN"
FULL_RUN="$FULL_RUN"
BASE_RUN="$BASE_RUN"

WMT="$WMT"
FLORES="$FLORES"
CHALLENGE="$CHALLENGE"

SUMMARY="$SUMMARY"


###############################################################################
# Full6565 training
###############################################################################

echo
echo "======================================================================"
echo "RQ2 FULL6565 — CLEAN PG-RKL RESUME"
echo "======================================================================"


if [[ ! -f "\$FULL_RUN/epoch3/config.json" ]]; then

    bash "\$MASTER"

fi


test -f \
  "\$FULL_RUN/epoch3/config.json"


echo
echo "RQ2_FULL6565_TRAINING_COMPLETE"


###############################################################################
# Fixed epoch3 evaluation
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
# Final summary
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


selected_run = Path(
    sys.argv[1]
)

random_run = Path(
    sys.argv[2]
)

full_run = Path(
    sys.argv[3]
)

base_run = Path(
    sys.argv[4]
)

summary_path = Path(
    sys.argv[5]
)


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


def load_rows(path):

    rows = []

    with Path(path).open(
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
# Frozen Base metric reconstruction
###############################################################################

base_scores = {}


for split in splits:

    rows = load_rows(
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


###############################################################################
# RQ2 table
###############################################################################

print(
    "=" * 128
)

print(
    "RQ2 — MT-PATCHER SOURCE SELECTION vs RANDOM vs FULL OPD"
)

print(
    "=" * 128
)

print(
    "PRIMARY PROTOCOL = FIXED EPOCH3"
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
                f"Missing metric: "
                f"{metric_path}"
            )


        metric = json.loads(
            metric_path.read_text(
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


        delta_bleu.append(
            bleu
            - base_scores[
                split
            ][
                "BLEU"
            ]
        )


        delta_chrf.append(
            chrf
            - base_scores[
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


selected_random = (
    selected
    - random_v
)

full_selected = (
    full_v
    - selected
)

full_random = (
    full_v
    - random_v
)


print()

print(
    "=" * 128
)

print(
    "PRIMARY COMPARISONS"
)

print(
    "=" * 128
)


print(
    f"Selected3732 - Random3732 = "
    f"{selected_random:+.6f} BLEU"
)

print(
    f"Full6565 - Selected3732   = "
    f"{full_selected:+.6f} BLEU"
)

print(
    f"Full6565 - Random3732     = "
    f"{full_random:+.6f} BLEU"
)


print()

print(
    "NOTE:"
)

print(
    "Selected vs Random is the same-N selection comparison."
)

print(
    "Full6565 uses more examples and more updates, "
    "so it is a coverage/reference comparison."
)


summary = {
    "protocol":
        "fixed_epoch3",

    "selected_random_same_n":
        True,

    "full_compute_matched":
        False,

    "systems":
        results,

    "selected_minus_random_bleu":
        selected_random,

    "full_minus_selected_bleu":
        full_selected,

    "full_minus_random_bleu":
        full_random,
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
  "$RESUME"

bash -n \
  "$RESUME"

echo
echo "RQ2_FULL_RESUME_STATIC_PASS"


###############################################################################
# STAGE 6 — Detached launch
###############################################################################

echo
echo "======================================================================"
echo "STAGE 6/6 — LAUNCH FULL6565"
echo "======================================================================"

RUNNING="$(
    pgrep -af \
    '[r]un_full6565_finish_rq2_v3.sh|[r]un_pgrkl_full6565_masterfix_v3.sh' \
    || true
)"


if [[ -n "$RUNNING" ]]; then

    echo
    echo "A Full6565 resume job is already running:"

    echo "$RUNNING"

    false

fi


nohup setsid bash "$RESUME" \
  > "$LOG" 2>&1 < /dev/null &


PID=$!


echo
echo "PID=$PID"
echo "LOG=$LOG"

echo
echo "RQ2_FULL6565_MASTERFIX_STARTED"


###############################################################################
# Wait until the OLD error is definitely cleared.
###############################################################################

PASS=0


for ROUND in $(seq 1 24)
do

    sleep 30

    echo
    echo "HEALTH_ROUND=$ROUND"


    grep -E \
'TRAIN_ROWS|OPD_TORCHNPU_PREFLIGHT_PASS|OPD_TRAINING_START|CLEAN_PG_RUNTIME_AUDIT_PASS|epoch=1 local_step=1/|expected 3732 rows|Traceback|RuntimeError|ChildFailedError' \
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
      'expected 3732 rows|Traceback|RuntimeError|ChildFailedError' \
      "$LOG" \
      2>/dev/null; then

        echo
        echo "FULL6565 MASTER FIX FAILED"

        tail -n 240 \
          "$LOG"

        false

    fi

done


if [[ "$PASS" -eq 1 ]]; then

    echo
    echo "======================================================================"
    echo "FULL6565 REAL TRAINING VERIFIED"
    echo "======================================================================"

    echo "SELECTED3732_FROZEN_PASS"
    echo "RANDOM3732_FROZEN_PASS"
    echo "FULL6565_DATA_PASS"
    echo "FULL6565_MASTER_ONLY_PATCH_PASS"
    echo "CLEAN_PG_RUNTIME_AUDIT_PASS"
    echo "FULL6565_REAL_UPDATE_PASS"

    echo
    echo "The detached job will continue through:"
    echo "  Full6565 epoch1"
    echo "  Full6565 epoch2"
    echo "  Full6565 epoch3"
    echo "  WMT/FLORES/Challenge eval"
    echo "  final RQ2 summary"

    echo
    echo "RQ2_FULL6565_LONG_RUN_SAFE"

else

    echo
    echo "Health window ended before first audited update."

    echo
    echo "This alone does not mean the detached process died."

    echo
    echo "PROCESS:"

    pgrep -af \
      'run_full6565_finish_rq2_v3.sh|run_pgrkl_full6565_masterfix_v3.sh|train_opd_pgrkl_cleanroom_v2_torchnpu.py' \
      || true

    echo
    echo "LATEST LOG:"

    tail -n 240 \
      "$LOG" \
      2>/dev/null \
      || true

fi

