#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

SCRIPT_DIR="$ROOT/scripts/mtpatcher_v10"

SRC_MASTER="$ROOT/scripts/mtpatcher_v4/run_opd_pgrkl_cleanroom_v2_oneclick.sh"
TRAINER="$ROOT/scripts/mtpatcher_v4/train_opd_pgrkl_cleanroom_v2_torchnpu.py"
EVALUATOR="$ROOT/scripts/mtpatcher_v6/eval_correction_fkl_torchnpu_v1.py"

FULL_POOL="$DATA_ROOT/$EXP/patch_pool6565_generation.jsonl"
SELECTED_DATA="$DATA_ROOT/$EXP/pe_k1_clean3732.jsonl"
RANDOM_CONTROL="$DATA_ROOT/$EXP/seqkd_equal3732_seed20260823.jsonl"

RANDOM_DATA="$DATA_ROOT/$EXP/opd_random_sources3732_seed20260823_v1.jsonl"
FULL_DATA="$DATA_ROOT/$EXP/opd_full_sources6565_v1.jsonl"

SELECTED_NAME="opd_torchnpu_pgrkl_cleanroom_pe3732_v2"
RANDOM_NAME="opd_torchnpu_pgrkl_random3732_v1"
FULL_NAME="opd_torchnpu_pgrkl_full6565_v1"

SELECTED_RUN="$RUN_ROOT/$EXP/$SELECTED_NAME"
RANDOM_RUN="$RUN_ROOT/$EXP/$RANDOM_NAME"
FULL_RUN="$RUN_ROOT/$EXP/$FULL_NAME"

RANDOM_MASTER="$SCRIPT_DIR/run_pgrkl_random3732_v1.sh"
FULL_MASTER="$SCRIPT_DIR/run_pgrkl_full6565_v1.sh"

RUNNER="$SCRIPT_DIR/run_rq2_selection_opd_queue_v1.sh"

QUEUE_LOG="$LOG_ROOT/$EXP/rq2_selection_opd_queue_v1.log"

mkdir -p \
  "$SCRIPT_DIR" \
  "$LOG_ROOT/$EXP"


###############################################################################
# 1. STATIC INPUT AUDIT
###############################################################################

echo
echo "======================================================================"
echo "STAGE 1/6 — INPUT AUDIT"
echo "======================================================================"

test -f "$SRC_MASTER"
test -f "$TRAINER"
test -f "$EVALUATOR"

test -f "$FULL_POOL"
test -f "$SELECTED_DATA"
test -f "$RANDOM_CONTROL"

test -d "$SELECTED_RUN"
test -f "$SELECTED_RUN/epoch3/config.json"

python -m py_compile \
  "$TRAINER" \
  "$EVALUATOR"

bash -n \
  "$SRC_MASTER"

echo "RQ2_INPUT_STATIC_PASS"


###############################################################################
# 2. BUILD SOURCE-ONLY RANDOM3732 + FULL6565 DATA
#
# Random subset is inherited from the already frozen SeqKD-Equal3732 control,
# so Selected-vs-Random uses the same source-selection control as earlier work.
#
# Target/reference fields are deliberately removed from the OPD training files.
###############################################################################

echo
echo "======================================================================"
echo "STAGE 2/6 — BUILD MATCHED OPD SOURCE SETS"
echo "======================================================================"

python - \
  "$FULL_POOL" \
  "$SELECTED_DATA" \
  "$RANDOM_CONTROL" \
  "$RANDOM_DATA" \
  "$FULL_DATA" <<'PY'
import hashlib
import json
import sys
from collections import defaultdict, deque
from pathlib import Path


full_path = Path(sys.argv[1])
selected_path = Path(sys.argv[2])
random_control_path = Path(sys.argv[3])

random_out = Path(sys.argv[4])
full_out = Path(sys.argv[5])


def load_jsonl(path):

    rows = []

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:

        for line_no, line in enumerate(
            f,
            start=1,
        ):

            if not line.strip():
                continue

            obj = json.loads(line)

            if not isinstance(obj, dict):

                raise RuntimeError(
                    f"Non-dict row: "
                    f"{path}:{line_no}"
                )

            rows.append(obj)

    return rows


def sha256(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):

            h.update(chunk)

    return h.hexdigest()


def get_messages(row):

    messages = row.get("messages")

    if (
        isinstance(messages, list)
        and messages
    ):
        return messages

    prompt = row.get("prompt")

    if (
        isinstance(prompt, list)
        and prompt
    ):
        return prompt

    raise RuntimeError(
        "Full-pool row lacks usable "
        "messages/prompt list"
    )


full_rows = load_jsonl(
    full_path
)

selected_rows = load_jsonl(
    selected_path
)

random_rows = load_jsonl(
    random_control_path
)


if len(full_rows) != 6565:

    raise RuntimeError(
        f"Expected full pool 6565, "
        f"got {len(full_rows)}"
    )


if len(selected_rows) != 3732:

    raise RuntimeError(
        f"Expected selected 3732, "
        f"got {len(selected_rows)}"
    )


if len(random_rows) != 3732:

    raise RuntimeError(
        f"Expected random 3732, "
        f"got {len(random_rows)}"
    )


###############################################################################
# Build stable lookup by original index where possible.
###############################################################################

by_index = {}

all_have_index = True


for row in full_rows:

    if "index" not in row:

        all_have_index = False
        break

    idx = int(row["index"])

    if idx in by_index:

        raise RuntimeError(
            f"Duplicate full-pool index: {idx}"
        )

    by_index[idx] = row


###############################################################################
# Fallback lookup supports duplicate source strings.
###############################################################################

by_source = defaultdict(deque)


for row in full_rows:

    source = str(
        row.get(
            "source",
            "",
        )
    )

    if not source:

        raise RuntimeError(
            "Empty source in full pool"
        )

    by_source[source].append(row)


def resolve_full_row(control_row):

    if (
        all_have_index
        and "index" in control_row
    ):

        idx = int(
            control_row["index"]
        )

        if idx not in by_index:

            raise RuntimeError(
                f"Control index absent "
                f"from full pool: {idx}"
            )

        full_row = by_index[idx]

        control_source = str(
            control_row.get(
                "source",
                "",
            )
        )

        full_source = str(
            full_row.get(
                "source",
                "",
            )
        )

        if (
            control_source
            and control_source != full_source
        ):

            raise RuntimeError(
                f"Index/source mismatch "
                f"for index {idx}"
            )

        return full_row


    source = str(
        control_row.get(
            "source",
            "",
        )
    )

    if not source:

        raise RuntimeError(
            "Control row lacks both "
            "usable index and source"
        )


    candidates = by_source.get(
        source
    )


    if not candidates:

        raise RuntimeError(
            "Control source absent "
            "from full pool"
        )


    return candidates[0]


def source_only(full_row, method):

    messages = get_messages(
        full_row
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
            "Training prompt leaks "
            "assistant target"
        )


    return {
        "index":
            int(
                full_row[
                    "index"
                ]
            )
            if "index" in full_row
            else None,

        "source":
            str(
                full_row[
                    "source"
                ]
            ),

        "messages":
            messages,

        "construction_method":
            method,
    }


###############################################################################
# Full source-only set.
###############################################################################

full_source_rows = [
    source_only(
        row,
        "opd_full_source_only",
    )
    for row in full_rows
]


###############################################################################
# Random source-only set.
###############################################################################

random_source_rows = []


for control_row in random_rows:

    full_row = resolve_full_row(
        control_row
    )

    random_source_rows.append(
        source_only(
            full_row,
            "opd_random_equal3732_source_only",
        )
    )


###############################################################################
# Selection/source-set audit.
###############################################################################

selected_indices = set()

for row in selected_rows:

    if "index" not in row:

        raise RuntimeError(
            "Selected PE rows lack index; "
            "cannot perform matched audit"
        )

    selected_indices.add(
        int(
            row[
                "index"
            ]
        )
    )


random_indices = [
    int(
        row[
            "index"
        ]
    )
    for row in random_source_rows
]


if len(
    set(
        random_indices
    )
) != 3732:

    raise RuntimeError(
        "Random source set contains "
        "duplicate original indices"
    )


if len(
    selected_indices
) != 3732:

    raise RuntimeError(
        "Selected source set contains "
        "duplicate original indices"
    )


random_index_set = set(
    random_indices
)

overlap = (
    selected_indices
    & random_index_set
)


if random_index_set == selected_indices:

    raise RuntimeError(
        "Random and selected source sets "
        "are accidentally identical"
    )


###############################################################################
# Save.
###############################################################################

random_out.parent.mkdir(
    parents=True,
    exist_ok=True,
)

full_out.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with random_out.open(
    "w",
    encoding="utf-8",
) as f:

    for row in random_source_rows:

        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


with full_out.open(
    "w",
    encoding="utf-8",
) as f:

    for row in full_source_rows:

        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


print(
    "RQ2_SOURCE_SET_AUDIT =",
    {
        "full_rows":
            len(
                full_source_rows
            ),

        "selected_rows":
            len(
                selected_indices
            ),

        "random_rows":
            len(
                random_index_set
            ),

        "selected_random_overlap":
            len(
                overlap
            ),

        "selected_random_overlap_fraction":
            len(
                overlap
            )
            / 3732,
    },
)


print(
    "RANDOM_DATA =",
    random_out,
)

print(
    "RANDOM_SHA256 =",
    sha256(
        random_out
    ),
)

print(
    "FULL_DATA =",
    full_out,
)

print(
    "FULL_SHA256 =",
    sha256(
        full_out
    ),
)

print(
    "RQ2_SOURCE_DATA_BUILD_PASS"
)
PY


###############################################################################
# 3. CLONE VALIDATED CLEAN PG-RKL MASTER
#
# Change ONLY:
#   NAME
#   TRAIN
#   HCCL port
#
# Student / Teacher / trainer / hyperparameters remain untouched.
###############################################################################

echo
echo "======================================================================"
echo "STAGE 3/6 — CLONE VALIDATED CLEAN PG-RKL"
echo "======================================================================"

export \
  SRC_MASTER \
  RANDOM_MASTER \
  FULL_MASTER \
  RANDOM_DATA \
  FULL_DATA \
  RANDOM_NAME \
  FULL_NAME

python - <<'PY'
import os
import re
from pathlib import Path


src_path = Path(
    os.environ[
        "SRC_MASTER"
    ]
)

src = src_path.read_text(
    encoding="utf-8"
)


OLD_NAME = (
    "opd_torchnpu_pgrkl_cleanroom_pe3732_v2"
)


if OLD_NAME not in src:

    raise RuntimeError(
        "Validated clean PG-RKL "
        "run name not found"
    )


if (
    "train_opd_pgrkl_cleanroom_v2_torchnpu.py"
    not in src
):

    raise RuntimeError(
        "Validated clean PG-RKL trainer "
        "reference missing"
    )


def make_clone(
    dst_path,
    new_name,
    new_train,
    new_port,
):

    text = src


    ############################################################################
    # Run name.
    ############################################################################

    text = text.replace(
        OLD_NAME,
        new_name,
    )


    ############################################################################
    # Training dataset assignment.
    ############################################################################

    train_pattern = re.compile(
        r'(?m)^'
        r'(\s*TRAIN\s*=\s*)'
        r'["\'][^"\']*'
        r'pe_k1_clean3732\.jsonl'
        r'["\']'
        r'\s*$'
    )


    text, train_n = (
        train_pattern.subn(
            lambda m:
                f'{m.group(1)}'
                f'"{new_train}"',
            text,
        )
    )


    if train_n != 1:

        raise RuntimeError(
            f"TRAIN replacement count "
            f"for {new_name}: "
            f"{train_n}"
        )


    ############################################################################
    # Fresh distributed port.
    ############################################################################

    port_n = 0


    for pattern, repl in (
        (
            r'(--master_port=)\d+',
            rf'\g<1>{new_port}',
        ),
        (
            r'(--master_port\s+)\d+',
            rf'\g<1>{new_port}',
        ),
        (
            r'(?m)^'
            r'(\s*MASTER_PORT\s*=\s*)'
            r'\d+\s*$',
            rf'\g<1>{new_port}',
        ),
    ):

        text, n = re.subn(
            pattern,
            repl,
            text,
        )

        port_n += n


    if port_n < 1:

        raise RuntimeError(
            f"Could not replace "
            f"distributed port "
            f"for {new_name}"
        )


    ############################################################################
    # Scientific invariants.
    ############################################################################

    required = (
        "train_opd_pgrkl_cleanroom_v2_torchnpu.py",
        "Qwen3-0.6B",
        "Qwen3-8B",
        new_name,
        str(
            new_train
        ),
    )


    for marker in required:

        if marker not in text:

            raise RuntimeError(
                f"Missing invariant "
                f"marker in {new_name}: "
                f"{marker}"
            )


    dst = Path(
        dst_path
    )

    dst.write_text(
        text,
        encoding="utf-8",
    )


    print(
        "MASTER_CLONE =",
        {
            "path":
                str(
                    dst
                ),

            "name":
                new_name,

            "train":
                str(
                    new_train
                ),

            "port":
                new_port,

            "train_replacements":
                train_n,

            "port_replacements":
                port_n,
        },
    )


make_clone(
    os.environ[
        "RANDOM_MASTER"
    ],
    os.environ[
        "RANDOM_NAME"
    ],
    os.environ[
        "RANDOM_DATA"
    ],
    29761,
)


make_clone(
    os.environ[
        "FULL_MASTER"
    ],
    os.environ[
        "FULL_NAME"
    ],
    os.environ[
        "FULL_DATA"
    ],
    29763,
)


print(
    "RQ2_MASTER_CLONE_PASS"
)
PY


chmod +x \
  "$RANDOM_MASTER" \
  "$FULL_MASTER"

bash -n \
  "$RANDOM_MASTER"

bash -n \
  "$FULL_MASTER"


echo
echo "===== RANDOM MASTER KEY LINES ====="

grep -nE \
'NAME=|TRAIN=|STUDENT=|TEACHER=|master_port|MASTER_PORT|train_opd_pgrkl_cleanroom' \
"$RANDOM_MASTER" \
| head -n 80


echo
echo "===== FULL MASTER KEY LINES ====="

grep -nE \
'NAME=|TRAIN=|STUDENT=|TEACHER=|master_port|MASTER_PORT|train_opd_pgrkl_cleanroom' \
"$FULL_MASTER" \
| head -n 80


echo "RQ2_MASTER_STATIC_PASS"


###############################################################################
# 4. BUILD SERIAL RUNNER
###############################################################################

echo
echo "======================================================================"
echo "STAGE 4/6 — BUILD SERIAL RQ2 RUNNER"
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

SELECTED_NAME="$SELECTED_NAME"
RANDOM_NAME="$RANDOM_NAME"
FULL_NAME="$FULL_NAME"

SELECTED_RUN="$SELECTED_RUN"
RANDOM_RUN="$RANDOM_RUN"
FULL_RUN="$FULL_RUN"

RANDOM_MASTER="$RANDOM_MASTER"
FULL_MASTER="$FULL_MASTER"

EVALUATOR="$EVALUATOR"

WMT="$DATA_ROOT/pilot_v2_qwen3_06b/wmt24_zh_en998.jsonl"
FLORES="$DATA_ROOT/pilot_v2_qwen3_06b/flores_zh_en1012.jsonl"
CHALLENGE="$DATA_ROOT/pilot_v2_qwen3_06b/challenge_zh_en197.jsonl"


###############################################################################
# Helper: fixed epoch3 evaluation.
###############################################################################

eval_epoch3 () {

    SYSTEM_NAME="\$1"
    RUN_DIR="\$2"

    MODEL="\$RUN_DIR/epoch3"

    test -f \
      "\$MODEL/config.json"

    EVAL="\$RUN_DIR/rq2_eval_epoch3"

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
          --model "\$MODEL" \
          --input "\$INPUT" \
          --output "\$EVAL/\$SPLIT/predictions.jsonl" \
          --metrics "\$EVAL/\$SPLIT/metrics.json" \
          --method "\${SYSTEM_NAME}_rq2_epoch3_\${SPLIT}" \
          --batch-size 16 \
          --max-new-tokens 256

    done


    echo "RQ2_EVAL_COMPLETE \$SYSTEM_NAME"
}


###############################################################################
# Selected — existing frozen checkpoint, re-evaluate only.
###############################################################################

echo
echo "======================================================================"
echo "RQ2 SELECTED3732 — RE-EVAL FROZEN EPOCH3"
echo "======================================================================"

eval_epoch3 \
  "Selected-OPD3732" \
  "\$SELECTED_RUN"


###############################################################################
# Random3732.
###############################################################################

echo
echo "======================================================================"
echo "RQ2 RANDOM3732 — CLEAN PG-RKL"
echo "======================================================================"

if [[ ! -f "\$RANDOM_RUN/epoch3/config.json" ]]; then

    bash "\$RANDOM_MASTER"

fi


test -f \
  "\$RANDOM_RUN/epoch3/config.json"


eval_epoch3 \
  "Random-OPD3732" \
  "\$RANDOM_RUN"


###############################################################################
# Full6565.
###############################################################################

echo
echo "======================================================================"
echo "RQ2 FULL6565 — CLEAN PG-RKL"
echo "======================================================================"

if [[ ! -f "\$FULL_RUN/epoch3/config.json" ]]; then

    bash "\$FULL_MASTER"

fi


test -f \
  "\$FULL_RUN/epoch3/config.json"


eval_epoch3 \
  "Full-OPD6565" \
  "\$FULL_RUN"


###############################################################################
# Final comparison.
###############################################################################

python - \
  "\$SELECTED_RUN" \
  "\$RANDOM_RUN" \
  "\$FULL_RUN" \
  "\$RUN_ROOT/\$EXP/_verified_base_eval_v3" <<'PY'
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

base_root = Path(
    sys.argv[4]
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


def load_jsonl(path):

    rows = []

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:

        for line in f:

            if line.strip():

                rows.append(
                    json.loads(
                        line
                    )
                )

    rows.sort(
        key=lambda x:
            int(
                x[
                    "index"
                ]
            )
    )

    return rows


###############################################################################
# Reconstruct frozen Base using the exact existing predictions.
###############################################################################

base_scores = {}


for split in splits:

    rows = load_jsonl(
        base_root
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
                [
                    refs
                ],
            ).score,

        "chrF":
            sacrebleu.corpus_chrf(
                hyps,
                [
                    refs
                ],
            ).score,
    }


results = {}


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
    "PRIMARY PROTOCOL: FIXED EPOCH3 FOR ALL SYSTEMS"
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


sizes = {
    "Selected-OPD3732":
        3732,

    "Random-OPD3732":
        3732,

    "Full-OPD6565":
        6565,
}


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


print()
print(
    "=" * 126
)

print(
    "CAUSAL / REFERENCE COMPARISONS"
)

print(
    "=" * 126
)


print(
    f"Selected3732 - Random3732 = "
    f"{selected-random_v:+.6f} BLEU"
)

print(
    f"Full6565 - Selected3732   = "
    f"{full_v-selected:+.6f} BLEU"
)

print(
    f"Full6565 - Random3732     = "
    f"{full_v-random_v:+.6f} BLEU"
)


print()
print(
    "INTERPRETATION RULE"
)

print(
    "Selected > Random materially: "
    "MT-Patcher source selection helps OPD."
)

print(
    "Selected ~= Random: "
    "selection contributes little to current OPD."
)

print(
    "Full >> Selected/Random: "
    "data coverage/quantity matters more than selection."
)

print(
    "All ~= Base: "
    "current clean PG-RKL itself is the bottleneck."
)


summary = {
    "protocol":
        "fixed epoch3",

    "systems":
        results,

    "selected_minus_random_bleu":
        selected
        - random_v,

    "full_minus_selected_bleu":
        full_v
        - selected,

    "full_minus_random_bleu":
        full_v
        - random_v,
}


summary_path = (
    full_run.parent
    / "rq2_selection_opd_summary_v1.json"
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

echo "RQ2_RUNNER_STATIC_PASS"


###############################################################################
# 5. DUPLICATE / FRESH OUTPUT GATES
###############################################################################

echo
echo "======================================================================"
echo "STAGE 5/6 — PROCESS AND OUTPUT GATES"
echo "======================================================================"

RUNNING="$(
    pgrep -af \
    '[r]un_rq2_selection_opd_queue_v1.sh|[r]un_pgrkl_random3732_v1.sh|[r]un_pgrkl_full6565_v1.sh' \
    || true
)"


if [[ -n "$RUNNING" ]]; then

    echo "Existing RQ2 process:"
    echo "$RUNNING"

    false

fi


if [[ -e "$RANDOM_RUN" ]]; then

    echo "Fresh Random-OPD output already exists:"
    echo "$RANDOM_RUN"

    false

fi


if [[ -e "$FULL_RUN" ]]; then

    echo "Fresh Full-OPD output already exists:"
    echo "$FULL_RUN"

    false

fi


echo "RQ2_FRESH_OUTPUT_GATE_PASS"


###############################################################################
# 6. LAUNCH
###############################################################################

echo
echo "======================================================================"
echo "STAGE 6/6 — LAUNCH RQ2 SELECTION OPD QUEUE"
echo "======================================================================"

nohup setsid bash "$RUNNER" \
  > "$QUEUE_LOG" 2>&1 < /dev/null &


PID=$!


echo "PID=$PID"

echo "QUEUE_LOG=$QUEUE_LOG"

echo "RQ2_SELECTION_OPD_QUEUE_STARTED"


###############################################################################
# Wait for Random-OPD to enter real training.
###############################################################################

PASS=0


for ROUND in \
    1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do

    sleep 30

    echo
    echo "HEALTH_ROUND=$ROUND"


    grep -E \
'CLEAN_PG_RUNTIME_AUDIT_PASS|OPD_TRAINING_START|epoch=1 local_step=1/|Traceback|RuntimeError|ChildFailedError|FAILED' \
    "$QUEUE_LOG" \
    2>/dev/null \
    | tail -n 50 \
    || true


    if grep -q \
    'CLEAN_PG_RUNTIME_AUDIT_PASS' \
    "$QUEUE_LOG" \
    2>/dev/null \
    && grep -q \
    'epoch=1 local_step=1/' \
    "$QUEUE_LOG" \
    2>/dev/null; then

        PASS=1

        break

    fi


    if grep -qE \
    'Traceback|RuntimeError|ChildFailedError' \
    "$QUEUE_LOG" \
    2>/dev/null; then

        echo
        echo "RQ2 QUEUE FAILED"

        tail -n 220 \
          "$QUEUE_LOG"

        false

    fi


    Q="$(
        pgrep -af \
        '[r]un_rq2_selection_opd_queue_v1.sh' \
        || true
    )"


    echo \
      "QUEUE_ALIVE=$([[ -n "$Q" ]] && echo YES || echo NO)"


    if [[ -z "$Q" ]]; then

        echo
        echo "RQ2 queue died before Random-OPD first update."

        tail -n 220 \
          "$QUEUE_LOG" \
          2>/dev/null \
          || true

        false

    fi

done


if [[ "$PASS" -ne 1 ]]; then

    echo
    echo "RQ2 queue did not reach Random-OPD audited first update."

    tail -n 220 \
      "$QUEUE_LOG" \
      2>/dev/null \
      || true

    false

fi


echo
echo "======================================================================"
echo "RQ2 SELECTION OPD LONG RUN VERIFIED"
echo "======================================================================"

echo "RQ2_SOURCE_DATA_BUILD_PASS"
echo "RQ2_MASTER_STATIC_PASS"
echo "RQ2_RUNNER_STATIC_PASS"
echo "RQ2_FRESH_OUTPUT_GATE_PASS"
echo "CLEAN_PG_RUNTIME_AUDIT_PASS"
echo "RQ2_SELECTION_OPD_LONG_QUEUE_RUNNING"

echo
echo "Sequence:"
echo "  1. Selected3732 frozen epoch3 re-eval"
echo "  2. Random3732 clean PG-RKL"
echo "  3. Random3732 epoch3 eval"
echo "  4. Full6565 clean PG-RKL"
echo "  5. Full6565 epoch3 eval"
echo "  6. Fixed-epoch3 RQ2 summary"

echo
echo "QUEUE_LOG=$QUEUE_LOG"

echo
echo "RQ2_SELECTION_OPD_LONG_RUN_SAFE"

