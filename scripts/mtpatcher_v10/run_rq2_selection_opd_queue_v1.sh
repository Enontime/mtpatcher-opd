#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

SELECTED_NAME="opd_torchnpu_pgrkl_cleanroom_pe3732_v2"
RANDOM_NAME="opd_torchnpu_pgrkl_random3732_v1"
FULL_NAME="opd_torchnpu_pgrkl_full6565_v1"

SELECTED_RUN="/workspace/mtpatcher/runs/mtpatcher_v3_full6565_20260823/opd_torchnpu_pgrkl_cleanroom_pe3732_v2"
RANDOM_RUN="/workspace/mtpatcher/runs/mtpatcher_v3_full6565_20260823/opd_torchnpu_pgrkl_random3732_v1"
FULL_RUN="/workspace/mtpatcher/runs/mtpatcher_v3_full6565_20260823/opd_torchnpu_pgrkl_full6565_v1"

RANDOM_MASTER="/workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend/scripts/mtpatcher_v10/run_pgrkl_random3732_v1.sh"
FULL_MASTER="/workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend/scripts/mtpatcher_v10/run_pgrkl_full6565_v1.sh"

EVALUATOR="/workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend/scripts/mtpatcher_v6/eval_correction_fkl_torchnpu_v1.py"

WMT="/workspace/mtpatcher/data/pilot_v2_qwen3_06b/wmt24_zh_en998.jsonl"
FLORES="/workspace/mtpatcher/data/pilot_v2_qwen3_06b/flores_zh_en1012.jsonl"
CHALLENGE="/workspace/mtpatcher/data/pilot_v2_qwen3_06b/challenge_zh_en197.jsonl"


###############################################################################
# Helper: fixed epoch3 evaluation.
###############################################################################

eval_epoch3 () {

    SYSTEM_NAME="$1"
    RUN_DIR="$2"

    MODEL="$RUN_DIR/epoch3"

    test -f       "$MODEL/config.json"

    EVAL="$RUN_DIR/rq2_eval_epoch3"

    mkdir -p       "$EVAL"


    for SPEC in       "wmt24:$WMT"       "flores:$FLORES"       "challenge:$CHALLENGE"
    do

        SPLIT="${SPEC%%:*}"
        INPUT="${SPEC#*:}"

        mkdir -p           "$EVAL/$SPLIT"


        python "$EVALUATOR"           --model "$MODEL"           --input "$INPUT"           --output "$EVAL/$SPLIT/predictions.jsonl"           --metrics "$EVAL/$SPLIT/metrics.json"           --method "${SYSTEM_NAME}_rq2_epoch3_${SPLIT}"           --batch-size 16           --max-new-tokens 256

    done


    echo "RQ2_EVAL_COMPLETE $SYSTEM_NAME"
}


###############################################################################
# Selected — existing frozen checkpoint, re-evaluate only.
###############################################################################

echo
echo "======================================================================"
echo "RQ2 SELECTED3732 — RE-EVAL FROZEN EPOCH3"
echo "======================================================================"

eval_epoch3   "Selected-OPD3732"   "$SELECTED_RUN"


###############################################################################
# Random3732.
###############################################################################

echo
echo "======================================================================"
echo "RQ2 RANDOM3732 — CLEAN PG-RKL"
echo "======================================================================"

if [[ ! -f "$RANDOM_RUN/epoch3/config.json" ]]; then

    bash "$RANDOM_MASTER"

fi


test -f   "$RANDOM_RUN/epoch3/config.json"


eval_epoch3   "Random-OPD3732"   "$RANDOM_RUN"


###############################################################################
# Full6565.
###############################################################################

echo
echo "======================================================================"
echo "RQ2 FULL6565 — CLEAN PG-RKL"
echo "======================================================================"

if [[ ! -f "$FULL_RUN/epoch3/config.json" ]]; then

    bash "$FULL_MASTER"

fi


test -f   "$FULL_RUN/epoch3/config.json"


eval_epoch3   "Full-OPD6565"   "$FULL_RUN"


###############################################################################
# Final comparison.
###############################################################################

python -   "$SELECTED_RUN"   "$RANDOM_RUN"   "$FULL_RUN"   "$RUN_ROOT/$EXP/_verified_base_eval_v3" <<'PY'
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

