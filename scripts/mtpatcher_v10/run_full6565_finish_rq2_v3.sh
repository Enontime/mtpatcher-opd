#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

MASTER="/workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend/scripts/mtpatcher_v10/run_pgrkl_full6565_masterfix_v3.sh"

EVALUATOR="/workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend/scripts/mtpatcher_v6/eval_correction_fkl_torchnpu_v1.py"

SELECTED_RUN="/workspace/mtpatcher/runs/mtpatcher_v3_full6565_20260823/opd_torchnpu_pgrkl_cleanroom_pe3732_v2"
RANDOM_RUN="/workspace/mtpatcher/runs/mtpatcher_v3_full6565_20260823/opd_torchnpu_pgrkl_random3732_v1"
FULL_RUN="/workspace/mtpatcher/runs/mtpatcher_v3_full6565_20260823/opd_torchnpu_pgrkl_full6565_v1"
BASE_RUN="/workspace/mtpatcher/runs/mtpatcher_v3_full6565_20260823/_verified_base_eval_v3"

WMT="/workspace/mtpatcher/data/pilot_v2_qwen3_06b/wmt24_zh_en998.jsonl"
FLORES="/workspace/mtpatcher/data/pilot_v2_qwen3_06b/flores_zh_en1012.jsonl"
CHALLENGE="/workspace/mtpatcher/data/pilot_v2_qwen3_06b/challenge_zh_en197.jsonl"

SUMMARY="/workspace/mtpatcher/runs/mtpatcher_v3_full6565_20260823/rq2_selection_opd_summary_v3.json"


###############################################################################
# Full6565 training
###############################################################################

echo
echo "======================================================================"
echo "RQ2 FULL6565 — CLEAN PG-RKL RESUME"
echo "======================================================================"


if [[ ! -f "$FULL_RUN/epoch3/config.json" ]]; then

    bash "$MASTER"

fi


test -f   "$FULL_RUN/epoch3/config.json"


echo
echo "RQ2_FULL6565_TRAINING_COMPLETE"


###############################################################################
# Fixed epoch3 evaluation
###############################################################################

EVAL="$FULL_RUN/rq2_eval_epoch3"

mkdir -p   "$EVAL"


for SPEC in   "wmt24:$WMT"   "flores:$FLORES"   "challenge:$CHALLENGE"
do

    SPLIT="${SPEC%%:*}"
    INPUT="${SPEC#*:}"

    mkdir -p       "$EVAL/$SPLIT"


    python "$EVALUATOR"       --model "$FULL_RUN/epoch3"       --input "$INPUT"       --output "$EVAL/$SPLIT/predictions.jsonl"       --metrics "$EVAL/$SPLIT/metrics.json"       --method "Full-OPD6565_rq2_epoch3_${SPLIT}"       --batch-size 16       --max-new-tokens 256

done


echo
echo "RQ2_EVAL_COMPLETE Full-OPD6565"


###############################################################################
# Final summary
###############################################################################

python -   "$SELECTED_RUN"   "$RANDOM_RUN"   "$FULL_RUN"   "$BASE_RUN"   "$SUMMARY" <<'PY'
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

