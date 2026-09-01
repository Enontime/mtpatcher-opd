#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

TRAINER="$ROOT/scripts/mtpatcher_v7/train_correction_nll_torchnpu_v1.py"

EVALUATOR="$ROOT/scripts/mtpatcher_v6/eval_correction_fkl_torchnpu_v1.py"

DATA="$DATA_ROOT/$EXP/patch_aware_k1_clean3732_v2.jsonl"

STUDENT="$MODEL_ROOT/Qwen3-0.6B"

WMT="$DATA_ROOT/pilot_v2_qwen3_06b/wmt24_zh_en998.jsonl"
FLORES="$DATA_ROOT/pilot_v2_qwen3_06b/flores_zh_en1012.jsonl"
CHALLENGE="$DATA_ROOT/pilot_v2_qwen3_06b/challenge_zh_en197.jsonl"


run_one () {

    MODE="$1"
    NAME="$2"
    PORT="$3"

    OUT="$RUN_ROOT/$EXP/$NAME"

    TRAIN_LOG="$LOG_ROOT/$EXP/${NAME}.log"


    echo
    echo "======================================================================"
    echo "START $NAME"
    date
    echo "======================================================================"


    if [[ ! -f "$OUT/training_manifest.json" ]]; then

        python -m torch.distributed.run \
          --nproc_per_node=16 \
          --master_port="$PORT" \
          "$TRAINER" \
          --student-model "$STUDENT" \
          --train "$DATA" \
          --output-dir "$OUT" \
          --mask-mode "$MODE" \
          --lr 2e-5 \
          --epochs 3 \
          --max-length 1024 \
          --warmup-ratio 0.03 \
          --weight-decay 0.01 \
          --max-grad-norm 1.0 \
          --seed 20260825 \
          > "$TRAIN_LOG" 2>&1

    fi


    test -f \
      "$OUT/training_manifest.json"

    test -f \
      "$OUT/epoch3/config.json"


    EVAL="$OUT/eval_epoch3"

    mkdir -p \
      "$EVAL"


    for SPEC in \
      "wmt24:$WMT" \
      "flores:$FLORES" \
      "challenge:$CHALLENGE"
    do

        SPLIT="${SPEC%%:*}"
        INPUT="${SPEC#*:}"

        mkdir -p \
          "$EVAL/$SPLIT"


        python "$EVALUATOR" \
          --model "$OUT/epoch3" \
          --input "$INPUT" \
          --output "$EVAL/$SPLIT/predictions.jsonl" \
          --metrics "$EVAL/$SPLIT/metrics.json" \
          --method "${NAME}_${SPLIT}" \
          --batch-size 16 \
          --max-new-tokens 256

    done


    echo
    echo "======================================================================"
    echo "COMPLETE $NAME"
    date
    echo "======================================================================"


    sleep 10
}


run_one \
  patch \
  corrnll_patch_pe3732_v1 \
  29701


run_one \
  random \
  corrnll_random_pe3732_v1 \
  29703


run_one \
  halo1 \
  corrnll_halo1_pe3732_v1 \
  29705


run_one \
  full \
  corrnll_full_pe3732_v1 \
  29707


###############################################################################
# FINAL SUMMARY
###############################################################################

python - <<'PY'
import json
import os
from pathlib import Path

import sacrebleu


RUN_ROOT = Path(
    os.environ[
        "RUN_ROOT"
    ]
)

EXP = (
    "mtpatcher_v3_full6565_20260823"
)

ROOT = RUN_ROOT / EXP


SYSTEMS = {
    "Patch-NLL":
        ROOT
        / "corrnll_patch_pe3732_v1"
        / "eval_epoch3",

    "Random-NLL":
        ROOT
        / "corrnll_random_pe3732_v1"
        / "eval_epoch3",

    "Halo1-NLL":
        ROOT
        / "corrnll_halo1_pe3732_v1"
        / "eval_epoch3",

    "FullCorr-NLL":
        ROOT
        / "corrnll_full_pe3732_v1"
        / "eval_epoch3",
}


BASE = (
    ROOT
    / "_verified_base_eval_v3"
)


SPLITS = (
    "wmt24",
    "flores",
    "challenge",
)


def load(
    path
):

    rows = []

    with Path(path).open(
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
        key=lambda x: int(
            x[
                "index"
            ]
        )
    )

    return rows


base = {}


for split in SPLITS:

    rows = load(
        BASE
        / split
        / "predictions.jsonl"
    )

    refs = [
        x[
            "reference"
        ]
        for x in rows
    ]

    hyp = [
        x[
            "student_translation"
        ]
        for x in rows
    ]


    base[
        split
    ] = {
        "BLEU":
            sacrebleu.corpus_bleu(
                hyp,
                [
                    refs
                ],
            ).score,

        "chrF":
            sacrebleu.corpus_chrf(
                hyp,
                [
                    refs
                ],
            ).score,
    }


results = {}


print(
    "=" * 118
)

print(
    "CORRECTION-TOKEN NLL FINAL RESULTS"
)

print(
    "=" * 118
)

print(
    f"{'SYSTEM':18s} "
    f"{'WMT':>10s} "
    f"{'FLORES':>10s} "
    f"{'CHALL':>10s} "
    f"{'AVG ΔBLEU':>12s} "
    f"{'AVG ΔchrF':>12s}"
)


for system, family in SYSTEMS.items():

    bleus = []
    db = []
    dc = []


    for split in SPLITS:

        metric = json.loads(
            (
                family
                / split
                / "metrics.json"
            ).read_text(
                encoding="utf-8"
            )
        )


        b = float(
            metric[
                "BLEU"
            ]
        )

        c = float(
            metric[
                "chrF"
            ]
        )


        bleus.append(
            b
        )

        db.append(
            b
            - base[
                split
            ][
                "BLEU"
            ]
        )

        dc.append(
            c
            - base[
                split
            ][
                "chrF"
            ]
        )


    results[
        system
    ] = {
        "wmt":
            bleus[
                0
            ],

        "flores":
            bleus[
                1
            ],

        "challenge":
            bleus[
                2
            ],

        "avg_delta_bleu":
            sum(
                db
            )
            / 3,

        "avg_delta_chrf":
            sum(
                dc
            )
            / 3,
    }


    print(
        f"{system:18s} "
        f"{bleus[0]:10.6f} "
        f"{bleus[1]:10.6f} "
        f"{bleus[2]:10.6f} "
        f"{sum(db)/3:+12.6f} "
        f"{sum(dc)/3:+12.6f}"
    )


print()
print(
    "=" * 118
)

print(
    "CAUSAL COMPARISONS"
)

print(
    "=" * 118
)


patch = results[
    "Patch-NLL"
][
    "avg_delta_bleu"
]

random_v = results[
    "Random-NLL"
][
    "avg_delta_bleu"
]

halo = results[
    "Halo1-NLL"
][
    "avg_delta_bleu"
]

full = results[
    "FullCorr-NLL"
][
    "avg_delta_bleu"
]


print(
    f"Patch - Random       = "
    f"{patch-random_v:+.6f} BLEU"
)

print(
    f"Halo1 - Patch        = "
    f"{halo-patch:+.6f} BLEU"
)

print(
    f"FullCorr - Patch     = "
    f"{full-patch:+.6f} BLEU"
)

print(
    f"FullCorr - Random    = "
    f"{full-random_v:+.6f} BLEU"
)


# Existing frozen baselines for direct context.
print()
print(
    "REFERENCE FROZEN BASELINES"
)

print(
    "PE-SFT3732 Avg ΔBLEU      = +0.390"
)

print(
    "SeqKD-Selected Avg ΔBLEU = +1.182"
)

print(
    "SeqKD-Full Avg ΔBLEU     = +1.465"
)

print()
print(
    "CORRECTION_NLL_QUEUE_ALL_PASS"
)
PY

