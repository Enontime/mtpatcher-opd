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

DATA="$DATA_ROOT/$EXP/patch_aware_k1_clean3732_v2.jsonl"

STUDENT="$MODEL_ROOT/Qwen3-0.6B"
TEACHER="$MODEL_ROOT/Qwen3-8B"

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
          --teacher-model "$TEACHER" \
          --train "$DATA" \
          --output-dir "$OUT" \
          --mask-mode "$MODE" \
          --lr 1e-6 \
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

    mkdir -p "$EVAL"


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


    sleep 20
}


run_one \
  patch \
  corrfkl_patch_pe3732_v1 \
  29671


run_one \
  random \
  corrfkl_random_pe3732_v1 \
  29673


run_one \
  halo1 \
  corrfkl_halo1_pe3732_v1 \
  29675


run_one \
  full \
  corrfkl_full_pe3732_v1 \
  29677


###############################################################################
# FINAL CONSOLIDATED SUMMARY
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
    "Patch-FKL":
        ROOT
        / "corrfkl_patch_pe3732_v1"
        / "eval_epoch3",

    "Random-FKL":
        ROOT
        / "corrfkl_random_pe3732_v1"
        / "eval_epoch3",

    "Halo1-FKL":
        ROOT
        / "corrfkl_halo1_pe3732_v1"
        / "eval_epoch3",

    "FullCorr-FKL":
        ROOT
        / "corrfkl_full_pe3732_v1"
        / "eval_epoch3",
}


BASE = (
    ROOT
    / "_verified_base_eval_v3"
)


def load_predictions(
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


splits = (
    "wmt24",
    "flores",
    "challenge",
)


base_scores = {}


for split in splits:

    rows = load_predictions(
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

    hyps = [
        x[
            "student_translation"
        ]
        for x in rows
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


print(
    "=" * 110
)

print(
    "CORRECTION-TRAJECTORY FKL FINAL RESULTS"
)

print(
    "=" * 110
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

    bleu_values = []

    delta_bleu = []
    delta_chrf = []


    for split in splits:

        metric = json.loads(
            (
                family
                / split
                / "metrics.json"
            ).read_text(
                encoding="utf-8"
            )
        )


        bleu_values.append(
            float(
                metric[
                    "BLEU"
                ]
            )
        )


        delta_bleu.append(
            float(
                metric[
                    "BLEU"
                ]
            )
            - base_scores[
                split
            ][
                "BLEU"
            ]
        )


        delta_chrf.append(
            float(
                metric[
                    "chrF"
                ]
            )
            - base_scores[
                split
            ][
                "chrF"
            ]
        )


    print(
        f"{system:18s} "
        f"{bleu_values[0]:10.6f} "
        f"{bleu_values[1]:10.6f} "
        f"{bleu_values[2]:10.6f} "
        f"{sum(delta_bleu)/3:+12.6f} "
        f"{sum(delta_chrf)/3:+12.6f}"
    )


print()
print(
    "CORRECTION_FKL_NIGHT_QUEUE_ALL_PASS"
)
PY

