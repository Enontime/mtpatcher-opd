#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

TRAINER="$ROOT/scripts/mtpatcher_v8/train_weighted_correction_nll_torchnpu_v1.py"

EVALUATOR="$ROOT/scripts/mtpatcher_v6/eval_correction_fkl_torchnpu_v1.py"

DATA="$DATA_ROOT/$EXP/patch_aware_k1_clean3732_v2.jsonl"

STUDENT="$MODEL_ROOT/Qwen3-0.6B"

WMT="$DATA_ROOT/pilot_v2_qwen3_06b/wmt24_zh_en998.jsonl"
FLORES="$DATA_ROOT/pilot_v2_qwen3_06b/flores_zh_en1012.jsonl"
CHALLENGE="$DATA_ROOT/pilot_v2_qwen3_06b/challenge_zh_en197.jsonl"


run_one () {

    MODE="$1"
    FACTOR="$2"
    NAME="$3"
    PORT="$4"

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
          --boost-mode "$MODE" \
          --boost-factor "$FACTOR" \
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
  2 \
  corrnll_patchboost2_pe3732_v1 \
  29721


run_one \
  patch \
  4 \
  corrnll_patchboost4_pe3732_v1 \
  29723


run_one \
  random \
  4 \
  corrnll_randomboost4_pe3732_v1 \
  29725


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
    "FullCorr-NLL":
        ROOT
        / "corrnll_full_pe3732_v1"
        / "eval_epoch3",

    "PatchBoost-2x":
        ROOT
        / "corrnll_patchboost2_pe3732_v1"
        / "eval_epoch3",

    "PatchBoost-4x":
        ROOT
        / "corrnll_patchboost4_pe3732_v1"
        / "eval_epoch3",

    "RandomBoost-4x":
        ROOT
        / "corrnll_randomboost4_pe3732_v1"
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


def load_predictions(path):

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


base_scores = {}


for split in SPLITS:

    rows = load_predictions(
        BASE
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
    "=" * 122
)

print(
    "MT-PATCHER PATCH-BOOST NLL FINAL RESULTS"
)

print(
    "=" * 122
)

print(
    f"{'SYSTEM':20s} "
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

        metrics_path = (
            family
            / split
            / "metrics.json"
        )


        if not metrics_path.exists():

            raise RuntimeError(
                f"Missing metrics: "
                f"{metrics_path}"
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


    results[
        system
    ] = {
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
        f"{system:20s} "
        f"{bleus[0]:10.6f} "
        f"{bleus[1]:10.6f} "
        f"{bleus[2]:10.6f} "
        f"{sum(db)/3:+12.6f} "
        f"{sum(dc)/3:+12.6f}"
    )


full = results[
    "FullCorr-NLL"
][
    "avg_delta_bleu"
]

p2 = results[
    "PatchBoost-2x"
][
    "avg_delta_bleu"
]

p4 = results[
    "PatchBoost-4x"
][
    "avg_delta_bleu"
]

r4 = results[
    "RandomBoost-4x"
][
    "avg_delta_bleu"
]


print()
print(
    "=" * 122
)

print(
    "CAUSAL COMPARISONS"
)

print(
    "=" * 122
)

print(
    f"PatchBoost2 - FullCorr  = "
    f"{p2-full:+.6f} BLEU"
)

print(
    f"PatchBoost4 - FullCorr  = "
    f"{p4-full:+.6f} BLEU"
)

print(
    f"PatchBoost4 - Patch2    = "
    f"{p4-p2:+.6f} BLEU"
)

print(
    f"PatchBoost4 - Random4   = "
    f"{p4-r4:+.6f} BLEU"
)

print(
    f"RandomBoost4 - FullCorr = "
    f"{r4-full:+.6f} BLEU"
)


print()
print(
    "REFERENCE FROZEN BASELINES"
)

print(
    "Patch-only NLL         = -0.644432"
)

print(
    "Random sparse NLL      = +0.150531"
)

print(
    "Halo1 NLL              = +0.099279"
)

print(
    "FullCorr NLL           = +0.285069"
)

print(
    "PE-SFT3732             = +0.390"
)

print(
    "SeqKD-Selected3732     = +1.182"
)

print(
    "SeqKD-Full6565         = +1.465"
)

print()
print(
    "PATCHBOOST_NLL_QUEUE_ALL_PASS"
)
PY

