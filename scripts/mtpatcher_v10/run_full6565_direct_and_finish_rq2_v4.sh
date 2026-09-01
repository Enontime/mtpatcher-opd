#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

TRAINER="/workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend/scripts/mtpatcher_v10/train_opd_pgrkl_cleanroom_full6565_v4_torchnpu.py"

STUDENT="/workspace/mtpatcher/models/Qwen3-0.6B"
TEACHER="/workspace/mtpatcher/models/Qwen3-8B"

TRAIN="/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/opd_full_sources6565_v1.jsonl"

SELECTED_RUN="/workspace/mtpatcher/runs/mtpatcher_v3_full6565_20260823/opd_torchnpu_pgrkl_cleanroom_pe3732_v2"
RANDOM_RUN="/workspace/mtpatcher/runs/mtpatcher_v3_full6565_20260823/opd_torchnpu_pgrkl_random3732_v1"
FULL_RUN="/workspace/mtpatcher/runs/mtpatcher_v3_full6565_20260823/opd_torchnpu_pgrkl_full6565_v1"

EVALUATOR="/workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend/scripts/mtpatcher_v6/eval_correction_fkl_torchnpu_v1.py"

WMT="/workspace/mtpatcher/data/pilot_v2_qwen3_06b/wmt24_zh_en998.jsonl"
FLORES="/workspace/mtpatcher/data/pilot_v2_qwen3_06b/flores_zh_en1012.jsonl"
CHALLENGE="/workspace/mtpatcher/data/pilot_v2_qwen3_06b/challenge_zh_en197.jsonl"

SUMMARY="/workspace/mtpatcher/runs/mtpatcher_v3_full6565_20260823/rq2_selection_opd_summary_v4.json"


echo
echo "======================================================================"
echo "RQ2 FULL6565 — DIRECT CLEAN PG-RKL"
echo "======================================================================"

echo "STUDENT=$STUDENT"
echo "TEACHER=$TEACHER"
echo "TRAIN=$TRAIN"
echo "OUTPUT=$FULL_RUN"

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

if [[ ! -f "$FULL_RUN/epoch3/config.json" ]]; then

    python -m torch.distributed.run       --nproc_per_node=16       --master_port=29781       "$TRAINER"       --student "$STUDENT"       --teacher "$TEACHER"       --train "$TRAIN"       --output-dir "$FULL_RUN"       --epochs 3       --lr 1e-6       --max-prompt-length 512       --max-new-tokens 256       --warmup-ratio 0.03       --weight-decay 0.01       --max-grad-norm 1.0       --temperature 0.7       --top-p 0.8       --top-k 20       --seed 20260824

fi


test -f   "$FULL_RUN/epoch3/config.json"


echo
echo "RQ2_FULL6565_TRAINING_COMPLETE"


###############################################################################
# Fixed epoch3 evaluation.
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
# Final RQ2 comparison.
###############################################################################

python -   "$SELECTED_RUN"   "$RANDOM_RUN"   "$FULL_RUN"   "$SUMMARY" <<'PY'
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

