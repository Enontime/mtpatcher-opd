#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

MASTER="/workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend/scripts/mtpatcher_v9/run_opd_after_patchboost4_pgrkl_v1.sh"

OUT="/workspace/mtpatcher/runs/mtpatcher_v3_full6565_20260823/opd_after_patchboost4_pgrkl_pe3732_v1"

EVALUATOR="/workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend/scripts/mtpatcher_v6/eval_correction_fkl_torchnpu_v1.py"

WMT="/workspace/mtpatcher/data/pilot_v2_qwen3_06b/wmt24_zh_en998.jsonl"
FLORES="/workspace/mtpatcher/data/pilot_v2_qwen3_06b/flores_zh_en1012.jsonl"
CHALLENGE="/workspace/mtpatcher/data/pilot_v2_qwen3_06b/challenge_zh_en197.jsonl"

PATCHBOOST_RUN="/workspace/mtpatcher/runs/mtpatcher_v3_full6565_20260823/corrnll_patchboost4_pe3732_v1"

SUMMARY="/workspace/mtpatcher/runs/mtpatcher_v3_full6565_20260823/opd_after_patchboost4_pgrkl_pe3732_v1/patchboost4_then_pgrkl_epoch_selection.json"


echo
echo "======================================================================"
echo "PATCHBOOST4 -> CLEAN PG-RKL"
echo "======================================================================"

echo "INITIALIZATION=/workspace/mtpatcher/runs/mtpatcher_v3_full6565_20260823/corrnll_patchboost4_pe3732_v1/epoch3"
echo "TRAIN_SOURCE_SET=/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/pe_k1_clean3732.jsonl"
echo "TEACHER=/workspace/mtpatcher/models/Qwen3-8B"
echo


bash "$MASTER"


###############################################################################
# Wait until epoch3 definitely exists in case the reused master
# internally detached any stage.
###############################################################################

READY=0

for ROUND in $(seq 1 180)
do

    if [[ -f "$OUT/epoch3/config.json" ]]; then

        READY=1
        break

    fi

    sleep 10

done


if [[ "$READY" -ne 1 ]]; then

    echo "PG-RKL epoch3 checkpoint did not appear."
    false

fi


echo
echo "PG_RKL_THREE_EPOCH_CHECKPOINTS_READY"


###############################################################################
# Evaluate EVERY epoch.
###############################################################################

for EPOCH in 1 2 3
do

    MODEL="$OUT/epoch${EPOCH}"

    test -f "$MODEL/config.json"

    EVAL="$OUT/eval_epoch${EPOCH}_all"

    mkdir -p "$EVAL"


    for SPEC in       "wmt24:$WMT"       "flores:$FLORES"       "challenge:$CHALLENGE"
    do

        SPLIT="${SPEC%%:*}"
        INPUT="${SPEC#*:}"

        mkdir -p           "$EVAL/$SPLIT"

        python "$EVALUATOR"           --model "$MODEL"           --input "$INPUT"           --output "$EVAL/$SPLIT/predictions.jsonl"           --metrics "$EVAL/$SPLIT/metrics.json"           --method "patchboost4_then_pgrkl_epoch${EPOCH}_${SPLIT}"           --batch-size 16           --max-new-tokens 256

    done

done


###############################################################################
# Summary vs frozen Base and vs PatchBoost4 initialization.
###############################################################################

python -   "$OUT"   "$PATCHBOOST_RUN"   "$SUMMARY" <<'PY'
import json
import sys
from pathlib import Path


out = Path(sys.argv[1])
patch = Path(sys.argv[2])
summary_path = Path(sys.argv[3])


splits = (
    "wmt24",
    "flores",
    "challenge",
)


base = {
    "wmt24": {
        "BLEU": 15.5362135559,
        "chrF": 45.537530,
    },

    "flores": {
        "BLEU": 19.9714797904,
        "chrF": 50.860857,
    },

    "challenge": {
        "BLEU": 16.5378710573,
        "chrF": 45.758287,
    },
}


def metrics(
    root,
    split,
):

    return json.loads(
        (
            root
            / split
            / "metrics.json"
        ).read_text(
            encoding="utf-8"
        )
    )


patch_scores = {}

patch_delta_bleu = []
patch_delta_chrf = []


for split in splits:

    m = metrics(
        patch
        / "eval_epoch3",
        split,
    )

    patch_scores[
        split
    ] = {
        "BLEU":
            float(m["BLEU"]),

        "chrF":
            float(m["chrF"]),
    }

    patch_delta_bleu.append(
        float(m["BLEU"])
        - base[split]["BLEU"]
    )

    patch_delta_chrf.append(
        float(m["chrF"])
        - base[split]["chrF"]
    )


patch_avg = (
    sum(patch_delta_bleu)
    / 3
)


print(
    "=" * 110
)

print(
    "PATCHBOOST4 -> CLEAN PG-RKL ALL-EPOCH RESULT"
)

print(
    "=" * 110
)

print(
    f"{'SYSTEM':20s} "
    f"{'WMT':>10s} "
    f"{'FLORES':>10s} "
    f"{'CHALL':>10s} "
    f"{'AVG ΔBLEU':>12s} "
    f"{'Δ vs PB4':>12s}"
)


print(
    f"{'PatchBoost4 init':20s} "
    f"{patch_scores['wmt24']['BLEU']:10.6f} "
    f"{patch_scores['flores']['BLEU']:10.6f} "
    f"{patch_scores['challenge']['BLEU']:10.6f} "
    f"{patch_avg:+12.6f} "
    f"{0.0:+12.6f}"
)


records = []


for epoch in (
    1,
    2,
    3,
):

    family = (
        out
        / f"eval_epoch{epoch}_all"
    )

    bleus = []
    chrfs = []

    delta_bleu = []
    delta_chrf = []


    for split in splits:

        m = metrics(
            family,
            split,
        )

        b = float(
            m["BLEU"]
        )

        c = float(
            m["chrF"]
        )

        bleus.append(b)
        chrfs.append(c)

        delta_bleu.append(
            b
            - base[split]["BLEU"]
        )

        delta_chrf.append(
            c
            - base[split]["chrF"]
        )


    avg_db = (
        sum(delta_bleu)
        / 3
    )

    avg_dc = (
        sum(delta_chrf)
        / 3
    )


    rec = {
        "epoch":
            epoch,

        "WMT_BLEU":
            bleus[0],

        "FLORES_BLEU":
            bleus[1],

        "CHALLENGE_BLEU":
            bleus[2],

        "avg_delta_bleu_vs_base":
            avg_db,

        "avg_delta_chrf_vs_base":
            avg_dc,

        "avg_delta_bleu_vs_patchboost4":
            avg_db - patch_avg,
    }


    records.append(
        rec
    )


    print(
        f"{('PG-RKL epoch'+str(epoch)):20s} "
        f"{bleus[0]:10.6f} "
        f"{bleus[1]:10.6f} "
        f"{bleus[2]:10.6f} "
        f"{avg_db:+12.6f} "
        f"{avg_db-patch_avg:+12.6f}"
    )


best = max(
    records,
    key=lambda x:
        x["avg_delta_bleu_vs_base"],
)


result = {
    "initialization":
        "PatchBoost4",

    "patchboost4_avg_delta_bleu":
        patch_avg,

    "epochs":
        records,

    "best_epoch":
        best,

    "criterion":
        "highest mean BLEU delta over WMT24/FLORES/Challenge",
}


summary_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

summary_path.write_text(
    json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print(
    "BEST_OPD_EPOCH =",
    best["epoch"],
)

print(
    "BEST_OPD_AVG_DELTA_BLEU =",
    best["avg_delta_bleu_vs_base"],
)

print(
    "BEST_OPD_DELTA_VS_PATCHBOOST4 =",
    best["avg_delta_bleu_vs_patchboost4"],
)

print(
    "SUMMARY_JSON =",
    summary_path,
)

print()
print(
    "PATCHBOOST4_THEN_PGRKL_ALL_PASS"
)
PY

