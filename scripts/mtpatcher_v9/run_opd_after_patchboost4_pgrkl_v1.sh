#!/usr/bin/env bash
set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

EXP="mtpatcher_v3_full6565_20260823"
NAME="opd_after_patchboost4_pgrkl_pe3732_v1"

TRAIN="$DATA_ROOT/$EXP/pe_k1_clean3732.jsonl"

STUDENT="/workspace/mtpatcher/runs/mtpatcher_v3_full6565_20260823/corrnll_patchboost4_pe3732_v1/epoch3"
TEACHER="$MODEL_ROOT/Qwen3-8B"

TRAINER="$ROOT/scripts/mtpatcher_v4/train_opd_pgrkl_cleanroom_v2_torchnpu.py"

OUT="$RUN_ROOT/$EXP/$NAME"
EVAL_ROOT="$OUT/eval_epoch3"

mkdir -p \
    "$OUT" \
    "$EVAL_ROOT"

echo "======================================================================"
echo "MT-PATCHER V3 — TORCH-NPU ON-POLICY PG-RKL-K1"
date
echo "======================================================================"

echo
echo "===== STAGE 1/4: PREFLIGHT ====="

export STUDENT TEACHER TRAIN

python - <<'PY'
import json
import os
from pathlib import Path

import torch
import torch_npu
import transformers

student = Path(
    os.environ["STUDENT"]
)

teacher = Path(
    os.environ["TEACHER"]
)

train = Path(
    os.environ["TRAIN"]
)

for p in (
    student,
    teacher,
    train,
):
    if not p.exists():
        raise RuntimeError(
            f"missing required path: {p}"
        )

rows = []

with train.open(
    "r",
    encoding="utf-8",
) as f:
    for line in f:
        if line.strip():
            rows.append(
                json.loads(line)
            )

if len(rows) != 3732:
    raise RuntimeError(
        f"expected 3732 rows, got {len(rows)}"
    )

for x in rows:
    if "reference" in x:
        raise RuntimeError(
            "reference leakage "
            f"index={x.get('index')}"
        )

    messages = x.get("messages")

    if not isinstance(
        messages,
        list,
    ):
        raise RuntimeError(
            "messages missing "
            f"index={x.get('index')}"
        )

    if any(
        m.get("role")
        == "assistant"
        for m in messages
    ):
        raise RuntimeError(
            "assistant leakage "
            f"index={x.get('index')}"
        )

print(
    "torch =",
    torch.__version__,
)

print(
    "torch_npu =",
    torch_npu.__version__,
)

print(
    "transformers =",
    transformers.__version__,
)

print(
    "NPU_COUNT =",
    torch.npu.device_count(),
)

if torch.npu.device_count() < 16:
    raise RuntimeError(
        "fewer than 16 visible NPUs"
    )

print(
    "TRAIN_ROWS =",
    len(rows),
)

print(
    "REFERENCE_USED = False"
)

print(
    "OPD_TORCHNPU_PREFLIGHT_PASS"
)
PY


echo
echo "===== STAGE 2/4: 16-NPU OPD TRAINING ====="

setsid python -m torch.distributed.run \
    --nproc_per_node=16 \
    --master_port=29741 \
    "$TRAINER" \
    --student "$STUDENT" \
    --teacher "$TEACHER" \
    --train "$TRAIN" \
    --output-dir "$OUT" \
    --epochs 3 \
    --lr 1e-6 \
    --max-prompt-length 512 \
    --max-new-tokens 256 \
    --warmup-ratio 0.03 \
    --weight-decay 0.01 \
    --max-grad-norm 1.0 \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --seed 20260824


MODEL="$OUT/epoch3"

if [[ ! -f "$MODEL/config.json" ]]; then
    echo "MISSING_EPOCH3_MODEL=$MODEL"
    false
fi

echo
echo "OPD_EPOCH3_MODEL=$MODEL"


echo
echo "===== STAGE 3/4: EVALUATION ====="

EVAL="$ROOT/scripts/pilot_v2/eval_qwen3_06b_mt_ascend.py"
SCORE="$ROOT/scripts/pilot_v2/score_mt_jsonl.py"

WMT="$DATA_ROOT/pilot_v2_qwen3_06b/wmt24_zh_en998.jsonl"
FLORES="$DATA_ROOT/pilot_v2_qwen3_06b/flores_zh_en1012.jsonl"
CHALLENGE="$DATA_ROOT/pilot_v2_qwen3_06b/challenge_zh_en197.jsonl"


run_eval () {
    NAME="$1"
    CARD="$2"
    DATA="$3"

    DIR="$EVAL_ROOT/$NAME"

    mkdir -p "$DIR"

    env \
        ASCEND_RT_VISIBLE_DEVICES="$CARD" \
        TRANSFORMERS_OFFLINE=1 \
        HF_HUB_OFFLINE=1 \
        TOKENIZERS_PARALLELISM=false \
        python "$EVAL" \
            --model "$MODEL" \
            --tokenizer "$STUDENT" \
            --input "$DATA" \
            --output "$DIR/predictions.jsonl" \
            --method "opd_torchnpu_fkl_pe3732_${NAME}" \
            --batch-size 16 \
            --max-new-tokens 256 \
            --attn-implementation sdpa \
        > "$DIR/generation.log" 2>&1

    python "$SCORE" \
        --input "$DIR/predictions.jsonl" \
        --output "$DIR/metrics.json" \
        >> "$DIR/generation.log" 2>&1
}


run_eval \
    wmt24 \
    0 \
    "$WMT" &

P1=$!


run_eval \
    flores \
    1 \
    "$FLORES" &

P2=$!


run_eval \
    challenge \
    2 \
    "$CHALLENGE" &

P3=$!


wait "$P1"
wait "$P2"
wait "$P3"

echo "OPD_TORCHNPU_EVAL_PASS"


echo
echo "===== STAGE 4/4: FINAL SUMMARY ====="

export EVAL_ROOT EXP RUN_ROOT

python - <<'PY'
import json
import os
from pathlib import Path

run = (
    Path(os.environ["RUN_ROOT"])
    / os.environ["EXP"]
)

eval_root = Path(
    os.environ["EVAL_ROOT"]
)

datasets = [
    "wmt24",
    "flores",
    "challenge",
]

base = {
    "wmt24": 15.536214,
    "flores": 19.971480,
    "challenge": 16.537871,
}


def load_system(paths):
    result = {}

    for ds in datasets:
        p = paths(ds)

        with p.open(
            "r",
            encoding="utf-8",
        ) as f:
            m = json.load(f)

        result[ds] = {
            "BLEU":
                float(m["BLEU"]),
            "chrF":
                float(m["chrF"]),
        }

    return result


opd = load_system(
    lambda ds:
        eval_root
        / ds
        / "metrics.json"
)


pe = load_system(
    lambda ds:
        run
        / "pe_k1_eval_epoch3"
        / ds
        / "metrics.json"
)


seqkd_selected = load_system(
    lambda ds:
        run
        / "seqkd_control_eval_epoch3"
        / "seqkd_selected3732_b4ga4"
        / ds
        / "metrics.json"
)


seqkd_full = load_system(
    lambda ds:
        run
        / "seqkd_control_eval_epoch3"
        / "seqkd_full6565_b4ga4"
        / ds
        / "metrics.json"
)


def avg_delta(x):
    return sum(
        x[ds]["BLEU"]
        - base[ds]
        for ds in datasets
    ) / 3.0


def avg_gap(a, b):
    return sum(
        a[ds]["BLEU"]
        - b[ds]["BLEU"]
        for ds in datasets
    ) / 3.0


print()
print("=" * 86)
print(
    "MT-PATCHER V3 — "
    "ON-POLICY PG-RKL-K1 RESULT"
)
print("=" * 86)

print(
    f"{'SYSTEM':28s}"
    f"{'WMT24':>10s}"
    f"{'FLORES':>10s}"
    f"{'CHALL':>10s}"
    f"{'AVGΔ':>10s}"
)

print("-" * 70)


systems = [
    (
        "Base",
        {
            ds: {
                "BLEU": base[ds],
                "chrF": 0.0,
            }
            for ds in datasets
        },
    ),
    (
        "PE3732",
        pe,
    ),
    (
        "SeqKD-Selected3732",
        seqkd_selected,
    ),
    (
        "SeqKD-Full6565",
        seqkd_full,
    ),
    (
        "PATCHBOOST4-THEN-PGRKL",
        opd,
    ),
]


for name, x in systems:

    delta = (
        0.0
        if name == "Base"
        else avg_delta(x)
    )

    print(
        f"{name:28s}"
        f"{x['wmt24']['BLEU']:10.3f}"
        f"{x['flores']['BLEU']:10.3f}"
        f"{x['challenge']['BLEU']:10.3f}"
        f"{delta:+10.3f}"
    )


print()
print("===== KEY COMPARISONS =====")

print(
    "OPD - PE3732 =",
    f"{avg_gap(opd, pe):+.4f}",
)

print(
    "OPD - SeqKD-Selected3732 =",
    f"{avg_gap(opd, seqkd_selected):+.4f}",
)

print(
    "OPD - SeqKD-Full6565 =",
    f"{avg_gap(opd, seqkd_full):+.4f}",
)


summary = {
    "base": base,
    "PE3732": pe,
    "SeqKD-Selected3732":
        seqkd_selected,
    "SeqKD-Full6565":
        seqkd_full,
    "PATCHBOOST4-THEN-PGRKL":
        opd,
}

out = (
    eval_root.parent
    / "opd_torchnpu_pgrkl_k1_audited_final_summary.json"
)

out.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print()
print(
    "SUMMARY_JSON =",
    out,
)

print(
    "MTPATCHER_V4_TORCHNPU_PGRKL_K1_ALL_PASS"
)
PY


echo
echo "======================================================================"
echo "OPD ONE-CLICK FINISHED"
date
echo "======================================================================"
