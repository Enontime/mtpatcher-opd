#!/usr/bin/env bash

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

EXP="mtpatcher_v3_full6565_20260823"

EXP_DATA="$DATA_ROOT/$EXP"
EXP_RUN="$RUN_ROOT/$EXP"
EXP_LOG="$LOG_ROOT/$EXP"

MASTER_LOG="$EXP_LOG/matched1800_oneclick_internal.log"

mkdir -p \
    "$EXP_DATA" \
    "$EXP_RUN" \
    "$EXP_LOG"

echo "================================================================"
echo "MT-PATCHER V3 MATCHED-1800 ONE-CLICK"
date
echo "================================================================"

echo
echo "STAGE 1/5: BUILD MATCHED DATASETS"
echo

export EXP_DATA EXP_RUN

python - <<'PY'
import hashlib
import json
import os
import random
from pathlib import Path

root = Path(os.environ["EXP_DATA"])
run = Path(os.environ["EXP_RUN"])

pool_path = root / "patch_pool6565_generation.jsonl"
pe1_path = root / "pe_k1_clean3732.jsonl"
pe2_path = root / "pe_k2_consistent.jsonl"
teacher_path = (
    run
    / "teacher_seqkd_full_qwen3_8b"
    / "predictions.jsonl"
)

required = [
    pool_path,
    pe1_path,
    pe2_path,
    teacher_path,
]

for p in required:
    if not p.exists():
        raise RuntimeError(f"missing required file: {p}")


def load(path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def teacher_text(x):
    for key in (
        "student_translation",
        "teacher_translation",
        "translation",
        "prediction",
    ):
        value = x.get(key)

        if (
            isinstance(value, str)
            and value.strip()
        ):
            return value.strip()

    raise RuntimeError(
        "missing teacher translation "
        f"index={x.get('index')}"
    )


def write_jsonl(path, rows):
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for x in rows:

            if "reference" in x:
                raise RuntimeError(
                    "reference leakage "
                    f"index={x.get('index')}"
                )

            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


pool_rows = load(pool_path)
pe1_rows = load(pe1_path)
pe2_rows = load(pe2_path)
teacher_rows = load(teacher_path)

pool = {
    int(x["index"]): x
    for x in pool_rows
}

pe1 = {
    int(x["index"]): x
    for x in pe1_rows
}

teacher = {
    int(x["index"]):
        teacher_text(x)
    for x in teacher_rows
}

if len(pool) != 6565:
    raise RuntimeError(
        f"pool rows={len(pool)}"
    )

if len(pe1) != 3732:
    raise RuntimeError(
        f"pe1 rows={len(pe1)}"
    )

if len(pe2_rows) != 1800:
    raise RuntimeError(
        f"pe2 rows={len(pe2_rows)}"
    )

if len(teacher) != 6565:
    raise RuntimeError(
        f"teacher rows={len(teacher)}"
    )


k2_indices = sorted(
    int(x["index"])
    for x in pe2_rows
)


# ------------------------------------------------------------
# PE-k1 random 1800
# ------------------------------------------------------------

rng_pe = random.Random(20260823)

pe1_random_indices = sorted(
    rng_pe.sample(
        sorted(pe1),
        1800,
    )
)

pe1_random = []

for idx in pe1_random_indices:
    x = dict(pe1[idx])

    x["construction_method"] = (
        "MT_PATCHER_PE_K1_RANDOM1800"
    )

    x["matched_experiment"] = (
        "matched1800"
    )

    pe1_random.append(x)


# ------------------------------------------------------------
# SeqKD on exactly the K2 source set
# ------------------------------------------------------------

seqkd_k2selected = []

for idx in k2_indices:
    p = pool[idx]

    seqkd_k2selected.append({
        "index": idx,
        "source": p["source"],
        "messages": p["messages"],
        "target_translation":
            teacher[idx],
        "construction_method":
            "SEQKD_K2_SELECTED1800_QWEN3_8B",
        "matched_experiment":
            "matched1800",
    })


# ------------------------------------------------------------
# Random equal-size SeqKD
# ------------------------------------------------------------

rng_equal = random.Random(
    20260824
)

equal_indices = sorted(
    rng_equal.sample(
        list(range(6565)),
        1800,
    )
)

seqkd_equal = []

for idx in equal_indices:
    p = pool[idx]

    seqkd_equal.append({
        "index": idx,
        "source": p["source"],
        "messages": p["messages"],
        "target_translation":
            teacher[idx],
        "construction_method":
            "SEQKD_EQUAL1800_QWEN3_8B",
        "matched_experiment":
            "matched1800",
    })


p_pe1rand = (
    root
    / "pe_k1_random1800_seed20260823.jsonl"
)

p_k2sel = (
    root
    / "seqkd_k2selected1800.jsonl"
)

p_equal = (
    root
    / "seqkd_equal1800_seed20260824.jsonl"
)

write_jsonl(
    p_pe1rand,
    pe1_random,
)

write_jsonl(
    p_k2sel,
    seqkd_k2selected,
)

write_jsonl(
    p_equal,
    seqkd_equal,
)


manifest = {
    "experiment":
        "matched1800",

    "reference_used_for_construction":
        False,

    "pe_k2_rows":
        len(pe2_rows),

    "pe_k1_random_rows":
        len(pe1_random),

    "seqkd_k2selected_rows":
        len(seqkd_k2selected),

    "seqkd_equal_rows":
        len(seqkd_equal),

    "k2_pe1random_overlap":
        len(
            set(k2_indices)
            & set(pe1_random_indices)
        ),

    "k2_equal_overlap":
        len(
            set(k2_indices)
            & set(equal_indices)
        ),

    "files": {
        p_pe1rand.name:
            sha256(p_pe1rand),

        p_k2sel.name:
            sha256(p_k2sel),

        p_equal.name:
            sha256(p_equal),
    },
}

manifest_path = (
    root
    / "matched1800_manifest.json"
)

manifest_path.write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print(
    "PE_K2_ROWS =",
    len(pe2_rows),
)

print(
    "PE_K1_RANDOM_ROWS =",
    len(pe1_random),
)

print(
    "SEQKD_K2SELECTED_ROWS =",
    len(seqkd_k2selected),
)

print(
    "SEQKD_EQUAL_ROWS =",
    len(seqkd_equal),
)

print(
    "K2_AND_PE1_RANDOM_OVERLAP =",
    manifest[
        "k2_pe1random_overlap"
    ],
)

print(
    "K2_AND_EQUAL_OVERLAP =",
    manifest[
        "k2_equal_overlap"
    ],
)

print(
    "REFERENCE_USED_FOR_CONSTRUCTION =",
    False,
)

print(
    "MTPATCHER_V3_MATCHED1800_BUILD_PASS"
)
PY


echo
echo "STAGE 2/5: TRAIN 4 MATCHED MODELS"
echo

MODEL="$MODEL_ROOT/Qwen3-0.6B"

TRAINER="$ROOT/scripts/pilot_v2/train_qwen3_06b_full_sft_ascend.py"


launch_train () {
    NAME="$1"
    CARD="$2"
    TRAIN="$3"

    OUT="$EXP_RUN/$NAME"
    LOG="$EXP_LOG/$NAME.log"

    mkdir -p "$OUT"

    if [
        -f "$OUT/epoch3/config.json"
    ]; then
        echo \
            "SKIP TRAIN: $NAME epoch3 exists"

        return
    fi

    (
        export \
            ASCEND_RT_VISIBLE_DEVICES="$CARD"

        python "$TRAINER" \
            --model "$MODEL" \
            --train "$TRAIN" \
            --output-dir "$OUT" \
            --lr 2e-5 \
            --epochs 3 \
            --batch-size 4 \
            --grad-accum 4 \
            --max-length 1024 \
            --warmup-ratio 0.03 \
            --weight-decay 0.01 \
            --max-grad-norm 1.0 \
            --seed 20260820 \
            --num-workers 2

    ) > "$LOG" 2>&1 &

    PID=$!

    TRAIN_PIDS+=("$PID")
    TRAIN_NAMES+=("$NAME")

    echo \
        "TRAIN STARTED: $NAME CARD=$CARD PID=$PID"
}


TRAIN_PIDS=()
TRAIN_NAMES=()

launch_train \
    "pe_k2_consistent1800_b4ga4" \
    0 \
    "$EXP_DATA/pe_k2_consistent.jsonl"

launch_train \
    "pe_k1_random1800_b4ga4" \
    1 \
    "$EXP_DATA/pe_k1_random1800_seed20260823.jsonl"

launch_train \
    "seqkd_k2selected1800_b4ga4" \
    2 \
    "$EXP_DATA/seqkd_k2selected1800.jsonl"

launch_train \
    "seqkd_equal1800_b4ga4" \
    3 \
    "$EXP_DATA/seqkd_equal1800_seed20260824.jsonl"


for i in "${!TRAIN_PIDS[@]}"
do
    PID="${TRAIN_PIDS[$i]}"
    NAME="${TRAIN_NAMES[$i]}"

    echo \
        "WAIT TRAIN: $NAME PID=$PID"

    wait "$PID"

    STATUS=$?

    echo \
        "TRAIN FINISHED: $NAME STATUS=$STATUS"
done


echo
echo "===== TRAIN CHECK ====="

for NAME in \
    pe_k2_consistent1800_b4ga4 \
    pe_k1_random1800_b4ga4 \
    seqkd_k2selected1800_b4ga4 \
    seqkd_equal1800_b4ga4
do
    MODEL_DIR="$EXP_RUN/$NAME/epoch3"

    if [
        -f "$MODEL_DIR/config.json"
    ]; then
        echo \
            "TRAIN_PASS $NAME"
    else
        echo \
            "TRAIN_MISSING $NAME"
    fi
done


echo
echo "STAGE 3/5: LAUNCH 12 EVALUATIONS"
echo

TOKENIZER="$MODEL_ROOT/Qwen3-0.6B"

EVAL="$ROOT/scripts/pilot_v2/eval_qwen3_06b_mt_ascend.py"

SCORE="$ROOT/scripts/pilot_v2/score_mt_jsonl.py"

EVAL_ROOT="$EXP_RUN/matched1800_eval_epoch3"

EVAL_LOG_ROOT="$EXP_LOG/matched1800_eval_epoch3"

mkdir -p \
    "$EVAL_ROOT" \
    "$EVAL_LOG_ROOT"

WMT="$DATA_ROOT/pilot_v2_qwen3_06b/wmt24_zh_en998.jsonl"

FLORES="$DATA_ROOT/pilot_v2_qwen3_06b/flores_zh_en1012.jsonl"

CHALLENGE="$DATA_ROOT/pilot_v2_qwen3_06b/challenge_zh_en197.jsonl"


launch_eval () {
    MODEL_NAME="$1"
    DATA_NAME="$2"
    CARD="$3"
    DATA="$4"

    MODEL_PATH="$EXP_RUN/$MODEL_NAME/epoch3"

    OUT="$EVAL_ROOT/$MODEL_NAME/$DATA_NAME"

    LOG="$EVAL_LOG_ROOT/${MODEL_NAME}_${DATA_NAME}.log"

    METRICS="$OUT/metrics.json"

    mkdir -p "$OUT"

    if [
        -f "$METRICS"
    ]; then
        echo \
            "SKIP EVAL: $MODEL_NAME / $DATA_NAME"

        return
    fi

    if [
        ! -f "$MODEL_PATH/config.json"
    ]; then
        echo \
            "EVAL BLOCKED: missing model $MODEL_PATH"

        return
    fi

    (
        export \
            ASCEND_RT_VISIBLE_DEVICES="$CARD"

        python "$EVAL" \
            --model "$MODEL_PATH" \
            --tokenizer "$TOKENIZER" \
            --input "$DATA" \
            --output "$OUT/predictions.jsonl" \
            --method "${MODEL_NAME}_${DATA_NAME}" \
            --batch-size 16 \
            --max-new-tokens 256 \
            --attn-implementation sdpa

        if [
            -f "$OUT/predictions.jsonl"
        ]; then
            python "$SCORE" \
                --input "$OUT/predictions.jsonl" \
                --output "$METRICS"
        fi

    ) > "$LOG" 2>&1 &

    PID=$!

    EVAL_PIDS+=("$PID")
    EVAL_NAMES+=(
        "$MODEL_NAME/$DATA_NAME"
    )

    echo \
        "EVAL STARTED: $MODEL_NAME / $DATA_NAME CARD=$CARD PID=$PID"
}


EVAL_PIDS=()
EVAL_NAMES=()


launch_eval \
    pe_k2_consistent1800_b4ga4 \
    wmt24 \
    4 \
    "$WMT"

launch_eval \
    pe_k2_consistent1800_b4ga4 \
    flores \
    5 \
    "$FLORES"

launch_eval \
    pe_k2_consistent1800_b4ga4 \
    challenge \
    6 \
    "$CHALLENGE"


launch_eval \
    pe_k1_random1800_b4ga4 \
    wmt24 \
    7 \
    "$WMT"

launch_eval \
    pe_k1_random1800_b4ga4 \
    flores \
    8 \
    "$FLORES"

launch_eval \
    pe_k1_random1800_b4ga4 \
    challenge \
    9 \
    "$CHALLENGE"


launch_eval \
    seqkd_k2selected1800_b4ga4 \
    wmt24 \
    10 \
    "$WMT"

launch_eval \
    seqkd_k2selected1800_b4ga4 \
    flores \
    11 \
    "$FLORES"

launch_eval \
    seqkd_k2selected1800_b4ga4 \
    challenge \
    12 \
    "$CHALLENGE"


launch_eval \
    seqkd_equal1800_b4ga4 \
    wmt24 \
    13 \
    "$WMT"

launch_eval \
    seqkd_equal1800_b4ga4 \
    flores \
    14 \
    "$FLORES"

launch_eval \
    seqkd_equal1800_b4ga4 \
    challenge \
    15 \
    "$CHALLENGE"


echo
echo "STAGE 4/5: WAIT FOR EVALUATIONS"
echo

for i in "${!EVAL_PIDS[@]}"
do
    PID="${EVAL_PIDS[$i]}"
    NAME="${EVAL_NAMES[$i]}"

    echo \
        "WAIT EVAL: $NAME PID=$PID"

    wait "$PID"

    STATUS=$?

    echo \
        "EVAL FINISHED: $NAME STATUS=$STATUS"
done


echo
echo "STAGE 5/5: FINAL SUMMARY"
echo

export EXP_RUN

python - <<'PY'
import json
import os
from pathlib import Path

run = Path(
    os.environ["EXP_RUN"]
)

base = {
    "wmt24": {
        "BLEU": 15.536214,
        "chrF": 45.537530,
    },
    "flores": {
        "BLEU": 19.971480,
        "chrF": 50.860857,
    },
    "challenge": {
        "BLEU": 16.537871,
        "chrF": 45.758287,
    },
}

systems = {
    "PE-k2-1800":
        run
        / "matched1800_eval_epoch3"
        / "pe_k2_consistent1800_b4ga4",

    "PE-k1-Random1800":
        run
        / "matched1800_eval_epoch3"
        / "pe_k1_random1800_b4ga4",

    "SeqKD-k2Selected1800":
        run
        / "matched1800_eval_epoch3"
        / "seqkd_k2selected1800_b4ga4",

    "SeqKD-Equal1800":
        run
        / "matched1800_eval_epoch3"
        / "seqkd_equal1800_b4ga4",
}

datasets = [
    "wmt24",
    "flores",
    "challenge",
]

result = {}
missing = []

for system, root in systems.items():
    result[system] = {}

    for ds in datasets:
        p = root / ds / "metrics.json"

        if not p.exists():
            missing.append(
                f"{system}/{ds}"
            )
            continue

        with p.open(
            "r",
            encoding="utf-8",
        ) as f:
            m = json.load(f)

        result[system][ds] = {
            "BLEU":
                float(m["BLEU"]),
            "chrF":
                float(m["chrF"]),
        }


print("=" * 82)
print("MT-PATCHER V3 MATCHED-1800 FINAL RESULTS")
print("=" * 82)

print()

print(
    f"{'SYSTEM':25s}"
    f"{'WMT24':>10s}"
    f"{'FLORES':>10s}"
    f"{'CHALL':>10s}"
    f"{'AVGΔ':>10s}"
)

print("-" * 65)

for system in systems:
    if not all(
        ds in result[system]
        for ds in datasets
    ):
        print(
            f"{system:25s}"
            " INCOMPLETE"
        )
        continue

    vals = result[system]

    avg_delta = sum(
        vals[ds]["BLEU"]
        - base[ds]["BLEU"]
        for ds in datasets
    ) / 3.0

    print(
        f"{system:25s}"
        f"{vals['wmt24']['BLEU']:10.3f}"
        f"{vals['flores']['BLEU']:10.3f}"
        f"{vals['challenge']['BLEU']:10.3f}"
        f"{avg_delta:+10.3f}"
    )


def complete(name):
    return all(
        ds in result[name]
        for ds in datasets
    )


def gap(a, b):
    return sum(
        result[a][ds]["BLEU"]
        - result[b][ds]["BLEU"]
        for ds in datasets
    ) / 3.0


print()
print("===== KEY COMPARISONS =====")

if (
    complete("PE-k2-1800")
    and complete(
        "PE-k1-Random1800"
    )
):
    print(
        "K2 - K1Random1800 =",
        gap(
            "PE-k2-1800",
            "PE-k1-Random1800",
        ),
    )

if (
    complete(
        "SeqKD-k2Selected1800"
    )
    and complete("PE-k2-1800")
):
    print(
        "TeacherSameSources - K2 =",
        gap(
            "SeqKD-k2Selected1800",
            "PE-k2-1800",
        ),
    )

if (
    complete(
        "SeqKD-k2Selected1800"
    )
    and complete(
        "SeqKD-Equal1800"
    )
):
    print(
        "TeacherK2Selected - Equal1800 =",
        gap(
            "SeqKD-k2Selected1800",
            "SeqKD-Equal1800",
        ),
    )


summary = {
    "base": base,
    "systems": result,
    "missing": missing,
}

summary_path = (
    run
    / "matched1800_final_summary.json"
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

if missing:
    print()
    print("MISSING RESULTS:")

    for x in missing:
        print(" -", x)

    print()
    print(
        "MTPATCHER_V3_MATCHED1800_PARTIAL"
    )

else:
    print()
    print(
        "MTPATCHER_V3_MATCHED1800_ALL_PASS"
    )
PY


echo
echo "================================================================"
echo "ONE-CLICK PIPELINE FINISHED"
date
echo "================================================================"
