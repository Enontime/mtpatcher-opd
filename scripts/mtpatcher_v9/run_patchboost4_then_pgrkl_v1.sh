#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

BASE_MODEL="$MODEL_ROOT/Qwen3-0.6B"

INIT_MODEL="$RUN_ROOT/$EXP/corrnll_patchboost4_pe3732_v1/epoch3"

PATCHBOOST_RUN="$RUN_ROOT/$EXP/corrnll_patchboost4_pe3732_v1"

TRAIN_DATA="$DATA_ROOT/$EXP/pe_k1_clean3732.jsonl"

PATCH_DATA="$DATA_ROOT/$EXP/patch_aware_k1_clean3732_v2.jsonl"

WMT="$DATA_ROOT/pilot_v2_qwen3_06b/wmt24_zh_en998.jsonl"
FLORES="$DATA_ROOT/pilot_v2_qwen3_06b/flores_zh_en1012.jsonl"
CHALLENGE="$DATA_ROOT/pilot_v2_qwen3_06b/challenge_zh_en197.jsonl"

SRC_MASTER="$ROOT/scripts/mtpatcher_v4/run_opd_pgrkl_cleanroom_v2_oneclick.sh"

SRC_TRAINER="$ROOT/scripts/mtpatcher_v4/train_opd_pgrkl_cleanroom_v2_torchnpu.py"

EVALUATOR="$ROOT/scripts/mtpatcher_v6/eval_correction_fkl_torchnpu_v1.py"

NEW_NAME="opd_after_patchboost4_pgrkl_pe3732_v1"

DST_MASTER="$ROOT/scripts/mtpatcher_v9/run_opd_after_patchboost4_pgrkl_v1.sh"

RUNNER="$ROOT/scripts/mtpatcher_v9/patchboost4_then_pgrkl_runner_v1.sh"

OUT="$RUN_ROOT/$EXP/$NEW_NAME"

LOG="$LOG_ROOT/$EXP/${NEW_NAME}.log"

SUMMARY="$OUT/patchboost4_then_pgrkl_epoch_selection.json"

mkdir -p \
  "$ROOT/scripts/mtpatcher_v9" \
  "$LOG_ROOT/$EXP"


###############################################################################
# STAGE 1 — INPUT / PROVENANCE
###############################################################################

echo
echo "======================================================================"
echo "STAGE 1/5 — INPUT AND PROVENANCE AUDIT"
echo "======================================================================"

test -d "$BASE_MODEL"
test -d "$INIT_MODEL"
test -f "$INIT_MODEL/config.json"
test -f "$PATCHBOOST_RUN/training_manifest.json"

test -f "$TRAIN_DATA"
test -f "$PATCH_DATA"

test -f "$SRC_MASTER"
test -f "$SRC_TRAINER"
test -f "$EVALUATOR"

python - "$PATCHBOOST_RUN" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])

manifest = json.loads(
    (run / "training_manifest.json").read_text(
        encoding="utf-8"
    )
)

print(
    "PATCHBOOST4_MANIFEST =",
    {
        "method":
            manifest.get("method"),

        "boost_mode":
            manifest.get("boost_mode"),

        "boost_factor":
            manifest.get("boost_factor"),

        "rows":
            manifest.get("rows"),

        "epochs":
            manifest.get("epochs"),

        "lr":
            manifest.get("lr"),

        "global_batch":
            manifest.get("global_batch"),
    },
)

if manifest.get("boost_mode") != "patch":
    raise RuntimeError(
        "Expected PatchBoost initialization"
    )

if abs(
    float(manifest.get("boost_factor", -1))
    - 4.0
) > 1e-12:
    raise RuntimeError(
        "Expected PatchBoost-4x initialization"
    )

if int(manifest.get("rows", -1)) != 3732:
    raise RuntimeError(
        "PatchBoost row mismatch"
    )

if int(manifest.get("epochs", -1)) != 3:
    raise RuntimeError(
        "PatchBoost epoch mismatch"
    )

print(
    "PATCHBOOST4_INITIALIZATION_PROVENANCE_PASS"
)
PY


###############################################################################
# STAGE 2 — TOKENIZER INTEGRITY AUDIT
#
# Do NOT automatically enable fix_mistral_regex.
# Prove actual Qwen3 tokenization is identical.
###############################################################################

echo
echo "======================================================================"
echo "STAGE 2/5 — BASE vs CHECKPOINT TOKENIZER INTEGRITY"
echo "======================================================================"

python - \
  "$BASE_MODEL" \
  "$INIT_MODEL" \
  "$PATCH_DATA" \
  "$WMT" \
  "$FLORES" \
  "$CHALLENGE" <<'PY'
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


base_path = sys.argv[1]
ckpt_path = sys.argv[2]

files = [
    Path(x)
    for x in sys.argv[3:]
]


base = AutoTokenizer.from_pretrained(
    base_path,
    local_files_only=True,
    trust_remote_code=True,
)

ckpt = AutoTokenizer.from_pretrained(
    ckpt_path,
    local_files_only=True,
    trust_remote_code=True,
)


print(
    "TOKENIZER_META =",
    {
        "base_class":
            type(base).__name__,

        "checkpoint_class":
            type(ckpt).__name__,

        "base_len":
            len(base),

        "checkpoint_len":
            len(ckpt),

        "base_vocab_size":
            base.vocab_size,

        "checkpoint_vocab_size":
            ckpt.vocab_size,

        "base_eos":
            base.eos_token_id,

        "checkpoint_eos":
            ckpt.eos_token_id,

        "base_pad":
            base.pad_token_id,

        "checkpoint_pad":
            ckpt.pad_token_id,
    },
)


if len(base) != len(ckpt):
    raise RuntimeError(
        "Tokenizer length mismatch"
    )

if base.vocab_size != ckpt.vocab_size:
    raise RuntimeError(
        "Tokenizer vocab_size mismatch"
    )

for attr in (
    "eos_token_id",
    "bos_token_id",
    "pad_token_id",
):

    if getattr(base, attr) != getattr(
        ckpt,
        attr,
    ):
        raise RuntimeError(
            f"Special token mismatch: {attr}"
        )


rows_checked = 0
text_encodes_checked = 0
prompt_encodes_checked = 0


for path in files:

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

            row = json.loads(line)

            rows_checked += 1


            for key in (
                "source",
                "reference",
                "target_translation",
                "student_translation",
            ):

                value = row.get(key)

                if (
                    isinstance(value, str)
                    and value
                ):

                    a = base(
                        value,
                        add_special_tokens=False,
                    )["input_ids"]

                    b = ckpt(
                        value,
                        add_special_tokens=False,
                    )["input_ids"]

                    text_encodes_checked += 1

                    if a != b:

                        raise RuntimeError(
                            f"Tokenizer ID mismatch "
                            f"path={path} "
                            f"line={line_no} "
                            f"field={key}"
                        )


            messages = row.get(
                "messages"
            )

            if (
                isinstance(messages, list)
                and messages
            ):

                pa = base.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )

                pb = ckpt.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )

                if pa != pb:

                    raise RuntimeError(
                        f"Chat-template text mismatch "
                        f"path={path} "
                        f"line={line_no}"
                    )


                ia = base(
                    pa,
                    add_special_tokens=False,
                )["input_ids"]

                ib = ckpt(
                    pb,
                    add_special_tokens=False,
                )["input_ids"]

                prompt_encodes_checked += 1

                if ia != ib:

                    raise RuntimeError(
                        f"Prompt token mismatch "
                        f"path={path} "
                        f"line={line_no}"
                    )


print(
    "TOKENIZER_EQUIVALENCE_AUDIT =",
    {
        "rows_checked":
            rows_checked,

        "text_encodes_checked":
            text_encodes_checked,

        "prompt_encodes_checked":
            prompt_encodes_checked,
    },
)

print(
    "QWEN3_BASE_CHECKPOINT_TOKENIZER_EXACT_ID_PASS"
)
PY


###############################################################################
# STAGE 3 — CLONE THE ALREADY VALIDATED CLEAN PG-RKL MASTER
#
# Preserve:
#   trainer
#   teacher
#   PE3732 source set
#   PG estimator
#   sampling
#   LR/schedule
#   16 NPU
#
# Change:
#   Student initialization only
#   output name
#   distributed port
###############################################################################

echo
echo "======================================================================"
echo "STAGE 3/5 — BUILD PATCHBOOST4 -> CLEAN PG-RKL MASTER"
echo "======================================================================"

python - \
  "$SRC_MASTER" \
  "$DST_MASTER" \
  "$INIT_MODEL" \
  "$NEW_NAME" <<'PY'
import re
import sys
from pathlib import Path


src = Path(sys.argv[1])
dst = Path(sys.argv[2])

init_model = sys.argv[3]
new_name = sys.argv[4]

old_name = (
    "opd_torchnpu_pgrkl_cleanroom_pe3732_v2"
)


text = src.read_text(
    encoding="utf-8"
)


if old_name not in text:

    raise RuntimeError(
        "Old clean PG-RKL run name "
        "not found in master"
    )


text = text.replace(
    old_name,
    new_name,
)


###############################################################################
# Find Student assignment.
###############################################################################

lines = text.splitlines()

student_candidates = []


for i, line in enumerate(lines):

    m = re.match(
        r'^(\s*)'
        r'([A-Za-z_][A-Za-z0-9_]*)'
        r'\s*=(.*)$',
        line,
    )

    if not m:
        continue

    var = m.group(2)

    if (
        "Qwen3-0.6B" in line
        and "STUDENT" in var.upper()
    ):
        student_candidates.append(
            (i, var, line)
        )


if len(student_candidates) == 0:

    for i, line in enumerate(lines):

        m = re.match(
            r'^(\s*)'
            r'([A-Za-z_][A-Za-z0-9_]*)'
            r'\s*=(.*)$',
            line,
        )

        if not m:
            continue

        var = m.group(2)

        if (
            "Qwen3-0.6B" in line
            and var.upper()
            in {
                "MODEL",
                "BASE_MODEL",
            }
        ):
            student_candidates.append(
                (i, var, line)
            )


print(
    "STUDENT_ASSIGNMENT_CANDIDATES =",
    student_candidates,
)


if len(student_candidates) != 1:

    raise RuntimeError(
        "Could not uniquely identify "
        "Student model assignment"
    )


idx, var, old_line = (
    student_candidates[0]
)


indent = old_line[
    :len(old_line)
    - len(old_line.lstrip())
]


lines[idx] = (
    f'{indent}{var}="{init_model}"'
)


text = "\n".join(
    lines
) + "\n"


###############################################################################
# Fresh HCCL port.
###############################################################################

port_changes = 0


patterns = [
    (
        r'(--master_port=)\d+',
        r'\g<1>29741',
    ),
    (
        r'(--master_port\s+)\d+',
        r'\g<1>29741',
    ),
    (
        r'(?m)^(\s*MASTER_PORT\s*=\s*)\d+\s*$',
        r'\g<1>29741',
    ),
]


for pattern, repl in patterns:

    text, n = re.subn(
        pattern,
        repl,
        text,
    )

    port_changes += n


print(
    "MASTER_PORT_REPLACEMENTS =",
    port_changes,
)


###############################################################################
# Label the result clearly if old result label exists.
###############################################################################

text = text.replace(
    "OPD-PGRKL-K1-CLEANROOM-PE3732",
    "PATCHBOOST4-THEN-PGRKL",
)


###############################################################################
# Hard invariants.
###############################################################################

if new_name not in text:

    raise RuntimeError(
        "New run name missing"
    )

if init_model not in text:

    raise RuntimeError(
        "PatchBoost4 init path missing"
    )

if (
    "train_opd_pgrkl_cleanroom_v2_torchnpu.py"
    not in text
):

    raise RuntimeError(
        "Validated clean PG-RKL trainer "
        "was not preserved"
    )

if "pe_k1_clean3732" not in text:

    raise RuntimeError(
        "PE3732 source set "
        "was not preserved"
    )

if "Qwen3-8B" not in text:

    raise RuntimeError(
        "Qwen3-8B Teacher "
        "was not preserved"
    )


dst.write_text(
    text,
    encoding="utf-8",
)


print(
    "PATCHBOOST4_TO_PGRKL_MASTER_BUILD_PASS"
)
PY


chmod +x \
"$DST_MASTER"

bash -n \
"$DST_MASTER"


echo
echo "===== CLONED MASTER KEY LINES ====="

grep -nE \
'NAME=|STUDENT|Qwen3-0.6B|Qwen3-8B|pe_k1_clean3732|master_port|MASTER_PORT|train_opd_pgrkl_cleanroom' \
"$DST_MASTER" \
| head -n 80


echo
echo "PATCHBOOST4_TO_PGRKL_MASTER_STATIC_PASS"


###############################################################################
# STAGE 4 — BUILD RUNNER
#
# The validated master performs training.
# Afterward evaluate epoch1/2/3 independently.
###############################################################################

echo
echo "======================================================================"
echo "STAGE 4/5 — BUILD TRAIN + ALL-EPOCH EVAL RUNNER"
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

MASTER="$DST_MASTER"

OUT="$OUT"

EVALUATOR="$EVALUATOR"

WMT="$WMT"
FLORES="$FLORES"
CHALLENGE="$CHALLENGE"

PATCHBOOST_RUN="$PATCHBOOST_RUN"

SUMMARY="$SUMMARY"


echo
echo "======================================================================"
echo "PATCHBOOST4 -> CLEAN PG-RKL"
echo "======================================================================"

echo "INITIALIZATION=$INIT_MODEL"
echo "TRAIN_SOURCE_SET=$TRAIN_DATA"
echo "TEACHER=$MODEL_ROOT/Qwen3-8B"
echo


bash "\$MASTER"


###############################################################################
# Wait until epoch3 definitely exists in case the reused master
# internally detached any stage.
###############################################################################

READY=0

for ROUND in \$(seq 1 180)
do

    if [[ -f "\$OUT/epoch3/config.json" ]]; then

        READY=1
        break

    fi

    sleep 10

done


if [[ "\$READY" -ne 1 ]]; then

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

    MODEL="\$OUT/epoch\${EPOCH}"

    test -f "\$MODEL/config.json"

    EVAL="\$OUT/eval_epoch\${EPOCH}_all"

    mkdir -p "\$EVAL"


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
          --method "patchboost4_then_pgrkl_epoch\${EPOCH}_\${SPLIT}" \
          --batch-size 16 \
          --max-new-tokens 256

    done

done


###############################################################################
# Summary vs frozen Base and vs PatchBoost4 initialization.
###############################################################################

python - \
  "\$OUT" \
  "\$PATCHBOOST_RUN" \
  "\$SUMMARY" <<'PY'
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

BASHRUN


chmod +x \
"$RUNNER"

bash -n \
"$RUNNER"

echo "PATCHBOOST4_PGRKL_RUNNER_STATIC_PASS"


###############################################################################
# STAGE 5 — CLEAN BACKGROUND LAUNCH
###############################################################################

echo
echo "======================================================================"
echo "STAGE 5/5 — LAUNCH"
echo "======================================================================"

RUNNING="$(
    pgrep -af \
    '[p]atchboost4_then_pgrkl_runner_v1.sh|[r]un_opd_after_patchboost4_pgrkl_v1.sh' \
    || true
)"


if [[ -n "$RUNNING" ]]; then

    echo "Existing PatchBoost4 -> PG-RKL process:"
    echo "$RUNNING"

    false

fi


if [[ -e "$OUT" ]]; then

    echo "Fresh output path already exists:"
    echo "$OUT"

    false

fi


nohup setsid bash "$RUNNER" \
  > "$LOG" 2>&1 < /dev/null &


PID=$!


echo "PID=$PID"
echo "LOG=$LOG"

echo "PATCHBOOST4_THEN_PGRKL_STARTED"


###############################################################################
# First live scientific gate.
###############################################################################

PASS=0


for ROUND in \
    1 2 3 4 5 6 7 8 9 10 11 12
do

    sleep 30

    echo
    echo "HEALTH_ROUND=$ROUND"


    grep -E \
'CLEAN_PG_RUNTIME_AUDIT_PASS|OPD_TRAINING_START|epoch=1 local_step=1/|Traceback|RuntimeError|FAILED|ChildFailedError' \
    "$LOG" \
    2>/dev/null \
    | tail -n 40 \
    || true


    if grep -q \
    'CLEAN_PG_RUNTIME_AUDIT_PASS' \
    "$LOG" \
    2>/dev/null \
    && grep -q \
    'epoch=1 local_step=1/' \
    "$LOG" \
    2>/dev/null; then

        PASS=1
        break

    fi


    if grep -qE \
    'Traceback|RuntimeError|ChildFailedError' \
    "$LOG" \
    2>/dev/null; then

        echo
        echo "PATCHBOOST4 -> PG-RKL FAILED"

        tail -n 200 \
          "$LOG"

        false

    fi

done


if [[ "$PASS" -ne 1 ]]; then

    echo
    echo "Training did not reach first audited update in health window."

    tail -n 180 \
      "$LOG" \
      2>/dev/null \
      || true

    false

fi


echo
echo "======================================================================"
echo "PATCHBOOST4 -> CLEAN PG-RKL VERIFIED"
echo "======================================================================"

echo "QWEN3_BASE_CHECKPOINT_TOKENIZER_EXACT_ID_PASS"
echo "PATCHBOOST4_INITIALIZATION_PROVENANCE_PASS"
echo "PATCHBOOST4_TO_PGRKL_MASTER_STATIC_PASS"
echo "CLEAN_PG_RUNTIME_AUDIT_PASS"
echo "PATCHBOOST4_THEN_PGRKL_RUNNING"

echo
echo "After training, epoch1/2/3 will all be evaluated."
echo "LOG=$LOG"

echo
echo "PATCHBOOST4_THEN_PGRKL_LONG_RUN_SAFE"

