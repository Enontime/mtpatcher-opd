#!/usr/bin/env bash
set -Eeuo pipefail

source /workspace/mtpatcher/project_env.sh

EXP="mtpatcher_v3_full6565_20260823"

GEN="$ROOT/scripts/mtpatcher_rq0/generate_seqkd50k_teacher_v1.py"
RQ0_RUNNER="$ROOT/scripts/mtpatcher_rq0/run_rq0b_seqkd_scaling50k_overnight_v1.sh"

OLD="$DATA_ROOT/$EXP/rq0_seqkd_scaling50k_v1/seqkd_newscrawl20000_qwen3_8b_v1.jsonl"

MODEL="$MODEL_ROOT/Qwen3-8B"

OUT="$DATA_ROOT/$EXP/teacher_seqkd_generation_spec_lock_v1"

PASS="$OUT/TEACHER_SEQKD_GENERATION_SPEC_LOCK_V1.PASS"
FAIL="$OUT/TEACHER_SEQKD_GENERATION_SPEC_LOCK_V1.FAIL"

REPORT="$OUT/teacher_seqkd_generation_spec_report_v1.txt"
JSON="$OUT/teacher_seqkd_generation_spec_report_v1.json"

mkdir -p "$OUT"

rm -f "$PASS" "$FAIL"

trap '
rc=$?
echo
echo "======================================================================"
echo "TEACHER SEQKD GENERATION SPEC LOCK FAILED"
echo "return_code=$rc"
date
echo "======================================================================"
touch "'"$FAIL"'"
' ERR


echo "======================================================================"
echo "TEACHER SEQKD GENERATION SPEC LOCK V1"
date
echo "======================================================================"

echo
echo "===== CONFIRMED PATHS ====="

for F in "$GEN" "$RQ0_RUNNER" "$OLD"; do
    if [ ! -f "$F" ]; then
        echo "MISSING: $F"
        false
    fi

    ls -lh "$F"
done


###############################################################################
# 1. Freeze exact generator and runner source.
###############################################################################

echo
echo "======================================================================"
echo "STAGE 1/4: FREEZE GENERATOR SOURCE"
echo "======================================================================"

cp "$GEN" \
   "$OUT/generate_seqkd50k_teacher_v1.FROZEN.py"

cp "$RQ0_RUNNER" \
   "$OUT/run_rq0b_seqkd_scaling50k_overnight_v1.FROZEN.sh"

sha256sum \
    "$GEN" \
    "$RQ0_RUNNER" \
    "$OLD" \
    "$MODEL/config.json" \
    "$MODEL/generation_config.json" \
    "$MODEL/tokenizer_config.json" \
    "$MODEL/tokenizer.json" \
    > "$OUT/frozen_sha256_v1.txt"

cat "$OUT/frozen_sha256_v1.txt"


###############################################################################
# 2. Extract ALL generation-relevant source lines with context.
###############################################################################

echo
echo "======================================================================"
echo "STAGE 2/4: EXTRACT GENERATION SPEC"
echo "======================================================================"

python - \
    "$GEN" \
    "$RQ0_RUNNER" \
    "$OLD" \
    "$MODEL" \
    "$REPORT" \
    "$JSON" <<'PY'

import json
import hashlib
import sys
from collections import Counter
from pathlib import Path

GEN = Path(sys.argv[1])
RUNNER = Path(sys.argv[2])
OLD = Path(sys.argv[3])
MODEL = Path(sys.argv[4])
REPORT = Path(sys.argv[5])
JSON_OUT = Path(sys.argv[6])


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(
                    f"json parse fail {path}:{ln}: {e}"
                )
    return rows


def extract_context(path, needles, radius=8):
    lines = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    hit_lines = set()

    for i, line in enumerate(lines):
        low = line.lower()

        if any(n.lower() in low for n in needles):
            for j in range(
                max(0, i-radius),
                min(len(lines), i+radius+1),
            ):
                hit_lines.add(j)

    blocks = []

    last = None
    current = []

    for j in sorted(hit_lines):
        if last is None or j == last + 1:
            current.append({
                "line": j + 1,
                "text": lines[j],
            })
        else:
            if current:
                blocks.append(current)

            current = [{
                "line": j + 1,
                "text": lines[j],
            }]

        last = j

    if current:
        blocks.append(current)

    return blocks


needles = [
    "apply_chat_template",
    "enable_thinking",
    "do_sample",
    "max_new_tokens",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "messages",
    "role",
    "translate",
    "translation",
    "source",
    "Qwen3-8B",
    "teacher",
    "generation_config",
    "batch-size",
    "device-id",
    "world-size",
    "seed",
]

gen_blocks = extract_context(
    GEN,
    needles,
    radius=10,
)

runner_blocks = extract_context(
    RUNNER,
    [
        "generate_seqkd50k_teacher_v1.py",
        "Qwen3-8B",
        "max-new-tokens",
        "batch-size",
        "device-id",
        "world-size",
        "seed",
        "seqkd_newscrawl",
    ],
    radius=12,
)


rows = read_jsonl(OLD)

if len(rows) != 20000:
    raise RuntimeError(
        f"old seqkd expected 20000, got {len(rows)}"
    )


keys = Counter()

for r in rows:
    for k in r:
        keys[k] += 1


interesting = [
    "teacher_model_path",
    "model",
    "model_path",
    "generation_config",
    "evaluation_method",
    "generation_method",
    "max_new_tokens",
    "enable_thinking",
    "do_sample",
]

metadata_values = {}

for key in interesting:

    c = Counter()

    for r in rows:

        if key not in r:
            continue

        value = json.dumps(
            r[key],
            ensure_ascii=False,
            sort_keys=True,
        )

        c[value] += 1

    if c:
        metadata_values[key] = dict(c)


def assistant_target(r):

    msgs = r.get("messages")

    if not isinstance(msgs, list):
        return None

    vals = []

    for m in msgs:
        if (
            isinstance(m, dict)
            and m.get("role") == "assistant"
            and isinstance(m.get("content"), str)
        ):
            vals.append(m["content"])

    return vals[-1] if vals else None


possible_target_fields = Counter()

for r in rows:

    for key in [
        "teacher_translation",
        "translation",
        "target",
        "response",
        "output",
        "student_translation",
        "reference",
    ]:

        if (
            isinstance(r.get(key), str)
            and r[key].strip()
        ):
            possible_target_fields[key] += 1

    if assistant_target(r):
        possible_target_fields[
            "messages:last_assistant"
        ] += 1


model_hashes = {}

for name in [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
]:

    p = MODEL / name

    if p.exists():
        model_hashes[name] = {
            "path": str(p),
            "sha256": sha256(p),
            "size_bytes": p.stat().st_size,
        }


report = {
    "protocol":
        "TEACHER_SEQKD_GENERATION_SPEC_LOCK_V1",

    "generator": {
        "path": str(GEN),
        "sha256": sha256(GEN),
        "relevant_blocks": gen_blocks,
    },

    "historical_runner": {
        "path": str(RUNNER),
        "sha256": sha256(RUNNER),
        "relevant_blocks": runner_blocks,
    },

    "old_seqkd20k": {
        "path": str(OLD),
        "sha256": sha256(OLD),
        "rows": len(rows),
        "field_presence": dict(keys),
        "metadata_values": metadata_values,
        "possible_target_fields":
            dict(possible_target_fields),
        "first_row": rows[0],
        "last_row": rows[-1],
    },

    "model": {
        "path": str(MODEL),
        "files": model_hashes,
    },

    "decision": {
        "status": "REVIEW_REQUIRED",
        "note": (
            "This audit extracts exact historical "
            "generation semantics. It does not yet "
            "authorize reuse."
        ),
    },
}

JSON_OUT.write_text(
    json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


out = []

def emit(x=""):
    out.append(str(x))


emit("=" * 80)
emit("TEACHER SEQKD GENERATION SPEC LOCK V1")
emit("=" * 80)

emit()
emit("===== GENERATOR =====")
emit(f"path={GEN}")
emit(f"sha256={sha256(GEN)}")

for block in gen_blocks:
    emit()
    emit("--- block ---")
    for x in block:
        emit(
            f"{x['line']:04d}: {x['text']}"
        )


emit()
emit("===== HISTORICAL RUNNER =====")
emit(f"path={RUNNER}")
emit(f"sha256={sha256(RUNNER)}")

for block in runner_blocks:
    emit()
    emit("--- block ---")
    for x in block:
        emit(
            f"{x['line']:04d}: {x['text']}"
        )


emit()
emit("===== OLD SEQKD20K =====")
emit(f"path={OLD}")
emit(f"sha256={sha256(OLD)}")
emit(f"rows={len(rows)}")
emit(f"keys={dict(keys)}")
emit(
    "possible_target_fields="
    f"{dict(possible_target_fields)}"
)

emit()
emit("METADATA_VALUES:")
emit(
    json.dumps(
        metadata_values,
        ensure_ascii=False,
        indent=2,
    )
)

emit()
emit("FIRST_ROW:")
emit(
    json.dumps(
        rows[0],
        ensure_ascii=False,
        indent=2,
    )
)

emit()
emit("LAST_ROW:")
emit(
    json.dumps(
        rows[-1],
        ensure_ascii=False,
        indent=2,
    )
)

emit()
emit("===== MODEL HASHES =====")

for k, v in model_hashes.items():
    emit(
        f"{k} "
        f"sha256={v['sha256']} "
        f"size={v['size_bytes']}"
    )

emit()
emit("===== DECISION =====")
emit("STATUS=REVIEW_REQUIRED")
emit(
    "No Teacher generation started."
)

REPORT.write_text(
    "\n".join(out) + "\n",
    encoding="utf-8",
)

print("ROWS =", len(rows))
print(
    "POSSIBLE_TARGET_FIELDS =",
    dict(possible_target_fields),
)
print(
    "GENERATOR_SHA256 =",
    sha256(GEN),
)
print(
    "HISTORICAL_RUNNER_SHA256 =",
    sha256(RUNNER),
)
print(
    "OLD_SEQKD_SHA256 =",
    sha256(OLD),
)
print("REPORT =", REPORT)
print("JSON =", JSON_OUT)
print(
    "TEACHER_SEQKD_GENERATION_SPEC_EXTRACTION_PASS"
)
PY


###############################################################################
# 3. Show the exact generator itself around important sections.
###############################################################################

echo
echo "======================================================================"
echo "STAGE 3/4: HUMAN-READABLE EXACT EVIDENCE"
echo "======================================================================"

{
    echo "===== GENERATOR: FIRST 340 LINES ====="
    sed -n '1,340p' "$GEN"

    echo
    echo "===== HISTORICAL RUNNER: FIRST 260 LINES ====="
    sed -n '1,260p' "$RQ0_RUNNER"

    echo
    echo "===== GENERATOR KEY GREP ====="

    grep -nE \
        'apply_chat_template|enable_thinking|do_sample|max_new_tokens|temperature|top_p|top_k|Translate|translate|translation|messages|source|teacher' \
        "$GEN" \
        || true

    echo
    echo "===== RUNNER INVOCATION GREP ====="

    grep -nE \
        'generate_seqkd50k_teacher_v1|Qwen3-8B|max-new-tokens|batch-size|device-id|world-size|seed|seqkd_newscrawl' \
        "$RQ0_RUNNER" \
        || true

} > "$OUT/exact_generation_source_evidence_v1.txt"

cat "$OUT/exact_generation_source_evidence_v1.txt"


###############################################################################
# 4. Finalize.
###############################################################################

echo
echo "======================================================================"
echo "STAGE 4/4: FINALIZE"
echo "======================================================================"

touch "$PASS"
rm -f "$FAIL"

echo
echo "======================================================================"
echo "TEACHER SEQKD GENERATION SPEC LOCK V1 PASS"
echo
echo "PASS means extraction completed."
echo "No Teacher generation has started."
echo
echo "REPORT=$REPORT"
echo "JSON=$JSON"
echo "SOURCE_EVIDENCE=$OUT/exact_generation_source_evidence_v1.txt"
echo
date
echo "======================================================================"
