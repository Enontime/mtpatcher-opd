#!/usr/bin/env bash
set -Eeuo pipefail

source /workspace/mtpatcher/project_env.sh

EXP="mtpatcher_v3_full6565_20260823"
D="$DATA_ROOT/$EXP"

B512="$D/broad512_conservative_feedback_calibration_v1"
OUT="$B512/k2_frozen_k1_construct_validation_v1"

GEN="$ROOT/scripts/mtpatcher_v3/generate_feedback_qwen3_8b.py"
MODEL="$MODEL_ROOT/Qwen3-8B"

PASS="$OUT/BROAD512_FROZEN_K1_K2_CONSTRUCT_V1.PASS"
FAIL="$OUT/BROAD512_FROZEN_K1_K2_CONSTRUCT_V1.FAIL"

mkdir -p \
    "$OUT/input_shards16" \
    "$OUT/output_shards16" \
    "$OUT/logs"

rm -f "$PASS" "$FAIL"

trap '
rc=$?
echo
echo "======================================================================"
echo "BROAD512 FROZEN K1 -> K2 CONSTRUCT V1 FAILED"
echo "return_code=$rc"
date
echo "======================================================================"
touch "'"$FAIL"'"
' ERR


echo "======================================================================"
echo "BROAD512 FROZEN K1 -> K2 CONSTRUCT VALIDATION V1"
date
echo "======================================================================"

echo
echo "B512=$B512"
echo "OUT=$OUT"
echo "GEN=$GEN"
echo "MODEL=$MODEL"
echo


###############################################################################
# STAGE 1
# Freeze the ORIGINAL Broad512 audited K1 post-edits.
###############################################################################

echo "======================================================================"
echo "STAGE 1/5: BUILD FROZEN K1 K2 INPUT"
echo "======================================================================"

python - "$B512" "$OUT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

B512 = Path(sys.argv[1])
OUT = Path(sys.argv[2])

src = B512 / "broad512_conservative_feedback_v1.jsonl"
all_out = OUT / "k2_input_frozen310.jsonl"
shard_dir = OUT / "input_shards16"

if not src.exists():
    raise RuntimeError(f"missing frozen Broad512 feedback: {src}")


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


rows = []

with src.open(encoding="utf-8") as f:
    for line in f:

        line = line.strip()

        if not line:
            continue

        x = json.loads(line)

        errors = x.get("errors")
        post = x.get("post_edit")
        student = x.get("student_translation")

        eligible = (
            x.get("parse_ok") is True
            and x.get("has_error") is True
            and isinstance(errors, list)
            and len(errors) > 0
            and isinstance(post, str)
            and bool(post.strip())
            and isinstance(student, str)
            and post.strip() != student.strip()
        )

        if not eligible:
            continue

        rows.append({
            "index": int(x["index"]),
            "source": x["source"],
            "student_translation": post.strip(),
        })


rows.sort(key=lambda x: x["index"])

if len(rows) != 310:
    raise RuntimeError(
        f"expected frozen PE eligible rows=310, got {len(rows)}"
    )

indices = [x["index"] for x in rows]

if len(indices) != len(set(indices)):
    raise RuntimeError("duplicate indices in frozen310 input")


with all_out.open("w", encoding="utf-8") as f:

    for x in rows:
        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )


writers = []

for i in range(16):

    p = shard_dir / f"shard_{i}.jsonl"

    writers.append(
        p.open(
            "w",
            encoding="utf-8",
        )
    )


try:

    for pos, x in enumerate(rows):

        writers[pos % 16].write(
            json.dumps(
                x,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )

finally:

    for w in writers:
        w.close()


print("FROZEN_K1_INPUT_ROWS =", len(rows))
print("INPUT_SHA256 =", sha256(all_out))

for i in range(16):

    p = shard_dir / f"shard_{i}.jsonl"

    n = sum(
        1
        for line in p.open(encoding="utf-8")
        if line.strip()
    )

    print(
        f"INPUT_SHARD card={i} rows={n}"
    )


print("FROZEN_K1_K2_INPUT_BUILD_PASS")
PY


###############################################################################
# STAGE 2
# Run K2 on 16 NPUs.
#
# IMPORTANT:
# Existing output shard files are NOT deleted.
# The historical conservative generator is resume-aware.
###############################################################################

echo
echo "======================================================================"
echo "STAGE 2/5: K2 GENERATION ON 16 NPUs"
echo "======================================================================"

declare -A PIDS


for CARD in $(seq 0 15); do

    INFILE="$OUT/input_shards16/shard_${CARD}.jsonl"
    OUTFILE="$OUT/output_shards16/shard_${CARD}.jsonl"
    CARDLOG="$OUT/logs/card_${CARD}.log"

    echo
    echo "K2 START card=$CARD"
    echo "input=$INFILE"
    echo "output=$OUTFILE"

    env ASCEND_RT_VISIBLE_DEVICES="$CARD" \
        python "$GEN" \
            --model "$MODEL" \
            --input "$INFILE" \
            --output "$OUTFILE" \
            --batch-size 2 \
            --max-new-tokens 768 \
            --max-prompt-tokens 1536 \
        > "$CARDLOG" 2>&1 &

    PIDS[$CARD]=$!

    echo "pid=${PIDS[$CARD]}"

done


FAILED_CARDS=()

for CARD in $(seq 0 15); do

    PID="${PIDS[$CARD]}"

    if wait "$PID"; then

        echo "K2 FINISHED card=$CARD status=0"

    else

        STATUS=$?

        echo "K2 FINISHED card=$CARD status=$STATUS"

        FAILED_CARDS+=("$CARD")

    fi

done


###############################################################################
# STAGE 3
# Integrity audit + one automatic retry.
###############################################################################

echo
echo "======================================================================"
echo "STAGE 3/5: SHARD INTEGRITY AUDIT"
echo "======================================================================"

BAD_CARDS="$(
python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1])

bad = []


def load_indices(path):

    if not path.exists():
        return None, None

    ids = []

    with path.open(encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            x = json.loads(line)

            if x.get("index") is not None:
                idx = int(x["index"])

            elif x.get("demo_id") is not None:
                idx = int(x["demo_id"])

            else:
                raise RuntimeError(
                    f"row missing index/demo_id: {path}"
                )

            ids.append(idx)

    return ids, set(ids)


for card in range(16):

    inp = (
        OUT
        / "input_shards16"
        / f"shard_{card}.jsonl"
    )

    out = (
        OUT
        / "output_shards16"
        / f"shard_{card}.jsonl"
    )

    in_list, in_set = load_indices(inp)
    out_list, out_set = load_indices(out)

    good = (
        in_list is not None
        and out_list is not None
        and len(in_list) == len(in_set)
        and len(out_list) == len(out_set)
        and in_set == out_set
    )

    if good:

        print(
            f"PASS card={card} "
            f"input={len(in_list)} "
            f"output={len(out_list)}",
            file=sys.stderr,
        )

    else:

        bad.append(card)

        print(
            f"NEEDS_RETRY card={card}",
            file=sys.stderr,
        )


print(" ".join(str(x) for x in bad))
PY
)"


if [[ -n "${BAD_CARDS// }" ]]; then

    echo
    echo "RETRY_REQUIRED cards=$BAD_CARDS"

    declare -A RETRY_PIDS

    for CARD in $BAD_CARDS; do

        INFILE="$OUT/input_shards16/shard_${CARD}.jsonl"
        OUTFILE="$OUT/output_shards16/shard_${CARD}.jsonl"
        CARDLOG="$OUT/logs/card_${CARD}.log"

        echo "K2 RETRY START card=$CARD"

        env ASCEND_RT_VISIBLE_DEVICES="$CARD" \
            python "$GEN" \
                --model "$MODEL" \
                --input "$INFILE" \
                --output "$OUTFILE" \
                --batch-size 2 \
                --max-new-tokens 768 \
                --max-prompt-tokens 1536 \
            >> "$CARDLOG" 2>&1 &

        RETRY_PIDS[$CARD]=$!

    done


    for CARD in $BAD_CARDS; do

        PID="${RETRY_PIDS[$CARD]}"

        if wait "$PID"; then
            echo "K2 RETRY FINISHED card=$CARD status=0"
        else
            STATUS=$?
            echo "K2 RETRY FINISHED card=$CARD status=$STATUS"
        fi

    done

else

    echo "NO_RETRY_REQUIRED"

fi


echo
echo "FINAL SHARD INTEGRITY CHECK"

python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1])


def load(path):

    if not path.exists():
        raise RuntimeError(
            f"missing file: {path}"
        )

    rows = []

    with path.open(encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            x = json.loads(line)

            if x.get("index") is not None:
                idx = int(x["index"])

            elif x.get("demo_id") is not None:
                idx = int(x["demo_id"])

            else:
                raise RuntimeError(
                    f"missing index/demo_id: {path}"
                )

            rows.append(idx)

    return rows


total_in = 0
total_out = 0

for card in range(16):

    inp = (
        OUT
        / "input_shards16"
        / f"shard_{card}.jsonl"
    )

    out = (
        OUT
        / "output_shards16"
        / f"shard_{card}.jsonl"
    )

    ii = load(inp)
    oo = load(out)

    if len(ii) != len(set(ii)):
        raise RuntimeError(
            f"duplicate input index card={card}"
        )

    if len(oo) != len(set(oo)):
        raise RuntimeError(
            f"duplicate output index card={card}"
        )

    if set(ii) != set(oo):
        missing = sorted(set(ii) - set(oo))
        extra = sorted(set(oo) - set(ii))

        raise RuntimeError(
            f"card={card} index mismatch "
            f"missing={missing[:20]} "
            f"extra={extra[:20]}"
        )

    print(
        f"shard={card} "
        f"input={len(ii)} "
        f"output={len(oo)} "
        f"PASS"
    )

    total_in += len(ii)
    total_out += len(oo)


print("TOTAL_INPUT =", total_in)
print("TOTAL_OUTPUT =", total_out)

if total_in != 310 or total_out != 310:
    raise RuntimeError(
        f"expected 310/310, got "
        f"{total_in}/{total_out}"
    )

print("ROUND_SHARD_INTEGRITY_PASS")
PY


###############################################################################
# STAGE 4
# Merge complete K2 output and construct verdict310.
###############################################################################

echo
echo "======================================================================"
echo "STAGE 4/5: MERGE + FREEZE K2 VERDICT310"
echo "======================================================================"

python - "$OUT" <<'PY'
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

OUT = Path(sys.argv[1])

input_path = OUT / "k2_input_frozen310.jsonl"

merged_path = OUT / "k2_feedback_frozen310.jsonl"

verdict_path = (
    OUT
    / "broad512_frozen_k1_k2_verdict310_v1.jsonl"
)

summary_path = OUT / "summary_v1.json"


def sha256(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def load_jsonl(path):

    rows = []

    with path.open(encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def get_idx(x):

    if x.get("index") is not None:
        return int(x["index"])

    if x.get("demo_id") is not None:
        return int(x["demo_id"])

    raise RuntimeError("missing index/demo_id")


inputs = load_jsonl(input_path)

outputs = []

for card in range(16):

    p = (
        OUT
        / "output_shards16"
        / f"shard_{card}.jsonl"
    )

    outputs.extend(
        load_jsonl(p)
    )


im = {
    get_idx(x): x
    for x in inputs
}

om = {
    get_idx(x): x
    for x in outputs
}


if len(inputs) != 310:
    raise RuntimeError(
        f"input rows != 310: {len(inputs)}"
    )

if len(outputs) != 310:
    raise RuntimeError(
        f"output rows != 310: {len(outputs)}"
    )

if len(im) != 310:
    raise RuntimeError(
        "duplicate input indices"
    )

if len(om) != 310:
    raise RuntimeError(
        "duplicate output indices"
    )

if set(im) != set(om):
    raise RuntimeError(
        "merged input/output index mismatch"
    )


with merged_path.open(
    "w",
    encoding="utf-8",
) as f:

    for idx in sorted(om):

        f.write(
            json.dumps(
                om[idx],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )


counts = Counter()
verdicts = []

source_mismatch = 0
recursion_mismatch = 0


for idx in sorted(im):

    inp = im[idx]
    out = om[idx]

    source_in = inp.get("source")
    source_out = out.get("source")

    if (
        isinstance(source_out, str)
        and isinstance(source_in, str)
        and source_out.strip() != source_in.strip()
    ):
        source_mismatch += 1


    expected_draft = inp.get("student_translation")
    actual_draft = out.get("student_translation")

    if (
        isinstance(expected_draft, str)
        and isinstance(actual_draft, str)
        and expected_draft.strip()
        != actual_draft.strip()
    ):
        recursion_mismatch += 1


    if out.get("parse_ok") is not True:

        state = "UNUSABLE"

    elif out.get("has_error") is True:

        state = "STILL_ERROR"

    elif out.get("has_error") is False:

        post = out.get("post_edit")
        draft = out.get("student_translation")

        if (
            isinstance(post, str)
            and isinstance(draft, str)
            and post.strip() == draft.strip()
        ):

            state = "ACCEPT_NO_ERROR_EXACT_COPY"

        else:

            state = "NO_ERROR_BUT_CHANGED_POSTEDIT"

    else:

        state = "UNUSABLE"


    counts[state] += 1

    verdicts.append({
        "index": idx,
        "k2_state": state,
        "k2_accept": (
            state
            == "ACCEPT_NO_ERROR_EXACT_COPY"
        ),
    })


if source_mismatch != 0:
    raise RuntimeError(
        f"source_mismatch={source_mismatch}"
    )

if recursion_mismatch != 0:
    raise RuntimeError(
        f"recursion_mismatch={recursion_mismatch}"
    )


with verdict_path.open(
    "w",
    encoding="utf-8",
) as f:

    for x in verdicts:

        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )


accept_n = counts[
    "ACCEPT_NO_ERROR_EXACT_COPY"
]


summary = {
    "protocol":
        "BROAD512_FROZEN_K1_K2_CONSTRUCT_VALIDATION_V1",

    "rows":
        len(verdicts),

    "state_counts":
        dict(counts),

    "accept_n":
        accept_n,

    "accept_rate":
        accept_n / len(verdicts),

    "source_mismatch":
        source_mismatch,

    "recursion_mismatch":
        recursion_mismatch,

    "semantic_labels_used":
        False,

    "student_training_started":
        False,

    "sha256": {
        "k2_input_frozen310.jsonl":
            sha256(input_path),

        "k2_feedback_frozen310.jsonl":
            sha256(merged_path),

        "broad512_frozen_k1_k2_verdict310_v1.jsonl":
            sha256(verdict_path),
    },

    "fidelity_note": (
        "K2 is run directly on the exact frozen "
        "Broad512 K1 post-edits that were previously "
        "used for semantic audit. "
        "This is a construct-validation adaptation, "
        "not PAPER-EXACT iterative feedback."
    ),
}


summary_path.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print("ROWS =", len(verdicts))
print("STATE_COUNTS =", dict(counts))
print("K2_ACCEPT =", accept_n)
print(
    "K2_ACCEPT_RATE =",
    accept_n / len(verdicts),
)
print("SOURCE_MISMATCH =", source_mismatch)
print(
    "RECURSION_MISMATCH =",
    recursion_mismatch,
)

print(
    "VERDICT_FILE =",
    verdict_path,
)

print(
    "SUMMARY =",
    summary_path,
)

print(
    "VERDICT_SHA256 =",
    sha256(verdict_path),
)

print(
    "BROAD512_FROZEN_K1_K2_FREEZE_PASS"
)
PY


###############################################################################
# STAGE 5
# PASS sentinel.
###############################################################################

echo
echo "======================================================================"
echo "STAGE 5/5: FINALIZE"
echo "======================================================================"

touch "$PASS"
rm -f "$FAIL"

echo
echo "======================================================================"
echo "BROAD512 FROZEN K1 -> K2 CONSTRUCT V1 PASS"
date
echo "PASS_SENTINEL=$PASS"
echo "======================================================================"

cat "$OUT/summary_v1.json"

