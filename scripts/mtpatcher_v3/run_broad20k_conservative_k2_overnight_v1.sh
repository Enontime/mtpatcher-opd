#!/usr/bin/env bash
set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

EXP="mtpatcher_v3_full6565_20260823"

D="$DATA_ROOT/$EXP"
P="$D/paperfaith_paper20k_v2"

OUT="$D/strong_repro_broad20k_conservative_k2_v1"
LOGDIR="$LOG_ROOT/$EXP/strong_repro_broad20k_conservative_k2_v1"

GEN="$ROOT/scripts/mtpatcher_v3/generate_feedback_qwen3_8b.py"
MODEL="$MODEL_ROOT/Qwen3-8B"

mkdir -p \
  "$OUT" \
  "$LOGDIR" \
  "$OUT/k1_input_shards16" \
  "$OUT/k1_output_shards16" \
  "$OUT/k2_input_shards16" \
  "$OUT/k2_output_shards16"

echo "======================================================================"
echo "BROAD20K CONSERVATIVE K1 -> K2 OVERNIGHT V1"
date
echo "======================================================================"

echo "EXP=$EXP"
echo "OUT=$OUT"
echo "GEN=$GEN"
echo "MODEL=$MODEL"
echo
echo "GENERATION_ONLY_NO_STUDENT_TRAINING=True"
echo

if [ -f "$OUT/BROAD20K_CONSERVATIVE_K2_OVERNIGHT_V1.PASS" ]; then

  echo "PASS sentinel already exists."
  echo "No work needed."

else

###############################################################################
# STAGE 0: freeze exact broad20k input
###############################################################################

echo
echo "======================================================================"
echo "STAGE 0/5: FREEZE EXACT BROAD20K CONSERVATIVE INPUT"
echo "======================================================================"

python - "$P" "$OUT" <<'PY'
import json
import sys
from pathlib import Path

P = Path(sys.argv[1])
OUT = Path(sys.argv[2])

src = P / "feedback_jobs.jsonl"
dst = OUT / "k1_input20000.jsonl"
shard_dir = OUT / "k1_input_shards16"

if not src.exists():
    raise RuntimeError(f"missing input: {src}")

rows = []

with src.open(encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        line = line.strip()

        if not line:
            continue

        x = json.loads(line)

        demo_id = x.get("demo_id")
        source = x.get("source")
        student = x.get("student_translation")

        if demo_id is None:
            raise RuntimeError(
                f"line={line_no}: missing demo_id"
            )

        if not isinstance(source, str) or not source.strip():
            raise RuntimeError(
                f"line={line_no}: bad source"
            )

        if not isinstance(student, str) or not student.strip():
            raise RuntimeError(
                f"line={line_no}: bad student_translation"
            )

        rows.append({
            "index": int(demo_id),
            "demo_id": int(demo_id),
            "source": source.strip(),
            "student_translation": student.strip(),
        })

rows.sort(key=lambda x: x["index"])

if len(rows) != 20000:
    raise RuntimeError(
        f"expected 20000 rows, got {len(rows)}"
    )

indices = [x["index"] for x in rows]

if len(indices) != len(set(indices)):
    raise RuntimeError("duplicate indices")

dst.parent.mkdir(parents=True, exist_ok=True)
shard_dir.mkdir(parents=True, exist_ok=True)

with dst.open("w", encoding="utf-8") as f:
    for x in rows:
        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )

writers = [
    (shard_dir / f"shard_{i}.jsonl").open(
        "w",
        encoding="utf-8",
    )
    for i in range(16)
]

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

counts = []

for i in range(16):
    p = shard_dir / f"shard_{i}.jsonl"

    n = sum(
        1
        for line in p.open(encoding="utf-8")
        if line.strip()
    )

    counts.append(n)

print("K1_INPUT_ROWS =", len(rows))
print("K1_INPUT_UNIQUE =", len(set(indices)))
print("K1_SHARD_COUNTS =", counts)

if counts != [1250] * 16:
    raise RuntimeError(
        f"unexpected shard counts: {counts}"
    )

print("BROAD20K_K1_INPUT_FREEZE_PASS")
PY


###############################################################################
# reusable 16-NPU generation function
###############################################################################

run_round () {

  local ROUND="$1"
  local INDIR="$2"
  local OUTDIR="$3"
  local RLOG="$4"

  mkdir -p "$OUTDIR" "$RLOG"

  echo
  echo "======================================================================"
  echo "START $ROUND ON 16 NPUs"
  date
  echo "======================================================================"

  declare -a PIDS
  declare -a FAILED

  for CARD in $(seq 0 15); do

    IN="$INDIR/shard_${CARD}.jsonl"
    O="$OUTDIR/shard_${CARD}.jsonl"
    L="$RLOG/card_${CARD}.log"

    echo "$ROUND START card=$CARD"

    env \
      ASCEND_RT_VISIBLE_DEVICES="$CARD" \
      python "$GEN" \
        --model "$MODEL" \
        --input "$IN" \
        --output "$O" \
        --batch-size 2 \
        --max-new-tokens 768 \
        --max-prompt-tokens 1536 \
        > "$L" 2>&1 &

    PIDS[$CARD]=$!

    echo \
      "$ROUND card=$CARD pid=${PIDS[$CARD]}" \
      "log=$L"

  done

  FAILED=()

  for CARD in $(seq 0 15); do

    if wait "${PIDS[$CARD]}"; then
      echo "$ROUND FINISHED card=$CARD status=0"
    else
      STATUS=$?
      echo "$ROUND FAILED card=$CARD status=$STATUS"
      FAILED+=("$CARD")
    fi

  done

  # One automatic resume/retry.
  # The generator already resumes from rows present
  # in its output JSONL.
  if [ "${#FAILED[@]}" -gt 0 ]; then

    echo
    echo "$ROUND RETRY CARDS = ${FAILED[*]}"

    declare -a RPIDS
    declare -a STILL_FAILED

    for CARD in "${FAILED[@]}"; do

      IN="$INDIR/shard_${CARD}.jsonl"
      O="$OUTDIR/shard_${CARD}.jsonl"
      L="$RLOG/card_${CARD}.retry1.log"

      env \
        ASCEND_RT_VISIBLE_DEVICES="$CARD" \
        python "$GEN" \
          --model "$MODEL" \
          --input "$IN" \
          --output "$O" \
          --batch-size 2 \
          --max-new-tokens 768 \
          --max-prompt-tokens 1536 \
          > "$L" 2>&1 &

      RPIDS[$CARD]=$!

    done

    STILL_FAILED=()

    for CARD in "${FAILED[@]}"; do

      if wait "${RPIDS[$CARD]}"; then
        echo "$ROUND RETRY PASS card=$CARD"
      else
        STATUS=$?
        echo \
          "$ROUND RETRY FAILED" \
          "card=$CARD status=$STATUS"
        STILL_FAILED+=("$CARD")
      fi

    done

    if [ "${#STILL_FAILED[@]}" -gt 0 ]; then
      echo \
        "$ROUND_FATAL_FAILED_CARDS=${STILL_FAILED[*]}" \
        >&2
      return 1
    fi

  fi

  echo
  echo "VERIFY $ROUND SHARDS"

  python - "$INDIR" "$OUTDIR" <<'PY'
import json
import sys
from pathlib import Path

INDIR = Path(sys.argv[1])
OUTDIR = Path(sys.argv[2])

total_in = 0
total_out = 0

for i in range(16):

    ip = INDIR / f"shard_{i}.jsonl"
    op = OUTDIR / f"shard_{i}.jsonl"

    if not ip.exists():
        raise RuntimeError(
            f"missing input shard: {ip}"
        )

    if not op.exists():
        raise RuntimeError(
            f"missing output shard: {op}"
        )

    inp = [
        json.loads(x)
        for x in ip.open(encoding="utf-8")
        if x.strip()
    ]

    out = [
        json.loads(x)
        for x in op.open(encoding="utf-8")
        if x.strip()
    ]

    ii = [int(x["index"]) for x in inp]
    oi = [int(x["index"]) for x in out]

    if len(oi) != len(set(oi)):
        raise RuntimeError(
            f"duplicate output index shard={i}"
        )

    if set(ii) != set(oi):
        missing = sorted(set(ii) - set(oi))
        extra = sorted(set(oi) - set(ii))

        raise RuntimeError(
            f"index mismatch shard={i} "
            f"missing={missing[:20]} "
            f"extra={extra[:20]}"
        )

    print(
        f"shard={i}",
        f"input={len(inp)}",
        f"output={len(out)}",
        "PASS",
    )

    total_in += len(inp)
    total_out += len(out)

print("TOTAL_INPUT =", total_in)
print("TOTAL_OUTPUT =", total_out)

if total_in != total_out:
    raise RuntimeError(
        f"total mismatch {total_in} != {total_out}"
    )

print("ROUND_SHARD_INTEGRITY_PASS")
PY

  echo "$ROUND COMPLETE"
  date
}


###############################################################################
# STAGE 1: K1 conservative feedback on all broad20k
###############################################################################

echo
echo "======================================================================"
echo "STAGE 1/5: FULL20K CONSERVATIVE K1"
echo "======================================================================"

run_round \
  "K1_FULL20K" \
  "$OUT/k1_input_shards16" \
  "$OUT/k1_output_shards16" \
  "$LOGDIR/k1"


###############################################################################
# STAGE 2: merge K1 and construct exact PE-eligible K2 input
###############################################################################

echo
echo "======================================================================"
echo "STAGE 2/5: MERGE K1 + BUILD K2 INPUT"
echo "======================================================================"

python - "$OUT" <<'PY'
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

OUT = Path(sys.argv[1])

k1_out_dir = OUT / "k1_output_shards16"
k2_in_dir = OUT / "k2_input_shards16"

k2_in_dir.mkdir(
    parents=True,
    exist_ok=True,
)

rows = []

for i in range(16):

    p = k1_out_dir / f"shard_{i}.jsonl"

    with p.open(encoding="utf-8") as f:
        for line in f:

            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

rows.sort(
    key=lambda x: int(x["index"])
)

if len(rows) != 20000:
    raise RuntimeError(
        f"K1 output rows={len(rows)}"
    )

indices = [
    int(x["index"])
    for x in rows
]

if len(indices) != len(set(indices)):
    raise RuntimeError(
        "duplicate K1 indices"
    )

k1_all = OUT / "k1_conservative_all20000.jsonl"

with k1_all.open(
    "w",
    encoding="utf-8",
) as f:

    for x in rows:
        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )

state = Counter()

eligible = []

for x in rows:

    if x.get("parse_ok") is not True:
        state["UNUSABLE"] += 1
        continue

    if x.get("has_error") is False:
        state["NO_ERROR"] += 1
        continue

    if x.get("has_error") is not True:
        state["UNUSABLE"] += 1
        continue

    state["ERROR"] += 1

    errors = x.get("errors")
    post = x.get("post_edit")
    student = x.get("student_translation")

    ok = (
        isinstance(errors, list)
        and len(errors) > 0
        and isinstance(post, str)
        and post.strip()
        and isinstance(student, str)
        and post.strip() != student.strip()
    )

    if not ok:
        state["ERROR_NOT_PE_ELIGIBLE"] += 1
        continue

    eligible.append(x)

k1_eligible = (
    OUT /
    "k1_conservative_pe_eligible.jsonl"
)

with k1_eligible.open(
    "w",
    encoding="utf-8",
) as f:

    for x in eligible:
        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )

k2_input = []

for x in eligible:

    k2_input.append({
        "index": int(x["index"]),
        "source": x["source"],
        "student_translation":
            x["post_edit"].strip(),
    })

k2_input.sort(
    key=lambda x: x["index"]
)

p_k2 = OUT / "k2_input_all.jsonl"

with p_k2.open(
    "w",
    encoding="utf-8",
) as f:

    for x in k2_input:
        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )

writers = [
    (
        k2_in_dir /
        f"shard_{i}.jsonl"
    ).open(
        "w",
        encoding="utf-8",
    )
    for i in range(16)
]

try:

    for pos, x in enumerate(k2_input):

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

counts = []

for i in range(16):

    p = k2_in_dir / f"shard_{i}.jsonl"

    counts.append(
        sum(
            1
            for line in p.open(
                encoding="utf-8"
            )
            if line.strip()
        )
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


print("K1_STATE_COUNTS =", dict(state))
print(
    "K1_PE_ELIGIBLE =",
    len(eligible),
    "/ 20000",
)
print(
    "K1_PE_ELIGIBLE_RATE =",
    len(eligible) / 20000,
)
print(
    "K2_SHARD_COUNTS =",
    counts,
)
print(
    "K1_ALL_SHA256 =",
    sha256(k1_all),
)
print(
    "K1_ELIGIBLE_SHA256 =",
    sha256(k1_eligible),
)
print(
    "K2_INPUT_SHA256 =",
    sha256(p_k2),
)

if sum(counts) != len(k2_input):
    raise RuntimeError(
        "K2 shard count mismatch"
    )

if not eligible:
    raise RuntimeError(
        "zero K1 PE-eligible rows"
    )

print(
    "BROAD20K_K1_TO_K2_INPUT_PASS"
)
PY


###############################################################################
# STAGE 3: K2 verification on every K1 PE-eligible post-edit
###############################################################################

echo
echo "======================================================================"
echo "STAGE 3/5: K2 STRICT SECOND-PASS VERIFICATION"
echo "======================================================================"

run_round \
  "K2_VERIFY" \
  "$OUT/k2_input_shards16" \
  "$OUT/k2_output_shards16" \
  "$LOGDIR/k2"


###############################################################################
# STAGE 4: freeze verified targets + audit
###############################################################################

echo
echo "======================================================================"
echo "STAGE 4/5: FREEZE K2 VERIFIED TARGETS + AUDIT"
echo "======================================================================"

python - "$OUT" <<'PY'
import hashlib
import json
import re
import sys
import unicodedata

from collections import Counter
from pathlib import Path

OUT = Path(sys.argv[1])


def load(path):

    rows = []

    with path.open(
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def write(path, rows):

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for x in rows:
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


k1_rows = load(
    OUT /
    "k1_conservative_pe_eligible.jsonl"
)

k1 = {
    int(x["index"]): x
    for x in k1_rows
}

k2_rows = []

for i in range(16):

    p = (
        OUT /
        "k2_output_shards16" /
        f"shard_{i}.jsonl"
    )

    k2_rows.extend(
        load(p)
    )

k2_rows.sort(
    key=lambda x: int(x["index"])
)

k2 = {
    int(x["index"]): x
    for x in k2_rows
}

if len(k2_rows) != len(k2):
    raise RuntimeError(
        "duplicate K2 output index"
    )

if set(k1) != set(k2):
    raise RuntimeError(
        "K1 eligible / K2 output "
        "index-set mismatch"
    )

source_mismatch = 0
recursion_mismatch = 0

for i in k1:

    a = k1[i]
    b = k2[i]

    if (
        a["source"].strip()
        !=
        b["source"].strip()
    ):
        source_mismatch += 1

    if (
        a["post_edit"].strip()
        !=
        b["student_translation"].strip()
    ):
        recursion_mismatch += 1

if source_mismatch:
    raise RuntimeError(
        f"source mismatch={source_mismatch}"
    )

if recursion_mismatch:
    raise RuntimeError(
        f"recursion mismatch="
        f"{recursion_mismatch}"
    )


###############################################################################
# Strict historical-style verification:
#
#   parse_ok == True
#   has_error == False
#   K2 post_edit exactly copies K2 input
#
# This explicitly excludes the historical
# "NO_ERROR but changed post_edit" anomaly.
###############################################################################

verified = []
persistent_error = []
unusable = []

state = Counter()

for i in sorted(k1):

    a = k1[i]
    b = k2[i]

    parse_ok = (
        b.get("parse_ok") is True
    )

    has_error = b.get(
        "has_error"
    )

    k2_post = b.get(
        "post_edit"
    )

    k2_input = b.get(
        "student_translation"
    )

    exact_copy = (
        isinstance(k2_post, str)
        and isinstance(k2_input, str)
        and
        k2_post.strip()
        ==
        k2_input.strip()
    )

    if (
        parse_ok
        and has_error is False
        and exact_copy
    ):

        state[
            "VERIFIED_NO_ERROR_EXACT_COPY"
        ] += 1

        verified.append({
            "index": i,
            "source": a["source"],

            "original_student_translation":
                a["student_translation"],

            "k1_errors":
                a.get("errors", []),

            "k1_post_edit":
                a["post_edit"],

            "k2_verdict":
                "no_error_exact_copy",

            "target_translation":
                a["post_edit"],

            "construction_method":
                "BROAD20K_CONSERVATIVE_"
                "K2_VERIFIED_V1",
        })

    elif (
        parse_ok
        and has_error is False
    ):

        state[
            "NO_ERROR_BUT_CHANGED_POSTEDIT"
        ] += 1

        unusable.append({
            "index": i,
            "reason":
                "k2_no_error_but_"
                "post_edit_changed",
            "k1": a,
            "k2": b,
        })

    elif (
        parse_ok
        and has_error is True
    ):

        state[
            "K2_STILL_ERROR"
        ] += 1

        persistent_error.append({
            "index": i,
            "source": a["source"],
            "original_student_translation":
                a["student_translation"],
            "k1_errors":
                a.get("errors", []),
            "k1_post_edit":
                a["post_edit"],
            "k2_errors":
                b.get("errors", []),
            "k2_post_edit":
                b.get("post_edit"),
        })

    else:

        state[
            "K2_UNUSABLE"
        ] += 1

        unusable.append({
            "index": i,
            "reason":
                "k2_parse_or_schema_unusable",
            "k1": a,
            "k2": b,
        })


###############################################################################
# Mechanical K1-vs-K2 source-span continuity audit.
# This is NOT semantic identity.
###############################################################################

def normalize(s):

    if not isinstance(s, str):
        return ""

    s = unicodedata.normalize(
        "NFKC",
        s,
    ).lower()

    return re.sub(
        r"[\W_]+",
        "",
        s,
        flags=re.UNICODE,
    )


def spans(errors):

    if not isinstance(
        errors,
        list,
    ):
        return []

    out = []

    for e in errors:

        if not isinstance(
            e,
            dict,
        ):
            continue

        for key in (
            "source_span",
            "error_source",
            "source_word",
            "source",
        ):

            v = e.get(key)

            if (
                isinstance(v, str)
                and v.strip()
            ):
                out.append(
                    v.strip()
                )
                break

    return out


def relation(A, B):

    A = [
        normalize(x)
        for x in A
        if normalize(x)
    ]

    B = [
        normalize(x)
        for x in B
        if normalize(x)
    ]

    if not A or not B:
        return "UNSCORABLE"

    for a in A:
        for b in B:
            if a == b:
                return (
                    "EXACT_SOURCE_SPAN"
                )

    for a in A:
        for b in B:

            if (
                min(
                    len(a),
                    len(b),
                )
                >= 2
                and (
                    a in b
                    or b in a
                )
            ):
                return (
                    "CONTAINMENT_SOURCE_SPAN"
                )

    return "NO_SOURCE_SPAN_OVERLAP"


continuity = Counter()

for x in persistent_error:

    continuity[
        relation(
            spans(
                x.get(
                    "k1_errors",
                    [],
                )
            ),
            spans(
                x.get(
                    "k2_errors",
                    [],
                )
            ),
        )
    ] += 1


###############################################################################
# Save frozen artifacts
###############################################################################

p_k2_all = (
    OUT /
    "k2_feedback_all.jsonl"
)

p_verified = (
    OUT /
    "k2_verified_clean.jsonl"
)

p_error = (
    OUT /
    "k2_persistent_error.jsonl"
)

p_unusable = (
    OUT /
    "k2_unusable.jsonl"
)

write(
    p_k2_all,
    k2_rows,
)

write(
    p_verified,
    verified,
)

write(
    p_error,
    persistent_error,
)

write(
    p_unusable,
    unusable,
)

summary = {
    "protocol":
        "BROAD20K_CONSERVATIVE_K2_"
        "VERIFICATION_V1",

    "generation_only":
        True,

    "student_training_started":
        False,

    "candidate_pool_rows":
        20000,

    "k1_pe_eligible_rows":
        len(k1),

    "k1_pe_eligible_rate":
        len(k1) / 20000,

    "k2_rows":
        len(k2),

    "k2_state_counts":
        dict(state),

    "k2_verified_clean_rows":
        len(verified),

    "k2_verified_given_k1_eligible":
        (
            len(verified) /
            len(k1)
        ),

    "k2_verified_rate_all20k":
        len(verified) / 20000,

    "k1_to_k2_error_source_span_"
    "continuity_mechanical":
        dict(continuity),

    "source_mismatch":
        source_mismatch,

    "recursion_mismatch":
        recursion_mismatch,

    "sha256": {
        p_k2_all.name:
            sha256(p_k2_all),

        p_verified.name:
            sha256(p_verified),

        p_error.name:
            sha256(p_error),

        p_unusable.name:
            sha256(p_unusable),
    },

    "fidelity_note":
        (
            "K1 uses frozen historical "
            "conservative Qwen3-8B treatment. "
            "K2 is a historical-style "
            "second-pass verification "
            "adaptation. This is not claimed "
            "as PAPER-EXACT iterative feedback."
        ),
}

summary_path = (
    OUT /
    "broad20k_conservative_k2_"
    "summary_v1.json"
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
    "K1_PE_ELIGIBLE =",
    len(k1),
    "/ 20000",
)

print(
    "K2_STATE_COUNTS =",
    dict(state),
)

print(
    "K2_VERIFIED_CLEAN =",
    len(verified),
)

print(
    "K2_VERIFIED_GIVEN_K1 =",
    len(verified) / len(k1),
)

print(
    "K2_VERIFIED_RATE_ALL20K =",
    len(verified) / 20000,
)

print(
    "SOURCE_SPAN_CONTINUITY =",
    dict(continuity),
)

print(
    "SUMMARY =",
    summary_path,
)

print(
    "BROAD20K_CONSERVATIVE_K2_"
    "FREEZE_PASS"
)
PY


###############################################################################
# STAGE 5: final sentinel
###############################################################################

echo
echo "======================================================================"
echo "STAGE 5/5: FINALIZE"
echo "======================================================================"

touch \
  "$OUT/BROAD20K_CONSERVATIVE_K2_OVERNIGHT_V1.PASS"

echo
echo "======================================================================"
echo "BROAD20K CONSERVATIVE K1 -> K2 OVERNIGHT V1 PASS"
date
echo "======================================================================"

cat \
  "$OUT/broad20k_conservative_k2_summary_v1.json"

fi
