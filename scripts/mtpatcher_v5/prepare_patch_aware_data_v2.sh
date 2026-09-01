#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

PE="$DATA_ROOT/$EXP/pe_k1_clean3732.jsonl"

FEEDBACK="$DATA_ROOT/$EXP/feedback_qwen3_8b_merged6565.jsonl"

OUTPUT="$DATA_ROOT/$EXP/patch_aware_k1_clean3732_v2.jsonl"

AUDIT_DIR="$ROOT/results/$EXP/patch_aware_data_v2"

SUMMARY="$AUDIT_DIR/summary.txt"

PREVIEW="$AUDIT_DIR/preview.txt"

MODEL="$MODEL_ROOT/Qwen3-0.6B"

mkdir -p "$AUDIT_DIR"

export PE FEEDBACK OUTPUT SUMMARY PREVIEW MODEL


python - <<'PY'
import hashlib
import json
import math
import os
import random
import statistics

from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from transformers import AutoTokenizer


PE = Path(os.environ["PE"])
FEEDBACK = Path(os.environ["FEEDBACK"])
OUTPUT = Path(os.environ["OUTPUT"])
SUMMARY = Path(os.environ["SUMMARY"])
PREVIEW = Path(os.environ["PREVIEW"])
MODEL = Path(os.environ["MODEL"])

SEED = 20260825


###############################################################################
# Helpers
###############################################################################

def norm(x):

    if x is None:
        return ""

    return " ".join(
        str(x)
        .replace("\r", " ")
        .replace("\n", " ")
        .split()
    )


def load_jsonl(path):

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:

        for lineno, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            obj = json.loads(line)

            if not isinstance(obj, dict):

                raise RuntimeError(
                    f"{path}:{lineno}: non-dict row"
                )

            rows.append(obj)

    return rows


def sha256(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):

            h.update(chunk)

    return h.hexdigest()


def nested(obj, path):

    cur = obj

    for key in path.split("."):

        if not isinstance(cur, dict):
            return None

        cur = cur.get(key)

    return cur


def feedback_post_edit(row):

    for path in (
        "parsed.post_edit",
        "parsed.postedit",
        "feedback.post_edit",
        "feedback.postedit",
        "post_edit",
        "postedit",
    ):

        value = nested(
            row,
            path,
        )

        if (
            isinstance(value, str)
            and value.strip()
        ):

            return norm(value), path

    return None, None


def pct(values, q):

    if not values:
        return 0.0

    values = sorted(values)

    pos = (
        len(values) - 1
    ) * q

    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return float(values[lo])

    w = pos - lo

    return float(
        values[lo] * (1.0 - w)
        + values[hi] * w
    )


###############################################################################
# Load
###############################################################################

if not PE.exists():

    raise RuntimeError(
        f"Missing PE file: {PE}"
    )


if not FEEDBACK.exists():

    raise RuntimeError(
        f"Missing feedback file: {FEEDBACK}"
    )


pe_rows = load_jsonl(
    PE
)

feedback_rows = load_jsonl(
    FEEDBACK
)


print("=" * 100)
print("PATCH-AWARE DATA PREPARATION V2")
print("=" * 100)

print(
    "PE_ROWS =",
    len(pe_rows),
)

print(
    "FEEDBACK_ROWS =",
    len(feedback_rows),
)


if len(pe_rows) != 3732:

    raise RuntimeError(
        f"Expected 3732 PE rows, got {len(pe_rows)}"
    )


###############################################################################
# Schema hard audit
###############################################################################

required = (
    "index",
    "source",
    "student_translation",
    "target_translation",
    "messages",
)


for i, row in enumerate(
    pe_rows
):

    for key in required:

        if key not in row:

            raise RuntimeError(
                f"PE row {i} missing {key}"
            )


    if not isinstance(
        row["source"],
        str,
    ) or not row["source"].strip():

        raise RuntimeError(
            f"Empty source at PE row {i}"
        )


    if not isinstance(
        row["student_translation"],
        str,
    ) or not row["student_translation"].strip():

        raise RuntimeError(
            f"Empty student_translation at PE row {i}"
        )


    if not isinstance(
        row["target_translation"],
        str,
    ) or not row["target_translation"].strip():

        raise RuntimeError(
            f"Empty target_translation at PE row {i}"
        )


    if not isinstance(
        row["messages"],
        list,
    ) or not row["messages"]:

        raise RuntimeError(
            f"Invalid messages at PE row {i}"
        )


print(
    "PE_SCHEMA_PASS"
)


###############################################################################
# Build feedback index
###############################################################################

feedback_map = {}


for row in feedback_rows:

    if "index" not in row:
        continue

    idx = int(
        row["index"]
    )

    if idx in feedback_map:

        raise RuntimeError(
            f"Duplicate feedback index={idx}"
        )

    feedback_map[
        idx
    ] = row


###############################################################################
# Most important provenance check:
#
# pe.target_translation must equal Feedbacker's post_edit.
###############################################################################

verified = 0
missing_feedback = 0
missing_feedback_postedit = 0
postedit_path_counter = Counter()

mismatches = []


for row in pe_rows:

    idx = int(
        row["index"]
    )

    fb = feedback_map.get(
        idx
    )

    if fb is None:

        missing_feedback += 1
        continue


    post_edit, used_path = feedback_post_edit(
        fb
    )


    if post_edit is None:

        missing_feedback_postedit += 1
        continue


    postedit_path_counter[
        used_path
    ] += 1


    target = norm(
        row[
            "target_translation"
        ]
    )


    if target != post_edit:

        mismatches.append(
            {
                "index": idx,
                "target_translation": target,
                "feedback_post_edit": post_edit,
                "feedback_path": used_path,
            }
        )

        continue


    verified += 1


print()
print(
    "TARGET_IS_POSTEDIT_AUDIT"
)

print(
    "verified_exact =",
    verified,
)

print(
    "missing_feedback =",
    missing_feedback,
)

print(
    "missing_feedback_postedit =",
    missing_feedback_postedit,
)

print(
    "mismatches =",
    len(mismatches),
)

print(
    "postedit_paths =",
    dict(postedit_path_counter),
)


if mismatches:

    print(
        "FIRST_MISMATCHES =",
        json.dumps(
            mismatches[:5],
            ensure_ascii=False,
            indent=2,
        ),
    )


if verified != len(
    pe_rows
):

    raise RuntimeError(
        "target_translation could not be proven equal "
        "to Feedbacker post_edit for all 3732 PE rows"
    )


print(
    "TARGET_TRANSLATION_IS_FEEDBACK_POSTEDIT_PASS"
)


###############################################################################
# Tokenizer
###############################################################################

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL),
    local_files_only=True,
    trust_remote_code=True,
)


if tokenizer.eos_token_id is None:

    raise RuntimeError(
        "Tokenizer has no EOS token"
    )


###############################################################################
# Build patch / halo / random / full masks.
#
# Mask positions are POST-EDIT response-token positions.
#
# EOS is represented by position len(post_ids).
# This matters for sentence-final pure deletions.
###############################################################################

out_rows = []

mask_fracs = []
halo_fracs = []

masked_counts = []
halo_counts = []

draft_lengths = []
post_lengths = []

operation_counts = Counter()

delete_only_rows = 0
sentence_final_delete_rows = 0

total_post_plus_eos = 0
total_patch = 0
total_halo = 0

preview_rows = []


for row_no, row in enumerate(
    pe_rows
):

    idx = int(
        row["index"]
    )


    draft = norm(
        row[
            "student_translation"
        ]
    )


    post_edit = norm(
        row[
            "target_translation"
        ]
    )


    draft_ids = tokenizer.encode(
        draft,
        add_special_tokens=False,
    )


    post_ids = tokenizer.encode(
        post_edit,
        add_special_tokens=False,
    )


    if draft_ids == post_ids:

        raise RuntimeError(
            f"PE clean row unexpectedly unchanged index={idx}"
        )


    draft_lengths.append(
        len(draft_ids)
    )

    post_lengths.append(
        len(post_ids)
    )


    matcher = SequenceMatcher(
        None,
        draft_ids,
        post_ids,
        autojunk=False,
    )


    patch = set()

    serialized_ops = []

    has_replace = False
    has_insert = False
    has_delete = False


    for (
        tag,
        i1,
        i2,
        j1,
        j2,
    ) in matcher.get_opcodes():

        operation_counts[
            tag
        ] += 1


        serialized_ops.append(
            {
                "tag": tag,
                "draft_start": i1,
                "draft_end": i2,
                "post_start": j1,
                "post_end": j2,
            }
        )


        if tag == "equal":
            continue


        if tag == "replace":

            has_replace = True

            patch.update(
                range(
                    j1,
                    j2,
                )
            )


        elif tag == "insert":

            has_insert = True

            patch.update(
                range(
                    j1,
                    j2,
                )
            )


        elif tag == "delete":

            has_delete = True

            # Supervise the next surviving post-edit token.
            #
            # If deletion happens at response end, supervise EOS.
            if j1 < len(
                post_ids
            ):

                patch.add(
                    j1
                )

            else:

                patch.add(
                    len(
                        post_ids
                    )
                )

                sentence_final_delete_rows += 1


    if (
        has_delete
        and not has_replace
        and not has_insert
    ):

        delete_only_rows += 1


    # Response positions include EOS at len(post_ids).
    response_position_count = (
        len(
            post_ids
        )
        + 1
    )


    if not patch:

        raise RuntimeError(
            f"Empty patch mask for changed row index={idx}"
        )


    if min(
        patch
    ) < 0:

        raise RuntimeError(
            f"Negative patch index={idx}"
        )


    if max(
        patch
    ) >= response_position_count:

        raise RuntimeError(
            f"Out-of-range patch index={idx}"
        )


    patch = sorted(
        patch
    )


    ###########################################################################
    # Halo-1 mask.
    ###########################################################################

    halo = set(
        patch
    )


    for p in patch:

        if p - 1 >= 0:

            halo.add(
                p - 1
            )


        if p + 1 < response_position_count:

            halo.add(
                p + 1
            )


    halo = sorted(
        halo
    )


    ###########################################################################
    # Deterministic equal-cardinality random control.
    #
    # Same row, same correction trajectory, same number of supervised tokens.
    ###########################################################################

    rng = random.Random(
        SEED + idx * 1000003
    )


    all_positions = list(
        range(
            response_position_count
        )
    )


    random_mask = sorted(
        rng.sample(
            all_positions,
            k=len(
                patch
            ),
        )
    )


    ###########################################################################
    # Full correction trajectory.
    ###########################################################################

    full_mask = all_positions


    patch_fraction = (
        len(patch)
        / response_position_count
    )


    halo_fraction = (
        len(halo)
        / response_position_count
    )


    mask_fracs.append(
        patch_fraction
    )

    halo_fracs.append(
        halo_fraction
    )

    masked_counts.append(
        len(patch)
    )

    halo_counts.append(
        len(halo)
    )


    total_post_plus_eos += (
        response_position_count
    )

    total_patch += len(
        patch
    )

    total_halo += len(
        halo
    )


    x = dict(
        row
    )


    x[
        "_patch_aware_v2"
    ] = {
        "postedit_field":
            "target_translation",

        "postedit_provenance":
            "feedback_qwen3_8b_merged6565.parsed.post_edit",

        "tokenizer":
            str(MODEL),

        "draft_token_count":
            len(draft_ids),

        "postedit_token_count":
            len(post_ids),

        "response_positions_with_eos":
            response_position_count,

        "patch_mask":
            patch,

        "patch_halo1_mask":
            halo,

        "random_equal_count_mask":
            random_mask,

        "full_correction_mask":
            full_mask,

        "patch_token_count":
            len(patch),

        "patch_halo1_token_count":
            len(halo),

        "patch_fraction":
            patch_fraction,

        "patch_halo1_fraction":
            halo_fraction,

        "diff_opcodes":
            serialized_ops,

        "random_seed":
            SEED + idx * 1000003,
    }


    out_rows.append(
        x
    )


    if (
        len(preview_rows) < 30
        and len(patch) >= 1
        and len(patch) <= 20
    ):

        display_tokens = []


        for p in patch:

            if p == len(
                post_ids
            ):

                display_tokens.append(
                    "<EOS>"
                )

            else:

                display_tokens.append(
                    tokenizer.convert_ids_to_tokens(
                        post_ids[
                            p
                        ]
                    )
                )


        preview_rows.append(
            {
                "index": idx,
                "source": norm(
                    row[
                        "source"
                    ]
                ),
                "draft": draft,
                "post_edit": post_edit,
                "patch_mask": patch,
                "patch_tokens": display_tokens,
                "random_mask": random_mask,
                "patch_fraction": patch_fraction,
            }
        )


###############################################################################
# Global scientific sanity
###############################################################################

global_patch_fraction = (
    total_patch
    / total_post_plus_eos
)


global_halo_fraction = (
    total_halo
    / total_post_plus_eos
)


if global_patch_fraction <= 0:

    raise RuntimeError(
        "Global patch mask is empty"
    )


if global_patch_fraction >= 0.80:

    raise RuntimeError(
        "Patch mask unexpectedly covers >=80% of correction trajectory"
    )


if len(
    out_rows
) != 3732:

    raise RuntimeError(
        "Output row count != 3732"
    )


###############################################################################
# Write atomically
###############################################################################

tmp = OUTPUT.with_suffix(
    OUTPUT.suffix
    + ".tmp"
)


with tmp.open(
    "w",
    encoding="utf-8",
) as f:

    for row in out_rows:

        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )


tmp.replace(
    OUTPUT
)


###############################################################################
# Re-read output validation
###############################################################################

check = load_jsonl(
    OUTPUT
)


if len(
    check
) != 3732:

    raise RuntimeError(
        "Materialized patch-aware file does not contain 3732 rows"
    )


for i, row in enumerate(
    check
):

    meta = row.get(
        "_patch_aware_v2"
    )


    if not isinstance(
        meta,
        dict,
    ):

        raise RuntimeError(
            f"Missing patch metadata row {i}"
        )


    p = meta.get(
        "patch_mask"
    )

    r = meta.get(
        "random_equal_count_mask"
    )


    if not p:

        raise RuntimeError(
            f"Empty patch mask after serialization row {i}"
        )


    if len(
        p
    ) != len(
        r
    ):

        raise RuntimeError(
            f"Random mask cardinality mismatch row {i}"
        )


###############################################################################
# Summary
###############################################################################

summary = []


summary.append(
    "=" * 100
)

summary.append(
    "MT-PATCHER PATCH-AWARE DATA V2"
)

summary.append(
    "=" * 100
)

summary.append(
    f"PE={PE}"
)

summary.append(
    f"PE_SHA256={sha256(PE)}"
)

summary.append(
    f"FEEDBACK={FEEDBACK}"
)

summary.append(
    f"OUTPUT={OUTPUT}"
)

summary.append(
    f"OUTPUT_SHA256={sha256(OUTPUT)}"
)

summary.append(
    ""
)

summary.append(
    f"rows={len(out_rows)}"
)

summary.append(
    "POSTEDIT_FIELD=target_translation"
)

summary.append(
    f"TARGET_POSTEDIT_EXACT_VERIFIED={verified}/3732"
)

summary.append(
    ""
)

summary.append(
    f"total_response_positions_with_eos={total_post_plus_eos}"
)

summary.append(
    f"total_patch_positions={total_patch}"
)

summary.append(
    f"total_halo1_positions={total_halo}"
)

summary.append(
    f"GLOBAL_PATCH_MASK_FRACTION={global_patch_fraction:.6f}"
)

summary.append(
    f"GLOBAL_HALO1_MASK_FRACTION={global_halo_fraction:.6f}"
)

summary.append(
    ""
)

summary.append(
    "PATCH FRACTION DISTRIBUTION"
)

for name, q in (
    ("p10", 0.10),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p90", 0.90),
    ("p95", 0.95),
    ("p99", 0.99),
):

    summary.append(
        f"{name}={pct(mask_fracs, q):.6f}"
    )


summary.append(
    ""
)

summary.append(
    "PATCH TOKEN COUNT DISTRIBUTION"
)

for name, q in (
    ("p10", 0.10),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p90", 0.90),
    ("p95", 0.95),
    ("p99", 0.99),
):

    summary.append(
        f"{name}={pct(masked_counts, q):.3f}"
    )


summary.append(
    ""
)

summary.append(
    f"draft_token_mean={statistics.mean(draft_lengths):.3f}"
)

summary.append(
    f"postedit_token_mean={statistics.mean(post_lengths):.3f}"
)

summary.append(
    f"delete_only_rows={delete_only_rows}"
)

summary.append(
    f"sentence_final_delete_rows={sentence_final_delete_rows}"
)

summary.append(
    f"diff_opcode_counts={dict(operation_counts)}"
)

summary.append(
    ""
)

summary.append(
    "AVAILABLE_CONTROLS:"
)

summary.append(
    "  patch_mask              = correction-localized"
)

summary.append(
    "  random_equal_count_mask = equal-token-count random control"
)

summary.append(
    "  full_correction_mask    = full corrected trajectory control"
)

summary.append(
    ""
)

summary.append(
    "PATCH_MASK_PREP_ALL_PASS"
)


SUMMARY.write_text(
    "\n".join(
        summary
    )
    + "\n",
    encoding="utf-8",
)


###############################################################################
# Preview
###############################################################################

with PREVIEW.open(
    "w",
    encoding="utf-8",
) as f:

    for row in preview_rows:

        f.write(
            "=" * 110
            + "\n"
        )

        f.write(
            f"INDEX={row['index']}\n"
        )

        f.write(
            f"PATCH_FRACTION={row['patch_fraction']:.6f}\n"
        )

        f.write(
            f"PATCH_MASK={row['patch_mask']}\n"
        )

        f.write(
            f"PATCH_TOKENS={row['patch_tokens']}\n"
        )

        f.write(
            f"RANDOM_MASK={row['random_mask']}\n\n"
        )

        f.write(
            "SOURCE:\n"
            + row[
                "source"
            ]
            + "\n\n"
        )

        f.write(
            "DRAFT:\n"
            + row[
                "draft"
            ]
            + "\n\n"
        )

        f.write(
            "POST_EDIT:\n"
            + row[
                "post_edit"
            ]
            + "\n\n"
        )


###############################################################################
# Terminal
###############################################################################

print()

print(
    "\n".join(
        summary
    )
)

print()

print(
    "SUMMARY_FILE=",
    SUMMARY,
)

print(
    "PREVIEW_FILE=",
    PREVIEW,
)

print(
    "PATCH_AWARE_DATA=",
    OUTPUT,
)

PY


echo
echo "======================================================================"
echo "FINAL FILES"
echo "======================================================================"

ls -lh \
"$OUTPUT" \
"$SUMMARY" \
"$PREVIEW"

echo
echo "PATCH_AWARE_DATA_PREP_SCRIPT_PASS"

