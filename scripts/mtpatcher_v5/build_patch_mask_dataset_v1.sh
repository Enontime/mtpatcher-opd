#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

INPUT="$DATA_ROOT/$EXP/pe_k1_clean3732.jsonl"

OUTPUT="$DATA_ROOT/$EXP/patch_mask_k1_clean3732_v1.jsonl"

OUT_DIR="$ROOT/results/$EXP/patch_mask_audit_v1"

SUMMARY="$OUT_DIR/summary.txt"
PREVIEW="$OUT_DIR/preview.txt"

MODEL="$MODEL_ROOT/Qwen3-0.6B"

mkdir -p "$OUT_DIR"

export INPUT OUTPUT SUMMARY PREVIEW MODEL

python - <<'PY'
import json
import math
import os
import statistics
from difflib import SequenceMatcher
from pathlib import Path

from transformers import AutoTokenizer


INPUT = Path(os.environ["INPUT"])
OUTPUT = Path(os.environ["OUTPUT"])
SUMMARY = Path(os.environ["SUMMARY"])
PREVIEW = Path(os.environ["PREVIEW"])
MODEL = Path(os.environ["MODEL"])


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
                    f"Non-dict row at line {lineno}"
                )

            rows.append(obj)

    return rows


def get_path(obj, path):

    cur = obj

    for part in path.split("."):

        if not isinstance(cur, dict):
            return None

        if part not in cur:
            return None

        cur = cur[part]

    return cur


def choose_path(rows, candidates):

    scored = []

    for path in candidates:

        valid = 0

        for row in rows:

            value = get_path(
                row,
                path,
            )

            if (
                isinstance(value, str)
                and value.strip()
            ):
                valid += 1

        scored.append(
            (
                valid,
                path,
            )
        )

    scored.sort(
        reverse=True
    )

    best_n, best_path = scored[0]

    if best_n < int(
        0.95 * len(rows)
    ):
        return None, scored[:10]

    return best_path, scored[:10]


def percentile(values, q):

    if not values:
        return 0.0

    xs = sorted(values)

    pos = (
        (len(xs) - 1)
        * q
    )

    lo = int(
        math.floor(pos)
    )

    hi = int(
        math.ceil(pos)
    )

    if lo == hi:
        return float(xs[lo])

    frac = pos - lo

    return float(
        xs[lo] * (1.0 - frac)
        + xs[hi] * frac
    )


###############################################################################
# Load
###############################################################################

if not INPUT.exists():

    raise RuntimeError(
        f"Missing PE dataset: {INPUT}"
    )


rows = load_jsonl(
    INPUT
)


if len(rows) != 3732:

    raise RuntimeError(
        f"Expected 3732 rows, got {len(rows)}"
    )


###############################################################################
# Robust field discovery
###############################################################################

SOURCE_CANDIDATES = (
    "source",
    "src",
    "source_text",
    "chinese",
    "input",
)


DRAFT_CANDIDATES = (
    "student_translation",
    "student_output",
    "draft",
    "student_draft",
    "hypothesis",
    "mt_output",
    "translation",
    "feedback.student_translation",
    "feedback.draft",
)


POSTEDIT_CANDIDATES = (
    "post_edit",
    "postedit",
    "post_edit_translation",
    "post_edited_translation",
    "corrected_translation",
    "corrected",
    "pe",
    "feedback.post_edit",
    "feedback.postedit",
    "parsed_feedback.post_edit",
    "parsed_feedback.postedit",
)


source_path, source_scores = choose_path(
    rows,
    SOURCE_CANDIDATES,
)

draft_path, draft_scores = choose_path(
    rows,
    DRAFT_CANDIDATES,
)

post_path, post_scores = choose_path(
    rows,
    POSTEDIT_CANDIDATES,
)


print(
    "SOURCE_PATH =",
    source_path,
)

print(
    "DRAFT_PATH =",
    draft_path,
)

print(
    "POSTEDIT_PATH =",
    post_path,
)


if source_path is None:

    print(
        "SOURCE FIELD CANDIDATES =",
        source_scores,
    )

    raise RuntimeError(
        "Could not identify source field"
    )


if draft_path is None:

    print(
        "DRAFT FIELD CANDIDATES =",
        draft_scores,
    )

    print(
        "FIRST ROW KEYS =",
        sorted(rows[0].keys()),
    )

    raise RuntimeError(
        "Could not identify Student draft field"
    )


if post_path is None:

    print(
        "POSTEDIT FIELD CANDIDATES =",
        post_scores,
    )

    print(
        "FIRST ROW KEYS =",
        sorted(rows[0].keys()),
    )

    raise RuntimeError(
        "Could not identify post-edit field"
    )


if draft_path == post_path:

    raise RuntimeError(
        "Draft and post-edit resolved to same field"
    )


###############################################################################
# Tokenizer
###############################################################################

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL),
    local_files_only=True,
    trust_remote_code=True,
)


###############################################################################
# Build token-level patch masks
###############################################################################

out_rows = []

mask_fracs = []
halo_fracs = []

draft_lens = []
post_lens = []

replace_rows = 0
insert_rows = 0
delete_rows = 0
delete_only_rows = 0

unchanged_rows = 0
changed_rows = 0

total_post_tokens = 0
total_mask_tokens = 0
total_halo_tokens = 0

opcode_counts = {
    "equal": 0,
    "replace": 0,
    "insert": 0,
    "delete": 0,
}


preview_items = []


for idx, row in enumerate(rows):

    source = norm(
        get_path(
            row,
            source_path,
        )
    )

    draft = norm(
        get_path(
            row,
            draft_path,
        )
    )

    post = norm(
        get_path(
            row,
            post_path,
        )
    )


    if not source:
        raise RuntimeError(
            f"Empty source at row {idx}"
        )


    if not draft:
        raise RuntimeError(
            f"Empty draft at row {idx}"
        )


    if not post:
        raise RuntimeError(
            f"Empty post-edit at row {idx}"
        )


    draft_ids = tokenizer.encode(
        draft,
        add_special_tokens=False,
    )

    post_ids = tokenizer.encode(
        post,
        add_special_tokens=False,
    )


    draft_lens.append(
        len(draft_ids)
    )

    post_lens.append(
        len(post_ids)
    )


    matcher = SequenceMatcher(
        a=draft_ids,
        b=post_ids,
        autojunk=False,
    )


    mask = set()

    row_has_replace = False
    row_has_insert = False
    row_has_delete = False

    opcodes_serialized = []


    for (
        tag,
        i1,
        i2,
        j1,
        j2,
    ) in matcher.get_opcodes():

        opcode_counts[
            tag
        ] += 1


        opcodes_serialized.append(
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

            row_has_replace = True

            for j in range(
                j1,
                j2,
            ):
                mask.add(j)


        elif tag == "insert":

            row_has_insert = True

            for j in range(
                j1,
                j2,
            ):
                mask.add(j)


        elif tag == "delete":

            row_has_delete = True

            # There is no target token corresponding to a pure deletion.
            # Anchor supervision on the token immediately after the
            # deleted region; if deletion is sentence-final, use the
            # final surviving token.
            if post_ids:

                anchor = min(
                    j1,
                    len(post_ids) - 1,
                )

                mask.add(
                    anchor
                )


    if row_has_replace:
        replace_rows += 1

    if row_has_insert:
        insert_rows += 1

    if row_has_delete:
        delete_rows += 1


    if (
        row_has_delete
        and not row_has_replace
        and not row_has_insert
    ):
        delete_only_rows += 1


    if draft_ids == post_ids:

        unchanged_rows += 1

    else:

        changed_rows += 1


    raw_mask = sorted(
        mask
    )


    halo = set(
        raw_mask
    )


    # One-token halo catches correction boundaries.
    for j in raw_mask:

        if j - 1 >= 0:
            halo.add(
                j - 1
            )

        if j + 1 < len(
            post_ids
        ):
            halo.add(
                j + 1
            )


    halo_mask = sorted(
        halo
    )


    npost = max(
        len(post_ids),
        1,
    )


    mask_frac = (
        len(raw_mask)
        / npost
    )


    halo_frac = (
        len(halo_mask)
        / npost
    )


    mask_fracs.append(
        mask_frac
    )

    halo_fracs.append(
        halo_frac
    )


    total_post_tokens += len(
        post_ids
    )

    total_mask_tokens += len(
        raw_mask
    )

    total_halo_tokens += len(
        halo_mask
    )


    x = dict(
        row
    )


    x[
        "_patch_mask_v1"
    ] = {
        "source_path":
            source_path,

        "draft_path":
            draft_path,

        "postedit_path":
            post_path,

        "draft_token_count":
            len(draft_ids),

        "postedit_token_count":
            len(post_ids),

        "mask_indices":
            raw_mask,

        "halo1_mask_indices":
            halo_mask,

        "mask_token_count":
            len(raw_mask),

        "halo1_token_count":
            len(halo_mask),

        "mask_fraction":
            mask_frac,

        "halo1_fraction":
            halo_frac,

        "opcodes":
            opcodes_serialized,
    }


    out_rows.append(
        x
    )


    # Representative previews:
    # prefer medium-size corrections rather than trivial punctuation.
    if (
        len(raw_mask) >= 2
        and len(raw_mask) <= 20
        and len(preview_items) < 40
    ):

        raw_tokens = tokenizer.convert_ids_to_tokens(
            [
                post_ids[j]
                for j in raw_mask
            ]
        )


        preview_items.append(
            {
                "index": idx,
                "source": source,
                "draft": draft,
                "post": post,
                "mask": raw_mask,
                "tokens": raw_tokens,
                "mask_fraction": mask_frac,
            }
        )


###############################################################################
# Hard scientific sanity checks
###############################################################################

if changed_rows < 3000:

    raise RuntimeError(
        f"Unexpectedly few changed rows: {changed_rows}/3732"
    )


if total_post_tokens <= 0:

    raise RuntimeError(
        "No post-edit tokens"
    )


global_mask_fraction = (
    total_mask_tokens
    / total_post_tokens
)


global_halo_fraction = (
    total_halo_tokens
    / total_post_tokens
)


if global_mask_fraction <= 0.0:

    raise RuntimeError(
        "Patch mask is empty"
    )


if global_mask_fraction >= 0.90:

    raise RuntimeError(
        "Patch mask covers implausibly large fraction of tokens"
    )


###############################################################################
# Write prepared dataset
###############################################################################

with OUTPUT.open(
    "w",
    encoding="utf-8",
) as f:

    for row in out_rows:

        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


###############################################################################
# Summary
###############################################################################

summary = []


summary.append(
    "=" * 100
)

summary.append(
    "MT-PATCHER CORRECTION MASK AUDIT V1"
)

summary.append(
    "=" * 100
)

summary.append(
    f"INPUT={INPUT}"
)

summary.append(
    f"OUTPUT={OUTPUT}"
)

summary.append(
    f"MODEL_TOKENIZER={MODEL}"
)

summary.append(
    ""
)

summary.append(
    f"rows={len(rows)}"
)

summary.append(
    f"changed_rows={changed_rows}"
)

summary.append(
    f"unchanged_rows={unchanged_rows}"
)

summary.append(
    ""
)

summary.append(
    f"source_path={source_path}"
)

summary.append(
    f"draft_path={draft_path}"
)

summary.append(
    f"postedit_path={post_path}"
)

summary.append(
    ""
)

summary.append(
    f"replace_rows={replace_rows}"
)

summary.append(
    f"insert_rows={insert_rows}"
)

summary.append(
    f"delete_rows={delete_rows}"
)

summary.append(
    f"delete_only_rows={delete_only_rows}"
)

summary.append(
    ""
)

summary.append(
    f"total_postedit_tokens={total_post_tokens}"
)

summary.append(
    f"total_patch_mask_tokens={total_mask_tokens}"
)

summary.append(
    f"total_halo1_tokens={total_halo_tokens}"
)

summary.append(
    f"GLOBAL_PATCH_MASK_FRACTION={global_mask_fraction:.6f}"
)

summary.append(
    f"GLOBAL_HALO1_MASK_FRACTION={global_halo_fraction:.6f}"
)

summary.append(
    ""
)

summary.append(
    "PATCH MASK FRACTION DISTRIBUTION"
)

for name, q in (
    ("p10", 0.10),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p90", 0.90),
    ("p95", 0.95),
):

    summary.append(
        f"{name}={percentile(mask_fracs, q):.6f}"
    )


summary.append(
    ""
)

summary.append(
    "HALO1 MASK FRACTION DISTRIBUTION"
)

for name, q in (
    ("p10", 0.10),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p90", 0.90),
    ("p95", 0.95),
):

    summary.append(
        f"{name}={percentile(halo_fracs, q):.6f}"
    )


summary.append(
    ""
)

summary.append(
    f"draft_len_mean={statistics.mean(draft_lens):.3f}"
)

summary.append(
    f"postedit_len_mean={statistics.mean(post_lens):.3f}"
)

summary.append(
    ""
)

summary.append(
    f"opcode_counts={opcode_counts}"
)

summary.append(
    ""
)

summary.append(
    "PATCH_MASK_DATASET_BUILD_PASS"
)


SUMMARY.write_text(
    "\n".join(summary)
    + "\n",
    encoding="utf-8",
)


###############################################################################
# Human-readable preview
###############################################################################

with PREVIEW.open(
    "w",
    encoding="utf-8",
) as f:

    for item in preview_items:

        f.write(
            "=" * 110
            + "\n"
        )

        f.write(
            f"ROW {item['index']}\n"
        )

        f.write(
            f"MASK_FRACTION={item['mask_fraction']:.4f}\n"
        )

        f.write(
            f"MASK_INDICES={item['mask']}\n"
        )

        f.write(
            f"MASK_TOKENS={item['tokens']}\n\n"
        )

        f.write(
            "SOURCE:\n"
            + item["source"]
            + "\n\n"
        )

        f.write(
            "DRAFT:\n"
            + item["draft"]
            + "\n\n"
        )

        f.write(
            "POST_EDIT:\n"
            + item["post"]
            + "\n\n"
        )


###############################################################################
# Terminal report
###############################################################################

print()
print(
    "\n".join(summary)
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
    "PATCH_DATASET=",
    OUTPUT,
)
PY

