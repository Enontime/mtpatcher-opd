#!/usr/bin/env python3

import hashlib
import json
import os
import statistics
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


EXP = os.environ["EXP"]

DATA_ROOT = Path(os.environ["DATA_ROOT"])
MODEL_ROOT = Path(os.environ["MODEL_ROOT"])

EXP_DATA = (
    DATA_ROOT
    / EXP
)

FEEDBACK = (
    EXP_DATA
    / "feedback_qwen3_8b_merged6565.jsonl"
)

PE = (
    EXP_DATA
    / "pe_k1_clean3732.jsonl"
)

OUT = (
    EXP_DATA
    / "ec_ropd_intervention_states_v1.jsonl"
)

REPORT = (
    EXP_DATA
    / "ec_ropd_intervention_states_v1_report.json"
)

MODEL = (
    MODEL_ROOT
    / "Qwen3-0.6B"
)


def load_jsonl(path):
    rows = []

    with path.open(
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def positions(
    text,
    needle,
):
    if not needle:
        return []

    out = []
    start = 0

    while True:
        p = text.find(
            needle,
            start,
        )

        if p < 0:
            break

        out.append(p)

        start = (
            p + 1
        )

    return out


feedback_rows = load_jsonl(
    FEEDBACK
)

pe_rows = load_jsonl(
    PE
)


if len(pe_rows) != 3732:
    raise RuntimeError(
        f"Expected PE3732, "
        f"got {len(pe_rows)}"
    )


fb_map = {}

for row in feedback_rows:
    idx = int(
        row["index"]
    )

    if idx in fb_map:
        raise RuntimeError(
            f"Duplicate feedback "
            f"index={idx}"
        )

    fb_map[idx] = row


pe_indices = []

for row in pe_rows:
    idx = int(
        row["index"]
    )

    if idx in pe_indices:
        raise RuntimeError(
            f"Duplicate PE "
            f"index={idx}"
        )

    pe_indices.append(idx)


missing_feedback = [
    idx
    for idx in pe_indices
    if idx not in fb_map
]

if missing_feedback:
    raise RuntimeError(
        f"Missing feedback rows: "
        f"{missing_feedback[:20]}"
    )


tokenizer = (
    AutoTokenizer
    .from_pretrained(
        MODEL,
        local_files_only=True,
    )
)


instances = []

error_type_counts = Counter()
error_count_hist = Counter()

reject_counts = Counter()

rows_all_errors_mapped = 0
rows_any_error_mapped = 0

prefix_token_lengths = []
correction_token_lengths = []

total_error_objects = 0


for pe_row in pe_rows:

    idx = int(
        pe_row["index"]
    )

    fb = fb_map[idx]

    draft = str(
        fb.get(
            "student_translation",
            "",
        )
        or ""
    )

    source = str(
        fb.get(
            "source",
            "",
        )
        or ""
    )

    post_edit = str(
        fb.get(
            "post_edit",
            "",
        )
        or ""
    )

    errors = fb.get(
        "errors",
        [],
    )

    if not isinstance(
        errors,
        list,
    ):
        raise RuntimeError(
            f"errors is not list "
            f"index={idx}"
        )


    error_count_hist[
        len(errors)
    ] += 1


    mapped_this_row = 0


    for error_id, error in enumerate(
        errors
    ):

        total_error_objects += 1

        if not isinstance(
            error,
            dict,
        ):
            reject_counts[
                "non_dict_error"
            ] += 1
            continue


        translation_span = str(
            error.get(
                "translation_span",
                "",
            )
            or ""
        ).strip()

        correction = str(
            error.get(
                "correction",
                "",
            )
            or ""
        ).strip()

        source_span = str(
            error.get(
                "source_span",
                "",
            )
            or ""
        ).strip()

        error_type = str(
            error.get(
                "error_type",
                "UNKNOWN",
            )
            or "UNKNOWN"
        ).strip()


        if not translation_span:
            reject_counts[
                "empty_translation_span"
            ] += 1
            continue

        if not correction:
            reject_counts[
                "empty_correction"
            ] += 1
            continue


        hits = positions(
            draft,
            translation_span,
        )


        if len(hits) == 0:
            reject_counts[
                "translation_span_missing"
            ] += 1
            continue


        if len(hits) > 1:
            reject_counts[
                "translation_span_ambiguous"
            ] += 1
            continue


        start = hits[0]

        end = (
            start
            + len(
                translation_span
            )
        )


        prefix_before = (
            draft[:start]
        )

        original_prefix = (
            draft[:end]
        )

        corrected_prefix = (
            prefix_before
            + correction
        )

        original_suffix = (
            draft[end:]
        )


        correction_ids = (
            tokenizer.encode(
                correction,
                add_special_tokens=False,
            )
        )

        original_prefix_ids = (
            tokenizer.encode(
                original_prefix,
                add_special_tokens=False,
            )
        )

        corrected_prefix_ids = (
            tokenizer.encode(
                corrected_prefix,
                add_special_tokens=False,
            )
        )


        if not correction_ids:
            reject_counts[
                "empty_correction_tokens"
            ] += 1
            continue


        source_span_exact = (
            bool(source_span)
            and source_span in source
        )

        correction_in_post_edit = (
            correction
            in post_edit
        )


        instance = {
            "parent_index":
                idx,

            # Pure provenance identifier.
            # No priority/order semantics.
            "error_id":
                error_id,

            "source":
                source,

            "student_translation":
                draft,

            "post_edit":
                post_edit,

            "source_span":
                source_span,

            "translation_span":
                translation_span,

            "correction":
                correction,

            "error_type":
                error_type,

            "explanation":
                str(
                    error.get(
                        "explanation",
                        "",
                    )
                    or ""
                ),

            "translation_char_start":
                start,

            "translation_char_end":
                end,

            "prefix_before_error":
                prefix_before,

            "original_prefix_through_error":
                original_prefix,

            "corrected_prefix":
                corrected_prefix,

            "original_suffix_after_error":
                original_suffix,

            "original_prefix_token_count":
                len(
                    original_prefix_ids
                ),

            "corrected_prefix_token_count":
                len(
                    corrected_prefix_ids
                ),

            "correction_token_count":
                len(
                    correction_ids
                ),

            "source_span_exact":
                source_span_exact,

            "correction_in_post_edit":
                correction_in_post_edit,

            "construction":
                (
                    "student_prefix_before_error"
                    "+patcher_correction"
                ),
        }


        instances.append(
            instance
        )

        mapped_this_row += 1

        error_type_counts[
            error_type
        ] += 1

        prefix_token_lengths.append(
            len(
                corrected_prefix_ids
            )
        )

        correction_token_lengths.append(
            len(
                correction_ids
            )
        )


    if mapped_this_row > 0:
        rows_any_error_mapped += 1

    if (
        len(errors) > 0
        and mapped_this_row
        == len(errors)
    ):
        rows_all_errors_mapped += 1


with OUT.open(
    "w",
    encoding="utf-8",
) as f:

    for x in instances:

        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )


report = {
    "method":
        "EC_ROPD_INTERVENTION_STATE_V1",

    "pe_rows":
        len(pe_rows),

    "feedback_rows":
        len(feedback_rows),

    "total_error_objects":
        total_error_objects,

    "eligible_intervention_instances":
        len(instances),

    "instance_coverage":
        (
            len(instances)
            / total_error_objects
            if total_error_objects
            else 0.0
        ),

    "rows_any_error_mapped":
        rows_any_error_mapped,

    "rows_all_errors_mapped":
        rows_all_errors_mapped,

    "row_any_coverage":
        rows_any_error_mapped
        / len(pe_rows),

    "row_all_coverage":
        rows_all_errors_mapped
        / len(pe_rows),

    "reject_counts":
        dict(
            sorted(
                reject_counts.items()
            )
        ),

    "error_count_histogram":
        {
            str(k): v
            for k, v
            in sorted(
                error_count_hist.items()
            )
        },

    "error_type_counts":
        dict(
            error_type_counts
            .most_common()
        ),

    "correction_in_post_edit":
        sum(
            bool(
                x[
                    "correction_in_post_edit"
                ]
            )
            for x in instances
        ),

    "source_span_exact":
        sum(
            bool(
                x[
                    "source_span_exact"
                ]
            )
            for x in instances
        ),

    "corrected_prefix_tokens_mean":
        (
            statistics.mean(
                prefix_token_lengths
            )
            if prefix_token_lengths
            else None
        ),

    "corrected_prefix_tokens_median":
        (
            statistics.median(
                prefix_token_lengths
            )
            if prefix_token_lengths
            else None
        ),

    "correction_tokens_mean":
        (
            statistics.mean(
                correction_token_lengths
            )
            if correction_token_lengths
            else None
        ),

    "correction_tokens_median":
        (
            statistics.median(
                correction_token_lengths
            )
            if correction_token_lengths
            else None
        ),

    "pe_sha256":
        sha256(
            PE
        ),

    "feedback_sha256":
        sha256(
            FEEDBACK
        ),

    "output_sha256":
        sha256(
            OUT
        ),
}


REPORT.write_text(
    json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print("=" * 100)
print("EC-ROPD INTERVENTION STATE FEASIBILITY")
print("=" * 100)

print(
    json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    )
)

print()
print("=" * 100)
print("FIRST 20 INTERVENTION STATES")
print("=" * 100)


for x in instances[:20]:

    print()
    print(
        f"PARENT={x['parent_index']} "
        f"ERROR_ID={x['error_id']} "
        f"TYPE={x['error_type']}"
    )

    print(
        "SOURCE_SPAN:",
        repr(
            x["source_span"]
        ),
    )

    print(
        "TRANSLATION_SPAN:",
        repr(
            x["translation_span"]
        ),
    )

    print(
        "CORRECTION:",
        repr(
            x["correction"]
        ),
    )

    print(
        "PREFIX_BEFORE:",
        repr(
            x["prefix_before_error"]
        ),
    )

    print(
        "CORRECTED_PREFIX:",
        repr(
            x["corrected_prefix"]
        ),
    )


print()
print(
    "OUTPUT=",
    OUT,
)

print(
    "REPORT=",
    REPORT,
)

print(
    "EC_ROPD_STATE_FEASIBILITY_PASS"
)
