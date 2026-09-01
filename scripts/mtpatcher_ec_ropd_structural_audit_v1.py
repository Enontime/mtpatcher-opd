#!/usr/bin/env python3

import json
import os
from collections import Counter, defaultdict
from pathlib import Path


DATA_ROOT = Path(os.environ["DATA_ROOT"])
EXP = os.environ["EXP"]

BASE = (
    DATA_ROOT
    / EXP
)

STATES = (
    BASE
    / "ec_ropd_intervention_states_v1.jsonl"
)

REPORT = (
    BASE
    / "ec_ropd_structural_audit_v1.json"
)


def load_jsonl(path):
    with path.open(
        encoding="utf-8",
    ) as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


rows = load_jsonl(
    STATES
)

if len(rows) != 6448:
    raise RuntimeError(
        f"Expected 6448 states, got {len(rows)}"
    )


by_parent = defaultdict(list)

for x in rows:
    by_parent[
        int(x["parent_index"])
    ].append(x)


# ------------------------------------------------------------
# Per-state structural checks.
# ------------------------------------------------------------

no_op = []
correction_not_in_postedit = []
source_span_not_exact = []
empty_prefix = []

for x in rows:

    if (
        x["translation_span"]
        == x["correction"]
    ):
        no_op.append(x)

    if not bool(
        x["correction_in_post_edit"]
    ):
        correction_not_in_postedit.append(
            x
        )

    if not bool(
        x["source_span_exact"]
    ):
        source_span_not_exact.append(
            x
        )

    if not x[
        "prefix_before_error"
    ]:
        empty_prefix.append(x)


# ------------------------------------------------------------
# Within-parent relations.
#
# IMPORTANT:
# Overlap itself does NOT invalidate an intervention state,
# because EC-ROPD v1 treats every error object independently.
# We inspect it only to understand the Patcher interface.
# ------------------------------------------------------------

overlap_pairs = []
same_span_pairs = []
conflicting_same_span = []

parents_with_overlap = set()
parents_with_same_span = set()

for parent, xs in by_parent.items():

    ordered = sorted(
        xs,
        key=lambda z: (
            int(
                z[
                    "translation_char_start"
                ]
            ),
            int(
                z[
                    "translation_char_end"
                ]
            ),
            int(
                z["error_id"]
            ),
        ),
    )

    for i in range(
        len(ordered)
    ):
        a = ordered[i]

        a1 = int(
            a[
                "translation_char_start"
            ]
        )

        a2 = int(
            a[
                "translation_char_end"
            ]
        )

        for j in range(
            i + 1,
            len(ordered),
        ):
            b = ordered[j]

            b1 = int(
                b[
                    "translation_char_start"
                ]
            )

            b2 = int(
                b[
                    "translation_char_end"
                ]
            )

            if b1 >= a2:
                break

            # Half-open interval overlap.
            if (
                max(a1, b1)
                < min(a2, b2)
            ):
                overlap_pairs.append(
                    (
                        parent,
                        a,
                        b,
                    )
                )

                parents_with_overlap.add(
                    parent
                )

            if (
                a1 == b1
                and a2 == b2
            ):
                same_span_pairs.append(
                    (
                        parent,
                        a,
                        b,
                    )
                )

                parents_with_same_span.add(
                    parent
                )

                if (
                    a["correction"]
                    != b["correction"]
                ):
                    conflicting_same_span.append(
                        (
                            parent,
                            a,
                            b,
                        )
                    )


# ------------------------------------------------------------
# Structural-clean candidate.
#
# This is ONLY reported.
# We are not yet training on it.
# ------------------------------------------------------------

conflicting_keys = set()

for parent, a, b in conflicting_same_span:

    conflicting_keys.add(
        (
            parent,
            int(a["error_id"]),
        )
    )

    conflicting_keys.add(
        (
            parent,
            int(b["error_id"]),
        )
    )


clean = []

reject_reason_counts = Counter()

for x in rows:

    key = (
        int(x["parent_index"]),
        int(x["error_id"]),
    )

    reasons = []

    if (
        x["translation_span"]
        == x["correction"]
    ):
        reasons.append(
            "no_op_correction"
        )

    if not bool(
        x["correction_in_post_edit"]
    ):
        reasons.append(
            "correction_not_in_postedit"
        )

    if not bool(
        x["source_span_exact"]
    ):
        reasons.append(
            "source_span_not_exact"
        )

    if key in conflicting_keys:
        reasons.append(
            "conflicting_same_span"
        )

    if not reasons:
        clean.append(x)

    else:
        for r in reasons:
            reject_reason_counts[r] += 1


# ------------------------------------------------------------
# Report helpers.
# ------------------------------------------------------------

def compact(x):
    return {
        "parent_index":
            int(
                x["parent_index"]
            ),

        "error_id":
            int(
                x["error_id"]
            ),

        "error_type":
            x["error_type"],

        "source_span":
            x["source_span"],

        "translation_span":
            x["translation_span"],

        "correction":
            x["correction"],

        "student_translation":
            x["student_translation"],

        "post_edit":
            x["post_edit"],
    }


report = {
    "states":
        len(rows),

    "parents":
        len(by_parent),

    "no_op_correction":
        len(no_op),

    "correction_not_in_postedit":
        len(
            correction_not_in_postedit
        ),

    "source_span_not_exact":
        len(
            source_span_not_exact
        ),

    "empty_prefix":
        len(empty_prefix),

    "overlap_pairs":
        len(overlap_pairs),

    "parents_with_overlap":
        len(
            parents_with_overlap
        ),

    "same_span_pairs":
        len(same_span_pairs),

    "parents_with_same_span":
        len(
            parents_with_same_span
        ),

    "conflicting_same_span_pairs":
        len(
            conflicting_same_span
        ),

    "structural_clean_states":
        len(clean),

    "structural_clean_fraction":
        len(clean)
        / len(rows),

    "reject_reason_counts":
        dict(
            reject_reason_counts
        ),

    "samples": {
        "no_op":
            [
                compact(x)
                for x in no_op[:10]
            ],

        "correction_not_in_postedit":
            [
                compact(x)
                for x
                in correction_not_in_postedit[
                    :10
                ]
            ],

        "source_span_not_exact":
            [
                compact(x)
                for x
                in source_span_not_exact[
                    :10
                ]
            ],

        "conflicting_same_span":
            [
                {
                    "parent_index":
                        parent,

                    "a":
                        compact(a),

                    "b":
                        compact(b),
                }
                for parent, a, b
                in conflicting_same_span[
                    :10
                ]
            ],

        "overlap":
            [
                {
                    "parent_index":
                        parent,

                    "a":
                        compact(a),

                    "b":
                        compact(b),
                }
                for parent, a, b
                in overlap_pairs[
                    :10
                ]
            ],
    },
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
print("EC-ROPD STRUCTURAL STATE AUDIT")
print("=" * 100)

print(
    f"STATES={len(rows)}"
)

print(
    f"PARENTS={len(by_parent)}"
)

print()

print(
    f"NO_OP_CORRECTION="
    f"{len(no_op)}"
)

print(
    f"CORRECTION_NOT_IN_POSTEDIT="
    f"{len(correction_not_in_postedit)}"
)

print(
    f"SOURCE_SPAN_NOT_EXACT="
    f"{len(source_span_not_exact)}"
)

print(
    f"EMPTY_PREFIX="
    f"{len(empty_prefix)}"
)

print()

print(
    f"OVERLAP_PAIRS="
    f"{len(overlap_pairs)}"
)

print(
    f"PARENTS_WITH_OVERLAP="
    f"{len(parents_with_overlap)}"
)

print(
    f"SAME_SPAN_PAIRS="
    f"{len(same_span_pairs)}"
)

print(
    f"CONFLICTING_SAME_SPAN_PAIRS="
    f"{len(conflicting_same_span)}"
)

print()

print(
    f"STRUCTURAL_CLEAN="
    f"{len(clean)}/{len(rows)} "
    f"({len(clean)/len(rows):.4%})"
)

print(
    "REJECT_REASONS=",
    dict(
        reject_reason_counts
    ),
)


def show(
    title,
    xs,
):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    if not xs:
        print("NONE")
        return

    for x in xs[:10]:

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
            "STUDENT:",
            x["student_translation"],
        )

        print(
            "POST_EDIT:",
            x["post_edit"],
        )


show(
    "NO-OP CORRECTION SAMPLES",
    no_op,
)

show(
    "CORRECTION NOT IN POST-EDIT SAMPLES",
    correction_not_in_postedit,
)

show(
    "SOURCE SPAN NOT EXACT SAMPLES",
    source_span_not_exact,
)


print()
print("=" * 100)
print("OVERLAP SAMPLES")
print("=" * 100)

if not overlap_pairs:
    print("NONE")

for parent, a, b in overlap_pairs[:10]:

    print()

    print(
        f"PARENT={parent}"
    )

    print(
        f"A ERROR_ID={a['error_id']} "
        f"{a['translation_span']!r}"
        f" -> "
        f"{a['correction']!r}"
    )

    print(
        f"B ERROR_ID={b['error_id']} "
        f"{b['translation_span']!r}"
        f" -> "
        f"{b['correction']!r}"
    )


print()
print("=" * 100)
print("CONFLICTING SAME-SPAN SAMPLES")
print("=" * 100)

if not conflicting_same_span:
    print("NONE")

for (
    parent,
    a,
    b,
) in conflicting_same_span[:10]:

    print()

    print(
        f"PARENT={parent}"
    )

    print(
        f"A ERROR_ID={a['error_id']} "
        f"{a['translation_span']!r}"
        f" -> "
        f"{a['correction']!r}"
    )

    print(
        f"B ERROR_ID={b['error_id']} "
        f"{b['translation_span']!r}"
        f" -> "
        f"{b['correction']!r}"
    )


print()
print(
    "REPORT=",
    REPORT,
)

print(
    "EC_ROPD_STRUCTURAL_AUDIT_V1_PASS"
)
