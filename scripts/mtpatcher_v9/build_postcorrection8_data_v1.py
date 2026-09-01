#!/usr/bin/env python3

import hashlib
import json
import os
import statistics
from collections import Counter
from pathlib import Path


EXP = "mtpatcher_v3_full6565_20260823"
HORIZON = 8

DATA_ROOT = Path(os.environ["DATA_ROOT"])
ROOT = Path(os.environ["ROOT"])

INPUT = (
    DATA_ROOT
    / EXP
    / "patch_aware_k1_clean3732_v2.jsonl"
)

OUTPUT = (
    DATA_ROOT
    / EXP
    / "patch_aware_k1_clean3732_postcorr8_v1.jsonl"
)

AUDIT_DIR = (
    ROOT
    / "results"
    / EXP
    / "postcorrection8_data_v1"
)

SUMMARY = (
    AUDIT_DIR
    / "summary.txt"
)

PREVIEW = (
    AUDIT_DIR
    / "preview.jsonl"
)


def load_jsonl(path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        for lineno, line in enumerate(
            f,
            start=1,
        ):
            if not line.strip():
                continue

            x = json.loads(line)

            if not isinstance(
                x,
                dict,
            ):
                raise RuntimeError(
                    f"Non-dict row "
                    f"{path}:{lineno}"
                )

            rows.append(x)

    return rows


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda:
                f.read(
                    1024 * 1024
                ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def q(xs, frac):
    if not xs:
        return 0.0

    ys = sorted(xs)

    if len(ys) == 1:
        return float(
            ys[0]
        )

    pos = (
        frac
        * (
            len(ys) - 1
        )
    )

    lo = int(pos)
    hi = min(
        lo + 1,
        len(ys) - 1,
    )

    w = pos - lo

    return float(
        ys[lo]
        * (1 - w)
        + ys[hi]
        * w
    )


rows = load_jsonl(
    INPUT
)

if len(rows) != 3732:
    raise RuntimeError(
        f"Expected 3732 rows, "
        f"got {len(rows)}"
    )


total_response = 0
total_patch = 0
total_halo1 = 0
total_post8 = 0

post8_counts = []
post8_fracs = []
patch_fracs = []
halo1_fracs = []

added_counts = []

op_counts = Counter()
changed_op_counts = Counter()

rows_post8_eq_patch = 0
rows_post8_eq_halo1 = 0
rows_post8_eq_full = 0

rows_post8_ge_50pct = 0
rows_post8_ge_75pct = 0
rows_post8_ge_90pct = 0

rows_with_eos = 0

out_rows = []
preview = []


for row_no, row in enumerate(
    rows
):
    meta = row.get(
        "_patch_aware_v2"
    )

    if not isinstance(
        meta,
        dict,
    ):
        raise RuntimeError(
            f"Missing _patch_aware_v2 "
            f"row={row_no}"
        )


    response_n = int(
        meta[
            "response_positions_with_eos"
        ]
    )

    post_n = int(
        meta[
            "postedit_token_count"
        ]
    )

    if response_n != post_n + 1:
        raise RuntimeError(
            f"Response/EOS mismatch "
            f"row={row_no}: "
            f"{response_n} != "
            f"{post_n}+1"
        )


    patch = [
        int(x)
        for x in meta[
            "patch_mask"
        ]
    ]

    halo1 = [
        int(x)
        for x in meta[
            "patch_halo1_mask"
        ]
    ]

    full = [
        int(x)
        for x in meta[
            "full_correction_mask"
        ]
    ]

    ops = meta.get(
        "diff_opcodes"
    )

    if not isinstance(
        ops,
        list,
    ):
        raise RuntimeError(
            f"Missing diff_opcodes "
            f"row={row_no}"
        )


    if full != list(
        range(
            response_n
        )
    ):
        raise RuntimeError(
            f"Unexpected full mask "
            f"row={row_no}"
        )


    post8 = set()


    for op in ops:
        tag = str(
            op.get(
                "tag",
                "",
            )
        )

        j1 = int(
            op[
                "post_start"
            ]
        )

        j2 = int(
            op[
                "post_end"
            ]
        )

        op_counts[
            tag
        ] += 1


        if tag == "equal":
            continue


        changed_op_counts[
            tag
        ] += 1


        if (
            j1 < 0
            or j2 < j1
            or j2 > post_n
        ):
            raise RuntimeError(
                f"Invalid opcode "
                f"row={row_no}: "
                f"{op}"
            )


        # ----------------------------------------------------
        # Frozen PostCorrection8 definition.
        #
        # replace / insert:
        #   corrected span [j1, j2)
        #   + at most 8 following response positions.
        #
        # delete:
        #   no corrected target span exists;
        #   begin at the first surviving post-edit position.
        #   Sentence-final deletion therefore selects EOS only.
        # ----------------------------------------------------

        if tag in {
            "replace",
            "insert",
        }:
            end = min(
                j2 + HORIZON,
                response_n,
            )

            post8.update(
                range(
                    j1,
                    end,
                )
            )

        elif tag == "delete":
            end = min(
                j1 + HORIZON,
                response_n,
            )

            post8.update(
                range(
                    j1,
                    end,
                )
            )

        else:
            raise RuntimeError(
                f"Unknown opcode tag "
                f"row={row_no}: "
                f"{tag}"
            )


    post8 = sorted(
        post8
    )


    if not post8:
        raise RuntimeError(
            f"Empty post8 "
            f"row={row_no}"
        )


    if post8 != sorted(
        set(
            post8
        )
    ):
        raise RuntimeError(
            f"Post8 not sorted/unique "
            f"row={row_no}"
        )


    if (
        min(post8) < 0
        or max(post8)
        >= response_n
    ):
        raise RuntimeError(
            f"Post8 out of range "
            f"row={row_no}"
        )


    # Existing patch supervision must be retained.
    if not set(
        patch
    ).issubset(
        set(
            post8
        )
    ):
        missing = sorted(
            set(patch)
            - set(post8)
        )

        raise RuntimeError(
            f"Patch not subset of post8 "
            f"row={row_no}: "
            f"{missing}"
        )


    post8_frac = (
        len(post8)
        / response_n
    )

    patch_frac = (
        len(patch)
        / response_n
    )

    halo1_frac = (
        len(halo1)
        / response_n
    )


    if post8 == patch:
        rows_post8_eq_patch += 1

    if post8 == halo1:
        rows_post8_eq_halo1 += 1

    if post8 == full:
        rows_post8_eq_full += 1

    if post8_frac >= 0.50:
        rows_post8_ge_50pct += 1

    if post8_frac >= 0.75:
        rows_post8_ge_75pct += 1

    if post8_frac >= 0.90:
        rows_post8_ge_90pct += 1

    if (
        response_n - 1
        in post8
    ):
        rows_with_eos += 1


    total_response += response_n
    total_patch += len(
        patch
    )
    total_halo1 += len(
        halo1
    )
    total_post8 += len(
        post8
    )

    post8_counts.append(
        len(post8)
    )

    post8_fracs.append(
        post8_frac
    )

    patch_fracs.append(
        patch_frac
    )

    halo1_fracs.append(
        halo1_frac
    )

    added_counts.append(
        len(post8)
        - len(patch)
    )


    x = dict(
        row
    )

    m = dict(
        meta
    )

    m[
        "post_correction8_mask"
    ] = post8

    m[
        "post_correction8_token_count"
    ] = len(
        post8
    )

    m[
        "post_correction8_fraction"
    ] = post8_frac

    m[
        "post_correction8_horizon"
    ] = HORIZON

    m[
        "post_correction8_definition"
    ] = (
        "replace/insert: "
        "[post_start, min(post_end+8, response_n)); "
        "delete: "
        "[post_start, min(post_start+8, response_n)); "
        "response_n includes EOS"
    )

    x[
        "_patch_aware_v2"
    ] = m

    out_rows.append(
        x
    )


    if (
        len(preview) < 30
        and len(post8)
        > len(patch)
    ):
        preview.append(
            {
                "row_no":
                    row_no,

                "index":
                    row.get(
                        "index"
                    ),

                "patch_count":
                    len(patch),

                "halo1_count":
                    len(halo1),

                "post8_count":
                    len(post8),

                "response_count":
                    response_n,

                "patch_fraction":
                    patch_frac,

                "halo1_fraction":
                    halo1_frac,

                "post8_fraction":
                    post8_frac,

                "patch_mask":
                    patch,

                "halo1_mask":
                    halo1,

                "post8_mask":
                    post8,

                "diff_opcodes":
                    ops,

                "target_translation":
                    row.get(
                        "target_translation"
                    ),
            }
        )


AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

with OUTPUT.open(
    "w",
    encoding="utf-8",
) as f:
    for x in out_rows:
        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )


with PREVIEW.open(
    "w",
    encoding="utf-8",
) as f:
    for x in preview:
        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )


global_patch = (
    total_patch
    / total_response
)

global_halo1 = (
    total_halo1
    / total_response
)

global_post8 = (
    total_post8
    / total_response
)


lines = []

lines.append(
    "=" * 100
)

lines.append(
    "POST-CORRECTION-8 DATA AUDIT V1"
)

lines.append(
    "=" * 100
)

lines.append(
    f"INPUT={INPUT}"
)

lines.append(
    f"OUTPUT={OUTPUT}"
)

lines.append(
    f"ROWS={len(rows)}"
)

lines.append(
    f"HORIZON={HORIZON}"
)

lines.append(
    ""
)

lines.append(
    f"total_response_positions_with_eos="
    f"{total_response}"
)

lines.append(
    f"total_patch_positions="
    f"{total_patch}"
)

lines.append(
    f"total_halo1_positions="
    f"{total_halo1}"
)

lines.append(
    f"total_post8_positions="
    f"{total_post8}"
)

lines.append(
    ""
)

lines.append(
    f"GLOBAL_PATCH_FRACTION="
    f"{global_patch:.6f}"
)

lines.append(
    f"GLOBAL_HALO1_FRACTION="
    f"{global_halo1:.6f}"
)

lines.append(
    f"GLOBAL_POST8_FRACTION="
    f"{global_post8:.6f}"
)

lines.append(
    f"POST8_SUPERVISION_SAVING_VS_FULL="
    f"{1.0 - global_post8:.6f}"
)

lines.append(
    ""
)

lines.append(
    f"ROWS_POST8_EQ_PATCH="
    f"{rows_post8_eq_patch}"
)

lines.append(
    f"ROWS_POST8_EQ_HALO1="
    f"{rows_post8_eq_halo1}"
)

lines.append(
    f"ROWS_POST8_EQ_FULL="
    f"{rows_post8_eq_full}"
)

lines.append(
    f"ROWS_POST8_GE_50PCT="
    f"{rows_post8_ge_50pct}"
)

lines.append(
    f"ROWS_POST8_GE_75PCT="
    f"{rows_post8_ge_75pct}"
)

lines.append(
    f"ROWS_POST8_GE_90PCT="
    f"{rows_post8_ge_90pct}"
)

lines.append(
    f"ROWS_POST8_INCLUDES_EOS="
    f"{rows_with_eos}"
)

lines.append(
    ""
)

lines.append(
    "POST8 TOKEN COUNT"
)

for name, frac in [
    ("p10", 0.10),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p90", 0.90),
    ("p95", 0.95),
]:
    lines.append(
        f"{name}="
        f"{q(post8_counts, frac):.3f}"
    )

lines.append(
    f"mean="
    f"{statistics.mean(post8_counts):.3f}"
)

lines.append(
    ""
)

lines.append(
    "POST8 FRACTION"
)

for name, frac in [
    ("p10", 0.10),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p90", 0.90),
    ("p95", 0.95),
]:
    lines.append(
        f"{name}="
        f"{q(post8_fracs, frac):.6f}"
    )

lines.append(
    f"mean="
    f"{statistics.mean(post8_fracs):.6f}"
)

lines.append(
    ""
)

lines.append(
    "ADDED TOKENS VS PATCH"
)

for name, frac in [
    ("p10", 0.10),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p90", 0.90),
    ("p95", 0.95),
]:
    lines.append(
        f"{name}="
        f"{q(added_counts, frac):.3f}"
    )

lines.append(
    f"mean="
    f"{statistics.mean(added_counts):.3f}"
)

lines.append(
    ""
)

lines.append(
    f"ALL_OPCODE_COUNTS="
    f"{dict(op_counts)}"
)

lines.append(
    f"CHANGED_OPCODE_COUNTS="
    f"{dict(changed_op_counts)}"
)

lines.append(
    ""
)

lines.append(
    f"INPUT_SHA256="
    f"{sha256(INPUT)}"
)

lines.append(
    f"OUTPUT_SHA256="
    f"{sha256(OUTPUT)}"
)

lines.append(
    ""
)

lines.append(
    "POSTCORRECTION8_DATA_V1_ALL_PASS"
)


text = "\n".join(
    lines
) + "\n"

SUMMARY.write_text(
    text,
    encoding="utf-8",
)

print(
    text
)

print(
    "PREVIEW=",
    PREVIEW
)
