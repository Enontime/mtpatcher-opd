#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.pe_pds_v1.eaec_core_v1 import (
    derive_edit_core,
    occurrences,
)


def eligible(x):
    errors = x.get("errors")
    post = x.get("post_edit")
    student = x.get(
        "student_translation"
    )

    return (
        x.get("parse_ok") is True
        and x.get(
            "has_error"
        ) is True
        and isinstance(
            errors,
            list,
        )
        and len(errors) > 0
        and isinstance(
            post,
            str,
        )
        and bool(post.strip())
        and isinstance(
            student,
            str,
        )
        and post.strip()
        != student.strip()
    )


def atomic_write(
    path: Path,
    data: bytes,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        str(path) + ".tmp"
    )

    tmp.write_bytes(data)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        required=True,
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    ap.add_argument(
        "--expect-selected",
        type=int,
        default=11792,
    )

    args = ap.parse_args()

    rows = []

    with open(
        args.input,
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                x = json.loads(line)

                if eligible(x):
                    rows.append(x)

    if (
        len(rows)
        != args.expect_selected
    ):
        raise RuntimeError(
            "PE selected population drift: "
            f"{len(rows)} != "
            f"{args.expect_selected}"
        )

    output_rows = []

    seen = set()

    stats = {
        "selected_sources":
            len(rows),
        "total_error_objects": 0,
        "exact_unique_anchor":
            0,
        "anchor_missing": 0,
        "anchor_ambiguous": 0,
        "zero_core": 0,
        "kept_patches": 0,
        "insertion_boundary_ops":
            0,
    }

    for x in rows:
        sid = int(
            x["index"]
        )

        if sid in seen:
            raise RuntimeError(
                f"duplicate source_id={sid}"
            )

        seen.add(sid)

        student = x[
            "student_translation"
        ]

        patches = []

        for e in x["errors"]:
            stats[
                "total_error_objects"
            ] += 1

            old = e[
                "translation_span"
            ]
            correction = e[
                "correction"
            ]

            if not (
                isinstance(old, str)
                and old
                and isinstance(
                    correction,
                    str,
                )
                and correction
            ):
                stats[
                    "zero_core"
                ] += 1
                continue

            hits = occurrences(
                student,
                old,
            )

            if len(hits) == 0:
                stats[
                    "anchor_missing"
                ] += 1
                continue

            if len(hits) > 1:
                stats[
                    "anchor_ambiguous"
                ] += 1
                continue

            stats[
                "exact_unique_anchor"
            ] += 1

            core, extra = (
                derive_edit_core(
                    old,
                    correction,
                )
            )

            stats[
                "insertion_boundary_ops"
            ] += extra[
                "insertion_boundary_ops"
            ]

            if not core:
                stats[
                    "zero_core"
                ] += 1
                continue

            patches.append(
                {
                    "old_span": old,
                    "correction":
                        correction,
                    "error_type":
                        e.get(
                            "error_type",
                            "",
                        ),
                    "core_char_intervals":
                        [
                            [a, b]
                            for a, b
                            in core
                        ],
                }
            )

            stats[
                "kept_patches"
            ] += 1

        output_rows.append(
            {
                "source_id": sid,
                "source":
                    x["source"],
                "base_student_translation":
                    student,
                "patches":
                    patches,
            }
        )

    output_rows.sort(
        key=lambda x:
        x["source_id"]
    )

    payload = (
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for row
            in output_rows
        )
        .encode("utf-8")
    )

    out = Path(
        args.output
    )

    atomic_write(
        out,
        payload,
    )

    sha = hashlib.sha256(
        payload
    ).hexdigest()

    atomic_write(
        Path(
            str(out)
            + ".sha256"
        ),
        (
            sha + "  "
            + out.name
            + "\n"
        ).encode("utf-8"),
    )

    summary = {
        "status":
            "PASS_EAEC_PATCHBANK_V1",
        "output":
            str(out),
        "sha256":
            sha,
        **stats,
    }

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
