#!/usr/bin/env python3

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--expected-rows", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.output} exists; use --overwrite explicitly"
        )

    rows = []

    with args.input.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue

            row = json.loads(line)

            messages = row.get("messages")
            target = row.get("target_translation")

            if not isinstance(messages, list) or len(messages) != 1:
                raise ValueError(
                    f"line {line_no}: expected exactly one source message"
                )

            msg = messages[0]
            if (
                not isinstance(msg, dict)
                or msg.get("role") != "user"
                or not isinstance(msg.get("content"), str)
            ):
                raise ValueError(
                    f"line {line_no}: invalid user message"
                )

            if not isinstance(target, str) or not target.strip():
                raise ValueError(
                    f"line {line_no}: invalid target_translation"
                )

            out = deepcopy(row)

            # Keep clean target_translation as semantic source of truth.
            out["messages"] = [
                deepcopy(msg),
                {
                    "role": "assistant",
                    "content": target,
                },
            ]

            # Make non-thinking semantics explicit per row.
            out["enable_thinking"] = False

            rows.append(out)

    if args.expected_rows is not None and len(rows) != args.expected_rows:
        raise AssertionError(
            f"row count {len(rows)} != expected {args.expected_rows}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_parquet(
        args.output,
        index=False,
    )

    print("source:", args.input)
    print("source_sha256:", sha256(args.input))
    print("output:", args.output)
    print("rows:", len(rows))
    print("output_sha256:", sha256(args.output))
    print("PREPARE_VERL_SFT_FULL_PASS")


if __name__ == "__main__":
    main()
