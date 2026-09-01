#!/usr/bin/env python3

import argparse
import json

from transformers import AutoTokenizer


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        required=True,
    )

    ap.add_argument(
        "--data",
        required=True,
    )

    ap.add_argument(
        "--max-length",
        type=int,
        default=1024,
    )

    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
    )

    lengths = []

    over = []

    with open(
        args.data,
        encoding="utf-8-sig",
    ) as f:

        for pos, line in enumerate(f):

            if not line.strip():
                continue

            x = json.loads(line)

            prompt = tok.apply_chat_template(
                x["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            pids = tok(
                prompt,
                add_special_tokens=False,
            )["input_ids"]

            tids = tok(
                x["target_translation"].strip(),
                add_special_tokens=False,
            )["input_ids"]

            n = (
                len(pids)
                + len(tids)
                + 1
            )

            lengths.append(n)

            if n > args.max_length:
                over.append(
                    (
                        pos,
                        x["index"],
                        n,
                    )
                )

    lengths.sort()

    def pct(q):

        if not lengths:
            return 0

        idx = int(
            (len(lengths) - 1)
            * q
        )

        return lengths[idx]

    result = {
        "rows":
            len(lengths),

        "p50":
            pct(0.50),

        "p95":
            pct(0.95),

        "p99":
            pct(0.99),

        "max":
            max(lengths),

        "over_1024":
            len(over),

        "first_over":
            over[:20],
    }

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    if over:
        raise RuntimeError(
            f"{len(over)} sequences "
            f"exceed max_length="
            f"{args.max_length}"
        )

    print(
        "RQ0_B_TOKEN_LENGTH_PASS"
    )


if __name__ == "__main__":
    main()
