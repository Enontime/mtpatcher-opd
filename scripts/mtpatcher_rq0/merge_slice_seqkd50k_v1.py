#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for b in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(b)

    return h.hexdigest()


def read_jsonl(path):
    rows = []

    with open(
        path,
        encoding="utf-8-sig",
    ) as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def write_jsonl(path, rows):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--sources",
        required=True,
    )

    ap.add_argument(
        "--shard-dir",
        required=True,
    )

    ap.add_argument(
        "--output-dir",
        required=True,
    )

    args = ap.parse_args()

    sources = read_jsonl(
        args.sources
    )

    if len(sources) != 50000:
        raise RuntimeError(
            f"sources={len(sources)}"
        )

    source_map = {
        int(x["index"]): x
        for x in sources
    }

    merged = {}

    duplicates = 0

    shard_dir = Path(
        args.shard_dir
    )

    for i in range(16):

        p = shard_dir / (
            f"device_{i}.jsonl"
        )

        if not p.exists():
            raise RuntimeError(
                f"missing shard={p}"
            )

        rows = read_jsonl(p)

        print(
            f"SHARD={i:02d} "
            f"ROWS={len(rows)}"
        )

        for x in rows:

            idx = int(
                x["index"]
            )

            if idx in merged:
                duplicates += 1

                # Resume may theoretically
                # duplicate exact records.
                if (
                    merged[idx]["target_translation"]
                    !=
                    x["target_translation"]
                ):
                    raise RuntimeError(
                        f"non-identical duplicate "
                        f"index={idx}"
                    )

                continue

            merged[idx] = x

    if set(merged) != set(
        range(50000)
    ):
        missing = sorted(
            set(range(50000))
            - set(merged)
        )

        extra = sorted(
            set(merged)
            - set(range(50000))
        )

        raise RuntimeError(
            f"coverage failure "
            f"missing={missing[:20]} "
            f"extra={extra[:20]}"
        )

    rows = []

    hitmax = []

    empty = []

    source_mismatch = []

    for idx in range(50000):

        x = merged[idx]

        if (
            x["source"]
            !=
            source_map[idx]["source"]
        ):
            source_mismatch.append(
                idx
            )

        target = str(
            x.get(
                "target_translation",
                "",
            )
        ).strip()

        if not target:
            empty.append(idx)

        if int(
            x.get(
                "new_tokens",
                0,
            )
        ) >= int(
            x.get(
                "max_new_tokens",
                512,
            )
        ):
            hitmax.append(idx)

        rows.append(x)

    if source_mismatch:
        raise RuntimeError(
            f"source mismatch "
            f"{source_mismatch[:20]}"
        )

    if empty:
        raise RuntimeError(
            f"empty targets "
            f"{empty[:20]}"
        )

    # With max_new_tokens=512 this
    # should be extremely rare. Stop
    # before training if truncation exists.
    if hitmax:
        raise RuntimeError(
            f"HIT_MAX_NEW_TOKENS "
            f"count={len(hitmax)} "
            f"first={hitmax[:30]}"
        )

    out_dir = Path(
        args.output_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    full = (
        out_dir
        / "seqkd_newscrawl50k_qwen3_8b_v1.jsonl"
    )

    write_jsonl(
        full,
        rows,
    )

    sizes = [
        6565,
        10000,
        20000,
        50000,
    ]

    slices = {}

    for n in sizes:

        p = (
            out_dir
            / f"seqkd_newscrawl{n}_qwen3_8b_v1.jsonl"
        )

        write_jsonl(
            p,
            rows[:n],
        )

        slices[str(n)] = {
            "path":
                str(p),

            "rows":
                n,

            "sha256":
                sha256(p),
        }

    audit = {
        "protocol":
            "RQ0_B_SEQKD_SCALING_V1",

        "source_rows":
            len(sources),

        "teacher_rows":
            len(rows),

        "duplicates_ignored":
            duplicates,

        "hitmax":
            len(hitmax),

        "empty":
            len(empty),

        "source_mismatch":
            len(source_mismatch),

        "teacher_full_sha256":
            sha256(full),

        "slices":
            slices,

        "nested_prefix":
            True,

        "teacher":
            "Qwen3-8B",

        "thinking":
            False,

        "sampling":
            "greedy",
    }

    audit_path = (
        out_dir
        / "seqkd_newscrawl50k_audit_v1.json"
    )

    audit_path.write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "RQ0_B_SEQKD_50K_MERGE_PASS"
    )

    print(
        "RQ0_B_SEQKD_NESTED_SLICE_PASS"
    )


if __name__ == "__main__":
    main()
