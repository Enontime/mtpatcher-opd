import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def load_jsonl(path):
    rows = []

    path = Path(path)

    if not path.exists():
        return rows

    with path.open(
        encoding="utf-8",
        errors="replace",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))
            except Exception:
                pass

    return rows


def valid(row):
    if not row.get("parse_ok"):
        return False

    a = row.get("analogs")

    if not isinstance(a, dict):
        return False

    seen = set()

    anchor = "".join(
        str(
            row.get(
                "source_span",
                "",
            )
        ).split()
    ).casefold()

    for aspect in (
        "category",
        "semantics",
    ):
        arr = a.get(aspect)

        if (
            not isinstance(arr, list)
            or len(arr) != 2
        ):
            return False

        for item in arr:
            if not isinstance(
                item,
                dict,
            ):
                return False

            src = item.get("source")
            tgt = item.get("target")

            if (
                not isinstance(src, str)
                or not src.strip()
                or not isinstance(tgt, str)
                or not tgt.strip()
            ):
                return False

            n = "".join(
                src.split()
            ).casefold()

            if n == anchor:
                return False

            if n in seen:
                return False

            seen.add(n)

    return len(seen) == 4


def read_dir(path):
    result = {}

    p = Path(path)

    if not p.exists():
        return result

    for f in sorted(
        p.glob("device_*.jsonl")
    ):
        for row in load_jsonl(f):
            if valid(row):
                result[
                    int(
                        row[
                            "analog_job_id"
                        ]
                    )
                ] = row

    return result


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base", required=True)
    ap.add_argument("--pairwise", required=True)
    ap.add_argument("--last31", required=True)
    ap.add_argument("--last8", required=True)
    ap.add_argument("--output", required=True)

    args = ap.parse_args()

    sources = [
        (
            "repaired_base",
            read_dir(args.base),
        ),
        (
            "pairwise",
            read_dir(args.pairwise),
        ),
        (
            "last31",
            read_dir(args.last31),
        ),
        (
            "last8_pool",
            read_dir(args.last8),
        ),
    ]

    final = {}
    origin = {}

    for name, rows in sources:
        for jid, row in rows.items():
            if jid not in final:
                final[jid] = row
                origin[jid] = name

    missing = sorted(
        set(range(3732))
        - set(final)
    )

    counts = Counter(
        origin.values()
    )

    print(
        "FINAL_SOURCE_COUNTS =",
        dict(counts),
    )

    print(
        "FINAL_VALID_ANALOG_ROWS =",
        len(final),
    )

    print(
        "FINAL_MISSING_IDS =",
        missing,
    )

    if missing:
        raise RuntimeError(
            f"WA still missing IDs: "
            f"{missing}"
        )

    if len(final) != 3732:
        raise RuntimeError(
            f"Expected 3732, "
            f"got {len(final)}"
        )

    out = Path(args.output)

    tmp = out.with_name(
        out.name + ".tmp"
    )

    if tmp.exists():
        shutil.rmtree(tmp)

    tmp.mkdir(
        parents=True,
        exist_ok=True,
    )

    handles = {
        i: (
            tmp
            / f"device_{i}.jsonl"
        ).open(
            "w",
            encoding="utf-8",
        )
        for i in range(16)
    }

    try:
        for jid in range(3732):
            row = dict(
                final[jid]
            )

            row[
                "wa_final_origin"
            ] = origin[jid]

            handles[
                jid % 16
            ].write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )
    finally:
        for h in handles.values():
            h.close()

    if out.exists():
        shutil.rmtree(out)

    tmp.rename(out)

    print(
        "WA_FINAL_3732_AUTHORITATIVE_PASS"
    )


if __name__ == "__main__":
    main()
