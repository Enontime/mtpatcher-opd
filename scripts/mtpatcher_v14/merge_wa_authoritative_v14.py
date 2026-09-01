import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def load(path):
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
                rows.append(
                    json.loads(line)
                )
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

        for x in arr:
            if not isinstance(
                x,
                dict,
            ):
                return False

            src = x.get("source")
            tgt = x.get("target")

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
        for row in load(f):
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

    ap.add_argument(
        "--base",
        required=True,
    )

    ap.add_argument(
        "--pairwise",
        required=True,
    )

    ap.add_argument(
        "--last31",
        required=True,
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    args = ap.parse_args()

    base = read_dir(args.base)
    pairwise = read_dir(
        args.pairwise
    )
    last31 = read_dir(
        args.last31
    )

    result = dict(base)

    source = {
        jid: "authoritative_repaired"
        for jid in result
    }

    for jid, row in pairwise.items():
        if jid not in result:
            result[jid] = row
            source[jid] = (
                "pairwise_final"
            )

    for jid, row in last31.items():
        if jid not in result:
            result[jid] = row
            source[jid] = (
                "last31_twostage"
            )

    missing = sorted(
        set(range(3732))
        - set(result)
    )

    counts = Counter(
        source.values()
    )

    print(
        "AUTHORITATIVE_COUNTS =",
        dict(counts),
    )

    print(
        "FINAL_VALID_ANALOG_ROWS =",
        len(result),
    )

    print(
        "FINAL_MISSING_ANALOG_IDS =",
        missing,
    )

    if missing:
        raise RuntimeError(
            "Still missing WA analog jobs"
        )

    if len(result) != 3732:
        raise RuntimeError(
            f"Expected 3732, got "
            f"{len(result)}"
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
                result[jid]
            )

            row[
                "wa_authoritative_source"
            ] = source[jid]

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
        "WA_AUTHORITATIVE_3732_MERGE_PASS"
    )


if __name__ == "__main__":
    main()
