import argparse
import json
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
            if not isinstance(item, dict):
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
    out = {}

    path = Path(path)

    if not path.exists():
        return out

    for f in sorted(
        path.glob("device_*.jsonl")
    ):
        for row in load_jsonl(f):
            if valid(row):
                out[
                    int(row["analog_job_id"])
                ] = row

    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--anchors", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--pairwise", required=True)
    ap.add_argument("--last31", required=True)
    ap.add_argument("--output", required=True)

    args = ap.parse_args()

    anchors = {
        int(x["analog_job_id"]): x
        for x in load_jsonl(args.anchors)
    }

    base = read_dir(args.base)
    pairwise = read_dir(args.pairwise)
    last31 = read_dir(args.last31)

    resolved = dict(base)

    counts = {
        "base": len(resolved),
        "pairwise_added": 0,
        "last31_added": 0,
    }

    for jid, row in pairwise.items():
        if jid not in resolved:
            resolved[jid] = row
            counts["pairwise_added"] += 1

    for jid, row in last31.items():
        if jid not in resolved:
            resolved[jid] = row
            counts["last31_added"] += 1

    missing = sorted(
        set(anchors)
        - set(resolved)
    )

    print("COUNTS =", counts)
    print(
        "RESOLVED_BEFORE_LAST8 =",
        len(resolved),
    )
    print(
        "MISSING8_COUNT =",
        len(missing),
    )
    print(
        "MISSING8_IDS =",
        missing,
    )

    expected = [
        318,
        360,
        362,
        796,
        1328,
        1839,
        3151,
        3261,
    ]

    if len(resolved) != 3724:
        raise RuntimeError(
            f"Expected 3724 resolved, "
            f"got {len(resolved)}"
        )

    if missing != expected:
        raise RuntimeError(
            f"Unexpected missing set: "
            f"{missing}"
        )

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as f:
        for jid in missing:
            f.write(
                json.dumps(
                    anchors[jid],
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("WA_EXACT_MISSING8_AUDIT_PASS")


if __name__ == "__main__":
    main()
