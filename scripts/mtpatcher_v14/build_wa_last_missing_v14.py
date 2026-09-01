import argparse
import json
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
            if not isinstance(x, dict):
                return False

            s = x.get("source")
            t = x.get("target")

            if (
                not isinstance(s, str)
                or not s.strip()
                or not isinstance(t, str)
                or not t.strip()
            ):
                return False

            n = "".join(
                s.split()
            ).casefold()

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
                    int(row["analog_job_id"])
                ] = row

    return result


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--anchors",
        required=True,
    )

    ap.add_argument(
        "--base",
        required=True,
    )

    ap.add_argument(
        "--pairwise",
        required=True,
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    args = ap.parse_args()

    anchors = {
        int(x["analog_job_id"]): x
        for x in load(args.anchors)
    }

    if len(anchors) != 3732:
        raise RuntimeError(
            f"anchors={len(anchors)}"
        )

    base = read_dir(args.base)
    pairwise = read_dir(
        args.pairwise
    )

    resolved = dict(base)

    pairwise_added = []

    for jid, row in pairwise.items():
        if jid not in resolved:
            resolved[jid] = row
            pairwise_added.append(jid)

    missing = sorted(
        set(anchors)
        - set(resolved)
    )

    print(
        "BASE_VALID =",
        len(base),
    )

    print(
        "PAIRWISE_VALID =",
        len(pairwise),
    )

    print(
        "PAIRWISE_NEW_IDS =",
        pairwise_added,
    )

    print(
        "RESOLVED_BEFORE_LAST_RETRY =",
        len(resolved),
    )

    print(
        "REAL_MISSING_COUNT =",
        len(missing),
    )

    print(
        "REAL_MISSING_IDS =",
        missing,
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

    if len(resolved) != 3701:
        raise RuntimeError(
            "Expected 3699 repaired "
            "+ 2 pairwise = 3701"
        )

    if len(missing) != 31:
        raise RuntimeError(
            f"Expected 31 real missing, "
            f"got {len(missing)}"
        )

    print(
        "WA_REAL_31_MISSING_AUDIT_PASS"
    )


if __name__ == "__main__":
    main()
