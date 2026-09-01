import argparse
import ast
import json
import re
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
                rows.append(
                    json.loads(line)
                )
            except Exception:
                pass

    return rows


def norm(x):
    return "".join(
        str(x).strip().split()
    ).casefold()


def validate(obj, anchor):
    if not isinstance(obj, dict):
        return None

    seen = set()
    clean = {}

    for aspect in (
        "category",
        "semantics",
    ):
        arr = obj.get(aspect)

        if (
            not isinstance(arr, list)
            or len(arr) != 2
        ):
            return None

        result = []

        for item in arr:
            if not isinstance(
                item,
                dict,
            ):
                return None

            src = item.get("source")
            tgt = item.get("target")

            if (
                not isinstance(src, str)
                or
                not isinstance(tgt, str)
            ):
                return None

            src = src.strip()
            tgt = tgt.strip()

            if not src or not tgt:
                return None

            n = norm(src)

            if n == norm(anchor):
                return None

            if n in seen:
                return None

            seen.add(n)

            result.append(
                {
                    "source": src,
                    "target": tgt,
                }
            )

        clean[aspect] = result

    if len(seen) != 4:
        return None

    return clean


def salvage_raw(row):
    raw = str(
        row.get(
            "raw_analogy",
            "",
        )
    ).strip()

    raw = (
        raw
        .replace("“", '"')
        .replace("”", '"')
    )

    raw = re.sub(
        r"^```(?:json)?\s*",
        "",
        raw,
        flags=re.I,
    )

    raw = re.sub(
        r"\s*```$",
        "",
        raw,
    )

    start = raw.find("{")
    end = raw.rfind("}")

    if (
        start < 0
        or end <= start
    ):
        return None

    chunk = raw[
        start:end + 1
    ]

    obj = None

    for parser in (
        json.loads,
        ast.literal_eval,
    ):
        try:
            obj = parser(chunk)
            break
        except Exception:
            pass

    if not isinstance(obj, dict):
        return None

    lowered = {
        str(k).strip().casefold():
            v
        for k, v in obj.items()
    }

    category = lowered.get(
        "category"
    )

    semantics = (
        lowered.get("semantics")
        or lowered.get("semantic")
    )

    candidate = {
        "category":
            category,

        "semantics":
            semantics,
    }

    return validate(
        candidate,
        row["source_span"],
    )


def load_map(directory):
    result = {}

    directory = Path(
        directory
    )

    if not directory.exists():
        return result

    for p in directory.glob(
        "device_*.jsonl"
    ):
        for row in load_jsonl(p):
            jid = int(
                row["analog_job_id"]
            )

            result[jid] = row

    return result


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--original-dir",
        required=True,
    )

    ap.add_argument(
        "--old-retry-dir",
        required=True,
    )

    ap.add_argument(
        "--final-retry-dir",
        required=True,
    )

    ap.add_argument(
        "--output-dir",
        required=True,
    )

    args = ap.parse_args()

    original = load_map(
        args.original_dir
    )

    old_retry = load_map(
        args.old_retry_dir
    )

    final_retry = load_map(
        args.final_retry_dir
    )

    if len(original) != 3732:
        raise RuntimeError(
            f"Original rows={len(original)}"
        )

    resolved = {}
    stats = Counter()

    unresolved = []

    for jid in range(3732):
        base = original[jid]

        analogs = None
        source = None

        if base.get("parse_ok"):
            analogs = validate(
                base.get("analogs"),
                base["source_span"],
            )

            if analogs is not None:
                source = "original"

        if analogs is None:
            analogs = salvage_raw(
                base
            )

            if analogs is not None:
                source = (
                    "original_salvage"
                )

        if (
            analogs is None
            and jid in old_retry
        ):
            candidate = (
                old_retry[jid]
            )

            if candidate.get(
                "parse_ok"
            ):
                analogs = validate(
                    candidate.get(
                        "analogs"
                    ),
                    base["source_span"],
                )

                if analogs is not None:
                    base = candidate
                    source = "old_retry"

        if (
            analogs is None
            and jid in final_retry
        ):
            candidate = (
                final_retry[jid]
            )

            if candidate.get(
                "parse_ok"
            ):
                analogs = validate(
                    candidate.get(
                        "analogs"
                    ),
                    original[jid][
                        "source_span"
                    ],
                )

                if analogs is not None:
                    base = candidate
                    source = (
                        "pairwise_final"
                    )

        if analogs is None:
            unresolved.append(
                jid
            )
            continue

        row = dict(base)

        row["analogs"] = analogs
        row["parse_ok"] = True
        row["parse_error"] = ""
        row[
            "wa_final_resolution"
        ] = source

        resolved[jid] = row

        stats[source] += 1

    print(
        "FINAL_RESOLUTION_COUNTS =",
        dict(stats),
    )

    print(
        "FINAL_RESOLVED =",
        len(resolved),
    )

    print(
        "FINAL_UNRESOLVED =",
        len(unresolved),
    )

    print(
        "UNRESOLVED_IDS =",
        unresolved,
    )

    if unresolved:
        raise RuntimeError(
            "WA final pairwise repair "
            "still has unresolved jobs"
        )

    if len(resolved) != 3732:
        raise RuntimeError(
            "WA resolution cardinality "
            "mismatch"
        )

    out_dir = Path(
        args.output_dir
    )

    tmp = out_dir.with_name(
        out_dir.name + ".finaltmp"
    )

    if tmp.exists():
        shutil.rmtree(tmp)

    tmp.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = {
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
            device = jid % 16

            files[
                device
            ].write(
                json.dumps(
                    resolved[jid],
                    ensure_ascii=False,
                )
                + "\n"
            )
    finally:
        for f in files.values():
            f.close()

    if out_dir.exists():
        shutil.rmtree(
            out_dir
        )

    tmp.rename(
        out_dir
    )

    print(
        "WA_ALL_3732_FINAL_REPAIR_PASS"
    )


if __name__ == "__main__":
    main()
