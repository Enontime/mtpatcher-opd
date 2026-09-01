import argparse
import ast
import json
import re
import shutil
from collections import Counter
from pathlib import Path


def load_jsonl(path):
    rows = []

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


def normalize_source(x):
    if not isinstance(x, str):
        return ""

    return "".join(
        x.strip().split()
    ).casefold()


def extract_container(raw):
    raw = (
        raw.strip()
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
        start >= 0
        and end > start
    ):
        raw = raw[
            start:end + 1
        ]

    parsers = [
        lambda x: json.loads(x),
        lambda x: ast.literal_eval(x),
    ]

    for parser in parsers:
        try:
            obj = parser(raw)

            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    return None


def find_group(obj, names):
    lowered = {
        str(k).strip().casefold():
            v
        for k, v in obj.items()
    }

    for name in names:
        if name in lowered:
            return lowered[name]

    return None


def parse_item(x):
    if isinstance(x, dict):
        lower = {
            str(k).strip().casefold():
                v
            for k, v in x.items()
        }

        src = None
        tgt = None

        for key in (
            "source",
            "chinese",
            "zh",
            "word",
            "phrase",
        ):
            if key in lower:
                src = lower[key]
                break

        for key in (
            "target",
            "english",
            "en",
            "translation",
        ):
            if key in lower:
                tgt = lower[key]
                break

        if (
            isinstance(src, str)
            and isinstance(tgt, str)
        ):
            src = src.strip()
            tgt = tgt.strip()

            if src and tgt:
                return {
                    "source": src,
                    "target": tgt,
                }

    if (
        isinstance(x, (list, tuple))
        and len(x) == 2
        and isinstance(x[0], str)
        and isinstance(x[1], str)
    ):
        return {
            "source":
                x[0].strip(),

            "target":
                x[1].strip(),
        }

    return None


def clean_group(
    arr,
    anchor,
    globally_seen,
):
    if not isinstance(
        arr,
        (list, tuple),
    ):
        return None

    clean = []

    anchor_n = normalize_source(
        anchor
    )

    for item in arr:
        pair = parse_item(item)

        if pair is None:
            continue

        src_n = normalize_source(
            pair["source"]
        )

        if not src_n:
            continue

        if src_n == anchor_n:
            continue

        if src_n in globally_seen:
            continue

        globally_seen.add(src_n)
        clean.append(pair)

        if len(clean) == 2:
            break

    if len(clean) != 2:
        return None

    return clean


def strict_existing(row):
    if not row.get("parse_ok"):
        return None

    obj = row.get("analogs")

    if not isinstance(obj, dict):
        return None

    seen = set()

    category = clean_group(
        obj.get("category"),
        row["source_span"],
        seen,
    )

    semantics = clean_group(
        obj.get("semantics"),
        row["source_span"],
        seen,
    )

    if (
        category is None
        or semantics is None
    ):
        return None

    return {
        "category": category,
        "semantics": semantics,
    }


def salvage(row):
    obj = extract_container(
        row.get(
            "raw_analogy",
            "",
        )
    )

    if obj is None:
        return None

    category_raw = find_group(
        obj,
        (
            "category",
            "categories",
            "categorical",
        ),
    )

    semantics_raw = find_group(
        obj,
        (
            "semantics",
            "semantic",
            "semantic association",
            "semantic associations",
            "co-occurrence",
            "cooccurrence",
        ),
    )

    seen = set()

    category = clean_group(
        category_raw,
        row["source_span"],
        seen,
    )

    semantics = clean_group(
        semantics_raw,
        row["source_span"],
        seen,
    )

    if (
        category is None
        or semantics is None
    ):
        return None

    return {
        "category": category,
        "semantics": semantics,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--original-dir",
        required=True,
    )

    ap.add_argument(
        "--retry-dir",
    )

    ap.add_argument(
        "--anchor-jobs",
        required=True,
    )

    ap.add_argument(
        "--repaired-dir",
        required=True,
    )

    ap.add_argument(
        "--retry-jobs",
        required=True,
    )

    ap.add_argument(
        "--audit",
        required=True,
    )

    args = ap.parse_args()

    original_dir = Path(
        args.original_dir
    )

    retry_dir = (
        Path(args.retry_dir)
        if args.retry_dir
        else None
    )

    repaired_dir = Path(
        args.repaired_dir
    )

    anchors = {
        int(x["analog_job_id"]): x
        for x in load_jsonl(
            Path(args.anchor_jobs)
        )
    }

    if len(anchors) != 3732:
        raise RuntimeError(
            f"Expected 3732 anchors, "
            f"got {len(anchors)}"
        )

    original = {}

    malformed_json_lines = 0

    for device in range(16):
        p = (
            original_dir
            / f"device_{device}.jsonl"
        )

        if not p.exists():
            raise RuntimeError(
                f"Missing original shard {p}"
            )

        rows = load_jsonl(p)

        for row in rows:
            jid = int(
                row["analog_job_id"]
            )

            if jid in original:
                raise RuntimeError(
                    f"Duplicate original "
                    f"analog_job_id={jid}"
                )

            original[jid] = row

    if len(original) != 3732:
        raise RuntimeError(
            f"Expected 3732 original outputs, "
            f"got {len(original)}"
        )

    retry_success = {}

    if (
        retry_dir is not None
        and retry_dir.exists()
    ):
        for device in range(16):
            p = (
                retry_dir
                / f"device_{device}.jsonl"
            )

            for row in load_jsonl(p):
                if not row.get("parse_ok"):
                    continue

                analogs = strict_existing(
                    row
                )

                if analogs is None:
                    continue

                retry_success[
                    int(
                        row[
                            "analog_job_id"
                        ]
                    )
                ] = (
                    row,
                    analogs,
                )

    stats = Counter()
    parse_errors = Counter()

    repaired = {}
    unresolved = []

    for jid in range(3732):
        row = original[jid]

        analogs = strict_existing(
            row
        )

        origin = None

        if analogs is not None:
            stats[
                "original_strict_valid"
            ] += 1

            origin = "original"

        else:
            stats[
                "original_invalid"
            ] += 1

            parse_errors[
                str(
                    row.get(
                        "parse_error",
                        "UNKNOWN",
                    )
                ).split(
                    ":",
                    1,
                )[0]
            ] += 1

            analogs = salvage(row)

            if analogs is not None:
                stats[
                    "salvaged_without_model"
                ] += 1

                origin = "salvage"

        if (
            analogs is None
            and jid in retry_success
        ):
            retry_row, analogs = (
                retry_success[jid]
            )

            row = retry_row

            stats[
                "recovered_by_retry"
            ] += 1

            origin = "retry"

        if analogs is None:
            unresolved.append(
                anchors[jid]
            )
            continue

        out = dict(row)

        out["parse_ok"] = True
        out["parse_error"] = ""
        out["analogs"] = analogs

        out[
            "wa_recovery_origin"
        ] = origin

        repaired[jid] = out

    stats[
        "resolved_total"
    ] = len(repaired)

    stats[
        "unresolved_total"
    ] = len(unresolved)

    # Rebuild repaired shards atomically.
    tmp_dir = repaired_dir.with_name(
        repaired_dir.name + ".tmp"
    )

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    tmp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = {
        i: (
            tmp_dir
            / f"device_{i}.jsonl"
        ).open(
            "w",
            encoding="utf-8",
        )
        for i in range(16)
    }

    try:
        for jid in sorted(repaired):
            device = jid % 16

            files[device].write(
                json.dumps(
                    repaired[jid],
                    ensure_ascii=False,
                )
                + "\n"
            )
    finally:
        for f in files.values():
            f.close()

    if repaired_dir.exists():
        shutil.rmtree(
            repaired_dir
        )

    tmp_dir.rename(
        repaired_dir
    )

    retry_jobs_path = Path(
        args.retry_jobs
    )

    with retry_jobs_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in unresolved:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    audit = {
        "original_outputs":
            len(original),

        "stats":
            dict(stats),

        "parse_error_type_counts":
            dict(parse_errors),

        "unresolved_ids_first100":
            [
                int(x["analog_job_id"])
                for x in unresolved[:100]
            ],

        "protocol":
            "RQ3_WA_V14_FORMAT_RECOVERY",
    }

    Path(args.audit).write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "WA_ANALOG_SALVAGE_PASS"
    )

    print(
        "WA_ANALOG_RETRY_REQUIRED =",
        len(unresolved),
    )


if __name__ == "__main__":
    main()
