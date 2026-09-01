import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path):
    rows = []

    with path.open(
        encoding="utf-8",
        errors="replace",
    ) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)

    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--merged", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--context-jobs", required=True)

    args = ap.parse_args()

    shard_dir = Path(args.shard_dir)

    rows = []

    seen_job = set()

    for device in range(16):
        p = shard_dir / f"device_{device}.jsonl"

        if not p.exists():
            raise RuntimeError(
                f"Missing analog shard {p}"
            )

        for row in load_jsonl(p):
            jid = int(row["analog_job_id"])

            if jid in seen_job:
                raise RuntimeError(
                    f"Duplicate analog job id {jid}"
                )

            seen_job.add(jid)
            rows.append(row)

    rows.sort(
        key=lambda x: int(
            x["analog_job_id"]
        )
    )

    if len(rows) != 3732:
        raise RuntimeError(
            f"Expected 3732 analog results, "
            f"got {len(rows)}"
        )

    parse_ok = sum(
        bool(x.get("parse_ok"))
        for x in rows
    )

    parse_fail = len(rows) - parse_ok

    merged_path = Path(args.merged)

    with merged_path.open(
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

    if parse_fail:
        print(
            "WA_ANALOG_PARSE_FAILURE "
            f"count={parse_fail}"
        )

        raise RuntimeError(
            "WA analog generation must produce "
            "exactly four valid pairs for all "
            "3732 PE examples before the "
            "paper-budget context stage."
        )

    context_jobs = []

    category_count = 0
    semantics_count = 0

    context_job_id = 0

    for row in rows:
        analogs = row["analogs"]

        for aspect in (
            "category",
            "semantics",
        ):
            for rank, pair in enumerate(
                analogs[aspect]
            ):
                context_jobs.append(
                    {
                        "wa_context_job_id":
                            context_job_id,

                        "analog_job_id":
                            row["analog_job_id"],

                        "parent_row_pos":
                            row["parent_row_pos"],

                        "parent_index":
                            row["parent_index"],

                        "aspect":
                            aspect,

                        "analog_rank":
                            rank,

                        "original_source":
                            row["source"],

                        "original_error_span":
                            row["source_span"],

                        "analog_source":
                            pair["source"],

                        "analog_target":
                            pair["target"],

                        "construction_method":
                            "MT_PATCHER_WA_CONTEXT_JOB_V14",
                    }
                )

                context_job_id += 1

                if aspect == "category":
                    category_count += 1
                else:
                    semantics_count += 1

    if len(context_jobs) != 14928:
        raise RuntimeError(
            f"Expected 14928 WA context jobs, "
            f"got {len(context_jobs)}"
        )

    if category_count != 7464:
        raise RuntimeError(
            f"Category count mismatch "
            f"{category_count}"
        )

    if semantics_count != 7464:
        raise RuntimeError(
            f"Semantics count mismatch "
            f"{semantics_count}"
        )

    context_path = Path(
        args.context_jobs
    )

    with context_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in context_jobs:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    audit = {
        "analog_rows":
            len(rows),

        "parse_ok":
            parse_ok,

        "parse_fail":
            parse_fail,

        "category_pairs":
            category_count,

        "semantics_pairs":
            semantics_count,

        "context_jobs":
            len(context_jobs),

        "contexts_per_pe":
            4,

        "analog_merged_sha256":
            sha256(merged_path),

        "context_jobs_sha256":
            sha256(context_path),

        "protocol":
            "MT_PATCHER_WA_V14_PAPER_BUDGET",
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

    print("WA_ANALOG_MERGE_PASS")
    print("WA_CONTEXT_14928_JOB_BUILD_PASS")


if __name__ == "__main__":
    main()
