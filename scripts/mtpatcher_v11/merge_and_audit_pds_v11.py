import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def norm(x):
    x = unicodedata.normalize(
        "NFKC",
        str(x)
    )
    x = re.sub(r"\s+", " ", x)
    return x.strip().casefold()


def norm_no_space(x):
    return re.sub(
        r"\s+",
        "",
        unicodedata.normalize(
            "NFKC",
            str(x)
        )
    ).casefold()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--world-size", type=int, default=16)
    args = ap.parse_args()

    jobs = {}

    with open(
        args.jobs,
        encoding="utf-8"
    ) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                jobs[int(row["job_id"])] = row

    results = {}

    shard_paths = []

    for device_id in range(args.world_size):
        path = (
            Path(args.shard_dir)
            / f"device_{device_id}.jsonl"
        )

        shard_paths.append(path)

        if not path.exists():
            raise RuntimeError(
                f"Missing PDS shard: {path}"
            )

        with path.open(
            encoding="utf-8"
        ) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                results[int(row["job_id"])] = row

    missing = sorted(
        set(jobs) - set(results)
    )

    if missing:
        raise RuntimeError(
            f"Missing generated jobs: "
            f"{len(missing)}, first={missing[:20]}"
        )

    reject_counts = Counter()
    coverage = defaultdict(int)

    valid = []
    seen_pairs = set()

    for jid in sorted(jobs):
        row = results[jid]

        src = str(
            row.get(
                "synthesized_source",
                ""
            )
        ).strip()

        tgt = str(
            row.get(
                "synthesized_target",
                ""
            )
        ).strip()

        original = str(
            row.get(
                "source",
                ""
            )
        ).strip()

        source_span = str(
            row.get(
                "source_span",
                ""
            )
        ).strip()

        correction = str(
            row.get(
                "correction",
                ""
            )
        ).strip()

        reason = None

        if not row.get("parse_ok"):
            reason = "parse_fail"

        elif not src or not tgt:
            reason = "empty_pair"

        elif (
            norm_no_space(source_span)
            not in norm_no_space(src)
        ):
            reason = "source_span_missing"

        elif norm(correction) not in norm(tgt):
            reason = "correction_missing"

        elif norm_no_space(src) == norm_no_space(
            original
        ):
            reason = "source_not_extended"

        elif norm(src) == norm(tgt):
            reason = "src_tgt_identical"

        pair_key = (
            norm(src),
            norm(tgt)
        )

        if reason is None and pair_key in seen_pairs:
            reason = "duplicate_pair"

        if reason is not None:
            reject_counts[reason] += 1
            continue

        seen_pairs.add(pair_key)

        parent_key = (
            row.get("parent_index"),
            row.get("error_index"),
        )

        coverage[parent_key] += 1

        valid.append(
            {
                "job_id": jid,
                "parent_index":
                    row.get("parent_index"),
                "parent_row_pos":
                    row.get("parent_row_pos"),
                "error_index":
                    row.get("error_index"),
                "pds_slot":
                    row.get("pds_slot"),
                "source":
                    src,
                "target_translation":
                    tgt,
                "source_span":
                    source_span,
                "correction":
                    correction,
                "error_type":
                    row.get("error_type"),
                "original_source":
                    original,
                "construction_method":
                    "MT_PATCHER_PDS_QWEN3_8B_V11",
            }
        )

    coverage_hist = Counter(
        coverage.values()
    )

    all_parent_keys = {
        (
            row.get("parent_index"),
            row.get("error_index")
        )
        for row in jobs.values()
    }

    zero_coverage = (
        len(all_parent_keys)
        - len(coverage)
    )

    coverage_hist[0] = zero_coverage

    Path(args.output).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        args.output,
        "w",
        encoding="utf-8"
    ) as f:
        for row in valid:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                ) + "\n"
            )

    valid_ratio = (
        len(valid) / len(jobs)
        if jobs else 0.0
    )

    audit = {
        "expected_jobs": len(jobs),
        "generated_jobs": len(results),
        "valid_pairs": len(valid),
        "valid_ratio": valid_ratio,
        "reject_counts":
            dict(reject_counts),
        "error_occurrences":
            len(all_parent_keys),
        "coverage_histogram":
            {
                str(k): coverage_hist[k]
                for k in sorted(
                    coverage_hist
                )
            },
        "jobs_sha256":
            sha256(args.jobs),
        "output_sha256":
            sha256(args.output),
        "world_size":
            args.world_size,
        "method":
            "MT_PATCHER_PDS_QWEN3_8B_V11",
    }

    with open(
        args.audit,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            audit,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False
        )
    )

    if valid_ratio < 0.50:
        raise RuntimeError(
            "PDS valid ratio below 0.50; "
            "do not train on this dataset"
        )

    print("PDS_MERGE_AUDIT_PASS")


if __name__ == "__main__":
    main()
