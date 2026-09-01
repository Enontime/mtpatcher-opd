#!/usr/bin/env python3

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def read_jsonl(path):
    ans = []

    with open(
        path,
        encoding="utf-8-sig",
    ) as f:
        for line in f:
            if line.strip():
                ans.append(
                    json.loads(line)
                )

    return ans


def write_jsonl(path, rows):
    path = Path(path)

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for x in rows:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )


def sha(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for b in iter(
            lambda:
                f.read(1024 * 1024),
            b"",
        ):
            h.update(b)

    return h.hexdigest()


def stats(rows):
    vals = sorted(
        float(
            x["student_mean_nll"]
        )
        for x in rows
    )

    def p(q):
        return vals[
            int(
                q * (len(vals) - 1)
            )
        ]

    return {
        "rows":
            len(vals),

        "mean":
            statistics.mean(vals),

        "median":
            statistics.median(vals),

        "p10":
            p(0.10),

        "p50":
            p(0.50),

        "p90":
            p(0.90),

        "p95":
            p(0.95),

        "p99":
            p(0.99),
    }


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--teacher-data",
        required=True,
    )

    ap.add_argument(
        "--scores",
        required=True,
    )

    ap.add_argument(
        "--out-dir",
        required=True,
    )

    args = ap.parse_args()

    data = read_jsonl(
        args.teacher_data
    )

    scores = read_jsonl(
        args.scores
    )

    if len(data) != 50000:
        raise RuntimeError(
            f"teacher rows={len(data)}"
        )

    if len(scores) != 50000:
        raise RuntimeError(
            f"score rows={len(scores)}"
        )

    score_map = {
        int(x["row_position"]):
            float(
                x["student_mean_nll"]
            )
        for x in scores
    }

    ranked = sorted(
        range(50000),
        key=lambda i:
            score_map[i],
        reverse=True,
    )

    out = Path(
        args.out_dir
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {}

    for n in [
        6565,
        10000,
    ]:

        idxs = ranked[:n]

        subset = [
            data[i]
            for i in idxs
        ]

        path = (
            out
            / f"seqkd_newscrawl_topnll{n}_v1.jsonl"
        )

        write_jsonl(
            path,
            subset,
        )

        subset_scores = [
            scores[i]
            for i in idxs
        ]

        manifest[str(n)] = {
            "path":
                str(path),

            "sha256":
                sha(path),

            "mean_nll":
                statistics.mean(
                    x["student_mean_nll"]
                    for x in subset_scores
                ),

            "min_selected_nll":
                min(
                    x["student_mean_nll"]
                    for x in subset_scores
                ),

            "max_selected_nll":
                max(
                    x["student_mean_nll"]
                    for x in subset_scores
                ),
        }

    audit = {
        "protocol":
            "RQ0_C_TOP_STUDENT_NLL_V1",

        "selection":
            "descending mean response NLL of Qwen3-8B teacher target under Base Qwen3-0.6B",

        "source_pool":
            "WMT NewsCrawl 2023 zh, fixed 50k",

        "sets":
            manifest,
    }

    audit_path = (
        out
        / "topnll_audit_v1.json"
    )

    audit_path.write_text(
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
        "RQ0_C_TOPNLL_DATA_READY_PASS"
    )


if __name__ == "__main__":
    main()
