import argparse
import hashlib
import json
import unicodedata
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)

    return h.hexdigest()


def text(x):
    return x.strip() if isinstance(x, str) else ""


def canonical(x):
    x = unicodedata.normalize(
        "NFKC",
        text(x),
    ).casefold()

    return "".join(
        c for c in x
        if unicodedata.category(c)[0] in {"L", "N"}
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--pe", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--audit", required=True)

    args = ap.parse_args()

    pe_path = Path(args.pe)
    out_path = Path(args.output)
    audit_path = Path(args.audit)

    rows = []

    with pe_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    if len(rows) != 3732:
        raise RuntimeError(
            f"Expected frozen PE3732, got {len(rows)}"
        )

    jobs = []

    parent_span_exact = 0
    parent_span_nonexact = 0
    missing_anchor = 0

    examples = []

    for row_pos, row in enumerate(rows):
        errors = row.get("feedback_errors")

        if not isinstance(errors, list) or not errors:
            raise RuntimeError(
                f"Invalid feedback_errors row_pos={row_pos}"
            )

        err = errors[0]

        if not isinstance(err, dict):
            raise RuntimeError(
                f"Invalid first error row_pos={row_pos}"
            )

        source = text(row.get("source"))
        source_span = text(err.get("source_span"))
        correction = text(err.get("correction"))

        if not source_span:
            missing_anchor += 1
            raise RuntimeError(
                f"Missing first-error source_span row_pos={row_pos}"
            )

        exact = (
            canonical(source_span)
            in canonical(source)
        )

        if exact:
            parent_span_exact += 1
        else:
            parent_span_nonexact += 1

            if len(examples) < 30:
                examples.append(
                    {
                        "row_pos":
                            row_pos,

                        "index":
                            row.get("index"),

                        "source":
                            source,

                        "source_span":
                            source_span,

                        "correction":
                            correction,
                    }
                )

        jobs.append(
            {
                "analog_job_id":
                    row_pos,

                "parent_row_pos":
                    row_pos,

                "parent_index":
                    row.get("index"),

                "source":
                    source,

                "student_translation":
                    text(
                        row.get(
                            "student_translation"
                        )
                    ),

                "source_span":
                    source_span,

                "correction":
                    correction,

                "error_type":
                    text(
                        err.get("error_type")
                    ),

                "explanation":
                    text(
                        err.get("explanation")
                    ),

                "parent_span_exact":
                    exact,

                "construction_method":
                    "MT_PATCHER_WA_ANCHOR_V14",
            }
        )

    if len(jobs) != 3732:
        raise RuntimeError(
            f"Expected 3732 WA jobs, got {len(jobs)}"
        )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with out_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in jobs:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    audit = {
        "pe_rows":
            len(rows),

        "wa_anchor_jobs":
            len(jobs),

        "anchor":
            "feedback_errors[0]",

        "parent_span_exact":
            parent_span_exact,

        "parent_span_nonexact":
            parent_span_nonexact,

        "missing_anchor":
            missing_anchor,

        "mismatch_examples":
            examples,

        "pe_sha256":
            sha256(pe_path),

        "jobs_sha256":
            sha256(out_path),

        "protocol":
            "MT_PATCHER_WA_V14",
    }

    audit_path.write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("PE_ROWS =", len(rows))
    print("WA_ANCHOR_JOBS =", len(jobs))
    print(
        "PARENT_SPAN_EXACT =",
        parent_span_exact,
    )
    print(
        "PARENT_SPAN_NONEXACT =",
        parent_span_nonexact,
    )
    print(
        "JOBS_SHA256 =",
        sha256(out_path),
    )

    print("WA_ANCHOR_JOB_BUILD_PASS")


if __name__ == "__main__":
    main()
