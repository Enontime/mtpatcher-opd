import argparse
import hashlib
import json
import random
from pathlib import Path


PROMPT_PREFIX = (
    "Translate the following text into English "
    "without additional explanations:\n\n"
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def pair_key(source, target):
    return (
        str(source).strip(),
        str(target).strip()
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pe", required=True)
    ap.add_argument("--pds", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--seed", type=int, default=20260825)
    args = ap.parse_args()

    pe_rows = []

    with open(
        args.pe,
        encoding="utf-8"
    ) as f:
        for line in f:
            if line.strip():
                pe_rows.append(
                    json.loads(line)
                )

    if len(pe_rows) != 3732:
        raise RuntimeError(
            f"Expected PE3732, got "
            f"{len(pe_rows)}"
        )

    pds_rows = []

    with open(
        args.pds,
        encoding="utf-8"
    ) as f:
        for line in f:
            if line.strip():
                pds_rows.append(
                    json.loads(line)
                )

    combined = []
    seen = set()

    pe_kept = 0
    pds_kept = 0
    duplicates = 0

    for row in pe_rows:
        source = str(
            row["source"]
        ).strip()

        target = str(
            row["target_translation"]
        ).strip()

        key = pair_key(
            source,
            target
        )

        if key in seen:
            duplicates += 1
            continue

        seen.add(key)

        new_row = dict(row)
        new_row[
            "rq3_data_component"
        ] = "PE"

        combined.append(new_row)
        pe_kept += 1

    for row in pds_rows:
        source = str(
            row["source"]
        ).strip()

        target = str(
            row["target_translation"]
        ).strip()

        key = pair_key(
            source,
            target
        )

        if key in seen:
            duplicates += 1
            continue

        seen.add(key)

        message = {
            "role": "user",
            "content":
                PROMPT_PREFIX
                + source
                + "\n\n",
        }

        new_row = {
            "index":
                f"pds_v11_{row['job_id']}",
            "source":
                source,
            "messages":
                [message],
            "target_translation":
                target,
            "student_translation":
                "",
            "feedback_errors":
                [
                    {
                        "source_span":
                            row["source_span"],
                        "translation_span":
                            "",
                        "error_type":
                            row.get(
                                "error_type",
                                ""
                            ),
                        "explanation":
                            "PDS synthesized context",
                        "correction":
                            row["correction"],
                    }
                ],
            "construction_method":
                "MT_PATCHER_PDS_QWEN3_8B_V11",
            "rq3_data_component":
                "PDS",
            "parent_index":
                row["parent_index"],
            "parent_row_pos":
                row["parent_row_pos"],
            "error_index":
                row["error_index"],
            "pds_slot":
                row["pds_slot"],
        }

        combined.append(new_row)
        pds_kept += 1

    rng = random.Random(args.seed)
    rng.shuffle(combined)

    Path(args.output).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        args.output,
        "w",
        encoding="utf-8"
    ) as f:
        for row in combined:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                ) + "\n"
            )

    audit = {
        "pe_input_rows":
            len(pe_rows),
        "pds_input_rows":
            len(pds_rows),
        "pe_kept":
            pe_kept,
        "pds_kept":
            pds_kept,
        "combined_rows":
            len(combined),
        "duplicates_removed":
            duplicates,
        "shuffle_seed":
            args.seed,
        "pe_sha256":
            sha256(args.pe),
        "pds_sha256":
            sha256(args.pds),
        "combined_sha256":
            sha256(args.output),
        "method":
            "MT_PATCHER_PE_PLUS_PDS_QWEN3_V11",
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

    print("PE_PLUS_PDS_BUILD_PASS")


if __name__ == "__main__":
    main()
