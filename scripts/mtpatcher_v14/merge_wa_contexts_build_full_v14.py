import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
import random


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


def text(x):
    return x.strip() if isinstance(x, str) else ""


def canonical(x):
    x = unicodedata.normalize(
        "NFKC",
        text(x),
    ).casefold()

    return "".join(
        ch
        for ch in x
        if unicodedata.category(ch)[0]
        in {"L", "N"}
    )


def contains_surface(needle, haystack):
    n = canonical(needle)
    h = canonical(haystack)

    return bool(n and h and n in h)


def pair_key(src, tgt):
    return (
        text(src),
        text(tgt),
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--shard-dir",
        required=True,
    )

    ap.add_argument(
        "--existing-pe-pds",
        required=True,
    )

    ap.add_argument(
        "--wa-output",
        required=True,
    )

    ap.add_argument(
        "--wa-audit",
        required=True,
    )

    ap.add_argument(
        "--combined",
        required=True,
    )

    ap.add_argument(
        "--combined-audit",
        required=True,
    )

    args = ap.parse_args()

    shard_dir = Path(
        args.shard_dir
    )

    generated = []

    seen_ids = set()

    for device in range(16):
        p = shard_dir / (
            f"device_{device}.jsonl"
        )

        if not p.exists():
            raise RuntimeError(
                f"Missing WA context shard {p}"
            )

        for row in load_jsonl(p):
            jid = int(
                row[
                    "wa_context_job_id"
                ]
            )

            if jid in seen_ids:
                raise RuntimeError(
                    f"Duplicate context job {jid}"
                )

            seen_ids.add(jid)
            generated.append(row)

    generated.sort(
        key=lambda x: int(
            x["wa_context_job_id"]
        )
    )

    if len(generated) != 14928:
        raise RuntimeError(
            f"Expected 14928 generated "
            f"contexts, got {len(generated)}"
        )

    parse_fail = 0
    duplicate_wa = 0

    quality = Counter()

    seen_pairs = set()

    valid = []

    for row in generated:
        src = text(
            row.get(
                "synthesized_source"
            )
        )

        tgt = text(
            row.get(
                "synthesized_target"
            )
        )

        if (
            not row.get("parse_ok")
            or not src
            or not tgt
        ):
            parse_fail += 1
            continue

        key = pair_key(src, tgt)

        if key in seen_pairs:
            duplicate_wa += 1
            continue

        seen_pairs.add(key)

        p_ok = contains_surface(
            row["analog_source"],
            src,
        )

        q_ok = contains_surface(
            row["analog_target"],
            tgt,
        )

        extended = (
            canonical(src)
            != canonical(
                row["original_source"]
            )
        )

        literal_placeholder = bool(
            re.search(
                r"(^|[\s:：])P"
                r"($|[\s,，。.;；:：])",
                src,
            )
            or re.search(
                r"(^|[\s:：])Q"
                r"($|[\s,，。.;；:：])",
                tgt,
            )
        )

        quality[
            "generated_contains_analog_source"
            if p_ok
            else
            "generated_missing_analog_source"
        ] += 1

        quality[
            "generated_contains_analog_target"
            if q_ok
            else
            "generated_missing_analog_target"
        ] += 1

        quality[
            "source_extended"
            if extended
            else
            "source_not_extended"
        ] += 1

        if literal_placeholder:
            quality[
                "literal_placeholder_suspect"
            ] += 1

        valid.append(
            {
                "index":
                    f"wa_v14_"
                    f"{row['wa_context_job_id']}",

                "source":
                    src,

                "messages":
                    [
                        {
                            "role": "user",
                            "content":
                                "Translate the following text into English "
                                "without additional explanations:\n\n"
                                + src
                                + "\n\n",
                        }
                    ],

                "target_translation":
                    tgt,

                "student_translation":
                    "",

                "feedback_errors":
                    [
                        {
                            "source_span":
                                row[
                                    "analog_source"
                                ],

                            "translation_span":
                                "",

                            "error_type":
                                "WA_"
                                + row[
                                    "aspect"
                                ].upper(),

                            "explanation":
                                "Word-analogy knowledge extension",

                            "correction":
                                row[
                                    "analog_target"
                                ],
                        }
                    ],

                "construction_method":
                    "MT_PATCHER_WA_QWEN3_8B_V14",

                "rq3_data_component":
                    "WA",

                "parent_index":
                    row["parent_index"],

                "parent_row_pos":
                    row["parent_row_pos"],

                "aspect":
                    row["aspect"],

                "analog_rank":
                    row["analog_rank"],

                "quality_audit":
                    {
                        "contains_P":
                            p_ok,

                        "contains_Q":
                            q_ok,

                        "source_extended":
                            extended,

                        "literal_placeholder_suspect":
                            literal_placeholder,
                    },
            }
        )

    wa_output = Path(
        args.wa_output
    )

    with wa_output.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in valid:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    wa_audit = {
        "requested_contexts":
            14928,

        "generated_contexts":
            len(generated),

        "parse_fail":
            parse_fail,

        "duplicate_wa_pair":
            duplicate_wa,

        "wa_after_postprocess":
            len(valid),

        "keep_ratio":
            len(valid) / 14928,

        "quality_flags":
            dict(quality),

        "wa_sha256":
            sha256(wa_output),

        "protocol":
            "MT_PATCHER_WA_V14_PAPER_STYLE",
    }

    Path(
        args.wa_audit
    ).write_text(
        json.dumps(
            wa_audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    ###########################################################################
    # Preserve the frozen PE+PDS-v13 baseline EXACTLY.
    ###########################################################################

    existing_path = Path(
        args.existing_pe_pds
    )

    existing = load_jsonl(
        existing_path
    )

    if len(existing) != 18610:
        raise RuntimeError(
            f"Expected frozen PE+PDS=18610, "
            f"got {len(existing)}"
        )

    existing_pairs = {
        pair_key(
            x["source"],
            x["target_translation"],
        )
        for x in existing
    }

    combined = list(existing)

    wa_overlap_existing = 0
    wa_kept_final = 0

    for row in valid:
        key = pair_key(
            row["source"],
            row["target_translation"],
        )

        if key in existing_pairs:
            wa_overlap_existing += 1
            continue

        existing_pairs.add(key)

        combined.append(row)

        wa_kept_final += 1

    rng = random.Random(
        20260825
    )

    rng.shuffle(combined)

    combined_path = Path(
        args.combined
    )

    with combined_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in combined:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    comp_counts = Counter(
        x.get(
            "rq3_data_component"
        )
        for x in combined
    )

    combined_audit = {
        "frozen_pe_pds_input_rows":
            len(existing),

        "wa_postprocessed":
            len(valid),

        "wa_overlap_existing_removed":
            wa_overlap_existing,

        "wa_kept":
            wa_kept_final,

        "combined_rows":
            len(combined),

        "component_counts":
            dict(comp_counts),

        "existing_pe_pds_sha256":
            sha256(existing_path),

        "wa_sha256":
            sha256(wa_output),

        "combined_sha256":
            sha256(combined_path),

        "protocol":
            "RQ3_PE_PDS_WA_V14",
    }

    Path(
        args.combined_audit
    ).write_text(
        json.dumps(
            combined_audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            wa_audit,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()

    print(
        json.dumps(
            combined_audit,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("WA_CONTEXT_MERGE_PASS")
    print(
        "FROZEN_PE_PDS_V13_PRESERVED_PASS"
    )
    print(
        "RQ3_PE_PDS_WA_V14_DATA_READY"
    )


if __name__ == "__main__":
    main()
