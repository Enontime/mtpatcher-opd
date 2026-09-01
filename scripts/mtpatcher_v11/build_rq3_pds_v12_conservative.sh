#!/usr/bin/env bash
set -euo pipefail

source /workspace/mtpatcher/project_env.sh

EXP="mtpatcher_v3_full6565_20260823"

SCRIPT_DIR="$ROOT/scripts/mtpatcher_v11"
EXP_DATA="$DATA_ROOT/$EXP"

JOBS="$EXP_DATA/rq3_pds_jobs_v11.jsonl"
PDS_DIR="$EXP_DATA/rq3_pds_v11"
PE="$EXP_DATA/pe_k1_clean3732.jsonl"

PDS_OUT="$EXP_DATA/rq3_pds_valid_v12_conservative.jsonl"
PDS_AUDIT="$EXP_DATA/rq3_pds_audit_v12_conservative.json"

COMBINED="$EXP_DATA/rq3_pe_plus_pds_v12_conservative.jsonl"
COMBINED_AUDIT="$EXP_DATA/rq3_pe_plus_pds_audit_v12_conservative.json"

PY="$SCRIPT_DIR/filter_pds_v12_conservative.py"

cat > "$PY" <<'PY'
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


EXP = "mtpatcher_v3_full6565_20260823"

DATA_ROOT = Path(os.environ["DATA_ROOT"])
EXP_DATA = DATA_ROOT / EXP

JOBS = EXP_DATA / "rq3_pds_jobs_v11.jsonl"
PDS_DIR = EXP_DATA / "rq3_pds_v11"
PE = EXP_DATA / "pe_k1_clean3732.jsonl"

PDS_OUT = EXP_DATA / "rq3_pds_valid_v12_conservative.jsonl"
PDS_AUDIT = EXP_DATA / "rq3_pds_audit_v12_conservative.json"

COMBINED = EXP_DATA / "rq3_pe_plus_pds_v12_conservative.jsonl"
COMBINED_AUDIT = EXP_DATA / "rq3_pe_plus_pds_audit_v12_conservative.json"


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


def canonical_chars(x):
    """
    Conservative surface normalization.

    Keep only Unicode letters/numbers, casefold English.
    This removes:
      whitespace
      straight/curly quotes
      Chinese/English punctuation
      hyphens
      percent-spacing differences

    It does NOT perform lexical or semantic paraphrase matching.
    """
    x = unicodedata.normalize("NFKC", text(x)).casefold()

    return "".join(
        ch
        for ch in x
        if unicodedata.category(ch)[0] in {"L", "N"}
    )


ELLIPSIS_RE = re.compile(r"(?:\.{2,}|…+)")


def surface_contains(needle, haystack):
    """
    Conservative containment.

    Normal case:
        punctuation/space-insensitive substring.

    Ellipsis source spans such as:
        送到...手中

    are interpreted as ordered literal fragments:
        送到 -> later -> 手中
    """
    needle = text(needle)
    haystack = text(haystack)

    if not needle or not haystack:
        return False

    pieces = [
        canonical_chars(p)
        for p in ELLIPSIS_RE.split(needle)
        if canonical_chars(p)
    ]

    h = canonical_chars(haystack)

    if not pieces or not h:
        return False

    if len(pieces) == 1:
        return pieces[0] in h

    pos = 0

    for piece in pieces:
        found = h.find(piece, pos)

        if found < 0:
            return False

        pos = found + len(piece)

    return True


def pair_key(src, tgt):
    return (
        canonical_chars(src),
        canonical_chars(tgt),
    )


def load_jsonl(path):
    rows = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


jobs = {
    int(r["job_id"]): r
    for r in load_jsonl(JOBS)
}

if len(jobs) != 26672:
    raise RuntimeError(
        f"Expected 26672 jobs, got {len(jobs)}"
    )


generated = {}

for device in range(16):
    shard = PDS_DIR / f"device_{device}.jsonl"

    if not shard.exists():
        raise RuntimeError(
            f"Missing shard: {shard}"
        )

    for row in load_jsonl(shard):
        jid = int(row["job_id"])

        if jid in generated:
            raise RuntimeError(
                f"Duplicate generated job_id: {jid}"
            )

        generated[jid] = row


if set(generated) != set(jobs):
    missing = sorted(set(jobs) - set(generated))
    extra = sorted(set(generated) - set(jobs))

    raise RuntimeError(
        f"Generation/job mismatch: "
        f"missing={len(missing)}, "
        f"extra={len(extra)}"
    )


flags = Counter()
exclusive_reject = Counter()

coverage = defaultdict(int)

valid = []
seen_pds = set()

examples = defaultdict(list)


for jid in sorted(jobs):
    job = jobs[jid]
    row = generated[jid]

    parent_source = text(job.get("source"))
    source_span = text(job.get("source_span"))
    correction = text(job.get("correction"))

    syn_src = text(row.get("synthesized_source"))
    syn_tgt = text(row.get("synthesized_target"))

    parse_ok = bool(row.get("parse_ok")) and syn_src and syn_tgt

    parent_span_ok = surface_contains(
        source_span,
        parent_source,
    )

    generated_span_ok = surface_contains(
        source_span,
        syn_src,
    )

    correction_ok = surface_contains(
        correction,
        syn_tgt,
    )

    extended_ok = (
        canonical_chars(syn_src)
        != canonical_chars(parent_source)
    )

    if parse_ok:
        flags["parse_ok"] += 1
    else:
        flags["parse_fail"] += 1

    if parent_span_ok:
        flags["parent_span_ok"] += 1
    else:
        flags["parent_span_invalid"] += 1

    if generated_span_ok:
        flags["generated_span_ok"] += 1
    else:
        flags["generated_span_missing"] += 1

    if correction_ok:
        flags["correction_ok"] += 1
    else:
        flags["correction_missing"] += 1

    if extended_ok:
        flags["extended_ok"] += 1
    else:
        flags["source_not_extended"] += 1

    # Keep diagnostics for the important failure classes.
    diag = {
        "job_id": jid,
        "parent_index": job.get("parent_index"),
        "error_index": job.get("error_index"),
        "source_span": source_span,
        "correction": correction,
        "parent_source": parent_source,
        "synthesized_source": syn_src,
        "synthesized_target": syn_tgt,
    }

    if (
        not parent_span_ok
        and len(examples["parent_span_invalid"]) < 30
    ):
        examples["parent_span_invalid"].append(diag)

    if (
        parent_span_ok
        and not generated_span_ok
        and len(examples["generated_span_missing"]) < 30
    ):
        examples["generated_span_missing"].append(diag)

    if (
        parent_span_ok
        and generated_span_ok
        and not correction_ok
        and len(examples["correction_missing"]) < 30
    ):
        examples["correction_missing"].append(diag)

    # Exclusive scientific rejection hierarchy.
    if not parse_ok:
        exclusive_reject["parse_fail"] += 1
        continue

    if not parent_span_ok:
        exclusive_reject["parent_span_invalid"] += 1
        continue

    if not generated_span_ok:
        exclusive_reject["generated_span_missing"] += 1
        continue

    if not correction_ok:
        exclusive_reject["correction_missing"] += 1
        continue

    if not extended_ok:
        exclusive_reject["source_not_extended"] += 1
        continue

    key = pair_key(syn_src, syn_tgt)

    if key in seen_pds:
        exclusive_reject["duplicate_pds_pair"] += 1
        continue

    seen_pds.add(key)

    parent_key = (
        job.get("parent_index"),
        job.get("error_index"),
    )

    coverage[parent_key] += 1

    valid.append(
        {
            "job_id": jid,
            "parent_index": job.get("parent_index"),
            "parent_row_pos": job.get("parent_row_pos"),
            "error_index": job.get("error_index"),
            "pds_slot": job.get("pds_slot"),

            "source": syn_src,
            "target_translation": syn_tgt,

            "source_span": source_span,
            "correction": correction,

            "error_type": job.get("error_type"),
            "original_source": parent_source,

            "construction_method":
                "MT_PATCHER_PDS_QWEN3_8B_V12_CONSERVATIVE",
        }
    )


all_error_pairs = {
    (
        job.get("parent_index"),
        job.get("error_index"),
    )
    for job in jobs.values()
}

coverage_hist = Counter(
    coverage.get(key, 0)
    for key in all_error_pairs
)


with PDS_OUT.open("w", encoding="utf-8") as f:
    for row in valid:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


pds_audit = {
    "protocol":
        "conservative_surface_normalization_v12",

    "generated_jobs":
        len(generated),

    "error_occurrences":
        len(all_error_pairs),

    "independent_flags":
        dict(flags),

    "exclusive_reject_counts":
        dict(exclusive_reject),

    "valid_pairs":
        len(valid),

    "valid_ratio":
        len(valid) / len(generated),

    "coverage_histogram":
        {
            str(k): coverage_hist[k]
            for k in sorted(coverage_hist)
        },

    "zero_coverage_error_pairs":
        coverage_hist[0],

    "zero_coverage_ratio":
        coverage_hist[0] / len(all_error_pairs),

    "mean_valid_contexts_per_error_pair":
        len(valid) / len(all_error_pairs),

    "examples":
        dict(examples),

    "jobs_sha256":
        sha256(JOBS),

    "output_sha256":
        sha256(PDS_OUT),

    "matching_policy": {
        "unicode_nfkc": True,
        "casefold": True,
        "ignore_punctuation": True,
        "ignore_whitespace": True,
        "ellipsis_as_ordered_fragments": True,
        "semantic_paraphrase_matching": False,
        "parent_span_must_be_valid": True,
    },
}


with PDS_AUDIT.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        pds_audit,
        f,
        indent=2,
        ensure_ascii=False,
    )


###############################################################################
# Build PE + PDS while preserving PE3732 EXACTLY.
###############################################################################

pe_rows = load_jsonl(PE)

if len(pe_rows) != 3732:
    raise RuntimeError(
        f"Expected PE3732, got {len(pe_rows)}"
    )


combined = []

# Preserve every frozen PE row, including any duplicate pair.
for row in pe_rows:
    x = dict(row)
    x["rq3_data_component"] = "PE"
    combined.append(x)


pe_pair_keys = {
    pair_key(
        row["source"],
        row["target_translation"],
    )
    for row in pe_rows
}


pds_kept = 0
pds_overlap_with_pe = 0
pds_duplicate_after_filter = 0

seen_new_pds = set()


PROMPT_PREFIX = (
    "Translate the following text into English "
    "without additional explanations:\n\n"
)


for row in valid:
    src = row["source"]
    tgt = row["target_translation"]

    key = pair_key(src, tgt)

    if key in pe_pair_keys:
        pds_overlap_with_pe += 1
        continue

    if key in seen_new_pds:
        pds_duplicate_after_filter += 1
        continue

    seen_new_pds.add(key)

    combined.append(
        {
            "index":
                f"pds_v12_{row['job_id']}",

            "source":
                src,

            "messages":
                [
                    {
                        "role": "user",
                        "content":
                            PROMPT_PREFIX
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
                            row["source_span"],

                        "translation_span":
                            "",

                        "error_type":
                            row.get(
                                "error_type",
                                "",
                            ),

                        "explanation":
                            "PDS synthesized context",

                        "correction":
                            row["correction"],
                    }
                ],

            "construction_method":
                "MT_PATCHER_PDS_QWEN3_8B_V12_CONSERVATIVE",

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
    )

    pds_kept += 1


# Deterministic shuffle while still preserving all PE rows.
import random

rng = random.Random(20260825)
rng.shuffle(combined)


with COMBINED.open(
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


combined_audit = {
    "pe_input_rows":
        len(pe_rows),

    "pe_kept":
        len(pe_rows),

    "pe_exactly_preserved":
        True,

    "pds_filtered_input":
        len(valid),

    "pds_kept":
        pds_kept,

    "pds_overlap_with_pe_removed":
        pds_overlap_with_pe,

    "pds_duplicate_after_filter":
        pds_duplicate_after_filter,

    "combined_rows":
        len(combined),

    "expected_combined_rows":
        len(pe_rows) + pds_kept,

    "shuffle_seed":
        20260825,

    "pe_sha256":
        sha256(PE),

    "pds_sha256":
        sha256(PDS_OUT),

    "combined_sha256":
        sha256(COMBINED),

    "method":
        "MT_PATCHER_PE_PLUS_PDS_QWEN3_V12_CONSERVATIVE",
}


with COMBINED_AUDIT.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        combined_audit,
        f,
        indent=2,
        ensure_ascii=False,
    )


print("======================================================================")
print("PDS V12 CONSERVATIVE AUDIT")
print("======================================================================")

print(
    json.dumps(
        pds_audit,
        indent=2,
        ensure_ascii=False,
    )
)

print()
print("======================================================================")
print("PE + PDS V12 DATASET AUDIT")
print("======================================================================")

print(
    json.dumps(
        combined_audit,
        indent=2,
        ensure_ascii=False,
    )
)

print()
print("PDS_V12_CONSERVATIVE_FILTER_PASS")
print("PE3732_EXACT_PRESERVATION_PASS")
print("RQ3_PE_PLUS_PDS_V12_DATA_READY_FOR_REVIEW")
PY

python -m py_compile "$PY"

python "$PY"

echo
echo "========== FINAL CARDINALITY =========="

wc -l \
  "$PE" \
  "$PDS_OUT" \
  "$COMBINED"

echo
echo "========== SHA256 =========="

sha256sum \
  "$PDS_OUT" \
  "$COMBINED"
