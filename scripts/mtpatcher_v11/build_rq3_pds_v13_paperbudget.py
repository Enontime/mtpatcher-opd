import hashlib
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
import random


EXP = "mtpatcher_v3_full6565_20260823"

ROOT = Path(os.environ["DATA_ROOT"]) / EXP

PE = ROOT / "pe_k1_clean3732.jsonl"
JOBS = ROOT / "rq3_pds_jobs_v11.jsonl"
SHARDS = ROOT / "rq3_pds_v11"

PDS_OUT = ROOT / "rq3_pds_v13_paperbudget.jsonl"
AUDIT_OUT = ROOT / "rq3_pds_v13_paperbudget_audit.json"

COMBINED_OUT = ROOT / "rq3_pe_plus_pds_v13_paperbudget.jsonl"
COMBINED_AUDIT_OUT = ROOT / "rq3_pe_plus_pds_v13_paperbudget_audit.json"


def load_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as f:
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


def txt(x):
    return x.strip() if isinstance(x, str) else ""


def canonical(x):
    x = unicodedata.normalize(
        "NFKC",
        txt(x),
    ).casefold()

    return "".join(
        ch for ch in x
        if unicodedata.category(ch)[0] in {"L", "N"}
    )


ELLIPSIS = re.compile(r"(?:\.{2,}|…+)")


def conservative_contains(needle, haystack):
    needle = txt(needle)
    haystack = txt(haystack)

    if not needle or not haystack:
        return False

    parts = [
        canonical(x)
        for x in ELLIPSIS.split(needle)
        if canonical(x)
    ]

    h = canonical(haystack)

    if not parts or not h:
        return False

    pos = 0

    for p in parts:
        idx = h.find(p, pos)

        if idx < 0:
            return False

        pos = idx + len(p)

    return True


###############################################################################
# 1. Frozen PE
###############################################################################

pe_rows = load_jsonl(PE)

if len(pe_rows) != 3732:
    raise RuntimeError(
        f"Expected PE3732, got {len(pe_rows)}"
    )


###############################################################################
# 2. Load original PDS jobs
###############################################################################

jobs_all = load_jsonl(JOBS)

if len(jobs_all) != 26672:
    raise RuntimeError(
        f"Expected 26672 jobs, got {len(jobs_all)}"
    )

jobs = {
    int(x["job_id"]): x
    for x in jobs_all
}


###############################################################################
# 3. Load all generated results
###############################################################################

generated = {}

for device in range(16):
    shard = SHARDS / f"device_{device}.jsonl"

    if not shard.exists():
        raise RuntimeError(
            f"Missing shard {shard}"
        )

    for row in load_jsonl(shard):
        jid = int(row["job_id"])

        if jid in generated:
            raise RuntimeError(
                f"Duplicate generation job_id={jid}"
            )

        generated[jid] = row


if len(generated) != 26672:
    raise RuntimeError(
        f"Expected 26672 generations, "
        f"got {len(generated)}"
    )


###############################################################################
# 4. PAPER-BUDGET SELECTION:
#    one knowledge anchor per selected PE example:
#    error_index == 0
#    four contexts per example.
###############################################################################

primary_jobs = [
    row
    for row in jobs_all
    if int(row["error_index"]) == 0
]

expected_candidates = 3732 * 4

if len(primary_jobs) != expected_candidates:
    raise RuntimeError(
        "Paper-budget cardinality mismatch: "
        f"primary_jobs={len(primary_jobs)}, "
        f"expected={expected_candidates}"
    )


per_parent = Counter(
    int(x["parent_row_pos"])
    for x in primary_jobs
)

bad_parent_counts = {
    k: v
    for k, v in per_parent.items()
    if v != 4
}

if bad_parent_counts:
    raise RuntimeError(
        f"Expected four PDS candidates per PE row; "
        f"bad={list(bad_parent_counts.items())[:20]}"
    )

if set(per_parent) != set(range(3732)):
    raise RuntimeError(
        "Not every PE row has a primary PDS anchor"
    )


###############################################################################
# 5. Official-style postprocessing.
#
# Public MT-Patcher collector effectively relies on:
#   - parseable source/target
#   - exact pair dedup
#
# P/Q compliance is audited below but is NOT used as a hard filter.
###############################################################################

valid = []

seen_pds_pairs = set()

parse_fail = 0
empty_pair = 0
duplicate_pds = 0

quality_flags = Counter()
coverage = Counter()

examples = defaultdict(list)


for job in sorted(
    primary_jobs,
    key=lambda x: int(x["job_id"]),
):
    jid = int(job["job_id"])

    row = generated[jid]

    src = txt(row.get("synthesized_source"))
    tgt = txt(row.get("synthesized_target"))

    if not row.get("parse_ok"):
        parse_fail += 1
        continue

    if not src or not tgt:
        empty_pair += 1
        continue

    # Exact string pair, matching public collector semantics.
    pair = (src, tgt)

    if pair in seen_pds_pairs:
        duplicate_pds += 1
        continue

    seen_pds_pairs.add(pair)

    source_span = txt(job.get("source_span"))
    correction = txt(job.get("correction"))
    parent_source = txt(job.get("source"))

    parent_span_ok = conservative_contains(
        source_span,
        parent_source,
    )

    generated_span_ok = conservative_contains(
        source_span,
        src,
    )

    correction_ok = conservative_contains(
        correction,
        tgt,
    )

    extended_ok = (
        canonical(src)
        != canonical(parent_source)
    )

    if parent_span_ok:
        quality_flags["parent_span_ok"] += 1
    else:
        quality_flags["parent_span_invalid"] += 1

    if generated_span_ok:
        quality_flags["generated_contains_P"] += 1
    else:
        quality_flags["generated_missing_P"] += 1

    if correction_ok:
        quality_flags["generated_contains_Q"] += 1
    else:
        quality_flags["generated_missing_Q"] += 1

    if extended_ok:
        quality_flags["source_extended"] += 1
    else:
        quality_flags["source_not_extended"] += 1

    # Suspicious literal placeholder leakage.
    placeholder_suspect = bool(
        re.search(
            r"(^|[\s:：])P($|[\s,，。.;；:：])",
            src,
        )
        or re.search(
            r"(^|[\s:：])Q($|[\s,，。.;；:：])",
            tgt,
        )
    )

    if placeholder_suspect:
        quality_flags["literal_placeholder_suspect"] += 1

        if len(examples["literal_placeholder_suspect"]) < 20:
            examples["literal_placeholder_suspect"].append(
                {
                    "job_id": jid,
                    "parent_index": job.get("parent_index"),
                    "source_span": source_span,
                    "correction": correction,
                    "synthesized_source": src,
                    "synthesized_target": tgt,
                }
            )

    if (
        not generated_span_ok
        and len(examples["generated_missing_P"]) < 20
    ):
        examples["generated_missing_P"].append(
            {
                "job_id": jid,
                "parent_index": job.get("parent_index"),
                "source_span": source_span,
                "correction": correction,
                "synthesized_source": src,
                "synthesized_target": tgt,
            }
        )

    if (
        not correction_ok
        and len(examples["generated_missing_Q"]) < 20
    ):
        examples["generated_missing_Q"].append(
            {
                "job_id": jid,
                "parent_index": job.get("parent_index"),
                "source_span": source_span,
                "correction": correction,
                "synthesized_source": src,
                "synthesized_target": tgt,
            }
        )

    parent_row = int(job["parent_row_pos"])

    coverage[parent_row] += 1

    valid.append(
        {
            "job_id": jid,
            "parent_row_pos": parent_row,
            "parent_index": job.get("parent_index"),
            "error_index": 0,
            "pds_slot": int(job["pds_slot"]),

            "source": src,
            "target_translation": tgt,

            "source_span": source_span,
            "correction": correction,
            "error_type": job.get("error_type"),

            "original_source": parent_source,

            "quality_audit": {
                "parent_span_ok": parent_span_ok,
                "generated_contains_P": generated_span_ok,
                "generated_contains_Q": correction_ok,
                "source_extended": extended_ok,
                "literal_placeholder_suspect":
                    placeholder_suspect,
            },

            "construction_method":
                "MT_PATCHER_PDS_QWEN3_8B_V13_PAPERBUDGET",
        }
    )


coverage_hist = Counter(
    coverage.get(i, 0)
    for i in range(3732)
)


###############################################################################
# 6. Save PDS
###############################################################################

with PDS_OUT.open(
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


audit = {
    "protocol":
        "MT_PATCHER_PDS_V13_PAPER_BUDGET",

    "design": {
        "selected_pe_examples":
            3732,

        "knowledge_anchor_per_pe":
            1,

        "anchor_selection":
            "feedback_errors[0]",

        "requested_contexts_per_pe":
            4,

        "expected_candidates":
            expected_candidates,

        "hard_postprocess":
            "parse_nonempty_exact_pair_dedup",

        "P_Q_containment":
            "audit_only",
    },

    "candidate_rows":
        len(primary_jobs),

    "parse_fail":
        parse_fail,

    "empty_pair":
        empty_pair,

    "duplicate_pds_pair":
        duplicate_pds,

    "pds_kept":
        len(valid),

    "pds_keep_ratio":
        len(valid) / expected_candidates,

    "quality_flags":
        dict(quality_flags),

    "coverage_histogram":
        {
            str(k): coverage_hist[k]
            for k in sorted(coverage_hist)
        },

    "zero_context_pe_rows":
        coverage_hist[0],

    "mean_kept_contexts_per_pe":
        len(valid) / 3732,

    "examples":
        dict(examples),

    "jobs_sha256":
        sha256(JOBS),

    "pds_sha256":
        sha256(PDS_OUT),
}


with AUDIT_OUT.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        audit,
        f,
        indent=2,
        ensure_ascii=False,
    )


###############################################################################
# 7. Build causal PE + PDS baseline.
#
# Preserve all 3732 PE examples exactly.
###############################################################################

combined = []

for row in pe_rows:
    x = dict(row)
    x["rq3_data_component"] = "PE"
    combined.append(x)


# Prevent added PDS examples from duplicating frozen PE pairs.
pe_pairs = {
    (
        txt(row["source"]),
        txt(row["target_translation"]),
    )
    for row in pe_rows
}


pds_kept_final = 0
pds_overlap_pe = 0

for row in valid:
    pair = (
        row["source"],
        row["target_translation"],
    )

    if pair in pe_pairs:
        pds_overlap_pe += 1
        continue

    combined.append(
        {
            "index":
                f"pds_v13_{row['job_id']}",

            "source":
                row["source"],

            "messages":
                [
                    {
                        "role": "user",
                        "content":
                            "Translate the following text into English "
                            "without additional explanations:\n\n"
                            + row["source"]
                            + "\n\n",
                    }
                ],

            "target_translation":
                row["target_translation"],

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
                            row.get("error_type", ""),

                        "explanation":
                            "PDS synthesized context",

                        "correction":
                            row["correction"],
                    }
                ],

            "construction_method":
                "MT_PATCHER_PDS_QWEN3_8B_V13_PAPERBUDGET",

            "rq3_data_component":
                "PDS",

            "parent_index":
                row["parent_index"],

            "parent_row_pos":
                row["parent_row_pos"],

            "error_index":
                0,

            "pds_slot":
                row["pds_slot"],

            "quality_audit":
                row["quality_audit"],
        }
    )

    pds_kept_final += 1


rng = random.Random(20260825)
rng.shuffle(combined)


with COMBINED_OUT.open(
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
    "protocol":
        "RQ3_PE_PLUS_PDS_V13_PAPER_BUDGET",

    "pe_input_rows":
        len(pe_rows),

    "pe_kept":
        len(pe_rows),

    "pe_exactly_preserved":
        True,

    "requested_pds_candidates":
        expected_candidates,

    "pds_after_postprocess":
        len(valid),

    "pds_overlap_with_pe_removed":
        pds_overlap_pe,

    "pds_kept":
        pds_kept_final,

    "combined_rows":
        len(combined),

    "theoretical_full_budget":
        3732 + (3732 * 4),

    "shuffle_seed":
        20260825,

    "pe_sha256":
        sha256(PE),

    "pds_sha256":
        sha256(PDS_OUT),

    "combined_sha256":
        sha256(COMBINED_OUT),
}


with COMBINED_AUDIT_OUT.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        combined_audit,
        f,
        indent=2,
        ensure_ascii=False,
    )


###############################################################################
# 8. Print compact result.
###############################################################################

print("======================================================================")
print("RQ3 PDS V13 — PAPER-BUDGET BASELINE")
print("======================================================================")

print(
    json.dumps(
        audit,
        indent=2,
        ensure_ascii=False,
    )
)

print()
print("======================================================================")
print("RQ3 PE + PDS V13")
print("======================================================================")

print(
    json.dumps(
        combined_audit,
        indent=2,
        ensure_ascii=False,
    )
)

if combined_audit["pe_kept"] != 3732:
    raise RuntimeError(
        "Frozen PE3732 was not preserved"
    )

if len(primary_jobs) != 14928:
    raise RuntimeError(
        "Paper PDS budget invariant failed"
    )

print()
print("PDS_V13_PAPER_BUDGET_PASS")
print("PE3732_EXACT_PRESERVATION_PASS")
print("RQ3_CANONICAL_PE_PLUS_PDS_DATA_READY")
