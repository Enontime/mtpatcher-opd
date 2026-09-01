#!/usr/bin/env bash
set -Eeuo pipefail

source /workspace/mtpatcher/project_env.sh

EXP="mtpatcher_v3_full6565_20260823"
D="$DATA_ROOT/$EXP"

OFFICIAL_CONSTRUCT="$ROOT/vendor/MT-Patcher-official/data_scripts/construct_combined_dataset.py"
OFFICIAL_CASE="$ROOT/vendor/MT-Patcher-official/pipeline/data_manager/llama_case_generation.py"

PF_JOBS="$D/paperfaith_paper20k_v2/pds_jobs.jsonl"

CURRENT_PAIRS="$D/strong_repro_pds_population_v1/pds_local_pairs_v1.jsonl"

OUT="$D/strong_repro_pds_multiplicity_lock_v1"

REPORT="$OUT/pds_multiplicity_lock_report_v1.txt"
JSON="$OUT/pds_multiplicity_lock_report_v1.json"
HASHES="$OUT/frozen_sha256_v1.txt"

PASS="$OUT/STRONG_REPRO_PDS_MULTIPLICITY_LOCK_V1.PASS"
FAIL="$OUT/STRONG_REPRO_PDS_MULTIPLICITY_LOCK_V1.FAIL"

mkdir -p "$OUT"
rm -f "$PASS" "$FAIL"

START="$(date +%s)"

echo "======================================================================"
echo "STRONG REPRO PDS MULTIPLICITY LOCK V1"
echo "======================================================================"
echo
echo "预计运行时长：2–10 秒；保守上限 1 分钟"
echo "当前时间："
echo "  北京时间: $(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
echo "  Server UTC: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
echo

trap '
rc=$?
echo "PDS_MULTIPLICITY_LOCK_FAILED rc=$rc"
touch "'"$FAIL"'"
' ERR


for f in \
    "$OFFICIAL_CONSTRUCT" \
    "$OFFICIAL_CASE" \
    "$PF_JOBS" \
    "$CURRENT_PAIRS"
do
    if [ ! -f "$f" ]; then
        echo "MISSING=$f"
        false
    fi
done


python - \
    "$OFFICIAL_CONSTRUCT" \
    "$OFFICIAL_CASE" \
    "$PF_JOBS" \
    "$CURRENT_PAIRS" \
    "$REPORT" \
    "$JSON" <<'PY'

import json
import re
import sys
from collections import Counter
from pathlib import Path

construct = Path(sys.argv[1])
casegen = Path(sys.argv[2])
pf_jobs = Path(sys.argv[3])
current_pairs = Path(sys.argv[4])
report_path = Path(sys.argv[5])
json_path = Path(sys.argv[6])


def load_jsonl(path):
    rows = []

    with path.open(
        encoding="utf-8-sig"
    ) as f:

        for n, line in enumerate(f, 1):

            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(
                    f"{path}:{n}: {e}"
                )

    return rows


def source_hits(path):

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    patterns = (
        "for ",
        ".append(",
        "range(",
        "repeat",
        "num_return_sequences",
        "num_samples",
        "n_samples",
        "sentence_analysis",
        "error_source",
        "correction",
        "word_pair",
        "Word Pair",
        "prompt",
        "generate(",
    )

    hits = []

    for i, line in enumerate(
        text.splitlines(),
        1,
    ):

        if any(
            p.lower() in line.lower()
            for p in patterns
        ):
            hits.append(
                f"{i:04d}: {line}"
            )

    return text, hits


construct_text, construct_hits = (
    source_hits(construct)
)

case_text, case_hits = (
    source_hits(casegen)
)


###############################################################################
# Historical paper-faithful PDS jobs
###############################################################################

jobs = load_jsonl(pf_jobs)

pair_keys = []
prompt_keys = []

missing_analysis = 0
missing_word_pair = 0


for row in jobs:

    analysis = str(
        row.get(
            "sentence_analysis",
            ""
        )
    ).strip()

    prompt = str(
        row.get(
            "prompt",
            ""
        )
    ).strip()

    if not analysis:
        missing_analysis += 1

    m = re.search(
        r"(?mi)^Word Pair:\s*(.+?)\s*$",
        prompt,
    )

    if m:
        wp = m.group(1).strip()
    else:
        wp = ""
        missing_word_pair += 1

    pair_keys.append(
        (
            analysis,
            wp,
        )
    )

    prompt_keys.append(prompt)


pair_counter = Counter(pair_keys)
prompt_counter = Counter(prompt_keys)

pair_mult_hist = Counter(
    pair_counter.values()
)

prompt_mult_hist = Counter(
    prompt_counter.values()
)


###############################################################################
# Current canonical local-pair population
###############################################################################

pairs = load_jsonl(current_pairs)


###############################################################################
# Look specifically for explicit repeat machinery in official source.
###############################################################################

combined = (
    construct_text
    + "\n"
    + case_text
)

repeat_patterns = {
    "repeat_token":
        r"\brepeat\b",

    "range_4":
        r"range\s*\(\s*4\s*\)",

    "num_return_sequences":
        r"num_return_sequences",

    "num_samples":
        r"num[_ ]?samples",

    "n_samples":
        r"\bn[_ ]?samples\b",
}

repeat_hits = {}

for name, pat in repeat_patterns.items():

    repeat_hits[name] = bool(
        re.search(
            pat,
            combined,
            flags=re.I,
        )
    )


###############################################################################
# Evidence-level decision.
#
# This is REPO/HISTORICAL implementation evidence.
# It is NOT automatically a PAPER-EXACT claim.
###############################################################################

historical_one_key_each = (
    len(pair_keys) > 0
    and
    len(pair_counter) == len(pair_keys)
    and
    max(pair_counter.values()) == 1
)

explicit_repeat_found = any(
    repeat_hits.values()
)


if (
    historical_one_key_each
    and
    not explicit_repeat_found
):

    decision = (
        "EVIDENCE_SUPPORTS_ONE_JOB_PER_LOCAL_PAIR"
    )

elif explicit_repeat_found:

    decision = (
        "EXPLICIT_REPEAT_MACHINERY_FOUND_REVIEW_REQUIRED"
    )

else:

    decision = (
        "MULTIPLICITY_REMAINS_UNRESOLVED"
    )


report = {
    "protocol":
        "STRONG_REPRO_PDS_MULTIPLICITY_LOCK_V1",

    "current_population": {
        "parent_rows":
            11792,

        "valid_local_pairs":
            len(pairs),

        "repeat4_provisional_jobs":
            len(pairs) * 4,
    },

    "historical_paperfaith": {
        "pds_job_rows":
            len(jobs),

        "unique_sentence_analysis_word_pair_keys":
            len(pair_counter),

        "max_key_multiplicity":
            max(
                pair_counter.values()
            ) if pair_counter else 0,

        "key_multiplicity_histogram":
            {
                str(k): v
                for k, v
                in sorted(
                    pair_mult_hist.items()
                )
            },

        "unique_exact_prompts":
            len(prompt_counter),

        "max_prompt_multiplicity":
            max(
                prompt_counter.values()
            ) if prompt_counter else 0,

        "prompt_multiplicity_histogram":
            {
                str(k): v
                for k, v
                in sorted(
                    prompt_mult_hist.items()
                )
            },

        "missing_sentence_analysis":
            missing_analysis,

        "missing_word_pair":
            missing_word_pair,
    },

    "official_repeat_markers":
        repeat_hits,

    "decision":
        decision,

    "fidelity_note":
        (
            "This locks released-repo and "
            "historical implementation multiplicity. "
            "It does not by itself claim PAPER-EXACT "
            "multiplicity if the paper is silent."
        ),
}


json_path.write_text(
    json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


lines = []

lines.append(
    "=" * 78
)

lines.append(
    "STRONG REPRO PDS MULTIPLICITY LOCK V1"
)

lines.append(
    "=" * 78
)

lines.append("")

lines.append(
    f"CURRENT_VALID_LOCAL_PAIRS = {len(pairs)}"
)

lines.append(
    f"PROVISIONAL_REPEAT4_JOBS = {len(pairs) * 4}"
)

lines.append("")

lines.append(
    f"PAPERFAITH_PDS_JOB_ROWS = {len(jobs)}"
)

lines.append(
    "PAPERFAITH_UNIQUE_ANALYSIS_WORDPAIR = "
    f"{len(pair_counter)}"
)

lines.append(
    "PAPERFAITH_MAX_KEY_MULTIPLICITY = "
    f"{max(pair_counter.values()) if pair_counter else 0}"
)

lines.append(
    "PAPERFAITH_KEY_MULTIPLICITY_HIST = "
    f"{dict(sorted(pair_mult_hist.items()))}"
)

lines.append(
    "PAPERFAITH_UNIQUE_EXACT_PROMPTS = "
    f"{len(prompt_counter)}"
)

lines.append(
    "PAPERFAITH_MAX_PROMPT_MULTIPLICITY = "
    f"{max(prompt_counter.values()) if prompt_counter else 0}"
)

lines.append(
    f"MISSING_SENTENCE_ANALYSIS = {missing_analysis}"
)

lines.append(
    f"MISSING_WORD_PAIR = {missing_word_pair}"
)

lines.append("")

lines.append(
    "OFFICIAL_REPEAT_MARKERS = "
    + json.dumps(
        repeat_hits,
        sort_keys=True,
    )
)

lines.append("")

lines.append(
    "===== OFFICIAL construct_combined_dataset.py ====="
)

lines.extend(construct_hits)

lines.append("")

lines.append(
    "===== OFFICIAL llama_case_generation.py ====="
)

lines.extend(case_hits)

lines.append("")

lines.append(
    f"DECISION = {decision}"
)

lines.append(
    "NOTE = repo/historical implementation evidence; "
    "not automatically PAPER-EXACT."
)


report_path.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

print("\n".join(lines))

print()
print(
    "PDS_MULTIPLICITY_EVIDENCE_EXTRACTION_PASS"
)
PY


{
    sha256sum \
        "$OFFICIAL_CONSTRUCT" \
        "$OFFICIAL_CASE" \
        "$PF_JOBS" \
        "$CURRENT_PAIRS" \
        "$REPORT" \
        "$JSON"
} > "$HASHES"


touch "$PASS"
rm -f "$FAIL"

END="$(date +%s)"

echo
echo "======================================================================"
echo "STRONG REPRO PDS MULTIPLICITY LOCK V1 PASS"
echo "======================================================================"
echo "实际运行时长: $((END - START)) 秒"
echo "完成时间:"
echo "  北京时间: $(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
echo "  Server UTC: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
echo
echo "REPORT=$REPORT"
echo "JSON=$JSON"
echo "PASS=$PASS"
echo "======================================================================"
