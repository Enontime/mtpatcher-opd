import csv
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


JOBS = Path(sys.argv[1])
RAW = Path(sys.argv[2])
OLD_HIDDEN = Path(sys.argv[3])
OUT = Path(sys.argv[4])

SEED = 20260831
TARGET_AUDIT_N = 100
MIN_AUDIT_N = 80


PARSED = OUT / "pds_structural_parse_all1100_v1.jsonl"
ACCEPTED = OUT / "pds_structural_accepted_A_v1.jsonl"
REJECTED = OUT / "pds_structural_rejected_v1.jsonl"

REPORT = OUT / "pds_structural_acceptance_report_v1.json"

BLIND = OUT / "accepted_set_semantic_blind100_v1.csv"
HIDDEN = OUT / "accepted_set_semantic_hidden100_v1.jsonl"
PROTOCOL = OUT / "accepted_set_semantic_protocol_v1.txt"

HASHES = OUT / "frozen_sha256_manifest_v1.txt"


###############################################################################
# IO
###############################################################################

def load_jsonl(path):
    rows = []

    with path.open(
        encoding="utf-8-sig",
        errors="replace",
    ) as f:

        for n, line in enumerate(f, 1):

            if not line.strip():
                continue

            try:
                rows.append(
                    json.loads(line)
                )

            except Exception as e:
                raise RuntimeError(
                    f"{path}:{n}: {e}"
                )

    return rows


###############################################################################
# NORMALIZATION
###############################################################################

def norm(s):
    s = unicodedata.normalize(
        "NFKC",
        str(s),
    ).lower()

    return "".join(
        ch
        for ch in s
        if ch.isalnum()
    )


def cjk_count(s):
    return len(
        re.findall(
            r"[\u3400-\u4dbf\u4e00-\u9fff]",
            str(s),
        )
    )


def en_words(s):
    return re.findall(
        r"[A-Za-z]+"
        r"(?:['’-][A-Za-z]+)?",
        str(s),
    )


def q_words(s):
    return len(
        en_words(s)
    )


def clean_md(s):
    s = str(s).strip()

    s = re.sub(
        r"^\s*#{1,6}\s*",
        "",
        s,
    )

    s = re.sub(
        r"^\s*[-•]\s*",
        "",
        s,
    )

    s = s.replace("**", "")
    s = s.replace("__", "")

    return s.strip()


def plausible_zh(s):
    cjk = cjk_count(s)
    latin = len(en_words(s))

    return (
        cjk >= 4
        and
        cjk >= max(
            4,
            latin,
        )
    )


def plausible_en(s):
    words = len(
        en_words(s)
    )

    cjk = cjk_count(s)

    return (
        words >= 4
        and
        cjk <= max(
            4,
            words // 3,
        )
    )


###############################################################################
# HEADING / META DETECTION
###############################################################################

ZH_LABEL = re.compile(
    r"^(?:"
    r"中文"
    r"(?:句子|新闻报道)?"
    r"(?:\s*[\(（][^()（）]*[\)）])?"
    r"|Chinese"
    r"(?:\s+Sentence)?"
    r"(?:\s*[\(（][^()（）]*[\)）])?"
    r"|中译英"
    r")"
    r"\s*[:：]\s*(.*)$",
    re.I,
)


EN_LABEL = re.compile(
    r"^(?:"
    r"英文"
    r"(?:句子|新闻报道)?"
    r"(?:\s*[\(（][^()（）]*[\)）])?"
    r"|English"
    r"(?:\s+Sentence)?"
    r"(?:\s*[\(（][^()（）]*[\)）])?"
    r"|英译中"
    r")"
    r"\s*[:：]\s*(.*)$",
    re.I,
)


META_HEAD = re.compile(
    r"^(?:"
    r"Explanation|Analysis|解析|说明|"
    r"Topic|Domain|Style|Summary|"
    r"Word\s*Pair|词对|双语词对"
    r")"
    r"\s*[:：]?",
    re.I,
)


def detect_lang_label(line):
    s = clean_md(line)

    m = ZH_LABEL.match(s)

    if m:
        return (
            "zh",
            clean_md(
                m.group(1)
            ),
        )

    m = EN_LABEL.match(s)

    if m:
        return (
            "en",
            clean_md(
                m.group(1)
            ),
        )

    return (
        None,
        None,
    )


def is_meta_heading(line):
    s = clean_md(line)

    return bool(
        META_HEAD.match(s)
    )


###############################################################################
# BLOCK EXTRACTION
###############################################################################

def extract_labeled_blocks(raw):

    lines = str(raw).splitlines()

    blocks = {
        "zh": [],
        "en": [],
    }

    i = 0

    while i < len(lines):

        lang, inline = detect_lang_label(
            lines[i]
        )

        if lang is None:
            i += 1
            continue

        buf = []

        if inline:
            buf.append(
                inline
            )

        j = i + 1

        while j < len(lines):

            next_lang, _ = detect_lang_label(
                lines[j]
            )

            if next_lang is not None:
                break

            if (
                is_meta_heading(
                    lines[j]
                )
                and
                buf
            ):
                break

            if (
                clean_md(lines[j])
                == "---"
                and
                buf
            ):
                break

            if lines[j].strip():
                buf.append(
                    lines[j].strip()
                )

            j += 1

        text = clean_md(
            "\n".join(buf)
        ).strip()

        if text:
            blocks[lang].append(
                text
            )

        i = max(
            j,
            i + 1,
        )


    def dedupe(items):
        seen = set()
        out = []

        for x in items:
            k = norm(x)

            if not k:
                continue

            if k in seen:
                continue

            seen.add(k)
            out.append(x)

        return out


    blocks["zh"] = dedupe(
        blocks["zh"]
    )

    blocks["en"] = dedupe(
        blocks["en"]
    )

    return blocks


###############################################################################
# FALLBACK SEGMENTS
###############################################################################

def useful_paragraphs(raw):

    paras = re.split(
        r"\n\s*\n+",
        str(raw),
    )

    out = []

    for p in paras:

        p = clean_md(p)

        if not p:
            continue

        if is_meta_heading(p):
            continue

        low = p.lower()

        if (
            "word pair:" in low
            or
            "双语词对" in p
            or
            "词对：" in p
        ):
            continue

        out.append(p)

    return out


###############################################################################
# PARSER
###############################################################################

def parse_pair(
    raw,
    P,
):

    raw = str(raw)

    ###########################################################################
    # Method 1: explicit labeled Chinese / English blocks.
    ###########################################################################

    blocks = extract_labeled_blocks(
        raw
    )

    zh = [
        x
        for x in blocks["zh"]
        if plausible_zh(x)
    ]

    en = [
        x
        for x in blocks["en"]
        if plausible_en(x)
    ]


    # Prefer a source candidate that actually contains P.
    zh_p = [
        x
        for x in zh
        if (
            norm(P)
            and
            norm(P) in norm(x)
        )
    ]


    if (
        len(zh_p) == 1
        and
        len(en) == 1
    ):
        return {
            "status":
                "PARSEABLE",

            "method":
                "EXPLICIT_LABELS",

            "X_prime":
                zh_p[0],

            "Y_prime":
                en[0],
        }


    if (
        len(zh_p) > 1
        or
        len(en) > 1
    ):
        return {
            "status":
                "AMBIGUOUS_PARSE",

            "method":
                "EXPLICIT_LABELS",

            "X_prime":
                None,

            "Y_prime":
                None,
        }


    ###########################################################################
    # Method 2: pipe-separated bilingual headline.
    ###########################################################################

    pipe_candidates = []

    for line in raw.splitlines():

        if "|" not in line:
            continue

        parts = [
            clean_md(x)
            for x in line.split("|")
            if clean_md(x)
        ]

        if len(parts) != 2:
            continue

        a, b = parts

        if (
            plausible_zh(a)
            and
            plausible_en(b)
            and
            norm(P) in norm(a)
        ):
            pipe_candidates.append(
                (a, b)
            )

        elif (
            plausible_zh(b)
            and
            plausible_en(a)
            and
            norm(P) in norm(b)
        ):
            pipe_candidates.append(
                (b, a)
            )


    uniq_pipe = {}

    for x, y in pipe_candidates:
        uniq_pipe[
            (
                norm(x),
                norm(y),
            )
        ] = (
            x,
            y,
        )


    if len(uniq_pipe) == 1:

        x, y = next(
            iter(
                uniq_pipe.values()
            )
        )

        return {
            "status":
                "PARSEABLE",

            "method":
                "PIPE_PAIR",

            "X_prime":
                x,

            "Y_prime":
                y,
        }


    if len(uniq_pipe) > 1:

        return {
            "status":
                "AMBIGUOUS_PARSE",

            "method":
                "PIPE_PAIR",

            "X_prime":
                None,

            "Y_prime":
                None,
        }


    ###########################################################################
    # Method 3: unlabeled paragraph pair.
    ###########################################################################

    paras = useful_paragraphs(
        raw
    )

    zh_candidates = [
        p
        for p in paras
        if (
            plausible_zh(p)
            and
            norm(P)
            and
            norm(P) in norm(p)
        )
    ]

    en_candidates = [
        p
        for p in paras
        if plausible_en(p)
    ]


    # Deduplicate.
    zh_map = {
        norm(x): x
        for x in zh_candidates
        if norm(x)
    }

    en_map = {
        norm(x): x
        for x in en_candidates
        if norm(x)
    }


    if (
        len(zh_map) == 1
        and
        len(en_map) == 1
    ):

        return {
            "status":
                "PARSEABLE",

            "method":
                "PARAGRAPH_PAIR",

            "X_prime":
                next(
                    iter(
                        zh_map.values()
                    )
                ),

            "Y_prime":
                next(
                    iter(
                        en_map.values()
                    )
                ),
        }


    if (
        len(zh_map) > 1
        or
        len(en_map) > 1
    ):

        return {
            "status":
                "AMBIGUOUS_PARSE",

            "method":
                "PARAGRAPH_PAIR",

            "X_prime":
                None,

            "Y_prime":
                None,
        }


    ###########################################################################
    # Method 4: Chinese sentence + substantial English parenthetical.
    ###########################################################################

    parenthetical = re.findall(
        r"[\(（]([^()（）]{20,500})[\)）]",
        raw,
    )

    en_parens = [
        clean_md(x)
        for x in parenthetical
        if len(
            en_words(x)
        ) >= 8
    ]


    source_paras = [
        p
        for p in paras
        if (
            plausible_zh(p)
            and
            norm(P)
            and
            norm(P) in norm(p)
        )
    ]


    if (
        len(source_paras) == 1
        and
        len(en_parens) == 1
    ):

        x = source_paras[0]

        y = en_parens[0]

        # Remove the English parenthetical from X if it is embedded there.
        x_clean = re.sub(
            r"[\(（]"
            + re.escape(y)
            + r"[\)）]",
            "",
            x,
        ).strip()

        if plausible_zh(x_clean):

            return {
                "status":
                    "PARSEABLE",

                "method":
                    "ENGLISH_PARENTHETICAL",

                "X_prime":
                    x_clean,

                "Y_prime":
                    y,
            }


    ###########################################################################
    # Failure taxonomy.
    ###########################################################################

    possible_en = [
        p
        for p in paras
        if plausible_en(p)
    ]

    possible_zh = [
        p
        for p in paras
        if plausible_zh(p)
    ]


    if not possible_en:
        reason = (
            "NO_ENGLISH_TARGET"
        )

    elif not possible_zh:
        reason = (
            "NO_SOURCE_BLOCK"
        )

    else:
        reason = (
            "UNPARSEABLE_OTHER"
        )


    return {
        "status":
            reason,

        "method":
            None,

        "X_prime":
            None,

        "Y_prime":
            None,
    }


###############################################################################
# LOAD FROZEN DATA
###############################################################################

job_rows = load_jsonl(
    JOBS
)

raw_rows = load_jsonl(
    RAW
)

old_hidden_rows = load_jsonl(
    OLD_HIDDEN
)


jobs = {
    int(x["job_id"]): x
    for x in job_rows
}

raw = {
    int(x["job_id"]): x
    for x in raw_rows
}


if len(jobs) != 1100:
    raise RuntimeError(
        f"expected1100 jobs got={len(jobs)}"
    )

if len(raw) != 1100:
    raise RuntimeError(
        f"expected1100 raw got={len(raw)}"
    )

if set(jobs) != set(raw):
    raise RuntimeError(
        "jobs/raw job_id mismatch"
    )


OLD_AUDITED_PAIR_IDS = {
    int(x["pair_id"])
    for x in old_hidden_rows
}


if len(OLD_AUDITED_PAIR_IDS) != 60:
    raise RuntimeError(
        "expected exactly 60 previously audited unique pair_ids; "
        f"got={len(OLD_AUDITED_PAIR_IDS)}"
    )


###############################################################################
# PARSE / STRUCTURAL FILTER
###############################################################################

parsed_rows = []
accepted_rows = []
rejected_rows = []

reject_counts = Counter()
method_counts = Counter()


for jid in sorted(jobs):

    j = jobs[jid]

    r = raw[jid]

    P = str(
        j["source_span"]
    ).strip()

    Q = str(
        j["correction"]
    ).strip()

    parent = str(
        j["original_source"]
    ).strip()

    text = str(
        r.get(
            "raw_generation",
            "",
        )
    )


    result = parse_pair(
        text,
        P,
    )


    row = {
        "job_id":
            jid,

        "pair_id":
            int(
                j["pair_id"]
            ),

        "parent_index":
            int(
                j["parent_index"]
            ),

        "pds_slot":
            int(
                j["pds_slot"]
            ),

        "parse_status":
            result["status"],

        "parse_method":
            result["method"],

        "P":
            P,

        "Q":
            Q,

        "X_prime":
            result["X_prime"],

        "Y_prime":
            result["Y_prime"],

        "P_exact_in_X":
            None,

        "parent_copy_in_X":
            None,

        "Q_exact_in_Y":
            None,

        "accepted_A":
            False,

        "reject_reason":
            None,
    }


    if (
        result["status"]
        !=
        "PARSEABLE"
    ):

        reason = result["status"]

        row["reject_reason"] = reason

        reject_counts[
            reason
        ] += 1

        rejected_rows.append(
            row
        )

        parsed_rows.append(
            row
        )

        continue


    X = result["X_prime"]
    Y = result["Y_prime"]


    method_counts[
        result["method"]
    ] += 1


    P_in_X = (
        bool(norm(P))
        and
        norm(P) in norm(X)
    )

    parent_copy = (
        bool(norm(parent))
        and
        norm(parent) in norm(X)
    )

    Q_in_Y = (
        bool(norm(Q))
        and
        norm(Q) in norm(Y)
    )


    row[
        "P_exact_in_X"
    ] = P_in_X

    row[
        "parent_copy_in_X"
    ] = parent_copy

    row[
        "Q_exact_in_Y"
    ] = Q_in_Y


    if parent_copy:

        row[
            "reject_reason"
        ] = (
            "PARENT_COPY_X"
        )

        reject_counts[
            "PARENT_COPY_X"
        ] += 1

        rejected_rows.append(
            row
        )

        parsed_rows.append(
            row
        )

        continue


    if not P_in_X:

        row[
            "reject_reason"
        ] = (
            "P_MISSING_X"
        )

        reject_counts[
            "P_MISSING_X"
        ] += 1

        rejected_rows.append(
            row
        )

        parsed_rows.append(
            row
        )

        continue


    # Q exact is DIAGNOSTIC ONLY.
    row[
        "accepted_A"
    ] = True

    accepted_rows.append(
        row
    )

    parsed_rows.append(
        row
    )


###############################################################################
# STRUCTURAL REPORT
###############################################################################

pair_to_rows = defaultdict(
    list
)

for row in accepted_rows:

    pair_to_rows[
        row["pair_id"]
    ].append(
        row
    )


accepted_pair_ids = set(
    pair_to_rows
)

unseen_pair_ids = sorted(
    accepted_pair_ids
    -
    OLD_AUDITED_PAIR_IDS
)


q_exact_accepted = sum(
    int(
        bool(
            x["Q_exact_in_Y"]
        )
    )
    for x in accepted_rows
)


report = {
    "protocol":
        "PDS_STRUCTURAL_ACCEPTANCE_V1",

    "generator_changed":
        False,

    "generator_treatment":
        "FROZEN_TOPIC_DOMAIN_STYLE_ONLY",

    "acceptance_definition_A": {
        "unique_parseable_bilingual_pair":
            True,

        "parent_copy_full_parent_in_X_prime":
            False,

        "P_normalized_containment_in_X_prime":
            True,

        "Q_exact_in_Y_prime":
            "DIAGNOSTIC_ONLY_NOT_A_GATE",
    },

    "raw_outputs":
        1100,

    "parseable_outputs":
        sum(
            1
            for x in parsed_rows
            if x["parse_status"]
            ==
            "PARSEABLE"
        ),

    "accepted_A_outputs":
        len(
            accepted_rows
        ),

    "accepted_A_output_rate":
        len(
            accepted_rows
        )
        / 1100,

    "accepted_A_unique_pairs":
        len(
            accepted_pair_ids
        ),

    "accepted_A_unseen_unique_pairs":
        len(
            unseen_pair_ids
        ),

    "previously_audited_pair_ids_excluded":
        len(
            OLD_AUDITED_PAIR_IDS
        ),

    "Q_exact_in_Y_among_A_outputs":
        q_exact_accepted,

    "Q_exact_in_Y_rate_among_A_outputs":
        (
            q_exact_accepted
            / len(accepted_rows)
            if accepted_rows
            else None
        ),

    "rejection_counts":
        dict(
            reject_counts
        ),

    "parse_method_counts":
        dict(
            method_counts
        ),
}


###############################################################################
# FREEZE FULL PARSE / ACCEPT / REJECT
###############################################################################

for path, rows in (
    (
        PARSED,
        parsed_rows,
    ),
    (
        ACCEPTED,
        accepted_rows,
    ),
    (
        REJECTED,
        rejected_rows,
    ),
):

    with path.open(
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


REPORT.write_text(
    json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


###############################################################################
# CONFIRMATORY AUDIT SAMPLE
#
# IMPORTANT ESTIMAND:
# uniform unique local pair, then one accepted output uniformly from that pair.
# This is PAIR-BALANCED, not output-weighted precision.
###############################################################################

if len(unseen_pair_ids) < MIN_AUDIT_N:

    raise RuntimeError(
        "insufficient unseen accepted unique pairs for confirmatory audit: "
        f"{len(unseen_pair_ids)} < {MIN_AUDIT_N}"
    )


rng = random.Random(
    SEED
)


audit_n = min(
    TARGET_AUDIT_N,
    len(
        unseen_pair_ids
    ),
)


sampled_pairs = rng.sample(
    unseen_pair_ids,
    audit_n,
)


selected = []

for pair_id in sampled_pairs:

    choices = list(
        pair_to_rows[
            pair_id
        ]
    )

    chosen = rng.choice(
        choices
    )

    chosen = dict(
        chosen
    )

    chosen[
        "accepted_outputs_for_pair"
    ] = len(
        choices
    )

    selected.append(
        chosen
    )


rng.shuffle(
    selected
)


blind_rows = []
hidden_rows = []


for i, row in enumerate(
    selected,
    1,
):

    audit_id = (
        f"PA{i:03d}"
    )

    blind_rows.append({
        "audit_id":
            audit_id,

        "P_source_knowledge":
            row["P"],

        "Q_intended_correction":
            row["Q"],

        "X_prime":
            row["X_prime"],

        "Y_prime":
            row["Y_prime"],

        "semantic_realization_status":
            "",

        "issue_note":
            "",

        "confidence":
            "",
    })


    hidden_rows.append({
        "audit_id":
            audit_id,

        "job_id":
            row["job_id"],

        "pair_id":
            row["pair_id"],

        "parent_index":
            row["parent_index"],

        "pds_slot":
            row["pds_slot"],

        "parse_method":
            row["parse_method"],

        "accepted_outputs_for_pair":
            row[
                "accepted_outputs_for_pair"
            ],

        "Q_words":
            q_words(
                row["Q"]
            ),

        "Q_exact_in_Y":
            row[
                "Q_exact_in_Y"
            ],

        "P_exact_in_X":
            row[
                "P_exact_in_X"
            ],

        "parent_copy_in_X":
            row[
                "parent_copy_in_X"
            ],

        "previous_longq_audit_pair_overlap":
            False,
    })


with BLIND.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "audit_id",
            "P_source_knowledge",
            "Q_intended_correction",
            "X_prime",
            "Y_prime",
            "semantic_realization_status",
            "issue_note",
            "confidence",
        ],
    )

    writer.writeheader()

    writer.writerows(
        blind_rows
    )


with HIDDEN.open(
    "w",
    encoding="utf-8",
) as f:

    for row in hidden_rows:

        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


protocol = f"""PDS ACCEPTED-SET SEMANTIC CONFIRMATORY AUDIT V1

SCIENTIFIC QUESTION
-------------------
Among local pairs that pass deterministic structural acceptance A,
how often does the extracted bilingual pair (X',Y') semantically realize
the intended local correction P -> Q?

GENERATOR
---------
FROZEN.
No generator change in this stage.

A — STRUCTURAL ACCEPTANCE
-------------------------
1. deterministic unique bilingual extraction -> (X',Y')
2. reject no-English / no-source / ambiguous / unparseable outputs
3. reject if full original parent source is contained in X'
4. require normalized P containment in X'
5. record normalized Q containment in Y' as diagnostic metadata only
6. Q exact DOES NOT affect retain/discard

CONFIRMATORY SAMPLING
---------------------
- sampling unit = unique local pair_id
- previously audited long-Q pair_ids are excluded
- uniform random sample over unseen accepted pair_ids
- one accepted output sampled uniformly within each selected pair
- seed = {SEED}
- target N = {TARGET_AUDIT_N}
- actual N = {audit_n}

ESTIMAND
--------
Pair-balanced semantic validity among accepted unseen local pairs.

This is intentionally NOT called output-weighted
P(valid | accepted output).

BLINDNESS
---------
Auditor sees:
- P
- Q
- extracted X'
- extracted Y'

Auditor does NOT see:
- Q_exact_in_Y
- pair_id / job_id / slot
- parse method
- number of accepted outputs for that pair

Q length is inherently observable from Q text.

RUBRIC
------
FULL:
  X' -> Y' fully realizes the intended correction meaning represented by P -> Q.
  Paraphrase is allowed. Exact Q string is not required.

PARTIAL:
  Core corrected meaning is present, but a material component is omitted,
  weakened, altered, or only partly realized.

FAIL:
  Q meaning is absent, contradicted, mistranslated, source-target mismatched,
  hallucinated, or materially incompatible.

UNJUDGEABLE:
  Despite structural extraction, the pair cannot be semantically judged
  with reasonable confidence.

IMPORTANT
---------
Do not count metadata/explanation outside extracted X',Y'.
Only judge the extracted pair.

DECISION USE
------------
This is the final PDS construct gate before full-scale generation.

No publication claim will use an arbitrary 95% threshold.

Internal GO signal:
- FULL rate clearly high (roughly around 90%+ is encouraging)
- gross FAIL rate only a few percent
- structural retention is sufficient
- observed failure modes are sparse and non-systematic

If passed:
  stop corpus diagnostics
  -> full PDS
  -> K1All+PDS vs K1RepeatMatched
"""


PROTOCOL.write_text(
    protocol,
    encoding="utf-8",
)


###############################################################################
# CONSOLE SUMMARY
###############################################################################

print("=" * 78)
print("PDS STRUCTURAL ACCEPTANCE V1")
print("=" * 78)

print()
print("RAW_OUTPUTS =", 1100)

print(
    "PARSEABLE_OUTPUTS =",
    report[
        "parseable_outputs"
    ],
)

print(
    "ACCEPTED_A_OUTPUTS =",
    report[
        "accepted_A_outputs"
    ],
)

print(
    "ACCEPTED_A_OUTPUT_RATE =",
    f"{report['accepted_A_output_rate']:.6f}",
)

print(
    "ACCEPTED_A_UNIQUE_PAIRS =",
    report[
        "accepted_A_unique_pairs"
    ],
)

print(
    "ACCEPTED_A_UNSEEN_UNIQUE_PAIRS =",
    report[
        "accepted_A_unseen_unique_pairs"
    ],
)

print()
print("===== REJECTION DECOMPOSITION =====")

for key, value in sorted(
    reject_counts.items()
):
    print(
        key,
        "=",
        value,
    )


print()
print("===== PARSE METHODS =====")

for key, value in sorted(
    method_counts.items()
):
    print(
        key,
        "=",
        value,
    )


print()
print(
    "Q_EXACT_IN_Y_RATE_AMONG_A =",
    report[
        "Q_exact_in_Y_rate_among_A_outputs"
    ],
)

print()
print(
    "PREVIOUS_AUDITED_PAIR_IDS_EXCLUDED =",
    len(
        OLD_AUDITED_PAIR_IDS
    ),
)

print(
    "CONFIRMATORY_AUDIT_ROWS =",
    audit_n,
)

print()
print(
    "BLIND =",
    BLIND,
)

print(
    "HIDDEN =",
    HIDDEN,
)

print(
    "REPORT =",
    REPORT,
)

print(
    "PROTOCOL =",
    PROTOCOL,
)

print()
print(
    "IMPORTANT: upload ONLY accepted_set_semantic_blind100_v1.csv"
)

print(
    "Do NOT inspect/upload the new hidden manifest before labels are frozen."
)

print()
print(
    "PDS_STRUCTURAL_ACCEPTANCE_V1_COMPLETE"
)
