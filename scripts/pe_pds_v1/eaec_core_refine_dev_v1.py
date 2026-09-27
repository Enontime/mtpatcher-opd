#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


# ---------------------------------------------------------------------
# DEV labels from the 100-case manual localization audit.
#
# IMPORTANT:
# These labels are for heuristic development only.
# They MUST NOT be reported later as held-out precision.
# ---------------------------------------------------------------------

GOOD = {
    2, 4, 6, 8, 9, 12, 14, 15, 16, 19,
    20, 21, 22, 24, 25, 27, 29, 35, 36, 37,
    40, 41, 44, 45, 48, 49, 52, 55, 56, 57,
    59, 61, 63, 64, 65, 67, 70, 71, 74, 75,
    77, 78, 79, 80, 81, 83, 84, 85, 86, 88,
    89, 90, 91, 92, 93, 94, 95, 96, 97, 99,
    100,
}

BROAD = {
    1, 3, 5, 7, 10, 13, 17, 23, 28, 30,
    31, 32, 33, 34, 38, 39, 42, 43, 46, 50,
    51, 53, 54, 58, 60, 62, 66, 68, 69, 72,
    73, 76, 82, 87, 98,
}

WRONG = {
    11, 18, 26, 47,
}


STOP = set("""
the a an and or of to in on at for from with by as
is are was were be been being this that these those
it its their his her them they he she we you i but
if then than so such into over under before after
during through about out up down no not only even
more most very also still when where who which while
will would can could should may might do does did
have has had said says stated according report reports
people china chinese year years
""".split())


WORD_RE = re.compile(
    r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[\u4e00-\u9fff]+"
)


def words_with_offsets(text: str):
    return [
        (
            m.group(0),
            m.start(),
            m.end(),
        )
        for m in WORD_RE.finditer(text)
    ]


def norm(x: str) -> str:
    return x.lower()


def content_query_weights(
    old_span: str,
    correction: str,
):
    old = [
        norm(x[0])
        for x in words_with_offsets(old_span)
    ]

    corr = [
        norm(x[0])
        for x in words_with_offsets(correction)
    ]

    sm = SequenceMatcher(
        None,
        old,
        corr,
        autojunk=False,
    )

    changed_old = set()
    changed_corr = set()

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue

        changed_old.update(
            old[i1:i2]
        )

        changed_corr.update(
            corr[j1:j2]
        )

    weights = {}

    for token in set(old + corr):
        if (
            token in STOP
            or len(token) < 3
        ):
            continue

        w = 1.0

        # Terms changed by MT-PATCHER's correction
        # carry stronger localization evidence.
        if (
            token in changed_old
            or token in changed_corr
        ):
            w += 1.0

        # Numbers are often highly discriminative
        # in translation patches.
        if any(ch.isdigit() for ch in token):
            w += 1.0

        weights[token] = w

    return weights


def weighted_unique_score(
    matches,
):
    best = {}

    for x in matches:
        token = x["token"]
        weight = x["weight"]

        best[token] = max(
            best.get(token, 0.0),
            weight,
        )

    return sum(
        best.values()
    )


def make_clusters(
    matches,
    max_non_support_gap: int = 3,
):
    if not matches:
        return []

    clusters = [
        [matches[0]]
    ]

    for item in matches[1:]:
        prev = clusters[-1][-1]

        token_distance = (
            item["token_index"]
            - prev["token_index"]
        )

        # distance <= 4 means at most 3
        # non-support words between them.
        if (
            token_distance
            <= max_non_support_gap + 1
        ):
            clusters[-1].append(
                item
            )
        else:
            clusters.append(
                [item]
            )

    return clusters


def cluster_score(cluster):
    unique_score = (
        weighted_unique_score(cluster)
    )

    span_words = (
        cluster[-1]["token_index"]
        - cluster[0]["token_index"]
        + 1
    )

    # Reward support density slightly;
    # penalize unnecessarily wide clusters.
    return (
        unique_score
        + 0.25 * len(cluster)
        - 0.05 * span_words
    )


def classify_label(index: int):
    if index in GOOD:
        return "GOOD"

    if index in BROAD:
        return "BROAD"

    if index in WRONG:
        return "WRONG"

    return "UNKNOWN"


def refine_case(row: dict):
    current = row[
        "current_translation"
    ]

    coarse = row[
        "projected_current_region"
    ]

    occurrences = []

    start = 0

    while True:
        pos = current.find(
            coarse,
            start,
        )

        if pos < 0:
            break

        occurrences.append(pos)

        start = pos + 1

    if len(occurrences) != 1:
        return {
            "status":
                "REJECT_COARSE_NOT_UNIQUE",
            "reason":
                f"coarse_occurrences={len(occurrences)}",
        }

    coarse_a = occurrences[0]
    coarse_b = (
        coarse_a + len(coarse)
    )

    words = words_with_offsets(
        current
    )

    query_weights = (
        content_query_weights(
            row["old_span"],
            row["correction"],
        )
    )

    inside = []
    outside = []

    for token_index, (
        token,
        char_a,
        char_b,
    ) in enumerate(words):

        token_norm = norm(token)

        if token_norm not in query_weights:
            continue

        item = {
            "token_index":
                token_index,
            "token":
                token_norm,
            "weight":
                query_weights[
                    token_norm
                ],
            "char_a":
                char_a,
            "char_b":
                char_b,
        }

        if (
            char_a >= coarse_a
            and char_b <= coarse_b
        ):
            inside.append(item)

        else:
            outside.append(item)

    inside_unique = len({
        x["token"]
        for x in inside
    })

    outside_unique = len({
        x["token"]
        for x in outside
    })

    inside_score = (
        weighted_unique_score(
            inside
        )
    )

    outside_score = (
        weighted_unique_score(
            outside
        )
    )

    base = {
        "inside_support_unique":
            inside_unique,
        "outside_support_unique":
            outside_unique,
        "inside_support_score":
            inside_score,
        "outside_support_score":
            outside_score,
    }

    # ---------------------------------------------------------
    # Safety veto:
    #
    # If coarse region carries almost no patch-specific lexical
    # evidence while stronger evidence exists elsewhere in the
    # same current trajectory, do not trust this projection.
    #
    # We do NOT automatically re-anchor elsewhere in v1.
    # ---------------------------------------------------------

    if (
        inside_unique <= 1
        and outside_score > inside_score
    ):
        return {
            **base,
            "status":
                "REJECT_OUTSIDE_STRONGER",
        }

    # Conservative aligned-active path requires at least
    # two distinct informative support terms.
    if inside_unique < 2:
        return {
            **base,
            "status":
                "REJECT_LOW_SUPPORT",
        }

    clusters = make_clusters(
        inside
    )

    if not clusters:
        return {
            **base,
            "status":
                "REJECT_NO_CLUSTER",
        }

    cluster = max(
        clusters,
        key=cluster_score,
    )

    # Add two local words on each side, but never escape
    # the coarse two-sided-alignment proposal.
    first_idx = (
        cluster[0][
            "token_index"
        ]
    )

    last_idx = (
        cluster[-1][
            "token_index"
        ]
    )

    start_idx = max(
        0,
        first_idx - 2,
    )

    end_idx = min(
        len(words),
        last_idx + 3,
    )

    while (
        start_idx < end_idx
        and words[start_idx][1]
        < coarse_a
    ):
        start_idx += 1

    while (
        end_idx > start_idx
        and words[end_idx - 1][2]
        > coarse_b
    ):
        end_idx -= 1

    if start_idx >= end_idx:
        return {
            **base,
            "status":
                "REJECT_EMPTY_REFINEMENT",
        }

    refined_a = (
        words[start_idx][1]
    )

    refined_b = (
        words[end_idx - 1][2]
    )

    refined = current[
        refined_a:refined_b
    ]

    coarse_words = len(
        words_with_offsets(coarse)
    )

    refined_words = len(
        words_with_offsets(refined)
    )

    return {
        **base,

        "status":
            "ALIGNED_CORE_CANDIDATE",

        "refined_region":
            refined,

        "refined_word_count":
            refined_words,

        "coarse_word_count":
            coarse_words,

        "shrink_ratio":
            (
                refined_words
                / max(
                    1,
                    coarse_words,
                )
            ),

        "support_cluster_terms":
            [
                x["token"]
                for x in cluster
            ],
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        required=True,
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    args = ap.parse_args()

    input_path = Path(
        args.input
    )

    output_path = Path(
        args.output
    )

    rows = [
        json.loads(line)
        for line in input_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(rows) != 100:
        raise RuntimeError(
            "DEV audit expects exactly "
            f"100 rows, got {len(rows)}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for i, row in enumerate(
        rows,
        1,
    ):
        result = refine_case(
            row
        )

        result = {
            "dev_index": i,
            "dev_label":
                classify_label(i),

            "source_id":
                row["source_id"],

            "error_type":
                row["error_type"],

            "old_span":
                row["old_span"],

            "correction":
                row["correction"],

            "coarse_region":
                row[
                    "projected_current_region"
                ],

            **result,
        }

        results.append(
            result
        )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in results:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    counts = Counter(
        (
            row["dev_label"],
            row["status"],
        )
        for row in results
    )

    accepted = [
        row
        for row in results
        if (
            row["status"]
            == "ALIGNED_CORE_CANDIDATE"
        )
    ]

    shrink = sorted(
        row["shrink_ratio"]
        for row in accepted
    )

    def median(xs):
        if not xs:
            return None

        n = len(xs)

        if n % 2:
            return xs[n // 2]

        return (
            xs[n // 2 - 1]
            + xs[n // 2]
        ) / 2

    print(
        "DEV_ROWS =",
        len(results),
    )

    print(
        "DEV_LABEL_COUNTS =",
        dict(
            Counter(
                row["dev_label"]
                for row in results
            )
        ),
    )

    print()

    for label in [
        "GOOD",
        "BROAD",
        "WRONG",
    ]:
        label_rows = [
            row
            for row in results
            if (
                row["dev_label"]
                == label
            )
        ]

        accepted_n = sum(
            row["status"]
            == "ALIGNED_CORE_CANDIDATE"
            for row in label_rows
        )

        print(
            f"{label}_ACCEPTED =",
            accepted_n,
            "/",
            len(label_rows),
        )

    print()

    print(
        "TOTAL_ALIGNED_CORE_CANDIDATES =",
        len(accepted),
    )

    print(
        "MEDIAN_SHRINK_RATIO =",
        median(shrink),
    )

    print()

    print(
        "STATUS_COUNTS =",
        dict(
            Counter(
                row["status"]
                for row in results
            )
        ),
    )

    print()

    print(
        "WRONG_ACCEPTED_INDICES =",
        [
            row["dev_index"]
            for row in results
            if (
                row["dev_label"]
                == "WRONG"
                and row["status"]
                == "ALIGNED_CORE_CANDIDATE"
            )
        ],
    )

    print()

    print(
        "EAEC_CORE_REFINE_DEV_V1=PASS"
    )


if __name__ == "__main__":
    main()
