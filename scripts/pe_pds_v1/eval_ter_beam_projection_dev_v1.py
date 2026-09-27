#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from statistics import mean

from sacrebleu.metrics import TER
from sacrebleu.metrics.lib_ter import BeamEditDistance


SEED = 20260923

EXPECTED_GOLD_SHA = (
    "021c20ec6075ee6794ba3078c8f905dd"
    "37d6556a311868652a3bb32217462340"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def exact_occurrences(
    text: str,
    sub: str,
) -> list[tuple[int, int]]:
    if not sub:
        return []

    out = []
    start = 0

    while True:
        pos = text.find(
            sub,
            start,
        )

        if pos < 0:
            break

        out.append(
            (
                pos,
                pos + len(sub),
            )
        )

        start = pos + 1

    return out


def whitespace_tokens_with_offsets(
    text: str,
) -> list[tuple[str, int, int]]:
    return [
        (
            m.group(0),
            m.start(),
            m.end(),
        )
        for m in re.finditer(
            r"\S+",
            text,
        )
    ]


def ter_tokens_with_offsets(
    text: str,
    ter: TER,
):
    """
    Default SacreBLEU TER tokenizer in this runtime is case-insensitive
    and, with default options, preserves whitespace token boundaries.

    We hard-gate that assumption for every sentence instead of silently
    approximating the tokenizer.
    """
    original = whitespace_tokens_with_offsets(
        text
    )

    ter_tokens = (
        ter.tokenizer(text)
        .split()
    )

    simple_normalized = [
        token.lower()
        for token, _, _ in original
    ]

    if simple_normalized != ter_tokens:
        return None

    return [
        {
            "text": token,
            "norm": norm,
            "start": start,
            "end": end,
        }
        for (
            (token, start, end),
            norm,
        ) in zip(
            original,
            ter_tokens,
        )
    ]


def historical_patch_token_indices(
    tokens,
    anchor_start: int,
    anchor_end: int,
) -> list[int]:
    out = []

    for i, tok in enumerate(tokens):
        if (
            tok["start"] < anchor_end
            and tok["end"] > anchor_start
        ):
            out.append(i)

    return out


def boundary_char(
    current_text: str,
    current_tokens,
    consumed_current_tokens: int,
) -> int:
    if consumed_current_tokens <= 0:
        return 0

    if consumed_current_tokens >= len(
        current_tokens
    ):
        return len(
            current_text
        )

    return int(
        current_tokens[
            consumed_current_tokens
        ]["start"]
    )


def parse_trace(
    trace: str,
    current_text: str,
    current_tokens,
):
    """
    Parse the RAW trace returned directly by:

        BeamEditDistance(
            historical_tokens
        )(
            current_tokens
        )

    Matrix orientation in SacreBLEU lib_ter:

        reference axis j  = historical translation
        hypothesis axis i = current translation

    Therefore RAW trace semantics are:

      ' ' / 's':
          consume historical + current

      'i':
          consume HISTORICAL/reference only
          (historical token has no current counterpart)

      'd':
          consume CURRENT/hypothesis only
          (current contains an inserted token)

    Important:
    lib_ter.trace_to_alignment() uses the inverse/flipped
    interpretation and must NOT be copied directly onto the
    raw BeamEditDistance trace.
    """
    ref_idx = 0
    hyp_idx = 0

    ref_to_hyp = {}
    ref_deleted_boundary = {}

    for op in trace:
        if op == " " or op == "s":
            ref_to_hyp[
                ref_idx
            ] = hyp_idx

            ref_idx += 1
            hyp_idx += 1

        elif op == "i":
            # RAW BeamEditDistance:
            # j -= 1, so historical/reference token is consumed
            # without consuming a current/hypothesis token.
            ref_deleted_boundary[
                ref_idx
            ] = boundary_char(
                current_text,
                current_tokens,
                hyp_idx,
            )

            ref_idx += 1

        elif op == "d":
            # RAW BeamEditDistance:
            # i -= 1, so current/hypothesis token is consumed
            # without consuming a historical/reference token.
            hyp_idx += 1

        else:
            raise RuntimeError(
                f"unknown trace op {op!r}"
            )

    return (
        ref_to_hyp,
        ref_deleted_boundary,
        ref_idx,
        hyp_idx,
    )


def consecutive_groups(
    xs: list[int],
) -> list[list[int]]:
    if not xs:
        return []

    xs = sorted(set(xs))

    groups = [
        [xs[0]]
    ]

    for x in xs[1:]:
        if x == groups[-1][-1] + 1:
            groups[-1].append(x)
        else:
            groups.append([x])

    return groups


def project_one(
    row,
    ter: TER,
):
    historical = row[
        "historical_student_translation"
    ]

    current = row[
        "current_translation"
    ]

    old_span = row[
        "translation_span"
    ]

    occurrences = exact_occurrences(
        historical,
        old_span,
    )

    if len(occurrences) == 0:
        return {
            "state":
                "HISTORICAL_ANCHOR_MISSING",
        }

    if len(occurrences) > 1:
        return {
            "state":
                "HISTORICAL_ANCHOR_AMBIGUOUS",
            "anchor_occurrences":
                len(occurrences),
        }

    historical_tokens = (
        ter_tokens_with_offsets(
            historical,
            ter,
        )
    )

    current_tokens = (
        ter_tokens_with_offsets(
            current,
            ter,
        )
    )

    if historical_tokens is None:
        return {
            "state":
                "HISTORICAL_TOKENIZER_CONTRACT_MISMATCH",
        }

    if current_tokens is None:
        return {
            "state":
                "CURRENT_TOKENIZER_CONTRACT_MISMATCH",
        }

    (
        anchor_start,
        anchor_end,
    ) = occurrences[0]

    patch_token_indices = (
        historical_patch_token_indices(
            historical_tokens,
            anchor_start,
            anchor_end,
        )
    )

    if not patch_token_indices:
        return {
            "state":
                "HISTORICAL_PATCH_TOKEN_EMPTY",
        }

    hist_norm = [
        x["norm"]
        for x in historical_tokens
    ]

    curr_norm = [
        x["norm"]
        for x in current_tokens
    ]

    ed = BeamEditDistance(
        hist_norm
    )

    distance, trace = ed(
        curr_norm
    )

    (
        ref_to_hyp,
        deleted_boundary,
        final_ref,
        final_hyp,
    ) = parse_trace(
        trace,
        current,
        current_tokens,
    )

    if final_ref != len(
        historical_tokens
    ):
        raise RuntimeError(
            (
                "trace reference length mismatch "
                f"{final_ref} != "
                f"{len(historical_tokens)}"
            )
        )

    if final_hyp != len(
        current_tokens
    ):
        raise RuntimeError(
            (
                "trace hypothesis length mismatch "
                f"{final_hyp} != "
                f"{len(current_tokens)}"
            )
        )

    mapped_current = [
        ref_to_hyp[i]
        for i in patch_token_indices
        if i in ref_to_hyp
    ]

    deleted_boundaries = [
        deleted_boundary[i]
        for i in patch_token_indices
        if i in deleted_boundary
    ]

    if mapped_current:
        groups = consecutive_groups(
            mapped_current
        )

        spans = []

        for group in groups:
            first = current_tokens[
                group[0]
            ]

            last = current_tokens[
                group[-1]
            ]

            spans.append(
                [
                    int(first["start"]),
                    int(last["end"]),
                ]
            )

        return {
            "state":
                "PROJECTED_SPAN",

            "pred_spans":
                spans,

            "edit_distance":
                int(distance),

            "historical_patch_tokens":
                len(
                    patch_token_indices
                ),

            "mapped_patch_tokens":
                len(
                    mapped_current
                ),

            "deleted_patch_tokens":
                len(
                    deleted_boundaries
                ),

            "trace":
                trace,
        }

    if deleted_boundaries:
        boundaries = sorted(
            set(
                int(x)
                for x in deleted_boundaries
            )
        )

        return {
            "state":
                "PROJECTED_BOUNDARY",

            "pred_spans":
                [
                    [x, x]
                    for x in boundaries
                ],

            "edit_distance":
                int(distance),

            "historical_patch_tokens":
                len(
                    patch_token_indices
                ),

            "mapped_patch_tokens":
                0,

            "deleted_patch_tokens":
                len(
                    deleted_boundaries
                ),

            "trace":
                trace,
        }

    return {
        "state":
            "UNLOCATED_AFTER_ALIGNMENT",

        "edit_distance":
            int(distance),

        "trace":
            trace,
    }


def char_union(
    spans,
):
    out = set()

    for start, end in spans:
        if start == end:
            continue

        out.update(
            range(
                int(start),
                int(end),
            )
        )

    return out


def span_iou_f1(
    gold_spans,
    pred_spans,
):
    g = char_union(
        gold_spans
    )

    p = char_union(
        pred_spans
    )

    if not g and not p:
        return 1.0, 1.0

    if not g or not p:
        return 0.0, 0.0

    inter = len(
        g & p
    )

    union = len(
        g | p
    )

    iou = (
        inter / union
        if union
        else 0.0
    )

    f1 = (
        2 * inter
        / (len(g) + len(p))
    )

    return iou, f1


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--gold",
        required=True,
    )

    ap.add_argument(
        "--out-dir",
        required=True,
    )

    args = ap.parse_args()

    gold_path = Path(
        args.gold
    )

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    got_sha = sha256(
        gold_path
    )

    if got_sha != EXPECTED_GOLD_SHA:
        raise RuntimeError(
            (
                "gold SHA mismatch "
                f"got={got_sha} "
                f"expected={EXPECTED_GOLD_SHA}"
            )
        )

    rows = [
        json.loads(line)
        for line in (
            gold_path
            .read_text(
                encoding="utf-8"
            )
            .splitlines()
        )
        if line.strip()
    ]

    source_ids = sorted({
        int(x["source_id"])
        for x in rows
    })

    rng = random.Random(
        SEED
    )

    rng.shuffle(
        source_ids
    )

    dev_sources = set(
        source_ids[:48]
    )

    dev = [
        x
        for x in rows
        if (
            int(x["source_id"])
            in dev_sources
            and x["gold"]["status"]
            != "INVALID_PATCH"
        )
    ]

    counts = Counter(
        x["gold"]["status"]
        for x in dev
    )

    if len(dev) != 89:
        raise RuntimeError(
            f"expected 89 valid DEV items, got {len(dev)}"
        )

    if counts["PRESENT"] != 60:
        raise RuntimeError(
            f"expected 60 DEV PRESENT, got {counts['PRESENT']}"
        )

    if counts["RESOLVED"] != 29:
        raise RuntimeError(
            f"expected 29 DEV RESOLVED, got {counts['RESOLVED']}"
        )

    ter = TER()

    predictions = []

    for row in dev:
        pred = project_one(
            row,
            ter,
        )

        predictions.append({
            "benchmark_id":
                row["benchmark_id"],

            "source_id":
                int(row["source_id"]),

            "error_type":
                row["error_type"],

            "gold_status":
                row["gold"]["status"],

            "gold_span_type":
                row["gold"]["span_type"],

            "gold_spans":
                row["gold"][
                    "current_spans"
                ],

            **pred,
        })

    pred_path = (
        out_dir
        / "predictions.jsonl"
    )

    with pred_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for x in predictions:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    states = Counter(
        x["state"]
        for x in predictions
    )

    present = [
        x
        for x in predictions
        if x["gold_status"]
        == "PRESENT"
    ]

    present_span_gold = [
        x
        for x in present
        if x["gold_span_type"]
        in (
            "TOKEN_SPAN",
            "MULTI_SPAN",
        )
    ]

    present_boundary_gold = [
        x
        for x in present
        if x["gold_span_type"]
        == "INSERTION_BOUNDARY"
    ]

    span_rows = []

    for x in present_span_gold:
        if x["state"] != "PROJECTED_SPAN":
            iou = 0.0
            f1 = 0.0
        else:
            iou, f1 = span_iou_f1(
                x["gold_spans"],
                x["pred_spans"],
            )

        span_rows.append(
            (
                x,
                iou,
                f1,
            )
        )

    projected_span_rows = [
        z
        for z in span_rows
        if z[0]["state"]
        == "PROJECTED_SPAN"
    ]

    boundary_exact = 0
    boundary_with_prediction = 0
    boundary_distances = []

    for x in present_boundary_gold:
        gold_boundary = int(
            x["gold_spans"][0][0]
        )

        if x["state"] == "PROJECTED_BOUNDARY":
            preds = [
                int(a)
                for a, b in x[
                    "pred_spans"
                ]
                if a == b
            ]

            if preds:
                boundary_with_prediction += 1

                d = min(
                    abs(
                        p
                        - gold_boundary
                    )
                    for p in preds
                )

                boundary_distances.append(
                    d
                )

                if d == 0:
                    boundary_exact += 1

    summary = {
        "method":
            "sacrebleu_ter_tokenizer_plus_"
            "beam_edit_distance_monotonic_"
            "historical_to_current_projection",

        "task":
            "localization_projection_"
            "conditioned_on_gold_present",

        "full_TER_shifts_used":
            False,

        "split":
            "DEV_ONLY",

        "split_seed":
            SEED,

        "dev_source_count":
            48,

        "dev_valid_items":
            len(dev),

        "dev_present":
            counts["PRESENT"],

        "dev_resolved":
            counts["RESOLVED"],

        "prediction_state_counts":
            dict(
                sorted(
                    states.items()
                )
            ),

        "present_projection_coverage":
            sum(
                x["state"]
                in (
                    "PROJECTED_SPAN",
                    "PROJECTED_BOUNDARY",
                )
                for x in present
            )
            / len(present),

        "span_gold_items":
            len(
                present_span_gold
            ),

        "span_projected_items":
            len(
                projected_span_rows
            ),

        "span_projection_coverage":
            len(
                projected_span_rows
            )
            / max(
                1,
                len(
                    present_span_gold
                ),
            ),

        "span_mean_char_iou_all":
            mean(
                iou
                for _, iou, _
                in span_rows
            )
            if span_rows
            else None,

        "span_mean_char_f1_all":
            mean(
                f1
                for _, _, f1
                in span_rows
            )
            if span_rows
            else None,

        "span_mean_char_iou_projected":
            mean(
                iou
                for _, iou, _
                in projected_span_rows
            )
            if projected_span_rows
            else None,

        "span_mean_char_f1_projected":
            mean(
                f1
                for _, _, f1
                in projected_span_rows
            )
            if projected_span_rows
            else None,

        "span_overlap_hit_rate_all":
            sum(
                iou > 0
                for _, iou, _
                in span_rows
            )
            / max(
                1,
                len(span_rows),
            ),

        "span_iou_ge_0_5_rate_all":
            sum(
                iou >= 0.5
                for _, iou, _
                in span_rows
            )
            / max(
                1,
                len(span_rows),
            ),

        "insertion_gold_items":
            len(
                present_boundary_gold
            ),

        "insertion_boundary_predictions":
            boundary_with_prediction,

        "insertion_boundary_exact":
            boundary_exact,

        "insertion_boundary_mean_abs_char_error":
            mean(
                boundary_distances
            )
            if boundary_distances
            else None,

        "note":
            (
                "This method is evaluated only as a "
                "historical-to-current span projector. "
                "It does not claim to decide PRESENT "
                "versus RESOLVED."
            ),
    }

    summary_path = (
        out_dir
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print(
        "TER_BEAM_PROJECTION_DEV_V1=PASS"
    )


if __name__ == "__main__":
    main()
