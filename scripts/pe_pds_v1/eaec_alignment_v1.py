#!/usr/bin/env python3
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

import torch

from scripts.pe_pds_v1.eaec_core_v1 import (
    occurrences,
    words_with_offsets,
)


def _subseq_count(
    seq: list[str],
    pat: list[str],
) -> int:
    if not pat or len(pat) > len(seq):
        return 0

    n = len(pat)

    return sum(
        seq[i:i+n] == pat
        for i in range(
            len(seq) - n + 1
        )
    )


def _word_range_for_chars(
    words,
    char_a: int,
    char_b: int,
):
    ids = []

    for i, (_, a, b) in enumerate(words):
        if max(a, char_a) < min(b, char_b):
            ids.append(i)

    if not ids:
        return None

    return min(ids), max(ids) + 1


def decode_response_text(
    tokenizer,
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> str:
    ids = (
        response_ids
        .detach()
        .cpu()
        .tolist()
    )

    valid = (
        response_mask
        .detach()
        .cpu()
        .bool()
        .tolist()
    )

    special = set(
        int(x)
        for x in tokenizer.all_special_ids
    )

    content_ids = [
        int(tok)
        for tok, keep in zip(ids, valid)
        if keep and int(tok) not in special
    ]

    return tokenizer.decode(
        content_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def strict_two_sided_projection(
    *,
    base_text: str,
    current_text: str,
    old_span: str,
    min_anchor_words: int = 2,
    max_historical_gap: int = 8,
    max_current_gap_floor: int = 12,
    max_current_gap_multiplier: int = 3,
) -> dict[str, Any]:
    """
    Diagnostic-only trajectory projection.

    Historical full Student translation:
        ... left anchor [old patch] right anchor ...

    Current Student translation:
        ... same left anchor [candidate region] same right anchor ...

    Requirements:
      * old_span occurs exactly once in historical translation;
      * exact matching blocks exist on BOTH sides;
      * each anchor has >= min_anchor_words;
      * anchors are near the historical patch;
      * anchor token sequences are unique in both trajectories;
      * current projected gap is non-empty and bounded.

    This only proposes a location.
    It does NOT declare the translation semantically wrong.
    """

    hits = occurrences(
        base_text,
        old_span,
    )

    if len(hits) != 1:
        return {
            "recovered": False,
            "reason": "historical_anchor_not_unique",
        }

    old_char_a = hits[0]
    old_char_b = (
        old_char_a + len(old_span)
    )

    base_words = words_with_offsets(
        base_text
    )
    current_words = words_with_offsets(
        current_text
    )

    if not base_words or not current_words:
        return {
            "recovered": False,
            "reason": "empty_word_sequence",
        }

    old_range = _word_range_for_chars(
        base_words,
        old_char_a,
        old_char_b,
    )

    if old_range is None:
        return {
            "recovered": False,
            "reason": "historical_word_projection_empty",
        }

    old_i1, old_i2 = old_range

    base_tokens = [
        x[0]
        for x in base_words
    ]

    current_tokens = [
        x[0]
        for x in current_words
    ]

    sm = SequenceMatcher(
        None,
        base_tokens,
        current_tokens,
        autojunk=False,
    )

    blocks = [
        b
        for b in sm.get_matching_blocks()
        if b.size > 0
    ]

    left_candidates = [
        b
        for b in blocks
        if (
            b.size >= min_anchor_words
            and b.a + b.size <= old_i1
            and (
                old_i1
                - (b.a + b.size)
                <= max_historical_gap
            )
        )
    ]

    if not left_candidates:
        return {
            "recovered": False,
            "reason": "no_left_anchor",
        }

    right_candidates = [
        b
        for b in blocks
        if (
            b.size >= min_anchor_words
            and b.a >= old_i2
            and (
                b.a - old_i2
                <= max_historical_gap
            )
        )
    ]

    if not right_candidates:
        return {
            "recovered": False,
            "reason": "no_right_anchor",
        }

    left = max(
        left_candidates,
        key=lambda b:
            b.a + b.size,
    )

    right = min(
        right_candidates,
        key=lambda b: b.a,
    )

    # Use up to 3 words nearest the patch as the actual
    # uniqueness-checked context anchors.
    left_k = min(
        3,
        left.size,
    )

    right_k = min(
        3,
        right.size,
    )

    left_anchor = base_tokens[
        left.a + left.size - left_k:
        left.a + left.size
    ]

    right_anchor = base_tokens[
        right.a:
        right.a + right_k
    ]

    if (
        _subseq_count(
            base_tokens,
            left_anchor,
        ) != 1
        or _subseq_count(
            current_tokens,
            left_anchor,
        ) != 1
    ):
        return {
            "recovered": False,
            "reason": "left_anchor_not_unique",
        }

    if (
        _subseq_count(
            base_tokens,
            right_anchor,
        ) != 1
        or _subseq_count(
            current_tokens,
            right_anchor,
        ) != 1
    ):
        return {
            "recovered": False,
            "reason": "right_anchor_not_unique",
        }

    current_j1 = (
        left.b + left.size
    )

    current_j2 = right.b

    if current_j2 <= current_j1:
        return {
            "recovered": False,
            "reason": "empty_or_reversed_current_region",
        }

    historical_patch_words = max(
        1,
        old_i2 - old_i1,
    )

    current_gap_words = (
        current_j2 - current_j1
    )

    allowed_current_gap = max(
        max_current_gap_floor,
        max_current_gap_multiplier
        * historical_patch_words,
    )

    if current_gap_words > allowed_current_gap:
        return {
            "recovered": False,
            "reason": "current_region_too_wide",
        }

    char_a = current_words[
        current_j1
    ][1]

    char_b = current_words[
        current_j2 - 1
    ][2]

    return {
        "recovered": True,
        "reason": "strict_two_sided_anchor",
        "current_char_interval": [
            int(char_a),
            int(char_b),
        ],
        "current_region":
            current_text[char_a:char_b],
        "left_anchor":
            " ".join(left_anchor),
        "right_anchor":
            " ".join(right_anchor),
        "historical_patch_words":
            historical_patch_words,
        "current_region_words":
            current_gap_words,
    }


def classify_and_try_alignment(
    *,
    base_text: str,
    current_text: str,
    patch: dict[str, Any],
) -> dict[str, Any]:
    old = patch["old_span"]
    correction = patch["correction"]

    old_hits = occurrences(
        current_text,
        old,
    )

    if len(old_hits) == 1:
        return {
            "state": "ACTIVE",
            "recovered": False,
            "reason": "exact_active",
        }

    if len(old_hits) > 1:
        return {
            "state": "AMBIGUOUS",
            "recovered": False,
            "reason": "old_span_multiple",
        }

    correction_hits = (
        occurrences(
            current_text,
            correction,
        )
        if correction
        else []
    )

    if len(correction_hits) == 1:
        return {
            "state": "RESOLVED",
            "recovered": False,
            "reason": "correction_exact_present",
        }

    projection = (
        strict_two_sided_projection(
            base_text=base_text,
            current_text=current_text,
            old_span=old,
        )
    )

    return {
        "state": "UNLOCATED",
        **projection,
    }
