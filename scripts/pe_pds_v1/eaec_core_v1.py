#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import torch


def occurrences(text: str, sub: str) -> list[int]:
    if not sub:
        return []

    out = []
    start = 0

    while True:
        p = text.find(sub, start)

        if p < 0:
            return out

        out.append(p)
        start = p + 1


def words_with_offsets(text: str):
    return [
        (m.group(0), m.start(), m.end())
        for m in re.finditer(r"\S+", text)
    ]


def merge_intervals(
    intervals: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if not intervals:
        return []

    intervals = sorted(intervals)

    out = []
    a, b = intervals[0]

    for c, d in intervals[1:]:
        if c <= b:
            b = max(b, d)
        else:
            out.append((a, b))
            a, b = c, d

    out.append((a, b))
    return out


def derive_edit_core(
    old_span: str,
    correction: str,
) -> tuple[list[tuple[int, int]], dict[str, int]]:
    """
    Return char intervals relative to old_span.

    Replacement/deletion:
        mark the old-side words that need changing.

    Pure insertion:
        no old token represents the missing material, so mark the
        nearest causal boundary token in the old sequence.
    """

    old_words = words_with_offsets(old_span)
    new_words = words_with_offsets(correction)

    if not old_words:
        return [], {
            "insertion_boundary_ops": 0,
        }

    old_tokens = [x[0] for x in old_words]
    new_tokens = [x[0] for x in new_words]

    sm = SequenceMatcher(
        None,
        old_tokens,
        new_tokens,
        autojunk=False,
    )

    marked: set[int] = set()
    insertion_ops = 0

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue

        if i1 < i2:
            # replacement / deletion
            marked.update(
                range(i1, i2)
            )
        else:
            # pure insertion
            insertion_ops += 1

            if i1 < len(old_words):
                marked.add(i1)
            elif i1 > 0:
                marked.add(i1 - 1)

    intervals = []

    for i in sorted(marked):
        _, a, b = old_words[i]
        intervals.append((a, b))

    return merge_intervals(intervals), {
        "insertion_boundary_ops": insertion_ops,
    }


def load_patchbank(path: str | Path):
    path = Path(path)

    bank = {}

    with path.open(
        encoding="utf-8"
    ) as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            sid = int(
                row["source_id"]
            )

            if sid in bank:
                raise RuntimeError(
                    f"duplicate source_id={sid}"
                )

            bank[sid] = row

    return bank


def _dense_weights(
    response_mask: torch.Tensor,
) -> torch.Tensor:
    w = torch.zeros_like(
        response_mask,
        dtype=torch.float32,
    )

    valid = response_mask.bool()

    w[valid] = 1.0

    w.requires_grad_(False)

    return w


def _normalize_valid_weights(
    w: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    valid = response_mask.bool()

    if not bool(valid.any()):
        return w

    mean = w[valid].mean()

    if not torch.isfinite(mean):
        raise RuntimeError(
            "non-finite EAEC weight mean"
        )

    if float(mean.item()) <= 0:
        raise RuntimeError(
            "non-positive EAEC weight mean"
        )

    w = w.clone()

    w[valid] = (
        w[valid] / mean
    )

    w[~valid] = 0.0

    w = w.detach()
    w.requires_grad_(False)

    got = w[valid].float().mean()

    if not torch.allclose(
        got,
        torch.tensor(
            1.0,
            device=got.device,
        ),
        atol=1e-6,
        rtol=0,
    ):
        raise RuntimeError(
            f"EAEC mean normalization failed: {got}"
        )

    return w


def build_sample_weights(
    *,
    tokenizer,
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    patches: list[dict[str, Any]],
    core_ratio: float = 2.0,
):
    """
    Build EAEC response-token weights.

    States:
      ACTIVE:
        old_span occurs exactly once -> project edit core.

      RESOLVED:
        old_span absent and correction occurs exactly once.

      AMBIGUOUS:
        old_span occurs multiple times.

      UNLOCATED:
        neither condition above.

    Any unsafe tokenization/projection falls back to ordinary dense OPD.
    """

    if core_ratio < 1.0:
        raise ValueError(
            "core_ratio must be >= 1"
        )

    weights = _dense_weights(
        response_mask
    )

    valid_positions = (
        torch.nonzero(
            response_mask.bool(),
            as_tuple=False,
        )
        .flatten()
        .tolist()
    )

    stats = {
        "active_patches": 0,
        "resolved_patches": 0,
        "ambiguous_patches": 0,
        "unlocated_patches": 0,
        "core_tokens": 0,
        "valid_response_tokens":
            len(valid_positions),
        "tokenization_fallback": 0,
        "projection_empty": 0,
    }

    if not valid_positions:
        return weights, stats

    if not patches:
        return weights, stats

    ids = (
        response_ids
        .detach()
        .cpu()
        .tolist()
    )

    special_ids = set(
        int(x)
        for x in tokenizer.all_special_ids
    )

    content_positions = [
        p
        for p in valid_positions
        if int(ids[p])
        not in special_ids
    ]

    content_ids = [
        int(ids[p])
        for p in content_positions
    ]

    if not content_ids:
        return weights, stats

    text = tokenizer.decode(
        content_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    roundtrip_ids = list(
        encoded["input_ids"]
    )

    offsets = [
        tuple(map(int, x))
        for x in encoded[
            "offset_mapping"
        ]
    ]

    # Critical safety gate:
    # char offsets are only valid if decode -> encode exactly
    # reconstructs the current generated token IDs.
    if (
        roundtrip_ids
        != content_ids
        or len(offsets)
        != len(content_positions)
    ):
        stats[
            "tokenization_fallback"
        ] = 1

        return weights, stats

    active_char_intervals = []

    for patch in patches:
        old = patch["old_span"]
        correction = patch[
            "correction"
        ]

        old_hits = occurrences(
            text,
            old,
        )

        if len(old_hits) == 1:
            stats[
                "active_patches"
            ] += 1

            base = old_hits[0]

            for a, b in patch[
                "core_char_intervals"
            ]:
                a = int(a)
                b = int(b)

                if not (
                    0 <= a < b <= len(old)
                ):
                    raise RuntimeError(
                        "invalid PatchBank core interval: "
                        f"{a=} {b=} len(old)={len(old)}"
                    )

                active_char_intervals.append(
                    (
                        base + a,
                        base + b,
                    )
                )

            continue

        if len(old_hits) > 1:
            stats[
                "ambiguous_patches"
            ] += 1
            continue

        correction_hits = (
            occurrences(
                text,
                correction,
            )
            if correction
            else []
        )

        if len(
            correction_hits
        ) == 1:
            stats[
                "resolved_patches"
            ] += 1
        else:
            stats[
                "unlocated_patches"
            ] += 1

    if not active_char_intervals:
        return weights, stats

    active_char_intervals = (
        merge_intervals(
            active_char_intervals
        )
    )

    core_response_positions = set()

    for token_i, (
        char_a,
        char_b,
    ) in enumerate(offsets):
        if char_b <= char_a:
            continue

        overlaps = any(
            max(char_a, core_a)
            < min(char_b, core_b)
            for core_a, core_b
            in active_char_intervals
        )

        if overlaps:
            core_response_positions.add(
                content_positions[
                    token_i
                ]
            )

    if not core_response_positions:
        stats[
            "projection_empty"
        ] = 1
        return weights, stats

    for p in (
        core_response_positions
    ):
        weights[p] = float(
            core_ratio
        )

    stats["core_tokens"] = len(
        core_response_positions
    )

    weights = (
        _normalize_valid_weights(
            weights,
            response_mask,
        )
    )

    return weights, stats
