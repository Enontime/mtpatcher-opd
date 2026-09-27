#!/usr/bin/env python3

from __future__ import annotations

import math
import os

import torch
from transformers import (
    AutoTokenizer,
)

from scripts.pe_pds_v1.eaec_core_v1 import (
    build_sample_weights,
    derive_edit_core,
)


MODEL = os.environ.get(
    "EAEC_TEST_MODEL",
    "/workspace/mtpatcher/models/Qwen3-0.6B",
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    local_files_only=True,
    trust_remote_code=True,
)


old = (
    "the market has already been in large "
    "quantities of procurement from the source"
)

correction = (
    "the market has already started large-scale "
    "procurement from the source"
)

core, extra = derive_edit_core(
    old,
    correction,
)

assert core
assert extra[
    "insertion_boundary_ops"
] >= 0


patch = {
    "old_span": old,
    "correction":
        correction,
    "error_type":
        "Grammar",
    "core_char_intervals":
        [
            [a, b]
            for a, b in core
        ],
}


def tensors(text, width=128):
    ids = tokenizer(
        text,
        add_special_tokens=False,
    )["input_ids"]

    eos = tokenizer.eos_token_id

    if eos is not None:
        ids = ids + [eos]

    assert len(ids) < width

    response = torch.full(
        (width,),
        tokenizer.pad_token_id,
        dtype=torch.long,
    )

    mask = torch.zeros(
        (width,),
        dtype=torch.long,
    )

    response[:len(ids)] = (
        torch.tensor(
            ids,
            dtype=torch.long,
        )
    )

    mask[:len(ids)] = 1

    return response, mask


# ACTIVE
response, mask = tensors(old)

w, s = build_sample_weights(
    tokenizer=tokenizer,
    response_ids=response,
    response_mask=mask,
    patches=[patch],
    core_ratio=2.0,
)

valid = mask.bool()

assert s["active_patches"] == 1
assert s["core_tokens"] > 0
assert s["tokenization_fallback"] == 0
assert torch.all(w[~valid] == 0)

assert math.isclose(
    float(w[valid].mean()),
    1.0,
    abs_tol=1e-6,
)

unique = sorted(
    set(
        round(
            float(x),
            6,
        )
        for x in w[valid]
    )
)

assert len(unique) == 2, unique

assert math.isclose(
    unique[1] / unique[0],
    2.0,
    rel_tol=1e-5,
)


# RESOLVED
response, mask = tensors(
    correction
)

w, s = build_sample_weights(
    tokenizer=tokenizer,
    response_ids=response,
    response_mask=mask,
    patches=[patch],
    core_ratio=2.0,
)

assert s[
    "resolved_patches"
] == 1

assert s[
    "active_patches"
] == 0

assert torch.allclose(
    w[mask.bool()],
    torch.ones_like(
        w[mask.bool()]
    ),
)


# AMBIGUOUS
response, mask = tensors(
    old + " ; " + old
)

w, s = build_sample_weights(
    tokenizer=tokenizer,
    response_ids=response,
    response_mask=mask,
    patches=[patch],
    core_ratio=2.0,
)

assert s[
    "ambiguous_patches"
] == 1

assert s[
    "active_patches"
] == 0


# UNLOCATED
response, mask = tensors(
    "a completely different translation"
)

w, s = build_sample_weights(
    tokenizer=tokenizer,
    response_ids=response,
    response_mask=mask,
    patches=[patch],
    core_ratio=2.0,
)

assert s[
    "unlocated_patches"
] == 1

assert s[
    "active_patches"
] == 0


# EMPTY PATCH BANK = exact dense OPD
response, mask = tensors(old)

w, s = build_sample_weights(
    tokenizer=tokenizer,
    response_ids=response,
    response_mask=mask,
    patches=[],
    core_ratio=2.0,
)

assert torch.allclose(
    w[mask.bool()],
    torch.ones_like(
        w[mask.bool()]
    ),
)

assert torch.all(
    w[~mask.bool()] == 0
)


print(
    "PASS_EAEC_CORE_V1_UNIT_TESTS"
)
