#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path("/workspace/mtpatcher")
SRC = ROOT / "data/verl_science_broad20k"

SEQKD = SRC / "seqkd_broad20k.parquet"
OPD = SRC / "opd_broad20k.parquet"

EXPECTED_ROWS = 20_000


# Structural reasoning markers only.
# Do not reject ordinary English words such as "think".
THINK_PATTERNS = [
    re.compile(r"<think>", re.I),
    re.compile(r"</think>", re.I),
    re.compile(r"<analysis>", re.I),
    re.compile(r"</analysis>", re.I),
]


def contains_reasoning_marker(text: str) -> bool:
    return any(p.search(text) for p in THINK_PATTERNS)


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def as_list(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def main() -> None:
    seq = pq.read_table(SEQKD).to_pylist()
    opd = pq.read_table(OPD).to_pylist()

    if len(seq) != EXPECTED_ROWS:
        fail(f"SeqKD rows={len(seq)}")

    if len(opd) != EXPECTED_ROWS:
        fail(f"OPD rows={len(opd)}")

    reasoning_targets = []
    target_message_mismatch = []
    reasoning_prompts = []

    for i, row in enumerate(seq):
        target = row.get("target_translation")
        messages = as_list(row.get("messages"))

        if not isinstance(target, str) or not target.strip():
            fail(f"SeqKD row {i}: invalid target_translation")

        if not isinstance(messages, list) or len(messages) != 2:
            fail(f"SeqKD row {i}: invalid messages")

        user = messages[0]
        assistant = messages[1]

        if user.get("role") != "user":
            fail(f"SeqKD row {i}: first role != user")

        if assistant.get("role") != "assistant":
            fail(f"SeqKD row {i}: second role != assistant")

        user_text = user.get("content")
        assistant_text = assistant.get("content")

        if not isinstance(user_text, str):
            fail(f"SeqKD row {i}: invalid user content")

        if not isinstance(assistant_text, str):
            fail(f"SeqKD row {i}: invalid assistant content")

        if contains_reasoning_marker(target):
            reasoning_targets.append(("target_translation", i))

        if contains_reasoning_marker(assistant_text):
            reasoning_targets.append(("assistant", i))

        # The actual supervised response must be answer-only.
        if assistant_text.strip() != target.strip():
            target_message_mismatch.append(i)

        if contains_reasoning_marker(user_text):
            reasoning_prompts.append(("seqkd", i))

    for i, row in enumerate(opd):
        prompt = as_list(row.get("prompt"))

        if not isinstance(prompt, list) or len(prompt) != 1:
            fail(f"OPD row {i}: invalid prompt")

        msg = prompt[0]

        if msg.get("role") != "user":
            fail(f"OPD row {i}: role != user")

        content = msg.get("content")

        if not isinstance(content, str):
            fail(f"OPD row {i}: invalid prompt content")

        if contains_reasoning_marker(content):
            reasoning_prompts.append(("opd", i))

    print("SEQKD_ROWS =", len(seq))
    print("OPD_ROWS =", len(opd))
    print("REASONING_TARGET_ROWS =", len(reasoning_targets))
    print(
        "TARGET_MESSAGE_MISMATCH_ROWS =",
        len(target_message_mismatch),
    )
    print("REASONING_PROMPT_ROWS =", len(reasoning_prompts))

    if reasoning_targets:
        print("FIRST_REASONING_TARGETS =", reasoning_targets[:20])
        fail("teacher reasoning marker found in SeqKD target")

    if target_message_mismatch:
        print(
            "FIRST_TARGET_MESSAGE_MISMATCH =",
            target_message_mismatch[:20],
        )
        fail(
            "assistant message differs from target_translation"
        )

    if reasoning_prompts:
        print("FIRST_REASONING_PROMPTS =", reasoning_prompts[:20])
        fail("reasoning marker found in training prompt")

    print("NO_THINKING_DATA_CONTRACT=PASS")


if __name__ == "__main__":
    main()
