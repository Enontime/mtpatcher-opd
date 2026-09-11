#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer


DATA_ROOT = Path("/workspace/mtpatcher/data")
MODEL = Path("/workspace/mtpatcher/models/Qwen3-0.6B")

SOURCE_UNIVERSE = DATA_ROOT / (
    "mtpatcher_v3_full6565_20260823/"
    "strong_repro_broad20k_seqkd_same_source_v1/"
    "teacher_input_broad20k_v1.jsonl"
)

OUT = DATA_ROOT / "verl_science_broad20k"

SEQKD_PARQUET = OUT / "seqkd_broad20k.parquet"
OPD_PARQUET = OUT / "opd_broad20k.parquet"
MANIFEST = OUT / "canonical_broad20k_assets_v1.json"

EXPECTED_SOURCE_SHA = (
    "c37543001b4d2e4330f638e7a523b7ac"
    "2e81f78c47bdf8781dc2cac7d3a1252c"
)

EXPECTED_ROWS = 20_000

PROMPT_TEMPLATE = (
    "Translate the following text into English "
    "without additional explanations:\n\n"
    "{source}\n\n"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path):
    rows = []

    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue

            obj = json.loads(line)

            if not isinstance(obj, dict):
                raise RuntimeError(
                    f"{path}:{line_no}: row is not dict"
                )

            rows.append(obj)

    return rows


def find_full_seqkd() -> Path:
    matches = sorted(
        DATA_ROOT.rglob("full_seqkd20000_v1.jsonl")
    )

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one full_seqkd20000_v1.jsonl; "
            f"found {len(matches)}:\n"
            + "\n".join(map(str, matches))
        )

    return matches[0]


def require_int_index(row, where):
    idx = row.get("index")

    if isinstance(idx, bool) or not isinstance(idx, int):
        raise RuntimeError(
            f"{where}: invalid index={idx!r}"
        )

    return idx


def validate_messages(messages, source, where):
    expected = [
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(
                source=source
            ),
        }
    ]

    if messages != expected:
        raise RuntimeError(
            f"{where}: prompt mismatch\n"
            f"expected={expected!r}\n"
            f"actual={messages!r}"
        )


def percentile(xs, q):
    xs = sorted(xs)

    if not xs:
        raise RuntimeError("empty percentile input")

    pos = int(round(
        (len(xs) - 1) * q
    ))

    return xs[pos]


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not SOURCE_UNIVERSE.is_file():
        raise RuntimeError(
            f"Broad20k source universe missing: "
            f"{SOURCE_UNIVERSE}"
        )

    source_sha = sha256(
        SOURCE_UNIVERSE
    )

    if source_sha != EXPECTED_SOURCE_SHA:
        raise RuntimeError(
            "Broad20k source SHA mismatch\n"
            f"expected={EXPECTED_SOURCE_SHA}\n"
            f"actual={source_sha}"
        )

    full_seqkd = find_full_seqkd()

    print(
        "SOURCE_UNIVERSE =",
        SOURCE_UNIVERSE,
    )
    print(
        "SOURCE_SHA256 =",
        source_sha,
    )
    print(
        "FULL_SEQKD =",
        full_seqkd,
    )
    print(
        "FULL_SEQKD_SHA256 =",
        sha256(full_seqkd),
    )

    source_rows = read_jsonl(
        SOURCE_UNIVERSE
    )
    full_rows = read_jsonl(
        full_seqkd
    )

    if len(source_rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"source rows={len(source_rows)}, "
            f"expected={EXPECTED_ROWS}"
        )

    if len(full_rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"FullSeqKD rows={len(full_rows)}, "
            f"expected={EXPECTED_ROWS}"
        )

    source_map = {}

    for i, row in enumerate(source_rows):
        idx = require_int_index(
            row,
            f"source[{i}]",
        )

        source = row.get("source")

        if not isinstance(source, str) or not source:
            raise RuntimeError(
                f"source[{i}]: invalid source"
            )

        if idx in source_map:
            raise RuntimeError(
                f"duplicate source index={idx}"
            )

        source_map[idx] = source

    full_map = {}

    for i, row in enumerate(full_rows):
        idx = require_int_index(
            row,
            f"full[{i}]",
        )

        if idx in full_map:
            raise RuntimeError(
                f"duplicate FullSeqKD index={idx}"
            )

        full_map[idx] = row

    expected_ids = set(
        range(EXPECTED_ROWS)
    )

    if set(source_map) != expected_ids:
        raise RuntimeError(
            "Broad20k source IDs are not exactly "
            "0..19999"
        )

    if set(full_map) != expected_ids:
        raise RuntimeError(
            "FullSeqKD IDs are not exactly "
            "0..19999"
        )

    sft_rows = []
    opd_rows = []

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        local_files_only=True,
    )

    prompt_lengths = []

    for idx in range(EXPECTED_ROWS):
        source = source_map[idx]
        row = full_map[idx]

        if row.get("source") != source:
            raise RuntimeError(
                f"index={idx}: source mismatch "
                "between canonical source universe "
                "and FullSeqKD"
            )

        messages = row.get("messages")

        if not isinstance(messages, list):
            raise RuntimeError(
                f"index={idx}: missing messages"
            )

        validate_messages(
            messages,
            source,
            f"index={idx}",
        )

        target = row.get(
            "target_translation"
        )

        if (
            not isinstance(target, str)
            or not target.strip()
        ):
            raise RuntimeError(
                f"index={idx}: "
                "missing target_translation"
            )

        # Render first, then tokenize explicitly.
        #
        # Do not use len(apply_chat_template(..., tokenize=True))
        # as a token-count oracle: Transformers return types may vary.
        rendered_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        prompt_ids = tokenizer(
            rendered_prompt,
            add_special_tokens=False,
        )["input_ids"]

        prompt_lengths.append(
            len(prompt_ids)
        )

        # -----------------------------
        # SeqKD SFT asset.
        #
        # Only fields required for the
        # response-only SFT semantics are
        # carried forward.
        # -----------------------------
        sft_rows.append(
            {
                "index": idx,
                "source": source,
                "messages": messages,
                "target_translation": target,
                "data_source": row.get(
                    "data_source",
                    "broad20k",
                ),
                "ability": "translation",
            }
        )

        # -----------------------------
        # OPD asset.
        #
        # IMPORTANT:
        # No Teacher target / reference
        # is present in the OPD trajectory
        # data.
        #
        # reward_model.ground_truth must
        # exist because Verl's generic
        # RewardLoop reads the field before
        # invoking our zero-reward callback.
        # It is a fixed sentinel and carries
        # no translation information.
        # -----------------------------
        opd_rows.append(
            {
                "index": idx,
                "prompt": messages,
                "data_source": "default",
                "ability": "translation",
                "reward_model": {
                    "style": "rule",
                    "ground_truth":
                        "UNUSED_ZERO_REWARD",
                },
                "extra_info": {
                    "index": idx,
                },
            }
        )

    max_prompt = max(
        prompt_lengths
    )

    p50 = percentile(
        prompt_lengths,
        0.50,
    )
    p95 = percentile(
        prompt_lengths,
        0.95,
    )
    p99 = percentile(
        prompt_lengths,
        0.99,
    )

    print(
        "PROMPT_LEN_P50 =",
        p50,
    )
    print(
        "PROMPT_LEN_P95 =",
        p95,
    )
    print(
        "PROMPT_LEN_P99 =",
        p99,
    )
    print(
        "PROMPT_LEN_MAX =",
        max_prompt,
    )

    # Formal recipe will use:
    # max_prompt_length=1024
    # max_response_length=256
    #
    # Fail rather than truncate source prompts.
    if max_prompt > 1024:
        raise RuntimeError(
            "Formal OPD prompt cap 1024 "
            f"is insufficient; max={max_prompt}"
        )

    pq.write_table(
        pa.Table.from_pylist(
            sft_rows
        ),
        SEQKD_PARQUET,
        compression="zstd",
    )

    pq.write_table(
        pa.Table.from_pylist(
            opd_rows
        ),
        OPD_PARQUET,
        compression="zstd",
    )

    # Re-read to make sure the bytes on disk
    # really have the expected row count.
    sft_table = pq.read_table(
        SEQKD_PARQUET
    )
    opd_table = pq.read_table(
        OPD_PARQUET
    )

    if sft_table.num_rows != EXPECTED_ROWS:
        raise RuntimeError(
            "SeqKD parquet row count mismatch"
        )

    if opd_table.num_rows != EXPECTED_ROWS:
        raise RuntimeError(
            "OPD parquet row count mismatch"
        )

    forbidden = {
        "reference",
        "target_translation",
        "teacher_translation",
        "teacher_target",
    }

    leaked_columns = (
        forbidden
        & set(opd_table.column_names)
    )

    if leaked_columns:
        raise RuntimeError(
            "Teacher/reference leakage in OPD "
            f"columns: {sorted(leaked_columns)}"
        )

    manifest = {
        "version": 1,
        "population": {
            "name": "frozen_broad20k",
            "rows": EXPECTED_ROWS,
            "source_path":
                str(SOURCE_UNIVERSE),
            "source_sha256":
                source_sha,
        },
        "full_seqkd": {
            "path": str(full_seqkd),
            "sha256":
                sha256(full_seqkd),
            "rows":
                len(full_rows),
        },
        "derived_assets": {
            "seqkd_parquet": {
                "path":
                    str(SEQKD_PARQUET),
                "sha256":
                    sha256(
                        SEQKD_PARQUET
                    ),
                "rows":
                    sft_table.num_rows,
            },
            "opd_parquet": {
                "path":
                    str(OPD_PARQUET),
                "sha256":
                    sha256(
                        OPD_PARQUET
                    ),
                "rows":
                    opd_table.num_rows,
                "contains_reference":
                    False,
                "contains_teacher_target":
                    False,
            },
        },
        "prompt_contract": {
            "template":
                PROMPT_TEMPLATE,
            "enable_thinking":
                False,
            "prompt_tokens": {
                "p50": p50,
                "p95": p95,
                "p99": p99,
                "max": max_prompt,
            },
        },
    }

    MANIFEST.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "SEQKD_PARQUET =",
        SEQKD_PARQUET,
    )
    print(
        "SEQKD_SHA256 =",
        sha256(
            SEQKD_PARQUET
        ),
    )
    print(
        "OPD_PARQUET =",
        OPD_PARQUET,
    )
    print(
        "OPD_SHA256 =",
        sha256(
            OPD_PARQUET
        ),
    )
    print(
        "MANIFEST =",
        MANIFEST,
    )

    print()
    print(
        "CANONICAL_BROAD20K_VERL_ASSETS_PASS"
    )


if __name__ == "__main__":
    main()
