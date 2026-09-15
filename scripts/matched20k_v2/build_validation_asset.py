#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path("/workspace/mtpatcher")

MATCHED = (
    ROOT
    / "data/verl_science_broad20k/matched20k_v2"
)

DATA = ROOT / "data/pilot_v2_qwen3_06b"

SOURCES = {
    "train_probe": MATCHED / "train_probe1024.jsonl",
    "wmt24": DATA / "wmt24_zh_en998.jsonl",
    "flores": DATA / "flores_zh_en1012.jsonl",
    "challenge": DATA / "challenge_zh_en197.jsonl",
}

EXPECTED_COUNTS = {
    "train_probe": 1024,
    "wmt24": 998,
    "flores": 1012,
    "challenge": 197,
}

EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except Exception as exc:
                raise RuntimeError(
                    f"{path}:{line_no}: invalid JSON"
                ) from exc

            rows.append(row)

    return rows


def normalize_row(
    dataset: str,
    local_index: int,
    row: dict,
    global_index: int,
) -> dict:
    source = row.get("source")
    reference = row.get("reference")
    messages = row.get("messages")

    if not isinstance(source, str) or not source:
        fail(f"{dataset}[{local_index}]: invalid source")

    if not isinstance(reference, str) or not reference.strip():
        fail(f"{dataset}[{local_index}]: invalid reference")

    if not isinstance(messages, list) or len(messages) != 1:
        fail(
            f"{dataset}[{local_index}]: "
            "messages must contain exactly one user message"
        )

    message = messages[0]

    if not isinstance(message, dict):
        fail(f"{dataset}[{local_index}]: message is not dict")

    if message.get("role") != "user":
        fail(f"{dataset}[{local_index}]: role != user")

    content = message.get("content")

    if not isinstance(content, str) or not content:
        fail(f"{dataset}[{local_index}]: invalid prompt content")

    # Preserve the original benchmark prompt exactly.
    # Only require that the actual source text is present.
    if source not in content:
        fail(
            f"{dataset}[{local_index}]: "
            "source is not present exactly in frozen prompt"
        )

    original_index = int(row.get("index", local_index))

    sample_id = row.get("sample_id")

    if not isinstance(sample_id, str) or not sample_id:
        sample_id = f"{dataset}:{original_index}"

    source_id = row.get("source_id")
    if source_id is not None:
        source_id = int(source_id)

    reference = reference.strip()

    return {
        "index": global_index,
        "dataset_index": original_index,
        "sample_id": sample_id,
        # Keep canonical OPD routing semantics.
        # Benchmark identity lives in eval_group.
        "data_source": "default",
        "eval_group": dataset,
        "ability": "translation",
        "source": source,
        "source_id": source_id,
        "reference": reference,

        # Exact frozen prompt used by both SeqKD and OPD validation.
        "messages": messages,
        "prompt": messages,

        # OPD task reward stays identically zero.
        # The real MT reference is stored separately above.
        "reward_model": {
            "style": "rule",
            "ground_truth": "UNUSED_ZERO_REWARD",
        },

        "extra_info": {
            "index": global_index,
            "sample_id": sample_id,
            "dataset": dataset,
            "eval_group": dataset,
            "dataset_index": original_index,
            "source": source,
            "source_id": source_id,
            "reference": reference,
        },
    }


def main() -> None:
    MATCHED.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    source_manifest = {}

    for dataset, path in SOURCES.items():
        if not path.is_file():
            fail(f"missing validation source: {path}")

        rows = read_jsonl(path)
        expected = EXPECTED_COUNTS[dataset]

        if len(rows) != expected:
            fail(
                f"{dataset}: rows={len(rows)}, expected={expected}"
            )

        source_manifest[dataset] = {
            "path": str(path),
            "rows": len(rows),
            "sha256": sha256_file(path),
        }

        for local_index, row in enumerate(rows):
            all_rows.append(
                normalize_row(
                    dataset=dataset,
                    local_index=local_index,
                    row=row,
                    global_index=len(all_rows),
                )
            )

    if len(all_rows) != EXPECTED_TOTAL:
        fail(
            f"total rows={len(all_rows)}, "
            f"expected={EXPECTED_TOTAL}"
        )

    indices = [x["index"] for x in all_rows]

    if indices != list(range(EXPECTED_TOTAL)):
        fail("global validation indices are not contiguous")

    sample_ids = [x["sample_id"] for x in all_rows]

    if len(set(sample_ids)) != len(sample_ids):
        fail("sample_id collision")

    jsonl_path = MATCHED / "validation3231.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    parquet_path = MATCHED / "validation3231.parquet"

    pq.write_table(
        pa.Table.from_pylist(all_rows),
        parquet_path,
        compression="zstd",
    )

    manifest = {
        "version": 3,
        "experiment": "matched20k_v2",
        "meaning": (
            "single shared frozen validation population "
            "for SeqKD and OPD Verl-native validation"
        ),
        "prompt_policy": {
            "preserve_original_messages_exactly": True,
            "enable_thinking": False,
            "normalization": "none",
        },
        "counts": EXPECTED_COUNTS,
        "total_rows": EXPECTED_TOTAL,
        "sources": source_manifest,
        "jsonl": {
            "path": str(jsonl_path),
            "sha256": sha256_file(jsonl_path),
        },
        "parquet": {
            "path": str(parquet_path),
            "sha256": sha256_file(parquet_path),
        },
        "metric_policy": {
            "train_probe": (
                "teacher-target fitting probe"
            ),
            "benchmark_macro": (
                "unweighted mean across "
                "WMT24/FLORES/Challenge dataset-level metrics"
            ),
        },
    }

    # matched20k_v2 routing policy freeze
    manifest["routing_policy"] = {
        "verl_data_source": "default",
        "metric_group_field": "eval_group",
        "reward_ground_truth": "UNUSED_ZERO_REWARD",
        "reference_field": "reference",
    }

    manifest_path = MATCHED / "validation_manifest.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("VALIDATION_ASSET_BUILD=PASS")
    print(f"TOTAL_ROWS={EXPECTED_TOTAL}")

    for name, count in EXPECTED_COUNTS.items():
        print(f"{name.upper()}_ROWS={count}")

    print(f"JSONL={jsonl_path}")
    print(f"PARQUET={parquet_path}")
    print(f"MANIFEST={manifest_path}")


if __name__ == "__main__":
    main()
