#!/usr/bin/env python3
# coding: utf-8

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq


def norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(s)).split()).strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


def convert(path: Path, *, is_train: bool):
    table = pq.read_table(path)
    raw = table.to_pylist()
    out = []

    for i, row in enumerate(raw):
        extra = row.get("extra_info") or {}
        reward = row.get("reward_model") or {}

        source = str(extra.get("src", "")).strip()
        reference = str(reward.get("ground_truth", "")).strip()
        messages = row.get("prompt")

        if not source:
            raise RuntimeError(f"{path}: empty source at row {i}")
        if not reference:
            raise RuntimeError(f"{path}: empty reference at row {i}")
        if not isinstance(messages, list) or not messages:
            raise RuntimeError(f"{path}: invalid prompt/messages at row {i}")

        item = {
            "index": i,
            "source": source,
            "reference": reference,
            "messages": messages,
            "src_lang": extra.get("src_lang"),
            "tgt_lang": extra.get("tgt_lang"),
            "data_source": row.get("data_source"),
            "ability": row.get("ability"),
        }
        if is_train:
            item["target_translation"] = reference
        out.append(item)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()

    files = {
        "human_train6565": (
            args.data_root / "train/train_mixed_zh_CN-en.parquet",
            True,
        ),
        "wmt24_zh_en998": (
            args.data_root / "test/wmt24_zh_CN-en.parquet",
            False,
        ),
        "flores_zh_en1012": (
            args.data_root / "test/flores_zh2en.parquet",
            False,
        ),
        "challenge_zh_en197": (
            args.data_root / "test/challenge_set_zh_CN-en.parquet",
            False,
        ),
    }

    converted = {}
    for name, (path, is_train) in files.items():
        if not path.exists():
            raise FileNotFoundError(path)
        rows = convert(path, is_train=is_train)
        out = args.output_dir / f"{name}.jsonl"
        write_jsonl(out, rows)
        converted[name] = {
            "source_parquet": str(path),
            "source_parquet_sha256": sha256(path),
            "jsonl": str(out),
            "jsonl_sha256": sha256(out),
            "rows": len(rows),
            "unique_normalized_source": len({norm(x["source"]) for x in rows}),
            "is_train": is_train,
        }
        print(
            f"{name}: rows={len(rows)} "
            f"unique_source={converted[name]['unique_normalized_source']} "
            f"output={out}"
        )

    train_rows = convert(files["human_train6565"][0], is_train=True)
    train_src = {norm(x["source"]) for x in train_rows}

    overlap = {}
    for name in ("wmt24_zh_en998", "flores_zh_en1012", "challenge_zh_en197"):
        test_rows = convert(files[name][0], is_train=False)
        test_src = {norm(x["source"]) for x in test_rows}
        overlap[name] = len(train_src & test_src)

    manifest = {
        "pilot": "pilot_v2_qwen3_06b",
        "prompt_policy": (
            "Use each parquet row's provided prompt messages; "
            "Qwen chat template is applied with enable_thinking=False."
        ),
        "train_policy": (
            "Human-SFT sanity check uses reward_model.ground_truth as target. "
            "Future MT-PATCHER construction must hide this reference."
        ),
        "datasets": converted,
        "exact_normalized_train_test_overlap": overlap,
    }

    manifest_path = args.output_dir / "data_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if any(overlap.values()):
        raise RuntimeError(f"train/test overlap found: {overlap}")

    print("PILOT_V2_DATA_PREP_PASS")
    print("manifest =", manifest_path)


if __name__ == "__main__":
    main()
