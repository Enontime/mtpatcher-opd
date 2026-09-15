#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path("/workspace/mtpatcher")
SRC = ROOT / "data/verl_science_broad20k"
OUT = SRC / "matched20k_v2"

CANONICAL_MANIFEST = SRC / "canonical_broad20k_assets_v1.json"
SEQKD = SRC / "seqkd_broad20k.parquet"
OPD = SRC / "opd_broad20k.parquet"

ROWS = 20_000
GLOBAL_BATCH = 16
PASSES = 6
TOTAL_STEPS = ROWS * PASSES // GLOBAL_BATCH

# Frozen experiment seeds.
SCHEDULE_SEED = 20260820
TRAIN_PROBE_SEED = 20260915
TRAIN_PROBE_ROWS = 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def as_messages(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def get_prompt_content(value, row_id: int) -> str:
    value = as_messages(value)

    if not isinstance(value, list) or len(value) != 1:
        fail(f"row {row_id}: OPD prompt must contain exactly one message")

    msg = value[0]

    if not isinstance(msg, dict):
        fail(f"row {row_id}: OPD prompt[0] is not dict")

    if msg.get("role") != "user":
        fail(f"row {row_id}: OPD prompt role != user")

    content = msg.get("content")

    if not isinstance(content, str):
        fail(f"row {row_id}: OPD prompt content is not str")

    return content


def get_seqkd_user_content(value, row_id: int) -> str:
    value = as_messages(value)

    if not isinstance(value, list) or len(value) != 2:
        fail(f"row {row_id}: SeqKD messages must contain two messages")

    if value[0].get("role") != "user":
        fail(f"row {row_id}: SeqKD first role != user")

    if value[1].get("role") != "assistant":
        fail(f"row {row_id}: SeqKD second role != assistant")

    content = value[0].get("content")

    if not isinstance(content, str):
        fail(f"row {row_id}: SeqKD user content is not str")

    return content


def append_int_column(
    table: pa.Table,
    name: str,
    values,
    type_=pa.int64(),
) -> pa.Table:
    return table.append_column(
        name,
        pa.array(values, type=type_),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    canonical = json.loads(
        CANONICAL_MANIFEST.read_text(encoding="utf-8")
    )

    expected_seqkd_sha = canonical["derived_assets"]["seqkd_parquet"]["sha256"]
    expected_opd_sha = canonical["derived_assets"]["opd_parquet"]["sha256"]

    actual_seqkd_sha = sha256_file(SEQKD)
    actual_opd_sha = sha256_file(OPD)

    if actual_seqkd_sha != expected_seqkd_sha:
        fail(
            "canonical SeqKD parquet SHA mismatch: "
            f"{actual_seqkd_sha} != {expected_seqkd_sha}"
        )

    if actual_opd_sha != expected_opd_sha:
        fail(
            "canonical OPD parquet SHA mismatch: "
            f"{actual_opd_sha} != {expected_opd_sha}"
        )

    seq = pq.read_table(SEQKD)
    opd = pq.read_table(OPD)

    if seq.num_rows != ROWS:
        fail(f"SeqKD rows={seq.num_rows}, expected={ROWS}")

    if opd.num_rows != ROWS:
        fail(f"OPD rows={opd.num_rows}, expected={ROWS}")

    seq_rows = seq.to_pylist()
    opd_rows = opd.to_pylist()

    template = canonical["prompt_contract"]["template"]

    # ------------------------------------------------------------
    # Canonical row/source identity.
    # This is a hard input invariant, not a filename assumption.
    # ------------------------------------------------------------
    for i, (s, o) in enumerate(zip(seq_rows, opd_rows)):
        s_index = int(s["index"])
        o_index = int(o["index"])

        if s_index != i or o_index != i:
            fail(
                f"row {i}: canonical index mismatch "
                f"seqkd={s_index}, opd={o_index}"
            )

        source = s["source"]

        if not isinstance(source, str) or not source:
            fail(f"row {i}: invalid SeqKD source")

        expected_prompt = template.format(source=source)

        seq_prompt = get_seqkd_user_content(s["messages"], i)
        opd_prompt = get_prompt_content(o["prompt"], i)

        if seq_prompt != expected_prompt:
            fail(f"row {i}: SeqKD prompt/source mismatch")

        if opd_prompt != expected_prompt:
            fail(f"row {i}: OPD prompt/source mismatch")

    print("CANONICAL_SOURCE_IDENTITY=PASS")
    print(f"ROWS={ROWS}")

    # ------------------------------------------------------------
    # One frozen source trajectory shared by both methods.
    # ------------------------------------------------------------
    rng = np.random.default_rng(SCHEDULE_SEED)

    source_order: list[int] = []
    schedule_rows: list[dict] = []

    for pass_id in range(1, PASSES + 1):
        permutation = rng.permutation(ROWS).tolist()

        if len(set(permutation)) != ROWS:
            fail(f"pass {pass_id}: permutation is not one-to-one")

        for within_pass, source_id in enumerate(permutation):
            schedule_position = len(source_order)
            global_step = schedule_position // GLOBAL_BATCH + 1
            position_in_batch = schedule_position % GLOBAL_BATCH

            source_order.append(int(source_id))

            schedule_rows.append(
                {
                    "schedule_position": schedule_position,
                    "global_step": global_step,
                    "pass": pass_id,
                    "position_in_pass": within_pass,
                    "position_in_batch": position_in_batch,
                    "source_id": int(source_id),
                }
            )

    if len(source_order) != ROWS * PASSES:
        fail("wrong schedule length")

    if TOTAL_STEPS != 7500:
        fail(f"unexpected total steps: {TOTAL_STEPS}")

    schedule_path = OUT / "source_order_manifest.jsonl"

    with schedule_path.open("w", encoding="utf-8") as f:
        for row in schedule_rows:
            f.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )

    order_arr = pa.array(source_order, type=pa.int64())

    seq_matched = seq.take(order_arr)
    opd_matched = opd.take(order_arr)

    schedule_position = np.arange(len(source_order), dtype=np.int64)
    global_step = schedule_position // GLOBAL_BATCH + 1
    pass_ids = schedule_position // ROWS + 1
    position_in_batch = schedule_position % GLOBAL_BATCH

    metadata_columns = {
        "source_id": np.asarray(source_order, dtype=np.int64),
        "schedule_position": schedule_position,
        "matched_global_step": global_step,
        "matched_pass": pass_ids,
        "position_in_global_batch": position_in_batch,
    }

    for name, values in metadata_columns.items():
        seq_matched = append_int_column(seq_matched, name, values)
        opd_matched = append_int_column(opd_matched, name, values)

    seq_out = OUT / "seqkd_matched120k.parquet"
    opd_out = OUT / "opd_matched120k.parquet"

    pq.write_table(seq_matched, seq_out, compression="zstd")
    pq.write_table(opd_matched, opd_out, compression="zstd")

    # ------------------------------------------------------------
    # Frozen train probe.
    # It measures teacher-target fitting, not benchmark generalization.
    # ------------------------------------------------------------
    probe_rng = np.random.default_rng(TRAIN_PROBE_SEED)

    probe_ids = probe_rng.choice(
        ROWS,
        size=TRAIN_PROBE_ROWS,
        replace=False,
    ).tolist()

    probe_manifest = {
        "version": 1,
        "meaning": "fixed teacher-target training probe",
        "population_rows": ROWS,
        "rows": TRAIN_PROBE_ROWS,
        "seed": TRAIN_PROBE_SEED,
        "source_ids": [int(x) for x in probe_ids],
    }

    probe_manifest_path = OUT / "train_probe_manifest.json"

    probe_manifest_path.write_text(
        json.dumps(probe_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    probe_jsonl = OUT / "train_probe1024.jsonl"

    with probe_jsonl.open("w", encoding="utf-8") as f:
        for probe_index, source_id in enumerate(probe_ids):
            row = seq_rows[int(source_id)]
            source = row["source"]
            reference = row["target_translation"]
            prompt = template.format(source=source)

            out = {
                "index": probe_index,
                "sample_id": f"train_probe:{source_id}",
                "source_id": int(source_id),
                "data_source": "train_probe",
                "source": source,
                "reference": reference,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            }

            f.write(
                json.dumps(out, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )

    # ------------------------------------------------------------
    # Formal asset manifest.
    # No PASS/gate files are created here.
    # ------------------------------------------------------------
    asset_manifest = {
        "version": 1,
        "experiment": "matched20k_v2",
        "population": {
            "rows": ROWS,
            "canonical_manifest": str(CANONICAL_MANIFEST),
            "seqkd_input": str(SEQKD),
            "seqkd_input_sha256": actual_seqkd_sha,
            "opd_input": str(OPD),
            "opd_input_sha256": actual_opd_sha,
        },
        "schedule": {
            "seed": SCHEDULE_SEED,
            "passes": PASSES,
            "global_batch_size": GLOBAL_BATCH,
            "total_rows": len(source_order),
            "total_optimizer_steps": TOTAL_STEPS,
            "manifest": str(schedule_path),
            "manifest_sha256": sha256_file(schedule_path),
        },
        "matched_assets": {
            "seqkd": {
                "path": str(seq_out),
                "rows": seq_matched.num_rows,
                "sha256": sha256_file(seq_out),
            },
            "opd": {
                "path": str(opd_out),
                "rows": opd_matched.num_rows,
                "sha256": sha256_file(opd_out),
            },
        },
        "train_probe": {
            "meaning": "teacher-target fitting probe",
            "rows": TRAIN_PROBE_ROWS,
            "seed": TRAIN_PROBE_SEED,
            "manifest": str(probe_manifest_path),
            "manifest_sha256": sha256_file(probe_manifest_path),
            "jsonl": str(probe_jsonl),
            "jsonl_sha256": sha256_file(probe_jsonl),
        },
    }

    asset_manifest_path = OUT / "asset_manifest.json"

    asset_manifest_path.write_text(
        json.dumps(asset_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("MATCHED_ASSET_BUILD=PASS")
    print(f"SCHEDULE_ROWS={len(source_order)}")
    print(f"OPTIMIZER_STEPS={TOTAL_STEPS}")
    print(f"SEQKD_MATCHED={seq_out}")
    print(f"OPD_MATCHED={opd_out}")
    print(f"TRAIN_PROBE={probe_jsonl}")
    print(f"ASSET_MANIFEST={asset_manifest_path}")


if __name__ == "__main__":
    main()
