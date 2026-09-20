#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path("/workspace/mtpatcher")
PROJECT = ROOT / "repo/MT-Patcher-Reproduction-Ascend"

FREEZE_ROOT = (
    ROOT
    / "data/mtpatcher_v3_full6565_20260823"
    / "strong_repro_student_arms_freeze_v1"
)

PE_JSONL = FREEZE_ROOT / "k1_all_pe11792_v1.jsonl"
PE_FREEZE_MANIFEST = (
    FREEZE_ROOT / "strong_repro_student_arms_manifest_v1.json"
)

OUT = (
    ROOT
    / "data/verl_science_pe_pds"
    / "pe11792_matched_v1"
)

SCHEDULE = OUT / "source_order_manifest.jsonl"
SFT_INPUT = OUT / "pe_sft_matched70752_input.jsonl"
SFT_PARQUET = OUT / "pe_sft_matched70752.parquet"
OPD_PARQUET = OUT / "pe_opd_matched70752.parquet"
MANIFEST = OUT / "asset_manifest.json"

PREPARE_SFT = PROJECT / "scripts/data/prepare_verl_sft.py"

EXPECTED_PE_SHA = (
    "7d8c5c1de8249db19e7e881174ef103c9ab6ebfa590623b6ef3190283684ed18"
)

ROWS = 11_792
GLOBAL_BATCH = 16
PASSES = 6
SCHEDULE_SEED = 20260820

TOTAL_EXPOSURES = ROWS * PASSES
STEPS_PER_PASS = ROWS // GLOBAL_BATCH
TOTAL_STEPS = TOTAL_EXPOSURES // GLOBAL_BATCH

PROMPT_TEMPLATE = (
    "Translate the following text into English "
    "without additional explanations:\n\n"
    "{source}\n\n"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open(encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except Exception as exc:
                fail(f"{path}:{line_no}: JSON parse error: {exc}")

            if not isinstance(row, dict):
                fail(f"{path}:{line_no}: row is not dict")

            rows.append(row)

    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    tmp.replace(path)


def as_list(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def validate_frozen_contract() -> dict:
    if not PE_JSONL.is_file():
        fail(f"missing frozen PE dataset: {PE_JSONL}")

    if not PE_FREEZE_MANIFEST.is_file():
        fail(f"missing frozen PE manifest: {PE_FREEZE_MANIFEST}")

    actual_sha = sha256_file(PE_JSONL)

    if actual_sha != EXPECTED_PE_SHA:
        fail(
            "frozen PE SHA drift: "
            f"{actual_sha} != {EXPECTED_PE_SHA}"
        )

    frozen = json.loads(
        PE_FREEZE_MANIFEST.read_text(encoding="utf-8")
    )

    checks = frozen.get("causal_checks", {})

    if checks.get(
        "all_pe_targets_from_frozen_k1_post_edit"
    ) is not True:
        fail(
            "freeze manifest does not certify "
            "PE target <- K1 post_edit"
        )

    dataset = (
        frozen
        .get("datasets", {})
        .get("K1_ALL_PE11792", {})
    )

    if int(dataset.get("rows", -1)) != ROWS:
        fail("freeze manifest PE row count drift")

    if dataset.get("sha256") != EXPECTED_PE_SHA:
        fail("freeze manifest PE SHA drift")

    if dataset.get("target") != "K1 conservative post_edit":
        fail(
            f"unexpected frozen target semantics: "
            f"{dataset.get('target')!r}"
        )

    return frozen


def validate_pe_rows(rows: list[dict]) -> None:
    if len(rows) != ROWS:
        fail(f"PE rows={len(rows)} expected={ROWS}")

    seen = set()

    for position, row in enumerate(rows):
        idx = row.get("index")
        source = row.get("source")
        target = row.get("target_translation")
        messages = row.get("messages")

        if not isinstance(idx, int):
            fail(f"row {position}: index must be int")

        if idx in seen:
            fail(f"row {position}: duplicate index={idx}")

        seen.add(idx)

        if not isinstance(source, str) or not source.strip():
            fail(f"row {position}: invalid source")

        if not isinstance(target, str) or not target.strip():
            fail(f"row {position}: invalid target_translation")

        if (
            row.get("target_provenance")
            != "K1_CONSERVATIVE_POST_EDIT"
        ):
            fail(
                f"row {position}: target provenance drift: "
                f"{row.get('target_provenance')!r}"
            )

        if not isinstance(messages, list) or len(messages) != 1:
            fail(
                f"row {position}: expected one user message"
            )

        message = messages[0]

        if (
            not isinstance(message, dict)
            or message.get("role") != "user"
            or not isinstance(message.get("content"), str)
        ):
            fail(f"row {position}: invalid user message")

        expected = PROMPT_TEMPLATE.format(source=source)

        if message["content"] != expected:
            fail(f"row {position}: prompt/source mismatch")


def build_schedule(rows: list[dict]) -> list[dict]:
    rng = np.random.default_rng(SCHEDULE_SEED)

    schedule = []

    for pass_id in range(1, PASSES + 1):
        permutation = rng.permutation(ROWS).tolist()

        if (
            len(permutation) != ROWS
            or len(set(permutation)) != ROWS
            or set(permutation) != set(range(ROWS))
        ):
            fail(f"pass {pass_id}: invalid permutation")

        for position_in_pass, population_position in enumerate(
            permutation
        ):
            schedule_position = len(schedule)

            schedule.append(
                {
                    "schedule_position": schedule_position,
                    "population_position":
                        int(population_position),
                    "source_id":
                        int(rows[population_position]["index"]),
                    "global_step":
                        schedule_position // GLOBAL_BATCH + 1,
                    "pass": pass_id,
                    "position_in_pass":
                        int(position_in_pass),
                    "position_in_batch":
                        schedule_position % GLOBAL_BATCH,
                }
            )

    if len(schedule) != TOTAL_EXPOSURES:
        fail(
            f"schedule rows={len(schedule)} "
            f"expected={TOTAL_EXPOSURES}"
        )

    return schedule


def matched_metadata(item: dict) -> dict:
    return {
        "source_id": int(item["source_id"]),
        "schedule_position":
            int(item["schedule_position"]),
        "matched_global_step":
            int(item["global_step"]),
        "matched_pass":
            int(item["pass"]),
        "position_in_global_batch":
            int(item["position_in_batch"]),
    }


def main() -> None:
    if ROWS % GLOBAL_BATCH != 0:
        fail("population does not divide global batch")

    if STEPS_PER_PASS != 737:
        fail(f"unexpected steps/pass={STEPS_PER_PASS}")

    if TOTAL_STEPS != 4422:
        fail(f"unexpected formal steps={TOTAL_STEPS}")

    if not PREPARE_SFT.is_file():
        fail(f"missing SFT materializer: {PREPARE_SFT}")

    frozen_manifest = validate_frozen_contract()

    pe_rows = load_jsonl(PE_JSONL)
    validate_pe_rows(pe_rows)

    OUT.mkdir(parents=True, exist_ok=True)

    schedule = build_schedule(pe_rows)
    write_jsonl(SCHEDULE, schedule)

    sft_input_rows = []
    opd_rows = []

    for item in schedule:
        base = pe_rows[item["population_position"]]
        metadata = matched_metadata(item)

        # ------------------------------------------------------
        # SFT:
        # Preserve frozen PE target and metadata.
        # Existing prepare_verl_sft.py owns assistant
        # materialization + enable_thinking=False.
        # ------------------------------------------------------
        sft_row = deepcopy(base)
        sft_row.update(metadata)
        sft_input_rows.append(sft_row)

        # ------------------------------------------------------
        # OPD:
        # Current canonical Verl source-only schema.
        # No PE post-edit / reference / teacher target.
        # ------------------------------------------------------
        opd_row = {
            "index": int(base["index"]),
            "prompt": deepcopy(base["messages"]),
            "data_source": "default",
            "ability": "translation",
            "reward_model": {
                "style": "rule",
                "ground_truth": "UNUSED_ZERO_REWARD",
            },
            "extra_info": {
                "index": int(base["index"]),
                "source": base["source"],
            },
            **metadata,
        }

        forbidden = {
            "target_translation",
            "reference",
            "post_edit",
            "student_translation",
        }

        leaked = forbidden.intersection(opd_row)

        if leaked:
            fail(
                f"OPD target leakage fields={sorted(leaked)}"
            )

        opd_rows.append(opd_row)

    write_jsonl(SFT_INPUT, sft_input_rows)

    # Reuse the frozen generic SFT materializer.
    subprocess.run(
        [
            sys.executable,
            str(PREPARE_SFT),
            "--input",
            str(SFT_INPUT),
            "--output",
            str(SFT_PARQUET),
            "--expected-rows",
            str(TOTAL_EXPOSURES),
            "--overwrite",
        ],
        cwd=str(PROJECT),
        check=True,
    )

    pq.write_table(
        pa.Table.from_pylist(opd_rows),
        OPD_PARQUET,
        compression="zstd",
    )

    # ----------------------------------------------------------
    # Zero-training round-trip gate.
    # ----------------------------------------------------------
    sft = pq.read_table(SFT_PARQUET)
    opd = pq.read_table(OPD_PARQUET)

    if sft.num_rows != TOTAL_EXPOSURES:
        fail("SFT parquet row count mismatch")

    if opd.num_rows != TOTAL_EXPOSURES:
        fail("OPD parquet row count mismatch")

    sft_rows = sft.to_pylist()
    opd_rows_back = opd.to_pylist()

    metadata_keys = (
        "source_id",
        "schedule_position",
        "matched_global_step",
        "matched_pass",
        "position_in_global_batch",
    )

    for i, (s, o, sch) in enumerate(
        zip(sft_rows, opd_rows_back, schedule)
    ):
        for key in metadata_keys:
            if int(s[key]) != int(o[key]):
                fail(
                    f"row {i}: SFT/OPD metadata mismatch "
                    f"for {key}"
                )

        if int(s["source_id"]) != int(sch["source_id"]):
            fail(f"row {i}: schedule source mismatch")

        s_messages = as_list(s["messages"])
        o_prompt = as_list(o["prompt"])

        if (
            not isinstance(s_messages, list)
            or len(s_messages) != 2
        ):
            fail(f"row {i}: invalid SFT messages")

        if (
            not isinstance(o_prompt, list)
            or len(o_prompt) != 1
        ):
            fail(f"row {i}: invalid OPD prompt")

        if s_messages[0] != o_prompt[0]:
            fail(f"row {i}: SFT/OPD prompt mismatch")

        if s_messages[1].get("role") != "assistant":
            fail(f"row {i}: invalid SFT assistant role")

        if (
            str(s_messages[1].get("content", "")).strip()
            != str(s["target_translation"]).strip()
        ):
            fail(f"row {i}: SFT target mismatch")

        if o["data_source"] != "default":
            fail(f"row {i}: OPD data_source drift")

        reward = o["reward_model"]

        if (
            reward.get("ground_truth")
            != "UNUSED_ZERO_REWARD"
        ):
            fail(f"row {i}: reward sentinel drift")

        for key in (
            "target_translation",
            "reference",
            "post_edit",
            "student_translation",
        ):
            if key in o:
                fail(
                    f"row {i}: OPD contains forbidden {key}"
                )

    eval_steps = list(range(0, 4401, 100))

    if TOTAL_STEPS not in eval_steps:
        eval_steps.append(TOTAL_STEPS)

    manifest = {
        "version": 1,
        "experiment": "pe11792_matched_v1",
        "classification":
            "LAB_REPRODUCTION_PAPER_ALIGNED_ADAPTATION",
        "population": {
            "rows": ROWS,
            "input_jsonl": str(PE_JSONL),
            "input_sha256": sha256_file(PE_JSONL),
            "freeze_manifest":
                str(PE_FREEZE_MANIFEST),
            "freeze_manifest_sha256":
                sha256_file(PE_FREEZE_MANIFEST),
            "lineage_certified_by_freeze_manifest": True,
            "target_provenance":
                "K1_CONSERVATIVE_POST_EDIT",
        },
        "schedule": {
            "seed": SCHEDULE_SEED,
            "passes": PASSES,
            "global_batch_size": GLOBAL_BATCH,
            "steps_per_pass": STEPS_PER_PASS,
            "formal_steps": TOTAL_STEPS,
            "total_exposures": TOTAL_EXPOSURES,
            "manifest": str(SCHEDULE),
            "manifest_sha256":
                sha256_file(SCHEDULE),
        },
        "comparison_contract": {
            "comparison_type":
                "SAMPLE_EXPOSURE_MATCHED",
            "compute_matched": False,
            "primary_endpoint_step": TOTAL_STEPS,
            "primary_metric":
                "benchmark_macro_bleu",
            "eval_steps": eval_steps,
            "retrospective_best_policy":
                "DIAGNOSTIC_ONLY",
        },
        "assets": {
            "sft_input_jsonl": {
                "path": str(SFT_INPUT),
                "rows": TOTAL_EXPOSURES,
                "sha256":
                    sha256_file(SFT_INPUT),
            },
            "sft_parquet": {
                "path": str(SFT_PARQUET),
                "rows": sft.num_rows,
                "sha256":
                    sha256_file(SFT_PARQUET),
                "materializer":
                    "scripts/data/prepare_verl_sft.py",
            },
            "opd_parquet": {
                "path": str(OPD_PARQUET),
                "rows": opd.num_rows,
                "sha256":
                    sha256_file(OPD_PARQUET),
                "contains_fixed_translation_target":
                    False,
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

    print("PE11792_MATCHED_ASSET_BUILD_PASS")
    print("ROWS =", ROWS)
    print("PASSES =", PASSES)
    print("STEPS_PER_PASS =", STEPS_PER_PASS)
    print("EXPOSURES =", TOTAL_EXPOSURES)
    print("FORMAL_STEPS =", TOTAL_STEPS)
    print("SCHEDULE =", SCHEDULE)
    print("SCHEDULE_SHA256 =", sha256_file(SCHEDULE))
    print("SFT_PARQUET =", SFT_PARQUET)
    print("SFT_SHA256 =", sha256_file(SFT_PARQUET))
    print("OPD_PARQUET =", OPD_PARQUET)
    print("OPD_SHA256 =", sha256_file(OPD_PARQUET))
    print("MANIFEST =", MANIFEST)


if __name__ == "__main__":
    main()
