#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("/workspace/mtpatcher")
PROJECT = ROOT / "repo/MT-Patcher-Reproduction-Ascend"

UPSTREAM_PDS = (
    ROOT
    / "data/mtpatcher_v3_full6565_20260823"
    / "strong_repro_pds_full_to_student_v2"
    / "pds_structural_accepted_v2.jsonl"
)
EXPECTED_UPSTREAM_SHA = (
    "078559128776c2dd2dd5d7ed2c9a8f22794457bf53382e3b4c4578404287f6f8"
)

OUT = ROOT / "data/verl_science_pe_pds/pds9952_matched_v1"
POPULATION = OUT / "pds9952_population_v1.jsonl"
POPULATION_MANIFEST = OUT / "pds9952_population_manifest_v1.json"
TEACHER_INPUT = OUT / "teacher_input_pds9952_v1.jsonl"
DEFAULT_TEACHER_TARGETS = OUT / "teacher_targets_pds9952_qwen3_8b_v1.jsonl"
SCHEDULE = OUT / "source_order_manifest.jsonl"
SFT_INPUT = OUT / "pds9952_seqkd_matched59712_input.jsonl"
SFT_PARQUET = OUT / "pds9952_seqkd_matched59712.parquet"
OPD_PARQUET = OUT / "pds9952_opd_matched59712.parquet"
ASSET_MANIFEST = OUT / "asset_manifest.json"

PREPARE_SFT = PROJECT / "scripts/data/prepare_verl_sft.py"

ROWS = 9_952
EXPECTED_ELIGIBLE_PARENTS = 9_957
EXPECTED_UPSTREAM_ROWS = 57_125
GLOBAL_BATCH = 16
PASSES = 6
SCHEDULE_SEED = 20260820
TOTAL_EXPOSURES = ROWS * PASSES
TOTAL_STEPS = TOTAL_EXPOSURES // GLOBAL_BATCH

ROW_SALT = "MTP_PDS9952_ROW_V1"
PARENT_SALT = "MTP_PDS9952_PARENT_V1"

PROMPT_TEMPLATE = (
    "Translate the following text into English "
    "without additional explanations:\n\n"
    "{source}\n\n"
)

FORBIDDEN_OPD_KEYS = {
    "target_translation",
    "reference",
    "post_edit",
    "student_translation",
    "historical_target_translation",
    "provenance_historical_target_translation",
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json_bytes(obj: dict[str, Any]) -> bytes:
    # FROZEN v1: no Unicode normalization and no whitespace normalization.
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def row_selection_hash(row: dict[str, Any]) -> str:
    obj = {
        "error_index": int(row["error_index"]),
        "pair_id": str(row["pair_id"]),
        "parent_index": int(row["parent_index"]),
        "pds_slot": int(row["pds_slot"]),
        "salt": ROW_SALT,
        "source": row["source"],
    }
    return sha256_bytes(canonical_json_bytes(obj))


def parent_rank_hash(parent_index: int) -> str:
    obj = {
        "parent_index": int(parent_index),
        "salt": PARENT_SALT,
    }
    return sha256_bytes(canonical_json_bytes(obj))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path: Path, obj: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    os.replace(tmp, path)


def load_jsonl(path: Path, *, with_line_no: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                fail(f"{path}:{line_no}: JSON parse error: {exc}")
            if not isinstance(row, dict):
                fail(f"{path}:{line_no}: row is not an object")
            if with_line_no:
                row = dict(row)
                row["_upstream_line_no"] = line_no
            rows.append(row)
    return rows


def as_list(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def git_capture(args: list[str]) -> str:
    try:
        p = subprocess.run(
            ["git", *args],
            cwd=str(PROJECT),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return p.stdout.rstrip("\n")
    except Exception as exc:
        return f"UNAVAILABLE: {exc!r}"


def validate_upstream_row(row: dict[str, Any], line_no: int) -> None:
    for key in ("parent_index", "error_index", "pair_id", "pds_slot", "source"):
        if key not in row:
            fail(f"upstream line {line_no}: missing required field {key!r}")
    try:
        int(row["parent_index"])
        int(row["error_index"])
        int(row["pds_slot"])
    except Exception as exc:
        fail(f"upstream line {line_no}: invalid integer field: {exc}")
    if not isinstance(row["source"], str) or not row["source"].strip():
        fail(f"upstream line {line_no}: source is empty/non-string")
    if not str(row["pair_id"]):
        fail(f"upstream line {line_no}: pair_id is empty")


def selected_population_identity(rows: list[dict[str, Any]]) -> str:
    identity = []
    for row in rows:
        identity.append(
            {
                "source_id": int(row["source_id"]),
                "parent_index": int(row["parent_index"]),
                "error_index": int(row["error_index"]),
                "pair_id": str(row["pair_id"]),
                "pds_slot": int(row["pds_slot"]),
                "source": row["source"],
                "selection_row_sha256": row["selection_row_sha256"],
                "parent_rank_sha256": row["parent_rank_sha256"],
            }
        )
    blob = ("\n".join(
        json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for x in identity
    ) + "\n").encode("utf-8")
    return sha256_bytes(blob)


def prepare() -> None:
    if not UPSTREAM_PDS.is_file():
        fail(f"missing frozen upstream PDS: {UPSTREAM_PDS}")

    actual_sha = sha256_file(UPSTREAM_PDS)
    if actual_sha != EXPECTED_UPSTREAM_SHA:
        fail(
            "frozen upstream PDS SHA drift: "
            f"{actual_sha} != {EXPECTED_UPSTREAM_SHA}"
        )

    raw = load_jsonl(UPSTREAM_PDS, with_line_no=True)
    if len(raw) != EXPECTED_UPSTREAM_ROWS:
        fail(
            f"upstream rows={len(raw)} expected={EXPECTED_UPSTREAM_ROWS}"
        )

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    hash_tie_events: list[dict[str, Any]] = []

    for row in raw:
        line_no = int(row.pop("_upstream_line_no"))
        validate_upstream_row(row, line_no)
        x = dict(row)
        x["upstream_line_no"] = line_no
        x["selection_row_sha256"] = row_selection_hash(x)
        parent = int(x["parent_index"])
        grouped[parent].append(x)

    if len(grouped) != EXPECTED_ELIGIBLE_PARENTS:
        fail(
            f"eligible parents={len(grouped)} "
            f"expected={EXPECTED_ELIGIBLE_PARENTS}"
        )

    one_per_parent: list[dict[str, Any]] = []
    for parent, candidates in grouped.items():
        candidates.sort(
            key=lambda x: (
                x["selection_row_sha256"],
                int(x["upstream_line_no"]),
            )
        )
        if (
            len(candidates) > 1
            and candidates[0]["selection_row_sha256"]
            == candidates[1]["selection_row_sha256"]
        ):
            hash_tie_events.append(
                {
                    "parent_index": parent,
                    "selection_row_sha256": candidates[0]["selection_row_sha256"],
                    "winner_upstream_line_no": int(candidates[0]["upstream_line_no"]),
                    "tied_upstream_line_nos": [
                        int(x["upstream_line_no"])
                        for x in candidates
                        if x["selection_row_sha256"]
                        == candidates[0]["selection_row_sha256"]
                    ],
                }
            )
        chosen = dict(candidates[0])
        chosen["parent_rank_sha256"] = parent_rank_hash(parent)
        one_per_parent.append(chosen)

    one_per_parent.sort(
        key=lambda x: (
            x["parent_rank_sha256"],
            int(x["parent_index"]),
        )
    )

    retained = one_per_parent[:ROWS]
    excluded = one_per_parent[ROWS:]

    if len(excluded) != EXPECTED_ELIGIBLE_PARENTS - ROWS:
        fail(f"excluded parent count={len(excluded)} expected=5")

    population: list[dict[str, Any]] = []
    for source_id, x in enumerate(retained):
        row = {
            "source_id": source_id,
            "parent_index": int(x["parent_index"]),
            "error_index": int(x["error_index"]),
            "pair_id": str(x["pair_id"]),
            "pds_slot": int(x["pds_slot"]),
            "source": x["source"],
            "upstream_line_no": int(x["upstream_line_no"]),
            "selection_row_sha256": x["selection_row_sha256"],
            "parent_rank_sha256": x["parent_rank_sha256"],
        }
        if x.get("job_id") is not None:
            row["job_id"] = x["job_id"]
        if isinstance(x.get("target_translation"), str):
            row["provenance_historical_target_translation"] = x[
                "target_translation"
            ]
        population.append(row)

    if len(population) != ROWS:
        fail(f"selected rows={len(population)} expected={ROWS}")
    if len({int(x["parent_index"]) for x in population}) != ROWS:
        fail("selected parent_index values are not unique")
    if len({x["source"] for x in population}) != ROWS:
        fail("selected source texts are not unique")
    if len({x["selection_row_sha256"] for x in population}) != ROWS:
        fail("selected row hashes are not unique")
    if any(not x["source"].strip() for x in population):
        fail("selected population contains empty source")

    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(POPULATION, population)

    teacher_input = [
        {"index": int(x["source_id"]), "source": x["source"]}
        for x in population
    ]
    write_jsonl(TEACHER_INPUT, teacher_input)

    population_identity_sha = selected_population_identity(population)

    manifest = {
        "version": 1,
        "experiment": "PDS9952 Parent-Diverse Low-Budget Adaptation",
        "classification": "ADAPTATION",
        "stage": "prepare",
        "upstream": {
            "path": str(UPSTREAM_PDS),
            "sha256": actual_sha,
            "rows": len(raw),
            "eligible_parent_count": len(grouped),
        },
        "selection_contract": {
            "row_salt": ROW_SALT,
            "parent_salt": PARENT_SALT,
            "serialization": {
                "ensure_ascii": False,
                "sort_keys": True,
                "separators": [",", ":"],
                "encoding": "utf-8",
                "unicode_normalization": "none",
                "whitespace_normalization": "none",
                "source_strip_before_hash": False,
            },
            "row_rank": "min(selection_row_sha256, upstream_line_no) per parent",
            "parent_rank": "ascending(parent_rank_sha256, parent_index)",
            "hash_tie_events": hash_tie_events,
        },
        "population": {
            "eligible_parent_count": EXPECTED_ELIGIBLE_PARENTS,
            "retained_parent_count": ROWS,
            "excluded_parent_count": len(excluded),
            "excluded_parents": [
                {
                    "parent_index": int(x["parent_index"]),
                    "parent_rank_sha256": x["parent_rank_sha256"],
                }
                for x in excluded
            ],
            "path": str(POPULATION),
            "sha256": sha256_file(POPULATION),
            "identity_sha256": population_identity_sha,
            "rows": ROWS,
            "unique_parents": len({x["parent_index"] for x in population}),
            "unique_sources": len({x["source"] for x in population}),
            "unique_selection_row_sha256": len(
                {x["selection_row_sha256"] for x in population}
            ),
        },
        "teacher_input": {
            "path": str(TEACHER_INPUT),
            "sha256": sha256_file(TEACHER_INPUT),
            "rows": len(teacher_input),
            "schema": ["index", "source"],
        },
        "prompt_contract": {
            "template": PROMPT_TEMPLATE,
            "enable_thinking": False,
        },
        "git": {
            "head": git_capture(["rev-parse", "HEAD"]),
            "branch": git_capture(["branch", "--show-current"]),
            "status_short": git_capture(["status", "--short"]),
        },
    }
    write_json(POPULATION_MANIFEST, manifest)

    print("PDS9952_PREPARE=PASS")
    print(f"UPSTREAM_SHA256={actual_sha}")
    print(f"ELIGIBLE_PARENTS={len(grouped)}")
    print(f"RETAINED_PARENTS={ROWS}")
    print(
        "EXCLUDED_PARENT_IDS="
        + json.dumps(
            [int(x["parent_index"]) for x in excluded],
            separators=(",", ":"),
        )
    )
    print(f"POPULATION_SHA256={sha256_file(POPULATION)}")
    print(f"POPULATION_IDENTITY_SHA256={population_identity_sha}")
    print(f"TEACHER_INPUT_SHA256={sha256_file(TEACHER_INPUT)}")
    print(f"POPULATION={POPULATION}")
    print(f"TEACHER_INPUT={TEACHER_INPUT}")
    print(f"MANIFEST={POPULATION_MANIFEST}")


def load_and_validate_population() -> list[dict[str, Any]]:
    if not POPULATION.is_file() or not POPULATION_MANIFEST.is_file():
        fail("prepare artifacts missing; run --stage prepare first")

    manifest = json.loads(POPULATION_MANIFEST.read_text(encoding="utf-8"))
    expected_sha = manifest["population"]["sha256"]
    actual_sha = sha256_file(POPULATION)
    if actual_sha != expected_sha:
        fail(f"population SHA drift: {actual_sha} != {expected_sha}")

    rows = load_jsonl(POPULATION)
    if len(rows) != ROWS:
        fail(f"population rows={len(rows)} expected={ROWS}")

    for pos, row in enumerate(rows):
        if int(row["source_id"]) != pos:
            fail(
                f"population row {pos}: source_id={row['source_id']} expected={pos}"
            )
        if not isinstance(row.get("source"), str) or not row["source"].strip():
            fail(f"population row {pos}: invalid source")

    if len({x["source"] for x in rows}) != ROWS:
        fail("population source texts are not unique")
    if len({int(x["parent_index"]) for x in rows}) != ROWS:
        fail("population parent IDs are not unique")
    return rows


def load_teacher_targets(path: Path, population: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not path.is_file():
        fail(f"missing frozen Teacher target asset: {path}")

    rows = load_jsonl(path)
    if len(rows) != ROWS:
        fail(f"Teacher target rows={len(rows)} expected={ROWS}")

    by_index: dict[int, dict[str, Any]] = {}
    for pos, row in enumerate(rows):
        try:
            idx = int(row["index"])
        except Exception as exc:
            fail(f"Teacher target row {pos}: invalid index: {exc}")
        if idx in by_index:
            fail(f"Teacher target duplicate index={idx}")
        target = row.get("target_translation")
        if not isinstance(target, str) or not target.strip():
            fail(f"Teacher target index={idx}: empty target_translation")
        by_index[idx] = row

    if set(by_index) != set(range(ROWS)):
        missing = sorted(set(range(ROWS)) - set(by_index))[:20]
        extra = sorted(set(by_index) - set(range(ROWS)))[:20]
        fail(f"Teacher target index set mismatch missing={missing} extra={extra}")

    ordered: list[dict[str, Any]] = []
    for source_id, pop in enumerate(population):
        row = by_index[source_id]
        if row.get("source") != pop["source"]:
            fail(f"Teacher target source mismatch at source_id={source_id}")
        ordered.append(row)
    return ordered


def build_schedule() -> list[dict[str, int]]:
    if ROWS % GLOBAL_BATCH != 0:
        fail("population does not divide global batch")
    if TOTAL_EXPOSURES % GLOBAL_BATCH != 0:
        fail("total exposures do not divide global batch")
    if TOTAL_STEPS != 3732:
        fail(f"unexpected TOTAL_STEPS={TOTAL_STEPS}")

    rng = np.random.default_rng(SCHEDULE_SEED)
    schedule: list[dict[str, int]] = []
    for pass_id in range(1, PASSES + 1):
        permutation = rng.permutation(ROWS).tolist()
        if len(permutation) != ROWS or len(set(permutation)) != ROWS:
            fail(f"pass {pass_id}: permutation is not one-to-one")
        for position_in_pass, source_id in enumerate(permutation):
            schedule_position = len(schedule)
            schedule.append(
                {
                    "schedule_position": schedule_position,
                    "population_position": int(source_id),
                    "source_id": int(source_id),
                    "global_step": schedule_position // GLOBAL_BATCH + 1,
                    "pass": pass_id,
                    "position_in_pass": int(position_in_pass),
                    "position_in_batch": schedule_position % GLOBAL_BATCH,
                }
            )
    if len(schedule) != TOTAL_EXPOSURES:
        fail(f"schedule rows={len(schedule)} expected={TOTAL_EXPOSURES}")
    if schedule[-1]["global_step"] != TOTAL_STEPS:
        fail(
            f"terminal step={schedule[-1]['global_step']} expected={TOTAL_STEPS}"
        )
    return schedule


def metadata(item: dict[str, int]) -> dict[str, int]:
    return {
        "source_id": int(item["source_id"]),
        "schedule_position": int(item["schedule_position"]),
        "matched_global_step": int(item["global_step"]),
        "matched_pass": int(item["pass"]),
        "position_in_global_batch": int(item["position_in_batch"]),
    }


def finalize(teacher_targets: Path) -> None:
    if not PREPARE_SFT.is_file():
        fail(f"missing existing SFT materializer: {PREPARE_SFT}")

    population = load_and_validate_population()
    teacher_rows = load_teacher_targets(teacher_targets, population)
    schedule = build_schedule()

    write_jsonl(SCHEDULE, schedule)

    sft_input_rows: list[dict[str, Any]] = []
    opd_rows: list[dict[str, Any]] = []

    for item in schedule:
        source_id = int(item["source_id"])
        pop = population[source_id]
        teacher = teacher_rows[source_id]
        source = pop["source"]
        target = teacher["target_translation"]
        prompt = PROMPT_TEMPLATE.format(source=source)
        md = metadata(item)

        sft_input_rows.append(
            {
                "index": source_id,
                "source": source,
                "messages": [{"role": "user", "content": prompt}],
                "target_translation": target,
                "data_source": "pds9952",
                "ability": "translation",
                "parent_index": int(pop["parent_index"]),
                "error_index": int(pop["error_index"]),
                "pair_id": str(pop["pair_id"]),
                "pds_slot": int(pop["pds_slot"]),
                "selection_row_sha256": pop["selection_row_sha256"],
                **md,
            }
        )

        opd_row = {
            "index": source_id,
            "prompt": [{"role": "user", "content": prompt}],
            "data_source": "default",
            "ability": "translation",
            "reward_model": {
                "style": "rule",
                "ground_truth": "UNUSED_ZERO_REWARD",
            },
            "extra_info": {
                "index": source_id,
                "source": source,
                "parent_index": int(pop["parent_index"]),
                "error_index": int(pop["error_index"]),
                "pair_id": str(pop["pair_id"]),
                "pds_slot": int(pop["pds_slot"]),
                "selection_row_sha256": pop["selection_row_sha256"],
            },
            **md,
        }

        leaked = FORBIDDEN_OPD_KEYS.intersection(opd_row)
        leaked_extra = FORBIDDEN_OPD_KEYS.intersection(opd_row["extra_info"])
        if leaked or leaked_extra:
            fail(
                "OPD target leakage: "
                f"top={sorted(leaked)} extra_info={sorted(leaked_extra)}"
            )
        opd_rows.append(opd_row)

    write_jsonl(SFT_INPUT, sft_input_rows)

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

    # Zero-training round-trip matched gate.
    sft = pq.read_table(SFT_PARQUET)
    opd = pq.read_table(OPD_PARQUET)
    if sft.num_rows != TOTAL_EXPOSURES:
        fail(f"SFT parquet rows={sft.num_rows} expected={TOTAL_EXPOSURES}")
    if opd.num_rows != TOTAL_EXPOSURES:
        fail(f"OPD parquet rows={opd.num_rows} expected={TOTAL_EXPOSURES}")

    sft_back = sft.to_pylist()
    opd_back = opd.to_pylist()
    metadata_keys = (
        "source_id",
        "schedule_position",
        "matched_global_step",
        "matched_pass",
        "position_in_global_batch",
    )

    for i, (s, o, sch) in enumerate(zip(sft_back, opd_back, schedule)):
        for key in metadata_keys:
            if int(s[key]) != int(o[key]):
                fail(f"exposure {i}: SFT/OPD metadata mismatch for {key}")
        if int(s["source_id"]) != int(sch["source_id"]):
            fail(f"exposure {i}: schedule/source_id mismatch")

        source_id = int(s["source_id"])
        expected_source = population[source_id]["source"]
        if s.get("source") != expected_source:
            fail(f"exposure {i}: SFT source mismatch")

        s_messages = as_list(s.get("messages"))
        o_prompt = as_list(o.get("prompt"))
        if not isinstance(s_messages, list) or len(s_messages) != 2:
            fail(f"exposure {i}: SFT messages must be [user,assistant]")
        if not isinstance(o_prompt, list) or len(o_prompt) != 1:
            fail(f"exposure {i}: OPD prompt must contain one user message")
        if s_messages[0].get("role") != "user":
            fail(f"exposure {i}: SFT first role != user")
        if s_messages[1].get("role") != "assistant":
            fail(f"exposure {i}: SFT second role != assistant")
        if o_prompt[0].get("role") != "user":
            fail(f"exposure {i}: OPD prompt role != user")

        expected_prompt = PROMPT_TEMPLATE.format(source=expected_source)
        if s_messages[0].get("content") != expected_prompt:
            fail(f"exposure {i}: SFT prompt/source mismatch")
        if o_prompt[0].get("content") != expected_prompt:
            fail(f"exposure {i}: OPD prompt/source mismatch")
        if s_messages[0].get("content") != o_prompt[0].get("content"):
            fail(f"exposure {i}: B/C prompt mismatch")

        expected_target = teacher_rows[source_id]["target_translation"]
        if s_messages[1].get("content") != expected_target:
            fail(f"exposure {i}: SFT assistant target mismatch")
        if s.get("target_translation") != expected_target:
            fail(f"exposure {i}: SFT target_translation mismatch")

        for key in FORBIDDEN_OPD_KEYS:
            if key in o:
                fail(f"exposure {i}: forbidden OPD column present: {key}")
        extra = o.get("extra_info") or {}
        if isinstance(extra, dict):
            for key in FORBIDDEN_OPD_KEYS:
                if key in extra:
                    fail(
                        f"exposure {i}: forbidden OPD extra_info field present: {key}"
                    )

    prepare_manifest = json.loads(
        POPULATION_MANIFEST.read_text(encoding="utf-8")
    )
    manifest = {
        "version": 1,
        "experiment": "PDS9952 Parent-Diverse Low-Budget Adaptation",
        "classification": "ADAPTATION",
        "stage": "finalized_matched_assets",
        "population": prepare_manifest["population"],
        "teacher_targets": {
            "path": str(teacher_targets),
            "sha256": sha256_file(teacher_targets),
            "rows": ROWS,
            "teacher_model": "/workspace/mtpatcher/models/Qwen3-8B",
            "enable_thinking": False,
            "do_sample": False,
            "max_new_tokens": 512,
            "prompt_template": PROMPT_TEMPLATE,
        },
        "schedule": {
            "seed": SCHEDULE_SEED,
            "passes": PASSES,
            "global_batch_size": GLOBAL_BATCH,
            "total_rows": TOTAL_EXPOSURES,
            "total_optimizer_steps": TOTAL_STEPS,
            "path": str(SCHEDULE),
            "sha256": sha256_file(SCHEDULE),
        },
        "matched_assets": {
            "seqkd_input_jsonl": {
                "path": str(SFT_INPUT),
                "sha256": sha256_file(SFT_INPUT),
                "rows": TOTAL_EXPOSURES,
            },
            "seqkd_parquet": {
                "path": str(SFT_PARQUET),
                "sha256": sha256_file(SFT_PARQUET),
                "rows": int(sft.num_rows),
            },
            "opd_parquet": {
                "path": str(OPD_PARQUET),
                "sha256": sha256_file(OPD_PARQUET),
                "rows": int(opd.num_rows),
                "contains_teacher_target": False,
                "contains_reference": False,
            },
        },
        "matched_gate": {
            "status": "PASS",
            "per_exposure_source_order_equal": True,
            "per_exposure_prompt_equal": True,
            "terminal_global_step": int(schedule[-1]["global_step"]),
        },
        "git": {
            "head": git_capture(["rev-parse", "HEAD"]),
            "branch": git_capture(["branch", "--show-current"]),
            "status_short": git_capture(["status", "--short"]),
        },
    }
    write_json(ASSET_MANIFEST, manifest)

    print("PDS9952_FINALIZE=PASS")
    print(f"TEACHER_TARGET_SHA256={sha256_file(teacher_targets)}")
    print(f"SCHEDULE_ROWS={TOTAL_EXPOSURES}")
    print(f"OPTIMIZER_STEPS={TOTAL_STEPS}")
    print(f"SCHEDULE_SHA256={sha256_file(SCHEDULE)}")
    print(f"SEQKD_PARQUET_SHA256={sha256_file(SFT_PARQUET)}")
    print(f"OPD_PARQUET_SHA256={sha256_file(OPD_PARQUET)}")
    print(f"ASSET_MANIFEST={ASSET_MANIFEST}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=("prepare", "finalize"),
    )
    parser.add_argument(
        "--teacher-targets",
        type=Path,
        default=DEFAULT_TEACHER_TARGETS,
        help=(
            "Merged fresh Qwen3-8B Teacher target JSONL. Required by finalize; "
            f"default: {DEFAULT_TEACHER_TARGETS}"
        ),
    )
    args = parser.parse_args()

    if args.stage == "prepare":
        prepare()
    else:
        finalize(args.teacher_targets)


if __name__ == "__main__":
    main()
