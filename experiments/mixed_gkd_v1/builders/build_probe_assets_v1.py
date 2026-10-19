#!/usr/bin/env python3

from __future__ import annotations

import collections
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


ROOT = Path("/workspace/mtpatcher")

POP = (
    ROOT
    / "data/verl_science_pe_pds/pds9952_matched_v1/"
      "pds9952_population_v1.jsonl"
)

FULL_SCHEDULE = (
    ROOT
    / "data/verl_science_pe_pds/pds9952_matched_v1/"
      "source_order_manifest.jsonl"
)

OUT = (
    ROOT
    / "data/verl_science_pe_pds/mixed_gkd_v1_probe"
)

POP_OUT = OUT / "pds2048_population_v1.jsonl"
RANK_OUT = OUT / "pds2048_selection_rank_v1.jsonl"
SCHEDULE_OUT = OUT / "pds2048_probe_schedule_v1.jsonl"
MASK_OUT = OUT / "lambda05_step_mask_v1.jsonl"
MANIFEST_OUT = OUT / "manifest_v1.json"
PASS_OUT = OUT / "PASS"


PDS_DOMAIN = b"MTPATCHER_MIXED_GKD_V1_PDS2048"

MASK_DOMAIN = (
    b"MTPATCHER_MIXED_GKD_V1_LAMBDA05_STEP_MASK"
)

EXPECTED_POPULATION_ROWS = 9952
SELECTED_ROWS = 2048
PASSES = 6
GLOBAL_BATCH = 16
EXPOSURES = SELECTED_ROWS * PASSES
STEPS = EXPOSURES // GLOBAL_BATCH

EXPECTED_POP_FILE_SHA256 = (
    "1491babf8cf2625fcf275cfee1d4a812"
    "a0f8989aad1bd6f0d024f5240b86b937"
)

EXPECTED_FULL_SCHEDULE_SHA256 = (
    "eb6473ac6fd09db7171107f2fcd4d89"
    "b65aa909bd5a1312a2c6dfb93e9d3dfa2"
)

EXPECTED = {
    "selected_source_ids_sha256":
        "8affe38d91fb1e8044d33ffc499754a4"
        "2d9bbcdff0f8f8a372690162285ab0df",

    "selected_rank_records_sha256":
        "86ee131acd15eaf685b8373959f044c4"
        "2adebd27b1949210e09ea9d6c70ebffb",

    "probe_schedule_identity_sha256":
        "2a624958f99cd6d8a1e1cef08c3d20d"
        "90b2a2591a95f57ca6d96ebf747e546f9",

    "probe_source_sequence_sha256":
        "630e8611ac5f766088568840df8915ac"
        "3864d189133d3757e2909af16a02d22c",

    "lambda05_mask_sha256":
        "b3d7f2784401c939fe4dd8fc95c7aa5"
        "fb20c36068d8b3188649b039c872d9e8e",

    "on_policy_step_ids_sha256":
        "ed79a6573cb421e713374f02d621c2cd"
        "906145aa519c8ac45f748fc096ebbd3e",
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def semantic_sha(obj: Any) -> str:
    return hashlib.sha256(
        canonical_json_bytes(obj)
    ).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            f.write("\n")


def main() -> None:
    print("===== INPUT GATES =====")

    if not POP.is_file():
        fail(f"missing population: {POP}")

    if not FULL_SCHEDULE.is_file():
        fail(
            f"missing schedule: {FULL_SCHEDULE}"
        )

    got_pop_sha = file_sha(POP)
    got_schedule_sha = file_sha(FULL_SCHEDULE)

    print(
        "population_file_sha256 =",
        got_pop_sha,
    )

    print(
        "full_schedule_file_sha256 =",
        got_schedule_sha,
    )

    if (
        got_pop_sha
        != EXPECTED_POP_FILE_SHA256
    ):
        fail(
            "PDS9952 population SHA drift"
        )

    if (
        got_schedule_sha
        != EXPECTED_FULL_SCHEDULE_SHA256
    ):
        fail(
            "PDS9952 source schedule SHA drift"
        )

    population = read_jsonl(POP)
    full_schedule = read_jsonl(
        FULL_SCHEDULE
    )

    if (
        len(population)
        != EXPECTED_POPULATION_ROWS
    ):
        fail(
            f"population rows={len(population)}"
        )

    if (
        len(full_schedule)
        != EXPECTED_POPULATION_ROWS * PASSES
    ):
        fail(
            f"schedule rows={len(full_schedule)}"
        )

    # ----------------------------------------
    # PDS2048 selection
    # ----------------------------------------

    ranked = []

    for pos, row in enumerate(population):
        source_id = int(row["source_id"])

        if source_id != pos:
            fail(
                "population source_id/order drift "
                f"pos={pos} source_id={source_id}"
            )

        selection_sha = str(
            row["selection_row_sha256"]
        ).lower()

        if len(selection_sha) != 64:
            fail(
                f"bad selection SHA source_id="
                f"{source_id}"
            )

        int(selection_sha, 16)

        payload = (
            PDS_DOMAIN
            + b"\x00"
            + str(source_id).encode("ascii")
            + b"\x00"
            + selection_sha.encode("ascii")
        )

        digest = hashlib.sha256(
            payload
        ).digest()

        ranked.append(
            {
                "source_id": source_id,
                "selection_row_sha256":
                    selection_sha,
                "rank_sha256":
                    digest.hex(),
                "_digest":
                    digest,
            }
        )

    ranked.sort(
        key=lambda x: (
            x["_digest"],
            x["source_id"],
        )
    )

    selected_ranked = ranked[
        :SELECTED_ROWS
    ]

    selected_ids = {
        int(x["source_id"])
        for x in selected_ranked
    }

    if len(selected_ids) != SELECTED_ROWS:
        fail(
            "selected source IDs are not unique"
        )

    selected_ids_sorted = sorted(
        selected_ids
    )

    selected_source_ids_sha = (
        semantic_sha(
            selected_ids_sorted
        )
    )

    rank_records = [
        {
            "source_id":
                int(x["source_id"]),
            "selection_row_sha256":
                x[
                    "selection_row_sha256"
                ],
            "rank_sha256":
                x["rank_sha256"],
        }
        for x in selected_ranked
    ]

    selected_rank_records_sha = (
        semantic_sha(rank_records)
    )

    if (
        selected_source_ids_sha
        != EXPECTED[
            "selected_source_ids_sha256"
        ]
    ):
        fail(
            "selected source identity drift"
        )

    if (
        selected_rank_records_sha
        != EXPECTED[
            "selected_rank_records_sha256"
        ]
    ):
        fail(
            "selection ranking drift"
        )

    # Population artifact is kept in original
    # PDS9952 source_id order.
    selected_population = [
        dict(row)
        for row in population
        if int(row["source_id"])
        in selected_ids
    ]

    if (
        len(selected_population)
        != SELECTED_ROWS
    ):
        fail(
            "selected population row mismatch"
        )

    # ----------------------------------------
    # Filter frozen exposure schedule
    # ----------------------------------------

    filtered = [
        row
        for row in full_schedule
        if int(row["source_id"])
        in selected_ids
    ]

    if len(filtered) != EXPOSURES:
        fail(
            f"filtered exposures={len(filtered)} "
            f"expected={EXPOSURES}"
        )

    source_counts = collections.Counter(
        int(row["source_id"])
        for row in filtered
    )

    if set(source_counts) != selected_ids:
        fail(
            "filtered source identity mismatch"
        )

    if set(source_counts.values()) != {
        PASSES
    }:
        fail(
            "selected source exposure counts "
            "are not all 6"
        )

    pass_counts = collections.Counter(
        int(row["pass"])
        for row in filtered
    )

    if (
        len(pass_counts) != PASSES
        or set(pass_counts.values())
        != {SELECTED_ROWS}
    ):
        fail(
            f"bad pass counts "
            f"{dict(pass_counts)}"
        )

    original_positions = [
        int(row["schedule_position"])
        for row in filtered
    ]

    if not all(
        a < b
        for a, b in zip(
            original_positions,
            original_positions[1:],
        )
    ):
        fail(
            "filtered schedule lost original "
            "relative ordering"
        )

    probe_schedule = []

    for probe_pos, row in enumerate(
        filtered
    ):
        probe_schedule.append(
            {
                "source_id":
                    int(row["source_id"]),

                "original_schedule_position":
                    int(
                        row[
                            "schedule_position"
                        ]
                    ),

                "original_global_step":
                    int(
                        row[
                            "global_step"
                        ]
                    ),

                "original_pass":
                    int(row["pass"]),

                "original_position_in_pass":
                    int(
                        row[
                            "position_in_pass"
                        ]
                    ),

                "probe_schedule_position":
                    probe_pos,

                "probe_global_step":
                    (
                        probe_pos
                        // GLOBAL_BATCH
                        + 1
                    ),

                "probe_position_in_batch":
                    (
                        probe_pos
                        % GLOBAL_BATCH
                    ),
            }
        )

    if (
        probe_schedule[-1][
            "probe_global_step"
        ]
        != STEPS
    ):
        fail(
            "terminal probe global step drift"
        )

    probe_schedule_identity_sha = (
        semantic_sha(
            probe_schedule
        )
    )

    probe_source_sequence_sha = (
        semantic_sha(
            [
                x["source_id"]
                for x in probe_schedule
            ]
        )
    )

    if (
        probe_schedule_identity_sha
        != EXPECTED[
            "probe_schedule_identity_sha256"
        ]
    ):
        fail(
            "probe schedule identity drift"
        )

    if (
        probe_source_sequence_sha
        != EXPECTED[
            "probe_source_sequence_sha256"
        ]
    ):
        fail(
            "probe source sequence drift"
        )

    # Every new batch must stay within
    # exactly one original pass.
    step_to_pass = {}

    for step in range(
        1,
        STEPS + 1,
    ):
        chunk = probe_schedule[
            (step - 1)
            * GLOBAL_BATCH:
            step
            * GLOBAL_BATCH
        ]

        if (
            len(chunk)
            != GLOBAL_BATCH
        ):
            fail(
                f"bad batch size step={step}"
            )

        pass_set = {
            x["original_pass"]
            for x in chunk
        }

        if len(pass_set) != 1:
            fail(
                "probe batch crosses original "
                f"pass step={step}: "
                f"{pass_set}"
            )

        step_to_pass[step] = next(
            iter(pass_set)
        )

    # ----------------------------------------
    # λ=.5 deterministic whole-step mask
    # ----------------------------------------

    mask_ranked = []

    for step in range(
        1,
        STEPS + 1,
    ):
        payload = (
            MASK_DOMAIN
            + b"\x00"
            + struct.pack(
                ">I",
                step,
            )
        )

        digest = hashlib.sha256(
            payload
        ).digest()

        mask_ranked.append(
            (
                digest,
                step,
            )
        )

    mask_ranked.sort(
        key=lambda x: (
            x[0],
            x[1],
        )
    )

    on_policy_steps = {
        step
        for _, step
        in mask_ranked[
            :STEPS // 2
        ]
    }

    if len(on_policy_steps) != 384:
        fail(
            "ON_POLICY step count drift"
        )

    mask = []

    for step in range(
        1,
        STEPS + 1,
    ):
        mask.append(
            {
                "step":
                    step,

                "branch":
                    (
                        "ON_POLICY"
                        if step
                        in on_policy_steps
                        else "FIXED"
                    ),

                "original_pass":
                    step_to_pass[step],
            }
        )

    branch_counts = (
        collections.Counter(
            x["branch"]
            for x in mask
        )
    )

    if branch_counts != {
        "ON_POLICY": 384,
        "FIXED": 384,
    }:
        fail(
            f"bad branch counts "
            f"{dict(branch_counts)}"
        )

    lambda05_mask_sha = (
        semantic_sha(mask)
    )

    on_policy_ids_sha = semantic_sha(
        sorted(on_policy_steps)
    )

    if (
        lambda05_mask_sha
        != EXPECTED[
            "lambda05_mask_sha256"
        ]
    ):
        fail(
            "lambda=.5 mask identity drift"
        )

    if (
        on_policy_ids_sha
        != EXPECTED[
            "on_policy_step_ids_sha256"
        ]
    ):
        fail(
            "on-policy step identity drift"
        )

    per_pass = collections.defaultdict(
        lambda: collections.Counter()
    )

    for row in mask:
        per_pass[
            int(row["original_pass"])
        ][row["branch"]] += 1

    per_pass_summary = {
        str(k): dict(v)
        for k, v in sorted(
            per_pass.items()
        )
    }

    # ----------------------------------------
    # Materialize immutable child artifacts
    # ----------------------------------------

    if OUT.exists():
        existing = [
            p
            for p in OUT.iterdir()
        ]

        if existing:
            fail(
                "output directory already "
                "contains files; refuse overwrite: "
                f"{OUT}"
            )
    else:
        OUT.mkdir(
            parents=True,
            exist_ok=False,
        )

    write_jsonl(
        POP_OUT,
        selected_population,
    )

    write_jsonl(
        RANK_OUT,
        rank_records,
    )

    write_jsonl(
        SCHEDULE_OUT,
        probe_schedule,
    )

    write_jsonl(
        MASK_OUT,
        mask,
    )

    manifest = {
        "schema_version": 1,
        "status":
            "PASS_MIXED_GKD_V1_PROBE_ASSETS",

        "scientific_contract":
            "Mixed-GKD v1 Scientific Contract",

        "source": {
            "population_path":
                str(POP),

            "population_sha256":
                got_pop_sha,

            "schedule_path":
                str(FULL_SCHEDULE),

            "schedule_sha256":
                got_schedule_sha,
        },

        "selection": {
            "domain":
                PDS_DOMAIN.decode(
                    "ascii"
                ),

            "parent_population":
                EXPECTED_POPULATION_ROWS,

            "selected":
                SELECTED_ROWS,

            "selected_source_ids_sha256":
                selected_source_ids_sha,

            "selected_rank_records_sha256":
                selected_rank_records_sha,
        },

        "schedule": {
            "passes":
                PASSES,

            "global_batch":
                GLOBAL_BATCH,

            "source_exposures":
                EXPOSURES,

            "optimizer_steps":
                STEPS,

            "probe_schedule_identity_sha256":
                probe_schedule_identity_sha,

            "probe_source_sequence_sha256":
                probe_source_sequence_sha,

            "pass_counts":
                dict(
                    sorted(
                        pass_counts.items()
                    )
                ),
        },

        "lambda05": {
            "domain":
                MASK_DOMAIN.decode(
                    "ascii"
                ),

            "on_policy_steps":
                384,

            "fixed_steps":
                384,

            "lambda05_mask_sha256":
                lambda05_mask_sha,

            "on_policy_step_ids_sha256":
                on_policy_ids_sha,

            "branch_counts_by_original_pass":
                per_pass_summary,
        },

        "artifact_files": {},
    }

    # Record physical file hashes separately from
    # semantic identity hashes.
    for path in (
        POP_OUT,
        RANK_OUT,
        SCHEDULE_OUT,
        MASK_OUT,
    ):
        manifest[
            "artifact_files"
        ][path.name] = {
            "path":
                str(path),

            "sha256":
                file_sha(path),
        }

    MANIFEST_OUT.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    PASS_OUT.write_text(
        "MIXED_GKD_V1_PROBE_ASSETS=PASS\n",
        encoding="utf-8",
    )

    print()
    print("===== RESULT =====")

    print(
        "selected_source_ids_sha256 =",
        selected_source_ids_sha,
    )

    print(
        "selected_rank_records_sha256 =",
        selected_rank_records_sha,
    )

    print(
        "probe_schedule_identity_sha256 =",
        probe_schedule_identity_sha,
    )

    print(
        "probe_source_sequence_sha256 =",
        probe_source_sequence_sha,
    )

    print(
        "lambda05_mask_sha256 =",
        lambda05_mask_sha,
    )

    print(
        "on_policy_step_ids_sha256 =",
        on_policy_ids_sha,
    )

    print(
        "branch_counts_by_original_pass =",
        json.dumps(
            per_pass_summary,
            sort_keys=True,
        ),
    )

    print()
    print(
        "manifest =",
        MANIFEST_OUT,
    )

    print(
        "MIXED_GKD_V1_PROBE_ASSET_BUILD=PASS"
    )


if __name__ == "__main__":
    main()
