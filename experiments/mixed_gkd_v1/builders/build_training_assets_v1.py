#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer


ROOT = Path("/workspace/mtpatcher")

PARENT = (
    ROOT
    / "data/verl_science_pe_pds/pds9952_matched_v1"
)

PROBE = (
    ROOT
    / "data/verl_science_pe_pds/mixed_gkd_v1_probe"
)

PROJECT = (
    ROOT
    / "repo/MT-Patcher-Reproduction-Ascend"
)

PROBE_MANIFEST = PROBE / "manifest_v1.json"
PROBE_POP = PROBE / "pds2048_population_v1.jsonl"
PROBE_SCHEDULE = PROBE / "pds2048_probe_schedule_v1.jsonl"
MASK = PROBE / "lambda05_step_mask_v1.jsonl"

PARENT_MANIFEST = PARENT / "asset_manifest.json"

TOKENIZER_PATH = (
    ROOT
    / "runs/science/pe_pds_v1/"
      "pds9952_stage_entry_v1/"
      "pe_opd_step4422_merged_hf"
)

FIXED_BANK = (
    PROBE
    / "fixed_response_bank_v1.jsonl"
)

ONPOLICY_PARQUET = (
    PROBE
    / "pds2048_onpolicy_v1.parquet"
)

MIXED_PARQUET = (
    PROBE
    / "pds2048_lambda05_v1.parquet"
)

MANIFEST_OUT = (
    PROBE
    / "training_assets_manifest_v1.json"
)

PASS_OUT = (
    PROBE
    / "TRAINING_ASSETS_PASS"
)

EXPECTED_FIXED_BANK_SEMANTIC_SHA = (
    "2fc2c8066e46a7a69775de51e17d21f"
    "47b3abb84c8ba919610ca5135e1f421de"
)

EXPECTED_RESPONSE_SEQUENCE_SHA = (
    "7ebee0da7e032f737319ad306b6eb321"
    "a1d12a3f41d6bfddd2025b3847cce6a2"
)

EXPECTED_ROWS = 12288
EXPECTED_POP = 2048
EXPECTED_STEPS = 768
GLOBAL_BATCH = 16


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


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
                + "\n"
            )


def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def semantic_sha(obj: Any) -> str:
    return hashlib.sha256(
        canonical_bytes(obj)
    ).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def ensure_absent(path: Path) -> None:
    if path.exists():
        fail(
            f"refuse overwrite: {path}"
        )


def main() -> None:
    for path in (
        FIXED_BANK,
        ONPOLICY_PARQUET,
        MIXED_PARQUET,
        MANIFEST_OUT,
        PASS_OUT,
    ):
        ensure_absent(path)

    print("===== LOAD FROZEN MANIFESTS =====")

    probe_manifest = json.loads(
        PROBE_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    parent_manifest = json.loads(
        PARENT_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if (
        probe_manifest["status"]
        != "PASS_MIXED_GKD_V1_PROBE_ASSETS"
    ):
        fail(
            "probe manifest status drift"
        )

    if (
        parent_manifest[
            "matched_gate"
        ]["status"]
        != "PASS"
    ):
        fail(
            "parent PDS9952 manifest not PASS"
        )

    parent_opd = Path(
        parent_manifest[
            "matched_assets"
        ][
            "opd_parquet"
        ][
            "path"
        ]
    )

    targets_path = Path(
        parent_manifest[
            "teacher_targets"
        ][
            "path"
        ]
    )

    if (
        file_sha(parent_opd)
        != parent_manifest[
            "matched_assets"
        ][
            "opd_parquet"
        ][
            "sha256"
        ]
    ):
        fail(
            "parent OPD parquet SHA drift"
        )

    if (
        file_sha(targets_path)
        != parent_manifest[
            "teacher_targets"
        ][
            "sha256"
        ]
    ):
        fail(
            "Teacher target asset SHA drift"
        )

    print(
        "parent_opd_sha256 =",
        file_sha(parent_opd),
    )

    print(
        "teacher_targets_sha256 =",
        file_sha(targets_path),
    )

    # --------------------------------------------------
    # Frozen response bank
    # --------------------------------------------------

    population = read_jsonl(
        PROBE_POP
    )

    targets = read_jsonl(
        targets_path
    )

    if len(population) != EXPECTED_POP:
        fail(
            f"probe population={len(population)}"
        )

    if len(targets) != 9952:
        fail(
            f"Teacher targets={len(targets)}"
        )

    targets_by_id = {}

    for row in targets:
        sid = int(row["index"])

        if sid in targets_by_id:
            fail(
                f"duplicate Teacher source_id={sid}"
            )

        targets_by_id[sid] = row

    if set(targets_by_id) != set(
        range(9952)
    ):
        fail(
            "Teacher target index set drift"
        )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            TOKENIZER_PATH,
            local_files_only=True,
        )
    )

    eos = tokenizer.eos_token_id

    if eos is None:
        fail(
            "tokenizer EOS missing"
        )

    eos = int(eos)

    response_bank = []

    for pop in population:
        sid = int(
            pop["source_id"]
        )

        target_row = (
            targets_by_id[sid]
        )

        if (
            target_row["source"]
            != pop["source"]
        ):
            fail(
                f"source mismatch sid={sid}"
            )

        target = target_row[
            "target_translation"
        ]

        if (
            not isinstance(target, str)
            or not target
            or target != target.strip()
        ):
            fail(
                f"bad frozen target sid={sid}"
            )

        target_ids = tokenizer(
            target,
            add_special_tokens=False,
        )["input_ids"]

        if (
            not target_ids
            or not all(
                isinstance(x, int)
                for x in target_ids
            )
        ):
            fail(
                f"bad target IDs sid={sid}"
            )

        response_ids = (
            list(target_ids)
            + [eos]
        )

        if len(response_ids) > 256:
            fail(
                "fixed response exceeds 256 "
                f"sid={sid} "
                f"len={len(response_ids)}"
            )

        response_bank.append(
            {
                "source_id":
                    sid,

                "target_utf8_sha256":
                    hashlib.sha256(
                        target.encode(
                            "utf-8"
                        )
                    ).hexdigest(),

                "response_ids":
                    response_ids,

                "response_len":
                    len(response_ids),
            }
        )

    fixed_bank_semantic_sha = (
        semantic_sha(
            response_bank
        )
    )

    response_sequence_sha = (
        semantic_sha(
            [
                row["response_ids"]
                for row
                in response_bank
            ]
        )
    )

    print()
    print("===== FIXED RESPONSE BANK =====")

    print(
        "fixed_response_bank_sha256 =",
        fixed_bank_semantic_sha,
    )

    print(
        "fixed_response_id_sequence_sha256 =",
        response_sequence_sha,
    )

    if (
        fixed_bank_semantic_sha
        != EXPECTED_FIXED_BANK_SEMANTIC_SHA
    ):
        fail(
            "fixed response bank "
            "semantic SHA drift"
        )

    if (
        response_sequence_sha
        != EXPECTED_RESPONSE_SEQUENCE_SHA
    ):
        fail(
            "fixed response sequence SHA drift"
        )

    write_jsonl(
        FIXED_BANK,
        response_bank,
    )

    # --------------------------------------------------
    # Frozen probe rows
    # --------------------------------------------------

    schedule = read_jsonl(
        PROBE_SCHEDULE
    )

    mask = read_jsonl(
        MASK
    )

    if len(schedule) != EXPECTED_ROWS:
        fail(
            f"probe schedule={len(schedule)}"
        )

    if len(mask) != EXPECTED_STEPS:
        fail(
            f"mask rows={len(mask)}"
        )

    mask_by_step = {
        int(row["step"]):
            str(row["branch"])
        for row in mask
    }

    if set(mask_by_step) != set(
        range(
            1,
            EXPECTED_STEPS + 1,
        )
    ):
        fail(
            "lambda mask step set drift"
        )

    parent_df = pd.read_parquet(
        parent_opd
    )

    if len(parent_df) != 59712:
        fail(
            f"parent OPD rows="
            f"{len(parent_df)}"
        )

    positions = [
        int(
            row[
                "original_schedule_position"
            ]
        )
        for row in schedule
    ]

    base = (
        parent_df
        .iloc[positions]
        .copy()
        .reset_index(drop=True)
    )

    expected_source_ids = [
        int(row["source_id"])
        for row in schedule
    ]

    actual_source_ids = [
        int(x)
        for x in base[
            "source_id"
        ].tolist()
    ]

    if (
        actual_source_ids
        != expected_source_ids
    ):
        fail(
            "parent OPD filtered "
            "source sequence drift"
        )

    # Verify original scheduling columns before
    # renaming them as parent provenance.
    for i, row in enumerate(
        schedule
    ):
        if (
            int(
                base.iloc[i][
                    "schedule_position"
                ]
            )
            != int(
                row[
                    "original_schedule_position"
                ]
            )
        ):
            fail(
                "parent schedule_position "
                f"drift row={i}"
            )

        if (
            int(
                base.iloc[i][
                    "matched_global_step"
                ]
            )
            != int(
                row[
                    "original_global_step"
                ]
            )
        ):
            fail(
                "parent global_step "
                f"drift row={i}"
            )

    base = base.rename(
        columns={
            "schedule_position":
                "parent_schedule_position",

            "matched_global_step":
                "parent_global_step",

            "matched_pass":
                "parent_pass",

            "position_in_global_batch":
                "parent_position_in_batch",
        }
    )

    base[
        "probe_schedule_position"
    ] = [
        int(
            x[
                "probe_schedule_position"
            ]
        )
        for x in schedule
    ]

    base[
        "probe_global_step"
    ] = [
        int(
            x[
                "probe_global_step"
            ]
        )
        for x in schedule
    ]

    base[
        "probe_position_in_batch"
    ] = [
        int(
            x[
                "probe_position_in_batch"
            ]
        )
        for x in schedule
    ]

    base[
        "probe_original_pass"
    ] = [
        int(
            x[
                "original_pass"
            ]
        )
        for x in schedule
    ]

    # --------------------------------------------------
    # λ = 1.0 dataset
    # --------------------------------------------------

    onpolicy = base.copy()

    onpolicy[
        "mixed_gkd_branch"
    ] = "ON_POLICY"

    onpolicy[
        "agent_name"
    ] = "single_turn_agent"

    onpolicy[
        "mixed_gkd_lambda"
    ] = 1.0

    # --------------------------------------------------
    # λ = .5 dataset
    # --------------------------------------------------

    mixed = base.copy()

    branches = [
        mask_by_step[
            int(step)
        ]
        for step in mixed[
            "probe_global_step"
        ].tolist()
    ]

    if (
        branches.count(
            "ON_POLICY"
        )
        != 384 * GLOBAL_BATCH
    ):
        fail(
            "ON_POLICY exposure count drift"
        )

    if (
        branches.count(
            "FIXED"
        )
        != 384 * GLOBAL_BATCH
    ):
        fail(
            "FIXED exposure count drift"
        )

    mixed[
        "mixed_gkd_branch"
    ] = branches

    mixed[
        "agent_name"
    ] = [
        (
            "single_turn_agent"
            if branch
            == "ON_POLICY"
            else
            "mtpatcher_fixed_response_v1"
        )
        for branch
        in branches
    ]

    mixed[
        "mixed_gkd_lambda"
    ] = 0.5

    # No Teacher target or response IDs may enter
    # either rollout parquet.
    forbidden = {
        "target_translation",
        "reference",
        "post_edit",
        "student_translation",
        "historical_target_translation",
        "provenance_historical_target_translation",
        "fixed_response_ids",
        "response_ids",
    }

    for name, df in (
        ("onpolicy", onpolicy),
        ("mixed", mixed),
    ):
        leaked = (
            forbidden
            .intersection(
                set(df.columns)
            )
        )

        if leaked:
            fail(
                f"{name} top-level target leakage: "
                f"{sorted(leaked)}"
            )

        # Exact whole-batch routing.
        for step in range(
            1,
            EXPECTED_STEPS + 1,
        ):
            part = df[
                df[
                    "probe_global_step"
                ]
                == step
            ]

            if len(part) != GLOBAL_BATCH:
                fail(
                    f"{name}: step {step} "
                    f"rows={len(part)}"
                )

            if (
                part[
                    "probe_position_in_batch"
                ]
                .tolist()
                != list(
                    range(
                        GLOBAL_BATCH
                    )
                )
            ):
                fail(
                    f"{name}: bad positions "
                    f"step={step}"
                )

            branches_step = set(
                part[
                    "mixed_gkd_branch"
                ].tolist()
            )

            agents_step = set(
                part[
                    "agent_name"
                ].tolist()
            )

            if len(branches_step) != 1:
                fail(
                    f"{name}: mixed branch "
                    f"inside step={step}"
                )

            if len(agents_step) != 1:
                fail(
                    f"{name}: mixed agents "
                    f"inside step={step}"
                )

    # --------------------------------------------------
    # Frozen semantic route identities
    # --------------------------------------------------

    def route_records(df):
        return [
            {
                "source_id":
                    int(row.source_id),

                "probe_schedule_position":
                    int(
                        row.probe_schedule_position
                    ),

                "probe_global_step":
                    int(
                        row.probe_global_step
                    ),

                "probe_position_in_batch":
                    int(
                        row.probe_position_in_batch
                    ),

                "branch":
                    str(
                        row.mixed_gkd_branch
                    ),

                "agent_name":
                    str(
                        row.agent_name
                    ),
            }
            for row
            in df.itertuples(
                index=False
            )
        ]

    onpolicy_route_sha = (
        semantic_sha(
            route_records(
                onpolicy
            )
        )
    )

    mixed_route_sha = (
        semantic_sha(
            route_records(
                mixed
            )
        )
    )

    print()
    print("===== ROUTING IDENTITIES =====")

    print(
        "onpolicy_route_sha256 =",
        onpolicy_route_sha,
    )

    print(
        "lambda05_route_sha256 =",
        mixed_route_sha,
    )

    # --------------------------------------------------
    # Materialize
    # --------------------------------------------------

    onpolicy.to_parquet(
        ONPOLICY_PARQUET,
        index=False,
    )

    mixed.to_parquet(
        MIXED_PARQUET,
        index=False,
    )

    manifest = {
        "schema_version": 1,

        "status":
            "PASS_MIXED_GKD_V1_TRAINING_ASSETS",

        "fixed_response_semantics": {
            "definition":
                "tokenize(frozen target_translation.strip(), "
                "add_special_tokens=False) + [EOS]",

            "rows":
                EXPECTED_POP,

            "semantic_sha256":
                fixed_bank_semantic_sha,

            "response_id_sequence_sha256":
                response_sequence_sha,

            "physical_file_sha256":
                file_sha(
                    FIXED_BANK
                ),

            "path":
                str(
                    FIXED_BANK
                ),
        },

        "routing": {
            "lambda1": {
                "rows":
                    EXPECTED_ROWS,

                "optimizer_steps":
                    EXPECTED_STEPS,

                "branch":
                    "ON_POLICY",

                "agent_name":
                    "single_turn_agent",

                "semantic_route_sha256":
                    onpolicy_route_sha,

                "path":
                    str(
                        ONPOLICY_PARQUET
                    ),

                "file_sha256":
                    file_sha(
                        ONPOLICY_PARQUET
                    ),
            },

            "lambda05": {
                "rows":
                    EXPECTED_ROWS,

                "optimizer_steps":
                    EXPECTED_STEPS,

                "on_policy_steps":
                    384,

                "fixed_steps":
                    384,

                "on_policy_exposures":
                    384 * GLOBAL_BATCH,

                "fixed_exposures":
                    384 * GLOBAL_BATCH,

                "semantic_route_sha256":
                    mixed_route_sha,

                "mask_semantic_sha256":
                    probe_manifest[
                        "lambda05"
                    ][
                        "lambda05_mask_sha256"
                    ],

                "path":
                    str(
                        MIXED_PARQUET
                    ),

                "file_sha256":
                    file_sha(
                        MIXED_PARQUET
                    ),
            },
        },

        "parent": {
            "opd_parquet":
                str(
                    parent_opd
                ),

            "opd_parquet_sha256":
                file_sha(
                    parent_opd
                ),

            "teacher_targets":
                str(
                    targets_path
                ),

            "teacher_targets_sha256":
                file_sha(
                    targets_path
                ),
        },
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
        "MIXED_GKD_V1_TRAINING_ASSETS=PASS\n",
        encoding="utf-8",
    )

    print()
    print("===== PHYSICAL ASSETS =====")

    for path in (
        FIXED_BANK,
        ONPOLICY_PARQUET,
        MIXED_PARQUET,
        MANIFEST_OUT,
    ):
        print(
            path.name,
            path.stat().st_size,
            file_sha(path),
        )

    print()
    print(
        "MIXED_GKD_V1_TRAINING_ASSET_BUILD=PASS"
    )


if __name__ == "__main__":
    main()
