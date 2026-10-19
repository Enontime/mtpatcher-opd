#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml


ROOT = Path(
    "/workspace/mtpatcher"
)

PROJECT = (
    ROOT
    / "repo/MT-Patcher-Reproduction-Ascend"
)

EXP = (
    PROJECT
    / "experiments/mixed_gkd_v1"
)

NEW = (
    ROOT
    / "repo/verl-v0.9.0-mixed-gkd-v1"
)

FORMAL = (
    ROOT
    / "repo/verl-v0.9.0-matched20k-v2-formal"
)

DATA = (
    ROOT
    / "data/verl_science_pe_pds/"
      "mixed_gkd_v1_probe"
)

TRAINING_MANIFEST = (
    DATA
    / "training_assets_manifest_v1.json"
)

EXPECTED = {
    "fixed_bank_file":
        "c995f56b2533bef831c4b88953910c06"
        "c0421acdb59250667aed86881e4f8711",

    "onpolicy_parquet":
        "65495714a6c30ad23bce75712318cd883"
        "5ced7ec93a0d88944f31f2fcf19444a",

    "lambda05_parquet":
        "e75120fcedc8bc872df2537cf71490120"
        "e9b52e0b447ba72f5891b5ac9c7301c",

    "lambda05_mask":
        "7d94f521893538bfe10a3481109c07f5"
        "ebd226d84d12cbdc86ea9d39d6db70bd",

    "lambda05_mask_semantic":
        "b3d7f2784401c939fe4dd8fc95c7aa5"
        "fb20c36068d8b3188649b039c872d9e8e",

    "lambda1_route":
        "5328928c2d376d3437753d01f8e4c17"
        "c5163096dc633e3280bac6b9838d410d5",

    "lambda05_route":
        "cdaa103352a77a5798dcef6d162ec6f8"
        "cdc346bef75a56926ef1207438f47de6",
}


def file_sha(
    path: Path,
) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def override(
    cfg: dict,
    key: str,
) -> str:
    hits = [
        str(item).split(
            "=",
            1,
        )[1]
        for item
        in cfg["overrides"]
        if str(item).startswith(
            key + "="
        )
    ]

    if len(hits) != 1:
        raise AssertionError(
            f"{key}: expected one override, "
            f"got {hits}"
        )

    return hits[0].strip(
        "'\""
    )


print(
    "===== [PREFLIGHT] FROZEN ASSETS ====="
)

manifest = json.loads(
    TRAINING_MANIFEST.read_text(
        encoding="utf-8"
    )
)

assert (
    manifest["status"]
    ==
    "PASS_MIXED_GKD_V1_TRAINING_ASSETS"
)

checks = [
    (
        Path(
            manifest[
                "fixed_response_semantics"
            ]["path"]
        ),
        EXPECTED[
            "fixed_bank_file"
        ],
        "fixed_response_bank",
    ),
    (
        Path(
            manifest[
                "routing"
            ]["lambda1"]["path"]
        ),
        EXPECTED[
            "onpolicy_parquet"
        ],
        "lambda1_parquet",
    ),
    (
        Path(
            manifest[
                "routing"
            ]["lambda05"]["path"]
        ),
        EXPECTED[
            "lambda05_parquet"
        ],
        "lambda05_parquet",
    ),
    (
        DATA
        / "lambda05_step_mask_v1.jsonl",
        EXPECTED[
            "lambda05_mask"
        ],
        "lambda05_mask",
    ),
]

for path, expected, label in checks:
    got = file_sha(
        path
    )

    print(
        label,
        got,
    )

    assert got == expected


assert (
    manifest["routing"][
        "lambda1"
    ][
        "semantic_route_sha256"
    ]
    == EXPECTED[
        "lambda1_route"
    ]
)

assert (
    manifest["routing"][
        "lambda05"
    ][
        "semantic_route_sha256"
    ]
    == EXPECTED[
        "lambda05_route"
    ]
)

assert (
    manifest["routing"][
        "lambda05"
    ][
        "mask_semantic_sha256"
    ]
    == EXPECTED[
        "lambda05_mask_semantic"
    ]
)


print()
print(
    "===== [PREFLIGHT] WORKTREES ====="
)

formal_status = subprocess.check_output(
    [
        "git",
        "-C",
        str(FORMAL),
        "status",
        "--porcelain",
    ],
    text=True,
)

assert (
    formal_status.strip()
    == ""
)

print(
    "formal_verl_clean = PASS"
)

new_head = subprocess.check_output(
    [
        "git",
        "-C",
        str(NEW),
        "rev-parse",
        "HEAD",
    ],
    text=True,
).strip()

print(
    "candidate_verl_head =",
    new_head,
)

diff = subprocess.check_output(
    [
        "git",
        "-C",
        str(NEW),
        "diff",
        "--binary",
        "--no-ext-diff",
        "HEAD",
        "--",
        "verl/trainer/distillation/fsdp/losses.py",
        "verl/trainer/distillation/losses.py",
    ]
)

assert diff

print(
    "candidate_verl_production_diff_sha256 =",
    hashlib.sha256(
        diff
    ).hexdigest(),
)


print()
print(
    "===== [PREFLIGHT] CONFIG CELLS ====="
)

specs = {
    "GKD-A":
        (
            EXP
            / "configs/probe/"
              "GKD-A_lambda1_fkl.yaml",
            1.0,
            "projected_fkl_topk_other",
            "pds2048_onpolicy_v1.parquet",
            768,
        ),

    "GKD-B":
        (
            EXP
            / "configs/probe/"
              "GKD-B_lambda1_jsd01.yaml",
            1.0,
            "projected_jsd01_topk_other",
            "pds2048_onpolicy_v1.parquet",
            768,
        ),

    "GKD-C":
        (
            EXP
            / "configs/probe/"
              "GKD-C_lambda05_fkl.yaml",
            0.5,
            "projected_fkl_topk_other",
            "pds2048_lambda05_v1.parquet",
            768,
        ),

    "GKD-D":
        (
            EXP
            / "configs/probe/"
              "GKD-D_lambda05_jsd01.yaml",
            0.5,
            "projected_jsd01_topk_other",
            "pds2048_lambda05_v1.parquet",
            768,
        ),
}

manager = (
    "mtpatcher_mixed_gkd."
    "agent_loop_manager_v1."
    "MixedGKDv1AgentLoopManagerTQ"
)

for (
    cell,
    (
        path,
        lam,
        loss,
        data_suffix,
        steps,
    ),
) in specs.items():
    cfg = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    meta = cfg[
        "mixed_gkd_v1"
    ]

    assert (
        meta["kind"]
        == "PROBE_SPEC"
    )

    assert (
        meta[
            "execution_status"
        ]
        ==
        "SPEC_ONLY_NOT_AUTHORIZED"
    )

    assert (
        meta[
            "formal_training_authorized"
        ]
        is False
    )

    assert float(
        meta[
            "trajectory_lambda"
        ]
    ) == lam

    assert (
        meta["divergence"]
        == loss
    )

    train = override(
        cfg,
        "data.train_files",
    )

    assert data_suffix in train

    assert (
        int(
            override(
                cfg,
                "data.train_batch_size",
            )
        )
        == 16
    )

    assert (
        override(
            cfg,
            "data.shuffle",
        )
        == "False"
    )

    assert (
        int(
            override(
                cfg,
                "trainer.total_training_steps",
            )
        )
        == steps
    )

    assert (
        override(
            cfg,
            "distillation.distillation_loss.loss_mode",
        )
        == loss
    )

    assert (
        int(
            override(
                cfg,
                "distillation.distillation_loss.topk",
            )
        )
        == 32
    )

    assert (
        override(
            cfg,
            "actor_rollout_ref.actor.loss_agg_mode",
        )
        == "token-mean"
    )

    assert (
        override(
            cfg,
            "actor_rollout_ref.rollout.calculate_log_probs",
        )
        == "False"
    )

    assert (
        override(
            cfg,
            "trainer.use_v1",
        )
        == "True"
    )

    assert (
        override(
            cfg,
            "trainer.v1.trainer_mode",
        )
        == "sync"
    )

    assert (
        override(
            cfg,
            "+actor_rollout_ref.rollout.agent.agent_loop_manager_class",
        )
        == manager
    )

    out = override(
        cfg,
        "trainer.default_local_dir",
    )

    assert (
        "/runs/science/"
        "mixed_gkd_v1/probe/"
        in out
    )

    print(
        cell,
        "PASS",
        f"lambda={lam}",
        loss,
    )


print()
print(
    "===== [PREFLIGHT] SMOKE SPECS ====="
)

for filename, expected_loss in (
    (
        "smoke-C_lambda05_fkl_2step.yaml",
        "projected_fkl_topk_other",
    ),
    (
        "smoke-D_lambda05_jsd01_2step.yaml",
        "projected_jsd01_topk_other",
    ),
):
    path = (
        EXP
        / "configs/smoke"
        / filename
    )

    cfg = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    meta = cfg[
        "mixed_gkd_v1"
    ]

    assert (
        meta["kind"]
        == "NPU_SMOKE"
    )

    assert (
        int(
            override(
                cfg,
                "trainer.total_training_steps",
            )
        )
        == 2
    )

    assert (
        override(
            cfg,
            "distillation.distillation_loss.loss_mode",
        )
        == expected_loss
    )

    train = override(
        cfg,
        "data.train_files",
    )

    assert (
        "pds2048_lambda05_v1.parquet"
        in train
    )

    out = override(
        cfg,
        "trainer.default_local_dir",
    )

    assert (
        "/smoke/mixed_gkd_v1/npu/"
        in out
    )

    print(
        filename,
        "PASS",
    )


print()
print(
    "MIXED_GKD_V1_CONFIG_PREFLIGHT=PASS"
)

print(
    "FORMAL_PROBE_EXECUTION="
    "BLOCKED_NOT_AUTHORIZED"
)

# Final-evaluation reservation is a formal execution gate.
import hashlib as _hashlib
import json as _json
from pathlib import Path as _Path

_final_eval_manifest = (
    _Path(__file__).resolve().parents[1]
    / "manifests"
    / "FINAL_EVAL_RESERVED_v1.json"
)

if not _final_eval_manifest.is_file():
    raise RuntimeError(
        "FINAL_EVAL_RESERVED_v1 manifest missing"
    )

_final_eval = _json.loads(
    _final_eval_manifest.read_text(
        encoding="utf-8"
    )
)

if (
    _final_eval.get("status")
    != "FINAL_EVAL_RESERVED_v1"
):
    raise RuntimeError(
        "FINAL_EVAL_RESERVED_v1 status drift"
    )

_policy = _final_eval["evaluation_policy"]

if _policy["endpoint_step"] != 768:
    raise RuntimeError(
        "final-eval endpoint drift"
    )

if _policy[
    "evaluate_only_after_all_four_cells_complete"
] is not True:
    raise RuntimeError(
        "final-eval ordering policy drift"
    )

if _policy[
    "checkpoint_selection_from_final_eval"
] is not False:
    raise RuntimeError(
        "final-eval checkpoint-selection policy drift"
    )

for _item in _final_eval["frozen_inputs"]:
    _path = _Path(_item["path"])

    if not _path.is_file():
        raise RuntimeError(
            f"final-eval input missing: {_path}"
        )

    _h = _hashlib.sha256()

    with _path.open("rb") as _f:
        while True:
            _chunk = _f.read(
                8 * 1024 * 1024
            )

            if not _chunk:
                break

            _h.update(_chunk)

    _got = _h.hexdigest()

    if _got != _item["sha256"]:
        raise RuntimeError(
            "final-eval SHA drift: "
            f"{_path}: {_got}"
        )

print(
    "FINAL_EVAL_RESERVED_v1=PASS"
)
