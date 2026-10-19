#!/usr/bin/env python3

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path("/workspace/mtpatcher")

PROJECT = (
    ROOT
    / "repo/MT-Patcher-Reproduction-Ascend"
)

EXP = (
    PROJECT
    / "experiments/mixed_gkd_v1"
)

BASE = (
    PROJECT
    / "configs/opd/"
      "pds9952_opd_formal_v1.yaml"
)

PROBE_DIR = (
    EXP
    / "configs/probe"
)

SMOKE_DIR = (
    EXP
    / "configs/smoke"
)

ONPOLICY_DATA = (
    ROOT
    / "data/verl_science_pe_pds/"
      "mixed_gkd_v1_probe/"
      "pds2048_onpolicy_v1.parquet"
)

MIXED_DATA = (
    ROOT
    / "data/verl_science_pe_pds/"
      "mixed_gkd_v1_probe/"
      "pds2048_lambda05_v1.parquet"
)

AGENT_CONFIG = (
    EXP
    / "configs/agent_loops_v1.yaml"
)

MANAGER = (
    "mtpatcher_mixed_gkd."
    "agent_loop_manager_v1."
    "MixedGKDv1AgentLoopManagerTQ"
)

RAY_PYTHONPATH = ":".join(
    [
        str(EXP / "src"),
        str(
            ROOT
            / "repo/verl-v0.9.0-mixed-gkd-v1"
        ),
        str(PROJECT),
    ]
)


def set_override(
    cfg: dict,
    key: str,
    value: str,
) -> None:
    prefix = key + "="

    hits = [
        i
        for i, item
        in enumerate(
            cfg["overrides"]
        )
        if str(item).startswith(
            prefix
        )
    ]

    encoded = (
        f"{key}={value}"
    )

    if len(hits) == 0:
        cfg["overrides"].append(
            encoded
        )

    elif len(hits) == 1:
        cfg["overrides"][
            hits[0]
        ] = encoded

    else:
        raise RuntimeError(
            f"duplicate override {key}: "
            f"{hits}"
        )


def materialize(
    *,
    filename: str,
    name: str,
    kind: str,
    execution_status: str,
    cell: str,
    lam: float,
    divergence: str,
    data_path: Path,
    total_steps: int,
    save_freq: int,
    output_root: Path,
) -> None:
    base = yaml.safe_load(
        BASE.read_text(
            encoding="utf-8"
        )
    )

    cfg = deepcopy(
        base
    )

    cfg["name"] = name

    cfg["mixed_gkd_v1"] = {
        "kind":
            kind,

        "execution_status":
            execution_status,

        "scientific_cell":
            cell,

        "trajectory_lambda":
            lam,

        "divergence":
            divergence,

        "support":
            "Teacher Top32 + OTHER",

        "aggregation":
            "token-mean",

        "optimizer_steps":
            total_steps,

        "formal_training_authorized":
            False,

        "notes": (
            "PASS of this config or a smoke run "
            "does not authorize the 768-step "
            "scientific probe."
        ),
    }

    # Mixed-GKD v1 provenance must describe the selected PDS2048
    # schedule, while retaining its PDS9952 selection parent.
    cfg["derivation"] = (
        "mechanical transform of frozen "
        "pds9952_opd_formal_v1 for Mixed-GKD v1 PDS2048"
    )

    cfg["materialized_schedule"] = {
        "rows": 12288,
        "original_source_population": 2048,
        "selection_parent_population": 9952,
        "original_source_passes": 6,
        "dataset_traversals": 1,
        "global_batch_size": 16,
        "optimizer_steps": 768,
    }

    # Frozen PDS2048 materialized schedule.
    set_override(
        cfg,
        "data.train_files",
        f"['{data_path}']",
    )

    set_override(
        cfg,
        "data.val_files",
        f"['{data_path}']",
    )

    set_override(
        cfg,
        "data.train_batch_size",
        "16",
    )

    set_override(
        cfg,
        "data.max_prompt_length",
        "1024",
    )

    set_override(
        cfg,
        "data.max_response_length",
        "256",
    )

    set_override(
        cfg,
        "data.filter_overlong_prompts",
        "False",
    )

    set_override(
        cfg,
        "data.truncation",
        "error",
    )

    set_override(
        cfg,
        "data.shuffle",
        "False",
    )

    # V1 synchronous routing is part of the
    # frozen implementation contract.
    set_override(
        cfg,
        "trainer.use_v1",
        "True",
    )

    set_override(
        cfg,
        "trainer.v1.trainer_mode",
        "sync",
    )

    # Student trajectory settings.
    set_override(
        cfg,
        "actor_rollout_ref.rollout.n",
        "1",
    )

    set_override(
        cfg,
        "actor_rollout_ref.rollout.temperature",
        "1.0",
    )

    set_override(
        cfg,
        "actor_rollout_ref.rollout.top_p",
        "1.0",
    )

    set_override(
        cfg,
        "actor_rollout_ref.rollout.top_k",
        "-1",
    )

    # Mixed-GKD does not use rollout log-probs.
    # The trainer recomputes old_log_probs.
    set_override(
        cfg,
        "actor_rollout_ref.rollout.calculate_log_probs",
        "False",
    )

    # Explicit custom routing / fixed-response plumbing.
    set_override(
        cfg,
        "+actor_rollout_ref.rollout.agent.agent_loop_manager_class",
        MANAGER,
    )

    set_override(
        cfg,
        "actor_rollout_ref.rollout.agent.agent_loop_config_path",
        str(
            AGENT_CONFIG
        ),
    )

    set_override(
        cfg,
        "actor_rollout_ref.rollout.agent.num_workers",
        "8",
    )

    # Explicitly propagate the experiment package
    # and candidate Verl runtime into Ray processes.

    # Frozen projected divergence.
    set_override(
        cfg,
        "distillation.distillation_loss.loss_mode",
        divergence,
    )

    set_override(
        cfg,
        "distillation.distillation_loss.topk",
        "32",
    )

    set_override(
        cfg,
        "distillation.distillation_loss.use_task_rewards",
        "False",
    )

    set_override(
        cfg,
        "distillation.distillation_loss.use_policy_gradient",
        "False",
    )

    set_override(
        cfg,
        "distillation.distillation_loss.loss_max_clamp",
        "null",
    )

    set_override(
        cfg,
        "distillation.distillation_loss.log_prob_min_clamp",
        "null",
    )

    # Preserve the matched PDS optimizer recipe.
    set_override(
        cfg,
        "actor_rollout_ref.actor.optim.lr",
        "1e-06",
    )

    set_override(
        cfg,
        "actor_rollout_ref.actor.ppo_mini_batch_size",
        "16",
    )

    set_override(
        cfg,
        "actor_rollout_ref.actor.ppo_epochs",
        "1",
    )

    set_override(
        cfg,
        "actor_rollout_ref.actor.loss_agg_mode",
        "token-mean",
    )

    # Endpoint / lifecycle.
    set_override(
        cfg,
        "trainer.total_epochs",
        "1",
    )

    set_override(
        cfg,
        "trainer.total_training_steps",
        str(
            total_steps
        ),
    )

    set_override(
        cfg,
        "trainer.save_freq",
        str(
            save_freq
        ),
    )

    set_override(
        cfg,
        "trainer.resume_mode",
        "disable",
    )

    set_override(
        cfg,
        "trainer.val_before_train",
        "False",
    )

    set_override(
        cfg,
        "trainer.test_freq",
        "-1",
    )

    set_override(
        cfg,
        "trainer.experiment_name",
        name,
    )

    set_override(
        cfg,
        "trainer.default_local_dir",
        str(
            output_root
            / "checkpoints"
        ),
    )

    out = (
        PROBE_DIR
        if kind
        == "PROBE_SPEC"
        else SMOKE_DIR
    ) / filename

    out.write_text(
        yaml.safe_dump(
            cfg,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        ),
        encoding="utf-8",
    )

    print(
        "WROTE",
        out,
    )


probe_root = (
    ROOT
    / "runs/science/"
      "mixed_gkd_v1/probe"
)

smoke_root = (
    ROOT
    / "smoke/mixed_gkd_v1/npu"
)


CELLS = [
    (
        "GKD-A_lambda1_fkl.yaml",
        "mixed-gkd-v1-GKD-A-lambda1-fkl",
        "GKD-A",
        1.0,
        "projected_fkl_topk_other",
        ONPOLICY_DATA,
    ),
    (
        "GKD-B_lambda1_jsd01.yaml",
        "mixed-gkd-v1-GKD-B-lambda1-jsd01",
        "GKD-B",
        1.0,
        "projected_jsd01_topk_other",
        ONPOLICY_DATA,
    ),
    (
        "GKD-C_lambda05_fkl.yaml",
        "mixed-gkd-v1-GKD-C-lambda05-fkl",
        "GKD-C",
        0.5,
        "projected_fkl_topk_other",
        MIXED_DATA,
    ),
    (
        "GKD-D_lambda05_jsd01.yaml",
        "mixed-gkd-v1-GKD-D-lambda05-jsd01",
        "GKD-D",
        0.5,
        "projected_jsd01_topk_other",
        MIXED_DATA,
    ),
]


for (
    filename,
    name,
    cell,
    lam,
    divergence,
    data_path,
) in CELLS:
    materialize(
        filename=filename,
        name=name,
        kind="PROBE_SPEC",
        execution_status=
            "SPEC_ONLY_NOT_AUTHORIZED",
        cell=cell,
        lam=lam,
        divergence=divergence,
        data_path=data_path,
        total_steps=768,
        save_freq=768,
        output_root=(
            probe_root
            / cell
        ),
    )


# Two-step mixed smokes cover:
# step 1 = ON_POLICY
# step 2 = FIXED
#
# Together C/D also exercise both projected losses.
SMOKES = [
    (
        "smoke-C_lambda05_fkl_2step.yaml",
        "mixed-gkd-v1-smoke-C-fkl-2step",
        "SMOKE-C",
        "projected_fkl_topk_other",
    ),
    (
        "smoke-D_lambda05_jsd01_2step.yaml",
        "mixed-gkd-v1-smoke-D-jsd01-2step",
        "SMOKE-D",
        "projected_jsd01_topk_other",
    ),
]


for (
    filename,
    name,
    cell,
    divergence,
) in SMOKES:
    materialize(
        filename=filename,
        name=name,
        kind="NPU_SMOKE",
        execution_status=
            "SMOKE_ONLY_ALLOWED_AFTER_PREFLIGHT",
        cell=cell,
        lam=0.5,
        divergence=divergence,
        data_path=MIXED_DATA,
        total_steps=2,
        save_freq=1,
        output_root=(
            smoke_root
            / cell
        ),
    )


print()
print(
    "MIXED_GKD_V1_CONFIG_BUILD=PASS"
)
