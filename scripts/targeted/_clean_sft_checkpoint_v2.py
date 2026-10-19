#!/usr/bin/env python3
"""PyTorch-compatible checkpoint loading and ledger validation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import _clean_sft_parity_core_v1 as c


def torch_load_compat(path: Path, map_location: str = "cpu") -> dict[str, Any]:
    """Load trusted local training state across pre/post weights_only defaults."""
    import torch

    try:
        state = torch.load(
            path, map_location=map_location, weights_only=False
        )
    except TypeError:
        state = torch.load(path, map_location=map_location)
    c.require(isinstance(state, dict), f"checkpoint state is not a dict: {path}")
    return state


def validate_pointer_meta_state(
    pointer: dict[str, Any],
    meta: dict[str, Any],
    state: dict[str, Any],
    contract_sha256: str,
) -> tuple[int, int]:
    fields = ("completed_epoch", "completed_updates", "run_contract_sha256")
    for field in fields:
        c.require(
            pointer.get(field) == meta.get(field) == state.get(field),
            f"CHECKPOINT_STATE_MISMATCH field={field}",
        )
    c.require(
        state.get("run_contract_sha256") == contract_sha256,
        "RESUME_CONTRACT_MISMATCH",
    )
    c.require(
        meta.get("state_format_version") == "kc_sft_checkpoint_v2"
        and state.get("state_format_version") == "kc_sft_checkpoint_v2",
        "unsupported checkpoint state format",
    )
    c.require(meta.get("status") == "PASS", "checkpoint meta status is not PASS")
    epoch = int(state["completed_epoch"])
    updates = int(state["completed_updates"])
    c.require(epoch >= 1 and updates >= 1, "invalid checkpoint completion counters")
    c.require(updates == epoch * 344, "checkpoint update count does not match epoch")
    for key in ("optimizer", "scheduler", "rng"):
        c.require(key in state, f"checkpoint state missing {key}")
    return epoch, updates


def committed_metric_prefix(
    records: list[dict[str, Any]],
    completed_updates: int,
    completed_epoch: int,
) -> list[dict[str, Any]]:
    committed = [
        record for record in records
        if int(record["optimizer_update"]) <= completed_updates
    ]
    actual = [int(record["optimizer_update"]) for record in committed]
    c.require(
        actual == list(range(1, completed_updates + 1)),
        "checkpoint/update metric ledger mismatch",
    )
    if committed:
        c.require(
            int(committed[-1]["epoch"]) == completed_epoch,
            "checkpoint epoch differs from final committed metric epoch",
        )
    return committed
