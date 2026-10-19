from __future__ import annotations

import numpy as np
import pytest
import torch

from verl.utils import (
    tensordict_utils as tu,
)

from mtpatcher_mixed_gkd.agent_loop_manager_v1 import (
    EXPECTED_MASK_FILE_SHA256,
    MASK_PATH,
    _file_sha256,
    _load_lambda05_mask,
    validate_mixed_gkd_train_batch,
)


BATCH = 16


def make_batch(
    *,
    step: int,
    lam: float,
    branch: str,
    agent: str,
    schedule_offset: int = 0,
):
    start = (
        step - 1
    ) * BATCH

    raw = {
        "dummy_tensor":
            torch.zeros(
                BATCH,
                1,
                dtype=torch.uint8,
            ),

        "probe_global_step":
            np.asarray(
                [step] * BATCH,
                dtype=object,
            ),

        "probe_position_in_batch":
            np.asarray(
                list(
                    range(BATCH)
                ),
                dtype=object,
            ),

        "probe_schedule_position":
            np.asarray(
                list(
                    range(
                        start + schedule_offset,
                        start + schedule_offset + BATCH,
                    )
                ),
                dtype=object,
            ),

        "mixed_gkd_lambda":
            np.asarray(
                [lam] * BATCH,
                dtype=object,
            ),

        "mixed_gkd_branch":
            np.asarray(
                [branch] * BATCH,
                dtype=object,
            ),

        "agent_name":
            np.asarray(
                [agent] * BATCH,
                dtype=object,
            ),
    }

    batch = tu.get_tensordict(
        raw
    )

    tu.assign_non_tensor_data(
        batch,
        "global_steps",
        step,
    )

    return batch


def test_frozen_lambda05_mask_identity():
    assert (
        _file_sha256(
            MASK_PATH
        )
        == EXPECTED_MASK_FILE_SHA256
    )

    mask = (
        _load_lambda05_mask()
    )

    assert len(mask) == 768
    assert (
        list(mask.values())
        .count("ON_POLICY")
        == 384
    )
    assert (
        list(mask.values())
        .count("FIXED")
        == 384
    )

    assert mask[1] == "ON_POLICY"
    assert mask[2] == "FIXED"


def test_lambda1_valid_batch():
    batch = make_batch(
        step=2,
        lam=1.0,
        branch="ON_POLICY",
        agent="single_turn_agent",
    )

    validate_mixed_gkd_train_batch(
        batch
    )


def test_lambda05_onpolicy_valid():
    batch = make_batch(
        step=1,
        lam=0.5,
        branch="ON_POLICY",
        agent="single_turn_agent",
    )

    validate_mixed_gkd_train_batch(
        batch
    )


def test_lambda05_fixed_valid():
    batch = make_batch(
        step=2,
        lam=0.5,
        branch="FIXED",
        agent=
            "mtpatcher_fixed_response_v1",
    )

    validate_mixed_gkd_train_batch(
        batch
    )


def test_global_step_mismatch_fails():
    batch = make_batch(
        step=2,
        lam=0.5,
        branch="FIXED",
        agent=
            "mtpatcher_fixed_response_v1",
    )

    tu.assign_non_tensor_data(
        batch,
        "global_steps",
        3,
    )

    with pytest.raises(
        RuntimeError,
        match="probe_global_step",
    ):
        validate_mixed_gkd_train_batch(
            batch
        )


def test_schedule_shift_fails():
    batch = make_batch(
        step=2,
        lam=0.5,
        branch="FIXED",
        agent=
            "mtpatcher_fixed_response_v1",
        schedule_offset=1,
    )

    with pytest.raises(
        RuntimeError,
        match="schedule position",
    ):
        validate_mixed_gkd_train_batch(
            batch
        )


def test_wrong_branch_fails():
    batch = make_batch(
        step=2,
        lam=0.5,
        branch="ON_POLICY",
        agent="single_turn_agent",
    )

    with pytest.raises(
        RuntimeError,
        match="frozen branch mismatch",
    ):
        validate_mixed_gkd_train_batch(
            batch
        )


def test_wrong_agent_fails():
    batch = make_batch(
        step=2,
        lam=0.5,
        branch="FIXED",
        agent="single_turn_agent",
    )

    with pytest.raises(
        RuntimeError,
        match="agent routing mismatch",
    ):
        validate_mixed_gkd_train_batch(
            batch
        )


def test_global_steps_nontensor_scalar_contract():
    batch = make_batch(
        step=7,
        lam=1.0,
        branch="ON_POLICY",
        agent="single_turn_agent",
    )

    # This is the actual tensordict behavior observed in
    # the current frozen environment: NonTensorData scalar
    # metadata is unwrapped on __getitem__.
    assert batch[
        "global_steps"
    ] == 7

    validate_mixed_gkd_train_batch(
        batch
    )


def test_probe_metadata_per_row_container_contract():
    batch = make_batch(
        step=7,
        lam=1.0,
        branch="ON_POLICY",
        agent="single_turn_agent",
    )

    value = batch[
        "probe_global_step"
    ]

    # In the frozen environment this currently materializes
    # as tensordict.utils.LinkedList.  The runtime contract only
    # depends on it preserving exactly one value per row.
    assert len(value) == BATCH

    validate_mixed_gkd_train_batch(
        batch
    )
