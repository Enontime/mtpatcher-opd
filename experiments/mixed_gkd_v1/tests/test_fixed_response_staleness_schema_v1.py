from __future__ import annotations

import numpy as np
import pytest

from mtpatcher_mixed_gkd.fixed_response_agent_v1 import (
    _fixed_branch_version_metadata,
)


@pytest.mark.parametrize(
    ("trainer_step", "model_version"),
    [
        (1, 0),
        (2, 1),
        (768, 767),
        (np.int64(768), 767),
    ],
)
def test_fixed_branch_maps_trainer_step_to_current_model_version(
    trainer_step,
    model_version,
):
    result = _fixed_branch_version_metadata(
        trainer_step
    )

    assert result == {
        "global_steps": model_version,
        "min_global_steps": model_version,
        "max_global_steps": model_version,
    }


@pytest.mark.parametrize(
    "bad_step",
    [
        0,
        -1,
    ],
)
def test_invalid_trainer_step_fails_closed(
    bad_step,
):
    with pytest.raises(
        ValueError,
        match="global_steps must be >= 1",
    ):
        _fixed_branch_version_metadata(
            bad_step
        )


def test_synchronous_staleness_formula_is_zero():
    # Mirrors Verl v1 trainer_base.py:
    # trajectory_staleness =
    #     (global_steps - 1) - max_global_steps
    for trainer_step in (
        1,
        2,
        768,
    ):
        metadata = (
            _fixed_branch_version_metadata(
                trainer_step
            )
        )

        staleness = (
            trainer_step
            - 1
            - metadata["max_global_steps"]
        )

        worst = (
            trainer_step
            - 1
            - metadata["min_global_steps"]
        )

        assert staleness == 0
        assert worst == 0
