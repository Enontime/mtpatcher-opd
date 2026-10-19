from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from tensordict import (
    NonTensorData,
    NonTensorStack,
    TensorDict,
)

from verl.trainer.ppo.v1.agent_loop_tq import (
    AgentLoopManagerTQ,
)


GLOBAL_BATCH = 16
TOTAL_STEPS = 768

MASK_PATH = Path(
    "/workspace/mtpatcher/data/"
    "verl_science_pe_pds/"
    "mixed_gkd_v1_probe/"
    "lambda05_step_mask_v1.jsonl"
)

EXPECTED_MASK_FILE_SHA256 = (
    "7d94f521893538bfe10a3481109c07f5"
    "ebd226d84d12cbdc86ea9d39d6db70bd"
)

EXPECTED_MASK_SEMANTIC_SHA256 = (
    "b3d7f2784401c939fe4dd8fc95c7aa5"
    "fb20c36068d8b3188649b039c872d9e8e"
)


def _canonical_json_bytes(
    obj: Any,
) -> bytes:
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _file_sha256(
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


@lru_cache(maxsize=1)
def _load_lambda05_mask() -> dict[int, str]:
    if not MASK_PATH.is_file():
        raise FileNotFoundError(
            MASK_PATH
        )

    got_file_sha = _file_sha256(
        MASK_PATH
    )

    if (
        got_file_sha
        != EXPECTED_MASK_FILE_SHA256
    ):
        raise RuntimeError(
            "Mixed-GKD mask physical "
            "SHA drift: "
            f"{got_file_sha}"
        )

    rows = [
        json.loads(line)
        for line in MASK_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    got_semantic_sha = (
        hashlib.sha256(
            _canonical_json_bytes(
                rows
            )
        ).hexdigest()
    )

    if (
        got_semantic_sha
        != EXPECTED_MASK_SEMANTIC_SHA256
    ):
        raise RuntimeError(
            "Mixed-GKD mask semantic "
            "SHA drift: "
            f"{got_semantic_sha}"
        )

    if len(rows) != TOTAL_STEPS:
        raise RuntimeError(
            "Mixed-GKD mask row count "
            f"{len(rows)} != "
            f"{TOTAL_STEPS}"
        )

    mask = {}

    for row in rows:
        step = int(
            row["step"]
        )

        branch = str(
            row["branch"]
        )

        if branch not in {
            "ON_POLICY",
            "FIXED",
        }:
            raise RuntimeError(
                "invalid mask branch "
                f"step={step} "
                f"branch={branch!r}"
            )

        if step in mask:
            raise RuntimeError(
                "duplicate mask step "
                f"{step}"
            )

        mask[step] = branch

    if set(mask) != set(
        range(
            1,
            TOTAL_STEPS + 1,
        )
    ):
        raise RuntimeError(
            "Mixed-GKD mask step set drift"
        )

    if (
        list(mask.values())
        .count("ON_POLICY")
        != 384
    ):
        raise RuntimeError(
            "ON_POLICY mask count drift"
        )

    if (
        list(mask.values())
        .count("FIXED")
        != 384
    ):
        raise RuntimeError(
            "FIXED mask count drift"
        )

    return mask


def _unwrap_scalar(
    value: Any,
) -> Any:
    if isinstance(
        value,
        NonTensorData,
    ):
        return value.data

    if isinstance(
        value,
        torch.Tensor,
    ):
        if value.numel() != 1:
            raise RuntimeError(
                "expected scalar Tensor, "
                f"got shape={value.shape}"
            )

        return value.item()

    if (
        hasattr(value, "item")
        and not isinstance(
            value,
            (
                str,
                bytes,
                dict,
                list,
                tuple,
            ),
        )
    ):
        try:
            return value.item()
        except Exception:
            pass

    return value


def _batch_values(
    batch: TensorDict,
    key: str,
) -> list[Any]:
    if key not in batch:
        raise KeyError(
            "Mixed-GKD runtime batch "
            f"missing {key!r}"
        )

    value = batch[key]

    if isinstance(
        value,
        NonTensorData,
    ):
        scalar = _unwrap_scalar(
            value
        )

        return [
            scalar
            for _ in range(
                len(batch)
            )
        ]

    if isinstance(
        value,
        NonTensorStack,
    ):
        return [
            _unwrap_scalar(
                value[i]
            )
            for i in range(
                len(value)
            )
        ]

    if isinstance(
        value,
        torch.Tensor,
    ):
        if value.ndim == 0:
            scalar = value.item()

            return [
                scalar
                for _ in range(
                    len(batch)
                )
            ]

        if value.shape[0] != len(
            batch
        ):
            raise RuntimeError(
                f"{key}: Tensor batch "
                "dimension mismatch"
            )

        return [
            _unwrap_scalar(
                value[i]
            )
            for i in range(
                len(batch)
            )
        ]

    # TensorDict may materialize per-row non-tensor metadata
    # as tensordict.utils.LinkedList rather than NonTensorStack.
    #
    # Keep this deliberately duck-typed: LinkedList is an internal
    # TensorDict representation, while the semantic contract here is
    # simply "one metadata value per batch row".
    if (
        not isinstance(
            value,
            (
                str,
                bytes,
                dict,
            ),
        )
        and hasattr(
            value,
            "__len__",
        )
        and hasattr(
            value,
            "__getitem__",
        )
    ):
        value_len = len(
            value
        )

        if value_len != len(
            batch
        ):
            raise RuntimeError(
                f"{key}: per-row metadata length "
                f"{value_len} != batch {len(batch)}"
            )

        return [
            _unwrap_scalar(
                value[i]
            )
            for i in range(
                value_len
            )
        ]

    # TensorDict may unwrap NonTensorData scalar metadata
    # on retrieval.  In the actual Verl path fields such as
    # global_steps / validate can therefore arrive as plain
    # Python scalars (or numpy scalar-like objects).
    scalar = _unwrap_scalar(
        value
    )

    if (
        scalar is not value
        or isinstance(
            value,
            (
                str,
                bytes,
                int,
                float,
                bool,
            ),
        )
    ):
        return [
            scalar
            for _ in range(
                len(batch)
            )
        ]

    raise TypeError(
        "unsupported Mixed-GKD "
        f"batch field {key!r}: "
        f"{type(value)!r}"
    )


def _single_global_step(
    batch: TensorDict,
) -> int:
    vals = _batch_values(
        batch,
        "global_steps",
    )

    parsed = [
        int(x)
        for x in vals
    ]

    if len(
        set(parsed)
    ) != 1:
        raise RuntimeError(
            "Mixed-GKD batch has "
            "multiple trainer global_steps: "
            f"{parsed}"
        )

    return parsed[0]


def validate_mixed_gkd_train_batch(
    batch: TensorDict,
) -> None:
    """Fail closed before any rollout is dispatched."""

    if len(batch) != GLOBAL_BATCH:
        raise RuntimeError(
            "Mixed-GKD requires exact "
            f"batch={GLOBAL_BATCH}, "
            f"got {len(batch)}"
        )

    global_step = (
        _single_global_step(
            batch
        )
    )

    if not (
        1
        <= global_step
        <= TOTAL_STEPS
    ):
        raise RuntimeError(
            "trainer global_steps outside "
            "frozen probe domain: "
            f"{global_step}"
        )

    probe_steps = [
        int(x)
        for x in _batch_values(
            batch,
            "probe_global_step",
        )
    ]

    if (
        probe_steps
        != [
            global_step
        ] * GLOBAL_BATCH
    ):
        raise RuntimeError(
            "probe_global_step != "
            "trainer global_steps: "
            f"trainer={global_step} "
            f"probe={probe_steps}"
        )

    positions = [
        int(x)
        for x in _batch_values(
            batch,
            "probe_position_in_batch",
        )
    ]

    expected_positions = list(
        range(
            GLOBAL_BATCH
        )
    )

    if positions != expected_positions:
        raise RuntimeError(
            "probe_position_in_batch "
            "order drift: "
            f"{positions}"
        )

    schedule_positions = [
        int(x)
        for x in _batch_values(
            batch,
            "probe_schedule_position",
        )
    ]

    expected_schedule_positions = list(
        range(
            (
                global_step - 1
            )
            * GLOBAL_BATCH,
            global_step
            * GLOBAL_BATCH,
        )
    )

    if (
        schedule_positions
        != expected_schedule_positions
    ):
        raise RuntimeError(
            "probe schedule position "
            "drift at step "
            f"{global_step}: "
            f"actual={schedule_positions} "
            "expected="
            f"{expected_schedule_positions}"
        )

    lambdas = [
        float(x)
        for x in _batch_values(
            batch,
            "mixed_gkd_lambda",
        )
    ]

    if len(
        set(lambdas)
    ) != 1:
        raise RuntimeError(
            "Mixed-GKD lambda mixed "
            "inside one optimizer step: "
            f"{lambdas}"
        )

    lam = lambdas[0]

    branches = [
        str(x)
        for x in _batch_values(
            batch,
            "mixed_gkd_branch",
        )
    ]

    if len(
        set(branches)
    ) != 1:
        raise RuntimeError(
            "Mixed-GKD branch mixed "
            "inside one optimizer step: "
            f"{branches}"
        )

    branch = branches[0]

    agents = [
        str(x)
        for x in _batch_values(
            batch,
            "agent_name",
        )
    ]

    if len(
        set(agents)
    ) != 1:
        raise RuntimeError(
            "Mixed-GKD agent_name mixed "
            "inside one optimizer step: "
            f"{agents}"
        )

    agent = agents[0]

    if lam == 1.0:
        expected_branch = (
            "ON_POLICY"
        )

    elif lam == 0.5:
        expected_branch = (
            _load_lambda05_mask()[
                global_step
            ]
        )

    else:
        raise RuntimeError(
            "unexpected Mixed-GKD "
            f"lambda={lam}"
        )

    if branch != expected_branch:
        raise RuntimeError(
            "Mixed-GKD frozen branch "
            "mismatch at "
            f"step={global_step}: "
            f"actual={branch} "
            f"expected={expected_branch}"
        )

    expected_agent = (
        "single_turn_agent"
        if expected_branch
        == "ON_POLICY"
        else
        "mtpatcher_fixed_response_v1"
    )

    if agent != expected_agent:
        raise RuntimeError(
            "Mixed-GKD agent routing "
            "mismatch at "
            f"step={global_step}: "
            f"actual={agent} "
            f"expected={expected_agent}"
        )


class MixedGKDv1AgentLoopManagerTQ(
    AgentLoopManagerTQ
):
    """TQ manager with frozen Mixed-GKD v1 routing gate."""

    def generate_sequences(
        self,
        prompts: TensorDict,
    ) -> None:
        validate = False

        if "validate" in prompts:
            vals = _batch_values(
                prompts,
                "validate",
            )

            validate = any(
                bool(x)
                for x in vals
            )

        if not validate:
            validate_mixed_gkd_train_batch(
                prompts
            )

        return super().generate_sequences(
            prompts
        )
