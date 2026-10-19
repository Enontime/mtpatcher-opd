from __future__ import annotations

import uuid

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import (
    NonTensorData,
    NonTensorStack,
)
from transformers import AutoTokenizer

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.rl_dataset import (
    RLHFDataset,
    collate_fn,
)

from mtpatcher_mixed_gkd.agent_loop_manager_v1 import (
    validate_mixed_gkd_train_batch,
)


DATA = (
    "/workspace/mtpatcher/data/"
    "verl_science_pe_pds/"
    "mixed_gkd_v1_probe/"
    "pds2048_lambda05_v1.parquet"
)

MODEL = (
    "/workspace/mtpatcher/runs/science/"
    "pe_pds_v1/"
    "pds9952_stage_entry_v1/"
    "pe_opd_step4422_merged_hf"
)

BATCH = 16


def build_dataset():
    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL,
            local_files_only=True,
        )
    )

    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": 1024,
            "max_response_length": 256,
            "filter_overlong_prompts": False,
            "truncation": "error",
            "shuffle": False,
            "apply_chat_template_kwargs": {
                "enable_thinking": False,
            },
        }
    )

    dataset = RLHFDataset(
        data_files=DATA,
        tokenizer=tokenizer,
        config=config,
        processor=None,
    )

    assert len(dataset) == 12288

    return dataset


def make_real_batch(
    dataset,
    *,
    step: int,
):
    start = (
        step - 1
    ) * BATCH

    rows = [
        dataset[i]
        for i in range(
            start,
            start + BATCH,
        )
    ]

    # Exact trainer v1 collate path.
    batch_dict = collate_fn(
        rows
    )

    batch_dict["uid"] = np.array(
        [
            str(uuid.uuid4())
            for _ in range(BATCH)
        ],
        dtype=object,
    )

    # Exact trainer v1 conversion path.
    batch = tu.get_tensordict(
        batch_dict
    )

    # Exact _next_train_batch() metadata path.
    tu.assign_non_tensor_data(
        batch,
        "global_steps",
        step,
    )

    return batch


def emulate_tq_worker_prompt_extraction(
    batch,
):
    """Mirror AgentLoopWorkerTQ's per-row extraction contract."""

    prompts = []

    # Default worker count = 8.
    chunks = batch.chunk(8)

    assert len(chunks) == 8

    for chunk in chunks:
        for i in range(
            len(chunk)
        ):
            prompt = {}

            for key, value in chunk.items():
                if isinstance(
                    value,
                    torch.Tensor,
                ):
                    prompt[key] = value[i]

                elif isinstance(
                    value,
                    NonTensorStack,
                ):
                    prompt[key] = (
                        value[i].data
                    )

                elif isinstance(
                    value,
                    NonTensorData,
                ):
                    prompt[key] = (
                        value.data
                    )

                else:
                    raise AssertionError(
                        "AgentLoopWorkerTQ would "
                        "drop unsupported field: "
                        f"{key} -> "
                        f"{type(value)!r}"
                    )

            prompts.append(
                prompt
            )

    assert len(prompts) == BATCH

    return prompts


def assert_real_step(
    dataset,
    *,
    step: int,
    expected_branch: str,
    expected_agent: str,
):
    batch = make_real_batch(
        dataset,
        step=step,
    )

    # Frozen Mixed-GKD runtime gate.
    validate_mixed_gkd_train_batch(
        batch
    )

    prompts = (
        emulate_tq_worker_prompt_extraction(
            batch
        )
    )

    assert [
        int(x["probe_global_step"])
        for x in prompts
    ] == [step] * BATCH

    assert [
        int(x["probe_position_in_batch"])
        for x in prompts
    ] == list(
        range(BATCH)
    )

    assert [
        str(x["mixed_gkd_branch"])
        for x in prompts
    ] == [
        expected_branch
    ] * BATCH

    assert [
        str(x["agent_name"])
        for x in prompts
    ] == [
        expected_agent
    ] * BATCH

    # Critical for FIXED branch lookup.
    source_ids = [
        int(x["source_id"])
        for x in prompts
    ]

    assert len(source_ids) == BATCH
    assert len(set(source_ids)) == BATCH

    # Worker must also receive the trainer model-version label.
    assert [
        int(x["global_steps"])
        for x in prompts
    ] == [step] * BATCH

    # Actual AgentLoop inputs must survive transport.
    assert all(
        isinstance(
            x["raw_prompt"],
            list,
        )
        for x in prompts
    )


def test_real_lambda05_step1_onpolicy_transport():
    dataset = build_dataset()

    assert_real_step(
        dataset,
        step=1,
        expected_branch="ON_POLICY",
        expected_agent=
            "single_turn_agent",
    )


def test_real_lambda05_step2_fixed_transport():
    dataset = build_dataset()

    assert_real_step(
        dataset,
        step=2,
        expected_branch="FIXED",
        expected_agent=
            "mtpatcher_fixed_response_v1",
    )
