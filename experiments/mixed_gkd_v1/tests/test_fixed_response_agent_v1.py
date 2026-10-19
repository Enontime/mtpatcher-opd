from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from mtpatcher_mixed_gkd.fixed_response_agent_v1 import (
    FixedResponseAgentLoopV1,
    _load_fixed_response_bank,
)


PROJECT = Path(
    "/workspace/mtpatcher/repo/"
    "MT-Patcher-Reproduction-Ascend"
)

BANK = Path(
    "/workspace/mtpatcher/data/"
    "verl_science_pe_pds/"
    "mixed_gkd_v1_probe/"
    "fixed_response_bank_v1.jsonl"
)

CFG = (
    PROJECT
    / "experiments/mixed_gkd_v1/"
      "configs/agent_loops_v1.yaml"
)

EXPECTED_FILE_SHA = (
    "c995f56b2533bef831c4b88953910c06"
    "c0421acdb59250667aed86881e4f8711"
)

EXPECTED_SEMANTIC_SHA = (
    "2fc2c8066e46a7a69775de51e17d21f"
    "47b3abb84c8ba919610ca5135e1f421de"
)

EOS = 151645


def file_sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def test_real_frozen_bank_identity():
    assert file_sha(
        BANK
    ) == EXPECTED_FILE_SHA

    bank = _load_fixed_response_bank(
        BANK,
        expected_file_sha256=
            EXPECTED_FILE_SHA,
        expected_semantic_sha256=
            EXPECTED_SEMANTIC_SHA,
        expected_rows=2048,
        eos_token_id=EOS,
        max_response_length=256,
    )

    assert len(bank) == 2048

    assert all(
        row[-1] == EOS
        for row in bank.values()
    )

    assert max(
        len(row)
        for row in bank.values()
    ) == 170


def test_agent_config_points_to_frozen_bank():
    cfg = yaml.safe_load(
        CFG.read_text(
            encoding="utf-8"
        )
    )

    assert isinstance(cfg, list)
    assert len(cfg) == 1

    row = cfg[0]

    assert (
        row["name"]
        == "mtpatcher_fixed_response_v1"
    )

    assert (
        row["_target_"]
        ==
        "mtpatcher_mixed_gkd."
        "fixed_response_agent_v1."
        "FixedResponseAgentLoopV1"
    )

    assert (
        row[
            "fixed_response_bank_file_sha256"
        ]
        == EXPECTED_FILE_SHA
    )

    assert (
        row[
            "fixed_response_bank_semantic_sha256"
        ]
        == EXPECTED_SEMANTIC_SHA
    )


class ExplodingServer:
    def __init__(self):
        self.generate_calls = 0

    async def generate(
        self,
        *args,
        **kwargs,
    ):
        self.generate_calls += 1

        raise AssertionError(
            "FIXED branch called "
            "Student server.generate()"
        )


def test_fixed_agent_returns_exact_ids_without_generation():
    rows = [
        json.loads(line)
        for line in BANK.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    sample = rows[17]

    source_id = int(
        sample["source_id"]
    )

    expected_response = list(
        sample["response_ids"]
    )

    # Build a minimal instance without invoking
    # heavyweight AgentLoopBase initialization.
    agent = (
        FixedResponseAgentLoopV1
        .__new__(
            FixedResponseAgentLoopV1
        )
    )

    agent.response_length = 256

    agent._fixed_response_bank = {
        source_id:
            tuple(
                expected_response
            )
    }

    server = ExplodingServer()

    agent.server_manager = server
    agent.enable_continuous_token = False
    agent.mm_processor_kwargs = {}

    prompt_seen = {}

    async def fake_process(
        messages,
    ):
        prompt_seen[
            "processed_messages"
        ] = messages

        return {}

    async def fake_template(
        messages,
        **kwargs,
    ):
        prompt_seen[
            "template_messages"
        ] = messages

        prompt_seen[
            "template_kwargs"
        ] = kwargs

        return [
            101,
            102,
            103,
        ]

    def fake_mm_kwargs(
        audios=None,
    ):
        assert audios is None
        return {}

    agent.process_multi_modal_info = (
        fake_process
    )

    agent.apply_chat_template = (
        fake_template
    )

    agent._get_mm_processor_kwargs = (
        fake_mm_kwargs
    )

    raw_prompt = [
        {
            "role": "user",
            "content": "hello",
        }
    ]

    output = asyncio.run(
        agent.run(
            {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
            },
            raw_prompt=raw_prompt,
            source_id=source_id,
            global_steps=7,
        )
    )

    assert (
        server.generate_calls
        == 0
    )

    assert output.prompt_ids == [
        101,
        102,
        103,
    ]

    assert (
        output.response_ids
        == expected_response
    )

    assert (
        output.response_mask
        == [1]
        * len(
            expected_response
        )
    )

    assert (
        output.response_logprobs
        is None
    )

    assert (
        output.extra_fields[
            "mixed_gkd_branch"
        ]
        == "FIXED"
    )

    assert (
        output.extra_fields[
            "fixed_response_source_id"
        ]
        == source_id
    )

    assert (
        output.extra_fields[
            "global_steps"
        ]
        == 6
    )

    assert (
        output.extra_fields[
            "min_global_steps"
        ]
        == 6
    )

    assert (
        output.extra_fields[
            "max_global_steps"
        ]
        == 6
    )

    assert (
        prompt_seen[
            "processed_messages"
        ]
        == raw_prompt
    )

    assert (
        prompt_seen[
            "template_messages"
        ]
        == raw_prompt
    )


def test_missing_source_fails_closed():
    agent = (
        FixedResponseAgentLoopV1
        .__new__(
            FixedResponseAgentLoopV1
        )
    )

    agent.response_length = 256
    agent._fixed_response_bank = {
        1: (
            7,
            EOS,
        )
    }

    with pytest.raises(
        KeyError,
        match="absent from frozen",
    ):
        asyncio.run(
            agent.run(
                {},
                raw_prompt=[
                    {
                        "role": "user",
                        "content": "x",
                    }
                ],
                source_id=999999,
            )
        )



def test_missing_global_steps_fails_closed():
    agent = (
        FixedResponseAgentLoopV1
        .__new__(
            FixedResponseAgentLoopV1
        )
    )

    agent.response_length = 256
    agent._fixed_response_bank = {
        1: (
            7,
            EOS,
        )
    }

    with pytest.raises(
        KeyError,
        match="missing global_steps",
    ):
        asyncio.run(
            agent.run(
                {},
                raw_prompt=[
                    {
                        "role": "user",
                        "content": "x",
                    }
                ],
                source_id=1,
            )
        )
