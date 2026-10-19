from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopMetrics,
    AgentLoopOutput,
)
from verl.experimental.agent_loop.single_turn_agent_loop import (
    SingleTurnAgentLoop,
)


EXPECTED_AGENT_NAME = "mtpatcher_fixed_response_v1"


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
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


@lru_cache(maxsize=8)
def _load_fixed_response_bank(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_semantic_sha256: str,
    expected_rows: int,
    eos_token_id: int,
    max_response_length: int,
) -> dict[int, tuple[int, ...]]:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"fixed response bank missing: {path}"
        )

    actual_file_sha = _file_sha256(
        path
    )

    if (
        actual_file_sha
        != expected_file_sha256
    ):
        raise RuntimeError(
            "fixed response bank physical SHA drift: "
            f"actual={actual_file_sha} "
            f"expected={expected_file_sha256}"
        )

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_no, line in enumerate(
            f,
            1,
        ):
            if not line.strip():
                continue

            try:
                row = json.loads(
                    line
                )
            except Exception as exc:
                raise ValueError(
                    "invalid fixed-response JSON "
                    f"line={line_no}: {exc}"
                ) from exc

            rows.append(row)

    if len(rows) != expected_rows:
        raise RuntimeError(
            "fixed response bank row-count drift: "
            f"actual={len(rows)} "
            f"expected={expected_rows}"
        )

    actual_semantic_sha = (
        hashlib.sha256(
            _canonical_json_bytes(
                rows
            )
        ).hexdigest()
    )

    if (
        actual_semantic_sha
        != expected_semantic_sha256
    ):
        raise RuntimeError(
            "fixed response bank semantic SHA drift: "
            f"actual={actual_semantic_sha} "
            f"expected={expected_semantic_sha256}"
        )

    result: dict[
        int,
        tuple[int, ...],
    ] = {}

    for row in rows:
        source_id = int(
            row["source_id"]
        )

        if source_id in result:
            raise RuntimeError(
                "duplicate fixed-response source_id="
                f"{source_id}"
            )

        response_ids = row.get(
            "response_ids"
        )

        if (
            not isinstance(
                response_ids,
                list,
            )
            or not response_ids
            or not all(
                isinstance(x, int)
                for x in response_ids
            )
        ):
            raise RuntimeError(
                "invalid fixed response IDs "
                f"source_id={source_id}"
            )

        declared_len = int(
            row["response_len"]
        )

        if (
            declared_len
            != len(response_ids)
        ):
            raise RuntimeError(
                "fixed response length drift "
                f"source_id={source_id}"
            )

        if (
            len(response_ids)
            > max_response_length
        ):
            raise RuntimeError(
                "fixed response exceeds "
                "rollout response_length "
                f"source_id={source_id} "
                f"length={len(response_ids)} "
                f"max={max_response_length}"
            )

        if (
            int(response_ids[-1])
            != int(eos_token_id)
        ):
            raise RuntimeError(
                "fixed response does not end "
                "in canonical EOS "
                f"source_id={source_id}"
            )

        result[source_id] = tuple(
            int(x)
            for x in response_ids
        )

    return result


def _fixed_branch_version_metadata(
    global_steps: Any,
) -> dict[str, int]:
    """Build Verl v1 model-version metadata for FIXED routing.

    Verl's trainer global_steps is the optimizer-step ordinal.
    During synchronous rollout for step N, the currently served
    Student weights are model version N - 1.  Native ON_POLICY
    trajectories obtain this version from the rollout server.

    FIXED performs zero Student generation, so tag the synthetic
    fixed trajectory with the same current Student model version
    that an ON_POLICY trajectory at this trainer step would use.
    """
    value = global_steps

    if hasattr(
        value,
        "item",
    ):
        value = value.item()

    trainer_step = int(value)

    if trainer_step < 1:
        raise ValueError(
            "global_steps must be >= 1"
        )

    model_version = trainer_step - 1

    return {
        "global_steps": model_version,
        "min_global_steps": model_version,
        "max_global_steps": model_version,
    }


class FixedResponseAgentLoopV1(
    SingleTurnAgentLoop
):
    """Mixed-GKD v1 fixed-prefix trajectory source.

    Semantics:

        raw MT prompt
        -> same prompt rendering as SingleTurnAgentLoop
        -> frozen canonical Teacher response IDs
        -> NO Student generation call

    Teacher scoring remains downstream in AgentLoopWorkerTQ,
    so fixed and on-policy branches share the same Teacher scorer.
    """

    def __init__(
        self,
        *args,
        fixed_response_bank_path: str,
        fixed_response_bank_file_sha256: str,
        fixed_response_bank_semantic_sha256: str,
        fixed_response_bank_rows: int,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        if self.processor is not None:
            raise ValueError(
                "Mixed-GKD v1 fixed-response "
                "agent supports text-only MT."
            )

        if self.enable_continuous_token:
            raise ValueError(
                "Mixed-GKD v1 fixed-response "
                "agent requires continuous_token=false."
            )

        if (
            self.tokenizer.eos_token_id
            is None
        ):
            raise ValueError(
                "tokenizer.eos_token_id is None"
            )

        self._fixed_response_bank = (
            _load_fixed_response_bank(
                fixed_response_bank_path,
                expected_file_sha256=
                    fixed_response_bank_file_sha256,
                expected_semantic_sha256=
                    fixed_response_bank_semantic_sha256,
                expected_rows=
                    int(
                        fixed_response_bank_rows
                    ),
                eos_token_id=
                    int(
                        self.tokenizer
                        .eos_token_id
                    ),
                max_response_length=
                    int(
                        self.response_length
                    ),
            )
        )

    async def run(
        self,
        sampling_params:
            dict[str, Any],
        priority: int = 0,
        **kwargs,
    ) -> AgentLoopOutput:
        # Explicitly unused: FIXED branch does not sample.
        del sampling_params
        del priority

        if "source_id" not in kwargs:
            raise KeyError(
                "fixed-response trajectory "
                "missing source_id"
            )

        raw_source_id = kwargs[
            "source_id"
        ]

        # Tensor scalars can arrive from a custom
        # dataset/collator; numpy/Python scalars are
        # handled by int(...) directly.
        if hasattr(
            raw_source_id,
            "item",
        ):
            raw_source_id = (
                raw_source_id.item()
            )

        source_id = int(
            raw_source_id
        )

        try:
            response_ids = list(
                self._fixed_response_bank[
                    source_id
                ]
            )
        except KeyError as exc:
            raise KeyError(
                "source_id absent from frozen "
                "fixed-response bank: "
                f"{source_id}"
            ) from exc

        if "global_steps" not in kwargs:
            raise KeyError(
                "fixed-response trajectory "
                "missing global_steps"
            )

        version_fields = (
            _fixed_branch_version_metadata(
                kwargs["global_steps"]
            )
        )

        messages = list(
            kwargs["raw_prompt"]
        )

        # Keep prompt construction identical to
        # SingleTurnAgentLoop.
        multi_modal_data = (
            await self
            .process_multi_modal_info(
                messages
            )
        )

        if multi_modal_data:
            raise ValueError(
                "Mixed-GKD v1 fixed branch "
                "received multimodal input"
            )

        mm_processor_kwargs = (
            self
            ._get_mm_processor_kwargs(
                None
            )
        )

        prompt_ids = (
            await self.apply_chat_template(
                messages,
                images=None,
                videos=None,
                audios=None,
                mm_processor_kwargs=
                    mm_processor_kwargs,
            )
        )

        if len(
            response_ids
        ) > self.response_length:
            # Should already be impossible from
            # construction-time bank validation.
            raise RuntimeError(
                "fixed response runtime "
                "length overflow"
            )

        output = AgentLoopOutput(
            prompt_ids=list(
                prompt_ids
            ),

            response_ids=
                response_ids,

            response_mask=[
                1
            ] * len(
                response_ids
            ),

            # Intentionally absent.
            # Mixed-GKD configs set
            # rollout.calculate_log_probs=false.
            response_logprobs=None,

            routed_experts=None,

            multi_modal_data={},

            mm_processor_kwargs=
                mm_processor_kwargs,

            num_turns=2,

            metrics=AgentLoopMetrics(
                generate_sequences=0.0,
                tool_calls=0.0,
                compute_score=0.0,
                num_preempted=-1,
            ),

            extra_fields={
                "turn_scores": [],
                "tool_rewards": [],
                "mixed_gkd_branch":
                    "FIXED",
                "fixed_response_source_id":
                    source_id,
                **version_fields,
            },
        )

        return output
