#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.agent_loop import AgentLoopManager
from verl.experimental.teacher_loop.teacher_manager import (
    AsyncTeacherLLMServerManager,
    _pad_teacher_outputs,
)
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.model import compute_position_id_with_mask
from verl.utils.ray_utils import auto_await
from verl.workers.config import HFModelConfig, RolloutConfig

DIRECT_PROMPT = (
    "Translate the following text into English without additional explanations:"
    "\n\n{source}\n\n"
)
DIRECT_PROMPT_SHA256 = "63101d0739ff2ee11b0e5d548b85dcd18be7a241777893326140f12778d9a8b8"

TEACHER_CHEM_TEMPLATE = """You are translating a Chinese sentence into English.

Teacher-only lexical knowledge:
Chinese chemistry term: {src_term}
Canonical English registry term: {en_name}

Use the canonical English registry term when translating that designated
chemistry term. Translate the complete source sentence accurately and
naturally. Do not explain the instruction.

Chinese source:
{source}

Return only the English translation."""

EXPECTED_ROWS = 64
EXPECTED_TOPK = 32
EXPECTED_ARM_SHA = {
    "S": "83231dd2da9eb018df0cc92de85a48a30c60ee01fab15149929555626481dccf",
    "T": "266a562b8a373446ede0dd7015dbcdbdbf3ae3f9dd3fc1e98de712b213f683e3",
}
EXPECTED_ROW_ORDER_SHA = "a13880e003fe1581dc0f945687a8d48533f957a4fce165229fd7d3b2f663d147"


def sha256_json(obj: Any) -> str:
    raw = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def as_python(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    if hasattr(value, "item") and not isinstance(value, (str, bytes, list, dict)):
        try:
            return value.item()
        except Exception:
            pass
    return value


def pad_ids(
    ids: list[int],
    *,
    width: int,
    pad_id: int,
    side: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    require(len(ids) <= width, f"sequence width overflow len={len(ids)} width={width}")
    n = width - len(ids)
    if side == "left":
        padded = [pad_id] * n + ids
        mask = [0] * n + [1] * len(ids)
    elif side == "right":
        padded = ids + [pad_id] * n
        mask = [1] * len(ids) + [0] * n
    else:
        raise ValueError(side)
    return (
        torch.tensor(padded, dtype=torch.long).unsqueeze(0),
        torch.tensor(mask, dtype=torch.long).unsqueeze(0),
    )


class _ForbiddenFreshRolloutClient:
    """Fail closed if replay code ever tries to call the Student rollout client."""

    def __init__(self, wrapped: Any):
        self._wrapped_type = type(wrapped).__name__
        self.accesses: list[str] = []

    def __getattr__(self, name: str):
        self.accesses.append(name)
        raise RuntimeError(
            "FRESH_ROLLOUT_FORBIDDEN: replay smoke attempted Student rollout client "
            f"attribute={name!r} wrapped_type={self._wrapped_type!r}"
        )


class OfflinePrefixSupportReplayManager(AgentLoopManager):
    """
    Driver-side frozen-response transport for the 64-row engineering smoke.

    This uses Verl's public custom AgentLoopManager extension point. It does not
    patch ray_trainer.py. Student generation is replaced by frozen response IDs;
    Teacher top-k scoring and actor update remain native Verl code paths.
    """

    def __init__(
        self,
        config,
        llm_client,
        teacher_client=None,
        reward_loop_worker_handles=None,
    ):
        self.config = config
        self.rollout_config: RolloutConfig = omega_conf_to_dataclass(
            config.actor_rollout_ref.rollout
        )
        self.model_config: HFModelConfig = omega_conf_to_dataclass(
            config.actor_rollout_ref.model
        )
        self.tokenizer = self.model_config.tokenizer
        self.teacher_client = teacher_client
        self.reward_loop_worker_handles = reward_loop_worker_handles

        require(
            hashlib.sha256(DIRECT_PROMPT.encode("utf-8")).hexdigest()
            == DIRECT_PROMPT_SHA256,
            "DIRECT_PROMPT source drift",
        )
        require(
            int(config.distillation.distillation_loss.topk) == EXPECTED_TOPK,
            "Teacher top-k drift",
        )
        require(
            str(config.distillation.distillation_loss.loss_mode)
            == "forward_kl_topk",
            "distillation loss mode drift",
        )
        require(
            bool(config.distillation.distillation_loss.use_task_rewards) is False,
            "task rewards must remain disabled",
        )
        require(
            bool(config.distillation.distillation_loss.use_policy_gradient) is False,
            "policy gradient must remain disabled",
        )
        require(teacher_client is not None, "Teacher client missing")

        # Deliberate tripwire. Any accidental Student generation fails before update.
        self.llm_client = _ForbiddenFreshRolloutClient(llm_client)
        self.teacher_key = str(config.distillation.teacher_key)
        self.teacher_server_manager = AsyncTeacherLLMServerManager(
            config=config,
            teacher_client=teacher_client,
        )

        checkpoint_dir = Path(str(config.trainer.default_local_dir))
        self.arm_run = checkpoint_dir.parent
        self.audit_path = self.arm_run / "replay_transport_rows.jsonl"
        self.summary_path = self.arm_run / "replay_preupdate_summary.json"

        if self.audit_path.exists():
            self.audit_path.unlink()

        self.calls = 0

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        # Intentionally skip AgentLoopWorker construction. Frozen replay is built
        # on the driver and only the native Teacher server is queried.
        return cls(*args, **kwargs)

    async def _score_teacher(
        self,
        *,
        teacher_prompt_ids: list[int],
        response_ids: list[int],
        routing_key: Any,
    ):
        return await self.teacher_server_manager.compute_teacher_logprobs_single(
            sequence_ids=teacher_prompt_ids + response_ids,
            multi_modal_data=None,
            mm_processor_kwargs={},
            routing_key=routing_key,
        )

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        self.calls += 1
        require(self.calls == 1, f"unexpected replay generate call count={self.calls}")
        require(len(prompts) == EXPECTED_ROWS, f"replay rows={len(prompts)} expected=64")

        infos_raw = prompts.non_tensor_batch.get("extra_info")
        require(infos_raw is not None, "extra_info missing from replay input")
        infos = [as_python(x) for x in infos_raw]
        require(all(isinstance(x, dict) for x in infos), "extra_info rows must be dicts")

        arms = {str(x.get("arm")) for x in infos}
        require(len(arms) == 1, f"mixed replay arms={arms}")
        arm = next(iter(arms))
        require(arm in EXPECTED_ARM_SHA, f"bad replay arm={arm!r}")

        row_ids = [str(x["row_id"]) for x in infos]
        require(sha256_json(row_ids) == EXPECTED_ROW_ORDER_SHA, "row-order SHA drift")

        frozen_responses: list[list[int]] = []
        student_prompt_ids_all: list[list[int]] = []
        teacher_prompt_ids_all: list[list[int]] = []
        teacher_routing_keys: list[Any] = []
        row_audits: list[dict[str, Any]] = []

        for idx, info in enumerate(infos):
            require(info.get("domain") == "chemistry", f"row {idx} non-chemistry domain")
            require(str(info.get("arm")) == arm, f"row {idx} arm mismatch")

            response_ids = [int(x) for x in info["frozen_response_ids"]]
            student_prompt_ids = [int(x) for x in info["student_prompt_ids"]]
            teacher_prompt_ids = [int(x) for x in info["teacher_prompt_ids"]]

            require(
                len(response_ids) == int(info["m_i"]),
                f"{info['row_id']} frozen response length mismatch",
            )
            require(
                sha256_json(response_ids) == str(info["frozen_response_sha256"]),
                f"{info['row_id']} frozen response SHA mismatch",
            )

            # Re-render Student prompt from its direct source-only template.
            student_user = DIRECT_PROMPT.format(source=str(info["src_text"]))
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": student_user}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            runtime_student_ids = self.tokenizer(
                rendered,
                add_special_tokens=False,
            )["input_ids"]
            runtime_student_ids = [int(x) for x in runtime_student_ids]

            require(
                runtime_student_ids == student_prompt_ids,
                f"{info['row_id']} Student prompt token drift",
            )
            require(
                sha256_json(student_prompt_ids) == str(info["student_prompt_sha256"]),
                f"{info['row_id']} Student prompt SHA drift",
            )
            require(
                sha256_json(teacher_prompt_ids) == str(info["teacher_prompt_sha256"]),
                f"{info['row_id']} Teacher prompt SHA drift",
            )

            # Privileged English lexical knowledge is inserted only in Teacher prompt.
            expected_teacher_user = TEACHER_CHEM_TEMPLATE.format(
                src_term=str(info["src_term"]),
                en_name=str(info["knowledge"]),
                source=str(info["src_text"]),
            )
            require(
                str(info["teacher_user_text"]) == expected_teacher_user,
                f"{info['row_id']} Teacher prompt text drift",
            )
            require(
                str(info["student_user_text"]) == student_user,
                f"{info['row_id']} Student prompt text drift",
            )

            frozen_responses.append(response_ids)
            student_prompt_ids_all.append(student_prompt_ids)
            teacher_prompt_ids_all.append(teacher_prompt_ids)

            data_source = "default"
            if "data_source" in prompts.non_tensor_batch:
                data_source = as_python(prompts.non_tensor_batch["data_source"][idx])
            teacher_routing_keys.append(data_source)

            row_audits.append(
                {
                    "row_id": str(info["row_id"]),
                    "arm": arm,
                    "prefix_sha": str(info["frozen_response_sha256"]),
                    "prefix_token_count": len(response_ids),
                    "first_token_ids": response_ids[:8],
                    "last_token_ids": response_ids[-8:],
                    "teacher_hint_present": True,
                    "student_hint_present": False,
                    "student_prompt_sha256": str(info["student_prompt_sha256"]),
                    "teacher_prompt_sha256": str(info["teacher_prompt_sha256"]),
                    "status": "PREUPDATE_REPLAY_VERIFIED",
                }
            )

        aggregate_sha = sha256_json(frozen_responses)
        require(
            aggregate_sha == EXPECTED_ARM_SHA[arm],
            f"aggregate replay SHA drift arm={arm} got={aggregate_sha}",
        )

        # Query native Verl Teacher service on the Teacher-only hinted source prompt
        # plus the exact frozen prefix. No Student sampling is performed.
        tasks = [
            self._score_teacher(
                teacher_prompt_ids=t_prompt,
                response_ids=response,
                routing_key=routing_key,
            )
            for t_prompt, response, routing_key in zip(
                teacher_prompt_ids_all,
                frozen_responses,
                teacher_routing_keys,
                strict=True,
            )
        ]
        teacher_results = await asyncio.gather(*tasks)

        prompt_width = int(self.rollout_config.prompt_length)
        response_width = int(self.rollout_config.response_length)
        pad_id = int(self.tokenizer.pad_token_id)
        require(response_width >= max(map(len, frozen_responses)), "response width too small")

        prompt_tensors = []
        response_tensors = []
        response_masks = []
        attention_masks = []
        input_tensors = []
        position_tensors = []
        teacher_ids_tensors = []
        teacher_logprob_tensors = []

        for i, (
            student_prompt_ids,
            teacher_prompt_ids,
            response_ids,
            teacher_result,
        ) in enumerate(
            zip(
                student_prompt_ids_all,
                teacher_prompt_ids_all,
                frozen_responses,
                teacher_results,
                strict=True,
            )
        ):
            prompt_ids_t, prompt_attn = pad_ids(
                student_prompt_ids,
                width=prompt_width,
                pad_id=pad_id,
                side="left",
            )
            response_ids_t, response_attn = pad_ids(
                response_ids,
                width=response_width,
                pad_id=pad_id,
                side="right",
            )

            response_mask = response_attn.clone()
            attention_mask = torch.cat([prompt_attn, response_attn], dim=1)
            input_ids = torch.cat([prompt_ids_t, response_ids_t], dim=1)
            position_ids = compute_position_id_with_mask(attention_mask)

            teacher_ids, teacher_logprobs = teacher_result
            teacher_ids, teacher_logprobs = _pad_teacher_outputs(
                teacher_ids,
                teacher_logprobs,
                prompt_width=prompt_width,
                response_width=response_width,
                prompt_length=len(teacher_prompt_ids),
                response_length=len(response_ids),
                pad_token_id=pad_id,
            )

            require(
                teacher_ids.shape[0] == 1 and teacher_logprobs.shape[0] == 1,
                f"{row_ids[i]} Teacher padded batch dimension mismatch",
            )
            require(
                teacher_ids.shape[-1] == EXPECTED_TOPK,
                f"{row_ids[i]} Teacher top-k IDs width mismatch",
            )
            require(
                teacher_logprobs.shape[-1] == EXPECTED_TOPK,
                f"{row_ids[i]} Teacher top-k logprob width mismatch",
            )
            require(
                bool(torch.isfinite(teacher_logprobs).all().item()),
                f"{row_ids[i]} non-finite Teacher logprobs",
            )

            prompt_tensors.append(prompt_ids_t)
            response_tensors.append(response_ids_t)
            response_masks.append(response_mask)
            attention_masks.append(attention_mask)
            input_tensors.append(input_ids)
            position_tensors.append(position_ids)
            teacher_ids_tensors.append(teacher_ids)
            teacher_logprob_tensors.append(teacher_logprobs)

        batch = TensorDict(
            {
                "prompts": torch.cat(prompt_tensors, dim=0),
                "responses": torch.cat(response_tensors, dim=0),
                "response_mask": torch.cat(response_masks, dim=0),
                "input_ids": torch.cat(input_tensors, dim=0),
                "attention_mask": torch.cat(attention_masks, dim=0),
                "position_ids": torch.cat(position_tensors, dim=0),
                "teacher_ids": torch.cat(teacher_ids_tensors, dim=0),
                "teacher_logprobs": torch.cat(teacher_logprob_tensors, dim=0),
                # Explicit zero reward so the generic PPO shell can execute while
                # use_task_rewards=False and use_policy_gradient=False.
                "rm_scores": torch.zeros((EXPECTED_ROWS, response_width), dtype=torch.float32),
            },
            batch_size=EXPECTED_ROWS,
        )

        non_tensor_batch = {
            "__num_turns__": np.ones(EXPECTED_ROWS, dtype=np.int32),
        }
        for key, val in prompts.non_tensor_batch.items():
            non_tensor_batch[key] = np.array(val, copy=True)

        # The trainer expects generation timing metadata even though no generation happened.
        timing = {
            "agent_loop/num_preempted/min": 0.0,
            "agent_loop/num_preempted/max": 0.0,
            "agent_loop/num_preempted/mean": 0.0,
            "agent_loop/generate_sequences/min": 0.0,
            "agent_loop/generate_sequences/max": 0.0,
            "agent_loop/generate_sequences/mean": 0.0,
            "agent_loop/tool_calls/min": 0.0,
            "agent_loop/tool_calls/max": 0.0,
            "agent_loop/tool_calls/mean": 0.0,
            "agent_loop/compute_score/min": 0.0,
            "agent_loop/compute_score/max": 0.0,
            "agent_loop/compute_score/mean": 0.0,
        }

        for row in row_audits:
            append_jsonl(self.audit_path, row)

        atomic_json(
            self.summary_path,
            {
                "status": "PASS_PREUPDATE_REPLAY_TRANSPORT",
                "arm": arm,
                "rows": EXPECTED_ROWS,
                "aggregate_response_ids_sha256": aggregate_sha,
                "expected_aggregate_response_ids_sha256": EXPECTED_ARM_SHA[arm],
                "row_order_sha256": sha256_json(row_ids),
                "student_rollout_calls": 0,
                "forbidden_rollout_client_accesses": list(self.llm_client.accesses),
                "teacher_top_k": EXPECTED_TOPK,
                "policy_gradient": False,
                "task_rewards": False,
                "teacher_hint_present": True,
                "student_hint_present": False,
                "formal_training_authorized": False,
            },
        )

        print(
            "PREFIX_REPLAY_PREUPDATE_PASS "
            f"arm={arm} rows={EXPECTED_ROWS} aggregate_sha={aggregate_sha}",
            flush=True,
        )

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info={
                "timing": timing,
                "metrics": [],
                "reward_extra_keys": [],
                "prefix_replay": {
                    "arm": arm,
                    "rows": EXPECTED_ROWS,
                    "aggregate_response_ids_sha256": aggregate_sha,
                    "fresh_student_rollout_calls": 0,
                },
            },
        )
