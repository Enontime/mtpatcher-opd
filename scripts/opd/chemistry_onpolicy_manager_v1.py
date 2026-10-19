#!/usr/bin/env python3
"""True on-policy, knowledge-conditioned Chemistry OPD adapter for native verl 0.9.

The superclass owns Student generation and rollout worker scheduling; the
native verl distillation actor owns KL, gradients, optimizer, and checkpoints.

Scientific boundary: Teacher gets canonical en_name, Student never does.
Only single-turn LLM trajectories are authorized. This is an ADAPTATION, not
paper-exact canonical OPD. Never use this manager with frozen replay inputs.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from verl.experimental.agent_loop import AgentLoopManager
from verl.experimental.teacher_loop.teacher_manager import (
    AsyncTeacherLLMServerManager, _pad_teacher_outputs,
)
from verl.utils.ray_utils import auto_await
from verl.utils.config import omega_conf_to_dataclass

DIRECT_PROMPT = (
    "Translate the following text into English without additional explanations:"
    "\n\n{source}\n\n"
)
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
DIRECT_PROMPT_SHA = "63101d0739ff2ee11b0e5d548b85dcd18be7a241777893326140f12778d9a8b8"
TOP_K = 32


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError("CHEM_ONPOLICY_CONTRACT_FAIL: " + message)


def render_prompt_ids(tokenizer, user_text: str) -> list[int]:
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return [int(x) for x in tokenizer(formatted, add_special_tokens=False)["input_ids"]]


def as_dict(value: Any) -> dict:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    check(isinstance(value, dict), "extra_info must be a dict")
    return value


def strip_left(ids, attn) -> list[int]:
    return [int(token) for token, mask in zip(ids.tolist(), attn.tolist()) if int(mask) == 1]


def extract_response(output, i: int) -> list[int]:
    """Single-turn only; response mask must exactly mark non-padding tokens."""
    response = output.batch["responses"][i]
    mask = output.batch["response_mask"][i]
    attn = output.batch["attention_mask"][i]
    response_width = int(response.shape[0])
    check(int(attn.shape[0]) >= response_width, "attention mask too short")
    actual = attn[-response_width:]
    check(torch.equal(mask.to(dtype=torch.bool), actual.to(dtype=torch.bool)),
          f"row {i} has tool/observation tokens or nonstandard response mask")
    positions = mask.nonzero(as_tuple=False).flatten().tolist()
    check(positions and positions == list(range(len(positions))),
          f"row {i} noncontiguous or empty Student response")
    return [int(x) for x in response[:len(positions)].tolist()]


class ChemistryOnPolicyManager(AgentLoopManager):
    """Extend native verl rollout with teacher side information, not training logic."""

    def __init__(self, config, llm_client, teacher_client=None,
                 reward_loop_worker_handles=None):
        super().__init__(config, llm_client, teacher_client,
                         reward_loop_worker_handles)
        self.tokenizer = omega_conf_to_dataclass(
            config.actor_rollout_ref.model
        ).tokenizer
        self.rollout = omega_conf_to_dataclass(
            config.actor_rollout_ref.rollout
        )
        check(hashlib.sha256(DIRECT_PROMPT.encode()).hexdigest() == DIRECT_PROMPT_SHA,
              "Student direct prompt changed")
        check(teacher_client is not None, "Teacher service absent")
        loss = config.distillation.distillation_loss
        check(bool(config.distillation.enabled), "native distillation disabled")
        check(str(loss.loss_mode) == "forward_kl_topk", "wrong KL mode")
        check(int(loss.topk) == TOP_K, "wrong Teacher top-k")
        check(not bool(loss.use_task_rewards) and not bool(loss.use_policy_gradient),
              "reward/policy gradient enabled")
        check(not bool(config.actor_rollout_ref.actor.use_kl_loss), "PPO actor KL enabled")
        check(not bool(config.algorithm.use_kl_in_reward), "reward KL enabled")
        check(int(config.actor_rollout_ref.rollout.n) == 1, "n!=1")
        check(float(config.actor_rollout_ref.rollout.temperature) == 1.0,
              "Student rollout temperature must be 1")
        check(float(config.actor_rollout_ref.rollout.top_p) == 1.0,
              "Student top-p changed")
        check(int(config.actor_rollout_ref.rollout.top_k) == -1,
              "native vLLM untruncated top_k must be -1")
        check(len(config.distillation.teacher_models) == 1,
              "only single Teacher supported")
        self.teacher_manager = AsyncTeacherLLMServerManager(
            config=config, teacher_client=teacher_client)
        self.pad_id = int(self.tokenizer.pad_token_id)
        self.calls = 0
        self.audit_dir = Path(str(config.trainer.default_local_dir)).parent
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.audit_path = self.audit_dir / "chem_onpolicy_rollout_metrics.jsonl"
        # Real, contemporaneous metrics. Kept separate from trainer's native
        # event writer to avoid event collisions; NEVER a post-hoc TB bridge.
        from torch.utils.tensorboard import SummaryWriter
        self.tb = SummaryWriter(log_dir=str(self.audit_dir / "tensorboard" / "rollout"))

    # IMPORTANT: do NOT override create(). Base create initializes Ray workers.

    async def _teacher_one(self, prompt_ids, response_ids):
        return await self.teacher_manager.compute_teacher_logprobs_single(
            sequence_ids=prompt_ids + response_ids,
            multi_modal_data=None, mm_processor_kwargs={}, routing_key=None,
        )

    @auto_await
    async def generate_sequences(self, prompts):
        # Verify authorization/data BEFORE model generation: no hint on Student side.
        extra = prompts.non_tensor_batch.get("extra_info")
        check(extra is not None and len(extra) == len(prompts), "missing extra_info")
        info_rows = [as_dict(x) for x in extra]
        seen = set()
        prompt_ids_all = []
        teacher_ids_all = []
        for i, info in enumerate(info_rows):
            job_id = str(info.get("job_id", ""))
            check(job_id and job_id not in seen, f"duplicate/missing job_id {job_id}")
            seen.add(job_id)
            check(info.get("domain") == "chemistry", f"non-Chemistry data row {i}")
            check(info.get("teacher_hint_present") is True, f"Teacher hint absent row {i}")
            check(info.get("student_hint_present") is False, f"Student hint leaked row {i}")
            source = info.get("src_text")
            term = info.get("src_term")
            en_name = info.get("en_name")
            check(all(isinstance(v, str) and v.strip() for v in (source,term,en_name)),
                  f"bad source/term/en_name at row {i}")
            check(info.get("student_prompt_sha256") is not None, "missing frozen prompt hash")
            s_prompt = render_prompt_ids(self.tokenizer, DIRECT_PROMPT.format(source=source))
            h = hashlib.sha256(json.dumps(s_prompt,separators=(',',':')).encode()).hexdigest()
            check(h == info["student_prompt_sha256"], f"Student prompt hash mismatch row {i}")
            # Input DataProto can have batch=None in native verl.
            # Validate native output prompt IDs against frozen data after rollout.
            t_prompt = TEACHER_CHEM_TEMPLATE.format(src_term=term,en_name=en_name,source=source)
            t_ids = render_prompt_ids(self.tokenizer, t_prompt)
            check(len(t_ids) + int(self.rollout.response_length) <= 1024,
                  f"Teacher max sequence length >1024 row {i}")
            check(len(s_prompt) + int(self.rollout.response_length) <= 1024,
                  f"Student max sequence length >1024 row {i}")
            if "teacher_prompt_sha256" in info:
                th = hashlib.sha256(json.dumps(t_ids,separators=(',',':')).encode()).hexdigest()
                check(th == info["teacher_prompt_sha256"], f"Teacher prompt hash changed row {i}")
            prompt_ids_all.append(s_prompt)
            teacher_ids_all.append(t_ids)

        # ONLY native Student generation. No custom generate()/sampling loop here.
        output = await super().generate_sequences(prompts)
        # Some verl revisions also compute a default (unhinted) Teacher on
        # the rollout path. We REPLACE those tensors completely with the
        # knowledge-conditioned Teacher scores below, never mix them.
        default_teacher_already_scored = (
            'teacher_ids' in output.batch.keys()
            or 'teacher_logprobs' in output.batch.keys()
        )
        check(('teacher_ids' in output.batch.keys()) ==
              ('teacher_logprobs' in output.batch.keys()),
              'partial default teacher result')
        check(len(output) == len(info_rows), "native rollout changed number of samples")
        prompt_width = int(output.batch["prompts"].shape[1])
        response_width = int(output.batch["responses"].shape[1])
        check(prompt_width == int(self.rollout.prompt_length), "rollout prompt width drift")
        check(response_width == int(self.rollout.response_length), "rollout response width drift")
        responses = []
        for i in range(len(output)):
            ids = strip_left(output.batch["prompts"][i],
                             output.batch["attention_mask"][i,:prompt_width])
            check(ids == prompt_ids_all[i], f"native rollout reordered prompt row {i}")
            responses.append(extract_response(output,i))
        check(all(len(t)+len(r)<=1024 for t,r in zip(teacher_ids_all,responses)),
              "Teacher prompt+response total length exceeded 1024")

        results = await asyncio.gather(*[
            self._teacher_one(t,r) for t,r in zip(teacher_ids_all,responses,strict=True)
        ])
        teacher_ids = []
        teacher_logprobs = []
        for i, (t,r,raw) in enumerate(zip(teacher_ids_all,responses,results,strict=True)):
            ids, logp = _pad_teacher_outputs(
                raw[0],raw[1],prompt_width=prompt_width,
                response_width=response_width,prompt_length=len(t),
                response_length=len(r),pad_token_id=self.pad_id,
            )
            check(tuple(ids.shape) == (1,prompt_width+response_width,TOP_K),
                  f"Teacher ids shape mismatch row {i}: {tuple(ids.shape)}")
            check(tuple(logp.shape) == (1,prompt_width+response_width,TOP_K),
                  f"Teacher logprobs shape mismatch row {i}: {tuple(logp.shape)}")
            check(torch.isfinite(logp).all().item(),f"nonfinite Teacher logits row {i}")
            teacher_ids.append(ids)
            teacher_logprobs.append(logp)
        output.batch["teacher_ids"] = torch.cat(teacher_ids,dim=0)
        output.batch["teacher_logprobs"] = torch.cat(teacher_logprobs,dim=0)
        # Native reward/advantage pathway is disabled; preserve native rm_scores
        # if present, otherwise add a constant zero score as validated replay does.
        if "rm_scores" not in output.batch.keys():
            output.batch["rm_scores"] = torch.zeros((len(output),response_width),dtype=torch.float32)
        self.calls += 1
        raw_step = prompts.meta_info.get("global_steps")
        outputs_text = [self.tokenizer.decode(r, skip_special_tokens=True).strip()
                        for r in responses]
        strict_hits = [int(str(info["en_name"]).casefold() in text.casefold())
                       for info, text in zip(info_rows,outputs_text,strict=True)]
        metrics = {
            "manager_call":self.calls,
            "default_teacher_was_scored_and_replaced":default_teacher_already_scored,
            "global_steps_from_verl":int(raw_step) if isinstance(raw_step,(int,np.integer)) else None,
            "rows":len(info_rows),
            "response_tokens":sum(map(len,responses)),
            "chem_train_rollout_hits":sum(strict_hits),
            "chem_train_rollout_strict_accuracy":float(np.mean(strict_hits)),
            "response_length_mean":float(np.mean([len(r) for r in responses])),
            "row_id_sha256":hashlib.sha256(json.dumps([str(v["job_id"]) for v in info_rows],
                                  separators=(',',':')).encode()).hexdigest(),
            "status":"NATIVE_FRESH_ROLLOUT_AND_TEACHER_TOPK_PASS",
        }
        # Write a genuine transport audit; native verl's own distillation,
        # optimizer and timing scalars are written by native trainer logger.
        with self.audit_path.open("a",encoding="utf8") as f:
            f.write(json.dumps(metrics,ensure_ascii=False,sort_keys=True)+"\n")
        # These are real Student on-policy *training* trajectories, not heldout
        # greedy test accuracy. Step is only labeled optimizer-aligned when verl
        # actually supplies a numeric global_steps field.
        tb_step = metrics["global_steps_from_verl"]
        tag_step = tb_step if tb_step is not None else self.calls
        prefix = "rollout/chem_train" if tb_step is not None else "rollout_manager_call/chem_train"
        self.tb.add_scalar(prefix + "/strict_hit", metrics["chem_train_rollout_strict_accuracy"], tag_step)
        self.tb.add_scalar(prefix + "/response_tokens", metrics["response_tokens"], tag_step)
        self.tb.add_scalar(prefix + "/mean_length", metrics["response_length_mean"], tag_step)
        self.tb.flush()
        print("CHEM_ONPOLICY_STEP " + json.dumps(metrics,sort_keys=True),flush=True)
        return output
