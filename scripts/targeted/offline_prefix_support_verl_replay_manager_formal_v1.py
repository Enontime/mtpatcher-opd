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
    'Translate the following text into English without additional explanations:'
    '\n\n{source}\n\n'
)
DIRECT_PROMPT_SHA256 = '63101d0739ff2ee11b0e5d548b85dcd18be7a241777893326140f12778d9a8b8'

TEACHER_CHEM_TEMPLATE = '''You are translating a Chinese sentence into English.

Teacher-only lexical knowledge:
Chinese chemistry term: {src_term}
Canonical English registry term: {en_name}

Use the canonical English registry term when translating that designated
chemistry term. Translate the complete source sentence accurately and
naturally. Do not explain the instruction.

Chinese source:
{source}

Return only the English translation.'''

EXPECTED_ROWS = 1000
EXPECTED_BATCH_ROWS = 40
EXPECTED_BATCHES_PER_PASS = 25
EXPECTED_PASSES = 3
EXPECTED_CALLS = 75
EXPECTED_TOPK = 32
EXPECTED_ROW_ORDER_SHA = '49763f7a369c48e87befb1417927753f5e3c02d4ec81748540c6b0ca50bb95da'
EXPECTED_ROW_SET_SHA = '92011d1090c65a99e1472b61e81a5ad5d5f3753e37c93d6538fc01f92eb23c8c'
EXPECTED_ARM_SHA = {
    'S': '2e37662fb949a43004d9842eb68166c751b2fd97273e6dc408d4b911a7a12308',
    'T': 'd68bff648787e93da34b550bb78b4e552c0c7bdff8f7b5db7138d36f215a87d1',
}


def sha256_json(obj: Any) -> str:
    raw = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
        f.flush()
        os.fsync(f.fileno())


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def as_python(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    if hasattr(value, 'item') and not isinstance(value, (str, bytes, list, dict)):
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
    require(len(ids) <= width, f'sequence width overflow len={len(ids)} width={width}')
    n = width - len(ids)
    if side == 'left':
        padded = [pad_id] * n + ids
        mask = [0] * n + [1] * len(ids)
    elif side == 'right':
        padded = ids + [pad_id] * n
        mask = [1] * len(ids) + [0] * n
    else:
        raise ValueError(side)
    return (
        torch.tensor(padded, dtype=torch.long).unsqueeze(0),
        torch.tensor(mask, dtype=torch.long).unsqueeze(0),
    )


class _ForbiddenFreshRolloutClient:
    def __init__(self, wrapped: Any):
        self._wrapped_type = type(wrapped).__name__
        self.accesses: list[str] = []

    def __getattr__(self, name: str):
        self.accesses.append(name)
        raise RuntimeError(
            'FRESH_ROLLOUT_FORBIDDEN: formal frozen replay attempted Student rollout client '
            f'attribute={name!r} wrapped_type={self._wrapped_type!r}'
        )


class OfflinePrefixSupportFormalReplayManager(AgentLoopManager):
    """Frozen 1000-row Chemistry S/T replay over 3 exact passes via native Verl."""

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
            hashlib.sha256(DIRECT_PROMPT.encode('utf-8')).hexdigest()
            == DIRECT_PROMPT_SHA256,
            'DIRECT_PROMPT source drift',
        )
        require(int(config.distillation.distillation_loss.topk) == EXPECTED_TOPK, 'Teacher top-k drift')
        require(
            str(config.distillation.distillation_loss.loss_mode) == 'forward_kl_topk',
            'distillation loss mode drift',
        )
        require(
            bool(config.distillation.distillation_loss.use_task_rewards) is False,
            'task rewards must remain disabled',
        )
        require(
            bool(config.distillation.distillation_loss.use_policy_gradient) is False,
            'policy gradient must remain disabled',
        )
        require(teacher_client is not None, 'Teacher client missing')
        require(int(config.data.train_batch_size) == EXPECTED_BATCH_ROWS, 'formal train batch drift')
        require(int(config.trainer.total_epochs) == EXPECTED_PASSES, 'formal pass count drift')
        require(int(config.trainer.total_training_steps) == EXPECTED_CALLS, 'formal update count drift')

        self.llm_client = _ForbiddenFreshRolloutClient(llm_client)
        self.teacher_key = str(config.distillation.teacher_key)
        self.teacher_server_manager = AsyncTeacherLLMServerManager(
            config=config,
            teacher_client=teacher_client,
        )

        checkpoint_dir = Path(str(config.trainer.default_local_dir))
        self.arm_run = checkpoint_dir.parent
        self.master_run = self.arm_run.parent.parent
        self.manifest_path = self.master_run / 'input_manifest.json'
        require(self.manifest_path.is_file(), f'missing formal input manifest: {self.manifest_path}')
        self.manifest = json.loads(self.manifest_path.read_text(encoding='utf-8'))
        require(self.manifest['status'] == 'PASS_FORMAL_CHEMISTRY_INPUTS_FROZEN', 'formal manifest status drift')
        require(self.manifest['chemistry_seed1_authorized'] is True, 'Chemistry seed1 not authorized')
        require(int(self.manifest['rows']) == EXPECTED_ROWS, 'formal manifest row drift')
        require(int(self.manifest['batch_size']) == EXPECTED_BATCH_ROWS, 'formal manifest batch drift')
        require(int(self.manifest['batches_per_pass']) == EXPECTED_BATCHES_PER_PASS, 'formal manifest batch/pass drift')
        require(int(self.manifest['passes']) == EXPECTED_PASSES, 'formal manifest passes drift')
        require(int(self.manifest['total_optimizer_updates_per_arm']) == EXPECTED_CALLS, 'formal manifest updates drift')
        require(self.manifest['row_order_sha256'] == EXPECTED_ROW_ORDER_SHA, 'formal row-order SHA drift')
        require(self.manifest['row_set_sha256'] == EXPECTED_ROW_SET_SHA, 'formal row-set SHA drift')
        require(self.manifest['matched_response_ids_sha256'] == EXPECTED_ARM_SHA, 'formal arm SHA drift')

        self.audit_path = self.arm_run / 'replay_transport_rows.jsonl'
        self.summary_path = self.arm_run / 'replay_progress.json'
        if self.audit_path.exists():
            self.audit_path.unlink()

        self.calls = 0
        self.active_arm: str | None = None
        self.pass_row_ids: list[str] = []
        self.pass_responses: list[list[int]] = []
        self.pass_tokens = 0
        self.completed_passes: list[int] = []

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
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

    def _write_progress(
        self,
        *,
        arm: str,
        pass_index: int,
        batch_index: int,
        status: str,
    ) -> None:
        atomic_json(
            self.summary_path,
            {
                'status': status,
                'arm': arm,
                'completed_calls': self.calls,
                'expected_calls': EXPECTED_CALLS,
                'current_pass': pass_index,
                'current_batch_one_based': batch_index + 1,
                'batches_per_pass': EXPECTED_BATCHES_PER_PASS,
                'completed_passes': list(self.completed_passes),
                'student_rollout_calls': 0,
                'forbidden_rollout_client_accesses': list(self.llm_client.accesses),
                'teacher_top_k': EXPECTED_TOPK,
                'policy_gradient': False,
                'task_rewards': False,
                'chemistry_seed1_authorized': True,
            },
        )

    def _close_pass(self, *, arm: str, pass_index: int) -> None:
        arm_cfg = self.manifest['arms'][arm]
        require(len(self.pass_row_ids) == EXPECTED_ROWS, f'pass {pass_index} row count drift')
        require(len(self.pass_responses) == EXPECTED_ROWS, f'pass {pass_index} response count drift')
        require(sha256_json(self.pass_row_ids) == EXPECTED_ROW_ORDER_SHA, f'pass {pass_index} row-order SHA drift')
        require(
            sha256_json(self.pass_responses) == EXPECTED_ARM_SHA[arm],
            f'pass {pass_index} aggregate response SHA drift',
        )
        expected_tokens = int(arm_cfg['effective_supervised_tokens_per_pass'])
        require(self.pass_tokens == expected_tokens, f'pass {pass_index} token-count drift')

        atomic_json(
            self.arm_run / f'replay_pass_{pass_index}_preupdate.json',
            {
                'status': 'PASS_FULL_FROZEN_PASS_PREUPDATE',
                'arm': arm,
                'pass': pass_index,
                'rows': EXPECTED_ROWS,
                'batches': EXPECTED_BATCHES_PER_PASS,
                'aggregate_response_ids_sha256': EXPECTED_ARM_SHA[arm],
                'row_order_sha256': EXPECTED_ROW_ORDER_SHA,
                'effective_supervised_token_count': self.pass_tokens,
                'fresh_student_rollout_calls': 0,
            },
        )
        self.completed_passes.append(pass_index)
        self.pass_row_ids = []
        self.pass_responses = []
        self.pass_tokens = 0

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        self.calls += 1
        require(self.calls <= EXPECTED_CALLS, f'unexpected replay generate call count={self.calls}')
        require(len(prompts) == EXPECTED_BATCH_ROWS, f'replay rows={len(prompts)} expected={EXPECTED_BATCH_ROWS}')

        zero_call = self.calls - 1
        pass_index = zero_call // EXPECTED_BATCHES_PER_PASS + 1
        batch_index = zero_call % EXPECTED_BATCHES_PER_PASS
        require(1 <= pass_index <= EXPECTED_PASSES, f'bad pass index {pass_index}')

        infos_raw = prompts.non_tensor_batch.get('extra_info')
        require(infos_raw is not None, 'extra_info missing from replay input')
        infos = [as_python(x) for x in infos_raw]
        require(all(isinstance(x, dict) for x in infos), 'extra_info rows must be dicts')

        arms = {str(x.get('arm')) for x in infos}
        require(len(arms) == 1, f'mixed replay arms={arms}')
        arm = next(iter(arms))
        require(arm in EXPECTED_ARM_SHA, f'bad replay arm={arm!r}')
        if self.active_arm is None:
            self.active_arm = arm
        require(self.active_arm == arm, f'arm changed inside manager {self.active_arm}->{arm}')

        arm_cfg = self.manifest['arms'][arm]
        expected_batch = arm_cfg['batches'][batch_index]
        row_ids = [str(x['row_id']) for x in infos]
        require(row_ids == list(expected_batch['row_ids']), f'call {self.calls} row IDs differ from frozen batch schedule')
        require(
            sha256_json(row_ids) == expected_batch['row_ids_sha256'],
            f'call {self.calls} batch row SHA drift',
        )

        frozen_responses: list[list[int]] = []
        student_prompt_ids_all: list[list[int]] = []
        teacher_prompt_ids_all: list[list[int]] = []
        teacher_routing_keys: list[Any] = []
        row_audits: list[dict[str, Any]] = []

        for idx, info in enumerate(infos):
            require(info.get('domain') == 'chemistry', f'row {idx} non-chemistry domain')
            require(str(info.get('arm')) == arm, f'row {idx} arm mismatch')
            require(info.get('student_hint_present') is False, f"{info['row_id']} Student hint flag drift")
            require(info.get('teacher_hint_present') is True, f"{info['row_id']} Teacher hint flag drift")

            response_ids = [int(x) for x in info['frozen_response_ids']]
            student_prompt_ids = [int(x) for x in info['student_prompt_ids']]
            teacher_prompt_ids = [int(x) for x in info['teacher_prompt_ids']]
            require(len(response_ids) == int(info['m_i']), f"{info['row_id']} frozen response length mismatch")
            require(
                sha256_json(response_ids) == str(info['frozen_response_sha256']),
                f"{info['row_id']} frozen response SHA mismatch",
            )

            student_user = DIRECT_PROMPT.format(source=str(info['src_text']))
            rendered = self.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': student_user}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            runtime_student_ids = [
                int(x)
                for x in self.tokenizer(rendered, add_special_tokens=False)['input_ids']
            ]
            require(runtime_student_ids == student_prompt_ids, f"{info['row_id']} Student prompt token drift")
            require(
                sha256_json(student_prompt_ids) == str(info['student_prompt_sha256']),
                f"{info['row_id']} Student prompt SHA drift",
            )
            require(
                sha256_json(teacher_prompt_ids) == str(info['teacher_prompt_sha256']),
                f"{info['row_id']} Teacher prompt SHA drift",
            )

            expected_teacher_user = TEACHER_CHEM_TEMPLATE.format(
                src_term=str(info['src_term']),
                en_name=str(info['knowledge']),
                source=str(info['src_text']),
            )
            require(
                str(info['teacher_user_text']) == expected_teacher_user,
                f"{info['row_id']} Teacher prompt text drift",
            )
            require(
                str(info['student_user_text']) == student_user,
                f"{info['row_id']} Student prompt text drift",
            )

            frozen_responses.append(response_ids)
            student_prompt_ids_all.append(student_prompt_ids)
            teacher_prompt_ids_all.append(teacher_prompt_ids)
            data_source = 'default'
            if 'data_source' in prompts.non_tensor_batch:
                data_source = as_python(prompts.non_tensor_batch['data_source'][idx])
            teacher_routing_keys.append(data_source)

            row_audits.append(
                {
                    'call': self.calls,
                    'pass': pass_index,
                    'batch_one_based': batch_index + 1,
                    'row_id': str(info['row_id']),
                    'arm': arm,
                    'prefix_sha': str(info['frozen_response_sha256']),
                    'prefix_token_count': len(response_ids),
                    'first_token_ids': response_ids[:8],
                    'last_token_ids': response_ids[-8:],
                    'teacher_hint_present': True,
                    'student_hint_present': False,
                    'student_prompt_sha256': str(info['student_prompt_sha256']),
                    'teacher_prompt_sha256': str(info['teacher_prompt_sha256']),
                    'status': 'PREUPDATE_REPLAY_VERIFIED',
                }
            )

        batch_response_sha = sha256_json(frozen_responses)
        require(
            batch_response_sha == expected_batch['response_ids_sha256'],
            f'call {self.calls} batch response SHA drift',
        )
        batch_tokens = sum(len(x) for x in frozen_responses)
        require(
            batch_tokens == int(expected_batch['effective_supervised_tokens']),
            f'call {self.calls} supervised-token count drift',
        )

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
        require(response_width >= max(map(len, frozen_responses)), 'response width too small')

        prompt_tensors = []
        response_tensors = []
        response_masks = []
        attention_masks = []
        input_tensors = []
        position_tensors = []
        teacher_ids_tensors = []
        teacher_logprob_tensors = []

        for i, (student_prompt_ids, teacher_prompt_ids, response_ids, teacher_result) in enumerate(
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
                side='left',
            )
            response_ids_t, response_attn = pad_ids(
                response_ids,
                width=response_width,
                pad_id=pad_id,
                side='right',
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
                f'{row_ids[i]} Teacher padded batch dimension mismatch',
            )
            require(teacher_ids.shape[-1] == EXPECTED_TOPK, f'{row_ids[i]} Teacher top-k IDs width mismatch')
            require(
                teacher_logprobs.shape[-1] == EXPECTED_TOPK,
                f'{row_ids[i]} Teacher top-k logprob width mismatch',
            )
            require(
                bool(torch.isfinite(teacher_logprobs).all().item()),
                f'{row_ids[i]} non-finite Teacher logprobs',
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
                'prompts': torch.cat(prompt_tensors, dim=0),
                'responses': torch.cat(response_tensors, dim=0),
                'response_mask': torch.cat(response_masks, dim=0),
                'input_ids': torch.cat(input_tensors, dim=0),
                'attention_mask': torch.cat(attention_masks, dim=0),
                'position_ids': torch.cat(position_tensors, dim=0),
                'teacher_ids': torch.cat(teacher_ids_tensors, dim=0),
                'teacher_logprobs': torch.cat(teacher_logprob_tensors, dim=0),
                'rm_scores': torch.zeros((EXPECTED_BATCH_ROWS, response_width), dtype=torch.float32),
            },
            batch_size=EXPECTED_BATCH_ROWS,
        )

        non_tensor_batch = {'__num_turns__': np.ones(EXPECTED_BATCH_ROWS, dtype=np.int32)}
        for key, val in prompts.non_tensor_batch.items():
            non_tensor_batch[key] = np.array(val, copy=True)

        multi_modal_inputs = np.empty(EXPECTED_BATCH_ROWS, dtype=object)
        multi_modal_inputs[:] = [{} for _ in range(EXPECTED_BATCH_ROWS)]
        non_tensor_batch['multi_modal_inputs'] = multi_modal_inputs
        for key in ('turn_scores', 'tool_rewards', 'min_global_steps', 'max_global_steps', 'extras'):
            if key not in non_tensor_batch:
                values = np.empty(EXPECTED_BATCH_ROWS, dtype=object)
                values[:] = [None] * EXPECTED_BATCH_ROWS
                non_tensor_batch[key] = values

        timing = {
            'agent_loop/num_preempted/min': 0.0,
            'agent_loop/num_preempted/max': 0.0,
            'agent_loop/num_preempted/mean': 0.0,
            'agent_loop/generate_sequences/min': 0.0,
            'agent_loop/generate_sequences/max': 0.0,
            'agent_loop/generate_sequences/mean': 0.0,
            'agent_loop/tool_calls/min': 0.0,
            'agent_loop/tool_calls/max': 0.0,
            'agent_loop/tool_calls/mean': 0.0,
            'agent_loop/compute_score/min': 0.0,
            'agent_loop/compute_score/max': 0.0,
            'agent_loop/compute_score/mean': 0.0,
        }

        for row in row_audits:
            append_jsonl(self.audit_path, row)

        self.pass_row_ids.extend(row_ids)
        self.pass_responses.extend(frozen_responses)
        self.pass_tokens += batch_tokens
        if batch_index + 1 == EXPECTED_BATCHES_PER_PASS:
            self._close_pass(arm=arm, pass_index=pass_index)

        final = self.calls == EXPECTED_CALLS
        if final:
            require(self.completed_passes == [1, 2, 3], f'completed pass drift {self.completed_passes}')
            require(self.pass_row_ids == [], 'residual pass row IDs at final call')
            require(self.pass_responses == [], 'residual pass responses at final call')
            require(self.pass_tokens == 0, 'residual pass tokens at final call')

        self._write_progress(
            arm=arm,
            pass_index=pass_index,
            batch_index=batch_index,
            status='PASS_FORMAL_REPLAY_COMPLETE' if final else 'PASS_BATCH_PREUPDATE_REPLAY',
        )

        print(
            'PREFIX_REPLAY_PREUPDATE_PASS '
            f'arm={arm} call={self.calls}/{EXPECTED_CALLS} pass={pass_index}/{EXPECTED_PASSES} '
            f'batch={batch_index + 1}/{EXPECTED_BATCHES_PER_PASS} rows={EXPECTED_BATCH_ROWS} '
            f'batch_sha={batch_response_sha}',
            flush=True,
        )
        if batch_index + 1 == EXPECTED_BATCHES_PER_PASS:
            print(
                'PREFIX_REPLAY_FULL_PASS_PREUPDATE_PASS '
                f'arm={arm} pass={pass_index} aggregate_sha={EXPECTED_ARM_SHA[arm]} '
                f'effective_tokens={int(arm_cfg["effective_supervised_tokens_per_pass"])}',
                flush=True,
            )
        if final:
            print(
                f'PREFIX_REPLAY_ALL_PASSES_PREUPDATE_PASS arm={arm} calls={self.calls} passes=3',
                flush=True,
            )

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info={
                'timing': timing,
                'metrics': [],
                'reward_extra_keys': [],
                'prefix_replay': {
                    'arm': arm,
                    'call': self.calls,
                    'pass': pass_index,
                    'batch': batch_index + 1,
                    'rows': EXPECTED_BATCH_ROWS,
                    'batch_response_ids_sha256': batch_response_sha,
                    'fresh_student_rollout_calls': 0,
                },
            },
        )
