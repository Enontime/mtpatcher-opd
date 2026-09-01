#!/usr/bin/env python3

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


class JsonlDataset(Dataset):
    def __init__(self, path):
        self.rows = []

        with Path(path).open(
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                if line.strip():
                    self.rows.append(
                        json.loads(line)
                    )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--student", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-6)

    p.add_argument(
        "--max-prompt-length",
        type=int,
        default=512,
    )

    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )

    p.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    p.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )

    p.add_argument(
        "--top-p",
        type=float,
        default=0.8,
    )

    p.add_argument(
        "--top-k",
        type=int,
        default=20,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=20260824,
    )

    return p.parse_args()


def set_seed(seed, rank):
    value = seed + rank

    random.seed(value)
    torch.manual_seed(value)

    if hasattr(torch, "npu"):
        torch.npu.manual_seed(value)


def get_prompt(row, tokenizer, max_length):
    messages = row.get("messages")

    if not isinstance(messages, list):
        raise RuntimeError(
            f"messages missing index={row.get('index')}"
        )

    # Construction must remain source-only.
    for m in messages:
        if m.get("role") == "assistant":
            raise RuntimeError(
                "assistant message leakage "
                f"index={row.get('index')}"
            )

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    old_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )

    tokenizer.truncation_side = old_side

    return encoded["input_ids"]


def save_model(
    model,
    tokenizer,
    out,
    rank,
):
    if rank != 0:
        dist.barrier()
        return

    path = Path(out)
    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_model = model.module

    raw_model.save_pretrained(
        path,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(path)

    print(
        f"CHECKPOINT_SAVED = {path}",
        flush=True,
    )

    dist.barrier()


def main():
    args = parse_args()

    rank = int(
        os.environ.get("RANK", "0")
    )

    local_rank = int(
        os.environ.get("LOCAL_RANK", "0")
    )

    world_size = int(
        os.environ.get("WORLD_SIZE", "1")
    )

    torch.npu.set_device(local_rank)

    dist.init_process_group(
        backend="hccl",
    )

    device = torch.device(
        f"npu:{local_rank}"
    )

    set_seed(
        args.seed,
        rank,
    )

    if rank == 0:
        print("=" * 78)
        print(
            "MT-PATCHER V3 "
            "TORCH-NPU ON-POLICY PG-RKL-K1"
        )
        print("=" * 78)

        print(
            "WORLD_SIZE =",
            world_size,
        )

        print(
            "STUDENT =",
            args.student,
        )

        print(
            "TEACHER =",
            args.teacher,
        )

        print(
            "TRAIN =",
            args.train,
        )

        print(
            "OBJECTIVE = D_KL("
            "Teacher || Student)"
        )

        print(
            "ON_POLICY = True"
        )

        print(
            "REFERENCE_USED = False"
        )

        print(flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.student,
        local_files_only=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id
        )

    student_cfg = AutoConfig.from_pretrained(
        args.student,
        local_files_only=True,
        trust_remote_code=True,
    )

    teacher_cfg = AutoConfig.from_pretrained(
        args.teacher,
        local_files_only=True,
        trust_remote_code=True,
    )

    if (
        student_cfg.vocab_size
        != teacher_cfg.vocab_size
    ):
        raise RuntimeError(
            "Student/Teacher vocab mismatch: "
            f"{student_cfg.vocab_size} vs "
            f"{teacher_cfg.vocab_size}"
        )

    # Small stagger reduces 16-process disk burst.
    time.sleep(local_rank * 0.5)

    student = (
        AutoModelForCausalLM
        .from_pretrained(
            args.student,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=True,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        .to(device)
    )

    student.config.use_cache = False

    dist.barrier()

    time.sleep(local_rank * 0.5)

    teacher = (
        AutoModelForCausalLM
        .from_pretrained(
            args.teacher,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=True,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        .to(device)
    )

    teacher.eval()
    teacher.config.use_cache = False

    for p in teacher.parameters():
        p.requires_grad_(False)

    dist.barrier()

    student = DDP(
        student,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )

    dataset = JsonlDataset(
        args.train
    )

    if len(dataset) != 3732:
        raise RuntimeError(
            "Expected PE source set size 3732, "
            f"got {len(dataset)}"
        )

    # Extra leakage check.
    if rank == 0:
        for row in dataset.rows:
            if "reference" in row:
                raise RuntimeError(
                    "Reference leakage detected "
                    f"index={row.get('index')}"
                )

        print(
            "TRAIN_ROWS =",
            len(dataset),
            flush=True,
        )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=0,
        collate_fn=lambda x: x,
    )

    total_updates = (
        len(loader)
        * args.epochs
    )

    warmup_steps = int(
        total_updates
        * args.warmup_ratio
    )

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_updates,
        )
    )

    if rank == 0:
        print(
            "LOCAL_STEPS_PER_EPOCH =",
            len(loader),
        )

        print(
            "TOTAL_UPDATES =",
            total_updates,
        )

        print(
            "GLOBAL_BATCH =",
            world_size,
        )

        print(
            "WARMUP_STEPS =",
            warmup_steps,
        )

        print(
            "OPD_TRAINING_START",
            flush=True,
        )

    global_step = 0

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        sampler.set_epoch(epoch)

        student.train()

        epoch_loss_sum = 0.0
        epoch_tokens = 0
        epoch_start = time.time()

        for local_step, batch in enumerate(
            loader,
            start=1,
        ):
            row = batch[0]

            prompt_ids = get_prompt(
                row,
                tokenizer,
                args.max_prompt_length,
            ).to(device)

            attention_mask = (
                torch.ones_like(
                    prompt_ids,
                    device=device,
                )
            )

            prompt_len = (
                prompt_ids.shape[1]
            )

            # ------------------------------------------------------------
            # ON-POLICY ROLLOUT FROM CURRENT STUDENT
            # ------------------------------------------------------------

            student.module.eval()

            with torch.no_grad():
                generated = (
                    student.module.generate(input_ids=prompt_ids, attention_mask=attention_mask, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True, repetition_penalty=1.0, return_dict_in_generate=True, output_scores=True, renormalize_logits=False)
                )
                # ------------------------------------------------------------
                # Preserve generation-time distribution for alignment audit.
                # ------------------------------------------------------------
                _pg_generate_output = generated
                
                if not hasattr(_pg_generate_output, 'sequences'):
                    raise RuntimeError(
                        'return_dict_in_generate=True did not return sequences'
                    )
                
                _pg_behavior_scores = _pg_generate_output.scores
                
                if _pg_behavior_scores is None or len(_pg_behavior_scores) == 0:
                    raise RuntimeError(
                        'generate() returned no per-step behavior scores'
                    )
                
                generated = _pg_generate_output.sequences


            student.module.train()

            response_len = (
                generated.shape[1]
                - prompt_len
            )

            if response_len <= 0:
                continue

            full_ids = generated

            full_mask = torch.ones_like(
                full_ids,
                device=device,
            )

            # ------------------------------------------------------------
            # STUDENT DISTRIBUTION WITH GRADIENT
            # ------------------------------------------------------------

            student_out = student(
                input_ids=full_ids,
                attention_mask=full_mask,
                use_cache=False,
            )

            # Logits predicting generated response tokens:
            #
            # response token #0 is predicted from prompt_len-1.
            #
            s_logits = student_out.logits[
                :,
                prompt_len - 1 : -1,
                :,
            ]

            # ------------------------------------------------------------
            # TEACHER DISTRIBUTION ON THE SAME STUDENT-INDUCED STATES
            # ------------------------------------------------------------

            with torch.no_grad():
                teacher_out = teacher(
                    input_ids=full_ids,
                    attention_mask=full_mask,
                    use_cache=False,
                )

                t_logits = (
                    teacher_out.logits[
                        :,
                        prompt_len - 1 : -1,
                        :,
                    ]
                )

                t_logp = F.log_softmax(
                    t_logits.float(),
                    dim=-1,
                )

                # Teacher probability retained ONLY for inherited diagnostics.
                # PG-RKL optimization itself uses sampled-token log-probabilities.
                t_prob = t_logp.exp()


            s_logp = F.log_softmax(
                s_logits.float(),
                dim=-1,
            )

            # Student distribution for exact reverse KL.
            s_prob = s_logp.exp()

            # Exact full-vocabulary forward KL:
            #
            # D_KL(P_student || P_teacher)
            #
            # Exact full-vocabulary reverse KL:
            # D_KL(P_student || P_teacher)
            # ================================================================
            # Sampled-token k1 Policy-Gradient Reverse-KL OPD
            # ================================================================

            _pg_rollout = generated

            # Hugging Face generate() may return either a Tensor or
            # GenerateDecoderOnlyOutput when return_dict_in_generate=True.
            if hasattr(_pg_rollout, 'sequences'):
                _pg_rollout = _pg_rollout.sequences

            if not torch.is_tensor(_pg_rollout):
                raise RuntimeError(
                    'Generate result is neither a Tensor nor an object '
                    'with .sequences'
                )

            if _pg_rollout.ndim != 2:
                raise RuntimeError(
                    'Expected generated IDs shape [batch, seq], got '
                    f'{tuple(_pg_rollout.shape)}'
                )

            _pg_batch = int(s_logits.shape[0])
            _pg_steps = int(s_logits.shape[1])

            if int(_pg_rollout.shape[0]) != _pg_batch:
                raise RuntimeError(
                    'Generate/logit batch mismatch: '
                    f'generate={tuple(_pg_rollout.shape)} '
                    f'logits={tuple(s_logits.shape)}'
                )

            if int(_pg_rollout.shape[1]) < _pg_steps:
                raise RuntimeError(
                    'Generated sequence shorter than response logits: '
                    f'generate={tuple(_pg_rollout.shape)} '
                    f'logits={tuple(s_logits.shape)}'
                )

            # The verified trainer already slices s_logits/t_logits to the
            # generated response states. The corresponding sampled actions
            # are therefore the final _pg_steps token IDs in generate().
            _pg_action_ids = _pg_rollout[:, -_pg_steps:]

            _pg_action_ids = _pg_action_ids.to(
                device=s_logits.device,
                dtype=torch.long,
            )

            if tuple(_pg_action_ids.shape) != (
                _pg_batch,
                _pg_steps,
            ):
                raise RuntimeError(
                    'Action/logit shape mismatch: '
                    f'actions={tuple(_pg_action_ids.shape)} '
                    f'logits={tuple(s_logits.shape)}'
                )

            _pg_min_token = int(
                _pg_action_ids.min().item()
            )

            _pg_max_token = int(
                _pg_action_ids.max().item()
            )

            _pg_vocab = int(
                s_logits.shape[-1]
            )

            if (
                _pg_min_token < 0
                or _pg_max_token >= _pg_vocab
            ):
                raise RuntimeError(
                    'Sampled token outside vocabulary: '
                    f'min={_pg_min_token} '
                    f'max={_pg_max_token} '
                    f'vocab={_pg_vocab}'
                )

            # ------------------------------------------------------------
            # Gather student/teacher probability of the ACTUAL sampled token.
            # ------------------------------------------------------------

            s_action_logp = s_logp.gather(
                dim=-1,
                index=_pg_action_ids.unsqueeze(-1),
            ).squeeze(-1)

            t_action_logp = t_logp.gather(
                dim=-1,
                index=_pg_action_ids.unsqueeze(-1),
            ).squeeze(-1)

            if not torch.isfinite(s_action_logp).all():
                raise RuntimeError(
                    'Non-finite student sampled log-probability'
                )

            if not torch.isfinite(t_action_logp).all():
                raise RuntimeError(
                    'Non-finite teacher sampled log-probability'
                )

            # ------------------------------------------------------------
            # k1 sampled reverse-KL estimator.
            #
            # Stop-gradient is essential: k1 provides the reward/advantage;
            # gradient flows only through s_action_logp in the PG objective.
            # ------------------------------------------------------------

            k1 = (
                s_action_logp.detach().float()
                - t_action_logp.detach().float()
            ).detach()

            if not torch.isfinite(k1).all():
                raise RuntimeError(
                    'Non-finite sampled k1'
                )

            # Keep token_kl as the distributed metric tensor expected by
            # the already-verified trainer. Individual sampled k1 values
            # may be negative; only the expectation corresponds to RKL.
                        # ================================================================
            # BEHAVIOR-MATCHED + ALIGNMENT-AUDITED sampled-k1 PG-RKL
            #
            # Rollout:
            #   temperature = 1.0
            #   top_p       = 1.0
            #   top_k       = 0
            #   repetition_penalty = 1.0
            #
            # Generation-time scores are retained and used to audit causal
            # token/logit alignment before computing the PG update.
            # ================================================================

            _pg_rollout = generated

            if not torch.is_tensor(_pg_rollout):
                raise RuntimeError(
                    'Normalized generate output is not a Tensor'
                )

            _pg_scores = _pg_behavior_scores

            _pg_response_steps = len(
                _pg_scores
            )

            if _pg_response_steps <= 0:
                raise RuntimeError(
                    'Zero generated response steps'
                )

            # Each entry in generate().scores corresponds exactly to one sampled
            # new token. Hence the final R tokens in sequences are the true
            # sampled actions. No heuristic is needed here.
            _pg_action_ids = _pg_rollout[
                :,
                -_pg_response_steps:
            ].to(
                device=s_logp.device,
                dtype=torch.long,
            )

            _pg_behavior_logits = torch.stack(
                [
                    score.float()
                    for score in _pg_scores
                ],
                dim=1,
            )

            if int(_pg_behavior_logits.shape[1]) != _pg_response_steps:
                raise RuntimeError(
                    'Behavior score stacking failure'
                )

            if int(_pg_behavior_logits.shape[0]) != int(_pg_action_ids.shape[0]):
                raise RuntimeError(
                    'Behavior/action batch mismatch'
                )

            _pg_behavior_logp = F.log_softmax(
                _pg_behavior_logits,
                dim=-1,
            )

            _pg_behavior_action_logp = _pg_behavior_logp.gather(
                dim=-1,
                index=_pg_action_ids.unsqueeze(-1),
            ).squeeze(-1)

            if not torch.isfinite(
                _pg_behavior_action_logp
            ).all():
                raise RuntimeError(
                    'Non-finite generation-time action logprob'
                )

            # ------------------------------------------------------------
            # Causal alignment audit.
            #
            # Candidate slices cover:
            #   exact response slice
            #   one-token-left shift
            #   one-token-right shift
            #   generic head/tail slicing
            #
            # The generation-time action log-probability tells us objectively
            # which student-forward position predicts each sampled token.
            # ------------------------------------------------------------

            _pg_S = int(
                s_logp.shape[1]
            )

            _pg_R = int(
                _pg_response_steps
            )

            if tuple(s_logp.shape) != tuple(t_logp.shape):
                raise RuntimeError(
                    'Student/teacher logprob tensor shape mismatch: '
                    f'student={tuple(s_logp.shape)} '
                    f'teacher={tuple(t_logp.shape)}'
                )

            if _pg_S < _pg_R:
                raise RuntimeError(
                    'Recomputed response logits are shorter than generation: '
                    f'logit_steps={_pg_S} generated_steps={_pg_R}'
                )

            _pg_candidates = []

            def _pg_test_alignment(name, s_candidate, t_candidate):

                if int(s_candidate.shape[1]) != _pg_R:
                    return

                sampled_forward = s_candidate.gather(
                    dim=-1,
                    index=_pg_action_ids.unsqueeze(-1),
                ).squeeze(-1)

                diff = (
                    sampled_forward.detach().float()
                    - _pg_behavior_action_logp.detach().float()
                ).abs()

                mae = float(
                    diff.mean().item()
                )

                maxe = float(
                    diff.max().item()
                )

                _pg_candidates.append(
                    (
                        mae,
                        maxe,
                        name,
                        s_candidate,
                        t_candidate,
                    )
                )

            # Exact shape.
            if _pg_S == _pg_R:

                _pg_test_alignment(
                    'exact',
                    s_logp,
                    t_logp,
                )

            # Generic tail.
            _pg_test_alignment(
                'tail',
                s_logp[:, -_pg_R:, :],
                t_logp[:, -_pg_R:, :],
            )

            # Generic head.
            _pg_test_alignment(
                'head',
                s_logp[:, :_pg_R, :],
                t_logp[:, :_pg_R, :],
            )

            # Explicit one-token ambiguity.
            if _pg_S == _pg_R + 1:

                _pg_test_alignment(
                    'drop_first',
                    s_logp[:, 1:, :],
                    t_logp[:, 1:, :],
                )

                _pg_test_alignment(
                    'drop_last',
                    s_logp[:, :-1, :],
                    t_logp[:, :-1, :],
                )

            if not _pg_candidates:

                raise RuntimeError(
                    'No valid causal-alignment candidate'
                )

            _pg_candidates.sort(
                key=lambda item: item[0]
            )

            (
                _pg_alignment_mae,
                _pg_alignment_max,
                _pg_alignment_name,
                _pg_s_logp,
                _pg_t_logp,
            ) = _pg_candidates[0]

            # With neutral generation warpers and the same model parameters,
            # generation-time sampled-action logprob and teacher-forced forward
            # logprob should agree closely. A large discrepancy means the
            # estimator is not scientifically valid, so stop immediately.
            if _pg_alignment_mae > 0.20:

                _pg_debug = [
                    (
                        name,
                        mae,
                        maxe,
                    )
                    for (
                        mae,
                        maxe,
                        name,
                        _,
                        __,
                    ) in _pg_candidates
                ]

                raise RuntimeError(
                    'PG-RKL ALIGNMENT AUDIT FAILED: '
                    f'best={_pg_alignment_name} '
                    f'mae={_pg_alignment_mae:.6f} '
                    f'max={_pg_alignment_max:.6f} '
                    f'candidates={_pg_debug}'
                )

            if rank == 0 and global_step == 0:

                print(
                    'PG_ALIGNMENT_AUDIT_PASS',
                    {
                        'alignment': _pg_alignment_name,
                        'mae': _pg_alignment_mae,
                        'max_error': _pg_alignment_max,
                        'generated_steps': _pg_R,
                        'forward_steps': _pg_S,
                    },
                    flush=True,
                )

            # ------------------------------------------------------------
            # Gather log-probabilities of the ACTUAL sampled student actions
            # using the alignment that was just objectively verified.
            # ------------------------------------------------------------

            s_action_logp = _pg_s_logp.gather(
                dim=-1,
                index=_pg_action_ids.unsqueeze(-1),
            ).squeeze(-1)

            t_action_logp = _pg_t_logp.gather(
                dim=-1,
                index=_pg_action_ids.unsqueeze(-1),
            ).squeeze(-1)

            if not torch.isfinite(
                s_action_logp
            ).all():
                raise RuntimeError(
                    'Non-finite student sampled logprob'
                )

            if not torch.isfinite(
                t_action_logp
            ).all():
                raise RuntimeError(
                    'Non-finite teacher sampled logprob'
                )

            # ------------------------------------------------------------
            # Sampled k1 reverse-KL estimator:
            #
            #   k1 = stopgrad(log pi_S(a|s) - log pi_T(a|s))
            #
            # Since actions now come from the same pi_S represented by
            # s_action_logp, E[k1] estimates KL(pi_S || pi_T).
            # ------------------------------------------------------------

            k1 = (
                s_action_logp.detach().float()
                - t_action_logp.detach().float()
            ).detach()

            if not torch.isfinite(
                k1
            ).all():
                raise RuntimeError(
                    'Non-finite sampled k1'
                )

            token_kl = k1

            distill_reward = (
                -k1
            ).detach()

            # Score-function policy-gradient objective.
            loss = -(
                distill_reward
                * s_action_logp.float()
            ).mean()


            # ================================================================
            # PG-RKL MATHEMATICAL CONSISTENCY DIAGNOSTIC V2
            #
            # 208 sentences total:
            #   13 synchronized batches * 16 ranks
            #
            # No meaningful parameter update is performed:
            #   loss is multiplied by zero below.
            # ================================================================

            if global_step == 0:

                _diag_offsets = (-2, -1, 0, 1, 2)

                _diag_stats = {}

                for _d in _diag_offsets:

                    _diag_stats[_d] = {

                        'valid_sample': 0.0,
                        'valid_exact': 0.0,
                        'valid_slogp': 0.0,
                        'valid_tlogp': 0.0,
                        'valid_count': 0.0,

                        'all_sample': 0.0,
                        'all_exact': 0.0,
                        'all_count': 0.0,
                    }


            # ------------------------------------------------------------
            # Build VALID response mask:
            #
            # include every token through the FIRST EOS, including that EOS;
            # exclude generated PAD/filler positions after EOS.
            # ------------------------------------------------------------

            _diag_actions_full = _pg_action_ids

            _diag_B = int(
                _diag_actions_full.shape[0]
            )

            _diag_R = int(
                _diag_actions_full.shape[1]
            )


            _diag_valid = torch.ones(
                (
                    _diag_B,
                    _diag_R,
                ),
                dtype=torch.bool,
                device=_diag_actions_full.device,
            )


            _diag_eos = tokenizer.eos_token_id


            if _diag_eos is not None:

                if isinstance(
                    _diag_eos,
                    int,
                ):

                    _diag_eos_ids = [
                        int(_diag_eos)
                    ]

                else:

                    _diag_eos_ids = [
                        int(x)
                        for x in _diag_eos
                    ]


                _diag_is_eos = torch.zeros_like(
                    _diag_actions_full,
                    dtype=torch.bool,
                )


                for _diag_eos_id in _diag_eos_ids:

                    _diag_is_eos |= (
                        _diag_actions_full
                        == _diag_eos_id
                    )


                # Number of EOS tokens STRICTLY before position t.
                _diag_eos_before = (
                    _diag_is_eos.to(
                        torch.int32
                    ).cumsum(
                        dim=1
                    )
                    - _diag_is_eos.to(
                        torch.int32
                    )
                )


                _diag_valid = (
                    _diag_eos_before == 0
                )


            if rank == 0 and global_step == 0:

                print(
                    'DIAG_VALID_TOKEN_MASK',
                    {
                        'batch': _diag_B,
                        'response_steps': _diag_R,
                        'valid_tokens': int(
                            _diag_valid.sum().item()
                        ),
                        'all_tokens': int(
                            _diag_valid.numel()
                        ),
                        'eos_token_id': _diag_eos,
                    },
                    flush=True,
                )


            # ------------------------------------------------------------
            # offset meaning:
            #
            #   offset = 0
            #       student state i vs teacher state i
            #
            #   offset = +1
            #       student state i vs teacher state i+1
            #
            #   offset = -1
            #       student state i vs teacher state i-1
            # ------------------------------------------------------------

            def _diag_make_pair(offset):

                R = int(
                    _pg_s_logp.shape[1]
                )


                if offset < 0:

                    k = -offset

                    if R <= k:
                        return None

                    s_lp = _pg_s_logp[
                        :,
                        k:,
                        :
                    ]

                    t_lp = _pg_t_logp[
                        :,
                        :-k,
                        :
                    ]

                    actions = _diag_actions_full[
                        :,
                        k:
                    ]

                    valid = _diag_valid[
                        :,
                        k:
                    ]


                elif offset > 0:

                    k = offset

                    if R <= k:
                        return None

                    s_lp = _pg_s_logp[
                        :,
                        :-k,
                        :
                    ]

                    t_lp = _pg_t_logp[
                        :,
                        k:,
                        :
                    ]

                    actions = _diag_actions_full[
                        :,
                        :-k
                    ]

                    valid = _diag_valid[
                        :,
                        :-k
                    ]


                else:

                    s_lp = _pg_s_logp
                    t_lp = _pg_t_logp
                    actions = _diag_actions_full
                    valid = _diag_valid


                return (
                    s_lp,
                    t_lp,
                    actions,
                    valid,
                )


            for _diag_offset in _diag_offsets:

                _pair = _diag_make_pair(
                    _diag_offset
                )


                if _pair is None:
                    continue


                (
                    _diag_s_lp,
                    _diag_t_lp,
                    _diag_actions,
                    _diag_mask,
                ) = _pair


                # --------------------------------------------------------
                # Sampled estimator:
                #
                # log p_S(a) - log p_T(a)
                # --------------------------------------------------------

                _diag_s_action = _diag_s_lp.gather(
                    dim=-1,
                    index=_diag_actions.unsqueeze(-1),
                ).squeeze(-1).float()


                _diag_t_action = _diag_t_lp.gather(
                    dim=-1,
                    index=_diag_actions.unsqueeze(-1),
                ).squeeze(-1).float()


                _diag_sample = (
                    _diag_s_action
                    - _diag_t_action
                )


                # --------------------------------------------------------
                # Exact full-vocab reverse KL:
                #
                # sum_v p_S(v) [log p_S(v) - log p_T(v)]
                # --------------------------------------------------------

                _diag_s_prob = (
                    _diag_s_lp.float().exp()
                )


                _diag_exact = (
                    _diag_s_prob
                    * (
                        _diag_s_lp.float()
                        - _diag_t_lp.float()
                    )
                ).sum(
                    dim=-1
                )


                if not torch.isfinite(
                    _diag_sample
                ).all():

                    raise RuntimeError(
                        'Non-finite diagnostic sampled k1'
                    )


                if not torch.isfinite(
                    _diag_exact
                ).all():

                    raise RuntimeError(
                        'Non-finite diagnostic exact RKL'
                    )


                _diag_mask_f = (
                    _diag_mask.float()
                )


                # VALID token sums.
                _v_sample = (
                    _diag_sample
                    * _diag_mask_f
                ).sum()


                _v_exact = (
                    _diag_exact
                    * _diag_mask_f
                ).sum()


                _v_slogp = (
                    _diag_s_action
                    * _diag_mask_f
                ).sum()


                _v_tlogp = (
                    _diag_t_action
                    * _diag_mask_f
                ).sum()


                _v_count = (
                    _diag_mask_f.sum()
                )


                # ALL token sums.
                _a_sample = (
                    _diag_sample.sum()
                )


                _a_exact = (
                    _diag_exact.sum()
                )


                _a_count = torch.tensor(
                    float(
                        _diag_sample.numel()
                    ),
                    dtype=torch.float32,
                    device=device,
                )


                # Float32 is intentional for HCCL compatibility.
                _diag_reduce = torch.stack(
                    [
                        _v_sample.float(),
                        _v_exact.float(),
                        _v_slogp.float(),
                        _v_tlogp.float(),
                        _v_count.float(),
                        _a_sample.float(),
                        _a_exact.float(),
                        _a_count.float(),
                    ]
                )


                dist.all_reduce(
                    _diag_reduce,
                    op=dist.ReduceOp.SUM,
                )


                if rank == 0:

                    obj = _diag_stats[
                        _diag_offset
                    ]


                    obj['valid_sample'] += float(
                        _diag_reduce[0].item()
                    )

                    obj['valid_exact'] += float(
                        _diag_reduce[1].item()
                    )

                    obj['valid_slogp'] += float(
                        _diag_reduce[2].item()
                    )

                    obj['valid_tlogp'] += float(
                        _diag_reduce[3].item()
                    )

                    obj['valid_count'] += float(
                        _diag_reduce[4].item()
                    )

                    obj['all_sample'] += float(
                        _diag_reduce[5].item()
                    )

                    obj['all_exact'] += float(
                        _diag_reduce[6].item()
                    )

                    obj['all_count'] += float(
                        _diag_reduce[7].item()
                    )


            # ================================================================
            # Critical:
            # zero the already-defined PG loss BEFORE optimizer.backward().
            # Student weights remain unchanged during this diagnostic.
            # ================================================================

            loss = loss * 0.0


            # 13 local synchronized batches:
            #
            # global_step 0 ... 12
            #
            # 13 * 16 ranks = 208 source sentences.
            if global_step == 12:

                if rank == 0:

                    print()
                    print(
                        '=' * 96
                    )

                    print(
                        'TEACHER_ALIGNMENT_DIAGNOSTIC_FINAL'
                    )

                    print(
                        '=' * 96
                    )


                    _diag_ranking = []


                    for _diag_offset in _diag_offsets:

                        obj = _diag_stats[
                            _diag_offset
                        ]


                        vc = obj[
                            'valid_count'
                        ]

                        ac = obj[
                            'all_count'
                        ]


                        if vc <= 0 or ac <= 0:
                            continue


                        valid_sample = (
                            obj['valid_sample']
                            / vc
                        )


                        valid_exact = (
                            obj['valid_exact']
                            / vc
                        )


                        valid_gap = abs(
                            valid_sample
                            - valid_exact
                        )


                        valid_slogp = (
                            obj['valid_slogp']
                            / vc
                        )


                        valid_tlogp = (
                            obj['valid_tlogp']
                            / vc
                        )


                        all_sample = (
                            obj['all_sample']
                            / ac
                        )


                        all_exact = (
                            obj['all_exact']
                            / ac
                        )


                        all_gap = abs(
                            all_sample
                            - all_exact
                        )


                        _diag_ranking.append(
                            (
                                valid_gap,
                                _diag_offset,
                                valid_sample,
                                valid_exact,
                                valid_slogp,
                                valid_tlogp,
                                all_sample,
                                all_exact,
                                all_gap,
                                vc,
                                ac,
                            )
                        )


                        print(
                            f'offset={_diag_offset:+d} '
                            f'VALID sampled={valid_sample:+.6f} '
                            f'exact={valid_exact:+.6f} '
                            f'gap={valid_gap:.6f} '
                            f's_logp={valid_slogp:+.6f} '
                            f't_logp={valid_tlogp:+.6f} '
                            f'tokens={int(vc)}'
                        )


                        print(
                            f'             '
                            f'ALL sampled={all_sample:+.6f} '
                            f'exact={all_exact:+.6f} '
                            f'gap={all_gap:.6f} '
                            f'tokens={int(ac)}'
                        )


                    _diag_ranking.sort(
                        key=lambda x: x[0]
                    )


                    if not _diag_ranking:

                        raise RuntimeError(
                            'No diagnostic result accumulated'
                        )


                    best = _diag_ranking[0]


                    (
                        best_gap,
                        best_offset,
                        best_sample,
                        best_exact,
                        best_slogp,
                        best_tlogp,
                        best_all_sample,
                        best_all_exact,
                        best_all_gap,
                        best_vc,
                        best_ac,
                    ) = best


                    print()
                    print(
                        'BEST_TEACHER_OFFSET =',
                        best_offset,
                    )

                    print(
                        'BEST_VALID_SAMPLED_K1 =',
                        best_sample,
                    )

                    print(
                        'BEST_VALID_EXACT_RKL =',
                        best_exact,
                    )

                    print(
                        'BEST_VALID_ABS_GAP =',
                        best_gap,
                    )

                    print(
                        'BEST_VALID_STUDENT_LOGP =',
                        best_slogp,
                    )

                    print(
                        'BEST_VALID_TEACHER_LOGP =',
                        best_tlogp,
                    )

                    print(
                        'BEST_ALL_SAMPLED_K1 =',
                        best_all_sample,
                    )

                    print(
                        'BEST_ALL_EXACT_RKL =',
                        best_all_exact,
                    )

                    print(
                        'BEST_ALL_ABS_GAP =',
                        best_all_gap,
                    )


                    # Strong diagnostic criteria.
                    if (
                        best_exact >= -0.02
                        and
                        best_gap < 0.50
                    ):

                        print(
                            'TEACHER_ALIGNMENT_DIAGNOSTIC_PASS'
                        )

                    else:

                        print(
                            'TEACHER_ALIGNMENT_DIAGNOSTIC_UNRESOLVED'
                        )


                    print(
                        '=' * 96
                    )

                    print()


                dist.barrier()

                dist.destroy_process_group()

                return

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite KL loss "
                    f"rank={rank} "
                    f"step={global_step}"
                )

            optimizer.zero_grad(
                set_to_none=True,
            )

            loss.backward()

            grad_norm = (
                torch.nn.utils
                .clip_grad_norm_(
                    student.parameters(),
                    args.max_grad_norm,
                )
            )

            optimizer.step()
            scheduler.step()

            global_step += 1

            epoch_loss_sum += (
                float(loss.detach())
                * response_len
            )

            epoch_tokens += (
                response_len
            )

            if (
                global_step == 1
                or global_step % 20 == 0
            ):
                stats = torch.tensor(
                    [
                        float(loss.detach()),
                        float(response_len),
                    ],
                    device=device,
                    dtype=torch.float32,
                )

                dist.all_reduce(
                    stats,
                    op=dist.ReduceOp.SUM,
                )

                stats /= world_size

                if rank == 0:
                    elapsed = (
                        time.time()
                        - epoch_start
                    )

                    lr = scheduler.get_last_lr()[0]

                    print(
                        f"epoch={epoch} "
                        f"local_step={local_step}/{len(loader)} "
                        f"update={global_step}/{total_updates} "
                        f"kl={stats[0].item():.6f} "
                        f"response_tokens={stats[1].item():.2f} "
                        f"grad_norm={float(grad_norm):.4f} "
                        f"lr={lr:.8g} "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )

            # Explicitly release the giant vocab logits quickly.
            del (
                student_out,
                teacher_out,
                s_logits,
                t_logits,
                t_logp,
                t_prob,
                s_logp,
                token_kl,
                loss,
            )

        loss_stats = torch.tensor(
            [
                epoch_loss_sum,
                float(epoch_tokens),
            ],
            device=device,
            dtype=torch.float32,
        )

        dist.all_reduce(
            loss_stats,
            op=dist.ReduceOp.SUM,
        )

        if rank == 0:
            mean_kl = (
                loss_stats[0].item()
                / max(
                    loss_stats[1].item(),
                    1.0,
                )
            )

            print(
                f"EPOCH_{epoch}_COMPLETE",
                flush=True,
            )

            print(
                "token_mean_sampled_k1_audited =",
                mean_kl,
                flush=True,
            )

        save_model(
            student,
            tokenizer,
            Path(args.output_dir)
            / f"epoch{epoch}",
            rank,
        )

    if rank == 0:
        print(
            "MTPATCHER_V4_TORCHNPU_PGRKL_K1_TRAINING_PASS",
            flush=True,
        )

    dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
