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
            "TORCH-NPU ON-POLICY REVERSE-KL"
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
                # CLEANROOM: preserve the ACTUAL behavior-policy distribution
                # used for each sampled Student action.
                # ------------------------------------------------------------

                _pg_generate_output = generated

                _pg_behavior_scores = (
                    _pg_generate_output.scores
                )

                generated = (
                    _pg_generate_output.sequences
                )

                if (
                    _pg_behavior_scores is None
                    or len(_pg_behavior_scores) == 0
                ):

                    raise RuntimeError(
                        "CLEANROOM generate() returned no behavior scores"
                    )


                _pg_response_steps = len(
                    _pg_behavior_scores
                )


                _pg_action_ids = generated[
                    :,
                    -_pg_response_steps:
                ]


                _pg_behavior_logp = F.log_softmax(
                    torch.stack(
                        tuple(_pg_behavior_scores),
                        dim=1,
                    ).float(),
                    dim=-1,
                )


                _pg_old_action_logp = (
                    _pg_behavior_logp.gather(
                        dim=-1,
                        index=_pg_action_ids.unsqueeze(-1),
                    ).squeeze(-1)
                ).detach()

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


            s_logp = F.log_softmax(
                s_logits.float(),
                dim=-1,
            )

            # Student distribution for exact sampled-k1 PG-RKL.
            s_prob = s_logp.exp()

            # Exact full-vocabulary forward KL:
            #
            # D_KL(P_student || P_teacher)
            #
            # Exact full-vocabulary sampled-k1 PG-RKL:
            # D_KL(P_student || P_teacher)

            # ============================================================
            # CLEAN-ROOM SAMPLED-K1 PG-RKL
            #
            # action a_t ~ behavior Student policy
            #
            # k1 = stopgrad(
            #     log pi_student(a_t | state_t)
            #     -
            #     log pi_teacher(a_t | state_t)
            # )
            #
            # advantage = -k1
            #
            # PPO-style importance ratio:
            #
            # ratio = pi_current(a_t) / pi_behavior(a_t)
            # ============================================================

            _pg_R = int(
                _pg_response_steps
            )


            if tuple(
                s_logp.shape
            ) != tuple(
                t_logp.shape
            ):

                raise RuntimeError(
                    "CLEANROOM student/teacher logprob shape mismatch: "
                    f"student={tuple(s_logp.shape)} "
                    f"teacher={tuple(t_logp.shape)}"
                )


            if int(
                s_logp.shape[1]
            ) < _pg_R:

                raise RuntimeError(
                    "CLEANROOM response logits shorter than rollout: "
                    f"logits={int(s_logp.shape[1])} "
                    f"rollout={_pg_R}"
                )


            # Exact response tail.
            _pg_s_logp = s_logp[
                :,
                -_pg_R:,
                :
            ]


            _pg_t_logp = t_logp[
                :,
                -_pg_R:,
                :
            ]


            s_action_logp = (
                _pg_s_logp.gather(
                    dim=-1,
                    index=_pg_action_ids.unsqueeze(-1),
                ).squeeze(-1)
            )


            t_action_logp = (
                _pg_t_logp.gather(
                    dim=-1,
                    index=_pg_action_ids.unsqueeze(-1),
                ).squeeze(-1)
            )


            if not torch.isfinite(
                s_action_logp
            ).all():

                raise RuntimeError(
                    "CLEANROOM non-finite student sampled logprob"
                )


            if not torch.isfinite(
                t_action_logp
            ).all():

                raise RuntimeError(
                    "CLEANROOM non-finite teacher sampled logprob"
                )


            # ------------------------------------------------------------
            # Behavior-policy / recomputed-policy audit.
            # ------------------------------------------------------------

            _pg_behavior_abs_diff = (
                s_action_logp.detach().float()
                -
                _pg_old_action_logp.detach().float()
            ).abs()


            _pg_alignment_mae = float(
                _pg_behavior_abs_diff.mean().item()
            )


            _pg_alignment_max = float(
                _pg_behavior_abs_diff.max().item()
            )


            if _pg_alignment_mae > 0.20:

                raise RuntimeError(
                    "CLEANROOM behavior alignment failed: "
                    f"mae={_pg_alignment_mae} "
                    f"max={_pg_alignment_max}"
                )


            if rank == 0 and global_step == 0:

                print(
                    "CLEAN_PG_RUNTIME_AUDIT_PASS",
                    {
                        "alignment": "exact_tail",
                        "mae": _pg_alignment_mae,
                        "max_error": _pg_alignment_max,
                        "generated_steps": _pg_R,
                        "forward_steps": int(
                            s_logp.shape[1]
                        ),
                    },
                    flush=True,
                )


            # ------------------------------------------------------------
            # UNIQUE sampled reverse-KL estimator.
            # ------------------------------------------------------------

            k1 = (
                s_action_logp.detach().float()
                -
                t_action_logp.detach().float()
            ).detach()


            if not torch.isfinite(
                k1
            ).all():

                raise RuntimeError(
                    "CLEANROOM non-finite sampled k1"
                )


            token_kl = k1


            distill_reward = (
                -k1
            ).detach()


            # ------------------------------------------------------------
            # PPO-style importance ratio.
            #
            # In one-step on-policy training this should remain close to 1.
            # Keeping it explicit matches the old-policy/current-policy
            # structure of PG-style OPD.
            # ------------------------------------------------------------

            _pg_log_ratio = (
                s_action_logp.float()
                -
                _pg_old_action_logp.detach().float()
            )


            _pg_log_ratio = torch.clamp(
                _pg_log_ratio,
                min=-20.0,
                max=20.0,
            )


            importance_ratio = torch.exp(
                _pg_log_ratio
            )


            _pg_clip_eps = 0.20


            pg_unclipped = (
                importance_ratio
                * distill_reward
            )


            pg_clipped = (
                torch.clamp(
                    importance_ratio,
                    1.0 - _pg_clip_eps,
                    1.0 + _pg_clip_eps,
                )
                * distill_reward
            )



            # ============================================================
            # ENTROPY-AWARE ON-POLICY DISTILLATION
            #
            # L_EOPD
            #   = L_clean_PG_RKL
            #   + alpha * I[H_teacher > tau] * FKL_top16
            #
            # Paper-style settings:
            #   tau   = 0.8
            #   alpha = 1.0
            #   k     = 16
            #
            # Reverse-KL PG remains active at ALL response positions.
            # Forward-KL is added only at high-teacher-entropy positions.
            # ============================================================

            _eopd_tau = 0.8
            _eopd_alpha = 1.0
            _eopd_topk = 16


            # ------------------------------------------------------------
            # Preserve the mathematically verified clean PG-RKL objective.
            # ------------------------------------------------------------

            _eopd_pg_loss = -torch.minimum(
                pg_unclipped,
                pg_clipped,
            ).mean()


            # ------------------------------------------------------------
            # Full teacher entropy:
            #
            # H_T(s_t) = -sum_v p_T(v|s_t) log p_T(v|s_t)
            # ------------------------------------------------------------

            _eopd_t_logp = t_logp.float()

            _eopd_t_prob = (
                _eopd_t_logp.exp()
            )


            _eopd_teacher_entropy = -(
                _eopd_t_prob
                * _eopd_t_logp
            ).sum(
                dim=-1
            )


            if not torch.isfinite(
                _eopd_teacher_entropy
            ).all():

                raise RuntimeError(
                    "EOPD teacher entropy contains non-finite values"
                )


            _eopd_high_mask = (
                _eopd_teacher_entropy
                > _eopd_tau
            ).detach()


            # ------------------------------------------------------------
            # Teacher top-16 FKL.
            #
            # Unlike the previous verl-style Top128 experiment,
            # EOPD RENORMALIZES teacher probability inside the selected
            # top-k support.
            # ------------------------------------------------------------

            _eopd_topk_t_logp, _eopd_topk_ids = torch.topk(
                _eopd_t_logp,
                k=_eopd_topk,
                dim=-1,
            )


            _eopd_topk_t_prob_raw = (
                _eopd_topk_t_logp.exp()
            )


            _eopd_topk_mass = (
                _eopd_topk_t_prob_raw.sum(
                    dim=-1,
                    keepdim=True,
                )
            ).clamp_min(
                1.0e-20
            )


            _eopd_topk_t_prob = (
                _eopd_topk_t_prob_raw
                / _eopd_topk_mass
            )


            _eopd_topk_t_logp_norm = (
                torch.log(
                    _eopd_topk_t_prob.clamp_min(
                        1.0e-20
                    )
                )
            )


            _eopd_topk_s_logp = torch.gather(
                s_logp.float(),
                dim=-1,
                index=_eopd_topk_ids,
            )


            _eopd_fkl_top16 = (
                _eopd_topk_t_prob
                * (
                    _eopd_topk_t_logp_norm
                    - _eopd_topk_s_logp
                )
            ).sum(
                dim=-1
            )


            if not torch.isfinite(
                _eopd_fkl_top16
            ).all():

                raise RuntimeError(
                    "EOPD top16 FKL contains non-finite values"
                )


            if float(
                _eopd_fkl_top16.min().detach().item()
            ) < -1.0e-4:

                raise RuntimeError(
                    "EOPD top16 FKL unexpectedly negative"
                )


            # Guard only against tiny floating-point negatives.
            _eopd_fkl_top16 = (
                _eopd_fkl_top16.clamp_min(
                    0.0
                )
            )


            # ------------------------------------------------------------
            # Paper objective averages the gated extra FKL over all
            # response-token positions.
            # ------------------------------------------------------------

            _eopd_fkl_loss = (
                _eopd_high_mask.to(
                    _eopd_fkl_top16.dtype
                )
                * _eopd_fkl_top16
            ).mean()


            loss = (
                _eopd_pg_loss
                + _eopd_alpha
                * _eopd_fkl_loss
            )


            if not torch.isfinite(
                loss
            ):

                raise RuntimeError(
                    "EOPD total loss is non-finite"
                )


            # ------------------------------------------------------------
            # First-batch scientific audit.
            # ------------------------------------------------------------

            if "_eopd_runtime_audit_done" not in locals():

                _eopd_entropy_cpu = (
                    _eopd_teacher_entropy
                    .detach()
                    .float()
                    .reshape(-1)
                    .cpu()
                )


                _eopd_q = torch.quantile(
                    _eopd_entropy_cpu,
                    torch.tensor(
                        [0.25, 0.50, 0.75, 0.90, 0.95],
                        dtype=torch.float32,
                    ),
                )


                _eopd_high_count = int(
                    _eopd_high_mask.sum().detach().item()
                )


                _eopd_token_count = int(
                    _eopd_high_mask.numel()
                )


                if _eopd_high_count > 0:

                    _eopd_high_fkl_mean = float(
                        _eopd_fkl_top16[
                            _eopd_high_mask
                        ].mean().detach().item()
                    )

                else:

                    _eopd_high_fkl_mean = 0.0


                if rank == 0:

                    print(
                        "EOPD_RUNTIME_AUDIT",
                        {
                            "tau": _eopd_tau,
                            "alpha": _eopd_alpha,
                            "topk": _eopd_topk,
                            "entropy_mean":
                                float(
                                    _eopd_entropy_cpu.mean().item()
                                ),
                            "entropy_q25":
                                float(_eopd_q[0].item()),
                            "entropy_q50":
                                float(_eopd_q[1].item()),
                            "entropy_q75":
                                float(_eopd_q[2].item()),
                            "entropy_q90":
                                float(_eopd_q[3].item()),
                            "entropy_q95":
                                float(_eopd_q[4].item()),
                            "high_entropy_fraction":
                                float(
                                    _eopd_high_count
                                    / max(_eopd_token_count, 1)
                                ),
                            "top16_teacher_mass_mean":
                                float(
                                    _eopd_topk_mass.mean()
                                    .detach().item()
                                ),
                            "top16_teacher_mass_min":
                                float(
                                    _eopd_topk_mass.min()
                                    .detach().item()
                                ),
                            "high_entropy_fkl_mean":
                                _eopd_high_fkl_mean,
                            "gated_fkl_loss":
                                float(
                                    _eopd_fkl_loss
                                    .detach().item()
                                ),
                            "pg_loss":
                                float(
                                    _eopd_pg_loss
                                    .detach().item()
                                ),
                            "total_loss":
                                float(
                                    loss.detach().item()
                                ),
                        },
                        flush=True,
                    )


                _eopd_runtime_audit_done = True

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
                "token_mean_reverse_kl =",
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
            "MTPATCHER_V4_TORCHNPU_REVERSEKL_TRAINING_PASS",
            flush=True,
        )

    dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
