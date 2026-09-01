#!/usr/bin/env python3

import argparse
import gc
import hashlib
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
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


MASK_KEYS = {
    "patch": "patch_mask",
    "random": "random_equal_count_mask",
    "halo1": "patch_halo1_mask",
    "full": "full_correction_mask",
}


def norm(x):

    return " ".join(
        str(x)
        .replace("\r", " ")
        .replace("\n", " ")
        .split()
    )


def sha256(path):

    h = hashlib.sha256()

    with Path(path).open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):

            h.update(chunk)

    return h.hexdigest()


def read_jsonl(path):

    rows = []

    with Path(path).open(
        "r",
        encoding="utf-8-sig",
    ) as f:

        for lineno, line in enumerate(
            f,
            start=1,
        ):

            if not line.strip():
                continue

            row = json.loads(line)

            if not isinstance(row, dict):

                raise RuntimeError(
                    f"non-dict row line={lineno}"
                )

            rows.append(row)

    return rows


def render_prompt(
    tokenizer,
    messages,
):

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


class CorrectionDataset(Dataset):

    def __init__(
        self,
        tokenizer,
        rows,
        mask_mode,
        max_length,
    ):

        self.items = []

        mask_key = MASK_KEYS[
            mask_mode
        ]

        mask_counts = []
        response_counts = []
        sequence_lengths = []

        for row_no, row in enumerate(
            rows
        ):

            messages = row.get(
                "messages"
            )

            if (
                not isinstance(
                    messages,
                    list,
                )
                or not messages
            ):

                raise RuntimeError(
                    f"invalid messages row={row_no}"
                )


            target = norm(
                row.get(
                    "target_translation",
                    "",
                )
            )


            if not target:

                raise RuntimeError(
                    f"empty target row={row_no}"
                )


            meta = row.get(
                "_patch_aware_v2"
            )


            if not isinstance(
                meta,
                dict,
            ):

                raise RuntimeError(
                    f"missing _patch_aware_v2 row={row_no}"
                )


            prompt_text = render_prompt(
                tokenizer,
                messages,
            )


            prompt_ids = tokenizer(
                prompt_text,
                add_special_tokens=False,
            )[
                "input_ids"
            ]


            target_ids = tokenizer(
                target,
                add_special_tokens=False,
            )[
                "input_ids"
            ]


            if not prompt_ids:

                raise RuntimeError(
                    f"empty prompt ids row={row_no}"
                )


            expected_target_n = int(
                meta[
                    "postedit_token_count"
                ]
            )


            if len(
                target_ids
            ) != expected_target_n:

                raise RuntimeError(
                    f"target tokenization mismatch "
                    f"row={row_no} "
                    f"now={len(target_ids)} "
                    f"prepared={expected_target_n}"
                )


            response_ids = (
                target_ids
                + [
                    tokenizer.eos_token_id
                ]
            )


            expected_response_n = int(
                meta[
                    "response_positions_with_eos"
                ]
            )


            if len(
                response_ids
            ) != expected_response_n:

                raise RuntimeError(
                    f"response count mismatch "
                    f"row={row_no}"
                )


            mask = meta.get(
                mask_key
            )


            if (
                not isinstance(
                    mask,
                    list,
                )
                or not mask
            ):

                raise RuntimeError(
                    f"empty mask "
                    f"mode={mask_mode} "
                    f"row={row_no}"
                )


            mask = [
                int(x)
                for x in mask
            ]


            if mask != sorted(
                set(mask)
            ):

                raise RuntimeError(
                    f"mask not sorted/unique "
                    f"row={row_no}"
                )


            if (
                min(mask) < 0
                or max(mask) >= len(
                    response_ids
                )
            ):

                raise RuntimeError(
                    f"mask range invalid "
                    f"row={row_no}"
                )


            input_ids = (
                prompt_ids
                + response_ids
            )


            if len(
                input_ids
            ) > max_length:

                raise RuntimeError(
                    f"sequence exceeds max_length "
                    f"row={row_no} "
                    f"len={len(input_ids)} "
                    f"max={max_length}"
                )


            self.items.append(
                {
                    "index":
                        int(
                            row[
                                "index"
                            ]
                        ),

                    "input_ids":
                        input_ids,

                    "prompt_len":
                        len(
                            prompt_ids
                        ),

                    "response_ids":
                        response_ids,

                    "mask":
                        mask,
                }
            )


            mask_counts.append(
                len(mask)
            )

            response_counts.append(
                len(response_ids)
            )

            sequence_lengths.append(
                len(input_ids)
            )


        print(
            "DATASET_MASK_MODE =",
            mask_mode,
        )

        print(
            "DATASET_ROWS =",
            len(
                self.items
            ),
        )

        print(
            "MEAN_MASK_TOKENS =",
            sum(mask_counts)
            / len(mask_counts),
        )

        print(
            "GLOBAL_MASK_FRACTION =",
            sum(mask_counts)
            / sum(response_counts),
        )

        print(
            "MAX_SEQUENCE_LENGTH =",
            max(sequence_lengths),
        )


    def __len__(
        self
    ):

        return len(
            self.items
        )


    def __getitem__(
        self,
        idx,
    ):

        return self.items[
            idx
        ]


def collate_one(
    batch
):

    if len(
        batch
    ) != 1:

        raise RuntimeError(
            "Correction-FKL expects local batch size 1"
        )

    return batch[
        0
    ]


def save_checkpoint(
    student,
    tokenizer,
    output_dir,
    epoch,
    rank,
):

    dist.barrier()

    if rank == 0:

        ckpt = (
            Path(
                output_dir
            )
            / f"epoch{epoch}"
        )

        ckpt.mkdir(
            parents=True,
            exist_ok=True,
        )

        student.module.save_pretrained(
            ckpt,
            safe_serialization=True,
        )

        tokenizer.save_pretrained(
            ckpt
        )

        print(
            "CHECKPOINT_SAVED =",
            ckpt,
            flush=True,
        )

    dist.barrier()


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--student-model",
        required=True,
    )

    ap.add_argument(
        "--teacher-model",
        required=True,
    )

    ap.add_argument(
        "--train",
        required=True,
        type=Path,
    )

    ap.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    ap.add_argument(
        "--mask-mode",
        required=True,
        choices=sorted(
            MASK_KEYS
        ),
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=1.0e-6,
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    ap.add_argument(
        "--max-length",
        type=int,
        default=1024,
    )

    ap.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
    )

    ap.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    ap.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=20260825,
    )

    ap.add_argument(
        "--preflight",
        action="store_true",
    )

    args = ap.parse_args()


    local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            "0",
        )
    )

    rank = int(
        os.environ.get(
            "RANK",
            "0",
        )
    )

    world_size = int(
        os.environ.get(
            "WORLD_SIZE",
            "1",
        )
    )


    torch.npu.set_device(
        local_rank
    )

    device = torch.device(
        f"npu:{local_rank}"
    )


    dist.init_process_group(
        backend="hccl"
    )


    random.seed(
        args.seed
        + rank
    )

    torch.manual_seed(
        args.seed
        + rank
    )

    torch.npu.manual_seed_all(
        args.seed
        + rank
    )


    rows = read_jsonl(
        args.train
    )


    if len(
        rows
    ) != 3732:

        raise RuntimeError(
            f"expected 3732 rows, got {len(rows)}"
        )


    tokenizer = AutoTokenizer.from_pretrained(
        args.student_model,
        local_files_only=True,
        trust_remote_code=True,
    )


    if tokenizer.eos_token_id is None:

        raise RuntimeError(
            "tokenizer has no EOS"
        )


    dataset = CorrectionDataset(
        tokenizer,
        rows,
        args.mask_mode,
        args.max_length,
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
        collate_fn=collate_one,
        drop_last=False,
    )


    if rank == 0:

        print(
            "=" * 100
        )

        print(
            "CORRECTION-TRAJECTORY EXACT FORWARD-KL"
        )

        print(
            "=" * 100
        )

        print(
            "TRAIN_SHA256 =",
            sha256(
                args.train
            ),
        )

        print(
            "MASK_MODE =",
            args.mask_mode,
        )

        print(
            "STUDENT =",
            args.student_model,
        )

        print(
            "TEACHER =",
            args.teacher_model,
        )

        print(
            "WORLD_SIZE =",
            world_size,
        )

        print(
            "LOCAL_STEPS_PER_EPOCH =",
            len(loader),
        )

        print(
            "GLOBAL_BATCH =",
            world_size,
        )

        print(
            "OBJECTIVE = teacher-forced correction trajectory exact FKL"
        )

        print(
            "LOSS_NORMALIZATION = mean over selected positions per example"
        )

        print(
            "PREFLIGHT =",
            args.preflight,
        )


    student_raw = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )


    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )


    if int(
        student_raw.config.vocab_size
    ) != int(
        teacher.config.vocab_size
    ):

        raise RuntimeError(
            "student/teacher vocab sizes differ"
        )



    # Qwen3 uses a padded model vocabulary.  The LM-head vocabulary
    # can legitimately exceed len(tokenizer).  Scientific compatibility
    # requires Student and Teacher output vocabularies to match; that
    # model-model check remains above.
    if rank == 0:
        print(
            "VOCAB_PADDING_AUDIT",
            {
                "student_model_vocab":
                    int(student_raw.config.vocab_size),
                "teacher_model_vocab":
                    int(teacher.config.vocab_size),
                "tokenizer_len":
                    len(tokenizer),
                "tokenizer_vocab_size":
                    int(tokenizer.vocab_size),
                "padding_rows":
                    int(student_raw.config.vocab_size)
                    - len(tokenizer),
            },
            flush=True,
        )


    student_raw.config.use_cache = False
    teacher.config.use_cache = False


    student_raw.to(
        device
    )

    teacher.to(
        device
    )


    teacher.eval()

    for p in teacher.parameters():

        p.requires_grad_(
            False
        )


    student_raw.train()


    student = DDP(
        student_raw,
        device_ids=[
            local_rank
        ],
        broadcast_buffers=False,
        find_unused_parameters=False,
    )


    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


    total_updates = (
        len(loader)
        * args.epochs
    )


    warmup_steps = int(
        round(
            total_updates
            * args.warmup_ratio
        )
    )


    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )


    if rank == 0:

        print(
            "TOTAL_UPDATES =",
            total_updates,
        )

        print(
            "WARMUP_STEPS =",
            warmup_steps,
        )

        print(
            "CORRECTION_FKL_TRAINING_START",
            flush=True,
        )


    global_step = 0

    metrics = []


    optimizer.zero_grad(
        set_to_none=True
    )


    for epoch in range(
        1,
        args.epochs + 1,
    ):

        sampler.set_epoch(
            epoch
        )

        epoch_start = time.time()

        epoch_loss_sum = 0.0
        epoch_mask_tokens = 0


        for local_step, item in enumerate(
            loader,
            start=1,
        ):

            input_ids = torch.tensor(
                [
                    item[
                        "input_ids"
                    ]
                ],
                dtype=torch.long,
                device=device,
            )


            attention_mask = torch.ones_like(
                input_ids,
                dtype=torch.long,
                device=device,
            )


            prompt_len = int(
                item[
                    "prompt_len"
                ]
            )


            mask = [
                int(x)
                for x in item[
                    "mask"
                ]
            ]


            response_ids = [
                int(x)
                for x in item[
                    "response_ids"
                ]
            ]


            prediction_positions = torch.tensor(
                [
                    prompt_len
                    - 1
                    + x
                    for x in mask
                ],
                dtype=torch.long,
                device=device,
            )


            selected_target_ids = torch.tensor(
                [
                    response_ids[
                        x
                    ]
                    for x in mask
                ],
                dtype=torch.long,
                device=device,
            )


            with torch.no_grad():

                teacher_out = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )


                t_selected = (
                    teacher_out.logits[
                        0
                    ]
                    .index_select(
                        0,
                        prediction_positions,
                    )
                    .float()
                )


                t_logp = F.log_softmax(
                    t_selected,
                    dim=-1,
                )


                t_prob = t_logp.exp()


            del (
                teacher_out,
                t_selected,
            )


            student_out = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )


            s_selected = (
                student_out.logits[
                    0
                ]
                .index_select(
                    0,
                    prediction_positions,
                )
                .float()
            )


            s_logp = F.log_softmax(
                s_selected,
                dim=-1,
            )


            token_kl = (
                t_prob
                * (
                    t_logp
                    - s_logp
                )
            ).sum(
                dim=-1
            )


            if not torch.isfinite(
                token_kl
            ).all():

                raise RuntimeError(
                    "non-finite Correction-FKL"
                )


            if float(
                token_kl.min()
                .detach()
                .item()
            ) < -1.0e-4:

                raise RuntimeError(
                    "unexpected materially negative exact FKL"
                )


            loss = token_kl.mean()


            if not torch.isfinite(
                loss
            ):

                raise RuntimeError(
                    "non-finite loss"
                )


            if (
                rank == 0
                and epoch == 1
                and local_step == 1
            ):

                with torch.no_grad():

                    teacher_target_logp = (
                        t_logp.gather(
                            1,
                            selected_target_ids[
                                :,
                                None
                            ],
                        )
                        .squeeze(
                            1
                        )
                    )


                    student_target_logp = (
                        s_logp.gather(
                            1,
                            selected_target_ids[
                                :,
                                None
                            ],
                        )
                        .squeeze(
                            1
                        )
                    )


                    teacher_top1 = (
                        t_logp.argmax(
                            dim=-1
                        )
                        == selected_target_ids
                    ).float().mean()


                    teacher_top5 = (
                        t_logp.topk(
                            k=5,
                            dim=-1,
                        ).indices
                        == selected_target_ids[
                            :,
                            None
                        ]
                    ).any(
                        dim=-1
                    ).float().mean()


                print(
                    "CORRECTION_FKL_RUNTIME_AUDIT",
                    {
                        "mask_mode":
                            args.mask_mode,

                        "row_index":
                            int(
                                item[
                                    "index"
                                ]
                            ),

                        "mask_tokens":
                            len(mask),

                        "response_tokens":
                            len(
                                response_ids
                            ),

                        "mask_fraction":
                            len(mask)
                            / len(
                                response_ids
                            ),

                        "exact_fkl_mean":
                            float(
                                loss.detach().item()
                            ),

                        "teacher_target_logp_mean":
                            float(
                                teacher_target_logp.mean()
                                .item()
                            ),

                        "student_target_logp_mean":
                            float(
                                student_target_logp.mean()
                                .item()
                            ),

                        "teacher_target_top1_fraction":
                            float(
                                teacher_top1.item()
                            ),

                        "teacher_target_top5_fraction":
                            float(
                                teacher_top5.item()
                            ),
                    },
                    flush=True,
                )


            loss.backward()


            grad_norm = clip_grad_norm_(
                student.parameters(),
                args.max_grad_norm,
            )


            if not torch.isfinite(
                torch.as_tensor(
                    grad_norm
                )
            ):

                raise RuntimeError(
                    "non-finite gradient norm"
                )


            if args.preflight:

                if rank == 0:

                    print(
                        "CORRECTION_FKL_PREFLIGHT",
                        {
                            "mask_mode":
                                args.mask_mode,

                            "loss":
                                float(
                                    loss.detach().item()
                                ),

                            "grad_norm":
                                float(
                                    grad_norm
                                ),

                            "mask_tokens":
                                len(mask),
                        },
                        flush=True,
                    )

                optimizer.zero_grad(
                    set_to_none=True
                )

                break


            optimizer.step()

            scheduler.step()

            optimizer.zero_grad(
                set_to_none=True
            )


            global_step += 1


            loss_value = float(
                loss.detach().item()
            )


            epoch_loss_sum += (
                loss_value
                * len(mask)
            )

            epoch_mask_tokens += len(
                mask
            )


            if (
                rank == 0
                and (
                    global_step == 1
                    or global_step % 20 == 0
                    or local_step == len(loader)
                )
            ):

                print(
                    f"epoch={epoch} "
                    f"local_step={local_step}/{len(loader)} "
                    f"update={global_step}/{total_updates} "
                    f"fkl={loss_value:.6f} "
                    f"mask_tokens={len(mask)} "
                    f"grad_norm={float(grad_norm):.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.8g} "
                    f"npu_alloc_gib="
                    f"{torch.npu.memory_allocated()/2**30:.3f} "
                    f"elapsed={time.time()-epoch_start:.1f}s",
                    flush=True,
                )


            del (
                input_ids,
                attention_mask,
                prediction_positions,
                selected_target_ids,
                t_logp,
                t_prob,
                student_out,
                s_selected,
                s_logp,
                token_kl,
                loss,
            )


        if args.preflight:

            break


        stats = torch.tensor(
            [
                epoch_loss_sum,
                float(
                    epoch_mask_tokens
                ),
            ],
            dtype=torch.float32,
            device=device,
        )


        dist.all_reduce(
            stats,
            op=dist.ReduceOp.SUM,
        )


        token_mean_fkl = (
            stats[
                0
            ].item()
            / max(
                stats[
                    1
                ].item(),
                1.0,
            )
        )


        if rank == 0:

            epoch_metric = {
                "epoch":
                    epoch,

                "mask_mode":
                    args.mask_mode,

                "token_mean_correction_fkl":
                    token_mean_fkl,

                "global_mask_tokens":
                    int(
                        stats[
                            1
                        ].item()
                    ),

                "global_updates_total":
                    global_step,

                "epoch_seconds":
                    time.time()
                    - epoch_start,
            }


            metrics.append(
                epoch_metric
            )


            args.output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )


            (
                args.output_dir
                / f"epoch{epoch}_metrics.json"
            ).write_text(
                json.dumps(
                    epoch_metric,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )


            print(
                f"EPOCH_{epoch}_COMPLETE"
            )

            print(
                "token_mean_correction_fkl =",
                token_mean_fkl,
            )


        save_checkpoint(
            student,
            tokenizer,
            args.output_dir,
            epoch,
            rank,
        )


        gc.collect()

        torch.npu.empty_cache()


    if args.preflight:

        dist.barrier()

        if rank == 0:

            print(
                "CORRECTION_FKL_PREFLIGHT_PASS",
                args.mask_mode,
                flush=True,
            )

        dist.destroy_process_group()

        return


    if rank == 0:

        manifest = {
            "method":
                "correction_trajectory_exact_forward_kl",

            "mask_mode":
                args.mask_mode,

            "student":
                args.student_model,

            "teacher":
                args.teacher_model,

            "train":
                str(
                    args.train
                ),

            "train_sha256":
                sha256(
                    args.train
                ),

            "rows":
                len(
                    rows
                ),

            "lr":
                args.lr,

            "epochs":
                args.epochs,

            "global_batch":
                world_size,

            "warmup_ratio":
                args.warmup_ratio,

            "weight_decay":
                args.weight_decay,

            "max_grad_norm":
                args.max_grad_norm,

            "seed":
                args.seed,

            "trajectory":
                "Feedbacker post-edit teacher-forced prefix",

            "divergence":
                "exact full-vocabulary KL(Teacher||Student)",

            "normalization":
                "mean selected-token FKL per example",

            "epoch_metrics":
                metrics,

            "training_complete":
                True,
        }


        (
            args.output_dir
            / "training_manifest.json"
        ).write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


        print(
            "CORRECTION_FKL_TRAINING_PASS",
            args.mask_mode,
            flush=True,
        )


    dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":

    main()
