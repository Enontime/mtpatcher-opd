#!/usr/bin/env python3

import argparse
import gc
import hashlib
import json
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

            obj = json.loads(line)

            if not isinstance(obj, dict):

                raise RuntimeError(
                    f"Non-dict row line={lineno}"
                )

            rows.append(obj)

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


class CorrectionNLLDataset(Dataset):

    def __init__(
        self,
        tokenizer,
        rows,
        mask_mode,
        max_length,
    ):

        self.items = []

        key = MASK_KEYS[
            mask_mode
        ]

        total_mask = 0
        total_response = 0
        max_seq = 0

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
                    f"Invalid messages row={row_no}"
                )


            if (
                str(
                    messages[-1].get(
                        "role",
                        "",
                    )
                )
                == "assistant"
            ):

                raise RuntimeError(
                    f"Prompt target leakage row={row_no}"
                )


            target = norm(
                row.get(
                    "target_translation",
                    "",
                )
            )


            if not target:

                raise RuntimeError(
                    f"Empty target row={row_no}"
                )


            meta = row.get(
                "_patch_aware_v2"
            )


            if not isinstance(
                meta,
                dict,
            ):

                raise RuntimeError(
                    f"Missing patch metadata row={row_no}"
                )


            prompt = render_prompt(
                tokenizer,
                messages,
            )


            prompt_ids = tokenizer(
                prompt,
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


            expected_target_n = int(
                meta[
                    "postedit_token_count"
                ]
            )


            if len(
                target_ids
            ) != expected_target_n:

                raise RuntimeError(
                    f"Target tokenization mismatch "
                    f"row={row_no}: "
                    f"{len(target_ids)} != "
                    f"{expected_target_n}"
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
                    f"Response length mismatch row={row_no}"
                )


            mask = meta.get(
                key
            )


            if (
                not isinstance(
                    mask,
                    list,
                )
                or not mask
            ):

                raise RuntimeError(
                    f"Empty mask row={row_no}"
                )


            mask = [
                int(x)
                for x in mask
            ]


            if mask != sorted(
                set(mask)
            ):

                raise RuntimeError(
                    f"Mask not sorted/unique row={row_no}"
                )


            if (
                min(mask) < 0
                or max(mask) >= len(
                    response_ids
                )
            ):

                raise RuntimeError(
                    f"Mask range invalid row={row_no}"
                )


            input_ids = (
                prompt_ids
                + response_ids
            )


            if len(
                input_ids
            ) > max_length:

                raise RuntimeError(
                    f"Sequence too long row={row_no}: "
                    f"{len(input_ids)}"
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


            total_mask += len(
                mask
            )

            total_response += len(
                response_ids
            )

            max_seq = max(
                max_seq,
                len(
                    input_ids
                ),
            )


        print(
            "DATASET_ROWS =",
            len(
                self.items
            ),
        )

        print(
            "MASK_MODE =",
            mask_mode,
        )

        print(
            "TOTAL_MASK_POSITIONS =",
            total_mask,
        )

        print(
            "TOTAL_RESPONSE_POSITIONS =",
            total_response,
        )

        print(
            "GLOBAL_MASK_FRACTION =",
            total_mask
            / total_response,
        )

        print(
            "MAX_SEQUENCE_LENGTH =",
            max_seq,
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
            "Expected local batch size 1"
        )

    return batch[
        0
    ]


def save_checkpoint(
    model,
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

        model.module.save_pretrained(
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
        default=2e-5,
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
        os.environ[
            "LOCAL_RANK"
        ]
    )

    rank = int(
        os.environ[
            "RANK"
        ]
    )

    world_size = int(
        os.environ[
            "WORLD_SIZE"
        ]
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
            f"Expected 3732 rows, "
            f"got {len(rows)}"
        )


    tokenizer = AutoTokenizer.from_pretrained(
        args.student_model,
        local_files_only=True,
        trust_remote_code=True,
    )


    if tokenizer.eos_token_id is None:

        raise RuntimeError(
            "Tokenizer has no EOS"
        )


    dataset = CorrectionNLLDataset(
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


    model_raw = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )


    model_raw.config.use_cache = False

    model_raw.to(
        device
    )

    model_raw.train()


    model = DDP(
        model_raw,
        device_ids=[
            local_rank
        ],
        broadcast_buffers=False,
        find_unused_parameters=False,
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),
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
            "=" * 100
        )

        print(
            "MT-PATCHER CORRECTION-TOKEN NLL"
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
            "WORLD_SIZE =",
            world_size,
        )

        print(
            "GLOBAL_BATCH =",
            world_size,
        )

        print(
            "LOCAL_STEPS_PER_EPOCH =",
            len(loader),
        )

        print(
            "TOTAL_UPDATES =",
            total_updates,
        )

        print(
            "LR =",
            args.lr,
        )

        print(
            "OBJECTIVE = selected correction-token NLL"
        )

        print(
            "TRAJECTORY = Feedbacker post-edit teacher-forced prefix"
        )

        print(
            "CORRECTION_NLL_TRAINING_START",
            flush=True,
        )


    global_step = 0

    epoch_records = []


    for epoch in range(
        1,
        args.epochs + 1,
    ):

        sampler.set_epoch(
            epoch
        )

        start_time = time.time()

        local_weighted_loss = 0.0
        local_token_count = 0


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


            response_ids = [
                int(x)
                for x in item[
                    "response_ids"
                ]
            ]


            mask = [
                int(x)
                for x in item[
                    "mask"
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


            target_ids = torch.tensor(
                [
                    response_ids[
                        x
                    ]
                    for x in mask
                ],
                dtype=torch.long,
                device=device,
            )


            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )


            selected_logits = (
                out.logits[
                    0
                ]
                .index_select(
                    0,
                    prediction_positions,
                )
                .float()
            )


            logp = F.log_softmax(
                selected_logits,
                dim=-1,
            )


            selected_target_logp = (
                logp.gather(
                    1,
                    target_ids[
                        :,
                        None
                    ],
                )
                .squeeze(
                    1
                )
            )


            token_nll = (
                -selected_target_logp
            )


            if not torch.isfinite(
                token_nll
            ).all():

                raise RuntimeError(
                    "Non-finite token NLL"
                )


            loss = token_nll.mean()


            if not torch.isfinite(
                loss
            ):

                raise RuntimeError(
                    "Non-finite loss"
                )


            if (
                rank == 0
                and epoch == 1
                and local_step == 1
            ):

                with torch.no_grad():

                    top1 = (
                        selected_logits.argmax(
                            dim=-1
                        )
                        == target_ids
                    ).float().mean()


                    top5 = (
                        selected_logits.topk(
                            k=5,
                            dim=-1,
                        ).indices
                        == target_ids[
                            :,
                            None
                        ]
                    ).any(
                        dim=-1
                    ).float().mean()


                print(
                    "CORRECTION_NLL_RUNTIME_AUDIT",
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

                        "mean_target_nll":
                            float(
                                loss.detach()
                                .item()
                            ),

                        "mean_target_logp":
                            float(
                                selected_target_logp.mean()
                                .detach()
                                .item()
                            ),

                        "student_target_top1_fraction":
                            float(
                                top1.item()
                            ),

                        "student_target_top5_fraction":
                            float(
                                top5.item()
                            ),
                    },
                    flush=True,
                )


            optimizer.zero_grad(
                set_to_none=True
            )


            loss.backward()


            grad_norm = clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm,
            )


            if not torch.isfinite(
                torch.as_tensor(
                    grad_norm
                )
            ):

                raise RuntimeError(
                    "Non-finite grad norm"
                )


            if args.preflight:

                if rank == 0:

                    print(
                        "CORRECTION_NLL_PREFLIGHT",
                        {
                            "mode":
                                args.mask_mode,

                            "loss":
                                float(
                                    loss.detach()
                                    .item()
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


            global_step += 1


            loss_value = float(
                loss.detach()
                .item()
            )


            local_weighted_loss += (
                loss_value
                * len(mask)
            )

            local_token_count += len(
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
                    f"nll={loss_value:.6f} "
                    f"mask_tokens={len(mask)} "
                    f"grad_norm={float(grad_norm):.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.8g} "
                    f"npu_alloc_gib="
                    f"{torch.npu.memory_allocated()/2**30:.3f} "
                    f"elapsed={time.time()-start_time:.1f}s",
                    flush=True,
                )


            del (
                input_ids,
                attention_mask,
                prediction_positions,
                target_ids,
                out,
                selected_logits,
                logp,
                selected_target_logp,
                token_nll,
                loss,
            )


        if args.preflight:

            break


        # HCCL-safe float32 statistics.
        stats = torch.tensor(
            [
                local_weighted_loss,
                float(
                    local_token_count
                ),
            ],
            dtype=torch.float32,
            device=device,
        )


        dist.all_reduce(
            stats,
            op=dist.ReduceOp.SUM,
        )


        mean_nll = (
            float(
                stats[
                    0
                ].item()
            )
            / max(
                float(
                    stats[
                        1
                    ].item()
                ),
                1.0,
            )
        )


        if rank == 0:

            rec = {
                "epoch":
                    epoch,

                "mask_mode":
                    args.mask_mode,

                "token_mean_nll":
                    mean_nll,

                "global_selected_tokens":
                    int(
                        stats[
                            1
                        ].item()
                    ),

                "global_updates_total":
                    global_step,

                "epoch_seconds":
                    time.time()
                    - start_time,
            }


            epoch_records.append(
                rec
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
                    rec,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )


            print(
                f"EPOCH_{epoch}_COMPLETE"
            )

            print(
                "token_mean_correction_nll =",
                mean_nll,
            )


        save_checkpoint(
            model,
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
                "CORRECTION_NLL_PREFLIGHT_PASS",
                args.mask_mode,
                flush=True,
            )

        dist.destroy_process_group()

        return


    if rank == 0:

        manifest = {
            "method":
                "correction_token_nll",

            "mask_mode":
                args.mask_mode,

            "student":
                args.student_model,

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

            "epochs":
                args.epochs,

            "lr":
                args.lr,

            "global_batch":
                world_size,

            "warmup_ratio":
                args.warmup_ratio,

            "weight_decay":
                args.weight_decay,

            "trajectory":
                "Feedbacker post-edit teacher-forced",

            "objective":
                "selected target-token negative log likelihood",

            "epoch_metrics":
                epoch_records,

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
            "CORRECTION_NLL_TRAINING_PASS",
            args.mask_mode,
            flush=True,
        )


    dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":

    main()
