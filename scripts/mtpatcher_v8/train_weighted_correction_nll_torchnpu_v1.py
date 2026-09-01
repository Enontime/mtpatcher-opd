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


BOOST_MASK_KEYS = {
    "patch":
        "patch_mask",

    "random":
        "random_equal_count_mask",
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

    with Path(path).open(
        "rb"
    ) as f:

        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):

            h.update(
                chunk
            )

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

            row = json.loads(
                line
            )

            if not isinstance(
                row,
                dict,
            ):

                raise RuntimeError(
                    f"Non-dict row "
                    f"line={lineno}"
                )

            rows.append(
                row
            )

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


class WeightedCorrectionDataset(
    Dataset
):

    def __init__(
        self,
        tokenizer,
        rows,
        boost_mode,
        boost_factor,
        max_length,
    ):

        self.items = []

        mask_key = BOOST_MASK_KEYS[
            boost_mode
        ]

        total_boost_positions = 0
        total_response_positions = 0
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
                    f"Invalid messages "
                    f"row={row_no}"
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
                    f"Target leakage "
                    f"row={row_no}"
                )


            target = norm(
                row.get(
                    "target_translation",
                    "",
                )
            )


            if not target:

                raise RuntimeError(
                    f"Empty target "
                    f"row={row_no}"
                )


            meta = row.get(
                "_patch_aware_v2"
            )


            if not isinstance(
                meta,
                dict,
            ):

                raise RuntimeError(
                    f"Missing patch metadata "
                    f"row={row_no}"
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


            expected_target = int(
                meta[
                    "postedit_token_count"
                ]
            )


            if len(
                target_ids
            ) != expected_target:

                raise RuntimeError(
                    f"Target tokenization "
                    f"changed row={row_no}: "
                    f"{len(target_ids)} "
                    f"vs {expected_target}"
                )


            response_ids = (
                target_ids
                + [
                    tokenizer.eos_token_id
                ]
            )


            expected_response = int(
                meta[
                    "response_positions_with_eos"
                ]
            )


            if len(
                response_ids
            ) != expected_response:

                raise RuntimeError(
                    f"Response-token count "
                    f"changed row={row_no}"
                )


            boost_mask = meta.get(
                mask_key
            )


            if (
                not isinstance(
                    boost_mask,
                    list,
                )
                or not boost_mask
            ):

                raise RuntimeError(
                    f"Empty boost mask "
                    f"row={row_no}"
                )


            boost_mask = [
                int(x)
                for x in boost_mask
            ]


            if boost_mask != sorted(
                set(
                    boost_mask
                )
            ):

                raise RuntimeError(
                    f"Boost mask not "
                    f"sorted/unique "
                    f"row={row_no}"
                )


            if (
                min(
                    boost_mask
                ) < 0
                or
                max(
                    boost_mask
                ) >= len(
                    response_ids
                )
            ):

                raise RuntimeError(
                    f"Boost mask "
                    f"out of range "
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
                    f"Sequence too long "
                    f"row={row_no}: "
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

                    "boost_mask":
                        boost_mask,
                }
            )


            total_boost_positions += len(
                boost_mask
            )

            total_response_positions += len(
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
            "BOOST_MODE =",
            boost_mode,
        )

        print(
            "BOOST_FACTOR =",
            boost_factor,
        )

        print(
            "TOTAL_BOOST_POSITIONS =",
            total_boost_positions,
        )

        print(
            "TOTAL_RESPONSE_POSITIONS =",
            total_response_positions,
        )

        print(
            "GLOBAL_BOOST_POSITION_FRACTION =",
            total_boost_positions
            / total_response_positions,
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
        index,
    ):

        return self.items[
            index
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
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--boost-mode",
        choices=sorted(
            BOOST_MASK_KEYS
        ),
        required=True,
    )

    ap.add_argument(
        "--boost-factor",
        type=float,
        required=True,
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


    if args.boost_factor < 1.0:

        raise RuntimeError(
            "boost_factor must be >= 1"
        )


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


    dataset = WeightedCorrectionDataset(
        tokenizer,
        rows,
        args.boost_mode,
        args.boost_factor,
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
        len(
            loader
        )
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
            "MT-PATCHER FULL-CORRECTION NLL + TOKEN BOOST"
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
            "BOOST_MODE =",
            args.boost_mode,
        )

        print(
            "BOOST_FACTOR =",
            args.boost_factor,
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
            len(
                loader
            ),
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
            "TRAJECTORY = full Feedbacker post-edit"
        )

        print(
            "OBJECTIVE = normalized weighted target-token NLL"
        )

        print(
            "BASE_TOKEN_WEIGHT = 1.0"
        )

        print(
            "BOOSTED_TOKEN_WEIGHT =",
            args.boost_factor,
        )

        print(
            "WEIGHTED_CORRECTION_NLL_TRAINING_START",
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

        epoch_start = time.time()


        local_weighted_numerator = 0.0
        local_weight_denominator = 0.0

        local_raw_nll_sum = 0.0
        local_raw_token_count = 0.0

        local_boost_nll_sum = 0.0
        local_boost_token_count = 0.0

        local_nonboost_nll_sum = 0.0
        local_nonboost_token_count = 0.0


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


            boost_mask = [
                int(x)
                for x in item[
                    "boost_mask"
                ]
            ]


            response_count = len(
                response_ids
            )


            prediction_positions = torch.arange(
                prompt_len - 1,
                prompt_len - 1 + response_count,
                dtype=torch.long,
                device=device,
            )


            target_ids = torch.tensor(
                response_ids,
                dtype=torch.long,
                device=device,
            )


            boost_bool = torch.zeros(
                response_count,
                dtype=torch.bool,
                device=device,
            )


            boost_index = torch.tensor(
                boost_mask,
                dtype=torch.long,
                device=device,
            )


            boost_bool[
                boost_index
            ] = True


            weights = torch.ones(
                response_count,
                dtype=torch.float32,
                device=device,
            )


            weights[
                boost_bool
            ] = float(
                args.boost_factor
            )


            optimizer.zero_grad(
                set_to_none=True
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


            target_logp = (
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
                -target_logp
            )


            if not torch.isfinite(
                token_nll
            ).all():

                raise RuntimeError(
                    "Non-finite token NLL"
                )


            weighted_numerator = (
                weights
                * token_nll
            ).sum()


            weight_denominator = (
                weights.sum()
            )


            loss = (
                weighted_numerator
                / weight_denominator
            )


            if not torch.isfinite(
                loss
            ):

                raise RuntimeError(
                    "Non-finite weighted loss"
                )


            boost_nll = (
                token_nll[
                    boost_bool
                ].mean()
            )


            if (
                ~boost_bool
            ).any():

                nonboost_nll = (
                    token_nll[
                        ~boost_bool
                    ].mean()
                )

            else:

                nonboost_nll = torch.tensor(
                    0.0,
                    dtype=torch.float32,
                    device=device,
                )


            unweighted_nll = (
                token_nll.mean()
            )


            if (
                rank == 0
                and epoch == 1
                and local_step == 1
            ):

                with torch.no_grad():

                    target_top1 = (
                        selected_logits.argmax(
                            dim=-1
                        )
                        == target_ids
                    ).float()


                    target_top5 = (
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
                    ).float()


                    effective_boost_weight_fraction = (
                        weights[
                            boost_bool
                        ].sum()
                        / weights.sum()
                    )


                print(
                    "WEIGHTED_NLL_RUNTIME_AUDIT",
                    {
                        "boost_mode":
                            args.boost_mode,

                        "boost_factor":
                            args.boost_factor,

                        "row_index":
                            int(
                                item[
                                    "index"
                                ]
                            ),

                        "response_tokens":
                            response_count,

                        "boost_tokens":
                            len(
                                boost_mask
                            ),

                        "boost_position_fraction":
                            len(
                                boost_mask
                            )
                            / response_count,

                        "effective_boost_weight_fraction":
                            float(
                                effective_boost_weight_fraction
                                .item()
                            ),

                        "full_unweighted_nll":
                            float(
                                unweighted_nll
                                .detach()
                                .item()
                            ),

                        "boost_position_nll":
                            float(
                                boost_nll
                                .detach()
                                .item()
                            ),

                        "nonboost_position_nll":
                            float(
                                nonboost_nll
                                .detach()
                                .item()
                            ),

                        "weighted_loss":
                            float(
                                loss
                                .detach()
                                .item()
                            ),

                        "student_target_top1_all":
                            float(
                                target_top1.mean()
                                .item()
                            ),

                        "student_target_top1_boost":
                            float(
                                target_top1[
                                    boost_bool
                                ].mean()
                                .item()
                            ),

                        "student_target_top5_all":
                            float(
                                target_top5.mean()
                                .item()
                            ),

                        "student_target_top5_boost":
                            float(
                                target_top5[
                                    boost_bool
                                ].mean()
                                .item()
                            ),
                    },
                    flush=True,
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
                    "Non-finite gradient norm"
                )


            if args.preflight:

                if rank == 0:

                    print(
                        "WEIGHTED_NLL_PREFLIGHT",
                        {
                            "boost_mode":
                                args.boost_mode,

                            "boost_factor":
                                args.boost_factor,

                            "loss":
                                float(
                                    loss
                                    .detach()
                                    .item()
                                ),

                            "full_nll":
                                float(
                                    unweighted_nll
                                    .detach()
                                    .item()
                                ),

                            "boost_nll":
                                float(
                                    boost_nll
                                    .detach()
                                    .item()
                                ),

                            "grad_norm":
                                float(
                                    grad_norm
                                ),

                            "boost_tokens":
                                len(
                                    boost_mask
                                ),
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


            weighted_num_value = float(
                weighted_numerator
                .detach()
                .item()
            )

            weight_den_value = float(
                weight_denominator
                .detach()
                .item()
            )

            raw_nll_sum_value = float(
                token_nll.sum()
                .detach()
                .item()
            )

            boost_nll_sum_value = float(
                token_nll[
                    boost_bool
                ].sum()
                .detach()
                .item()
            )


            nonboost_count = (
                response_count
                - len(
                    boost_mask
                )
            )


            if nonboost_count > 0:

                nonboost_nll_sum_value = float(
                    token_nll[
                        ~boost_bool
                    ].sum()
                    .detach()
                    .item()
                )

            else:

                nonboost_nll_sum_value = 0.0


            local_weighted_numerator += (
                weighted_num_value
            )

            local_weight_denominator += (
                weight_den_value
            )

            local_raw_nll_sum += (
                raw_nll_sum_value
            )

            local_raw_token_count += (
                response_count
            )

            local_boost_nll_sum += (
                boost_nll_sum_value
            )

            local_boost_token_count += len(
                boost_mask
            )

            local_nonboost_nll_sum += (
                nonboost_nll_sum_value
            )

            local_nonboost_token_count += (
                nonboost_count
            )


            if (
                rank == 0
                and (
                    global_step == 1
                    or
                    global_step % 20 == 0
                    or
                    local_step == len(
                        loader
                    )
                )
            ):

                print(
                    f"epoch={epoch} "
                    f"local_step={local_step}/{len(loader)} "
                    f"update={global_step}/{total_updates} "
                    f"weighted_nll="
                    f"{float(loss.detach().item()):.6f} "
                    f"full_nll="
                    f"{float(unweighted_nll.detach().item()):.6f} "
                    f"boost_nll="
                    f"{float(boost_nll.detach().item()):.6f} "
                    f"boost_tokens={len(boost_mask)} "
                    f"grad_norm={float(grad_norm):.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.8g} "
                    f"npu_alloc_gib="
                    f"{torch.npu.memory_allocated()/2**30:.3f} "
                    f"elapsed="
                    f"{time.time()-epoch_start:.1f}s",
                    flush=True,
                )


            del (
                input_ids,
                attention_mask,
                prediction_positions,
                target_ids,
                boost_bool,
                boost_index,
                weights,
                out,
                selected_logits,
                logp,
                target_logp,
                token_nll,
                weighted_numerator,
                weight_denominator,
                loss,
                boost_nll,
                nonboost_nll,
                unweighted_nll,
            )


        if args.preflight:

            break


        #######################################################################
        # HCCL-safe float32 metric reduction.
        #######################################################################

        stats = torch.tensor(
            [
                local_weighted_numerator,
                local_weight_denominator,

                local_raw_nll_sum,
                local_raw_token_count,

                local_boost_nll_sum,
                local_boost_token_count,

                local_nonboost_nll_sum,
                local_nonboost_token_count,
            ],
            dtype=torch.float32,
            device=device,
        )


        dist.all_reduce(
            stats,
            op=dist.ReduceOp.SUM,
        )


        weighted_mean = (
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


        raw_mean = (
            float(
                stats[
                    2
                ].item()
            )
            / max(
                float(
                    stats[
                        3
                    ].item()
                ),
                1.0,
            )
        )


        boost_mean = (
            float(
                stats[
                    4
                ].item()
            )
            / max(
                float(
                    stats[
                        5
                    ].item()
                ),
                1.0,
            )
        )


        nonboost_mean = (
            float(
                stats[
                    6
                ].item()
            )
            / max(
                float(
                    stats[
                        7
                    ].item()
                ),
                1.0,
            )
        )


        if rank == 0:

            rec = {
                "epoch":
                    epoch,

                "boost_mode":
                    args.boost_mode,

                "boost_factor":
                    args.boost_factor,

                "weighted_mean_nll":
                    weighted_mean,

                "full_unweighted_mean_nll":
                    raw_mean,

                "boost_position_mean_nll":
                    boost_mean,

                "nonboost_position_mean_nll":
                    nonboost_mean,

                "global_response_tokens":
                    int(
                        stats[
                            3
                        ].item()
                    ),

                "global_boost_tokens":
                    int(
                        stats[
                            5
                        ].item()
                    ),

                "global_updates_total":
                    global_step,

                "epoch_seconds":
                    time.time()
                    - epoch_start,
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
                "weighted_mean_nll =",
                weighted_mean,
            )

            print(
                "full_unweighted_mean_nll =",
                raw_mean,
            )

            print(
                "boost_position_mean_nll =",
                boost_mean,
            )

            print(
                "nonboost_position_mean_nll =",
                nonboost_mean,
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
                "WEIGHTED_NLL_PREFLIGHT_PASS",
                args.boost_mode,
                args.boost_factor,
                flush=True,
            )

        dist.destroy_process_group()

        return


    if rank == 0:

        manifest = {
            "method":
                "full_correction_weighted_nll",

            "boost_mode":
                args.boost_mode,

            "boost_factor":
                args.boost_factor,

            "base_token_weight":
                1.0,

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
                "full Feedbacker post-edit teacher-forced",

            "objective":
                "normalized weighted target-token negative log likelihood",

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
            "WEIGHTED_CORRECTION_NLL_TRAINING_PASS",
            args.boost_mode,
            args.boost_factor,
            flush=True,
        )


    dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":

    main()
