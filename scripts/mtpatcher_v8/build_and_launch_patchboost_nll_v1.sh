#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

SCRIPT_DIR="$ROOT/scripts/mtpatcher_v8"

TRAINER="$SCRIPT_DIR/train_weighted_correction_nll_torchnpu_v1.py"
QUEUE="$SCRIPT_DIR/run_patchboost_nll_queue_v1.sh"

EVALUATOR="$ROOT/scripts/mtpatcher_v6/eval_correction_fkl_torchnpu_v1.py"

DATA="$DATA_ROOT/$EXP/patch_aware_k1_clean3732_v2.jsonl"

STUDENT="$MODEL_ROOT/Qwen3-0.6B"

FULL_BASELINE="$RUN_ROOT/$EXP/corrnll_full_pe3732_v1"

QUEUE_LOG="$LOG_ROOT/$EXP/patchboost_nll_queue_v1.log"

mkdir -p \
"$SCRIPT_DIR" \
"$LOG_ROOT/$EXP"


###############################################################################
# TRAINER
###############################################################################

cat > "$TRAINER" <<'PY'
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
PY


###############################################################################
# QUEUE
###############################################################################

cat > "$QUEUE" <<'BASHQ'
#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

TRAINER="$ROOT/scripts/mtpatcher_v8/train_weighted_correction_nll_torchnpu_v1.py"

EVALUATOR="$ROOT/scripts/mtpatcher_v6/eval_correction_fkl_torchnpu_v1.py"

DATA="$DATA_ROOT/$EXP/patch_aware_k1_clean3732_v2.jsonl"

STUDENT="$MODEL_ROOT/Qwen3-0.6B"

WMT="$DATA_ROOT/pilot_v2_qwen3_06b/wmt24_zh_en998.jsonl"
FLORES="$DATA_ROOT/pilot_v2_qwen3_06b/flores_zh_en1012.jsonl"
CHALLENGE="$DATA_ROOT/pilot_v2_qwen3_06b/challenge_zh_en197.jsonl"


run_one () {

    MODE="$1"
    FACTOR="$2"
    NAME="$3"
    PORT="$4"

    OUT="$RUN_ROOT/$EXP/$NAME"

    TRAIN_LOG="$LOG_ROOT/$EXP/${NAME}.log"


    echo
    echo "======================================================================"
    echo "START $NAME"
    date
    echo "======================================================================"


    if [[ ! -f "$OUT/training_manifest.json" ]]; then

        python -m torch.distributed.run \
          --nproc_per_node=16 \
          --master_port="$PORT" \
          "$TRAINER" \
          --student-model "$STUDENT" \
          --train "$DATA" \
          --output-dir "$OUT" \
          --boost-mode "$MODE" \
          --boost-factor "$FACTOR" \
          --lr 2e-5 \
          --epochs 3 \
          --max-length 1024 \
          --warmup-ratio 0.03 \
          --weight-decay 0.01 \
          --max-grad-norm 1.0 \
          --seed 20260825 \
          > "$TRAIN_LOG" 2>&1

    fi


    test -f \
      "$OUT/training_manifest.json"

    test -f \
      "$OUT/epoch3/config.json"


    EVAL="$OUT/eval_epoch3"

    mkdir -p \
      "$EVAL"


    for SPEC in \
      "wmt24:$WMT" \
      "flores:$FLORES" \
      "challenge:$CHALLENGE"
    do

        SPLIT="${SPEC%%:*}"
        INPUT="${SPEC#*:}"

        mkdir -p \
          "$EVAL/$SPLIT"


        python "$EVALUATOR" \
          --model "$OUT/epoch3" \
          --input "$INPUT" \
          --output "$EVAL/$SPLIT/predictions.jsonl" \
          --metrics "$EVAL/$SPLIT/metrics.json" \
          --method "${NAME}_${SPLIT}" \
          --batch-size 16 \
          --max-new-tokens 256

    done


    echo
    echo "======================================================================"
    echo "COMPLETE $NAME"
    date
    echo "======================================================================"


    sleep 10
}


run_one \
  patch \
  2 \
  corrnll_patchboost2_pe3732_v1 \
  29721


run_one \
  patch \
  4 \
  corrnll_patchboost4_pe3732_v1 \
  29723


run_one \
  random \
  4 \
  corrnll_randomboost4_pe3732_v1 \
  29725


###############################################################################
# FINAL SUMMARY
###############################################################################

python - <<'PY'
import json
import os
from pathlib import Path

import sacrebleu


RUN_ROOT = Path(
    os.environ[
        "RUN_ROOT"
    ]
)

EXP = (
    "mtpatcher_v3_full6565_20260823"
)

ROOT = RUN_ROOT / EXP


SYSTEMS = {
    "FullCorr-NLL":
        ROOT
        / "corrnll_full_pe3732_v1"
        / "eval_epoch3",

    "PatchBoost-2x":
        ROOT
        / "corrnll_patchboost2_pe3732_v1"
        / "eval_epoch3",

    "PatchBoost-4x":
        ROOT
        / "corrnll_patchboost4_pe3732_v1"
        / "eval_epoch3",

    "RandomBoost-4x":
        ROOT
        / "corrnll_randomboost4_pe3732_v1"
        / "eval_epoch3",
}


BASE = (
    ROOT
    / "_verified_base_eval_v3"
)


SPLITS = (
    "wmt24",
    "flores",
    "challenge",
)


def load_predictions(path):

    rows = []

    with Path(path).open(
        "r",
        encoding="utf-8-sig",
    ) as f:

        for line in f:

            if line.strip():

                rows.append(
                    json.loads(
                        line
                    )
                )

    rows.sort(
        key=lambda x: int(
            x[
                "index"
            ]
        )
    )

    return rows


base_scores = {}


for split in SPLITS:

    rows = load_predictions(
        BASE
        / split
        / "predictions.jsonl"
    )

    refs = [
        row[
            "reference"
        ]
        for row in rows
    ]

    hyps = [
        row[
            "student_translation"
        ]
        for row in rows
    ]


    base_scores[
        split
    ] = {
        "BLEU":
            sacrebleu.corpus_bleu(
                hyps,
                [
                    refs
                ],
            ).score,

        "chrF":
            sacrebleu.corpus_chrf(
                hyps,
                [
                    refs
                ],
            ).score,
    }


results = {}


print(
    "=" * 122
)

print(
    "MT-PATCHER PATCH-BOOST NLL FINAL RESULTS"
)

print(
    "=" * 122
)

print(
    f"{'SYSTEM':20s} "
    f"{'WMT':>10s} "
    f"{'FLORES':>10s} "
    f"{'CHALL':>10s} "
    f"{'AVG ΔBLEU':>12s} "
    f"{'AVG ΔchrF':>12s}"
)


for system, family in SYSTEMS.items():

    bleus = []
    db = []
    dc = []


    for split in SPLITS:

        metrics_path = (
            family
            / split
            / "metrics.json"
        )


        if not metrics_path.exists():

            raise RuntimeError(
                f"Missing metrics: "
                f"{metrics_path}"
            )


        metric = json.loads(
            metrics_path.read_text(
                encoding="utf-8"
            )
        )


        bleu = float(
            metric[
                "BLEU"
            ]
        )

        chrf = float(
            metric[
                "chrF"
            ]
        )


        bleus.append(
            bleu
        )

        db.append(
            bleu
            - base_scores[
                split
            ][
                "BLEU"
            ]
        )

        dc.append(
            chrf
            - base_scores[
                split
            ][
                "chrF"
            ]
        )


    results[
        system
    ] = {
        "avg_delta_bleu":
            sum(
                db
            )
            / 3,

        "avg_delta_chrf":
            sum(
                dc
            )
            / 3,
    }


    print(
        f"{system:20s} "
        f"{bleus[0]:10.6f} "
        f"{bleus[1]:10.6f} "
        f"{bleus[2]:10.6f} "
        f"{sum(db)/3:+12.6f} "
        f"{sum(dc)/3:+12.6f}"
    )


full = results[
    "FullCorr-NLL"
][
    "avg_delta_bleu"
]

p2 = results[
    "PatchBoost-2x"
][
    "avg_delta_bleu"
]

p4 = results[
    "PatchBoost-4x"
][
    "avg_delta_bleu"
]

r4 = results[
    "RandomBoost-4x"
][
    "avg_delta_bleu"
]


print()
print(
    "=" * 122
)

print(
    "CAUSAL COMPARISONS"
)

print(
    "=" * 122
)

print(
    f"PatchBoost2 - FullCorr  = "
    f"{p2-full:+.6f} BLEU"
)

print(
    f"PatchBoost4 - FullCorr  = "
    f"{p4-full:+.6f} BLEU"
)

print(
    f"PatchBoost4 - Patch2    = "
    f"{p4-p2:+.6f} BLEU"
)

print(
    f"PatchBoost4 - Random4   = "
    f"{p4-r4:+.6f} BLEU"
)

print(
    f"RandomBoost4 - FullCorr = "
    f"{r4-full:+.6f} BLEU"
)


print()
print(
    "REFERENCE FROZEN BASELINES"
)

print(
    "Patch-only NLL         = -0.644432"
)

print(
    "Random sparse NLL      = +0.150531"
)

print(
    "Halo1 NLL              = +0.099279"
)

print(
    "FullCorr NLL           = +0.285069"
)

print(
    "PE-SFT3732             = +0.390"
)

print(
    "SeqKD-Selected3732     = +1.182"
)

print(
    "SeqKD-Full6565         = +1.465"
)

print()
print(
    "PATCHBOOST_NLL_QUEUE_ALL_PASS"
)
PY

BASHQ


###############################################################################
# STATIC / PROVENANCE CHECK
###############################################################################

chmod +x \
"$TRAINER" \
"$QUEUE"


python -m py_compile \
"$TRAINER" \
"$EVALUATOR"

bash -n \
"$QUEUE"

test -f "$DATA"

test -d "$STUDENT"

echo
echo "PATCHBOOST_STATIC_COMPILE_PASS"


###############################################################################
# VERIFY EXISTING FULL-CORRECTION BASELINE IS MATCHED
###############################################################################

export FULL_BASELINE DATA

python - <<'PY'
import json
import os
from pathlib import Path


baseline = Path(
    os.environ[
        "FULL_BASELINE"
    ]
)

data = Path(
    os.environ[
        "DATA"
    ]
)


manifest_path = (
    baseline
    / "training_manifest.json"
)


if not manifest_path.exists():

    raise RuntimeError(
        f"Missing existing FullCorr manifest: "
        f"{manifest_path}"
    )


manifest = json.loads(
    manifest_path.read_text(
        encoding="utf-8"
    )
)


print(
    "FULLCORR_BASELINE_MANIFEST =",
    {
        "method":
            manifest.get(
                "method"
            ),

        "mask_mode":
            manifest.get(
                "mask_mode"
            ),

        "rows":
            manifest.get(
                "rows"
            ),

        "epochs":
            manifest.get(
                "epochs"
            ),

        "lr":
            manifest.get(
                "lr"
            ),

        "global_batch":
            manifest.get(
                "global_batch"
            ),

        "train_sha256":
            manifest.get(
                "train_sha256"
            ),
    },
)


if int(
    manifest.get(
        "rows",
        -1,
    )
) != 3732:

    raise RuntimeError(
        "FullCorr row count mismatch"
    )


if int(
    manifest.get(
        "epochs",
        -1,
    )
) != 3:

    raise RuntimeError(
        "FullCorr epoch mismatch"
    )


if abs(
    float(
        manifest.get(
            "lr",
            -1,
        )
    )
    - 2e-5
) > 1e-12:

    raise RuntimeError(
        "FullCorr LR mismatch"
    )


if int(
    manifest.get(
        "global_batch",
        -1,
    )
) != 16:

    raise RuntimeError(
        "FullCorr global batch mismatch"
    )


for split in (
    "wmt24",
    "flores",
    "challenge",
):

    path = (
        baseline
        / "eval_epoch3"
        / split
        / "metrics.json"
    )

    if not path.exists():

        raise RuntimeError(
            f"Missing FullCorr eval "
            f"{split}"
        )


print(
    "FULLCORR_MATCHED_BASELINE_GATE_PASS"
)
PY


###############################################################################
# MASK MATCHING AUDIT
###############################################################################

python - "$DATA" <<'PY'
import json
import sys
from pathlib import Path


path = Path(
    sys.argv[
        1
    ]
)


rows = [
    json.loads(
        x
    )
    for x in path.read_text(
        encoding="utf-8"
    ).splitlines()
    if x.strip()
]


if len(
    rows
) != 3732:

    raise RuntimeError(
        len(
            rows
        )
    )


patch_total = 0
random_total = 0
response_total = 0


for i, row in enumerate(
    rows
):

    m = row[
        "_patch_aware_v2"
    ]

    patch = m[
        "patch_mask"
    ]

    random_mask = m[
        "random_equal_count_mask"
    ]

    full = m[
        "full_correction_mask"
    ]


    if len(
        patch
    ) != len(
        random_mask
    ):

        raise RuntimeError(
            f"Per-row matched count "
            f"failed row={i}"
        )


    if not set(
        patch
    ).issubset(
        set(
            full
        )
    ):

        raise RuntimeError(
            f"Patch outside trajectory "
            f"row={i}"
        )


    if not set(
        random_mask
    ).issubset(
        set(
            full
        )
    ):

        raise RuntimeError(
            f"Random outside trajectory "
            f"row={i}"
        )


    patch_total += len(
        patch
    )

    random_total += len(
        random_mask
    )

    response_total += len(
        full
    )


print(
    "BOOST_MASK_AUDIT =",
    {
        "rows":
            len(
                rows
            ),

        "patch_total":
            patch_total,

        "random_total":
            random_total,

        "response_total":
            response_total,

        "patch_fraction":
            patch_total
            / response_total,
    },
)


if patch_total != 26665:

    raise RuntimeError(
        patch_total
    )


if random_total != patch_total:

    raise RuntimeError(
        "Patch/random count mismatch"
    )


if response_total != 146817:

    raise RuntimeError(
        response_total
    )


print(
    "PATCH_RANDOM_WEIGHT_BUDGET_MATCH_PASS"
)
PY


###############################################################################
# PREFLIGHT — PATCH BOOST 4X
###############################################################################

PATCH_PREFLIGHT="$LOG_ROOT/$EXP/patchboost4_preflight_v1.log"

echo
echo "======================================================================"
echo "PATCHBOOST-4X PREFLIGHT"
echo "======================================================================"

python -m torch.distributed.run \
  --nproc_per_node=16 \
  --master_port=29715 \
  "$TRAINER" \
  --student-model "$STUDENT" \
  --train "$DATA" \
  --output-dir "$RUN_ROOT/$EXP/_patchboost4_preflight_v1" \
  --boost-mode patch \
  --boost-factor 4 \
  --lr 2e-5 \
  --epochs 1 \
  --max-length 1024 \
  --warmup-ratio 0.03 \
  --weight-decay 0.01 \
  --max-grad-norm 1.0 \
  --seed 20260825 \
  --preflight \
  > "$PATCH_PREFLIGHT" 2>&1


grep -E \
'WEIGHTED_NLL_RUNTIME_AUDIT|WEIGHTED_NLL_PREFLIGHT|WEIGHTED_NLL_PREFLIGHT_PASS|Traceback|RuntimeError|ERROR|ChildFailedError' \
"$PATCH_PREFLIGHT" \
| tail -n 80


grep -q \
'WEIGHTED_NLL_PREFLIGHT_PASS patch 4' \
"$PATCH_PREFLIGHT"

echo "PATCHBOOST4_PREFLIGHT_GATE_PASS"


###############################################################################
# PREFLIGHT — RANDOM BOOST 4X
###############################################################################

RANDOM_PREFLIGHT="$LOG_ROOT/$EXP/randomboost4_preflight_v1.log"

echo
echo "======================================================================"
echo "RANDOMBOOST-4X PREFLIGHT"
echo "======================================================================"

python -m torch.distributed.run \
  --nproc_per_node=16 \
  --master_port=29717 \
  "$TRAINER" \
  --student-model "$STUDENT" \
  --train "$DATA" \
  --output-dir "$RUN_ROOT/$EXP/_randomboost4_preflight_v1" \
  --boost-mode random \
  --boost-factor 4 \
  --lr 2e-5 \
  --epochs 1 \
  --max-length 1024 \
  --warmup-ratio 0.03 \
  --weight-decay 0.01 \
  --max-grad-norm 1.0 \
  --seed 20260825 \
  --preflight \
  > "$RANDOM_PREFLIGHT" 2>&1


grep -E \
'WEIGHTED_NLL_RUNTIME_AUDIT|WEIGHTED_NLL_PREFLIGHT|WEIGHTED_NLL_PREFLIGHT_PASS|Traceback|RuntimeError|ERROR|ChildFailedError' \
"$RANDOM_PREFLIGHT" \
| tail -n 80


grep -q \
'WEIGHTED_NLL_PREFLIGHT_PASS random 4' \
"$RANDOM_PREFLIGHT"

echo "RANDOMBOOST4_PREFLIGHT_GATE_PASS"


###############################################################################
# DUPLICATE GATE
###############################################################################

RUNNING="$(
    pgrep -af \
    '[r]un_patchboost_nll_queue_v1.sh|[t]rain_weighted_correction_nll_torchnpu_v1.py' \
    || true
)"


if [[ -n "$RUNNING" ]]; then

    echo
    echo "Existing PatchBoost process:"
    echo "$RUNNING"

    false

fi


###############################################################################
# LAUNCH THREE-WAY QUEUE
###############################################################################

echo
echo "======================================================================"
echo "LAUNCH PATCHBOOST NLL QUEUE"
echo "======================================================================"

nohup setsid bash "$QUEUE" \
  > "$QUEUE_LOG" 2>&1 < /dev/null &


PID=$!


echo "PID=$PID"

echo "QUEUE_LOG=$QUEUE_LOG"

echo "PATCHBOOST_NLL_QUEUE_STARTED"


###############################################################################
# WAIT UNTIL PATCHBOOST-2X EPOCH 1 CHECKPOINT EXISTS
###############################################################################

FIRST_LOG="$LOG_ROOT/$EXP/corrnll_patchboost2_pe3732_v1.log"

PASS=0


for ROUND in \
    1 2 3 4 5 6 7 8 9 10 11 12
do

    sleep 20

    echo
    echo "HEALTH_ROUND=$ROUND"


    if [[ -f "$FIRST_LOG" ]]; then

        grep -E \
'WEIGHTED_NLL_RUNTIME_AUDIT|WEIGHTED_CORRECTION_NLL_TRAINING_START|epoch=1 local_step=|EPOCH_1_COMPLETE|weighted_mean_nll|full_unweighted_mean_nll|boost_position_mean_nll|CHECKPOINT_SAVED|Traceback|RuntimeError|ERROR|ChildFailedError' \
        "$FIRST_LOG" \
        | tail -n 70 \
        || true


        if grep -q \
        'EPOCH_1_COMPLETE' \
        "$FIRST_LOG" \
        && grep -q \
        'CHECKPOINT_SAVED = .*corrnll_patchboost2_pe3732_v1/epoch1' \
        "$FIRST_LOG"; then

            PASS=1

            break

        fi


        if grep -qE \
        'Traceback|RuntimeError|ChildFailedError' \
        "$FIRST_LOG"; then

            echo
            echo "PATCHBOOST TRAINING FAILED"

            tail -n 180 \
            "$FIRST_LOG"

            false

        fi

    fi


    Q="$(
        pgrep -af \
        '[r]un_patchboost_nll_queue_v1.sh' \
        || true
    )"


    T="$(
        pgrep -af \
        '[t]rain_weighted_correction_nll_torchnpu_v1.py' \
        || true
    )"


    echo \
      "QUEUE_ALIVE=$([[ -n "$Q" ]] && echo YES || echo NO)"

    echo \
      "TRAINER_ALIVE=$([[ -n "$T" ]] && echo YES || echo NO)"


    if [[ -z "$Q" ]] && [[ -z "$T" ]]; then

        echo
        echo "PatchBoost queue died before Epoch 1 checkpoint."

        echo
        echo "QUEUE LOG:"

        tail -n 160 \
          "$QUEUE_LOG" \
          2>/dev/null \
          || true

        echo
        echo "TRAIN LOG:"

        tail -n 200 \
          "$FIRST_LOG" \
          2>/dev/null \
          || true

        false

    fi

done


if [[ "$PASS" -ne 1 ]]; then

    echo
    echo "PatchBoost Epoch-1 confirmation timeout."

    tail -n 180 \
      "$FIRST_LOG" \
      2>/dev/null \
      || true

    false

fi


echo
echo "======================================================================"
echo "PATCHBOOST LONG RUN VERIFIED"
echo "======================================================================"

echo "FULLCORR_MATCHED_BASELINE_GATE_PASS"
echo "PATCH_RANDOM_WEIGHT_BUDGET_MATCH_PASS"
echo "PATCHBOOST4_PREFLIGHT_GATE_PASS"
echo "RANDOMBOOST4_PREFLIGHT_GATE_PASS"
echo "PATCHBOOST2_EPOCH1_CHECKPOINT_PASS"
echo "PATCHBOOST_NLL_LONG_QUEUE_RUNNING"

echo
echo "Queue:"
echo "  PatchBoost-2x"
echo "  PatchBoost-4x"
echo "  RandomBoost-4x"

echo
echo "Existing comparison baseline:"
echo "  FullCorr-NLL"

echo
echo "QUEUE_LOG=$QUEUE_LOG"

echo
echo "PATCHBOOST_NLL_LONG_RUN_SAFE"

