#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

SCRIPT_DIR="$ROOT/scripts/mtpatcher_v6"

TRAINER="$SCRIPT_DIR/train_correction_fkl_torchnpu_v1.py"
EVALUATOR="$SCRIPT_DIR/eval_correction_fkl_torchnpu_v1.py"
QUEUE="$SCRIPT_DIR/run_correction_fkl_night_queue_v1.sh"

DATA="$DATA_ROOT/$EXP/patch_aware_k1_clean3732_v2.jsonl"

STUDENT="$MODEL_ROOT/Qwen3-0.6B"
TEACHER="$MODEL_ROOT/Qwen3-8B"

QUEUE_LOG="$LOG_ROOT/$EXP/correction_fkl_night_queue_v1.log"

mkdir -p "$LOG_ROOT/$EXP"


###############################################################################
# TRAINER
###############################################################################

cat > "$TRAINER" <<'PY'
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


    if int(
        student_raw.config.vocab_size
    ) != len(
        tokenizer
    ):

        raise RuntimeError(
            "tokenizer/model vocab mismatch"
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
            dtype=torch.float64,
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
PY


###############################################################################
# EVALUATOR
###############################################################################

cat > "$EVALUATOR" <<'PY'
#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

import sacrebleu
import torch
import torch_npu

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


def read_jsonl(
    path
):

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


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        required=True,
    )

    ap.add_argument(
        "--input",
        required=True,
        type=Path,
    )

    ap.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    ap.add_argument(
        "--metrics",
        required=True,
        type=Path,
    )

    ap.add_argument(
        "--method",
        required=True,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )

    args = ap.parse_args()


    torch.npu.set_device(
        0
    )

    device = torch.device(
        "npu:0"
    )


    rows = read_jsonl(
        args.input
    )


    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
    )


    tokenizer.padding_side = "left"


    if tokenizer.pad_token_id is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )


    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )


    model.to(
        device
    )

    model.eval()


    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    output_rows = []

    started = time.time()


    with torch.inference_mode():

        for start in range(
            0,
            len(rows),
            args.batch_size,
        ):

            batch = rows[
                start:
                start
                + args.batch_size
            ]


            prompts = [
                tokenizer.apply_chat_template(
                    row[
                        "messages"
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )

                for row in batch
            ]


            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )


            encoded = {
                k:
                    v.to(
                        device
                    )

                for k, v in encoded.items()
            }


            input_width = encoded[
                "input_ids"
            ].shape[
                1
            ]


            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=
                    args.max_new_tokens,
                pad_token_id=
                    tokenizer.pad_token_id,
                eos_token_id=
                    tokenizer.eos_token_id,
            )


            new_ids = generated[
                :,
                input_width:
            ]


            texts = tokenizer.batch_decode(
                new_ids,
                skip_special_tokens=True,
            )


            for row, text in zip(
                batch,
                texts,
            ):

                translation = (
                    text.strip()
                )


                if not translation:

                    raise RuntimeError(
                        f"empty translation "
                        f"index={row['index']}"
                    )


                output_rows.append(
                    {
                        "index":
                            int(
                                row[
                                    "index"
                                ]
                            ),

                        "source":
                            row[
                                "source"
                            ],

                        "reference":
                            row[
                                "reference"
                            ],

                        "student_translation":
                            translation,

                        "evaluation_method":
                            args.method,

                        "student_model_path":
                            args.model,
                    }
                )


            print(
                f"EVAL_PROGRESS "
                f"{len(output_rows)}/{len(rows)}",
                flush=True,
            )


    output_rows.sort(
        key=lambda x: int(
            x[
                "index"
            ]
        )
    )


    if [
        int(
            x[
                "index"
            ]
        )
        for x in output_rows
    ] != list(
        range(
            len(rows)
        )
    ):

        raise RuntimeError(
            "evaluation index coverage mismatch"
        )


    with args.output.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in output_rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(
                        ",",
                        ":",
                    ),
                )
                + "\n"
            )


    refs = [
        x[
            "reference"
        ]
        for x in output_rows
    ]

    hyps = [
        x[
            "student_translation"
        ]
        for x in output_rows
    ]


    bleu = sacrebleu.corpus_bleu(
        hyps,
        [
            refs
        ],
    ).score


    chrf = sacrebleu.corpus_chrf(
        hyps,
        [
            refs
        ],
    ).score


    result = {
        "rows":
            len(
                output_rows
            ),

        "BLEU":
            bleu,

        "chrF":
            chrf,

        "seconds":
            time.time()
            - started,

        "method":
            args.method,

        "model":
            args.model,
    }


    args.metrics.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    args.metrics.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    print(
        f"BLEU={bleu:.6f}"
    )

    print(
        f"CHRF={chrf:.6f}"
    )

    print(
        "CORRECTION_FKL_EVAL_PASS"
    )


if __name__ == "__main__":

    main()
PY


###############################################################################
# NIGHT QUEUE
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

TRAINER="$ROOT/scripts/mtpatcher_v6/train_correction_fkl_torchnpu_v1.py"

EVALUATOR="$ROOT/scripts/mtpatcher_v6/eval_correction_fkl_torchnpu_v1.py"

DATA="$DATA_ROOT/$EXP/patch_aware_k1_clean3732_v2.jsonl"

STUDENT="$MODEL_ROOT/Qwen3-0.6B"
TEACHER="$MODEL_ROOT/Qwen3-8B"

WMT="$DATA_ROOT/pilot_v2_qwen3_06b/wmt24_zh_en998.jsonl"
FLORES="$DATA_ROOT/pilot_v2_qwen3_06b/flores_zh_en1012.jsonl"
CHALLENGE="$DATA_ROOT/pilot_v2_qwen3_06b/challenge_zh_en197.jsonl"


run_one () {

    MODE="$1"
    NAME="$2"
    PORT="$3"

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
          --teacher-model "$TEACHER" \
          --train "$DATA" \
          --output-dir "$OUT" \
          --mask-mode "$MODE" \
          --lr 1e-6 \
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

    mkdir -p "$EVAL"


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


    sleep 20
}


run_one \
  patch \
  corrfkl_patch_pe3732_v1 \
  29671


run_one \
  random \
  corrfkl_random_pe3732_v1 \
  29673


run_one \
  halo1 \
  corrfkl_halo1_pe3732_v1 \
  29675


run_one \
  full \
  corrfkl_full_pe3732_v1 \
  29677


###############################################################################
# FINAL CONSOLIDATED SUMMARY
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
    "Patch-FKL":
        ROOT
        / "corrfkl_patch_pe3732_v1"
        / "eval_epoch3",

    "Random-FKL":
        ROOT
        / "corrfkl_random_pe3732_v1"
        / "eval_epoch3",

    "Halo1-FKL":
        ROOT
        / "corrfkl_halo1_pe3732_v1"
        / "eval_epoch3",

    "FullCorr-FKL":
        ROOT
        / "corrfkl_full_pe3732_v1"
        / "eval_epoch3",
}


BASE = (
    ROOT
    / "_verified_base_eval_v3"
)


def load_predictions(
    path
):

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


splits = (
    "wmt24",
    "flores",
    "challenge",
)


base_scores = {}


for split in splits:

    rows = load_predictions(
        BASE
        / split
        / "predictions.jsonl"
    )

    refs = [
        x[
            "reference"
        ]
        for x in rows
    ]

    hyps = [
        x[
            "student_translation"
        ]
        for x in rows
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


print(
    "=" * 110
)

print(
    "CORRECTION-TRAJECTORY FKL FINAL RESULTS"
)

print(
    "=" * 110
)


print(
    f"{'SYSTEM':18s} "
    f"{'WMT':>10s} "
    f"{'FLORES':>10s} "
    f"{'CHALL':>10s} "
    f"{'AVG ΔBLEU':>12s} "
    f"{'AVG ΔchrF':>12s}"
)


for system, family in SYSTEMS.items():

    bleu_values = []

    delta_bleu = []
    delta_chrf = []


    for split in splits:

        metric = json.loads(
            (
                family
                / split
                / "metrics.json"
            ).read_text(
                encoding="utf-8"
            )
        )


        bleu_values.append(
            float(
                metric[
                    "BLEU"
                ]
            )
        )


        delta_bleu.append(
            float(
                metric[
                    "BLEU"
                ]
            )
            - base_scores[
                split
            ][
                "BLEU"
            ]
        )


        delta_chrf.append(
            float(
                metric[
                    "chrF"
                ]
            )
            - base_scores[
                split
            ][
                "chrF"
            ]
        )


    print(
        f"{system:18s} "
        f"{bleu_values[0]:10.6f} "
        f"{bleu_values[1]:10.6f} "
        f"{bleu_values[2]:10.6f} "
        f"{sum(delta_bleu)/3:+12.6f} "
        f"{sum(delta_chrf)/3:+12.6f}"
    )


print()
print(
    "CORRECTION_FKL_NIGHT_QUEUE_ALL_PASS"
)
PY

BASHQ


chmod +x \
"$TRAINER" \
"$EVALUATOR" \
"$QUEUE"


###############################################################################
# STATIC CHECK
###############################################################################

echo
echo "======================================================================"
echo "STATIC CHECK"
echo "======================================================================"

python -m py_compile \
"$TRAINER" \
"$EVALUATOR"

bash -n \
"$QUEUE"

test -f "$DATA"

test -d "$STUDENT"

test -d "$TEACHER"

echo "CORRECTION_FKL_STATIC_CHECK_PASS"


###############################################################################
# DATA CONTROL AUDIT
###############################################################################

python - "$DATA" <<'PY'
import json
import sys
from pathlib import Path


path = Path(
    sys.argv[1]
)

rows = [
    json.loads(x)
    for x in path.read_text(
        encoding="utf-8"
    ).splitlines()
    if x.strip()
]


if len(
    rows
) != 3732:

    raise RuntimeError(
        len(rows)
    )


patch = 0
random_n = 0
halo = 0
full = 0


for i, row in enumerate(
    rows
):

    meta = row[
        "_patch_aware_v2"
    ]

    p = meta[
        "patch_mask"
    ]

    r = meta[
        "random_equal_count_mask"
    ]

    h = meta[
        "patch_halo1_mask"
    ]

    f = meta[
        "full_correction_mask"
    ]


    if len(
        p
    ) != len(
        r
    ):

        raise RuntimeError(
            f"patch/random cardinality differs row={i}"
        )


    if not set(
        p
    ).issubset(
        set(
            h
        )
    ):

        raise RuntimeError(
            f"patch not subset halo row={i}"
        )


    if not set(
        h
    ).issubset(
        set(
            f
        )
    ):

        raise RuntimeError(
            f"halo not subset full row={i}"
        )


    patch += len(
        p
    )

    random_n += len(
        r
    )

    halo += len(
        h
    )

    full += len(
        f
    )


print(
    "PATCH_TOKENS =",
    patch,
)

print(
    "RANDOM_TOKENS =",
    random_n,
)

print(
    "HALO1_TOKENS =",
    halo,
)

print(
    "FULL_TOKENS =",
    full,
)


if patch != random_n:

    raise RuntimeError(
        "global patch/random count mismatch"
    )


print(
    "PATCH_RANDOM_EQUAL_CARDINALITY_PASS"
)

print(
    "PATCH_MASK_NESTING_PASS"
)

print(
    "CORRECTION_FKL_DATA_CONTROL_AUDIT_PASS"
)
PY


###############################################################################
# 16-NPU PREFLIGHT: PATCH
###############################################################################

PATCH_PREFLIGHT_LOG="$LOG_ROOT/$EXP/corrfkl_patch_preflight_v1.log"

echo
echo "======================================================================"
echo "16-NPU PATCH PREFLIGHT"
echo "======================================================================"

python -m torch.distributed.run \
  --nproc_per_node=16 \
  --master_port=29667 \
  "$TRAINER" \
  --student-model "$STUDENT" \
  --teacher-model "$TEACHER" \
  --train "$DATA" \
  --output-dir "$RUN_ROOT/$EXP/_corrfkl_patch_preflight_v1" \
  --mask-mode patch \
  --lr 1e-6 \
  --epochs 1 \
  --max-length 1024 \
  --warmup-ratio 0.03 \
  --weight-decay 0.01 \
  --max-grad-norm 1.0 \
  --seed 20260825 \
  --preflight \
  > "$PATCH_PREFLIGHT_LOG" 2>&1


grep -E \
'CORRECTION_FKL_RUNTIME_AUDIT|CORRECTION_FKL_PREFLIGHT|CORRECTION_FKL_PREFLIGHT_PASS|Traceback|RuntimeError' \
"$PATCH_PREFLIGHT_LOG" \
| tail -n 30


grep -q \
'CORRECTION_FKL_PREFLIGHT_PASS patch' \
"$PATCH_PREFLIGHT_LOG"

echo "PATCH_PREFLIGHT_GATE_PASS"


###############################################################################
# 16-NPU PREFLIGHT: FULL
###############################################################################

FULL_PREFLIGHT_LOG="$LOG_ROOT/$EXP/corrfkl_full_preflight_v1.log"

echo
echo "======================================================================"
echo "16-NPU FULL-CORRECTION PREFLIGHT"
echo "======================================================================"

python -m torch.distributed.run \
  --nproc_per_node=16 \
  --master_port=29669 \
  "$TRAINER" \
  --student-model "$STUDENT" \
  --teacher-model "$TEACHER" \
  --train "$DATA" \
  --output-dir "$RUN_ROOT/$EXP/_corrfkl_full_preflight_v1" \
  --mask-mode full \
  --lr 1e-6 \
  --epochs 1 \
  --max-length 1024 \
  --warmup-ratio 0.03 \
  --weight-decay 0.01 \
  --max-grad-norm 1.0 \
  --seed 20260825 \
  --preflight \
  > "$FULL_PREFLIGHT_LOG" 2>&1


grep -E \
'CORRECTION_FKL_RUNTIME_AUDIT|CORRECTION_FKL_PREFLIGHT|CORRECTION_FKL_PREFLIGHT_PASS|Traceback|RuntimeError' \
"$FULL_PREFLIGHT_LOG" \
| tail -n 30


grep -q \
'CORRECTION_FKL_PREFLIGHT_PASS full' \
"$FULL_PREFLIGHT_LOG"

echo "FULL_PREFLIGHT_GATE_PASS"


###############################################################################
# PROTECT AGAINST DUPLICATE NIGHT QUEUE
###############################################################################

RUNNING="$(
    pgrep -af \
    'run_correction_fkl_night_queue_v1.sh' \
    || true
)"


if [[ -n "$RUNNING" ]]; then

    echo
    echo "EXISTING NIGHT QUEUE FOUND:"
    echo "$RUNNING"

    echo "DUPLICATE NIGHT QUEUE NOT STARTED."

else

    nohup setsid bash "$QUEUE" \
      > "$QUEUE_LOG" 2>&1 < /dev/null &


    PID=$!


    echo
    echo "======================================================================"
    echo "CORRECTION FKL NIGHT QUEUE STARTED"
    echo "======================================================================"

    echo "PID=$PID"
    echo "QUEUE=$QUEUE"
    echo "LOG=$QUEUE_LOG"

    echo
    echo "CORRECTION_FKL_NIGHT_QUEUE_STARTED"

fi


###############################################################################
# FIRST NIGHT-QUEUE HEALTH CHECK
###############################################################################

sleep 120

echo
echo "======================================================================"
echo "NIGHT QUEUE FIRST HEALTH CHECK"
echo "======================================================================"


if [[ -f "$QUEUE_LOG" ]]; then

    tail -n 80 \
    "$QUEUE_LOG"

fi


echo
echo "======================================================================"
echo "READY FOR UNATTENDED RUN"
echo "======================================================================"

echo "QUEUE_LOG=$QUEUE_LOG"

echo
echo "To inspect later:"
echo "tail -n 100 $QUEUE_LOG"

echo
echo "CORRECTION_FKL_UNATTENDED_QUEUE_GATE_PASS"

