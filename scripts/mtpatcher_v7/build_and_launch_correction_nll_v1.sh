#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

SCRIPT_DIR="$ROOT/scripts/mtpatcher_v7"

TRAINER="$SCRIPT_DIR/train_correction_nll_torchnpu_v1.py"
QUEUE="$SCRIPT_DIR/run_correction_nll_queue_v1.sh"

# Reuse the already validated evaluator.
EVALUATOR="$ROOT/scripts/mtpatcher_v6/eval_correction_fkl_torchnpu_v1.py"

DATA="$DATA_ROOT/$EXP/patch_aware_k1_clean3732_v2.jsonl"

STUDENT="$MODEL_ROOT/Qwen3-0.6B"

QUEUE_LOG="$LOG_ROOT/$EXP/correction_nll_queue_v1.log"

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

TRAINER="$ROOT/scripts/mtpatcher_v7/train_correction_nll_torchnpu_v1.py"

EVALUATOR="$ROOT/scripts/mtpatcher_v6/eval_correction_fkl_torchnpu_v1.py"

DATA="$DATA_ROOT/$EXP/patch_aware_k1_clean3732_v2.jsonl"

STUDENT="$MODEL_ROOT/Qwen3-0.6B"

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
          --train "$DATA" \
          --output-dir "$OUT" \
          --mask-mode "$MODE" \
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
  corrnll_patch_pe3732_v1 \
  29701


run_one \
  random \
  corrnll_random_pe3732_v1 \
  29703


run_one \
  halo1 \
  corrnll_halo1_pe3732_v1 \
  29705


run_one \
  full \
  corrnll_full_pe3732_v1 \
  29707


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
    "Patch-NLL":
        ROOT
        / "corrnll_patch_pe3732_v1"
        / "eval_epoch3",

    "Random-NLL":
        ROOT
        / "corrnll_random_pe3732_v1"
        / "eval_epoch3",

    "Halo1-NLL":
        ROOT
        / "corrnll_halo1_pe3732_v1"
        / "eval_epoch3",

    "FullCorr-NLL":
        ROOT
        / "corrnll_full_pe3732_v1"
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


def load(
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


base = {}


for split in SPLITS:

    rows = load(
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

    hyp = [
        x[
            "student_translation"
        ]
        for x in rows
    ]


    base[
        split
    ] = {
        "BLEU":
            sacrebleu.corpus_bleu(
                hyp,
                [
                    refs
                ],
            ).score,

        "chrF":
            sacrebleu.corpus_chrf(
                hyp,
                [
                    refs
                ],
            ).score,
    }


results = {}


print(
    "=" * 118
)

print(
    "CORRECTION-TOKEN NLL FINAL RESULTS"
)

print(
    "=" * 118
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

    bleus = []
    db = []
    dc = []


    for split in SPLITS:

        metric = json.loads(
            (
                family
                / split
                / "metrics.json"
            ).read_text(
                encoding="utf-8"
            )
        )


        b = float(
            metric[
                "BLEU"
            ]
        )

        c = float(
            metric[
                "chrF"
            ]
        )


        bleus.append(
            b
        )

        db.append(
            b
            - base[
                split
            ][
                "BLEU"
            ]
        )

        dc.append(
            c
            - base[
                split
            ][
                "chrF"
            ]
        )


    results[
        system
    ] = {
        "wmt":
            bleus[
                0
            ],

        "flores":
            bleus[
                1
            ],

        "challenge":
            bleus[
                2
            ],

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
        f"{system:18s} "
        f"{bleus[0]:10.6f} "
        f"{bleus[1]:10.6f} "
        f"{bleus[2]:10.6f} "
        f"{sum(db)/3:+12.6f} "
        f"{sum(dc)/3:+12.6f}"
    )


print()
print(
    "=" * 118
)

print(
    "CAUSAL COMPARISONS"
)

print(
    "=" * 118
)


patch = results[
    "Patch-NLL"
][
    "avg_delta_bleu"
]

random_v = results[
    "Random-NLL"
][
    "avg_delta_bleu"
]

halo = results[
    "Halo1-NLL"
][
    "avg_delta_bleu"
]

full = results[
    "FullCorr-NLL"
][
    "avg_delta_bleu"
]


print(
    f"Patch - Random       = "
    f"{patch-random_v:+.6f} BLEU"
)

print(
    f"Halo1 - Patch        = "
    f"{halo-patch:+.6f} BLEU"
)

print(
    f"FullCorr - Patch     = "
    f"{full-patch:+.6f} BLEU"
)

print(
    f"FullCorr - Random    = "
    f"{full-random_v:+.6f} BLEU"
)


# Existing frozen baselines for direct context.
print()
print(
    "REFERENCE FROZEN BASELINES"
)

print(
    "PE-SFT3732 Avg ΔBLEU      = +0.390"
)

print(
    "SeqKD-Selected Avg ΔBLEU = +1.182"
)

print(
    "SeqKD-Full Avg ΔBLEU     = +1.465"
)

print()
print(
    "CORRECTION_NLL_QUEUE_ALL_PASS"
)
PY

BASHQ


###############################################################################
# STATIC CHECK
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
echo "CORRECTION_NLL_STATIC_PASS"


###############################################################################
# VERIFY MASK COUNTS + FULL-NLL SANITY
###############################################################################

python - "$DATA" <<'PY'
import json
import sys
from pathlib import Path


rows = [
    json.loads(
        x
    )
    for x in Path(
        sys.argv[
            1
        ]
    ).read_text(
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


counts = {
    "patch": 0,
    "random": 0,
    "halo1": 0,
    "full": 0,
}


for row in rows:

    m = row[
        "_patch_aware_v2"
    ]


    counts[
        "patch"
    ] += len(
        m[
            "patch_mask"
        ]
    )

    counts[
        "random"
    ] += len(
        m[
            "random_equal_count_mask"
        ]
    )

    counts[
        "halo1"
    ] += len(
        m[
            "patch_halo1_mask"
        ]
    )

    counts[
        "full"
    ] += len(
        m[
            "full_correction_mask"
        ]
    )


print(
    "MASK_COUNTS =",
    counts,
)


expected = {
    "patch": 26665,
    "random": 26665,
    "halo1": 46490,
    "full": 146817,
}


if counts != expected:

    raise RuntimeError(
        f"Mask totals changed: "
        f"{counts}"
    )


print(
    "CORRECTION_NLL_MASK_AUDIT_PASS"
)
PY


###############################################################################
# PREFLIGHT PATCH
###############################################################################

PATCH_PREFLIGHT="$LOG_ROOT/$EXP/corrnll_patch_preflight_v1.log"

echo
echo "======================================================================"
echo "PATCH-NLL PREFLIGHT"
echo "======================================================================"

python -m torch.distributed.run \
  --nproc_per_node=16 \
  --master_port=29695 \
  "$TRAINER" \
  --student-model "$STUDENT" \
  --train "$DATA" \
  --output-dir "$RUN_ROOT/$EXP/_corrnll_patch_preflight_v1" \
  --mask-mode patch \
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
'CORRECTION_NLL_RUNTIME_AUDIT|CORRECTION_NLL_PREFLIGHT|CORRECTION_NLL_PREFLIGHT_PASS|Traceback|RuntimeError|ERROR' \
"$PATCH_PREFLIGHT" \
| tail -n 60


grep -q \
'CORRECTION_NLL_PREFLIGHT_PASS patch' \
"$PATCH_PREFLIGHT"

echo "PATCH_NLL_PREFLIGHT_GATE_PASS"


###############################################################################
# PREFLIGHT FULL
###############################################################################

FULL_PREFLIGHT="$LOG_ROOT/$EXP/corrnll_full_preflight_v1.log"

echo
echo "======================================================================"
echo "FULL-CORRECTION NLL PREFLIGHT"
echo "======================================================================"

python -m torch.distributed.run \
  --nproc_per_node=16 \
  --master_port=29697 \
  "$TRAINER" \
  --student-model "$STUDENT" \
  --train "$DATA" \
  --output-dir "$RUN_ROOT/$EXP/_corrnll_full_preflight_v1" \
  --mask-mode full \
  --lr 2e-5 \
  --epochs 1 \
  --max-length 1024 \
  --warmup-ratio 0.03 \
  --weight-decay 0.01 \
  --max-grad-norm 1.0 \
  --seed 20260825 \
  --preflight \
  > "$FULL_PREFLIGHT" 2>&1


grep -E \
'CORRECTION_NLL_RUNTIME_AUDIT|CORRECTION_NLL_PREFLIGHT|CORRECTION_NLL_PREFLIGHT_PASS|Traceback|RuntimeError|ERROR' \
"$FULL_PREFLIGHT" \
| tail -n 60


grep -q \
'CORRECTION_NLL_PREFLIGHT_PASS full' \
"$FULL_PREFLIGHT"

echo "FULL_NLL_PREFLIGHT_GATE_PASS"


###############################################################################
# DUPLICATE PROCESS GATE
###############################################################################

RUNNING="$(
    pgrep -af \
    '[r]un_correction_nll_queue_v1.sh|[t]rain_correction_nll_torchnpu_v1.py' \
    || true
)"


if [[ -n "$RUNNING" ]]; then

    echo
    echo "Correction-NLL queue already running:"
    echo "$RUNNING"

    false

fi


###############################################################################
# LAUNCH
###############################################################################

echo
echo "======================================================================"
echo "LAUNCH CORRECTION-NLL FOUR-WAY QUEUE"
echo "======================================================================"

nohup setsid bash "$QUEUE" \
  > "$QUEUE_LOG" 2>&1 < /dev/null &


PID=$!


echo "PID=$PID"

echo "QUEUE_LOG=$QUEUE_LOG"

echo "CORRECTION_NLL_QUEUE_STARTED"


###############################################################################
# CONFIRM FIRST REAL EPOCH COMPLETES
###############################################################################

PATCH_LOG="$LOG_ROOT/$EXP/corrnll_patch_pe3732_v1.log"

PASS=0


for ROUND in \
    1 2 3 4 5 6 7 8 9 10 11 12
do

    sleep 20

    echo
    echo "HEALTH_ROUND=$ROUND"


    if [[ -f "$PATCH_LOG" ]]; then

        grep -E \
'CORRECTION_NLL_RUNTIME_AUDIT|CORRECTION_NLL_TRAINING_START|epoch=1 local_step=|EPOCH_1_COMPLETE|token_mean_correction_nll|CHECKPOINT_SAVED|Traceback|RuntimeError|ERROR|ChildFailedError' \
        "$PATCH_LOG" \
        | tail -n 60 \
        || true


        if grep -q \
        'EPOCH_1_COMPLETE' \
        "$PATCH_LOG" \
        && grep -q \
        'CHECKPOINT_SAVED = .*corrnll_patch_pe3732_v1/epoch1' \
        "$PATCH_LOG"; then

            PASS=1

            break

        fi


        if grep -qE \
        'Traceback|RuntimeError|ChildFailedError' \
        "$PATCH_LOG"; then

            echo
            echo "CORRECTION-NLL TRAINING FAILED"

            tail -n 180 \
            "$PATCH_LOG"

            false

        fi

    fi


    Q="$(
        pgrep -af \
        '[r]un_correction_nll_queue_v1.sh' \
        || true
    )"


    T="$(
        pgrep -af \
        '[t]rain_correction_nll_torchnpu_v1.py' \
        || true
    )"


    echo \
      "QUEUE_ALIVE=$([[ -n "$Q" ]] && echo YES || echo NO)"

    echo \
      "TRAINER_ALIVE=$([[ -n "$T" ]] && echo YES || echo NO)"


    if [[ -z "$Q" ]] && [[ -z "$T" ]]; then

        echo
        echo "Queue died before Epoch-1 checkpoint."

        tail -n 150 \
          "$QUEUE_LOG" \
          2>/dev/null \
          || true

        tail -n 180 \
          "$PATCH_LOG" \
          2>/dev/null \
          || true

        false

    fi

done


if [[ "$PASS" -ne 1 ]]; then

    echo
    echo "Epoch-1 confirmation timeout."

    tail -n 160 \
      "$PATCH_LOG" \
      2>/dev/null \
      || true

    false

fi


echo
echo "======================================================================"
echo "CORRECTION-NLL LONG RUN VERIFIED"
echo "======================================================================"

echo "PATCH_NLL_PREFLIGHT_GATE_PASS"
echo "FULL_NLL_PREFLIGHT_GATE_PASS"
echo "PATCH_NLL_EPOCH1_CHECKPOINT_PASS"
echo "CORRECTION_NLL_LONG_QUEUE_RUNNING"

echo
echo "Queued:"
echo "  1. Patch-NLL"
echo "  2. Random equal-count NLL"
echo "  3. Halo1-NLL"
echo "  4. Full correction NLL"

echo
echo "QUEUE_LOG=$QUEUE_LOG"

echo
echo "CORRECTION_NLL_LONG_RUN_SAFE"

