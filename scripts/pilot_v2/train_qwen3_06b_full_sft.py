#!/usr/bin/env python3
# coding: utf-8

import argparse
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("messages"), list) or not row["messages"]:
                raise RuntimeError(f"line {line_no}: invalid messages")
            target = row.get("target_translation")
            if not isinstance(target, str) or not target.strip():
                raise RuntimeError(f"line {line_no}: missing target_translation")
            rows.append(row)
    return rows


def render_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


class TokenizedDataset(Dataset):
    def __init__(self, tokenizer, rows, max_length):
        self.items = []
        lengths = []
        response_lengths = []

        eos = tokenizer.eos_token_id
        if eos is None:
            raise RuntimeError("tokenizer.eos_token_id is None")

        overlong = []

        for i, row in enumerate(rows):
            prompt_text = render_prompt(tokenizer, row["messages"])
            prompt_ids = tokenizer(
                prompt_text,
                add_special_tokens=False,
            )["input_ids"]

            target = row["target_translation"].strip()
            target_ids = tokenizer(
                target,
                add_special_tokens=False,
            )["input_ids"]

            input_ids = prompt_ids + target_ids + [eos]
            labels = ([-100] * len(prompt_ids)) + target_ids + [eos]

            if len(input_ids) > max_length:
                overlong.append((i, len(input_ids), row.get("index")))
                continue

            self.items.append(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                    "sequence_length": len(input_ids),
                    "response_tokens": len(target_ids) + 1,
                }
            )
            lengths.append(len(input_ids))
            response_lengths.append(len(target_ids) + 1)

        if overlong:
            first = overlong[:10]
            raise RuntimeError(
                f"{len(overlong)} examples exceed max_length={max_length}; "
                f"first={first}. Increase --max-length rather than silently dropping."
            )

        lengths_sorted = sorted(lengths)
        def pct(p):
            if not lengths_sorted:
                return 0
            k = min(len(lengths_sorted) - 1, int(round((len(lengths_sorted) - 1) * p)))
            return lengths_sorted[k]

        print("TOKENIZATION rows =", len(self.items))
        print("seq_len_min =", min(lengths))
        print("seq_len_mean =", sum(lengths) / len(lengths))
        print("seq_len_p50 =", pct(0.50))
        print("seq_len_p95 =", pct(0.95))
        print("seq_len_p99 =", pct(0.99))
        print("seq_len_max =", max(lengths))
        print("response_len_mean =", sum(response_lengths) / len(response_lengths))
        print("response_len_max =", max(response_lengths))
        print("PILOT_V2_TOKENIZATION_PASS")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids = []
    labels = []
    attention = []
    response_tokens = 0

    for x in batch:
        n = len(x["input_ids"])
        pad = max_len - n
        input_ids.append(x["input_ids"] + [pad_id] * pad)
        labels.append(x["labels"] + [-100] * pad)
        attention.append([1] * n + [0] * pad)
        response_tokens += x["response_tokens"]

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "response_tokens": response_tokens,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This Pilot-v2 trainer expects BF16 support (RTX 3090 supports BF16).")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda:0")

    rows = read_jsonl(args.train)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PILOT-V2 QWEN3-0.6B FULL SFT")
    print("=" * 80)
    print("model =", args.model)
    print("train =", args.train)
    print("train_sha256 =", sha256(args.train))
    print("rows =", len(rows))
    print("objective = response-only next-token loss")
    print("prompt_source = parquet-provided prompt messages")
    print("enable_thinking = False")
    print("dtype = bfloat16")
    print("full_finetune = True")
    print("lr =", args.lr)
    print("epochs =", args.epochs)
    print("batch_size =", args.batch_size)
    print("grad_accum =", args.grad_accum)
    print("effective_batch_size =", args.batch_size * args.grad_accum)
    print("max_length =", args.max_length)
    print("warmup_ratio =", args.warmup_ratio)
    print("weight_decay =", args.weight_decay)
    print("seed =", args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = TokenizedDataset(tokenizer, rows, args.max_length)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
        drop_last=False,
    )

    print("Loading model...")
    load_start = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.config.use_cache = False
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable != total:
        raise RuntimeError(f"Expected full finetuning: trainable={trainable}, total={total}")

    print("model_load_seconds =", round(time.time() - load_start, 3))
    print("trainable_parameters =", trainable)
    print("total_parameters =", total)
    print("gpu_memory_allocated_gib =", round(torch.cuda.memory_allocated() / 2**30, 3))

    params = [p for p in model.parameters() if p.requires_grad]
    try:
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            fused=True,
        )
        optimizer_impl = "AdamW(fused=True)"
    except Exception:
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        optimizer_impl = "AdamW"
    print("optimizer =", optimizer_impl)

    updates_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = int(round(total_updates * args.warmup_ratio))

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    print("batches_per_epoch =", len(loader))
    print("optimizer_updates_per_epoch =", updates_per_epoch)
    print("total_optimizer_updates =", total_updates)
    print("warmup_steps =", warmup_steps)

    global_update = 0
    metrics = []
    total_start = time.time()

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        loss_num = 0.0
        token_num = 0

        print()
        print("=" * 80)
        print(f"EPOCH {epoch}/{args.epochs}")
        print("=" * 80)

        for batch_no, batch in enumerate(loader, 1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            response_tokens = int(batch["response_tokens"])

            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                )
                loss = outputs.loss

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss epoch={epoch} batch={batch_no}: {loss.detach().float().item()}"
                )

            (loss / args.grad_accum).backward()

            loss_value = float(loss.detach().float().cpu().item())
            loss_num += loss_value * response_tokens
            token_num += response_tokens

            do_step = (
                batch_no % args.grad_accum == 0
                or batch_no == len(loader)
            )

            if do_step:
                grad_norm = clip_grad_norm_(params, args.max_grad_norm)
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    raise RuntimeError(
                        f"Non-finite grad norm epoch={epoch} batch={batch_no}: {grad_norm}"
                    )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1

                if (
                    global_update == 1
                    or global_update % 20 == 0
                    or batch_no == len(loader)
                ):
                    print(
                        f"epoch={epoch} batch={batch_no}/{len(loader)} "
                        f"update={global_update}/{total_updates} "
                        f"loss={loss_value:.6f} "
                        f"grad_norm={float(grad_norm):.6f} "
                        f"lr={scheduler.get_last_lr()[0]:.8g} "
                        f"gpu_alloc_gib={torch.cuda.memory_allocated()/2**30:.3f} "
                        f"elapsed={time.time()-epoch_start:.1f}s",
                        flush=True,
                    )

            del outputs, loss, input_ids, labels, attention_mask

        epoch_loss = loss_num / token_num
        ckpt = args.output_dir / f"epoch{epoch}"
        ckpt.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(
            ckpt,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(ckpt)

        epoch_metric = {
            "epoch": epoch,
            "token_mean_response_loss": epoch_loss,
            "response_tokens": token_num,
            "optimizer_updates_total": global_update,
            "epoch_seconds": time.time() - epoch_start,
            "checkpoint": str(ckpt),
        }
        metrics.append(epoch_metric)

        (args.output_dir / f"epoch{epoch}_metrics.json").write_text(
            json.dumps(epoch_metric, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"EPOCH_{epoch}_COMPLETE")
        print("token_mean_response_loss =", epoch_loss)

        gc.collect()
        torch.cuda.empty_cache()

    manifest = {
        "pilot": "pilot_v2_qwen3_06b",
        "model": args.model,
        "train": str(args.train),
        "train_sha256": sha256(args.train),
        "rows": len(rows),
        "objective": "response_only_translation_next_token_loss",
        "prompt_source": "parquet_provided_prompt_messages",
        "enable_thinking": False,
        "dtype": "bfloat16",
        "full_finetune": True,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "max_length": args.max_length,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "epoch_metrics": metrics,
        "total_training_seconds": time.time() - total_start,
        "training_complete": True,
    }

    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("PILOT_V2_HUMAN_SFT_TRAINING_PASS")


if __name__ == "__main__":
    main()
