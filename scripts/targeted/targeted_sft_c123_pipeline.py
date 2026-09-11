#!/usr/bin/env python3
"""
Targeted WA-SFT positive-control C1/C2/C3 training and diagnostic evaluation.

Scientific class:
    LAB ADAPTATION / TARGETED WA-SFT POSITIVE CONTROL

Frozen matrix:
    C0 = /workspace/mtpatcher/models/Qwen3-0.6B
    C1 = C0 + Idiom5500 SFT, 3 epochs
    C2 = C0 + Chemistry5500 SFT, 3 epochs
    C3 = C0 + Combined11000 SFT, 3 epochs

Exposure contract:
    same per-example exposure. C3 intentionally has about 2x total updates of a
    single-domain arm because it contains both domains.

Training semantics:
    response-only causal-LM loss
    physical batch = 4
    grad accumulation = 4
    effective batch ~= 16
    AdamW(foreach=False)
    lr = 2e-5
    weight_decay = 0.01
    warmup_ratio = 0.03
    linear scheduler
    max_grad_norm = 1.0
    BF16 full-parameter fine-tuning
    seed = 20260820
    max_length = 1024
    user-only prompt; target_translation is appended once as supervision

Evaluation:
    exact frozen diagnostic2000
    exact direct-translation prompt
    greedy, enable_thinking=False, max_new_tokens=512
    Chemistry: case-insensitive canonical English target substring accuracy
    Idiom: materialize 1000 rows/arm for the already-frozen external judge

Engineering:
    durable progress JSON
    per-epoch resumable model+optimizer+scheduler checkpoint
    per-row durable eval shards
    dynamic ETA
    no UC/UW selection changes
"""

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

TZ8 = timezone(timedelta(hours=8))

PROMPT = (
    "Translate the following text into English without additional explanations:"
    "\n\n{source}\n\n"
)
PROMPT_SHA = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
EXPECTED_PROMPT_SHA = "63101d0739ff2ee11b0e5d548b85dcd18be7a241777893326140f12778d9a8b8"

EXPECTED_TARGET_HASHES = {
    "chemistry_train5500_sft.jsonl":
        "9703cee06bf97ee15afa4c79d0fe935b4acf59e9b60a9d33abc31f58d348f914",
    "idiom_train5500_sft.jsonl":
        "ef8491397614ae673e644de0554ea0d304a82d953bdcfca8ae91cf9ba40cd448",
    "combined_train11000_sft.jsonl":
        "69b89e8c30822c81febedd4f4e4f861d61e9e8ccf95b30806e63c6cc2afcc21a",
}
EXPECTED_DIAG_SHA = "7c5082217de8071bd0b519e83cac255896f704d0b41d9bd49ae2cd5b41af6db2"

EXPECTED_C0_CONFIG_SHA = "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"
EXPECTED_C0_TOKENIZER_CONFIG_SHA = "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"


def now():
    return datetime.now(TZ8).isoformat(timespec="seconds")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"{path}:{ln}: {e}") from e
    return rows


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_jsonl_atomic(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def verify_frozen_inputs(c0, targets, diagnostic):
    c0 = Path(c0)
    targets = Path(targets)
    diagnostic = Path(diagnostic)

    if PROMPT_SHA != EXPECTED_PROMPT_SHA:
        raise SystemExit(f"PROMPT_SHA_FAIL {PROMPT_SHA}")

    checks = {
        "c0_config": (
            c0 / "config.json",
            EXPECTED_C0_CONFIG_SHA,
        ),
        "c0_tokenizer_config": (
            c0 / "tokenizer_config.json",
            EXPECTED_C0_TOKENIZER_CONFIG_SHA,
        ),
    }
    for name, (path, expected) in checks.items():
        got = sha256_file(path)
        if got != expected:
            raise SystemExit(f"{name.upper()}_SHA_FAIL expected={expected} got={got}")

    manifest_path = targets / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN":
        raise SystemExit(f"TARGET_MANIFEST_NOT_FROZEN {manifest.get('status')}")

    for name, expected in EXPECTED_TARGET_HASHES.items():
        path = targets / name
        got = sha256_file(path)
        if got != expected:
            raise SystemExit(
                f"TARGET_SHA_FAIL file={name} expected={expected} got={got}"
            )

    diag_path = diagnostic / "targeted_diagnostic2000.jsonl"
    got = sha256_file(diag_path)
    if got != EXPECTED_DIAG_SHA:
        raise SystemExit(
            f"DIAGNOSTIC_SHA_FAIL expected={EXPECTED_DIAG_SHA} got={got}"
        )

    rows = read_jsonl(diag_path)
    if len(rows) != 2000:
        raise SystemExit(f"DIAGNOSTIC_COUNT_FAIL got={len(rows)}")

    counts = Counter((r["eval_domain"], r["eval_split"]) for r in rows)
    expected_counts = Counter({
        ("chemistry", "uc"): 500,
        ("chemistry", "uw"): 500,
        ("idiom", "uc"): 500,
        ("idiom", "uw"): 500,
    })
    if counts != expected_counts:
        raise SystemExit(f"DIAGNOSTIC_SPLIT_FAIL {dict(counts)}")

    return {
        "c0_config_sha256": EXPECTED_C0_CONFIG_SHA,
        "c0_tokenizer_config_sha256": EXPECTED_C0_TOKENIZER_CONFIG_SHA,
        "target_hashes": EXPECTED_TARGET_HASHES,
        "diagnostic_sha256": EXPECTED_DIAG_SHA,
        "prompt_sha256": EXPECTED_PROMPT_SHA,
    }


def set_all_seeds(seed):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

    import torch
    torch.manual_seed(seed)
    if hasattr(torch, "npu"):
        try:
            torch.npu.manual_seed(seed)
            torch.npu.manual_seed_all(seed)
        except Exception:
            pass


def prepare_tokenized_rows(tokenizer, rows, max_length):
    out = []
    overlong = []

    for i, r in enumerate(rows):
        messages = r.get("messages")
        if not isinstance(messages, list) or len(messages) != 1:
            raise SystemExit(
                f"USER_ONLY_MESSAGES_FAIL row={i} target_id={r.get('target_id')}"
            )
        if messages[0].get("role") != "user":
            raise SystemExit(
                f"USER_ONLY_ROLE_FAIL row={i} role={messages[0].get('role')}"
            )
        if any(m.get("role") == "assistant" for m in messages):
            raise SystemExit(f"ASSISTANT_PROMPT_LEAK row={i}")

        target = r.get("target_translation")
        if not isinstance(target, str) or not target.strip():
            raise SystemExit(f"MISSING_TARGET row={i}")

        if r.get("student_prompt_sha256") != EXPECTED_PROMPT_SHA:
            raise SystemExit(
                f"ROW_PROMPT_SHA_FAIL row={i} got={r.get('student_prompt_sha256')}"
            )

        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        prompt_ids = tokenizer(
            rendered,
            add_special_tokens=False,
        )["input_ids"]

        target_ids = tokenizer(
            target,
            add_special_tokens=False,
        )["input_ids"]

        eos = tokenizer.eos_token_id
        if eos is None:
            raise SystemExit("TOKENIZER_EOS_IS_NONE")

        input_ids = prompt_ids + target_ids + [eos]
        labels = [-100] * len(prompt_ids) + target_ids + [eos]

        if len(input_ids) > max_length:
            overlong.append(
                {
                    "row": i,
                    "target_id": r.get("target_id"),
                    "tokens": len(input_ids),
                }
            )
        else:
            out.append({
                "input_ids": input_ids,
                "labels": labels,
                "target_id": r.get("target_id"),
            })

    if overlong:
        raise SystemExit(
            f"TRAIN_MAX_LENGTH_FAIL n={len(overlong)} "
            f"max={max(x['tokens'] for x in overlong)}"
        )

    return out


class TokenRows:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def make_collate(tokenizer):
    import torch

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise SystemExit("TOKENIZER_PAD_IS_NONE")

    def collate(batch):
        maxlen = max(len(x["input_ids"]) for x in batch)
        input_ids = []
        labels = []
        attention = []

        for x in batch:
            n = len(x["input_ids"])
            pad = maxlen - n
            input_ids.append(x["input_ids"] + [pad_id] * pad)
            labels.append(x["labels"] + [-100] * pad)
            attention.append([1] * n + [0] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }

    return collate


def save_resume_checkpoint(run_dir, model, tokenizer, optimizer, scheduler,
                           completed_epoch, completed_updates):
    import torch

    run_dir = Path(run_dir)
    resume_hf = run_dir / "resume_hf"
    resume_hf_tmp = run_dir / "resume_hf.tmp"

    if resume_hf_tmp.exists():
        shutil.rmtree(resume_hf_tmp)
    resume_hf_tmp.mkdir(parents=True, exist_ok=True)

    old_use_cache = getattr(model.config, "use_cache", None)
    if old_use_cache is not None:
        model.config.use_cache = True

    model.save_pretrained(resume_hf_tmp, safe_serialization=True)
    tokenizer.save_pretrained(resume_hf_tmp)

    if old_use_cache is not None:
        model.config.use_cache = old_use_cache

    if resume_hf.exists():
        shutil.rmtree(resume_hf)
    os.replace(resume_hf_tmp, resume_hf)

    state_tmp = run_dir / "resume_state.pt.tmp"
    state_path = run_dir / "resume_state.pt"
    torch.save({
        "completed_epoch": completed_epoch,
        "completed_updates": completed_updates,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, state_tmp)
    os.replace(state_tmp, state_path)

    atomic_json(
        run_dir / "resume_meta.json",
        {
            "completed_epoch": completed_epoch,
            "completed_updates": completed_updates,
            "timestamp": now(),
        },
    )


def train_main(args):
    import torch
    import torch_npu  # noqa: F401
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    final_dir = run_dir / "final_hf"
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists() and final_dir.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("status") == "PASS":
            print(f"{now()} TRAIN_ALREADY_PASS arm={args.arm}", flush=True)
            return

    provenance = verify_frozen_inputs(
        args.c0, args.targets_dir, args.diagnostic_dir
    )

    train_path = Path(args.train)
    expected_hash = EXPECTED_TARGET_HASHES.get(train_path.name)
    if expected_hash is None:
        raise SystemExit(f"UNAPPROVED_TRAIN_FILE {train_path.name}")
    if sha256_file(train_path) != expected_hash:
        raise SystemExit(f"TRAIN_FILE_HASH_FAIL {train_path}")

    rows = read_jsonl(train_path)
    expected_rows = 11000 if args.arm == "C3" else 5500
    if len(rows) != expected_rows:
        raise SystemExit(
            f"TRAIN_COUNT_FAIL arm={args.arm} expected={expected_rows} got={len(rows)}"
        )

    device = "npu:0"
    torch.npu.set_device(device)
    set_all_seeds(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.c0)
    tokenized = prepare_tokenized_rows(tokenizer, rows, args.max_length)

    n = len(tokenized)
    microbatches_per_epoch = math.ceil(n / args.batch_size)
    updates_per_epoch = math.ceil(microbatches_per_epoch / args.grad_accum)
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = int(total_updates * args.warmup_ratio)

    resume_state = run_dir / "resume_state.pt"
    resume_hf = run_dir / "resume_hf"

    start_epoch = 1
    completed_updates = 0

    if resume_state.exists() and resume_hf.exists():
        saved = torch.load(resume_state, map_location="cpu")
        completed_epoch = int(saved["completed_epoch"])
        start_epoch = completed_epoch + 1
        completed_updates = int(saved["completed_updates"])
        model_source = resume_hf
        print(
            f"{now()} TRAIN_RESUME arm={args.arm} "
            f"completed_epoch={completed_epoch} updates={completed_updates}",
            flush=True,
        )
    else:
        saved = None
        model_source = Path(args.c0)

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)

    model.config.use_cache = False
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        foreach=False,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    if saved is not None:
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])

    collate = make_collate(tokenizer)
    started = time.time()
    recent_losses = []

    resolved_attn = getattr(model.config, "_attn_implementation", None)

    atomic_json(
        run_dir / "progress.json",
        {
            "status": "RUNNING",
            "arm": args.arm,
            "epoch": start_epoch,
            "epochs": args.epochs,
            "completed_updates": completed_updates,
            "total_updates": total_updates,
            "updated": now(),
        },
    )

    print("============================================================", flush=True)
    print(f"TARGETED SFT TRAIN {args.arm}", flush=True)
    print(f"rows={n}", flush=True)
    print(f"base={args.c0}", flush=True)
    print(f"train={train_path}", flush=True)
    print(f"epochs={args.epochs}", flush=True)
    print(f"batch_size={args.batch_size}", flush=True)
    print(f"grad_accum={args.grad_accum}", flush=True)
    print(f"updates_per_epoch={updates_per_epoch}", flush=True)
    print(f"total_updates={total_updates}", flush=True)
    print(f"lr={args.lr}", flush=True)
    print(f"warmup_steps={warmup_steps}", flush=True)
    print(f"weight_decay={args.weight_decay}", flush=True)
    print(f"max_length={args.max_length}", flush=True)
    print(f"seed={args.seed}", flush=True)
    print(f"attention_implementation={resolved_attn}", flush=True)
    print("prompt_leak=FALSE", flush=True)
    print("objective=response_only_translation_next_token_loss", flush=True)
    print("============================================================", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        g = torch.Generator()
        g.manual_seed(args.seed + epoch - 1)

        loader = DataLoader(
            TokenRows(tokenized),
            batch_size=args.batch_size,
            shuffle=True,
            generator=g,
            num_workers=0,
            collate_fn=collate,
            drop_last=False,
        )

        num_micro = len(loader)
        micro_idx = 0
        optimizer.zero_grad(set_to_none=True)

        for batch in loader:
            micro_idx += 1

            group_start = ((micro_idx - 1) // args.grad_accum) * args.grad_accum + 1
            group_end = min(group_start + args.grad_accum - 1, num_micro)
            group_size = group_end - group_start + 1

            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            raw_loss = outputs.loss
            loss = raw_loss / group_size
            loss.backward()

            recent_losses.append(float(raw_loss.detach().cpu()))
            if len(recent_losses) > 50:
                recent_losses.pop(0)

            end_group = (micro_idx == group_end)
            if not end_group:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            completed_updates += 1

            elapsed = max(time.time() - started, 1e-9)
            updates_this_session = max(
                completed_updates - ((start_epoch - 1) * updates_per_epoch), 1
            )
            session_total = total_updates - ((start_epoch - 1) * updates_per_epoch)
            rate = updates_this_session / elapsed
            remaining = total_updates - completed_updates
            eta = remaining / rate if rate else None

            if completed_updates % args.report_every == 0 or remaining == 0:
                mean_loss = sum(recent_losses) / len(recent_losses)
                lr = scheduler.get_last_lr()[0]
                print(
                    f"{now()} arm={args.arm} epoch={epoch}/{args.epochs} "
                    f"update={completed_updates}/{total_updates} "
                    f"pct={100*completed_updates/total_updates:.1f}% "
                    f"loss50={mean_loss:.6f} grad_norm={float(grad_norm):.4f} "
                    f"lr={lr:.8g} ETA={'?' if eta is None else f'{eta/60:.1f}m'}",
                    flush=True,
                )
                atomic_json(
                    run_dir / "progress.json",
                    {
                        "status": "RUNNING",
                        "arm": args.arm,
                        "epoch": epoch,
                        "epochs": args.epochs,
                        "completed_updates": completed_updates,
                        "total_updates": total_updates,
                        "percentage": round(
                            100 * completed_updates / total_updates, 2
                        ),
                        "mean_recent_loss": mean_loss,
                        "grad_norm": float(grad_norm),
                        "lr": lr,
                        "eta_seconds": round(eta, 1) if eta is not None else None,
                        "updated": now(),
                    },
                )

        # Durable epoch checkpoint; restart resumes from this exact optimizer/scheduler state.
        save_resume_checkpoint(
            run_dir,
            model,
            tokenizer,
            optimizer,
            scheduler,
            completed_epoch=epoch,
            completed_updates=completed_updates,
        )

        print(
            f"{now()} arm={args.arm} EPOCH_CHECKPOINT=PASS "
            f"epoch={epoch}/{args.epochs} updates={completed_updates}",
            flush=True,
        )

    final_tmp = run_dir / "final_hf.tmp"
    if final_tmp.exists():
        shutil.rmtree(final_tmp)
    final_tmp.mkdir(parents=True, exist_ok=True)

    model.config.use_cache = True
    model.save_pretrained(final_tmp, safe_serialization=True)
    tokenizer.save_pretrained(final_tmp)

    if final_dir.exists():
        shutil.rmtree(final_dir)
    os.replace(final_tmp, final_dir)

    manifest = {
        "status": "PASS",
        "scientific_class": "LAB ADAPTATION / TARGETED WA-SFT POSITIVE CONTROL",
        "arm": args.arm,
        "base_model": args.c0,
        "train_file": str(train_path),
        "train_sha256": sha256_file(train_path),
        "train_rows": n,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.grad_accum,
            "effective_batch_nominal": args.batch_size * args.grad_accum,
            "updates_per_epoch": updates_per_epoch,
            "total_updates": total_updates,
            "optimizer": "AdamW(foreach=False)",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": warmup_steps,
            "scheduler": "linear",
            "max_grad_norm": args.max_grad_norm,
            "precision": "bfloat16",
            "full_parameter_finetune": True,
            "seed": args.seed,
            "epoch_shuffle_seed_rule": "seed + epoch - 1",
            "max_length": args.max_length,
            "response_only": True,
            "messages_user_only": True,
            "enable_thinking": False,
            "attention_implementation": resolved_attn,
        },
        "exposure_contract": (
            "same per-example exposure; C3 intentionally has about 2x total "
            "updates of a single-domain arm"
        ),
        "frozen_input_provenance": provenance,
        "final_model": str(final_dir),
        "created": now(),
    }
    atomic_json(manifest_path, manifest)
    atomic_json(
        run_dir / "progress.json",
        {
            "status": "PASS",
            "arm": args.arm,
            "completed_updates": total_updates,
            "total_updates": total_updates,
            "percentage": 100.0,
            "updated": now(),
        },
    )

    print(f"{now()} {args.arm}_SFT_TRAIN=PASS", flush=True)
    print(f"{args.arm}_FINAL_MODEL={final_dir}", flush=True)


def eval_completed(path):
    if not Path(path).exists():
        return {}
    out = {}
    for r in read_jsonl(path):
        if r.get("status") == "PASS":
            out[r["eval_id"]] = r
    return out


def eval_worker_main(args):
    import torch
    import torch_npu  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = read_jsonl(args.input)
    assigned = [
        r for i, r in enumerate(rows)
        if i % args.num_workers == args.worker_id
    ]
    done_map = eval_completed(args.output)

    device = f"npu:{args.worker_id}"
    torch.npu.set_device(device)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    resolved_attn = getattr(model.config, "_attn_implementation", None)

    initial = sum(r["eval_id"] in done_map for r in assigned)
    done = initial
    total = len(assigned)
    started = time.time()

    print(
        f"{now()} EVAL_WORKER_READY arm={args.arm} worker={args.worker_id} "
        f"resume={initial}/{total} attn={resolved_attn}",
        flush=True,
    )

    for r in assigned:
        if r["eval_id"] in done_map:
            continue

        user_text = PROMPT.format(source=r["src_text"])
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        batch = tok(rendered, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            ids = model.generate(
                **batch,
                do_sample=False,
                max_new_tokens=512,
                use_cache=True,
            )

        gen = ids[:, batch["input_ids"].shape[1]:]
        hyp = tok.batch_decode(gen, skip_special_tokens=True)[0].strip()

        append_jsonl(
            args.output,
            {
                "status": "PASS",
                "arm": args.arm,
                "eval_id": r["eval_id"],
                "eval_domain": r["eval_domain"],
                "eval_split": r["eval_split"],
                "job_id": r["job_id"],
                "entity_key": r.get("entity_key"),
                "src_term": r["src_term"],
                "definition": r.get("definition"),
                "src_text": r["src_text"],
                "canonical_en_target": r.get("canonical_en_target"),
                "translation": hyp,
                "model_path": args.model,
                "prompt_sha256": EXPECTED_PROMPT_SHA,
                "enable_thinking": False,
                "do_sample": False,
                "max_new_tokens": 512,
                "attention_implementation": resolved_attn,
                "worker_id": args.worker_id,
                "timestamp": now(),
            },
        )

        done += 1
        elapsed = max(time.time() - started, 1e-9)
        fresh = done - initial
        rate = fresh / elapsed if fresh else 0.0
        eta = (total - done) / rate if rate else None

        if done % 20 == 0 or done == total:
            print(
                f"{now()} arm={args.arm} worker={args.worker_id} "
                f"done={done}/{total} pct={100*done/total:.1f}% "
                f"rate={rate:.2f}/s "
                f"ETA={'?' if eta is None else f'{eta/60:.1f}m'}",
                flush=True,
            )

    print(
        f"{now()} EVAL_WORKER_PASS arm={args.arm} worker={args.worker_id}",
        flush=True,
    )


def count_pass_lines(path):
    p = Path(path)
    if not p.exists():
        return 0
    return len(eval_completed(p))


def eval_master_main(args):
    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    provenance = verify_frozen_inputs(
        args.c0, args.targets_dir, args.diagnostic_dir
    )

    source_path = Path(args.diagnostic_dir) / "targeted_diagnostic2000.jsonl"
    source_rows = read_jsonl(source_path)

    shard_dir = eval_dir / "shards"
    log_dir = eval_dir / "worker_logs"
    shard_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    procs = []
    handles = []

    for wid in range(args.num_workers):
        shard = shard_dir / f"part_{wid:02d}.jsonl"
        log = log_dir / f"worker_{wid:02d}.log"
        fh = open(log, "a", encoding="utf-8")
        handles.append(fh)

        cmd = [
            sys.executable, "-u", str(Path(__file__).resolve()),
            "eval-worker",
            "--arm", args.arm,
            "--model", args.model,
            "--input", str(source_path),
            "--output", str(shard),
            "--worker-id", str(wid),
            "--num-workers", str(args.num_workers),
        ]
        p = subprocess.Popen(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        procs.append((wid, p, shard, log))

    started = time.time()
    last = None

    while True:
        done = sum(count_pass_lines(shard) for _, _, shard, _ in procs)
        running = sum(p.poll() is None for _, p, _, _ in procs)
        elapsed = max(time.time() - started, 1e-9)
        rate = done / elapsed if done else 0.0
        eta = (2000 - done) / rate if rate else None

        atomic_json(
            eval_dir / "progress.json",
            {
                "status": "RUNNING" if running else "FINALIZING",
                "arm": args.arm,
                "done": done,
                "total": 2000,
                "percentage": round(100 * done / 2000, 2),
                "workers_running": running,
                "workers_total": args.num_workers,
                "rate_rows_per_sec": round(rate, 3),
                "eta_seconds": round(eta, 1) if eta is not None else None,
                "updated": now(),
            },
        )

        state = (done, running)
        if state != last:
            print(
                f"{now()} arm={args.arm} phase=diagnostic_generate "
                f"done={done}/2000 pct={100*done/2000:.1f}% "
                f"workers={running}/{args.num_workers} rate={rate:.2f}/s "
                f"ETA={'?' if eta is None else f'{eta/60:.1f}m'}",
                flush=True,
            )
            last = state

        if running == 0:
            break
        time.sleep(5)

    for fh in handles:
        fh.close()

    bad = [
        (wid, p.returncode, str(log))
        for wid, p, _, log in procs
        if p.returncode != 0
    ]
    if bad:
        atomic_json(eval_dir / "worker_failures.json", bad)
        raise SystemExit(f"EVAL_WORKER_FAIL arm={args.arm} bad={bad}")

    merged = {}
    for _, _, shard, _ in procs:
        for r in read_jsonl(shard):
            if r.get("status") != "PASS":
                continue
            eid = r["eval_id"]
            if eid in merged:
                raise SystemExit(f"EVAL_DUP arm={args.arm} eval_id={eid}")
            merged[eid] = r

    source_ids = [r["eval_id"] for r in source_rows]
    if set(merged) != set(source_ids):
        missing = sorted(set(source_ids) - set(merged))[:20]
        extra = sorted(set(merged) - set(source_ids))[:20]
        raise SystemExit(
            f"EVAL_IDENTITY_FAIL arm={args.arm} "
            f"missing={missing} extra={extra}"
        )

    ordered = [merged[eid] for eid in source_ids]
    translations_path = eval_dir / f"{args.arm.lower()}_diagnostic2000_translations.jsonl"
    write_jsonl_atomic(translations_path, ordered)

    chem = []
    for r in ordered:
        if r["eval_domain"] != "chemistry":
            continue
        target = r["canonical_en_target"]
        hit = target.casefold() in r["translation"].casefold()
        x = dict(r)
        x["canonical_target_hit"] = hit
        chem.append(x)

    chem_path = eval_dir / f"{args.arm.lower()}_chemistry_diagnostic1000_scored.jsonl"
    write_jsonl_atomic(chem_path, chem)

    def chem_stats(split=None):
        xs = chem if split is None else [
            r for r in chem if r["eval_split"] == split
        ]
        hits = sum(bool(r["canonical_target_hit"]) for r in xs)
        return {"n": len(xs), "hits": hits, "accuracy": hits / len(xs)}

    chem_summary = {
        "arm": args.arm,
        "metric": "case-insensitive canonical English target substring accuracy",
        "scientific_role": "primary Chemistry targeted diagnostic metric",
        "overall": chem_stats(),
        "uc500": chem_stats("uc"),
        "uw500": chem_stats("uw"),
    }
    atomic_json(eval_dir / "chemistry_summary.json", chem_summary)

    idiom = []
    for r in ordered:
        if r["eval_domain"] != "idiom":
            continue
        if not r.get("definition"):
            raise SystemExit(
                f"IDIOM_DEFINITION_MISSING arm={args.arm} eval_id={r['eval_id']}"
            )
        idiom.append({
            "eval_id": f"{args.arm}|{r['eval_id']}",
            "base_eval_id": r["eval_id"],
            "arm": args.arm,
            "job_id": r["job_id"],
            "split": r["eval_split"],
            "src_term": r["src_term"],
            "definition": r["definition"],
            "src_text": r["src_text"],
            "model_translation": r["translation"],
        })

    if Counter(r["split"] for r in idiom) != Counter({"uc": 500, "uw": 500}):
        raise SystemExit(f"IDIOM_SPLIT_FAIL arm={args.arm}")

    idiom_path = eval_dir / f"{args.arm.lower()}_idiom_diagnostic1000_for_judge.jsonl"
    write_jsonl_atomic(idiom_path, idiom)

    manifest = {
        "status": "PASS",
        "scientific_class": "DIAGNOSTIC ONLY / LAB ADAPTATION",
        "arm": args.arm,
        "model": args.model,
        "model_config_sha256": sha256_file(Path(args.model) / "config.json"),
        "generation_contract": {
            "prompt": PROMPT,
            "prompt_sha256": EXPECTED_PROMPT_SHA,
            "enable_thinking": False,
            "do_sample": False,
            "max_new_tokens": 512,
        },
        "frozen_input_provenance": provenance,
        "outputs": {
            translations_path.name: sha256_file(translations_path),
            chem_path.name: sha256_file(chem_path),
            "chemistry_summary.json": sha256_file(
                eval_dir / "chemistry_summary.json"
            ),
            idiom_path.name: sha256_file(idiom_path),
        },
        "created": now(),
    }
    atomic_json(eval_dir / "manifest.json", manifest)
    atomic_json(
        eval_dir / "progress.json",
        {
            "status": "PASS",
            "arm": args.arm,
            "done": 2000,
            "total": 2000,
            "percentage": 100.0,
            "updated": now(),
        },
    )

    print(f"{now()} {args.arm}_TARGETED_EVAL_GENERATION=PASS", flush=True)
    print(
        f"{args.arm}_CHEMISTRY_SUMMARY="
        + json.dumps(chem_summary, ensure_ascii=False),
        flush=True,
    )
    print(f"{args.arm}_IDIOM_JUDGE_INPUT={idiom_path}", flush=True)


def combine_main(args):
    run = Path(args.run_root)
    out = run / "idiom_c123_diagnostic3000_for_judge.jsonl"

    all_rows = []
    chemistry = {}

    for arm in ("C1", "C2", "C3"):
        edir = run / arm / "eval"
        jp = edir / f"{arm.lower()}_idiom_diagnostic1000_for_judge.jsonl"
        rows = read_jsonl(jp)

        if len(rows) != 1000:
            raise SystemExit(f"{arm}_IDIOM_COUNT_FAIL {len(rows)}")
        if Counter(r["split"] for r in rows) != Counter({"uc": 500, "uw": 500}):
            raise SystemExit(f"{arm}_IDIOM_SPLIT_FAIL")
        if any(r["arm"] != arm for r in rows):
            raise SystemExit(f"{arm}_ARM_FIELD_FAIL")

        all_rows.extend(rows)

        chemistry[arm] = json.loads(
            (edir / "chemistry_summary.json").read_text(encoding="utf-8")
        )

    if len(all_rows) != 3000 or len({r["eval_id"] for r in all_rows}) != 3000:
        raise SystemExit("COMBINED_IDIOM_IDENTITY_FAIL")

    write_jsonl_atomic(out, all_rows)

    c0_chem = json.loads(
        Path(args.c0_chemistry_summary).read_text(encoding="utf-8")
    )
    comparison = {
        "status": "PASS",
        "scientific_class": "DIAGNOSTIC ONLY / LAB ADAPTATION",
        "C0": c0_chem,
        "C1": chemistry["C1"],
        "C2": chemistry["C2"],
        "C3": chemistry["C3"],
        "deltas_vs_C0_overall": {
            arm: chemistry[arm]["overall"]["accuracy"] - c0_chem["overall"]["accuracy"]
            for arm in ("C1", "C2", "C3")
        },
        "idiom_c123_input": {
            "rows": 3000,
            "sha256": sha256_file(out),
        },
        "created": now(),
    }
    atomic_json(run / "chemistry_c0_c123_comparison.json", comparison)

    print("============================================================", flush=True)
    print("C1/C2/C3 SERVER PIPELINE COMPLETE", flush=True)
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)
    print(f"IDIOM_C123_JUDGE_INPUT={out}", flush=True)
    print(f"IDIOM_C123_JUDGE_INPUT_SHA256={sha256_file(out)}", flush=True)
    print("SERVER_STAGE=PASS", flush=True)
    print("NEXT=RUN_FROZEN_DEEPSEEK_JUDGE_ON_WINDOWS", flush=True)
    print("============================================================", flush=True)


def build_parser():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    t = sp.add_parser("train")
    t.add_argument("--arm", required=True, choices=["C1", "C2", "C3"])
    t.add_argument("--c0", required=True)
    t.add_argument("--targets-dir", required=True)
    t.add_argument("--diagnostic-dir", required=True)
    t.add_argument("--train", required=True)
    t.add_argument("--run-dir", required=True)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--grad-accum", type=int, default=4)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--warmup-ratio", type=float, default=0.03)
    t.add_argument("--max-grad-norm", type=float, default=1.0)
    t.add_argument("--max-length", type=int, default=1024)
    t.add_argument("--seed", type=int, default=20260820)
    t.add_argument("--report-every", type=int, default=10)

    ew = sp.add_parser("eval-worker")
    ew.add_argument("--arm", required=True)
    ew.add_argument("--model", required=True)
    ew.add_argument("--input", required=True)
    ew.add_argument("--output", required=True)
    ew.add_argument("--worker-id", type=int, required=True)
    ew.add_argument("--num-workers", type=int, required=True)

    em = sp.add_parser("eval-master")
    em.add_argument("--arm", required=True, choices=["C1", "C2", "C3"])
    em.add_argument("--model", required=True)
    em.add_argument("--c0", required=True)
    em.add_argument("--targets-dir", required=True)
    em.add_argument("--diagnostic-dir", required=True)
    em.add_argument("--eval-dir", required=True)
    em.add_argument("--num-workers", type=int, default=4)

    c = sp.add_parser("combine")
    c.add_argument("--run-root", required=True)
    c.add_argument("--c0-chemistry-summary", required=True)

    return ap


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.cmd == "train":
        train_main(args)
    elif args.cmd == "eval-worker":
        eval_worker_main(args)
    elif args.cmd == "eval-master":
        eval_master_main(args)
    elif args.cmd == "combine":
        combine_main(args)
