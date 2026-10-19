#!/usr/bin/env python3
"""Training, evaluation, comparison, and CLI for Clean-SFT parity v1."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import _clean_sft_parity_core_v1 as c
import _clean_sft_checkpoint_v2 as checkpoint_v2
import _clean_sft_parity_kc_v1 as kc
import _clean_sft_materialize_v2 as materialize_v2
import _clean_sft_train_contract_v1 as train_contract_v2
import _targeted_tensorboard_v1 as targeted_tb


def set_seeds(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2**32 - 1))
    except Exception:
        pass
    import torch
    torch.manual_seed(seed)
    try:
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)
    except Exception:
        pass


def optimizer_to(optimizer: Any, device: str) -> None:
    import torch
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def capture_rng() -> dict[str, Any]:
    import torch
    state: dict[str, Any] = {
        "python": random.getstate(), "torch": torch.get_rng_state()
    }
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except Exception:
        pass
    try:
        state["npu"] = torch.npu.get_rng_state_all()
    except Exception:
        pass
    return state


def restore_rng(state: dict[str, Any]) -> None:
    import torch
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "numpy" in state:
        import numpy as np
        np.random.set_state(state["numpy"])
    if "npu" in state:
        try:
            torch.npu.set_rng_state_all(state["npu"])
        except Exception:
            pass


def tokenize_training_rows(
    tokenizer: Any, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    c.require(
        tokenizer.eos_token_id is not None
        and tokenizer.pad_token_id is not None,
        "tokenizer EOS/PAD missing",
    )
    output = []
    for index, row in enumerate(rows):
        messages = row.get("messages")
        c.require(
            isinstance(messages, list)
            and len(messages) == 1
            and messages[0].get("role") == "user",
            f"USER_ONLY_MESSAGES_FAIL index={index}",
        )
        c.require(
            messages[0].get("content")
            == c.DIRECT_PROMPT.format(source=row["source"]),
            f"PROMPT_CONTENT_FAIL index={index}",
        )
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = tokenizer(
            rendered, add_special_tokens=False
        )["input_ids"]
        target_ids = tokenizer(
            row["target_translation"], add_special_tokens=False
        )["input_ids"]
        input_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
        labels = [-100] * len(prompt_ids) + target_ids + [
            tokenizer.eos_token_id
        ]
        c.require(
            len(input_ids) <= c.MAX_TOTAL_LENGTH,
            f"TRAIN_MAX_LENGTH_FAIL target_id={row['target_id']} "
            f"tokens={len(input_ids)}",
        )
        output.append({
            "input_ids": input_ids, "labels": labels,
            "target_id": row["target_id"],
            "response_tokens": len(target_ids) + 1,
        })
    return output


def collate_rows(
    tokenizer: Any, rows: list[dict[str, Any]], device: str
) -> dict[str, Any]:
    import torch
    width = max(len(x["input_ids"]) for x in rows)
    ids, labels, masks = [], [], []
    for row in rows:
        padding = width - len(row["input_ids"])
        ids.append(row["input_ids"] + [tokenizer.pad_token_id] * padding)
        labels.append(row["labels"] + [-100] * padding)
        masks.append([1] * len(row["input_ids"]) + [0] * padding)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(
            masks, dtype=torch.long, device=device
        ),
    }


def save_epoch_checkpoint(
    run_dir: Path, model: Any, tokenizer: Any, optimizer: Any,
    scheduler: Any, epoch: int, updates: int, run_contract_sha256: str,
) -> Path:
    """Transactionally commit all state needed to resume at the next epoch."""
    import torch
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    final = checkpoints / f"epoch_{epoch:03d}"
    if final.exists():
        meta = c.read_json(final / "meta.json")
        c.require(
            meta.get("completed_epoch") == epoch
            and meta.get("completed_updates") == updates
            and meta.get("run_contract_sha256") == run_contract_sha256,
            f"checkpoint collision {final}",
        )
        return final
    tmp = checkpoints / f"epoch_{epoch:03d}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "hf").mkdir(parents=True)
    old_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = True
    model.save_pretrained(tmp / "hf", safe_serialization=True)
    tokenizer.save_pretrained(tmp / "hf")
    model.config.use_cache = old_cache
    torch.save(
        {
            "completed_epoch": epoch, "completed_updates": updates,
            "state_format_version": "kc_sft_checkpoint_v2",
            "run_contract_sha256": run_contract_sha256,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "rng": capture_rng(),
        },
        tmp / "state.pt",
    )
    c.atomic_json(
        tmp / "meta.json",
        {
            "status": "PASS", "completed_epoch": epoch,
            "completed_updates": updates,
            "state_format_version": "kc_sft_checkpoint_v2",
            "run_contract_sha256": run_contract_sha256,
            "torch_version": torch.__version__,
            "torch_load_weights_only": False,
            "created": c.now(),
        },
    )
    os.replace(tmp, final)
    c.atomic_json(
        run_dir / "latest_resume.json",
        {
            "checkpoint": str(final), "completed_epoch": epoch,
            "completed_updates": updates,
            "run_contract_sha256": run_contract_sha256,
            "updated": c.now(),
        },
    )
    return final


def training_contract(
    arm: str, rows: list[dict[str, Any]], target_path: Path
) -> dict[str, Any]:
    return train_contract_v2.build(arm, rows, target_path)


def cmd_train(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as functional
    import torch_npu  # noqa: F401
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )
    arm, cfg = args.arm, kc.ARM_CONFIG[args.arm]
    c.require(
        args.tensorboard_write_every >= 1,
        "--tensorboard-write-every must be >= 1",
    )
    c.require(
        args.tensorboard or args.tensorboard_logdir is None,
        "--tensorboard-logdir requires --tensorboard",
    )
    target_path = (
        Path(args.targets) if args.targets else Path(cfg["targets"])
    )
    rows = c.read_jsonl(target_path)
    c.require(len(rows) == 5500, f"TRAIN_COUNT_FAIL {len(rows)}")
    contract = train_contract_v2.build(arm, rows, target_path)
    run_dir = (
        Path(args.run_dir) if args.run_dir
        else c.RUN_ROOT / arm / "train"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    final_hf, manifest_path = run_dir / "final_hf", run_dir / "manifest.json"
    run_contract_sha256 = kc.ensure_immutable_contract(
        run_dir / "run_contract.json", contract
    )
    if (
        manifest_path.is_file() and final_hf.is_dir()
        and c.read_json(manifest_path).get("status") == "PASS"
    ):
        print(f"TRAIN_ALREADY_PASS arm={arm}")
        return
    device = args.device
    torch.npu.set_device(device)
    set_seeds(c.SEED)
    tokenizer = AutoTokenizer.from_pretrained(c.C0)
    tokenized = tokenize_training_rows(tokenizer, rows)
    micro_per_epoch = math.ceil(len(rows) / c.MICRO_BATCH)
    updates_per_epoch = math.ceil(micro_per_epoch / c.GRAD_ACCUM)
    total_updates = updates_per_epoch * c.EPOCHS
    c.require(
        (updates_per_epoch, total_updates) == (344, 1032),
        f"UPDATE_BUDGET_FAIL {(updates_per_epoch, total_updates)}",
    )
    warmup_steps = int(total_updates * c.WARMUP_RATIO)
    latest = run_dir / "latest_resume.json"
    kc.resolve_latest_checkpoint(run_dir, run_contract_sha256)
    saved, start_epoch, completed_updates, model_source = None, 1, 0, c.C0
    if latest.is_file():
        pointer = c.read_json(latest)
        c.require(
            pointer.get("run_contract_sha256") == run_contract_sha256,
            "RESUME_CONTRACT_MISMATCH before state load",
        )
        checkpoint = Path(pointer["checkpoint"])
        c.require(
            checkpoint.is_dir()
            and (checkpoint / "state.pt").is_file()
            and (checkpoint / "hf").is_dir(),
            "resume checkpoint incomplete",
        )
        meta = c.read_json(checkpoint / "meta.json")
        saved = checkpoint_v2.torch_load_compat(checkpoint / "state.pt")
        saved_epoch, saved_updates = checkpoint_v2.validate_pointer_meta_state(
            pointer, meta, saved, run_contract_sha256
        )
        start_epoch = saved_epoch + 1
        completed_updates = saved_updates
        model_source = checkpoint / "hf"
    model = AutoModelForCausalLM.from_pretrained(
        model_source, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    model.config.use_cache = False
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"],
        weight_decay=c.WEIGHT_DECAY, foreach=False,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    if saved is not None:
        optimizer.load_state_dict(saved["optimizer"])
        optimizer_to(optimizer, device)
        scheduler.load_state_dict(saved["scheduler"])
        restore_rng(saved["rng"])
    metrics_path = run_dir / "update_metrics.jsonl"
    # Roll the metric ledger back to the same durable boundary as the
    # transactional model/optimizer checkpoint before replaying an epoch.
    if metrics_path.is_file():
        committed = checkpoint_v2.committed_metric_prefix(
            c.read_jsonl(metrics_path), completed_updates,
            max(start_epoch - 1, 0),
        )
        c.atomic_jsonl(metrics_path, committed)
    else:
        c.require(
            completed_updates == 0,
            "resume checkpoint exists without committed update metrics",
        )
    tensorboard_logdir = (
        Path(args.tensorboard_logdir).resolve()
        if args.tensorboard_logdir else run_dir / "tensorboard"
    )
    tensorboard = targeted_tb.NativeTensorBoard(
        enabled=args.tensorboard, log_dir=tensorboard_logdir,
        run_id=f"{arm}:{run_contract_sha256}",
        durable_step=completed_updates, rank=0,
        write_every=args.tensorboard_write_every,
    )
    started, session_start_updates = time.time(), completed_updates
    epoch_token_totals = {}
    for epoch in range(start_epoch, c.EPOCHS + 1):
        order = c.epoch_order(len(rows), epoch)
        c.require(
            c.sha256_json(order)
            == contract["epoch_order_sha256"][str(epoch)],
            "epoch order hash mismatch",
        )
        micros = [
            order[i:i + c.MICRO_BATCH]
            for i in range(0, len(order), c.MICRO_BATCH)
        ]
        optimizer.zero_grad(set_to_none=True)
        group_tokens = group_micros = 0
        group_ce_sum = 0.0
        epoch_tokens = 0
        group_started = time.time()
        for micro_index, indices in enumerate(micros, 1):
            batch = collate_rows(
                tokenizer, [tokenized[i] for i in indices], device
            )
            labels = batch.pop("labels")
            output = model(**batch, use_cache=False)
            shift_logits = output.logits[:, :-1, :].float().contiguous()
            shift_labels = labels[:, 1:].contiguous()
            tokens = int(shift_labels.ne(-100).sum().item())
            c.require(tokens > 0, f"zero supervised tokens micro={micro_index}")
            ce_sum = functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1), ignore_index=-100, reduction="sum",
            )
            c.require(
                bool(torch.isfinite(ce_sum).item()),
                f"non-finite CE micro={micro_index}",
            )
            ce_sum.backward()
            group_tokens += tokens
            epoch_tokens += tokens
            group_ce_sum += float(ce_sum.detach().float().cpu())
            group_micros += 1
            boundary = (
                group_micros == c.GRAD_ACCUM
                or micro_index == len(micros)
            )
            if not boundary:
                continue
            c.require(group_tokens > 0, "update group has zero tokens")
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(group_tokens)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), c.MAX_GRAD_NORM
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            completed_updates += 1
            update_seconds = time.time() - group_started
            grad_norm_value = float(grad_norm)
            record = {
                "status": "PASS", "arm": arm, "epoch": epoch,
                "optimizer_update": completed_updates,
                "microbatches": group_micros,
                "group_response_tokens": group_tokens,
                "group_ce_sum": group_ce_sum,
                "group_token_mean_ce": group_ce_sum / group_tokens,
                "grad_norm_before_clip": grad_norm_value,
                "max_grad_norm": c.MAX_GRAD_NORM,
                "lr_after_step": scheduler.get_last_lr()[0],
                "step_seconds": update_seconds,
                "teacher_calls_during_training": 0,
                "elapsed_seconds": time.time() - started,
                "timestamp": c.now(),
            }
            c.append_jsonl(metrics_path, record)
            tensorboard.log_sft_update(
                step=completed_updates,
                ce_per_response_token=record["group_token_mean_ce"],
                lr=record["lr_after_step"],
                grad_norm_preclip=grad_norm_value,
                max_grad_norm=c.MAX_GRAD_NORM,
                response_tokens=group_tokens,
                step_seconds=update_seconds,
            )
            if (
                completed_updates % args.report_every == 0
                or completed_updates == total_updates
            ):
                fresh = max(completed_updates - session_start_updates, 1)
                rate = fresh / max(time.time() - started, 1e-9)
                eta = (
                    (total_updates - completed_updates) / rate
                    if rate else None
                )
                print(
                    f"{c.now()} arm={arm} epoch={epoch}/{c.EPOCHS} "
                    f"update={completed_updates}/{total_updates} "
                    f"tokens={group_tokens} "
                    f"ce={record['group_token_mean_ce']:.6f} "
                    f"grad={grad_norm_value:.4f} "
                    f"lr={record['lr_after_step']:.8g} "
                    f"ETA={'?' if eta is None else f'{eta/60:.1f}m'}",
                    flush=True,
                )
                c.atomic_json(
                    run_dir / "progress.json",
                    {
                        "status": "RUNNING", "arm": arm, "epoch": epoch,
                        "completed_updates": completed_updates,
                        "total_updates": total_updates,
                        "last_group_response_tokens": group_tokens,
                        "updated": c.now(),
                    },
                )
            group_tokens = group_micros = 0
            group_ce_sum = 0.0
            group_started = time.time()
        epoch_token_totals[str(epoch)] = epoch_tokens
        checkpoint_v2.committed_metric_prefix(
            c.read_jsonl(metrics_path), completed_updates, epoch
        )
        checkpoint = save_epoch_checkpoint(
            run_dir, model, tokenizer, optimizer, scheduler,
            epoch, completed_updates, run_contract_sha256,
        )
        print(
            f"EPOCH_CHECKPOINT_PASS epoch={epoch} "
            f"updates={completed_updates} path={checkpoint}"
        )
        tensorboard.flush()
    c.require(
        completed_updates == total_updates,
        f"TRAIN_UPDATE_COUNT_FAIL {completed_updates} != {total_updates}",
    )
    final_tmp = run_dir / "final_hf.tmp"
    if final_tmp.exists():
        shutil.rmtree(final_tmp)
    final_tmp.mkdir(parents=True)
    model.config.use_cache = True
    model.save_pretrained(final_tmp, safe_serialization=True)
    tokenizer.save_pretrained(final_tmp)
    if final_hf.exists():
        shutil.rmtree(final_hf)
    os.replace(final_tmp, final_hf)
    all_metrics = c.read_jsonl(metrics_path)
    c.require(
        len(all_metrics) == total_updates,
        f"UPDATE_METRIC_COUNT_FAIL {len(all_metrics)} != {total_updates}",
    )
    c.require(
        [int(x["optimizer_update"]) for x in all_metrics]
        == list(range(1, total_updates + 1)),
        "update metric ledger is not a unique contiguous sequence",
    )
    token_counts = [int(x["group_response_tokens"]) for x in all_metrics]
    epoch_token_totals = {
        str(epoch): sum(
            int(x["group_response_tokens"])
            for x in all_metrics if int(x["epoch"]) == epoch
        )
        for epoch in range(1, c.EPOCHS + 1)
    }
    manifest = {
        "status": "PASS", "arm": arm,
        "run_contract_sha256": run_contract_sha256,
        "scientific_class": (
            "FORMAL_KC_SFT_MATCHED_PROMPT_BASELINE"
            if cfg["formal_matched_baseline"]
            else "KC_SFT_SELECTIVE_QUALITY_CONTROL"
            if cfg["dataset"] == "selective_clean"
            else "OLD_TARGET_MATCHED_TRAINER_CONTROL"
        ),
        "claim_boundary": (
            "Offline teacher-forced sequence CE; not equivalent to OPD "
            "on-policy forward-KL or compute cost."
        ),
        "init_model": str(c.C0),
        "teacher_model_for_target_materialization": str(c.TEACHER),
        "train_file": str(target_path),
        "train_sha256": c.sha256_file(target_path),
        "rows": len(rows), "contract": contract,
        "training": {
            "epochs": c.EPOCHS, "micro_batch": c.MICRO_BATCH,
            "gradient_accumulation": c.GRAD_ACCUM,
            "effective_batch_nominal": 16,
            "updates_per_epoch": updates_per_epoch,
            "total_updates": total_updates,
            "optimizer": "AdamW(foreach=False)", "lr": cfg["lr"],
            "weight_decay": c.WEIGHT_DECAY,
            "warmup_ratio": c.WARMUP_RATIO,
            "warmup_steps": warmup_steps, "scheduler": "linear",
            "max_grad_norm": c.MAX_GRAD_NORM, "precision": "bfloat16",
            "ce_compute_dtype": "float32",
            "full_parameter_student_update": True, "seed": c.SEED,
            "epoch_shuffle_seed_rule": (
                "random.Random(seed + epoch - 1).shuffle(order)"
            ),
            "epoch_order_sha256": contract["epoch_order_sha256"],
            "max_total_length": c.MAX_TOTAL_LENGTH,
            "objective": "response-only teacher-forced target cross-entropy",
            "loss_reduction": (
                "global valid response-token mean across accumulated "
                "microbatches"
            ),
            "gradient_order": (
                "FP32 CE sum backward -> divide gradients by update-group response "
                "tokens -> clip -> optimizer -> scheduler"
            ),
            "response_tokens_per_epoch": epoch_token_totals,
            "update_group_tokens": {
                "count": len(token_counts), "min": min(token_counts),
                "max": max(token_counts),
                "mean": sum(token_counts) / len(token_counts),
                "sum": sum(token_counts),
            },
            "update_metrics_sha256": c.sha256_file(metrics_path),
            "teacher_calls_during_training": 0,
        },
        "compute_accounting": {
            "training_elapsed_seconds": time.time() - started,
            "target_materialization_teacher_cost": (
                "see clean target manifest"
                if cfg["dataset"] != "old_target"
                else "historical artifact; not recomputed"
            ),
            "same_optimizer_steps_does_not_imply_same_compute_as_opd": True,
        },
        "tensorboard": {
            "enabled": bool(args.tensorboard),
            "logdir": str(tensorboard_logdir) if args.tensorboard else None,
            "write_every_optimizer_updates": args.tensorboard_write_every,
            "step_rule": "actual optimizer update",
            "resume_policy": "purge events after durable checkpoint step",
        },
        "final_model": str(final_hf), "created": c.now(),
    }
    c.atomic_json(manifest_path, manifest)
    c.atomic_json(
        run_dir / "progress.json",
        {
            "status": "PASS", "arm": arm,
            "completed_updates": total_updates,
            "total_updates": total_updates, "updated": c.now(),
        },
    )
    tensorboard.close()
    print(f"TRAIN_PASS arm={arm} updates={total_updates} final={final_hf}")


def generate_translation(
    model: Any, tokenizer: Any, source: str, device: str
) -> str:
    import torch
    prompt = c.DIRECT_PROMPT.format(source=source)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    batch = tokenizer(
        rendered, return_tensors="pt", add_special_tokens=False
    )
    batch = {key: value.to(device) for key, value in batch.items()}
    width = batch["input_ids"].shape[1]
    with torch.no_grad():
        sequence = model.generate(
            **batch, do_sample=False, max_new_tokens=512,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id, use_cache=True,
        )
    return tokenizer.decode(
        sequence[0, width:].detach().cpu().tolist(),
        skip_special_tokens=True,
    ).strip()


def pass_map(path: Path, key: str, contract_sha: str) -> dict[str, dict[str, Any]]:
    return kc.pass_map(path, key, contract_sha)


def metric_stats(
    rows: list[dict[str, Any]], key: str
) -> dict[str, Any]:
    hits = sum(bool(row[key]) for row in rows)
    return {
        "n": len(rows), "hits": hits,
        "accuracy": hits / len(rows) if rows else None,
    }


def score_targeted(rows: list[dict[str, Any]]) -> dict[str, Any]:
    chemistry = [
        row for row in rows if row["eval_domain"] == "chemistry"
    ]
    c.require(
        len(chemistry) == 1000,
        f"chem diagnostic rows={len(chemistry)}",
    )
    for row in chemistry:
        text, canonical = c.translation_of(row), row["canonical_en_target"]
        audit = c.audit_translation(text, canonical)
        row["strict_canonical_hit"] = c.strict_hit(canonical, text)
        row["ci_spacing_normalized_hit"] = (
            row["strict_canonical_hit"]
            or c.ci_spacing_hit(canonical, text)
        )
        non_hit_flags = [
            flag for flag in audit["flags"]
            if flag != "canonical_term_missing"
        ]
        row["meta_free"] = not non_hit_flags
        row["meta_free_target_hit"] = (
            row["strict_canonical_hit"] and row["meta_free"]
        )
        row["is_ci"] = c.is_ci_row(row)
    groups = {
        "all": chemistry,
        "uc": [x for x in chemistry if x["eval_split"] == "uc"],
        "uw": [x for x in chemistry if x["eval_split"] == "uw"],
        "ci": [x for x in chemistry if x["is_ci"]],
        "non_ci": [x for x in chemistry if not x["is_ci"]],
        "uc_non_ci": [
            x for x in chemistry
            if x["eval_split"] == "uc" and not x["is_ci"]
        ],
        "uw_non_ci": [
            x for x in chemistry
            if x["eval_split"] == "uw" and not x["is_ci"]
        ],
    }
    return {
        name: {
            "strict_canonical": metric_stats(xs, "strict_canonical_hit"),
            "ci_spacing_normalized": metric_stats(
                xs, "ci_spacing_normalized_hit"
            ),
            "meta_free_target_hit": metric_stats(
                xs, "meta_free_target_hit"
            ),
            "meta_free_outputs": metric_stats(xs, "meta_free"),
        }
        for name, xs in groups.items()
    }


def evaluation_contract(arm: str, model_path: Path) -> dict[str, Any]:
    c.require(arm in kc.ARM_CONFIG, f"unknown arm {arm}")
    return {
        "contract_version": "kc_sft_eval_contract_v2",
        "evaluation_version": "chemistry_parity_eval_v2",
        "arm": arm,
        "model": kc.model_identity(model_path),
        "inputs": {
            "targeted": {"path": str(c.DIAG), "sha256": c.sha256_file(c.DIAG)},
            "general": {
                name: {"path": str(path), "sha256": c.sha256_file(path)}
                for name, path in c.GENERAL_EVAL.items()
            },
        },
        "generation": {
            "prompt_sha256": c.EXPECTED_DIRECT_PROMPT_SHA,
            "enable_thinking": False, "do_sample": False,
            "max_new_tokens": 512,
        },
        "scoring": {
            "targeted": "strict_case_insensitive_canonical_substring_v1",
            "general": "sacrebleu_corpus_bleu_chrf_v1",
        },
    }


def cmd_eval(args: argparse.Namespace) -> None:
    import torch
    import torch_npu  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_path = (
        Path(args.model) if args.model
        else c.RUN_ROOT / args.arm / "train/final_hf"
    )
    out_dir = (
        Path(args.out_dir) if args.out_dir
        else c.RUN_ROOT / args.arm / "eval"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    c.require(model_path.is_dir(), f"model missing: {model_path}")
    eval_contract = evaluation_contract(args.arm, model_path)
    eval_contract_sha256 = kc.ensure_immutable_contract(
        out_dir / "eval_contract.json", eval_contract
    )
    frozen = c.frozen_input_checks()
    c.require(
        not any(x.get("status") == "FAIL" for x in frozen.values()),
        "frozen eval input check failed",
    )
    device = args.device
    torch.npu.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device).eval()
    started = time.time()
    if args.scope in ("all", "targeted"):
        source_rows = c.read_jsonl(c.DIAG)
        pred_path = out_dir / "targeted_diagnostic2000_translations.jsonl"
        done = pass_map(pred_path, "eval_id", eval_contract_sha256)
        for row in source_rows:
            eval_id = str(row["eval_id"])
            if eval_id in done:
                continue
            hypothesis = generate_translation(
                model, tokenizer, row["src_text"], device
            )
            record = {
                "status": "PASS", "arm": args.arm,
                "eval_id": row["eval_id"],
                "eval_contract_sha256": eval_contract_sha256,
                "eval_domain": row["eval_domain"],
                "eval_split": row["eval_split"], "job_id": row["job_id"],
                "entity_key": row.get("entity_key"),
                "src_term": row["src_term"],
                "definition": row.get("definition"),
                "src_text": row["src_text"],
                "canonical_en_target": row.get("canonical_en_target"),
                "model_translation": hypothesis,
                "prompt_sha256": c.EXPECTED_DIRECT_PROMPT_SHA,
                "enable_thinking": False, "do_sample": False,
                "max_new_tokens": 512, "timestamp": c.now(),
            }
            c.append_jsonl(pred_path, record)
            done[eval_id] = record
            if len(done) % args.report_every == 0:
                print(
                    f"targeted done={len(done)}/{len(source_rows)}",
                    flush=True,
                )
        ordered = [done[str(row["eval_id"])] for row in source_rows]
        summary = score_targeted(ordered)
        chemistry = [
            row for row in ordered if row["eval_domain"] == "chemistry"
        ]
        c.atomic_jsonl(
            out_dir / "chemistry_diagnostic1000_scored.jsonl",
            chemistry,
        )
        c.atomic_json(
            out_dir / "chemistry_summary.json",
            {
                "status": "PASS",
                "eval_contract_sha256": eval_contract_sha256,
                "primary_metric": (
                    "case-insensitive exact canonical substring"
                ),
                "supplemental_metrics_do_not_replace_primary": True,
                "slices": summary, "created": c.now(),
            },
        )
        c.atomic_jsonl(
            out_dir / "semantic_lexical_review_template.jsonl",
            [{
                "eval_id": row["eval_id"], "split": row["eval_split"],
                "src_term": row["src_term"], "source": row["src_text"],
                "canonical_en_target": row["canonical_en_target"],
                "model_translation": row["model_translation"],
                "semantic_lexical_correct": None,
                "reviewer_notes": "",
            } for row in chemistry],
        )
        print("TARGETED_EVAL_PASS")
    if args.scope in ("all", "general"):
        import sacrebleu
        dataset_metrics = {}
        for name, path in c.GENERAL_EVAL.items():
            rows = c.read_jsonl(path)
            pred_path = out_dir / f"{name.lower()}_predictions.jsonl"
            done = pass_map(pred_path, "row_id", eval_contract_sha256)
            for index, row in enumerate(rows):
                row_id = f"{name}:{index}"
                if row_id in done:
                    continue
                hypothesis = generate_translation(
                    model, tokenizer, row["source"], device
                )
                record = {
                    "status": "PASS", "arm": args.arm,
                    "dataset": name, "row_id": row_id,
                    "eval_contract_sha256": eval_contract_sha256,
                    "source": row["source"],
                    "reference": row["reference"],
                    "student_translation": hypothesis,
                    "prompt_sha256": c.EXPECTED_DIRECT_PROMPT_SHA,
                    "enable_thinking": False, "do_sample": False,
                    "max_new_tokens": 512, "timestamp": c.now(),
                }
                c.append_jsonl(pred_path, record)
                done[row_id] = record
                if len(done) % args.report_every == 0:
                    print(
                        f"general {name} done={len(done)}/{len(rows)}",
                        flush=True,
                    )
            ordered = [
                done[f"{name}:{index}"] for index in range(len(rows))
            ]
            references = [x["reference"] for x in ordered]
            hypotheses = [x["student_translation"] for x in ordered]
            dataset_metrics[name] = {
                "rows": len(rows),
                "BLEU": float(
                    sacrebleu.corpus_bleu(
                        hypotheses, [references]
                    ).score
                ),
                "chrF": float(
                    sacrebleu.corpus_chrf(
                        hypotheses, [references]
                    ).score
                ),
                "predictions_sha256": c.sha256_file(pred_path),
            }
        macro_bleu = (
            sum(x["BLEU"] for x in dataset_metrics.values()) / 3
        )
        macro_chrf = (
            sum(x["chrF"] for x in dataset_metrics.values()) / 3
        )
        c.atomic_json(
            out_dir / "general_summary.json",
            {
                "status": "PASS",
                "eval_contract_sha256": eval_contract_sha256,
                "generation": {
                    "prompt_sha256": c.EXPECTED_DIRECT_PROMPT_SHA,
                    "enable_thinking": False, "do_sample": False,
                    "max_new_tokens": 512,
                },
                "datasets": dataset_metrics, "macro_bleu": macro_bleu,
                "macro_chrf": macro_chrf,
                "delta_macro_bleu_vs_C0": (
                    macro_bleu - c.BASE_GENERAL_MACRO_BLEU
                ),
                "created": c.now(),
            },
        )
        print("GENERAL_EVAL_PASS")
    c.atomic_json(
        out_dir / "manifest.json",
        {
            "status": "PASS", "arm": args.arm,
            "model": str(model_path),
            "eval_contract_sha256": eval_contract_sha256,
            "eval_contract": eval_contract,
            "model_config_sha256": c.sha256_file(
                model_path / "config.json"
            ),
            "scope": args.scope, "frozen_inputs": frozen,
            "elapsed_seconds": time.time() - started,
            "created": c.now(),
        },
    )


def keyed_chem(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for row in c.read_jsonl(path):
        if row.get("eval_domain") != "chemistry":
            continue
        eval_id = str(row["eval_id"])
        canonical, text = row["canonical_en_target"], c.translation_of(row)
        result[eval_id] = {
            **row, "_translation": text,
            "_strict_hit": c.strict_hit(canonical, text),
            "_ci_hit": (
                c.strict_hit(canonical, text)
                or c.ci_spacing_hit(canonical, text)
            ),
        }
    c.require(
        len(result) == 1000,
        f"expected chemistry1000 at {path}, got {len(result)}",
    )
    return result


def paired_counts(
    base: dict[str, dict[str, Any]],
    other: dict[str, dict[str, Any]],
    ids: list[str],
) -> dict[str, int]:
    gain = sum(
        not base[i]["_strict_hit"] and other[i]["_strict_hit"]
        for i in ids
    )
    loss = sum(
        base[i]["_strict_hit"] and not other[i]["_strict_hit"]
        for i in ids
    )
    return {
        "gain": gain, "loss": loss, "net": gain - loss,
        "retained": sum(
            base[i]["_strict_hit"] and other[i]["_strict_hit"]
            for i in ids
        ),
        "both_miss": sum(
            not base[i]["_strict_hit"] and not other[i]["_strict_hit"]
            for i in ids
        ),
    }


def cmd_compare(args: argparse.Namespace) -> None:
    cfg = kc.ARM_CONFIG[args.arm]
    sft_path = (
        Path(args.sft_predictions) if args.sft_predictions
        else c.RUN_ROOT / args.arm / "eval/targeted_diagnostic2000_translations.jsonl"
    )
    opd_path = (
        Path(args.opd_predictions) if args.opd_predictions
        else c.OPD_RUNS[cfg["opd"]]
        / "O2/targeted_eval/targeted_diagnostic2000_translations.jsonl"
    )
    c0_path, old_c2_path = (
        Path(args.c0_predictions), Path(args.old_c2_predictions)
    )
    sft, opd, c0, old_c2 = map(
        keyed_chem, (sft_path, opd_path, c0_path, old_c2_path)
    )
    ids = list(c0)
    c.require(
        set(ids) == set(sft) == set(opd) == set(old_c2),
        "paired eval_id sets differ",
    )
    for eval_id in ids:
        c.require(
            c0[eval_id]["src_text"]
            == sft[eval_id]["src_text"]
            == opd[eval_id]["src_text"],
            f"source mismatch {eval_id}",
        )
        c.require(
            c0[eval_id]["canonical_en_target"]
            == sft[eval_id]["canonical_en_target"]
            == opd[eval_id]["canonical_en_target"],
            f"target mismatch {eval_id}",
        )
    groups = {
        "all": ids,
        "uc": [i for i in ids if c0[i]["eval_split"] == "uc"],
        "uw": [i for i in ids if c0[i]["eval_split"] == "uw"],
        "non_ci": [i for i in ids if not c.is_ci_row(c0[i])],
        "uc_non_ci": [
            i for i in ids
            if c0[i]["eval_split"] == "uc" and not c.is_ci_row(c0[i])
        ],
        "uw_non_ci": [
            i for i in ids
            if c0[i]["eval_split"] == "uw" and not c.is_ci_row(c0[i])
        ],
    }
    table = {}
    for name, subset in groups.items():
        table[name] = {
            "n": len(subset),
            "C0": sum(c0[i]["_strict_hit"] for i in subset) / len(subset),
            "OLD_C2": (
                sum(old_c2[i]["_strict_hit"] for i in subset) / len(subset)
            ),
            "SFT": (
                sum(sft[i]["_strict_hit"] for i in subset) / len(subset)
            ),
            "OPD": (
                sum(opd[i]["_strict_hit"] for i in subset) / len(subset)
            ),
            "SFT_vs_C0": paired_counts(c0, sft, subset),
            "OPD_vs_C0": paired_counts(c0, opd, subset),
            "SFT_vs_OPD": paired_counts(opd, sft, subset),
        }
    disagreements = []
    for eval_id in ids:
        if sft[eval_id]["_strict_hit"] == opd[eval_id]["_strict_hit"]:
            continue
        disagreements.append({
            "eval_id": eval_id, "split": c0[eval_id]["eval_split"],
            "source": c0[eval_id]["src_text"],
            "src_term": c0[eval_id]["src_term"],
            "canonical_en_target": c0[eval_id]["canonical_en_target"],
            "old_c2_hit": old_c2[eval_id]["_strict_hit"],
            "sft_hit": sft[eval_id]["_strict_hit"],
            "opd_hit": opd[eval_id]["_strict_hit"],
            "c0_translation": c0[eval_id]["_translation"],
            "old_c2_translation": old_c2[eval_id]["_translation"],
            "sft_translation": sft[eval_id]["_translation"],
            "opd_translation": opd[eval_id]["_translation"],
            "disagreement": (
                "SFT_ONLY" if sft[eval_id]["_strict_hit"] else "OPD_ONLY"
            ),
        })
    out = (
        Path(args.out_dir) if args.out_dir
        else c.RUN_ROOT / args.arm / "compare"
    )
    out.mkdir(parents=True, exist_ok=True)
    historical_disagreements = []
    for eval_id in ids:
        if old_c2[eval_id]["_strict_hit"] == opd[eval_id]["_strict_hit"]:
            continue
        historical_disagreements.append({
            "eval_id": eval_id,
            "split": c0[eval_id]["eval_split"],
            "source": c0[eval_id]["src_text"],
            "src_term": c0[eval_id]["src_term"],
            "canonical_en_target": c0[eval_id]["canonical_en_target"],
            "historical_disagreement": (
                "HISTORICAL_SFT_ONLY"
                if old_c2[eval_id]["_strict_hit"]
                else "HISTORICAL_OPD_ONLY"
            ),
            "old_c2_hit": old_c2[eval_id]["_strict_hit"],
            "opd_hit": opd[eval_id]["_strict_hit"],
            "current_sft_hit": sft[eval_id]["_strict_hit"],
            "old_c2_translation": old_c2[eval_id]["_translation"],
            "opd_translation": opd[eval_id]["_translation"],
            "current_sft_translation": sft[eval_id]["_translation"],
        })
    c.atomic_jsonl(
        out / "historical_disagreement_recheck.jsonl",
        historical_disagreements,
    )
    c.atomic_jsonl(out / "strict_disagreements.jsonl", disagreements)
    blinded, key = [], []
    for eval_id in ids:
        candidates = [
            ("C0", c0[eval_id]["_translation"]),
            ("OLD_C2", old_c2[eval_id]["_translation"]),
            (args.arm, sft[eval_id]["_translation"]),
            (f"OPD_{cfg['opd']}", opd[eval_id]["_translation"]),
        ]
        seed = int(
            hashlib.sha256(
                ("semantic-blind-v1|" + eval_id).encode()
            ).hexdigest()[:16],
            16,
        )
        random.Random(seed).shuffle(candidates)
        for position, (label, text) in enumerate(candidates):
            blind_id = hashlib.sha256(
                f"semantic-blind-v1|{eval_id}|{position}".encode()
            ).hexdigest()[:20]
            blinded.append({
                "blind_id": blind_id, "eval_id": eval_id,
                "split": c0[eval_id]["eval_split"],
                "source": c0[eval_id]["src_text"],
                "src_term": c0[eval_id]["src_term"],
                "canonical_en_target": c0[eval_id]["canonical_en_target"],
                "translation": text, "semantic_lexical_correct": None,
                "faithful_complete_translation": None,
                "quality_issue": "", "reviewer_notes": "",
            })
            key.append({
                "blind_id": blind_id, "arm": label, "eval_id": eval_id
            })
    c.atomic_jsonl(out / "semantic_blind_review.jsonl", blinded)
    c.atomic_jsonl(out / "semantic_blind_key.jsonl", key)
    general = {}
    general_paths = {
        "SFT": (
            Path(args.sft_general) if args.sft_general
            else c.RUN_ROOT / args.arm / "eval/general_summary.json"
        ),
        "OPD": (
            Path(args.opd_general) if args.opd_general
            else c.OPD_RUNS[cfg["opd"]] / "general_bleu_opd/O2/summary.json"
        ),
    }
    for label, path in general_paths.items():
        general[label] = (
            c.read_json(path) if path.is_file()
            else {"status": "NOT_VERIFIED", "path": str(path)}
        )
    semantic = {
        "status": "PENDING_BLIND_HUMAN_REVIEW",
        "input": str(out / "semantic_blind_review.jsonl"),
        "key": str(out / "semantic_blind_key.jsonl"),
    }
    if args.semantic_judgments:
        judgment_path = Path(args.semantic_judgments)
        judgments = {
            str(x["blind_id"]): x for x in c.read_jsonl(judgment_path)
        }
        key_map = {x["blind_id"]: x["arm"] for x in key}
        c.require(
            set(judgments) == set(key_map),
            "semantic judgment blind_id set mismatch",
        )
        by_arm: dict[str, list[bool]] = {}
        for blind_id, row in judgments.items():
            c.require(
                isinstance(row.get("semantic_lexical_correct"), bool),
                f"missing boolean semantic judgment {blind_id}",
            )
            by_arm.setdefault(key_map[blind_id], []).append(
                row["semantic_lexical_correct"]
            )
        semantic = {
            "status": "PASS",
            "summary": {
                arm: {
                    "n": len(values), "hits": sum(values),
                    "accuracy": sum(values) / len(values),
                }
                for arm, values in by_arm.items()
            },
            "judgments_sha256": c.sha256_file(judgment_path),
        }
    summary = {
        "status": "PASS", "arm": args.arm,
        "primary_metric": "strict case-insensitive canonical substring",
        "strict_table": table,
        "disagreements": {
            "total": len(disagreements),
            "sft_only": sum(
                x["disagreement"] == "SFT_ONLY" for x in disagreements
            ),
            "opd_only": sum(
                x["disagreement"] == "OPD_ONLY" for x in disagreements
            ),
            "artifact": str(out / "strict_disagreements.jsonl"),
        },
        "historical_disagreement_recheck": {
            "total": len(historical_disagreements),
            "historical_sft_only": sum(
                x["historical_disagreement"] == "HISTORICAL_SFT_ONLY"
                for x in historical_disagreements
            ),
            "historical_opd_only": sum(
                x["historical_disagreement"] == "HISTORICAL_OPD_ONLY"
                for x in historical_disagreements
            ),
            "artifact": str(
                out / "historical_disagreement_recheck.jsonl"
            ),
        },
        "general_mt": general, "semantic_lexical": semantic,
        "claim_boundary": (
            "SFT and OPD share base configuration but differ in objective, "
            "prefixes, target/rollout tokens, Teacher calls, and compute."
        ),
        "created": c.now(),
    }
    c.atomic_json(out / "summary.json", summary)
    print("PAIRED_COMPARISON_PASS")
    print(json.dumps(summary["strict_table"], ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean Chemistry SFT parity controls"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--json-out")
    materialize = sub.add_parser("materialize")
    materialize.add_argument("--audit-only", action="store_true")
    materialize.add_argument("--dataset", required=True, choices=tuple(kc.DATASETS))
    materialize.add_argument("--finalize", action="store_true")
    materialize.add_argument("--manual-review-jsonl")
    materialize.add_argument("--device", default="npu:0")
    materialize.add_argument(
        "--max-retries", type=int, default=3, choices=(1, 2, 3)
    )
    materialize.add_argument("--max-new-tokens", type=int, default=384)
    materialize.add_argument("--review-sample-size", type=int, default=100)
    train = sub.add_parser("train")
    train.add_argument(
        "--arm", required=True, choices=tuple(kc.ARM_CONFIG)
    )
    train.add_argument("--targets")
    train.add_argument("--run-dir")
    train.add_argument("--device", default="npu:0")
    train.add_argument("--report-every", type=int, default=10)
    train.add_argument("--tensorboard", action="store_true")
    train.add_argument("--tensorboard-logdir")
    train.add_argument(
        "--tensorboard-write-every", type=int, default=1
    )
    evaluate = sub.add_parser("eval")
    evaluate.add_argument(
        "--arm", required=True, choices=tuple(kc.ARM_CONFIG)
    )
    evaluate.add_argument("--model")
    evaluate.add_argument("--out-dir")
    evaluate.add_argument("--device", default="npu:0")
    evaluate.add_argument(
        "--scope", choices=("all", "targeted", "general"), default="all"
    )
    evaluate.add_argument("--report-every", type=int, default=50)
    compare = sub.add_parser("compare")
    compare.add_argument(
        "--arm", required=True, choices=tuple(kc.ARM_CONFIG)
    )
    compare.add_argument("--sft-predictions")
    compare.add_argument("--opd-predictions")
    compare.add_argument(
        "--c0-predictions",
        default=str(
            c.ROOT / "runs/targeted/c0_targeted_diagnostic1000_20260910/"
            "c0_targeted_diagnostic2000_translations.jsonl"
        ),
    )
    compare.add_argument(
        "--old-c2-predictions",
        default=str(
            c.ROOT / "runs/targeted/wa_sft_positive_control_c123_20260910/"
            "C2/eval/c2_diagnostic2000_translations.jsonl"
        ),
    )
    compare.add_argument("--sft-general")
    compare.add_argument("--opd-general")
    compare.add_argument("--semantic-judgments")
    compare.add_argument("--out-dir")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    commands = {
        "preflight": kc.cmd_preflight,
        "materialize": materialize_v2.cmd_materialize,
        "train": cmd_train,
        "eval": cmd_eval,
        "compare": cmd_compare,
    }
    commands[args.command](args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"{c.now()} INTERRUPTED", file=sys.stderr)
        raise
