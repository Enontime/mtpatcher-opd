#!/usr/bin/env python3
# coding: utf-8

import argparse
import importlib.util
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_trainer(path: Path):
    script_dir = str(path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    spec = importlib.util.spec_from_file_location(
        "ecropd_frozen_trainer",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import trainer: {path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def teachability_mass(
    student,
    teacher,
    input_ids,
    top_k,
):
    mask = torch.ones_like(input_ids)

    with torch.no_grad():
        s_out = student(
            input_ids=input_ids,
            attention_mask=mask,
            use_cache=False,
        )

        t_out = teacher(
            input_ids=input_ids,
            attention_mask=mask,
            use_cache=False,
        )

        # Exact convention corresponding to the first resumed
        # token in the frozen EC trainer:
        # prefix length L -> logits at L-1 predict token L.
        s_logits = s_out.logits[:, -1, :].float()
        t_logits = t_out.logits[:, -1, :].float()

        s_top = torch.topk(
            s_logits,
            k=top_k,
            dim=-1,
        ).indices

        t_prob = F.softmax(
            t_logits,
            dim=-1,
        )

        mass = torch.gather(
            t_prob,
            dim=-1,
            index=s_top,
        ).sum(dim=-1)

    return float(mass.item())


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--trainer", required=True, type=Path)
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--source-map", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)

    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--epoch", type=int, default=1)

    ap.add_argument("--top-k", type=int, default=16)

    ap.add_argument("--max-prompt-length", type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=256)

    ap.add_argument(
        "--feedback-max-prompt-tokens",
        type=int,
        default=1536,
    )
    ap.add_argument(
        "--feedback-max-new-tokens",
        type=int,
        default=768,
    )

    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k-rollout", type=int, default=20)

    args = ap.parse_args()

    trainer = load_trainer(args.trainer)

    device = torch.device("npu:0")
    torch.npu.set_device(0)

    print("=" * 80)
    print("H1 REPAIR-INDUCED TEACHABILITY PROBE")
    print("=" * 80)
    print("QUESTION = Does localized semantic repair increase local teachability?")
    print("TRAINING = False")
    print("STUDENT =", args.student)
    print("TEACHER =", args.teacher)
    print("TRAIN =", args.train)
    print("LIMIT =", args.limit)
    print("SEED =", args.seed)
    print("EPOCH =", args.epoch)
    print("TEACHABILITY_TOPK =", args.top_k)
    print()

    student_cfg = AutoConfig.from_pretrained(
        args.student,
        local_files_only=True,
    )
    teacher_cfg = AutoConfig.from_pretrained(
        args.teacher,
        local_files_only=True,
    )

    if student_cfg.vocab_size != teacher_cfg.vocab_size:
        raise RuntimeError("Student/Teacher vocab mismatch")

    student_tok = AutoTokenizer.from_pretrained(
        args.student,
        local_files_only=True,
    )
    teacher_tok = AutoTokenizer.from_pretrained(
        args.teacher,
        local_files_only=True,
    )

    if student_tok.eos_token_id != teacher_tok.eos_token_id:
        raise RuntimeError("Student/Teacher EOS mismatch")

    # Stronger runtime invariant than the trainer itself needs:
    # H1 interprets Student top-K token IDs directly in Teacher space.
    if student_tok.get_vocab() != teacher_tok.get_vocab():
        raise RuntimeError(
            "Student/Teacher token-id vocabulary mapping mismatch"
        )

    print("TOKEN_ID_SPACE_EXACT_PASS")

    student = AutoModelForCausalLM.from_pretrained(
        args.student,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)

    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)

    student.eval()
    teacher.eval()

    student.config.use_cache = False
    teacher.config.use_cache = False

    for p in student.parameters():
        p.requires_grad_(False)

    for p in teacher.parameters():
        p.requires_grad_(False)

    dataset = trainer.frozen.JsonlDataset(args.train)
    source_map = trainer.read_source_map(args.source_map)

    rows = dataset.rows[:args.limit]

    results = []

    sources_seen = 0
    sources_parse_ok = 0
    sources_has_error = 0
    valid_errors_total = 0
    retained_errors_total = 0
    rejected_error_end_boundary = 0

    for n, row in enumerate(rows, start=1):
        idx = int(row["index"])
        source = source_map[idx]

        prompt_ids = trainer.frozen.get_prompt(
            row,
            student_tok,
            args.max_prompt_length,
        ).to(device)

        prompt_len = int(prompt_ids.shape[1])
        attention_mask = torch.ones_like(prompt_ids)

        current_seed = trainer.rollout_seed(
            args.seed,
            args.epoch,
            idx,
            "current",
        )

        trainer.set_rollout_seed(current_seed)

        with torch.no_grad():
            generated = student.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k_rollout,
                pad_token_id=student_tok.pad_token_id,
                eos_token_id=student_tok.eos_token_id,
                use_cache=True,
            )

        response_ids_all = generated[
            0,
            prompt_len:,
        ].tolist()

        response_ids = trainer.trim_at_eos(
            response_ids_all,
            student_tok.eos_token_id,
        )

        draft = trainer.decode_ids(
            student_tok,
            response_ids,
        )

        sources_seen += 1

        if not draft.strip():
            if n % 8 == 0:
                print(
                    f"progress={n}/{len(rows)} "
                    f"valid_errors={valid_errors_total} "
                    f"retained={retained_errors_total}"
                )
            continue

        _, parsed = trainer.run_feedbacker(
            teacher,
            teacher_tok,
            source,
            draft,
            device,
            args.feedback_max_prompt_tokens,
            args.feedback_max_new_tokens,
        )

        if parsed["parse_ok"]:
            sources_parse_ok += 1

        if parsed["has_error"] is True:
            sources_has_error += 1

        valid, _ = trainer.valid_errors(
            parsed,
            source,
            draft,
            response_ids,
            student_tok,
        )

        valid_errors_total += len(valid)

        for error in valid:
            span = error["translation_span"]

            start = trainer.find_unique(
                draft,
                span,
            )

            if start is None:
                raise RuntimeError(
                    "Invariant broken: valid error lost unique span"
                )

            end = start + len(span)

            # Primary H1 restriction:
            # original side must be a genuine Student-visited
            # token state at the same semantic error-end boundary.
            original_boundary = trainer.find_token_boundary(
                student_tok,
                response_ids,
                draft[:end],
            )

            if original_boundary is None:
                rejected_error_end_boundary += 1
                continue

            original_response_ids = response_ids[
                :original_boundary
            ]

            corrected_response_ids = (
                error["raw_prefix_ids"]
                + error["correction_ids"]
            )

            if trainer.decode_ids(
                student_tok,
                original_response_ids,
            ) != draft[:end]:
                raise RuntimeError(
                    "Original-state roundtrip invariant failed"
                )

            if trainer.decode_ids(
                student_tok,
                corrected_response_ids,
            ) != error["corrected_prefix"]:
                raise RuntimeError(
                    "Corrected-state roundtrip invariant failed"
                )

            original_input_ids = torch.cat(
                [
                    prompt_ids,
                    torch.tensor(
                        [original_response_ids],
                        dtype=torch.long,
                        device=device,
                    ),
                ],
                dim=1,
            )

            corrected_input_ids = torch.cat(
                [
                    prompt_ids,
                    torch.tensor(
                        [corrected_response_ids],
                        dtype=torch.long,
                        device=device,
                    ),
                ],
                dim=1,
            )

            c_minus = teachability_mass(
                student,
                teacher,
                original_input_ids,
                args.top_k,
            )

            c_plus = teachability_mass(
                student,
                teacher,
                corrected_input_ids,
                args.top_k,
            )

            delta = c_plus - c_minus

            retained_errors_total += 1

            results.append({
                "index": idx,
                "error_id": int(error["error_id"]),
                "error_type": error["error_type"],
                "source_span": error["source_span"],
                "translation_span": span,
                "correction": error["correction"],
                "original_prefix": draft[:end],
                "corrected_prefix": error["corrected_prefix"],
                "original_response_tokens": len(original_response_ids),
                "corrected_response_tokens": len(corrected_response_ids),
                "C_minus": c_minus,
                "C_plus": c_plus,
                "delta_C": delta,
            })

        if n % 8 == 0:
            print(
                f"progress={n}/{len(rows)} "
                f"valid_errors={valid_errors_total} "
                f"retained={retained_errors_total}"
            )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as f:
        for r in results:
            f.write(
                json.dumps(
                    r,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print("=" * 80)
    print("H1 SMOKE SUMMARY")
    print("=" * 80)

    print("sources_seen =", sources_seen)
    print("sources_parse_ok =", sources_parse_ok)
    print("sources_has_error =", sources_has_error)
    print("valid_errors_total =", valid_errors_total)
    print("retained_errors_total =", retained_errors_total)
    print(
        "error_end_boundary_rejects =",
        rejected_error_end_boundary,
    )

    retention = (
        retained_errors_total / valid_errors_total
        if valid_errors_total
        else 0.0
    )

    print("exact_end_boundary_retention =", retention)

    if not results:
        raise RuntimeError(
            "No exact-boundary H1 states retained"
        )

    by_source = defaultdict(list)

    for r in results:
        by_source[r["index"]].append(
            r["delta_C"]
        )

    source_deltas = [
        sum(xs) / len(xs)
        for xs in by_source.values()
    ]

    mean_delta = (
        sum(source_deltas)
        / len(source_deltas)
    )

    median_delta = statistics.median(
        source_deltas
    )

    positive_rate = (
        sum(x > 0 for x in source_deltas)
        / len(source_deltas)
    )

    print("retained_sources =", len(source_deltas))
    print("SOURCE_MEAN_DELTA_C =", mean_delta)
    print("SOURCE_MEDIAN_DELTA_C =", median_delta)
    print("SOURCE_POSITIVE_RATE =", positive_rate)
    print("OUTPUT =", args.output)
    print()
    print("H1_REPAIR_TEACHABILITY_SMOKE_PASS")


if __name__ == "__main__":
    main()
