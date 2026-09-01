#!/usr/bin/env python3
# coding: utf-8

import argparse
import json
import time
from pathlib import Path

import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("messages"), list) or not row["messages"]:
                raise RuntimeError(f"line {line_no}: invalid messages")
            if not isinstance(row.get("reference"), str) or not row["reference"].strip():
                raise RuntimeError(f"line {line_no}: missing reference")
            rows.append(row)
    return rows


def read_done(path: Path):
    if not path.exists():
        return {}
    out = {}
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out[int(row["index"])] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--method", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--attn-implementation", choices=["sdpa", "eager"], default="sdpa")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    if hasattr(torch.npu, "manual_seed_all"):
        torch.npu.manual_seed_all(args.seed)

    if not torch.npu.is_available():
        raise RuntimeError("Ascend NPU required")
    torch.npu.set_device(0)

    rows = read_jsonl(args.input)
    done = read_done(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PILOT-V2 QWEN3 MT EVALUATION")
    print("=" * 80)
    print("model =", args.model)
    print("input =", args.input)
    print("rows =", len(rows))
    print("output =", args.output)
    print("method =", args.method)
    print("already_done =", len(done))
    print("batch_size =", args.batch_size)
    print("max_new_tokens =", args.max_new_tokens)
    print("enable_thinking = False")
    print("do_sample = True")
    print("seed =", args.seed)
    print("temperature =", args.temperature)
    print("top_p =", args.top_p)
    print("top_k =", args.top_k)
    print("device = npu:0")
    print("device_name =", torch.npu.get_device_name(0))
    print("attn_implementation =", args.attn_implementation)

    tokenizer_path = args.tokenizer or args.model
    print("tokenizer =", tokenizer_path)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    )
    model.to("npu:0")
    model.eval()

    pending = [x for x in rows if int(x["index"]) not in done]
    generated = 0
    total_seconds = 0.0

    with args.output.open("a", encoding="utf-8") as out:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]

            prompt_texts = [
                tokenizer.apply_chat_template(
                    x["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for x in batch
            ]

            encoded = tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            encoded = {k: v.to("npu:0", non_blocking=True) for k, v in encoded.items()}
            input_width = encoded["input_ids"].shape[1]

            torch.npu.synchronize()
            t0 = time.time()
            with torch.inference_mode():
                outputs = model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            torch.npu.synchronize()
            elapsed = time.time() - t0
            total_seconds += elapsed

            new = outputs[:, input_width:]
            texts = tokenizer.batch_decode(new, skip_special_tokens=True)

            for row, text, token_ids in zip(batch, texts, new):
                translation = text.strip()
                if not translation:
                    raise RuntimeError(f"empty translation at index={row['index']}")

                result = {
                    "index": int(row["index"]),
                    "source": row["source"],
                    "reference": row["reference"],
                    "student_translation": translation,
                    "evaluation_method": args.method,
                    "student_model_path": args.model,
                    "generation_config": {
                        "enable_thinking": False,
                        "do_sample": True,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "seed": args.seed,
                        "max_new_tokens": args.max_new_tokens,
                        "dtype": str(dtype),
                    },
                    "new_tokens": int(token_ids.numel()),
                }
                out.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                out.flush()
                generated += 1

            done_total = len(done) + generated
            print(
                f"generated={done_total}/{len(rows)} "
                f"batch_seconds={elapsed:.3f}",
                flush=True,
            )

    final = read_done(args.output)
    expected = {int(x["index"]) for x in rows}
    if set(final) != expected:
        raise RuntimeError(
            f"prediction coverage mismatch: got={len(final)} expected={len(expected)}"
        )

    print("generated_this_run =", generated)
    print("total_output_rows =", len(final))
    print(
        "mean_seconds_per_generated_row =",
        (total_seconds / generated if generated else 0.0),
    )
    print("PILOT_V2_MT_GENERATION_PASS")


if __name__ == "__main__":
    main()
