#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT = (
    "Translate the following text into English "
    "without additional explanations:\n\n"
    "{source}\n\n"
)


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            x = json.loads(line)
            if "index" not in x:
                raise RuntimeError(f"line={line_no}: missing index")
            if not str(x.get("source", "")).strip():
                raise RuntimeError(f"line={line_no}: missing source")
            rows.append(x)
    return rows


def load_done(path):
    done = {}
    if not path.exists():
        return done

    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            x = json.loads(line)
            done[int(x["index"])] = x

    return done


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True)

    ap.add_argument("--device-id", type=int, required=True)
    ap.add_argument("--world-size", type=int, default=16)

    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=512)

    args = ap.parse_args()

    torch.npu.set_device(
        f"npu:{args.device_id}"
    )

    device = torch.device(
        f"npu:{args.device_id}"
    )

    all_rows = load_jsonl(
        args.input
    )

    assigned = [
        x for x in all_rows
        if int(x["index"]) % args.world_size
        == args.device_id
    ]

    output = Path(args.output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    done = load_done(output)

    pending = [
        x for x in assigned
        if int(x["index"]) not in done
    ]

    print(
        f"DEVICE={args.device_id} "
        f"ASSIGNED={len(assigned)} "
        f"COMPLETED={len(done)} "
        f"PENDING={len(pending)}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
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
        attn_implementation="sdpa",
    )

    model.to(device)
    model.eval()

    print(
        f"SEQKD_TEACHER_MODEL_READY "
        f"device={args.device_id}",
        flush=True,
    )

    generated = 0
    start_all = time.time()

    with output.open(
        "a",
        encoding="utf-8",
    ) as fout:

        for start in range(
            0,
            len(pending),
            args.batch_size,
        ):

            batch = pending[
                start:
                start + args.batch_size
            ]

            messages = [
                [
                    {
                        "role": "user",
                        "content": PROMPT.format(
                            source=x["source"]
                        ),
                    }
                ]
                for x in batch
            ]

            prompt_texts = [
                tokenizer.apply_chat_template(
                    m,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for m in messages
            ]

            encoded = tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )

            encoded = {
                k: v.to(
                    device,
                    non_blocking=True,
                )
                for k, v in encoded.items()
            }

            input_width = (
                encoded["input_ids"].shape[1]
            )

            t0 = time.time()

            with torch.inference_mode():
                outputs = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            new = outputs[
                :,
                input_width:
            ]

            texts = tokenizer.batch_decode(
                new,
                skip_special_tokens=True,
            )

            for row, msg, text, ids in zip(
                batch,
                messages,
                texts,
                new,
            ):
                target = text.strip()

                if not target:
                    raise RuntimeError(
                        f"empty translation "
                        f"index={row['index']}"
                    )

                result = {
                    "index":
                        int(row["index"]),

                    "source":
                        row["source"],

                    "messages":
                        msg,

                    "target_translation":
                        target,

                    "construction_method":
                        "RQ0_SEQKD_NEWCRAWL_QWEN3_8B",

                    "teacher_model":
                        args.model,

                    "teacher_thinking":
                        False,

                    "do_sample":
                        False,

                    "max_new_tokens":
                        args.max_new_tokens,

                    "new_tokens":
                        int(ids.numel()),
                }

                fout.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                fout.flush()

                generated += 1

            completed = (
                len(done) + generated
            )

            if (
                generated <= args.batch_size
                or generated % 160 == 0
                or completed == len(assigned)
            ):
                print(
                    f"DEVICE={args.device_id} "
                    f"GENERATED={completed}/"
                    f"{len(assigned)} "
                    f"batch_seconds="
                    f"{time.time()-t0:.3f}",
                    flush=True,
                )

    print(
        f"DEVICE={args.device_id} "
        f"SEQKD_TEACHER_WORKER_PASS "
        f"seconds={time.time()-start_all:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
