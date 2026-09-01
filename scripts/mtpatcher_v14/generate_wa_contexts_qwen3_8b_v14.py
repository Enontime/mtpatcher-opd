import argparse
import json
import random
import re
from pathlib import Path

import torch
import torch_npu
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


def load_jsonl(path):
    rows = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def load_completed(path):
    done = set()

    if not path.exists():
        return done

    with path.open(
        encoding="utf-8",
        errors="replace",
    ) as f:
        for line in f:
            try:
                row = json.loads(line)

                done.add(
                    int(
                        row[
                            "wa_context_job_id"
                        ]
                    )
                )
            except Exception:
                continue

    return done


def parse_pair(raw):
    zh = None
    en = None

    for line in raw.splitlines():
        line = line.strip()

        if line.startswith("中文句子:"):
            zh = line.split(
                ":",
                1,
            )[1].strip()

        elif line.startswith("中文句子："):
            zh = line.split(
                "：",
                1,
            )[1].strip()

        elif line.startswith("英文句子:"):
            en = line.split(
                ":",
                1,
            )[1].strip()

        elif line.startswith("英文句子："):
            en = line.split(
                "：",
                1,
            )[1].strip()

    if not zh or not en:
        return None, None

    return zh, en


def prompt(row):
    return f"""You are a Chinese-English parallel-data synthesizer.

Use the ORIGINAL SOURCE only as a loose guide for domain, register and style.

Create ONE new Chinese-English parallel sentence pair containing the given bilingual ANALOG WORD PAIR.

Requirements:
- The new Chinese sentence must naturally use the Chinese analog phrase.
- The English sentence must naturally express its given English translation.
- Preserve approximately the same domain/register/style as the original source.
- The new sentence should describe a different situation or semantic content from the original source.
- Produce fluent natural Chinese and English.
- Do not mention placeholders P or Q.
- Output exactly two lines and nothing else:

中文句子: <new Chinese sentence>
英文句子: <new English translation>

ORIGINAL SOURCE:
{row["original_source"]}

ANALOG WORD PAIR:
Chinese: {row["analog_source"]}
English: {row["analog_target"]}
"""


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--jobs", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True)

    ap.add_argument(
        "--device-id",
        type=int,
        required=True,
    )

    ap.add_argument(
        "--world-size",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=192,
    )

    ap.add_argument(
        "--temperature",
        type=float,
        default=1.5,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=20260825,
    )

    args = ap.parse_args()

    torch.npu.set_device(
        args.device_id
    )

    device = f"npu:{args.device_id}"

    random.seed(
        args.seed + args.device_id
    )

    torch.manual_seed(
        args.seed + args.device_id
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )

    model.to(device)
    model.eval()

    all_jobs = load_jsonl(args.jobs)

    assigned = [
        x
        for x in all_jobs
        if int(
            x["wa_context_job_id"]
        ) % args.world_size
        == args.device_id
    ]

    out_path = Path(args.output)

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed = load_completed(
        out_path
    )

    pending = [
        x for x in assigned
        if int(
            x["wa_context_job_id"]
        ) not in completed
    ]

    print(
        f"DEVICE={args.device_id} "
        f"ASSIGNED={len(assigned)} "
        f"COMPLETED={len(completed)} "
        f"PENDING={len(pending)}",
        flush=True,
    )

    print(
        "WA_CONTEXT_MODEL_READY "
        f"device={args.device_id}",
        flush=True,
    )

    generated = 0

    with out_path.open(
        "a",
        encoding="utf-8",
    ) as fout:

        for start in range(
            0,
            len(pending),
            args.batch_size,
        ):
            batch = pending[
                start:start
                + args.batch_size
            ]

            rendered = []

            for row in batch:
                chat = [
                    {
                        "role": "user",
                        "content": prompt(row),
                    }
                ]

                rendered.append(
                    tokenizer.apply_chat_template(
                        chat,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )

            enc = tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )

            enc = {
                k: v.to(device)
                for k, v in enc.items()
            }

            with torch.inference_mode():
                outputs = model.generate(
                    **enc,
                    do_sample=True,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            prompt_len = enc[
                "input_ids"
            ].shape[1]

            for row, output in zip(
                batch,
                outputs,
            ):
                raw = tokenizer.decode(
                    output[prompt_len:],
                    skip_special_tokens=True,
                ).strip()

                zh, en = parse_pair(raw)

                fout.write(
                    json.dumps(
                        {
                            **row,

                            "raw_generation":
                                raw,

                            "parse_ok":
                                bool(zh and en),

                            "synthesized_source":
                                zh,

                            "synthesized_target":
                                en,

                            "temperature":
                                args.temperature,

                            "synthesis_model":
                                args.model,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            fout.flush()

            generated += len(batch)

            if (
                generated == len(batch)
                or generated % 80 == 0
                or generated == len(pending)
            ):
                print(
                    f"DEVICE={args.device_id} "
                    f"GENERATED={generated}/"
                    f"{len(pending)}",
                    flush=True,
                )

    print(
        f"WA_CONTEXT_DEVICE_"
        f"{args.device_id}_COMPLETE",
        flush=True,
    )


if __name__ == "__main__":
    main()
