import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_rows(path):
    rows = []

    with Path(path).open(
        encoding="utf-8-sig"
    ) as f:

        for line_no, line in enumerate(f, 1):

            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))

            except Exception as e:
                raise RuntimeError(
                    f"{path}:{line_no}: {e}"
                )

    return rows


def completed_ids(path):
    p = Path(path)

    if not p.exists():
        return set()

    ids = set()

    with p.open(
        encoding="utf-8",
        errors="replace",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            try:
                x = json.loads(line)
                ids.add(int(x["job_id"]))

            except Exception:
                continue

    return ids


def apply_chat(tokenizer, prompt):

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    try:

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    except TypeError:

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        required=True,
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    ap.add_argument(
        "--model",
        required=True,
    )

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
        required=True,
    )

    ap.add_argument(
        "--mode",
        choices=[
            "analysis",
            "case",
        ],
        required=True,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=20260831,
    )

    args = ap.parse_args()

    random.seed(
        args.seed
        +
        args.device_id
    )

    torch.manual_seed(
        args.seed
        +
        args.device_id
    )

    device = (
        f"npu:{args.device_id}"
    )

    if hasattr(torch, "npu"):
        torch.npu.set_device(device)

    rows = load_rows(
        args.input
    )

    mine = [
        x
        for x in rows
        if int(x["job_id"])
        % args.world_size
        ==
        args.device_id
    ]

    done = completed_ids(
        args.output
    )

    pending = [
        x
        for x in mine
        if int(x["job_id"])
        not in done
    ]

    print(
        f"MODE={args.mode}"
    )

    print(
        f"DEVICE={args.device_id}"
    )

    print(
        f"INPUT_TOTAL={len(rows)}"
    )

    print(
        f"DEVICE_TOTAL={len(mine)}"
    )

    print(
        f"ALREADY_DONE={len(done)}"
    )

    print(
        f"PENDING={len(pending)}"
    )

    if not pending:
        print(
            "WORKER_ALREADY_COMPLETE"
        )
        return

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            args.model,
            local_files_only=True,
            trust_remote_code=True,
        )
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = "left"

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            args.model,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    )

    model.to(device)
    model.eval()

    out = Path(
        args.output
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with out.open(
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

            rendered = [
                apply_chat(
                    tokenizer,
                    str(x["prompt"]),
                )
                for x in batch
            ]

            enc = tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1536,
                add_special_tokens=False,
            )

            enc = {
                k: v.to(device)
                for k, v in enc.items()
            }

            input_len = (
                enc["input_ids"]
                .shape[1]
            )

            generation_kwargs = {
                "max_new_tokens":
                    args.max_new_tokens,

                "pad_token_id":
                    tokenizer.pad_token_id,

                "eos_token_id":
                    tokenizer.eos_token_id,

                "use_cache":
                    True,

                "num_beams":
                    1,
            }

            if args.mode == "analysis":

                generation_kwargs.update({
                    "do_sample":
                        False,
                })

            else:

                generation_kwargs.update({
                    "do_sample":
                        True,

                    "temperature":
                        1.0,

                    "top_p":
                        1.0,
                })

            with torch.inference_mode():

                generated = (
                    model.generate(
                        **enc,
                        **generation_kwargs,
                    )
                )

            continuations = (
                generated[
                    :,
                    input_len:
                ]
            )

            texts = (
                tokenizer.batch_decode(
                    continuations,
                    skip_special_tokens=True,
                )
            )

            for row, text, toks in zip(
                batch,
                texts,
                continuations,
            ):

                result = dict(row)

                result.update({
                    "raw_generation":
                        text.strip(),

                    "generator_model":
                        "Qwen3-8B",

                    "generator_mode":
                        args.mode,

                    "enable_thinking":
                        False,

                    "num_beams":
                        1,

                    "temperature":
                        (
                            None
                            if args.mode
                            == "analysis"
                            else 1.0
                        ),

                    "do_sample":
                        (
                            args.mode
                            == "case"
                        ),

                    "max_new_tokens":
                        args.max_new_tokens,

                    "generated_tokens":
                        int(
                            toks.numel()
                        ),
                })

                fout.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                fout.flush()

            print(
                f"device={args.device_id} "
                f"mode={args.mode} "
                f"done="
                f"{min(start + len(batch), len(pending))}"
                f"/{len(pending)}"
            )

    print(
        f"WORKER_PASS "
        f"device={args.device_id} "
        f"mode={args.mode}"
    )


if __name__ == "__main__":
    main()
