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
                done.add(int(row["analog_job_id"]))
            except Exception:
                continue

    return done


def extract_json(text):
    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.I,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    start = text.find("{")
    end = text.rfind("}")

    if start < 0 or end <= start:
        raise ValueError("JSON object not found")

    return json.loads(
        text[start:end + 1]
    )


def validate(obj, anchor):
    if not isinstance(obj, dict):
        raise ValueError("root not dict")

    out = {}

    seen_src = set()

    for key in ("category", "semantics"):
        arr = obj.get(key)

        if not isinstance(arr, list):
            raise ValueError(
                f"{key} is not list"
            )

        if len(arr) != 2:
            raise ValueError(
                f"{key} requires exactly two pairs"
            )

        clean = []

        for x in arr:
            if not isinstance(x, dict):
                raise ValueError(
                    f"{key} entry not dict"
                )

            src = x.get("source")
            tgt = x.get("target")

            if not isinstance(src, str):
                raise ValueError("source invalid")

            if not isinstance(tgt, str):
                raise ValueError("target invalid")

            src = src.strip()
            tgt = tgt.strip()

            if not src or not tgt:
                raise ValueError(
                    "empty bilingual pair"
                )

            norm = "".join(src.split()).casefold()

            if norm == "".join(
                anchor.split()
            ).casefold():
                raise ValueError(
                    "analog equals original anchor"
                )

            if norm in seen_src:
                raise ValueError(
                    "duplicate analogous source"
                )

            seen_src.add(norm)

            clean.append(
                {
                    "source": src,
                    "target": tgt,
                }
            )

        out[key] = clean

    if len(seen_src) != 4:
        raise ValueError(
            "expected four distinct analogs"
        )

    return out


def build_prompt(row):
    return f"""Assume you are a Chinese-English language expert with broad knowledge and strong associative ability.

A Chinese machine-translation student made an error involving the following Chinese word or phrase P in sentence X.

Associate rare and challenging Chinese words or phrases from exactly two perspectives:

1. Category:
   Words or phrases belonging to the same category or type as P.

2. Semantics:
   Words or phrases that frequently co-occur with P or naturally occur in closely related semantic contexts.

Generate exactly TWO Chinese-English bilingual pairs for Category and exactly TWO for Semantics.

Requirements:
- All four Chinese entries must be different from P and from each other.
- Prefer relatively rare or challenging translation knowledge.
- Each entry should be a word or short phrase, not a full sentence.
- Give a natural English translation for every Chinese entry.
- Return ONLY one JSON object in the exact structure below.
- Do not add markdown or explanations.

{{
  "category": [
    {{"source": "Chinese phrase 1", "target": "English translation 1"}},
    {{"source": "Chinese phrase 2", "target": "English translation 2"}}
  ],
  "semantics": [
    {{"source": "Chinese phrase 3", "target": "English translation 3"}},
    {{"source": "Chinese phrase 4", "target": "English translation 4"}}
  ]
}}

X: {row["source"]}
P: {row["source_span"]}
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
        default=256,
    )

    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=20260825,
    )

    args = ap.parse_args()

    device_id = args.device_id

    torch.npu.set_device(device_id)

    device = f"npu:{device_id}"

    random.seed(args.seed + device_id)
    torch.manual_seed(args.seed + device_id)

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

    jobs = load_jsonl(args.jobs)

    assigned = [
        x
        for x in jobs
        if int(x["analog_job_id"])
        % args.world_size
        == device_id
    ]

    out_path = Path(args.output)

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed = load_completed(out_path)

    pending = [
        x for x in assigned
        if int(x["analog_job_id"])
        not in completed
    ]

    print(
        f"DEVICE={device_id} "
        f"ASSIGNED={len(assigned)} "
        f"COMPLETED={len(completed)} "
        f"PENDING={len(pending)}",
        flush=True,
    )

    print(
        f"WA_ANALOG_MODEL_READY device={device_id}",
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
                start:start + args.batch_size
            ]

            prompts = [
                build_prompt(row)
                for row in batch
            ]

            chats = [
                [
                    {
                        "role": "user",
                        "content": p,
                    }
                ]
                for p in prompts
            ]

            rendered = [
                tokenizer.apply_chat_template(
                    c,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for c in chats
            ]

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

                parse_ok = False
                parsed = None
                parse_error = ""

                try:
                    parsed = validate(
                        extract_json(raw),
                        row["source_span"],
                    )
                    parse_ok = True
                except Exception as exc:
                    parse_error = (
                        type(exc).__name__
                        + ": "
                        + str(exc)
                    )

                fout.write(
                    json.dumps(
                        {
                            **row,

                            "raw_analogy":
                                raw,

                            "parse_ok":
                                parse_ok,

                            "parse_error":
                                parse_error,

                            "analogs":
                                parsed,

                            "temperature":
                                args.temperature,

                            "analog_model":
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
                    f"DEVICE={device_id} "
                    f"GENERATED={generated}/"
                    f"{len(pending)}",
                    flush=True,
                )

    print(
        f"WA_ANALOG_DEVICE_{device_id}_COMPLETE",
        flush=True,
    )


if __name__ == "__main__":
    main()
