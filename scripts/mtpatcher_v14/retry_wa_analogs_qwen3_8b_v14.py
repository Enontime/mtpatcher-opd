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

    if not path.exists():
        return rows

    with path.open(
        encoding="utf-8",
        errors="replace",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            try:
                rows.append(
                    json.loads(line)
                )
            except Exception:
                continue

    return rows


def normalize(x):
    return "".join(
        x.strip().split()
    ).casefold()


def validate(obj, anchor):
    if not isinstance(obj, dict):
        raise ValueError(
            "root_not_dict"
        )

    seen = set()

    result = {}

    for key in (
        "category",
        "semantics",
    ):
        arr = obj.get(key)

        if not isinstance(arr, list):
            raise ValueError(
                f"{key}_not_list"
            )

        if len(arr) != 2:
            raise ValueError(
                f"{key}_count_{len(arr)}"
            )

        clean = []

        for item in arr:
            if not isinstance(
                item,
                dict,
            ):
                raise ValueError(
                    "item_not_dict"
                )

            src = item.get("source")
            tgt = item.get("target")

            if not isinstance(
                src,
                str,
            ):
                raise ValueError(
                    "source_invalid"
                )

            if not isinstance(
                tgt,
                str,
            ):
                raise ValueError(
                    "target_invalid"
                )

            src = src.strip()
            tgt = tgt.strip()

            if not src or not tgt:
                raise ValueError(
                    "empty_pair"
                )

            n = normalize(src)

            if n == normalize(anchor):
                raise ValueError(
                    "analog_equals_anchor"
                )

            if n in seen:
                raise ValueError(
                    "duplicate_analog"
                )

            seen.add(n)

            clean.append(
                {
                    "source": src,
                    "target": tgt,
                }
            )

        result[key] = clean

    if len(seen) != 4:
        raise ValueError(
            "not_four_unique"
        )

    return result


def extract_json(raw):
    raw = raw.strip()

    raw = re.sub(
        r"^```(?:json)?\s*",
        "",
        raw,
        flags=re.I,
    )

    raw = re.sub(
        r"\s*```$",
        "",
        raw,
    )

    start = raw.find("{")
    end = raw.rfind("}")

    if (
        start < 0
        or end <= start
    ):
        raise ValueError(
            "json_object_missing"
        )

    return json.loads(
        raw[start:end + 1]
    )


def prompt(row):
    return f"""Generate exactly four Chinese-English analogous phrase pairs for the translation weakness below.

You MUST return valid JSON only.

There are exactly two groups:
- category: exactly 2 pairs from the same category/type
- semantics: exactly 2 semantically associated or commonly co-occurring pairs

All four Chinese phrases must:
- be different from the original phrase
- be different from each other
- preferably be relatively rare or translation-challenging
- be words or short phrases, not sentences

Use EXACTLY this schema:

{{
  "category": [
    {{"source": "中文短语1", "target": "English phrase 1"}},
    {{"source": "中文短语2", "target": "English phrase 2"}}
  ],
  "semantics": [
    {{"source": "中文短语3", "target": "English phrase 3"}},
    {{"source": "中文短语4", "target": "English phrase 4"}}
  ]
}}

Do not output markdown.
Do not output commentary.
Do not add extra keys.

Original Chinese sentence:
{row["source"]}

Original problematic Chinese phrase:
{row["source_span"]}
"""


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--jobs",
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
        "--max-attempts",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=384,
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

    torch.npu.set_device(
        args.device_id
    )

    device = (
        f"npu:{args.device_id}"
    )

    random.seed(
        args.seed
        + args.device_id
    )

    torch.manual_seed(
        args.seed
        + args.device_id
    )

    jobs = load_jsonl(
        Path(args.jobs)
    )

    assigned = [
        x
        for x in jobs
        if int(
            x["analog_job_id"]
        ) % args.world_size
        == args.device_id
    ]

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Preserve only previous successful retries.
    previous_success = {}

    for row in load_jsonl(
        output_path
    ):
        if row.get("parse_ok"):
            previous_success[
                int(
                    row[
                        "analog_job_id"
                    ]
                )
            ] = row

    pending = [
        row
        for row in assigned
        if int(
            row["analog_job_id"]
        ) not in previous_success
    ]

    print(
        f"DEVICE={args.device_id} "
        f"ASSIGNED={len(assigned)} "
        f"PREVIOUS_SUCCESS="
        f"{len(previous_success)} "
        f"PENDING={len(pending)}",
        flush=True,
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            args.model,
            local_files_only=True,
            trust_remote_code=True,
        )
    )

    tokenizer.padding_side = (
        "left"
    )

    if (
        tokenizer.pad_token_id
        is None
    ):
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id
        )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            args.model,
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
    )

    model.to(device)
    model.eval()

    print(
        "WA_RETRY_MODEL_READY "
        f"device={args.device_id}",
        flush=True,
    )

    final_rows = dict(
        previous_success
    )

    for pos, row in enumerate(
        pending,
        1,
    ):
        success = None
        last_raw = ""
        last_error = ""

        for attempt in range(
            1,
            args.max_attempts + 1,
        ):
            chat = [
                {
                    "role": "user",
                    "content":
                        prompt(row),
                }
            ]

            rendered = (
                tokenizer
                .apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

            enc = tokenizer(
                rendered,
                return_tensors="pt",
                add_special_tokens=False,
            )

            enc = {
                k: v.to(device)
                for k, v
                in enc.items()
            }

            with torch.inference_mode():
                output = model.generate(
                    **enc,
                    do_sample=True,
                    temperature=
                        args.temperature,
                    max_new_tokens=
                        args.max_new_tokens,
                    pad_token_id=
                        tokenizer.pad_token_id,
                    eos_token_id=
                        tokenizer.eos_token_id,
                )

            prompt_len = (
                enc["input_ids"]
                .shape[1]
            )

            raw = tokenizer.decode(
                output[0][prompt_len:],
                skip_special_tokens=True,
            ).strip()

            last_raw = raw

            try:
                analogs = validate(
                    extract_json(raw),
                    row["source_span"],
                )

                success = {
                    **row,

                    "raw_analogy":
                        raw,

                    "parse_ok":
                        True,

                    "parse_error":
                        "",

                    "analogs":
                        analogs,

                    "retry_attempt":
                        attempt,

                    "analog_model":
                        args.model,

                    "temperature":
                        args.temperature,

                    "construction_method":
                        "MT_PATCHER_WA_ANALOG_RETRY_V14",
                }

                break

            except Exception as exc:
                last_error = (
                    type(exc).__name__
                    + ": "
                    + str(exc)
                )

        if success is None:
            success = {
                **row,

                "raw_analogy":
                    last_raw,

                "parse_ok":
                    False,

                "parse_error":
                    last_error,

                "analogs":
                    None,

                "retry_attempt":
                    args.max_attempts,

                "construction_method":
                    "MT_PATCHER_WA_ANALOG_RETRY_V14",
            }

        final_rows[
            int(
                row[
                    "analog_job_id"
                ]
            )
        ] = success

        if (
            pos == 1
            or pos % 10 == 0
            or pos == len(pending)
        ):
            ok_now = sum(
                bool(x.get("parse_ok"))
                for x in
                final_rows.values()
            )

            print(
                f"DEVICE={args.device_id} "
                f"PROCESSED={pos}/"
                f"{len(pending)} "
                f"SUCCESS_TOTAL={ok_now}",
                flush=True,
            )

    tmp = output_path.with_suffix(
        ".jsonl.tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as f:
        for jid in sorted(
            final_rows
        ):
            f.write(
                json.dumps(
                    final_rows[jid],
                    ensure_ascii=False,
                )
                + "\n"
            )

    tmp.replace(
        output_path
    )

    failures = sum(
        not bool(
            x.get("parse_ok")
        )
        for x
        in final_rows.values()
    )

    print(
        f"WA_RETRY_DEVICE_"
        f"{args.device_id}_COMPLETE "
        f"FAILURES={failures}",
        flush=True,
    )


if __name__ == "__main__":
    main()
