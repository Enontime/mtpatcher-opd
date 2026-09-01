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

    path = Path(path)

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
                rows.append(json.loads(line))
            except Exception:
                pass

    return rows


def norm(x):
    return "".join(
        str(x).strip().split()
    ).casefold()


def render(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generate(
    model,
    tokenizer,
    device,
    prompt,
    do_sample,
    temperature=1.0,
    max_new_tokens=256,
):
    text = render(
        tokenizer,
        prompt,
    )

    enc = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    enc = {
        k: v.to(device)
        for k, v in enc.items()
    }

    kwargs = {
        "max_new_tokens":
            max_new_tokens,

        "pad_token_id":
            tokenizer.pad_token_id,

        "eos_token_id":
            tokenizer.eos_token_id,

        "do_sample":
            do_sample,
    }

    if do_sample:
        kwargs.update(
            temperature=temperature,
            top_p=0.95,
        )

    with torch.inference_mode():
        out = model.generate(
            **enc,
            **kwargs,
        )

    prompt_len = (
        enc["input_ids"].shape[1]
    )

    return tokenizer.decode(
        out[0][prompt_len:],
        skip_special_tokens=True,
    ).strip()


def clean_candidate(line):
    x = line.strip()

    if not x:
        return ""

    x = re.sub(
        r"^[\-\*\•\·\s]+",
        "",
        x,
    )

    x = re.sub(
        r"^\d+\s*[\.\)、\):：]\s*",
        "",
        x,
    )

    for prefix in (
        "候选词：",
        "候选词:",
        "候选短语：",
        "候选短语:",
        "中文：",
        "中文:",
        "短语：",
        "短语:",
        "答案：",
        "答案:",
    ):
        if x.startswith(prefix):
            x = x[len(prefix):].strip()

    x = x.strip(
        " \"'“”‘’`"
    )

    # Some models occasionally append explanations.
    for sep in (
        "\t",
        " —— ",
        " -- ",
        " -> ",
        " → ",
    ):
        if sep in x:
            x = x.split(
                sep,
                1,
            )[0].strip()

    return x


def parse_pool(raw):
    raw = re.sub(
        r"```.*?",
        "",
        raw,
        flags=re.I,
    )

    raw = raw.replace(
        "```",
        "",
    )

    out = []

    for line in raw.splitlines():
        x = clean_candidate(line)

        if not x:
            continue

        # Keep phrase-level material.
        if len(x) > 80:
            continue

        if x in {
            "Category",
            "Semantics",
            "类别",
            "语义",
        }:
            continue

        out.append(x)

    return out


def pool_prompt(
    row,
    aspect,
    forbidden,
):
    forbidden_text = "\n".join(
        f"- {x}"
        for x in forbidden
    )

    if aspect == "category":
        relation = """请给出 8 个与原短语属于相同类别、
相同实体类型、相同概念类别或相似术语类别的中文词或短语。"""
    else:
        relation = """请给出 8 个与原短语语义相关、
经常共现、属于同一事件场景或自然出现在相近语境中的中文词或短语。"""

    return f"""你是一名中英机器翻译专家。

{relation}

优先选择：
- 相对少见；
- 对机器翻译具有一定难度；
- 可以作为独立翻译知识的词或短语。

要求：
1. 每行只输出一个中文词或短语。
2. 总共输出 8 行。
3. 不输出英文。
4. 不解释。
5. 不写完整句子。
6. 不得输出下列禁用短语。

原始句子：
{row["source"]}

原始错误短语：
{row["source_span"]}

禁用短语：
{forbidden_text}

现在直接输出 8 个中文候选：
"""


def translation_prompt(phrase):
    return f"""Translate the following Chinese word or short phrase into natural English.

Return only the English translation.
Do not explain.
Do not use quotation marks.

Chinese:
{phrase}
"""


def clean_translation(raw):
    lines = [
        x.strip()
        for x in raw.splitlines()
        if x.strip()
    ]

    if not lines:
        return ""

    x = lines[0]

    for prefix in (
        "English:",
        "Translation:",
        "English translation:",
        "英文：",
        "英文:",
        "翻译：",
        "翻译:",
    ):
        if x.startswith(prefix):
            x = x[len(prefix):].strip()

    x = x.strip(
        " \"'“”‘’`"
    )

    return x


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
        default=2,
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
        args.jobs
    )

    assigned = [
        row
        for row in jobs
        if int(
            row["analog_job_id"]
        ) % args.world_size
        == args.device_id
    ]

    output = Path(
        args.output
    )

    previous = {
        int(x["analog_job_id"]): x
        for x in load_jsonl(output)
        if x.get("parse_ok")
    }

    pending = [
        row
        for row in assigned
        if int(
            row["analog_job_id"]
        ) not in previous
    ]

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            args.model,
            local_files_only=True,
            trust_remote_code=True,
        )
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
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
        f"LAST8_DEVICE={args.device_id} "
        f"ASSIGNED={len(assigned)} "
        f"PENDING={len(pending)}",
        flush=True,
    )

    final = dict(previous)

    for pos, row in enumerate(
        pending,
        1,
    ):
        anchor = row["source_span"]

        selected = []

        analogs = {
            "category": [],
            "semantics": [],
        }

        errors = []

        for aspect in (
            "category",
            "semantics",
        ):
            candidates = []

            for attempt in range(
                1,
                13,
            ):
                raw = generate(
                    model,
                    tokenizer,
                    device,
                    pool_prompt(
                        row,
                        aspect,
                        [
                            anchor,
                            *selected,
                            *candidates,
                        ],
                    ),
                    do_sample=True,
                    temperature=(
                        0.8
                        if attempt <= 6
                        else 1.1
                    ),
                    max_new_tokens=320,
                )

                pool = parse_pool(raw)

                for candidate in pool:
                    n = norm(candidate)

                    if not n:
                        continue

                    if n == norm(anchor):
                        continue

                    if any(
                        n == norm(x)
                        for x in selected
                    ):
                        continue

                    if any(
                        n == norm(x)
                        for x in candidates
                    ):
                        continue

                    candidates.append(
                        candidate
                    )

                if len(candidates) >= 2:
                    break

            if len(candidates) < 2:
                errors.append(
                    f"{aspect}:"
                    f"only_{len(candidates)}_candidates"
                )
                break

            chosen = candidates[:2]

            for phrase in chosen:
                english = ""

                for tr_attempt in range(
                    1,
                    5,
                ):
                    raw_en = generate(
                        model,
                        tokenizer,
                        device,
                        translation_prompt(
                            phrase
                        ),
                        do_sample=False,
                        max_new_tokens=96,
                    )

                    english = (
                        clean_translation(
                            raw_en
                        )
                    )

                    if english:
                        break

                if not english:
                    errors.append(
                        f"{aspect}:"
                        f"translation_failed:"
                        f"{phrase}"
                    )
                    break

                analogs[
                    aspect
                ].append(
                    {
                        "source":
                            phrase,

                        "target":
                            english,
                    }
                )

                selected.append(
                    phrase
                )

            if len(
                analogs[aspect]
            ) != 2:
                break

        unique = {
            norm(x["source"])
            for arr
            in analogs.values()
            for x in arr
        }

        parse_ok = (
            len(
                analogs["category"]
            ) == 2
            and
            len(
                analogs["semantics"]
            ) == 2
            and
            len(unique) == 4
            and
            norm(anchor)
            not in unique
        )

        result = {
            **row,

            "parse_ok":
                parse_ok,

            "parse_error":
                ""
                if parse_ok
                else ";".join(
                    errors
                ),

            "analogs":
                analogs
                if parse_ok
                else None,

            "wa_recovery_origin":
                "last8_candidate_pool",

            "construction_method":
                "MT_PATCHER_WA_LAST8_POOL_V14",
        }

        final[
            int(
                row["analog_job_id"]
            )
        ] = result

        print(
            f"LAST8_DEVICE={args.device_id} "
            f"JOB={row['analog_job_id']} "
            f"OK={parse_ok} "
            f"PROGRESS={pos}/{len(pending)}",
            flush=True,
        )

    tmp = Path(
        str(output) + ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as f:
        for jid in sorted(final):
            f.write(
                json.dumps(
                    final[jid],
                    ensure_ascii=False,
                )
                + "\n"
            )

    tmp.replace(output)

    failures = sum(
        not bool(x.get("parse_ok"))
        for x in final.values()
    )

    print(
        f"LAST8_DEVICE_{args.device_id}_COMPLETE "
        f"FAILURES={failures}",
        flush=True,
    )


if __name__ == "__main__":
    main()
