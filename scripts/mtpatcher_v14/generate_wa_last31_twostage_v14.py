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


def load(path):
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
                rows.append(
                    json.loads(line)
                )
            except Exception:
                pass

    return rows


def norm(x):
    return "".join(
        str(x).strip().split()
    ).casefold()


def clean_phrase(raw):
    raw = raw.strip()

    raw = re.sub(
        r"^```.*?\n?",
        "",
        raw,
        flags=re.I,
    )

    raw = re.sub(
        r"\n?```$",
        "",
        raw,
    )

    lines = [
        x.strip()
        for x in raw.splitlines()
        if x.strip()
    ]

    if not lines:
        return ""

    x = lines[0]

    x = re.sub(
        r"^[\-\*\d\.\)\s]+",
        "",
        x,
    )

    # Common labels.
    for prefix in (
        "中文短语：",
        "中文短语:",
        "短语：",
        "短语:",
        "答案：",
        "答案:",
        "Phrase:",
        "Chinese:",
    ):
        if x.startswith(prefix):
            x = x[len(prefix):].strip()

    x = x.strip(
        " \t\r\n\"'“”‘’`"
    )

    # JSON fallback.
    if x.startswith("{"):
        try:
            obj = json.loads(x)

            for key in (
                "source",
                "phrase",
                "chinese",
            ):
                v = obj.get(key)

                if (
                    isinstance(v, str)
                    and v.strip()
                ):
                    return v.strip()
        except Exception:
            pass

    return x.strip()


def clean_translation(raw):
    raw = raw.strip()

    raw = re.sub(
        r"^```.*?\n?",
        "",
        raw,
        flags=re.I,
    )

    raw = re.sub(
        r"\n?```$",
        "",
        raw,
    )

    lines = [
        x.strip()
        for x in raw.splitlines()
        if x.strip()
    ]

    if not lines:
        return ""

    x = lines[0]

    for prefix in (
        "English translation:",
        "English:",
        "Translation:",
        "英文翻译：",
        "英文翻译:",
        "答案：",
        "答案:",
    ):
        if x.startswith(prefix):
            x = x[len(prefix):].strip()

    x = x.strip(
        " \t\r\n\"'“”‘’`"
    )

    if x.startswith("{"):
        try:
            obj = json.loads(x)

            for key in (
                "target",
                "translation",
                "english",
            ):
                v = obj.get(key)

                if (
                    isinstance(v, str)
                    and v.strip()
                ):
                    return v.strip()
        except Exception:
            pass

    return x.strip()


def phrase_prompt(
    row,
    aspect,
    selected,
):
    anchor = row["source_span"]

    forbidden = [
        anchor,
        *selected,
    ]

    forbidden_text = "、".join(
        forbidden
    )

    if aspect == "category":
        instruction = (
            "请给出一个与原短语属于同一类别、"
            "同一事物类型或同一概念类别的"
            "较少见、较有翻译难度的中文词或短语。"
        )
    else:
        instruction = (
            "请给出一个与原短语在语义上紧密相关、"
            "经常共现或自然出现在相近语境中的"
            "较少见、较有翻译难度的中文词或短语。"
        )

    return f"""你是一名中英机器翻译专家。

{instruction}

要求：
1. 只输出一个中文词或短语。
2. 不要输出完整句子。
3. 不要解释。
4. 不要加编号。
5. 不要输出英文。
6. 不得与禁用短语相同。
7. 尽量选择对机器翻译具有挑战性的表达。

原句：
{row["source"]}

原错误短语：
{anchor}

禁用短语：
{forbidden_text}

只输出新的中文词或短语：
"""


def translation_prompt(
    phrase,
):
    return f"""Translate the following Chinese word or short phrase into natural English.

Output only the English translation.
Do not explain.
Do not add quotation marks.

Chinese phrase:
{phrase}
"""


def render(
    tokenizer,
    prompt,
):
    return tokenizer.apply_chat_template(
        [
            {
                "role":
                    "user",

                "content":
                    prompt,
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
    sample,
    temperature=1.0,
):
    rendered = render(
        tokenizer,
        prompt,
    )

    enc = tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
    )

    enc = {
        k: v.to(device)
        for k, v in enc.items()
    }

    kwargs = dict(
        max_new_tokens=96,
        pad_token_id=
            tokenizer.pad_token_id,
        eos_token_id=
            tokenizer.eos_token_id,
    )

    if sample:
        kwargs.update(
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
        )
    else:
        kwargs.update(
            do_sample=False,
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
        default=4,
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

    jobs = load(
        args.jobs
    )

    assigned = [
        x
        for x in jobs
        if int(
            x["analog_job_id"]
        ) % args.world_size
        == args.device_id
    ]

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    previous = {
        int(x["analog_job_id"]): x
        for x in load(output)
        if x.get("parse_ok")
    }

    pending = [
        x
        for x in assigned
        if int(
            x["analog_job_id"]
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
        f"DEVICE={args.device_id} "
        f"ASSIGNED={len(assigned)} "
        f"PENDING={len(pending)}",
        flush=True,
    )

    final = dict(
        previous
    )

    for pos, row in enumerate(
        pending,
        1,
    ):
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
            for rank in range(2):

                phrase = None

                # Diversity is needed here, so phrase
                # generation remains sampled.
                for attempt in range(
                    1,
                    17,
                ):
                    temp = (
                        0.7
                        if attempt <= 8
                        else 1.0
                    )

                    raw = generate(
                        model,
                        tokenizer,
                        device,
                        phrase_prompt(
                            row,
                            aspect,
                            selected,
                        ),
                        sample=True,
                        temperature=temp,
                    )

                    candidate = (
                        clean_phrase(raw)
                    )

                    n = norm(candidate)

                    if not candidate:
                        errors.append(
                            f"{aspect}/{rank}:"
                            "empty_phrase"
                        )
                        continue

                    if len(candidate) > 80:
                        errors.append(
                            f"{aspect}/{rank}:"
                            "phrase_too_long"
                        )
                        continue

                    if n == norm(
                        row["source_span"]
                    ):
                        errors.append(
                            f"{aspect}/{rank}:"
                            "equals_anchor"
                        )
                        continue

                    if any(
                        n == norm(x)
                        for x in selected
                    ):
                        errors.append(
                            f"{aspect}/{rank}:"
                            "duplicate"
                        )
                        continue

                    phrase = candidate
                    break

                if phrase is None:
                    break

                # Translation is deterministic.
                raw_en = generate(
                    model,
                    tokenizer,
                    device,
                    translation_prompt(
                        phrase
                    ),
                    sample=False,
                )

                english = (
                    clean_translation(
                        raw_en
                    )
                )

                if not english:
                    errors.append(
                        f"{aspect}/{rank}:"
                        "empty_translation"
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

        parse_ok = (
            len(
                analogs["category"]
            ) == 2
            and
            len(
                analogs["semantics"]
            ) == 2
            and
            len({
                norm(x["source"])
                for arr
                in analogs.values()
                for x in arr
            }) == 4
        )

        final[
            int(
                row[
                    "analog_job_id"
                ]
            )
        ] = {
            **row,

            "parse_ok":
                parse_ok,

            "parse_error":
                ""
                if parse_ok
                else ";".join(
                    errors[-20:]
                ),

            "analogs":
                analogs
                if parse_ok
                else None,

            "wa_recovery_origin":
                "last31_twostage",

            "construction_method":
                "MT_PATCHER_WA_TWOSTAGE_RECOVERY_V14",
        }

        print(
            f"DEVICE={args.device_id} "
            f"JOB={row['analog_job_id']} "
            f"OK={parse_ok} "
            f"{pos}/{len(pending)}",
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

    fail = sum(
        not x.get("parse_ok")
        for x in final.values()
    )

    print(
        f"LAST31_DEVICE_"
        f"{args.device_id}_COMPLETE "
        f"FAILURES={fail}",
        flush=True,
    )


if __name__ == "__main__":
    main()
