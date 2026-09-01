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

    if not Path(path).exists():
        return rows

    with open(
        path,
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


def extract_pair(raw):
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
            "JSON object missing"
        )

    obj = json.loads(
        raw[start:end + 1]
    )

    if not isinstance(obj, dict):
        raise ValueError(
            "root is not dict"
        )

    src = obj.get("source")
    tgt = obj.get("target")

    if (
        not isinstance(src, str)
        or not isinstance(tgt, str)
    ):
        raise ValueError(
            "source/target invalid"
        )

    src = src.strip()
    tgt = tgt.strip()

    if not src or not tgt:
        raise ValueError(
            "empty bilingual pair"
        )

    return {
        "source": src,
        "target": tgt,
    }


def build_prompt(
    row,
    aspect,
    selected,
):
    original = row["source_span"]

    forbidden = [
        original,
        *selected,
    ]

    forbidden_text = "\n".join(
        f"- {x}"
        for x in forbidden
    )

    if aspect == "category":
        relation = (
            "belong to the same category or type "
            "as the original Chinese phrase"
        )
    else:
        relation = (
            "be semantically associated with, "
            "frequently co-occur with, or naturally "
            "appear in a closely related context to "
            "the original Chinese phrase"
        )

    return f"""You are a Chinese-English language expert.

Generate ONE Chinese-English bilingual phrase pair.

The new Chinese phrase should {relation}.

Prefer relatively rare or translation-challenging knowledge.

Rules:
- Output a word or short phrase, not a full sentence.
- Do not reuse the original phrase.
- Do not reuse any previously selected phrase.
- Give a natural English translation.
- Return ONLY valid JSON.
- Use exactly two keys: source and target.
- No markdown.
- No explanation.

Forbidden Chinese phrases:
{forbidden_text}

Original sentence:
{row["source"]}

Original problematic phrase:
{original}

Required JSON:
{{"source":"中文短语","target":"English translation"}}
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
        default=4,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=20260825,
    )

    args = ap.parse_args()

    device_id = args.device_id

    torch.npu.set_device(
        device_id
    )

    device = f"npu:{device_id}"

    random.seed(
        args.seed + device_id
    )

    torch.manual_seed(
        args.seed + device_id
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
        == device_id
    ]

    output_path = Path(
        args.output
    )

    previous = {}

    for row in load_jsonl(
        output_path
    ):
        if row.get("parse_ok"):
            previous[
                int(row["analog_job_id"])
            ] = row

    pending = [
        row
        for row in assigned
        if int(row["analog_job_id"])
        not in previous
    ]

    print(
        f"DEVICE={device_id} "
        f"ASSIGNED={len(assigned)} "
        f"PREVIOUS={len(previous)} "
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
        f"PAIRWISE_MODEL_READY device={device_id}",
        flush=True,
    )

    final_rows = dict(
        previous
    )

    for pos, row in enumerate(
        pending,
        1,
    ):
        selected = []
        result = {
            "category": [],
            "semantics": [],
        }

        failure = ""

        for aspect in (
            "category",
            "semantics",
        ):
            for rank in range(2):

                pair = None

                for attempt in range(
                    1,
                    9,
                ):
                    chat = [
                        {
                            "role": "user",
                            "content":
                                build_prompt(
                                    row,
                                    aspect,
                                    selected,
                                ),
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
                        out = model.generate(
                            **enc,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            max_new_tokens=128,
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
                        out[0][prompt_len:],
                        skip_special_tokens=True,
                    ).strip()

                    try:
                        candidate = (
                            extract_pair(raw)
                        )

                        n = norm(
                            candidate["source"]
                        )

                        if n == norm(
                            row["source_span"]
                        ):
                            raise ValueError(
                                "equals original anchor"
                            )

                        if any(
                            n == norm(x)
                            for x in selected
                        ):
                            raise ValueError(
                                "duplicate selected analog"
                            )

                        pair = candidate
                        break

                    except Exception as exc:
                        failure = (
                            f"{aspect}/{rank}/"
                            f"attempt{attempt}: "
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        )

                if pair is None:
                    break

                result[
                    aspect
                ].append(pair)

                selected.append(
                    pair["source"]
                )

            if len(
                result[aspect]
            ) != 2:
                break

        parse_ok = (
            len(result["category"]) == 2
            and
            len(result["semantics"]) == 2
            and
            len({
                norm(x["source"])
                for aspect in result.values()
                for x in aspect
            }) == 4
        )

        final_rows[
            int(row["analog_job_id"])
        ] = {
            **row,

            "parse_ok":
                parse_ok,

            "parse_error":
                "" if parse_ok
                else failure,

            "analogs":
                result if parse_ok
                else None,

            "wa_recovery_origin":
                "pairwise_final_retry",

            "construction_method":
                "MT_PATCHER_WA_PAIRWISE_RETRY_V14",
        }

        print(
            f"DEVICE={device_id} "
            f"JOB={row['analog_job_id']} "
            f"OK={parse_ok} "
            f"PROGRESS={pos}/{len(pending)}",
            flush=True,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = output_path.with_suffix(
        ".tmp"
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
            row.get("parse_ok")
        )
        for row
        in final_rows.values()
    )

    print(
        f"PAIRWISE_DEVICE_{device_id}_COMPLETE "
        f"FAILURES={failures}",
        flush=True,
    )


if __name__ == "__main__":
    main()
