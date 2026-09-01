import argparse
import json
import os
import random
import re
from pathlib import Path

import torch
import torch_npu
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


def clean_text(x):
    if not isinstance(x, str):
        return ""
    return x.strip()


def parse_completion(text):
    src = ""
    tgt = ""

    m_src = re.search(
        r"中文句子\s*[:：]\s*(.+)",
        text
    )
    m_tgt = re.search(
        r"英文句子\s*[:：]\s*(.+)",
        text
    )

    if m_src:
        src = m_src.group(1).strip()

    if m_tgt:
        tgt = m_tgt.group(1).strip()

    return src, tgt


def build_prompt(job):
    original = job["source"]
    source_span = job["source_span"]
    correction = job["correction"]
    slot = int(job["pds_slot"]) + 1

    return f"""你是一名高质量的中英平行语料合成器。

下面给出一个学生翻译模型曾经出错的中文短语 P，
以及它在英语中的正确翻译 Q。

原始中文句子：
{original}

P：{source_span}
Q：{correction}

请生成一个新的中英平行句对，用于让学生在新的语境中学习这一翻译知识。

要求：
1. 新中文句子必须原样包含 P。
2. 新英文句子必须自然地包含 Q。
3. 中英文必须语义完全对应。
4. 新句子应与原句保持大致相似的领域、语体或风格。
5. 新句子的具体语义和场景应与原句明显不同，不能只是改几个词。
6. 英文必须自然、完整、符合母语表达。
7. 这是针对该知识点生成的第 {slot} 个独立语境，请尽量避免与其他语境雷同。
8. 不要解释，不要输出分析过程。

严格只输出两行：

中文句子: <新的中文句子>
英文句子: <对应英文翻译>
"""


def load_jobs(path, device_id, world_size):
    jobs = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            jid = int(row["job_id"])

            if jid % world_size == device_id:
                jobs.append(row)

    return jobs


def load_completed(output):
    completed = set()

    if not Path(output).exists():
        return completed

    with open(output, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                completed.add(int(row["job_id"]))
            except Exception:
                continue

    return completed


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
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device-id", type=int, required=True)
    ap.add_argument("--world-size", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--seed", type=int, default=20260825)
    args = ap.parse_args()

    device_id = args.device_id
    device = f"npu:{device_id}"

    torch.npu.set_device(device_id)

    seed = args.seed + device_id
    random.seed(seed)
    torch.manual_seed(seed)

    jobs = load_jobs(
        args.jobs,
        device_id,
        args.world_size
    )

    completed = load_completed(args.output)

    pending = [
        j for j in jobs
        if int(j["job_id"]) not in completed
    ]

    print(
        f"DEVICE={device_id} "
        f"ASSIGNED={len(jobs)} "
        f"COMPLETED={len(completed)} "
        f"PENDING={len(pending)}",
        flush=True
    )

    if not pending:
        print(
            f"PDS_DEVICE_{device_id}_ALREADY_COMPLETE",
            flush=True
        )
        return

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )

    model.to(device)
    model.eval()

    print(
        f"PDS_MODEL_READY device={device_id}",
        flush=True
    )

    Path(args.output).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    done_now = 0

    with open(
        args.output,
        "a",
        encoding="utf-8"
    ) as fout:

        for start in range(
            0,
            len(pending),
            args.batch_size
        ):
            batch_jobs = pending[
                start:start + args.batch_size
            ]

            prompts = [
                apply_chat(
                    tokenizer,
                    build_prompt(job)
                )
                for job in batch_jobs
            ]

            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=768,
            )

            encoded = {
                k: v.to(device)
                for k, v in encoded.items()
            }

            input_len = encoded[
                "input_ids"
            ].shape[1]

            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=1.5,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

            new_tokens = generated[
                :,
                input_len:
            ]

            decoded = tokenizer.batch_decode(
                new_tokens,
                skip_special_tokens=True,
            )

            for job, raw in zip(
                batch_jobs,
                decoded
            ):
                synthesized_source, synthesized_target = (
                    parse_completion(raw)
                )

                result = dict(job)

                result.update(
                    {
                        "generator_model":
                            "Qwen3-8B",
                        "generator_temperature":
                            1.5,
                        "enable_thinking":
                            False,
                        "raw_generation":
                            raw,
                        "synthesized_source":
                            clean_text(
                                synthesized_source
                            ),
                        "synthesized_target":
                            clean_text(
                                synthesized_target
                            ),
                        "parse_ok":
                            bool(
                                synthesized_source
                                and synthesized_target
                            ),
                    }
                )

                fout.write(
                    json.dumps(
                        result,
                        ensure_ascii=False
                    ) + "\n"
                )

                fout.flush()
                done_now += 1

            if (
                done_now == len(batch_jobs)
                or done_now % 80 == 0
            ):
                print(
                    f"DEVICE={device_id} "
                    f"GENERATED={done_now}/"
                    f"{len(pending)}",
                    flush=True
                )

    print(
        f"PDS_DEVICE_{device_id}_COMPLETE "
        f"generated={done_now}",
        flush=True
    )


if __name__ == "__main__":
    main()
