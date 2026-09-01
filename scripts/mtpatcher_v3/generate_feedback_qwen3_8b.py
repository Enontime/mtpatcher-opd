import argparse
import json
import time
from pathlib import Path

import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = """You are a high-precision machine translation error annotator.

Your task is to evaluate a Chinese-to-English translation draft.

Focus on translation correctness and faithfulness:
- mistranslation
- omission
- addition
- terminology
- named entities
- numbers
- grammar that changes or obscures meaning
- serious fluency problems

Do not rewrite an already correct translation merely because you prefer another wording.
Do not invent errors.
You do not have access to a human reference translation.

Return JSON only.
"""


def build_messages(source, draft):
    user = f"""Source Chinese text:
{source}

Student English translation:
{draft}

Determine whether the Student translation contains a substantive translation error.

Return exactly one JSON object with this schema:

{{
  "has_error": true,
  "errors": [
    {{
      "source_span": "the relevant span copied from the Chinese source",
      "translation_span": "the problematic span copied from the Student translation",
      "error_type": "Mistranslation | Omission | Addition | Terminology | Named Entity | Number | Grammar | Fluency | Other",
      "explanation": "why this is an error",
      "correction": "the corrected English rendering for this error"
    }}
  ],
  "post_edit": "the complete corrected English translation"
}}

If the translation is already accurate, complete, and natural enough, return:

{{
  "has_error": false,
  "errors": [],
  "post_edit": "{draft}"
}}

When has_error is false, post_edit must copy the Student translation exactly.
When has_error is true, post_edit must be a complete translation integrating all listed corrections.
Do not output markdown or any text outside the JSON object."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            x = json.loads(line)

            if "index" not in x:
                raise RuntimeError(f"line {line_no}: missing index")
            if not x.get("source", "").strip():
                raise RuntimeError(f"line {line_no}: empty source")
            if not x.get("student_translation", "").strip():
                raise RuntimeError(
                    f"line {line_no}: empty student_translation"
                )

            rows.append(x)
    return rows


def load_done(path):
    done = set()
    if not path.exists():
        return done

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                x = json.loads(line)
                done.add(int(x["index"]))
            except Exception:
                pass
    return done


def parse_json(raw):
    text = raw.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    candidates = [text]

    left = text.find("{")
    right = text.rfind("}")

    if left >= 0 and right > left:
        candidates.append(text[left:right + 1])

    obj = None
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                obj = value
                break
        except Exception:
            pass

    if obj is None:
        return {
            "parse_ok": False,
            "has_error": None,
            "errors": [],
            "post_edit": None,
        }

    has_error = obj.get("has_error")

    if isinstance(has_error, str):
        z = has_error.strip().lower()
        if z in {"true", "yes", "1"}:
            has_error = True
        elif z in {"false", "no", "0"}:
            has_error = False
        else:
            has_error = None

    if not isinstance(has_error, bool):
        has_error = None

    errors = obj.get("errors", [])
    if not isinstance(errors, list):
        errors = []

    clean_errors = []
    for error in errors:
        if isinstance(error, dict):
            clean_errors.append(error)

    post_edit = obj.get("post_edit")
    if not isinstance(post_edit, str):
        post_edit = None
    elif not post_edit.strip():
        post_edit = None
    else:
        post_edit = post_edit.strip()

    parse_ok = (
        isinstance(has_error, bool)
        and post_edit is not None
        and isinstance(errors, list)
    )

    return {
        "parse_ok": parse_ok,
        "has_error": has_error,
        "errors": clean_errors,
        "post_edit": post_edit,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-prompt-tokens", type=int, default=1536)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = read_jsonl(input_path)
    done = load_done(output_path)

    pending = [
        x for x in rows
        if int(x["index"]) not in done
    ]

    print("=" * 80)
    print("MTPATCHER V3 QWEN3-8B FEEDBACKER")
    print("=" * 80)

    print("model =", args.model)
    print("input =", input_path)
    print("output =", output_path)

    print("rows =", len(rows))
    print("already_done =", len(done))
    print("pending =", len(pending))

    print("batch_size =", args.batch_size)
    print("max_new_tokens =", args.max_new_tokens)

    print("npu_available =", torch.npu.is_available())
    print("npu_count_visible =", torch.npu.device_count())

    if not torch.npu.is_available():
        raise RuntimeError("NPU unavailable")

    device = torch.device("npu:0")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )

    model.to(device)
    model.eval()

    print("model loaded")
    print("device =", device)

    generated = 0
    t_all = time.time()

    with output_path.open(
        "a",
        encoding="utf-8",
        buffering=1,
    ) as fout:

        for start in range(
            0,
            len(pending),
            args.batch_size,
        ):
            batch = pending[
                start:start + args.batch_size
            ]

            prompts = []

            for x in batch:
                messages = build_messages(
                    x["source"],
                    x["student_translation"],
                )

                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )

                prompts.append(rendered)

            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_prompt_tokens,
            )

            enc = {
                k: v.to(device)
                for k, v in enc.items()
            }

            prompt_width = enc["input_ids"].shape[1]

            t0 = time.time()

            with torch.inference_mode():
                out = model.generate(
                    **enc,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

            torch.npu.synchronize()

            new_ids = out[:, prompt_width:]

            texts = tokenizer.batch_decode(
                new_ids,
                skip_special_tokens=True,
            )

            for row, raw, ids in zip(
                batch,
                texts,
                new_ids,
            ):
                parsed = parse_json(raw)

                nonpad = int(
                    (
                        ids
                        != tokenizer.pad_token_id
                    )
                    .sum()
                    .item()
                )

                result = {
                    "index": int(row["index"]),
                    "source": row["source"],
                    "student_translation":
                        row["student_translation"],

                    "parse_ok":
                        parsed["parse_ok"],
                    "has_error":
                        parsed["has_error"],
                    "errors":
                        parsed["errors"],
                    "post_edit":
                        parsed["post_edit"],

                    "raw_feedback": raw,

                    "new_tokens": nonpad,
                    "hit_max_new_tokens":
                        nonpad >= args.max_new_tokens,

                    "feedbacker_model":
                        args.model,
                    "enable_thinking": False,
                    "do_sample": False,
                }

                fout.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

                generated += 1

            dt = time.time() - t0

            print(
                f"generated={generated}/{len(pending)} "
                f"batch_seconds={dt:.3f}",
                flush=True,
            )

    total_time = time.time() - t_all

    print(
        "generated_this_run =",
        generated,
    )

    print(
        "mean_seconds_per_generated_row =",
        total_time / max(generated, 1),
    )

    print(
        "MTPATCHER_V3_FEEDBACK_GENERATION_PASS"
    )


if __name__ == "__main__":
    main()
