import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def batched_generate(
    model,
    tokenizer,
    prompts,
    batch_size,
    max_new_tokens,
):
    outputs_all = []

    for start in range(0, len(prompts), batch_size):
        part = prompts[start:start + batch_size]

        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for p in part
        ]

        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )

        encoded = {
            k: v.to(model.device)
            for k, v in encoded.items()
        }

        input_len = encoded["input_ids"].shape[1]

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                repetition_penalty=1.05,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        new_tokens = generated[:, input_len:]

        decoded = tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True,
        )

        decoded = [
            x.strip()
            for x in decoded
        ]

        outputs_all.extend(decoded)

        print(
            f"generated={min(start + len(part), len(prompts))}"
            f"/{len(prompts)}",
            flush=True,
        )

    return outputs_all


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)

    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=512)

    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    if hasattr(torch, "npu"):
        torch.npu.manual_seed_all(args.seed)

    rows = load_jsonl(args.input)

    print("=" * 80)
    print("PEGRL TWO-STAGE EVALUATION")
    print("=" * 80)

    print("model =", args.model)
    print("rows =", len(rows))
    print("seed =", args.seed)
    print("temperature = 0.6")
    print("top_p = 0.95")
    print("top_k = 20")
    print("repetition_penalty = 1.05")
    print("thinking = False")
    print("max_new_tokens =", args.max_new_tokens)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
        trust_remote_code=True,
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    model = model.to("npu:0")
    model.eval()

    sources = [
        x["source"]
        for x in rows
    ]

    refs = [
        x["reference"]
        for x in rows
    ]

    # ========================================================
    # Stage 1: MT
    # ========================================================

    mt_prompts = [
        (
            "Translate the following text into English "
            "without additional explanations:\n"
            + src
        )
        for src in sources
    ]

    print()
    print("===== STAGE 1: TRANSLATION =====")

    t0 = time.time()

    drafts = batched_generate(
        model=model,
        tokenizer=tokenizer,
        prompts=mt_prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    print(
        "stage1_seconds =",
        time.time() - t0,
    )

    # ========================================================
    # Stage 2: PE
    # Exact Appendix-D / official use_test_prompt form.
    # ========================================================

    pe_prompts = [
        (
            "Given the source text:\n"
            f"{src}\n"
            "Improve the following draft English translation "
            "into a high-quality English version, "
            "without explanations:\n"
            f"{draft}"
        )
        for src, draft in zip(sources, drafts)
    ]

    print()
    print("===== STAGE 2: POST-EDIT =====")

    t1 = time.time()

    post_edits = batched_generate(
        model=model,
        tokenizer=tokenizer,
        prompts=pe_prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    print(
        "stage2_seconds =",
        time.time() - t1,
    )

    draft_path = out_dir / "draft_predictions.jsonl"
    pe_path = out_dir / "postedit_predictions.jsonl"

    with draft_path.open("w", encoding="utf-8") as fd, \
         pe_path.open("w", encoding="utf-8") as fp:

        for i, (row, draft, pe) in enumerate(
            zip(rows, drafts, post_edits)
        ):
            common = {
                "index": i,
                "source": row["source"],
                "reference": row["reference"],
            }

            d = {
                **common,
                "student_translation": draft,
                "draft_translation": draft,
                "postedit_translation": pe,
                "stage": "mt",
            }

            p = {
                **common,
                "student_translation": pe,
                "draft_translation": draft,
                "postedit_translation": pe,
                "stage": "postedit",
            }

            fd.write(
                json.dumps(
                    d,
                    ensure_ascii=False,
                ) + "\n"
            )

            fp.write(
                json.dumps(
                    p,
                    ensure_ascii=False,
                ) + "\n"
            )

    noop = sum(
        a.strip() == b.strip()
        for a, b in zip(drafts, post_edits)
    )

    print()
    print("draft_output =", draft_path)
    print("postedit_output =", pe_path)

    print(
        "exact_noop =",
        f"{noop}/{len(rows)}",
        f"({100.0 * noop / len(rows):.2f}%)",
    )

    print("PEGRL_TWO_STAGE_GENERATION_PASS")


if __name__ == "__main__":
    main()
