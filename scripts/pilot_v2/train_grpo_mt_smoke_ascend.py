#!/usr/bin/env python3

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import torch
import torch_npu
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from trl import GRPOConfig, GRPOTrainer


# ============================================================
# Completion extraction
# ============================================================

def completion_to_text(x):
    if isinstance(x, str):
        return x

    if isinstance(x, dict):
        return str(x.get("content", ""))

    if isinstance(x, list):
        # conversational completion:
        # [{"role": "assistant", "content": "..."}]
        for item in reversed(x):
            if isinstance(item, dict) and "content" in item:
                return str(item["content"])

        return "".join(str(v) for v in x)

    return str(x)


# ============================================================
# Dependency-free chrF2-like reward
#
# Character n-grams: 1..6
# beta = 2
# whitespace removed
# score returned in [0, 1]
# ============================================================

def normalize_for_chrf(text):
    return re.sub(r"\s+", "", text.strip())


def ngram_counter(text, n):
    if len(text) < n:
        return Counter()

    return Counter(
        text[i:i+n]
        for i in range(len(text) - n + 1)
    )


def local_chrf2(hypothesis, reference, max_order=6, beta=2.0):
    hyp = normalize_for_chrf(hypothesis)
    ref = normalize_for_chrf(reference)

    if not hyp or not ref:
        return 0.0

    precisions = []
    recalls = []

    for n in range(1, max_order + 1):
        hc = ngram_counter(hyp, n)
        rc = ngram_counter(ref, n)

        htotal = sum(hc.values())
        rtotal = sum(rc.values())

        if htotal == 0 or rtotal == 0:
            continue

        common = sum(
            min(count, rc.get(gram, 0))
            for gram, count in hc.items()
        )

        precisions.append(common / htotal)
        recalls.append(common / rtotal)

    if not precisions:
        return 0.0

    p = sum(precisions) / len(precisions)
    r = sum(recalls) / len(recalls)

    beta2 = beta * beta
    denom = beta2 * p + r

    if denom <= 0:
        return 0.0

    return (1.0 + beta2) * p * r / denom


# ============================================================
# GRPO reward function
# ============================================================

_reward_calls = 0


def chrf_reward(completions, reference, **kwargs):
    global _reward_calls
    _reward_calls += 1

    texts = [completion_to_text(x) for x in completions]

    if len(texts) != len(reference):
        raise RuntimeError(
            f"reward length mismatch: "
            f"completions={len(texts)} references={len(reference)}"
        )

    rewards = [
        local_chrf2(hyp, ref)
        for hyp, ref in zip(texts, reference)
    ]

    if _reward_calls <= 3:
        print()
        print(f"[REWARD_CALL {_reward_calls}]")
        print("num_completions =", len(texts))
        print(
            "reward_min/mean/max =",
            f"{min(rewards):.6f}",
            f"{sum(rewards)/len(rewards):.6f}",
            f"{max(rewards):.6f}",
        )

        print("sample_hyp =", repr(texts[0][:300]))
        print("sample_ref =", repr(reference[0][:300]))

    if any(not math.isfinite(x) for x in rewards):
        raise RuntimeError("non-finite reward detected")

    return rewards


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=20)

    parser.add_argument("--seed", type=int, default=20260821)

    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load frozen pilot
    # --------------------------------------------------------

    raw = []

    with open(args.train, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw.append(json.loads(line))

    if args.rows > len(raw):
        raise RuntimeError(
            f"requested rows={args.rows}, dataset has {len(raw)}"
        )

    raw = raw[:args.rows]

    records = []

    for row in raw:
        messages = row["messages"]
        reference = row["reference"]

        if not isinstance(messages, list) or not messages:
            raise RuntimeError("invalid messages")

        if not reference:
            raise RuntimeError("empty reference")

        records.append({
            "prompt": messages,
            "reference": reference,
            "source": row["source"],
            "pilot_index": row["pilot_index"],
            "source_index": row["source_index"],
        })

    dataset = Dataset.from_list(records)

    print("=" * 80)
    print("MT-PATCHER PILOT-V2 GRPO SMOKE")
    print("=" * 80)

    print("model =", args.model)
    print("train =", args.train)
    print("rows =", len(dataset))
    print("max_steps =", args.max_steps)
    print("seed =", args.seed)

    print()
    print("reward = local_chrF2")
    print("num_generations = 4")
    print("temperature = 0.7")
    print("top_p = 0.8")
    print("top_k = 20")
    print("thinking = False")
    print("beta = 0")
    print("loss_type = grpo")

    # Reward sanity check
    exact = local_chrf2(
        "The company announced the plan.",
        "The company announced the plan.",
    )

    unrelated = local_chrf2(
        "A completely different sentence.",
        "The company announced the plan.",
    )

    print()
    print("reward_exact_match =", exact)
    print("reward_unrelated =", unrelated)

    if abs(exact - 1.0) > 1e-8:
        raise RuntimeError(
            f"reward exact-match self-test failed: {exact}"
        )

    print("GRPO_REWARD_SELFTEST_PASS")

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print()
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    # Verify non-thinking template on one example
    rendered = tokenizer.apply_chat_template(
        records[0]["prompt"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    print()
    print("===== CHAT TEMPLATE SAMPLE =====")
    print(rendered[:1000])

    if records[0]["reference"] in rendered:
        raise RuntimeError(
            "REFERENCE LEAKAGE: reference appears in prompt"
        )

    print("GRPO_REFERENCE_LEAKAGE_CHECK_PASS")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("Loading model...")

    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="sdpa",
    )

    # Qwen3 input embeddings and lm_head share the same parameter.
    # Preserve tied-weight metadata so save/reload remains consistent.
    model.config.tie_word_embeddings = True
    model.tie_weights()
    model.config.use_cache = False

    print(
        "model_load_seconds =",
        round(time.time() - t0, 3),
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("total_parameters =", total_params)
    print("trainable_parameters =", trainable_params)
    print("full_finetune =", trainable_params == total_params)

    # --------------------------------------------------------
    # GRPO
    # --------------------------------------------------------

    config = GRPOConfig(
        output_dir=str(output_dir),

        # 4 completions form one GRPO group.
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,

        max_steps=args.max_steps,

        learning_rate=1e-6,
        lr_scheduler_type="constant",
        warmup_steps=0,

        bf16=True,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },

        max_grad_norm=1.0,

        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        report_to="none",

        save_strategy="no",

        seed=args.seed,
        data_seed=args.seed,

        remove_unused_columns=False,

        num_generations=4,
        max_completion_length=128,

        temperature=0.7,
        top_p=0.8,
        top_k=20,

        chat_template_kwargs={
            "enable_thinking": False,
        },

        # First controlled smoke: plain GRPO.
        beta=0.0,
        loss_type="grpo",
        scale_rewards="group",

        log_completions=True,
        num_completions_to_print=2,

        use_vllm=False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=chrf_reward,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print()
    print("GRPO_TRAINER_INIT_PASS")

    t1 = time.time()

    result = trainer.train()

    train_seconds = time.time() - t1

    print()
    print("GRPO_TRAIN_PASS")
    print("train_seconds =", train_seconds)

    print()
    print("===== TRAIN METRICS =====")

    for k, v in sorted(result.metrics.items()):
        print(f"{k} = {v}")

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    final_dir = output_dir / "final"

    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    manifest = {
        "experiment": "pilot_v2_grpo_smoke",
        "model": args.model,
        "train": args.train,
        "rows": len(dataset),
        "max_steps": args.max_steps,
        "seed": args.seed,
        "reward": "local_chrF2",
        "num_generations": 4,
        "max_completion_length": 128,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "enable_thinking": False,
        "learning_rate": 1e-6,
        "beta": 0.0,
        "loss_type": "grpo",
        "scale_rewards": "group",
        "full_finetune": True,
    }

    (output_dir / "smoke_manifest.json").write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("final_model =", final_dir)

    if not final_dir.exists():
        raise RuntimeError("final model directory missing")

    print("PILOT_V2_GRPO_SMOKE_PASS")


if __name__ == "__main__":
    main()
