#!/usr/bin/env python3

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer


# ============================================================
# Completion
# ============================================================

def completion_to_text(x):
    if isinstance(x, str):
        return x

    if isinstance(x, dict):
        return str(x.get("content", ""))

    if isinstance(x, list):
        for item in reversed(x):
            if isinstance(item, dict) and "content" in item:
                return str(item["content"])
        return "".join(str(v) for v in x)

    return str(x)


# ============================================================
# Exact SacreBLEU sentence chrF2 equivalent
#
# Verified against:
# sacrebleu==2.5.1
# sentence_chrf(...)
#
# parity over 998 WMT24 outputs:
# MAE=0, MAX_ABS=0, RMSE=0
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
# Reward
# ============================================================

reward_calls = 0


def chrf_reward(completions, reference, **kwargs):
    global reward_calls
    reward_calls += 1

    texts = [
        completion_to_text(x)
        for x in completions
    ]

    if len(texts) != len(reference):
        raise RuntimeError(
            f"reward length mismatch: "
            f"{len(texts)} vs {len(reference)}"
        )

    rewards = [
        local_chrf2(h, r)
        for h, r in zip(texts, reference)
    ]

    if any(not math.isfinite(x) for x in rewards):
        raise RuntimeError("non-finite reward")

    # Sparse diagnostics: don't flood logs.
    if reward_calls <= 3 or reward_calls % 100 == 0:
        print()
        print(f"[REWARD_CALL {reward_calls}]")
        print(
            "reward_min/mean/max =",
            f"{min(rewards):.6f}",
            f"{sum(rewards)/len(rewards):.6f}",
            f"{max(rewards):.6f}",
        )
        print("sample_hyp =", repr(texts[0][:240]))
        print("sample_ref =", repr(reference[0][:240]))

    return rewards


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--seed", type=int, default=20260821)

    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # Load exactly the frozen 1024 dataset
    # ========================================================

    raw = []

    with open(args.train, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw.append(json.loads(line))

    if len(raw) != 1024:
        raise RuntimeError(
            f"Expected frozen 1024 rows, got {len(raw)}"
        )

    records = []

    for row in raw:
        records.append({
            "prompt": row["messages"],
            "reference": row["reference"],
            "source": row["source"],
            "pilot_index": row["pilot_index"],
            "source_index": row["source_index"],
        })

    dataset = Dataset.from_list(records)

    print("=" * 80)
    print("MT-PATCHER PILOT-V2 GRPO chrF-1024")
    print("=" * 80)

    print("model =", args.model)
    print("train =", args.train)
    print("rows =", len(dataset))
    print("seed =", args.seed)

    print()
    print("reward = exact_sentence_chrF2")
    print("num_generations = 4")
    print("max_completion_length = 128")
    print("temperature = 0.7")
    print("top_p = 0.8")
    print("top_k = 20")
    print("learning_rate = 1e-6")
    print("beta = 0")
    print("loss_type = grpo")
    print("full_finetune = True")

    # ========================================================
    # Reward self test
    # ========================================================

    exact = local_chrf2(
        "The company announced the plan.",
        "The company announced the plan.",
    )

    print()
    print("reward_exact_match =", exact)

    if abs(exact - 1.0) > 1e-12:
        raise RuntimeError("chrF reward self-test failed")

    print("GRPO_REWARD_SELFTEST_PASS")

    # ========================================================
    # Tokenizer
    # ========================================================

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    rendered = tokenizer.apply_chat_template(
        records[0]["prompt"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    if records[0]["reference"] in rendered:
        raise RuntimeError("REFERENCE LEAKAGE")

    print("GRPO_REFERENCE_LEAKAGE_CHECK_PASS")

    # ========================================================
    # Model
    # ========================================================

    print()
    print("Loading model...")

    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="sdpa",
    )

    model.config.tie_word_embeddings = False
    model.config.use_cache = False

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "model_load_seconds =",
        round(time.time() - t0, 3),
    )

    print("total_parameters =", total_params)
    print("trainable_parameters =", trainable_params)
    print(
        "full_finetune =",
        total_params == trainable_params,
    )

    # ========================================================
    # GRPO config
    #
    # Smoke established:
    # 64 rows -> 64 optimizer steps / epoch.
    # Therefore frozen 1024 rows -> 1024 steps / epoch.
    # ========================================================

    config = GRPOConfig(
        output_dir=str(output_dir),

        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,

        max_steps=1024,

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
        logging_steps=10,
        logging_first_step=True,

        report_to="none",

        # Learning-curve checkpoints
        save_strategy="steps",
        save_steps=256,
        save_total_limit=4,

        # We only need models for evaluation.
        # Avoid huge Adam optimizer-state checkpoints.
        save_only_model=True,

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

        beta=0.0,
        loss_type="grpo",
        scale_rewards="group",

        log_completions=False,

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

    # ========================================================
    # Train
    # ========================================================

    start = time.time()

    result = trainer.train()

    elapsed = time.time() - start

    print()
    print("GRPO_TRAIN_PASS")
    print("train_seconds =", elapsed)

    print()
    print("===== FINAL TRAIN METRICS =====")

    for key, value in sorted(result.metrics.items()):
        print(f"{key} = {value}")

    # ========================================================
    # Final model
    # ========================================================

    final_dir = output_dir / "final"

    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    manifest = {
        "experiment": "pilot_v2_grpo_chrf1024",
        "student": args.model,
        "train": args.train,
        "rows": 1024,
        "steps": 1024,
        "seed": args.seed,

        "reward": "sentence_chrF2",
        "reward_parity": {
            "sacrebleu_version": "2.5.1",
            "rows_checked": 998,
            "MAE": 0.0,
            "MAX_ABS": 0.0,
            "RMSE": 0.0,
        },

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

        "checkpoints": [
            256,
            512,
            768,
            1024,
        ],
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("final_model =", final_dir)
    print("PILOT_V2_GRPO_CHRF1024_PASS")


if __name__ == "__main__":
    main()
