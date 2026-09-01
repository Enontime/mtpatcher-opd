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
# Completion extraction
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
# SacreBLEU 2.5.1 Tokenizer13a equivalent
#
# Verified on 998 WMT24 outputs:
# MAE = 0
# MAX_ABS = 0
# RMSE = 0
# ============================================================

def tokenize_13a(line):
    line = line.replace("<skipped>", "")
    line = line.replace("-\n", "")
    line = line.replace("\n", " ")

    if "&" in line:
        line = line.replace("&quot;", '"')
        line = line.replace("&amp;", "&")
        line = line.replace("&lt;", "<")
        line = line.replace("&gt;", ">")

    # Official Tokenizer13a pads both sides before regexp tokenization.
    line = f" {line} "

    line = re.sub(
        r'([\{-\~\[-\` -\&\(-\+\:-\@\/])',
        r' \1 ',
        line,
    )

    line = re.sub(
        r'([^0-9])([\.,])',
        r'\1 \2 ',
        line,
    )

    line = re.sub(
        r'([\.,])([^0-9])',
        r' \1 \2',
        line,
    )

    line = re.sub(
        r'([0-9])(-)',
        r'\1 \2 ',
        line,
    )

    return " ".join(line.split())


def ngrams(tokens, n):
    return Counter(
        tuple(tokens[i:i+n])
        for i in range(len(tokens) - n + 1)
    )


# ============================================================
# SacreBLEU sentence BLEU equivalent
#
# smooth_method = "exp"
# effective_order = True
# max_ngram_order = 4
#
# Returns BLEU in [0, 100].
# ============================================================

def local_sentence_bleu(hypothesis, reference):
    hyp_tokens = tokenize_13a(hypothesis).split()
    ref_tokens = tokenize_13a(reference).split()

    sys_len = len(hyp_tokens)
    ref_len = len(ref_tokens)

    correct = []
    total = []

    for n in range(1, 5):
        hc = ngrams(hyp_tokens, n)
        rc = ngrams(ref_tokens, n)

        total_n = sum(hc.values())

        correct_n = sum(
            min(count, rc.get(gram, 0))
            for gram, count in hc.items()
        )

        correct.append(correct_n)
        total.append(total_n)

    # SacreBLEU brevity penalty
    bp = 1.0

    if sys_len < ref_len:
        bp = (
            math.exp(1.0 - ref_len / sys_len)
            if sys_len > 0
            else 0.0
        )

    # SacreBLEU early stop when there is no overlap at all
    if not any(correct):
        return 0.0

    precisions = [0.0] * 4

    smooth_mteval = 1.0
    eff_order = 4

    for n in range(1, 5):
        if total[n - 1] == 0:
            break

        # effective_order=True
        eff_order = n

        if correct[n - 1] == 0:
            smooth_mteval *= 2.0

            precisions[n - 1] = (
                100.0
                / (
                    smooth_mteval
                    * total[n - 1]
                )
            )
        else:
            precisions[n - 1] = (
                100.0
                * correct[n - 1]
                / total[n - 1]
            )

    score = bp * math.exp(
        sum(
            math.log(p)
            for p in precisions[:eff_order]
        )
        / eff_order
    )

    return score


# ============================================================
# GRPO reward
#
# BLEU / 100 -> [0, 1]
# ============================================================

reward_calls = 0


def bleu_reward(completions, reference, **kwargs):
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
        local_sentence_bleu(h, r) / 100.0
        for h, r in zip(texts, reference)
    ]

    if any(not math.isfinite(x) for x in rewards):
        raise RuntimeError("non-finite reward")

    if reward_calls <= 3 or reward_calls % 100 == 0:
        print()
        print(f"[REWARD_CALL {reward_calls}]")
        print(
            "reward_min/mean/max =",
            f"{min(rewards):.6f}",
            f"{sum(rewards) / len(rewards):.6f}",
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
    # Frozen 1024 dataset
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
    print("MT-PATCHER PILOT-V2 GRPO BLEU-1024")
    print("=" * 80)

    print("model =", args.model)
    print("train =", args.train)
    print("rows =", len(dataset))
    print("seed =", args.seed)

    print()
    print("reward = sacrebleu251_sentence_bleu_equivalent")
    print("reward_scale = BLEU/100")
    print("num_generations = 4")
    print("max_completion_length = 128")
    print("temperature = 0.7")
    print("top_p = 0.8")
    print("top_k = 20")
    print("learning_rate = 1e-6")
    print("beta = 0")
    print("loss_type = grpo")
    print("scale_rewards = group")
    print("full_finetune = True")

    # ========================================================
    # Reward self-tests
    # ========================================================

    exact = local_sentence_bleu(
        "The company announced the plan.",
        "The company announced the plan.",
    )

    zero = local_sentence_bleu(
        "abc xyz",
        "hello world",
    )

    print()
    print("reward_exact_match_bleu =", exact)
    print("reward_zero_overlap_bleu =", zero)

    if abs(exact - 100.0) > 1e-12:
        raise RuntimeError(
            "BLEU exact-match self-test failed"
        )

    if abs(zero) > 1e-12:
        raise RuntimeError(
            "BLEU zero-overlap self-test failed"
        )

    print("GRPO_BLEU_REWARD_SELFTEST_PASS")

    # ========================================================
    # Tokenizer / leakage check
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
    # Kept identical to chrF-1024 except reward.
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

        save_strategy="steps",
        save_steps=256,
        save_total_limit=4,
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
        reward_funcs=bleu_reward,
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
    # Save final
    # ========================================================

    final_dir = output_dir / "final"

    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    manifest = {
        "experiment": "pilot_v2_grpo_bleu1024",
        "student": args.model,
        "train": args.train,
        "rows": 1024,
        "steps": 1024,
        "seed": args.seed,

        "reward": "sentence_BLEU",
        "reward_scale": "BLEU/100",

        "reward_definition": {
            "sacrebleu_version": "2.5.1",
            "tokenizer": "13a",
            "smooth_method": "exp",
            "effective_order": True,
            "max_ngram_order": 4,
        },

        "reward_parity": {
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
    print("PILOT_V2_GRPO_BLEU1024_PASS")


if __name__ == "__main__":
    main()
