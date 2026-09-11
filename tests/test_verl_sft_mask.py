#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import torch

from omegaconf import OmegaConf
from transformers import AutoTokenizer
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


MODEL = "/workspace/mtpatcher/models/Qwen3-0.6B"
DATA = Path(
    "/workspace/mtpatcher/data/"
    "verl_sft_qwen3_06b/human64.parquet"
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    local_files_only=True,
)

config = OmegaConf.create(
    {
        "messages_key": "messages",
        "tools_key": "tools",
        "enable_thinking_key": "enable_thinking",
        "enable_thinking_default": False,
        "pad_mode": "no_padding",
        "max_length": 1024,
        "truncation": "error",
        "apply_chat_template_kwargs": {
            "enable_thinking": False,
        },
        "use_shm": False,
        "ignore_input_ids_mismatch": True,
    }
)

dataset = MultiTurnSFTDataset(
    parquet_files=[str(DATA)],
    tokenizer=tokenizer,
    config=config,
    processor=None,
    max_samples=-1,
)

df = pd.read_parquet(DATA)

print("dataset_len:", len(dataset))

bad = 0

for i in range(min(64, len(dataset))):
    item = dataset[i]

    input_ids = item["input_ids"]
    loss_mask = item["loss_mask"].bool()

    assert isinstance(input_ids, torch.Tensor)
    assert input_ids.shape == loss_mask.shape

    ids = input_ids.tolist()
    mask = loss_mask.tolist()

    target = df.iloc[i]["target_translation"]

    # Exact target tokenization without chat-template wrapper.
    target_ids = tokenizer.encode(
        target,
        add_special_tokens=False,
    )

    # Locate target tokens exactly inside the final dataset sequence.
    starts = []
    for j in range(len(ids) - len(target_ids) + 1):
        if ids[j:j + len(target_ids)] == target_ids:
            starts.append(j)

    if len(starts) != 1:
        print(f"\nSAMPLE {i}: TARGET_LOCATION_ERROR")
        print("matches:", starts)
        bad += 1
        continue

    start = starts[0]
    end = start + len(target_ids)

    target_mask = mask[start:end]

    missing = [
        k for k, m in enumerate(target_mask)
        if not m
    ]

    print(f"\n=== SAMPLE {i} ===")
    print("target_start:", start)
    print("target_tokens:", len(target_ids))
    print("target_supervised:", sum(target_mask))
    print("target_missing:", len(missing))

    if missing:
        bad += 1

        prefix_len = 0
        for m in target_mask:
            if m:
                break
            prefix_len += 1

        print("masked_target_prefix_tokens:", prefix_len)

        prefix_ids = target_ids[:prefix_len]
        print(
            "masked_target_prefix_text:",
            repr(
                tokenizer.decode(
                    prefix_ids,
                    skip_special_tokens=False,
                )
            ),
        )

    # The user instruction/source must never be supervised.
    user_text = df.iloc[i]["messages"][0]["content"]
    supervised_ids = [
        tok for tok, m in zip(ids, mask)
        if m
    ]
    supervised_text = tokenizer.decode(
        supervised_ids,
        skip_special_tokens=False,
    )

    assert user_text not in supervised_text

print("\nMASK_BAD_ROWS:", bad)
print("MASK_TOTAL_ROWS:", min(64, len(dataset)))

assert bad == 0, (
    f"Verl SFT response-only parity failed on {bad} rows"
)

print("VERL_SFT_MASK_PARITY_PASS")
