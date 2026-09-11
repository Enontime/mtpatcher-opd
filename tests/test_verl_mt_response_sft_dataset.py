#!/usr/bin/env python3

from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from verl.utils.import_utils import load_extern_object


PROJECT = Path(
    "/workspace/mtpatcher/repo/"
    "MT-Patcher-Reproduction-Ascend"
)

DATASET_PATH = (
    PROJECT
    / "scripts/data/verl_mt_response_sft_dataset.py"
)

DATA = Path(
    "/workspace/mtpatcher/data/"
    "verl_sft_qwen3_06b/human64.parquet"
)

MODEL = "/workspace/mtpatcher/models/Qwen3-0.6B"


tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    local_files_only=True,
)

DatasetCls = load_extern_object(
    str(DATASET_PATH),
    "MTResponseOnlySFTDataset",
)

config = OmegaConf.create({
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
    "ignore_input_ids_mismatch": False,
})

dataset = DatasetCls(
    parquet_files=[str(DATA)],
    tokenizer=tokenizer,
    config=config,
    processor=None,
    max_samples=-1,
)

assert len(dataset) == 64, len(dataset)

eos = tokenizer.eos_token_id
assert eos is not None


def tokenize_plain(text):
    return tokenizer(
        text,
        add_special_tokens=False,
    )["input_ids"]


bad = 0
lengths = []
supervised_counts = []

for i in range(len(dataset)):
    item = dataset[i]
    row = dataset.dataframe.iloc[i].to_dict()
    messages = dataset._build_messages(row)

    # Historical prompt construction.
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    legacy_prompt_ids = tokenize_plain(prompt_text)

    target = row["target_translation"].strip()
    legacy_target_ids = tokenize_plain(target)

    legacy_input_ids = (
        legacy_prompt_ids
        + legacy_target_ids
        + [eos]
    )

    legacy_label_mask = (
        [0] * len(legacy_prompt_ids)
        + [1] * (len(legacy_target_ids) + 1)
    )

    new_ids = item["input_ids"].tolist()
    new_mask = item["loss_mask"].tolist()
    new_pos = item["position_ids"].tolist()

    checks = {
        "input_ids": new_ids == legacy_input_ids,
        "loss_mask": new_mask == legacy_label_mask,
        "position_ids": new_pos == list(range(len(new_ids))),
        "last_is_eos": new_ids[-1] == eos,
        "eos_supervised": new_mask[-1] == 1,
    }

    # Exact causal-mask interpretation used by Verl no-padding sft_loss:
    # token i's log-prob receives loss_mask[i+1].
    #
    # For a single sample, the final prediction position must have zero loss
    # because there is no next token after EOS.
    mask_tensor = torch.tensor(new_mask, dtype=torch.long)

    verl_shifted = torch.roll(
        mask_tensor,
        shifts=-1,
        dims=0,
    )
    verl_shifted[-1] = 0

    expected_prediction_mask = torch.tensor(
        new_mask[1:] + [0],
        dtype=torch.long,
    )

    checks["causal_shift"] = torch.equal(
        verl_shifted,
        expected_prediction_mask,
    )

    if not all(checks.values()):
        bad += 1
        print(
            f"ROW {i} FAIL:",
            {k: v for k, v in checks.items() if not v},
        )

    lengths.append(len(new_ids))
    supervised_counts.append(sum(new_mask))

    if i < 2:
        print(f"\n=== ROW {i} ===")
        print("sequence_length:", len(new_ids))
        print("prompt_tokens:", len(legacy_prompt_ids))
        print("target_tokens:", len(legacy_target_ids))
        print("supervised_tokens:", sum(new_mask))
        print("last_token_id:", new_ids[-1])
        print(
            "last_token:",
            repr(
                tokenizer.decode(
                    [new_ids[-1]],
                    skip_special_tokens=False,
                )
            ),
        )

print("\n=== SUMMARY ===")
print("rows:", len(dataset))
print("bad_rows:", bad)
print("max_length:", max(lengths))
print("min_length:", min(lengths))
print("max_supervised:", max(supervised_counts))
print("min_supervised:", min(supervised_counts))

assert bad == 0, f"{bad} rows failed parity"

print("\nMT_RESPONSE_SFT_64_PARITY_PASS")
