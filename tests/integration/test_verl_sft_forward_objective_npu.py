#!/usr/bin/env python3
"""
Gate 3: numerical forward-objective parity for MT response-only SFT.

This is deliberately an integration diagnostic, not a training loop.

Checks on representative Human6565 examples:
1. independently reconstructed historical supervised target tokens
   equal the adapter's supervised tokens;
2. target + EOS token counts agree;
3. HF causal-LM loss with historical labels agrees numerically with
   the response-only NLL implied by Verl's loss_mask semantics;
4. losses are finite.

Training/distributed infrastructure remains owned by Verl.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    "verl_sft_qwen3_06b/human6565.parquet"
)

MODEL = "/workspace/mtpatcher/models/Qwen3-0.6B"

DEVICE = "npu:0"
MAX_LENGTH = 1024

# BF16 model outputs are converted to FP32 for the explicit NLL calculation.
# We compare two loss formulations applied to the same forward pass.
ATOL = 5e-4


def plain_tokenize(tokenizer, text: str) -> list[int]:
    ids = tokenizer(
        text,
        add_special_tokens=False,
    )["input_ids"]

    if not isinstance(ids, list):
        raise TypeError("tokenizer did not return list input_ids")

    return ids


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
    "max_length": MAX_LENGTH,
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

assert len(dataset) == 6565


# ------------------------------------------------------------
# Choose representative lengths WITHOUT using the model.
# ------------------------------------------------------------

lengths = np.empty(len(dataset), dtype=np.int64)

for i in range(len(dataset)):
    lengths[i] = dataset[i]["input_ids"].numel()

median_value = float(np.percentile(lengths, 50))
p99_value = float(np.percentile(lengths, 99))

median_idx = int(np.argmin(np.abs(lengths - median_value)))
p99_idx = int(np.argmin(np.abs(lengths - p99_value)))
max_idx = int(np.argmax(lengths))

selected = []

for idx in [0, median_idx, p99_idx, max_idx]:
    if idx not in selected:
        selected.append(idx)

print("=== SELECTED ROWS ===")
for idx in selected:
    print(
        "row=",
        idx,
        "length=",
        int(lengths[idx]),
    )


# ------------------------------------------------------------
# Load ONE model, ONE NPU, inference/eval only.
# ------------------------------------------------------------

if not torch.npu.is_available():
    raise RuntimeError("Ascend NPU is required for Gate 3")

torch.npu.set_device(0)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
)

model.to(DEVICE)
model.eval()

eos = tokenizer.eos_token_id

if eos is None:
    raise RuntimeError("tokenizer.eos_token_id is None")


results = []

with torch.no_grad():
    for idx in selected:
        item = dataset[idx]

        ids_cpu = item["input_ids"].to(torch.long)
        mask_cpu = item["loss_mask"].to(torch.long)

        row = dataset.dataframe.iloc[idx].to_dict()
        messages = dataset._build_messages(row)

        # ----------------------------------------------------
        # Independent historical objective reconstruction.
        # Do not derive this from loss_mask.
        # ----------------------------------------------------

        prompt_text = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        legacy_prompt_ids = plain_tokenize(
            tokenizer,
            prompt_text,
        )

        target = row["target_translation"].strip()

        legacy_target_ids = plain_tokenize(
            tokenizer,
            target,
        )

        legacy_ids = (
            legacy_prompt_ids
            + legacy_target_ids
            + [eos]
        )

        legacy_labels = (
            [-100] * len(legacy_prompt_ids)
            + legacy_target_ids
            + [eos]
        )

        # Current migration happens to have exact serialization parity.
        # Keep this as a diagnostic, NOT as the long-term semantic contract.
        serialization_equal = (
            ids_cpu.tolist() == legacy_ids
        )

        if not serialization_equal:
            raise AssertionError(
                f"row {idx}: current migration lost serialized parity"
            )

        labels_cpu = torch.tensor(
            legacy_labels,
            dtype=torch.long,
        )

        supervised_legacy = labels_cpu[
            labels_cpu != -100
        ].tolist()

        supervised_adapter = ids_cpu[
            mask_cpu.bool()
        ].tolist()

        if supervised_legacy != supervised_adapter:
            raise AssertionError(
                f"row {idx}: supervised token IDs differ"
            )

        if len(supervised_adapter) != len(legacy_target_ids) + 1:
            raise AssertionError(
                f"row {idx}: supervised token count mismatch"
            )

        if supervised_adapter[-1] != eos:
            raise AssertionError(
                f"row {idx}: final supervised token is not EOS"
            )

        # ----------------------------------------------------
        # ONE actual Qwen3 forward.
        # ----------------------------------------------------

        input_ids = ids_cpu.unsqueeze(0).to(DEVICE)
        labels = labels_cpu.unsqueeze(0).to(DEVICE)

        output = model(
            input_ids=input_ids,
            labels=labels,
            use_cache=False,
        )

        legacy_loss = output.loss.float()

        if not torch.isfinite(legacy_loss):
            raise AssertionError(
                f"row {idx}: HF legacy loss is non-finite"
            )

        # ----------------------------------------------------
        # Verl response-mask semantics.
        #
        # A causal logit at position i predicts token i+1.
        # Therefore dataset loss_mask on token positions becomes
        # mask[1:] on prediction positions.
        #
        # This is the per-sample equivalent of Verl v0.9's
        # left-shifted loss-mask semantics.
        # ----------------------------------------------------

        shift_logits = (
            output.logits[:, :-1, :]
            .float()
            .contiguous()
        )

        shift_targets = input_ids[:, 1:].contiguous()

        token_nll = F.cross_entropy(
            shift_logits.reshape(
                -1,
                shift_logits.shape[-1],
            ),
            shift_targets.reshape(-1),
            reduction="none",
        )

        prediction_mask = (
            mask_cpu[1:]
            .to(DEVICE)
            .bool()
            .reshape(-1)
        )

        verl_supervised_count = int(
            prediction_mask.sum().item()
        )

        expected_count = len(legacy_target_ids) + 1

        if verl_supervised_count != expected_count:
            raise AssertionError(
                f"row {idx}: Verl prediction-mask count "
                f"{verl_supervised_count} != {expected_count}"
            )

        verl_equiv_loss = (
            token_nll[prediction_mask].mean()
        )

        if not torch.isfinite(verl_equiv_loss):
            raise AssertionError(
                f"row {idx}: Verl-equivalent loss is non-finite"
            )

        diff = abs(
            float(legacy_loss.item())
            - float(verl_equiv_loss.item())
        )

        ok = diff <= ATOL

        print(f"\n=== ROW {idx} ===")
        print("sequence_length:", len(legacy_ids))
        print("prompt_tokens:", len(legacy_prompt_ids))
        print("target_tokens:", len(legacy_target_ids))
        print(
            "supervised_tokens:",
            len(supervised_adapter),
        )
        print(
            "serialization_equal:",
            serialization_equal,
        )
        print(
            "supervised_ids_equal:",
            supervised_legacy == supervised_adapter,
        )
        print(
            "legacy_hf_loss:",
            float(legacy_loss.item()),
        )
        print(
            "verl_equiv_loss:",
            float(verl_equiv_loss.item()),
        )
        print("abs_diff:", diff)
        print("within_tolerance:", ok)

        if not ok:
            raise AssertionError(
                f"row {idx}: numerical loss mismatch "
                f"{diff} > {ATOL}"
            )

        results.append(
            {
                "row": idx,
                "length": len(legacy_ids),
                "supervised": len(supervised_adapter),
                "legacy_loss": float(
                    legacy_loss.item()
                ),
                "verl_equiv_loss": float(
                    verl_equiv_loss.item()
                ),
                "abs_diff": diff,
            }
        )


assert len(results) == len(selected)

print("\n=== GATE 3 SUMMARY ===")
print("rows_tested:", len(results))
print(
    "max_abs_diff:",
    max(x["abs_diff"] for x in results),
)

print(
    "\nVERL_SFT_FORWARD_OBJECTIVE_PARITY_PASS"
)
