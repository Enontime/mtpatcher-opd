#!/usr/bin/env python3
"""Static optimizer contracts shared by both Chemistry KC-SFT learning rates."""
from __future__ import annotations

from typing import Any

import _clean_sft_parity_core_v1 as c
import _clean_sft_parity_kc_v1 as kc


def build(arm: str) -> dict[str, Any]:
    cfg = kc.ARM_CONFIG[arm]
    return {
        "optimizer": "AdamW(foreach=False)",
        "lr": cfg["lr"],
        "weight_decay": c.WEIGHT_DECAY,
        "scheduler": "linear",
        "warmup_ratio": c.WARMUP_RATIO,
        "warmup_steps": int(1032 * c.WARMUP_RATIO),
        "epochs": c.EPOCHS,
        "micro_batch": c.MICRO_BATCH,
        "gradient_accumulation": c.GRAD_ACCUM,
        "updates_per_epoch": 344,
        "total_updates": 1032,
        "precision": "bfloat16",
        "ce_compute_dtype": "float32",
        "max_grad_norm": c.MAX_GRAD_NORM,
        "seed": c.SEED,
        "loss_reduction": (
            "FP32 CE sum then divide gradients by all valid response tokens "
            "in the update group"
        ),
    }
