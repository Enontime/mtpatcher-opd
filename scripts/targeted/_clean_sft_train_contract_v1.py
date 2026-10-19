#!/usr/bin/env python3
"""Immutable training contract construction for Chemistry KC-SFT."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import _clean_sft_optimization_contract_v2 as optimization_v2
import _clean_sft_parity_core_v1 as c
import _clean_sft_parity_kc_v1 as kc


def build(
    arm: str, rows: list[dict[str, Any]], target_path: Path
) -> dict[str, Any]:
    cfg = kc.ARM_CONFIG[arm]
    frozen = c.frozen_input_checks()
    c.require(
        not any(item.get("status") == "FAIL" for item in frozen.values()),
        "frozen training input check failed",
    )
    opd = c.inspect_opd_contract(cfg["opd"])
    c.require(opd["status"] == "PASS", f"OPD contract failed: {cfg['opd']}")
    identity = c.source_identity(c.read_jsonl(c.CONTEXT), rows)
    c.require(
        identity["status"] == "PASS",
        f"SOURCE_IDENTITY_FAIL {identity['issues']}",
    )
    dataset = cfg["dataset"]
    target_manifest = None
    if dataset != "old_target":
        spec = kc.DATASETS[dataset]
        manifest_path = spec["directory"] / "manifest.json"
        c.require(manifest_path.is_file(), f"{dataset} target manifest missing")
        manifest = c.read_json(manifest_path)
        if dataset == "matched_prompt":
            c.require(
                manifest.get("status") == "FROZEN"
                and manifest.get("technical_validity_status") == "PASS"
                and manifest.get("target_quality_status")
                == "RAW_DISTRIBUTION_AUDITED",
                "MatchedPrompt is not technically valid raw-distribution frozen data",
            )
        else:
            c.require(
                manifest.get("status") == "FROZEN"
                and manifest.get("technical_validity_status") == "PASS"
                and manifest.get("target_quality_status") == "PASS",
                f"{dataset} targets are not FROZEN/TARGET_QUALITY_PASS",
            )
        c.require(
            manifest.get("output", {}).get("sha256")
            == c.sha256_file(target_path),
            f"{dataset} target hash differs from frozen manifest",
        )
        if cfg["formal_matched_baseline"]:
            c.require(
                manifest.get("teacher_condition_parity_status") == "PASS"
                and manifest.get("formal_matched_sft_baseline") is True,
                "formal KC-SFT arm lacks TEACHER_CONDITION_PARITY_PASS",
            )
        target_manifest = {
            "path": str(manifest_path),
            "sha256": c.sha256_file(manifest_path),
            "teacher_condition_parity_status": manifest.get(
                "teacher_condition_parity_status"
            ),
            "technical_validity_status": manifest.get(
                "technical_validity_status"
            ),
            "target_quality_status": manifest.get("target_quality_status"),
            "content_quality_gate": manifest.get("content_quality_gate"),
        }
    return {
        "contract_version": "kc_sft_train_contract_v3",
        "arm": arm,
        "dataset": dataset,
        "formal_matched_baseline": cfg["formal_matched_baseline"],
        "frozen_inputs": frozen,
        "source_identity": identity,
        "opd_contract": opd,
        "target_path": str(target_path.resolve()),
        "target_sha256": c.sha256_file(target_path),
        "target_manifest": target_manifest,
        "base_model": {
            "path": str(c.C0),
            "config_sha256": c.sha256_file(c.C0 / "config.json"),
            "tokenizer_config_sha256": c.sha256_file(
                c.C0 / "tokenizer_config.json"
            ),
        },
        "optimization": optimization_v2.build(arm),
        "epoch_order_sha256": {
            str(epoch): c.epoch_order_sha(len(rows), epoch)
            for epoch in range(1, c.EPOCHS + 1)
        },
    }
