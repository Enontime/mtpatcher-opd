#!/usr/bin/env python3
"""Convert frozen targeted OPD/SFT logs and evaluations to TensorBoard.

Source experiments are read-only.  Every conversion is content-addressed and
written below a separate versioned output root.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import _targeted_tensorboard_v1 as tb


VERSION = "targeted_tensorboard_bridge_v1"
ROOT = Path("/workspace/mtpatcher")
TARGETED_RUNS = ROOT / "runs/targeted"
DEFAULT_OUTPUT = TARGETED_RUNS / "tensorboard_bridge_v1_20261015"

PRESETS: dict[str, dict[str, Path]] = {
    "chemistry": {
        "opd_lr1e6_horizon5": TARGETED_RUNS / "wa_opd_horizon5_o12_20260911",
        "opd_lr1e5": TARGETED_RUNS / "wa_opd_lr1e5_chemistry_20261014",
        "opd_lr2e5": TARGETED_RUNS / "wa_opd_lr2e5_chemistry_20261014",
        "opd_lr3e5": TARGETED_RUNS / "wa_opd_lr3e5_chemistry_20261014",
        "sft_c2_old": TARGETED_RUNS / "wa_sft_positive_control_c123_20260910",
        "sft_a1_horizon5": TARGETED_RUNS / "a1_sft_horizon5_c12_v3_20260912",
        "kc_sft_matched_prompt_lr2e5": (
            TARGETED_RUNS / "chemistry_clean_sft_parity_20261015"
        ),
    },
}

PRESET_ARM_FILTERS: dict[str, set[str]] = {
    "opd_lr1e6_horizon5": {"O2"},
    "opd_lr1e5": {"O2"},
    "opd_lr2e5": {"O2"},
    "opd_lr3e5": {"O2"},
    "sft_c2_old": {"C2"},
    "kc_sft_matched_prompt_lr2e5": {"KC_SFT_MATCHED_PROMPT_LR2E5"},
}

BAD_PATH_PARTS = ("INVALID", "failed", "aborted", "DUPLICATE")
KEY_VALUE_RE = re.compile(r"(?P<key>[A-Za-z0-9_]+)=(?P<value>[^\s]+)")
TIMESTAMP_RE = re.compile(r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T[^\s]+)")
PASS_RE = re.compile(r"^P(?P<pass_number>[1-9][0-9]*)$")


@dataclass(frozen=True)
class Experiment:
    run_id: str
    root: Path
    arm: str
    arm_dir: Path
    manifest_path: Path
    training_log: Path | None
    method: str


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    tb.require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
        tb.require(isinstance(value, dict), f"JSONL object required: {path}:{line_number}")
        rows.append(value)
    return rows


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    tb.require(bool(cleaned), f"empty/unsafe run id: {value!r}")
    return cleaned


def _method(manifest: dict[str, Any], log: Path | None) -> str:
    objective = str(manifest.get("training", {}).get("objective", "")).lower()
    name = str(log or "").lower()
    if "kl" in objective or "opd" in objective or "opd" in name:
        return "opd"
    if "cross" in objective or "sft" in objective or "sft" in name:
        return "sft"
    return "unknown"


def _candidate_logs(root: Path, arm: str, arm_dir: Path) -> list[Path]:
    return [
        arm_dir / "train.log",
        root / f"{arm}_train.log",
        root / "logs" / f"{arm}_train.log",
        root / "logs" / "train.log",
        root / f"{arm.lower()}_train.log",
    ]


def discover_experiments(alias: str, root: Path) -> list[Experiment]:
    tb.require(root.is_dir(), f"run directory missing: {root}")
    manifests = sorted(root.glob("**/train/manifest.json"))
    manifests = [
        path for path in manifests
        if not any(marker.lower() in str(path).lower() for marker in BAD_PATH_PARTS)
    ]
    experiments: list[Experiment] = []
    for manifest_path in manifests:
        arm_dir = manifest_path.parent.parent
        manifest = read_json(manifest_path)
        arm = str(manifest.get("arm") or arm_dir.name)
        candidates = [p for p in _candidate_logs(root, arm, arm_dir) if p.is_file()]
        training_log = candidates[0] if candidates else None
        method = _method(manifest, training_log)
        experiments.append(Experiment(
            run_id=_safe_id(f"{alias}__{arm}"),
            root=root,
            arm=arm,
            arm_dir=arm_dir,
            manifest_path=manifest_path,
            training_log=training_log,
            method=method,
        ))
    return experiments


def parse_run_specs(preset: str | None, specs: list[str]) -> dict[str, Path]:
    result = dict(PRESETS.get(preset or "", {}))
    for spec in specs:
        tb.require("=" in spec, f"--run must be ID=PATH: {spec}")
        name, raw_path = spec.split("=", 1)
        result[_safe_id(name)] = Path(raw_path).resolve()
    tb.require(bool(result), "no runs selected; use --preset or --run ID=PATH")
    return result


def _float(fields: dict[str, str], name: str) -> float:
    try:
        value = float(fields[name])
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f"missing/invalid log field {name}") from exc
    tb.require(math.isfinite(value), f"non-finite log field {name}")
    return value


def _wall_time(line: str) -> float | None:
    match = TIMESTAMP_RE.match(line)
    if not match:
        return None
    try:
        return datetime.fromisoformat(match.group("timestamp")).timestamp()
    except ValueError:
        return None


def parse_training_log(
    path: Path,
    *,
    max_grad_norm: float | None,
) -> tuple[list[tb.ScalarRecord], dict[str, Any]]:
    records: list[tb.ScalarRecord] = []
    values_seen: dict[tuple[str, int], float] = {}
    prior_source_step = -1
    parsed_lines = 0

    def add(tag: str, step: int, value: float, source_field: str, wall_time: float | None) -> None:
        key = (tag, step)
        if key in values_seen:
            tb.require(values_seen[key] == value, f"conflicting duplicate {tag}@{step} in {path}")
            return
        values_seen[key] = value
        records.append(tb.ScalarRecord(tag, step, value, wall_time, source_field, str(path)))

    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        fields = {match.group("key"): match.group("value") for match in KEY_VALUE_RE.finditer(line)}
        if "update" not in fields or not ({"kl100", "loss50"} & set(fields)):
            continue
        try:
            step = int(fields["update"].split("/", 1)[0])
        except ValueError as exc:
            raise RuntimeError(f"invalid update field {path}:{line_number}") from exc
        tb.require(step >= prior_source_step, f"non-monotonic optimizer update in {path}:{line_number}")
        prior_source_step = step
        parsed_lines += 1
        wall_time = _wall_time(line)
        grad_norm = _float(fields, "grad_norm")
        lr = _float(fields, "lr")
        add("optim/grad_norm_preclip", step, grad_norm, "grad_norm", wall_time)
        add("optim/lr", step, lr, "lr", wall_time)
        if max_grad_norm is not None:
            add("optim/grad_clip_scale", step, tb.clip_scale(grad_norm, max_grad_norm), "derived:max_grad_norm/grad_norm", wall_time)
        if "kl100" in fields:
            add("opd/kl_recent100", step, _float(fields, "kl100"), "kl100", wall_time)
            if "rollout_len200" in fields:
                add("train/rollout_length_recent200", step, _float(fields, "rollout_len200"), "rollout_len200", wall_time)
        if "loss50" in fields:
            add("sft/ce_microbatch_mean_recent50", step, _float(fields, "loss50"), "loss50", wall_time)
    ordered = sorted(records, key=lambda item: (item.step, item.tag))
    tb.validate_scalar_records(ordered)
    return ordered, {"parsed_metric_lines": parsed_lines, "last_logged_step": max((r.step for r in records), default=None)}


def parse_native_update_metrics(path: Path) -> list[tb.ScalarRecord]:
    records: list[tb.ScalarRecord] = []
    previous = 0
    for index, row in enumerate(read_jsonl(path), 1):
        step = int(row["optimizer_update"])
        tb.require(step == previous + 1, f"non-contiguous update_metrics at {path}:{index}")
        previous = step
        mapping = {
            "sft/ce_per_response_token": row.get("group_token_mean_ce"),
            "optim/lr": row.get("lr_after_step"),
            "optim/grad_norm_preclip": row.get("grad_norm_before_clip"),
            "train/response_tokens": row.get("group_response_tokens"),
            "train/step_seconds": row.get("step_seconds"),
        }
        max_grad_norm = row.get("max_grad_norm")
        if max_grad_norm is not None and row.get("grad_norm_before_clip") is not None:
            mapping["optim/grad_clip_scale"] = tb.clip_scale(float(row["grad_norm_before_clip"]), float(max_grad_norm))
        for tag, value in mapping.items():
            if value is not None:
                records.append(tb.ScalarRecord(tag, step, float(value), source_field=tag, source_path=str(path)))
    return tb.validate_scalar_records(sorted(records, key=lambda item: (item.step, item.tag)))


def model_identity(path: Path) -> dict[str, Any]:
    tb.require(path.is_dir(), f"model checkpoint missing: {path}")
    files: list[dict[str, Any]] = []
    for item in sorted(path.iterdir()):
        if item.is_file() and (item.suffix == ".safetensors" or item.name in {"config.json", "tokenizer_config.json", "generation_config.json"}):
            files.append({"name": item.name, "size": item.stat().st_size, "sha256": tb.sha256_file(item)})
    tb.require(any(item["name"] == "config.json" for item in files), f"model config missing: {path}")
    tb.require(any(item["name"].endswith(".safetensors") for item in files), f"model weights missing: {path}")
    return {"path": str(path.resolve()), "files": files, "identity_sha256": tb.sha256_json(files)}


def _accuracy(container: dict[str, Any], key: str) -> float | None:
    value = container.get(key)
    if isinstance(value, dict):
        value = value.get("accuracy")
    return float(value) if isinstance(value, (int, float)) else None


def _nested_accuracy(container: dict[str, Any], *keys: str) -> float | None:
    value: Any = container
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, dict):
        value = value.get("accuracy")
    return float(value) if isinstance(value, (int, float)) else None


def _first_available(*values: float | None) -> float | None:
    return next((value for value in values if value is not None), None)


def chemistry_records(
    summary: dict[str, Any],
    *,
    step: int,
    path: Path,
    eval_id: str,
) -> list[tb.ScalarRecord]:
    slices = summary.get("slices")
    slices = slices if isinstance(slices, dict) else {}
    mapping = {
        "knowledge/chemistry_strict": _first_available(
            _accuracy(summary, "overall"),
            _nested_accuracy(slices, "all", "strict_canonical"),
        ),
        "knowledge/chemistry_spacing_normalized": _first_available(
            _accuracy(summary, "spacing_normalized"),
            _nested_accuracy(slices, "all", "ci_spacing_normalized"),
        ),
        "knowledge/chemistry_ci": _nested_accuracy(slices, "ci", "strict_canonical"),
        "knowledge/chemistry_non_ci": _nested_accuracy(slices, "non_ci", "strict_canonical"),
        "knowledge/chemistry_uc": _first_available(
            _accuracy(summary, "uc500"), _accuracy(summary, "uc"),
            _nested_accuracy(slices, "uc", "strict_canonical"),
        ),
        "knowledge/chemistry_uw": _first_available(
            _accuracy(summary, "uw500"), _accuracy(summary, "uw"),
            _nested_accuracy(slices, "uw", "strict_canonical"),
        ),
    }
    return [tb.ScalarRecord(tag, step, value, source_field=tag, source_path=str(path), eval_id=eval_id) for tag, value in mapping.items() if value is not None]


def _slice_accuracy(summary: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        group = summary.get(key)
        if not isinstance(group, dict):
            continue
        for field in ("accuracy", "new_accuracy"):
            value = group.get(field)
            if isinstance(value, (int, float)):
                return float(value)
    return None


def slice_records(summary: dict[str, Any], *, step: int, path: Path, eval_id: str) -> list[tb.ScalarRecord]:
    mapping = {
        "knowledge/chemistry_ci": _slice_accuracy(summary, "CI110", "ci"),
        "knowledge/chemistry_non_ci": _slice_accuracy(summary, "NONCI890", "non_ci"),
    }
    return [tb.ScalarRecord(tag, step, value, source_field=tag, source_path=str(path), eval_id=eval_id) for tag, value in mapping.items() if value is not None]


def quality_records(summary: dict[str, Any], *, step: int, path: Path, eval_id: str) -> list[tb.ScalarRecord]:
    mapping = {
        "quality/bleu": summary.get("macro_bleu"),
        "quality/chrf": summary.get("macro_chrf"),
    }
    return [tb.ScalarRecord(tag, step, float(value), source_field=tag, source_path=str(path), eval_id=eval_id) for tag, value in mapping.items() if isinstance(value, (int, float))]


def _final_quality_path(experiment: Experiment) -> Path | None:
    candidates = [
        experiment.root / "general_bleu_opd" / experiment.arm / "summary.json",
        experiment.root / "general_bleu" / experiment.arm / "summary.json",
        experiment.arm_dir / "general_eval" / "summary.json",
        experiment.arm_dir / "eval" / "general_summary.json",
    ]
    return next((path for path in candidates if path.is_file()), None)


def _final_checkpoint_step(
    experiment: Experiment, manifest: dict[str, Any]
) -> tuple[int, list[Path], dict[str, int]]:
    """Resolve a completed checkpoint step from durable artifacts, not epoch labels."""
    train_dir = experiment.arm_dir / "train"
    evidence: list[Path] = []
    values: dict[str, int] = {}
    for name in ("resume_meta.json", "progress.json", "latest_resume.json"):
        path = train_dir / name
        if not path.is_file():
            continue
        payload = read_json(path)
        step = payload.get("completed_updates")
        if isinstance(step, int):
            values[name] = step
            evidence.append(path)
        if name == "progress.json":
            tb.require(payload.get("status") == "PASS", f"final progress is not PASS: {path}")
    ledger = train_dir / "update_metrics.jsonl"
    expected_ledger_sha = manifest.get("training", {}).get("update_metrics_sha256")
    if ledger.is_file() and isinstance(expected_ledger_sha, str):
        tb.require(tb.sha256_file(ledger) == expected_ledger_sha, f"update ledger SHA mismatch: {ledger}")
        ledger_rows = read_jsonl(ledger)
        steps = [int(row["optimizer_update"]) for row in ledger_rows]
        tb.require(steps == list(range(1, len(steps) + 1)), f"non-contiguous final update ledger: {ledger}")
        values["update_metrics.jsonl"] = steps[-1] if steps else 0
        evidence.append(ledger)
    tb.require(bool(values), f"durable final checkpoint step unavailable: {train_dir}")
    tb.require(len(set(values.values())) == 1, f"final checkpoint step disagreement: {values}")
    step = next(iter(values.values()))
    planned = manifest.get("training", {}).get("total_updates")
    if isinstance(planned, int):
        tb.require(step == planned, f"completed checkpoint step {step} != planned {planned}")
    return step, evidence, values


def validate_bound_evaluation_contract(
    experiment: Experiment,
    train_manifest: dict[str, Any],
    eval_dir: Path,
    eval_manifest: dict[str, Any],
    chemistry_summary: dict[str, Any] | None,
    general_summary: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[Path]]:
    """Validate the immutable Clean-SFT eval/checkpoint identity closure."""
    contract_path = eval_dir / "eval_contract.json"
    tb.require(contract_path.is_file(), f"frozen eval contract missing: {contract_path}")
    envelope = read_json(contract_path)
    contract = envelope.get("contract")
    contract_sha = envelope.get("contract_sha256")
    tb.require(envelope.get("status") == "BOUND", f"eval contract is not BOUND: {contract_path}")
    tb.require(isinstance(contract, dict), f"eval contract payload missing: {contract_path}")
    tb.require(
        isinstance(contract_sha, str) and tb.sha256_json(contract) == contract_sha,
        f"eval contract SHA mismatch: {contract_path}",
    )
    tb.require(eval_manifest.get("status") == "PASS", f"eval manifest is not PASS: {eval_dir}")
    tb.require(eval_manifest.get("arm") == experiment.arm, f"eval manifest arm mismatch: {eval_dir}")
    tb.require(eval_manifest.get("eval_contract") == contract, f"eval manifest contract payload mismatch: {eval_dir}")
    tb.require(eval_manifest.get("eval_contract_sha256") == contract_sha, f"eval manifest contract SHA mismatch: {eval_dir}")

    actual_inputs: dict[str, str] = {}
    inputs = contract.get("inputs")
    tb.require(isinstance(inputs, dict), f"eval inputs missing: {contract_path}")
    input_entries: list[dict[str, Any]] = []
    targeted = inputs.get("targeted")
    if isinstance(targeted, dict):
        input_entries.append(targeted)
    general_inputs = inputs.get("general")
    if isinstance(general_inputs, dict):
        input_entries.extend(
            value for value in general_inputs.values() if isinstance(value, dict)
        )
    tb.require(bool(input_entries), f"no frozen eval inputs: {contract_path}")
    for entry in input_entries:
        path = Path(str(entry.get("path", "")))
        expected = entry.get("sha256")
        tb.require(path.is_file() and isinstance(expected, str), f"invalid frozen eval input: {entry}")
        actual = tb.sha256_file(path)
        tb.require(actual == expected, f"frozen eval input SHA mismatch: {path}")
        actual_inputs[str(path.resolve())] = actual

    final_model = Path(str(train_manifest.get("final_model", ""))).resolve()
    declared_model = contract.get("model")
    tb.require(isinstance(declared_model, dict), f"eval model identity missing: {contract_path}")
    tb.require(
        Path(str(declared_model.get("resolved_path", ""))).resolve() == final_model,
        f"eval contract model path mismatch: {contract_path}",
    )
    declared_files = declared_model.get("files")
    tb.require(isinstance(declared_files, list) and declared_files, f"eval model files missing: {contract_path}")

    def verify_declared_files(model_dir: Path, label: str) -> None:
        tb.require(model_dir.is_dir(), f"{label} model directory missing: {model_dir}")
        for item in declared_files:
            tb.require(isinstance(item, dict), f"invalid declared model file: {item}")
            path = model_dir / str(item.get("name", ""))
            tb.require(path.is_file(), f"{label} model file missing: {path}")
            tb.require(path.stat().st_size == item.get("size"), f"{label} model size mismatch: {path}")
            tb.require(tb.sha256_file(path) == item.get("sha256"), f"{label} model SHA mismatch: {path}")

    verify_declared_files(final_model, "final")
    tb.require(
        tb.sha256_json(declared_files) == declared_model.get("identity_sha256"),
        f"declared model identity SHA mismatch: {contract_path}",
    )

    train_dir = experiment.arm_dir / "train"
    run_contract_path = train_dir / "run_contract.json"
    latest_path = train_dir / "latest_resume.json"
    tb.require(run_contract_path.is_file() and latest_path.is_file(), f"training contract/checkpoint pointer missing: {train_dir}")
    run_contract = read_json(run_contract_path)
    run_contract_sha = train_manifest.get("run_contract_sha256")
    tb.require(tb.sha256_json(run_contract["contract"]) == run_contract_sha, f"training contract SHA mismatch: {run_contract_path}")
    latest = read_json(latest_path)
    tb.require(latest.get("run_contract_sha256") == run_contract_sha, f"checkpoint pointer contract mismatch: {latest_path}")
    checkpoint = Path(str(latest.get("checkpoint", "")))
    meta_path = checkpoint / "meta.json"
    tb.require(meta_path.is_file(), f"checkpoint metadata missing: {checkpoint}")
    meta = read_json(meta_path)
    expected_step = train_manifest.get("training", {}).get("total_updates")
    tb.require(
        meta.get("status") == "PASS"
        and meta.get("run_contract_sha256") == run_contract_sha
        and meta.get("completed_updates") == expected_step
        and latest.get("completed_updates") == expected_step,
        f"durable checkpoint identity mismatch: {checkpoint}",
    )
    verify_declared_files(checkpoint / "hf", "durable checkpoint")

    frozen_inputs = eval_manifest.get("frozen_inputs")
    tb.require(isinstance(frozen_inputs, dict), f"eval frozen_inputs missing: {eval_dir}")
    for name, item in frozen_inputs.items():
        tb.require(isinstance(item, dict) and item.get("status") == "PASS", f"frozen input status failed: {name}")
        if "actual" in item or "expected" in item:
            tb.require(item.get("actual") == item.get("expected"), f"frozen input hash mismatch: {name}")

    for summary_name, summary in (
        ("chemistry_summary.json", chemistry_summary),
        ("general_summary.json", general_summary),
    ):
        if summary is None:
            continue
        tb.require(summary.get("status") == "PASS", f"summary is not PASS: {summary_name}")
        tb.require(summary.get("eval_contract_sha256") == contract_sha, f"summary contract mismatch: {summary_name}")

    prediction_hashes: dict[str, str] = {}
    if general_summary is not None:
        datasets = general_summary.get("datasets")
        tb.require(isinstance(datasets, dict), f"general datasets missing: {eval_dir}")
        for dataset_name, payload in datasets.items():
            tb.require(isinstance(payload, dict), f"invalid general dataset summary: {dataset_name}")
            prediction_path = eval_dir / f"{str(dataset_name).lower()}_predictions.jsonl"
            expected = payload.get("predictions_sha256")
            tb.require(prediction_path.is_file() and isinstance(expected, str), f"prediction identity missing: {dataset_name}")
            actual = tb.sha256_file(prediction_path)
            tb.require(actual == expected, f"prediction SHA mismatch: {prediction_path}")
            prediction_hashes[str(prediction_path.resolve())] = actual

    evidence = {
        "status": "PASS",
        "eval_contract_sha256": contract_sha,
        "evaluation_version": contract.get("evaluation_version"),
        "scoring": contract.get("scoring"),
        "generation": contract.get("generation"),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_step": expected_step,
        "run_contract_sha256": run_contract_sha,
        "model_identity_sha256": declared_model.get("identity_sha256"),
        "verified_eval_inputs": actual_inputs,
        "verified_prediction_files": prediction_hashes,
    }
    return evidence, [contract_path, run_contract_path, latest_path, meta_path]


def collect_final_evaluation(
    experiment: Experiment, manifest: dict[str, Any]
) -> tuple[list[tb.ScalarRecord], list[dict[str, Any]], list[Path], list[str]]:
    step, step_sources, step_evidence = _final_checkpoint_step(experiment, manifest)
    eval_dir_candidates = [experiment.arm_dir / "targeted_eval", experiment.arm_dir / "eval"]
    eval_dir = next((path for path in eval_dir_candidates if path.is_dir()), None)
    sources: list[Path] = list(step_sources)
    records: list[tb.ScalarRecord] = []
    missing: list[str] = []
    eval_id = f"{experiment.run_id}:final:{step}"
    evaluation_contract: dict[str, Any] = {
        "training_manifest_frozen_inputs": manifest.get("frozen_inputs", "NOT_AVAILABLE"),
        "checkpoint_step_evidence": step_evidence,
    }
    eval_manifest_payload: dict[str, Any] | None = None
    chemistry_payload: dict[str, Any] | None = None
    quality_payload: dict[str, Any] | None = None
    if eval_dir is not None:
        eval_manifest = eval_dir / "manifest.json"
        if eval_manifest.is_file():
            eval_manifest_payload = read_json(eval_manifest)
            declared_model = eval_manifest_payload.get("model")
            if isinstance(declared_model, str) and declared_model:
                tb.require(
                    Path(declared_model).resolve() == Path(str(manifest.get("final_model", ""))).resolve(),
                    f"evaluation manifest model does not match training manifest: {eval_manifest}",
                )
            sources.append(eval_manifest)
            evaluation_contract["evaluation_manifest"] = eval_manifest_payload
        chemistry = eval_dir / "chemistry_summary.json"
        slices = eval_dir / "slice_summary.json"
        if chemistry.is_file():
            sources.append(chemistry)
            chemistry_payload = read_json(chemistry)
            records.extend(chemistry_records(chemistry_payload, step=step, path=chemistry, eval_id=eval_id))
            evaluation_contract["chemistry_metric_definition"] = chemistry_payload.get(
                "metric", chemistry_payload.get("primary_metric", "NOT_AVAILABLE")
            )
        else:
            missing.append("knowledge/chemistry_strict NOT_AVAILABLE")
        if slices.is_file():
            sources.append(slices)
            records.extend(slice_records(read_json(slices), step=step, path=slices, eval_id=eval_id))
        elif not (
            isinstance(chemistry_payload, dict)
            and isinstance(chemistry_payload.get("slices"), dict)
            and "ci" in chemistry_payload["slices"]
            and "non_ci" in chemistry_payload["slices"]
        ):
            missing.append("knowledge/chemistry_ci and non_ci NOT_AVAILABLE")
    quality = _final_quality_path(experiment)
    if quality is not None:
        sources.append(quality)
        quality_payload = read_json(quality)
        declared_model = quality_payload.get("model")
        if isinstance(declared_model, str) and declared_model:
            tb.require(
                Path(declared_model).resolve() == Path(str(manifest.get("final_model", ""))).resolve(),
                f"quality summary model does not match training manifest: {quality}",
            )
        records.extend(quality_records(quality_payload, step=step, path=quality, eval_id=eval_id))
        evaluation_contract["quality_generation"] = quality_payload.get("generation", "NOT_AVAILABLE")
    else:
        missing.append("quality/bleu and quality/chrf NOT_AVAILABLE")
    if eval_dir is not None and (eval_dir / "eval_contract.json").is_file():
        tb.require(eval_manifest_payload is not None, f"eval manifest missing beside frozen contract: {eval_dir}")
        bound_evidence, bound_sources = validate_bound_evaluation_contract(
            experiment, manifest, eval_dir, eval_manifest_payload,
            chemistry_payload, quality_payload,
        )
        sources.extend(bound_sources)
        evaluation_contract["frozen_contract_validation"] = bound_evidence
    if not records:
        return [], [], sources, missing
    model_path = Path(str(manifest.get("final_model", "")))
    identity = model_identity(model_path)
    provenance = [{
        "eval_id": eval_id,
        "optimizer_step": step,
        "checkpoint": identity,
        "metric_sources": [str(path) for path in sources],
        "evaluation_contract": evaluation_contract,
        "baseline": None,
    }]
    return records, provenance, sources, missing


def collect_a1_passes(
    experiment: Experiment,
) -> tuple[list[tb.ScalarRecord], list[dict[str, Any]], list[Path], list[str]]:
    summary_path = experiment.arm_dir / "summary.json"
    if not summary_path.is_file():
        return [], [], [], []
    summary = read_json(summary_path)
    passes = summary.get("passes")
    if not isinstance(passes, dict):
        return [], [], [summary_path], []
    train_manifest = read_json(experiment.manifest_path)
    frozen_inputs = train_manifest.get(
        "frozen_inputs", train_manifest.get("frozen_input_provenance", "NOT_AVAILABLE")
    )
    records: list[tb.ScalarRecord] = []
    provenance: list[dict[str, Any]] = []
    sources = [summary_path]
    missing: list[str] = []
    for pass_name, payload in sorted(passes.items()):
        match = PASS_RE.match(str(pass_name))
        tb.require(match is not None and isinstance(payload, dict), f"malformed pass entry in {summary_path}: {pass_name}")
        meta_path = experiment.arm_dir / "checkpoints" / f"{pass_name}_resume_meta.json"
        checkpoint = Path(str(payload.get("checkpoint", "")))
        if not meta_path.is_file() or not checkpoint.is_dir():
            missing.append(f"{pass_name} evaluation omitted: checkpoint/meta identity unavailable")
            continue
        meta = read_json(meta_path)
        step = meta.get("completed_updates")
        tb.require(isinstance(step, int), f"completed_updates missing: {meta_path}")
        eval_id = f"{experiment.run_id}:{pass_name}:{step}"
        pass_sources = [summary_path, meta_path]
        evaluation_contract: dict[str, Any] = {"frozen_inputs": frozen_inputs}
        chemistry_path = experiment.arm_dir / "targeted_eval" / pass_name / "chemistry_summary.json"
        chemistry = read_json(chemistry_path) if chemistry_path.is_file() else payload.get("chemistry")
        chemistry_source = chemistry_path if chemistry_path.is_file() else summary_path
        if isinstance(chemistry, dict):
            records.extend(chemistry_records(chemistry, step=step, path=chemistry_source, eval_id=eval_id))
            evaluation_contract["chemistry_metric_definition"] = chemistry.get("metric", "NOT_AVAILABLE")
            if chemistry_source != summary_path:
                sources.append(chemistry_source)
                pass_sources.append(chemistry_source)
        general_path = experiment.arm_dir / "general_eval" / pass_name / "summary.json"
        general = read_json(general_path) if general_path.is_file() else payload.get("general")
        general_source = general_path if general_path.is_file() else summary_path
        if isinstance(general, dict):
            declared_model = general.get("model")
            if isinstance(declared_model, str) and declared_model:
                tb.require(Path(declared_model).resolve() == checkpoint.resolve(), f"A1 quality model mismatch: {general_path}")
            records.extend(quality_records(general, step=step, path=general_source, eval_id=eval_id))
            evaluation_contract["quality_generation"] = general.get("generation", "NOT_AVAILABLE")
            if general_source != summary_path:
                sources.append(general_source)
                pass_sources.append(general_source)
        sources.append(meta_path)
        provenance.append({
            "eval_id": eval_id,
            "optimizer_step": step,
            "checkpoint": model_identity(checkpoint),
            "metric_sources": [str(path) for path in pass_sources],
            "evaluation_contract": evaluation_contract,
            "baseline": None,
        })
    return records, provenance, sources, missing


def collect_opd_horizon_passes(
    experiment: Experiment, manifest: dict[str, Any]
) -> tuple[list[tb.ScalarRecord], list[dict[str, Any]], list[Path], list[str]]:
    heldout_root = experiment.root / "learning_curve" / "heldout" / experiment.arm
    if not heldout_root.is_dir():
        return [], [], [], []
    training = manifest.get("training", {})
    updates_per_epoch = training.get("updates_per_epoch")
    epochs = training.get("epochs")
    tb.require(
        isinstance(updates_per_epoch, int) and updates_per_epoch > 0,
        f"updates_per_epoch unavailable for OPD horizon: {experiment.manifest_path}",
    )
    tb.require(isinstance(epochs, int) and epochs > 0, f"epochs unavailable for OPD horizon: {experiment.manifest_path}")
    snapshot_log = experiment.root / "logs" / f"{experiment.arm}_snapshot.log"
    tb.require(snapshot_log.is_file(), f"snapshot evidence missing: {snapshot_log}")
    snapshot_text = snapshot_log.read_text(encoding="utf-8")
    records: list[tb.ScalarRecord] = []
    provenance: list[dict[str, Any]] = []
    sources: list[Path] = [snapshot_log]
    missing: list[str] = []
    for pass_dir in sorted(heldout_root.iterdir()):
        match = PASS_RE.match(pass_dir.name)
        if not pass_dir.is_dir() or match is None:
            continue
        pass_number = int(match.group("pass_number"))
        tb.require(pass_number <= epochs, f"horizon pass exceeds training epochs: {pass_dir}")
        checkpoint = (
            experiment.arm_dir / "train" / "epoch_checkpoints"
            / f"pass_{pass_number}_hf"
        )
        expected_snapshot = f"pass={pass_number} SNAPSHOT_PASS path={checkpoint}"
        tb.require(expected_snapshot in snapshot_text, f"snapshot/checkpoint evidence mismatch: {checkpoint}")
        summary_path = pass_dir / "chemistry_summary.json"
        if not summary_path.is_file() or not checkpoint.is_dir():
            missing.append(f"{pass_dir.name} heldout evaluation omitted: summary/checkpoint missing")
            continue
        step = pass_number * updates_per_epoch
        eval_id = f"{experiment.run_id}:heldout:{pass_dir.name}:{step}"
        summary = read_json(summary_path)
        records.extend(chemistry_records(summary, step=step, path=summary_path, eval_id=eval_id))
        sources.append(summary_path)
        provenance.append({
            "eval_id": eval_id,
            "optimizer_step": step,
            "checkpoint": model_identity(checkpoint),
            "metric_sources": [str(summary_path), str(snapshot_log)],
            "evaluation_contract": {
                "evaluation_split": "heldout",
                "metric_definition": summary.get("metric", "NOT_AVAILABLE"),
                "frozen_inputs": manifest.get("frozen_inputs", "NOT_AVAILABLE"),
                "decoding_config": "NOT_AVAILABLE_IN_HISTORICAL_SUMMARY",
                "step_evidence": {
                    "pass_number": pass_number,
                    "updates_per_epoch": updates_per_epoch,
                    "snapshot_log": str(snapshot_log),
                },
            },
            "baseline": None,
        })
    if records:
        missing.append("OPD Horizon-5 per-pass quality/bleu and quality/chrf NOT_AVAILABLE")
    return records, provenance, sources, missing


def collect_eval_index(path: Path | None, run_id: str) -> tuple[list[tb.ScalarRecord], list[dict[str, Any]], list[Path]]:
    if path is None:
        return [], [], []
    payload = read_json(path)
    rows = payload.get("eval_records")
    tb.require(isinstance(rows, list), f"eval_records list required: {path}")
    records: list[tb.ScalarRecord] = []
    provenance: list[dict[str, Any]] = []
    source_paths = [path]
    seen_eval_ids: set[str] = set()
    for row in rows:
        tb.require(isinstance(row, dict), f"invalid eval record in {path}")
        if row.get("run_id") != run_id:
            continue
        eval_id = str(row.get("eval_id", ""))
        tb.require(eval_id and eval_id not in seen_eval_ids, f"duplicate/empty eval_id: {eval_id}")
        seen_eval_ids.add(eval_id)
        step = row.get("optimizer_step")
        checkpoint_identity = row.get("checkpoint_identity_sha256")
        tb.require(isinstance(step, int) and step >= 0, f"invalid optimizer_step for {eval_id}")
        tb.require(isinstance(checkpoint_identity, str) and checkpoint_identity, f"checkpoint identity required for {eval_id}")
        checkpoint_path_value = row.get("checkpoint_path")
        tb.require(isinstance(checkpoint_path_value, str) and checkpoint_path_value, f"checkpoint path required for {eval_id}")
        checkpoint_path = Path(checkpoint_path_value)
        actual_checkpoint = model_identity(checkpoint_path)
        tb.require(
            actual_checkpoint["identity_sha256"] == checkpoint_identity,
            f"checkpoint identity mismatch for {eval_id}",
        )
        identity_fields = {}
        for field in (
            "eval_data_sha256", "prompt_sha256",
            "decoding_config_sha256", "evaluation_version",
        ):
            value = row.get(field)
            tb.require(isinstance(value, str) and value, f"{field} required for {eval_id}")
            identity_fields[field] = value
        source_files = row.get("source_files")
        tb.require(isinstance(source_files, list) and source_files, f"source_files required for {eval_id}")
        verified_sources: list[str] = []
        verified_source_hashes: set[str] = set()
        for source in source_files:
            source_path = Path(str(source["path"]))
            tb.require(source_path.is_file(), f"evaluation source missing: {source_path}")
            tb.require(tb.sha256_file(source_path) == source.get("sha256"), f"evaluation SHA mismatch: {source_path}")
            verified_sources.append(str(source_path))
            verified_source_hashes.add(str(source.get("sha256")))
            source_paths.append(source_path)
        tb.require(
            identity_fields["eval_data_sha256"] in verified_source_hashes,
            f"eval_data_sha256 is not bound to a source file for {eval_id}",
        )
        metrics = row.get("metrics")
        tb.require(isinstance(metrics, dict), f"metrics object required for {eval_id}")
        if {"knowledge/gain", "knowledge/loss"} & set(metrics):
            baseline = row.get("baseline")
            tb.require(isinstance(baseline, dict) and baseline.get("model_identity_sha256") and baseline.get("eval_id") == eval_id, f"GAIN/LOSS baseline identity/eval_id required for {eval_id}")
        for tag, value in metrics.items():
            tb.require(tag in tb.METRIC_DEFINITIONS, f"unsupported eval metric {tag}")
            records.append(tb.ScalarRecord(tag, step, float(value), source_field=tag, source_path=verified_sources[0], eval_id=eval_id))
        provenance.append({
            "eval_id": eval_id,
            "optimizer_step": step,
            "checkpoint": actual_checkpoint,
            "metric_sources": verified_sources,
            "evaluation_contract": identity_fields,
            "baseline": row.get("baseline"),
        })
    return records, provenance, source_paths


def _source_entry(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": tb.sha256_file(path)}


def build_plan(experiment: Experiment, eval_index: Path | None = None) -> dict[str, Any]:
    manifest = read_json(experiment.manifest_path)
    training = manifest.get("training", {})
    max_grad_norm = training.get("max_grad_norm")
    records: list[tb.ScalarRecord] = []
    sources: list[Path] = [experiment.manifest_path]
    missing: list[str] = []
    parser_report: dict[str, Any] = {}
    native_metrics = experiment.arm_dir / "train" / "update_metrics.jsonl"
    if native_metrics.is_file():
        sources.append(native_metrics)
        records.extend(parse_native_update_metrics(native_metrics))
        parser_report["training_source"] = "native_update_metrics_jsonl"
    elif experiment.training_log is not None:
        sources.append(experiment.training_log)
        parsed, parser_report = parse_training_log(
            experiment.training_log,
            max_grad_norm=float(max_grad_norm) if isinstance(max_grad_norm, (int, float)) else None,
        )
        records.extend(parsed)
        parser_report["training_source"] = "historical_text_log"
    else:
        missing.append("training metrics NOT_AVAILABLE: no supported log")
    final_records, final_provenance, final_sources, final_missing = collect_final_evaluation(experiment, manifest)
    records.extend(final_records)
    sources.extend(final_sources)
    missing.extend(final_missing)
    pass_records, pass_provenance, pass_sources, pass_missing = collect_a1_passes(experiment)
    records.extend(pass_records)
    sources.extend(pass_sources)
    missing.extend(pass_missing)
    horizon_records, horizon_provenance, horizon_sources, horizon_missing = (
        collect_opd_horizon_passes(experiment, manifest)
    )
    records.extend(horizon_records)
    sources.extend(horizon_sources)
    missing.extend(horizon_missing)
    indexed_records, indexed_provenance, indexed_sources = collect_eval_index(eval_index, experiment.run_id)
    records.extend(indexed_records)
    sources.extend(indexed_sources)
    evaluation_provenance = (
        final_provenance + pass_provenance + horizon_provenance
        + indexed_provenance
    )
    # A tag/step may be present in both final and pass summaries.  Exact matches
    # are deduplicated; conflicting values are blocked.
    unique: dict[tuple[str, int], tb.ScalarRecord] = {}
    for record in sorted(records, key=lambda item: (item.step, item.tag, item.eval_id or "")):
        key = (record.tag, record.step)
        if key in unique:
            tb.require(float(unique[key].value) == float(record.value), f"conflicting metric {record.tag}@{record.step}")
            continue
        unique[key] = record
    ordered = sorted(unique.values(), key=lambda item: (item.step, item.tag))
    tb.validate_scalar_records(ordered)
    source_entries = [_source_entry(path) for path in sorted(set(sources))]
    converter_code = [
        _source_entry(Path(__file__).resolve()),
        _source_entry(Path(tb.__file__).resolve()),
    ]
    tag_counts = Counter(record.tag for record in ordered)
    metric_definitions = {tag: tb.METRIC_DEFINITIONS[tag] for tag in sorted(tag_counts)}
    identity_payload = {
        "converter_version": VERSION,
        "converter_code": converter_code,
        "run_id": experiment.run_id,
        "sources": source_entries,
        "records": [record.to_json() for record in ordered],
        "evaluation_provenance": evaluation_provenance,
    }
    return {
        "status": "READY" if ordered else "NOT_AVAILABLE",
        "converter_version": VERSION,
        "converter_code": converter_code,
        "run_id": experiment.run_id,
        "run_root": str(experiment.root),
        "arm": experiment.arm,
        "method": experiment.method,
        "source_manifest": str(experiment.manifest_path),
        "step_rule": "actual optimizer update from log/update_metrics/checkpoint metadata",
        "records": [record.to_json() for record in ordered],
        "scalar_counts": dict(sorted(tag_counts.items())),
        "metric_definitions": metric_definitions,
        "evaluation_provenance": evaluation_provenance,
        "sources": source_entries,
        "missing": missing,
        "parser_report": parser_report,
        "mechanism_diagnostics": {
            "schema": tb.MECHANISM_DIAGNOSTIC_SCHEMA,
            "student_prefix_term_probability": "NOT_AVAILABLE",
            "teacher_prefix_term_probability": "NOT_AVAILABLE",
            "term_position_kl": "NOT_AVAILABLE",
            "non_term_position_kl": "NOT_AVAILABLE",
            "effective_term_event_coverage": "NOT_AVAILABLE",
        },
        "conversion_identity_sha256": tb.sha256_json(identity_payload),
    }


def write_plan(plan: dict[str, Any], output: Path, writer_class: Any | None = None) -> tuple[Path, str]:
    tb.require(plan["status"] == "READY", f"no convertible metrics for {plan['run_id']}")
    identity = str(plan["conversion_identity_sha256"])
    run_root = output / VERSION / _safe_id(str(plan["run_id"]))
    destination = run_root / identity[:16]
    manifest_path = destination / "event_manifest.json"
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        tb.require(existing.get("conversion_identity_sha256") == identity, f"managed conversion identity mismatch: {destination}")
        return destination, "ALREADY_CURRENT"
    tb.require(not destination.exists(), f"refuse unmanaged/incomplete destination: {destination}")
    run_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{identity[:16]}.", dir=run_root))
    try:
        events = staging / "events"
        events.mkdir()
        cls = writer_class or tb._summary_writer_class()
        writer = cls(log_dir=str(events))
        for raw in plan["records"]:
            writer.add_scalar(raw["tag"], raw["value"], global_step=raw["step"], walltime=raw.get("wall_time"))
        writer.flush()
        writer.close()
        manifest = {key: value for key, value in plan.items() if key != "records"}
        manifest.update({
            "status": "PASS",
            "event_directory": str(destination / "events"),
            "record_count": len(plan["records"]),
            "idempotence": "content-addressed; identical sources and records are a no-op",
        })
        (staging / "event_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(staging, destination)
    except Exception:
        # Keep the failed staging directory for forensic inspection.  It is
        # hidden and never treated as a completed conversion.
        raise
    return destination, "CREATED"


def validate_event_directory(destination: Path) -> dict[str, Any]:
    manifest_path = destination / "event_manifest.json"
    events = destination / "events"
    tb.require(manifest_path.is_file() and events.is_dir(), f"incomplete event conversion: {destination}")
    manifest = read_json(manifest_path)
    accumulator = tb.event_accumulator_class()(str(events), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    expected = manifest.get("scalar_counts", {})
    tb.require(available == set(expected), f"unexpected/missing event tags: actual={sorted(available)} expected={sorted(expected)}")
    result: dict[str, Any] = {}
    for tag, count in expected.items():
        tb.require(tag in available, f"event tag missing: {tag}")
        values = accumulator.Scalars(tag)
        steps = [int(value.step) for value in values]
        tb.require(len(values) == int(count), f"event scalar count mismatch: {tag}")
        tb.require(steps == sorted(set(steps)), f"event steps duplicate/non-monotonic: {tag}")
        result[tag] = {"count": len(values), "min_step": min(steps), "max_step": max(steps)}
    return {"status": "PASS", "destination": str(destination), "tags": result}


def selected_experiments(args: argparse.Namespace) -> list[Experiment]:
    runs = parse_run_specs(args.preset, args.run)
    result: list[Experiment] = []
    for alias, root in runs.items():
        discovered = discover_experiments(alias, root)
        arm_filter = PRESET_ARM_FILTERS.get(alias)
        if arm_filter is not None:
            discovered = [item for item in discovered if item.arm in arm_filter]
        result.extend(discovered)
    tb.require(bool(result), "no train manifests discovered")
    ids = [item.run_id for item in result]
    tb.require(len(ids) == len(set(ids)), f"duplicate discovered run IDs: {ids}")
    return result


def cmd_inspect(args: argparse.Namespace) -> None:
    reports = []
    eval_index = Path(args.eval_index) if args.eval_index else None
    for experiment in selected_experiments(args):
        plan = build_plan(experiment, eval_index)
        reports.append({key: value for key, value in plan.items() if key != "records"})
    print(json.dumps({"status": "PASS", "read_only": True, "experiments": reports}, ensure_ascii=False, indent=2))


def cmd_convert(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    eval_index = Path(args.eval_index) if args.eval_index else None
    converted = []
    for experiment in selected_experiments(args):
        plan = build_plan(experiment, eval_index)
        if plan["status"] != "READY":
            converted.append({"run_id": experiment.run_id, "status": "NOT_AVAILABLE", "missing": plan["missing"]})
            continue
        destination, status = write_plan(plan, output)
        converted.append({"run_id": experiment.run_id, "status": status, "destination": str(destination)})
    print(json.dumps({"status": "PASS", "output": str(output), "experiments": converted}, ensure_ascii=False, indent=2))


def cmd_validate(args: argparse.Namespace) -> None:
    destinations = [Path(path).resolve() for path in args.event_dir]
    if args.output:
        destinations.extend(path.parent for path in Path(args.output).resolve().glob(f"{VERSION}/**/event_manifest.json"))
    unique = sorted(set(destinations))
    tb.require(bool(unique), "no event directories selected")
    print(json.dumps({"status": "PASS", "validated": [validate_event_directory(path) for path in unique]}, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "convert"):
        command = sub.add_parser(name)
        command.add_argument("--preset", choices=sorted(PRESETS), default="chemistry")
        command.add_argument("--run", action="append", default=[], metavar="ID=PATH")
        command.add_argument("--eval-index", help="Optional frozen evaluation index with checkpoint identities")
    sub.choices["convert"].add_argument("--output", default=str(DEFAULT_OUTPUT))
    validate = sub.add_parser("validate")
    validate.add_argument("--event-dir", action="append", default=[])
    validate.add_argument("--output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    {"inspect": cmd_inspect, "convert": cmd_convert, "validate": cmd_validate}[args.command](args)


if __name__ == "__main__":
    main()

