#!/usr/bin/env python3
"""Shared TensorBoard contracts for targeted OPD/SFT experiments.

The module is intentionally side-effect free until TensorBoard is explicitly
enabled.  It never imports torch or tensorboard on the disabled path.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


VERSION = "targeted_tensorboard_v1"


METRIC_DEFINITIONS: dict[str, dict[str, str]] = {
    "opd/kl_recent100": {
        "unit": "nats_per_response_token_row_mean",
        "definition": "Mean of the most recent 100 per-row OPD KL values; historical field kl100.",
        "comparable_with": "the same OPD objective and implementation only",
    },
    "sft/ce_microbatch_mean_recent50": {
        "unit": "nats_per_token_microbatch_mean",
        "definition": "Mean of the most recent 50 historical microbatch HF losses; historical field loss50.",
        "comparable_with": "the same historical SFT loss reduction only",
    },
    "sft/ce_per_response_token": {
        "unit": "nats_per_response_token",
        "definition": "Whole-GA-group response CE sum divided by the exact valid response-token count.",
        "comparable_with": "SFT runs using the same token-normalized CE contract",
    },
    "optim/lr": {"unit": "learning_rate", "definition": "Scheduler LR after optimizer update."},
    "optim/grad_norm_preclip": {"unit": "l2_norm", "definition": "Global gradient norm returned before clipping."},
    "optim/grad_clip_scale": {"unit": "ratio", "definition": "min(1, max_grad_norm / grad_norm_preclip)."},
    "train/response_tokens": {"unit": "tokens", "definition": "Valid response tokens in the completed optimizer update group."},
    "train/rollout_length_recent200": {"unit": "tokens", "definition": "Historical rolling mean of the most recent 200 Student rollout lengths; field rollout_len200."},
    "train/rollout_length_mean": {"unit": "tokens", "definition": "Mean Student rollout length in the current optimizer update group."},
    "train/step_seconds": {"unit": "seconds", "definition": "Wall-clock seconds for the completed optimizer update group."},
    "knowledge/chemistry_strict": {"unit": "accuracy", "definition": "Frozen evaluator strict canonical-term accuracy."},
    "knowledge/chemistry_spacing_normalized": {"unit": "accuracy", "definition": "Frozen evaluator spacing-normalized chemistry accuracy."},
    "knowledge/chemistry_ci": {"unit": "accuracy", "definition": "Frozen C.I. slice accuracy."},
    "knowledge/chemistry_non_ci": {"unit": "accuracy", "definition": "Frozen non-C.I. slice accuracy."},
    "knowledge/chemistry_uc": {"unit": "accuracy", "definition": "Frozen UC split accuracy."},
    "knowledge/chemistry_uw": {"unit": "accuracy", "definition": "Frozen UW split accuracy."},
    "knowledge/idiom_strict": {"unit": "accuracy", "definition": "Frozen idiom strict accuracy, only when explicitly present."},
    "quality/bleu": {"unit": "BLEU", "definition": "Existing evaluation macro BLEU; never recomputed by TensorBoard."},
    "quality/chrf": {"unit": "chrF", "definition": "Existing evaluation macro chrF; never recomputed by TensorBoard."},
    "knowledge/gain": {"unit": "samples", "definition": "Per-eval-id incorrect-to-correct count versus an explicitly identified baseline."},
    "knowledge/loss": {"unit": "samples", "definition": "Per-eval-id correct-to-incorrect count versus an explicitly identified baseline."},
}


MECHANISM_DIAGNOSTIC_SCHEMA: dict[str, Any] = {
    "schema_version": "targeted_prefix_diagnostics_v1",
    "required_identity_fields": [
        "run_id", "checkpoint_identity_sha256", "optimizer_step",
        "eval_id", "eval_data_sha256", "prompt_sha256",
        "decoding_config_sha256", "scorer_version",
    ],
    "event_fields": [
        "sample_id", "domain", "ci_group", "prefix_kind",
        "token_position", "is_term_position", "teacher_probability",
        "student_probability", "token_kl", "effective_term_event",
    ],
    "prefix_kind_values": ["student_prefix", "teacher_prefix"],
    "ci_group_values": ["ci", "non_ci", "not_applicable"],
    "missing_policy": "NOT_AVAILABLE; never infer from aggregate accuracy or KL",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def clip_scale(grad_norm_preclip: float, max_grad_norm: float) -> float:
    require(math.isfinite(grad_norm_preclip) and grad_norm_preclip >= 0, "invalid pre-clip gradient norm")
    require(math.isfinite(max_grad_norm) and max_grad_norm > 0, "invalid max gradient norm")
    return 1.0 if grad_norm_preclip <= max_grad_norm else max_grad_norm / grad_norm_preclip


def _summary_writer_class() -> Any:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "TensorBoard requested but unavailable. Install/activate the project "
            "environment containing torch.utils.tensorboard, or omit --tensorboard."
        ) from exc
    return SummaryWriter


def event_accumulator_class() -> Any:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "TensorBoard event validation requested but tensorboard is unavailable."
        ) from exc
    return EventAccumulator


@dataclass(frozen=True)
class ScalarRecord:
    tag: str
    step: int
    value: float
    wall_time: float | None = None
    source_field: str | None = None
    source_path: str | None = None
    eval_id: str | None = None

    def validate(self) -> None:
        require(self.tag in METRIC_DEFINITIONS, f"unknown TensorBoard tag: {self.tag}")
        require(isinstance(self.step, int) and self.step >= 0, f"invalid step for {self.tag}")
        require(math.isfinite(float(self.value)), f"non-finite scalar for {self.tag} step={self.step}")

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def validate_scalar_records(records: Iterable[ScalarRecord]) -> list[ScalarRecord]:
    checked = list(records)
    seen: dict[tuple[str, int], float] = {}
    previous: dict[str, int] = {}
    for record in checked:
        record.validate()
        key = (record.tag, record.step)
        require(key not in seen, f"duplicate scalar tag/step: {record.tag}@{record.step}")
        require(record.step > previous.get(record.tag, -1), f"non-monotonic scalar step: {record.tag}@{record.step}")
        seen[key] = float(record.value)
        previous[record.tag] = record.step
    return checked


class NativeTensorBoard:
    """Optional main-process writer for future targeted trainers.

    Resume uses SummaryWriter's purge_step contract.  Events at and after
    durable_step + 1 are hidden, so updates replayed after the durable
    checkpoint do not create a second visible trajectory.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        log_dir: Path,
        run_id: str,
        durable_step: int,
        rank: int = 0,
        write_every: int = 1,
        writer_class: Any | None = None,
    ) -> None:
        require(write_every >= 1, "tensorboard write_every must be >= 1")
        require(durable_step >= 0, "tensorboard durable_step must be >= 0")
        self.enabled = bool(enabled and rank == 0)
        self.log_dir = Path(log_dir)
        self.run_id = run_id
        self.durable_step = durable_step
        self.write_every = write_every
        self._last_step: dict[str, int] = {}
        self._writer = None
        if self.enabled:
            cls = writer_class or _summary_writer_class()
            self.log_dir.mkdir(parents=True, exist_ok=True)
            contract_path = self.log_dir / "native_tensorboard_contract.json"
            if contract_path.is_file():
                existing = json.loads(contract_path.read_text(encoding="utf-8"))
                require(
                    existing.get("run_id") == run_id,
                    f"TensorBoard logdir belongs to another run: {self.log_dir}",
                )
                require(
                    existing.get("version") == VERSION
                    and existing.get("write_every") == write_every,
                    f"TensorBoard logdir contract changed: {self.log_dir}",
                )
            purge_step = durable_step + 1 if durable_step > 0 else None
            self._writer = cls(log_dir=str(self.log_dir), purge_step=purge_step)
            metadata = {
                "version": VERSION,
                "run_id": run_id,
                "durable_step_at_open": durable_step,
                "resume_policy": "purge_events_at_or_after_durable_step_plus_one",
                "write_every": write_every,
            }
            contract_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    def add_scalars(
        self,
        step: int,
        values: Mapping[str, float | int],
        *,
        wall_time: float | None = None,
        force: bool = False,
    ) -> None:
        if not self.enabled:
            return
        require(step > self.durable_step, f"TensorBoard step {step} is not after durable step {self.durable_step}")
        if not force and step % self.write_every:
            return
        for tag, raw_value in values.items():
            require(tag in METRIC_DEFINITIONS, f"unknown TensorBoard tag: {tag}")
            value = float(raw_value)
            require(math.isfinite(value), f"non-finite TensorBoard scalar: {tag}@{step}")
            require(step > self._last_step.get(tag, self.durable_step), f"duplicate/non-monotonic native scalar: {tag}@{step}")
            self._writer.add_scalar(tag, value, global_step=step, walltime=wall_time)
            self._last_step[tag] = step

    def log_sft_update(
        self,
        *,
        step: int,
        ce_per_response_token: float,
        lr: float,
        grad_norm_preclip: float,
        max_grad_norm: float,
        response_tokens: int,
        step_seconds: float,
    ) -> None:
        self.add_scalars(step, {
            "sft/ce_per_response_token": ce_per_response_token,
            "optim/lr": lr,
            "optim/grad_norm_preclip": grad_norm_preclip,
            "optim/grad_clip_scale": clip_scale(grad_norm_preclip, max_grad_norm),
            "train/response_tokens": response_tokens,
            "train/step_seconds": step_seconds,
        })

    def log_opd_update(
        self,
        *,
        step: int,
        kl_recent100: float,
        lr: float,
        grad_norm_preclip: float,
        max_grad_norm: float,
        response_tokens: int,
        rollout_length_mean: float,
        step_seconds: float,
    ) -> None:
        self.add_scalars(step, {
            "opd/kl_recent100": kl_recent100,
            "optim/lr": lr,
            "optim/grad_norm_preclip": grad_norm_preclip,
            "optim/grad_clip_scale": clip_scale(grad_norm_preclip, max_grad_norm),
            "train/response_tokens": response_tokens,
            "train/rollout_length_mean": rollout_length_mean,
            "train/step_seconds": step_seconds,
        })

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None

