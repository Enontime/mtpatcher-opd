#!/usr/bin/env python3

from pathlib import Path
from hashlib import sha256
import json
import shutil

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter


ROOT = Path("/workspace/mtpatcher")

SEQKD_EVENT = ROOT / (
    "runs/sft/"
    "canonical_seqkd_broad20k_qwen3_06b_8b_20260903_133457/"
    "tensorboard_log/mtpatcher-sft/"
    "canonical-seqkd-broad20k-qwen3-06b-8b/"
    "events.out.tfevents.1788442529.8442009ba1f5.415103.0"
)

OPD_EVENT = ROOT / (
    "repo/verl-v0.9.0/tensorboard_log/mtpatcher-opd/"
    "canonical-fkl-topk-broad20k-qwen3-06b-8b/"
    "events.out.tfevents.1788587310.8442009ba1f5.1473680.0"
)

OUT = ROOT / "runs/science/paper_tensorboard_clean_20260915"


SEQKD_TAGS = {
    "train/loss": "train/loss",
    "train/lr": "train/lr",
    "train/grad_norm": "train/grad_norm",
    "train/mfu": "system/mfu",
}


OPD_TAGS = {
    "actor/grad_norm": "train/grad_norm",
    "actor/lr": "train/lr",
    "actor/entropy": "train/entropy",
    "response_length/mean": "train/response_length",

    "actor/distillation/loss": "distill/kl_loss",
    "actor/distillation/teacher_mass": "distill/teacher_mass",
    "actor/distillation/student_mass": "distill/student_mass",
    "actor/distillation/overlap_ratio": "distill/overlap_ratio",

    "perf/throughput": "system/throughput",
    "perf/time_per_step": "system/step_time_s",
}


def file_sha256(path: Path) -> str:
    h = sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def convert(source: Path, destination: Path, mapping: dict[str, str]):
    if not source.is_file():
        raise RuntimeError(f"missing source event: {source}")

    destination.mkdir(parents=True, exist_ok=True)

    ea = EventAccumulator(
        str(source),
        size_guidance={"scalars": 0},
    )
    ea.Reload()

    available = set(ea.Tags().get("scalars", []))

    missing = sorted(set(mapping) - available)
    if missing:
        raise RuntimeError(
            f"missing expected tags in {source}: {missing}"
        )

    writer = SummaryWriter(log_dir=str(destination))

    counts = {}

    for raw_tag, clean_tag in mapping.items():
        values = ea.Scalars(raw_tag)

        for item in values:
            writer.add_scalar(
                clean_tag,
                item.value,
                global_step=item.step,
                walltime=item.wall_time,
            )

        counts[clean_tag] = len(values)

    writer.flush()
    writer.close()

    return counts


def main():
    if OUT.exists():
        marker = OUT / ".generated_clean_tensorboard"

        if not marker.is_file():
            raise RuntimeError(
                f"refuse to replace unmanaged directory: {OUT}"
            )

        shutil.rmtree(OUT)

    seqkd_out = OUT / "seqkd_full_20k"

    opd_out = (
        OUT
        / "opd_full_20k"
        / "steps_1251_3750"
    )

    seqkd_counts = convert(
        SEQKD_EVENT,
        seqkd_out,
        SEQKD_TAGS,
    )

    opd_counts = convert(
        OPD_EVENT,
        opd_out,
        OPD_TAGS,
    )

    provenance = {
        "kind": "DERIVED_FROM_RAW_TENSORBOARD",
        "created_for": "paper-facing human-readable TensorBoard",
        "sources": {
            "seqkd": {
                "path": str(SEQKD_EVENT),
                "sha256": file_sha256(SEQKD_EVENT),
            },
            "opd": {
                "path": str(OPD_EVENT),
                "sha256": file_sha256(OPD_EVENT),
            },
        },
        "tag_mapping": {
            "seqkd": SEQKD_TAGS,
            "opd": OPD_TAGS,
        },
        "scalar_counts": {
            "seqkd": seqkd_counts,
            "opd": opd_counts,
        },
        "notes": [
            "Raw TensorBoard events remain unchanged.",
            "Historical events contain no native benchmark-validation curve.",
            "Zero-valued critic bookkeeping is intentionally omitted.",
        ],
    }

    OUT.mkdir(parents=True, exist_ok=True)

    (OUT / "provenance.json").write_text(
        json.dumps(
            provenance,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    (OUT / ".generated_clean_tensorboard").write_text(
        "DERIVED_FROM_RAW_TENSORBOARD\n",
        encoding="utf-8",
    )

    print("CLEAN_TENSORBOARD_BUILD=PASS")
    print(f"OUT={OUT}")
    print(f"SEQKD_TAGS={len(SEQKD_TAGS)}")
    print(f"OPD_TAGS={len(OPD_TAGS)}")


if __name__ == "__main__":
    main()
