#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import sacrebleu


EXPECTED_COUNTS = {
    "train_probe": 1024,
    "wmt24": 998,
    "flores": 1012,
    "challenge": 197,
}

BENCHMARKS = (
    "wmt24",
    "flores",
    "challenge",
)


def _as_text(value, name: str, index: int) -> str:
    if not isinstance(value, str):
        raise TypeError(
            f"{name}[{index}] must be str, "
            f"got {type(value)!r}"
        )

    value = value.strip()

    if not value:
        raise ValueError(
            f"{name}[{index}] is empty"
        )

    return value


def compute_mt_metrics(
    *,
    data_sources: Iterable[str],
    references: Iterable[str],
    predictions: Iterable[str],
    strict_counts: bool = True,
) -> dict[str, float]:
    data_sources = list(data_sources)
    references = list(references)
    predictions = list(predictions)

    n = len(data_sources)

    if len(references) != n or len(predictions) != n:
        raise ValueError(
            "validation vector length mismatch: "
            f"data_sources={n}, "
            f"references={len(references)}, "
            f"predictions={len(predictions)}"
        )

    grouped_refs: dict[str, list[str]] = defaultdict(list)
    grouped_preds: dict[str, list[str]] = defaultdict(list)

    for i, (dataset, ref, pred) in enumerate(
        zip(
            data_sources,
            references,
            predictions,
        )
    ):
        if dataset not in EXPECTED_COUNTS:
            raise ValueError(
                f"unknown data_source[{i}]={dataset!r}"
            )

        grouped_refs[dataset].append(
            _as_text(ref, "reference", i)
        )

        grouped_preds[dataset].append(
            _as_text(pred, "prediction", i)
        )

    if strict_counts:
        for dataset, expected in EXPECTED_COUNTS.items():
            got = len(grouped_refs[dataset])

            if got != expected:
                raise ValueError(
                    f"{dataset}: rows={got}, "
                    f"expected={expected}"
                )

    scores = {}

    for dataset in EXPECTED_COUNTS:
        refs = grouped_refs[dataset]
        preds = grouped_preds[dataset]

        if not refs:
            continue

        bleu = sacrebleu.corpus_bleu(
            preds,
            [refs],
        ).score

        chrf = sacrebleu.corpus_chrf(
            preds,
            [refs],
        ).score

        scores[dataset] = {
            "bleu": float(bleu),
            "chrf": float(chrf),
        }

    metrics: dict[str, float] = {}

    if "train_probe" in scores:
        metrics[
            "compare/train_probe/teacher_bleu"
        ] = scores["train_probe"]["bleu"]

        metrics[
            "compare/train_probe/teacher_chrf"
        ] = scores["train_probe"]["chrf"]

    for dataset in BENCHMARKS:
        if dataset not in scores:
            continue

        metrics[
            f"compare/benchmark/{dataset}_bleu"
        ] = scores[dataset]["bleu"]

        metrics[
            f"compare/benchmark/{dataset}_chrf"
        ] = scores[dataset]["chrf"]

    if all(x in scores for x in BENCHMARKS):
        metrics[
            "compare/benchmark/macro_bleu"
        ] = sum(
            scores[x]["bleu"]
            for x in BENCHMARKS
        ) / len(BENCHMARKS)

        metrics[
            "compare/benchmark/macro_chrf"
        ] = sum(
            scores[x]["chrf"]
            for x in BENCHMARKS
        ) / len(BENCHMARKS)

    return metrics
