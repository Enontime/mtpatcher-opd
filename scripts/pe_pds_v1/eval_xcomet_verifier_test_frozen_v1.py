#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20260923
THRESHOLD = 0.8


def load_dev_module(path):
    spec = importlib.util.spec_from_file_location(
        "verifier_dev_v1",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "cannot load validated DEV evaluator"
        )

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_jsonl(path):
    return [
        json.loads(x)
        for x in Path(path).read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]


def write_jsonl(path, rows):
    with Path(path).open(
        "w",
        encoding="utf-8",
    ) as f:
        for x in rows:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def get_test_rows(gold_path):
    rows = load_jsonl(gold_path)

    source_ids = sorted({
        int(x["source_id"])
        for x in rows
    })

    rng = random.Random(SEED)
    rng.shuffle(source_ids)

    dev_sources = set(
        source_ids[:48]
    )

    test_sources = set(
        source_ids[48:]
    )

    if dev_sources & test_sources:
        raise RuntimeError(
            "DEV/TEST source leakage"
        )

    test_all = [
        x
        for x in rows
        if int(x["source_id"])
        in test_sources
    ]

    valid = [
        x
        for x in test_all
        if x["gold"]["status"]
        in {
            "PRESENT",
            "RESOLVED",
        }
    ]

    return (
        rows,
        dev_sources,
        test_sources,
        test_all,
        valid,
    )


def prepare(args):
    mod = load_dev_module(
        args.dev_evaluator
    )

    (
        all_rows,
        dev_sources,
        test_sources,
        test_all,
        valid,
    ) = get_test_rows(
        args.gold
    )

    out = Path(args.out)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    cases = []

    for x in valid:
        st = mod.toks(
            x["source"],
            mod.SRC_RE,
        )

        tt = mod.toks(
            x["current_translation"],
            mod.TGT_RE,
        )

        state, patch_ids = (
            mod.source_patch_indices(
                x["source"],
                x["source_span"],
                st,
            )
        )

        cases.append({
            "benchmark_id":
                x["benchmark_id"],

            "source_id":
                int(x["source_id"]),

            "gold_status":
                x["gold"]["status"],

            "error_type":
                x["error_type"],

            "source":
                x["source"],

            "source_span":
                x["source_span"],

            "current_translation":
                x["current_translation"],

            "source_tokens":
                st,

            "target_tokens":
                tt,

            "source_anchor_state":
                state,

            "patch_source_indices":
                patch_ids,
        })

    # Verify every source_id has one current sentence.
    grouped = defaultdict(list)

    for x in cases:
        grouped[
            x["source_id"]
        ].append(x)

    sentences = []

    for sid in sorted(grouped):
        xs = grouped[sid]

        signatures = {
            (
                x["source"],
                x["current_translation"],
            )
            for x in xs
        }

        if len(signatures) != 1:
            raise RuntimeError(
                f"source_id {sid} has "
                f"multiple current translations"
            )

        sentences.append(xs[0])

    write_jsonl(
        out / "cases_test.jsonl",
        cases,
    )

    write_jsonl(
        out / "sentences_test.jsonl",
        sentences,
    )

    with (
        out / "awesome_input_test.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        for x in sentences:
            src = " ".join(
                t["text"]
                for t in x["source_tokens"]
            )

            tgt = " ".join(
                t["text"]
                for t in x["target_tokens"]
            )

            if "|||" in src or "|||" in tgt:
                raise RuntimeError(
                    "awesome-align delimiter collision"
                )

            f.write(
                src
                + " ||| "
                + tgt
                + "\n"
            )

    print(
        "ALL_GOLD_ITEMS =",
        len(all_rows),
    )

    print(
        "DEV_SOURCES =",
        len(dev_sources),
    )

    print(
        "TEST_SOURCES =",
        len(test_sources),
    )

    print(
        "TEST_ALL_ITEMS =",
        len(test_all),
    )

    print(
        "TEST_VALID_ITEMS =",
        len(valid),
    )

    print(
        "TEST_VALID_SENTENCES =",
        len(sentences),
    )

    print(
        "TEST_STATUS_COUNTS =",
        dict(
            Counter(
                x["gold_status"]
                for x in cases
            )
        ),
    )

    print(
        "TEST_SOURCE_ANCHORS =",
        dict(
            Counter(
                x["source_anchor_state"]
                for x in cases
            )
        ),
    )

    print(
        "FROZEN_TEST_PREPARE=PASS"
    )


def parse_awesome(args):
    mod = load_dev_module(
        args.dev_evaluator
    )

    out = Path(args.out)

    cases = load_jsonl(
        out / "cases_test.jsonl"
    )

    sentences = load_jsonl(
        out / "sentences_test.jsonl"
    )

    lines = (
        out / "awesome_output_test.txt"
    ).read_text(
        encoding="utf-8"
    ).splitlines()

    if len(lines) != len(sentences):
        raise RuntimeError(
            f"awesome output lines "
            f"{len(lines)} != "
            f"{len(sentences)} sentences"
        )

    pair_map = {}

    for x, line in zip(
        sentences,
        lines,
    ):
        pairs = []

        for item in line.split():
            i, j = item.split("-")

            i = int(i)
            j = int(j)

            if not (
                0 <= i
                < len(x["source_tokens"])
            ):
                raise RuntimeError(
                    "awesome source index "
                    "out of range"
                )

            if not (
                0 <= j
                < len(x["target_tokens"])
            ):
                raise RuntimeError(
                    "awesome target index "
                    "out of range"
                )

            pairs.append([i, j])

        pair_map[
            x["source_id"]
        ] = pairs

    preds = []

    for x in cases:
        if (
            x["source_anchor_state"]
            != "OK"
        ):
            spans = []
            state = x[
                "source_anchor_state"
            ]

        else:
            spans = mod.project_pairs(
                pair_map[
                    x["source_id"]
                ],
                x[
                    "patch_source_indices"
                ],
                x["target_tokens"],
            )

            state = (
                "PROJECTED_SPAN"
                if spans
                else "UNALIGNED"
            )

        preds.append({
            "benchmark_id":
                x["benchmark_id"],
            "state":
                state,
            "pred_spans":
                spans,
        })

    write_jsonl(
        out / "c3_test.jsonl",
        preds,
    )

    print(
        "TEST_C3_PROPOSALS =",
        dict(
            Counter(
                x["state"]
                for x in preds
            )
        ),
    )

    print(
        "FROZEN_TEST_AWESOME_PARSE=PASS"
    )


def confusion(rows, field):
    covered = [
        x
        for x in rows
        if x[field]
        != "ABSTAIN"
    ]

    tp = sum(
        x["gold_status"] == "PRESENT"
        and x[field] == "PRESENT"
        for x in covered
    )

    fp = sum(
        x["gold_status"] == "RESOLVED"
        and x[field] == "PRESENT"
        for x in covered
    )

    tn = sum(
        x["gold_status"] == "RESOLVED"
        and x[field] == "RESOLVED"
        for x in covered
    )

    fn = sum(
        x["gold_status"] == "PRESENT"
        and x[field] == "RESOLVED"
        for x in covered
    )

    recall = (
        tp / (tp + fn)
        if tp + fn else None
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp else None
    )

    precision = (
        tp / (tp + fp)
        if tp + fp else None
    )

    return {
        "total_items":
            len(rows),

        "covered_n":
            len(covered),

        "abstain_n":
            len(rows) - len(covered),

        "coverage":
            len(covered) / len(rows),

        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,

        "precision_present":
            precision,

        "recall_present":
            recall,

        "specificity_resolved":
            specificity,

        "accuracy":
            (
                (tp + tn) / len(covered)
                if covered else None
            ),

        "balanced_accuracy":
            (
                (recall + specificity) / 2
                if (
                    recall is not None
                    and specificity
                    is not None
                )
                else None
            ),
    }


def xcomet_eval(args):
    import torch
    from comet.models import XCOMETMetric

    mod = load_dev_module(
        args.dev_evaluator
    )

    out = Path(args.out)

    cases = load_jsonl(
        out / "cases_test.jsonl"
    )

    sentences = load_jsonl(
        out / "sentences_test.jsonl"
    )

    proposals = {
        x["benchmark_id"]: x
        for x in load_jsonl(
            out / "c3_test.jsonl"
        )
    }

    print(
        "===== LOAD FROZEN XCOMET ====="
    )

    t0 = time.time()

    model = XCOMETMetric.load_from_checkpoint(
        checkpoint_path=args.ckpt,
        pretrained_model=args.base,
        load_pretrained_weights=False,
        local_files_only=True,
        map_location=torch.device(
            "cpu"
        ),
        strict=False,
        layer_transformation="softmax",
    )

    print(
        "LOAD_SECONDS =",
        round(
            time.time() - t0,
            2,
        ),
    )

    data = [
        {
            "src": x["source"],
            "mt":
                x["current_translation"],
        }
        for x in sentences
    ]

    print(
        "===== PREDICT FROZEN TEST ====="
    )

    t1 = time.time()

    pred = model.predict(
        data,
        batch_size=1,
        gpus=0,
        accelerator="cpu",
        num_workers=0,
        progress_bar=False,
        length_batching=False,
    )

    sec = time.time() - t1

    print(
        "PREDICT_SECONDS =",
        round(sec, 2),
    )

    print(
        "SECONDS_PER_SENTENCE =",
        round(
            sec / len(sentences),
            3,
        ),
    )

    xrows = []
    offset_states = Counter()

    for x, score, spans in zip(
        sentences,
        pred.scores,
        pred.metadata.error_spans,
    ):
        fixed = [
            mod.canonicalize(
                x["current_translation"],
                s,
            )
            for s in spans
        ]

        offset_states.update(
            s["offset_state"]
            for s in fixed
        )

        xrows.append({
            "source_id":
                x["source_id"],
            "score":
                float(score),
            "error_spans":
                fixed,
        })

    write_jsonl(
        out
        / "xcomet_test_sentences.jsonl",
        xrows,
    )

    xmap = {
        int(x["source_id"]): x
        for x in xrows
    }

    result_rows = []

    for x in cases:
        prop = proposals[
            x["benchmark_id"]
        ]

        proposal = prop.get(
            "pred_spans",
            [],
        )

        xc = xmap[
            int(x["source_id"])
        ]

        valid_errors = [
            s
            for s in xc["error_spans"]
            if not s[
                "offset_state"
            ].startswith(
                "INVALID"
            )
        ]

        invalid_errors = [
            s
            for s in xc["error_spans"]
            if s[
                "offset_state"
            ].startswith(
                "INVALID"
            )
        ]

        if not proposal:
            primary = "ABSTAIN"
            reason = "NO_PROPOSAL"
            coverage = None

        else:
            pchars = mod.char_union(
                proposal
            )

            invalid_overlap = False

            for bad in invalid_errors:
                a = bad.get(
                    "raw_start"
                )
                b = bad.get(
                    "raw_end"
                )

                if (
                    a is None
                    or b is None
                ):
                    continue

                badchars = (
                    mod.char_union([
                        [
                            int(a),
                            int(b),
                        ]
                    ])
                )

                if pchars & badchars:
                    invalid_overlap = True
                    break

            if invalid_overlap:
                primary = "ABSTAIN"
                reason = (
                    "INVALID_XCOMET_"
                    "OFFSET_OVERLAP"
                )
                coverage = None

            else:
                echars = set()

                for s in valid_errors:
                    echars |= (
                        mod.char_union([
                            [
                                int(
                                    s["start"]
                                ),
                                int(
                                    s["end"]
                                ),
                            ]
                        ])
                    )

                intersection = len(
                    pchars & echars
                )

                coverage = (
                    intersection
                    / len(pchars)
                )

                primary = (
                    "PRESENT"
                    if coverage
                    >= THRESHOLD
                    else "RESOLVED"
                )

                reason = None

        any_overlap = (
            "ABSTAIN"
            if primary == "ABSTAIN"
            else (
                "PRESENT"
                if coverage > 0
                else "RESOLVED"
            )
        )

        result_rows.append({
            "benchmark_id":
                x["benchmark_id"],
            "source_id":
                x["source_id"],
            "error_type":
                x["error_type"],
            "gold_status":
                x["gold_status"],

            "proposal_state":
                prop["state"],

            "proposal_spans":
                proposal,

            "proposal_coverage":
                coverage,

            "threshold":
                THRESHOLD,

            "primary_pred":
                primary,

            "any_overlap_pred":
                any_overlap,

            "abstain_reason":
                reason,
        })

    write_jsonl(
        out
        / "frozen_test_items.jsonl",
        result_rows,
    )

    report = {
        "protocol":
            "frozen_verifier_v1",

        "split":
            "TEST",

        "threshold":
            THRESHOLD,

        "gold_counts":
            dict(
                Counter(
                    x["gold_status"]
                    for x in result_rows
                )
            ),

        "offset_states":
            dict(offset_states),

        "abstain_reasons":
            dict(
                Counter(
                    x["abstain_reason"]
                    for x in result_rows
                    if x[
                        "primary_pred"
                    ]
                    == "ABSTAIN"
                )
            ),

        "primary_threshold_0_8":
            confusion(
                result_rows,
                "primary_pred",
            ),

        "secondary_any_overlap":
            confusion(
                result_rows,
                "any_overlap_pred",
            ),
    }

    (
        out
        / "frozen_test_summary.json"
    ).write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "===== FROZEN TEST GOLD ====="
    )
    print(
        report["gold_counts"]
    )

    print()
    print(
        "===== OFFSET STATES ====="
    )
    print(
        report["offset_states"]
    )

    print()
    print(
        "===== PRIMARY: "
        "C3 + XCOMET + COVERAGE>=0.8 ====="
    )

    print(
        json.dumps(
            report[
                "primary_threshold_0_8"
            ],
            indent=2,
        )
    )

    print()
    print(
        "===== SECONDARY: "
        "SAME PIPELINE ANY-OVERLAP ====="
    )

    print(
        json.dumps(
            report[
                "secondary_any_overlap"
            ],
            indent=2,
        )
    )

    print()
    print(
        "ABSTAIN_REASONS =",
        report[
            "abstain_reasons"
        ],
    )

    print()
    print(
        "FROZEN_TEST_EVALUATION=PASS"
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "mode",
        choices=[
            "prepare",
            "parse-awesome",
            "xcomet-eval",
        ],
    )

    ap.add_argument("--gold")
    ap.add_argument("--dev-evaluator")
    ap.add_argument("--ckpt")
    ap.add_argument("--base")

    ap.add_argument(
        "--out",
        required=True,
    )

    args = ap.parse_args()

    {
        "prepare": prepare,
        "parse-awesome":
            parse_awesome,
        "xcomet-eval":
            xcomet_eval,
    }[args.mode](args)


if __name__ == "__main__":
    main()
