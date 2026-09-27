#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

SEED = 20260923

SRC_RE = re.compile(
    r"[A-Za-z0-9]+|"
    r"[\u3400-\u4DBF\u4E00-\u9FFF]|"
    r"[^\s]"
)

TGT_RE = re.compile(
    r"[A-Za-z]+(?:['’\-][A-Za-z]+)*|"
    r"\d+(?:[.,]\d+)*|"
    r"[\u3400-\u4DBF\u4E00-\u9FFF]|"
    r"[^\s]"
)


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


def toks(text, pattern):
    return [
        {
            "text": m.group(0),
            "start": m.start(),
            "end": m.end(),
        }
        for m in pattern.finditer(text)
    ]


def occurrences(text, sub):
    if not sub:
        return []

    out = []
    pos = 0

    while True:
        i = text.find(sub, pos)

        if i < 0:
            return out

        out.append(
            [i, i + len(sub)]
        )
        pos = i + 1


def source_patch_indices(
    source,
    source_span,
    src_tokens,
):
    occ = occurrences(
        source,
        source_span,
    )

    if len(occ) == 0:
        return "SOURCE_ANCHOR_MISSING", []

    if len(occ) > 1:
        return "SOURCE_ANCHOR_AMBIGUOUS", []

    a, b = occ[0]

    ids = [
        i
        for i, t in enumerate(src_tokens)
        if t["start"] < b
        and t["end"] > a
    ]

    if not ids:
        return "SOURCE_PATCH_TOKEN_EMPTY", []

    return "OK", ids


def consecutive_groups(xs):
    xs = sorted(set(xs))

    if not xs:
        return []

    out = [[xs[0]]]

    for x in xs[1:]:
        if x == out[-1][-1] + 1:
            out[-1].append(x)
        else:
            out.append([x])

    return out


def project_pairs(
    pairs,
    patch_ids,
    target_tokens,
):
    patch_ids = set(patch_ids)

    target_ids = sorted({
        int(j)
        for i, j in pairs
        if int(i) in patch_ids
    })

    spans = []

    for g in consecutive_groups(
        target_ids
    ):
        spans.append([
            int(
                target_tokens[
                    g[0]
                ]["start"]
            ),
            int(
                target_tokens[
                    g[-1]
                ]["end"]
            ),
        ])

    return spans


def char_union(spans):
    out = set()

    for a, b in spans:
        if a < b:
            out.update(
                range(
                    int(a),
                    int(b),
                )
            )

    return out


def span_relation(
    proposal,
    errors,
):
    p = char_union(proposal)
    e = char_union(errors)

    inter = len(p & e)

    if not p:
        return {
            "overlap": False,
            "overlap_chars": 0,
            "proposal_coverage": None,
            "error_coverage": None,
            "iou": None,
        }

    return {
        "overlap":
            inter > 0,

        "overlap_chars":
            inter,

        "proposal_coverage":
            inter / len(p),

        "error_coverage":
            (
                inter / len(e)
                if e else 0.0
            ),

        "iou":
            (
                inter / len(p | e)
                if (p | e)
                else 0.0
            ),
    }


def canonicalize(mt, span):
    s = int(span["start"])
    e = int(span["end"])
    text = span["text"]

    base = {
        "text": text,
        "confidence":
            float(
                span.get(
                    "confidence",
                    0.0,
                )
            ),

        "severity":
            span.get(
                "severity"
            ),

        "raw_start": s,
        "raw_end": e,
    }

    if not (
        0 <= s <= e <= len(mt)
    ):
        return {
            **base,
            "offset_state":
                "INVALID_BOUNDS",
            "start": None,
            "end": None,
        }

    raw = mt[s:e]

    if raw == text:
        return {
            **base,
            "offset_state": "EXACT",
            "start": s,
            "end": e,
        }

    if raw.strip() == text:
        left = (
            len(raw)
            - len(raw.lstrip())
        )

        right = (
            len(raw)
            - len(raw.rstrip())
        )

        cs = s + left
        ce = e - right

        if mt[cs:ce] != text:
            raise RuntimeError(
                "whitespace repair "
                "invariant failed"
            )

        return {
            **base,
            "offset_state":
                "WHITESPACE_REPAIRED",
            "start": cs,
            "end": ce,
        }

    return {
        **base,
        "offset_state":
            "INVALID_MISMATCH",
        "start": None,
        "end": None,
        "raw_slice": raw,
    }


def dev_rows(gold_path):
    rows = load_jsonl(
        gold_path
    )

    source_ids = sorted({
        int(x["source_id"])
        for x in rows
    })

    rng = random.Random(SEED)
    rng.shuffle(source_ids)

    dev_sources = set(
        source_ids[:48]
    )

    dev = [
        x
        for x in rows
        if (
            int(x["source_id"])
            in dev_sources
            and x["gold"]["status"]
            in {
                "PRESENT",
                "RESOLVED",
            }
        )
    ]

    assert len(dev) == 89

    assert sum(
        x["gold"]["status"]
        == "PRESENT"
        for x in dev
    ) == 60

    assert sum(
        x["gold"]["status"]
        == "RESOLVED"
        for x in dev
    ) == 29

    return dev


def prepare(args):
    out = Path(args.out)

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    dev = dev_rows(
        args.gold
    )

    cases = []

    for x in dev:
        st = toks(
            x["source"],
            SRC_RE,
        )

        tt = toks(
            x["current_translation"],
            TGT_RE,
        )

        state, patch_ids = (
            source_patch_indices(
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
                x[
                    "current_translation"
                ],

            "source_tokens":
                st,

            "target_tokens":
                tt,

            "source_anchor_state":
                state,

            "patch_source_indices":
                patch_ids,
        })

    write_jsonl(
        out / "cases89.jsonl",
        cases,
    )

    # One sentence pair per DEV source.
    unique = {}

    for x in cases:
        key = (
            x["source_id"],
            x["source"],
            x[
                "current_translation"
            ],
        )

        unique[key] = x

    sentence_rows = sorted(
        unique.values(),
        key=lambda x:
            x["source_id"],
    )

    valid_sentence_count = len(sentence_rows)

    if not (1 <= valid_sentence_count <= 48):
        raise RuntimeError(
            f"unexpected valid sentence count "
            f"{valid_sentence_count}"
        )

    write_jsonl(
        out
        / "sentences48.jsonl",
        sentence_rows,
    )

    with (
        out
        / "awesome_input48.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        for x in sentence_rows:
            src = " ".join(
                t["text"]
                for t in x[
                    "source_tokens"
                ]
            )

            tgt = " ".join(
                t["text"]
                for t in x[
                    "target_tokens"
                ]
            )

            assert "|||" not in src
            assert "|||" not in tgt

            f.write(
                src
                + " ||| "
                + tgt
                + "\n"
            )

    print(
        "CASES89 =",
        len(cases),
    )

    print(
        "UNIQUE_VALID_SENTENCES =",
        len(sentence_rows),
    )

    print(
        "GOLD_STATUS =",
        dict(
            Counter(
                x["gold_status"]
                for x in cases
            )
        ),
    )

    print(
        "SOURCE_ANCHORS =",
        dict(
            Counter(
                x[
                    "source_anchor_state"
                ]
                for x in cases
            )
        ),
    )

    print(
        "D_PREPARE=PASS"
    )


def simalign(args):
    from simalign import SentenceAligner

    out = Path(args.out)

    cases = load_jsonl(
        out / "cases89.jsonl"
    )

    sentences = load_jsonl(
        out / "sentences48.jsonl"
    )

    aligner = SentenceAligner(
        model=args.mbert,
        token_type="bpe",
        matching_methods="i",
        device="cpu",
        layer=8,
    )

    pair_map = {}

    for n, x in enumerate(
        sentences,
        start=1,
    ):
        src = [
            t["text"]
            for t in x[
                "source_tokens"
            ]
        ]

        tgt = [
            t["text"]
            for t in x[
                "target_tokens"
            ]
        ]

        a = aligner.get_word_aligns(
            src,
            tgt,
        )

        pairs = sorted([
            [int(i), int(j)]
            for i, j
            in a["itermax"]
        ])

        pair_map[
            x["source_id"]
        ] = pairs

        if n % 10 == 0:
            print(
                "C2_SENTENCES_DONE",
                n,
                "/48",
            )

    preds = []

    for x in cases:
        if (
            x["source_anchor_state"]
            != "OK"
        ):
            pred = []
            state = x[
                "source_anchor_state"
            ]

        else:
            pairs = pair_map[
                x["source_id"]
            ]

            pred = project_pairs(
                pairs,
                x[
                    "patch_source_indices"
                ],
                x["target_tokens"],
            )

            state = (
                "PROJECTED_SPAN"
                if pred
                else "UNALIGNED"
            )

        preds.append({
            "benchmark_id":
                x["benchmark_id"],
            "state":
                state,
            "pred_spans":
                pred,
        })

    write_jsonl(
        out
        / "c2_itermax89.jsonl",
        preds,
    )

    print(
        "C2_89=PASS"
    )


def parse_awesome(args):
    out = Path(args.out)

    cases = load_jsonl(
        out / "cases89.jsonl"
    )

    sentences = load_jsonl(
        out / "sentences48.jsonl"
    )

    lines = (
        out
        / "awesome_output48.txt"
    ).read_text(
        encoding="utf-8"
    ).splitlines()

    if len(lines) != len(sentences):
        raise RuntimeError(
            f"awesome lines "
            f"{len(lines)} != "
            f"{len(sentences)}"
        )

    pair_map = {}

    for x, line in zip(
        sentences,
        lines,
    ):
        pairs = []

        for z in line.split():
            i, j = z.split("-")
            pairs.append([
                int(i),
                int(j),
            ])

        pair_map[
            x["source_id"]
        ] = pairs

    preds = []

    for x in cases:
        if (
            x["source_anchor_state"]
            != "OK"
        ):
            pred = []
            state = x[
                "source_anchor_state"
            ]

        else:
            pred = project_pairs(
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
                if pred
                else "UNALIGNED"
            )

        preds.append({
            "benchmark_id":
                x["benchmark_id"],
            "state":
                state,
            "pred_spans":
                pred,
        })

    write_jsonl(
        out
        / "c3_awesome89.jsonl",
        preds,
    )

    print(
        "C3_89=PASS"
    )


def run_xcomet(args):
    import torch
    from comet.models import XCOMETMetric

    out = Path(args.out)

    sentences = load_jsonl(
        out / "sentences48.jsonl"
    )

    print(
        "===== LOAD XCOMET ====="
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
            "mt": x[
                "current_translation"
            ],
        }
        for x in sentences
    ]

    print(
        "===== PREDICT UNIQUE VALID SENTENCES ====="
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

    rows = []

    states = Counter()

    for x, score, spans in zip(
        sentences,
        pred.scores,
        pred.metadata.error_spans,
    ):
        fixed = [
            canonicalize(
                x[
                    "current_translation"
                ],
                s,
            )
            for s in spans
        ]

        states.update(
            s["offset_state"]
            for s in fixed
        )

        rows.append({
            "source_id":
                x["source_id"],

            "score":
                float(score),

            "error_spans":
                fixed,
        })

    write_jsonl(
        out
        / "xcomet_sentences48.jsonl",
        rows,
    )

    invalid = sum(
        v
        for k, v in states.items()
        if k.startswith(
            "INVALID"
        )
    )

    print(
        "OFFSET_STATES =",
        dict(states),
    )

    print(
        "INVALID_OFFSET_COUNT =",
        invalid,
    )

    if invalid != 0:
        raise RuntimeError(
            "XCOMET invalid offsets "
            "present; verifier evaluation "
            "must not silently continue."
        )

    print(
        "XCOMET_48=PASS"
    )


def confusion(rows):
    tp = sum(
        x["gold"] == "PRESENT"
        and x["pred"] == "PRESENT"
        for x in rows
    )

    fp = sum(
        x["gold"] == "RESOLVED"
        and x["pred"] == "PRESENT"
        for x in rows
    )

    tn = sum(
        x["gold"] == "RESOLVED"
        and x["pred"] == "RESOLVED"
        for x in rows
    )

    fn = sum(
        x["gold"] == "PRESENT"
        and x["pred"] == "RESOLVED"
        for x in rows
    )

    p = tp + fn
    n = tn + fp

    precision = (
        tp / (tp + fp)
        if tp + fp else None
    )

    recall = (
        tp / p
        if p else None
    )

    specificity = (
        tn / n
        if n else None
    )

    accuracy = (
        (tp + tn)
        / (p + n)
    )

    balanced = (
        (
            recall
            + specificity
        ) / 2
        if (
            recall is not None
            and specificity is not None
        )
        else None
    )

    return {
        "n": len(rows),
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
            accuracy,
        "balanced_accuracy":
            balanced,
    }


def evaluate(args):
    out = Path(args.out)

    cases = load_jsonl(
        out / "cases89.jsonl"
    )

    c2 = {
        x["benchmark_id"]: x
        for x in load_jsonl(
            out
            / "c2_itermax89.jsonl"
        )
    }

    c3 = {
        x["benchmark_id"]: x
        for x in load_jsonl(
            out
            / "c3_awesome89.jsonl"
        )
    }

    xcomet = {
        int(x["source_id"]): x
        for x in load_jsonl(
            out
            / "xcomet_sentences48.jsonl"
        )
    }

    item_rows = []

    for x in cases:
        xc = xcomet[
            int(x["source_id"])
        ]

        all_errors = xc[
            "error_spans"
        ]

        valid_errors = [
            s
            for s in all_errors
            if not s[
                "offset_state"
            ].startswith(
                "INVALID"
            )
        ]

        invalid_errors = [
            s
            for s in all_errors
            if s[
                "offset_state"
            ].startswith(
                "INVALID"
            )
        ]

        err_spans = [
            [
                int(s["start"]),
                int(s["end"]),
            ]
            for s in valid_errors
        ]

        # D0 asks only whether XCOMET detected
        # any current error in the sentence.
        # Offset validity is irrelevant to D0.
        whole_has_error = (
            len(all_errors) > 0
        )

        row = {
            "benchmark_id":
                x["benchmark_id"],
            "source_id":
                x["source_id"],
            "gold_status":
                x["gold_status"],
            "error_type":
                x["error_type"],
            "xcomet_score":
                xc["score"],
            "xcomet_error_count":
                len(valid_errors),
            "xcomet_error_spans":
                valid_errors,
        }

        row["D0_pred"] = (
            "PRESENT"
            if whole_has_error
            else "RESOLVED"
        )

        for name, proj in [
            ("D1_C2", c2),
            ("D2_C3", c3),
        ]:
            p = proj[
                x["benchmark_id"]
            ]

            proposal = p.get(
                "pred_spans",
                [],
            )

            # An INVALID XCOMET offset must never
            # silently become negative evidence.
            #
            # If the historical-patch proposal
            # intersects an uncertain raw XCOMET
            # interval, abstain on this patch.
            invalid_offset_overlap = False

            if proposal:
                proposal_chars = char_union(
                    proposal
                )

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

                    bad_chars = char_union([
                        [
                            int(a),
                            int(b),
                        ]
                    ])

                    if (
                        proposal_chars
                        & bad_chars
                    ):
                        invalid_offset_overlap = True
                        break

            rel = span_relation(
                proposal,
                err_spans,
            )

            row[
                name
                + "_proposal_state"
            ] = p["state"]

            row[
                name
                + "_proposal_spans"
            ] = proposal

            row[
                name
                + "_overlap"
            ] = rel["overlap"]

            row[
                name
                + "_proposal_coverage"
            ] = rel[
                "proposal_coverage"
            ]

            row[
                name
                + "_iou"
            ] = rel["iou"]

            overlapping = []

            if proposal:
                pchars = char_union(
                    proposal
                )

                for s in valid_errors:
                    schars = char_union([
                        [
                            s["start"],
                            s["end"],
                        ]
                    ])

                    if pchars & schars:
                        overlapping.append(
                            s
                        )

            row[
                name
                + "_max_overlap_confidence"
            ] = (
                max(
                    s["confidence"]
                    for s
                    in overlapping
                )
                if overlapping
                else None
            )

            if not proposal:
                row[
                    name + "_pred"
                ] = "ABSTAIN"

                row[
                    name + "_abstain_reason"
                ] = "NO_PROPOSAL"

            elif invalid_offset_overlap:
                row[
                    name + "_pred"
                ] = "ABSTAIN"

                row[
                    name + "_abstain_reason"
                ] = (
                    "INVALID_XCOMET_OFFSET_OVERLAP"
                )

            else:
                row[
                    name + "_pred"
                ] = (
                    "PRESENT"
                    if rel["overlap"]
                    else "RESOLVED"
                )

                row[
                    name + "_abstain_reason"
                ] = None

        item_rows.append(row)

    write_jsonl(
        out
        / "verifier_items89.jsonl",
        item_rows,
    )

    d0_rows = [
        {
            "gold":
                x["gold_status"],
            "pred":
                x["D0_pred"],
        }
        for x in item_rows
    ]

    report = {
        "split": "DEV_ONLY",
        "seed": SEED,
        "gold_counts":
            dict(
                Counter(
                    x["gold_status"]
                    for x in item_rows
                )
            ),
        "D0_WHOLE_SENTENCE_ANY_ERROR":
            confusion(
                d0_rows
            ),
    }

    for name in [
        "D1_C2",
        "D2_C3",
    ]:
        covered = [
            {
                "gold":
                    x["gold_status"],
                "pred":
                    x[name + "_pred"],
            }
            for x in item_rows
            if x[name + "_pred"]
            != "ABSTAIN"
        ]

        report[name] = {
            "coverage":
                len(covered)
                / len(item_rows),

            "abstain":
                len(item_rows)
                - len(covered),

            "classification":
                confusion(
                    covered
                ),

            "present_overlap_rate":
                mean([
                    1.0
                    if x[
                        name
                        + "_overlap"
                    ]
                    else 0.0
                    for x in item_rows
                    if (
                        x["gold_status"]
                        == "PRESENT"
                        and x[
                            name
                            + "_pred"
                        ]
                        != "ABSTAIN"
                    )
                ]),

            "resolved_overlap_rate":
                mean([
                    1.0
                    if x[
                        name
                        + "_overlap"
                    ]
                    else 0.0
                    for x in item_rows
                    if (
                        x["gold_status"]
                        == "RESOLVED"
                        and x[
                            name
                            + "_pred"
                        ]
                        != "ABSTAIN"
                    )
                ]),

            "mean_proposal_coverage_present":
                mean([
                    x[
                        name
                        + "_proposal_coverage"
                    ]
                    for x in item_rows
                    if (
                        x["gold_status"]
                        == "PRESENT"
                        and x[
                            name
                            + "_proposal_coverage"
                        ]
                        is not None
                    )
                ]),

            "mean_proposal_coverage_resolved":
                mean([
                    x[
                        name
                        + "_proposal_coverage"
                    ]
                    for x in item_rows
                    if (
                        x["gold_status"]
                        == "RESOLVED"
                        and x[
                            name
                            + "_proposal_coverage"
                        ]
                        is not None
                    )
                ]),
        }

    # Same-current-translation groups with mixed
    # historical patch statuses are especially
    # informative for patch-specific verification.
    by_source = defaultdict(list)

    for x in item_rows:
        by_source[
            x["source_id"]
        ].append(x)

    mixed = []

    for sid, xs in by_source.items():
        statuses = {
            x["gold_status"]
            for x in xs
        }

        if statuses == {
            "PRESENT",
            "RESOLVED",
        }:
            mixed.append({
                "source_id": sid,
                "n_patches": len(xs),
                "gold_statuses":
                    [
                        x["gold_status"]
                        for x in xs
                    ],
                "D0_preds":
                    [
                        x["D0_pred"]
                        for x in xs
                    ],
                "D1_preds":
                    [
                        x["D1_C2_pred"]
                        for x in xs
                    ],
                "D2_preds":
                    [
                        x["D2_C3_pred"]
                        for x in xs
                    ],
                "benchmark_ids":
                    [
                        x["benchmark_id"]
                        for x in xs
                    ],
            })

    report[
        "mixed_status_same_sentence_groups"
    ] = mixed

    (
        out / "verifier_summary.json"
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
        "===== GOLD ====="
    )
    print(
        report["gold_counts"]
    )

    print()
    print(
        "===== D0 WHOLE SENTENCE ====="
    )
    print(
        json.dumps(
            report[
                "D0_WHOLE_SENTENCE_ANY_ERROR"
            ],
            indent=2,
        )
    )

    for name in [
        "D1_C2",
        "D2_C3",
    ]:
        print()
        print(
            "=====",
            name,
            "====="
        )

        print(
            json.dumps(
                report[name],
                indent=2,
            )
        )

    print()
    print(
        "===== MIXED-STATUS SAME-SENTENCE GROUPS ====="
    )
    print(
        "COUNT =",
        len(mixed),
    )

    for x in mixed:
        print(
            json.dumps(
                x,
                ensure_ascii=False,
            )
        )

    print()
    print(
        "XCOMET_VERIFIER_DEV_V1=PASS"
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "mode",
        choices=[
            "prepare",
            "simalign",
            "parse-awesome",
            "xcomet",
            "evaluate",
        ],
    )

    ap.add_argument("--gold")
    ap.add_argument("--mbert")
    ap.add_argument("--ckpt")
    ap.add_argument("--base")

    ap.add_argument(
        "--out",
        required=True,
    )

    args = ap.parse_args()

    {
        "prepare": prepare,
        "simalign": simalign,
        "parse-awesome":
            parse_awesome,
        "xcomet": run_xcomet,
        "evaluate": evaluate,
    }[args.mode](args)


if __name__ == "__main__":
    main()
