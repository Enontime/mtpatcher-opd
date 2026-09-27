#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import re
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
        j = text.find(sub, pos)

        if j < 0:
            break

        out.append(
            (j, j + len(sub))
        )

        pos = j + 1

    return out


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
        return (
            "SOURCE_ANCHOR_MISSING",
            [],
        )

    if len(occ) > 1:
        return (
            "SOURCE_ANCHOR_AMBIGUOUS",
            [],
        )

    a, b = occ[0]

    ids = [
        i
        for i, t in enumerate(src_tokens)
        if (
            t["start"] < b
            and t["end"] > a
        )
    ]

    if not ids:
        return (
            "SOURCE_PATCH_TOKEN_EMPTY",
            [],
        )

    return "OK", ids


def groups(xs):
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

    js = sorted({
        int(j)
        for i, j in pairs
        if int(i) in patch_ids
    })

    if not js:
        return []

    spans = []

    for g in groups(js):
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
    s = set()

    for a, b in spans:
        if a == b:
            continue

        s.update(
            range(
                int(a),
                int(b),
            )
        )

    return s


def iou_f1(gold, pred):
    g = char_union(gold)
    p = char_union(pred)

    if not g and not p:
        return 1.0, 1.0

    if not g or not p:
        return 0.0, 0.0

    inter = len(g & p)

    return (
        inter / len(g | p),
        2 * inter / (
            len(g) + len(p)
        ),
    )


def load_jsonl(path):
    return [
        json.loads(x)
        for x in Path(path)
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
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


def build_cases(gold_path):
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
            == "PRESENT"
        )
    ]

    assert len(dev) == 60

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

            "error_type":
                x["error_type"],

            "source":
                x["source"],

            "source_span":
                x["source_span"],

            "current_translation":
                x["current_translation"],

            "gold_span_type":
                x["gold"]["span_type"],

            "gold_spans":
                x["gold"]["current_spans"],

            "source_tokens":
                st,

            "target_tokens":
                tt,

            "source_anchor_state":
                state,

            "patch_source_indices":
                patch_ids,
        })

    return cases


def prepare(args):
    out = Path(args.out)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    cases = build_cases(
        args.gold
    )

    write_jsonl(
        out / "cases.jsonl",
        cases,
    )

    with (
        out
        / "awesome_input.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        for x in cases:
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

    c = Counter(
        x["source_anchor_state"]
        for x in cases
    )

    print(
        "CASES =",
        len(cases),
    )

    print(
        "SOURCE_ANCHORS =",
        dict(c),
    )

    print(
        "C_PREPARE=PASS"
    )


def run_simalign(args):
    from simalign import (
        SentenceAligner,
    )

    cases = load_jsonl(
        Path(args.out)
        / "cases.jsonl"
    )

    aligner = SentenceAligner(
        model=args.mbert,
        token_type="bpe",
        matching_methods="ai",
        device="cpu",
        layer=8,
    )

    argmax_rows = []
    itermax_rows = []

    for n, x in enumerate(
        cases,
        start=1,
    ):
        base = {
            "benchmark_id":
                x["benchmark_id"],

            "source_anchor_state":
                x[
                    "source_anchor_state"
                ],
        }

        if (
            x["source_anchor_state"]
            != "OK"
        ):
            argmax_rows.append({
                **base,
                "state":
                    x[
                        "source_anchor_state"
                    ],
                "pairs": [],
                "pred_spans": [],
            })

            itermax_rows.append({
                **base,
                "state":
                    x[
                        "source_anchor_state"
                    ],
                "pairs": [],
                "pred_spans": [],
            })

            continue

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

        for key, dest in [
            (
                "inter",
                argmax_rows,
            ),
            (
                "itermax",
                itermax_rows,
            ),
        ]:
            pairs = sorted(
                [
                    [
                        int(i),
                        int(j),
                    ]
                    for i, j
                    in a[key]
                ]
            )

            pred = project_pairs(
                pairs,
                x[
                    "patch_source_indices"
                ],
                x["target_tokens"],
            )

            dest.append({
                **base,
                "state":
                    (
                        "PROJECTED_SPAN"
                        if pred
                        else
                        "UNALIGNED"
                    ),

                "pairs":
                    pairs,

                "pred_spans":
                    pred,
            })

        if n % 10 == 0:
            print(
                "SIMALIGN_DONE",
                n,
                "/",
                len(cases),
            )

    write_jsonl(
        Path(args.out)
        / "c1_simalign_argmax.jsonl",
        argmax_rows,
    )

    write_jsonl(
        Path(args.out)
        / "c2_simalign_itermax.jsonl",
        itermax_rows,
    )

    print(
        "SIMALIGN_DEV=PASS"
    )


def parse_awesome(args):
    out = Path(args.out)

    cases = load_jsonl(
        out / "cases.jsonl"
    )

    lines = (
        out
        / "awesome_output.txt"
    ).read_text(
        encoding="utf-8"
    ).splitlines()

    if len(lines) != len(cases):
        raise RuntimeError(
            "awesome line count "
            f"{len(lines)} != "
            f"{len(cases)}"
        )

    rows = []

    for x, line in zip(
        cases,
        lines,
    ):
        base = {
            "benchmark_id":
                x["benchmark_id"],

            "source_anchor_state":
                x[
                    "source_anchor_state"
                ],
        }

        if (
            x["source_anchor_state"]
            != "OK"
        ):
            rows.append({
                **base,
                "state":
                    x[
                        "source_anchor_state"
                    ],
                "pairs": [],
                "pred_spans": [],
            })

            continue

        pairs = []

        for z in line.split():
            i, j = z.split("-")

            i = int(i)
            j = int(j)

            if not (
                0 <= i
                < len(
                    x[
                        "source_tokens"
                    ]
                )
            ):
                raise RuntimeError(
                    "awesome source index "
                    "out of range"
                )

            if not (
                0 <= j
                < len(
                    x[
                        "target_tokens"
                    ]
                )
            ):
                raise RuntimeError(
                    "awesome target index "
                    "out of range"
                )

            pairs.append(
                [i, j]
            )

        pred = project_pairs(
            pairs,
            x[
                "patch_source_indices"
            ],
            x["target_tokens"],
        )

        rows.append({
            **base,

            "state":
                (
                    "PROJECTED_SPAN"
                    if pred
                    else
                    "UNALIGNED"
                ),

            "pairs":
                sorted(pairs),

            "pred_spans":
                pred,
        })

    write_jsonl(
        out
        / "c3_awesome_align.jsonl",
        rows,
    )

    print(
        "AWESOME_PARSE=PASS"
    )


def metric_block(
    cases,
    preds,
    subset_ids=None,
):
    pmap = {
        x["benchmark_id"]: x
        for x in preds
    }

    xs = [
        x
        for x in cases
        if (
            subset_ids is None
            or x["benchmark_id"]
            in subset_ids
        )
    ]

    span_xs = [
        x
        for x in xs
        if x[
            "gold_span_type"
        ] in (
            "TOKEN_SPAN",
            "MULTI_SPAN",
        )
    ]

    vals = []

    for x in span_xs:
        p = pmap[
            x["benchmark_id"]
        ]

        pred = p.get(
            "pred_spans",
            [],
        )

        iou, f1 = iou_f1(
            x["gold_spans"],
            pred,
        )

        vals.append(
            (iou, f1, bool(pred))
        )

    insertion = [
        x
        for x in xs
        if x[
            "gold_span_type"
        ]
        == "INSERTION_BOUNDARY"
    ]

    insertion_unaligned = sum(
        not pmap[
            x["benchmark_id"]
        ].get(
            "pred_spans",
            [],
        )
        for x in insertion
    )

    return {
        "items":
            len(xs),

        "span_items":
            len(span_xs),

        "projection_coverage_all":
            sum(
                bool(
                    pmap[
                        x["benchmark_id"]
                    ].get(
                        "pred_spans",
                        [],
                    )
                )
                for x in xs
            )
            / max(
                1,
                len(xs),
            ),

        "span_projection_coverage":
            sum(
                z[2]
                for z in vals
            )
            / max(
                1,
                len(vals),
            ),

        "mean_char_iou":
            mean(
                z[0]
                for z in vals
            )
            if vals
            else None,

        "mean_char_f1":
            mean(
                z[1]
                for z in vals
            )
            if vals
            else None,

        "overlap_hit_rate":
            sum(
                z[0] > 0
                for z in vals
            )
            / max(
                1,
                len(vals),
            ),

        "iou_ge_0_5_rate":
            sum(
                z[0] >= 0.5
                for z in vals
            )
            / max(
                1,
                len(vals),
            ),

        "insertion_items":
            len(insertion),

        "insertion_no_alignment":
            insertion_unaligned,
    }


def b1_pred_for_compare(
    cases,
    b1_path,
):
    raw = {
        x["benchmark_id"]: x
        for x in load_jsonl(
            b1_path
        )
    }

    out = []

    for x in cases:
        p = raw[
            x["benchmark_id"]
        ]

        out.append({
            "benchmark_id":
                x["benchmark_id"],

            "state":
                p["state"],

            "pred_spans":
                p.get(
                    "pred_spans",
                    [],
                ),
        })

    return out


def evaluate(args):
    out = Path(args.out)

    cases = load_jsonl(
        out / "cases.jsonl"
    )

    methods = {
        "B1_EDIT":
            b1_pred_for_compare(
                cases,
                args.b1,
            ),

        "C1_SIMALIGN_ARGMAX":
            load_jsonl(
                out
                / "c1_simalign_argmax.jsonl"
            ),

        "C2_SIMALIGN_ITERMAX":
            load_jsonl(
                out
                / "c2_simalign_itermax.jsonl"
            ),

        "C3_AWESOME_ALIGN":
            load_jsonl(
                out
                / "c3_awesome_align.jsonl"
            ),
    }

    b1map = {
        x["benchmark_id"]: x
        for x in methods[
            "B1_EDIT"
        ]
    }

    anchor_missing = {
        x["benchmark_id"]
        for x in cases
        if b1map[
            x["benchmark_id"]
        ]["state"]
        == "HISTORICAL_ANCHOR_MISSING"
    }

    low_b1 = set()

    for x in cases:
        if x[
            "gold_span_type"
        ] not in (
            "TOKEN_SPAN",
            "MULTI_SPAN",
        ):
            continue

        pred = b1map[
            x["benchmark_id"]
        ].get(
            "pred_spans",
            [],
        )

        iou, _ = iou_f1(
            x["gold_spans"],
            pred,
        )

        if iou < 0.5:
            low_b1.add(
                x["benchmark_id"]
            )

    slices = {
        "ALL_PRESENT":
            {
                x["benchmark_id"]
                for x in cases
            },

        "NAMED_ENTITY":
            {
                x["benchmark_id"]
                for x in cases
                if x["error_type"]
                == "Named Entity"
            },

        "TERMINOLOGY":
            {
                x["benchmark_id"]
                for x in cases
                if x["error_type"]
                == "Terminology"
            },

        "B1_ANCHOR_MISSING":
            anchor_missing,

        "B1_IOU_LT_0_5":
            low_b1,

        "MULTI_SPAN":
            {
                x["benchmark_id"]
                for x in cases
                if x[
                    "gold_span_type"
                ]
                == "MULTI_SPAN"
            },
    }

    report = {
        "split":
            "DEV_ONLY",

        "seed":
            SEED,

        "methods": {},

        "slices": {},
    }

    for name, preds in methods.items():
        report[
            "methods"
        ][name] = metric_block(
            cases,
            preds,
        )

    for sname, ids in slices.items():
        report[
            "slices"
        ][sname] = {
            name:
                metric_block(
                    cases,
                    preds,
                    ids,
                )
            for name, preds
            in methods.items()
        }

    (out / "comparison.json").write_text(
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
    print("===== MAIN DEV TABLE =====")

    print(
        "METHOD"
        "\tCOVERAGE"
        "\tSPAN_COV"
        "\tMEAN_IOU"
        "\tMEAN_F1"
        "\tOVERLAP"
        "\tIOU>=.5"
    )

    for name, m in report[
        "methods"
    ].items():
        print(
            name,
            f"{m['projection_coverage_all']:.4f}",
            f"{m['span_projection_coverage']:.4f}",
            f"{m['mean_char_iou']:.4f}",
            f"{m['mean_char_f1']:.4f}",
            f"{m['overlap_hit_rate']:.4f}",
            f"{m['iou_ge_0_5_rate']:.4f}",
            sep="\t",
        )

    for sname in [
        "NAMED_ENTITY",
        "TERMINOLOGY",
        "B1_ANCHOR_MISSING",
        "B1_IOU_LT_0_5",
        "MULTI_SPAN",
    ]:
        print()
        print(
            "===== SLICE",
            sname,
            "====="
        )

        for name, m in report[
            "slices"
        ][sname].items():
            print(
                name,
                "n=",
                m["items"],
                "cov=",
                round(
                    m[
                        "projection_coverage_all"
                    ],
                    4,
                ),
                "iou=",
                (
                    round(
                        m["mean_char_iou"],
                        4,
                    )
                    if m[
                        "mean_char_iou"
                    ]
                    is not None
                    else None
                ),
                "f1=",
                (
                    round(
                        m["mean_char_f1"],
                        4,
                    )
                    if m[
                        "mean_char_f1"
                    ]
                    is not None
                    else None
                ),
            )

    print()
    print(
        "SOURCE_PIVOT_DEV_EVAL=PASS"
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "mode",
        choices=[
            "prepare",
            "simalign",
            "parse-awesome",
            "evaluate",
        ],
    )

    ap.add_argument(
        "--gold",
    )

    ap.add_argument(
        "--b1",
    )

    ap.add_argument(
        "--mbert",
    )

    ap.add_argument(
        "--out",
        required=True,
    )

    args = ap.parse_args()

    if args.mode == "prepare":
        prepare(args)

    elif args.mode == "simalign":
        run_simalign(args)

    elif args.mode == "parse-awesome":
        parse_awesome(args)

    elif args.mode == "evaluate":
        evaluate(args)


if __name__ == "__main__":
    main()
