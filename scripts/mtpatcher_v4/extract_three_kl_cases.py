#!/usr/bin/env python3

import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from difflib import SequenceMatcher

from sacrebleu.metrics import BLEU, CHRF


EXP = "mtpatcher_v3_full6565_20260823"

ROOT = Path(os.environ["ROOT"])
RUN_ROOT = Path(os.environ["RUN_ROOT"])
DATA_ROOT = Path(os.environ["DATA_ROOT"])

RESULTS_BASE = Path(
    os.environ.get(
        "RESULTS_ROOT",
        os.environ.get(
            "RESULTS",
            str(ROOT / "results"),
        ),
    )
)

EXP_RUN = RUN_ROOT / EXP

OUT = (
    RESULTS_BASE
    / EXP
    / "three_kl_case_analysis_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


###############################################################################
# Known held-out datasets
###############################################################################

DATASETS = {
    "WMT24": (
        DATA_ROOT
        / "pilot_v2_qwen3_06b"
        / "wmt24_zh_en998.jsonl"
    ),
    "FLORES": (
        DATA_ROOT
        / "pilot_v2_qwen3_06b"
        / "flores_zh_en1012.jsonl"
    ),
    "CHALLENGE": (
        DATA_ROOT
        / "pilot_v2_qwen3_06b"
        / "challenge_zh_en197.jsonl"
    ),
}


###############################################################################
# Known corpus BLEU results.
#
# These are ONLY used to identify which existing prediction file belongs
# to which system.
###############################################################################

TARGET_BLEU = {
    "Base": {
        "WMT24": 15.536,
        "FLORES": 19.971,
        "CHALLENGE": 16.538,
    },

    "FKL": {
        "WMT24": 15.301,
        "FLORES": 19.795,
        "CHALLENGE": 16.941,
    },

    "RKL": {
        "WMT24": 15.575,
        "FLORES": 19.827,
        "CHALLENGE": 16.670,
    },

    "PG_RKL": {
        "WMT24": 15.490,
        "FLORES": 19.878,
        "CHALLENGE": 16.596,
    },

    "SEQKD_SELECTED": {
        "WMT24": 16.838,
        "FLORES": 21.114,
        "CHALLENGE": 17.638,
    },
}


SYSTEM_HINTS = {
    "Base": (
        "base",
    ),

    "FKL": (
        "fkl",
        "forward",
    ),

    "RKL": (
        "rkl",
        "reversekl",
    ),

    "PG_RKL": (
        "cleanroom",
        "pgrkl",
    ),

    "SEQKD_SELECTED": (
        "seqkd",
        "selected",
    ),
}


PRED_KEYS = (
    "student_translation",
    "prediction",
    "pred",
    "hypothesis",
    "generated_text",
    "generation",
    "output",
    "response",
    "translation",
    "text",
)


SRC_KEYS = (
    "source",
    "src",
    "input",
    "source_text",
    "zh",
    "chinese",
)


REF_KEYS = (
    "reference",
    "ref",
    "target",
    "tgt",
    "target_text",
    "en",
    "english",
)


bleu_metric = BLEU(
    effective_order=True,
)

chrf_metric = CHRF(
    word_order=2,
)


###############################################################################
# Utility
###############################################################################

def norm_text(x):

    if x is None:
        return ""

    x = str(x)

    return " ".join(
        x.replace(
            "\r",
            " ",
        ).replace(
            "\n",
            " ",
        ).split()
    )


def load_jsonl(path):

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:

        for lineno, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(
                    line
                )

            except Exception as e:
                raise RuntimeError(
                    f"JSON parse failed: {path}:{lineno}: {e}"
                )

            if isinstance(
                obj,
                dict,
            ):
                rows.append(
                    obj
                )

    return rows


def choose_text_key(
    rows,
    preferred,
):

    if not rows:
        return None

    counts = defaultdict(
        int
    )

    for row in rows:

        for key, value in row.items():

            if (
                isinstance(
                    value,
                    str,
                )
                and value.strip()
            ):
                counts[
                    key
                ] += 1

    n = len(
        rows
    )

    for key in preferred:

        if counts.get(
            key,
            0,
        ) >= int(
            n * 0.95
        ):
            return key

    return None


def all_string_fields(
    rows,
):

    if not rows:
        return []

    counts = defaultdict(
        int
    )

    for row in rows:

        for key, value in row.items():

            if (
                isinstance(
                    value,
                    str,
                )
                and value.strip()
            ):
                counts[
                    key
                ] += 1

    n = len(
        rows
    )

    keys = [
        key
        for key, count in counts.items()
        if count >= int(
            n * 0.95
        )
    ]

    # preferred prediction-like keys first
    keys.sort(
        key=lambda k: (
            PRED_KEYS.index(k)
            if k in PRED_KEYS
            else 999,
            k,
        )
    )

    return keys


###############################################################################
# Load references.
###############################################################################

dataset_rows = {}

references = {}

sources = {}


for split, path in DATASETS.items():

    if not path.exists():

        raise RuntimeError(
            f"Missing held-out dataset: {path}"
        )

    rows = load_jsonl(
        path
    )

    src_key = choose_text_key(
        rows,
        SRC_KEYS,
    )

    ref_key = choose_text_key(
        rows,
        REF_KEYS,
    )

    if src_key is None:

        raise RuntimeError(
            f"Cannot identify source field in {path}; "
            f"keys={sorted(rows[0].keys())}"
        )

    if ref_key is None:

        raise RuntimeError(
            f"Cannot identify reference field in {path}; "
            f"keys={sorted(rows[0].keys())}"
        )

    dataset_rows[
        split
    ] = rows

    sources[
        split
    ] = [
        norm_text(
            row[src_key]
        )
        for row in rows
    ]

    references[
        split
    ] = [
        norm_text(
            row[ref_key]
        )
        for row in rows
    ]

    print(
        f"DATASET {split:10s} "
        f"rows={len(rows):4d} "
        f"source_key={src_key} "
        f"reference_key={ref_key}"
    )


###############################################################################
# Bounded discovery.
#
# Search ONLY this experiment's run directory.
# No scanning across /workspace or model/data roots.
###############################################################################

candidate_files = []


if not EXP_RUN.exists():

    raise RuntimeError(
        f"Experiment run directory missing: {EXP_RUN}"
    )


base_depth = len(
    EXP_RUN.parts
)


for path in EXP_RUN.rglob(
    "*.jsonl"
):

    depth = (
        len(path.parts)
        - base_depth
    )

    # Keep discovery bounded.
    if depth > 6:
        continue

    try:
        size = path.stat().st_size

    except OSError:
        continue

    if size <= 0:
        continue

    # Skip enormous construction files if any were copied under runs.
    if size > 100 * 1024 * 1024:
        continue

    candidate_files.append(
        path
    )


print()
print(
    "CANDIDATE_JSONL_FILES =",
    len(candidate_files),
)


###############################################################################
# Score every viable prediction field.
###############################################################################

candidate_outputs = defaultdict(
    list
)


for path in candidate_files:

    try:
        rows = load_jsonl(
            path
        )

    except Exception:
        continue

    n = len(
        rows
    )

    possible_splits = [
        split
        for split in DATASETS
        if len(
            dataset_rows[split]
        ) == n
    ]

    if not possible_splits:
        continue

    fields = all_string_fields(
        rows
    )

    if not fields:
        continue

    for split in possible_splits:

        refs = references[
            split
        ]

        for field in fields:

            hyps = [
                norm_text(
                    row.get(
                        field,
                        "",
                    )
                )
                for row in rows
            ]

            if any(
                not x
                for x in hyps
            ):
                continue

            try:
                score = bleu_metric.corpus_score(
                    hyps,
                    [
                        refs
                    ],
                ).score

            except Exception:
                continue

            candidate_outputs[
                split
            ].append(
                {
                    "path": str(
                        path
                    ),
                    "field": field,
                    "bleu": float(
                        score
                    ),
                    "rows": rows,
                    "hyps": hyps,
                }
            )


###############################################################################
# Identify system files by target corpus BLEU + pathname hints.
###############################################################################

selected = defaultdict(
    dict
)


mapping_lines = []


def hint_bonus(
    system,
    path,
):

    low = path.lower()

    hints = SYSTEM_HINTS[
        system
    ]

    matched = sum(
        1
        for h in hints
        if h in low
    )

    bonus = (
        0.025
        * matched
    )

    # Exact RKL must not accidentally select PG-RKL.
    if (
        system == "RKL"
        and (
            "pgrkl" in low
            or "cleanroom" in low
        )
    ):
        bonus -= 0.20

    if (
        system == "FKL"
        and "pgrkl" in low
    ):
        bonus -= 0.20

    return bonus


print()
print(
    "=" * 110
)

print(
    "AUTO-MAPPING EXISTING PREDICTIONS"
)

print(
    "=" * 110
)


for split in DATASETS:

    candidates = candidate_outputs[
        split
    ]

    if not candidates:

        raise RuntimeError(
            f"No viable prediction JSONL found for {split}"
        )

    for system in TARGET_BLEU:

        target = TARGET_BLEU[
            system
        ][
            split
        ]

        ranked = []

        for cand in candidates:

            delta = abs(
                cand[
                    "bleu"
                ]
                - target
            )

            adjusted = (
                delta
                - hint_bonus(
                    system,
                    cand[
                        "path"
                    ],
                )
            )

            ranked.append(
                (
                    adjusted,
                    delta,
                    cand,
                )
            )

        ranked.sort(
            key=lambda x: (
                x[0],
                x[1],
            )
        )

        best = ranked[
            0
        ][
            2
        ]

        raw_delta = ranked[
            0
        ][
            1
        ]

        # Known summary numbers are rounded to 3 decimals.
        # Anything wildly farther than this is suspicious.
        if raw_delta > 0.30:

            print()
            print(
                f"FAILED TO IDENTIFY {system} / {split}"
            )

            print(
                f"target BLEU = {target}"
            )

            print(
                "closest candidates:"
            )

            for _, d, cand in ranked[
                :12
            ]:

                print(
                    f"  BLEU={cand['bleu']:.6f} "
                    f"diff={d:.6f} "
                    f"field={cand['field']} "
                    f"path={cand['path']}"
                )

            raise RuntimeError(
                f"Prediction mapping ambiguous for {system}/{split}"
            )

        selected[
            system
        ][
            split
        ] = best

        line = (
            f"{system:16s} {split:10s} "
            f"target={target:7.3f} "
            f"found={best['bleu']:9.6f} "
            f"diff={raw_delta:8.6f} "
            f"field={best['field']} "
            f"path={best['path']}"
        )

        mapping_lines.append(
            line
        )

        print(
            line
        )


###############################################################################
# Basic safety: same split/system predictions should have correct row count.
###############################################################################

for system in TARGET_BLEU:

    for split in DATASETS:

        hyps = selected[
            system
        ][
            split
        ][
            "hyps"
        ]

        if len(
            hyps
        ) != len(
            references[split]
        ):

            raise RuntimeError(
                f"Row mismatch {system}/{split}"
            )


###############################################################################
# Sentence-level analysis.
###############################################################################

all_rows = []


for split in DATASETS:

    refs = references[
        split
    ]

    srcs = sources[
        split
    ]

    systems = {
        system: selected[
            system
        ][
            split
        ][
            "hyps"
        ]
        for system in TARGET_BLEU
    }

    for i, (
        src,
        ref,
    ) in enumerate(
        zip(
            srcs,
            refs,
        )
    ):

        item = {
            "split": split,
            "index": i,
            "source": src,
            "reference": ref,
        }

        for system, hyps in systems.items():

            hyp = hyps[
                i
            ]

            sb = bleu_metric.sentence_score(
                hyp,
                [
                    ref
                ],
            ).score

            sc = chrf_metric.sentence_score(
                hyp,
                [
                    ref
                ],
            ).score

            item[
                system
            ] = hyp

            item[
                f"{system}_BLEU"
            ] = float(
                sb
            )

            item[
                f"{system}_chrF"
            ] = float(
                sc
            )

        base_chrf = item[
            "Base_chrF"
        ]

        base_bleu = item[
            "Base_BLEU"
        ]

        for system in (
            "FKL",
            "RKL",
            "PG_RKL",
            "SEQKD_SELECTED",
        ):

            item[
                f"{system}_dchrF"
            ] = (
                item[
                    f"{system}_chrF"
                ]
                - base_chrf
            )

            item[
                f"{system}_dBLEU"
            ] = (
                item[
                    f"{system}_BLEU"
                ]
                - base_bleu
            )

            item[
                f"{system}_changed"
            ] = (
                item[
                    system
                ]
                != item[
                    "Base"
                ]
            )

        kl_d = [
            item[
                "FKL_dchrF"
            ],
            item[
                "RKL_dchrF"
            ],
            item[
                "PG_RKL_dchrF"
            ],
        ]

        item[
            "KL_mean_dchrF"
        ] = sum(
            kl_d
        ) / 3.0

        item[
            "KL_min_dchrF"
        ] = min(
            kl_d
        )

        item[
            "KL_max_dchrF"
        ] = max(
            kl_d
        )

        item[
            "KL_spread_dchrF"
        ] = (
            max(
                kl_d
            )
            - min(
                kl_d
            )
        )

        item[
            "RKL_PG_output_similarity"
        ] = SequenceMatcher(
            None,
            item[
                "RKL"
            ],
            item[
                "PG_RKL"
            ],
        ).ratio()

        all_rows.append(
            item
        )


###############################################################################
# Corpus-level behavioral summary.
###############################################################################

summary_lines = []

summary_lines.append(
    "=" * 100
)

summary_lines.append(
    "THREE-KL CASE ANALYSIS SUMMARY"
)

summary_lines.append(
    "=" * 100
)

summary_lines.append(
    ""
)

summary_lines.append(
    "SYSTEM BEHAVIOR VS BASE"
)

summary_lines.append(
    "-" * 100
)


for system in (
    "FKL",
    "RKL",
    "PG_RKL",
    "SEQKD_SELECTED",
):

    changed = sum(
        1
        for x in all_rows
        if x[
            f"{system}_changed"
        ]
    )

    positive = sum(
        1
        for x in all_rows
        if x[
            f"{system}_dchrF"
        ] > 0.5
    )

    negative = sum(
        1
        for x in all_rows
        if x[
            f"{system}_dchrF"
        ] < -0.5
    )

    neutral = (
        len(
            all_rows
        )
        - positive
        - negative
    )

    avg_dchrf = sum(
        x[
            f"{system}_dchrF"
        ]
        for x in all_rows
    ) / len(
        all_rows
    )

    summary_lines.append(
        f"{system:16s} "
        f"changed={changed:4d}/{len(all_rows)} "
        f"({100.0 * changed / len(all_rows):6.2f}%) "
        f"chrF_up={positive:4d} "
        f"chrF_down={negative:4d} "
        f"neutral={neutral:4d} "
        f"mean_sentence_dchrF={avg_dchrf:+.4f}"
    )


###############################################################################
# Categories.
###############################################################################

def choose_unique(
    rows,
    key,
    reverse,
    n,
    used,
):

    rows = sorted(
        rows,
        key=key,
        reverse=reverse,
    )

    out = []

    for row in rows:

        uid = (
            row[
                "split"
            ],
            row[
                "index"
            ],
        )

        if uid in used:
            continue

        used.add(
            uid
        )

        out.append(
            row
        )

        if len(
            out
        ) >= n:
            break

    return out


categories = []


# 1. All three KL hurt the same example.
categories.append(
    (
        "ALL_3_KL_WORSE",
        [
            x
            for x in all_rows
            if (
                x[
                    "FKL_dchrF"
                ] < -0.5
                and x[
                    "RKL_dchrF"
                ] < -0.5
                and x[
                    "PG_RKL_dchrF"
                ] < -0.5
            )
        ],
        lambda x: x[
            "KL_mean_dchrF"
        ],
        False,
    )
)


# 2. All three KL improve.
categories.append(
    (
        "ALL_3_KL_BETTER",
        [
            x
            for x in all_rows
            if (
                x[
                    "FKL_dchrF"
                ] > 0.5
                and x[
                    "RKL_dchrF"
                ] > 0.5
                and x[
                    "PG_RKL_dchrF"
                ] > 0.5
            )
        ],
        lambda x: x[
            "KL_mean_dchrF"
        ],
        True,
    )
)


# 3. Forward improves while exact reverse hurts.
categories.append(
    (
        "FKL_UP_RKL_DOWN",
        [
            x
            for x in all_rows
            if (
                x[
                    "FKL_dchrF"
                ] > 1.0
                and x[
                    "RKL_dchrF"
                ] < -1.0
            )
        ],
        lambda x: (
            x[
                "FKL_dchrF"
            ]
            - x[
                "RKL_dchrF"
            ]
        ),
        True,
    )
)


# 4. Reverse improves while forward hurts.
categories.append(
    (
        "RKL_UP_FKL_DOWN",
        [
            x
            for x in all_rows
            if (
                x[
                    "RKL_dchrF"
                ] > 1.0
                and x[
                    "FKL_dchrF"
                ] < -1.0
            )
        ],
        lambda x: (
            x[
                "RKL_dchrF"
            ]
            - x[
                "FKL_dchrF"
            ]
        ),
        True,
    )
)


# 5. Sampled PG behaves differently from exact reverse KL.
categories.append(
    (
        "PG_VS_EXACT_RKL_DIVERGE",
        [
            x
            for x in all_rows
            if (
                x[
                    "PG_RKL"
                ]
                != x[
                    "RKL"
                ]
            )
        ],
        lambda x: abs(
            x[
                "PG_RKL_dchrF"
            ]
            - x[
                "RKL_dchrF"
            ]
        ),
        True,
    )
)


# 6. SeqKD wins where all KLs fail to produce useful gain.
categories.append(
    (
        "SEQKD_WINS_KL_FAILS",
        [
            x
            for x in all_rows
            if (
                x[
                    "SEQKD_SELECTED_dchrF"
                ] > 2.0
                and x[
                    "KL_max_dchrF"
                ] < 0.5
            )
        ],
        lambda x: (
            x[
                "SEQKD_SELECTED_dchrF"
            ]
            - x[
                "KL_mean_dchrF"
            ]
        ),
        True,
    )
)


# 7. KL output changed but metric barely moved:
# candidate lexical/style/paraphrase drift.
categories.append(
    (
        "KL_CHANGED_BUT_CHRF_NEUTRAL",
        [
            x
            for x in all_rows
            if (
                (
                    x[
                        "FKL_changed"
                    ]
                    or x[
                        "RKL_changed"
                    ]
                    or x[
                        "PG_RKL_changed"
                    ]
                )
                and abs(
                    x[
                        "KL_mean_dchrF"
                    ]
                ) < 0.30
            )
        ],
        lambda x: (
            int(
                x[
                    "FKL_changed"
                ]
            )
            + int(
                x[
                    "RKL_changed"
                ]
            )
            + int(
                x[
                    "PG_RKL_changed"
                ]
            )
        ),
        True,
    )
)


# 8. All 3 KL decode exactly like Base, while SeqKD changes and improves.
categories.append(
    (
        "ALL_KL_SAME_AS_BASE_SEQKD_IMPROVES",
        [
            x
            for x in all_rows
            if (
                not x[
                    "FKL_changed"
                ]
                and not x[
                    "RKL_changed"
                ]
                and not x[
                    "PG_RKL_changed"
                ]
                and x[
                    "SEQKD_SELECTED_changed"
                ]
                and x[
                    "SEQKD_SELECTED_dchrF"
                ] > 1.0
            )
        ],
        lambda x: x[
            "SEQKD_SELECTED_dchrF"
        ],
        True,
    )
)


###############################################################################
# Select at most 6 unique examples from each category.
###############################################################################

used = set()

selected_cases = []


for (
    category,
    rows,
    key,
    reverse,
) in categories:

    chosen = choose_unique(
        rows,
        key,
        reverse,
        6,
        used,
    )

    summary_lines.append(
        ""
    )

    summary_lines.append(
        f"{category}: candidates={len(rows)}, selected={len(chosen)}"
    )

    for rank, row in enumerate(
        chosen,
        start=1,
    ):

        z = dict(
            row
        )

        z[
            "category"
        ] = category

        z[
            "category_rank"
        ] = rank

        selected_cases.append(
            z
        )


###############################################################################
# Write machine-readable full table.
###############################################################################

all_jsonl = OUT / "all_sentence_comparisons.jsonl"


with all_jsonl.open(
    "w",
    encoding="utf-8",
) as f:

    for row in all_rows:

        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


selected_jsonl = OUT / "selected_cases.jsonl"


with selected_jsonl.open(
    "w",
    encoding="utf-8",
) as f:

    for row in selected_cases:

        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


###############################################################################
# CSV for easy filtering.
###############################################################################

csv_path = OUT / "all_sentence_comparisons.csv"


fieldnames = [
    "split",
    "index",
    "source",
    "reference",

    "Base",
    "Base_BLEU",
    "Base_chrF",

    "FKL",
    "FKL_BLEU",
    "FKL_chrF",
    "FKL_dBLEU",
    "FKL_dchrF",
    "FKL_changed",

    "RKL",
    "RKL_BLEU",
    "RKL_chrF",
    "RKL_dBLEU",
    "RKL_dchrF",
    "RKL_changed",

    "PG_RKL",
    "PG_RKL_BLEU",
    "PG_RKL_chrF",
    "PG_RKL_dBLEU",
    "PG_RKL_dchrF",
    "PG_RKL_changed",

    "SEQKD_SELECTED",
    "SEQKD_SELECTED_BLEU",
    "SEQKD_SELECTED_chrF",
    "SEQKD_SELECTED_dBLEU",
    "SEQKD_SELECTED_dchrF",
    "SEQKD_SELECTED_changed",

    "KL_mean_dchrF",
    "KL_min_dchrF",
    "KL_max_dchrF",
    "KL_spread_dchrF",
    "RKL_PG_output_similarity",
]


with csv_path.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
        extrasaction="ignore",
    )

    writer.writeheader()

    writer.writerows(
        all_rows
    )


###############################################################################
# Pretty human-readable selected cases.
###############################################################################

txt_path = OUT / "selected_cases.txt"


with txt_path.open(
    "w",
    encoding="utf-8",
) as f:

    current = None

    for row in selected_cases:

        category = row[
            "category"
        ]

        if category != current:

            current = category

            f.write(
                "\n"
                + "=" * 120
                + "\n"
            )

            f.write(
                category
                + "\n"
            )

            f.write(
                "=" * 120
                + "\n"
            )

        f.write(
            "\n"
            f"[{row['split']} #{row['index']}] "
            f"rank={row['category_rank']}\n"
        )

        f.write(
            f"SOURCE:\n{row['source']}\n\n"
        )

        f.write(
            f"REFERENCE:\n{row['reference']}\n\n"
        )

        f.write(
            "BASE "
            f"(BLEU={row['Base_BLEU']:.2f}, "
            f"chrF={row['Base_chrF']:.2f}):\n"
            f"{row['Base']}\n\n"
        )

        for system in (
            "FKL",
            "RKL",
            "PG_RKL",
            "SEQKD_SELECTED",
        ):

            f.write(
                f"{system} "
                f"(BLEU={row[f'{system}_BLEU']:.2f}, "
                f"ΔBLEU={row[f'{system}_dBLEU']:+.2f}, "
                f"chrF={row[f'{system}_chrF']:.2f}, "
                f"ΔchrF={row[f'{system}_dchrF']:+.2f}):\n"
                f"{row[system]}\n\n"
            )

        f.write(
            "-" * 120
            + "\n"
        )


###############################################################################
# Mapping + summary.
###############################################################################

mapping_path = OUT / "prediction_mapping.txt"


mapping_path.write_text(
    "\n".join(
        mapping_lines
    )
    + "\n",
    encoding="utf-8",
)


summary_path = OUT / "summary.txt"


summary_path.write_text(
    "\n".join(
        summary_lines
    )
    + "\n",
    encoding="utf-8",
)


###############################################################################
# Terminal preview:
# first 18 selected examples.
###############################################################################

print()
print(
    "=" * 110
)

print(
    "BEHAVIOR SUMMARY"
)

print(
    "=" * 110
)


for line in summary_lines:

    print(
        line
    )


print()
print(
    "=" * 110
)

print(
    "SELECTED CASE PREVIEW"
)

print(
    "=" * 110
)


for row in selected_cases[
    :18
]:

    print()
    print(
        f"[{row['category']}] "
        f"{row['split']} #{row['index']}"
    )

    print(
        "SRC :",
        row[
            "source"
        ],
    )

    print(
        "REF :",
        row[
            "reference"
        ],
    )

    print(
        "BASE:",
        row[
            "Base"
        ],
    )

    print(
        f"FKL : {row['FKL']} "
        f"[ΔchrF {row['FKL_dchrF']:+.2f}]"
    )

    print(
        f"RKL : {row['RKL']} "
        f"[ΔchrF {row['RKL_dchrF']:+.2f}]"
    )

    print(
        f"PG  : {row['PG_RKL']} "
        f"[ΔchrF {row['PG_RKL_dchrF']:+.2f}]"
    )

    print(
        f"SEQ : {row['SEQKD_SELECTED']} "
        f"[ΔchrF {row['SEQKD_SELECTED_dchrF']:+.2f}]"
    )


print()
print(
    "=" * 110
)

print(
    "KL_CASE_EXTRACTION_PASS"
)

print(
    "OUT =",
    OUT,
)

print(
    "MAPPING =",
    mapping_path,
)

print(
    "SUMMARY =",
    summary_path,
)

print(
    "SELECTED_TXT =",
    txt_path,
)

print(
    "SELECTED_JSONL =",
    selected_jsonl,
)

print(
    "ALL_JSONL =",
    all_jsonl,
)

print(
    "ALL_CSV =",
    csv_path,
)

print(
    "=" * 110
)
