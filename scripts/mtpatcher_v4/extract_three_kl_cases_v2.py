#!/usr/bin/env python3

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from difflib import SequenceMatcher

from sacrebleu.metrics import BLEU, CHRF


###############################################################################
# CONFIG
###############################################################################

EXP = "mtpatcher_v3_full6565_20260823"

ROOT = Path(os.environ["ROOT"])
RUN_ROOT = Path(os.environ["RUN_ROOT"])
DATA_ROOT = Path(os.environ["DATA_ROOT"])

EXP_RUN = RUN_ROOT / EXP

OUT = (
    ROOT
    / "results"
    / EXP
    / "three_kl_case_analysis_v2"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


###############################################################################
# Canonical held-out data
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
# Exact known corpus BLEU vectors
#
# These identify systems as a 3-D vector:
#
#   (WMT24, FLORES, CHALLENGE)
#
# Base values come from the frozen base evaluation.
# Other values come from their completed evaluation runs.
###############################################################################

TARGET = {
    "Base": {
        "WMT24": 15.536214,
        "FLORES": 19.971480,
        "CHALLENGE": 16.537871,
    },

    "FKL": {
        "WMT24": 15.300985,
        "FLORES": 19.795267,
        "CHALLENGE": 16.940660,
    },

    "RKL": {
        "WMT24": 15.575108,
        "FLORES": 19.827007,
        "CHALLENGE": 16.670469,
    },

    "PG_RKL": {
        "WMT24": 15.489656,
        "FLORES": 19.877694,
        "CHALLENGE": 16.595995,
    },

    "SEQKD_SELECTED": {
        "WMT24": 16.838339,
        "FLORES": 21.113730,
        "CHALLENGE": 17.638367,
    },
}


###############################################################################
# Known system-family directories.
#
# DO NOT auto-map these again.
###############################################################################

KNOWN_FAMILIES = {
    "FKL": (
        EXP_RUN
        / "opd_torchnpu_fkl_pe3732_v1"
        / "eval_epoch3"
    ),

    "RKL": (
        EXP_RUN
        / "opd_torchnpu_rkl_pe3732_v2"
        / "eval_epoch3"
    ),

    "PG_RKL": (
        EXP_RUN
        / "opd_torchnpu_pgrkl_cleanroom_pe3732_v2"
        / "eval_epoch3"
    ),

    "SEQKD_SELECTED": (
        EXP_RUN
        / "seqkd_control_eval_epoch3"
        / "seqkd_selected3732_b4ga4"
    ),
}


SPLIT_DIRS = {
    "WMT24": "wmt24",
    "FLORES": "flores",
    "CHALLENGE": "challenge",
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
)


SRC_KEYS = (
    "source",
    "src",
    "input",
    "source_text",
)


REF_KEYS = (
    "reference",
    "ref",
    "target",
    "tgt",
    "target_text",
)


bleu = BLEU(
    effective_order=True,
)

chrf = CHRF(
    word_order=2,
)


###############################################################################
# Helpers
###############################################################################

def norm(x):

    if x is None:
        return ""

    return " ".join(
        str(x)
        .replace("\r", " ")
        .replace("\n", " ")
        .split()
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
                    f"JSON parse error "
                    f"{path}:{lineno}: {e}"
                )

            if not isinstance(
                obj,
                dict,
            ):

                raise RuntimeError(
                    f"Non-dict JSON row "
                    f"{path}:{lineno}"
                )

            rows.append(
                obj
            )

    return rows


def choose_key(
    rows,
    candidates,
):

    for key in candidates:

        valid = sum(
            1
            for row in rows
            if (
                key in row
                and isinstance(
                    row[key],
                    str,
                )
                and row[key].strip()
            )
        )

        if valid == len(
            rows
        ):
            return key

    return None


def fingerprint(
    split_hyps,
):

    h = hashlib.sha256()

    for split in (
        "WMT24",
        "FLORES",
        "CHALLENGE",
    ):

        h.update(
            split.encode()
        )

        for hyp in split_hyps[
            split
        ]:

            h.update(
                hyp.encode(
                    "utf-8",
                    errors="replace",
                )
            )

            h.update(
                b"\n"
            )

    return h.hexdigest()


###############################################################################
# Load canonical held-out datasets
###############################################################################

canonical = {}


print(
    "=" * 110
)

print(
    "LOAD CANONICAL HELD-OUT DATA"
)

print(
    "=" * 110
)


for split, path in DATASETS.items():

    if not path.exists():

        raise RuntimeError(
            f"Missing dataset: {path}"
        )

    rows = load_jsonl(
        path
    )

    src_key = choose_key(
        rows,
        SRC_KEYS,
    )

    ref_key = choose_key(
        rows,
        REF_KEYS,
    )

    if src_key is None:

        raise RuntimeError(
            f"No source key in {path}; "
            f"keys={sorted(rows[0])}"
        )

    if ref_key is None:

        raise RuntimeError(
            f"No reference key in {path}; "
            f"keys={sorted(rows[0])}"
        )

    canonical[
        split
    ] = {
        "rows": rows,

        "source": [
            norm(
                row[src_key]
            )
            for row in rows
        ],

        "reference": [
            norm(
                row[ref_key]
            )
            for row in rows
        ],
    }

    print(
        f"{split:10s} "
        f"rows={len(rows):4d} "
        f"src={src_key} "
        f"ref={ref_key} "
        f"path={path}"
    )


###############################################################################
# Read one complete system family
###############################################################################

def read_family(
    family,
    label,
):

    result = {
        "family": family,
        "hyps": {},
        "bleu": {},
        "paths": {},
    }

    for split in (
        "WMT24",
        "FLORES",
        "CHALLENGE",
    ):

        pred = (
            family
            / SPLIT_DIRS[split]
            / "predictions.jsonl"
        )

        if not pred.exists():

            raise RuntimeError(
                f"{label}: missing prediction "
                f"for {split}: {pred}"
            )

        rows = load_jsonl(
            pred
        )

        expected_n = len(
            canonical[
                split
            ][
                "rows"
            ]
        )

        if len(rows) != expected_n:

            raise RuntimeError(
                f"{label}/{split}: "
                f"row mismatch "
                f"{len(rows)} != {expected_n}"
            )

        pred_key = choose_key(
            rows,
            PRED_KEYS,
        )

        if pred_key is None:

            raise RuntimeError(
                f"{label}/{split}: "
                f"no prediction field; "
                f"keys={sorted(rows[0])}"
            )

        src_key = choose_key(
            rows,
            SRC_KEYS,
        )

        ref_key = choose_key(
            rows,
            REF_KEYS,
        )


        # ------------------------------------------------------------
        # Alignment verification.
        # ------------------------------------------------------------

        if src_key is not None:

            got_src = [
                norm(
                    row[src_key]
                )
                for row in rows
            ]

            expected_src = canonical[
                split
            ][
                "source"
            ]

            src_match = sum(
                a == b
                for a, b in zip(
                    got_src,
                    expected_src,
                )
            )

            src_ratio = (
                src_match
                / expected_n
            )

            if src_ratio < 0.999:

                raise RuntimeError(
                    f"{label}/{split}: "
                    f"source alignment failed "
                    f"{src_match}/{expected_n}"
                )


        if ref_key is not None:

            got_ref = [
                norm(
                    row[ref_key]
                )
                for row in rows
            ]

            expected_ref = canonical[
                split
            ][
                "reference"
            ]

            ref_match = sum(
                a == b
                for a, b in zip(
                    got_ref,
                    expected_ref,
                )
            )

            ref_ratio = (
                ref_match
                / expected_n
            )

            if ref_ratio < 0.999:

                raise RuntimeError(
                    f"{label}/{split}: "
                    f"reference alignment failed "
                    f"{ref_match}/{expected_n}"
                )


        hyps = [
            norm(
                row[pred_key]
            )
            for row in rows
        ]


        score = bleu.corpus_score(
            hyps,
            [
                canonical[
                    split
                ][
                    "reference"
                ]
            ],
        ).score


        result[
            "hyps"
        ][
            split
        ] = hyps

        result[
            "bleu"
        ][
            split
        ] = float(
            score
        )

        result[
            "paths"
        ][
            split
        ] = pred


    result[
        "fingerprint"
    ] = fingerprint(
        result[
            "hyps"
        ]
    )

    return result


###############################################################################
# Read known systems first.
###############################################################################

systems = {}


print()
print(
    "=" * 110
)

print(
    "READ FIXED SYSTEM FAMILIES"
)

print(
    "=" * 110
)


for label, family in KNOWN_FAMILIES.items():

    if not family.exists():

        raise RuntimeError(
            f"Known family missing: "
            f"{label}: {family}"
        )

    system = read_family(
        family,
        label,
    )

    systems[
        label
    ] = system


    print(
        f"{label:16s} "
        f"WMT={system['bleu']['WMT24']:.6f} "
        f"FLORES={system['bleu']['FLORES']:.6f} "
        f"CHALL={system['bleu']['CHALLENGE']:.6f}"
    )

    print(
        f"  FAMILY={family}"
    )


###############################################################################
# Verify known systems against their expected 3-D BLEU vectors.
###############################################################################

print()
print(
    "=" * 110
)

print(
    "VERIFY FIXED SYSTEM BLEU VECTORS"
)

print(
    "=" * 110
)


for label in KNOWN_FAMILIES:

    diffs = {
        split: abs(
            systems[
                label
            ][
                "bleu"
            ][
                split
            ]
            - TARGET[
                label
            ][
                split
            ]
        )
        for split in (
            "WMT24",
            "FLORES",
            "CHALLENGE",
        )
    }

    max_diff = max(
        diffs.values()
    )


    print(
        f"{label:16s} "
        f"max_abs_diff={max_diff:.8f} "
        f"diffs={diffs}"
    )


    if max_diff > 0.01:

        raise RuntimeError(
            f"{label}: known family BLEU "
            f"does not match expected system"
        )


###############################################################################
# Discover Base as ONE family with all 3 splits.
#
# Critical change from V1:
#
# We never choose a Base prediction separately for each split.
# A Base candidate must contain ALL THREE:
#
#   family/wmt24/predictions.jsonl
#   family/flores/predictions.jsonl
#   family/challenge/predictions.jsonl
###############################################################################

print()
print(
    "=" * 110
)

print(
    "DISCOVER TRUE BASE AS A 3-SPLIT FAMILY"
)

print(
    "=" * 110
)


family_dirs = set()


base_depth = len(
    EXP_RUN.parts
)


for pred in EXP_RUN.rglob(
    "predictions.jsonl"
):

    depth = (
        len(
            pred.parts
        )
        - base_depth
    )

    # bounded search under this experiment only
    if depth > 8:
        continue

    split_dir = pred.parent.name.lower()

    if split_dir not in {
        "wmt24",
        "flores",
        "challenge",
    }:
        continue

    family_dirs.add(
        pred.parent.parent
    )


complete_candidates = []


for family in sorted(
    family_dirs,
    key=str,
):

    required = [
        family
        / "wmt24"
        / "predictions.jsonl",

        family
        / "flores"
        / "predictions.jsonl",

        family
        / "challenge"
        / "predictions.jsonl",
    ]

    if not all(
        p.exists()
        for p in required
    ):
        continue

    try:

        system = read_family(
            family,
            "BASE_CANDIDATE",
        )

    except Exception:

        continue


    diffs = {
        split: abs(
            system[
                "bleu"
            ][
                split
            ]
            - TARGET[
                "Base"
            ][
                split
            ]
        )

        for split in (
            "WMT24",
            "FLORES",
            "CHALLENGE",
        )
    }


    max_diff = max(
        diffs.values()
    )

    l1 = sum(
        diffs.values()
    )


    complete_candidates.append(
        {
            "system": system,
            "max_diff": max_diff,
            "l1": l1,
            "diffs": diffs,
        }
    )


complete_candidates.sort(
    key=lambda x: (
        x[
            "max_diff"
        ],
        x[
            "l1"
        ],
        str(
            x[
                "system"
            ][
                "family"
            ]
        ),
    )
)


print(
    f"COMPLETE_3_SPLIT_FAMILIES = "
    f"{len(complete_candidates)}"
)


print()
print(
    "TOP BASE CANDIDATES"
)


for cand in complete_candidates[
    :20
]:

    s = cand[
        "system"
    ]

    print(
        f"maxdiff={cand['max_diff']:.8f} "
        f"L1={cand['l1']:.8f} "
        f"WMT={s['bleu']['WMT24']:.6f} "
        f"FLORES={s['bleu']['FLORES']:.6f} "
        f"CHALL={s['bleu']['CHALLENGE']:.6f}"
    )

    print(
        f"  {s['family']}"
    )


strict = [
    cand
    for cand in complete_candidates
    if cand[
        "max_diff"
    ] <= 0.01
]


if not strict:

    raise RuntimeError(
        "TRUE BASE NOT FOUND: "
        "no single 3-split family matches "
        "the Base BLEU vector within 0.01"
    )


###############################################################################
# If multiple Base copies match, allow only if their outputs are identical.
###############################################################################

best_diff = strict[
    0
][
    "max_diff"
]


near_best = [
    cand
    for cand in strict
    if (
        cand[
            "max_diff"
        ]
        <= best_diff + 0.001
    )
]


fingerprints = {
    cand[
        "system"
    ][
        "fingerprint"
    ]

    for cand in near_best
}


if len(
    fingerprints
) > 1:

    print()
    print(
        "AMBIGUOUS BASE CANDIDATES:"
    )

    for cand in near_best:

        print(
            cand[
                "system"
            ][
                "family"
            ],
            cand[
                "system"
            ][
                "fingerprint"
            ],
        )

    raise RuntimeError(
        "Multiple non-identical Base families "
        "match the 3-D BLEU vector"
    )


# Prefer a path containing "base" if several identical copies exist.
near_best.sort(
    key=lambda cand: (
        0
        if "base" in str(
            cand[
                "system"
            ][
                "family"
            ]
        ).lower()
        else 1,

        cand[
            "l1"
        ],
    )
)


base_system = near_best[
    0
][
    "system"
]


systems[
    "Base"
] = base_system


print()
print(
    "TRUE_BASE_FOUND"
)

print(
    "FAMILY =",
    base_system[
        "family"
    ],
)

print(
    "BLEU VECTOR =",
    {
        split: base_system[
            "bleu"
        ][
            split
        ]

        for split in (
            "WMT24",
            "FLORES",
            "CHALLENGE",
        )
    },
)

print(
    "FINGERPRINT =",
    base_system[
        "fingerprint"
    ],
)


###############################################################################
# Final mapping hard audit
###############################################################################

print()
print(
    "=" * 110
)

print(
    "FINAL MAPPING AUDIT"
)

print(
    "=" * 110
)


ORDER = (
    "Base",
    "FKL",
    "RKL",
    "PG_RKL",
    "SEQKD_SELECTED",
)


mapping_lines = []


families = []


for label in ORDER:

    system = systems[
        label
    ]

    family = system[
        "family"
    ]

    families.append(
        str(
            family
        )
    )


    max_diff = max(
        abs(
            system[
                "bleu"
            ][
                split
            ]
            - TARGET[
                label
            ][
                split
            ]
        )

        for split in (
            "WMT24",
            "FLORES",
            "CHALLENGE",
        )
    )


    line = (
        f"{label:16s} "
        f"WMT={system['bleu']['WMT24']:.6f} "
        f"FLORES={system['bleu']['FLORES']:.6f} "
        f"CHALL={system['bleu']['CHALLENGE']:.6f} "
        f"maxdiff={max_diff:.8f} "
        f"family={family}"
    )

    mapping_lines.append(
        line
    )

    print(
        line
    )


# All five logical systems must be different families.
if len(
    set(
        families
    )
) != len(
    families
):

    raise RuntimeError(
        "Two logical systems mapped to the same family"
    )


# Additional semantic path assertions.
if "fkl" not in str(
    systems[
        "FKL"
    ][
        "family"
    ]
).lower():

    raise RuntimeError(
        "FKL path audit failed"
    )


rkl_path = str(
    systems[
        "RKL"
    ][
        "family"
    ]
).lower()


if (
    "rkl" not in rkl_path
    or "pgrkl" in rkl_path
    or "cleanroom" in rkl_path
):

    raise RuntimeError(
        "Exact RKL path audit failed"
    )


pg_path = str(
    systems[
        "PG_RKL"
    ][
        "family"
    ]
).lower()


if (
    "pgrkl" not in pg_path
    or "cleanroom" not in pg_path
):

    raise RuntimeError(
        "PG-RKL path audit failed"
    )


seq_path = str(
    systems[
        "SEQKD_SELECTED"
    ][
        "family"
    ]
).lower()


if (
    "seqkd" not in seq_path
    or "selected" not in seq_path
):

    raise RuntimeError(
        "SeqKD-selected path audit failed"
    )


print()
print(
    "SYSTEM_MAPPING_V2_PASS"
)


###############################################################################
# Row-level comparison
###############################################################################

all_rows = []


for split in (
    "WMT24",
    "FLORES",
    "CHALLENGE",
):

    srcs = canonical[
        split
    ][
        "source"
    ]

    refs = canonical[
        split
    ][
        "reference"
    ]


    for i, (
        src,
        ref,
    ) in enumerate(
        zip(
            srcs,
            refs,
        )
    ):

        row = {
            "split": split,
            "index": i,
            "source": src,
            "reference": ref,
        }


        for label in ORDER:

            hyp = systems[
                label
            ][
                "hyps"
            ][
                split
            ][
                i
            ]


            sb = bleu.sentence_score(
                hyp,
                [
                    ref
                ],
            ).score


            sc = chrf.sentence_score(
                hyp,
                [
                    ref
                ],
            ).score


            row[
                label
            ] = hyp

            row[
                f"{label}_BLEU"
            ] = float(
                sb
            )

            row[
                f"{label}_chrF"
            ] = float(
                sc
            )


        for label in (
            "FKL",
            "RKL",
            "PG_RKL",
            "SEQKD_SELECTED",
        ):

            row[
                f"{label}_dBLEU"
            ] = (
                row[
                    f"{label}_BLEU"
                ]
                - row[
                    "Base_BLEU"
                ]
            )


            row[
                f"{label}_dchrF"
            ] = (
                row[
                    f"{label}_chrF"
                ]
                - row[
                    "Base_chrF"
                ]
            )


            row[
                f"{label}_changed"
            ] = (
                row[
                    label
                ]
                != row[
                    "Base"
                ]
            )


        kl_d = [
            row[
                "FKL_dchrF"
            ],

            row[
                "RKL_dchrF"
            ],

            row[
                "PG_RKL_dchrF"
            ],
        ]


        row[
            "KL_mean_dchrF"
        ] = sum(
            kl_d
        ) / 3.0


        row[
            "KL_min_dchrF"
        ] = min(
            kl_d
        )


        row[
            "KL_max_dchrF"
        ] = max(
            kl_d
        )


        row[
            "KL_spread_dchrF"
        ] = (
            max(
                kl_d
            )
            - min(
                kl_d
            )
        )


        row[
            "RKL_PG_similarity"
        ] = SequenceMatcher(
            None,
            row[
                "RKL"
            ],
            row[
                "PG_RKL"
            ],
        ).ratio()


        all_rows.append(
            row
        )


###############################################################################
# Behavior summary
###############################################################################

summary = []


summary.append(
    "=" * 100
)

summary.append(
    "THREE-KL CASE ANALYSIS V2"
)

summary.append(
    "=" * 100
)

summary.append(
    ""
)

summary.append(
    "IMPORTANT: Base is one verified 3-split family."
)

summary.append(
    ""
)

summary.append(
    "SYSTEM BEHAVIOR VS TRUE BASE"
)

summary.append(
    "-" * 100
)


for label in (
    "FKL",
    "RKL",
    "PG_RKL",
    "SEQKD_SELECTED",
):

    changed = sum(
        row[
            f"{label}_changed"
        ]

        for row in all_rows
    )


    up = sum(
        row[
            f"{label}_dchrF"
        ] > 0.5

        for row in all_rows
    )


    down = sum(
        row[
            f"{label}_dchrF"
        ] < -0.5

        for row in all_rows
    )


    neutral = (
        len(
            all_rows
        )
        - up
        - down
    )


    mean_delta = sum(
        row[
            f"{label}_dchrF"
        ]

        for row in all_rows
    ) / len(
        all_rows
    )


    summary.append(
        f"{label:16s} "
        f"changed={changed:4d}/{len(all_rows)} "
        f"({100*changed/len(all_rows):6.2f}%) "
        f"chrF_up={up:4d} "
        f"chrF_down={down:4d} "
        f"neutral={neutral:4d} "
        f"mean_sentence_dchrF={mean_delta:+.4f}"
    )


###############################################################################
# Split-level summary
###############################################################################

summary.append(
    ""
)

summary.append(
    "PER-SPLIT BEHAVIOR"
)

summary.append(
    "-" * 100
)


for split in (
    "WMT24",
    "FLORES",
    "CHALLENGE",
):

    rows = [
        x
        for x in all_rows
        if x[
            "split"
        ] == split
    ]


    summary.append(
        f"[{split}]"
    )


    for label in (
        "FKL",
        "RKL",
        "PG_RKL",
        "SEQKD_SELECTED",
    ):

        changed = sum(
            x[
                f"{label}_changed"
            ]
            for x in rows
        )


        mean_delta = sum(
            x[
                f"{label}_dchrF"
            ]
            for x in rows
        ) / len(
            rows
        )


        summary.append(
            f"  {label:16s} "
            f"changed={changed:4d}/{len(rows)} "
            f"mean_dchrF={mean_delta:+.4f}"
        )


###############################################################################
# Sampling categories
###############################################################################

CATEGORY_DEFS = []


CATEGORY_DEFS.append(
    (
        "ALL_3_KL_WORSE",

        lambda x: (
            x[
                "FKL_dchrF"
            ] < -0.5
            and
            x[
                "RKL_dchrF"
            ] < -0.5
            and
            x[
                "PG_RKL_dchrF"
            ] < -0.5
        ),

        lambda x: x[
            "KL_mean_dchrF"
        ],

        False,
    )
)


CATEGORY_DEFS.append(
    (
        "ALL_3_KL_BETTER",

        lambda x: (
            x[
                "FKL_dchrF"
            ] > 0.5
            and
            x[
                "RKL_dchrF"
            ] > 0.5
            and
            x[
                "PG_RKL_dchrF"
            ] > 0.5
        ),

        lambda x: x[
            "KL_mean_dchrF"
        ],

        True,
    )
)


CATEGORY_DEFS.append(
    (
        "FKL_UP_RKL_DOWN",

        lambda x: (
            x[
                "FKL_dchrF"
            ] > 1.0
            and
            x[
                "RKL_dchrF"
            ] < -1.0
        ),

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


CATEGORY_DEFS.append(
    (
        "RKL_UP_FKL_DOWN",

        lambda x: (
            x[
                "RKL_dchrF"
            ] > 1.0
            and
            x[
                "FKL_dchrF"
            ] < -1.0
        ),

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


CATEGORY_DEFS.append(
    (
        "PG_VS_EXACT_RKL_DIVERGE",

        lambda x: (
            x[
                "PG_RKL"
            ]
            != x[
                "RKL"
            ]
        ),

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


CATEGORY_DEFS.append(
    (
        "SEQKD_WINS_KL_FAILS",

        lambda x: (
            x[
                "SEQKD_SELECTED_dchrF"
            ] > 2.0
            and
            x[
                "KL_max_dchrF"
            ] < 0.5
        ),

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


CATEGORY_DEFS.append(
    (
        "KL_CHANGED_BUT_CHRF_NEUTRAL",

        lambda x: (
            (
                x[
                    "FKL_changed"
                ]
                or
                x[
                    "RKL_changed"
                ]
                or
                x[
                    "PG_RKL_changed"
                ]
            )
            and
            abs(
                x[
                    "KL_mean_dchrF"
                ]
            ) < 0.30
        ),

        lambda x: (
            int(
                x[
                    "FKL_changed"
                ]
            )
            +
            int(
                x[
                    "RKL_changed"
                ]
            )
            +
            int(
                x[
                    "PG_RKL_changed"
                ]
            )
        ),

        True,
    )
)


CATEGORY_DEFS.append(
    (
        "ALL_KL_SAME_AS_BASE_SEQKD_IMPROVES",

        lambda x: (
            not x[
                "FKL_changed"
            ]
            and
            not x[
                "RKL_changed"
            ]
            and
            not x[
                "PG_RKL_changed"
            ]
            and
            x[
                "SEQKD_SELECTED_changed"
            ]
            and
            x[
                "SEQKD_SELECTED_dchrF"
            ] > 1.0
        ),

        lambda x: x[
            "SEQKD_SELECTED_dchrF"
        ],

        True,
    )
)


selected_cases = []


for (
    category,
    predicate,
    key,
    reverse,
) in CATEGORY_DEFS:

    candidates = [
        row
        for row in all_rows
        if predicate(
            row
        )
    ]


    candidates.sort(
        key=key,
        reverse=reverse,
    )


    chosen = candidates[
        :6
    ]


    summary.append(
        ""
    )

    summary.append(
        f"{category}: "
        f"candidates={len(candidates)}, "
        f"selected={len(chosen)}"
    )


    for rank, row in enumerate(
        chosen,
        start=1,
    ):

        x = dict(
            row
        )

        x[
            "category"
        ] = category

        x[
            "category_rank"
        ] = rank

        selected_cases.append(
            x
        )


###############################################################################
# Write mapping
###############################################################################

mapping_path = (
    OUT
    / "prediction_mapping_v2.txt"
)


mapping_path.write_text(
    "\n".join(
        mapping_lines
    )
    + "\n",
    encoding="utf-8",
)


###############################################################################
# Write summary
###############################################################################

summary_path = (
    OUT
    / "summary.txt"
)


summary_path.write_text(
    "\n".join(
        summary
    )
    + "\n",
    encoding="utf-8",
)


###############################################################################
# Write full JSONL
###############################################################################

all_jsonl = (
    OUT
    / "all_sentence_comparisons.jsonl"
)


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


###############################################################################
# Write selected JSONL
###############################################################################

selected_jsonl = (
    OUT
    / "selected_cases.jsonl"
)


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
# Write CSV
###############################################################################

csv_path = (
    OUT
    / "all_sentence_comparisons.csv"
)


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
    "RKL_PG_similarity",
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
# Human-readable selected cases
###############################################################################

selected_txt = (
    OUT
    / "selected_cases.txt"
)


with selected_txt.open(
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
            "SOURCE:\n"
            + row[
                "source"
            ]
            + "\n\n"
        )


        f.write(
            "REFERENCE:\n"
            + row[
                "reference"
            ]
            + "\n\n"
        )


        f.write(
            "BASE "
            f"(BLEU={row['Base_BLEU']:.2f}, "
            f"chrF={row['Base_chrF']:.2f}):\n"
            + row[
                "Base"
            ]
            + "\n\n"
        )


        for label in (
            "FKL",
            "RKL",
            "PG_RKL",
            "SEQKD_SELECTED",
        ):

            f.write(
                f"{label} "
                f"(BLEU={row[f'{label}_BLEU']:.2f}, "
                f"ΔBLEU={row[f'{label}_dBLEU']:+.2f}, "
                f"chrF={row[f'{label}_chrF']:.2f}, "
                f"ΔchrF={row[f'{label}_dchrF']:+.2f}):\n"
                + row[
                    label
                ]
                + "\n\n"
            )


        f.write(
            "-" * 120
            + "\n"
        )


###############################################################################
# Terminal report
###############################################################################

print()
print(
    "=" * 110
)

print(
    "FINAL V2 BEHAVIOR SUMMARY"
)

print(
    "=" * 110
)


for line in summary:

    print(
        line
    )


print()
print(
    "=" * 110
)

print(
    "V2 CASE EXTRACTION PASS"
)

print(
    "=" * 110
)


print(
    "OUT_DIR="
    + str(
        OUT
    )
)


print(
    "SELECTED_CASES="
    + str(
        selected_txt
    )
)


print(
    "SUMMARY="
    + str(
        summary_path
    )
)


print(
    "MAPPING="
    + str(
        mapping_path
    )
)


print(
    "ALL_CSV="
    + str(
        csv_path
    )
)


print(
    "ALL_JSONL="
    + str(
        all_jsonl
    )
)


print(
    "SELECTED_JSONL="
    + str(
        selected_jsonl
    )
)


print()
print(
    "IMPORTANT:"
)

print(
    "Upload this file to ChatGPT:"
)

print(
    selected_txt
)


print(
    "=" * 110
)
