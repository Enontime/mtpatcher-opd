#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

EXP="mtpatcher_v3_full6565_20260823"

V2="$ROOT/scripts/mtpatcher_v4/extract_three_kl_cases_v2.py"
V3="$ROOT/scripts/mtpatcher_v4/extract_three_kl_cases_v3.py"

CANONICAL_BASE="$RUN_ROOT/$EXP/_verified_base_eval_v3"

echo "======================================================================"
echo "FIND TRUE BASE + THREE-KL CASE ANALYSIS V3"
date
echo "======================================================================"


###############################################################################
# STAGE 1 — VERIFY INPUTS
###############################################################################

echo
echo "===== STAGE 1/5: VERIFY INPUTS ====="

test -f "$V2"

python -m py_compile "$V2"

echo "V2_SCRIPT=$V2"
echo "RUN_ROOT=$RUN_ROOT"
echo "CANONICAL_BASE=$CANONICAL_BASE"

echo "INPUT_CHECK_PASS"


###############################################################################
# STAGE 2 — FIND EXACT BASE PREDICTIONS ACROSS RUN_ROOT
###############################################################################

echo
echo "===== STAGE 2/5: FIND TRUE BASE PREDICTIONS ====="

export CANONICAL_BASE

python - <<'PY'
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from sacrebleu.metrics import BLEU


RUN_ROOT = Path(
    os.environ["RUN_ROOT"]
)

DATA_ROOT = Path(
    os.environ["DATA_ROOT"]
)

CANONICAL = Path(
    os.environ["CANONICAL_BASE"]
)


DATASETS = {
    "wmt24": {
        "data": (
            DATA_ROOT
            / "pilot_v2_qwen3_06b"
            / "wmt24_zh_en998.jsonl"
        ),
        "rows": 998,
        "target": 15.536214,
    },

    "flores": {
        "data": (
            DATA_ROOT
            / "pilot_v2_qwen3_06b"
            / "flores_zh_en1012.jsonl"
        ),
        "rows": 1012,
        "target": 19.971480,
    },

    "challenge": {
        "data": (
            DATA_ROOT
            / "pilot_v2_qwen3_06b"
            / "challenge_zh_en197.jsonl"
        ),
        "rows": 197,
        "target": 16.537871,
    },
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


metric = BLEU(
    effective_order=True
)


def norm(x):

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

            except Exception:
                return None

            if not isinstance(
                obj,
                dict,
            ):
                return None

            rows.append(
                obj
            )

    return rows


def choose_key(
    rows,
    keys,
):

    for key in keys:

        if all(
            isinstance(
                row.get(key),
                str,
            )
            and row[key].strip()
            for row in rows
        ):
            return key

    return None


###############################################################################
# Canonical refs/sources
###############################################################################

canonical = {}


for split, cfg in DATASETS.items():

    rows = load_jsonl(
        cfg["data"]
    )

    if rows is None:

        raise RuntimeError(
            f"Could not load dataset: {cfg['data']}"
        )

    if len(rows) != cfg["rows"]:

        raise RuntimeError(
            f"Dataset row mismatch {split}: "
            f"{len(rows)} != {cfg['rows']}"
        )

    src_key = choose_key(
        rows,
        SRC_KEYS,
    )

    ref_key = choose_key(
        rows,
        REF_KEYS,
    )

    if src_key is None or ref_key is None:

        raise RuntimeError(
            f"Cannot identify source/reference "
            f"for {split}"
        )

    canonical[
        split
    ] = {
        "source": [
            norm(
                x[src_key]
            )
            for x in rows
        ],

        "reference": [
            norm(
                x[ref_key]
            )
            for x in rows
        ],
    }


###############################################################################
# BOUNDED search.
#
# Search only RUN_ROOT.
# Search only files literally called predictions.jsonl.
# Reject files whose row counts do not match held-out sets.
###############################################################################

candidates = []


for path in RUN_ROOT.rglob(
    "predictions.jsonl"
):

    try:

        rel_depth = len(
            path.relative_to(
                RUN_ROOT
            ).parts
        )

    except Exception:
        continue

    # Hard bound.
    if rel_depth > 12:
        continue

    try:
        size = path.stat().st_size

    except OSError:
        continue

    if (
        size <= 0
        or size > 100 * 1024 * 1024
    ):
        continue

    candidates.append(
        path
    )


print(
    "PREDICTION_FILES_SCANNED =",
    len(candidates),
)


results = defaultdict(
    list
)


for path in candidates:

    rows = load_jsonl(
        path
    )

    if not rows:
        continue

    possible = [
        split
        for split, cfg in DATASETS.items()
        if len(rows) == cfg["rows"]
    ]

    if not possible:
        continue

    pred_key = choose_key(
        rows,
        PRED_KEYS,
    )

    if pred_key is None:
        continue

    hyps = [
        norm(
            row[pred_key]
        )
        for row in rows
    ]


    for split in possible:

        expected_src = canonical[
            split
        ][
            "source"
        ]

        refs = canonical[
            split
        ][
            "reference"
        ]


        # ------------------------------------------------------------
        # Strong source alignment check whenever source exists.
        # ------------------------------------------------------------

        src_key = choose_key(
            rows,
            SRC_KEYS,
        )

        if src_key is not None:

            got_src = [
                norm(
                    row[src_key]
                )
                for row in rows
            ]

            matches = sum(
                a == b
                for a, b in zip(
                    got_src,
                    expected_src,
                )
            )

            if matches != len(
                expected_src
            ):
                continue


        # ------------------------------------------------------------
        # Strong reference check whenever reference exists.
        # ------------------------------------------------------------

        ref_key = choose_key(
            rows,
            REF_KEYS,
        )

        if ref_key is not None:

            got_ref = [
                norm(
                    row[ref_key]
                )
                for row in rows
            ]

            matches = sum(
                a == b
                for a, b in zip(
                    got_ref,
                    refs,
                )
            )

            if matches != len(
                refs
            ):
                continue


        score = metric.corpus_score(
            hyps,
            [
                refs
            ],
        ).score


        diff = abs(
            score
            - DATASETS[
                split
            ][
                "target"
            ]
        )


        h = hashlib.sha256()

        for hyp in hyps:

            h.update(
                hyp.encode(
                    "utf-8",
                    errors="replace",
                )
            )

            h.update(
                b"\n"
            )


        results[
            split
        ].append(
            {
                "path": path,
                "score": float(
                    score
                ),
                "diff": float(
                    diff
                ),
                "fingerprint": h.hexdigest(),
                "pred_key": pred_key,
            }
        )


###############################################################################
# Print closest candidates and select exact Base.
###############################################################################

chosen = {}


for split in (
    "wmt24",
    "flores",
    "challenge",
):

    arr = sorted(
        results[
            split
        ],
        key=lambda x: x[
            "diff"
        ],
    )


    print()
    print(
        "=" * 110
    )

    print(
        f"BASE SEARCH: {split.upper()}"
    )

    print(
        f"TARGET_BLEU = "
        f"{DATASETS[split]['target']:.6f}"
    )

    print(
        "=" * 110
    )


    for cand in arr[
        :15
    ]:

        print(
            f"BLEU={cand['score']:.9f} "
            f"diff={cand['diff']:.9f} "
            f"field={cand['pred_key']}"
        )

        print(
            f"  {cand['path']}"
        )


    # Exact stored eval should reproduce to far better than 0.002 BLEU.
    exact = [
        x
        for x in arr
        if x[
            "diff"
        ] <= 0.002
    ]


    if not exact:

        raise RuntimeError(
            f"TRUE BASE {split.upper()} NOT FOUND "
            f"ANYWHERE UNDER RUN_ROOT. "
            f"Closest diff="
            f"{arr[0]['diff'] if arr else 'NONE'}"
        )


    # Multiple copies are okay only if hypotheses are identical.
    fingerprints = {
        x[
            "fingerprint"
        ]
        for x in exact
    }


    if len(
        fingerprints
    ) > 1:

        print()
        print(
            "NON-IDENTICAL EXACT BASE CANDIDATES:"
        )

        for x in exact:

            print(
                x[
                    "fingerprint"
                ],
                x[
                    "path"
                ],
            )

        raise RuntimeError(
            f"Ambiguous true Base for {split}: "
            f"multiple non-identical predictions "
            f"match exact BLEU"
        )


    # Prefer paths explicitly mentioning base.
    exact.sort(
        key=lambda x: (
            0
            if "base" in str(
                x[
                    "path"
                ]
            ).lower()
            else 1,

            x[
                "diff"
            ],

            len(
                str(
                    x[
                        "path"
                    ]
                )
            ),
        )
    )


    chosen[
        split
    ] = exact[
        0
    ]


    print()
    print(
        "TRUE_BASE_SPLIT_FOUND"
    )

    print(
        "PATH =",
        chosen[
            split
        ][
            "path"
        ],
    )

    print(
        "BLEU =",
        chosen[
            split
        ][
            "score"
        ],
    )

    print(
        "DIFF =",
        chosen[
            split
        ][
            "diff"
        ],
    )


###############################################################################
# Build canonical 3-split Base family using symlinks.
#
# We deliberately do NOT copy data or modify original eval outputs.
###############################################################################

CANONICAL.mkdir(
    parents=True,
    exist_ok=True,
)


for split in (
    "wmt24",
    "flores",
    "challenge",
):

    dst_dir = (
        CANONICAL
        / split
    )

    dst_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dst = (
        dst_dir
        / "predictions.jsonl"
    )

    src = chosen[
        split
    ][
        "path"
    ].resolve()


    if dst.is_symlink():

        dst.unlink()

    elif dst.exists():

        raise RuntimeError(
            f"Refusing to overwrite "
            f"non-symlink canonical file: {dst}"
        )


    dst.symlink_to(
        src
    )


###############################################################################
# Verify canonical family.
###############################################################################

print()
print(
    "=" * 110
)

print(
    "CANONICAL BASE FAMILY"
)

print(
    "=" * 110
)


for split in (
    "wmt24",
    "flores",
    "challenge",
):

    dst = (
        CANONICAL
        / split
        / "predictions.jsonl"
    )

    print(
        f"{split:10s} "
        f"{dst} -> {dst.resolve()}"
    )


print()
print(
    "TRUE_BASE_CANONICALIZATION_PASS"
)
PY


###############################################################################
# STAGE 3 — CREATE A FRESH V3 CASE SCRIPT
###############################################################################

echo
echo "===== STAGE 3/5: BUILD CASE ANALYSIS V3 ====="

export V2
export V3

python - <<'PY'
import os
from pathlib import Path


src = Path(
    os.environ["V2"]
)

dst = Path(
    os.environ["V3"]
)


text = src.read_text(
    encoding="utf-8"
)


old = (
    '"three_kl_case_analysis_v2"'
)

new = (
    '"three_kl_case_analysis_v3"'
)


if old not in text:

    raise RuntimeError(
        "Could not find V2 output-directory marker"
    )


text = text.replace(
    old,
    new,
    1,
)


###############################################################################
# Tighten Base discovery preference:
# canonical verified family should be preferred over duplicate exact copies.
###############################################################################

needle = '''
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
'''


replacement = '''
# Prefer the explicitly canonicalized verified Base family.
near_best.sort(
    key=lambda cand: (
        0
        if "_verified_base_eval_v3" in str(
            cand[
                "system"
            ][
                "family"
            ]
        )
        else 1,

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
'''


if needle not in text:

    raise RuntimeError(
        "Could not patch Base preference block"
    )


text = text.replace(
    needle,
    replacement,
    1,
)


dst.write_text(
    text,
    encoding="utf-8",
)


print(
    "CASE_ANALYSIS_V3_BUILD_PASS"
)
PY


python -m py_compile \
"$V3"


###############################################################################
# STAGE 4 — RUN V3 EXTRACTION
###############################################################################

echo
echo "===== STAGE 4/5: RUN V3 EXTRACTION ====="

python \
"$V3"


###############################################################################
# STAGE 5 — FINAL PATH / MAPPING AUDIT
###############################################################################

echo
echo "===== STAGE 5/5: FINAL RESULT ====="

DIR="$ROOT/results/$EXP/three_kl_case_analysis_v3"

MAP="$DIR/prediction_mapping_v2.txt"

TXT="$DIR/selected_cases.txt"

SUMMARY="$DIR/summary.txt"


echo
echo "======================================================================"
echo "FINAL VERIFIED MAPPING"
echo "======================================================================"

cat "$MAP"


echo
echo "======================================================================"
echo "SUMMARY"
echo "======================================================================"

cat "$SUMMARY"


echo
echo "======================================================================"
echo "FILES"
echo "======================================================================"

ls -lh "$DIR"


echo
echo "SELECTED_CASES=$TXT"
echo "SUMMARY=$SUMMARY"
echo "MAPPING=$MAP"

echo
echo "THREE_KL_CASE_ANALYSIS_V3_ALL_PASS"

