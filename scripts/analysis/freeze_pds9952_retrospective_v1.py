#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path("/workspace/mtpatcher")

RETRO = (
    ROOT
    / "runs/science/pe_pds_v1/pds9952_retrospective_v1"
)

SEQ = RETRO / "seqkd_native_persistent_v1"
OPD = RETRO / "opd"

END = (
    ROOT
    / "runs/science/pe_pds_v1/pds9952_endpoint_validation_v4"
)

OUT = RETRO / "freeze_v1"

EXPECTED_ROWS = 3231
TOL = 2e-6

STEPS_CORE = list(range(0, 3701, 100))
STEPS_NONZERO = list(range(100, 3701, 100))
STEPS = STEPS_CORE + [3732]

EXPECTED_SEQ_ENDPOINT_SHA = (
    "3d67351ac25232f9483b152634e09d2e39621aa68558a5dffd34cbcd29f2f98b"
)

FATAL_PATTERNS = [
    "Error executing job",
    "RayTaskError",
    "RuntimeError",
    "ValueError",
    "ActorDiedError",
    "ACL_ERROR",
    "EZ9999",
    "MTE error",
    "OutOfMemory",
    "OOM",
]


def die(msg: str) -> None:
    raise RuntimeError(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canonical_hash(items) -> str:
    """
    Order-insensitive but multiplicity-sensitive hash.
    Each item is serialized independently, then sorted.
    """
    encoded = [
        json.dumps(
            x,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=False,
        ).encode("utf-8")
        for x in items
    ]
    encoded.sort()

    h = hashlib.sha256()
    for b in encoded:
        h.update(len(b).to_bytes(8, "big"))
        h.update(b)
    return h.hexdigest()


def read_generation(path: Path, expected_step: int) -> dict:
    if not path.is_file():
        die(f"MISSING_GENERATION: {path}")

    rows = []

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as e:
                die(f"JSON_PARSE_FAIL {path}:{lineno}: {e}")

            for key in ("input", "output", "step", "uid"):
                if key not in row:
                    die(
                        f"MISSING_KEY path={path} "
                        f"line={lineno} key={key}"
                    )

            if not isinstance(row["input"], str):
                die(f"INPUT_NOT_STRING {path}:{lineno}")

            if not isinstance(row["output"], str):
                die(f"OUTPUT_NOT_STRING {path}:{lineno}")

            if int(row["step"]) != int(expected_step):
                die(
                    f"STEP_MISMATCH {path}:{lineno} "
                    f"observed={row['step']} expected={expected_step}"
                )

            rows.append(row)

    if len(rows) != EXPECTED_ROWS:
        die(
            f"ROW_COUNT_FAIL {path}: "
            f"{len(rows)} != {EXPECTED_ROWS}"
        )

    inputs = [r["input"] for r in rows]
    pairs = [[r["input"], r["output"]] for r in rows]
    uids = [str(r["uid"]) for r in rows]

    uid_unique = len(set(uids))

    if uid_unique != EXPECTED_ROWS:
        die(
            f"UID_UNIQUENESS_FAIL {path}: "
            f"{uid_unique} != {EXPECTED_ROWS}"
        )

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "input_multiset_sha256": canonical_hash(inputs),
        "pair_multiset_sha256": canonical_hash(pairs),
        "uid_multiset_sha256": canonical_hash(uids),
        "uid_unique": uid_unique,
    }


def generation_steps(directory: Path) -> list[int]:
    steps = []

    for p in directory.glob("*.jsonl"):
        try:
            steps.append(int(p.stem))
        except ValueError:
            die(f"NON_NUMERIC_GENERATION_FILENAME: {p}")

    return sorted(steps)


def read_compare_metrics(tb_dir: Path) -> dict:
    if not tb_dir.is_dir():
        die(f"MISSING_TENSORBOARD_DIR: {tb_dir}")

    events = list(tb_dir.glob("events.out.tfevents.*"))
    if not events:
        die(f"NO_TENSORBOARD_EVENTS: {tb_dir}")

    ea = EventAccumulator(
        str(tb_dir),
        size_guidance={"scalars": 0},
    )
    ea.Reload()

    scalar_tags = ea.Tags().get("scalars", [])
    tags = sorted(
        t for t in scalar_tags
        if t.startswith("compare/")
    )

    if len(tags) != 10:
        print("OBSERVED_COMPARE_TAGS:")
        for t in tags:
            print("  ", t)

        die(
            f"COMPARE_TAG_COUNT_FAIL {tb_dir}: "
            f"{len(tags)} != 10"
        )

    result = {}

    for tag in tags:
        by_step = defaultdict(list)

        for ev in ea.Scalars(tag):
            by_step[int(ev.step)].append(float(ev.value))

        for step, vals in by_step.items():
            if max(vals) - min(vals) > TOL:
                die(
                    f"TB_DUPLICATE_CONFLICT "
                    f"tag={tag} step={step} vals={vals}"
                )

            result.setdefault(step, {})[tag] = vals[-1]

    return {
        "tags": tags,
        "metrics": result,
        "event_files": [
            {
                "path": str(p),
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
            }
            for p in sorted(events)
        ],
    }


def require_metrics(
    source: dict,
    step: int,
    tagset: list[str],
    label: str,
) -> dict:
    metrics = source["metrics"].get(step)

    if metrics is None:
        die(f"MISSING_TB_STEP {label}: {step}")

    if sorted(metrics) != sorted(tagset):
        die(
            f"TB_TAGSET_MISMATCH {label} step={step}"
        )

    return metrics


def metric_equal(a: dict, b: dict, tags: list[str]) -> bool:
    for tag in tags:
        if abs(float(a[tag]) - float(b[tag])) > TOL:
            return False
    return True


def metric_max_abs_diff(
    a: dict,
    b: dict,
    tags: list[str],
) -> float:
    return max(
        abs(float(a[t]) - float(b[t]))
        for t in tags
    )


def find_macro_tag(
    tags: list[str],
    token: str,
) -> str | None:
    candidates = [
        t for t in tags
        if "macro" in t.lower()
        and token.lower() in t.lower()
    ]

    if len(candidates) == 1:
        return candidates[0]

    return None


def sign(x: float, eps: float = 1e-12) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def count_sign_crossings(values: list[float]) -> int:
    signs = [sign(x) for x in values]
    signs = [x for x in signs if x != 0]

    return sum(
        1 for a, b in zip(signs, signs[1:])
        if a != b
    )


OUT.mkdir(parents=True, exist_ok=True)

print("============================================================")
print("PDS9952 RETROSPECTIVE FREEZE V1")
print("============================================================")


# ------------------------------------------------------------
# 1. Exact generation inventories
# ------------------------------------------------------------

seq_actual_steps = generation_steps(SEQ / "generations")
opd_actual_steps = generation_steps(OPD / "generations")

if seq_actual_steps != STEPS_CORE:
    die(
        "SEQKD_STEP_INVENTORY_FAIL\n"
        f"observed={seq_actual_steps}\n"
        f"expected={STEPS_CORE}"
    )

if opd_actual_steps != STEPS:
    die(
        "OPD_STEP_INVENTORY_FAIL\n"
        f"observed={opd_actual_steps}\n"
        f"expected={STEPS}"
    )

print("SEQKD_GENERATION_INVENTORY=PASS")
print("OPD_GENERATION_INVENTORY=PASS")


# ------------------------------------------------------------
# 2. SeqKD replay PASS contract + fatal gate
# ------------------------------------------------------------

seq_log = (SEQ / "run.log").read_text(
    encoding="utf-8",
    errors="replace",
)

passes = re.findall(
    r"MATCHED20K_PERSISTENT_REPLAY_STEP\s+([0-9]+)=PASS",
    seq_log,
)

pass_counts = Counter(int(x) for x in passes)

if sorted(pass_counts) != STEPS_NONZERO:
    die(
        "SEQKD_PASS_STEP_SET_FAIL\n"
        f"observed={sorted(pass_counts)}"
    )

bad_counts = {
    step: count
    for step, count in pass_counts.items()
    if count != 1
}

if bad_counts:
    die(f"SEQKD_PASS_DUPLICATE_FAIL: {bad_counts}")

fatal_hits = [
    p for p in FATAL_PATTERNS
    if p in seq_log
]

if fatal_hits:
    die(
        "SEQKD_FATAL_GATE_FAIL: "
        + repr(fatal_hits)
    )

print("SEQKD_REPLAY_PASS_37_37=PASS")
print("SEQKD_FATAL_GATE=PASS")


# ------------------------------------------------------------
# 3. Read raw evidence
# ------------------------------------------------------------

seq_gen = {}
opd_gen = {}

for step in STEPS_CORE:
    seq_gen[step] = read_generation(
        SEQ / "generations" / f"{step}.jsonl",
        step,
    )

seq_endpoint_path = (
    END
    / "seqkd_step3732/generations/3732.jsonl"
)

seq_endpoint_sha = sha256_file(seq_endpoint_path)

if seq_endpoint_sha != EXPECTED_SEQ_ENDPOINT_SHA:
    die(
        "SEQKD_ENDPOINT_SHA_DRIFT\n"
        f"observed={seq_endpoint_sha}\n"
        f"expected={EXPECTED_SEQ_ENDPOINT_SHA}"
    )

seq_gen[3732] = read_generation(
    seq_endpoint_path,
    3732,
)

for step in STEPS:
    opd_gen[step] = read_generation(
        OPD / "generations" / f"{step}.jsonl",
        step,
    )

common0 = read_generation(
    END / "common_step0/generations/0.jsonl",
    0,
)

opd_endpoint = read_generation(
    END / "opd_step3732/generations/3732.jsonl",
    3732,
)

print("RAW_GENERATION_ROW_GATE=PASS")
print("SEQKD_FROZEN_3732_SHA=PASS")


# ------------------------------------------------------------
# 4. Stable validation input population
# ------------------------------------------------------------

reference_input_hash = common0["input_multiset_sha256"]

for arm_name, arm in (
    ("seqkd", seq_gen),
    ("opd", opd_gen),
):
    for step in STEPS:
        observed = arm[step]["input_multiset_sha256"]

        if observed != reference_input_hash:
            die(
                f"INPUT_MULTISET_DRIFT "
                f"arm={arm_name} step={step}\n"
                f"observed={observed}\n"
                f"reference={reference_input_hash}"
            )

print("ALL_78_TRAJECTORY_INPUT_MULTISETS=PASS")


# ------------------------------------------------------------
# 5. Required semantic parity anchors
# ------------------------------------------------------------

if (
    seq_gen[0]["pair_multiset_sha256"]
    != common0["pair_multiset_sha256"]
):
    die("SEQKD_STEP0_PAIR_PARITY_FAIL")

if (
    opd_gen[0]["pair_multiset_sha256"]
    != common0["pair_multiset_sha256"]
):
    die("OPD_STEP0_PAIR_PARITY_FAIL")

if (
    seq_gen[0]["pair_multiset_sha256"]
    != opd_gen[0]["pair_multiset_sha256"]
):
    die("SEQKD_OPD_STEP0_PAIR_PARITY_FAIL")

if (
    opd_gen[3732]["pair_multiset_sha256"]
    != opd_endpoint["pair_multiset_sha256"]
):
    die("OPD_3732_ENDPOINT_PAIR_PARITY_FAIL")

print("STEP0_PAIR_MULTISET_PARITY=PASS")
print("OPD_3732_ENDPOINT_PAIR_MULTISET_PARITY=PASS")


# ------------------------------------------------------------
# 6. TensorBoard metric extraction
# ------------------------------------------------------------

seq_tb = read_compare_metrics(
    SEQ / "tensorboard"
)

opd_tb = read_compare_metrics(
    OPD / "tensorboard"
)

common0_tb = read_compare_metrics(
    END / "common_step0/tensorboard"
)

seq_endpoint_tb = read_compare_metrics(
    END / "seqkd_step3732/tensorboard"
)

opd_endpoint_tb = read_compare_metrics(
    END / "opd_step3732/tensorboard"
)

tagsets = [
    seq_tb["tags"],
    opd_tb["tags"],
    common0_tb["tags"],
    seq_endpoint_tb["tags"],
    opd_endpoint_tb["tags"],
]

if not all(x == tagsets[0] for x in tagsets[1:]):
    print("TAGSETS:")
    for i, x in enumerate(tagsets):
        print(i, x)
    die("COMPARE_METRIC_TAGSET_GLOBAL_MISMATCH")

tags = tagsets[0]

print("COMPARE_METRIC_TAGS=10")
for t in tags:
    print("METRIC_TAG", t)


# ------------------------------------------------------------
# 7. Build exact 39-point metric trajectories
# ------------------------------------------------------------

seq_metrics = {}
opd_metrics = {}

common0_metrics = require_metrics(
    common0_tb,
    0,
    tags,
    "common_step0",
)

# Step 0 is the single shared frozen stage-entry evaluation.
seq_metrics[0] = dict(common0_metrics)
opd_metrics[0] = dict(common0_metrics)

# If retrospective TB itself contains step0, require agreement.
if 0 in seq_tb["metrics"]:
    x = require_metrics(seq_tb, 0, tags, "seq_retro_step0")
    if not metric_equal(x, common0_metrics, tags):
        die(
            "SEQ_RETRO_STEP0_METRIC_PARITY_FAIL "
            f"maxdiff={metric_max_abs_diff(x, common0_metrics, tags)}"
        )

if 0 in opd_tb["metrics"]:
    x = require_metrics(opd_tb, 0, tags, "opd_retro_step0")
    if not metric_equal(x, common0_metrics, tags):
        die(
            "OPD_RETRO_STEP0_METRIC_PARITY_FAIL "
            f"maxdiff={metric_max_abs_diff(x, common0_metrics, tags)}"
        )


for step in STEPS_NONZERO:
    seq_metrics[step] = require_metrics(
        seq_tb,
        step,
        tags,
        f"seqkd_{step}",
    )

    opd_metrics[step] = require_metrics(
        opd_tb,
        step,
        tags,
        f"opd_{step}",
    )


seq_ep_metrics = require_metrics(
    seq_endpoint_tb,
    3732,
    tags,
    "seqkd_endpoint_3732",
)

opd_retro_ep_metrics = require_metrics(
    opd_tb,
    3732,
    tags,
    "opd_retro_3732",
)

opd_frozen_ep_metrics = require_metrics(
    opd_endpoint_tb,
    3732,
    tags,
    "opd_endpoint_3732",
)

if not metric_equal(
    opd_retro_ep_metrics,
    opd_frozen_ep_metrics,
    tags,
):
    die(
        "OPD_3732_METRIC_PARITY_FAIL "
        f"maxdiff="
        f"{metric_max_abs_diff(opd_retro_ep_metrics, opd_frozen_ep_metrics, tags)}"
    )

seq_metrics[3732] = dict(seq_ep_metrics)
opd_metrics[3732] = dict(opd_retro_ep_metrics)

print("METRIC_EXTRACTION_39x2=PASS")
print("OPD_3732_METRIC_ENDPOINT_PARITY=PASS")


# ------------------------------------------------------------
# 8. Write paired CSV
# ------------------------------------------------------------

csv_path = OUT / "trajectory_summary_v1.csv"

header = [
    "step",
    "seqkd_generation_source",
    "opd_generation_source",
]

for tag in tags:
    header.extend(
        [
            f"seqkd::{tag}",
            f"opd::{tag}",
            f"delta_opd_minus_seqkd::{tag}",
        ]
    )

with csv_path.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:
    w = csv.DictWriter(f, fieldnames=header)
    w.writeheader()

    for step in STEPS:
        row = {
            "step": step,
            "seqkd_generation_source": seq_gen[step]["path"],
            "opd_generation_source": opd_gen[step]["path"],
        }

        for tag in tags:
            s = float(seq_metrics[step][tag])
            o = float(opd_metrics[step][tag])

            row[f"seqkd::{tag}"] = repr(s)
            row[f"opd::{tag}"] = repr(o)
            row[f"delta_opd_minus_seqkd::{tag}"] = repr(o - s)

        w.writerow(row)


# ------------------------------------------------------------
# 9. Descriptive trajectory geometry
#    No endpoint selection. No causal interpretation.
# ------------------------------------------------------------

macro_bleu = find_macro_tag(tags, "bleu")
macro_chrf = find_macro_tag(tags, "chrf")

descriptive = {}

for label, tag in (
    ("macro_bleu", macro_bleu),
    ("macro_chrf", macro_chrf),
):
    if tag is None:
        descriptive[label] = {
            "tag": None,
            "status": "AMBIGUOUS_OR_NOT_FOUND",
        }
        continue

    deltas = [
        float(opd_metrics[s][tag])
        - float(seq_metrics[s][tag])
        for s in STEPS
    ]

    nonzero_deltas = [
        float(opd_metrics[s][tag])
        - float(seq_metrics[s][tag])
        for s in STEPS
        if s != 0
    ]

    positive = sum(x > 0 for x in nonzero_deltas)
    negative = sum(x < 0 for x in nonzero_deltas)
    tie = len(nonzero_deltas) - positive - negative

    descriptive[label] = {
        "tag": tag,
        "endpoint_step": 3732,
        "endpoint_seqkd": float(seq_metrics[3732][tag]),
        "endpoint_opd": float(opd_metrics[3732][tag]),
        "endpoint_delta_opd_minus_seqkd": (
            float(opd_metrics[3732][tag])
            - float(seq_metrics[3732][tag])
        ),
        "nonzero_step_delta_mean": statistics.mean(
            nonzero_deltas
        ),
        "nonzero_step_delta_median": statistics.median(
            nonzero_deltas
        ),
        "positive_steps": positive,
        "negative_steps": negative,
        "tie_steps": tie,
        "sign_crossings": count_sign_crossings(
            nonzero_deltas
        ),
        "min_delta": min(nonzero_deltas),
        "min_delta_step": STEPS[1:][
            nonzero_deltas.index(min(nonzero_deltas))
        ],
        "max_delta": max(nonzero_deltas),
        "max_delta_step": STEPS[1:][
            nonzero_deltas.index(max(nonzero_deltas))
        ],
        "note": (
            "Extrema are descriptive trajectory geometry only; "
            "they do not select or redefine an endpoint."
        ),
    }


# ------------------------------------------------------------
# 10. JSON freeze record
# ------------------------------------------------------------

record = {
    "version": 1,
    "experiment": (
        "PDS9952 retrospective SeqKD vs OPD learning dynamics"
    ),
    "classification": "RETROSPECTIVE_DYNAMICS_ONLY",
    "status": "PASS",
    "endpoint_policy": {
        "primary_endpoint": 3732,
        "endpoint_reselection_allowed": False,
        "best_checkpoint_selection_allowed": False,
        "seqkd_step3732_source": (
            "frozen endpoint_validation_v4"
        ),
        "opd_step3732_source": (
            "retrospective raw generation, "
            "cross-checked against frozen endpoint_validation_v4"
        ),
    },
    "trajectory": {
        "steps": STEPS,
        "point_count_per_arm": len(STEPS),
        "rows_per_point": EXPECTED_ROWS,
        "compare_metric_count": len(tags),
        "compare_metric_tags": tags,
    },
    "gates": {
        "seqkd_generation_inventory": "PASS",
        "opd_generation_inventory": "PASS",
        "seqkd_replay_pass_count": "37/37",
        "seqkd_fatal_gate": "PASS",
        "raw_generation_rows": "PASS",
        "all_input_multisets": "PASS",
        "step0_pair_multiset_parity": "PASS",
        "opd_3732_endpoint_pair_multiset_parity": "PASS",
        "seqkd_3732_frozen_sha": "PASS",
        "metric_extraction_39x2": "PASS",
        "opd_3732_metric_endpoint_parity": "PASS",
    },
    "reference_input_multiset_sha256": reference_input_hash,
    "seqkd_frozen_endpoint_sha256": seq_endpoint_sha,
    "generation_evidence": {
        "seqkd": seq_gen,
        "opd": opd_gen,
        "common_step0_endpoint": common0,
        "opd_frozen_endpoint": opd_endpoint,
    },
    "tensorboard": {
        "seqkd_retrospective": seq_tb["event_files"],
        "opd_retrospective": opd_tb["event_files"],
        "common_step0_endpoint": common0_tb["event_files"],
        "seqkd_step3732_endpoint": seq_endpoint_tb["event_files"],
        "opd_step3732_endpoint": opd_endpoint_tb["event_files"],
    },
    "metrics": {
        str(step): {
            "seqkd": seq_metrics[step],
            "opd": opd_metrics[step],
            "delta_opd_minus_seqkd": {
                tag: (
                    float(opd_metrics[step][tag])
                    - float(seq_metrics[step][tag])
                )
                for tag in tags
            },
        }
        for step in STEPS
    },
    "descriptive_trajectory_geometry": descriptive,
    "interpretation_boundary": [
        "Retrospective checkpoints are for dynamics analysis only.",
        "The primary endpoint remains frozen at step 3732.",
        "No best-checkpoint selection is permitted.",
        "Single-seed results are descriptive only.",
        "Trajectory shape alone does not establish a causal mechanism.",
    ],
}

json_path = OUT / "trajectory_summary_v1.json"
json_path.write_text(
    json.dumps(
        record,
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


# ------------------------------------------------------------
# 11. Human-readable interpretation record
# ------------------------------------------------------------

md = []

md.append("# PDS9952 Retrospective Trajectory Freeze v1")
md.append("")
md.append("Status: **PASS**")
md.append("")
md.append("## Frozen scope")
md.append("")
md.append(
    "- Paired trajectory: `0,100,...,3700,3732`."
)
md.append("- 39 points per arm.")
md.append("- 3231 validation rows per point.")
md.append("- 10 `compare/*` metrics per point.")
md.append(
    "- Primary endpoint remains **step3732**."
)
md.append(
    "- Retrospective trajectory MUST NOT redefine the endpoint."
)
md.append("")
md.append("## Evidence construction")
md.append("")
md.append(
    "- SeqKD `0..3700`: persistent native FSDP replay."
)
md.append(
    "- SeqKD `3732`: frozen endpoint-validation artifact."
)
md.append(
    "- OPD `0..3732`: persistent retrospective replay."
)
md.append(
    "- OPD `3732`: cross-checked against frozen endpoint validation."
)
md.append("")
md.append("## Gates")
md.append("")
for k, v in record["gates"].items():
    md.append(f"- `{k}`: **{v}**")

md.append("")
md.append("## Descriptive trajectory geometry")
md.append("")

for label in ("macro_bleu", "macro_chrf"):
    d = descriptive[label]

    md.append(f"### {label}")

    if d.get("tag") is None:
        md.append("")
        md.append(
            "Macro tag could not be identified unambiguously."
        )
        md.append("")
        continue

    md.append("")
    md.append(f"- TensorBoard tag: `{d['tag']}`")
    md.append(
        f"- Endpoint SeqKD: `{d['endpoint_seqkd']:.9f}`"
    )
    md.append(
        f"- Endpoint OPD: `{d['endpoint_opd']:.9f}`"
    )
    md.append(
        "- Endpoint OPD−SeqKD: "
        f"`{d['endpoint_delta_opd_minus_seqkd']:+.9f}`"
    )
    md.append(
        "- Mean OPD−SeqKD over nonzero trajectory points: "
        f"`{d['nonzero_step_delta_mean']:+.9f}`"
    )
    md.append(
        "- Median OPD−SeqKD over nonzero trajectory points: "
        f"`{d['nonzero_step_delta_median']:+.9f}`"
    )
    md.append(
        "- Positive / negative / tie points: "
        f"`{d['positive_steps']} / "
        f"{d['negative_steps']} / "
        f"{d['tie_steps']}`"
    )
    md.append(
        f"- Sign crossings: `{d['sign_crossings']}`"
    )
    md.append(
        "- Minimum observed gap: "
        f"`{d['min_delta']:+.9f}` at step "
        f"`{d['min_delta_step']}`"
    )
    md.append(
        "- Maximum observed gap: "
        f"`{d['max_delta']:+.9f}` at step "
        f"`{d['max_delta_step']}`"
    )
    md.append("")
    md.append(
        "These extrema describe trajectory geometry only. "
        "They do not select a checkpoint."
    )
    md.append("")

md.append("## Interpretation boundary")
md.append("")
md.append(
    "This artifact describes single-seed learning dynamics. "
    "It does not establish general superiority, significance, "
    "or a causal mechanism."
)
md.append("")
md.append(
    "The paper-facing primary endpoint remains the pre-frozen "
    "step3732 comparison."
)
md.append("")

md_path = OUT / "trajectory_interpretation_v1.md"
md_path.write_text(
    "\n".join(md) + "\n",
    encoding="utf-8",
)


# ------------------------------------------------------------
# 12. Hash manifest
# ------------------------------------------------------------

evidence_files = set()

for meta in seq_gen.values():
    evidence_files.add(Path(meta["path"]))

for meta in opd_gen.values():
    evidence_files.add(Path(meta["path"]))

evidence_files.update(
    [
        END / "common_step0/generations/0.jsonl",
        END / "opd_step3732/generations/3732.jsonl",
        END / "common_step0/status.json",
        END / "seqkd_step3732/status.json",
        END / "opd_step3732/status.json",
        END / "endpoint_summary.json",
        END / "PASS",
        SEQ / "config/persistent16_full.yaml",
        OPD / "config/persistent.yaml",
        SEQ / "run.log",
        OPD / "run.log",
        csv_path,
        json_path,
        md_path,
    ]
)

for d in (
    SEQ / "provenance",
    OPD / "provenance",
    END / "provenance",
):
    if d.is_dir():
        for p in d.rglob("*"):
            if p.is_file():
                evidence_files.add(p)

for source in (
    seq_tb,
    opd_tb,
    common0_tb,
    seq_endpoint_tb,
    opd_endpoint_tb,
):
    for item in source["event_files"]:
        evidence_files.add(Path(item["path"]))

manifest_path = OUT / "trajectory_hash_manifest_v1.sha256"

with manifest_path.open(
    "w",
    encoding="utf-8",
) as f:
    for path in sorted(
        evidence_files,
        key=lambda p: str(p),
    ):
        if not path.is_file():
            die(f"HASH_MANIFEST_MISSING_FILE: {path}")

        digest = sha256_file(path)

        try:
            rel = path.relative_to(ROOT)
            display = str(rel)
        except ValueError:
            display = str(path)

        f.write(f"{digest}  {display}\n")


manifest_self_sha = sha256_file(manifest_path)

(
    OUT / "trajectory_hash_manifest_v1.sha256.self"
).write_text(
    manifest_self_sha
    + "  trajectory_hash_manifest_v1.sha256\n",
    encoding="utf-8",
)


# ------------------------------------------------------------
# 13. Final PASS
# ------------------------------------------------------------

(OUT / "PASS").write_text(
    "PDS9952_RETROSPECTIVE_FREEZE_V1=PASS\n",
    encoding="utf-8",
)

print()
print("============================================================")
print("PDS9952_RETROSPECTIVE_FREEZE_V1=PASS")
print("============================================================")
print(f"OUT={OUT}")
print(f"CSV={csv_path}")
print(f"JSON={json_path}")
print(f"INTERPRETATION={md_path}")
print(f"HASH_MANIFEST={manifest_path}")
print(f"HASH_MANIFEST_SHA256={manifest_self_sha}")
print()
print("DESCRIPTIVE_TRAJECTORY_GEOMETRY=")
print(
    json.dumps(
        descriptive,
        ensure_ascii=False,
        indent=2,
    )
)
