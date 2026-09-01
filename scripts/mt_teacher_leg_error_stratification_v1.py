#!/usr/bin/env python3

import json
import os
import statistics
from pathlib import Path


RUN_ROOT = Path(os.environ["RUN_ROOT"])
EXP = os.environ["EXP"]

BASE = (
    RUN_ROOT
    / EXP
    / "prefix_failure_probe_v1"
)

DATA = (
    BASE
    / "teacher_leg_probe_common26_v3.jsonl"
)

OUT = (
    BASE
    / "teacher_leg_error_stratification_v1.jsonl"
)


LABELS = {
    # H0 has no clear independent substantive
    # translation error after Patcher correction.
    227:  "single_error_clean",
    303:  "single_error_clean",
    1000: "single_error_clean",
    1018: "single_error_clean",
    1301: "single_error_clean",
    1332: "single_error_clean",
    1343: "single_error_clean",
    1374: "single_error_clean",
    1394: "single_error_clean",
    1452: "single_error_clean",
    1494: "single_error_clean",
    1671: "single_error_clean",
    1706: "single_error_clean",
    1710: "single_error_clean",
    1871: "single_error_clean",
    1983: "single_error_clean",

    # H0 still contains at least one clear,
    # independently identifiable adequacy error.
    548:  "extra_error",
    998:  "extra_error",
    1224: "extra_error",
    1234: "extra_error",
    1496: "extra_error",
    1651: "extra_error",
    1751: "extra_error",
    1980: "extra_error",

    # Conservative exclusions.
    1090: "ambiguous",
    1204: "invalid_patcher",
}


RATIONALE = {
    548:  "St. George remains mistranslated; source says Sturgis.",
    998:  "H0 gets current diabetes status/polarity wrong.",
    1224: "H0 omits the independent fell-off-stage event.",
    1234: "H0 retains independent title/content inaccuracies.",
    1496: "H0 retains self-sufficient agriculture instead of subsistence agriculture.",
    1651: "H0 retains magma for source lava.",
    1751: "H0 retains horse racing instead of polo.",
    1980: "H0 retains abandoned instead of obsolete/phased out.",
    1090: "in a galaxy vs around a galaxy is semantically debatable.",
    1204: "Patcher Before->Since correction is itself semantically invalid.",
}


def mean(xs):
    return sum(xs) / len(xs)


def summarize(name, xs):
    print(
        f"{name}: "
        f"n={len(xs)} "
        f"mean={mean(xs):+.4f} "
        f"median={statistics.median(xs):+.4f} "
        f">+1={sum(x > 1 for x in xs)}/{len(xs)} "
        f"<-1={sum(x < -1 for x in xs)}/{len(xs)}"
    )


with DATA.open(
    encoding="utf-8",
) as f:
    rows = [
        json.loads(line)
        for line in f
        if line.strip()
    ]


if len(rows) != 26:
    raise RuntimeError(
        f"Expected 26 rows, got {len(rows)}"
    )


seen = {
    int(x["job_id"])
    for x in rows
}

if seen != set(LABELS):
    raise RuntimeError(
        "Label/artifact job IDs do not match"
    )


out_rows = []

for x in rows:
    jid = int(
        x["job_id"]
    )

    y = {
        "job_id": jid,
        "dataset": x["dataset"],
        "label": LABELS[jid],
        "rationale": RATIONALE.get(
            jid,
            "No clear independent substantive error in H0.",
        ),
        "h0_ref_chrf": float(
            x["h0"]["ref_chrf"]
        ),
        "t2_delta": (
            float(
                x["teacher"]["horizons"]["2"]["ref_chrf"]
            )
            - float(
                x["h0"]["ref_chrf"]
            )
        ),
        "t4_delta": (
            float(
                x["teacher"]["horizons"]["4"]["ref_chrf"]
            )
            - float(
                x["h0"]["ref_chrf"]
            )
        ),
        "t8_delta": (
            float(
                x["teacher"]["horizons"]["8"]["ref_chrf"]
            )
            - float(
                x["h0"]["ref_chrf"]
            )
        ),
        "full_delta": (
            float(
                x["teacher"]["full_ref_chrf"]
            )
            - float(
                x["h0"]["ref_chrf"]
            )
        ),
    }

    out_rows.append(y)


with OUT.open(
    "w",
    encoding="utf-8",
) as f:
    for x in out_rows:
        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )


print("=" * 100)
print("ERROR-STRATIFIED TEACHER-LEG AUDIT")
print("=" * 100)

for label in [
    "single_error_clean",
    "extra_error",
    "ambiguous",
    "invalid_patcher",
]:
    xs = [
        x
        for x in out_rows
        if x["label"] == label
    ]

    print()
    print(
        f"{label}: N={len(xs)}"
    )

    if label in {
        "ambiguous",
        "invalid_patcher",
    }:
        for x in xs:
            print(
                f"JOB={x['job_id']} "
                f"{x['rationale']}"
            )
        continue

    for key, name in [
        ("t2_delta", "T2-H0"),
        ("t4_delta", "T4-H0"),
        ("t8_delta", "T8-H0"),
        ("full_delta", "TFULL-H0"),
    ]:
        summarize(
            name,
            [
                x[key]
                for x in xs
            ],
        )

    print()
    for x in xs:
        print(
            f"JOB={x['job_id']} "
            f"T2={x['t2_delta']:+.3f} "
            f"T4={x['t4_delta']:+.3f} "
            f"T8={x['t8_delta']:+.3f} "
            f"FULL={x['full_delta']:+.3f}"
        )


print()
print(
    f"OUTPUT={OUT}"
)

print(
    "ERROR_STRATIFICATION_V1_PASS"
)
