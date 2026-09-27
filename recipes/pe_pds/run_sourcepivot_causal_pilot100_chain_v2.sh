#!/usr/bin/env bash

set +e
set +u
set +o pipefail 2>/dev/null

source /workspace/mtpatcher/project_env.sh

ROOT=/workspace/mtpatcher
PROJECT="$ROOT/repo/MT-Patcher-Reproduction-Ascend"

BASE="$ROOT/runs/diagnostics/pe_pds_v1/sourcepivot_causal_pilot100_v2"

STATUS="$BASE/chain_status.txt"

mkdir -p "$BASE"
: > "$STATUS"

echo "===== C SHIFT CONTROL: 1-STEP GATE ====="

bash \
  "$PROJECT/recipes/pe_pds/run_pe11792_shiftctrl_smoke1_v2.sh"

S_RC=$?

echo "C_SMOKE_RC=$S_RC" \
    | tee -a "$STATUS"

if [[ "$S_RC" -ne 0 ]]; then
    echo "CHAIN_STOP=C_SMOKE" \
        | tee -a "$STATUS"
    exit "$S_RC"
fi

echo
echo "===== C CONTROL AUDIT ====="

PY="$ROOT/envs/mtpatcher-npu-py311/bin/python"

"$PY" - "$BASE/C_smoke1/audit" <<'PYAUDIT'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])

rows = []

for p in sorted(
    root.glob(
        "worker_*.jsonl"
    )
):
    for line in p.read_text(
        encoding="utf-8"
    ).splitlines():
        if line.strip():
            rows.append(
                json.loads(line)
            )

print(
    "AUDIT_ROWS =",
    len(rows),
)

assert len(rows) == 16, len(rows)

projected = [
    x
    for x in rows
    if int(
        x.get(
            "projected_active_tokens",
            0,
        )
    ) > 0
]

print(
    "PROJECTED_ROWS =",
    len(projected),
)

assert projected

for x in projected:
    assert (
        int(x["active_tokens"])
        == int(
            x[
                "projected_active_tokens"
            ]
        )
    )

    assert (
        x.get(
            "control_changed"
        )
        is True
    )

    assert int(
        x.get(
            "control_shift",
            0,
        )
    ) != 0

    assert abs(
        float(
            x["weight_mean"]
        )
        - 1.0
    ) < 1e-5

print(
    "MATCHED_GLOBAL_STEPS =",
    sorted({
        int(
            x.get(
                "matched_global_step",
                -1,
            )
        )
        for x in rows
    }),
)

print(
    "SCHEDULE_POSITIONS =",
    sorted(
        int(
            x.get(
                "schedule_position",
                -1,
            )
        )
        for x in rows
    ),
)

print(
    "BATCH_POSITIONS =",
    sorted(
        int(
            x.get(
                "position_in_global_batch",
                -1,
            )
        )
        for x in rows
    ),
)

assert {
    int(
        x.get(
            "matched_global_step",
            -1,
        )
    )
    for x in rows
} == {1}

assert sorted(
    int(
        x.get(
            "schedule_position",
            -1,
        )
    )
    for x in rows
) == list(range(16))

assert sorted(
    int(
        x.get(
            "position_in_global_batch",
            -1,
        )
    )
    for x in rows
) == list(range(16))

print(
    "CIRCULAR_SHIFT_MATCHED_CONTROL=PASS"
)
PYAUDIT

AUDIT_RC=$?

echo "C_SMOKE_AUDIT_RC=$AUDIT_RC" \
    | tee -a "$STATUS"

if [[ "$AUDIT_RC" -ne 0 ]]; then
    echo "CHAIN_STOP=C_AUDIT" \
        | tee -a "$STATUS"
    exit "$AUDIT_RC"
fi

grep -q \
    'PATCHAWARE_LOSS_CONSUMED_PASS' \
    "$BASE/C_smoke1/train.log"

TRACE_RC=$?

echo "C_SMOKE_LOSS_TRACE_RC=$TRACE_RC" \
    | tee -a "$STATUS"

if [[ "$TRACE_RC" -ne 0 ]]; then
    echo "CHAIN_STOP=C_LOSS_TRACE" \
        | tee -a "$STATUS"
    exit "$TRACE_RC"
fi

echo
echo "===== B SOURCE-PIVOT: 100 STEPS ====="

bash \
  "$PROJECT/recipes/pe_pds/run_pe11792_sourcepivot_pilot100_v2.sh"

B_RC=$?

echo "B100_RC=$B_RC" \
    | tee -a "$STATUS"

if [[ "$B_RC" -ne 0 ]]; then
    echo "CHAIN_STOP=B100" \
        | tee -a "$STATUS"
    exit "$B_RC"
fi

grep -q \
    'PATCHAWARE_LOSS_CONSUMED_PASS' \
    "$BASE/B_sourcepivot/train.log"

B_TRACE=$?

echo "B100_LOSS_TRACE_RC=$B_TRACE" \
    | tee -a "$STATUS"

if [[ "$B_TRACE" -ne 0 ]]; then
    echo "CHAIN_STOP=B100_TRACE" \
        | tee -a "$STATUS"
    exit "$B_TRACE"
fi

echo
echo "===== C SHIFT CONTROL: 100 STEPS ====="

bash \
  "$PROJECT/recipes/pe_pds/run_pe11792_shiftctrl_pilot100_v2.sh"

C_RC=$?

echo "C100_RC=$C_RC" \
    | tee -a "$STATUS"

if [[ "$C_RC" -ne 0 ]]; then
    echo "CHAIN_STOP=C100" \
        | tee -a "$STATUS"
    exit "$C_RC"
fi

grep -q \
    'PATCHAWARE_LOSS_CONSUMED_PASS' \
    "$BASE/C_shiftctrl/train.log"

C_TRACE=$?

echo "C100_LOSS_TRACE_RC=$C_TRACE" \
    | tee -a "$STATUS"

if [[ "$C_TRACE" -ne 0 ]]; then
    echo "CHAIN_STOP=C100_TRACE" \
        | tee -a "$STATUS"
    exit "$C_TRACE"
fi

echo \
  "SOURCEPIVOT_CAUSAL_PILOT100_TRAINING=PASS" \
  | tee -a "$STATUS"

exit 0
