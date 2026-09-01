#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

SRC_TRAINER="$ROOT/scripts/mtpatcher_v4/train_opd_pgrkl_k1_audited_torchnpu.py"
DST_TRAINER="$ROOT/scripts/mtpatcher_v4/train_opd_pgrkl_k1_audited_v2_torchnpu.py"

SRC_MASTER="$ROOT/scripts/mtpatcher_v4/run_opd_pgrkl_k1_audited_oneclick.sh"
DST_MASTER="$ROOT/scripts/mtpatcher_v4/run_opd_pgrkl_k1_audited_v2_oneclick.sh"

OLD_NAME="opd_torchnpu_pgrkl_k1_audited_pe3732_v1"
NEW_NAME="opd_torchnpu_pgrkl_k1_audited_pe3732_v2"

LOG="$LOG_ROOT/$EXP/${NEW_NAME}.log"


echo "======================================================================"
echo "FIX AUDIT PRINT GUARD + RUN PG-RKL AUDITED V2"
date
echo "======================================================================"


###############################################################################
# STAGE 1 — PATCH ONLY THE REAL ROOT CAUSE
###############################################################################

echo
echo "===== STAGE 1/5: PATCH TRAINER ====="

test -f "$SRC_TRAINER"

export SRC_TRAINER
export DST_TRAINER

python - <<'PY'
import ast
import os
from pathlib import Path


src = Path(os.environ["SRC_TRAINER"])
dst = Path(os.environ["DST_TRAINER"])

text = src.read_text(
    encoding="utf-8"
)


old = "if rank == 0 and step == 0:"
new = "if rank == 0 and global_step == 0:"


count = text.count(old)

print(
    "BROKEN_GUARD_COUNT =",
    count,
)


if count != 1:
    raise RuntimeError(
        f"Expected exactly one broken audit guard; found {count}"
    )


text = text.replace(
    old,
    new,
    1,
)


# ----------------------------------------------------------------------
# Clean the stale human-readable objective label too.
# This does NOT alter the optimization.
# ----------------------------------------------------------------------

text = text.replace(
    "OBJECTIVE = D_KL(Teacher || Student)",
    "OBJECTIVE = sampled-token k1 PG-RKL [behavior-matched, alignment-audited]",
)

text = text.replace(
    '"D_KL(Teacher || Student)"',
    '"sampled-token k1 PG-RKL [behavior-matched, alignment-audited]"',
)


# ----------------------------------------------------------------------
# Syntax verification.
# ----------------------------------------------------------------------

tree = ast.parse(text)


# ----------------------------------------------------------------------
# Verify there are no remaining LOADS of an undefined variable named step.
#
# method calls such as optimizer.step() are ast.Attribute and do not count.
# ----------------------------------------------------------------------

step_loads = []

global_step_loads = []


for node in ast.walk(tree):

    if (
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
    ):

        if node.id == "step":
            step_loads.append(
                (
                    node.lineno,
                    node.col_offset,
                )
            )

        if node.id == "global_step":
            global_step_loads.append(
                (
                    node.lineno,
                    node.col_offset,
                )
            )


print(
    "UNQUALIFIED_STEP_LOADS =",
    step_loads,
)

print(
    "GLOBAL_STEP_LOAD_COUNT =",
    len(global_step_loads),
)


if step_loads:
    raise RuntimeError(
        "There are still executable references to variable `step`: "
        + str(step_loads)
    )


if not global_step_loads:
    raise RuntimeError(
        "global_step is unexpectedly absent from trainer"
    )


# ----------------------------------------------------------------------
# Scientific mechanism must remain intact.
# ----------------------------------------------------------------------

checks = {
    "behavior temperature 1":
        "temperature=1.0" in text,

    "behavior top_p 1":
        "top_p=1.0" in text,

    "behavior top_k 0":
        "top_k=0" in text,

    "generation scores":
        "_pg_behavior_scores" in text,

    "alignment candidates":
        "_pg_candidates" in text,

    "alignment threshold":
        "_pg_alignment_mae > 0.20" in text,

    "alignment success print":
        "PG_ALIGNMENT_AUDIT_PASS" in text,

    "sampled student logp":
        "s_action_logp" in text,

    "sampled teacher logp":
        "t_action_logp" in text,

    "sampled k1":
        "k1 = (" in text,

    "negative k1 reward":
        "distill_reward" in text,

    "PG objective":
        "s_action_logp.float()" in text,

    "gradient clipping":
        "clip_grad_norm_" in text,

    "finite loss":
        "torch.isfinite(loss)" in text,
}


print()
print("SCIENTIFIC MECHANISM CHECK")


for name, ok in checks.items():
    print(
        f"{name:32s} = {ok}"
    )


if not all(checks.values()):
    raise RuntimeError(
        "Scientific mechanism changed unexpectedly"
    )


dst.write_text(
    text,
    encoding="utf-8",
)


print()
print(
    "AUDIT_GUARD_PATCH_PASS"
)
PY


python -m py_compile "$DST_TRAINER"

echo "TRAINER_COMPILE_PASS"


###############################################################################
# STAGE 2 — BUILD FRESH MASTER
###############################################################################

echo
echo "===== STAGE 2/5: BUILD MASTER ====="

test -f "$SRC_MASTER"

export SRC_MASTER
export DST_MASTER
export OLD_NAME
export NEW_NAME

python - <<'PY'
import os
import re
from pathlib import Path


src = Path(os.environ["SRC_MASTER"])
dst = Path(os.environ["DST_MASTER"])

old_name = os.environ["OLD_NAME"]
new_name = os.environ["NEW_NAME"]

text = src.read_text(
    encoding="utf-8"
)


# ----------------------------------------------------------------------
# NAME
# ----------------------------------------------------------------------

matches = re.findall(
    r"(?m)^NAME=.*$",
    text,
)

if len(matches) != 1:
    raise RuntimeError(
        f"Expected one top-level NAME, got {matches}"
    )


text = re.sub(
    r"(?m)^NAME=.*$",
    f'NAME="{new_name}"',
    text,
    count=1,
)


# ----------------------------------------------------------------------
# TRAINER
# ----------------------------------------------------------------------

matches = re.findall(
    r"(?m)^TRAINER=.*$",
    text,
)

if len(matches) != 1:
    raise RuntimeError(
        f"Expected one top-level TRAINER, got {matches}"
    )


text = re.sub(
    r"(?m)^TRAINER=.*$",
    'TRAINER="$ROOT/scripts/mtpatcher_v4/'
    'train_opd_pgrkl_k1_audited_v2_torchnpu.py"',
    text,
    count=1,
)


# ----------------------------------------------------------------------
# Fresh distributed port.
# ----------------------------------------------------------------------

ports = re.findall(
    r"--master_port=\d+",
    text,
)

if len(ports) != 1:
    raise RuntimeError(
        f"Expected one master_port, got {ports}"
    )


text = re.sub(
    r"--master_port=\d+",
    "--master_port=29644",
    text,
    count=1,
)


# Replace inherited run directory references.
text = text.replace(
    old_name,
    new_name,
)


checks = {
    "new name":
        new_name in text,

    "new trainer":
        "train_opd_pgrkl_k1_audited_v2_torchnpu.py"
        in text,

    "16 NPU":
        "--nproc_per_node=16"
        in text,

    "same PE3732":
        "pe_k1_clean3732.jsonl"
        in text,

    "fresh port":
        "--master_port=29644"
        in text,

    "audited result":
        "OPD-PGRKL-K1-AUDITED-PE3732"
        in text,
}


print("MASTER CHECK")


for name, ok in checks.items():
    print(
        f"{name:28s} = {ok}"
    )


if not all(checks.values()):
    raise RuntimeError(
        "Master verification failed"
    )


dst.write_text(
    text,
    encoding="utf-8",
)

dst.chmod(
    0o755
)


print()
print(
    "AUDITED_V2_MASTER_BUILD_PASS"
)
PY


bash -n "$DST_MASTER"

echo "MASTER_SYNTAX_PASS"


###############################################################################
# STAGE 3 — FINAL STATIC AUDIT
###############################################################################

echo
echo "===== STAGE 3/5: FINAL STATIC AUDIT ====="


echo
echo "--- fixed print guard ---"

grep -n -A8 -B8 \
'PG_ALIGNMENT_AUDIT_PASS' \
"$DST_TRAINER"


echo
echo "--- no invalid `step` variable ---"

python - "$DST_TRAINER" <<'PY'
import ast
import sys
from pathlib import Path

p = Path(sys.argv[1])

tree = ast.parse(
    p.read_text(
        encoding="utf-8"
    )
)

bad = []

for node in ast.walk(tree):

    if (
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == "step"
    ):

        bad.append(
            node.lineno
        )

print(
    "UNQUALIFIED_STEP_LOAD_LINES =",
    bad,
)

if bad:
    raise RuntimeError(
        f"Undefined step references remain: {bad}"
    )

print(
    "NO_UNDEFINED_STEP_PASS"
)
PY


echo
echo "--- master ---"

grep -nE \
'^NAME=|^TRAINER=|nproc_per_node|master_port|AUDITED|pe_k1_clean3732' \
"$DST_MASTER" \
| head -100


echo
echo "FINAL_STATIC_AUDIT_PASS"


###############################################################################
# STAGE 4 — DUPLICATE PROCESS GUARD
###############################################################################

echo
echo "===== STAGE 4/5: PROCESS CHECK ====="

RUNNING="$(
    pgrep -af \
    'train_opd_pgrkl_k1_audited_v2_torchnpu|run_opd_pgrkl_k1_audited_v2' \
    || true
)"


if [[ -n "$RUNNING" ]]; then

    echo "Existing V2 process:"
    echo "$RUNNING"

    echo "DUPLICATE_LAUNCH_BLOCKED"

else

    echo "No active audited V2 process."

fi


###############################################################################
# STAGE 5 — LAUNCH
###############################################################################

echo
echo "===== STAGE 5/5: LAUNCH ====="


if [[ -z "$RUNNING" ]]; then

    mkdir -p \
    "$LOG_ROOT/$EXP"


    nohup setsid bash "$DST_MASTER" \
        > "$LOG" 2>&1 < /dev/null &


    PID=$!


    echo
    echo "======================================================================"
    echo "AUDITED PG-RKL V2 STARTED"
    echo "PID=$PID"
    echo "LOG=$LOG"
    echo "TRAINER=$DST_TRAINER"
    echo "MASTER=$DST_MASTER"
    echo "======================================================================"

fi


###############################################################################
# AUTOMATIC SCIENTIFIC HEALTH CHECK
###############################################################################

sleep 100


echo
echo "======================================================================"
echo "SCIENTIFIC HEALTH CHECK"
echo "======================================================================"


python - "$LOG" <<'PY'
import re
import sys
from pathlib import Path


p = Path(sys.argv[1])

lines = p.read_text(
    errors="replace"
).splitlines()


interesting = []

for line in lines:

    if any(
        marker in line
        for marker in (
            "OBJECTIVE =",
            "TRAIN_ROWS =",
            "LOCAL_STEPS_PER_EPOCH",
            "TOTAL_UPDATES",
            "GLOBAL_BATCH",
            "OPD_TRAINING_START",
            "PG_ALIGNMENT_AUDIT_PASS",
            "PG-RKL ALIGNMENT AUDIT FAILED",
            "local_step=",
            "Traceback",
            "NameError:",
            "RuntimeError:",
        )
    ):

        interesting.append(
            line
        )


for line in interesting[-120:]:
    print(
        line
    )


# Fail the health check loudly if a primary Python exception appeared.
fatal = [
    line
    for line in lines
    if (
        "NameError:" in line
        or "PG-RKL ALIGNMENT AUDIT FAILED" in line
        or (
            "RuntimeError:" in line
            and "repository_manager" not in line
        )
    )
]


if fatal:

    print()
    print(
        "HEALTH_CHECK_FOUND_PRIMARY_ERROR"
    )

    for line in fatal[:30]:
        print(
            line
        )

else:

    print()
    print(
        "FIRST_HEALTH_CHECK_PASS"
    )
PY

