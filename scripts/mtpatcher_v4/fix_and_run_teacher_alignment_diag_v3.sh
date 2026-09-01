#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

SRC_DIAG="$ROOT/scripts/mtpatcher_v4/diag_pgrkl_teacher_alignment_v2_torchnpu.py"

DST_DIAG="$ROOT/scripts/mtpatcher_v4/diag_pgrkl_teacher_alignment_v3_torchnpu.py"

MASTER_SRC="$ROOT/scripts/mtpatcher_v4/run_opd_pgrkl_k1_audited_v2_oneclick.sh"

MASTER="$ROOT/scripts/mtpatcher_v4/run_pgrkl_teacher_alignment_diag_v3.sh"

NAME="opd_teacher_alignment_diag_208_v3"

LOG="$LOG_ROOT/$EXP/${NAME}.log"


echo "======================================================================"
echo "PG-RKL MATHEMATICAL CONSISTENCY DIAGNOSTIC V3"
echo "208 examples / ZERO optimizer updates"
date
echo "======================================================================"


###############################################################################
# STAGE 1 — PATCH THE ALREADY-SUCCESSFUL V2 DIAGNOSTIC
###############################################################################

echo
echo "===== STAGE 1/5: PATCH DIAGNOSTIC TRAINER ====="

test -f "$SRC_DIAG"

python -m py_compile "$SRC_DIAG"

export SRC_DIAG
export DST_DIAG

python - <<'PY'
import ast
import os
from pathlib import Path


src = Path(os.environ["SRC_DIAG"])
dst = Path(os.environ["DST_DIAG"])

text = src.read_text(
    encoding="utf-8"
)


marker = (
    "# PG-RKL MATHEMATICAL CONSISTENCY DIAGNOSTIC V2"
)


if marker not in text:

    raise RuntimeError(
        "Could not find V2 diagnostic block"
    )


prefix, diag = text.split(
    marker,
    1,
)


###############################################################################
# 1. Stop using global_step to initialize diagnostic state.
#
# Because we're going to skip optimizer/update bookkeeping completely,
# global_step may stay at 0.
###############################################################################

old = "if global_step == 0:"

pos = diag.find(old)


if pos < 0:

    raise RuntimeError(
        "Could not find diagnostic initialization guard"
    )


replacement = (
    "if '_diag_batch_index' not in locals():\n"
    "\n"
    "                _diag_batch_index = 0"
)


diag = (
    diag[:pos]
    + replacement
    + diag[pos + len(old):]
)


###############################################################################
# 2. Replace final global_step==12 trigger with our own independent counter.
###############################################################################

old_final = "if global_step == 12:"


if old_final not in diag:

    raise RuntimeError(
        "Could not find V2 13-batch finish guard"
    )


new_final = (
    "_diag_batch_index += 1\n"
    "\n"
    "            if _diag_batch_index >= 13:"
)


diag = diag.replace(
    old_final,
    new_final,
    1,
)


###############################################################################
# 3. Completely skip optimizer/backward/scheduler on diagnostic batches.
#
# Insert `continue` immediately before the original optimizer.zero_grad().
#
# Final batch returns before reaching continue.
###############################################################################

optimizer_marker = "optimizer.zero_grad("

optimizer_pos = diag.find(
    optimizer_marker
)


if optimizer_pos < 0:

    raise RuntimeError(
        "Could not find optimizer.zero_grad after diagnostic"
    )


# Determine indentation of optimizer.zero_grad line.
before_optimizer = diag[:optimizer_pos]

line_start = before_optimizer.rfind("\n") + 1

optimizer_indent = diag[
    line_start:optimizer_pos
]


if optimizer_indent.strip():

    raise RuntimeError(
        "Unexpected text before optimizer.zero_grad"
    )


continue_text = (
    optimizer_indent
    + "# Diagnostic mode: absolutely no backward / optimizer / scheduler.\n"
    + optimizer_indent
    + "continue\n"
    + "\n"
)


diag = (
    diag[:line_start]
    + continue_text
    + diag[line_start:]
)


###############################################################################
# 4. Remove misleading zero-loss statement.
#
# It is unnecessary now because optimizer path is unreachable.
###############################################################################

diag = diag.replace(
    "loss = loss * 0.0",
    (
        "# loss intentionally left untouched; "
        "optimizer path is skipped below"
    ),
    1,
)


###############################################################################
# Reassemble.
###############################################################################

result = (
    prefix
    + marker
    + diag
)


###############################################################################
# Syntax validation.
###############################################################################

tree = ast.parse(
    result
)


###############################################################################
# Scientific/static verification.
###############################################################################

checks = {
    "custom batch counter":
        "_diag_batch_index = 0"
        in result,

    "13 batches":
        "_diag_batch_index >= 13"
        in result,

    "five offsets":
        "_diag_offsets = (-2, -1, 0, 1, 2)"
        in result,

    "EOS/PAD mask":
        "_diag_eos_before"
        in result,

    "sampled k1":
        "_diag_sample"
        in result,

    "exact reverse KL":
        "_diag_exact"
        in result,

    "valid token result":
        "VALID sampled="
        in result,

    "all token result":
        "ALL sampled="
        in result,

    "teacher offset result":
        "BEST_TEACHER_OFFSET"
        in result,

    "diagnostic final":
        "TEACHER_ALIGNMENT_DIAGNOSTIC_FINAL"
        in result,

    "optimizer bypass":
        "continue"
        in result,

    "old finish condition removed":
        "if global_step == 12:"
        not in result,
}


print("V3 DIAGNOSTIC CHECK")


for name, ok in checks.items():

    print(
        f"{name:34s} = {ok}"
    )


if not all(
    checks.values()
):

    raise RuntimeError(
        "V3 diagnostic scientific verification failed"
    )


###############################################################################
# Ensure continue occurs BEFORE optimizer.zero_grad in diagnostic section.
###############################################################################

diag_start = result.index(
    marker
)

continue_pos = result.index(
    "continue",
    diag_start,
)

optimizer_pos = result.index(
    "optimizer.zero_grad",
    diag_start,
)


print()
print(
    "CONTINUE_POS =",
    continue_pos,
)

print(
    "OPTIMIZER_POS =",
    optimizer_pos,
)


if continue_pos >= optimizer_pos:

    raise RuntimeError(
        "Diagnostic continue does not precede optimizer"
    )


dst.write_text(
    result,
    encoding="utf-8",
)


print()
print(
    "TEACHER_ALIGNMENT_DIAG_V3_PATCH_PASS"
)
PY


python -m py_compile "$DST_DIAG"

echo "DIAGNOSTIC_V3_COMPILE_PASS"


###############################################################################
# STAGE 2 — BUILD MASTER WITHOUT ASSUMING ANY EVAL BOUNDARY VARIABLE
#
# We locate the ACTUAL torch.distributed.run command and truncate the master
# immediately after that shell command.
###############################################################################

echo
echo "===== STAGE 2/5: BUILD ROBUST SHORT MASTER ====="

test -f "$MASTER_SRC"

export MASTER_SRC
export MASTER
export NAME

python - <<'PY'
import os
import re
from pathlib import Path


src = Path(os.environ["MASTER_SRC"])
dst = Path(os.environ["MASTER"])

name = os.environ["NAME"]


text = src.read_text(
    encoding="utf-8"
)


###############################################################################
# Rewrite top-level run identity.
###############################################################################

name_matches = re.findall(
    r"(?m)^NAME=.*$",
    text,
)


if len(name_matches) != 1:

    raise RuntimeError(
        "Expected exactly one top-level NAME: "
        + str(name_matches)
    )


text = re.sub(
    r"(?m)^NAME=.*$",
    f'NAME="{name}"',
    text,
    count=1,
)


###############################################################################
# Rewrite trainer.
###############################################################################

trainer_matches = re.findall(
    r"(?m)^TRAINER=.*$",
    text,
)


if len(trainer_matches) != 1:

    raise RuntimeError(
        "Expected exactly one top-level TRAINER: "
        + str(trainer_matches)
    )


text = re.sub(
    r"(?m)^TRAINER=.*$",
    (
        'TRAINER="$ROOT/scripts/mtpatcher_v4/'
        'diag_pgrkl_teacher_alignment_v3_torchnpu.py"'
    ),
    text,
    count=1,
)


###############################################################################
# Fresh HCCL port.
###############################################################################

ports = re.findall(
    r"--master_port=\d+",
    text,
)


if len(ports) != 1:

    raise RuntimeError(
        "Expected exactly one master_port: "
        + str(ports)
    )


text = re.sub(
    r"--master_port=\d+",
    "--master_port=29647",
    text,
    count=1,
)


###############################################################################
# Find the distributed launch command itself.
#
# Example:
#
# "$PYTHON" -m torch.distributed.run \
#     --nproc_per_node=16 \
#     ...
#     "$TRAINER" ...
#
# We do NOT depend on OPD_EPOCH3_MODEL or evaluation-stage wording.
###############################################################################

lines = text.splitlines()


launch_candidates = [
    i
    for i, line in enumerate(lines)
    if (
        "torch.distributed.run"
        in line
        or re.search(
            r"(^|\s)torchrun(\s|$)",
            line,
        )
    )
]


print(
    "DISTRIBUTED_LAUNCH_CANDIDATES =",
    [
        i + 1
        for i in launch_candidates
    ],
)


if len(
    launch_candidates
) != 1:

    raise RuntimeError(
        "Could not uniquely locate distributed launch command"
    )


start = launch_candidates[0]


###############################################################################
# Follow shell backslash continuation until command ends.
###############################################################################

end = start


while (
    end < len(lines) - 1
    and lines[end].rstrip().endswith("\\")
):

    end += 1


print(
    "DISTRIBUTED_COMMAND_LINES =",
    start + 1,
    "-",
    end + 1,
)


###############################################################################
# Sanity check the command includes our training controls.
###############################################################################

command_text = "\n".join(
    lines[start:end + 1]
)


if "--nproc_per_node=16" not in command_text:

    raise RuntimeError(
        "Distributed command no longer uses 16 ranks"
    )


if "$TRAINER" not in command_text:

    raise RuntimeError(
        "Distributed command does not invoke TRAINER variable"
    )


###############################################################################
# Truncate immediately after distributed diagnostic process returns.
###############################################################################

short_lines = lines[:end + 1]


short_lines.extend(
    [
        "",
        'echo',
        'echo "======================================================================"',
        'echo "TEACHER_ALIGNMENT_DIAGNOSTIC_LAUNCHER_PASS"',
        'date',
        'echo "======================================================================"',
        "",
    ]
)


short_text = "\n".join(
    short_lines
) + "\n"


###############################################################################
# Verify evaluation/checkpoint pipeline is gone.
###############################################################################

checks = {
    "diagnostic trainer":
        "diag_pgrkl_teacher_alignment_v3_torchnpu.py"
        in short_text,

    "16 ranks":
        "--nproc_per_node=16"
        in short_text,

    "PE3732":
        "pe_k1_clean3732.jsonl"
        in short_text,

    "fresh port":
        "--master_port=29647"
        in short_text,

    "no evaluation":
        "OPD_TORCHNPU_EVAL_PASS"
        not in short_text,

    "no final BLEU summary":
        "FINAL SUMMARY"
        not in short_text,

    "no epoch3 model dependency":
        "OPD_EPOCH3_MODEL"
        not in short_text,
}


print()
print("SHORT MASTER CHECK")


for name, ok in checks.items():

    print(
        f"{name:34s} = {ok}"
    )


if not all(
    checks.values()
):

    raise RuntimeError(
        "Short master verification failed"
    )


dst.write_text(
    short_text,
    encoding="utf-8",
)

dst.chmod(
    0o755
)


print()
print(
    "ROBUST_SHORT_MASTER_BUILD_PASS"
)
PY


bash -n "$MASTER"

echo "ROBUST_SHORT_MASTER_SYNTAX_PASS"


###############################################################################
# STAGE 3 — FINAL AUDIT
###############################################################################

echo
echo "===== STAGE 3/5: FINAL AUDIT ====="


echo
echo "--- diagnostic mechanics ---"

grep -nE \
'_diag_batch_index|VALID sampled=|ALL sampled=|BEST_TEACHER_OFFSET|TEACHER_ALIGNMENT_DIAGNOSTIC_FINAL|continue|optimizer.zero_grad' \
"$DST_DIAG" \
| tail -n 80


echo
echo "--- launcher ---"

grep -nE \
'^NAME=|^TRAINER=|torch.distributed.run|nproc_per_node|master_port|pe_k1_clean3732' \
"$MASTER" \
| head -100


echo
echo "FINAL_DIAGNOSTIC_AUDIT_PASS"


###############################################################################
# STAGE 4 — LAUNCH
###############################################################################

echo
echo "===== STAGE 4/5: LAUNCH ====="


RUNNING="$(
    pgrep -af \
    'diag_pgrkl_teacher_alignment_v3_torchnpu' \
    || true
)"


if [[ -n "$RUNNING" ]]; then

    echo "Existing diagnostic process:"
    echo "$RUNNING"

    echo "Duplicate launch skipped."

else

    mkdir -p \
    "$LOG_ROOT/$EXP"


    nohup setsid bash "$MASTER" \
        > "$LOG" 2>&1 < /dev/null &


    PID=$!


    echo
    echo "======================================================================"
    echo "TEACHER ALIGNMENT DIAGNOSTIC V3 STARTED"
    echo "PID=$PID"
    echo "LOG=$LOG"
    echo "======================================================================"

fi


###############################################################################
# STAGE 5 — WAIT FOR FINAL MATHEMATICAL RESULT
###############################################################################

echo
echo "===== STAGE 5/5: WAIT FOR RESULT ====="


for _i in $(seq 1 90); do

    if grep -q \
        'TEACHER_ALIGNMENT_DIAGNOSTIC_FINAL' \
        "$LOG" 2>/dev/null; then

        break

    fi


    if grep -qE \
        '\[rank[0-9]+\]: Traceback|NameError:|TypeError:|ValueError:' \
        "$LOG" 2>/dev/null; then

        break

    fi


    sleep 5

done


echo
echo "======================================================================"
echo "MATHEMATICAL DIAGNOSTIC RESULT"
echo "======================================================================"


grep -E \
'PG_ALIGNMENT|DIAG_VALID_TOKEN_MASK|TEACHER_ALIGNMENT|offset=|BEST_|Traceback|RuntimeError:|NameError:|TypeError:|ValueError:' \
"$LOG" \
| tail -n 160

