#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1


EXP="mtpatcher_v3_full6565_20260823"

SRC_TRAINER="$ROOT/scripts/mtpatcher_v4/train_opd_reversekl_torchnpu.py"
DST_TRAINER="$ROOT/scripts/mtpatcher_v4/train_opd_reversekl_torchnpu_v2.py"

SRC_MASTER="$ROOT/scripts/mtpatcher_v4/run_opd_reversekl_torchnpu_oneclick.sh"
DST_MASTER="$ROOT/scripts/mtpatcher_v4/run_opd_reversekl_torchnpu_v2_oneclick.sh"

NAME="opd_torchnpu_rkl_pe3732_v2"

LOG="$LOG_ROOT/$EXP/${NAME}.log"


echo "======================================================================"
echo "MT-PATCHER — FIX + RUN REVERSE-KL OPD V2"
date
echo "======================================================================"


###############################################################################
# STAGE 1 — VERIFY FAILED V1 SOURCES
###############################################################################

echo
echo "===== STAGE 1/6: VERIFY V1 SOURCE ====="

test -f "$SRC_TRAINER"
test -f "$SRC_MASTER"

echo "SRC_TRAINER=$SRC_TRAINER"
echo "SRC_MASTER=$SRC_MASTER"

echo "RKL_V1_SOURCE_FOUND"


###############################################################################
# STAGE 2 — FIX THE EXACT STALE t_prob CLEANUP REFERENCE
###############################################################################

echo
echo "===== STAGE 2/6: BUILD FIXED RKL V2 TRAINER ====="

export SRC_TRAINER
export DST_TRAINER

python - <<'PY'
import ast
import os
import re
from pathlib import Path


src = Path(os.environ["SRC_TRAINER"])
dst = Path(os.environ["DST_TRAINER"])


text = src.read_text(
    encoding="utf-8"
)

lines = text.splitlines()


print("SOURCE =", src)
print("TARGET =", dst)


# ======================================================================
# A. Verify that the actual Reverse-KL objective is already correct.
# ======================================================================

required = [
    "s_prob = s_logp.exp()",
    "s_logp",
    "t_logp",
    "token_kl",
    "student.module.generate",
    "clip_grad_norm_",
    "DistributedSampler",
]


for marker in required:
    if marker not in text:
        raise RuntimeError(
            f"RKL v1 trainer missing expected marker: {marker}"
        )


# Verify the expected reverse-KL expression structurally from text.
if not re.search(
    r"""
    token_kl\s*=\s*\(
        .*?
        s_prob
        .*?
        s_logp
        .*?
        -\s*t_logp
        .*?
    \)\.sum\(dim=-1\)
    """,
    text,
    flags=re.VERBOSE | re.DOTALL,
):
    raise RuntimeError(
        "Could not verify KL(Student || Teacher) objective"
    )


print("REVERSE_KL_OBJECTIVE_ALREADY_CORRECT")


# ======================================================================
# B. Locate EXACT standalone:
#
#       t_prob,
#
# We expect this to be inside the cleanup `del (...)` block.
# We deliberately do NOT globally replace every textual occurrence.
# ======================================================================

matches = [
    i
    for i, line in enumerate(lines)
    if re.match(
        r"^\s*t_prob,\s*$",
        line,
    )
]


print(
    "STANDALONE_TPROB_LINES =",
    [i + 1 for i in matches],
)


if len(matches) != 1:

    for i in matches:
        lo = max(0, i - 8)
        hi = min(len(lines), i + 9)

        print()
        print(
            f"CONTEXT AROUND LINE {i + 1}"
        )

        for j in range(lo, hi):
            print(
                f"{j + 1:5d}: {lines[j]}"
            )

    raise RuntimeError(
        "Expected exactly one standalone stale 't_prob,' "
        f"but found {len(matches)}"
    )


idx = matches[0]


# ======================================================================
# C. Make sure it really belongs to a `del (...)` cleanup block.
# ======================================================================

lo = max(0, idx - 30)

del_anchor = None


for j in range(idx - 1, lo - 1, -1):

    stripped = lines[j].strip()

    if (
        stripped.startswith("del (")
        or stripped.startswith("del(")
    ):
        del_anchor = j
        break


if del_anchor is None:

    print()
    print("UNSAFE PATCH CONTEXT:")

    for j in range(
        max(0, idx - 15),
        min(len(lines), idx + 15),
    ):
        print(
            f"{j + 1:5d}: {lines[j]}"
        )

    raise RuntimeError(
        "The stale t_prob reference was not verified "
        "to be inside a del(...) cleanup block"
    )


print(
    f"DEL_BLOCK_START_LINE={del_anchor + 1}"
)

print(
    f"STALE_TPROB_LINE={idx + 1}: {lines[idx].strip()}"
)


# ======================================================================
# D. Replace only the cleanup variable:
#
#       t_prob,
#
# ->
#
#       s_prob,
#
# because s_prob is now the large probability tensor that actually
# needs cleanup after reverse-KL.
# ======================================================================

indent = (
    lines[idx]
    [:len(lines[idx]) - len(lines[idx].lstrip())]
)

lines[idx] = indent + "s_prob,"


print(
    f"FIXED_LINE={idx + 1}: {lines[idx].strip()}"
)


text = "\n".join(lines) + "\n"


# ======================================================================
# E. Clean stale comment text.
# This has no numerical effect, but keeps the experimental source honest.
# ======================================================================

text = text.replace(
    "# Exact full-vocabulary forward KL:",
    "# Exact full-vocabulary reverse KL:",
)


# ======================================================================
# F. Python AST audit:
#
# There must be ZERO executable references to t_prob after the fix.
# Comments do not count.
# ======================================================================

tree = ast.parse(text)


t_prob_nodes = [
    node
    for node in ast.walk(tree)
    if isinstance(node, ast.Name)
    and node.id == "t_prob"
]


s_prob_nodes = [
    node
    for node in ast.walk(tree)
    if isinstance(node, ast.Name)
    and node.id == "s_prob"
]


print()
print(
    "EXECUTABLE_t_prob_REFERENCES =",
    len(t_prob_nodes),
)

print(
    "EXECUTABLE_s_prob_REFERENCES =",
    len(s_prob_nodes),
)


if t_prob_nodes:

    details = [
        (
            getattr(node, "lineno", None),
            type(node.ctx).__name__,
        )
        for node in t_prob_nodes
    ]

    raise RuntimeError(
        f"Executable t_prob references still remain: {details}"
    )


if len(s_prob_nodes) < 2:
    raise RuntimeError(
        "Unexpectedly few executable s_prob references"
    )


# ======================================================================
# G. Final mathematical/source checks.
# ======================================================================

final_checks = {
    "student rollout":
        "student.module.generate" in text,

    "student probability":
        "s_prob = s_logp.exp()" in text,

    "reverse KL":
        "s_logp" in text
        and "- t_logp" in text,

    "old teacher probability assignment absent":
        "t_prob = t_logp.exp()" not in text,

    "gradient clipping":
        "clip_grad_norm_" in text,

    "finite loss check":
        "torch.isfinite(loss)" in text,

    "distributed sampler":
        "DistributedSampler" in text,
}


print()
print("FINAL TRAINER CHECKS")


for name, value in final_checks.items():

    print(
        f"{name:40s} = {value}"
    )


if not all(final_checks.values()):

    raise RuntimeError(
        "Fixed RKL trainer verification failed"
    )


dst.write_text(
    text,
    encoding="utf-8",
)


print()
print("RKL_V2_TRAINER_BUILD_PASS")
PY


###############################################################################
# STAGE 3 — COMPILE BEFORE DOING ANYTHING ELSE
###############################################################################

echo
echo "===== STAGE 3/6: COMPILE TRAINER ====="

python -m py_compile "$DST_TRAINER"

echo "RKL_V2_TRAINER_COMPILE_PASS"


###############################################################################
# STAGE 4 — BUILD FRESH V2 MASTER
###############################################################################

echo
echo "===== STAGE 4/6: BUILD V2 MASTER ====="

export SRC_MASTER
export DST_MASTER

python - <<'PY'
import os
from pathlib import Path


src = Path(os.environ["SRC_MASTER"])
dst = Path(os.environ["DST_MASTER"])


text = src.read_text(
    encoding="utf-8"
)


required = [
    "opd_torchnpu_rkl_pe3732_v1",
    "train_opd_reversekl_torchnpu.py",
    "--nproc_per_node=16",
    "pe_k1_clean3732.jsonl",
]


for marker in required:

    if marker not in text:

        raise RuntimeError(
            f"RKL v1 master missing expected marker: {marker}"
        )


# Fresh run/output identity.
text = text.replace(
    "opd_torchnpu_rkl_pe3732_v1",
    "opd_torchnpu_rkl_pe3732_v2",
)


# Point to fixed trainer.
text = text.replace(
    "train_opd_reversekl_torchnpu.py",
    "train_opd_reversekl_torchnpu_v2.py",
)


# Fresh rendezvous port.
text = text.replace(
    "--master_port=29635",
    "--master_port=29636",
)


checks = {
    "v2 run name":
        "opd_torchnpu_rkl_pe3732_v2"
        in text,

    "v2 trainer":
        "train_opd_reversekl_torchnpu_v2.py"
        in text,

    "old trainer absent":
        "train_opd_reversekl_torchnpu.py"
        not in text,

    "16 NPU":
        "--nproc_per_node=16"
        in text,

    "new port":
        "--master_port=29636"
        in text,

    "PE 3732":
        "pe_k1_clean3732.jsonl"
        in text,
}


print("MASTER CHECKS")


for name, value in checks.items():

    print(
        f"{name:30s} = {value}"
    )


if not all(checks.values()):

    raise RuntimeError(
        "RKL v2 master verification failed"
    )


dst.write_text(
    text,
    encoding="utf-8",
)

dst.chmod(0o755)


print()
print("RKL_V2_MASTER_BUILD_PASS")
PY


bash -n "$DST_MASTER"

echo "RKL_V2_MASTER_COMPILE_PASS"


###############################################################################
# STAGE 5 — SCIENTIFIC + CLEANUP AUDIT
###############################################################################

echo
echo "===== STAGE 5/6: FINAL AUDIT ====="

echo
echo "--- Reverse-KL objective ---"

grep -n -A18 -B8 \
'D_KL(P_student || P_teacher)' \
"$DST_TRAINER" \
| head -80


echo
echo "--- Probability tensor cleanup ---"

grep -n -A25 -B5 \
'del (' \
"$DST_TRAINER" \
| tail -60


echo
echo "--- Any remaining textual t_prob references ---"

grep -n '\bt_prob\b' \
"$DST_TRAINER" \
|| true


echo
echo "--- Run configuration ---"

grep -nE \
'NAME=|TRAINER=|nproc_per_node|master_port|pe_k1_clean3732|REVERSE-KL|RKL' \
"$DST_MASTER" \
| head -100


echo
echo "RKL_V2_FINAL_AUDIT_PASS"


###############################################################################
# STAGE 6 — MAKE SURE OLD FAILED RKL IS GONE, THEN LAUNCH FRESH V2
###############################################################################

echo
echo "===== STAGE 6/6: PROCESS CHECK + LAUNCH ====="


RUNNING="$(
    pgrep -af \
    'train_opd_reversekl_torchnpu.py|opd_torchnpu_rkl_pe3732_v1' \
    || true
)"


if [[ -n "$RUNNING" ]]; then

    echo "WARNING: old RKL-related process text still exists:"
    echo "$RUNNING"
    echo
    echo "The new run uses a different trainer, run directory, and port."

else

    echo "No old failed RKL v1 process remains."

fi


mkdir -p "$LOG_ROOT/$EXP"


nohup setsid bash "$DST_MASTER" \
    > "$LOG" 2>&1 < /dev/null &


PID=$!


echo
echo "======================================================================"
echo "RKL_V2_STARTED"
echo "PID=$PID"
echo "LOG=$LOG"
echo "TRAINER=$DST_TRAINER"
echo "MASTER=$DST_MASTER"
echo "======================================================================"

