#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1


EXP="mtpatcher_v3_full6565_20260823"

SRC_TRAINER="$ROOT/scripts/mtpatcher_v3/train_opd_forwardkl_torchnpu.py"
DST_TRAINER="$ROOT/scripts/mtpatcher_v4/train_opd_reversekl_torchnpu.py"

SRC_MASTER="$ROOT/scripts/mtpatcher_v3/run_opd_forwardkl_torchnpu_oneclick.sh"
DST_MASTER="$ROOT/scripts/mtpatcher_v4/run_opd_reversekl_torchnpu_oneclick.sh"

RKL_NAME="opd_torchnpu_rkl_pe3732_v1"

LOG="$LOG_ROOT/$EXP/${RKL_NAME}.log"


echo "======================================================================"
echo "MT-PATCHER — BUILD + RUN VERIFIED REVERSE-KL OPD"
date
echo "======================================================================"


###############################################################################
# STAGE 0 — CHECK VERIFIED SOURCE FILES
###############################################################################

echo
echo "===== STAGE 0/6: VERIFY SOURCE FILES ====="

test -f "$SRC_TRAINER"
test -f "$SRC_MASTER"

echo "SRC_TRAINER=$SRC_TRAINER"
echo "SRC_MASTER=$SRC_MASTER"

echo "VERIFIED_SOURCE_FILES_PASS"


###############################################################################
# STAGE 1 — STOP ONLY THE ABANDONED BROKEN V2
###############################################################################

echo
echo "===== STAGE 1/6: CLEAN BROKEN V2 PROCESSES ====="

OLD_PIDS="$(
    pgrep -f \
    'train_pe_opd_reversekl_v2.py|run_pe_opd_reversekl_v2_oneclick.sh' \
    || true
)"

if [[ -n "$OLD_PIDS" ]]; then
    echo "Found broken-v2 processes:"
    echo "$OLD_PIDS"

    kill $OLD_PIDS || true
    sleep 3
else
    echo "No broken-v2 process is running."
fi

echo "BROKEN_V2_PROCESS_CHECK_PASS"


###############################################################################
# STAGE 2 — BUILD RKL TRAINER FROM THE ALREADY SUCCESSFUL FKL TRAINER
###############################################################################

echo
echo "===== STAGE 2/6: BUILD REVERSE-KL TRAINER ====="

export SRC_TRAINER
export DST_TRAINER

python - <<'PY'
import os
import re
from pathlib import Path


src = Path(os.environ["SRC_TRAINER"])
dst = Path(os.environ["DST_TRAINER"])


print("SOURCE_TRAINER =", src)
print("TARGET_TRAINER =", dst)


text = src.read_text(encoding="utf-8")
lines = text.splitlines()


# ----------------------------------------------------------------------
# Verify that this is the trainer which already completed the FKL run.
# ----------------------------------------------------------------------

required = [
    "student.module.generate",
    "DistributedSampler",
    "t_logp",
    "t_prob",
    "s_logp",
    "token_kl",
    "clip_grad_norm_",
    "ON_POLICY",
]

for marker in required:
    if marker not in text:
        raise RuntimeError(
            f"Verified FKL trainer is missing marker: {marker}"
        )

print("FKL_TRAINER_IDENTITY_PASS")


# ----------------------------------------------------------------------
# Utility: exact unique line match.
# ----------------------------------------------------------------------

def unique_index(predicate, description):

    indices = [
        i
        for i, line in enumerate(lines)
        if predicate(line)
    ]

    if len(indices) != 1:
        print()
        print(
            f"DEBUG: {description} candidates = {len(indices)}"
        )

        for i in indices:
            print(
                f"{i + 1}: {lines[i]}"
            )

        raise RuntimeError(
            f"{description}: expected exactly one match, "
            f"found {len(indices)}"
        )

    return indices[0]


def expression_end(start):

    """
    Find the final line of a multiline Python expression by balancing
    parentheses/brackets/braces.
    """

    balance = 0
    seen_open = False

    for i in range(start, len(lines)):

        # These known expressions contain no relevant bracket-like
        # characters in string literals, so simple balancing is safe here.
        code = lines[i].split("#", 1)[0]

        opens = (
            code.count("(")
            + code.count("[")
            + code.count("{")
        )

        closes = (
            code.count(")")
            + code.count("]")
            + code.count("}")
        )

        if opens > 0:
            seen_open = True

        balance += opens
        balance -= closes

        if seen_open and balance == 0:
            return i

    raise RuntimeError(
        f"Could not find expression end starting at line {start + 1}"
    )


# ----------------------------------------------------------------------
# Anchor 1:
#
#   t_prob = t_logp.exp()
#
# This is unique to FKL.
# ----------------------------------------------------------------------

tprob_idx = unique_index(
    lambda line:
        re.match(
            r"^\s*t_prob\s*=\s*t_logp\.exp\(\)\s*$",
            line,
        )
        is not None,
    "t_prob assignment",
)

print(
    f"FOUND_TPROB_LINE={tprob_idx + 1}:",
    lines[tprob_idx].strip(),
)


# ----------------------------------------------------------------------
# Anchor 2:
#
# Find the student log-softmax AFTER t_prob.
# ----------------------------------------------------------------------

slog_candidates = [
    i
    for i in range(tprob_idx + 1, len(lines))
    if re.match(
        r"^\s*s_logp\s*=\s*F\.log_softmax\(",
        lines[i],
    )
]


if not slog_candidates:
    raise RuntimeError(
        "Could not find s_logp after t_prob"
    )


slog_idx = slog_candidates[0]
slog_end = expression_end(slog_idx)


print(
    f"FOUND_SLOG_LINE={slog_idx + 1}:",
    lines[slog_idx].strip(),
)


# ----------------------------------------------------------------------
# Anchor 3:
#
# STRICT left-hand-side match:
#
#   token_kl = ...
#
# This deliberately does NOT match:
#
#   loss = token_kl.mean()
# ----------------------------------------------------------------------

token_candidates = [
    i
    for i in range(slog_end + 1, len(lines))
    if re.match(
        r"^\s*token_kl\s*=",
        lines[i],
    )
]


if not token_candidates:
    raise RuntimeError(
        "Could not find token_kl assignment after s_logp"
    )


token_idx = token_candidates[0]
token_end = expression_end(token_idx)


print(
    f"FOUND_TOKEN_KL_LINE={token_idx + 1}:",
    lines[token_idx].strip(),
)


# ----------------------------------------------------------------------
# Anchor 4:
#
# Ensure this token_kl assignment belongs to:
#
#   loss = token_kl.mean()
#
# ----------------------------------------------------------------------

loss_candidates = [
    i
    for i in range(token_end + 1, min(token_end + 20, len(lines)))
    if re.match(
        r"^\s*loss\s*=\s*token_kl\.mean\(\)\s*$",
        lines[i],
    )
]


if len(loss_candidates) != 1:
    raise RuntimeError(
        "The located token_kl block is not followed by the expected "
        "'loss = token_kl.mean()'"
    )


loss_idx = loss_candidates[0]


print(
    f"FOUND_LOSS_LINE={loss_idx + 1}:",
    lines[loss_idx].strip(),
)


# ----------------------------------------------------------------------
# Save indentation from original code.
# ----------------------------------------------------------------------

indent = (
    lines[token_idx]
    [:len(lines[token_idx]) - len(lines[token_idx].lstrip())]
)


# ----------------------------------------------------------------------
# STEP A:
# Remove:
#
#   t_prob = t_logp.exp()
# ----------------------------------------------------------------------

del lines[tprob_idx]


# Deleting a previous line shifts all later indices by one.
slog_idx -= 1
slog_end -= 1
token_idx -= 1
token_end -= 1
loss_idx -= 1


# ----------------------------------------------------------------------
# STEP B:
# Add student probability directly after the completed s_logp expression.
#
# s_logp is computed from float32 logits in the verified trainer.
# Therefore:
#
#   s_prob = exp(log_softmax(float32_logits))
#
# avoids the fp16 softmax/log instability from the abandoned v2.
# ----------------------------------------------------------------------

slog_indent = (
    lines[slog_idx]
    [:len(lines[slog_idx]) - len(lines[slog_idx].lstrip())]
)


insert_lines = [
    "",
    slog_indent + "# Student distribution for exact reverse KL.",
    slog_indent + "s_prob = s_logp.exp()",
]


insert_pos = slog_end + 1

lines[
    insert_pos:insert_pos
] = insert_lines


shift = len(insert_lines)

token_idx += shift
token_end += shift
loss_idx += shift


# ----------------------------------------------------------------------
# STEP C:
#
# Replace ONLY:
#
# FKL:
#
#   KL(T || S)
#   = sum P_T (log P_T - log P_S)
#
# with:
#
# RKL:
#
#   KL(S || T)
#   = sum P_S (log P_S - log P_T)
#
# Student rollout, Teacher, data, sampler, LR, schedule, checkpoints,
# evaluation, etc. remain inherited from the verified experiment.
# ----------------------------------------------------------------------

reverse_block = [
    indent + "# Exact full-vocabulary reverse KL:",
    indent + "# D_KL(P_student || P_teacher)",
    indent + "token_kl = (",
    indent + "    s_prob",
    indent + "    * (",
    indent + "        s_logp",
    indent + "        - t_logp",
    indent + "    )",
    indent + ").sum(dim=-1)",
]


lines[
    token_idx:token_end + 1
] = reverse_block


text = "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# Diagnostic labels.
# ----------------------------------------------------------------------

text = text.replace(
    "ON-POLICY FORWARD-KL",
    "ON-POLICY REVERSE-KL",
)

text = text.replace(
    "D_KL(Teacher || Student)",
    "D_KL(Student || Teacher)",
)

text = text.replace(
    "D_KL(P_teacher || P_student)",
    "D_KL(P_student || P_teacher)",
)

text = text.replace(
    "token_mean_forward_kl",
    "token_mean_reverse_kl",
)

text = text.replace(
    "MTPATCHER_V3_TORCHNPU_OPD_TRAINING_PASS",
    "MTPATCHER_V4_TORCHNPU_REVERSEKL_TRAINING_PASS",
)


# ----------------------------------------------------------------------
# Final trainer verification.
# ----------------------------------------------------------------------

verification = {
    "student rollout":
        "student.module.generate" in text,

    "on-policy marker":
        "ON_POLICY" in text,

    "student logprob":
        "s_logp" in text,

    "teacher logprob":
        "t_logp" in text,

    "student probability":
        "s_prob = s_logp.exp()" in text,

    "teacher probability removed":
        "t_prob = t_logp.exp()" not in text,

    "reverse KL marker":
        "D_KL(P_student || P_teacher)" in text,

    "reverse log ratio":
        "- t_logp" in text,

    "gradient clipping":
        "clip_grad_norm_" in text,

    "distributed sampler":
        "DistributedSampler" in text,
}


print()
print("TRAINER VERIFICATION")

for name, value in verification.items():
    print(
        f"{name:30s} = {value}"
    )


if not all(verification.values()):
    raise RuntimeError(
        "Reverse-KL trainer verification failed"
    )


dst.write_text(
    text,
    encoding="utf-8",
)


print()
print("RKL_TRAINER_BUILD_PASS")
PY


###############################################################################
# STAGE 3 — SYNTAX CHECK TRAINER
###############################################################################

echo
echo "===== STAGE 3/6: TRAINER SYNTAX CHECK ====="

python -m py_compile "$DST_TRAINER"

echo "RKL_TRAINER_SYNTAX_PASS"


###############################################################################
# STAGE 4 — BUILD MASTER FROM VERIFIED FKL ONE-CLICK
###############################################################################

echo
echo "===== STAGE 4/6: BUILD RKL ONE-CLICK MASTER ====="

export SRC_MASTER
export DST_MASTER

python - <<'PY'
import os
from pathlib import Path


src = Path(os.environ["SRC_MASTER"])
dst = Path(os.environ["DST_MASTER"])


master = src.read_text(
    encoding="utf-8"
)


required = [
    "opd_torchnpu_fkl_pe3732_v1",
    "train_opd_forwardkl_torchnpu.py",
    "--nproc_per_node=16",
    "pe_k1_clean3732.jsonl",
]


for marker in required:
    if marker not in master:
        raise RuntimeError(
            f"Verified FKL master missing marker: {marker}"
        )


# ----------------------------------------------------------------------
# Experiment run name/output directory.
# ----------------------------------------------------------------------

master = master.replace(
    "opd_torchnpu_fkl_pe3732_v1",
    "opd_torchnpu_rkl_pe3732_v1",
)


# ----------------------------------------------------------------------
# Trainer path.
# ----------------------------------------------------------------------

old_full = (
    "$ROOT/scripts/mtpatcher_v3/"
    "train_opd_forwardkl_torchnpu.py"
)

new_full = (
    "$ROOT/scripts/mtpatcher_v4/"
    "train_opd_reversekl_torchnpu.py"
)


if old_full in master:

    master = master.replace(
        old_full,
        new_full,
    )

else:

    # Robust fallback if the verified shell used a different quoting style.
    master = master.replace(
        "train_opd_forwardkl_torchnpu.py",
        "train_opd_reversekl_torchnpu.py",
    )

    master = master.replace(
        "scripts/mtpatcher_v3/train_opd_reversekl_torchnpu.py",
        "scripts/mtpatcher_v4/train_opd_reversekl_torchnpu.py",
    )


# ----------------------------------------------------------------------
# Separate rendezvous port.
# ----------------------------------------------------------------------

master = master.replace(
    "29631",
    "29635",
)


# ----------------------------------------------------------------------
# Labels used by result summary.
# ----------------------------------------------------------------------

master = master.replace(
    "ON-POLICY FORWARD-KL",
    "ON-POLICY REVERSE-KL",
)

master = master.replace(
    "OPD-FKL-PE3732",
    "OPD-RKL-PE3732",
)

master = master.replace(
    "OPD_FKL_PE3732",
    "OPD_RKL_PE3732",
)

master = master.replace(
    "MTPATCHER_V3_TORCHNPU_OPD_ALL_PASS",
    "MTPATCHER_V4_TORCHNPU_REVERSEKL_ALL_PASS",
)


# ----------------------------------------------------------------------
# Separate summary name if that exact filename occurs.
# ----------------------------------------------------------------------

master = master.replace(
    "opd_torchnpu_final_summary.json",
    "opd_torchnpu_reversekl_final_summary.json",
)


verification = {
    "RKL run name":
        "opd_torchnpu_rkl_pe3732_v1" in master,

    "RKL trainer":
        "train_opd_reversekl_torchnpu.py" in master,

    "no FKL trainer":
        "train_opd_forwardkl_torchnpu.py" not in master,

    "16 NPU":
        "--nproc_per_node=16" in master,

    "PE3732 source":
        "pe_k1_clean3732.jsonl" in master,
}


print("MASTER VERIFICATION")

for name, value in verification.items():
    print(
        f"{name:30s} = {value}"
    )


if not all(verification.values()):
    raise RuntimeError(
        "Reverse-KL master verification failed"
    )


dst.write_text(
    master,
    encoding="utf-8",
)

dst.chmod(0o755)


print()
print("RKL_MASTER_BUILD_PASS")
PY


bash -n "$DST_MASTER"

echo "RKL_MASTER_SYNTAX_PASS"


###############################################################################
# STAGE 5 — PRINT EXACT SCIENTIFIC DIFF
###############################################################################

echo
echo "===== STAGE 5/6: SCIENTIFIC AUDIT ====="

echo
echo "--- Reverse-KL objective block ---"

grep -n -A12 -B10 \
'D_KL(P_student || P_teacher)' \
"$DST_TRAINER"

echo
echo "--- Run configuration ---"

grep -nE \
'NAME=|TRAINER=|nproc_per_node|master_port|pe_k1_clean3732|RKL|REVERSE-KL' \
"$DST_MASTER" \
| head -100


echo
echo "SCIENTIFIC_AUDIT_PASS"


###############################################################################
# STAGE 6 — LAUNCH
###############################################################################

echo
echo "===== STAGE 6/6: LAUNCH RKL OPD ====="

mkdir -p "$LOG_ROOT/$EXP"


# We deliberately use a fresh experiment output path.
# No successful FKL artifact is modified.

nohup setsid bash "$DST_MASTER" \
    > "$LOG" 2>&1 < /dev/null &


PID=$!


echo
echo "======================================================================"
echo "RKL_OPD_STARTED"
echo "PID=$PID"
echo "LOG=$LOG"
echo "TRAINER=$DST_TRAINER"
echo "MASTER=$DST_MASTER"
echo "======================================================================"

