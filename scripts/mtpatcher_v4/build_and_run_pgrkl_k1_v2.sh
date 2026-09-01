#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1


EXP="mtpatcher_v3_full6565_20260823"

# ----------------------------------------------------------------------
# Use the RKL pipeline that has ALREADY completed all 3 epochs + eval.
# ----------------------------------------------------------------------

SRC_TRAINER="$ROOT/scripts/mtpatcher_v4/train_opd_reversekl_torchnpu.py"

SRC_MASTER="$ROOT/scripts/mtpatcher_v4/run_opd_reversekl_torchnpu_v2_oneclick.sh"


DST_TRAINER="$ROOT/scripts/mtpatcher_v4/train_opd_pgrkl_k1_torchnpu_v2.py"

DST_MASTER="$ROOT/scripts/mtpatcher_v4/run_opd_pgrkl_k1_v2_oneclick.sh"


OLD_NAME="opd_torchnpu_rkl_pe3732_v2"

NEW_NAME="opd_torchnpu_pgrkl_k1_pe3732_v2"


LOG="$LOG_ROOT/$EXP/${NEW_NAME}.log"


echo "======================================================================"
echo "MT-PATCHER V4 — BUILD SAMPLED-TOKEN K1 PG-RKL OPD V2"
date
echo "======================================================================"


###############################################################################
# STAGE 1/6 — VERIFY SUCCESSFUL SOURCE PIPELINE
###############################################################################

echo
echo "===== STAGE 1/6: VERIFY SOURCE PIPELINE ====="

test -f "$SRC_TRAINER"
test -f "$SRC_MASTER"

echo "SRC_TRAINER=$SRC_TRAINER"
echo "SRC_MASTER=$SRC_MASTER"

echo "SOURCE_FILES_PASS"


###############################################################################
# STAGE 2/6 — BUILD TRAINER
###############################################################################

echo
echo "===== STAGE 2/6: BUILD PG-RKL TRAINER ====="

export SRC_TRAINER
export DST_TRAINER

python - <<'PY'
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
# A. Verify only genuinely essential pieces.
#
# DO NOT require t_prob / s_prob.
# They are irrelevant to sampled-token k1 PG-RKL.
# ======================================================================

required = [
    "student.module.generate",
    "t_logp",
    "s_logp",
    "token_kl",
    "clip_grad_norm_",
    "torch.isfinite(loss)",
    "DistributedSampler",
]


for marker in required:

    if marker not in text:

        raise RuntimeError(
            f"Successful RKL trainer missing essential marker: {marker}"
        )


print("SUCCESSFUL_RKL_SOURCE_PASS")


# ======================================================================
# B. Resolve the actual tensor returned by student.module.generate(...)
#
# Example:
#
# generated = student.module.generate(...)
#
# We extract "generated" statically from the successful source instead
# of guessing tensor names at runtime.
# ======================================================================

generate_matches = re.findall(
    r"(?m)^[ \t]*([A-Za-z_][A-Za-z0-9_]*)"
    r"[ \t]*=[ \t]*student\.module\.generate[ \t]*\(",
    text,
)


if len(generate_matches) != 1:

    print(
        "student.module.generate assignment candidates =",
        generate_matches,
    )

    raise RuntimeError(
        "Expected exactly one direct student.module.generate assignment; "
        f"found {len(generate_matches)}"
    )


GENERATED_VAR = generate_matches[0]


print(
    "GENERATED_VAR =",
    GENERATED_VAR,
)


# ======================================================================
# C. Helpers
# ======================================================================

def exact_indices(pattern):

    regex = re.compile(pattern)

    return [
        i
        for i, line in enumerate(lines)
        if regex.match(line)
    ]


def expression_end(start):

    balance = 0
    seen_open = False

    for i in range(start, len(lines)):

        code = lines[i].split(
            "#",
            1,
        )[0]

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

        if opens:
            seen_open = True

        balance += opens
        balance -= closes

        if seen_open and balance == 0:
            return i

    raise RuntimeError(
        f"Could not resolve multiline expression from line {start + 1}"
    )


# ======================================================================
# D. Locate exactly:
#
# token_kl = ...
# ======================================================================

token_candidates = exact_indices(
    r"^\s*token_kl\s*="
)


if len(token_candidates) != 1:

    print(
        "token_kl candidates =",
        [
            (i + 1, lines[i])
            for i in token_candidates
        ],
    )

    raise RuntimeError(
        "Expected exactly one token_kl assignment; "
        f"found {len(token_candidates)}"
    )


token_idx = token_candidates[0]

token_end = expression_end(
    token_idx
)


print(
    "TOKEN_KL_START =",
    token_idx + 1,
)

print(
    "TOKEN_KL_END   =",
    token_end + 1,
)


# ======================================================================
# E. Locate the loss belonging to this KL block.
# ======================================================================

loss_candidates = []

for i in range(
    token_end + 1,
    min(
        token_end + 30,
        len(lines),
    ),
):

    if re.match(
        r"^\s*loss\s*=",
        lines[i],
    ):

        loss_candidates.append(i)


if len(loss_candidates) != 1:

    print(
        "nearby loss candidates =",
        [
            (i + 1, lines[i])
            for i in loss_candidates
        ],
    )

    raise RuntimeError(
        "Expected exactly one loss assignment after token_kl; "
        f"found {len(loss_candidates)}"
    )


loss_idx = loss_candidates[0]

loss_end = expression_end(
    loss_idx
)


print(
    "LOSS_START =",
    loss_idx + 1,
)

print(
    "LOSS_END   =",
    loss_end + 1,
)


indent = (
    lines[token_idx]
    [
        :len(lines[token_idx])
        - len(lines[token_idx].lstrip())
    ]
)


# ======================================================================
# F. Replace exact full-vocab RKL objective with sampled-token k1 PG.
#
# Student rollout:
#
#       a_t ~ pi_student(. | s_t)
#
# Sampled-token estimator:
#
#       k1_t =
#           stopgrad(
#               log pi_student(a_t | s_t)
#               -
#               log pi_teacher(a_t | s_t)
#           )
#
# Distillation reward:
#
#       r_t = -k1_t
#
# Score-function loss:
#
#       L =
#           -mean(
#               r_t *
#               log pi_student(a_t | s_t)
#           )
#
# Because r_t is detached, gradient only flows through student sampled
# action log-probabilities.
# ======================================================================

g = GENERATED_VAR


new_block = [

    indent + "# ================================================================",
    indent + "# Sampled-token k1 Policy-Gradient Reverse-KL OPD",
    indent + "# ================================================================",
    "",

    indent + "_pg_batch = int(s_logits.shape[0])",
    indent + "_pg_steps = int(s_logits.shape[1])",
    "",

    indent + f"_pg_rollout_ids = {g}",
    "",

    indent + "if not torch.is_tensor(_pg_rollout_ids):",
    indent + "    raise RuntimeError(",
    indent + f"        'student.module.generate output {g} is not a Tensor'",
    indent + "    )",
    "",

    indent + "if _pg_rollout_ids.ndim != 2:",
    indent + "    raise RuntimeError(",
    indent + "        'Expected generate output [batch, sequence], got '",
    indent + "        f'{tuple(_pg_rollout_ids.shape)}'",
    indent + "    )",
    "",

    indent + "if int(_pg_rollout_ids.shape[0]) != _pg_batch:",
    indent + "    raise RuntimeError(",
    indent + "        'Rollout/logit batch mismatch: '",
    indent + "        f'rollout={tuple(_pg_rollout_ids.shape)} '",
    indent + "        f'logits={tuple(s_logits.shape)}'",
    indent + "    )",
    "",

    indent + "if int(_pg_rollout_ids.shape[1]) < _pg_steps:",
    indent + "    raise RuntimeError(",
    indent + "        'Rollout sequence shorter than response logits: '",
    indent + "        f'rollout={tuple(_pg_rollout_ids.shape)} '",
    indent + "        f'logits={tuple(s_logits.shape)}'",
    indent + "    )",
    "",

    indent + "# For decoder-only generate(), output is prompt + generated response.",
    indent + "# The verified trainer has already sliced s_logits/t_logits to",
    indent + "# response prediction states, so the last _pg_steps token IDs are",
    indent + "# the sampled actions corresponding to those logits.",
    "",

    indent + "_pg_action_ids = _pg_rollout_ids[:, -_pg_steps:]",
    "",

    indent + "_pg_action_ids = _pg_action_ids.to(",
    indent + "    device=s_logits.device,",
    indent + "    dtype=torch.long,",
    indent + ")",
    "",

    indent + "if tuple(_pg_action_ids.shape) != (_pg_batch, _pg_steps):",
    indent + "    raise RuntimeError(",
    indent + "        'Sampled action alignment failure: '",
    indent + "        f'actions={tuple(_pg_action_ids.shape)} '",
    indent + "        f'logits={tuple(s_logits.shape)}'",
    indent + "    )",
    "",

    indent + "_pg_min_id = int(_pg_action_ids.min().item())",
    indent + "_pg_max_id = int(_pg_action_ids.max().item())",
    indent + "_pg_vocab = int(s_logits.shape[-1])",
    "",

    indent + "if _pg_min_id < 0 or _pg_max_id >= _pg_vocab:",
    indent + "    raise RuntimeError(",
    indent + "        'Invalid sampled token ID: '",
    indent + "        f'min={_pg_min_id} max={_pg_max_id} vocab={_pg_vocab}'",
    indent + "    )",
    "",

    indent + "# ------------------------------------------------------------",
    indent + "# Gather log-probability of the ACTUALLY sampled student token.",
    indent + "# ------------------------------------------------------------",
    "",

    indent + "s_action_logp = s_logp.gather(",
    indent + "    dim=-1,",
    indent + "    index=_pg_action_ids.unsqueeze(-1),",
    indent + ").squeeze(-1)",
    "",

    indent + "t_action_logp = t_logp.gather(",
    indent + "    dim=-1,",
    indent + "    index=_pg_action_ids.unsqueeze(-1),",
    indent + ").squeeze(-1)",
    "",

    indent + "if tuple(s_action_logp.shape) != (_pg_batch, _pg_steps):",
    indent + "    raise RuntimeError(",
    indent + "        'Student sampled logprob shape mismatch'",
    indent + "    )",
    "",

    indent + "if tuple(t_action_logp.shape) != (_pg_batch, _pg_steps):",
    indent + "    raise RuntimeError(",
    indent + "        'Teacher sampled logprob shape mismatch'",
    indent + "    )",
    "",

    indent + "# ------------------------------------------------------------",
    indent + "# k1 sampled-token reverse-KL estimator.",
    indent + "#",
    indent + "# Important:",
    indent + "# k1 itself MUST be stop-gradient for the PG estimator.",
    indent + "# ------------------------------------------------------------",
    "",

    indent + "s_action_for_k1 = torch.clamp(",
    indent + "    s_action_logp.detach().float(),",
    indent + "    min=-10.0,",
    indent + ")",
    "",

    indent + "t_action_for_k1 = torch.clamp(",
    indent + "    t_action_logp.detach().float(),",
    indent + "    min=-10.0,",
    indent + ")",
    "",

    indent + "k1 = (",
    indent + "    s_action_for_k1",
    indent + "    - t_action_for_k1",
    indent + ").detach()",
    "",

    indent + "k1 = torch.clamp(",
    indent + "    k1,",
    indent + "    min=-10.0,",
    indent + "    max=10.0,",
    indent + ")",
    "",

    indent + "if not torch.isfinite(k1).all():",
    indent + "    raise RuntimeError(",
    indent + "        'Non-finite sampled k1 detected'",
    indent + "    )",
    "",

    indent + "# Preserve the verified trainer's metric variable.",
    indent + "# Individual sampled k1 values may be negative.",
    indent + "token_kl = k1",
    "",

    indent + "# Teacher preference over the student's sampled action.",
    indent + "distill_reward = (-k1).detach()",
    "",

    indent + "# ------------------------------------------------------------",
    indent + "# Vanilla score-function policy-gradient objective.",
    indent + "#",
    indent + "# L = - E[r_t * log pi_student(a_t|s_t)]",
    indent + "# ------------------------------------------------------------",
    "",

    indent + "loss = -(",
    indent + "    distill_reward",
    indent + "    * s_action_logp.float()",
    indent + ").mean()",
]


lines[
    token_idx:loss_end + 1
] = new_block


text = "\n".join(lines) + "\n"


# ======================================================================
# G. Update labels only.
# ======================================================================

replacements = {

    "ON-POLICY REVERSE-KL":
        "ON-POLICY PG-RKL-K1",

    "D_KL(Teacher || Student)":
        "PG-RKL-k1 sampled-token",

    "D_KL(Student || Teacher)":
        "PG-RKL-k1 sampled-token",

    "token_mean_reverse_kl":
        "token_mean_sampled_k1",

    "MTPATCHER_V4_TORCHNPU_REVERSEKL_TRAINING_PASS":
        "MTPATCHER_V4_TORCHNPU_PGRKL_K1_TRAINING_PASS",
}


for old, new in replacements.items():

    text = text.replace(
        old,
        new,
    )


# Also fix the stale FKL label that appeared in the successful RKL log.
text = text.replace(
    'OBJECTIVE = D_KL(Teacher || Student)',
    'OBJECTIVE = PG-RKL-k1 sampled-token',
)


# ======================================================================
# H. Verify SCIENTIFIC objective, without requiring t_prob/s_prob.
# ======================================================================

checks = {

    "student rollout preserved":
        "student.module.generate" in text,

    "actual rollout variable":
        f"_pg_rollout_ids = {GENERATED_VAR}" in text,

    "sampled student logp":
        "s_action_logp = s_logp.gather" in text,

    "sampled teacher logp":
        "t_action_logp = t_logp.gather" in text,

    "k1 student minus teacher":
        "s_action_for_k1" in text
        and "- t_action_for_k1" in text,

    "k1 stop gradient":
        ").detach()" in text
        and "k1 =" in text,

    "negative-k1 reward":
        "distill_reward = (-k1).detach()" in text,

    "policy gradient":
        "distill_reward" in text
        and "s_action_logp.float()" in text,

    "finite k1 guard":
        "torch.isfinite(k1).all()" in text,

    "finite loss guard":
        "torch.isfinite(loss)" in text,

    "gradient clipping":
        "clip_grad_norm_" in text,

    "distributed sampler":
        "DistributedSampler" in text,
}


print()
print("PG-RKL SCIENTIFIC VERIFICATION")


for name, value in checks.items():

    print(
        f"{name:34s} = {value}"
    )


if not all(checks.values()):

    raise RuntimeError(
        "PG-RKL scientific verification failed"
    )


dst.write_text(
    text,
    encoding="utf-8",
)


print()
print("PGRKL_K1_V2_TRAINER_BUILD_PASS")
PY


###############################################################################
# STAGE 3/6 — PYTHON SYNTAX
###############################################################################

echo
echo "===== STAGE 3/6: TRAINER SYNTAX CHECK ====="

python -m py_compile "$DST_TRAINER"

echo "PGRKL_K1_V2_TRAINER_SYNTAX_PASS"


###############################################################################
# STAGE 4/6 — BUILD MASTER
###############################################################################

echo
echo "===== STAGE 4/6: BUILD ONE-CLICK MASTER ====="

export SRC_MASTER
export DST_MASTER
export OLD_NAME
export NEW_NAME

python - <<'PY'
import os
from pathlib import Path


src = Path(
    os.environ["SRC_MASTER"]
)

dst = Path(
    os.environ["DST_MASTER"]
)

old_name = os.environ["OLD_NAME"]
new_name = os.environ["NEW_NAME"]


text = src.read_text(
    encoding="utf-8"
)


required = [
    old_name,
    "train_opd_reversekl_torchnpu.py",
    "--nproc_per_node=16",
    "pe_k1_clean3732.jsonl",
]


for marker in required:

    if marker not in text:

        raise RuntimeError(
            f"Successful RKL master missing marker: {marker}"
        )


# ----------------------------------------------------------------------
# Fresh run identity.
# ----------------------------------------------------------------------

text = text.replace(
    old_name,
    new_name,
)


# ----------------------------------------------------------------------
# New PG trainer.
# ----------------------------------------------------------------------

text = text.replace(
    "$ROOT/scripts/mtpatcher_v4/train_opd_reversekl_torchnpu.py",
    "$ROOT/scripts/mtpatcher_v4/train_opd_pgrkl_k1_torchnpu_v2.py",
)


text = text.replace(
    "train_opd_reversekl_torchnpu.py",
    "train_opd_pgrkl_k1_torchnpu_v2.py",
)


# ----------------------------------------------------------------------
# New HCCL rendezvous port.
# ----------------------------------------------------------------------

text = text.replace(
    "--master_port=29636",
    "--master_port=29638",
)


# ----------------------------------------------------------------------
# Summary labels.
# ----------------------------------------------------------------------

text = text.replace(
    "ON-POLICY REVERSE-KL",
    "ON-POLICY PG-RKL-K1",
)


text = text.replace(
    "OPD-RKL-PE3732",
    "OPD-PGRKL-K1-PE3732",
)


text = text.replace(
    "OPD_RKL_PE3732",
    "OPD_PGRKL_K1_PE3732",
)


text = text.replace(
    "opd_torchnpu_reversekl_final_summary.json",
    "opd_torchnpu_pgrkl_k1_final_summary.json",
)


text = text.replace(
    "MTPATCHER_V4_TORCHNPU_REVERSEKL_ALL_PASS",
    "MTPATCHER_V4_TORCHNPU_PGRKL_K1_ALL_PASS",
)


checks = {

    "new run name":
        new_name in text,

    "new trainer":
        "train_opd_pgrkl_k1_torchnpu_v2.py" in text,

    "old trainer removed":
        "train_opd_reversekl_torchnpu.py" not in text,

    "16 NPUs":
        "--nproc_per_node=16" in text,

    "PE3732":
        "pe_k1_clean3732.jsonl" in text,

    "port 29638":
        "--master_port=29638" in text,
}


print("MASTER VERIFICATION")


for name, value in checks.items():

    print(
        f"{name:28s} = {value}"
    )


if not all(checks.values()):

    raise RuntimeError(
        "PG-RKL V2 master verification failed"
    )


dst.write_text(
    text,
    encoding="utf-8",
)

dst.chmod(
    0o755
)


print()
print("PGRKL_K1_V2_MASTER_BUILD_PASS")
PY


bash -n "$DST_MASTER"

echo "PGRKL_K1_V2_MASTER_SYNTAX_PASS"


###############################################################################
# STAGE 5/6 — AUDIT
###############################################################################

echo
echo "===== STAGE 5/6: SCIENTIFIC AUDIT ====="

echo
echo "--- Actual sampled-token PG-RKL block ---"

grep -n -A130 -B15 \
'Sampled-token k1 Policy-Gradient Reverse-KL OPD' \
"$DST_TRAINER" \
| head -180


echo
echo "--- Run configuration ---"

grep -nE \
'NAME=|TRAINER=|nproc_per_node|master_port|PGRKL|PG-RKL|pe_k1_clean3732' \
"$DST_MASTER" \
| head -100


echo
echo "PGRKL_K1_V2_SCIENTIFIC_AUDIT_PASS"


###############################################################################
# STAGE 6/6 — LAUNCH
###############################################################################

echo
echo "===== STAGE 6/6: LAUNCH ====="

mkdir -p \
"$LOG_ROOT/$EXP"


nohup setsid bash "$DST_MASTER" \
    > "$LOG" 2>&1 < /dev/null &


PID=$!


echo
echo "======================================================================"
echo "PG-RKL K1 V2 STARTED"
echo "PID=$PID"
echo "LOG=$LOG"
echo "TRAINER=$DST_TRAINER"
echo "MASTER=$DST_MASTER"
echo "======================================================================"

