#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

###############################################################################
# SOURCE: the RKL trainer which has already completed all 3 epochs.
###############################################################################

SRC_TRAINER="$ROOT/scripts/mtpatcher_v4/train_opd_reversekl_torchnpu.py"

DST_TRAINER="$ROOT/scripts/mtpatcher_v4/train_opd_pgrkl_k1_torchnpu.py"

SRC_MASTER="$ROOT/scripts/mtpatcher_v4/run_opd_reversekl_torchnpu_v2_oneclick.sh"

DST_MASTER="$ROOT/scripts/mtpatcher_v4/run_opd_pgrkl_k1_torchnpu_oneclick.sh"

OLD_NAME="opd_torchnpu_rkl_pe3732_v2"

NEW_NAME="opd_torchnpu_pgrkl_k1_pe3732_v1"

LOG="$LOG_ROOT/$EXP/${NEW_NAME}.log"


echo "======================================================================"
echo "MT-PATCHER V4 — SAMPLED-TOKEN K1 POLICY-GRADIENT OPD"
date
echo "======================================================================"


###############################################################################
# STAGE 1 — VERIFIED SOURCES
###############################################################################

echo
echo "===== STAGE 1/6: VERIFY SOURCE PIPELINE ====="

test -f "$SRC_TRAINER"
test -f "$SRC_MASTER"

echo "SRC_TRAINER=$SRC_TRAINER"
echo "SRC_MASTER=$SRC_MASTER"

echo "SOURCE_PIPELINE_PASS"


###############################################################################
# STAGE 2 — BUILD PG-RKL K1 TRAINER
###############################################################################

echo
echo "===== STAGE 2/6: BUILD PG-RKL K1 TRAINER ====="

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


# ============================================================================
# Verify that this is the already-successful exact-RKL trainer.
# ============================================================================

required = [
    "student.module.generate",
    "DistributedSampler",
    "t_logp",
    "s_logp",
    "s_prob = s_logp.exp()",
    "t_prob = t_logp.exp()",
    "token_kl",
    "clip_grad_norm_",
    "torch.isfinite(loss)",
]


for marker in required:

    if marker not in text:

        raise RuntimeError(
            f"Verified RKL trainer marker missing: {marker}"
        )


print("VERIFIED_RKL_SOURCE_PASS")


# ============================================================================
# Helpers
# ============================================================================

def unique_index(predicate, description):

    found = [
        i
        for i, line in enumerate(lines)
        if predicate(line)
    ]

    if len(found) != 1:

        print()
        print(
            f"{description}: matches={len(found)}"
        )

        for i in found:
            print(
                f"{i + 1}: {lines[i]}"
            )

        raise RuntimeError(
            f"{description}: expected one match, got {len(found)}"
        )

    return found[0]


def expression_end(start):

    balance = 0
    opened = False

    for i in range(
        start,
        len(lines),
    ):

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
            opened = True

        balance += opens
        balance -= closes

        if opened and balance == 0:
            return i

    raise RuntimeError(
        f"Could not find expression end from line {start + 1}"
    )


# ============================================================================
# Locate exact RKL token_kl assignment.
# ============================================================================

token_idx = unique_index(
    lambda line:
        re.match(
            r"^\s*token_kl\s*=",
            line,
        )
        is not None,
    "token_kl assignment",
)


token_end = expression_end(
    token_idx
)


# ============================================================================
# Locate:
#
#     loss = token_kl.mean()
# ============================================================================

loss_candidates = [
    i
    for i in range(
        token_end + 1,
        min(
            token_end + 25,
            len(lines),
        ),
    )
    if re.match(
        r"^\s*loss\s*=\s*token_kl\.mean\(\)\s*$",
        lines[i],
    )
]


if len(loss_candidates) != 1:

    raise RuntimeError(
        "Expected exact-RKL loss immediately after token_kl; "
        f"found={len(loss_candidates)}"
    )


loss_idx = loss_candidates[0]


indent = (
    lines[token_idx]
    [:len(lines[token_idx])
      - len(lines[token_idx].lstrip())]
)


print(
    "TOKEN_KL_LINE =",
    token_idx + 1,
)

print(
    "LOSS_LINE =",
    loss_idx + 1,
)


# ============================================================================
# Replace exact full-vocabulary RKL with sampled-token k1 PG objective.
#
# Latest verl PG-OPD definition:
#
#   k1_t = sg(
#       log pi_student(y_t | s_t)
#       -
#       log pi_teacher(y_t | s_t)
#   )
#
#   reward_t = -k1_t
#
#   policy loss:
#
#       - reward_t * log pi_student(y_t | s_t)
#
# Here rollout and update happen immediately, so this is the vanilla
# score-function form. There is no meaningful PPO importance ratio because
# rollout-policy == pre-update policy for this single update.
# ============================================================================

block = [

    indent + "# ================================================================",
    indent + "# Sampled-token k1 Policy-Gradient OPD",
    indent + "#",
    indent + "# k1 = sg(log pi_student(a|s) - log pi_teacher(a|s))",
    indent + "# reward = -k1",
    indent + "# loss = -reward * log pi_student(a|s)",
    indent + "#",
    indent + "# This reproduces the core PG-RKL objective used by verl OPD,",
    indent + "# while retaining this project's verified torch_npu/DDP runtime.",
    indent + "# ================================================================",
    "",

    indent + "_pg_batch = int(s_logits.shape[0])",
    indent + "_pg_steps = int(s_logits.shape[1])",
    "",

    indent + "# ------------------------------------------------------------",
    indent + "# Resolve the sampled rollout token IDs already present in this",
    indent + "# verified trainer. Prefer response/generated tensors with exact",
    indent + "# [batch, response_steps] shape.",
    indent + "# ------------------------------------------------------------",
    "",

    indent + "_pg_exact = []",
    indent + "_pg_suffix = []",
    "",

    indent + "for _pg_name, _pg_value in list(locals().items()):",
    indent + "    if _pg_name.startswith('_pg_'):",
    indent + "        continue",
    "",

    indent + "    if not torch.is_tensor(_pg_value):",
    indent + "        continue",
    "",

    indent + "    if _pg_value.ndim != 2:",
    indent + "        continue",
    "",

    indent + "    if _pg_value.dtype not in (",
    indent + "        torch.int64,",
    indent + "        torch.int32,",
    indent + "        torch.long,",
    indent + "    ):",
    indent + "        continue",
    "",

    indent + "    if int(_pg_value.shape[0]) != _pg_batch:",
    indent + "        continue",
    "",

    indent + "    _pg_lower = _pg_name.lower()",
    "",

    indent + "    if any(",
    indent + "        bad in _pg_lower",
    indent + "        for bad in (",
    indent + "            'mask',",
    indent + "            'length',",
    indent + "            'position',",
    indent + "        )",
    indent + "    ):",
    indent + "        continue",
    "",

    indent + "    _pg_semantic = any(",
    indent + "        key in _pg_lower",
    indent + "        for key in (",
    indent + "            'response',",
    indent + "            'generated',",
    indent + "            'completion',",
    indent + "            'rollout',",
    indent + "            'sequence',",
    indent + "            'full',",
    indent + "            'output',",
    indent + "        )",
    indent + "    )",
    "",

    indent + "    if int(_pg_value.shape[1]) == _pg_steps:",
    indent + "        if _pg_semantic:",
    indent + "            _pg_exact.append(",
    indent + "                (_pg_name, _pg_value)",
    indent + "            )",
    "",

    indent + "    elif int(_pg_value.shape[1]) > _pg_steps:",
    indent + "        if _pg_semantic:",
    indent + "            _pg_suffix.append(",
    indent + "                (_pg_name, _pg_value)",
    indent + "            )",
    "",

    indent + "# Prefer names that explicitly denote generated response IDs.",
    indent + "_pg_priority = (",
    indent + "    'response_ids',",
    indent + "    'generated_ids',",
    indent + "    'completion_ids',",
    indent + "    'rollout_ids',",
    indent + "    'new_ids',",
    indent + "    'response',",
    indent + "    'generated',",
    indent + "    'completion',",
    indent + "    'rollout',",
    indent + ")",
    "",

    indent + "def _pg_rank_name(name):",
    indent + "    low = name.lower()",
    indent + "    for j, key in enumerate(_pg_priority):",
    indent + "        if low == key:",
    indent + "            return (0, j, len(low))",
    indent + "        if key in low:",
    indent + "            return (1, j, len(low))",
    indent + "    return (2, 999, len(low))",
    "",

    indent + "if _pg_exact:",
    indent + "    _pg_exact.sort(",
    indent + "        key=lambda x: _pg_rank_name(x[0])",
    indent + "    )",
    "",

    indent + "    _pg_action_source, _pg_action_ids = _pg_exact[0]",
    "",

    indent + "elif _pg_suffix:",
    "",

    indent + "    _pg_suffix.sort(",
    indent + "        key=lambda x: _pg_rank_name(x[0])",
    indent + "    )",
    "",

    indent + "    _pg_action_source, _pg_full_ids = _pg_suffix[0]",
    "",

    indent + "    # Response tokens are the suffix of the full generated",
    indent + "    # prompt+response sequence. s_logits/t_logits have already",
    indent + "    # been sliced by the verified trainer to response states.",
    indent + "    _pg_action_ids = _pg_full_ids[:, -_pg_steps:]",
    "",

    indent + "else:",
    "",

    indent + "    _pg_debug = []",
    "",

    indent + "    for _pg_name, _pg_value in list(locals().items()):",
    indent + "        if torch.is_tensor(_pg_value):",
    indent + "            _pg_debug.append(",
    indent + "                (",
    indent + "                    _pg_name,",
    indent + "                    tuple(_pg_value.shape),",
    indent + "                    str(_pg_value.dtype),",
    indent + "                )",
    indent + "            )",
    "",

    indent + "    raise RuntimeError(",
    indent + "        'Could not resolve sampled response token IDs. '",
    indent + "        f's_logits_shape={tuple(s_logits.shape)} '",
    indent + "        f'tensor_locals={_pg_debug}'",
    indent + "    )",
    "",

    indent + "_pg_action_ids = _pg_action_ids.to(",
    indent + "    device=s_logits.device,",
    indent + "    dtype=torch.long,",
    indent + ")",
    "",

    indent + "if tuple(_pg_action_ids.shape) != (_pg_batch, _pg_steps):",
    indent + "    raise RuntimeError(",
    indent + "        'PG action/logit alignment failure: '",
    indent + "        f'action_shape={tuple(_pg_action_ids.shape)} '",
    indent + "        f'logit_shape={tuple(s_logits.shape)} '",
    indent + "        f'source={_pg_action_source}'",
    indent + "    )",
    "",

    indent + "if int(_pg_action_ids.min().item()) < 0:",
    indent + "    raise RuntimeError(",
    indent + "        'Negative sampled token ID detected'",
    indent + "    )",
    "",

    indent + "if int(_pg_action_ids.max().item()) >= int(s_logits.shape[-1]):",
    indent + "    raise RuntimeError(",
    indent + "        'Sampled token ID exceeds vocabulary: '",
    indent + "        f'max={int(_pg_action_ids.max().item())} '",
    indent + "        f'vocab={int(s_logits.shape[-1])}'",
    indent + "    )",
    "",

    indent + "# Gather log-probabilities only at the token actually sampled",
    indent + "# by the student rollout.",
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

    indent + "# Match the numerical safeguards of current verl defaults:",
    indent + "# clamp very small sampled-token log-probabilities, then clamp",
    indent + "# the per-token distillation signal.",
    "",

    indent + "s_action_for_k1 = torch.clamp(",
    indent + "    s_action_logp.detach(),",
    indent + "    min=-10.0,",
    indent + ")",
    "",

    indent + "t_action_for_k1 = torch.clamp(",
    indent + "    t_action_logp.detach(),",
    indent + "    min=-10.0,",
    indent + ")",
    "",

    indent + "# k1 is explicitly stop-gradient.",
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

    indent + "# Keep token_kl name because the verified trainer's distributed",
    indent + "# metric aggregation expects this tensor.",
    indent + "# Individual k1 samples may be negative; their expectation is RKL.",
    indent + "token_kl = k1",
    "",

    indent + "distill_reward = (-k1).detach()",
    "",

    indent + "# Score-function / vanilla policy-gradient objective.",
    indent + "#",
    indent + "#   loss = - reward * log pi_student(a|s)",
    indent + "#        =   k1     * log pi_student(a|s)",
    "",

    indent + "loss = -(",
    indent + "    distill_reward",
    indent + "    * s_action_logp",
    indent + ").mean()",
]


lines[
    token_idx:loss_idx + 1
] = block


text = "\n".join(lines) + "\n"


# ============================================================================
# Clean old RKL descriptions.
# ============================================================================

text = text.replace(
    "ON-POLICY REVERSE-KL",
    "ON-POLICY PG-RKL-K1",
)

text = text.replace(
    "D_KL(Teacher || Student)",
    "PG-RKL-k1 sampled-token",
)

text = text.replace(
    "D_KL(Student || Teacher)",
    "PG-RKL-k1 sampled-token",
)

text = text.replace(
    "token_mean_reverse_kl",
    "token_mean_sampled_k1",
)

text = text.replace(
    "# Exact full-vocabulary reverse KL:",
    "# Sampled-token reverse-KL estimator:",
)

text = text.replace(
    "# D_KL(P_student || P_teacher)",
    "# k1 = sg(log P_student(a) - log P_teacher(a))",
)

text = text.replace(
    "MTPATCHER_V4_TORCHNPU_REVERSEKL_TRAINING_PASS",
    "MTPATCHER_V4_TORCHNPU_PGRKL_K1_TRAINING_PASS",
)


# ============================================================================
# Verification.
# ============================================================================

checks = {
    "student rollout":
        "student.module.generate" in text,

    "sampled-token gather":
        "s_action_logp = s_logp.gather" in text,

    "teacher sampled logp":
        "t_action_logp = t_logp.gather" in text,

    "k1 stop gradient":
        ").detach()" in text
        and "k1 =" in text,

    "negative k1 reward":
        "distill_reward = (-k1).detach()" in text,

    "policy-gradient loss":
        "distill_reward"
        in text
        and "s_action_logp"
        in text,

    "teacher probability diagnostics retained":
        "t_prob = t_logp.exp()" in text,

    "student probability diagnostics retained":
        "s_prob = s_logp.exp()" in text,

    "finite guard":
        "torch.isfinite(loss)" in text,

    "gradient clip":
        "clip_grad_norm_" in text,

    "DDP sampler":
        "DistributedSampler" in text,
}


print()
print("PG-RKL TRAINER VERIFICATION")


for name, ok in checks.items():

    print(
        f"{name:38s} = {ok}"
    )


if not all(checks.values()):

    raise RuntimeError(
        "PG-RKL trainer verification failed"
    )


dst.write_text(
    text,
    encoding="utf-8",
)


print()
print("PGRKL_K1_TRAINER_BUILD_PASS")
PY


###############################################################################
# STAGE 3 — SYNTAX CHECK
###############################################################################

echo
echo "===== STAGE 3/6: TRAINER SYNTAX CHECK ====="

python -m py_compile "$DST_TRAINER"

echo "PGRKL_K1_TRAINER_SYNTAX_PASS"


###############################################################################
# STAGE 4 — CREATE ONE-CLICK MASTER
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


src = Path(os.environ["SRC_MASTER"])
dst = Path(os.environ["DST_MASTER"])

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
            f"Verified RKL master marker missing: {marker}"
        )


# Fresh run identity.
text = text.replace(
    old_name,
    new_name,
)


# New trainer.
text = text.replace(
    "$ROOT/scripts/mtpatcher_v4/train_opd_reversekl_torchnpu.py",
    "$ROOT/scripts/mtpatcher_v4/train_opd_pgrkl_k1_torchnpu.py",
)


text = text.replace(
    "train_opd_reversekl_torchnpu.py",
    "train_opd_pgrkl_k1_torchnpu.py",
)


# Separate rendezvous port.
text = text.replace(
    "--master_port=29636",
    "--master_port=29637",
)


# Labels.
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


# Fresh summary artifact.
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

    "PG trainer":
        "train_opd_pgrkl_k1_torchnpu.py" in text,

    "old trainer removed":
        "train_opd_reversekl_torchnpu.py" not in text,

    "16 NPU":
        "--nproc_per_node=16" in text,

    "PE3732":
        "pe_k1_clean3732.jsonl" in text,

    "new port":
        "--master_port=29637" in text,
}


print("MASTER VERIFICATION")


for name, ok in checks.items():

    print(
        f"{name:30s} = {ok}"
    )


if not all(checks.values()):

    raise RuntimeError(
        "PG-RKL master verification failed"
    )


dst.write_text(
    text,
    encoding="utf-8",
)

dst.chmod(
    0o755
)


print()
print("PGRKL_K1_MASTER_BUILD_PASS")
PY


bash -n "$DST_MASTER"

echo "PGRKL_K1_MASTER_SYNTAX_PASS"


###############################################################################
# STAGE 5 — SCIENTIFIC AUDIT
###############################################################################

echo
echo "===== STAGE 5/6: SCIENTIFIC AUDIT ====="

echo
echo "--- PG-RKL k1 objective ---"

grep -n -A120 -B12 \
'Sampled-token k1 Policy-Gradient OPD' \
"$DST_TRAINER" \
| head -160


echo
echo "--- Master config ---"

grep -nE \
'NAME=|TRAINER=|nproc_per_node|master_port|PGRKL|PG-RKL|pe_k1_clean3732' \
"$DST_MASTER" \
| head -100


echo
echo "PGRKL_K1_SCIENTIFIC_AUDIT_PASS"


###############################################################################
# STAGE 6 — LAUNCH
###############################################################################

echo
echo "===== STAGE 6/6: LAUNCH ====="

mkdir -p "$LOG_ROOT/$EXP"


nohup setsid bash "$DST_MASTER" \
    > "$LOG" 2>&1 < /dev/null &


PID=$!


echo
echo "======================================================================"
echo "PG-RKL K1 OPD STARTED"
echo "PID=$PID"
echo "LOG=$LOG"
echo "TRAINER=$DST_TRAINER"
echo "MASTER=$DST_MASTER"
echo "======================================================================"

