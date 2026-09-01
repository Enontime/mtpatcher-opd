#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

###############################################################################
# CLEAN SOURCE:
# verified full-vocabulary reverse-KL trainer.
#
# We deliberately do NOT derive this experiment from either of the previous
# PG-k1 trainers.
###############################################################################

RKL_SRC="$ROOT/scripts/mtpatcher_v4/train_opd_reversekl_torchnpu.py"

CLEAN_TRAINER="$ROOT/scripts/mtpatcher_v4/train_opd_pgrkl_cleanroom_v1_torchnpu.py"

PREFLIGHT_TRAINER="$ROOT/scripts/mtpatcher_v4/preflight_opd_pgrkl_cleanroom_v1_torchnpu.py"

MASTER_SRC="$ROOT/scripts/mtpatcher_v4/run_opd_pgrkl_k1_audited_v2_oneclick.sh"

FULL_MASTER="$ROOT/scripts/mtpatcher_v4/run_opd_pgrkl_cleanroom_v1_oneclick.sh"

PREFLIGHT_MASTER="$ROOT/scripts/mtpatcher_v4/run_opd_pgrkl_cleanroom_v1_preflight.sh"

FULL_NAME="opd_torchnpu_pgrkl_cleanroom_pe3732_v1"

PREFLIGHT_NAME="opd_torchnpu_pgrkl_cleanroom_preflight208_v1"

FULL_LOG="$LOG_ROOT/$EXP/${FULL_NAME}.log"

PREFLIGHT_LOG="$LOG_ROOT/$EXP/${PREFLIGHT_NAME}.log"


echo "======================================================================"
echo "CLEAN-ROOM PG-RKL V1"
echo "SOURCE = verified exact-RKL trainer"
echo "PRE-FLIGHT = 208 frozen Student rollouts"
echo "FULL RUN = launched only after mathematical consistency passes"
date
echo "======================================================================"


###############################################################################
# STAGE 1 — SOURCE CHECK
###############################################################################

echo
echo "===== STAGE 1/7: SOURCE CHECK ====="

test -f "$RKL_SRC"
test -f "$MASTER_SRC"

python -m py_compile "$RKL_SRC"

echo "RKL_SRC=$RKL_SRC"
echo "MASTER_SRC=$MASTER_SRC"

echo "CLEAN_SOURCE_PASS"


###############################################################################
# STAGE 2 — BUILD CLEAN-ROOM TRAINER
###############################################################################

echo
echo "===== STAGE 2/7: BUILD CLEAN TRAINER ====="

export RKL_SRC
export CLEAN_TRAINER

python - <<'PY'
import ast
import os
from pathlib import Path


src = Path(os.environ["RKL_SRC"])
dst = Path(os.environ["CLEAN_TRAINER"])

text = src.read_text(
    encoding="utf-8"
)


###############################################################################
# Utilities
###############################################################################

def source_segment(source, node):
    x = ast.get_source_segment(
        source,
        node,
    )
    return x or ""


def target_names(node):

    if isinstance(node, ast.Assign):
        targets = node.targets

    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]

    else:
        return []

    result = []

    def walk_target(t):

        if isinstance(t, ast.Name):
            result.append(t.id)

        elif isinstance(
            t,
            (ast.Tuple, ast.List),
        ):

            for child in t.elts:
                walk_target(child)

    for target in targets:
        walk_target(target)

    return result


def absolute_position(source, lineno, col):

    lines = source.splitlines(
        keepends=True
    )

    return (
        sum(
            len(x)
            for x in lines[:lineno - 1]
        )
        + col
    )


###############################################################################
# Locate exactly one generate() assignment.
###############################################################################

tree = ast.parse(text)

generate_assignments = []


for node in ast.walk(tree):

    if not isinstance(
        node,
        (ast.Assign, ast.AnnAssign),
    ):
        continue

    value = node.value

    if not isinstance(value, ast.Call):
        continue

    func = value.func

    if (
        isinstance(func, ast.Attribute)
        and func.attr == "generate"
    ):

        names = target_names(node)

        if len(names) == 1:

            generate_assignments.append(
                (
                    node,
                    value,
                    names[0],
                )
            )


print(
    "GENERATE_ASSIGNMENT_COUNT =",
    len(generate_assignments),
)


if len(generate_assignments) != 1:

    for node, call, name in generate_assignments:

        print(
            node.lineno,
            name,
            source_segment(text, node),
        )

    raise RuntimeError(
        "Clean exact-RKL source must contain exactly one "
        "assigned generate() call"
    )


gen_assign, gen_call, gen_var = (
    generate_assignments[0]
)


print(
    "GENERATED_VARIABLE =",
    gen_var,
)

print(
    "GENERATE_LINES =",
    gen_assign.lineno,
    "-",
    gen_assign.end_lineno,
)


###############################################################################
# Force behavior policy q == raw Student policy as closely as generate permits.
###############################################################################

kw = {
    item.arg: item
    for item in gen_call.keywords
    if item.arg is not None
}


def set_keyword(name, value):

    value_node = ast.parse(
        value,
        mode="eval",
    ).body

    if name in kw:

        kw[name].value = value_node

    else:

        new_kw = ast.keyword(
            arg=name,
            value=value_node,
        )

        gen_call.keywords.append(
            new_kw
        )

        kw[name] = new_kw


set_keyword(
    "do_sample",
    "True",
)

set_keyword(
    "temperature",
    "1.0",
)

set_keyword(
    "top_p",
    "1.0",
)

set_keyword(
    "top_k",
    "0",
)

set_keyword(
    "repetition_penalty",
    "1.0",
)

set_keyword(
    "return_dict_in_generate",
    "True",
)

set_keyword(
    "output_scores",
    "True",
)

set_keyword(
    "renormalize_logits",
    "False",
)


new_generate_call = ast.unparse(
    gen_call
)


call_start = absolute_position(
    text,
    gen_call.lineno,
    gen_call.col_offset,
)

call_end = absolute_position(
    text,
    gen_call.end_lineno,
    gen_call.end_col_offset,
)


text = (
    text[:call_start]
    + new_generate_call
    + text[call_end:]
)


###############################################################################
# Reparse and insert capture of actual rollout actions + behavior logprobs.
###############################################################################

tree = ast.parse(text)

generate_assignments = []


for node in ast.walk(tree):

    if not isinstance(
        node,
        (ast.Assign, ast.AnnAssign),
    ):
        continue

    if not isinstance(
        node.value,
        ast.Call,
    ):
        continue

    f = node.value.func

    if (
        isinstance(f, ast.Attribute)
        and f.attr == "generate"
    ):

        names = target_names(node)

        if len(names) == 1:

            generate_assignments.append(
                (
                    node,
                    names[0],
                )
            )


if len(generate_assignments) != 1:

    raise RuntimeError(
        "generate assignment became ambiguous after rewrite"
    )


gen_assign, gen_var = (
    generate_assignments[0]
)

lines = text.splitlines()


first_line = lines[
    gen_assign.lineno - 1
]

indent = first_line[
    :len(first_line)
    - len(first_line.lstrip())
]


capture = f'''

{indent}# ------------------------------------------------------------
{indent}# CLEANROOM: preserve the ACTUAL behavior-policy distribution
{indent}# used for each sampled Student action.
{indent}# ------------------------------------------------------------

{indent}_pg_generate_output = {gen_var}

{indent}_pg_behavior_scores = (
{indent}    _pg_generate_output.scores
{indent})

{indent}{gen_var} = (
{indent}    _pg_generate_output.sequences
{indent})

{indent}if (
{indent}    _pg_behavior_scores is None
{indent}    or len(_pg_behavior_scores) == 0
{indent}):

{indent}    raise RuntimeError(
{indent}        "CLEANROOM generate() returned no behavior scores"
{indent}    )


{indent}_pg_response_steps = len(
{indent}    _pg_behavior_scores
{indent})


{indent}_pg_action_ids = {gen_var}[
{indent}    :,
{indent}    -_pg_response_steps:
{indent}]


{indent}_pg_behavior_logp = F.log_softmax(
{indent}    torch.stack(
{indent}        tuple(_pg_behavior_scores),
{indent}        dim=1,
{indent}    ).float(),
{indent}    dim=-1,
{indent})


{indent}_pg_old_action_logp = (
{indent}    _pg_behavior_logp.gather(
{indent}        dim=-1,
{indent}        index=_pg_action_ids.unsqueeze(-1),
{indent}    ).squeeze(-1)
{indent}).detach()
'''


lines[
    gen_assign.end_lineno:
    gen_assign.end_lineno
] = capture.splitlines()


text = "\n".join(lines) + "\n"


###############################################################################
# Locate VERIFIED full-vocab RKL token_kl + its corresponding loss.
###############################################################################

tree = ast.parse(text)

rkl_token_nodes = []


for node in ast.walk(tree):

    if not isinstance(
        node,
        (ast.Assign, ast.AnnAssign),
    ):
        continue

    if "token_kl" not in target_names(node):
        continue

    seg = source_segment(
        text,
        node,
    )

    if (
        "s_logp" in seg
        and "t_logp" in seg
    ):

        rkl_token_nodes.append(
            node
        )


print(
    "RKL_TOKEN_KL_CANDIDATES =",
    [
        (
            x.lineno,
            x.end_lineno,
            source_segment(text, x),
        )
        for x in rkl_token_nodes
    ],
)


if len(rkl_token_nodes) != 1:

    raise RuntimeError(
        "Could not uniquely identify exact-RKL token_kl"
    )


rkl_token = rkl_token_nodes[0]


loss_nodes = []


for node in ast.walk(tree):

    if not isinstance(
        node,
        (ast.Assign, ast.AnnAssign),
    ):
        continue

    if "loss" not in target_names(node):
        continue

    if node.lineno <= rkl_token.end_lineno:
        continue

    if (
        node.lineno
        - rkl_token.end_lineno
        > 40
    ):
        continue

    loss_nodes.append(
        node
    )


loss_nodes.sort(
    key=lambda x: x.lineno
)


if not loss_nodes:

    raise RuntimeError(
        "No nearby RKL loss after token_kl"
    )


rkl_loss = loss_nodes[0]


print(
    "RKL_LOSS_SELECTED =",
    rkl_loss.lineno,
    "-",
    rkl_loss.end_lineno,
)

print(
    source_segment(
        text,
        rkl_loss,
    )
)


###############################################################################
# Replace ONLY the old RKL token-loss block.
#
# s_logp / t_logp construction is inherited untouched from the validated
# exact-RKL implementation.
###############################################################################

lines = text.splitlines()

line0 = lines[
    rkl_token.lineno - 1
]

indent = line0[
    :len(line0)
    - len(line0.lstrip())
]


pg_block = f'''
{indent}# ============================================================
{indent}# CLEAN-ROOM SAMPLED-K1 PG-RKL
{indent}#
{indent}# action a_t ~ behavior Student policy
{indent}#
{indent}# k1 = stopgrad(
{indent}#     log pi_student(a_t | state_t)
{indent}#     -
{indent}#     log pi_teacher(a_t | state_t)
{indent}# )
{indent}#
{indent}# advantage = -k1
{indent}#
{indent}# PPO-style importance ratio:
{indent}#
{indent}# ratio = pi_current(a_t) / pi_behavior(a_t)
{indent}# ============================================================

{indent}_pg_R = int(
{indent}    _pg_response_steps
{indent})


{indent}if tuple(
{indent}    s_logp.shape
{indent}) != tuple(
{indent}    t_logp.shape
{indent}):

{indent}    raise RuntimeError(
{indent}        "CLEANROOM student/teacher logprob shape mismatch: "
{indent}        f"student={{tuple(s_logp.shape)}} "
{indent}        f"teacher={{tuple(t_logp.shape)}}"
{indent}    )


{indent}if int(
{indent}    s_logp.shape[1]
{indent}) < _pg_R:

{indent}    raise RuntimeError(
{indent}        "CLEANROOM response logits shorter than rollout: "
{indent}        f"logits={{int(s_logp.shape[1])}} "
{indent}        f"rollout={{_pg_R}}"
{indent}    )


{indent}# Exact response tail.
{indent}_pg_s_logp = s_logp[
{indent}    :,
{indent}    -_pg_R:,
{indent}    :
{indent}]


{indent}_pg_t_logp = t_logp[
{indent}    :,
{indent}    -_pg_R:,
{indent}    :
{indent}]


{indent}s_action_logp = (
{indent}    _pg_s_logp.gather(
{indent}        dim=-1,
{indent}        index=_pg_action_ids.unsqueeze(-1),
{indent}    ).squeeze(-1)
{indent})


{indent}t_action_logp = (
{indent}    _pg_t_logp.gather(
{indent}        dim=-1,
{indent}        index=_pg_action_ids.unsqueeze(-1),
{indent}    ).squeeze(-1)
{indent})


{indent}if not torch.isfinite(
{indent}    s_action_logp
{indent}).all():

{indent}    raise RuntimeError(
{indent}        "CLEANROOM non-finite student sampled logprob"
{indent}    )


{indent}if not torch.isfinite(
{indent}    t_action_logp
{indent}).all():

{indent}    raise RuntimeError(
{indent}        "CLEANROOM non-finite teacher sampled logprob"
{indent}    )


{indent}# ------------------------------------------------------------
{indent}# Behavior-policy / recomputed-policy audit.
{indent}# ------------------------------------------------------------

{indent}_pg_behavior_abs_diff = (
{indent}    s_action_logp.detach().float()
{indent}    -
{indent}    _pg_old_action_logp.detach().float()
{indent}).abs()


{indent}_pg_alignment_mae = float(
{indent}    _pg_behavior_abs_diff.mean().item()
{indent})


{indent}_pg_alignment_max = float(
{indent}    _pg_behavior_abs_diff.max().item()
{indent})


{indent}if _pg_alignment_mae > 0.20:

{indent}    raise RuntimeError(
{indent}        "CLEANROOM behavior alignment failed: "
{indent}        f"mae={{_pg_alignment_mae}} "
{indent}        f"max={{_pg_alignment_max}}"
{indent}    )


{indent}if rank == 0 and global_step == 0:

{indent}    print(
{indent}        "CLEAN_PG_RUNTIME_AUDIT_PASS",
{indent}        {{
{indent}            "alignment": "exact_tail",
{indent}            "mae": _pg_alignment_mae,
{indent}            "max_error": _pg_alignment_max,
{indent}            "generated_steps": _pg_R,
{indent}            "forward_steps": int(
{indent}                s_logp.shape[1]
{indent}            ),
{indent}        }},
{indent}        flush=True,
{indent}    )


{indent}# ------------------------------------------------------------
{indent}# UNIQUE sampled reverse-KL estimator.
{indent}# ------------------------------------------------------------

{indent}k1 = (
{indent}    s_action_logp.detach().float()
{indent}    -
{indent}    t_action_logp.detach().float()
{indent}).detach()


{indent}if not torch.isfinite(
{indent}    k1
{indent}).all():

{indent}    raise RuntimeError(
{indent}        "CLEANROOM non-finite sampled k1"
{indent}    )


{indent}token_kl = k1


{indent}distill_reward = (
{indent}    -k1
{indent}).detach()


{indent}# ------------------------------------------------------------
{indent}# PPO-style importance ratio.
{indent}#
{indent}# In one-step on-policy training this should remain close to 1.
{indent}# Keeping it explicit matches the old-policy/current-policy
{indent}# structure of PG-style OPD.
{indent}# ------------------------------------------------------------

{indent}_pg_log_ratio = (
{indent}    s_action_logp.float()
{indent}    -
{indent}    _pg_old_action_logp.detach().float()
{indent})


{indent}_pg_log_ratio = torch.clamp(
{indent}    _pg_log_ratio,
{indent}    min=-20.0,
{indent}    max=20.0,
{indent})


{indent}importance_ratio = torch.exp(
{indent}    _pg_log_ratio
{indent})


{indent}_pg_clip_eps = 0.20


{indent}pg_unclipped = (
{indent}    importance_ratio
{indent}    * distill_reward
{indent})


{indent}pg_clipped = (
{indent}    torch.clamp(
{indent}        importance_ratio,
{indent}        1.0 - _pg_clip_eps,
{indent}        1.0 + _pg_clip_eps,
{indent}    )
{indent}    * distill_reward
{indent})


{indent}loss = -torch.minimum(
{indent}    pg_unclipped,
{indent}    pg_clipped,
{indent}).mean()
'''


replacement = pg_block.splitlines()


lines[
    rkl_token.lineno - 1:
    rkl_loss.end_lineno
] = replacement


text = "\n".join(
    lines
) + "\n"


###############################################################################
# Clean old human-readable objective labels.
###############################################################################

label_replacements = {
    "D_KL(Student || Teacher)":
        "sampled-token k1 PG-RKL [clean-room, behavior-matched]",

    "D_KL(Student||Teacher)":
        "sampled-token k1 PG-RKL [clean-room, behavior-matched]",

    "reverse KL":
        "sampled-k1 PG-RKL",
}


for old, new in label_replacements.items():

    text = text.replace(
        old,
        new,
    )


###############################################################################
# Final AST verification.
###############################################################################

tree = ast.parse(text)


def assignment_nodes(name):

    out = []

    for node in ast.walk(tree):

        if not isinstance(
            node,
            (ast.Assign, ast.AnnAssign),
        ):
            continue

        if name in target_names(node):
            out.append(node)

    return sorted(
        out,
        key=lambda x: x.lineno,
    )


k1_assignments = assignment_nodes(
    "k1"
)

token_kl_assignments = assignment_nodes(
    "token_kl"
)


pg_loss_assignments = []


for node in assignment_nodes(
    "loss"
):

    seg = source_segment(
        text,
        node,
    )

    if (
        "pg_unclipped" in seg
        and "pg_clipped" in seg
    ):

        pg_loss_assignments.append(
            node
        )


print()
print("CLEANROOM OBJECTIVE COUNTS")

print(
    "K1_ASSIGNMENT_COUNT =",
    len(k1_assignments),
)

print(
    "TOKEN_KL_ASSIGNMENT_COUNT =",
    len(token_kl_assignments),
)

print(
    "PG_LOSS_ASSIGNMENT_COUNT =",
    len(pg_loss_assignments),
)


if len(k1_assignments) != 1:

    raise RuntimeError(
        "Cleanroom trainer must contain exactly one k1 assignment"
    )


if len(token_kl_assignments) != 1:

    raise RuntimeError(
        "Cleanroom trainer must contain exactly one token_kl assignment"
    )


if len(pg_loss_assignments) != 1:

    raise RuntimeError(
        "Cleanroom trainer must contain exactly one PG loss"
    )


checks = {
    "neutral temperature":
        "temperature=1.0" in text,

    "neutral top_p":
        "top_p=1.0" in text,

    "neutral top_k":
        "top_k=0" in text,

    "generation scores":
        "output_scores=True" in text,

    "return generate dict":
        "return_dict_in_generate=True"
        in text,

    "behavior old logprob":
        "_pg_old_action_logp"
        in text,

    "runtime alignment":
        "CLEAN_PG_RUNTIME_AUDIT_PASS"
        in text,

    "sampled k1":
        "k1 = (" in text,

    "teacher action logprob":
        "t_action_logp"
        in text,

    "student action logprob":
        "s_action_logp"
        in text,

    "importance ratio":
        "importance_ratio"
        in text,

    "clip":
        "_pg_clip_eps = 0.20"
        in text,

    "policy loss":
        "pg_unclipped"
        in text
        and "pg_clipped"
        in text,
}


print()
print("CLEANROOM SCIENTIFIC CHECK")


for name, ok in checks.items():

    print(
        f"{name:30s} = {ok}"
    )


if not all(
    checks.values()
):

    raise RuntimeError(
        "Cleanroom scientific verification failed"
    )


dst.write_text(
    text,
    encoding="utf-8",
)


print()
print(
    "CLEANROOM_PGRKL_TRAINER_BUILD_PASS"
)
PY


python -m py_compile "$CLEAN_TRAINER"

echo "CLEANROOM_TRAINER_COMPILE_PASS"


###############################################################################
# STAGE 3 — BUILD FROZEN 208-SAMPLE PREFLIGHT FROM THAT EXACT CLEAN TRAINER
###############################################################################

echo
echo "===== STAGE 3/7: BUILD PREFLIGHT ====="

export CLEAN_TRAINER
export PREFLIGHT_TRAINER

python - <<'PY'
import ast
import os
from pathlib import Path


src = Path(os.environ["CLEAN_TRAINER"])
dst = Path(os.environ["PREFLIGHT_TRAINER"])

text = src.read_text(
    encoding="utf-8"
)

tree = ast.parse(text)


def targets(node):

    if isinstance(node, ast.Assign):
        xs = node.targets

    elif isinstance(node, ast.AnnAssign):
        xs = [node.target]

    else:
        return []

    out = []

    for x in xs:

        if isinstance(x, ast.Name):
            out.append(x.id)

    return out


def seg(node):

    return (
        ast.get_source_segment(
            text,
            node,
        )
        or ""
    )


###############################################################################
# Find unique clean PG loss.
###############################################################################

loss_candidates = []


for node in ast.walk(tree):

    if not isinstance(
        node,
        (ast.Assign, ast.AnnAssign),
    ):
        continue

    if "loss" not in targets(node):
        continue

    s = seg(node)

    if (
        "pg_unclipped" in s
        and "pg_clipped" in s
    ):

        loss_candidates.append(
            node
        )


if len(loss_candidates) != 1:

    raise RuntimeError(
        "Preflight could not uniquely find clean PG loss"
    )


pg_loss = loss_candidates[0]


###############################################################################
# Locate optimizer.zero_grad after PG loss.
###############################################################################

lines = text.splitlines()

optimizer_lines = [
    i
    for i, line in enumerate(
        lines,
        start=1,
    )
    if (
        "optimizer.zero_grad"
        in line
        and i > pg_loss.end_lineno
    )
]


if not optimizer_lines:

    raise RuntimeError(
        "No optimizer.zero_grad after clean PG loss"
    )


optimizer_line = optimizer_lines[0]


line_text = lines[
    optimizer_line - 1
]

indent = line_text[
    :len(line_text)
    - len(line_text.lstrip())
]


###############################################################################
# Frozen mathematical preflight.
###############################################################################

block = f'''
{indent}# ============================================================
{indent}# CLEAN PG-RKL FROZEN PREFLIGHT
{indent}#
{indent}# 13 synchronized local batches * 16 ranks = 208 sentences.
{indent}#
{indent}# Absolutely no backward / optimizer / scheduler occurs.
{indent}# ============================================================

{indent}if "_clean_preflight_batches" not in locals():

{indent}    _clean_preflight_batches = 0

{indent}    _clean_preflight_sample_sum = 0.0

{indent}    _clean_preflight_exact_sum = 0.0

{indent}    _clean_preflight_behavior_sum = 0.0

{indent}    _clean_preflight_ratio_sum = 0.0

{indent}    _clean_preflight_count = 0.0


{indent}_clean_exact_rkl = (
{indent}    _pg_s_logp.float().exp()
{indent}    * (
{indent}        _pg_s_logp.float()
{indent}        -
{indent}        _pg_t_logp.float()
{indent}    )
{indent}).sum(
{indent}    dim=-1
{indent})


{indent}if not torch.isfinite(
{indent}    _clean_exact_rkl
{indent}).all():

{indent}    raise RuntimeError(
{indent}        "CLEAN PREFLIGHT non-finite exact RKL"
{indent}    )


{indent}_clean_n = float(
{indent}    k1.numel()
{indent})


{indent}_clean_reduce = torch.stack(
{indent}    [
{indent}        k1.float().sum(),
{indent}        _clean_exact_rkl.float().sum(),
{indent}        _pg_behavior_abs_diff.float().sum(),
{indent}        importance_ratio.detach().float().sum(),
{indent}        torch.tensor(
{indent}            _clean_n,
{indent}            dtype=torch.float32,
{indent}            device=device,
{indent}        ),
{indent}    ]
{indent})


{indent}dist.all_reduce(
{indent}    _clean_reduce,
{indent}    op=dist.ReduceOp.SUM,
{indent})


{indent}if rank == 0:

{indent}    _clean_preflight_sample_sum += float(
{indent}        _clean_reduce[0].item()
{indent}    )

{indent}    _clean_preflight_exact_sum += float(
{indent}        _clean_reduce[1].item()
{indent}    )

{indent}    _clean_preflight_behavior_sum += float(
{indent}        _clean_reduce[2].item()
{indent}    )

{indent}    _clean_preflight_ratio_sum += float(
{indent}        _clean_reduce[3].item()
{indent}    )

{indent}    _clean_preflight_count += float(
{indent}        _clean_reduce[4].item()
{indent}    )


{indent}_clean_preflight_batches += 1


{indent}if _clean_preflight_batches >= 13:

{indent}    _clean_pass_flag = torch.zeros(
{indent}        1,
{indent}        dtype=torch.int32,
{indent}        device=device,
{indent}    )


{indent}    if rank == 0:

{indent}        c = _clean_preflight_count

{indent}        sampled_mean = (
{indent}            _clean_preflight_sample_sum
{indent}            / c
{indent}        )

{indent}        exact_mean = (
{indent}            _clean_preflight_exact_sum
{indent}            / c
{indent}        )

{indent}        gap = abs(
{indent}            sampled_mean
{indent}            - exact_mean
{indent}        )

{indent}        behavior_mae = (
{indent}            _clean_preflight_behavior_sum
{indent}            / c
{indent}        )

{indent}        ratio_mean = (
{indent}            _clean_preflight_ratio_sum
{indent}            / c
{indent}        )


{indent}        print()
{indent}        print(
{indent}            "=" * 94
{indent}        )

{indent}        print(
{indent}            "CLEAN_PG_PREFLIGHT_FINAL"
{indent}        )

{indent}        print(
{indent}            f"sampled_k1_mean = {{sampled_mean:+.8f}}"
{indent}        )

{indent}        print(
{indent}            f"exact_rkl_mean = {{exact_mean:+.8f}}"
{indent}        )

{indent}        print(
{indent}            f"sample_exact_gap = {{gap:.8f}}"
{indent}        )

{indent}        print(
{indent}            f"behavior_forward_mae = {{behavior_mae:.8f}}"
{indent}        )

{indent}        print(
{indent}            f"importance_ratio_mean = {{ratio_mean:.8f}}"
{indent}        )

{indent}        print(
{indent}            f"sampled_tokens = {{int(c)}}"
{indent}        )


{indent}        clean_ok = (
{indent}            exact_mean >= -0.02
{indent}            and
{indent}            gap < 0.50
{indent}            and
{indent}            behavior_mae < 0.20
{indent}            and
{indent}            0.75 < ratio_mean < 1.25
{indent}        )


{indent}        if clean_ok:

{indent}            _clean_pass_flag.fill_(
{indent}                1
{indent}            )

{indent}            print(
{indent}                "CLEAN_PG_PREFLIGHT_PASS"
{indent}            )

{indent}        else:

{indent}            print(
{indent}                "CLEAN_PG_PREFLIGHT_FAIL"
{indent}            )


{indent}        print(
{indent}            "=" * 94
{indent}        )

{indent}        print(
{indent}            flush=True
{indent}        )


{indent}    dist.broadcast(
{indent}        _clean_pass_flag,
{indent}        src=0,
{indent}    )


{indent}    clean_ok_all = bool(
{indent}        int(
{indent}            _clean_pass_flag.item()
{indent}        )
{indent}    )


{indent}    dist.barrier()

{indent}    dist.destroy_process_group()


{indent}    if not clean_ok_all:

{indent}        raise RuntimeError(
{indent}            "CLEAN PG-RKL mathematical preflight failed"
{indent}        )


{indent}    return


{indent}# Critical:
{indent}# every non-final preflight batch skips the ENTIRE update path.
{indent}continue
'''


lines[
    optimizer_line - 1:
    optimizer_line - 1
] = block.splitlines()


result = "\n".join(
    lines
) + "\n"


###############################################################################
# Syntax/static verification.
###############################################################################

ast.parse(
    result
)


diag_pos = result.index(
    "CLEAN PG-RKL FROZEN PREFLIGHT"
)

continue_pos = result.index(
    "continue",
    diag_pos,
)

optimizer_pos = result.index(
    "optimizer.zero_grad",
    diag_pos,
)


if continue_pos >= optimizer_pos:

    raise RuntimeError(
        "Preflight optimizer bypass is misplaced"
    )


checks = {
    "13 batches":
        "_clean_preflight_batches >= 13"
        in result,

    "sampled k1":
        "sampled_k1_mean"
        in result,

    "exact RKL":
        "exact_rkl_mean"
        in result,

    "gap":
        "sample_exact_gap"
        in result,

    "behavior audit":
        "behavior_forward_mae"
        in result,

    "importance ratio":
        "importance_ratio_mean"
        in result,

    "PASS gate":
        "CLEAN_PG_PREFLIGHT_PASS"
        in result,

    "optimizer bypass":
        continue_pos < optimizer_pos,
}


print("PREFLIGHT CHECK")


for name, ok in checks.items():

    print(
        f"{name:28s} = {ok}"
    )


if not all(
    checks.values()
):

    raise RuntimeError(
        "Preflight verification failed"
    )


dst.write_text(
    result,
    encoding="utf-8",
)


print()
print(
    "CLEANROOM_PREFLIGHT_BUILD_PASS"
)
PY


python -m py_compile "$PREFLIGHT_TRAINER"

echo "CLEANROOM_PREFLIGHT_COMPILE_PASS"


###############################################################################
# STAGE 4 — BUILD FULL MASTER + SHORT PREFLIGHT MASTER
###############################################################################

echo
echo "===== STAGE 4/7: BUILD LAUNCHERS ====="

export MASTER_SRC
export FULL_MASTER
export PREFLIGHT_MASTER
export FULL_NAME
export PREFLIGHT_NAME

python - <<'PY'
import os
import re
from pathlib import Path


src = Path(
    os.environ["MASTER_SRC"]
)

full_dst = Path(
    os.environ["FULL_MASTER"]
)

pre_dst = Path(
    os.environ["PREFLIGHT_MASTER"]
)

full_name = os.environ[
    "FULL_NAME"
]

pre_name = os.environ[
    "PREFLIGHT_NAME"
]


base = src.read_text(
    encoding="utf-8"
)


###############################################################################
# Helper
###############################################################################

def rewrite_master(
    text,
    name,
    trainer_filename,
    port,
):

    name_hits = re.findall(
        r"(?m)^NAME=.*$",
        text,
    )

    trainer_hits = re.findall(
        r"(?m)^TRAINER=.*$",
        text,
    )

    port_hits = re.findall(
        r"--master_port=\d+",
        text,
    )


    if len(name_hits) != 1:
        raise RuntimeError(
            f"NAME count={{len(name_hits)}}"
        )


    if len(trainer_hits) != 1:
        raise RuntimeError(
            f"TRAINER count={{len(trainer_hits)}}"
        )


    if len(port_hits) != 1:
        raise RuntimeError(
            f"PORT count={{len(port_hits)}}"
        )


    text = re.sub(
        r"(?m)^NAME=.*$",
        f'NAME="{name}"',
        text,
        count=1,
    )


    text = re.sub(
        r"(?m)^TRAINER=.*$",
        (
            'TRAINER="$ROOT/scripts/mtpatcher_v4/'
            + trainer_filename
            + '"'
        ),
        text,
        count=1,
    )


    text = re.sub(
        r"--master_port=\d+",
        f"--master_port={port}",
        text,
        count=1,
    )


    return text


###############################################################################
# Full master
###############################################################################

full = rewrite_master(
    base,
    full_name,
    "train_opd_pgrkl_cleanroom_v1_torchnpu.py",
    29649,
)


# Replace inherited result identity.
full = full.replace(
    "OPD-PGRKL-K1-AUDITED-PE3732",
    "OPD-PGRKL-K1-CLEANROOM-PE3732",
)


# Replace inherited run-name literals wherever present.
full = full.replace(
    "opd_torchnpu_pgrkl_k1_audited_pe3732_v2",
    full_name,
)

full = full.replace(
    "opd_torchnpu_pgrkl_k1_audited_pe3732_v1",
    full_name,
)


full_dst.write_text(
    full,
    encoding="utf-8",
)

full_dst.chmod(
    0o755
)


###############################################################################
# Preflight master
###############################################################################

pre = rewrite_master(
    base,
    pre_name,
    "preflight_opd_pgrkl_cleanroom_v1_torchnpu.py",
    29648,
)


lines = pre.splitlines()


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
    "PREFLIGHT_DISTRIBUTED_LAUNCH_CANDIDATES =",
    [
        i + 1
        for i in launch_candidates
    ],
)


if len(
    launch_candidates
) != 1:

    raise RuntimeError(
        "Could not uniquely locate distributed command"
    )


start = launch_candidates[0]

end = start


while (
    end < len(lines) - 1
    and lines[end].rstrip().endswith("\\")
):

    end += 1


command = "\n".join(
    lines[start:end + 1]
)


if "--nproc_per_node=16" not in command:

    raise RuntimeError(
        "Preflight does not use 16 ranks"
    )


if "$TRAINER" not in command:

    raise RuntimeError(
        "Preflight command does not invoke TRAINER"
    )


short = lines[:end + 1]


short.extend(
    [
        "",
        'echo',
        'echo "CLEAN_PG_PREFLIGHT_LAUNCHER_PASS"',
        "",
    ]
)


pre_text = "\n".join(
    short
) + "\n"


pre_dst.write_text(
    pre_text,
    encoding="utf-8",
)

pre_dst.chmod(
    0o755
)


###############################################################################
# Verify.
###############################################################################

checks = {
    "full clean trainer":
        "train_opd_pgrkl_cleanroom_v1_torchnpu.py"
        in full,

    "full 16 NPU":
        "--nproc_per_node=16"
        in full,

    "full fresh port":
        "--master_port=29649"
        in full,

    "full PE3732":
        "pe_k1_clean3732.jsonl"
        in full,

    "full result label":
        "OPD-PGRKL-K1-CLEANROOM-PE3732"
        in full,

    "preflight trainer":
        "preflight_opd_pgrkl_cleanroom_v1_torchnpu.py"
        in pre_text,

    "preflight fresh port":
        "--master_port=29648"
        in pre_text,

    "preflight no eval":
        "FINAL SUMMARY"
        not in pre_text,
}


print()
print("MASTER CHECK")


for name, ok in checks.items():

    print(
        f"{name:28s} = {ok}"
    )


if not all(
    checks.values()
):

    raise RuntimeError(
        "Launcher verification failed"
    )


print()
print(
    "CLEANROOM_LAUNCHERS_BUILD_PASS"
)
PY


bash -n "$FULL_MASTER"
bash -n "$PREFLIGHT_MASTER"

echo "CLEANROOM_LAUNCHERS_SYNTAX_PASS"


###############################################################################
# STAGE 5 — FINAL STATIC SCIENTIFIC AUDIT
###############################################################################

echo
echo "===== STAGE 5/7: FINAL STATIC AUDIT ====="

python - "$CLEAN_TRAINER" <<'PY'
import ast
import sys
from pathlib import Path


p = Path(sys.argv[1])

text = p.read_text(
    encoding="utf-8"
)

tree = ast.parse(
    text
)


def assigned(node, name):

    if isinstance(node, ast.Assign):
        xs = node.targets

    elif isinstance(node, ast.AnnAssign):
        xs = [node.target]

    else:
        return False

    return any(
        isinstance(x, ast.Name)
        and x.id == name
        for x in xs
    )


k1 = [
    n
    for n in ast.walk(tree)
    if isinstance(
        n,
        (ast.Assign, ast.AnnAssign),
    )
    and assigned(n, "k1")
]


token = [
    n
    for n in ast.walk(tree)
    if isinstance(
        n,
        (ast.Assign, ast.AnnAssign),
    )
    and assigned(n, "token_kl")
]


pg_loss = []

for n in ast.walk(tree):

    if not isinstance(
        n,
        (ast.Assign, ast.AnnAssign),
    ):
        continue

    if not assigned(
        n,
        "loss",
    ):
        continue

    seg = (
        ast.get_source_segment(
            text,
            n,
        )
        or ""
    )

    if (
        "pg_unclipped" in seg
        and
        "pg_clipped" in seg
    ):

        pg_loss.append(
            n
        )


print(
    "K1_ASSIGNMENT_COUNT =",
    len(k1),
)

print(
    "TOKEN_KL_ASSIGNMENT_COUNT =",
    len(token),
)

print(
    "PG_LOSS_ASSIGNMENT_COUNT =",
    len(pg_loss),
)


if not (
    len(k1) == 1
    and
    len(token) == 1
    and
    len(pg_loss) == 1
):

    raise RuntimeError(
        "CLEANROOM UNIQUE-OBJECTIVE AUDIT FAILED"
    )


print(
    "CLEANROOM_UNIQUE_OBJECTIVE_AUDIT_PASS"
)
PY


echo
echo "--- key clean-room code ---"

grep -nE \
'CLEAN_PG_RUNTIME_AUDIT_PASS|k1 =|token_kl =|importance_ratio =|pg_unclipped|pg_clipped|loss = -torch.minimum' \
"$CLEAN_TRAINER"


echo
echo "FINAL_STATIC_SCIENTIFIC_AUDIT_PASS"


###############################################################################
# STAGE 6 — RUN 208-SAMPLE FROZEN PREFLIGHT SYNCHRONOUSLY
###############################################################################

echo
echo "===== STAGE 6/7: FROZEN MATHEMATICAL PREFLIGHT ====="

mkdir -p \
"$LOG_ROOT/$EXP"


if bash "$PREFLIGHT_MASTER" \
    > "$PREFLIGHT_LOG" 2>&1; then

    PREFLIGHT_RC=0

else

    PREFLIGHT_RC=$?

fi


echo
echo "======================================================================"
echo "PREFLIGHT RESULT"
echo "======================================================================"


grep -E \
'CLEAN_PG_RUNTIME_AUDIT_PASS|CLEAN_PG_PREFLIGHT|sampled_k1_mean|exact_rkl_mean|sample_exact_gap|behavior_forward_mae|importance_ratio_mean|sampled_tokens|Traceback|RuntimeError:|NameError:|TypeError:|ValueError:' \
"$PREFLIGHT_LOG" \
| tail -n 120


PREFLIGHT_OK=0


if (
    [[ "$PREFLIGHT_RC" -eq 0 ]]
    &&
    grep -q \
        'CLEAN_PG_PREFLIGHT_PASS' \
        "$PREFLIGHT_LOG"
); then

    PREFLIGHT_OK=1

fi


###############################################################################
# STAGE 7 — ONLY LAUNCH FULL RUN IF PREFLIGHT IS MATHEMATICALLY VALID
###############################################################################

echo
echo "===== STAGE 7/7: FULL RUN GATE ====="


if [[ "$PREFLIGHT_OK" -eq 1 ]]; then

    echo "CLEAN PG-RKL PREFLIGHT PASSED."
    echo "Launching full fresh 3-epoch experiment."


    RUNNING="$(
        pgrep -af \
        'train_opd_pgrkl_cleanroom_v1_torchnpu' \
        || true
    )"


    if [[ -n "$RUNNING" ]]; then

        echo
        echo "Existing clean-room training process:"
        echo "$RUNNING"

        echo "DUPLICATE_FULL_LAUNCH_SKIPPED"

    else

        nohup setsid bash "$FULL_MASTER" \
            > "$FULL_LOG" 2>&1 < /dev/null &


        PID=$!


        echo
        echo "======================================================================"
        echo "CLEANROOM PG-RKL FULL RUN STARTED"
        echo "PID=$PID"
        echo "LOG=$FULL_LOG"
        echo "TRAINER=$CLEAN_TRAINER"
        echo "MASTER=$FULL_MASTER"
        echo "======================================================================"

    fi

else

    echo
    echo "======================================================================"
    echo "CLEANROOM FULL TRAINING NOT LAUNCHED"
    echo "Reason: frozen mathematical preflight did not pass."
    echo "PREFLIGHT_RC=$PREFLIGHT_RC"
    echo "PREFLIGHT_LOG=$PREFLIGHT_LOG"
    echo "======================================================================"

fi

