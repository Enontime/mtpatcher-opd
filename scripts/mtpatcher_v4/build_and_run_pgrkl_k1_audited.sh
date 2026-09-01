#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

# 已经完整跑通 PG-RKL 的 trainer，只在它上面修 behavior + alignment。
SRC_TRAINER="$ROOT/scripts/mtpatcher_v4/train_opd_pgrkl_k1_torchnpu_v4.py"

DST_TRAINER="$ROOT/scripts/mtpatcher_v4/train_opd_pgrkl_k1_audited_torchnpu.py"

SRC_MASTER="$ROOT/scripts/mtpatcher_v4/run_opd_pgrkl_k1_v5_oneclick.sh"

DST_MASTER="$ROOT/scripts/mtpatcher_v4/run_opd_pgrkl_k1_audited_oneclick.sh"

OLD_NAME="opd_torchnpu_pgrkl_k1_pe3732_v5"

NEW_NAME="opd_torchnpu_pgrkl_k1_audited_pe3732_v1"

LOG="$LOG_ROOT/$EXP/${NEW_NAME}.log"


echo "======================================================================"
echo "MT-PATCHER — BEHAVIOR-MATCHED + ALIGNMENT-AUDITED PG-RKL"
date
echo "======================================================================"


###############################################################################
# STAGE 1 — VERIFY SOURCE
###############################################################################

echo
echo "===== STAGE 1/6: VERIFY SOURCE ====="

test -f "$SRC_TRAINER"
test -f "$SRC_MASTER"

python -m py_compile "$SRC_TRAINER"

echo "SRC_TRAINER=$SRC_TRAINER"
echo "SRC_MASTER=$SRC_MASTER"

echo "SOURCE_PASS"


###############################################################################
# STAGE 2 — BUILD AUDITED TRAINER
###############################################################################

echo
echo "===== STAGE 2/6: BUILD AUDITED TRAINER ====="

export SRC_TRAINER
export DST_TRAINER

python - <<'PY'
import ast
import copy
import os
import re
from pathlib import Path


src = Path(os.environ["SRC_TRAINER"])
dst = Path(os.environ["DST_TRAINER"])

source = src.read_text(
    encoding="utf-8"
)

tree = ast.parse(source)

lines_keep = source.splitlines(
    keepends=True
)

lines_plain = source.splitlines()


print("SOURCE =", src)
print("TARGET =", dst)


###############################################################################
# Utilities
###############################################################################

def abs_offset(lineno, col):

    return (
        sum(
            len(x)
            for x in lines_keep[:lineno - 1]
        )
        + col
    )


def assigned_names(node):

    targets = []

    if isinstance(node, ast.Assign):
        targets = node.targets

    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]

    result = []

    for target in targets:

        if isinstance(target, ast.Name):

            result.append(
                target.id
            )

        elif isinstance(
            target,
            (ast.Tuple, ast.List),
        ):

            for elt in target.elts:

                if isinstance(
                    elt,
                    ast.Name,
                ):
                    result.append(
                        elt.id
                    )

    return result


def assignment_value(node):

    if isinstance(node, ast.Assign):
        return node.value

    if isinstance(node, ast.AnnAssign):
        return node.value

    return None


def attr_chain(node):

    names = []

    while isinstance(
        node,
        ast.Attribute,
    ):

        names.append(
            node.attr
        )

        node = node.value

    if isinstance(
        node,
        ast.Name,
    ):

        names.append(
            node.id
        )

    return list(
        reversed(names)
    )


def contains_generate(node):

    if node is None:
        return False

    for sub in ast.walk(node):

        if not isinstance(
            sub,
            ast.Call,
        ):
            continue

        if attr_chain(
            sub.func
        )[-3:] == [
            "student",
            "module",
            "generate",
        ]:

            return True

    return False


###############################################################################
# Verify the existing PG-RKL implementation.
###############################################################################

required = {
    "student generate":
        "student.module.generate"
        in source,

    "student sampled logp":
        "s_action_logp = s_logp.gather"
        in source,

    "teacher sampled logp":
        "t_action_logp = t_logp.gather"
        in source,

    "k1":
        "k1 = ("
        in source,

    "reward":
        "distill_reward = (-k1).detach()"
        in source,

    "PG gradient path":
        "s_action_logp.float()"
        in source,

    "metric":
        "token_kl = k1"
        in source,

    "finite loss":
        "torch.isfinite(loss)"
        in source,

    "gradient clipping":
        "clip_grad_norm_"
        in source,

    "distributed sampler":
        "DistributedSampler"
        in source,
}


print()
print("SOURCE PG-RKL CHECK")

for name, ok in required.items():

    print(
        f"{name:30s} = {ok}"
    )


if not all(
    required.values()
):

    raise RuntimeError(
        "Source is not the verified PG-RKL trainer"
    )


###############################################################################
# Locate:
#
#   generated = student.module.generate(...)
#
###############################################################################

generate_assignments = []


for node in ast.walk(tree):

    if not isinstance(
        node,
        (ast.Assign, ast.AnnAssign),
    ):
        continue

    value = assignment_value(
        node
    )

    if not contains_generate(
        value
    ):
        continue

    names = assigned_names(
        node
    )

    if len(names) != 1:

        raise RuntimeError(
            "generate() assignment must have exactly one target: "
            + str(names)
        )

    generate_assignments.append(
        (
            node,
            names[0],
        )
    )


if len(
    generate_assignments
) != 1:

    raise RuntimeError(
        "Expected exactly one student generate assignment, "
        f"found {len(generate_assignments)}"
    )


generate_assignment, generated_var = (
    generate_assignments[0]
)


###############################################################################
# Find the exact Call object inside the assignment.
###############################################################################

generate_calls = []


for sub in ast.walk(
    assignment_value(
        generate_assignment
    )
):

    if not isinstance(
        sub,
        ast.Call,
    ):
        continue

    if attr_chain(
        sub.func
    )[-3:] == [
        "student",
        "module",
        "generate",
    ]:

        generate_calls.append(
            sub
        )


if len(generate_calls) != 1:

    raise RuntimeError(
        "Expected one student.module.generate Call"
    )


generate_call = generate_calls[0]


print()
print(
    "GENERATED_VAR =",
    generated_var,
)

print(
    "GENERATE ASSIGNMENT LINES =",
    generate_assignment.lineno,
    "-",
    generate_assignment.end_lineno,
)


###############################################################################
# Force a behavior policy that exactly corresponds to raw student logits.
#
# Crucially, we do NOT care whether the original code used:
#
#     temperature=args.temperature
#
# or:
#
#     temperature=0.7
#
# AST replaces the expression itself.
###############################################################################

new_call = copy.deepcopy(
    generate_call
)


def set_kw(call, name, value):

    existing = None

    for kw in call.keywords:

        if kw.arg == name:

            existing = kw
            break

    node = ast.Constant(
        value=value
    )

    if existing is None:

        call.keywords.append(
            ast.keyword(
                arg=name,
                value=node,
            )
        )

    else:

        existing.value = node


set_kw(
    new_call,
    "do_sample",
    True,
)

set_kw(
    new_call,
    "temperature",
    1.0,
)

set_kw(
    new_call,
    "top_p",
    1.0,
)

set_kw(
    new_call,
    "top_k",
    0,
)

# Eliminate another possible inherited sampling transformation.
set_kw(
    new_call,
    "repetition_penalty",
    1.0,
)

# We need generation-time scores to PROVE alignment.
set_kw(
    new_call,
    "return_dict_in_generate",
    True,
)

set_kw(
    new_call,
    "output_scores",
    True,
)

# Keep output_scores as the actual processed scores.
set_kw(
    new_call,
    "renormalize_logits",
    False,
)


ast.fix_missing_locations(
    new_call
)


new_call_source = ast.unparse(
    new_call
)


print()
print(
    "PATCHED GENERATE CALL:"
)

print(
    new_call_source
)


###############################################################################
# Locate token_kl and its corresponding loss via AST.
###############################################################################

token_nodes = []


for node in ast.walk(tree):

    if not isinstance(
        node,
        (ast.Assign, ast.AnnAssign),
    ):
        continue

    if (
        "token_kl"
        in assigned_names(node)
    ):

        token_nodes.append(
            node
        )


if len(token_nodes) != 1:

    raise RuntimeError(
        "Expected exactly one token_kl assignment, "
        f"found {len(token_nodes)}"
    )


token_node = token_nodes[0]


loss_nodes = []


for node in ast.walk(tree):

    if not isinstance(
        node,
        (ast.Assign, ast.AnnAssign),
    ):
        continue

    if (
        "loss"
        not in assigned_names(node)
    ):
        continue

    if (
        node.lineno
        > token_node.end_lineno
        and
        node.lineno
        <= token_node.end_lineno + 40
    ):

        loss_nodes.append(
            node
        )


if len(loss_nodes) != 1:

    raise RuntimeError(
        "Expected exactly one loss assignment after token_kl, "
        f"found {len(loss_nodes)}"
    )


loss_node = loss_nodes[0]


print()
print(
    "TOKEN_KL LINES =",
    token_node.lineno,
    "-",
    token_node.end_lineno,
)

print(
    "LOSS LINES =",
    loss_node.lineno,
    "-",
    loss_node.end_lineno,
)


###############################################################################
# Indentation for inserted code.
###############################################################################

token_line = lines_plain[
    token_node.lineno - 1
]

indent = token_line[
    :len(token_line)
    - len(token_line.lstrip())
]


gen_line = lines_plain[
    generate_assignment.lineno - 1
]

gen_indent = gen_line[
    :len(gen_line)
    - len(gen_line.lstrip())
]


###############################################################################
# Immediately after generate:
#
# Preserve:
#   behavior scores
#
# Restore:
#   generated = Tensor
#
# so the already-successful downstream pipeline keeps working.
###############################################################################

post_generate = (
    "\n"
    + gen_indent
    + "# ------------------------------------------------------------\n"
    + gen_indent
    + "# Preserve generation-time distribution for alignment audit.\n"
    + gen_indent
    + "# ------------------------------------------------------------\n"
    + gen_indent
    + "_pg_generate_output = "
    + generated_var
    + "\n"
    + gen_indent
    + "\n"
    + gen_indent
    + "if not hasattr(_pg_generate_output, 'sequences'):\n"
    + gen_indent
    + "    raise RuntimeError(\n"
    + gen_indent
    + "        'return_dict_in_generate=True did not return sequences'\n"
    + gen_indent
    + "    )\n"
    + gen_indent
    + "\n"
    + gen_indent
    + "_pg_behavior_scores = _pg_generate_output.scores\n"
    + gen_indent
    + "\n"
    + gen_indent
    + "if _pg_behavior_scores is None or len(_pg_behavior_scores) == 0:\n"
    + gen_indent
    + "    raise RuntimeError(\n"
    + gen_indent
    + "        'generate() returned no per-step behavior scores'\n"
    + gen_indent
    + "    )\n"
    + gen_indent
    + "\n"
    + gen_indent
    + generated_var
    + " = _pg_generate_output.sequences\n"
)


###############################################################################
# New PG block.
#
# This performs an explicit alignment audit every batch.
#
# generation scores are the ground truth distribution used at sampling time.
#
# We compare them with possible slices of recomputed s_logp.
#
# The slice whose sampled-action log-probs agree with generation scores is
# selected automatically.
#
# If no candidate agrees sufficiently well:
#
#     STOP BEFORE optimizer update.
###############################################################################

pg_block = f'''
{indent}# ================================================================
{indent}# BEHAVIOR-MATCHED + ALIGNMENT-AUDITED sampled-k1 PG-RKL
{indent}#
{indent}# Rollout:
{indent}#   temperature = 1.0
{indent}#   top_p       = 1.0
{indent}#   top_k       = 0
{indent}#   repetition_penalty = 1.0
{indent}#
{indent}# Generation-time scores are retained and used to audit causal
{indent}# token/logit alignment before computing the PG update.
{indent}# ================================================================

{indent}_pg_rollout = {generated_var}

{indent}if not torch.is_tensor(_pg_rollout):
{indent}    raise RuntimeError(
{indent}        'Normalized generate output is not a Tensor'
{indent}    )

{indent}_pg_scores = _pg_behavior_scores

{indent}_pg_response_steps = len(
{indent}    _pg_scores
{indent})

{indent}if _pg_response_steps <= 0:
{indent}    raise RuntimeError(
{indent}        'Zero generated response steps'
{indent}    )

{indent}# Each entry in generate().scores corresponds exactly to one sampled
{indent}# new token. Hence the final R tokens in sequences are the true
{indent}# sampled actions. No heuristic is needed here.
{indent}_pg_action_ids = _pg_rollout[
{indent}    :,
{indent}    -_pg_response_steps:
{indent}].to(
{indent}    device=s_logp.device,
{indent}    dtype=torch.long,
{indent})

{indent}_pg_behavior_logits = torch.stack(
{indent}    [
{indent}        score.float()
{indent}        for score in _pg_scores
{indent}    ],
{indent}    dim=1,
{indent})

{indent}if int(_pg_behavior_logits.shape[1]) != _pg_response_steps:
{indent}    raise RuntimeError(
{indent}        'Behavior score stacking failure'
{indent}    )

{indent}if int(_pg_behavior_logits.shape[0]) != int(_pg_action_ids.shape[0]):
{indent}    raise RuntimeError(
{indent}        'Behavior/action batch mismatch'
{indent}    )

{indent}_pg_behavior_logp = F.log_softmax(
{indent}    _pg_behavior_logits,
{indent}    dim=-1,
{indent})

{indent}_pg_behavior_action_logp = _pg_behavior_logp.gather(
{indent}    dim=-1,
{indent}    index=_pg_action_ids.unsqueeze(-1),
{indent}).squeeze(-1)

{indent}if not torch.isfinite(
{indent}    _pg_behavior_action_logp
{indent}).all():
{indent}    raise RuntimeError(
{indent}        'Non-finite generation-time action logprob'
{indent}    )

{indent}# ------------------------------------------------------------
{indent}# Causal alignment audit.
{indent}#
{indent}# Candidate slices cover:
{indent}#   exact response slice
{indent}#   one-token-left shift
{indent}#   one-token-right shift
{indent}#   generic head/tail slicing
{indent}#
{indent}# The generation-time action log-probability tells us objectively
{indent}# which student-forward position predicts each sampled token.
{indent}# ------------------------------------------------------------

{indent}_pg_S = int(
{indent}    s_logp.shape[1]
{indent})

{indent}_pg_R = int(
{indent}    _pg_response_steps
{indent})

{indent}if tuple(s_logp.shape) != tuple(t_logp.shape):
{indent}    raise RuntimeError(
{indent}        'Student/teacher logprob tensor shape mismatch: '
{indent}        f'student={{tuple(s_logp.shape)}} '
{indent}        f'teacher={{tuple(t_logp.shape)}}'
{indent}    )

{indent}if _pg_S < _pg_R:
{indent}    raise RuntimeError(
{indent}        'Recomputed response logits are shorter than generation: '
{indent}        f'logit_steps={{_pg_S}} generated_steps={{_pg_R}}'
{indent}    )

{indent}_pg_candidates = []

{indent}def _pg_test_alignment(name, s_candidate, t_candidate):

{indent}    if int(s_candidate.shape[1]) != _pg_R:
{indent}        return

{indent}    sampled_forward = s_candidate.gather(
{indent}        dim=-1,
{indent}        index=_pg_action_ids.unsqueeze(-1),
{indent}    ).squeeze(-1)

{indent}    diff = (
{indent}        sampled_forward.detach().float()
{indent}        - _pg_behavior_action_logp.detach().float()
{indent}    ).abs()

{indent}    mae = float(
{indent}        diff.mean().item()
{indent}    )

{indent}    maxe = float(
{indent}        diff.max().item()
{indent}    )

{indent}    _pg_candidates.append(
{indent}        (
{indent}            mae,
{indent}            maxe,
{indent}            name,
{indent}            s_candidate,
{indent}            t_candidate,
{indent}        )
{indent}    )

{indent}# Exact shape.
{indent}if _pg_S == _pg_R:

{indent}    _pg_test_alignment(
{indent}        'exact',
{indent}        s_logp,
{indent}        t_logp,
{indent}    )

{indent}# Generic tail.
{indent}_pg_test_alignment(
{indent}    'tail',
{indent}    s_logp[:, -_pg_R:, :],
{indent}    t_logp[:, -_pg_R:, :],
{indent})

{indent}# Generic head.
{indent}_pg_test_alignment(
{indent}    'head',
{indent}    s_logp[:, :_pg_R, :],
{indent}    t_logp[:, :_pg_R, :],
{indent})

{indent}# Explicit one-token ambiguity.
{indent}if _pg_S == _pg_R + 1:

{indent}    _pg_test_alignment(
{indent}        'drop_first',
{indent}        s_logp[:, 1:, :],
{indent}        t_logp[:, 1:, :],
{indent}    )

{indent}    _pg_test_alignment(
{indent}        'drop_last',
{indent}        s_logp[:, :-1, :],
{indent}        t_logp[:, :-1, :],
{indent}    )

{indent}if not _pg_candidates:

{indent}    raise RuntimeError(
{indent}        'No valid causal-alignment candidate'
{indent}    )

{indent}_pg_candidates.sort(
{indent}    key=lambda item: item[0]
{indent})

{indent}(
{indent}    _pg_alignment_mae,
{indent}    _pg_alignment_max,
{indent}    _pg_alignment_name,
{indent}    _pg_s_logp,
{indent}    _pg_t_logp,
{indent}) = _pg_candidates[0]

{indent}# With neutral generation warpers and the same model parameters,
{indent}# generation-time sampled-action logprob and teacher-forced forward
{indent}# logprob should agree closely. A large discrepancy means the
{indent}# estimator is not scientifically valid, so stop immediately.
{indent}if _pg_alignment_mae > 0.20:

{indent}    _pg_debug = [
{indent}        (
{indent}            name,
{indent}            mae,
{indent}            maxe,
{indent}        )
{indent}        for (
{indent}            mae,
{indent}            maxe,
{indent}            name,
{indent}            _,
{indent}            __,
{indent}        ) in _pg_candidates
{indent}    ]

{indent}    raise RuntimeError(
{indent}        'PG-RKL ALIGNMENT AUDIT FAILED: '
{indent}        f'best={{_pg_alignment_name}} '
{indent}        f'mae={{_pg_alignment_mae:.6f}} '
{indent}        f'max={{_pg_alignment_max:.6f}} '
{indent}        f'candidates={{_pg_debug}}'
{indent}    )

{indent}if rank == 0 and step == 0:

{indent}    print(
{indent}        'PG_ALIGNMENT_AUDIT_PASS',
{indent}        {{
{indent}            'alignment': _pg_alignment_name,
{indent}            'mae': _pg_alignment_mae,
{indent}            'max_error': _pg_alignment_max,
{indent}            'generated_steps': _pg_R,
{indent}            'forward_steps': _pg_S,
{indent}        }},
{indent}        flush=True,
{indent}    )

{indent}# ------------------------------------------------------------
{indent}# Gather log-probabilities of the ACTUAL sampled student actions
{indent}# using the alignment that was just objectively verified.
{indent}# ------------------------------------------------------------

{indent}s_action_logp = _pg_s_logp.gather(
{indent}    dim=-1,
{indent}    index=_pg_action_ids.unsqueeze(-1),
{indent}).squeeze(-1)

{indent}t_action_logp = _pg_t_logp.gather(
{indent}    dim=-1,
{indent}    index=_pg_action_ids.unsqueeze(-1),
{indent}).squeeze(-1)

{indent}if not torch.isfinite(
{indent}    s_action_logp
{indent}).all():
{indent}    raise RuntimeError(
{indent}        'Non-finite student sampled logprob'
{indent}    )

{indent}if not torch.isfinite(
{indent}    t_action_logp
{indent}).all():
{indent}    raise RuntimeError(
{indent}        'Non-finite teacher sampled logprob'
{indent}    )

{indent}# ------------------------------------------------------------
{indent}# Sampled k1 reverse-KL estimator:
{indent}#
{indent}#   k1 = stopgrad(log pi_S(a|s) - log pi_T(a|s))
{indent}#
{indent}# Since actions now come from the same pi_S represented by
{indent}# s_action_logp, E[k1] estimates KL(pi_S || pi_T).
{indent}# ------------------------------------------------------------

{indent}k1 = (
{indent}    s_action_logp.detach().float()
{indent}    - t_action_logp.detach().float()
{indent}).detach()

{indent}if not torch.isfinite(
{indent}    k1
{indent}).all():
{indent}    raise RuntimeError(
{indent}        'Non-finite sampled k1'
{indent}    )

{indent}token_kl = k1

{indent}distill_reward = (
{indent}    -k1
{indent}).detach()

{indent}# Score-function policy-gradient objective.
{indent}loss = -(
{indent}    distill_reward
{indent}    * s_action_logp.float()
{indent}).mean()
'''.strip("\n")


###############################################################################
# Three edits are applied to ORIGINAL SOURCE using ORIGINAL AST positions:
#
# 1. replace generate Call
# 2. insert normalization immediately after generate assignment
# 3. replace old PG loss block
#
# Apply from highest source offset downward.
###############################################################################

replacements = []


# A. generate Call
replacements.append(
    (
        abs_offset(
            generate_call.lineno,
            generate_call.col_offset,
        ),
        abs_offset(
            generate_call.end_lineno,
            generate_call.end_col_offset,
        ),
        new_call_source,
        "generate_call",
    )
)


# B. post-generate normalization
assignment_end = abs_offset(
    generate_assignment.end_lineno,
    generate_assignment.end_col_offset,
)

replacements.append(
    (
        assignment_end,
        assignment_end,
        post_generate,
        "post_generate",
    )
)


# C. PG objective block
replacements.append(
    (
        abs_offset(
            token_node.lineno,
            token_node.col_offset,
        ),
        abs_offset(
            loss_node.end_lineno,
            loss_node.end_col_offset,
        ),
        pg_block,
        "pg_block",
    )
)


result = source


for start, end, replacement, name in sorted(
    replacements,
    key=lambda item: item[0],
    reverse=True,
):

    print(
        "APPLY PATCH:",
        name,
    )

    result = (
        result[:start]
        + replacement
        + result[end:]
    )


###############################################################################
# Clean labels.
###############################################################################

result = result.replace(
    "D_KL(Teacher || Student)",
    "sampled-token k1 PG-RKL [behavior-matched, alignment-audited]",
)

result = result.replace(
    "token_mean_sampled_k1",
    "token_mean_sampled_k1_audited",
)


###############################################################################
# Verify generated source is syntactically valid BEFORE saving.
###############################################################################

ast.parse(
    result
)


###############################################################################
# Final scientific checks.
###############################################################################

final_checks = {
    "behavior T=1":
        "temperature=1.0"
        in result,

    "behavior top_p=1":
        "top_p=1.0"
        in result,

    "behavior top_k=0":
        "top_k=0"
        in result,

    "output scores":
        "output_scores=True"
        in result,

    "return dict":
        "return_dict_in_generate=True"
        in result,

    "behavior score preservation":
        "_pg_behavior_scores"
        in result,

    "actual sampled actions":
        "_pg_action_ids"
        in result,

    "alignment audit":
        "PG_ALIGNMENT_AUDIT_PASS"
        in result,

    "alignment fail guard":
        "PG-RKL ALIGNMENT AUDIT FAILED"
        in result,

    "student sampled logp":
        "s_action_logp"
        in result,

    "teacher sampled logp":
        "t_action_logp"
        in result,

    "sampled k1":
        "k1 = ("
        in result,

    "PG reward":
        "distill_reward"
        in result,

    "finite loss":
        "torch.isfinite(loss)"
        in result,

    "gradient clipping":
        "clip_grad_norm_"
        in result,
}


print()
print("FINAL SCIENTIFIC CHECK")


for name, ok in final_checks.items():

    print(
        f"{name:34s} = {ok}"
    )


if not all(
    final_checks.values()
):

    raise RuntimeError(
        "Final audited PG-RKL verification failed"
    )


dst.write_text(
    result,
    encoding="utf-8",
)


print()
print(
    "AUDITED_PGRKL_TRAINER_BUILD_PASS"
)
PY


python -m py_compile "$DST_TRAINER"

echo "AUDITED_PGRKL_TRAINER_COMPILE_PASS"


###############################################################################
# STAGE 3 — BUILD FRESH MASTER
###############################################################################

echo
echo "===== STAGE 3/6: BUILD MASTER ====="

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


###############################################################################
# Replace top-level NAME regardless of previous literal spelling.
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
    f'NAME="{new_name}"',
    text,
    count=1,
)


###############################################################################
# Replace trainer regardless of old path spelling.
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
    'TRAINER="$ROOT/scripts/mtpatcher_v4/'
    'train_opd_pgrkl_k1_audited_torchnpu.py"',
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
    "--master_port=29643",
    text,
    count=1,
)


###############################################################################
# Replace inherited run name wherever it occurs.
###############################################################################

text = text.replace(
    old_name,
    new_name,
)


###############################################################################
# Results labels.
###############################################################################

text = text.replace(
    "OPD-PGRKL-K1-PE3732",
    "OPD-PGRKL-K1-AUDITED-PE3732",
)

text = text.replace(
    "OPD_PGRKL_K1_PE3732",
    "OPD_PGRKL_K1_AUDITED_PE3732",
)

text = text.replace(
    "opd_torchnpu_pgrkl_k1_final_summary.json",
    "opd_torchnpu_pgrkl_k1_audited_final_summary.json",
)


###############################################################################
# Verify controls remain unchanged.
###############################################################################

checks = {
    "new run name":
        new_name
        in text,

    "audited trainer":
        "train_opd_pgrkl_k1_audited_torchnpu.py"
        in text,

    "16 NPU":
        "--nproc_per_node=16"
        in text,

    "PE3732":
        "pe_k1_clean3732.jsonl"
        in text,

    "new port":
        "--master_port=29643"
        in text,

    "audited result label":
        "OPD-PGRKL-K1-AUDITED-PE3732"
        in text,
}


print("MASTER CHECK")


for name, ok in checks.items():

    print(
        f"{name:30s} = {ok}"
    )


if not all(
    checks.values()
):

    raise RuntimeError(
        "Audited master verification failed"
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
    "AUDITED_PGRKL_MASTER_BUILD_PASS"
)
PY


bash -n "$DST_MASTER"

echo "AUDITED_PGRKL_MASTER_SYNTAX_PASS"


###############################################################################
# STAGE 4 — SCIENTIFIC AUDIT
###############################################################################

echo
echo "===== STAGE 4/6: SCIENTIFIC AUDIT ====="


echo
echo "--- Generate call ---"

grep -n -A18 -B6 \
'student.module.generate' \
"$DST_TRAINER" \
| head -80


echo
echo "--- Alignment audit ---"

grep -n -A140 -B10 \
'Causal alignment audit' \
"$DST_TRAINER" \
| head -190


echo
echo "--- PG k1 ---"

grep -n -A45 -B10 \
'Sampled k1 reverse-KL estimator' \
"$DST_TRAINER" \
| head -100


echo
echo "--- Master ---"

grep -nE \
'^NAME=|^TRAINER=|nproc_per_node|master_port|AUDITED|pe_k1_clean3732' \
"$DST_MASTER" \
| head -100


echo "AUDITED_PGRKL_SCIENTIFIC_AUDIT_PASS"


###############################################################################
# STAGE 5 — PROCESS CHECK
###############################################################################

echo
echo "===== STAGE 5/6: PROCESS CHECK ====="

RUNNING="$(
    pgrep -af \
    'train_opd_pgrkl_k1_audited|run_opd_pgrkl_k1_audited' \
    || true
)"


if [[ -n "$RUNNING" ]]; then

    echo "Matching audited PG-RKL processes already exist:"
    echo "$RUNNING"

    echo "Refusing to launch duplicate job."

else

    echo "No active audited PG-RKL job."

fi


###############################################################################
# STAGE 6 — LAUNCH
###############################################################################

echo
echo "===== STAGE 6/6: LAUNCH ====="


if [[ -z "$RUNNING" ]]; then

    mkdir -p \
    "$LOG_ROOT/$EXP"


    nohup setsid bash "$DST_MASTER" \
        > "$LOG" 2>&1 < /dev/null &


    PID=$!


    echo
    echo "======================================================================"
    echo "AUDITED PG-RKL STARTED"
    echo "PID=$PID"
    echo "LOG=$LOG"
    echo "TRAINER=$DST_TRAINER"
    echo "MASTER=$DST_MASTER"
    echo "======================================================================"

fi


###############################################################################
# Automatic first scientific health check.
###############################################################################

sleep 90


echo
echo "======================================================================"
echo "FIRST SCIENTIFIC HEALTH CHECK"
echo "======================================================================"


tail -n 200 "$LOG"

