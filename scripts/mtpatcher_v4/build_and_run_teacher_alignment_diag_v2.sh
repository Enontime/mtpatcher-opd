#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

SRC="$ROOT/scripts/mtpatcher_v4/train_opd_pgrkl_k1_audited_v2_torchnpu.py"

DST="$ROOT/scripts/mtpatcher_v4/diag_pgrkl_teacher_alignment_v2_torchnpu.py"

MASTER_SRC="$ROOT/scripts/mtpatcher_v4/run_opd_pgrkl_k1_audited_v2_oneclick.sh"

MASTER="$ROOT/scripts/mtpatcher_v4/run_pgrkl_teacher_alignment_diag_v2.sh"

NAME="opd_teacher_alignment_diag_208_v2"

LOG="$LOG_ROOT/$EXP/${NAME}.log"


echo "======================================================================"
echo "PG-RKL TEACHER ALIGNMENT + EOS/PAD DIAGNOSTIC V2"
date
echo "======================================================================"


###############################################################################
# STAGE 1 — VERIFY SOURCE
###############################################################################

echo
echo "===== STAGE 1/5: VERIFY SOURCE ====="

test -f "$SRC"
test -f "$MASTER_SRC"

python -m py_compile "$SRC"

echo "SRC=$SRC"
echo "MASTER_SRC=$MASTER_SRC"

echo "SOURCE_PASS"


###############################################################################
# STAGE 2 — BUILD DIAGNOSTIC TRAINER
###############################################################################

echo
echo "===== STAGE 2/5: BUILD DIAGNOSTIC TRAINER ====="

export SRC
export DST

python - <<'PY'
import ast
import os
from pathlib import Path


src = Path(os.environ["SRC"])
dst = Path(os.environ["DST"])

text = src.read_text(
    encoding="utf-8"
)

tree = ast.parse(text)

lines = text.splitlines()


print("SOURCE =", src)
print("TARGET =", dst)


###############################################################################
# Helpers
###############################################################################

def assigned_names(node):

    if isinstance(node, ast.Assign):
        targets = node.targets

    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]

    else:
        return []

    out = []

    for target in targets:

        if isinstance(target, ast.Name):
            out.append(target.id)

        elif isinstance(
            target,
            (ast.Tuple, ast.List),
        ):

            for item in target.elts:

                if isinstance(item, ast.Name):
                    out.append(item.id)

    return out


def src_segment(node):

    seg = ast.get_source_segment(
        text,
        node,
    )

    return seg or ""


###############################################################################
# Source scientific prerequisites
###############################################################################

required = [
    "_pg_s_logp",
    "_pg_t_logp",
    "_pg_action_ids",
    "PG_ALIGNMENT_AUDIT_PASS",
    "distill_reward",
    "s_action_logp",
    "optimizer.zero_grad",
    "clip_grad_norm_",
]


for marker in required:

    if marker not in text:

        raise RuntimeError(
            f"Missing prerequisite: {marker}"
        )


###############################################################################
# Find ALL k1 assignments, just for diagnostics.
###############################################################################

k1_nodes = []


for node in ast.walk(tree):

    if not isinstance(
        node,
        (ast.Assign, ast.AnnAssign),
    ):
        continue

    if "k1" in assigned_names(node):

        k1_nodes.append(node)


k1_nodes.sort(
    key=lambda x: x.lineno
)


print()
print("ALL K1 ASSIGNMENTS")


for node in k1_nodes:

    print(
        f"  lines {node.lineno}-{node.end_lineno}: "
        + src_segment(node).replace("\n", " ")
    )


if len(k1_nodes) < 1:

    raise RuntimeError(
        "No k1 assignment exists"
    )


###############################################################################
# Find the REAL PG loss.
#
# Required structure:
#
#   loss = -(distill_reward * s_action_logp...).mean()
#
###############################################################################

pg_loss_nodes = []


for node in ast.walk(tree):

    if not isinstance(
        node,
        (ast.Assign, ast.AnnAssign),
    ):
        continue

    if "loss" not in assigned_names(node):
        continue

    seg = src_segment(node)

    if (
        "distill_reward" in seg
        and
        "s_action_logp" in seg
    ):

        pg_loss_nodes.append(node)


if len(pg_loss_nodes) != 1:

    print()
    print(
        "PG LOSS CANDIDATES =",
        len(pg_loss_nodes),
    )

    for node in pg_loss_nodes:

        print(
            node.lineno,
            "-",
            node.end_lineno,
            src_segment(node),
        )

    raise RuntimeError(
        "Expected exactly one PG policy loss"
    )


pg_loss = pg_loss_nodes[0]


print()
print(
    "SELECTED PG LOSS =",
    pg_loss.lineno,
    "-",
    pg_loss.end_lineno,
)

print(
    src_segment(pg_loss)
)


###############################################################################
# Select the nearest executable k1 preceding the PG loss.
###############################################################################

preceding_k1 = [
    node
    for node in k1_nodes
    if node.end_lineno < pg_loss.lineno
]


if not preceding_k1:

    raise RuntimeError(
        "No k1 precedes the real PG loss"
    )


selected_k1 = max(
    preceding_k1,
    key=lambda x: x.end_lineno,
)


if (
    pg_loss.lineno
    - selected_k1.end_lineno
    > 100
):

    raise RuntimeError(
        "Nearest k1 is implausibly far from PG loss"
    )


print()
print(
    "SELECTED K1 =",
    selected_k1.lineno,
    "-",
    selected_k1.end_lineno,
)

print(
    src_segment(selected_k1)
)


###############################################################################
# Make sure token_kl / reward occur between k1 and PG loss.
###############################################################################

between = "\n".join(
    lines[
        selected_k1.end_lineno:
        pg_loss.lineno - 1
    ]
)


if "distill_reward" not in between:

    raise RuntimeError(
        "Selected k1 does not feed distill_reward"
    )


###############################################################################
# Insertion indentation.
###############################################################################

loss_line = lines[
    pg_loss.lineno - 1
]

indent = loss_line[
    :len(loss_line)
    - len(loss_line.lstrip())
]


###############################################################################
# Diagnostic block
#
# Insert AFTER the PG loss exists and BEFORE optimizer.zero_grad.
###############################################################################

block = f'''

{indent}# ================================================================
{indent}# PG-RKL MATHEMATICAL CONSISTENCY DIAGNOSTIC V2
{indent}#
{indent}# 208 sentences total:
{indent}#   13 synchronized batches * 16 ranks
{indent}#
{indent}# No meaningful parameter update is performed:
{indent}#   loss is multiplied by zero below.
{indent}# ================================================================

{indent}if global_step == 0:

{indent}    _diag_offsets = (-2, -1, 0, 1, 2)

{indent}    _diag_stats = {{}}

{indent}    for _d in _diag_offsets:

{indent}        _diag_stats[_d] = {{

{indent}            'valid_sample': 0.0,
{indent}            'valid_exact': 0.0,
{indent}            'valid_slogp': 0.0,
{indent}            'valid_tlogp': 0.0,
{indent}            'valid_count': 0.0,

{indent}            'all_sample': 0.0,
{indent}            'all_exact': 0.0,
{indent}            'all_count': 0.0,
{indent}        }}


{indent}# ------------------------------------------------------------
{indent}# Build VALID response mask:
{indent}#
{indent}# include every token through the FIRST EOS, including that EOS;
{indent}# exclude generated PAD/filler positions after EOS.
{indent}# ------------------------------------------------------------

{indent}_diag_actions_full = _pg_action_ids

{indent}_diag_B = int(
{indent}    _diag_actions_full.shape[0]
{indent})

{indent}_diag_R = int(
{indent}    _diag_actions_full.shape[1]
{indent})


{indent}_diag_valid = torch.ones(
{indent}    (
{indent}        _diag_B,
{indent}        _diag_R,
{indent}    ),
{indent}    dtype=torch.bool,
{indent}    device=_diag_actions_full.device,
{indent})


{indent}_diag_eos = tokenizer.eos_token_id


{indent}if _diag_eos is not None:

{indent}    if isinstance(
{indent}        _diag_eos,
{indent}        int,
{indent}    ):

{indent}        _diag_eos_ids = [
{indent}            int(_diag_eos)
{indent}        ]

{indent}    else:

{indent}        _diag_eos_ids = [
{indent}            int(x)
{indent}            for x in _diag_eos
{indent}        ]


{indent}    _diag_is_eos = torch.zeros_like(
{indent}        _diag_actions_full,
{indent}        dtype=torch.bool,
{indent}    )


{indent}    for _diag_eos_id in _diag_eos_ids:

{indent}        _diag_is_eos |= (
{indent}            _diag_actions_full
{indent}            == _diag_eos_id
{indent}        )


{indent}    # Number of EOS tokens STRICTLY before position t.
{indent}    _diag_eos_before = (
{indent}        _diag_is_eos.to(
{indent}            torch.int32
{indent}        ).cumsum(
{indent}            dim=1
{indent}        )
{indent}        - _diag_is_eos.to(
{indent}            torch.int32
{indent}        )
{indent}    )


{indent}    _diag_valid = (
{indent}        _diag_eos_before == 0
{indent}    )


{indent}if rank == 0 and global_step == 0:

{indent}    print(
{indent}        'DIAG_VALID_TOKEN_MASK',
{indent}        {{
{indent}            'batch': _diag_B,
{indent}            'response_steps': _diag_R,
{indent}            'valid_tokens': int(
{indent}                _diag_valid.sum().item()
{indent}            ),
{indent}            'all_tokens': int(
{indent}                _diag_valid.numel()
{indent}            ),
{indent}            'eos_token_id': _diag_eos,
{indent}        }},
{indent}        flush=True,
{indent}    )


{indent}# ------------------------------------------------------------
{indent}# offset meaning:
{indent}#
{indent}#   offset = 0
{indent}#       student state i vs teacher state i
{indent}#
{indent}#   offset = +1
{indent}#       student state i vs teacher state i+1
{indent}#
{indent}#   offset = -1
{indent}#       student state i vs teacher state i-1
{indent}# ------------------------------------------------------------

{indent}def _diag_make_pair(offset):

{indent}    R = int(
{indent}        _pg_s_logp.shape[1]
{indent}    )


{indent}    if offset < 0:

{indent}        k = -offset

{indent}        if R <= k:
{indent}            return None

{indent}        s_lp = _pg_s_logp[
{indent}            :,
{indent}            k:,
{indent}            :
{indent}        ]

{indent}        t_lp = _pg_t_logp[
{indent}            :,
{indent}            :-k,
{indent}            :
{indent}        ]

{indent}        actions = _diag_actions_full[
{indent}            :,
{indent}            k:
{indent}        ]

{indent}        valid = _diag_valid[
{indent}            :,
{indent}            k:
{indent}        ]


{indent}    elif offset > 0:

{indent}        k = offset

{indent}        if R <= k:
{indent}            return None

{indent}        s_lp = _pg_s_logp[
{indent}            :,
{indent}            :-k,
{indent}            :
{indent}        ]

{indent}        t_lp = _pg_t_logp[
{indent}            :,
{indent}            k:,
{indent}            :
{indent}        ]

{indent}        actions = _diag_actions_full[
{indent}            :,
{indent}            :-k
{indent}        ]

{indent}        valid = _diag_valid[
{indent}            :,
{indent}            :-k
{indent}        ]


{indent}    else:

{indent}        s_lp = _pg_s_logp
{indent}        t_lp = _pg_t_logp
{indent}        actions = _diag_actions_full
{indent}        valid = _diag_valid


{indent}    return (
{indent}        s_lp,
{indent}        t_lp,
{indent}        actions,
{indent}        valid,
{indent}    )


{indent}for _diag_offset in _diag_offsets:

{indent}    _pair = _diag_make_pair(
{indent}        _diag_offset
{indent}    )


{indent}    if _pair is None:
{indent}        continue


{indent}    (
{indent}        _diag_s_lp,
{indent}        _diag_t_lp,
{indent}        _diag_actions,
{indent}        _diag_mask,
{indent}    ) = _pair


{indent}    # --------------------------------------------------------
{indent}    # Sampled estimator:
{indent}    #
{indent}    # log p_S(a) - log p_T(a)
{indent}    # --------------------------------------------------------

{indent}    _diag_s_action = _diag_s_lp.gather(
{indent}        dim=-1,
{indent}        index=_diag_actions.unsqueeze(-1),
{indent}    ).squeeze(-1).float()


{indent}    _diag_t_action = _diag_t_lp.gather(
{indent}        dim=-1,
{indent}        index=_diag_actions.unsqueeze(-1),
{indent}    ).squeeze(-1).float()


{indent}    _diag_sample = (
{indent}        _diag_s_action
{indent}        - _diag_t_action
{indent}    )


{indent}    # --------------------------------------------------------
{indent}    # Exact full-vocab reverse KL:
{indent}    #
{indent}    # sum_v p_S(v) [log p_S(v) - log p_T(v)]
{indent}    # --------------------------------------------------------

{indent}    _diag_s_prob = (
{indent}        _diag_s_lp.float().exp()
{indent}    )


{indent}    _diag_exact = (
{indent}        _diag_s_prob
{indent}        * (
{indent}            _diag_s_lp.float()
{indent}            - _diag_t_lp.float()
{indent}        )
{indent}    ).sum(
{indent}        dim=-1
{indent}    )


{indent}    if not torch.isfinite(
{indent}        _diag_sample
{indent}    ).all():

{indent}        raise RuntimeError(
{indent}            'Non-finite diagnostic sampled k1'
{indent}        )


{indent}    if not torch.isfinite(
{indent}        _diag_exact
{indent}    ).all():

{indent}        raise RuntimeError(
{indent}            'Non-finite diagnostic exact RKL'
{indent}        )


{indent}    _diag_mask_f = (
{indent}        _diag_mask.float()
{indent}    )


{indent}    # VALID token sums.
{indent}    _v_sample = (
{indent}        _diag_sample
{indent}        * _diag_mask_f
{indent}    ).sum()


{indent}    _v_exact = (
{indent}        _diag_exact
{indent}        * _diag_mask_f
{indent}    ).sum()


{indent}    _v_slogp = (
{indent}        _diag_s_action
{indent}        * _diag_mask_f
{indent}    ).sum()


{indent}    _v_tlogp = (
{indent}        _diag_t_action
{indent}        * _diag_mask_f
{indent}    ).sum()


{indent}    _v_count = (
{indent}        _diag_mask_f.sum()
{indent}    )


{indent}    # ALL token sums.
{indent}    _a_sample = (
{indent}        _diag_sample.sum()
{indent}    )


{indent}    _a_exact = (
{indent}        _diag_exact.sum()
{indent}    )


{indent}    _a_count = torch.tensor(
{indent}        float(
{indent}            _diag_sample.numel()
{indent}        ),
{indent}        dtype=torch.float32,
{indent}        device=device,
{indent}    )


{indent}    # Float32 is intentional for HCCL compatibility.
{indent}    _diag_reduce = torch.stack(
{indent}        [
{indent}            _v_sample.float(),
{indent}            _v_exact.float(),
{indent}            _v_slogp.float(),
{indent}            _v_tlogp.float(),
{indent}            _v_count.float(),
{indent}            _a_sample.float(),
{indent}            _a_exact.float(),
{indent}            _a_count.float(),
{indent}        ]
{indent}    )


{indent}    dist.all_reduce(
{indent}        _diag_reduce,
{indent}        op=dist.ReduceOp.SUM,
{indent}    )


{indent}    if rank == 0:

{indent}        obj = _diag_stats[
{indent}            _diag_offset
{indent}        ]


{indent}        obj['valid_sample'] += float(
{indent}            _diag_reduce[0].item()
{indent}        )

{indent}        obj['valid_exact'] += float(
{indent}            _diag_reduce[1].item()
{indent}        )

{indent}        obj['valid_slogp'] += float(
{indent}            _diag_reduce[2].item()
{indent}        )

{indent}        obj['valid_tlogp'] += float(
{indent}            _diag_reduce[3].item()
{indent}        )

{indent}        obj['valid_count'] += float(
{indent}            _diag_reduce[4].item()
{indent}        )

{indent}        obj['all_sample'] += float(
{indent}            _diag_reduce[5].item()
{indent}        )

{indent}        obj['all_exact'] += float(
{indent}            _diag_reduce[6].item()
{indent}        )

{indent}        obj['all_count'] += float(
{indent}            _diag_reduce[7].item()
{indent}        )


{indent}# ================================================================
{indent}# Critical:
{indent}# zero the already-defined PG loss BEFORE optimizer.backward().
{indent}# Student weights remain unchanged during this diagnostic.
{indent}# ================================================================

{indent}loss = loss * 0.0


{indent}# 13 local synchronized batches:
{indent}#
{indent}# global_step 0 ... 12
{indent}#
{indent}# 13 * 16 ranks = 208 source sentences.
{indent}if global_step == 12:

{indent}    if rank == 0:

{indent}        print()
{indent}        print(
{indent}            '=' * 96
{indent}        )

{indent}        print(
{indent}            'TEACHER_ALIGNMENT_DIAGNOSTIC_FINAL'
{indent}        )

{indent}        print(
{indent}            '=' * 96
{indent}        )


{indent}        _diag_ranking = []


{indent}        for _diag_offset in _diag_offsets:

{indent}            obj = _diag_stats[
{indent}                _diag_offset
{indent}            ]


{indent}            vc = obj[
{indent}                'valid_count'
{indent}            ]

{indent}            ac = obj[
{indent}                'all_count'
{indent}            ]


{indent}            if vc <= 0 or ac <= 0:
{indent}                continue


{indent}            valid_sample = (
{indent}                obj['valid_sample']
{indent}                / vc
{indent}            )


{indent}            valid_exact = (
{indent}                obj['valid_exact']
{indent}                / vc
{indent}            )


{indent}            valid_gap = abs(
{indent}                valid_sample
{indent}                - valid_exact
{indent}            )


{indent}            valid_slogp = (
{indent}                obj['valid_slogp']
{indent}                / vc
{indent}            )


{indent}            valid_tlogp = (
{indent}                obj['valid_tlogp']
{indent}                / vc
{indent}            )


{indent}            all_sample = (
{indent}                obj['all_sample']
{indent}                / ac
{indent}            )


{indent}            all_exact = (
{indent}                obj['all_exact']
{indent}                / ac
{indent}            )


{indent}            all_gap = abs(
{indent}                all_sample
{indent}                - all_exact
{indent}            )


{indent}            _diag_ranking.append(
{indent}                (
{indent}                    valid_gap,
{indent}                    _diag_offset,
{indent}                    valid_sample,
{indent}                    valid_exact,
{indent}                    valid_slogp,
{indent}                    valid_tlogp,
{indent}                    all_sample,
{indent}                    all_exact,
{indent}                    all_gap,
{indent}                    vc,
{indent}                    ac,
{indent}                )
{indent}            )


{indent}            print(
{indent}                f'offset={{_diag_offset:+d}} '
{indent}                f'VALID sampled={{valid_sample:+.6f}} '
{indent}                f'exact={{valid_exact:+.6f}} '
{indent}                f'gap={{valid_gap:.6f}} '
{indent}                f's_logp={{valid_slogp:+.6f}} '
{indent}                f't_logp={{valid_tlogp:+.6f}} '
{indent}                f'tokens={{int(vc)}}'
{indent}            )


{indent}            print(
{indent}                f'             '
{indent}                f'ALL sampled={{all_sample:+.6f}} '
{indent}                f'exact={{all_exact:+.6f}} '
{indent}                f'gap={{all_gap:.6f}} '
{indent}                f'tokens={{int(ac)}}'
{indent}            )


{indent}        _diag_ranking.sort(
{indent}            key=lambda x: x[0]
{indent}        )


{indent}        if not _diag_ranking:

{indent}            raise RuntimeError(
{indent}                'No diagnostic result accumulated'
{indent}            )


{indent}        best = _diag_ranking[0]


{indent}        (
{indent}            best_gap,
{indent}            best_offset,
{indent}            best_sample,
{indent}            best_exact,
{indent}            best_slogp,
{indent}            best_tlogp,
{indent}            best_all_sample,
{indent}            best_all_exact,
{indent}            best_all_gap,
{indent}            best_vc,
{indent}            best_ac,
{indent}        ) = best


{indent}        print()
{indent}        print(
{indent}            'BEST_TEACHER_OFFSET =',
{indent}            best_offset,
{indent}        )

{indent}        print(
{indent}            'BEST_VALID_SAMPLED_K1 =',
{indent}            best_sample,
{indent}        )

{indent}        print(
{indent}            'BEST_VALID_EXACT_RKL =',
{indent}            best_exact,
{indent}        )

{indent}        print(
{indent}            'BEST_VALID_ABS_GAP =',
{indent}            best_gap,
{indent}        )

{indent}        print(
{indent}            'BEST_VALID_STUDENT_LOGP =',
{indent}            best_slogp,
{indent}        )

{indent}        print(
{indent}            'BEST_VALID_TEACHER_LOGP =',
{indent}            best_tlogp,
{indent}        )

{indent}        print(
{indent}            'BEST_ALL_SAMPLED_K1 =',
{indent}            best_all_sample,
{indent}        )

{indent}        print(
{indent}            'BEST_ALL_EXACT_RKL =',
{indent}            best_all_exact,
{indent}        )

{indent}        print(
{indent}            'BEST_ALL_ABS_GAP =',
{indent}            best_all_gap,
{indent}        )


{indent}        # Strong diagnostic criteria.
{indent}        if (
{indent}            best_exact >= -0.02
{indent}            and
{indent}            best_gap < 0.50
{indent}        ):

{indent}            print(
{indent}                'TEACHER_ALIGNMENT_DIAGNOSTIC_PASS'
{indent}            )

{indent}        else:

{indent}            print(
{indent}                'TEACHER_ALIGNMENT_DIAGNOSTIC_UNRESOLVED'
{indent}            )


{indent}        print(
{indent}            '=' * 96
{indent}        )

{indent}        print()


{indent}    dist.barrier()

{indent}    dist.destroy_process_group()

{indent}    return
'''


###############################################################################
# Insert AFTER PG loss.
###############################################################################

insert_at = pg_loss.end_lineno


lines[
    insert_at:insert_at
] = block.splitlines()


result = "\n".join(lines) + "\n"


###############################################################################
# Syntax verification
###############################################################################

ast.parse(
    result
)


###############################################################################
# Make sure diagnostic is before optimizer.zero_grad().
###############################################################################

diag_pos = result.find(
    "PG-RKL MATHEMATICAL CONSISTENCY DIAGNOSTIC V2"
)

optimizer_pos = result.find(
    "optimizer.zero_grad",
    diag_pos,
)


if diag_pos < 0:
    raise RuntimeError(
        "Diagnostic block not inserted"
    )


if optimizer_pos < 0:
    raise RuntimeError(
        "Could not locate optimizer after diagnostic"
    )


###############################################################################
# Hard checks
###############################################################################

checks = {
    "selected real PG loss":
        "distill_reward" in src_segment(pg_loss),

    "five offsets":
        "_diag_offsets = (-2, -1, 0, 1, 2)"
        in result,

    "EOS mask":
        "_diag_eos_before"
        in result,

    "valid token stats":
        "VALID sampled="
        in result,

    "all token stats":
        "ALL sampled="
        in result,

    "exact RKL":
        "_diag_exact"
        in result,

    "sampled k1":
        "_diag_sample"
        in result,

    "student logp mean":
        "BEST_VALID_STUDENT_LOGP"
        in result,

    "teacher logp mean":
        "BEST_VALID_TEACHER_LOGP"
        in result,

    "float32 allreduce":
        "_diag_reduce"
        in result
        and "dtype=torch.float32"
        in result,

    "zero loss after definition":
        "loss = loss * 0.0"
        in result,

    "early finish":
        "global_step == 12"
        in result,

    "no model persistence required":
        "TEACHER_ALIGNMENT_DIAGNOSTIC_FINAL"
        in result,
}


print()
print("DIAGNOSTIC BUILD CHECK")


for name, ok in checks.items():

    print(
        f"{name:34s} = {ok}"
    )


if not all(
    checks.values()
):

    raise RuntimeError(
        "Diagnostic build verification failed"
    )


dst.write_text(
    result,
    encoding="utf-8",
)


print()
print(
    "TEACHER_ALIGNMENT_DIAG_V2_BUILD_PASS"
)
PY


python -m py_compile "$DST"

echo "DIAGNOSTIC_TRAINER_COMPILE_PASS"


###############################################################################
# STAGE 3 — BUILD SHORT MASTER
###############################################################################

echo
echo "===== STAGE 3/5: BUILD SHORT MASTER ====="

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
# Replace run identity.
###############################################################################

text = re.sub(
    r'(?m)^NAME=.*$',
    f'NAME="{name}"',
    text,
    count=1,
)


text = re.sub(
    r'(?m)^TRAINER=.*$',
    'TRAINER="$ROOT/scripts/mtpatcher_v4/'
    'diag_pgrkl_teacher_alignment_v2_torchnpu.py"',
    text,
    count=1,
)


text = re.sub(
    r"--master_port=\d+",
    "--master_port=29646",
    text,
    count=1,
)


###############################################################################
# IMPORTANT:
# remove evaluation/checkpoint stages.
#
# Keep everything through the distributed training command,
# stop immediately before OPD_EPOCH3_MODEL is expected.
###############################################################################

lines = text.splitlines()


cut_candidates = [
    i
    for i, line in enumerate(lines)
    if line.startswith(
        "OPD_EPOCH3_MODEL="
    )
]


if len(cut_candidates) != 1:

    raise RuntimeError(
        "Could not uniquely find OPD_EPOCH3_MODEL boundary: "
        + str(cut_candidates)
    )


cut = cut_candidates[0]


short_lines = lines[:cut]


short_lines.extend(
    [
        "",
        'echo',
        'echo "======================================================================"',
        'echo "TEACHER ALIGNMENT DIAGNOSTIC MASTER FINISHED"',
        'date',
        'echo "======================================================================"',
        "",
    ]
)


short_text = "\n".join(
    short_lines
) + "\n"


###############################################################################
# Verify short master.
###############################################################################

checks = {
    "diagnostic trainer":
        "diag_pgrkl_teacher_alignment_v2_torchnpu.py"
        in short_text,

    "16 NPU":
        "--nproc_per_node=16"
        in short_text,

    "fresh port":
        "--master_port=29646"
        in short_text,

    "PE3732":
        "pe_k1_clean3732.jsonl"
        in short_text,

    "evaluation removed":
        "STAGE 3/4: EVALUATION"
        not in short_text,

    "epoch3 dependency removed":
        "OPD_EPOCH3_MODEL="
        not in short_text,
}


print("SHORT MASTER CHECK")


for name, ok in checks.items():

    print(
        f"{name:30s} = {ok}"
    )


if not all(
    checks.values()
):

    raise RuntimeError(
        "Short diagnostic master verification failed"
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
    "SHORT_DIAGNOSTIC_MASTER_BUILD_PASS"
)
PY


bash -n "$MASTER"

echo "SHORT_MASTER_SYNTAX_PASS"


###############################################################################
# STAGE 4 — LAUNCH
###############################################################################

echo
echo "===== STAGE 4/5: LAUNCH ====="

mkdir -p \
"$LOG_ROOT/$EXP"


RUNNING="$(
    pgrep -af \
    'diag_pgrkl_teacher_alignment_v2_torchnpu' \
    || true
)"


if [[ -n "$RUNNING" ]]; then

    echo "Existing diagnostic process detected:"
    echo "$RUNNING"
    echo "Duplicate launch skipped."

else

    nohup setsid bash "$MASTER" \
        > "$LOG" 2>&1 < /dev/null &


    PID=$!


    echo
    echo "======================================================================"
    echo "TEACHER ALIGNMENT DIAGNOSTIC V2 STARTED"
    echo "PID=$PID"
    echo "LOG=$LOG"
    echo "======================================================================"

fi


###############################################################################
# STAGE 5 — WAIT FOR RESULT
###############################################################################

echo
echo "===== STAGE 5/5: WAIT FOR DIAGNOSTIC ====="


for _i in $(seq 1 60); do

    if grep -q \
        'TEACHER_ALIGNMENT_DIAGNOSTIC_FINAL' \
        "$LOG" 2>/dev/null; then

        break

    fi


    if grep -qE \
        'Traceback|RuntimeError:|NameError:|TypeError:|ValueError:' \
        "$LOG" 2>/dev/null; then

        break

    fi


    sleep 5

done


echo
echo "======================================================================"
echo "DIAGNOSTIC RESULT"
echo "======================================================================"


grep -E \
'PG_ALIGNMENT|DIAG_VALID_TOKEN_MASK|TEACHER_ALIGNMENT|offset=|BEST_|Traceback|RuntimeError:|NameError:|TypeError:|ValueError:' \
"$LOG" \
| tail -n 120

