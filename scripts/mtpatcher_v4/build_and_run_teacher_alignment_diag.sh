#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

SRC="$ROOT/scripts/mtpatcher_v4/train_opd_pgrkl_k1_audited_v2_torchnpu.py"
DST="$ROOT/scripts/mtpatcher_v4/diag_pgrkl_teacher_alignment_torchnpu.py"

MASTER_SRC="$ROOT/scripts/mtpatcher_v4/run_opd_pgrkl_k1_audited_v2_oneclick.sh"
MASTER="$ROOT/scripts/mtpatcher_v4/run_pgrkl_teacher_alignment_diag.sh"

NAME="opd_teacher_alignment_diag_208_v1"
LOG="$LOG_ROOT/$EXP/${NAME}.log"

echo "======================================================================"
echo "PG-RKL TEACHER CAUSAL-ALIGNMENT DIAGNOSTIC"
date
echo "======================================================================"

test -f "$SRC"
test -f "$MASTER_SRC"

export SRC
export DST

python - <<'PY'
import ast
import os
from pathlib import Path

src = Path(os.environ["SRC"])
dst = Path(os.environ["DST"])

text = src.read_text(encoding="utf-8")
tree = ast.parse(text)

required = [
    "_pg_s_logp",
    "_pg_t_logp",
    "_pg_action_ids",
    "PG_ALIGNMENT_AUDIT_PASS",
    "k1 = (",
    "optimizer.zero_grad",
]

for x in required:
    if x not in text:
        raise RuntimeError(f"missing diagnostic prerequisite: {x}")

# Find sampled-k1 assignment.
k1_nodes = []

for node in ast.walk(tree):
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        continue

    targets = node.targets if isinstance(node, ast.Assign) else [node.target]

    if any(
        isinstance(t, ast.Name) and t.id == "k1"
        for t in targets
    ):
        k1_nodes.append(node)

if len(k1_nodes) != 1:
    raise RuntimeError(
        f"expected one k1 assignment, found {len(k1_nodes)}"
    )

k1_node = k1_nodes[0]

lines = text.splitlines()

line = lines[k1_node.end_lineno - 1]
indent = line[:len(line) - len(line.lstrip())]

block = f'''

{indent}# ================================================================
{indent}# TEACHER CAUSAL ALIGNMENT DIAGNOSTIC
{indent}#
{indent}# No optimizer update is allowed in this diagnostic.
{indent}#
{indent}# For candidate teacher offset d:
{indent}#
{indent}#   student state i  <-> teacher state i+d
{indent}#
{indent}# We compare:
{indent}#
{indent}#   Monte-Carlo sampled k1
{indent}#   exact full-vocabulary KL(student || teacher)
{indent}#
{indent}# A causally valid teacher state should make the two agree
{indent}# over many student-sampled actions.
{indent}# ================================================================

{indent}if '_diag_sum' not in locals():
{indent}    _diag_sum = {{
{indent}        d: {{
{indent}            'sample': 0.0,
{indent}            'exact': 0.0,
{indent}            'abs_gap': 0.0,
{indent}            'count': 0.0,
{indent}        }}
{indent}        for d in (-2, -1, 0, 1, 2)
{indent}    }}

{indent}def _diag_pair(offset):

{indent}    R = int(_pg_s_logp.shape[1])

{indent}    if offset < 0:
{indent}        k = -offset

{indent}        if R <= k:
{indent}            return None

{indent}        s_lp = _pg_s_logp[:, k:, :]
{indent}        t_lp = _pg_t_logp[:, :-k, :]
{indent}        actions = _pg_action_ids[:, k:]

{indent}    elif offset > 0:

{indent}        k = offset

{indent}        if R <= k:
{indent}            return None

{indent}        s_lp = _pg_s_logp[:, :-k, :]
{indent}        t_lp = _pg_t_logp[:, k:, :]
{indent}        actions = _pg_action_ids[:, :-k]

{indent}    else:

{indent}        s_lp = _pg_s_logp
{indent}        t_lp = _pg_t_logp
{indent}        actions = _pg_action_ids

{indent}    # Sampled reverse-KL estimator.
{indent}    s_a = s_lp.gather(
{indent}        -1,
{indent}        actions.unsqueeze(-1),
{indent}    ).squeeze(-1).float()

{indent}    t_a = t_lp.gather(
{indent}        -1,
{indent}        actions.unsqueeze(-1),
{indent}    ).squeeze(-1).float()

{indent}    sampled = s_a - t_a

{indent}    # Exact full-vocabulary reverse KL.
{indent}    s_prob_diag = s_lp.float().exp()

{indent}    exact = (
{indent}        s_prob_diag
{indent}        * (
{indent}            s_lp.float()
{indent}            - t_lp.float()
{indent}        )
{indent}    ).sum(dim=-1)

{indent}    n = float(sampled.numel())

{indent}    return (
{indent}        sampled.sum(),
{indent}        exact.sum(),
{indent}        n,
{indent}    )


{indent}for _diag_offset in (-2, -1, 0, 1, 2):

{indent}    _diag_result = _diag_pair(
{indent}        _diag_offset
{indent}    )

{indent}    if _diag_result is None:
{indent}        continue

{indent}    _diag_sample_sum, _diag_exact_sum, _diag_n = (
{indent}        _diag_result
{indent}    )

{indent}    # Aggregate all 16 ranks every batch.
{indent}    _diag_tensor = torch.tensor(
{indent}        [
{indent}            float(_diag_sample_sum.item()),
{indent}            float(_diag_exact_sum.item()),
{indent}            float(_diag_n),
{indent}        ],
{indent}        dtype=torch.float32,
{indent}        device=device,
{indent}    )

{indent}    dist.all_reduce(
{indent}        _diag_tensor,
{indent}        op=dist.ReduceOp.SUM,
{indent}    )

{indent}    if rank == 0:

{indent}        _diag_sum[_diag_offset]['sample'] += float(
{indent}            _diag_tensor[0].item()
{indent}        )

{indent}        _diag_sum[_diag_offset]['exact'] += float(
{indent}            _diag_tensor[1].item()
{indent}        )

{indent}        _diag_sum[_diag_offset]['count'] += float(
{indent}            _diag_tensor[2].item()
{indent}        )


{indent}# 13 synchronized local batches x 16 ranks = 208 sentences.
{indent}if global_step == 12:

{indent}    if rank == 0:

{indent}        print()
{indent}        print('=' * 86)
{indent}        print('TEACHER_ALIGNMENT_DIAGNOSTIC_FINAL')
{indent}        print('=' * 86)

{indent}        _diag_results = []

{indent}        for _diag_offset in (-2, -1, 0, 1, 2):

{indent}            obj = _diag_sum[_diag_offset]

{indent}            if obj['count'] <= 0:
{indent}                continue

{indent}            sampled_mean = (
{indent}                obj['sample']
{indent}                / obj['count']
{indent}            )

{indent}            exact_mean = (
{indent}                obj['exact']
{indent}                / obj['count']
{indent}            )

{indent}            gap = abs(
{indent}                sampled_mean
{indent}                - exact_mean
{indent}            )

{indent}            _diag_results.append(
{indent}                (
{indent}                    gap,
{indent}                    _diag_offset,
{indent}                    sampled_mean,
{indent}                    exact_mean,
{indent}                    obj['count'],
{indent}                )
{indent}            )

{indent}            print(
{indent}                f'offset={{_diag_offset:+d}} '
{indent}                f'sampled_k1={{sampled_mean:+.6f}} '
{indent}                f'exact_rkl={{exact_mean:+.6f}} '
{indent}                f'abs_gap={{gap:.6f}} '
{indent}                f'tokens={{int(obj["count"])}}',
{indent}                flush=True,
{indent}            )

{indent}        _diag_results.sort()

{indent}        (
{indent}            best_gap,
{indent}            best_offset,
{indent}            best_sample,
{indent}            best_exact,
{indent}            best_count,
{indent}        ) = _diag_results[0]

{indent}        print()
{indent}        print(
{indent}            'BEST_TEACHER_OFFSET =',
{indent}            best_offset,
{indent}        )

{indent}        print(
{indent}            'BEST_SAMPLE_MEAN =',
{indent}            best_sample,
{indent}        )

{indent}        print(
{indent}            'BEST_EXACT_RKL =',
{indent}            best_exact,
{indent}        )

{indent}        print(
{indent}            'BEST_ABS_GAP =',
{indent}            best_gap,
{indent}        )

{indent}        if best_gap < 0.30:
{indent}            print(
{indent}                'TEACHER_ALIGNMENT_DIAGNOSTIC_PASS'
{indent}            )
{indent}        else:
{indent}            print(
{indent}                'TEACHER_ALIGNMENT_DIAGNOSTIC_UNRESOLVED'
{indent}            )

{indent}        print('=' * 86)
{indent}        print()

{indent}    dist.barrier()
{indent}    dist.destroy_process_group()
{indent}    return

{indent}# Diagnostic mode must NEVER update model parameters.
{indent}loss = loss * 0.0
'''

# Insert immediately after k1 is computed, before optimizer path.
insert_at = k1_node.end_lineno

lines[insert_at:insert_at] = block.splitlines()

result = "\n".join(lines) + "\n"

ast.parse(result)

# Hard guards.
checks = {
    "zero update":
        "loss = loss * 0.0" in result,

    "five offsets":
        "for _diag_offset in (-2, -1, 0, 1, 2)"
        in result,

    "exact RKL":
        "s_prob_diag" in result,

    "all reduce":
        "dist.all_reduce" in result,

    "208 examples":
        "global_step == 12" in result,

    "early return":
        "dist.destroy_process_group()"
        in result,

    "original PG objective retained":
        "distill_reward" in result,
}

print("DIAGNOSTIC BUILD CHECK")

for name, ok in checks.items():
    print(f"{name:32s} = {ok}")

if not all(checks.values()):
    raise RuntimeError(
        "diagnostic build verification failed"
    )

dst.write_text(
    result,
    encoding="utf-8",
)

print()
print("TEACHER_ALIGNMENT_DIAG_BUILD_PASS")
PY

python -m py_compile "$DST"

echo "DIAGNOSTIC_COMPILE_PASS"


###############################################################################
# Build a short 16-NPU launcher from the successful master.
###############################################################################

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

text = re.sub(
    r'(?m)^NAME=.*$',
    f'NAME="{name}"',
    text,
    count=1,
)

text = re.sub(
    r'(?m)^TRAINER=.*$',
    'TRAINER="$ROOT/scripts/mtpatcher_v4/'
    'diag_pgrkl_teacher_alignment_torchnpu.py"',
    text,
    count=1,
)

text = re.sub(
    r"--master_port=\d+",
    "--master_port=29645",
    text,
    count=1,
)

dst.write_text(
    text,
    encoding="utf-8",
)

dst.chmod(0o755)

print("DIAGNOSTIC_MASTER_BUILD_PASS")
PY

bash -n "$MASTER"

echo "DIAGNOSTIC_MASTER_SYNTAX_PASS"


###############################################################################
# Launch.
###############################################################################

mkdir -p "$LOG_ROOT/$EXP"

nohup setsid bash "$MASTER" \
    > "$LOG" 2>&1 < /dev/null &

PID=$!

echo
echo "======================================================================"
echo "TEACHER ALIGNMENT DIAGNOSTIC STARTED"
echo "PID=$PID"
echo "LOG=$LOG"
echo "======================================================================"

