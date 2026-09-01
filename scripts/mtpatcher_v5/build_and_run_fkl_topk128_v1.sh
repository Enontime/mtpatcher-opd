#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

SRC_TRAINER="$ROOT/scripts/mtpatcher_v3/train_opd_forwardkl_torchnpu.py"

DST_TRAINER="$ROOT/scripts/mtpatcher_v5/train_opd_forwardkl_topk128_torchnpu.py"

SRC_MASTER="$ROOT/scripts/mtpatcher_v3/run_opd_forwardkl_torchnpu_oneclick.sh"

DST_MASTER="$ROOT/scripts/mtpatcher_v5/run_opd_forwardkl_topk128_oneclick.sh"

OLD_NAME="opd_torchnpu_fkl_pe3732_v1"

NEW_NAME="opd_torchnpu_fkl_topk128_pe3732_v1"

FULL_LOG="$LOG_ROOT/$EXP/${NEW_NAME}.log"

RUN_DIR="$RUN_ROOT/$EXP/$NEW_NAME"


echo "======================================================================"
echo "TEACHER-TOP-128 FORWARD-KL OPD"
echo "Official-style verl GKD loss approximation"
date
echo "======================================================================"


###############################################################################
# STAGE 1 — VERIFY THE KNOWN-GOOD EXACT FKL SOURCE
###############################################################################

echo
echo "===== STAGE 1/6: VERIFY EXACT-FKL SOURCE ====="

test -f "$SRC_TRAINER"
test -f "$SRC_MASTER"

python -m py_compile \
"$SRC_TRAINER"

bash -n \
"$SRC_MASTER"

echo "SOURCE_TRAINER=$SRC_TRAINER"
echo "SOURCE_MASTER=$SRC_MASTER"

echo "EXACT_FKL_SOURCE_PASS"


###############################################################################
# STAGE 2 — BUILD TOP-128 FKL TRAINER
###############################################################################

echo
echo "===== STAGE 2/6: BUILD TOPK128 TRAINER ====="

export SRC_TRAINER
export DST_TRAINER

python - <<'PY'
import ast
import os
from pathlib import Path


src = Path(
    os.environ["SRC_TRAINER"]
)

dst = Path(
    os.environ["DST_TRAINER"]
)

text = src.read_text(
    encoding="utf-8"
)

tree = ast.parse(
    text
)


###############################################################################
# Helpers
###############################################################################

def assigned_names(node):

    if isinstance(
        node,
        ast.Assign,
    ):
        targets = node.targets

    elif isinstance(
        node,
        ast.AnnAssign,
    ):
        targets = [
            node.target
        ]

    else:
        return []

    out = []

    for target in targets:

        if isinstance(
            target,
            ast.Name,
        ):
            out.append(
                target.id
            )

    return out


def segment(node):

    return (
        ast.get_source_segment(
            text,
            node,
        )
        or ""
    )


###############################################################################
# Find the VERIFIED exact-FKL token_kl:
#
# token_kl =
#     (t_prob * (t_logp - s_logp)).sum(-1)
###############################################################################

candidates = []


for node in ast.walk(
    tree
):

    if not isinstance(
        node,
        (
            ast.Assign,
            ast.AnnAssign,
        ),
    ):
        continue


    if "token_kl" not in assigned_names(
        node
    ):
        continue


    s = segment(
        node
    )


    if (
        "t_prob" in s
        and
        "t_logp" in s
        and
        "s_logp" in s
    ):

        candidates.append(
            node
        )


print(
    "EXACT_FKL_TOKEN_KL_CANDIDATES =",
    len(candidates),
)


for node in candidates:

    print(
        f"  lines {node.lineno}-{node.end_lineno}:",
        segment(
            node
        ).replace(
            "\n",
            " ",
        ),
    )


if len(
    candidates
) != 1:

    raise RuntimeError(
        "Expected exactly one verified exact-FKL token_kl"
    )


node = candidates[
    0
]


###############################################################################
# Preserve indentation.
###############################################################################

lines = text.splitlines()

first = lines[
    node.lineno - 1
]

indent = first[
    :len(first)
    - len(
        first.lstrip()
    )
]


###############################################################################
# Current verl-style teacher-top-k forward KL.
#
# Important:
# - top-k is selected from TEACHER full-distribution logprobs.
# - teacher probabilities are NOT renormalized inside top-k.
# - therefore truncated KL can be negative.
# - current verl clamps top-k distillation loss to >= 0.
#
# We also print a first-batch audit:
#   teacher top-128 probability mass
#   exact full-vocab FKL
#   top128 raw/clamped FKL
###############################################################################

replacement = f'''
{indent}# ============================================================
{indent}# TEACHER-TOP-128 FORWARD KL
{indent}#
{indent}# Matches the current verl GKD/OPD loss structure:
{indent}#
{indent}# sum_{{v in TopK(T)}} p_T(v)
{indent}#     [log p_T(v) - log p_S(v)]
{indent}#
{indent}# Top-k teacher probabilities remain probabilities under
{indent}# the FULL teacher distribution; no top-k renormalization.
{indent}#
{indent}# Because truncated masses need not sum to one, the
{indent}# truncated quantity can be negative. Current verl clamps
{indent}# the per-token top-k loss at zero.
{indent}# ============================================================

{indent}_topk_k = 128


{indent}if int(
{indent}    t_logp.shape[-1]
{indent}) < _topk_k:

{indent}    raise RuntimeError(
{indent}        "Vocabulary smaller than requested TOPK=128: "
{indent}        f"vocab={{int(t_logp.shape[-1])}}"
{indent}    )


{indent}_topk_t_logp, _topk_ids = torch.topk(
{indent}    t_logp.float(),
{indent}    k=_topk_k,
{indent}    dim=-1,
{indent})


{indent}_topk_t_prob = (
{indent}    _topk_t_logp.exp()
{indent})


{indent}_topk_s_logp = torch.gather(
{indent}    s_logp.float(),
{indent}    dim=-1,
{indent}    index=_topk_ids,
{indent})


{indent}_topk_teacher_mass = (
{indent}    _topk_t_prob.sum(
{indent}        dim=-1
{indent}    )
{indent})


{indent}_topk_raw = (
{indent}    _topk_t_prob
{indent}    * (
{indent}        _topk_t_logp
{indent}        - _topk_s_logp
{indent}    )
{indent}).sum(
{indent}    dim=-1
{indent})


{indent}token_kl = (
{indent}    _topk_raw.clamp_min(
{indent}        0.0
{indent}    )
{indent})


{indent}if not torch.isfinite(
{indent}    token_kl
{indent}).all():

{indent}    raise RuntimeError(
{indent}        "Non-finite teacher-top128 FKL"
{indent}    )


{indent}# ------------------------------------------------------------
{indent}# First-batch scientific audit.
{indent}#
{indent}# The exact full-vocab FKL is computed only once for
{indent}# comparison. The source trainer already materializes t_prob,
{indent}# so this does not introduce a new full-vocab tensor.
{indent}# ------------------------------------------------------------

{indent}if "_topk128_audit_done" not in locals():

{indent}    _topk128_exact_fkl = (
{indent}        t_prob.float()
{indent}        * (
{indent}            t_logp.float()
{indent}            - s_logp.float()
{indent}        )
{indent}    ).sum(
{indent}        dim=-1
{indent}    )


{indent}    if rank == 0:

{indent}        print(
{indent}            "TOPK128_FKL_RUNTIME_AUDIT",
{indent}            {{
{indent}                "teacher_mass_mean":
{indent}                    float(
{indent}                        _topk_teacher_mass.mean().item()
{indent}                    ),
{indent}                "teacher_mass_min":
{indent}                    float(
{indent}                        _topk_teacher_mass.min().item()
{indent}                    ),
{indent}                "exact_fkl_mean":
{indent}                    float(
{indent}                        _topk128_exact_fkl.mean().item()
{indent}                    ),
{indent}                "topk_raw_mean":
{indent}                    float(
{indent}                        _topk_raw.mean().item()
{indent}                    ),
{indent}                "topk_clamped_mean":
{indent}                    float(
{indent}                        token_kl.mean().item()
{indent}                    ),
{indent}                "negative_token_fraction":
{indent}                    float(
{indent}                        (
{indent}                            _topk_raw < 0
{indent}                        ).float().mean().item()
{indent}                    ),
{indent}                "topk":
{indent}                    _topk_k,
{indent}            }},
{indent}            flush=True,
{indent}        )


{indent}    _topk128_audit_done = True
'''


lines[
    node.lineno - 1:
    node.end_lineno
] = replacement.splitlines()


result = "\n".join(
    lines
) + "\n"


###############################################################################
# Parse final source.
###############################################################################

tree2 = ast.parse(
    result
)


###############################################################################
# Unique objective audit.
###############################################################################

def count_assignments(
    tree,
    name,
):

    count = 0

    for n in ast.walk(
        tree
    ):

        if not isinstance(
            n,
            (
                ast.Assign,
                ast.AnnAssign,
            ),
        ):
            continue

        if name in assigned_names(
            n
        ):
            count += 1

    return count


token_count = count_assignments(
    tree2,
    "token_kl",
)


print()
print(
    "TOKEN_KL_ASSIGNMENT_COUNT =",
    token_count,
)


if token_count != 1:

    raise RuntimeError(
        "Top-k trainer must have exactly one token_kl assignment"
    )


###############################################################################
# Required markers.
###############################################################################

checks = {
    "topk=128":
        "_topk_k = 128"
        in result,

    "teacher topk":
        "torch.topk("
        in result,

    "teacher probability":
        "_topk_t_prob"
        in result,

    "student gather":
        "_topk_s_logp"
        in result,

    "teacher mass":
        "_topk_teacher_mass"
        in result,

    "no renormalization":
        "_topk_t_prob.sum("
        in result,

    "negative clamp":
        "_topk_raw.clamp_min("
        in result,

    "runtime audit":
        "TOPK128_FKL_RUNTIME_AUDIT"
        in result,

    "exact comparison":
        "_topk128_exact_fkl"
        in result,

    "finite check":
        "Non-finite teacher-top128 FKL"
        in result,
}


print()
print(
    "TOPK128 SCIENTIFIC STATIC CHECK"
)


for name, ok in checks.items():

    print(
        f"{name:28s} = {ok}"
    )


if not all(
    checks.values()
):

    raise RuntimeError(
        "Top-k FKL scientific verification failed"
    )


###############################################################################
# Ensure the exact original token_kl expression is gone.
###############################################################################

for n in ast.walk(
    tree2
):

    if not isinstance(
        n,
        (
            ast.Assign,
            ast.AnnAssign,
        ),
    ):
        continue

    if "token_kl" not in assigned_names(
        n
    ):
        continue

    s = (
        ast.get_source_segment(
            result,
            n,
        )
        or ""
    )

    if (
        "t_prob" in s
        and
        "t_logp" in s
        and
        "s_logp" in s
    ):

        raise RuntimeError(
            "Old full-vocab exact-FKL token_kl survived"
        )


dst.write_text(
    result,
    encoding="utf-8",
)


print()
print(
    "TOPK128_FKL_TRAINER_BUILD_PASS"
)
PY


python -m py_compile \
"$DST_TRAINER"

echo "TOPK128_TRAINER_COMPILE_PASS"


###############################################################################
# STAGE 3 — BUILD FRESH MASTER
###############################################################################

echo
echo "===== STAGE 3/6: BUILD FRESH MASTER ====="

export SRC_MASTER
export DST_MASTER
export OLD_NAME
export NEW_NAME

python - <<'PY'
import os
import re
from pathlib import Path


src = Path(
    os.environ["SRC_MASTER"]
)

dst = Path(
    os.environ["DST_MASTER"]
)

old_name = os.environ[
    "OLD_NAME"
]

new_name = os.environ[
    "NEW_NAME"
]


text = src.read_text(
    encoding="utf-8"
)


###############################################################################
# NAME
###############################################################################

name_hits = re.findall(
    r"(?m)^NAME=.*$",
    text,
)


if len(
    name_hits
) != 1:

    raise RuntimeError(
        f"Expected one NAME line, got {name_hits}"
    )


text = re.sub(
    r"(?m)^NAME=.*$",
    f'NAME="{new_name}"',
    text,
    count=1,
)


###############################################################################
# TRAINER
###############################################################################

trainer_hits = re.findall(
    r"(?m)^TRAINER=.*$",
    text,
)


if len(
    trainer_hits
) != 1:

    raise RuntimeError(
        f"Expected one TRAINER line, got {trainer_hits}"
    )


text = re.sub(
    r"(?m)^TRAINER=.*$",
    (
        'TRAINER="$ROOT/scripts/mtpatcher_v5/'
        'train_opd_forwardkl_topk128_torchnpu.py"'
    ),
    text,
    count=1,
)


###############################################################################
# Fresh distributed port.
###############################################################################

port_hits = re.findall(
    r"--master_port=\d+",
    text,
)


if len(
    port_hits
) != 1:

    raise RuntimeError(
        f"Expected one master_port, got {port_hits}"
    )


text = re.sub(
    r"--master_port=\d+",
    "--master_port=29653",
    text,
    count=1,
)


###############################################################################
# Replace inherited run-path literals.
###############################################################################

text = text.replace(
    old_name,
    new_name,
)


###############################################################################
# Human-readable result labels.
###############################################################################

text = text.replace(
    "OPD-FKL-PE3732",
    "OPD-FKL-TOPK128-PE3732",
)

text = text.replace(
    "OPD-FKL",
    "OPD-FKL-TOPK128",
)


###############################################################################
# Verify scientific experiment remains controlled.
###############################################################################

checks = {
    "new name":
        new_name in text,

    "new trainer":
        "train_opd_forwardkl_topk128_torchnpu.py"
        in text,

    "16 ranks":
        "--nproc_per_node=16"
        in text,

    "PE3732":
        "pe_k1_clean3732.jsonl"
        in text,

    "fresh port":
        "--master_port=29653"
        in text,

    "old name removed":
        old_name
        not in text,
}


print(
    "TOPK128 MASTER CHECK"
)


for name, ok in checks.items():

    print(
        f"{name:28s} = {ok}"
    )


if not all(
    checks.values()
):

    raise RuntimeError(
        "Top-k master verification failed"
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
    "TOPK128_MASTER_BUILD_PASS"
)
PY


bash -n \
"$DST_MASTER"

echo "TOPK128_MASTER_SYNTAX_PASS"


###############################################################################
# STAGE 4 — FINAL STATIC AUDIT
###############################################################################

echo
echo "===== STAGE 4/6: FINAL STATIC AUDIT ====="

python - "$DST_TRAINER" <<'PY'
import ast
import sys
from pathlib import Path


p = Path(
    sys.argv[1]
)

text = p.read_text(
    encoding="utf-8"
)

tree = ast.parse(
    text
)


token_nodes = []


for node in ast.walk(
    tree
):

    if not isinstance(
        node,
        (
            ast.Assign,
            ast.AnnAssign,
        ),
    ):
        continue


    targets = (
        node.targets
        if isinstance(
            node,
            ast.Assign,
        )
        else [
            node.target
        ]
    )


    if any(
        isinstance(
            x,
            ast.Name,
        )
        and x.id == "token_kl"

        for x in targets
    ):

        token_nodes.append(
            node
        )


print(
    "TOKEN_KL_ASSIGNMENT_COUNT =",
    len(
        token_nodes
    ),
)


if len(
    token_nodes
) != 1:

    raise RuntimeError(
        "Expected one token_kl"
    )


seg = (
    ast.get_source_segment(
        text,
        token_nodes[
            0
        ],
    )
    or ""
)


print(
    "TOKEN_KL_SOURCE =",
    seg,
)


if "_topk_raw.clamp_min" not in seg:

    raise RuntimeError(
        "token_kl is not top-k clamped loss"
    )


required = [
    "_topk_k = 128",
    "_topk_t_logp",
    "_topk_t_prob",
    "_topk_s_logp",
    "_topk_teacher_mass",
    "_topk_raw",
    "TOPK128_FKL_RUNTIME_AUDIT",
]


for marker in required:

    if marker not in text:

        raise RuntimeError(
            f"Missing marker: {marker}"
        )


print(
    "TOPK128_FINAL_STATIC_AUDIT_PASS"
)
PY


###############################################################################
# STAGE 5 — FRESH LAUNCH GATE
###############################################################################

echo
echo "===== STAGE 5/6: LAUNCH ====="

CAN_LAUNCH=1


if [[ -e "$RUN_DIR" ]]; then

    echo "RUN DIRECTORY ALREADY EXISTS:"
    echo "$RUN_DIR"

    echo "FRESH-LAUNCH BLOCKED TO PROTECT EXISTING OUTPUT."

    CAN_LAUNCH=0

fi


RUNNING="$(
    pgrep -af \
    'train_opd_forwardkl_topk128_torchnpu.py' \
    || true
)"


if [[ -n "$RUNNING" ]]; then

    echo "EXISTING TOPK128 PROCESS:"
    echo "$RUNNING"

    echo "DUPLICATE-LAUNCH BLOCKED."

    CAN_LAUNCH=0

fi


if [[ "$CAN_LAUNCH" -eq 1 ]]; then

    mkdir -p \
    "$LOG_ROOT/$EXP"


    nohup setsid bash "$DST_MASTER" \
        > "$FULL_LOG" 2>&1 < /dev/null &


    PID=$!


    echo
    echo "======================================================================"
    echo "TOPK128 FORWARD-KL FULL RUN STARTED"
    echo "PID=$PID"
    echo "LOG=$FULL_LOG"
    echo "TRAINER=$DST_TRAINER"
    echo "MASTER=$DST_MASTER"
    echo "RUN_DIR=$RUN_DIR"
    echo "======================================================================"

fi


###############################################################################
# STAGE 6 — FIRST HEALTH / SCIENTIFIC CHECK
###############################################################################

echo
echo "===== STAGE 6/6: FIRST HEALTH CHECK ====="


if [[ "$CAN_LAUNCH" -eq 1 ]]; then

    sleep 120

fi


if [[ -f "$FULL_LOG" ]]; then

    echo
    echo "======================================================================"
    echo "TOPK128 FKL HEALTH"
    echo "======================================================================"


    grep -E \
'TOPK128_FKL_RUNTIME_AUDIT|TRAIN_ROWS|LOCAL_STEPS_PER_EPOCH|TOTAL_UPDATES|GLOBAL_BATCH|OPD_TRAINING_START|epoch=|EPOCH_|token_mean|CHECKPOINT|TRAINING_PASS|EVAL_PASS|FINAL SUMMARY|KEY COMPARISONS|ALL_PASS|Traceback|RuntimeError:|NameError:|FAILED' \
    "$FULL_LOG" \
    | tail -n 120

fi

