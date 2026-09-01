#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

SRC_TRAINER="$ROOT/scripts/mtpatcher_v4/train_opd_pgrkl_cleanroom_v2_torchnpu.py"
SRC_MASTER="$ROOT/scripts/mtpatcher_v4/run_opd_pgrkl_cleanroom_v2_oneclick.sh"

DST_TRAINER="$ROOT/scripts/mtpatcher_v5/train_eopd_tau08_k16_torchnpu.py"
DST_MASTER="$ROOT/scripts/mtpatcher_v5/run_eopd_tau08_k16_oneclick.sh"

OLD_NAME="opd_torchnpu_pgrkl_cleanroom_pe3732_v2"
NEW_NAME="opd_torchnpu_eopd_tau08_k16_pe3732_v1"

RUN_DIR="$RUN_ROOT/$EXP/$NEW_NAME"
LOG="$LOG_ROOT/$EXP/${NEW_NAME}.log"


echo "======================================================================"
echo "ENTROPY-AWARE ON-POLICY DISTILLATION"
echo "tau=0.8 alpha=1.0 teacher-topk=16"
date
echo "======================================================================"


###############################################################################
# STAGE 1 — VERIFY CLEAN PG-RKL SOURCE
###############################################################################

echo
echo "===== STAGE 1/6: VERIFY CLEAN PG-RKL SOURCE ====="

test -f "$SRC_TRAINER"
test -f "$SRC_MASTER"

python -m py_compile "$SRC_TRAINER"
bash -n "$SRC_MASTER"

echo "SOURCE_TRAINER=$SRC_TRAINER"
echo "SOURCE_MASTER=$SRC_MASTER"
echo "CLEAN_PGRKL_SOURCE_PASS"


###############################################################################
# STAGE 2 — BUILD EOPD TRAINER
###############################################################################

echo
echo "===== STAGE 2/6: BUILD EOPD TRAINER ====="

export SRC_TRAINER
export DST_TRAINER

python - <<'PY'
import ast
import os
from pathlib import Path


src = Path(os.environ["SRC_TRAINER"])
dst = Path(os.environ["DST_TRAINER"])

text = src.read_text(encoding="utf-8")
tree = ast.parse(text)


def target_names(node):

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

    return out


def seg(node):
    return ast.get_source_segment(text, node) or ""


###############################################################################
# Locate the verified clean PG objective.
#
# Expected structure:
#
# loss = -torch.minimum(
#     pg_unclipped,
#     pg_clipped
# ).mean()
###############################################################################

loss_candidates = []


for node in ast.walk(tree):

    if not isinstance(
        node,
        (ast.Assign, ast.AnnAssign),
    ):
        continue

    if "loss" not in target_names(node):
        continue

    s = seg(node)

    if (
        "pg_unclipped" in s
        and "pg_clipped" in s
        and "minimum" in s
    ):
        loss_candidates.append(node)


print(
    "CLEAN_PG_LOSS_CANDIDATES =",
    len(loss_candidates),
)


for node in loss_candidates:
    print(
        f"  lines {node.lineno}-{node.end_lineno}:",
        seg(node).replace("\n", " "),
    )


if len(loss_candidates) != 1:
    raise RuntimeError(
        "Expected exactly one clean PG-RKL loss assignment"
    )


loss_node = loss_candidates[0]


###############################################################################
# Verify teacher/student full-distribution logprobs exist before PG loss.
###############################################################################

t_logp_nodes = []
s_logp_nodes = []


for node in ast.walk(tree):

    if not isinstance(
        node,
        (ast.Assign, ast.AnnAssign),
    ):
        continue

    names = target_names(node)

    if "t_logp" in names:
        t_logp_nodes.append(node)

    if "s_logp" in names:
        s_logp_nodes.append(node)


if not t_logp_nodes or not s_logp_nodes:
    raise RuntimeError(
        "Could not find t_logp / s_logp in clean trainer"
    )


if min(x.lineno for x in t_logp_nodes) >= loss_node.lineno:
    raise RuntimeError(
        "t_logp is not available before PG loss"
    )


if min(x.lineno for x in s_logp_nodes) >= loss_node.lineno:
    raise RuntimeError(
        "s_logp is not available before PG loss"
    )


###############################################################################
# Preserve the exact verified PG loss RHS.
###############################################################################

if isinstance(loss_node, ast.Assign):
    pg_rhs = ast.get_source_segment(
        text,
        loss_node.value,
    )
else:
    pg_rhs = ast.get_source_segment(
        text,
        loss_node.value,
    )


if not pg_rhs:
    raise RuntimeError(
        "Could not recover original clean PG loss expression"
    )


lines = text.splitlines()

first_line = lines[
    loss_node.lineno - 1
]

indent = first_line[
    :len(first_line) - len(first_line.lstrip())
]


###############################################################################
# EOPD:
#
#   L = L_PG-RKL
#       + alpha * I[H_teacher > tau] * FKL_top16
#
# Paper settings:
#   tau   = 0.8
#   alpha = 1.0
#   top-k = 16
#
# Teacher top-k distribution is RENORMALIZED inside top-k.
###############################################################################

replacement = f'''
{indent}# ============================================================
{indent}# ENTROPY-AWARE ON-POLICY DISTILLATION
{indent}#
{indent}# L_EOPD
{indent}#   = L_clean_PG_RKL
{indent}#   + alpha * I[H_teacher > tau] * FKL_top16
{indent}#
{indent}# Paper-style settings:
{indent}#   tau   = 0.8
{indent}#   alpha = 1.0
{indent}#   k     = 16
{indent}#
{indent}# Reverse-KL PG remains active at ALL response positions.
{indent}# Forward-KL is added only at high-teacher-entropy positions.
{indent}# ============================================================

{indent}_eopd_tau = 0.8
{indent}_eopd_alpha = 1.0
{indent}_eopd_topk = 16


{indent}# ------------------------------------------------------------
{indent}# Preserve the mathematically verified clean PG-RKL objective.
{indent}# ------------------------------------------------------------

{indent}_eopd_pg_loss = {pg_rhs}


{indent}# ------------------------------------------------------------
{indent}# Full teacher entropy:
{indent}#
{indent}# H_T(s_t) = -sum_v p_T(v|s_t) log p_T(v|s_t)
{indent}# ------------------------------------------------------------

{indent}_eopd_t_logp = t_logp.float()

{indent}_eopd_t_prob = (
{indent}    _eopd_t_logp.exp()
{indent})


{indent}_eopd_teacher_entropy = -(
{indent}    _eopd_t_prob
{indent}    * _eopd_t_logp
{indent}).sum(
{indent}    dim=-1
{indent})


{indent}if not torch.isfinite(
{indent}    _eopd_teacher_entropy
{indent}).all():

{indent}    raise RuntimeError(
{indent}        "EOPD teacher entropy contains non-finite values"
{indent}    )


{indent}_eopd_high_mask = (
{indent}    _eopd_teacher_entropy
{indent}    > _eopd_tau
{indent}).detach()


{indent}# ------------------------------------------------------------
{indent}# Teacher top-16 FKL.
{indent}#
{indent}# Unlike the previous verl-style Top128 experiment,
{indent}# EOPD RENORMALIZES teacher probability inside the selected
{indent}# top-k support.
{indent}# ------------------------------------------------------------

{indent}_eopd_topk_t_logp, _eopd_topk_ids = torch.topk(
{indent}    _eopd_t_logp,
{indent}    k=_eopd_topk,
{indent}    dim=-1,
{indent})


{indent}_eopd_topk_t_prob_raw = (
{indent}    _eopd_topk_t_logp.exp()
{indent})


{indent}_eopd_topk_mass = (
{indent}    _eopd_topk_t_prob_raw.sum(
{indent}        dim=-1,
{indent}        keepdim=True,
{indent}    )
{indent}).clamp_min(
{indent}    1.0e-20
{indent})


{indent}_eopd_topk_t_prob = (
{indent}    _eopd_topk_t_prob_raw
{indent}    / _eopd_topk_mass
{indent})


{indent}_eopd_topk_t_logp_norm = (
{indent}    torch.log(
{indent}        _eopd_topk_t_prob.clamp_min(
{indent}            1.0e-20
{indent}        )
{indent}    )
{indent})


{indent}_eopd_topk_s_logp = torch.gather(
{indent}    s_logp.float(),
{indent}    dim=-1,
{indent}    index=_eopd_topk_ids,
{indent})


{indent}_eopd_fkl_top16 = (
{indent}    _eopd_topk_t_prob
{indent}    * (
{indent}        _eopd_topk_t_logp_norm
{indent}        - _eopd_topk_s_logp
{indent}    )
{indent}).sum(
{indent}    dim=-1
{indent})


{indent}if not torch.isfinite(
{indent}    _eopd_fkl_top16
{indent}).all():

{indent}    raise RuntimeError(
{indent}        "EOPD top16 FKL contains non-finite values"
{indent}    )


{indent}if float(
{indent}    _eopd_fkl_top16.min().detach().item()
{indent}) < -1.0e-4:

{indent}    raise RuntimeError(
{indent}        "EOPD top16 FKL unexpectedly negative"
{indent}    )


{indent}# Guard only against tiny floating-point negatives.
{indent}_eopd_fkl_top16 = (
{indent}    _eopd_fkl_top16.clamp_min(
{indent}        0.0
{indent}    )
{indent})


{indent}# ------------------------------------------------------------
{indent}# Paper objective averages the gated extra FKL over all
{indent}# response-token positions.
{indent}# ------------------------------------------------------------

{indent}_eopd_fkl_loss = (
{indent}    _eopd_high_mask.to(
{indent}        _eopd_fkl_top16.dtype
{indent}    )
{indent}    * _eopd_fkl_top16
{indent}).mean()


{indent}loss = (
{indent}    _eopd_pg_loss
{indent}    + _eopd_alpha
{indent}    * _eopd_fkl_loss
{indent})


{indent}if not torch.isfinite(
{indent}    loss
{indent}):

{indent}    raise RuntimeError(
{indent}        "EOPD total loss is non-finite"
{indent}    )


{indent}# ------------------------------------------------------------
{indent}# First-batch scientific audit.
{indent}# ------------------------------------------------------------

{indent}if "_eopd_runtime_audit_done" not in locals():

{indent}    _eopd_entropy_cpu = (
{indent}        _eopd_teacher_entropy
{indent}        .detach()
{indent}        .float()
{indent}        .reshape(-1)
{indent}        .cpu()
{indent}    )


{indent}    _eopd_q = torch.quantile(
{indent}        _eopd_entropy_cpu,
{indent}        torch.tensor(
{indent}            [0.25, 0.50, 0.75, 0.90, 0.95],
{indent}            dtype=torch.float32,
{indent}        ),
{indent}    )


{indent}    _eopd_high_count = int(
{indent}        _eopd_high_mask.sum().detach().item()
{indent}    )


{indent}    _eopd_token_count = int(
{indent}        _eopd_high_mask.numel()
{indent}    )


{indent}    if _eopd_high_count > 0:

{indent}        _eopd_high_fkl_mean = float(
{indent}            _eopd_fkl_top16[
{indent}                _eopd_high_mask
{indent}            ].mean().detach().item()
{indent}        )

{indent}    else:

{indent}        _eopd_high_fkl_mean = 0.0


{indent}    if rank == 0:

{indent}        print(
{indent}            "EOPD_RUNTIME_AUDIT",
{indent}            {{
{indent}                "tau": _eopd_tau,
{indent}                "alpha": _eopd_alpha,
{indent}                "topk": _eopd_topk,
{indent}                "entropy_mean":
{indent}                    float(
{indent}                        _eopd_entropy_cpu.mean().item()
{indent}                    ),
{indent}                "entropy_q25":
{indent}                    float(_eopd_q[0].item()),
{indent}                "entropy_q50":
{indent}                    float(_eopd_q[1].item()),
{indent}                "entropy_q75":
{indent}                    float(_eopd_q[2].item()),
{indent}                "entropy_q90":
{indent}                    float(_eopd_q[3].item()),
{indent}                "entropy_q95":
{indent}                    float(_eopd_q[4].item()),
{indent}                "high_entropy_fraction":
{indent}                    float(
{indent}                        _eopd_high_count
{indent}                        / max(_eopd_token_count, 1)
{indent}                    ),
{indent}                "top16_teacher_mass_mean":
{indent}                    float(
{indent}                        _eopd_topk_mass.mean()
{indent}                        .detach().item()
{indent}                    ),
{indent}                "top16_teacher_mass_min":
{indent}                    float(
{indent}                        _eopd_topk_mass.min()
{indent}                        .detach().item()
{indent}                    ),
{indent}                "high_entropy_fkl_mean":
{indent}                    _eopd_high_fkl_mean,
{indent}                "gated_fkl_loss":
{indent}                    float(
{indent}                        _eopd_fkl_loss
{indent}                        .detach().item()
{indent}                    ),
{indent}                "pg_loss":
{indent}                    float(
{indent}                        _eopd_pg_loss
{indent}                        .detach().item()
{indent}                    ),
{indent}                "total_loss":
{indent}                    float(
{indent}                        loss.detach().item()
{indent}                    ),
{indent}            }},
{indent}            flush=True,
{indent}        )


{indent}    _eopd_runtime_audit_done = True
'''


lines[
    loss_node.lineno - 1:
    loss_node.end_lineno
] = replacement.splitlines()


result = "\n".join(lines) + "\n"


###############################################################################
# Static validation
###############################################################################

tree2 = ast.parse(result)


required = {
    "tau08":
        "_eopd_tau = 0.8" in result,

    "alpha1":
        "_eopd_alpha = 1.0" in result,

    "topk16":
        "_eopd_topk = 16" in result,

    "teacher entropy":
        "_eopd_teacher_entropy" in result,

    "hard entropy gate":
        "_eopd_high_mask" in result,

    "teacher topk":
        "torch.topk(" in result,

    "renormalization":
        "_eopd_topk_t_prob_raw" in result
        and "_eopd_topk_mass" in result,

    "gated fkl":
        "_eopd_fkl_loss" in result,

    "pg preserved":
        "_eopd_pg_loss" in result,

    "combined objective":
        "_eopd_pg_loss" in result
        and "_eopd_alpha" in result
        and "_eopd_fkl_loss" in result,

    "runtime audit":
        "EOPD_RUNTIME_AUDIT" in result,
}


print()
print("EOPD STATIC SCIENTIFIC CHECK")


for name, ok in required.items():
    print(
        f"{name:28s} = {ok}"
    )


if not all(required.values()):
    raise RuntimeError(
        "EOPD scientific static audit failed"
    )


###############################################################################
# Make sure clean k1 machinery survived.
###############################################################################

for marker in (
    "k1",
    "old_action_logp",
    "importance_ratio",
    "pg_unclipped",
    "pg_clipped",
):
    if marker not in result:
        raise RuntimeError(
            f"Clean PG-RKL marker missing after EOPD build: {marker}"
        )


dst.write_text(
    result,
    encoding="utf-8",
)


print()
print(
    "EOPD_TRAINER_BUILD_PASS"
)
PY


python -m py_compile "$DST_TRAINER"

echo "EOPD_TRAINER_COMPILE_PASS"


###############################################################################
# STAGE 3 — BUILD EOPD MASTER
###############################################################################

echo
echo "===== STAGE 3/6: BUILD EOPD MASTER ====="

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
# Replace experiment name.
###############################################################################

name_hits = re.findall(
    r"(?m)^NAME=.*$",
    text,
)


if len(name_hits) != 1:
    raise RuntimeError(
        f"Expected exactly one NAME line: {name_hits}"
    )


text = re.sub(
    r"(?m)^NAME=.*$",
    f'NAME="{new_name}"',
    text,
    count=1,
)


###############################################################################
# Replace trainer.
###############################################################################

trainer_hits = re.findall(
    r"(?m)^TRAINER=.*$",
    text,
)


if len(trainer_hits) != 1:
    raise RuntimeError(
        f"Expected exactly one TRAINER line: {trainer_hits}"
    )


text = re.sub(
    r"(?m)^TRAINER=.*$",
    (
        'TRAINER="$ROOT/scripts/mtpatcher_v5/'
        'train_eopd_tau08_k16_torchnpu.py"'
    ),
    text,
    count=1,
)


###############################################################################
# Fresh port.
###############################################################################

ports = re.findall(
    r"--master_port=\d+",
    text,
)


if len(ports) != 1:
    raise RuntimeError(
        f"Expected exactly one master_port: {ports}"
    )


text = re.sub(
    r"--master_port=\d+",
    "--master_port=29655",
    text,
    count=1,
)


###############################################################################
# Replace inherited path literals.
###############################################################################

text = text.replace(
    old_name,
    new_name,
)


###############################################################################
# Improve displayed final system label when present.
###############################################################################

text = text.replace(
    "OPD-PGRKL-K1-CLEANROOM-PE3732",
    "EOPD-TAU08-K16-PE3732",
)


checks = {
    "new name":
        new_name in text,

    "new trainer":
        "train_eopd_tau08_k16_torchnpu.py"
        in text,

    "16 ranks":
        "--nproc_per_node=16"
        in text,

    "PE3732":
        "pe_k1_clean3732.jsonl"
        in text,

    "fresh port":
        "--master_port=29655"
        in text,

    "old name removed":
        old_name not in text,
}


print("EOPD MASTER CHECK")


for name, ok in checks.items():
    print(
        f"{name:28s} = {ok}"
    )


if not all(checks.values()):
    raise RuntimeError(
        "EOPD master audit failed"
    )


dst.write_text(
    text,
    encoding="utf-8",
)

dst.chmod(0o755)


print()
print(
    "EOPD_MASTER_BUILD_PASS"
)
PY


bash -n "$DST_MASTER"

echo "EOPD_MASTER_SYNTAX_PASS"


###############################################################################
# STAGE 4 — FINAL SOURCE AUDIT
###############################################################################

echo
echo "===== STAGE 4/6: FINAL SOURCE AUDIT ====="

python - "$DST_TRAINER" <<'PY'
import ast
import sys
from pathlib import Path


p = Path(sys.argv[1])

text = p.read_text(
    encoding="utf-8"
)

ast.parse(text)


required = (
    "_eopd_tau = 0.8",
    "_eopd_alpha = 1.0",
    "_eopd_topk = 16",
    "_eopd_teacher_entropy",
    "_eopd_high_mask",
    "_eopd_topk_t_prob",
    "_eopd_fkl_top16",
    "_eopd_pg_loss",
    "_eopd_fkl_loss",
    "EOPD_RUNTIME_AUDIT",
)


for marker in required:

    n = text.count(marker)

    print(
        f"{marker:34s} count={n}"
    )

    if n == 0:
        raise RuntimeError(
            f"Missing EOPD marker: {marker}"
        )


if text.count(
    "EOPD_RUNTIME_AUDIT"
) != 1:

    raise RuntimeError(
        "Expected exactly one EOPD runtime audit"
    )


print()
print(
    "EOPD_FINAL_STATIC_AUDIT_PASS"
)
PY


###############################################################################
# STAGE 5 — FRESH LAUNCH
###############################################################################

echo
echo "===== STAGE 5/6: FRESH LAUNCH ====="

CAN_LAUNCH=1


if [[ -e "$RUN_DIR" ]]; then

    echo "RUN DIRECTORY ALREADY EXISTS:"
    echo "$RUN_DIR"

    echo "Existing output protected; launch blocked."

    CAN_LAUNCH=0

fi


RUNNING="$(
    pgrep -af \
    'train_eopd_tau08_k16_torchnpu.py' \
    || true
)"


if [[ -n "$RUNNING" ]]; then

    echo "EOPD PROCESS ALREADY RUNNING:"
    echo "$RUNNING"

    CAN_LAUNCH=0

fi


if [[ "$CAN_LAUNCH" -eq 1 ]]; then

    mkdir -p "$LOG_ROOT/$EXP"

    nohup setsid bash "$DST_MASTER" \
        > "$LOG" 2>&1 < /dev/null &

    PID=$!

    echo
    echo "======================================================================"
    echo "EOPD FULL RUN STARTED"
    echo "PID=$PID"
    echo "LOG=$LOG"
    echo "RUN_DIR=$RUN_DIR"
    echo "TRAINER=$DST_TRAINER"
    echo "MASTER=$DST_MASTER"
    echo "======================================================================"

fi


###############################################################################
# STAGE 6 — FIRST SCIENTIFIC HEALTH CHECK
###############################################################################

echo
echo "===== STAGE 6/6: FIRST SCIENTIFIC HEALTH CHECK ====="


if [[ "$CAN_LAUNCH" -eq 1 ]]; then
    sleep 120
fi


if [[ -f "$LOG" ]]; then

    echo
    echo "======================================================================"
    echo "EOPD HEALTH"
    echo "======================================================================"

    grep -E \
'EOPD_RUNTIME_AUDIT|CLEAN_PG_RUNTIME|TRAIN_ROWS|LOCAL_STEPS_PER_EPOCH|TOTAL_UPDATES|GLOBAL_BATCH|OPD_TRAINING_START|epoch=|EPOCH_|token_mean|CHECKPOINT|TRAINING_PASS|EVAL_PASS|FINAL SUMMARY|KEY COMPARISONS|ALL_PASS|Traceback|RuntimeError:|NameError:|FAILED' \
    "$LOG" \
    | tail -n 120

fi

