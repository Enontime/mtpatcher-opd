#!/usr/bin/env bash

set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

SRC_TRAINER="$ROOT/scripts/mtpatcher_v4/train_opd_pgrkl_cleanroom_v1_torchnpu.py"
DST_TRAINER="$ROOT/scripts/mtpatcher_v4/train_opd_pgrkl_cleanroom_v2_torchnpu.py"

SRC_MASTER="$ROOT/scripts/mtpatcher_v4/run_opd_pgrkl_cleanroom_v1_oneclick.sh"
DST_MASTER="$ROOT/scripts/mtpatcher_v4/run_opd_pgrkl_cleanroom_v2_oneclick.sh"

OLD_NAME="opd_torchnpu_pgrkl_cleanroom_pe3732_v1"
NEW_NAME="opd_torchnpu_pgrkl_cleanroom_pe3732_v2"

PREFLIGHT_LOG="$LOG_ROOT/$EXP/opd_torchnpu_pgrkl_cleanroom_preflight208_v1.log"

FULL_LOG="$LOG_ROOT/$EXP/${NEW_NAME}.log"


echo "======================================================================"
echo "CLEANROOM PG-RKL V2"
echo "PATCH: REMOVE STALE t_prob CLEANUP ONLY"
date
echo "======================================================================"


###############################################################################
# STAGE 1 — VERIFY EXISTING SCIENTIFIC PREFLIGHT
###############################################################################

echo
echo "===== STAGE 1/6: VERIFY PREFLIGHT ====="

test -f "$PREFLIGHT_LOG"

grep -E \
'CLEAN_PG_PREFLIGHT_FINAL|sampled_k1_mean|exact_rkl_mean|sample_exact_gap|behavior_forward_mae|importance_ratio_mean|sampled_tokens|CLEAN_PG_PREFLIGHT_PASS' \
"$PREFLIGHT_LOG"


if ! grep -q \
    'CLEAN_PG_PREFLIGHT_PASS' \
    "$PREFLIGHT_LOG"
then
    echo "PREFLIGHT PASS MARKER MISSING"
else
    echo "EXISTING_PREFLIGHT_PASS"
fi


###############################################################################
# STAGE 2 — PATCH ONLY THE STALE CLEANUP
###############################################################################

echo
echo "===== STAGE 2/6: PATCH TRAINER ====="

test -f "$SRC_TRAINER"

export SRC_TRAINER
export DST_TRAINER

python - <<'PY'
import ast
import difflib
import os
import re
from pathlib import Path


src = Path(os.environ["SRC_TRAINER"])
dst = Path(os.environ["DST_TRAINER"])

old = src.read_text(
    encoding="utf-8"
)

lines = old.splitlines(
    keepends=True
)


###############################################################################
# Find the specific cleanup del(...) block containing t_prob.
###############################################################################

remove_indices = []

inside_del = False
del_depth = 0


for i, line in enumerate(lines):

    stripped = line.strip()

    if not inside_del:

        if stripped == "del (":
            inside_del = True
            del_depth = 1

        continue


    # We are inside del(...)
    if re.fullmatch(
        r"t_prob,\s*",
        stripped,
    ):
        remove_indices.append(i)


    # The cleanup form used here closes on a line containing only ')'.
    if stripped == ")":
        inside_del = False
        del_depth = 0


print(
    "STALE_T_PROB_CLEANUP_LINES =",
    [
        i + 1
        for i in remove_indices
    ],
)


if len(remove_indices) != 1:

    raise RuntimeError(
        "Expected exactly one stale t_prob cleanup line, "
        f"found {len(remove_indices)}"
    )


new_lines = [
    line
    for i, line in enumerate(lines)
    if i not in remove_indices
]

new = "".join(
    new_lines
)


###############################################################################
# Syntax verification.
###############################################################################

tree = ast.parse(
    new
)


###############################################################################
# Verify there is no executable t_prob reference remaining.
###############################################################################

t_prob_loads = []

t_prob_dels = []


for node in ast.walk(tree):

    if (
        isinstance(node, ast.Name)
        and node.id == "t_prob"
    ):

        if isinstance(
            node.ctx,
            ast.Load,
        ):
            t_prob_loads.append(
                node.lineno
            )

        elif isinstance(
            node.ctx,
            ast.Del,
        ):
            t_prob_dels.append(
                node.lineno
            )


print(
    "T_PROB_LOAD_LINES =",
    t_prob_loads,
)

print(
    "T_PROB_DEL_LINES =",
    t_prob_dels,
)


if t_prob_loads or t_prob_dels:

    raise RuntimeError(
        "Executable t_prob reference remains after cleanup patch"
    )


###############################################################################
# Verify the scientific objective is still unique.
###############################################################################

def assignment_names(node):

    if isinstance(node, ast.Assign):
        targets = node.targets

    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]

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


k1_nodes = []

token_kl_nodes = []

pg_loss_nodes = []


for node in ast.walk(tree):

    if not isinstance(
        node,
        (
            ast.Assign,
            ast.AnnAssign,
        ),
    ):
        continue

    names = assignment_names(
        node
    )

    if "k1" in names:
        k1_nodes.append(node)

    if "token_kl" in names:
        token_kl_nodes.append(node)

    if "loss" in names:

        segment = (
            ast.get_source_segment(
                new,
                node,
            )
            or ""
        )

        if (
            "pg_unclipped" in segment
            and
            "pg_clipped" in segment
        ):
            pg_loss_nodes.append(
                node
            )


print()
print(
    "K1_ASSIGNMENT_COUNT =",
    len(k1_nodes),
)

print(
    "TOKEN_KL_ASSIGNMENT_COUNT =",
    len(token_kl_nodes),
)

print(
    "PG_LOSS_ASSIGNMENT_COUNT =",
    len(pg_loss_nodes),
)


if not (
    len(k1_nodes) == 1
    and
    len(token_kl_nodes) == 1
    and
    len(pg_loss_nodes) == 1
):

    raise RuntimeError(
        "Scientific objective changed unexpectedly"
    )


###############################################################################
# Required clean-room mechanics.
###############################################################################

required = [
    "CLEAN_PG_RUNTIME_AUDIT_PASS",
    "_pg_old_action_logp",
    "s_action_logp",
    "t_action_logp",
    "importance_ratio",
    "_pg_clip_eps = 0.20",
    "pg_unclipped",
    "pg_clipped",
]


for marker in required:

    if marker not in new:

        raise RuntimeError(
            f"Missing scientific marker after patch: {marker}"
        )


###############################################################################
# Show exact textual diff.
###############################################################################

diff = list(
    difflib.unified_diff(
        old.splitlines(),
        new.splitlines(),
        fromfile=str(src),
        tofile=str(dst),
        lineterm="",
    )
)


print()
print("PATCH DIFF")


for line in diff:
    print(line)


removed_content = [
    line
    for line in diff
    if (
        line.startswith("-")
        and not line.startswith("---")
    )
]


added_content = [
    line
    for line in diff
    if (
        line.startswith("+")
        and not line.startswith("+++")
    )
]


if len(removed_content) != 1:

    raise RuntimeError(
        "Patch changed more than one source line"
    )


if "t_prob," not in removed_content[0]:

    raise RuntimeError(
        "The single removed line was not t_prob cleanup"
    )


if added_content:

    raise RuntimeError(
        "Cleanup patch unexpectedly added source lines"
    )


dst.write_text(
    new,
    encoding="utf-8",
)


print()
print(
    "STALE_T_PROB_CLEANUP_PATCH_PASS"
)
PY


python -m py_compile \
"$DST_TRAINER"

echo "TRAINER_V2_COMPILE_PASS"


###############################################################################
# STAGE 3 — BUILD FRESH V2 MASTER
###############################################################################

echo
echo "===== STAGE 3/6: BUILD V2 MASTER ====="

test -f "$SRC_MASTER"

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

names = re.findall(
    r"(?m)^NAME=.*$",
    text,
)


if len(names) != 1:

    raise RuntimeError(
        f"Expected one NAME, got {names}"
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

trainers = re.findall(
    r"(?m)^TRAINER=.*$",
    text,
)


if len(trainers) != 1:

    raise RuntimeError(
        f"Expected one TRAINER, got {trainers}"
    )


text = re.sub(
    r"(?m)^TRAINER=.*$",
    (
        'TRAINER="$ROOT/scripts/mtpatcher_v4/'
        'train_opd_pgrkl_cleanroom_v2_torchnpu.py"'
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
        f"Expected one master_port, got {ports}"
    )


text = re.sub(
    r"--master_port=\d+",
    "--master_port=29651",
    text,
    count=1,
)


###############################################################################
# Replace any inherited explicit v1 run-path literals.
###############################################################################

text = text.replace(
    old_name,
    new_name,
)


###############################################################################
# Verify.
###############################################################################

checks = {
    "v2 name":
        new_name in text,

    "v2 trainer":
        "train_opd_pgrkl_cleanroom_v2_torchnpu.py"
        in text,

    "16 NPU":
        "--nproc_per_node=16"
        in text,

    "fresh port":
        "--master_port=29651"
        in text,

    "same PE3732":
        "pe_k1_clean3732.jsonl"
        in text,

    "clean result label":
        "OPD-PGRKL-K1-CLEANROOM-PE3732"
        in text,
}


print(
    "MASTER V2 CHECK"
)


for name, ok in checks.items():

    print(
        f"{name:28s} = {ok}"
    )


if not all(
    checks.values()
):

    raise RuntimeError(
        "Master V2 verification failed"
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
    "CLEANROOM_V2_MASTER_BUILD_PASS"
)
PY


bash -n \
"$DST_MASTER"

echo "MASTER_V2_SYNTAX_PASS"


###############################################################################
# STAGE 4 — FINAL SCIENTIFIC AUDIT
###############################################################################

echo
echo "===== STAGE 4/6: FINAL AUDIT ====="

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


def names(node):

    if isinstance(
        node,
        ast.Assign,
    ):
        xs = node.targets

    elif isinstance(
        node,
        ast.AnnAssign,
    ):
        xs = [node.target]

    else:
        return []

    return [
        x.id
        for x in xs
        if isinstance(
            x,
            ast.Name,
        )
    ]


counts = {
    "k1": 0,
    "token_kl": 0,
    "pg_loss": 0,
    "t_prob_exec": 0,
}


for node in ast.walk(tree):

    if isinstance(
        node,
        ast.Name,
    ) and node.id == "t_prob":

        counts[
            "t_prob_exec"
        ] += 1


    if not isinstance(
        node,
        (
            ast.Assign,
            ast.AnnAssign,
        ),
    ):
        continue


    ns = names(
        node
    )


    if "k1" in ns:
        counts["k1"] += 1


    if "token_kl" in ns:
        counts["token_kl"] += 1


    if "loss" in ns:

        seg = (
            ast.get_source_segment(
                text,
                node,
            )
            or ""
        )

        if (
            "pg_unclipped" in seg
            and
            "pg_clipped" in seg
        ):

            counts[
                "pg_loss"
            ] += 1


print(
    counts
)


if counts != {
    "k1": 1,
    "token_kl": 1,
    "pg_loss": 1,
    "t_prob_exec": 0,
}:

    raise RuntimeError(
        "Final scientific/static audit failed"
    )


print(
    "CLEANROOM_V2_FINAL_AUDIT_PASS"
)
PY


###############################################################################
# STAGE 5 — PROCESS CHECK + FRESH LAUNCH
###############################################################################

echo
echo "===== STAGE 5/6: LAUNCH ====="


if ! grep -q \
    'CLEAN_PG_PREFLIGHT_PASS' \
    "$PREFLIGHT_LOG"
then

    echo "FULL RUN BLOCKED: PREFLIGHT MARKER MISSING"

else

    RUNNING="$(
        pgrep -af \
        'train_opd_pgrkl_cleanroom_v2_torchnpu.py' \
        || true
    )"


    if [[ -n "$RUNNING" ]]
    then

        echo "Existing V2 clean-room process:"
        echo "$RUNNING"

        echo "DUPLICATE_LAUNCH_SKIPPED"

    else

        mkdir -p \
        "$LOG_ROOT/$EXP"


        nohup setsid bash "$DST_MASTER" \
            > "$FULL_LOG" 2>&1 < /dev/null &


        PID=$!


        echo
        echo "======================================================================"
        echo "CLEANROOM PG-RKL V2 STARTED"
        echo "PID=$PID"
        echo "LOG=$FULL_LOG"
        echo "TRAINER=$DST_TRAINER"
        echo "MASTER=$DST_MASTER"
        echo "======================================================================"

    fi

fi


###############################################################################
# STAGE 6 — VERIFY IT SURVIVES WELL BEYOND STEP 1
###############################################################################

echo
echo "===== STAGE 6/6: RUNTIME CHECK ====="

sleep 120


echo
echo "======================================================================"
echo "CLEANROOM PG-RKL V2 HEALTH"
echo "======================================================================"


grep -E \
'CLEAN_PG_RUNTIME_AUDIT_PASS|TRAIN_ROWS|LOCAL_STEPS_PER_EPOCH|TOTAL_UPDATES|GLOBAL_BATCH|OPD_TRAINING_START|epoch=|EPOCH_|token_mean|CHECKPOINT|TRAINING_PASS|FINAL SUMMARY|ALL_PASS|Traceback|UnboundLocalError|RuntimeError:|NameError:|FAILED' \
"$FULL_LOG" \
| tail -n 100

