#!/usr/bin/env bash
set -euo pipefail

source /workspace/mtpatcher/project_env.sh

ROOT=/workspace/mtpatcher
PROJECT="$ROOT/repo/MT-Patcher-Reproduction-Ascend"
PY="$ROOT/envs/verl-v0.9.0-a3/bin/python"

EAEC_VERL="$ROOT/repo/verl-v0.9.0-matched20k-v2-eaec-v1"

CFG="$PROJECT/configs/opd/pe11792_eaec_opd_r2_v1.yaml"

PATCHBANK="$ROOT/data/verl_science_pe_pds/pe11792_matched_v1/eaec_patchbank_v1.jsonl"

WRAPPER="$PROJECT/scripts/targeted/main_ppo_sync_wrapper_v1.py"

RUN="$ROOT/runs/diagnostics/pe_pds_v1/eaec_one_step_smoke_v1"

die() {
    echo "EAEC_SMOKE_FAIL: $*" >&2
    return 1
}


###############################################################################
# Environment
###############################################################################

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Critical ordering:
# EAEC Verl must resolve before the frozen formal checkout already present
# in project_env.sh.
export PYTHONPATH="$PROJECT:$EAEC_VERL:${PYTHONPATH:-}"

export PATH="$ROOT/envs/verl-v0.9.0-a3/bin:$PATH"
unset PYTHONHOME || true
hash -r

export EAEC_PATCHBANK_JSONL="$PATCHBANK"
export EAEC_CORE_RATIO=2.0
export EAEC_AUDIT_JSONL="$RUN/eaec_runtime_audit.jsonl"


###############################################################################
# Hard preflight
###############################################################################

[[ -s "$PATCHBANK" ]] || die "PatchBank absent"
[[ -f "$CFG" ]] || die "EAEC config absent"
[[ -f "$WRAPPER" ]] || die "Verl wrapper absent"

EXPECTED_SHA="2a35b53dc397110b531260b27ed49cfc5400a83f4e70db95e06c3f56addc99f3"

GOT_SHA="$(sha256sum "$PATCHBANK" | awk '{print $1}')"

[[ "$GOT_SHA" == "$EXPECTED_SHA" ]] \
    || die "PatchBank SHA drift: $GOT_SHA"


###############################################################################
# Ensure the EAEC Verl checkout is the imported runtime.
###############################################################################

"$PY" - "$EAEC_VERL" <<'PY'
import sys
from pathlib import Path

expected = Path(sys.argv[1]).resolve()

import verl
import verl.trainer.distillation.losses as losses

verl_file = Path(verl.__file__).resolve()
loss_file = Path(losses.__file__).resolve()

print("VERL_FILE =", verl_file)
print("LOSS_FILE =", loss_file)

assert expected in verl_file.parents, (
    expected,
    verl_file,
)

assert expected in loss_file.parents, (
    expected,
    loss_file,
)

text = loss_file.read_text(
    encoding="utf-8",
)

assert (
    "MTP_EAEC_DISTILLATION_TOKEN_WEIGHTS_V1"
    in text
)

print("EAEC_RUNTIME_IMPORT_GATE=PASS")
PY


###############################################################################
# Make sure there is no already-running Verl training job.
###############################################################################

"$PY" - <<'PY'
import subprocess

text = subprocess.check_output(
    [
        "ps",
        "-eo",
        "pid=,stat=,args=",
    ],
    text=True,
    errors="replace",
)

bad = []

for line in text.splitlines():
    x = line.strip().split(
        None,
        2,
    )

    if len(x) < 3:
        continue

    pid, stat, cmd = x

    if stat.startswith("Z"):
        continue

    if (
        "main_ppo_sync_wrapper_v1.py" in cmd
        or "verl.trainer.main_ppo" in cmd
    ):
        bad.append(line)

if bad:
    print(
        "\n".join(bad)
    )
    raise SystemExit(
        "ACTIVE_VERL_TRAINER_GATE=FAIL"
    )

print(
    "ACTIVE_VERL_TRAINER_GATE=PASS"
)
PY


###############################################################################
# Clean only this smoke's output.
###############################################################################

rm -rf "$RUN"

mkdir -p \
    "$RUN/checkpoints" \
    "$RUN/tensorboard"


###############################################################################
# Convert the frozen EAEC formal protocol into ONE-STEP overrides.
#
# Everything stays identical except:
#   total_training_steps = 1
#   save_freq            = 1
#   experiment_name
#   output directory
###############################################################################

"$PY" - "$CFG" "$RUN/overrides.txt" "$RUN" <<'PY'
import sys
from pathlib import Path

from omegaconf import OmegaConf


cfg_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
run = Path(sys.argv[3])

meta = OmegaConf.load(
    cfg_path
)

overrides = list(
    meta.overrides
)

replace = {
    "trainer.total_training_steps":
        "trainer.total_training_steps=1",

    "trainer.save_freq":
        "trainer.save_freq=1",

    "trainer.experiment_name":
        "trainer.experiment_name="
        "pe11792-eaec-r2-one-step-smoke-v1",

    "trainer.default_local_dir":
        "trainer.default_local_dir="
        + str(run / "checkpoints"),

    "trainer.resume_mode":
        "trainer.resume_mode=disable",

    "trainer.val_before_train":
        "trainer.val_before_train=False",

    "trainer.test_freq":
        "trainer.test_freq=-1",
}


seen = set()
new = []

for item in overrides:
    text = str(item)

    normalized = text

    while normalized.startswith("+"):
        normalized = normalized[1:]

    key = normalized.split(
        "=",
        1,
    )[0]

    if key in replace:
        new.append(
            replace[key]
        )
        seen.add(key)
    else:
        new.append(text)


for key, value in replace.items():
    if key not in seen:
        new.append(value)


manager_key = (
    "actor_rollout_ref.rollout.agent."
    "agent_loop_manager_class"
)

manager_values = []

for x in new:
    y = x

    while y.startswith("+"):
        y = y[1:]

    if (
        y.split("=", 1)[0]
        == manager_key
    ):
        manager_values.append(x)

assert len(manager_values) == 1, (
    manager_values
)

assert (
    "EAECExactActiveEditCoreManagerV1"
    in manager_values[0]
)


# Freeze the exact scientific baseline constraints.
required = {
    "data.train_batch_size":
        "16",

    "actor_rollout_ref.actor.optim.lr":
        "1e-06",

    "actor_rollout_ref.rollout.n":
        "1",

    "actor_rollout_ref.rollout.temperature":
        "1.0",

    "distillation.distillation_loss.loss_mode":
        "forward_kl_topk",

    "distillation.distillation_loss.topk":
        "32",
}


actual = {}

for x in new:
    y = x

    while y.startswith("+"):
        y = y[1:]

    if "=" not in y:
        continue

    k, v = y.split(
        "=",
        1,
    )

    actual[k] = v


for k, expected in required.items():
    assert actual.get(k) == expected, (
        k,
        actual.get(k),
        expected,
    )


out_path.write_text(
    "\n".join(new)
    + "\n",
    encoding="utf-8",
)

print(
    "EAEC_ONE_STEP_OVERRIDE_GATE=PASS"
)

for x in new:
    print(x)
PY


###############################################################################
# Resolve config before spending accelerator time.
###############################################################################

mapfile -t OVERRIDES < "$RUN/overrides.txt"

(
    cd "$RUN"

    "$PY" "$WRAPPER" \
        --cfg job \
        "${OVERRIDES[@]}"
) > "$RUN/resolved_config.yaml"

[[ -s "$RUN/resolved_config.yaml" ]] \
    || die "resolved config absent"


###############################################################################
# Launch exactly one fresh-rollout + Teacher FKL update.
###############################################################################

echo "EAEC_ONE_STEP_START"

set +e

(
    cd "$RUN"

    export TENSORBOARD_DIR="$RUN/tensorboard"

    "$PY" "$WRAPPER" \
        "${OVERRIDES[@]}"
) > "$RUN/train.log" 2>&1

RC=$?

set -e

echo "EAEC_TRAIN_RC=$RC"

if [[ "$RC" -ne 0 ]]; then
    echo
    echo "===== TRAIN LOG TAIL ====="
    tail -n 240 "$RUN/train.log" || true
    die "training process failed rc=$RC"
fi


###############################################################################
# Runtime semantic gate.
###############################################################################

"$PY" - "$RUN/eaec_runtime_audit.jsonl" <<'PY'
import json
import math
import sys
from pathlib import Path


path = Path(
    sys.argv[1]
)

assert path.is_file(), path

rows = [
    json.loads(x)
    for x in path.read_text(
        encoding="utf-8"
    ).splitlines()
    if x.strip()
]

assert rows, (
    "EAEC manager produced no audit rows"
)

print(
    "EAEC_AUDIT_ROWS =",
    len(rows),
)

for row in rows:
    print(
        json.dumps(
            row,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )

    assert (
        row["status"]
        == "PASS_EAEC_RUNTIME_BATCH"
    )

    assert (
        row[
            "missing_patchbank_sources"
        ]
        == 0
    )

    assert (
        row[
            "valid_response_tokens"
        ]
        > 0
    )

    assert math.isclose(
        float(
            row[
                "token_weight_mean"
            ]
        ),
        1.0,
        rel_tol=0,
        abs_tol=1e-6,
    )

    assert (
        row[
            "token_weight_min"
        ]
        > 0
    )

    assert (
        row[
            "token_weight_max"
        ]
        <= 2.0 + 1e-6
    )


active = sum(
    int(
        x["active_patches"]
    )
    for x in rows
)

core = sum(
    int(
        x["core_tokens"]
    )
    for x in rows
)

fallback = sum(
    int(
        x[
            "tokenization_fallback"
        ]
    )
    for x in rows
)

print(
    "TOTAL_ACTIVE_PATCHES =",
    active,
)

print(
    "TOTAL_CORE_TOKENS =",
    core,
)

print(
    "TOTAL_TOKENIZATION_FALLBACK =",
    fallback,
)


# Active=0 is scientifically meaningful:
# exact historical patches did not survive this sampled current trajectory.
#
# Therefore don't fake PASS for the weighting mechanism.
if active == 0 or core == 0:
    print(
        "EAEC_ACTIVE_WEIGHT_PATH="
        "NO_ACTIVE_PATCH_IN_THIS_SMOKE"
    )
else:
    print(
        "EAEC_ACTIVE_WEIGHT_PATH=PASS"
    )

print(
    "EAEC_RUNTIME_SEMANTIC_GATE=PASS"
)
PY


###############################################################################
# Confirm the weighted-loss branch was actually executed.
###############################################################################

echo
echo "===== EAEC LOG EVIDENCE ====="

grep -E \
    'EAEC_MANAGER_INIT_PASS|EAEC_RUNTIME_BATCH_PASS|token_weight_mean|token_weight_min|token_weight_max|distillation/loss|grad_norm' \
    "$RUN/train.log" \
    | tail -n 120 || true


###############################################################################
# Checkpoint / one-step gate.
###############################################################################

echo
echo "===== CHECKPOINT TREE ====="

find "$RUN/checkpoints" \
    -maxdepth 3 \
    -type f \
    -o -type d \
    | head -160


if find "$RUN/checkpoints" \
    -mindepth 1 \
    -print \
    -quit \
    | grep -q .
then
    echo "EAEC_ONE_STEP_CHECKPOINT_GATE=PASS"
else
    die "one-step checkpoint absent"
fi


###############################################################################
# Frozen baseline checkout remains untouched.
###############################################################################

FORMAL_LOSS="$ROOT/repo/verl-v0.9.0-matched20k-v2-formal/verl/trainer/distillation/losses.py"

if grep -q \
    'MTP_EAEC_DISTILLATION_TOKEN_WEIGHTS_V1' \
    "$FORMAL_LOSS"
then
    die "formal baseline runtime contaminated"
fi

echo "FORMAL_RUNTIME_POST_SMOKE=PASS"

echo
echo "EAEC_ONE_STEP_INTEGRATION_SMOKE=PASS"
