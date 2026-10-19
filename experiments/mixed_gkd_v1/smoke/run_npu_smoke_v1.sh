#!/usr/bin/env bash

set +e
set +u
set +o pipefail 2>/dev/null

source /workspace/mtpatcher/project_env.sh

ROOT=/workspace/mtpatcher
PROJECT="$ROOT/repo/MT-Patcher-Reproduction-Ascend"
EXP="$PROJECT/experiments/mixed_gkd_v1"

NEW="$ROOT/repo/verl-v0.9.0-mixed-gkd-v1"
FORMAL="$ROOT/repo/verl-v0.9.0-matched20k-v2-formal"

A3="$ROOT/envs/verl-v0.9.0-a3"
PY="$A3/bin/python"
RAY="$A3/bin/ray"

PREFLIGHT="$EXP/preflight/preflight_configs_v1.py"

MODE="${1:-}"

case "$MODE" in
    --smoke-c)
        CFG="$EXP/configs/smoke/smoke-C_lambda05_fkl_2step.yaml"
        RUN="$ROOT/smoke/mixed_gkd_v1/npu/SMOKE-C"
        EXPECTED_LOSS="projected_fkl_topk_other"
        ;;

    --smoke-d)
        CFG="$EXP/configs/smoke/smoke-D_lambda05_jsd01_2step.yaml"
        RUN="$ROOT/smoke/mixed_gkd_v1/npu/SMOKE-D"
        EXPECTED_LOSS="projected_jsd01_topk_other"
        ;;

    --preflight-c)
        CFG="$EXP/configs/smoke/smoke-C_lambda05_fkl_2step.yaml"
        RUN="$ROOT/smoke/mixed_gkd_v1/npu/SMOKE-C"
        EXPECTED_LOSS="projected_fkl_topk_other"
        PREFLIGHT_ONLY=1
        ;;

    --preflight-d)
        CFG="$EXP/configs/smoke/smoke-D_lambda05_jsd01_2step.yaml"
        RUN="$ROOT/smoke/mixed_gkd_v1/npu/SMOKE-D"
        EXPECTED_LOSS="projected_jsd01_topk_other"
        PREFLIGHT_ONLY=1
        ;;

    *)
        echo "usage:"
        echo "  $0 --preflight-c"
        echo "  $0 --preflight-d"
        echo "  $0 --smoke-c"
        echo "  $0 --smoke-d"
        echo
        echo "FORMAL/PROBE EXECUTION IS NOT EXPOSED BY THIS LAUNCHER."
        exit 64
        ;;
esac


sha_file() {
    sha256sum "$1" | awk '{print $1}'
}


clean_ray() {
    "$RAY" stop --force \
        > /tmp/mixed_gkd_v1_ray_stop.log \
        2>&1 || true

    sleep 5

    local active

    active="$(
        ps -eo stat,pid,cmd |
        awk '$1 !~ /^Z/' |
        grep -E \
'raylet|gcs_server|ray/dashboard|TaskRunnerV1|vLLMHttpServer|GlobalRequestLoadBalancer' |
        grep -v grep || true
    )"

    if [[ -n "$active" ]]; then
        echo "RAY_CLEANUP=FAIL"
        echo "$active"
        return 1
    fi

    echo "RAY_CLEANUP=PASS"
    return 0
}


echo "============================================================"
echo "MIXED-GKD v1 — NPU INTEGRATION SMOKE"
echo "mode=$MODE"
echo "config=$CFG"
echo "run=$RUN"
echo "formal_probe_authorized=NO"
date -u
echo "============================================================"

cd "$PROJECT" || exit 2

export PYTHONPATH="$EXP/src:$EXP/reference:$NEW:$PROJECT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo
echo "===== [GATE 1] GLOBAL CONFIG PREFLIGHT ====="

"$PY" "$PREFLIGHT"
RC=$?

echo "GLOBAL_PREFLIGHT_RC=$RC"

if [[ "$RC" -ne 0 ]]; then
    echo "SMOKE=BLOCKED_GLOBAL_PREFLIGHT"
    exit 10
fi


echo
echo "===== [GATE 2] SELECTED CONFIG CONTRACT ====="

"$PY" - "$CFG" "$RUN" "$EXPECTED_LOSS" <<'PY'
from pathlib import Path
import sys
import yaml

cfg_path = Path(sys.argv[1])
run = str(Path(sys.argv[2]))
expected_loss = sys.argv[3]

cfg = yaml.safe_load(
    cfg_path.read_text(
        encoding="utf-8"
    )
)

meta = cfg["mixed_gkd_v1"]

assert meta["kind"] == "NPU_SMOKE"
assert (
    meta["execution_status"]
    == "SMOKE_ONLY_ALLOWED_AFTER_PREFLIGHT"
)
assert (
    meta["formal_training_authorized"]
    is False
)

assert float(
    meta["trajectory_lambda"]
) == 0.5


def override(key):
    hits = [
        str(x).split("=", 1)[1]
        for x in cfg["overrides"]
        if str(x).startswith(
            key + "="
        )
    ]

    assert len(hits) == 1, (
        key,
        hits,
    )

    return hits[0].strip("'\"")


assert int(
    override(
        "trainer.total_training_steps"
    )
) == 2

assert int(
    override(
        "data.train_batch_size"
    )
) == 16

assert (
    override(
        "data.shuffle"
    )
    == "False"
)

assert (
    "pds2048_lambda05_v1.parquet"
    in override(
        "data.train_files"
    )
)

assert (
    override(
        "distillation.distillation_loss.loss_mode"
    )
    == expected_loss
)

assert (
    int(
        override(
            "distillation.distillation_loss.topk"
        )
    )
    == 32
)

assert (
    override(
        "actor_rollout_ref.rollout.calculate_log_probs"
    )
    == "False"
)

assert (
    override(
        "trainer.use_v1"
    )
    == "True"
)

assert (
    override(
        "trainer.v1.trainer_mode"
    )
    == "sync"
)

manager = override(
    "+actor_rollout_ref.rollout.agent.agent_loop_manager_class"
)

assert manager == (
    "mtpatcher_mixed_gkd."
    "agent_loop_manager_v1."
    "MixedGKDv1AgentLoopManagerTQ"
)

ckpt = override(
    "trainer.default_local_dir"
)

assert ckpt == (
    run + "/checkpoints"
)

assert (
    "/runs/science/"
    not in ckpt
)

assert (
    "/smoke/mixed_gkd_v1/npu/"
    in ckpt
)

print(
    "SELECTED_SMOKE_CONFIG_GATE=PASS"
)
print(
    "loss_mode=",
    expected_loss,
)
print(
    "optimizer_steps=2"
)
print(
    "expected_step1_branch=ON_POLICY"
)
print(
    "expected_step2_branch=FIXED"
)
PY

RC=$?

echo "SELECTED_CONFIG_RC=$RC"

if [[ "$RC" -ne 0 ]]; then
    echo "SMOKE=BLOCKED_SELECTED_CONFIG"
    exit 11
fi


echo
echo "===== [GATE 3] FORMAL RUNTIME IMMUTABILITY ====="

if [[ -n "$(git -C "$FORMAL" status --porcelain)" ]]; then
    echo "FORMAL_VERL_DIRTY=BLOCKED"
    git -C "$FORMAL" status --short
    exit 12
fi

echo "FORMAL_VERL_CLEAN=PASS"


echo
echo "===== [GATE 4] ACTIVE RAY / TRAINING ====="

ACTIVE="$(
    ps -eo stat,pid,cmd |
    awk '$1 !~ /^Z/' |
    grep -E \
'raylet|gcs_server|ray/dashboard|TaskRunnerV1|vLLMHttpServer|GlobalRequestLoadBalancer|verl.trainer.main_ppo' |
    grep -v grep || true
)"

if [[ -n "$ACTIVE" ]]; then
    echo "ACTIVE_RAY_OR_TRAINING=BLOCKED"
    echo "$ACTIVE"
    exit 13
fi

echo "ACTIVE_RAY_OR_TRAINING=NONE"


echo
echo "===== [GATE 5] EXISTING RUN ====="

if [[ -e "$RUN/PASS" ]] || \
   [[ -e "$RUN/train.log" ]] || \
   [[ -d "$RUN/checkpoints" ]]
then
    echo "EXISTING_SMOKE_RUN=BLOCKED"
    echo "$RUN"
    exit 14
fi

echo "EXISTING_SMOKE_RUN=NONE"


echo
echo "===== [PROVENANCE] ====="

mkdir -p \
    "$RUN/provenance" \
    "$RUN/tensorboard"

git -C "$NEW" rev-parse HEAD \
    > "$RUN/provenance/candidate_verl_head.txt"

git -C "$FORMAL" rev-parse HEAD \
    > "$RUN/provenance/formal_verl_head.txt"

git -C "$PROJECT" rev-parse HEAD \
    > "$RUN/provenance/project_head.txt"

git -C "$PROJECT" status --short \
    > "$RUN/provenance/project_status.txt"

git -C "$NEW" status --short \
    > "$RUN/provenance/candidate_verl_status.txt"

git -C "$FORMAL" status --short \
    > "$RUN/provenance/formal_verl_status.txt"

sha256sum \
    "$CFG" \
    "$PREFLIGHT" \
    "$EXP/configs/agent_loops_v1.yaml" \
    "$EXP/src/mtpatcher_mixed_gkd/agent_loop_manager_v1.py" \
    "$EXP/src/mtpatcher_mixed_gkd/fixed_response_agent_v1.py" \
    "$ROOT/data/verl_science_pe_pds/mixed_gkd_v1_probe/training_assets_manifest_v1.json" \
    "$ROOT/data/verl_science_pe_pds/mixed_gkd_v1_probe/pds2048_lambda05_v1.parquet" \
    "$ROOT/data/verl_science_pe_pds/mixed_gkd_v1_probe/fixed_response_bank_v1.jsonl" \
    "$ROOT/data/verl_science_pe_pds/mixed_gkd_v1_probe/lambda05_step_mask_v1.jsonl" \
    > "$RUN/provenance/input_hashes.sha256"

git -C "$NEW" diff \
    --binary \
    --no-ext-diff \
    HEAD \
    -- \
    verl/trainer/distillation/fsdp/losses.py \
    verl/trainer/distillation/losses.py \
    > "$RUN/provenance/candidate_verl_production.patch"

sha256sum \
    "$RUN/provenance/candidate_verl_production.patch" \
    > "$RUN/provenance/candidate_verl_production.patch.sha256"

{
    echo "timestamp_utc=$(date -u +%FT%TZ)"
    echo "mode=$MODE"
    echo "config=$CFG"
    echo "run=$RUN"
    echo "expected_loss=$EXPECTED_LOSS"
    echo "formal_probe_authorized=NO"
    echo "PYTHONPATH=$PYTHONPATH"
} > "$RUN/provenance/run_identity.txt"

echo "PROVENANCE_CAPTURE=PASS"


if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo
    echo "MIXED_GKD_NPU_SMOKE_PREFLIGHT_ONLY=PASS"

    rm -rf "$RUN"

    printf '\a'
    exit 0
fi


echo
echo "===== [RAY CLEAN] ====="

clean_ray || exit 15


echo
echo "===== [LOAD OVERRIDES] ====="

mapfile -t OVERRIDES < <(
    "$PY" - "$CFG" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(
    open(
        sys.argv[1],
        encoding="utf-8",
    )
)

for item in cfg["overrides"]:
    print(item)
PY
)

printf 'override_count=%s\n' \
    "${#OVERRIDES[@]}"

printf '%s\n' \
    "${OVERRIDES[@]}" \
    > "$RUN/provenance/overrides.txt"


echo
echo "===== [NPU SMOKE START] ====="

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export TENSORBOARD_DIR="$RUN/tensorboard"

(
    cd "$RUN" || exit 1

    "$PY" \
        -m verl.trainer.main_ppo \
        "${OVERRIDES[@]}"
) > "$RUN/train.log" 2>&1

TRAIN_RC=$?

echo "$TRAIN_RC" \
    > "$RUN/train_rc.txt"

echo "NPU_SMOKE_TRAIN_RC=$TRAIN_RC"


echo
echo "===== [RAY CLEAN AFTER RUN] ====="

clean_ray
RAY_RC=$?

echo "$RAY_RC" \
    > "$RUN/ray_cleanup_rc.txt"

echo "NPU_SMOKE_RAY_CLEAN_RC=$RAY_RC"


if [[ "$TRAIN_RC" -ne 0 ]]; then
    echo
    echo "===== TRAIN FAILURE TAIL ====="

    tail -n 320 \
        "$RUN/train.log" \
        || true

    echo
    echo "MIXED_GKD_NPU_SMOKE=FAIL_TRAIN"

    printf '\a\a\a'
    exit 20
fi

if [[ "$RAY_RC" -ne 0 ]]; then
    echo "MIXED_GKD_NPU_SMOKE=FAIL_RAY_CLEANUP"
    printf '\a\a\a'
    exit 21
fi


echo
echo "===== [POST GATE] ====="

"$PY" - "$RUN" "$EXPECTED_LOSS" <<'PY'
from pathlib import Path
import math
import sys


run = Path(sys.argv[1])
expected_loss = sys.argv[2]

log = (
    run
    / "train.log"
)

assert log.is_file()
assert log.stat().st_size > 0

text = log.read_text(
    encoding="utf-8",
    errors="replace",
)

# The selected projected mode must appear in
# the resolved/runtime output.
assert expected_loss in text, (
    "selected projected loss mode "
    "not observed in train.log"
)

# Two-step run must reach its configured endpoint.
endpoint_markers = (
    "global_step",
    "global steps",
    "training/global_step",
)

assert any(
    marker in text
    for marker in endpoint_markers
), "no global-step evidence in log"

# The run must not silently invoke formal/probe paths.
assert (
    "/runs/science/mixed_gkd_v1/probe/"
    not in text
)

assert (
    "SPEC_ONLY_NOT_AUTHORIZED"
    not in text
)

# At least one checkpoint must have been produced
# under the smoke directory. save_freq=1.
ckpt = (
    run
    / "checkpoints"
)

assert ckpt.is_dir(), ckpt

steps = []

for p in ckpt.glob(
    "global_step_*"
):
    try:
        steps.append(
            int(
                p.name.rsplit(
                    "_",
                    1,
                )[1]
            )
        )
    except Exception:
        pass

assert steps, (
    "no smoke checkpoints"
)

assert max(steps) >= 2, (
    "2-step endpoint checkpoint missing: "
    f"{steps}"
)

print(
    "SMOKE_ENDPOINT_STEPS=",
    sorted(steps),
)

print(
    "MIXED_GKD_NPU_SMOKE_POST_GATE=PASS"
)
PY

POST_RC=$?

echo "NPU_SMOKE_POST_RC=$POST_RC"

if [[ "$POST_RC" -ne 0 ]]; then
    echo "MIXED_GKD_NPU_SMOKE=FAIL_POST_GATE"
    printf '\a\a\a'
    exit 22
fi

touch "$RUN/PASS"

echo
echo "============================================================"
echo "MIXED_GKD_NPU_SMOKE=PASS"
echo "mode=$MODE"
echo "run=$RUN"
echo "FORMAL_PROBE_EXECUTION=STILL_BLOCKED_NOT_AUTHORIZED"
echo "============================================================"

printf '\a\a'
exit 0
