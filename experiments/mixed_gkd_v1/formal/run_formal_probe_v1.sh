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

CFG_A="$EXP/configs/probe/GKD-A_lambda1_fkl.yaml"
CFG_C="$EXP/configs/probe/GKD-C_lambda05_fkl.yaml"
FINAL_EVAL="$EXP/manifests/FINAL_EVAL_RESERVED_v1.json"
AUTH="$EXP/manifests/FORMAL_PROBE_AUTHORIZATION_AC_v1.json"

EXPECTED_VERL_HEAD="ee2060725c3d5deff0d77ab4ffab4f30cd283dab"
EXPECTED_PRODUCTION_PATCH="aeb31078322f2206bba053109a8502baa3495457d487b87b0d1a3cc37fa61a8a"

MODE="${1:---preflight-ac}"

cd "$PROJECT"

export PYTHONPATH="$EXP/src:$EXP/reference:$NEW:$PROJECT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

get_override() {
    local cfg="$1"
    local key="$2"

    "$PY" - "$cfg" "$key" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(
    open(
        sys.argv[1],
        encoding="utf-8",
    )
)

key = sys.argv[2]

hits = [
    str(x).split("=", 1)[1]
    for x in cfg["overrides"]
    if str(x).startswith(
        key + "="
    )
]

if len(hits) != 1:
    raise SystemExit(
        f"{key}: expected one override, got {hits}"
    )

print(
    hits[0].strip("'\"")
)
PY
}

load_overrides() {
    local cfg="$1"

    "$PY" - "$cfg" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(
    open(
        sys.argv[1],
        encoding="utf-8",
    )
)

for item in cfg["overrides"]:
    print(str(item))
PY
}

active_payload() {
    ps -eo stat,pid,cmd \
      | awk '$1 !~ /^Z/' \
      | grep -E \
'raylet|gcs_server|ray/dashboard|TaskRunnerV1|vLLMHttpServer|GlobalRequestLoadBalancer|verl.trainer.main_ppo|verl.trainer.sft_trainer' \
      | grep -v grep \
      || true
}

clean_ray() {
    timeout 60s ray stop --force \
      >/tmp/mixed_gkd_formal_ray_stop.log \
      2>&1 || true

    sleep 2

    local active
    active="$(active_payload)"

    if [[ -n "$active" ]]; then
        echo "RAY_CLEANUP=FAIL"
        echo "$active"
        return 1
    fi

    echo "RAY_CLEANUP=PASS"
    return 0
}

reap_zombies_if_idle() {
    local active
    active="$(active_payload)"

    if [[ -n "$active" ]]; then
        echo "ZOMBIE_REAP=SKIP_ACTIVE_PAYLOAD"
        return 1
    fi

    local z
    z="$(
      ps -eo stat= \
        | awk '$1 ~ /^Z/ {n++} END {print n+0}'
    )"

    echo "zombies_before=$z"

    if [[ "$z" -gt 0 ]] \
       && [[ -f /tmp/reap_pid1.gdb ]] \
       && grep -qE \
            '^TracerPid:[[:space:]]+0$' \
            /proc/1/status
    then
        timeout 30s \
          gdb -q -n -batch \
          -p 1 \
          -x /tmp/reap_pid1.gdb \
          >/tmp/mixed_gkd_formal_reap.log \
          2>&1
    fi

    z="$(
      ps -eo stat= \
        | awk '$1 ~ /^Z/ {n++} END {print n+0}'
    )"

    echo "zombies_after=$z"

    [[ "$z" -eq 0 ]]
}

check_cell() {
    local cfg="$1"
    local expected_cell="$2"
    local auth_mode="$3"

    "$PY" - \
      "$cfg" \
      "$expected_cell" \
      "$auth_mode" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(
    open(
        sys.argv[1],
        encoding="utf-8",
    )
)

cell = sys.argv[2]
mode = sys.argv[3]
meta = cfg["mixed_gkd_v1"]

assert meta["kind"] == "PROBE_SPEC"
assert meta["scientific_cell"] == cell
assert int(meta["optimizer_steps"]) == 768

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
) == 768

assert int(
    override(
        "trainer.save_freq"
    )
) == 768

assert (
    override(
        "trainer.resume_mode"
    )
    == "disable"
)

assert int(
    override(
        "data.train_batch_size"
    )
) == 16

if mode == "prepared":
    assert (
        meta["execution_status"]
        == "SPEC_ONLY_NOT_AUTHORIZED"
    )
    assert (
        meta["formal_training_authorized"]
        is False
    )
elif mode == "authorized":
    assert (
        meta["execution_status"]
        == "FORMAL_AUTHORIZED"
    )
    assert (
        meta["formal_training_authorized"]
        is True
    )
else:
    raise AssertionError(mode)

print(
    f"{cell}_CONTRACT_{mode.upper()}=PASS"
)
PY
}

verify_authorization() {
    if [[ ! -f "$AUTH" ]]; then
        echo "FORMAL_AUTHORIZATION=MISSING"
        return 70
    fi

    "$PY" - "$AUTH" "$FINAL_EVAL" "$NEW" <<'PY_AUTH'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

auth_path = Path(sys.argv[1])
final_eval = Path(sys.argv[2])
candidate = Path(sys.argv[3])

auth = json.loads(
    auth_path.read_text(
        encoding="utf-8"
    )
)

assert auth["status"] == "FORMAL_AUTHORIZED"
assert auth["authorized_cells"] == [
    "GKD-A",
    "GKD-C",
]
assert auth["chain_order"] == [
    "GKD-A",
    "GKD-C",
]
assert auth["unauthorized_cells"] == [
    "GKD-B",
    "GKD-D",
]

assert (
    auth["scope"]["optimizer_steps_per_cell"]
    == 768
)

assert (
    auth["scope"]["evaluate_during_or_after_ac"]
    is False
)

assert (
    auth["scope"]["final_eval_after_all_four_cells"]
    is True
)

final_sha = hashlib.sha256(
    final_eval.read_bytes()
).hexdigest()

assert (
    final_sha
    == auth["final_eval_manifest_sha256"]
)

head = subprocess.check_output(
    [
        "git",
        "-C",
        str(candidate),
        "rev-parse",
        "HEAD",
    ],
    text=True,
).strip()

assert head == auth["candidate_verl_head"]

patch = subprocess.check_output(
    [
        "git",
        "-C",
        str(candidate),
        "diff",
        "--",
        "verl/trainer/distillation/fsdp/losses.py",
        "verl/trainer/distillation/losses.py",
    ]
)

patch_sha = hashlib.sha256(
    patch
).hexdigest()

assert (
    patch_sha
    == auth[
        "candidate_verl_production_patch_sha256"
    ]
)

print("FORMAL_AC_AUTHORIZATION=PASS")
print("AUTHORIZED_CELLS=GKD-A,GKD-C")
print("GKD-B_D_AUTHORIZED=NO")
PY_AUTH

    return $?
}


preflight_common() {
    local auth_mode="$1"

    echo "===== FORMAL COMMON PREFLIGHT ====="

    "$PY" \
      "$EXP/preflight/preflight_configs_v1.py"
    local global_rc=$?

    if [[ "$global_rc" -ne 0 ]]; then
        echo "GLOBAL_PREFLIGHT=FAIL"
        return 10
    fi

    if [[ ! -f "$FINAL_EVAL" ]]; then
        echo "FINAL_EVAL_MANIFEST=MISSING"
        return 11
    fi

    local candidate_head
    candidate_head="$(
      git -C "$NEW" rev-parse HEAD
    )"

    echo "candidate_head=$candidate_head"

    if [[ "$candidate_head" != "$EXPECTED_VERL_HEAD" ]]; then
        echo "CANDIDATE_HEAD=FAIL"
        return 12
    fi

    local formal_head
    formal_head="$(
      git -C "$FORMAL" rev-parse HEAD
    )"

    echo "formal_head=$formal_head"

    if [[ "$formal_head" != "$EXPECTED_VERL_HEAD" ]]; then
        echo "FORMAL_HEAD=FAIL"
        return 13
    fi

    local formal_status
    formal_status="$(
      git -C "$FORMAL" status --short
    )"

    if [[ -n "$formal_status" ]]; then
        echo "FORMAL_VERL_CLEAN=FAIL"
        echo "$formal_status"
        return 14
    fi

    echo "FORMAL_VERL_CLEAN=PASS"

    local patch_sha
    patch_sha="$(
      cp \
      "$AUTH" \
      "$run/provenance/FORMAL_PROBE_AUTHORIZATION_AC_v1.json"

    git -C "$NEW" diff -- \
        verl/trainer/distillation/fsdp/losses.py \
        verl/trainer/distillation/losses.py \
        | sha256sum \
        | awk '{print $1}'
    )"

    echo "candidate_production_patch=$patch_sha"

    if [[ "$patch_sha" != "$EXPECTED_PRODUCTION_PATCH" ]]; then
        echo "CANDIDATE_PRODUCTION_PATCH=FAIL"
        return 15
    fi

    echo "CANDIDATE_PRODUCTION_PATCH=PASS"

    local got

    got="$(
      sha256sum \
        "$ROOT/data/verl_science_pe_pds/mixed_gkd_v1_probe/pds2048_onpolicy_v1.parquet" \
        | awk '{print $1}'
    )"

    [[ "$got" == "65495714a6c30ad23bce75712318cd8835ced7ec93a0d88944f31f2fcf19444a" ]] \
      || return 16

    got="$(
      sha256sum \
        "$ROOT/data/verl_science_pe_pds/mixed_gkd_v1_probe/pds2048_lambda05_v1.parquet" \
        | awk '{print $1}'
    )"

    [[ "$got" == "e75120fcedc8bc872df2537cf71490120e9b52e0b447ba72f5891b5ac9c7301c" ]] \
      || return 17

    got="$(
      sha256sum \
        "$ROOT/data/verl_science_pe_pds/mixed_gkd_v1_probe/lambda05_step_mask_v1.jsonl" \
        | awk '{print $1}'
    )"

    [[ "$got" == "7d94f521893538bfe10a3481109c07f5ebd226d84d12cbdc86ea9d39d6db70bd" ]] \
      || return 18

    echo "FROZEN_ASSET_HASHES=PASS"

    check_cell \
      "$CFG_A" \
      "GKD-A" \
      "$auth_mode" \
      || return 19

    check_cell \
      "$CFG_C" \
      "GKD-C" \
      "$auth_mode" \
      || return 20

    local active
    active="$(active_payload)"

    if [[ -n "$active" ]]; then
        echo "ACTIVE_PAYLOAD=BLOCKED"
        echo "$active"
        return 21
    fi

    echo "ACTIVE_PAYLOAD=NONE"

    local run_a
    local run_c

    run_a="$(
      dirname "$(
        get_override \
          "$CFG_A" \
          "trainer.default_local_dir"
      )"
    )"

    run_c="$(
      dirname "$(
        get_override \
          "$CFG_C" \
          "trainer.default_local_dir"
      )"
    )"

    if [[ -e "$run_a" ]]; then
        echo "EXISTING_GKD_A_RUN=$run_a"
        return 22
    fi

    if [[ -e "$run_c" ]]; then
        echo "EXISTING_GKD_C_RUN=$run_c"
        return 23
    fi

    echo "EXISTING_FORMAL_RUNS=NONE"

    return 0
}

capture_provenance() {
    local cfg="$1"
    local run="$2"

    mkdir -p \
      "$run/provenance"

    cp \
      "$cfg" \
      "$run/provenance/config.yaml"

    cp \
      "$FINAL_EVAL" \
      "$run/provenance/FINAL_EVAL_RESERVED_v1.json"

    git -C "$NEW" diff -- \
      verl/trainer/distillation/fsdp/losses.py \
      verl/trainer/distillation/losses.py \
      > "$run/provenance/candidate_verl_production.patch"

    {
        echo "timestamp_utc=$(date -u +%FT%TZ)"
        echo "project_head=$(git -C "$PROJECT" rev-parse HEAD)"
        echo "candidate_verl_head=$(git -C "$NEW" rev-parse HEAD)"
        echo "formal_verl_head=$(git -C "$FORMAL" rev-parse HEAD)"
        echo "candidate_verl_production_patch_sha256=$(
          sha256sum \
            "$run/provenance/candidate_verl_production.patch" \
            | awk '{print $1}'
        )"
    } > "$run/provenance/source_state.txt"

    sha256sum \
      "$cfg" \
      "$FINAL_EVAL" \
      "$ROOT/data/verl_science_pe_pds/mixed_gkd_v1_probe/fixed_response_bank_v1.jsonl" \
      "$ROOT/data/verl_science_pe_pds/mixed_gkd_v1_probe/pds2048_onpolicy_v1.parquet" \
      "$ROOT/data/verl_science_pe_pds/mixed_gkd_v1_probe/pds2048_lambda05_v1.parquet" \
      "$ROOT/data/verl_science_pe_pds/mixed_gkd_v1_probe/lambda05_step_mask_v1.jsonl" \
      > "$run/provenance/input_hashes.sha256"
}

post_gate() {
    local run="$1"
    local cell="$2"

    if [[ ! -f "$run/train_rc.txt" ]]; then
        echo "${cell}_POST_GATE=NO_TRAIN_RC"
        return 30
    fi

    local rc
    rc="$(
      cat "$run/train_rc.txt"
    )"

    if [[ "$rc" -ne 0 ]]; then
        echo "${cell}_POST_GATE=TRAIN_FAIL"
        return 31
    fi

    if [[ ! -d "$run/checkpoints/global_step_768" ]]; then
        echo "${cell}_POST_GATE=STEP768_CHECKPOINT_MISSING"
        return 32
    fi

    if ! grep -q \
      'training/global_step:768' \
      "$run/train.log"
    then
        echo "${cell}_POST_GATE=STEP768_LOG_MISSING"
        return 33
    fi

    echo "${cell}_ENDPOINT_STEP=768"
    echo "${cell}_POST_GATE=PASS"

    touch "$run/PASS"

    return 0
}

run_cell() {
    local cfg="$1"
    local cell="$2"

    local checkpoint_dir
    local run

    checkpoint_dir="$(
      get_override \
        "$cfg" \
        "trainer.default_local_dir"
    )"

    run="$(
      dirname "$checkpoint_dir"
    )"

    mkdir -p "$run"

    capture_provenance \
      "$cfg" \
      "$run"

    mapfile -t OVERRIDES < <(
      load_overrides "$cfg"
    )

    echo "CELL=$cell"
    echo "RUN=$run"
    echo "override_count=${#OVERRIDES[@]}"
    echo "START_UTC=$(date -u +%FT%TZ)"

    export TENSORBOARD_DIR="$run/tensorboard"

    (
      cd "$run" || return 40

      "$PY" \
        -m verl.trainer.main_ppo \
        "${OVERRIDES[@]}"
    ) > "$run/train.log" 2>&1

    local train_rc=$?

    echo "$train_rc" \
      > "$run/train_rc.txt"

    echo "END_UTC=$(date -u +%FT%TZ)"
    echo "${cell}_TRAIN_RC=$train_rc"

    clean_ray
    local ray_rc=$?

    echo "$ray_rc" \
      > "$run/ray_cleanup_rc.txt"

    if [[ "$train_rc" -ne 0 ]]; then
        echo "${cell}=FAIL_TRAIN"
        tail -n 120 "$run/train.log"
        return 41
    fi

    if [[ "$ray_rc" -ne 0 ]]; then
        echo "${cell}=FAIL_RAY_CLEAN"
        return 42
    fi

    post_gate \
      "$run" \
      "$cell"

    return $?
}

echo "============================================================"
echo "MIXED-GKD FORMAL PROBE LAUNCHER"
echo "mode=$MODE"
date -u
echo "============================================================"

case "$MODE" in
    --preflight-ac)
        preflight_common prepared
        RC=$?

        if [[ "$RC" -eq 0 ]]; then
            echo "FORMAL_AC_PREP=PASS"
            echo "FORMAL_EXECUTION=BLOCKED_PENDING_EXPLICIT_AUTHORIZATION"
        else
            echo "FORMAL_AC_PREP=FAIL rc=$RC"
        fi
        ;;

    --preflight-authorized-ac)
        preflight_common prepared
        RC=$?

        if [[ "$RC" -eq 0 ]]; then
            verify_authorization
            RC=$?
        fi

        if [[ "$RC" -eq 0 ]]; then
            echo "FORMAL_AC_AUTHORIZED_PREFLIGHT=PASS"
            echo "FORMAL_EXECUTION=AUTHORIZED_NOT_STARTED_BY_PREFLIGHT"
        else
            echo "FORMAL_AC_AUTHORIZED_PREFLIGHT=FAIL rc=$RC"
        fi
        ;;

    --run-ac)
        preflight_common prepared
        RC=$?

        if [[ "$RC" -eq 0 ]]; then
            verify_authorization
            RC=$?
        fi

        if [[ "$RC" -ne 0 ]]; then
            echo "FORMAL_AC_RUN=BLOCKED_PREFLIGHT rc=$RC"
        else
            clean_ray
            RAY0=$?

            reap_zombies_if_idle
            REAP0=$?

            if [[ "$RAY0" -ne 0 ]] \
               || [[ "$REAP0" -ne 0 ]]
            then
                echo "FORMAL_AC_RUN=BLOCKED_HYGIENE"
                RC=50
            else
                run_cell \
                  "$CFG_A" \
                  "GKD-A"

                A_RC=$?

                if [[ "$A_RC" -ne 0 ]]; then
                    echo "FORMAL_AC_CHAIN=STOP_AFTER_A"
                    RC="$A_RC"
                else
                    reap_zombies_if_idle
                    MID_REAP=$?

                    if [[ "$MID_REAP" -ne 0 ]]; then
                        echo "FORMAL_AC_CHAIN=STOP_HYGIENE_BEFORE_C"
                        RC=51
                    else
                        run_cell \
                          "$CFG_C" \
                          "GKD-C"

                        C_RC=$?

                        if [[ "$C_RC" -ne 0 ]]; then
                            RC="$C_RC"
                        else
                            echo "FORMAL_AC_CHAIN=PASS"
                            RC=0
                        fi
                    fi
                fi
            fi
        fi
        ;;

    *)
        echo "Usage: $0 [--preflight-ac|--preflight-authorized-ac|--run-ac]"
        RC=2
        ;;
esac

echo "FORMAL_LAUNCHER_RC=$RC"

printf '\a\a\a'

exit "$RC"
