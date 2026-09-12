#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/mtpatcher
REPO="$ROOT/repo/MT-Patcher-Reproduction-Ascend"
VERL="$ROOT/repo/verl-v0.9.0"
VERL_ENV_ROOT="$ROOT/envs/verl-v0.9.0-a3"

RUN=${RUN:?RUN must be provided}
CONFIG="$REPO/configs/targeted/offline_prefix_support_native_verl_smoke64_v1.yaml"
MANAGER="$REPO/scripts/targeted/offline_prefix_support_verl_replay_manager_v1.py"
ZERO_REWARD="$REPO/scripts/opd/constant_zero_reward.py"

STUDENT="$ROOT/models/Qwen3-0.6B"
TEACHER="$ROOT/models/Qwen3-8B"

EXPECTED_HEAD_FILE="$RUN/implementation_commit.txt"
STATE="$RUN/state.json"
MASTER_SUMMARY="$RUN/summary.json"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO/scripts/targeted:$REPO:$VERL:${PYTHONPATH:-}"

if [[ -f "$ROOT/project_env.sh" ]]; then
    # shellcheck disable=SC1090
    source "$ROOT/project_env.sh"
fi

# project_env.sh configures the Ascend stack but does not select the Verl Python
# environment. Force the exact validated env before any Verl/module import.
[[ -x "$VERL_ENV_ROOT/bin/python" ]] || {
    echo "FAIL: missing Verl Python: $VERL_ENV_ROOT/bin/python" >&2
    exit 1
}
export PATH="$VERL_ENV_ROOT/bin:$PATH"
unset PYTHONHOME || true
hash -r

python - "$VERL_ENV_ROOT" <<'PY_ENV_GATE'
import importlib
import os
import sys
from pathlib import Path

env = Path(sys.argv[1]).resolve()
exe_raw = Path(sys.executable)
exe = exe_raw.resolve()
expected_raw = env / "bin/python"
expected = expected_raw.resolve()
assert exe == expected, f"wrong python executable: got={exe} expected={expected}"
assert Path(sys.prefix).resolve() == env, (
    f"wrong python prefix: got={Path(sys.prefix).resolve()} expected={env}"
)
for name in (
    "tensordict",
    "torch",
    "torch_npu",
    "ray",
    "pyarrow",
    "omegaconf",
    "tensorboard",
    "transformers",
):
    importlib.import_module(name)
print(f"VERL_ENV_GATE=PASS python={exe_raw} prefix={Path(sys.prefix).resolve()}")
PY_ENV_GATE

die() {
    echo "FAIL: $*" >&2
    exit 1
}

atomic_state() {
    local status="$1"
    local phase="$2"
    python - "$STATE" "$status" "$phase" <<'PY'
import json, os, sys, time
p, status, phase = sys.argv[1:]
obj = {
    "status": status,
    "phase": phase,
    "formal_training_authorized": False,
    "scientific_class": "DIAGNOSTIC ONLY / OFFLINE PREFIX-SUPPORT REPLAY",
    "updated_unix": time.time(),
}
tmp = p + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
    f.write("\n")
    f.flush()
    os.fsync(f.fileno())
os.replace(tmp, p)
PY
}

sha256_file() {
    sha256sum "$1" | awk '{print $1}'
}

mkdir -p "$RUN"

on_exit() {
    local rc=$?
    if [[ "$rc" -ne 0 ]]; then
        set +e
        atomic_state "FAIL" "engineering_smoke"
        touch "$RUN/FAIL"
        echo "ENGINEERING_SMOKE_FAILED rc=$rc" >&2
    fi
    return "$rc"
}
trap on_exit EXIT

atomic_state "RUNNING" "preflight"

echo "Question: does native Verl consume exact frozen S/T response IDs and produce a finite one-update top-k32 FKL backward path without any fresh Student rollout?"
echo "Competing explanations: replay transport works; custom hook silently fresh-rolls; Teacher tensors misalign; native actor backward path rejects replay batch."
echo "Falsifiable prediction: exact replay SHA survives transport, fresh-rollout tripwire stays at zero, Teacher top-k32 aligns, actor grad_norm>0 and TensorBoard records one step for both arms."
echo "Decision after result: PASS authorizes scientific review only; it does not authorize formal Chemistry/Idiom S/T training."

[[ -s "$CONFIG" ]] || die "missing config"
[[ -s "$MANAGER" ]] || die "missing manager"
[[ -s "$ZERO_REWARD" ]] || die "missing zero reward"
[[ -s "$RUN/input_manifest.json" ]] || die "missing input manifest"
[[ -s "$EXPECTED_HEAD_FILE" ]] || die "missing implementation commit file"

IMPLEMENTATION_HEAD="$(cat "$EXPECTED_HEAD_FILE")"
CURRENT_HEAD="$(git -C "$REPO" rev-parse HEAD)"
[[ "$CURRENT_HEAD" == "$IMPLEMENTATION_HEAD" ]] \
    || die "repo HEAD drift current=$CURRENT_HEAD expected=$IMPLEMENTATION_HEAD"

# No tracked edits are allowed after the implementation commit.
[[ -z "$(git -C "$REPO" diff --name-only)" ]] || die "tracked worktree edits exist"
[[ -z "$(git -C "$REPO" diff --cached --name-only)" ]] || die "staged edits exist"

python - "$RUN/input_manifest.json" "$CONFIG" <<'PY'
import json, sys, yaml
m = json.load(open(sys.argv[1], encoding="utf-8"))
c = yaml.safe_load(open(sys.argv[2], encoding="utf-8"))

assert m["status"] == "PASS_INPUTS_FROZEN_FOR_ENGINEERING_SMOKE"
assert m["formal_training_authorized"] is False
assert m["rows"] == 64
assert m["domain"] == "chemistry"
assert set(m["arms"]) == {"S", "T"}

assert c["formal_training_authorized"] is False
assert c["rows"] == 64
assert c["objective"]["loss_mode"] == "forward_kl_topk"
assert c["objective"]["teacher_topk"] == 32
assert c["objective"]["use_policy_gradient"] is False
assert c["optimization"]["optimizer_updates_per_arm"] == 1
assert c["replay_transport"]["fresh_student_rollout_forbidden"] is True
assert c["tensorboard"]["required"] is True
print("RUNNER_STATIC_CONTRACT=PASS")
PY

# Detect module FQN import before Ray is started.
python - <<'PY'
import importlib
m = importlib.import_module("offline_prefix_support_verl_replay_manager_v1")
assert hasattr(m, "OfflinePrefixSupportReplayManager")
print("CUSTOM_MANAGER_IMPORT=PASS")
PY

# Fail rather than competing with another formal main_ppo process.
if pgrep -af '[p]ython.*verl\.trainer\.main_ppo' > "$RUN/preexisting_main_ppo.txt"; then
    cat "$RUN/preexisting_main_ppo.txt"
    die "pre-existing Verl main_ppo process detected"
fi

npu-smi info > "$RUN/npu_before.txt" 2>&1 || true

build_overrides() {
    local arm="$1"
    local arm_run="$2"
    local parquet="$RUN/inputs/chemistry_${arm}_first64.parquet"
    local project="mtpatcher-prefix-replay-smoke"
    local exp="chemistry-${arm}-native-verl-oneupdate-v1"

    cat <<EOF
data.train_files=['$parquet']
data.val_files=['$parquet']
data.train_batch_size=64
data.max_prompt_length=1024
data.max_response_length=256
data.filter_overlong_prompts=False
data.truncation=error
data.shuffle=False
+data.apply_chat_template_kwargs.enable_thinking=False
algorithm.adv_estimator=grpo
algorithm.use_kl_in_reward=False
reward.custom_reward_function.path=$ZERO_REWARD
reward.custom_reward_function.name=compute_score
actor_rollout_ref.model.path=$STUDENT
actor_rollout_ref.model.use_remove_padding=True
actor_rollout_ref.model.enable_gradient_checkpointing=False
actor_rollout_ref.actor.use_torch_compile=False
actor_rollout_ref.actor.optim.lr=1.0e-6
actor_rollout_ref.actor.ppo_mini_batch_size=64
actor_rollout_ref.actor.ppo_epochs=1
actor_rollout_ref.actor.data_loader_seed=20260820
actor_rollout_ref.actor.shuffle=False
actor_rollout_ref.actor.loss_agg_mode=token-mean
actor_rollout_ref.actor.use_dynamic_bsz=True
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096
actor_rollout_ref.actor.fsdp_config.param_offload=False
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
actor_rollout_ref.actor.use_kl_loss=False
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.tensor_model_parallel_size=2
actor_rollout_ref.rollout.n=1
actor_rollout_ref.rollout.temperature=1.0
actor_rollout_ref.rollout.top_p=1.0
actor_rollout_ref.rollout.top_k=-1
actor_rollout_ref.rollout.dtype=bfloat16
actor_rollout_ref.rollout.gpu_memory_utilization=0.35
actor_rollout_ref.rollout.max_model_len=1281
actor_rollout_ref.rollout.enforce_eager=True
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096
++actor_rollout_ref.rollout.agent.agent_loop_manager_class=offline_prefix_support_verl_replay_manager_v1.OfflinePrefixSupportReplayManager
distillation.enabled=True
distillation.distillation_loss.loss_mode=forward_kl_topk
distillation.distillation_loss.topk=32
distillation.distillation_loss.use_task_rewards=False
distillation.distillation_loss.use_policy_gradient=False
distillation.distillation_loss.loss_max_clamp=null
distillation.distillation_loss.log_prob_min_clamp=null
distillation.n_gpus_per_node=4
distillation.nnodes=1
distillation.teacher_models.teacher_model.model_path=$TEACHER
distillation.teacher_models.teacher_model.inference.name=vllm
distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=2
distillation.teacher_models.teacher_model.inference.data_parallel_size=1
distillation.teacher_models.teacher_model.inference.dtype=bfloat16
distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.40
distillation.teacher_models.teacher_model.inference.enforce_eager=True
distillation.teacher_models.teacher_model.inference.max_model_len=1281
trainer.nnodes=1
trainer.n_gpus_per_node=8
trainer.project_name=$project
trainer.experiment_name=$exp
trainer.logger=['console','tensorboard']
trainer.val_before_train=False
trainer.test_freq=-1
trainer.total_epochs=1
trainer.total_training_steps=1
trainer.save_freq=-1
trainer.default_local_dir=$arm_run/checkpoints
trainer.resume_mode=disable
trainer.balance_batch=False
trainer.critic_warmup=0
EOF
}

compose_arm() {
    local arm="$1"
    local arm_run="$RUN/chemistry/$arm"
    mkdir -p "$arm_run/checkpoints" "$arm_run/tensorboard"
    build_overrides "$arm" "$arm_run" > "$arm_run/overrides.txt"
    mapfile -t OVS < "$arm_run/overrides.txt"

    (
        cd "$arm_run"
        python -m verl.trainer.main_ppo_v0 --cfg job "${OVS[@]}"
    ) > "$arm_run/resolved_config.yaml"

    python - "$arm" "$arm_run/resolved_config.yaml" "$RUN/input_manifest.json" <<'PY'
import json, sys
from omegaconf import OmegaConf

arm, path, manifest_path = sys.argv[1:]
cfg = OmegaConf.load(path)
m = json.load(open(manifest_path, encoding="utf-8"))

expected_parquet = m["arms"][arm]["parquet"]
assert list(cfg.data.train_files) == [expected_parquet]
assert cfg.data.train_batch_size == 64
assert cfg.data.shuffle is False
assert cfg.actor_rollout_ref.model.path == "/workspace/mtpatcher/models/Qwen3-0.6B"
assert cfg.actor_rollout_ref.rollout.n == 1
assert float(cfg.actor_rollout_ref.rollout.temperature) == 1.0
assert int(cfg.actor_rollout_ref.rollout.top_k) == -1
assert cfg.distillation.enabled is True
assert cfg.distillation.distillation_loss.loss_mode == "forward_kl_topk"
assert int(cfg.distillation.distillation_loss.topk) == 32
assert cfg.distillation.distillation_loss.use_task_rewards is False
assert cfg.distillation.distillation_loss.use_policy_gradient is False
assert cfg.actor_rollout_ref.actor.use_kl_loss is False
assert cfg.algorithm.use_kl_in_reward is False
assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 64
assert cfg.actor_rollout_ref.actor.ppo_epochs == 1
assert int(cfg.actor_rollout_ref.actor.data_loader_seed) == 20260820
assert cfg.actor_rollout_ref.actor.shuffle is False
assert cfg.trainer.total_training_steps == 1
assert cfg.trainer.save_freq == -1
assert cfg.trainer.val_before_train is False
assert cfg.trainer.balance_batch is False
assert list(cfg.trainer.logger) == ["console", "tensorboard"]
fqn = cfg.actor_rollout_ref.rollout.agent.agent_loop_manager_class
assert fqn == "offline_prefix_support_verl_replay_manager_v1.OfflinePrefixSupportReplayManager"
print(f"RESOLVED_CONFIG_GATE=PASS arm={arm}")
PY
}

echo "=== COMPOSE BOTH ARMS BEFORE ANY UPDATE ==="
compose_arm S
compose_arm T

python - "$RUN/chemistry/S/resolved_config.yaml" "$RUN/chemistry/T/resolved_config.yaml" <<'PY'
from pathlib import Path
from omegaconf import OmegaConf
import json
import sys

def load(path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)

def arm_tokens(cfg):
    train_files = list(cfg["data"]["train_files"])
    val_files = list(cfg["data"]["val_files"])
    if len(train_files) != 1 or len(val_files) != 1:
        raise SystemExit(
            f"expected one train/val file, got train={train_files!r} val={val_files!r}"
        )

    local_dir = str(cfg["trainer"]["default_local_dir"])
    arm_run = str(Path(local_dir).parent)
    exp = str(cfg["trainer"]["experiment_name"])

    if not arm_run or not exp or not train_files[0]:
        raise SystemExit("empty arm-specific resolved-config token")

    return sorted(
        {
            str(train_files[0]): "<ARM_PARQUET>",
            str(val_files[0]): "<ARM_PARQUET>",
            arm_run: "<ARM_RUN>",
            exp: "<ARM_EXPERIMENT>",
        }.items(),
        key=lambda kv: len(kv[0]),
        reverse=True,
    )

def canonicalize(obj, replacements):
    if isinstance(obj, dict):
        return {k: canonicalize(v, replacements) for k, v in obj.items()}
    if isinstance(obj, list):
        return [canonicalize(v, replacements) for v in obj]
    if isinstance(obj, str):
        out = obj
        for raw, marker in replacements:
            out = out.replace(raw, marker)
        return out
    return obj

def collect_diffs(a, b, path="$", out=None, limit=80):
    if out is None:
        out = []
    if len(out) >= limit:
        return out
    if type(a) is not type(b):
        out.append((path, a, b))
        return out
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if len(out) >= limit:
                break
            if k not in a or k not in b:
                out.append((f"{path}.{k}", a.get(k, "<MISSING>"), b.get(k, "<MISSING>")))
            else:
                collect_diffs(a[k], b[k], f"{path}.{k}", out, limit)
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            out.append((path + ".length", len(a), len(b)))
            return out
        for i, (x, y) in enumerate(zip(a, b)):
            if len(out) >= limit:
                break
            collect_diffs(x, y, f"{path}[{i}]", out, limit)
        return out
    if a != b:
        out.append((path, a, b))
    return out

s_raw = load(sys.argv[1])
t_raw = load(sys.argv[2])

raw_diffs = collect_diffs(s_raw, t_raw)
print(f"S_T_RAW_RESOLVED_CONFIG_DIFF_COUNT={len(raw_diffs)}")
for path, sv, tv in raw_diffs[:40]:
    print(
        "ARM_SPECIFIC_RAW_DIFF "
        + json.dumps(
            {"path": path, "S": sv, "T": tv},
            ensure_ascii=False,
            default=str,
        )
    )

s = canonicalize(s_raw, arm_tokens(s_raw))
t = canonicalize(t_raw, arm_tokens(t_raw))

residual = collect_diffs(s, t)
if residual:
    print(f"S_T_RESIDUAL_CONFIG_DIFF_COUNT={len(residual)}")
    for path, sv, tv in residual:
        print(
            "UNEXPECTED_RESOLVED_CONFIG_DIFF "
            + json.dumps(
                {"path": path, "S": sv, "T": tv},
                ensure_ascii=False,
                default=str,
            )
        )
    raise SystemExit(
        "S/T resolved configs differ after recursive normalization of known arm-specific values"
    )

print("S_T_RESOLVED_CONFIG_MATCH=PASS")
print("S_T_CONFIG_EQUIVALENCE_MODE=recursive_known_arm_value_normalization")
PY

atomic_state "RUNNING" "native_verl_oneupdate_arms"

postcheck_arm() {
    local arm="$1"
    local arm_run="$RUN/chemistry/$arm"
    local expected_sha
    if [[ "$arm" == "S" ]]; then
        expected_sha="83231dd2da9eb018df0cc92de85a48a30c60ee01fab15149929555626481dccf"
    else
        expected_sha="266a562b8a373446ede0dd7015dbcdbdbf3ae3f9dd3fc1e98de712b213f683e3"
    fi

    python - "$arm" "$arm_run" "$expected_sha" <<'PY'
import glob
import json
import math
import os
import re
import sys
from pathlib import Path

arm, arm_run_s, expected_sha = sys.argv[1:]
arm_run = Path(arm_run_s)

summary_path = arm_run / "replay_preupdate_summary.json"
audit_path = arm_run / "replay_transport_rows.jsonl"
train_log = arm_run / "train.log"
tb = arm_run / "tensorboard"

assert summary_path.is_file(), summary_path
assert audit_path.is_file(), audit_path
assert train_log.is_file(), train_log

summary = json.load(open(summary_path, encoding="utf-8"))
assert summary["status"] == "PASS_PREUPDATE_REPLAY_TRANSPORT"
assert summary["arm"] == arm
assert summary["rows"] == 64
assert summary["aggregate_response_ids_sha256"] == expected_sha
assert summary["student_rollout_calls"] == 0
assert summary["forbidden_rollout_client_accesses"] == []
assert summary["teacher_top_k"] == 32
assert summary["policy_gradient"] is False
assert summary["student_hint_present"] is False
assert summary["teacher_hint_present"] is True

rows = [json.loads(x) for x in audit_path.read_text(encoding="utf-8").splitlines() if x.strip()]
assert len(rows) == 64
assert all(r["arm"] == arm for r in rows)
assert all(r["student_hint_present"] is False for r in rows)
assert all(r["teacher_hint_present"] is True for r in rows)
assert all(r["prefix_token_count"] > 0 for r in rows)

text = train_log.read_text(encoding="utf-8", errors="replace")
assert "FRESH_ROLLOUT_FORBIDDEN" not in text
assert "PREFIX_REPLAY_PREUPDATE_PASS" in text
assert re.search(r"Total training steps:\s*1\b", text), "trainer did not resolve to exactly one step"

events = [Path(p) for p in glob.glob(str(tb / "**/events.out.tfevents.*"), recursive=True)]
events = [p for p in events if p.is_file() and p.stat().st_size > 0]
assert events, f"no nonempty TensorBoard event under {tb}"

# TensorBoard must contain actual first-step training metrics.
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

tags = set()
scalars = {}
for event_dir in sorted({p.parent for p in events}):
    ea = EventAccumulator(str(event_dir))
    ea.Reload()
    tags.update(ea.Tags().get("scalars", []))
    for tag in ea.Tags().get("scalars", []):
        vals = ea.Scalars(tag)
        if vals:
            scalars.setdefault(tag, []).extend(v.value for v in vals)

def one_of(candidates):
    for name in candidates:
        if name in scalars and scalars[name]:
            return name, scalars[name][-1]
    return None, None

grad_tag, grad = one_of(["actor/grad_norm", "grad_norm"])
loss_tag, loss = one_of(["actor/distillation/loss", "distillation/loss"])

assert grad_tag is not None, f"grad_norm scalar absent; tags={sorted(tags)}"
assert loss_tag is not None, f"distillation loss scalar absent; tags={sorted(tags)}"
assert math.isfinite(float(grad)) and float(grad) > 0.0, (grad_tag, grad)
assert math.isfinite(float(loss)), (loss_tag, loss)

out = {
    "status": "PASS_ONE_UPDATE_ENGINEERING_SMOKE",
    "arm": arm,
    "rows": 64,
    "aggregate_response_ids_sha256": expected_sha,
    "fresh_student_rollout_calls": 0,
    "teacher_top_k": 32,
    "policy_gradient": False,
    "grad_norm_tag": grad_tag,
    "grad_norm": float(grad),
    "distillation_loss_tag": loss_tag,
    "distillation_loss": float(loss),
    "tensorboard_events": [
        {"path": str(p), "bytes": p.stat().st_size}
        for p in events
    ],
    "formal_training_authorized": False,
}

tmp = arm_run / "postcheck.json.tmp"
final = arm_run / "postcheck.json"
tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, final)
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
}

run_arm() {
    local arm="$1"
    local arm_run="$RUN/chemistry/$arm"
    local project="mtpatcher-prefix-replay-smoke"
    local exp="chemistry-${arm}-native-verl-oneupdate-v1"
    local tb="$arm_run/tensorboard"

    [[ ! -f "$arm_run/DONE" ]] || die "arm $arm already has DONE marker; refuse duplicate update"

    # Belt and suspenders for this Verl version:
    # 1) explicit absolute TENSORBOARD_DIR contract;
    # 2) relative native Tracking path is symlinked into the same absolute arm dir.
    mkdir -p "$tb" "$arm_run/tensorboard_log/$project"
    rm -f "$arm_run/tensorboard_log/$project/$exp"
    ln -s "$tb" "$arm_run/tensorboard_log/$project/$exp"

    mapfile -t OVS < "$arm_run/overrides.txt"

    echo "=== START ARM $arm ==="
    echo "TENSORBOARD_DIR=$tb"
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "$arm_run/started_utc.txt"

    (
        cd "$arm_run"
        export TENSORBOARD_DIR="$tb"
        python -m verl.trainer.main_ppo_v0 "${OVS[@]}"
    ) > "$arm_run/train.log" 2>&1 &
    local pid=$!
    echo "$pid" > "$arm_run/main_ppo.pid"

    while kill -0 "$pid" 2>/dev/null; do
        python - "$arm_run" "$pid" <<'PY'
import json, os, sys, time
run, pid = sys.argv[1:]
p = os.path.join(run, "heartbeat.json")
tmp = p + ".tmp"
events = []
for root, _, files in os.walk(os.path.join(run, "tensorboard")):
    for f in files:
        if f.startswith("events.out.tfevents."):
            fp = os.path.join(root, f)
            events.append({"path": fp, "bytes": os.path.getsize(fp)})
with open(tmp, "w", encoding="utf-8") as h:
    json.dump(
        {
            "status": "RUNNING",
            "pid": int(pid),
            "updated_unix": time.time(),
            "tensorboard_events": events,
        },
        h,
        indent=2,
        sort_keys=True,
    )
    h.write("\n")
    h.flush()
    os.fsync(h.fileno())
os.replace(tmp, p)
PY
        sleep 30
    done

    set +e
    wait "$pid"
    local rc=$?
    set -e
    echo "$rc" > "$arm_run/main_ppo_status.txt"
    if [[ "$rc" -ne 0 ]]; then
        tail -n 200 "$arm_run/train.log" || true
        die "arm $arm main_ppo failed rc=$rc"
    fi

    postcheck_arm "$arm"
    touch "$arm_run/DONE"
    echo "ARM_${arm}_ENGINEERING_SMOKE=PASS"
}

run_arm S
run_arm T

python - "$RUN" <<'PY'
import json, os, sys
from pathlib import Path

run = Path(sys.argv[1])
s = json.load(open(run / "chemistry/S/postcheck.json", encoding="utf-8"))
t = json.load(open(run / "chemistry/T/postcheck.json", encoding="utf-8"))

assert s["status"] == t["status"] == "PASS_ONE_UPDATE_ENGINEERING_SMOKE"
assert s["fresh_student_rollout_calls"] == t["fresh_student_rollout_calls"] == 0
assert s["teacher_top_k"] == t["teacher_top_k"] == 32
assert s["policy_gradient"] is t["policy_gradient"] is False

summary = {
    "status": "PASS_ENGINEERING_SMOKE_FORMAL_TRAINING_NOT_AUTHORIZED",
    "scientific_class": "DIAGNOSTIC ONLY / OFFLINE PREFIX-SUPPORT REPLAY",
    "problem": "Verify native Verl frozen-prefix replay transport before any scientific S/T training.",
    "result": {
        "S": s,
        "T": t,
    },
    "interpretation": (
        "Both arms completed exactly one disposable native-Verl actor update from C0 "
        "using exact frozen response IDs, native Teacher top-k32 scoring, direct FKL, "
        "and zero fresh Student rollout calls. This is an engineering result only."
    ),
    "next_step": (
        "Do not launch formal S/T training automatically. Review the smoke, compute the "
        "preregistered practical-effect floor from existing per-example outputs, and "
        "complete the frozen Idiom external semantic/judge analogue."
    ),
    "formal_training_authorized": False,
}
tmp = run / "summary.json.tmp"
tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, run / "summary.json")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

atomic_state "PASS_ENGINEERING_SMOKE_FORMAL_TRAINING_NOT_AUTHORIZED" "complete"
npu-smi info > "$RUN/npu_after.txt" 2>&1 || true

echo
echo "============================================================"
echo "Problem -> Result -> Interpretation -> Next step"
echo "============================================================"
echo "Problem: prove frozen S/T replay reaches native Verl Teacher-topk32 FKL backward/update without fresh Student rollout."
echo "Result: S and T each passed one disposable update with exact response SHA, finite grad/loss, and nonempty TensorBoard."
echo "Interpretation: engineering replay contract PASS only; no scientific S-vs-T conclusion is drawn."
echo "Next step: review smoke + practical-effect floor + external Idiom judge. Formal S/T training remains blocked."
echo
echo "FINAL_RESULT=PASS_ENGINEERING_SMOKE_FORMAL_TRAINING_NOT_AUTHORIZED"
trap - EXIT
