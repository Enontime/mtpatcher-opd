#!/usr/bin/env bash

set +e
set +u
set +o pipefail 2>/dev/null

ROOT=/workspace/mtpatcher

PROJECT=$ROOT/repo/MT-Patcher-Reproduction-Ascend
VERL=$ROOT/repo/verl-v0.9.0
ENV=$ROOT/envs/verl-v0.9.0-a3

ORIG_RUN=$ROOT/runs/opd/canonical_fkl_topk_broad20k_qwen3_06b_8b_20260903_034058

SOURCE_CKPT=$ORIG_RUN/checkpoints/global_step_1250
ORIG_OVERRIDES=$ORIG_RUN/overrides.txt

RUN=$ROOT/runs/opd/opd_recovery_20260905_v2

RECOVERY_OVERRIDES=$RUN/recovery_overrides.txt

CONFIG=$PROJECT/configs/opd/canonical_fkl_topk_broad20k_qwen3_06b_8b.yaml
RECIPE=$PROJECT/recipes/opd/canonical_fkl_topk_broad20k_qwen3_06b_8b.sh

mkdir -p "$RUN/checkpoints"

echo "$RUN" > /tmp/mtpatcher_opd_recovery_run_dir

echo "============================================================"
echo "CANONICAL OPD RECOVERY — 20260904_v1"
echo "============================================================"

echo
echo "source_run=$ORIG_RUN"
echo "source_checkpoint=$SOURCE_CKPT"
echo "recovery_run=$RUN"

READY=1


# ============================================================
# 1. Exact canonical environment
# ============================================================

echo
echo "=== 1. ENVIRONMENT ==="

if [[ -f "$ENV/bin/activate" ]]; then
  source "$ENV/bin/activate"
  ENV_STATUS=$?
else
  ENV_STATUS=99
fi

echo "env_status=$ENV_STATUS"

if [[ "$ENV_STATUS" -ne 0 ]]; then
  READY=0
fi

cd "$VERL"
CD_STATUS=$?

echo "verl_cd_status=$CD_STATUS"

if [[ "$CD_STATUS" -ne 0 ]]; then
  READY=0
fi

echo "python=$(command -v python)"
python --version 2>&1


# ============================================================
# 2. Recovery source integrity / availability
# ============================================================

echo
echo "=== 2. SOURCE CHECKPOINT AVAILABILITY ==="

for f in \
  "$ORIG_OVERRIDES" \
  "$SOURCE_CKPT/data.pt" \
  "$SOURCE_CKPT/actor/fsdp_config.json"
do
  if [[ -f "$f" ]]; then
    echo "PASS $f"
  else
    echo "MISSING $f"
    READY=0
  fi
done

for rank in $(seq 0 7)
do
  for kind in model optim extra_state
  do
    f="$SOURCE_CKPT/actor/${kind}_world_size_8_rank_${rank}.pt"

    if [[ -f "$f" ]]; then
      :
    else
      echo "MISSING $f"
      READY=0
    fi
  done
done

if [[ "$READY" -eq 1 ]]; then
  echo "SOURCE_CHECKPOINT_8RANK_COMPLETE_PASS"
fi


# ============================================================
# 3. Exact known-infrastructure preflight
#
# Attempt1 died because this exact npu-smi board query failed.
# Do not enter another 20h-scale run if it is already broken.
# ============================================================

echo
echo "=== 3. NPU PREFLIGHT ==="

npu-smi info -t board -i 1 \
  > "$RUN/npu_board1_preflight.txt" 2>&1

NPU_STATUS=$?

echo "npu_board1_status=$NPU_STATUS"

if [[ "$NPU_STATUS" -ne 0 ]]; then
  echo "NPU_PREFLIGHT_FAIL"
  READY=0
else
  echo "NPU_PREFLIGHT_PASS"
fi


# ============================================================
# 4. Build recovery overrides from the ACTUAL Attempt1
#    resolved override set.
#
# Allowed differences:
#   1. trainer.default_local_dir
#   2. trainer.resume_mode
#   3. trainer.resume_from_path (new)
#
# Everything else must remain byte-for-byte identical.
# ============================================================

echo
echo "=== 4. BUILD RECOVERY OVERRIDES ==="

if [[ "$READY" -eq 1 ]]; then

python - \
  "$ORIG_OVERRIDES" \
  "$RECOVERY_OVERRIDES" \
  "$RUN" \
  "$SOURCE_CKPT" \
<<'PY'
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
run = sys.argv[3]
ckpt = sys.argv[4]

lines = src.read_text(
    encoding="utf-8"
).splitlines()

n_local = sum(
    x.startswith("trainer.default_local_dir=")
    for x in lines
)

n_mode = sum(
    x == "trainer.resume_mode=disable"
    for x in lines
)

n_resume_path = sum(
    x.startswith("trainer.resume_from_path=")
    for x in lines
)

assert n_local == 1, (
    f"expected one default_local_dir, got {n_local}"
)

assert n_mode == 1, (
    f"expected exactly one resume_mode=disable, got {n_mode}"
)

assert n_resume_path == 0, (
    "Attempt1 overrides unexpectedly already contain "
    "resume_from_path"
)

out = []

for x in lines:
    if x.startswith("trainer.default_local_dir="):
        out.append(
            f"trainer.default_local_dir={run}/checkpoints"
        )
    elif x == "trainer.resume_mode=disable":
        out.append(
            "trainer.resume_mode=resume_path"
        )
    else:
        out.append(x)

out.append(
    f"trainer.resume_from_path={ckpt}"
)

dst.write_text(
    "\n".join(out) + "\n",
    encoding="utf-8",
)

print("RECOVERY_OVERRIDE_BUILD_PASS")
print("original_lines =", len(lines))
print("recovery_lines =", len(out))
PY

  BUILD_STATUS=$?

else
  BUILD_STATUS=99
fi

echo "override_build_status=$BUILD_STATUS"

if [[ "$BUILD_STATUS" -ne 0 ]]; then
  READY=0
fi


# ============================================================
# 5. Prove science overrides did not drift.
# ============================================================

echo
echo "=== 5. OVERRIDE FIDELITY CONTRACT ==="

if [[ "$READY" -eq 1 ]]; then

python - \
  "$ORIG_OVERRIDES" \
  "$RECOVERY_OVERRIDES" \
  "$RUN" \
  "$SOURCE_CKPT" \
<<'PY'
import sys
from pathlib import Path

old_path = Path(sys.argv[1])
new_path = Path(sys.argv[2])
run = sys.argv[3]
ckpt = sys.argv[4]

old = old_path.read_text(
    encoding="utf-8"
).splitlines()

new = new_path.read_text(
    encoding="utf-8"
).splitlines()

runtime_prefixes = (
    "trainer.default_local_dir=",
    "trainer.resume_mode=",
    "trainer.resume_from_path=",
)

old_science = [
    x for x in old
    if not x.startswith(runtime_prefixes)
]

new_science = [
    x for x in new
    if not x.startswith(runtime_prefixes)
]

assert old_science == new_science, (
    "science override drift detected"
)

expected_runtime = {
    f"trainer.default_local_dir={run}/checkpoints",
    "trainer.resume_mode=resume_path",
    f"trainer.resume_from_path={ckpt}",
}

actual_runtime = {
    x for x in new
    if x.startswith(runtime_prefixes)
}

assert actual_runtime == expected_runtime, (
    "unexpected recovery runtime override set:\n"
    f"{actual_runtime}"
)

assert len(new) == len(old) + 1

# Canonical science sentinels.
required = [
    "data.train_batch_size=16",
    "data.shuffle=False",
    "actor_rollout_ref.actor.optim.lr=1e-06",
    "actor_rollout_ref.actor.ppo_epochs=1",
    "actor_rollout_ref.actor.loss_agg_mode=token-mean",
    "actor_rollout_ref.rollout.n=1",
    "actor_rollout_ref.rollout.temperature=1.0",
    "actor_rollout_ref.rollout.top_p=1.0",
    "actor_rollout_ref.rollout.top_k=-1",
    "distillation.enabled=True",
    "distillation.distillation_loss.loss_mode=forward_kl_topk",
    "distillation.distillation_loss.topk=32",
    "distillation.distillation_loss.use_task_rewards=False",
    "distillation.distillation_loss.use_policy_gradient=False",
    "trainer.total_epochs=3",
    "trainer.save_freq=1250",
    "trainer.n_gpus_per_node=8",
    "distillation.n_gpus_per_node=4",
]

missing = [
    x for x in required
    if x not in new
]

assert not missing, (
    f"missing canonical sentinels: {missing}"
)

print("RECOVERY_OVERRIDE_FIDELITY_PASS")
print("science_override_sequence_exact = True")
print("runtime_delta_count = 3")
PY

  FIDELITY_STATUS=$?

else
  FIDELITY_STATUS=99
fi

echo "override_fidelity_status=$FIDELITY_STATUS"

if [[ "$FIDELITY_STATUS" -ne 0 ]]; then
  READY=0
fi

diff -u \
  "$ORIG_OVERRIDES" \
  "$RECOVERY_OVERRIDES" \
  > "$RUN/recovery_override.diff" 2>&1


# ============================================================
# 6. Provenance snapshots
# ============================================================

echo
echo "=== 6. PROVENANCE ==="

cp "$CONFIG" \
  "$RUN/canonical_config_snapshot.yaml" \
  2>/dev/null

cp "$RECIPE" \
  "$RUN/canonical_recipe_snapshot.sh" \
  2>/dev/null

cp "$ORIG_OVERRIDES" \
  "$RUN/attempt1_overrides_snapshot.txt" \
  2>/dev/null

sha256sum \
  "$CONFIG" \
  "$RECIPE" \
  "$ORIG_OVERRIDES" \
  > "$RUN/recovery_provenance.sha256" \
  2>/dev/null

cat > "$RUN/recovery_manifest.txt" <<EOF
recovery_id=opd_recovery_20260905_v2
source_run=$ORIG_RUN
source_checkpoint=$SOURCE_CKPT
source_global_step=1250
discarded_attempt1_suffix=1251-1549
target_pass2_step=2500
target_pass3_step=3750
primary_endpoint=3750
resume_mode=resume_path
recovery_checkpoint_dir=$RUN/checkpoints
EOF


# ============================================================
# 7. Hydra compose before entering Ray.
# ============================================================

echo
echo "=== 7. RECOVERY HYDRA COMPOSE ==="

if [[ "$READY" -eq 1 ]]; then

  mapfile -t OVERRIDES \
    < "$RECOVERY_OVERRIDES"

  python -m verl.trainer.main_ppo \
    --cfg job \
    "${OVERRIDES[@]}" \
    > "$RUN/resolved_config.yaml"

  COMPOSE_STATUS=$?

else
  COMPOSE_STATUS=99
fi

echo "$COMPOSE_STATUS" \
  > "$RUN/compose_status.txt"

echo "compose_status=$COMPOSE_STATUS"

if [[ "$COMPOSE_STATUS" -ne 0 ]]; then
  READY=0
fi


# ============================================================
# 8. Recovery-specific resolved config contract.
# ============================================================

echo
echo "=== 8. RESOLVED RECOVERY CONTRACT ==="

if [[ "$READY" -eq 1 ]]; then

python - \
  "$RUN/resolved_config.yaml" \
  "$RUN" \
  "$SOURCE_CKPT" \
<<'PY'
import sys

from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])

run = sys.argv[2]
ckpt = sys.argv[3]

DATA = (
    "/workspace/mtpatcher/data/"
    "verl_science_broad20k/"
    "opd_broad20k.parquet"
)

assert list(cfg.data.train_files) == [DATA]
assert cfg.data.train_batch_size == 16
assert cfg.data.shuffle is False

assert cfg.trainer.nnodes == 1
assert cfg.trainer.n_gpus_per_node == 8
assert cfg.trainer.total_epochs == 3
assert cfg.trainer.save_freq == 1250

assert (
    str(cfg.trainer.default_local_dir)
    == f"{run}/checkpoints"
)

assert (
    str(cfg.trainer.resume_mode)
    == "resume_path"
)

assert (
    str(cfg.trainer.resume_from_path)
    == ckpt
)

a = cfg.actor_rollout_ref.actor
r = cfg.actor_rollout_ref.rollout
d = cfg.distillation

assert float(a.optim.lr) == 1e-6
assert a.ppo_mini_batch_size == 16
assert a.ppo_epochs == 1
assert a.loss_agg_mode == "token-mean"

assert r.n == 1
assert float(r.temperature) == 1.0
assert float(r.top_p) == 1.0
assert int(r.top_k) == -1
assert r.tensor_model_parallel_size == 2

assert d.enabled is True
assert (
    d.distillation_loss.loss_mode
    == "forward_kl_topk"
)
assert d.distillation_loss.topk == 32
assert (
    d.distillation_loss.use_task_rewards
    is False
)
assert (
    d.distillation_loss.use_policy_gradient
    is False
)
assert d.n_gpus_per_node == 4
assert d.nnodes == 1

print("RECOVERY_RESOLVED_CONTRACT_PASS")
print("resume_global_step = 1250")
print("target_global_step = 3750")
print("actor_pool = 8")
print("teacher_pool = 4")
print("teacher_topk = 32")
print("sampling = 1.0 / 1.0 / -1")
PY

  CONTRACT_STATUS=$?

else
  CONTRACT_STATUS=99
fi

echo "$CONTRACT_STATUS" \
  > "$RUN/resolved_contract_status.txt"

echo "resolved_contract_status=$CONTRACT_STATUS"

if [[ "$CONTRACT_STATUS" -ne 0 ]]; then
  READY=0
fi


# ============================================================
# 9. Formal Verl-native recovery.
# ============================================================

echo
echo "=== 9. FORMAL RECOVERY ==="

if [[ "$READY" -eq 1 ]]; then

  date -Is \
    > "$RUN/formal_recovery_started_at.txt"

  echo "OPD_RECOVERY_PRECHECK_PASS"
  echo
  echo "scientific trajectory:"
  echo "0 -> 1250"
  echo "+ faithful Verl-native restart"
  echo "+ 1251 -> 3750"
  echo
  echo "Attempt1 steps 1251-1549 are discarded."
  echo
  echo "STARTING_VERL_NATIVE_RECOVERY"

  python -m verl.trainer.main_ppo \
    "${OVERRIDES[@]}" \
    2>&1 |
    tee "$RUN/train.log"

  MAIN_STATUS=${PIPESTATUS[0]}

  echo "$MAIN_STATUS" \
    > "$RUN/main_ppo_status.txt"

  echo "$MAIN_STATUS" \
    > "$RUN/final_status.txt"

  date -Is \
    > "$RUN/formal_recovery_finished_at.txt"

  if [[ "$MAIN_STATUS" -eq 0 ]]; then
    echo "OPD_RECOVERY_MAIN_PPO_PASS"
  else
    echo \
      "OPD_RECOVERY_MAIN_PPO_FAIL status=$MAIN_STATUS"
  fi

else

  echo "99" \
    > "$RUN/final_status.txt"

  echo "OPD_RECOVERY_PRECHECK_FAIL"
  echo "FORMAL_RECOVERY_NOT_STARTED"

fi


echo
echo "============================================================"
echo "OPD RECOVERY WRAPPER FINISHED"
echo "============================================================"

echo "run=$RUN"

if [[ -f "$RUN/final_status.txt" ]]; then
  echo -n "recorded_status="
  cat "$RUN/final_status.txt"
fi
