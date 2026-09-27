#!/usr/bin/env bash
set -euo pipefail

source /workspace/mtpatcher/project_env.sh

ROOT=/workspace/mtpatcher
PROJECT="$ROOT/repo/MT-Patcher-Reproduction-Ascend"
EAEC_VERL="$ROOT/repo/verl-v0.9.0-matched20k-v2-eaec-v1"
PY="$ROOT/envs/verl-v0.9.0-a3/bin/python"

WRAPPER="$PROJECT/scripts/targeted/main_ppo_sync_wrapper_v1.py"

DATA="$ROOT/data/verl_science_pe_pds/pe11792_matched_v1/localization_benchmark_pool128_v1.parquet"
PATCHBANK="$ROOT/data/verl_science_pe_pds/pe11792_matched_v1/eaec_patchbank_v1.jsonl"

RUN="$ROOT/runs/diagnostics/pe_pds_v1/localization_benchmark128_v1"

die() {
    echo "EAEC_PERSISTENCE_FAIL: $*" >&2
    exit 1
}

###############################################################################
# 0. Environment / immutable inputs
###############################################################################

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

export PYTHONPATH="$PROJECT:$EAEC_VERL:${PYTHONPATH:-}"
export PATH="$ROOT/envs/verl-v0.9.0-a3/bin:$PATH"
unset PYTHONHOME || true
hash -r

export EAEC_PATCHBANK_JSONL="$PATCHBANK"
export EAEC_CORE_RATIO=2.0
export EAEC_AUDIT_JSONL="$RUN/eaec_runtime_audit.jsonl"

[[ -s "$DATA" ]] || die "1024-source dataset absent"
[[ -s "$PATCHBANK" ]] || die "PatchBank absent"
[[ -f "$WRAPPER" ]] || die "wrapper absent"

EXPECTED_PATCHBANK_SHA="2a35b53dc397110b531260b27ed49cfc5400a83f4e70db95e06c3f56addc99f3"

GOT_PATCHBANK_SHA="$(sha256sum "$PATCHBANK" | awk '{print $1}')"

[[ "$GOT_PATCHBANK_SHA" == "$EXPECTED_PATCHBANK_SHA" ]] \
    || die "PatchBank SHA drift: $GOT_PATCHBANK_SHA"


###############################################################################
# 1. Check dataset contract
###############################################################################

"$PY" - "$DATA" <<'PY'
import sys
import pyarrow.parquet as pq

p = sys.argv[1]
t = pq.read_table(p)

assert t.num_rows == 128, t.num_rows
assert "source_id" in t.column_names

ids = [int(x) for x in t["source_id"].to_pylist()]

assert len(ids) == 128
assert len(set(ids)) == 128

print("PERSISTENCE_DATA_ROWS =", len(ids))
print("PERSISTENCE_UNIQUE_IDS =", len(set(ids)))
print("PERSISTENCE_DATA_GATE=PASS")
PY


###############################################################################
# 2. Ensure EAEC runtime is the imported Verl
###############################################################################

"$PY" - "$EAEC_VERL" <<'PY'
import sys
from pathlib import Path

expected = Path(sys.argv[1]).resolve()

import verl

actual = Path(verl.__file__).resolve()

print("VERL_FILE =", actual)

assert expected in actual.parents, (
    expected,
    actual,
)

print("EAEC_VERL_IMPORT_GATE=PASS")
PY


###############################################################################
# 3. No live trainer from a previous run
###############################################################################

"$PY" - <<'PY'
import subprocess

text = subprocess.check_output(
    ["ps", "-eo", "pid=,stat=,args="],
    text=True,
    errors="replace",
)

bad = []

for line in text.splitlines():
    x = line.strip().split(None, 2)

    if len(x) < 3:
        continue

    pid, stat, cmd = x

    # Zombies are already dead.
    if stat.startswith("Z"):
        continue

    if (
        "main_ppo_sync_wrapper_v1.py" in cmd
        or "verl.trainer.main_ppo" in cmd
    ):
        bad.append(line)

if bad:
    print("\n".join(bad))
    raise SystemExit("ACTIVE_TRAINER_GATE=FAIL")

print("ACTIVE_TRAINER_GATE=PASS")
PY


###############################################################################
# 4. Fresh output directory
###############################################################################

rm -rf "$RUN"
mkdir -p "$RUN"


###############################################################################
# 5. 1024-source persistence protocol
#
# Scientific contract:
# - same Qwen3-0.6B initialization
# - 1024 unique PE-selected sources
# - batch=16
# - temperature=1, top_p=1, top_k=-1
# - one rollout/source
# - lr=0 => Student policy stays frozen
# - distillation disabled => Qwen3-8B Teacher is not needed
# - 64 steps => exactly 1024 source rollouts
###############################################################################

O=(
"data.train_files=['$DATA']"
"data.val_files=['$DATA']"
"data.train_batch_size=16"
"data.max_prompt_length=1024"
"data.max_response_length=256"
"data.filter_overlong_prompts=False"
"data.truncation=error"
"data.shuffle=False"
"+data.apply_chat_template_kwargs.enable_thinking=False"

"algorithm.adv_estimator=grpo"
"algorithm.use_kl_in_reward=False"

"reward.custom_reward_function.path=$PROJECT/scripts/opd/constant_zero_reward.py"
"reward.custom_reward_function.name=compute_score"

"actor_rollout_ref.actor.use_kl_loss=False"

"actor_rollout_ref.model.path=$ROOT/models/Qwen3-0.6B"
"actor_rollout_ref.model.use_remove_padding=True"
"actor_rollout_ref.model.enable_gradient_checkpointing=False"

"actor_rollout_ref.actor.use_torch_compile=False"
"actor_rollout_ref.actor.optim.lr=0.0"
"actor_rollout_ref.actor.ppo_mini_batch_size=16"
"actor_rollout_ref.actor.ppo_epochs=1"
"actor_rollout_ref.actor.loss_agg_mode=token-mean"
"actor_rollout_ref.actor.use_dynamic_bsz=True"
"actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096"

"actor_rollout_ref.actor.fsdp_config.param_offload=False"
"actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
"actor_rollout_ref.actor.fsdp_config.use_torch_compile=False"

"actor_rollout_ref.rollout.name=vllm"
"actor_rollout_ref.rollout.tensor_model_parallel_size=2"
"actor_rollout_ref.rollout.n=1"
"actor_rollout_ref.rollout.temperature=1.0"
"actor_rollout_ref.rollout.top_p=1.0"
"actor_rollout_ref.rollout.top_k=-1"
"actor_rollout_ref.rollout.dtype=bfloat16"
"actor_rollout_ref.rollout.gpu_memory_utilization=0.35"
"actor_rollout_ref.rollout.max_model_len=1281"
"actor_rollout_ref.rollout.enforce_eager=True"

"++actor_rollout_ref.rollout.agent.agent_loop_manager_class=scripts.pe_pds_v1.localization_benchmark_export_manager_v1.LocalizationBenchmarkExportManagerV1"

"distillation.enabled=False"

"trainer.nnodes=1"
"trainer.n_gpus_per_node=8"
"trainer.project_name=mtpatcher-eaec-diagnostic"
"trainer.experiment_name=localization-benchmark128-v1"
"trainer.logger=['console']"
"trainer.val_before_train=False"
"trainer.test_freq=-1"
"trainer.total_epochs=1"
"trainer.total_training_steps=8"
"trainer.save_freq=-1"
"trainer.default_local_dir=$RUN/checkpoints"
"trainer.resume_mode=disable"
)


###############################################################################
# 6. Run
###############################################################################

echo "EAEC_PERSISTENCE1024_START"

set +e

(
    cd "$RUN"

    "$PY" "$WRAPPER" \
        "${O[@]}"
) > "$RUN/train.log" 2>&1

RC=$?

set -e

echo "EAEC_PERSISTENCE1024_RC=$RC"

if [[ "$RC" -ne 0 ]]; then
    echo
    echo "===== TRAIN LOG TAIL ====="
    tail -n 260 "$RUN/train.log" || true
    die "persistence run failed rc=$RC"
fi


###############################################################################
# 7. Aggregate exact-active persistence
###############################################################################

"$PY" - "$RUN/eaec_runtime_audit.jsonl" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])

assert p.is_file(), p

rows = [
    json.loads(x)
    for x in p.read_text(
        encoding="utf-8"
    ).splitlines()
    if x.strip()
]

assert len(rows) == 8, len(rows)

keys = [
    "samples",
    "samples_with_patches",
    "candidate_patches",
    "active_patches",
    "resolved_patches",
    "ambiguous_patches",
    "unlocated_patches",
    "core_tokens",
    "valid_response_tokens",
    "tokenization_fallback",
    "projection_empty",
    "missing_patchbank_sources",
]

tot = {
    k: sum(
        int(row.get(k, 0))
        for row in rows
    )
    for k in keys
}

classified = (
    tot["active_patches"]
    + tot["resolved_patches"]
    + tot["ambiguous_patches"]
    + tot["unlocated_patches"]
)

unclassified = (
    tot["candidate_patches"]
    - classified
)

assert tot["samples"] == 128, tot["samples"]
assert tot["missing_patchbank_sources"] == 0

print("BATCHES =", len(rows))

for k in keys:
    print(f"{k} =", tot[k])

print()

print(
    "ACTIVE_PER_CANDIDATE =",
    tot["active_patches"]
    / max(1, tot["candidate_patches"]),
)

print(
    "ACTIVE_PER_CLASSIFIED =",
    tot["active_patches"]
    / max(1, classified),
)

print(
    "RESOLVED_PER_CANDIDATE =",
    tot["resolved_patches"]
    / max(1, tot["candidate_patches"]),
)

print(
    "UNLOCATED_PER_CANDIDATE =",
    tot["unlocated_patches"]
    / max(1, tot["candidate_patches"]),
)

print(
    "CORE_TOKEN_RATIO =",
    tot["core_tokens"]
    / max(1, tot["valid_response_tokens"]),
)

print(
    "TOKENIZATION_FALLBACK_SAMPLE_RATIO =",
    tot["tokenization_fallback"]
    / max(1, tot["samples"]),
)

print(
    "PROJECTION_EMPTY_RATIO =",
    tot["projection_empty"]
    / max(1, tot["samples"]),
)

print(
    "UNCLASSIFIED_PATCHES =",
    unclassified,
)

print()
print("EAEC_PERSISTENCE1024_SUMMARY=PASS")
PY


###############################################################################
# 8. Compact runtime evidence
###############################################################################

echo
echo "===== FIRST EAEC BATCHES ====="

grep 'EAEC_RUNTIME_BATCH_PASS' \
    "$RUN/train.log" |
head -10 || true

echo
echo "===== LAST EAEC BATCHES ====="

grep 'EAEC_RUNTIME_BATCH_PASS' \
    "$RUN/train.log" |
tail -10 || true

echo
echo "EAEC_PERSISTENCE1024=PASS"
