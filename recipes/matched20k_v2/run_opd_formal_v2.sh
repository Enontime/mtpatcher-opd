#!/usr/bin/env bash

set +e
set +u
set +o pipefail 2>/dev/null

ROOT=/workspace/mtpatcher

PROJECT="$ROOT/repo/MT-Patcher-Reproduction-Ascend"
FORMAL_VERL="$ROOT/repo/verl-v0.9.0-matched20k-v2-formal"

VERL_PY="$ROOT/envs/verl-v0.9.0-a3/bin/python"

CFG="$PROJECT/configs/opd/matched20k_v2_opd_formal_v2.yaml"

PROTOCOL="$PROJECT/manifests/matched20k_v2/formal_protocol_v1.json"

OPD_CONTRACT="$PROJECT/manifests/matched20k_v2/opd_formal_contract_v1.json"

RUN="$ROOT/runs/science/matched20k_v2/opd_formal_v2"

if [[ -e "$RUN" ]]; then
    echo "OPD_FORMAL_RUN_PATH_EXISTS=BLOCKED"
else
    mkdir -p \
        "$RUN/provenance" \
        "$RUN/tensorboard"

    cp \
        "$CFG" \
        "$RUN/provenance/formal_config.yaml"

    cp \
        "$PROTOCOL" \
        "$RUN/provenance/formal_protocol_v1.json"

    cp \
        "$OPD_CONTRACT" \
        "$RUN/provenance/opd_formal_contract_v1.json"

    cp \
        "$ROOT/smoke/matched20k_v2/formal_preflight/opd_v2/formal_resolved_config.yaml" \
        "$RUN/provenance/resolved_config.yaml"

    export PYTHONPATH="$PROJECT:$FORMAL_VERL:${PYTHONPATH:-}"

    # Formal training uses native Verl PPOTrainerSync.
    unset VERL_CUSTOM_TRAINER_MODULE
    unset MATCHED20K_VALIDATION_JSONL

    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

    export TOKENIZERS_PARALLELISM=false
    export PYTORCH_ALLOC_CONF=expandable_segments:True

    export TENSORBOARD_DIR="$RUN/tensorboard"

    "$VERL_PY" - \
        "$CFG" \
        "$RUN/provenance/runtime_manifest.json" \
        "$PROJECT" \
        "$FORMAL_VERL" <<'PY'
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import yaml

cfg_path = Path(sys.argv[1])
out = Path(sys.argv[2])
project = Path(sys.argv[3])
verl = Path(sys.argv[4])


def head(path):
    return subprocess.check_output(
        [
            "git",
            "-C",
            str(path),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


spec = yaml.safe_load(
    cfg_path.read_text(
        encoding="utf-8"
    )
)

payload = {
    "schema_version": 1,
    "arm": "opd",
    "formal": True,
    "project_commit": head(project),
    "verl_commit": head(verl),
    "formal_config_sha256": sha(cfg_path),
    "max_optimizer_steps": 7500,
    "global_batch_size": 16,
    "source_exposures": 120000,
    "dataset_traversals": 1,
    "original_source_passes": 6,
    "checkpoint_interval": 100,
    "trainer": "native Verl PPOTrainerSync",
    "objective": "forward_kl_topk",
    "teacher_topk": 32,
    "runtime_compatibility_fix": {
        "override": "actor_rollout_ref.actor.fsdp_config.use_torch_compile=False",
        "reason": "Ascend torch.compile entropy kernel triggers deterministic 507035 in formal OPD pipeline",
        "evidence_pass": "/workspace/mtpatcher/smoke/matched20k_v2/opd_entropy_compile_off_step1_v5",
        "evidence_fail_positive_control": "/workspace/mtpatcher/smoke/matched20k_v2/opd_entropy_compile_on_step1_v6"
    },
    "teacher":
        "/workspace/mtpatcher/models/Qwen3-8B",
    "student":
        "/workspace/mtpatcher/models/Qwen3-0.6B",
}

out.write_text(
    json.dumps(
        payload,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

print(
    json.dumps(
        payload,
        indent=2,
    )
)

print("OPD_FORMAL_RUNTIME_MANIFEST=PASS")
PY

    mapfile -t OVERRIDES < <(
        "$VERL_PY" - "$CFG" <<'PY'
import sys
import yaml

spec = yaml.safe_load(
    open(
        sys.argv[1],
        encoding="utf-8",
    )
)

for item in spec["overrides"]:
    print(item)
PY
    )

    echo
    echo "===== OPD FORMAL START ====="
    echo "run=$RUN"
    echo "override_count=${#OVERRIDES[@]}"

    "$VERL_PY" \
        -m verl.trainer.main_ppo \
        "${OVERRIDES[@]}" \
        > "$RUN/train.log" 2>&1

    RC=$?

    echo "$RC" \
        > "$RUN/final_status.txt"

    echo "OPD_FORMAL_RC=$RC"
fi
