#!/usr/bin/env bash

set +e
set +u
set +o pipefail 2>/dev/null

ROOT=/workspace/mtpatcher

PROJECT="$ROOT/repo/MT-Patcher-Reproduction-Ascend"
FORMAL_VERL="$ROOT/repo/verl-v0.9.0-matched20k-v2-formal"

VERL_PY="$ROOT/envs/verl-v0.9.0-a3/bin/python"
TORCHRUN="$ROOT/envs/verl-v0.9.0-a3/bin/torchrun"

CFG="$PROJECT/configs/sft/matched20k_v2_seqkd_formal_v1.yaml"
PROTOCOL="$PROJECT/manifests/matched20k_v2/formal_protocol_v1.json"
MODEL_PROV="$PROJECT/manifests/matched20k_v2/model_provenance_v1.json"

RUN="$ROOT/runs/science/matched20k_v2/seqkd_formal_v1"

if [[ -e "$RUN" ]]; then
    echo "SEQKD_FORMAL_RUN_PATH_EXISTS=BLOCKED"
    echo "Existing formal run is preserved:"
    echo "$RUN"
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
        "$MODEL_PROV" \
        "$RUN/provenance/model_provenance_v1.json"

    export PYTHONPATH="$PROJECT:$FORMAL_VERL:${PYTHONPATH:-}"

    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    export TOKENIZERS_PARALLELISM=false
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    export TENSORBOARD_DIR="$RUN/tensorboard"

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

assert spec["nproc_per_node"] == 16

for item in spec["overrides"]:
    print(item)
PY
    )

    echo "===== SEQKD FORMAL START ====="
    echo "run=$RUN"
    echo "override_count=${#OVERRIDES[@]}"

    "$TORCHRUN" \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=16 \
        -m verl.trainer.sft_trainer \
        "${OVERRIDES[@]}" \
        > "$RUN/train.log" 2>&1

    RC=$?

    echo "$RC" \
        > "$RUN/final_status.txt"

    echo "SEQKD_FORMAL_RC=$RC"
fi
