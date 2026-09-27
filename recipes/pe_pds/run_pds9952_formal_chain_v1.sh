#!/usr/bin/env bash
# PDS9952 v1 orchestration scaffold.
# Safe phase-A modes are implemented now. Training modes deliberately remain gated
# until the prepare/finalize/stage-entry assets have passed human review.

set +e
set +u
set +o pipefail 2>/dev/null

source /workspace/mtpatcher/project_env.sh

ROOT=/workspace/mtpatcher
PROJECT="$ROOT/repo/MT-Patcher-Reproduction-Ascend"
A3="$ROOT/envs/verl-v0.9.0-a3"
PY="$A3/bin/python"
TEACHER_PY="$ROOT/envs/mtpatcher-npu-py311/bin/python"
FORMAL_VERL="$ROOT/repo/verl-v0.9.0-matched20k-v2-formal"

BUILDER="$PROJECT/scripts/pe_pds/build_pds9952_matched_assets.py"
PROTOCOL="$PROJECT/manifests/pe_pds/pds9952_matched_formal_protocol_v1.json"
SEQCFG="$PROJECT/configs/sft/pds9952_seqkd_formal_v1.yaml"
OPDCFG="$PROJECT/configs/opd/pds9952_opd_formal_v1.yaml"

DATA="$ROOT/data/verl_science_pe_pds/pds9952_matched_v1"
TEACHER_INPUT="$DATA/teacher_input_pds9952_v1.jsonl"
TEACHER_TARGETS="${PDS9952_TEACHER_TARGETS:-$DATA/teacher_targets_pds9952_qwen3_8b_v1.jsonl}"

STAGE_ACTOR="$ROOT/runs/science/pe_pds_v1/pe_opd_formal_v1/checkpoints/global_step_4422/actor"
STAGE_ROOT="$ROOT/runs/science/pe_pds_v1/pds9952_stage_entry_v1"
STAGE_MERGED="$STAGE_ROOT/pe_opd_step4422_merged_hf"

MODE="${1:---prepare-only}"

cd "$PROJECT" || exit 2
export PYTHONPATH="$PROJECT:$FORMAL_VERL:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

preflight() {
    echo "===== PDS9952 PREFLIGHT ====="
    echo "MODE=$MODE"
    echo "PROJECT=$PROJECT"
    echo "HEAD=$(git rev-parse HEAD 2>/dev/null)"
    echo "BRANCH=$(git branch --show-current 2>/dev/null)"
    echo "===== WORKTREE (informational; unrelated dirt is allowed) ====="
    git status --short
    echo

    for f in "$BUILDER" "$PROTOCOL" "$SEQCFG" "$OPDCFG"; do
        if [[ ! -f "$f" ]]; then
            echo "MISSING_REQUIRED_FILE=$f"
            return 10
        fi
    done

    "$PY" -m py_compile "$BUILDER" || return 11
    "$PY" - "$PROTOCOL" "$SEQCFG" "$OPDCFG" <<'PY'
import json, sys, yaml
from pathlib import Path
p=Path(sys.argv[1]); s=Path(sys.argv[2]); o=Path(sys.argv[3])
protocol=json.loads(p.read_text(encoding='utf-8'))
seq=yaml.safe_load(s.read_text(encoding='utf-8'))
opd=yaml.safe_load(o.read_text(encoding='utf-8'))
assert protocol['budget']['population']==9952
assert protocol['budget']['source_exposures']==59712
assert protocol['budget']['optimizer_steps']==3732
assert protocol['primary_endpoint_step']==3732
assert 3732 in protocol['validation_steps'] and 0 in protocol['validation_steps']
assert seq['name']=='pds9952_seqkd_formal_v1'
assert opd['name']=='pds9952_opd_formal_v1'
print('PDS9952_STATIC_SPEC_PARSE=PASS')
PY
    return $?
}

merge_teacher_shards() {
    local shard_dir="$DATA/teacher_shards16"
    mkdir -p "$shard_dir"
    "$PY" - "$TEACHER_INPUT" "$shard_dir" "$TEACHER_TARGETS" <<'PY'
import json, os, sys
from pathlib import Path
inp=Path(sys.argv[1]); shard_dir=Path(sys.argv[2]); out=Path(sys.argv[3])
expected={}
for ln,line in enumerate(inp.open(encoding='utf-8-sig'),1):
    if not line.strip(): continue
    x=json.loads(line); i=int(x['index'])
    if i in expected: raise RuntimeError(f'duplicate input index={i}')
    expected[i]=x['source']
assert set(expected)==set(range(9952)), (len(expected), min(expected), max(expected))
merged={}
for p in sorted(shard_dir.glob('device_*.jsonl')):
    for ln,line in enumerate(p.open(encoding='utf-8-sig'),1):
        if not line.strip(): continue
        x=json.loads(line); i=int(x['index'])
        if i not in expected: raise RuntimeError(f'{p}:{ln}: unexpected index={i}')
        if x.get('source') != expected[i]: raise RuntimeError(f'{p}:{ln}: source mismatch index={i}')
        t=x.get('target_translation')
        if not isinstance(t,str) or not t.strip(): raise RuntimeError(f'{p}:{ln}: empty target index={i}')
        if i in merged:
            # Exact duplicate is tolerable for resume-aware shards; conflicting duplicate is not.
            old=merged[i]
            if old.get('source')!=x.get('source') or old.get('target_translation')!=x.get('target_translation'):
                raise RuntimeError(f'conflicting duplicate target index={i}')
            continue
        merged[i]=x
missing=sorted(set(expected)-set(merged))
if missing:
    raise RuntimeError(f'missing Teacher targets count={len(missing)} first={missing[:30]}')
if len(merged)!=9952:
    raise RuntimeError(f'merged target rows={len(merged)} expected=9952')
tmp=out.with_suffix(out.suffix+'.tmp'); out.parent.mkdir(parents=True,exist_ok=True)
with tmp.open('w',encoding='utf-8') as f:
    for i in range(9952):
        f.write(json.dumps(merged[i],ensure_ascii=False,separators=(',',':'))+'\n')
os.replace(tmp,out)
print('PDS9952_TEACHER_MERGE=PASS')
print('TEACHER_TARGETS=',out)
PY
}

run_teacher_generation() {
    local gen="$PROJECT/scripts/mtpatcher_rq0/generate_seqkd50k_teacher_v1.py"
    local model="$ROOT/models/Qwen3-8B"
    local shard_dir="$DATA/teacher_shards16"
    local log_dir="$DATA/teacher_logs16"
    mkdir -p "$shard_dir" "$log_dir"

    if [[ ! -f "$TEACHER_INPUT" ]]; then
        echo "TEACHER_INPUT_MISSING=$TEACHER_INPUT"
        return 20
    fi
    if [[ ! -f "$gen" ]]; then
        echo "TEACHER_GENERATOR_MISSING=$gen"
        return 21
    fi

    echo "===== LAUNCH FROZEN EXISTING TEACHER GENERATOR ON 16 NPUs ====="
    declare -A PIDS
    for card in $(seq 0 15); do
        local out="$shard_dir/device_${card}.jsonl"
        local log="$log_dir/device_${card}.log"
        echo "TEACHER_START card=$card out=$out"
        "$TEACHER_PY" -u "$gen" \
                --input "$TEACHER_INPUT" \
                --output "$out" \
                --model "$model" \
                --device-id "$card" \
                --world-size 16 \
                --batch-size 16 \
                --max-new-tokens 512 \
            > "$log" 2>&1 &
        PIDS[$card]=$!
    done

    local failed=0
    for card in $(seq 0 15); do
        wait "${PIDS[$card]}"
        local rc=$?
        echo "TEACHER_DONE card=$card rc=$rc"
        if [[ "$rc" -ne 0 ]]; then
            failed=1
            tail -80 "$log_dir/device_${card}.log" 2>/dev/null || true
        fi
    done
    if [[ "$failed" -ne 0 ]]; then
        echo "PDS9952_TEACHER_GENERATION=FAIL"
        echo "Successful shard rows are preserved. Repair/rerun failed IDs before merge."
        return 22
    fi

    merge_teacher_shards || return $?
    "$PY" "$BUILDER" --stage finalize --teacher-targets "$TEACHER_TARGETS"
}

merge_stage_entry() {
    if [[ ! -d "$STAGE_ACTOR" ]]; then
        echo "STAGE_ACTOR_MISSING=$STAGE_ACTOR"
        return 30
    fi
    if [[ -e "$STAGE_MERGED" ]]; then
        echo "STAGE_MERGED_EXISTS=BLOCKED"
        echo "$STAGE_MERGED"
        echo "Refusing to overwrite a potentially frozen stage-entry artifact."
        return 31
    fi
    mkdir -p "$STAGE_ROOT"
    "$PY" -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$STAGE_ACTOR" \
        --target_dir "$STAGE_MERGED" || return $?
    echo "PDS9952_STAGE_ENTRY_MERGE=PASS"
    echo "STAGE_MERGED=$STAGE_MERGED"
    find "$STAGE_MERGED" -maxdepth 1 -type f -print0 2>/dev/null \
        | sort -z | xargs -0 -r sha256sum > "$STAGE_ROOT/merged_hf_files.sha256"
}

static_audit() {
    "$PY" - "$DATA" "$STAGE_MERGED" "$SEQCFG" "$OPDCFG" <<'PY'
import json, sys, yaml
from pathlib import Path
D=Path(sys.argv[1]); stage=Path(sys.argv[2]); seqcfg=Path(sys.argv[3]); opdcfg=Path(sys.argv[4])
required=[
 D/'pds9952_population_v1.jsonl',
 D/'pds9952_population_manifest_v1.json',
 D/'teacher_targets_pds9952_qwen3_8b_v1.jsonl',
 D/'source_order_manifest.jsonl',
 D/'pds9952_seqkd_matched59712.parquet',
 D/'pds9952_opd_matched59712.parquet',
 D/'asset_manifest.json',
]
for p in required:
    if not p.is_file(): raise RuntimeError(f'missing finalized asset: {p}')
if not stage.is_dir(): raise RuntimeError(f'missing common stage-entry HF: {stage}')
manifest=json.loads((D/'asset_manifest.json').read_text(encoding='utf-8'))
assert manifest['schedule']['total_rows']==59712
assert manifest['schedule']['total_optimizer_steps']==3732
assert manifest['matched_gate']['status']=='PASS'
seq=yaml.safe_load(seqcfg.read_text(encoding='utf-8'))
opd=yaml.safe_load(opdcfg.read_text(encoding='utf-8'))
needle=str(stage)
assert any(x==f'model.path={needle}' for x in seq['overrides'])
assert any(x==f'actor_rollout_ref.model.path={needle}' for x in opd['overrides'])
print('PDS9952_STATIC_ASSET_AUDIT=PASS')
PY
}

preflight || exit $?

case "$MODE" in
    --prepare-only)
        "$PY" "$BUILDER" --stage prepare
        exit $?
        ;;
    --teacher-and-finalize)
        run_teacher_generation
        exit $?
        ;;
    --finalize-assets)
        "$PY" "$BUILDER" --stage finalize --teacher-targets "$TEACHER_TARGETS"
        exit $?
        ;;
    --merge-stage-entry)
        merge_stage_entry
        exit $?
        ;;
    --static-audit)
        static_audit
        exit $?
        ;;
    --smoke|--formal)
        echo "PDS9952_TRAINING_MODE=GATED"
        echo "This phase-A scaffold intentionally refuses training until prepare/finalize/stage-entry audits are reviewed."
        echo "Run --static-audit and inspect the resulting hashes before adding/authorizing training launch commands."
        exit 64
        ;;
    *)
        echo "Usage: $0 [--prepare-only|--teacher-and-finalize|--finalize-assets|--merge-stage-entry|--static-audit|--smoke|--formal]"
        exit 2
        ;;
esac
