#!/usr/bin/env bash
set -u -o pipefail

main() {
    local MTP_ROOT=/workspace/mtpatcher
    local REPO="$MTP_ROOT/repo/MT-Patcher-Reproduction-Ascend"
    local VERL="$MTP_ROOT/repo/verl-v0.9.0"
    local VERL_ENV_ROOT="$MTP_ROOT/envs/verl-v0.9.0-a3"
    local RUN="${RUN:-}"
    local CONFIG="$REPO/configs/targeted/offline_prefix_support_chemistry_seed1_native_verl_v1.yaml"
    local MANAGER="$REPO/scripts/targeted/offline_prefix_support_verl_replay_manager_formal_v1.py"
    local WRAPPER="$REPO/scripts/targeted/main_ppo_sync_wrapper_v1.py"
    local ZERO_REWARD="$REPO/scripts/opd/constant_zero_reward.py"
    local STUDENT="$MTP_ROOT/models/Qwen3-0.6B"
    local TEACHER="$MTP_ROOT/models/Qwen3-8B"
    local STATE="$RUN/state.json"

    if [[ -f "$MTP_ROOT/project_env.sh" ]]; then
        # shellcheck disable=SC1090
        source "$MTP_ROOT/project_env.sh"
    fi

    # project_env.sh may define generic ROOT/ENV_ROOT; keep collision-resistant paths.
    MTP_ROOT=/workspace/mtpatcher
    REPO="$MTP_ROOT/repo/MT-Patcher-Reproduction-Ascend"
    VERL="$MTP_ROOT/repo/verl-v0.9.0"
    VERL_ENV_ROOT="$MTP_ROOT/envs/verl-v0.9.0-a3"
    CONFIG="$REPO/configs/targeted/offline_prefix_support_chemistry_seed1_native_verl_v1.yaml"
    MANAGER="$REPO/scripts/targeted/offline_prefix_support_verl_replay_manager_formal_v1.py"
    WRAPPER="$REPO/scripts/targeted/main_ppo_sync_wrapper_v1.py"
    ZERO_REWARD="$REPO/scripts/opd/constant_zero_reward.py"
    STUDENT="$MTP_ROOT/models/Qwen3-0.6B"
    TEACHER="$MTP_ROOT/models/Qwen3-8B"
    STATE="$RUN/state.json"

    export PYTHONUNBUFFERED=1
    export TOKENIZERS_PARALLELISM=false
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    export PYTHONPATH="$REPO/scripts/targeted:$REPO:$VERL:${PYTHONPATH:-}"

    fail() {
        echo "FAIL: $*" >&2
        if [[ -n "$RUN" && -d "$RUN" ]]; then
            python - "$STATE" "$*" <<'PY_FAIL' 2>/dev/null
import json, os, sys, time
p, reason = sys.argv[1:]
obj = {
    'status': 'FAIL_FORMAL_CHEMISTRY_TRAINING',
    'phase': 'formal_training',
    'reason': reason,
    'updated_unix': time.time(),
    'scientific_class': 'DIAGNOSTIC ONLY / OFFLINE PREFIX-SUPPORT REPLAY',
}
tmp = p + '.tmp'
with open(tmp, 'w', encoding='utf-8') as f:
    json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
    f.write('\n')
    f.flush(); os.fsync(f.fileno())
os.replace(tmp, p)
PY_FAIL
        fi
        return 1
    }

    atomic_state() {
        local status="$1"
        local phase="$2"
        python - "$STATE" "$status" "$phase" <<'PY_STATE'
import json, os, sys, time
p, status, phase = sys.argv[1:]
obj = {
    'status': status,
    'phase': phase,
    'chemistry_seed1_authorized': True,
    'idiom_training_authorized': False,
    'scientific_class': 'DIAGNOSTIC ONLY / OFFLINE PREFIX-SUPPORT REPLAY',
    'updated_unix': time.time(),
}
tmp = p + '.tmp'
with open(tmp, 'w', encoding='utf-8') as f:
    json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
    f.write('\n'); f.flush(); os.fsync(f.fileno())
os.replace(tmp, p)
PY_STATE
    }

    if [[ -z "$RUN" ]]; then
        echo "FAIL: RUN must be provided" >&2
        return 1
    fi
    mkdir -p "$RUN" || { fail "cannot create run dir"; return 1; }

    echo "============================================================"
    echo "0. AGENT3 FORMAL CHEMISTRY CONTRACT"
    echo "============================================================"
    echo "Question: with rows/objective/dose fixed, does Teacher-supported frozen replay outperform Student-supported frozen replay on Chemistry targeted transfer?"
    echo "Competing explanations: trajectory support matters; soft-KL/update efficiency dominates; Teacher targets are intrinsically stronger on T prefixes."
    echo "Falsifiable prediction: both arms consume exactly 1000 frozen rows for 3 passes with zero fresh Student rollout, 75 native Verl updates, checkpoints at 25/50/75, and distinct TensorBoard traces."
    echo "Decision after result: training completion alone makes no effect claim; strict Chemistry evaluation + paired bootstrap + prereg recovery-ratio screen are required."

    [[ -x "$VERL_ENV_ROOT/bin/python" ]] || { fail "missing Verl Python $VERL_ENV_ROOT/bin/python"; return 1; }
    export PATH="$VERL_ENV_ROOT/bin:$PATH"
    unset PYTHONHOME || true
    hash -r

    python - "$VERL_ENV_ROOT" <<'PY_ENV'
import importlib, sys
from pathlib import Path
env = Path(sys.argv[1]).resolve()
assert Path(sys.executable).resolve() == (env / 'bin/python').resolve(), (sys.executable, env)
assert Path(sys.prefix).resolve() == env, (sys.prefix, env)
for name in ('tensordict','torch','torch_npu','ray','pyarrow','omegaconf','tensorboard','transformers'):
    importlib.import_module(name)
print(f'VERL_ENV_GATE=PASS python={sys.executable} prefix={sys.prefix}')
PY_ENV
    [[ $? -eq 0 ]] || { fail "Verl environment gate failed"; return 1; }

    for f in "$CONFIG" "$MANAGER" "$WRAPPER" "$ZERO_REWARD" "$RUN/input_manifest.json" "$RUN/implementation_commit.txt"; do
        [[ -s "$f" ]] || { fail "missing required file $f"; return 1; }
    done

    local expected_head current_head
    expected_head="$(cat "$RUN/implementation_commit.txt")"
    current_head="$(git -C "$REPO" rev-parse HEAD 2>/dev/null)" || { fail "cannot read repo HEAD"; return 1; }
    [[ "$current_head" == "$expected_head" ]] || { fail "repo HEAD drift current=$current_head expected=$expected_head"; return 1; }
    [[ -z "$(git -C "$REPO" status --porcelain)" ]] || { git -C "$REPO" status --short; fail "worktree must be clean"; return 1; }

    python - "$RUN/input_manifest.json" "$CONFIG" <<'PY_STATIC'
import json, sys, yaml
m = json.load(open(sys.argv[1], encoding='utf-8'))
c = yaml.safe_load(open(sys.argv[2], encoding='utf-8'))
assert m['status'] == 'PASS_FORMAL_CHEMISTRY_INPUTS_FROZEN'
assert m['chemistry_seed1_authorized'] is True
assert m['rows'] == 1000
assert m['batch_size'] == 40
assert m['batches_per_pass'] == 25
assert m['passes'] == 3
assert m['total_optimizer_updates_per_arm'] == 75
assert set(m['arms']) == {'S','T'}
assert m['matched_response_ids_sha256']['S'] == '2e37662fb949a43004d9842eb68166c751b2fd97273e6dc408d4b911a7a12308'
assert m['matched_response_ids_sha256']['T'] == 'd68bff648787e93da34b550bb78b4e552c0c7bdff8f7b5db7138d36f215a87d1'
assert c['authorization_scope'] == 'Chemistry S/T seed1 under frozen prereg v2 only'
assert c['idiom_training_authorized'] is False
assert c['rows'] == 1000
assert c['optimization']['train_batch_size'] == 40
assert c['optimization']['ppo_mini_batch_size'] == 40
assert c['optimization']['passes'] == 3
assert c['optimization']['updates_per_pass'] == 25
assert c['optimization']['total_updates'] == 75
assert c['optimization']['save_freq'] == 25
assert c['objective']['loss_mode'] == 'forward_kl_topk'
assert c['objective']['teacher_topk'] == 32
assert c['objective']['use_policy_gradient'] is False
assert c['replay_transport']['fresh_student_rollout_forbidden'] is True
print('FORMAL_STATIC_CONTRACT=PASS')
PY_STATIC
    [[ $? -eq 0 ]] || { fail "formal static contract failed"; return 1; }

    python - <<'PY_IMPORT'
import importlib
m = importlib.import_module('offline_prefix_support_verl_replay_manager_formal_v1')
assert hasattr(m, 'OfflinePrefixSupportFormalReplayManager')
print('FORMAL_MANAGER_IMPORT=PASS')
PY_IMPORT
    [[ $? -eq 0 ]] || { fail "formal manager import failed"; return 1; }

    if pgrep -af '[p]ython.*main_ppo_sync_wrapper_v1.py' > "$RUN/preexisting_sync_verl.txt"; then
        cat "$RUN/preexisting_sync_verl.txt"
        fail "pre-existing synchronous Verl training process detected"
        return 1
    fi

    npu-smi info > "$RUN/npu_before.txt" 2>&1 || true
    atomic_state "RUNNING" "compose_both_arms" || { fail "cannot write initial state"; return 1; }

    build_overrides() {
        local arm="$1"
        local arm_run="$2"
        local parquet="$RUN/inputs/chemistry_${arm}_full1000.parquet"
        local project="mtpatcher-prefix-replay-formal"
        local exp="chemistry-${arm}-offline-prefix-seed1-v1"
        cat <<EOF
data.train_files=['$parquet']
data.val_files=['$parquet']
data.train_batch_size=40
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
actor_rollout_ref.actor.optim.weight_decay=0.01
actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0
actor_rollout_ref.actor.optim.lr_warmup_steps=-1
actor_rollout_ref.actor.ppo_mini_batch_size=40
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
++actor_rollout_ref.rollout.agent.agent_loop_manager_class=offline_prefix_support_verl_replay_manager_formal_v1.OfflinePrefixSupportFormalReplayManager
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
trainer.total_epochs=3
trainer.total_training_steps=75
trainer.save_freq=25
trainer.default_local_dir=$arm_run/checkpoints
trainer.resume_mode=disable
trainer.balance_batch=False
trainer.critic_warmup=0
EOF
    }

    compose_arm() {
        local arm="$1"
        local arm_run="$RUN/chemistry/$arm"
        mkdir -p "$arm_run/checkpoints" "$arm_run/tensorboard" || return 1
        build_overrides "$arm" "$arm_run" > "$arm_run/overrides.txt" || return 1
        local -a ovs
        mapfile -t ovs < "$arm_run/overrides.txt"
        (
            cd "$arm_run" || return 1
            python "$WRAPPER" --cfg job "${ovs[@]}"
        ) > "$arm_run/resolved_config.yaml"
        [[ $? -eq 0 ]] || return 1

        python - "$arm" "$arm_run/resolved_config.yaml" "$RUN/input_manifest.json" <<'PY_COMPOSE'
import json, sys
from omegaconf import OmegaConf
arm, path, manifest_path = sys.argv[1:]
cfg = OmegaConf.load(path)
m = json.load(open(manifest_path, encoding='utf-8'))
assert list(cfg.data.train_files) == [m['arms'][arm]['parquet']]
assert cfg.data.train_batch_size == 40
assert cfg.data.shuffle is False
assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 40
assert cfg.actor_rollout_ref.actor.ppo_epochs == 1
assert int(cfg.actor_rollout_ref.actor.data_loader_seed) == 20260820
assert cfg.actor_rollout_ref.actor.shuffle is False
assert float(cfg.actor_rollout_ref.actor.optim.lr) == 1e-6
assert float(cfg.actor_rollout_ref.actor.optim.weight_decay) == 0.01
assert float(cfg.actor_rollout_ref.actor.optim.lr_warmup_steps_ratio) == 0.0
assert cfg.distillation.enabled is True
assert cfg.distillation.distillation_loss.loss_mode == 'forward_kl_topk'
assert int(cfg.distillation.distillation_loss.topk) == 32
assert cfg.distillation.distillation_loss.use_task_rewards is False
assert cfg.distillation.distillation_loss.use_policy_gradient is False
assert cfg.actor_rollout_ref.actor.use_kl_loss is False
assert cfg.algorithm.use_kl_in_reward is False
assert cfg.trainer.total_epochs == 3
assert cfg.trainer.total_training_steps == 75
assert cfg.trainer.save_freq == 25
assert cfg.trainer.resume_mode == 'disable'
assert cfg.trainer.val_before_train is False
assert cfg.trainer.balance_batch is False
assert list(cfg.trainer.logger) == ['console','tensorboard']
fqn = cfg.actor_rollout_ref.rollout.agent.agent_loop_manager_class
assert fqn == 'offline_prefix_support_verl_replay_manager_formal_v1.OfflinePrefixSupportFormalReplayManager'
print(f'RESOLVED_CONFIG_GATE=PASS arm={arm}')
PY_COMPOSE
    }

    echo "=== COMPOSE BOTH ARMS BEFORE ANY UPDATE ==="
    compose_arm S || { fail "S Hydra compose/config gate failed"; return 1; }
    compose_arm T || { fail "T Hydra compose/config gate failed"; return 1; }

    python - "$RUN/chemistry/S/resolved_config.yaml" "$RUN/chemistry/T/resolved_config.yaml" <<'PY_EQ'
from pathlib import Path
from omegaconf import OmegaConf
import json, sys

def load(p):
    return OmegaConf.to_container(OmegaConf.load(p), resolve=True)

def arm_tokens(cfg):
    train_files = list(cfg['data']['train_files']); val_files = list(cfg['data']['val_files'])
    local_dir = str(cfg['trainer']['default_local_dir']); arm_run = str(Path(local_dir).parent)
    exp = str(cfg['trainer']['experiment_name'])
    return sorted({str(train_files[0]): '<ARM_PARQUET>', str(val_files[0]): '<ARM_PARQUET>', arm_run: '<ARM_RUN>', exp: '<ARM_EXPERIMENT>'}.items(), key=lambda kv: len(kv[0]), reverse=True)

def canon(o, repl):
    if isinstance(o, dict): return {k: canon(v, repl) for k,v in o.items()}
    if isinstance(o, list): return [canon(v, repl) for v in o]
    if isinstance(o, str):
        for raw, marker in repl: o = o.replace(raw, marker)
    return o

def diffs(a,b,path='$',out=None):
    if out is None: out=[]
    if type(a) is not type(b): out.append((path,a,b)); return out
    if isinstance(a,dict):
        for k in sorted(set(a)|set(b)):
            if k not in a or k not in b: out.append((f'{path}.{k}',a.get(k,'<MISSING>'),b.get(k,'<MISSING>')))
            else: diffs(a[k],b[k],f'{path}.{k}',out)
        return out
    if isinstance(a,list):
        if len(a)!=len(b): out.append((path+'.length',len(a),len(b))); return out
        for i,(x,y) in enumerate(zip(a,b)): diffs(x,y,f'{path}[{i}]',out)
        return out
    if a!=b: out.append((path,a,b))
    return out

s0=load(sys.argv[1]); t0=load(sys.argv[2])
s=canon(s0,arm_tokens(s0)); t=canon(t0,arm_tokens(t0))
r=diffs(s,t)
if r:
    print('S_T_RESIDUAL_CONFIG_DIFF_COUNT='+str(len(r)))
    for x in r[:80]: print('UNEXPECTED_RESOLVED_CONFIG_DIFF '+json.dumps(x,ensure_ascii=False,default=str))
    raise SystemExit(1)
print('S_T_RESOLVED_CONFIG_MATCH=PASS')
PY_EQ
    [[ $? -eq 0 ]] || { fail "S/T resolved config mismatch"; return 1; }

    atomic_state "RUNNING" "formal_chemistry_S_then_T" || { fail "cannot update state"; return 1; }

    write_heartbeat() {
        local arm="$1"
        local arm_run="$2"
        local pid="$3"
        python - "$arm" "$arm_run" "$pid" <<'PY_HB'
import glob, json, os, re, sys, time
from pathlib import Path
arm, arm_run_s, pid = sys.argv[1:]
arm_run = Path(arm_run_s)
events=[Path(p) for p in glob.glob(str(arm_run/'tensorboard/**/events.out.tfevents.*'),recursive=True)]
events=[{'path':str(p),'bytes':p.stat().st_size} for p in events if p.is_file()]
ckpts=sorted([p.name for p in (arm_run/'checkpoints').glob('global_step_*') if p.is_dir()])
progress=arm_run/'replay_progress.json'
replay=None
if progress.is_file():
    try: replay=json.load(open(progress,encoding='utf-8'))
    except Exception: replay=None
obj={'status':'RUNNING','arm':arm,'pid':int(pid),'updated_unix':time.time(),'replay':replay,'checkpoints':ckpts,'tensorboard_events':events}
tmp=arm_run/'heartbeat.json.tmp'; final=arm_run/'heartbeat.json'
tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
os.replace(tmp,final)
PY_HB
    }

    postcheck_arm() {
        local arm="$1"
        local arm_run="$RUN/chemistry/$arm"
        python - "$arm" "$arm_run" "$RUN/input_manifest.json" <<'PY_POST'
import glob, json, math, os, re, sys
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

arm, arm_run_s, manifest_s = sys.argv[1:]
arm_run=Path(arm_run_s); m=json.load(open(manifest_s,encoding='utf-8')); a=m['arms'][arm]
progress=json.load(open(arm_run/'replay_progress.json',encoding='utf-8'))
assert progress['status']=='PASS_FORMAL_REPLAY_COMPLETE', progress
assert progress['completed_calls']==75 and progress['completed_passes']==[1,2,3]
assert progress['student_rollout_calls']==0
assert progress['forbidden_rollout_client_accesses']==[]
assert progress['teacher_top_k']==32 and progress['policy_gradient'] is False

audit=[json.loads(x) for x in (arm_run/'replay_transport_rows.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
assert len(audit)==3000, len(audit)
assert all(r['arm']==arm for r in audit)
assert all(r['student_hint_present'] is False and r['teacher_hint_present'] is True for r in audit)

pass_summaries=[]
for p in (1,2,3):
    q=json.load(open(arm_run/f'replay_pass_{p}_preupdate.json',encoding='utf-8'))
    assert q['status']=='PASS_FULL_FROZEN_PASS_PREUPDATE'
    assert q['rows']==1000 and q['batches']==25
    assert q['aggregate_response_ids_sha256']==a['aggregate_response_ids_sha256']
    assert q['row_order_sha256']==m['row_order_sha256']
    assert q['effective_supervised_token_count']==a['effective_supervised_tokens_per_pass']
    assert q['fresh_student_rollout_calls']==0
    pass_summaries.append(q)

text=(arm_run/'train.log').read_text(encoding='utf-8',errors='replace')
assert 'FRESH_ROLLOUT_FORBIDDEN' not in text
assert 'PREFIX_REPLAY_ALL_PASSES_PREUPDATE_PASS' in text
assert re.search(r'Total training steps:\s*75\b',text), 'trainer did not resolve to 75 steps'

ckpts=[]
for step in (25,50,75):
    d=arm_run/'checkpoints'/f'global_step_{step}'/'actor'
    assert d.is_dir(), d
    models=sorted(d.glob('model_world_size_8_rank_*.pt'))
    assert len(models)==8, (d,len(models))
    ckpts.append({'step':step,'actor':str(d),'model_shards':len(models)})

events=[Path(p) for p in glob.glob(str(arm_run/'tensorboard/**/events.out.tfevents.*'),recursive=True)]
events=[p for p in events if p.is_file() and p.stat().st_size>0]
assert events, 'no TensorBoard events'
scalars={}
for event_dir in sorted({p.parent for p in events}):
    ea=EventAccumulator(str(event_dir)); ea.Reload()
    for tag in ea.Tags().get('scalars',[]):
        for x in ea.Scalars(tag): scalars.setdefault(tag,{})[int(x.step)]=float(x.value)

def pick(names):
    for n in names:
        if n in scalars and scalars[n]: return n,scalars[n]
    raise AssertionError(f'missing tags {names}; have={sorted(scalars)}')
grad_tag,grads=pick(['actor/grad_norm','grad_norm'])
loss_tag,losses=pick(['actor/distillation/loss','distillation/loss'])
for step in range(1,76):
    assert step in grads, f'missing grad step {step}'
    assert step in losses, f'missing loss step {step}'
    assert math.isfinite(grads[step]) and grads[step]>0, (step,grads[step])
    assert math.isfinite(losses[step]), (step,losses[step])

batches=a['batches']; assert len(batches)==25
pass_dose=[]
for p in (1,2,3):
    start=(p-1)*25+1; stop=p*25
    toks=[int(b['effective_supervised_tokens']) for b in batches]
    total=sum(toks)
    weighted=sum(losses[start+i]*toks[i] for i in range(25))/total
    mean_grad=sum(grads[s] for s in range(start,stop+1))/25
    pass_dose.append({
        'pass':p,
        'optimizer_updates':25,
        'effective_supervised_token_count':total,
        'loss_per_token':weighted,
        'mean_grad_norm':mean_grad,
        'parameter_update_norm_l2':None,
        'parameter_update_norm_l2_status':'PENDING_OFFLINE_POSTTRAIN_CHECKPOINT_DELTA_FROM_SAME_C0',
        'checkpoint':str(arm_run/'checkpoints'/f'global_step_{stop}'),
    })

out={
    'status':'PASS_FORMAL_ARM_TRAINING_PENDING_EVAL',
    'arm':arm,
    'rows_per_pass':1000,
    'passes':3,
    'optimizer_updates':75,
    'fresh_student_rollout_calls':0,
    'teacher_top_k':32,
    'policy_gradient':False,
    'aggregate_response_ids_sha256':a['aggregate_response_ids_sha256'],
    'effective_supervised_tokens_per_pass':a['effective_supervised_tokens_per_pass'],
    'pass_dose':pass_dose,
    'grad_norm_tag':grad_tag,
    'distillation_loss_tag':loss_tag,
    'checkpoints':ckpts,
    'tensorboard_events':[{'path':str(p),'bytes':p.stat().st_size} for p in events],
    'scientific_effect_claim':None,
    'next_step':'Frozen Chemistry strict evaluation + paired bootstrap + prereg recovery-ratio screen; offline parameter_update_norm_l2 closure.',
}
tmp=arm_run/'postcheck.json.tmp'; final=arm_run/'postcheck.json'
tmp.write_text(json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8'); os.replace(tmp,final)
print(json.dumps(out,ensure_ascii=False,indent=2))
PY_POST
    }

    run_arm() {
        local arm="$1"
        local arm_run="$RUN/chemistry/$arm"
        local project="mtpatcher-prefix-replay-formal"
        local exp="chemistry-${arm}-offline-prefix-seed1-v1"
        local tb="$arm_run/tensorboard"
        [[ ! -e "$arm_run/DONE" ]] || { fail "arm $arm already DONE; refuse duplicate formal training"; return 1; }
        mkdir -p "$tb" "$arm_run/tensorboard_log/$project" || return 1
        rm -f "$arm_run/tensorboard_log/$project/$exp"
        ln -s "$tb" "$arm_run/tensorboard_log/$project/$exp" || return 1

        local -a ovs
        mapfile -t ovs < "$arm_run/overrides.txt"
        echo "=== START FORMAL ARM $arm ==="
        echo "TENSORBOARD_DIR=$tb"
        date -u +"%Y-%m-%dT%H:%M:%SZ" > "$arm_run/started_utc.txt"
        (
            cd "$arm_run" || return 1
            export TENSORBOARD_DIR="$tb"
            python "$WRAPPER" "${ovs[@]}"
        ) > "$arm_run/train.log" 2>&1 &
        local pid=$!
        echo "$pid" > "$arm_run/main_ppo.pid"

        while kill -0 "$pid" 2>/dev/null; do
            write_heartbeat "$arm" "$arm_run" "$pid" || true
            sleep 30
        done
        wait "$pid"
        local rc=$?
        echo "$rc" > "$arm_run/main_ppo_status.txt"
        if [[ "$rc" -ne 0 ]]; then
            tail -n 220 "$arm_run/train.log" || true
            fail "arm $arm main_ppo failed rc=$rc"
            return 1
        fi
        postcheck_arm "$arm" || { fail "arm $arm postcheck failed"; return 1; }
        touch "$arm_run/DONE"
        echo "ARM_${arm}_FORMAL_TRAINING=PASS_PENDING_EVAL"
        return 0
    }

    run_arm S || return 1
    run_arm T || return 1

    python - "$RUN" <<'PY_MASTER'
import json, os, sys
from pathlib import Path
run=Path(sys.argv[1])
s=json.load(open(run/'chemistry/S/postcheck.json',encoding='utf-8'))
t=json.load(open(run/'chemistry/T/postcheck.json',encoding='utf-8'))
assert s['status']==t['status']=='PASS_FORMAL_ARM_TRAINING_PENDING_EVAL'
assert s['optimizer_updates']==t['optimizer_updates']==75
assert s['effective_supervised_tokens_per_pass']==t['effective_supervised_tokens_per_pass']
assert s['fresh_student_rollout_calls']==t['fresh_student_rollout_calls']==0
summary={
    'status':'PASS_FORMAL_CHEMISTRY_ST_SEED1_TRAINING_PENDING_EVAL',
    'scientific_class':'DIAGNOSTIC ONLY / OFFLINE PREFIX-SUPPORT REPLAY',
    'problem':'Train frozen Student-prefix and Teacher-prefix Chemistry replay arms under the preregistered matched native-Verl contract.',
    'result':{'S':s,'T':t},
    'interpretation':'Formal training transport/dose completed. No S-vs-T scientific effect claim is made before frozen Chemistry evaluation and preregistered screening.',
    'next_step':'Evaluate Chemistry strict accuracy at P1/P2/P3, run paired bootstrap, compute recovery-ratio screen, and close offline parameter_update_norm_l2. Do not launch Idiom automatically.',
    'idiom_training_authorized':False,
}
tmp=run/'summary.json.tmp'; final=run/'summary.json'
tmp.write_text(json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8'); os.replace(tmp,final)
print(json.dumps(summary,ensure_ascii=False,indent=2))
PY_MASTER
    [[ $? -eq 0 ]] || { fail "master summary failed"; return 1; }

    atomic_state "PASS_FORMAL_CHEMISTRY_ST_SEED1_TRAINING_PENDING_EVAL" "complete_pending_eval" || { fail "final state write failed"; return 1; }
    npu-smi info > "$RUN/npu_after.txt" 2>&1 || true

    echo
    echo "============================================================"
    echo "Problem -> Result -> Interpretation -> Next step"
    echo "============================================================"
    echo "Problem: execute formal Chemistry S/T seed1 frozen-prefix replay under native Verl."
    echo "Result: S and T each completed 3 exact 1000-row passes / 75 updates with checkpoints at 25/50/75 and TensorBoard."
    echo "Interpretation: formal training PASS only; scientific S-vs-T effect remains pending frozen evaluation."
    echo "Next step: Chemistry strict evaluation + paired bootstrap + recovery ratio + offline parameter delta norm. Idiom remains blocked."
    echo "FINAL_RESULT=PASS_FORMAL_CHEMISTRY_ST_SEED1_TRAINING_PENDING_EVAL"
    return 0
}

main "$@"
