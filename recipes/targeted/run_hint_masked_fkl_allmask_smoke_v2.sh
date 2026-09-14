#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/mtpatcher
REPO="$ROOT/repo/MT-Patcher-Reproduction-Ascend"
VERL="$ROOT/repo/verl-v0.9.0"
ENV="$ROOT/envs/verl-v0.9.0-a3"
PY="$ENV/bin/python"
RUN=${RUN:?}
S_INPUT=${S_INPUT:?}
IMPLEMENTATION_HEAD=${IMPLEMENTATION_HEAD:?}
WRAPPER="$REPO/scripts/targeted/main_ppo_sync_wrapper_v1.py"
ZERO="$REPO/scripts/opd/constant_zero_reward.py"
STUDENT="$ROOT/models/Qwen3-0.6B"
TEACHER="$ROOT/models/Qwen3-8B"
BASE=offline_prefix_support_verl_replay_manager_formal_v1.OfflinePrefixSupportFormalReplayManager
ONES=hint_masked_fkl_replay_manager_v2.OfflinePrefixSupportAllOnesReplayManagerV2
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO/scripts/targeted:$REPO:$VERL:${PYTHONPATH:-}"
[[ -f "$ROOT/project_env.sh" ]] && source "$ROOT/project_env.sh"
export PATH="$ENV/bin:$PATH";unset PYTHONHOME || true;hash -r
die(){ echo "FAIL: $*" >&2;exit 1; }
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$IMPLEMENTATION_HEAD" ]] || die "HEAD drift"
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || die "repo dirty"
[[ -s "$S_INPUT" ]] || die "S input absent"

overrides(){
 local arm="$1" manager="$2" dir="$3"
 cat <<EOF
data.train_files=['$S_INPUT']
data.val_files=['$S_INPUT']
data.train_batch_size=40
data.max_prompt_length=1024
data.max_response_length=256
data.filter_overlong_prompts=False
data.truncation=error
data.shuffle=False
+data.apply_chat_template_kwargs.enable_thinking=False
algorithm.adv_estimator=grpo
algorithm.use_kl_in_reward=False
reward.custom_reward_function.path=$ZERO
reward.custom_reward_function.name=compute_score
actor_rollout_ref.model.path=$STUDENT
actor_rollout_ref.model.use_remove_padding=False
actor_rollout_ref.model.enable_gradient_checkpointing=False
actor_rollout_ref.actor.use_torch_compile=False
actor_rollout_ref.actor.optim.lr=1.0e-6
actor_rollout_ref.actor.optim.weight_decay=0.01
actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0
actor_rollout_ref.actor.optim.lr_warmup_steps=-1
actor_rollout_ref.actor.ppo_mini_batch_size=40
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
actor_rollout_ref.actor.ppo_epochs=1
actor_rollout_ref.actor.data_loader_seed=20260820
actor_rollout_ref.actor.shuffle=False
actor_rollout_ref.actor.loss_agg_mode=token-mean
actor_rollout_ref.actor.use_dynamic_bsz=False
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
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096
++actor_rollout_ref.rollout.agent.agent_loop_manager_class=$manager
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
trainer.project_name=mtpatcher-hint-masked-allmask-smoke
trainer.experiment_name=chemistry-S-${arm}-oneupdate-v1
trainer.logger=['console','tensorboard']
trainer.val_before_train=False
trainer.test_freq=-1
trainer.total_epochs=1
trainer.total_training_steps=1
trainer.save_freq=1
trainer.default_local_dir=$dir/checkpoints
trainer.resume_mode=disable
trainer.balance_batch=False
trainer.critic_warmup=0
EOF
}
compose(){
 local arm="$1" manager="$2" dir="$RUN/$arm"
 mkdir -p "$dir/checkpoints" "$dir/tensorboard"
 overrides "$arm" "$manager" "$dir" > "$dir/overrides.txt"
 mapfile -t O < "$dir/overrides.txt"
 (cd "$dir";"$PY" "$WRAPPER" --cfg job "${O[@]}") > "$dir/resolved_config.yaml"
}
runarm(){
 local arm="$1" dir="$RUN/$arm" project=mtpatcher-hint-masked-allmask-smoke exp="chemistry-S-${arm}-oneupdate-v1"
 mkdir -p "$dir/tensorboard" "$dir/tensorboard_log/$project"
 rm -f "$dir/tensorboard_log/$project/$exp";ln -s "$dir/tensorboard" "$dir/tensorboard_log/$project/$exp"
 mapfile -t O < "$dir/overrides.txt"
 (cd "$dir";export TENSORBOARD_DIR="$dir/tensorboard";"$PY" "$WRAPPER" "${O[@]}") > "$dir/train.log" 2>&1 &
 pid=$!;echo "$pid" > "$dir/main_ppo.pid";rc=0;wait "$pid" || rc=$?
 [[ "$rc" -eq 0 ]] || { tail -n 160 "$dir/train.log" || true;die "$arm rc=$rc"; }
 touch "$dir/DONE"
}

compose DENSE "$BASE"
compose ONES "$ONES"
"$PY" - "$RUN/DENSE/resolved_config.yaml" "$RUN/ONES/resolved_config.yaml" <<'PY_CFG'
import sys
from omegaconf import OmegaConf
def load(p):return OmegaConf.to_container(OmegaConf.load(p),resolve=True)
a,b=load(sys.argv[1]),load(sys.argv[2])
for x in (a,b):
    x["trainer"]["experiment_name"]="<EXP>";x["trainer"]["default_local_dir"]="<DIR>"
    x["actor_rollout_ref"]["rollout"]["agent"]["agent_loop_manager_class"]="<MANAGER>"
assert a==b
assert int(a["trainer"]["total_training_steps"])==1 and int(a["trainer"]["save_freq"])==1
assert int(a["data"]["train_batch_size"])==40
assert a["actor_rollout_ref"]["model"]["use_remove_padding"] is False
assert a["actor_rollout_ref"]["actor"]["use_dynamic_bsz"] is False
assert int(a["actor_rollout_ref"]["actor"]["ppo_micro_batch_size_per_gpu"])==1
assert a["actor_rollout_ref"]["rollout"]["log_prob_use_dynamic_bsz"] is False
assert int(a["actor_rollout_ref"]["rollout"]["log_prob_micro_batch_size_per_gpu"])==1
assert float(a["actor_rollout_ref"]["actor"]["optim"]["weight_decay"])==0.01
print("DENSE_ONES_CONFIG_MATCH=PASS")
print("VALIDATED_FORMAL_STACK_MATCH=PASS")
PY_CFG

echo "=== DENSE ===";runarm DENSE
sleep 3
echo "=== ONES ===";runarm ONES

"$PY" - "$RUN" <<'PY_EQ'
import json,math,os,sys
from pathlib import Path
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
R=Path(sys.argv[1])

def scalars(arm):
    out={}
    events=list((R/arm/"tensorboard").rglob("events.out.tfevents.*"))
    assert any(p.stat().st_size>0 for p in events)
    for d in {p.parent for p in events if p.stat().st_size>0}:
        e=EventAccumulator(str(d));e.Reload()
        for tag in e.Tags().get("scalars",[]):
            v=e.Scalars(tag)
            if v:out.setdefault(tag,[]).extend(x.value for x in v)
    return out
def pick(s,names):
    for n in names:
        if n in s and s[n]:return n,float(s[n][-1])
    raise AssertionError((names,sorted(s)))
D,O=scalars("DENSE"),scalars("ONES")
ldn,ld=pick(D,["actor/distillation/loss","distillation/loss"])
lon,lo=pick(O,["actor/distillation/loss","distillation/loss"])
gdn,gd=pick(D,["actor/grad_norm","grad_norm"])
gon,go=pick(O,["actor/grad_norm","grad_norm"])
rn,ratio=pick(O,["actor/distillation/token_weight_selected_ratio","distillation/token_weight_selected_ratio"])
assert abs(ratio-1)<=1e-7
assert abs(ld-lo)<=1e-6,(ld,lo)
assert abs(gd-go)<=1e-5*max(1.,abs(gd),abs(go)),(gd,go)
t=json.load(open(R/"ONES"/"token_weight_transport.json",encoding="utf-8"))
assert t["selected_tokens"]==t["valid_response_tokens"] and t["padding_weight_nonzero"]==0 and t["requires_grad"] is False

def files(arm):
    p=R/arm/"checkpoints"/"global_step_1"/"actor"
    fs=sorted(p.glob("model_world_size_*_rank_*.pt"))
    assert len(fs)==8,(arm,[str(x) for x in fs])
    return fs
def walk(x,p=""):
    if torch.is_tensor(x):yield p,x
    elif isinstance(x,dict):
        for k in sorted(x,key=str):yield from walk(x[k],p+"."+str(k) if p else str(k))
    elif isinstance(x,(list,tuple)):
        for i,v in enumerate(x):yield from walk(v,f"{p}[{i}]")
exact=True;mx=0.;count=0;elems=0
for a,b in zip(files("DENSE"),files("ONES"),strict=True):
    A=torch.load(a,map_location="cpu",weights_only=False)
    B=torch.load(b,map_location="cpu",weights_only=False)
    ta=dict(walk(A));tb=dict(walk(B));assert ta.keys()==tb.keys()
    for k in ta:
        x,y=ta[k],tb[k];assert x.shape==y.shape and x.dtype==y.dtype
        count+=1;elems+=x.numel()
        if not torch.equal(x,y):
            exact=False
            d=(x.float()-y.float()).abs().max().item() if x.is_floating_point() else float((x!=y).any())
            mx=max(mx,d)
    del A,B,ta,tb
assert exact,f"saved model parameters differ max_abs={mx}"
obj={
 "status":"PASS_ALL_MASK_EQUIVALENCE_SMOKE",
 "loss":{"dense":ld,"ones":lo,"abs_diff":abs(ld-lo),"dense_tag":ldn,"ones_tag":lon},
 "grad_norm":{"dense":gd,"ones":go,"abs_diff":abs(gd-go),"dense_tag":gdn,"ones_tag":gon},
 "ones_selected_ratio":{"tag":rn,"value":ratio},
 "transport":t,
 "checkpoint_parameter_equivalence":{"exact":exact,"max_abs_diff":mx,"tensor_count":count,"elements":elems},
 "formal_training_authorized":False,
}
tmp=R/"equivalence.json.tmp";tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n",encoding="utf-8");os.replace(tmp,R/"equivalence.json")
print(json.dumps(obj,indent=2,sort_keys=True))
print("ALL_MASK_EQUIVALENCE_GATE=PASS")
PY_EQ
echo "FINAL_RESULT=PASS_HINT_MASKED_FKL_ALL_MASK_EQUIVALENCE_SMOKE"
