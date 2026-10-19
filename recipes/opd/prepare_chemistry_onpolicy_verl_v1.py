#!/usr/bin/env python3
"""Freeze a versioned native-verl Hydra launch contract. NO training occurs.

Formal launches are deliberately disabled pending a validated 1-step native
smoke, exact last-batch accounting, scheduler/optimizer and fixed-trajectory
numerical parity. Do not edit this guard without creating a new reviewed run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path('/workspace/mtpatcher')
PROJECT=ROOT/'repo/MT-Patcher-Reproduction-Ascend'
VERL=ROOT/'repo/verl-v0.9.0'
DATA_ROOT=ROOT/'data/chem_onpolicy_verl_v1_20261016'


def sha(path:Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def git_head(repo:Path):
    return subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()


def git_status(repo:Path):
    return subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['smoke','formal'],required=True)
    ap.add_argument('--lr',choices=['1e-5','3e-5'],default='1e-5')
    ap.add_argument('--run',type=Path,required=True)
    args=ap.parse_args()
    if args.mode=='formal':
        raise SystemExit('FORMAL_NOT_AUTHORIZED: verify on-policy smoke, 5500-row tail batch, '
                         'optimizer/scheduler/GA4 equivalence first. No formal launcher emitted.')
    manifest_path=DATA_ROOT/'data_manifest.json'
    if not manifest_path.exists():
        raise SystemExit(f'Build frozen data first: {manifest_path}')
    manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    assert manifest['status']=='DATA_ONLY_NO_TRAINING_AUTHORIZATION'
    inp=Path(manifest['files']['smoke16']['path'])
    assert sha(inp)==manifest['files']['smoke16']['sha256']
    assert manifest['files']['smoke16']['rows']==16
    assert args.mode == 'smoke', 'RUN_CLASS must be smoke'
    expected_parent = ROOT/'runs/smoke/targeted'
    if args.run.resolve().parent != expected_parent.resolve():
        raise SystemExit(f'SMOKE_PATH_VIOLATION: expected under {expected_parent}, got {args.run}')
    if args.run.exists():
        raise SystemExit(f'Run directory already exists: {args.run}; refusing to reuse')
    assert args.run.is_absolute(), 'run must be an absolute path'
    run=args.run
    project_head=git_head(PROJECT)
    verl_head=git_head(VERL)
    # Only untracked files are expected; we DO NOT modify or stage any of them.
    dirty=git_status(PROJECT)
    repo_script=PROJECT/'scripts/opd/chemistry_onpolicy_manager_v1.py'
    assert repo_script.is_file(),f'Copy manager to {repo_script}'
    assert (PROJECT/'scripts/opd/chemistry_zero_reward_v1.py').is_file()
    assert (PROJECT/'scripts/targeted/main_ppo_sync_wrapper_v1.py').is_file()
    assert (VERL/'verl/experimental/agent_loop/agent_loop.py').is_file()
    assert (ROOT/'envs/verl-v0.9.0-a3/bin/python').is_file()
    p=run/'checkpoints'
    opts=[
        f"data.train_files=['{inp}']",
        f"data.val_files=['{inp}']",
        'data.train_batch_size=16',
        'data.max_prompt_length=767',
        'data.max_response_length=256',
        'data.filter_overlong_prompts=False',
        'data.truncation=error',
        'data.shuffle=False',
        '+data.apply_chat_template_kwargs.enable_thinking=False',
        'algorithm.adv_estimator=grpo',
        'algorithm.use_kl_in_reward=False',
        f'reward.custom_reward_function.path={PROJECT}/scripts/opd/chemistry_zero_reward_v1.py',
        'reward.custom_reward_function.name=compute_score',
        f'actor_rollout_ref.model.path={ROOT}/models/Qwen3-0.6B',
        'actor_rollout_ref.model.use_remove_padding=True',
        'actor_rollout_ref.model.enable_gradient_checkpointing=False',
        'actor_rollout_ref.actor.use_torch_compile=False',
        f'actor_rollout_ref.actor.optim.lr={args.lr}',
        'actor_rollout_ref.actor.optim.weight_decay=0.01',
        'actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0', # smoke only
        'actor_rollout_ref.actor.optim.lr_warmup_steps=-1',
        'actor_rollout_ref.actor.ppo_mini_batch_size=16',
        'actor_rollout_ref.actor.ppo_epochs=1',
        'actor_rollout_ref.actor.data_loader_seed=20260820',
        'actor_rollout_ref.actor.shuffle=False',
        'actor_rollout_ref.actor.loss_agg_mode=token-mean',
        'actor_rollout_ref.actor.use_dynamic_bsz=True',
        'actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096',
        'actor_rollout_ref.actor.fsdp_config.param_offload=False',
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload=False',
        'actor_rollout_ref.actor.use_kl_loss=False',
        'actor_rollout_ref.rollout.name=vllm',
        'actor_rollout_ref.rollout.tensor_model_parallel_size=2',
        'actor_rollout_ref.rollout.n=1',
        'actor_rollout_ref.rollout.temperature=1.0',
        'actor_rollout_ref.rollout.top_p=1.0',
        'actor_rollout_ref.rollout.top_k=-1',
        'actor_rollout_ref.rollout.dtype=bfloat16',
        'actor_rollout_ref.rollout.gpu_memory_utilization=0.35',
        'actor_rollout_ref.rollout.max_model_len=1024',
        'actor_rollout_ref.rollout.enforce_eager=True',
        'actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True',
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2',
        'actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096',
        '++actor_rollout_ref.rollout.agent.agent_loop_manager_class=chemistry_onpolicy_manager_v1.ChemistryOnPolicyManager',
        'distillation.enabled=True',
        'distillation.distillation_loss.loss_mode=forward_kl_topk',
        'distillation.distillation_loss.topk=32',
        'distillation.distillation_loss.use_task_rewards=False',
        'distillation.distillation_loss.use_policy_gradient=False',
        'distillation.distillation_loss.loss_max_clamp=null',
        'distillation.distillation_loss.log_prob_min_clamp=null',
        'distillation.n_gpus_per_node=4',
        'distillation.nnodes=1',
        f'distillation.teacher_models.teacher_model.model_path={ROOT}/models/Qwen3-8B',
        'distillation.teacher_models.teacher_model.inference.name=vllm',
        'distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=2',
        'distillation.teacher_models.teacher_model.inference.data_parallel_size=1',
        'distillation.teacher_models.teacher_model.inference.dtype=bfloat16',
        'distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.40',
        'distillation.teacher_models.teacher_model.inference.enforce_eager=True',
        'distillation.teacher_models.teacher_model.inference.max_model_len=1024',
        'trainer.nnodes=1',
        'trainer.n_gpus_per_node=8',
        'trainer.project_name=mtpatcher-chem-onpolicy-smoke',
        f'trainer.experiment_name=smoke-chem-onpolicy-{run.name}',
        "trainer.logger=['console','tensorboard']",
        'trainer.val_before_train=False',
        'trainer.test_freq=-1',
        'trainer.total_epochs=1',
        'trainer.total_training_steps=1',
        'trainer.save_freq=1',
        f'trainer.default_local_dir={p}',
        'trainer.resume_mode=disable',
        'trainer.balance_batch=False',
        'trainer.critic_warmup=0',
    ]
    # verl teacher inference requires one additional token of context.
    assert 767 + 256 + 1 <= 1024
    assert manifest['max_student_prompt_tokens'] + 256 <= 1024
    assert manifest['max_teacher_prompt_tokens'] + 256 <= 1024
    run.mkdir(parents=True,exist_ok=False)
    (run/'overrides.txt').write_text('\n'.join(opts)+'\n',encoding='utf8')
    contract={
        'status':'PREPARED_NOT_EXECUTED', 'RUN_CLASS':'smoke', 'mode':args.mode,
        'run':str(run),'population':16,'lr':args.lr,
        'project_commit':project_head,'project_dirty':dirty,
        'verl_commit':verl_head,
        'data_file':str(inp),'data_sha256':sha(inp),
        'manager_sha256':sha(repo_script),
        'native_model_update':'NOT_YET_VERIFIED',
        'scheduler_and_ga4_parity':'NOT_YET_VERIFIED',
        'tail_batch_5500':'NOT_YET_VERIFIED',
        'evaluation':'NOT_RUN',
        'authorization':'ONE_STEP_ENGINEERING_SMOKE_ONLY',
    }
    (run/'run_contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('PREPARE=PASS_NOT_EXECUTED')
    print(f'RUN={run}')
    print(f'OVERRIDES={run}/overrides.txt')
    print('RUN_SMOKE_WITH=recipes/opd/run_chemistry_onpolicy_smoke_v1.sh '+str(run))

if __name__=='__main__':
    main()
