#!/usr/bin/env python3
"""Rewrite an existing Stage1 SFT override template; never guess a full Verl config.

The existing template must contain all scientific knobs, so a missing field fails
rather than silently letting native Verl apply a different default.
"""
import argparse
import json
import re
from pathlib import Path
import yaml

ROOT = Path('/workspace/mtpatcher')
PROJECT = ROOT/'repo/MT-Patcher-Reproduction-Ascend'
DATASET = PROJECT/'scripts/data/verl_mt_response_sft_dataset.py'
C0 = ROOT/'models/Qwen3-0.6B'


def read_template(path):
    obj = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    assert isinstance(obj, dict) and isinstance(obj.get('overrides'), list), 'EXPECTED_STAGE1_YAML_WITH_OVERRIDES_ARRAY'
    mp, order = {}, []
    for item in obj['overrides']:
        assert isinstance(item, str) and '=' in item, f'BAD_OVERRIDE {item!r}'
        key, value = item.split('=',1)
        name = key.lstrip('+')
        assert name not in mp, f'DUPLICATE_OVERRIDE {name}'
        mp[name] = (key, value)
        order.append(name)
    return obj, mp, order

def existing_one(mp, *keys):
    present = [x for x in keys if x in mp]
    assert len(present) == 1, f'EXPECTED_EXACTLY_ONE_OVERRIDE_{keys}_PRESENT_{present}'
    return present[0]

def make(args):
    obj, mp, order = read_template(args.template)
    def set_value(name, value, *, must_exist=True):
        if name not in mp:
            if must_exist:
                raise RuntimeError(f'MISSING_REQUIRED_STAGE1_OVERRIDE {name}; inspect real Stage1 YAML')
            order.append(name)
            mp[name] = ('++'+name, str(value))
        else:
            mp[name] = (mp[name][0], str(value))
    def verify_value(name, want):
        key = existing_one(mp, name)
        got = mp[key][1].strip(" \"'")
        assert got == want, f'UNMATCHED_SCIENTIFIC_KNOB {name} actual={got} expected={want}'

    # These keys are known from the actual user-presented native SFT source and
    # Stage1 planned config. Ignore neither dataset nor checkpoint location.
    set_value('data.train_files', f"['{args.data}']")
    set_value('data.train_batch_size', '16')
    set_value('data.custom_cls.path', str(DATASET))
    set_value('data.custom_cls.name', 'MTResponseOnlySFTDataset')
    set_value('trainer.total_epochs', str(args.epochs))
    set_value('trainer.total_training_steps', str(args.horizon))
    set_value('trainer.save_freq', str(args.save_freq))
    set_value('trainer.test_freq', '-1')
    set_value('trainer.logger', "['console','tensorboard']")
    set_value('trainer.project_name', 'chem_clean_sft_matched_v2')
    set_value('trainer.experiment_name', args.run_name)
    set_value('trainer.default_local_dir', str(args.ckpt_dir))
    set_value('data.train_sampler_shuffle', 'false', must_exist=False)
    # Optional native config properties, present in several released Verl v0.9
    # variants. Trainer uses .get(..., True), thus setting false is mandatory.
    if 'data.val_files' in mp:
        set_value('data.val_files', 'null')
    if 'data.max_length' in mp:
        set_value('data.max_length','1024')
    if 'data.truncation' in mp:
        set_value('data.truncation','error')
    set_value('data.num_workers','0',must_exist=False)
    if 'trainer.n_gpus_per_node' in mp:
        set_value('trainer.n_gpus_per_node', '8')
    if 'trainer.nnodes' in mp:
        set_value('trainer.nnodes', '1')
    if 'data.micro_batch_size_per_gpu' in mp:
        set_value('data.micro_batch_size_per_gpu', '1')
    for key in ('model.partial_pretrain','model.path'):
        if key in mp:
            set_value(key, str(C0))
    assert any(k in mp for k in ('model.partial_pretrain','model.path')), 'STUDENT_MODEL_OVERRIDE_MISSING'
    if 'model.tokenizer_path' in mp:
        set_value('model.tokenizer_path', str(C0))
    if 'optim.lr' in mp:
        set_value('optim.lr', '3e-5')
    else:
        raise RuntimeError('STAGE1_TEMPLATE_MISSING_OPTIM_LR')
    if 'optim.weight_decay' in mp:
        set_value('optim.weight_decay','0.01')
    else:
        raise RuntimeError('STAGE1_TEMPLATE_MISSING_WEIGHT_DECAY')
    sched = existing_one(mp, 'optim.lr_scheduler', 'optim.lr_scheduler_type')
    set_value(sched,'cosine')
    warmup = existing_one(mp, 'optim.lr_warmup_steps_ratio', 'optim.warmup_steps_ratio')
    set_value(warmup,'0.03')
    # Explicitly match 1 optimizer-update per 16-source batch under 8 DP ranks.
    if 'model.lora_rank' in mp:
        set_value('model.lora_rank', '0')
    if args.resume:
        ckpt = Path(args.resume).resolve(strict=True)
        assert ckpt.is_dir() and re.fullmatch(r'global_step_\d+', ckpt.name), \
            'RESUME_MUST_POINT_TO_GLOBAL_STEP_DIRECTORY'
        assert (ckpt / 'data_0.pt').is_file(), \
            'DATALOADER_STATE_MISSING_IN_CHECKPOINT'
        set_value('trainer.resume_mode', 'resume_path', must_exist=False)
        set_value('trainer.resume_from_path', str(ckpt), must_exist=False)
    else:
        if 'trainer.resume_mode' in mp:
            set_value('trainer.resume_mode', 'disable')
        if 'trainer.resume_from_path' in mp:
            set_value('trainer.resume_from_path', 'null')
    # Only one specified save directory; no cleanup or overwrite.
    obj['overrides'] = [mp[name][0] + '=' + mp[name][1] for name in order]
    out = Path(args.output)
    assert not out.exists(), f'REFUSING_OVERWRITE {out}'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(obj, allow_unicode=True, sort_keys=False), encoding='utf-8')
    print('CONFIG_CREATED', out)
    for k in ('data.train_files','data.custom_cls.name','trainer.total_training_steps',
              'trainer.total_epochs','optim.lr','trainer.default_local_dir','trainer.logger'):
        print('CONFIG_KNOB',k,mp[k][1])
    print('NATIVE_CONFIG_COMPOSE_AND_SMOKE_STILL_REQUIRED')

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--template', required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--data',required=True)
    ap.add_argument('--ckpt-dir',required=True)
    ap.add_argument('--run-name',required=True)
    ap.add_argument('--epochs',type=int,required=True)
    ap.add_argument('--horizon',type=int,default=1032)
    ap.add_argument('--save-freq',type=int,default=160)
    ap.add_argument('--resume',default='')
    make(ap.parse_args())
