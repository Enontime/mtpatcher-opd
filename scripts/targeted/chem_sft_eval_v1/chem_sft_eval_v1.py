#!/usr/bin/env python3
"""Sequential frozen Chemistry1000 eval + audited evaluation TensorBoard.

Run on Ascend server, not during training. Does not modify the frozen evaluator.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT=Path('/workspace/mtpatcher')
RUN=ROOT/'runs/formal/chem_clean_sft_matched_v2'
EVAL=ROOT/'native_chem12h_v2/chemistry1000_native_parity_eval_v1.py'
MERGER=ROOT/'repo/MT-Patcher-Reproduction-Ascend/scripts/targeted/chem_clean_sft_v2/merge_sft_step.sh'
PYEVAL=ROOT/'envs/mtpatcher-npu-py311/bin/python'
PYVERL=ROOT/'envs/verl-v0.9.0-a3/bin/python'
STEPS=(160,320,480,640,800,960,1029)
BASE=RUN/'evaluation/chemistry1000_sft_curve_v1'
RECORD_SUFFIXES={'.json','.jsonl'}
HIT_KEYS=('canonical_strict_hit','strict_hit','is_strict_hit','canonical_hit','hit','is_hit')
GROUP_KEYS=('subset','split','group','bucket','category','category_name','dataset','type','source_group','eval_group')
COUNT_KEYS=('hits','hit_count','strict_hits','canonical_hits','correct','n_correct','num_correct','matched','count_correct')
TOTAL_KEYS=('total','n','num_samples','num_examples','count','evaluated','n_samples')
ACCURACY_KEYS=('strict_accuracy','canonical_accuracy','accuracy','hit_rate','strict_hit_rate')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()


def normalized_group(s):
    if not isinstance(s,str):return None
    u=re.sub(r'[^A-Z0-9]','',s.upper())
    if u in ('UC','CHEMISTRYUC') or u.endswith('UC500'): return 'UC'
    if u in ('UW','CHEMISTRYUW') or u.endswith('UW500'): return 'UW'
    if u in ('ALL','OVERALL','CHEMISTRY','CHEMISTRY1000','TOTAL','ALL1000','OVERALL1000'): return 'ALL'
    return None


def valid_count(a,total):
    if isinstance(a,bool):return None
    if isinstance(a,str) and re.fullmatch(r'\d+',a):a=int(a)
    if isinstance(a,(int,float)) and math.isfinite(a) and float(a).is_integer() and 0<=a<=total:return int(a)
    return None


def numeric_rate(a):
    if isinstance(a,bool):return None
    if isinstance(a,str):
        if a.endswith('%'):
            try:return float(a[:-1])/100
            except ValueError:return None
        try:a=float(a)
        except ValueError:return None
    if isinstance(a,(float,int)) and math.isfinite(a):
        if 0 <= a <= 1:return float(a)
        if 1 < a <= 100:return float(a)/100
    return None


def extract_count(obj,expected):
    if not isinstance(obj,dict):return None
    obj={str(k).lower():v for k,v in obj.items()}
    for k in TOTAL_KEYS:
        if k in obj:
            total=valid_count(obj[k],10**8)
            if total is not None and total!=expected:return None
    for k in COUNT_KEYS:
        if k in obj:
            count=valid_count(obj[k],expected)
            if count is not None:
                for rk in ACCURACY_KEYS:
                    if rk in obj:
                        rate=numeric_rate(obj[rk])
                        if rate is not None and abs(count/expected-rate)>0.002:return None
                return count
    for k in ACCURACY_KEYS:
        if k in obj:
            r=numeric_rate(obj[k])
            if r is not None:
                count=round(r*expected)
                if abs(count/expected-r) < 0.00005:return count
    return None


def walk_dicts(obj, depth=0):
    if depth>10:return
    if isinstance(obj,dict):
        yield obj
        for v in obj.values():yield from walk_dicts(v,depth+1)
    elif isinstance(obj,list) and len(obj)<30:
        for v in obj:yield from walk_dicts(v,depth+1)


def extract_from_summary(obj):
    # Summary must explicitly identify each subgroup; the 1000 denominator is mandatory.
    for d in walk_dicts(obj):
        flat={str(k).lower():v for k,v in d.items()}
        flat_count={}
        for g,keys,denom in (('overall',('strict_hits','overall_hits','total_hits','canonical_hits'),1000),
                              ('UC',('uc_hits','uc500_hits','uc_correct'),500),
                              ('UW',('uw_hits','uw500_hits','uw_correct'),500)):
            for k in keys:
                if k in flat:
                    n=valid_count(flat[k],denom)
                    if n is not None:flat_count[g]=n;break
        if len(flat_count)==3 and flat_count['overall']==flat_count['UC']+flat_count['UW']:
            return flat_count
        groups={}
        for k,v in d.items():
            label=normalized_group(k)
            if label and isinstance(v,dict):
                n=extract_count(v,1000 if label=='ALL' else 500)
                if n is not None:groups[label]=n
        if len(groups)==3 and groups['ALL']==groups['UC']+groups['UW']:
            return {'overall':groups['ALL'],'UC':groups['UC'],'UW':groups['UW']}
    return None


def collect_rows(obj):
    if isinstance(obj,list) and len(obj)==1000 and all(isinstance(v,dict) for v in obj):
        return obj
    if isinstance(obj,dict):
        for k in ('rows','results','predictions','samples','cases','records','items','evaluations'):
            v=obj.get(k)
            if isinstance(v,list) and len(v)==1000 and all(isinstance(x,dict) for x in v):return v
    return None


def extract_from_rows(rows):
    if len(rows)!=1000:return None
    counts={'UC':0,'UW':0}; totals={'UC':0,'UW':0}
    for row in rows:
        category=next((normalized_group(row[k]) for k in GROUP_KEYS if k in row and normalized_group(row[k]) in ('UC','UW')),None)
        if category is None:return None
        val=next((row[k] for k in HIT_KEYS if k in row),None)
        if isinstance(val,bool):hit=int(val)
        elif val in (0,1):hit=int(val)
        else:return None
        totals[category]+=1;counts[category]+=hit
    if totals!={'UC':500,'UW':500}:return None
    return {'overall':counts['UC']+counts['UW'],'UC':counts['UC'],'UW':counts['UW']}


def load_candidate_files(directory):
    files=[]
    for p in directory.rglob('*'):
        if p.is_file() and p.suffix.lower() in RECORD_SUFFIXES and p.stat().st_size<25_000_000:
            files.append(p)
    return files


def parse_metrics(eval_dir):
    candidates=[]; inspected=[]
    for file in load_candidate_files(eval_dir):
        try:
            if file.suffix=='.jsonl':
                with file.open(encoding='utf-8') as f: obj=[json.loads(line) for line in f if line.strip()]
            else: obj=json.loads(file.read_text(encoding='utf-8'))
        except (OSError,ValueError,UnicodeError) as e:
            inspected.append(f'{file}: unreadable {e}');continue
        metrics=extract_from_rows(collect_rows(obj)) if collect_rows(obj) is not None else extract_from_summary(obj)
        sample_keys=(list(obj[0].keys()) if isinstance(obj,list) and obj and isinstance(obj[0],dict) else list(obj.keys()) if isinstance(obj,dict) else [])
        inspected.append(f'{file}: metrics={metrics} keys={sample_keys[:35]}')
        if metrics:candidates.append((file,metrics))
    if not candidates:
        raise RuntimeError('EVALUATOR_METRIC_SCHEMA_UNKNOWN. Preserve original output; inspect files and adapt parser rather than fabricate results.\n'+'\n'.join(inspected))
    reference=candidates[0][1]
    for p,metrics in candidates[1:]:
        if metrics!=reference:
            raise RuntimeError(f'CONFLICTING_EVALUATOR_METRICS: {candidates}')
    return reference,[str(p) for p,_ in candidates],inspected


def ensure_idle():
    p=subprocess.run(['pgrep','-af',r'[v]erl.trainer.sft_trainer|[m]ain_ppo_sync_wrapper_v1.py|[t]orchrun'],text=True,capture_output=True)
    if p.returncode==0 and p.stdout.strip():raise RuntimeError('TRAINING_PROCESSES_ACTIVE; stop evaluation start\n'+p.stdout)


def check_contract():
    assert EVAL.is_file(),f'FROZEN_EVALUATOR_NOT_FOUND: {EVAL}'
    assert MERGER.is_file(),f'NATIVE_MERGER_NOT_FOUND: {MERGER}'
    assert (RUN/'SMOKE_V2_PASS').is_file() or (ROOT/'smoke/chem_clean_sft_v2/20261017T121524Z/SMOKE_V2_PASS').is_file()
    data=ROOT/'data/chem_clean_sft_matched_v1/chem_clean_sft_5500.parquet'
    assert sha(data)=='4cb1bac020062067db77333fde5c47700aa80f15bc0499c191855a4b7ad3cd4f', 'FORMAL_DATA_CHANGED'
    assert all((RUN/f'checkpoints/global_step_{st}').is_dir() for st in STEPS), 'MISSING_SFT_CHECKPOINT'
    return sha(EVAL)


def run_checked(cmd,log,env=None):
    print('COMMAND', ' '.join(map(str,cmd)),flush=True)
    with log.open('x',encoding='utf-8') as f:
        f.write('COMMAND: '+' '.join(map(str,cmd))+'\n');f.flush()
        p=subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,env=env)
        f.write(f'\nEXIT_CODE={p.returncode}\n')
    if p.returncode!=0:raise RuntimeError(f'PROCESS_FAILED exit={p.returncode} log={log}')


def verified_merged(path):
    return path.is_dir() and (path/'config.json').is_file() and any(path.glob('*.safetensors')) and any((path/x).is_file() for x in ('tokenizer.json','tokenizer_config.json'))


def read_saved(stepdir):
    manifest=stepdir/'verified_result.json'
    if not manifest.is_file():return None
    obj=json.loads(manifest.read_text(encoding='utf-8'))
    m,_,_=parse_metrics(stepdir/'frozen_evaluator_output')
    assert m==obj['counts'], 'SAVED_METRICS_CHANGED'
    return obj


def do_run(args):
    assert not args.steps or all(s in STEPS for s in args.steps)
    ensure_idle()
    expected_eval_sha=check_contract()
    BASE.mkdir(parents=True,exist_ok=True)
    print('EVALUATOR_SHA256',expected_eval_sha)
    wanted=args.steps or STEPS
    for step in wanted:
        stepdir=BASE/f'step_{step}'
        saved=read_saved(stepdir) if stepdir.exists() else None
        if saved:
            assert saved['evaluator_sha256']==expected_eval_sha,'EVALUATOR_CHANGED_SINCE_PREVIOUS_STEP'
            print('SKIP_VERIFIED_EVAL',step,saved['counts'],flush=True);continue
        if stepdir.exists():
            out=stepdir/'frozen_evaluator_output'
            log=stepdir/'evaluator.log'
            if not (out.is_dir() and log.is_file() and 'EXIT_CODE=0' in log.read_text(encoding='utf-8',errors='replace')):
                raise RuntimeError(f'INCOMPLETE_EVALUATION_DIRECTORY: {stepdir}. Inspect; never overwrite.')
            print('REUSE_COMPLETED_EVALUATOR_OUTPUT',step)
        merged=RUN/f'evaluation/merged_step_{step}'
        if not merged.exists():
            merge_log=BASE/f'merge_{step}.log'
            run_checked(['bash',MERGER,step],merge_log)
        assert verified_merged(merged),f'MERGED_HF_UNVERIFIED {merged}'
        assert sha(EVAL)==expected_eval_sha,'FROZEN_EVALUATOR_MODIFIED_MID_RUN'
        if not stepdir.exists():
            stepdir.mkdir()
            out=stepdir/'frozen_evaluator_output'
            log=stepdir/'evaluator.log'
            # Preserve the frozen evaluator's prompt, tokenizer, decoding, and metric logic.
            env=os.environ.copy();env.setdefault('TOKENIZERS_PARALLELISM','false')
            run_checked([PYEVAL,EVAL,'eval','--arm','SFT_HIST','--model',merged,'--out',out],log,env=env)
        out=stepdir/'frozen_evaluator_output'
        assert out.is_dir(),f'EVALUATOR_OUTPUT_DIRECTORY_MISSING: {out}'
        counts,metric_files,inspected=parse_metrics(out)
        record={'step':step,'counts':counts,'accuracy':{'overall':counts['overall']/1000,'UC':counts['UC']/500,'UW':counts['UW']/500},
                'evaluator_sha256':expected_eval_sha,'evaluator':str(EVAL), 'model':str(merged),
                'checkpoint':str(RUN/f'checkpoints/global_step_{step}'), 'metric_evidence':metric_files,
                'verified_at_utc':datetime.now(timezone.utc).isoformat(),
                'contract':'frozen Chemistry1000; student direct prompt; thinking off; greedy; max_new_tokens=512; casefold canonical substring'}
        (stepdir/'verified_result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print('FROZEN_EVAL_VERIFIED',step,counts,flush=True)
    print('RUN_COMPLETE_REQUESTED_STEPS',wanted)
    # Emit a single TensorBoard series from all seven independently verified checkpoint evaluations.
    if all((BASE/f'step_{s}/verified_result.json').is_file() for s in STEPS):
        if not (BASE/'EVAL_TENSORBOARD_PASS').exists(): do_tb(args)
        else:print('EVAL_TENSORBOARD_ALREADY_WRITTEN',BASE/'tensorboard_eval')
    else:
        print('TB_WAITING_FOR_ALL_SEVEN_STEPS; partial outputs remain auditable')


def do_tb(args):
    records=[]
    for step in STEPS:
        path=BASE/f'step_{step}/verified_result.json'
        assert path.is_file(),f'MISSING_EVAL {step}'
        obj=read_saved(path.parent)
        assert obj['step']==step
        assert obj['evaluator_sha256']==sha(EVAL),f'EVALUATOR_CHANGED {step}'
        counts=obj['counts']
        assert counts['UC']+counts['UW']==counts['overall']
        records.append(obj)
    tb=BASE/'tensorboard_eval'
    assert not tb.exists(),f'TB_ALREADY_EXISTS; refusing to duplicate events {tb}'
    # Write an independent evaluation run; NEVER inject synthetic events into native train/tensorboard.
    from torch.utils.tensorboard import SummaryWriter
    tb.mkdir()
    writer=SummaryWriter(str(tb))
    try:
        for r in records:
            s=r['step']; acc=r['accuracy']; c=r['counts']
            for tag,value in (('eval/strict_accuracy',acc['overall']),('eval/uc_accuracy',acc['UC']),
                              ('eval/uw_accuracy',acc['UW']),('eval/strict_hits',c['overall']),
                              ('eval/uc_hits',c['UC']),('eval/uw_hits',c['UW'])):
                writer.add_scalar(tag,value,global_step=s)
        writer.flush()
    finally:writer.close()
    from tensorboard.backend.event_processing import event_accumulator
    ea=event_accumulator.EventAccumulator(str(tb),size_guidance={'scalars':0});ea.Reload()
    for tag in ('eval/strict_accuracy','eval/uc_accuracy','eval/uw_accuracy'):
        steps=[int(e.step) for e in ea.Scalars(tag)]
        assert steps==list(STEPS),f'TB_STEP_COVERAGE_MISMATCH {tag} {steps}'
    report=BASE/'curve_summary.json'
    report.write_text(json.dumps({'metric_type':'frozen_evaluator_accuracy_fraction','results':records},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (BASE/'curve_summary.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['step','overall_hits','overall_n','overall_accuracy','UC_hits','UC_n','UC_accuracy','UW_hits','UW_n','UW_accuracy'])
        for r in records:
            c=r['counts'];a=r['accuracy'];w.writerow([r['step'],c['overall'],1000,a['overall'],c['UC'],500,a['UC'],c['UW'],500,a['UW']])
    (BASE/'EVAL_TENSORBOARD_PASS').write_text('seven frozen Chemistry1000 evaluations checked; events are derived only from verified evaluator output\n',encoding='utf-8')
    print('SEVEN_STEP_EVAL_TENSORBOARD_PASS',tb)
    for r in records:print('EVAL',r['step'],r['counts'],r['accuracy'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['run','tensorboard','inspect'])
    p.add_argument('--steps',type=int,nargs='*',choices=STEPS)
    args=p.parse_args()
    if args.command=='run':do_run(args)
    elif args.command=='tensorboard':do_tb(args)
    else:
        print('EVAL',EVAL, 'sha256',sha(EVAL) if EVAL.exists() else 'MISSING')
        print('MERGER',MERGER)
        for st in STEPS:
            merged=RUN/f'evaluation/merged_step_{st}'
            out=BASE/f'step_{st}'
            print(st,'ckpt',(RUN/f'checkpoints/global_step_{st}').exists(),
                  'merged',verified_merged(merged),'evaluation',out.exists(),
                  'verified',(out/'verified_result.json').exists())
        print('INDEPENDENT_TB',BASE/'tensorboard_eval')

if __name__=='__main__':main()
