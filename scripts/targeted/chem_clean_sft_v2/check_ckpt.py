#!/usr/bin/env python3
"""Structural checkpoint audit and model-shard change evidence. Never mutates ckpt."""
import argparse
import hashlib
from pathlib import Path


def files_for(step_root):
    d=Path(step_root)
    assert d.is_dir(),f'CHECKPOINT_MISSING {d}'
    return [x for x in d.rglob('*') if x.is_file()]

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1<<20),b''):
            h.update(c)
    return h.hexdigest()

def checkpoint(base, step):
    matches=[p for p in Path(base).rglob(f'global_step_{step}') if p.is_dir()]
    assert len(matches)==1,f'EXPECTED_ONE_CHECKPOINT_FOR_{step}: {matches}'
    fs=files_for(matches[0]); assert fs
    model=[p for p in fs if 'model' in p.name.lower() and p.suffix in ('.pt','.bin','.safetensors','.distcp')]
    optimizer=[p for p in fs if any(s in p.name.lower() for s in ('optim','optimizer'))]
    extra=[p for p in fs if any(s in p.name.lower() for s in ('extra','state','loader','scheduler'))]
    print('CKPT',step,str(matches[0]),'files',len(fs),'model',len(model),'optimizer',len(optimizer),'extra',len(extra))
    assert model and optimizer, 'NO_MODEL_OR_OPTIMIZER_STATE'
    return matches[0],model

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('ckpt_root')
    ap.add_argument('--compare', nargs=2,type=int,default=None)
    args=ap.parse_args()
    if args.compare:
        p1,m1=checkpoint(args.ckpt_root,args.compare[0]); p2,m2=checkpoint(args.ckpt_root,args.compare[1])
        one={str(f.relative_to(p1)):sha(f) for f in m1}
        two={str(f.relative_to(p2)):sha(f) for f in m2}
        overlap=set(one)&set(two)
        assert overlap and any(one[k]!=two[k] for k in overlap),'MODEL_SHARDS_UNCHANGED_OR_NOT_COMPARABLE'
        print('MODEL_WEIGHTS_CHANGED_BETWEEN_CHECKPOINTS')
    else:
        checkpoint(args.ckpt_root,5)

if __name__=='__main__':main()
