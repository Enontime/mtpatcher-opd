#!/usr/bin/env python3
"""Materialize FROZEN Chemistry 5500 context data for native verl; no data edits.

Writes separate smoke16 and full5500 parquets. Student prompt contains no
canonical English target; teacher side info is isolated in extra_info.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DIRECT_PROMPT=(
    'Translate the following text into English without additional explanations:'
    '\n\n{source}\n\n'
)
TEACHER_CHEM_TEMPLATE = '''You are translating a Chinese sentence into English.

Teacher-only lexical knowledge:
Chinese chemistry term: {src_term}
Canonical English registry term: {en_name}

Use the canonical English registry term when translating that designated
chemistry term. Translate the complete source sentence accurately and
naturally. Do not explain the instruction.

Chinese source:
{source}

Return only the English translation.'''
EXPECTED_CONTEXT_SHA='ca6532cacce64f24f14226a4dca509b494aaa5f5bbeecbc0f1d2f157c317044f'
BASE=Path('/workspace/mtpatcher')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):
            h.update(b)
    return h.hexdigest()


def sha_ids(ids):
    return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()


def token_ids(tok,text):
    s=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,
                              add_generation_prompt=True,enable_thinking=False)
    return [int(v) for v in tok(s,add_special_tokens=False)['input_ids']]


def build(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    source=args.source
    assert sha(source)==EXPECTED_CONTEXT_SHA, 'FROZEN_CHEMISTRY_SHA_MISMATCH: refuse dataset materialization'
    rows=[json.loads(line) for line in source.read_text(encoding='utf8').splitlines() if line.strip()]
    assert len(rows)==5500, f'Expected exactly 5500 rows, got {len(rows)}'
    assert hashlib.sha256(DIRECT_PROMPT.encode()).hexdigest()=='63101d0739ff2ee11b0e5d548b85dcd18be7a241777893326140f12778d9a8b8'
    tok=AutoTokenizer.from_pretrained(args.student)
    output=[]
    seen=set()
    max_student=0
    max_teacher=0
    for pos,r in enumerate(rows):
        assert r.get('domain')=='chemistry',f'Unexpected domain at {pos}'
        jid=int(r['job_id'])
        assert jid not in seen,f'Duplicate job_id {jid}'
        seen.add(jid)
        src=str(r['src_text'])
        term=str(r['src_term'])
        name=str(r['lexical_record']['en_name']).strip()
        assert name and term and src
        student_text=DIRECT_PROMPT.format(source=src)
        teacher_text=TEACHER_CHEM_TEMPLATE.format(src_term=term,en_name=name,source=src)
        sid=token_ids(tok,student_text)
        tid=token_ids(tok,teacher_text)
        assert len(sid)+256<=1024,(jid,'student sequence exceeds 1024',len(sid))
        assert len(tid)+256<=1024,(jid,'teacher sequence exceeds 1024',len(tid))
        max_student=max(max_student,len(sid))
        max_teacher=max(max_teacher,len(tid))
        output.append({
            'data_source':'chemistry',
            'prompt':[{'role':'user','content':student_text}],
            'ability':'translation',
            'reward_model':{'style':'rule','ground_truth':'UNUSED_ZERO_REWARD'},
            'extra_info':{
                'index':pos,'job_id':jid,'domain':'chemistry',
                'src_text':src,'src_term':term,'en_name':name,
                'student_hint_present':False,'teacher_hint_present':True,
                'student_prompt_sha256':sha_ids(sid),
                'teacher_prompt_sha256':sha_ids(tid),
            },
        })
    args.out.mkdir(parents=True,exist_ok=True)
    sizes={'smoke16':16,'full5500':5500}
    files={}
    for label,n in sizes.items():
        path=args.out/f'chem_onpolicy_{label}.parquet'
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite existing {path}')
        temp=path.with_suffix('.parquet.tmp')
        if temp.exists():
            raise FileExistsError(f'Temporary output already exists: {temp}')
        pq.write_table(pa.Table.from_pylist(output[:n]),temp)
        temp.replace(path)
        files[label]={'path':str(path),'sha256':sha(path),'rows':n}
    manifest={'status':'DATA_ONLY_NO_TRAINING_AUTHORIZATION',
              'source':str(source),'source_sha256':EXPECTED_CONTEXT_SHA,
              'student_model':str(args.student),'tokenizer_config_sha256':sha(args.student/'tokenizer_config.json'),
              'full_unique_rows':5500, 'max_student_prompt_tokens':max_student,
              'max_teacher_prompt_tokens':max_teacher,
              'student_prompt_sha256':hashlib.sha256(DIRECT_PROMPT.encode()).hexdigest(),
              'files':files,'seed':20260820,
              'note':'Native verl training data is an adaptation and does not guarantee old per-row stochastic RNG equivalence.'}
    (args.out/'data_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps(manifest,indent=2,ensure_ascii=False))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=BASE/'data/mtpatcher_v3_full6565_20260823/targeted_section43_contexts_qwen3_8b_final_20260910/chemistry_train.jsonl')
    p.add_argument('--student',type=Path,default=BASE/'models/Qwen3-0.6B')
    p.add_argument('--out',type=Path,default=BASE/'data/chem_onpolicy_verl_v1_20261016')
    build(p.parse_args())

if __name__=='__main__':
    main()
