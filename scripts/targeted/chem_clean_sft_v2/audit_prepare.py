#!/usr/bin/env python3
"""Read-only frozen-input audit, then optional reuse of Stage1's existing build.
No Teacher model generation is ever performed here.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path('/workspace/mtpatcher')
RAW = ROOT / 'data/chem_clean_sft_matched_v1/teacher_targets.jsonl'
SOURCE = ROOT / 'data/chem_onpolicy_verl_v1_20261016/chem_onpolicy_full5500.parquet'
BUILT = ROOT / 'data/chem_clean_sft_matched_v1/chem_clean_sft_5500.parquet'
STAGE1 = ROOT / 'native_chem12h_v2/clean_sft_stage1/chem_clean_sft_data_v1.py'
EXPECTED_RAW_SHA = 'cf1d23edad79ba23c90bbb00d00d9011207955b56da3803e49165372c8096615'
EXPECTED_SOURCE_SHA = '489db3fae79bfde3ba7a59ba43ec177eedf4502e928212d759f2d9dd75f81caf'

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def audit_raw():
    import pyarrow.parquet as pq
    for p in (RAW, SOURCE):
        if not p.is_file():
            raise RuntimeError(f'MISSING {p}')
    assert digest(RAW) == EXPECTED_RAW_SHA, 'RAW_SHA_DRIFT'
    assert digest(SOURCE) == EXPECTED_SOURCE_SHA, 'SOURCE_SHA_DRIFT'
    records = [json.loads(x) for x in RAW.read_text(encoding='utf-8').splitlines() if x.strip()]
    assert len(records) == 5500, f'RECORD_COUNT_{len(records)}'
    counts = Counter(int(r['index']) for r in records)
    assert set(counts) == set(range(5500)) and max(counts.values()) == 1, 'INDEX_IDENTITY_FAIL'
    originals = pq.read_table(SOURCE, columns=['extra_info']).to_pylist()
    assert len(originals) == 5500
    by_id = {int(r['extra_info']['index']): r['extra_info'] for r in originals}
    assert len(by_id) == 5500
    assert all(r['source_text'] == by_id[int(r['index'])]['src_text'] and
               r['canonical_en_target'] == by_id[int(r['index'])]['en_name'] for r in records), 'SOURCE_OR_TERM_MISMATCH'
    assert all(r['teacher_translation'].strip() and not r.get('empty') and
               not r.get('ended_at_max_new_tokens') for r in records), 'BAD_TARGET_OR_TRUNCATION'
    strict = sum(bool(r['canonical_strict_hit']) for r in records)
    print('RAW_AUDIT_PASS rows=5500 strict_hit=%s/5500 sha256=%s' % (strict, EXPECTED_RAW_SHA), flush=True)
    return records, originals

def audit_tokens(records, originals):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen3-0.6B'), local_files_only=True)
    bad = []
    import pyarrow.parquet as pq
    src = pq.read_table(SOURCE, columns=['prompt', 'extra_info']).to_pylist()
    records_by_id = {int(r['index']): r for r in records}
    for row in src:
        i = int(row['extra_info']['index'])
        text = tok.apply_chat_template(row['prompt'], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        pids = tok(text, add_special_tokens=False)['input_ids']
        rids = tok(records_by_id[i]['teacher_translation'], add_special_tokens=False)['input_ids']
        if len(pids) + len(rids) + 1 > 1024 or not pids or not rids:
            bad.append([i, len(pids), len(rids)])
    assert not bad, f'STUDENT_PROMPT_RESPONSE_EOS_TOO_LONG examples={bad[:10]}'
    print('TOKEN_BOUND_PASS n=5500 max_combined<=1024 (independent; dataset loss mask still requires audit)', flush=True)

def audit_parquet():
    import pyarrow.parquet as pq
    assert BUILT.is_file(), f'BUILT_DATA_MISSING {BUILT}'
    pf = pq.ParquetFile(BUILT)
    assert pf.metadata.num_rows == 5500, f'PARQUET_ROWS_{pf.metadata.num_rows}'
    print('PARQUET_BUILT rows=5500 sha256=' + digest(BUILT), flush=True)
    print('PARQUET_SCHEMA', str(pf.schema_arrow), flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['audit','build','verify-built'])
    args = ap.parse_args()
    records, originals = audit_raw()
    if args.action == 'audit':
        audit_tokens(records, originals)
    elif args.action == 'build':
        audit_tokens(records, originals)
        if BUILT.exists():
            print('BUILD_ALREADY_EXISTS refusing rewrite:', BUILT)
        else:
            assert STAGE1.is_file(), f'NEED_EXISTING_STAGE1_BUILDER {STAGE1}'
            subprocess.run([sys.executable, str(STAGE1), 'build'], check=True)
        audit_parquet()
    else:
        audit_parquet()

if __name__ == '__main__':
    main()
