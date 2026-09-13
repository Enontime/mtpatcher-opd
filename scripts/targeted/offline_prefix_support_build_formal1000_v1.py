#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

MTP_ROOT = Path('/workspace/mtpatcher')
REPO = MTP_ROOT / 'repo/MT-Patcher-Reproduction-Ascend'
PREP = MTP_ROOT / 'runs/targeted/offline_prefix_support_preupdate_signal1000_v2_20260913'
PAIR = PREP / 'paired/chemistry_pairs_first1000.jsonl'
STUDENT_BANK = PREP / 'banks/chemistry/S.jsonl'
TEACHER_BANK = PREP / 'banks/chemistry/T.jsonl'
H5 = REPO / 'scripts/targeted/targeted_wa_opd_horizon5_o12_v2.py'

STUDENT = MTP_ROOT / 'models/Qwen3-0.6B'
TEACHER = MTP_ROOT / 'models/Qwen3-8B'

ROWS = 1000
BATCH_SIZE = 40
BATCHES_PER_PASS = 25
PASSES = 3
TOTAL_UPDATES = 75

EXPECTED_H5_SHA = '38ac62c0a93f1b762b7536db60421dd8c1e7aca49a57c25851306c3cb76b6e16'
EXPECTED_PAIR_SHA = 'a7a78e0556933b642e21ac7b6f57392aebfb50077aff47850c78279dc448fe88'
EXPECTED_ROW_ORDER_SHA = '49763f7a369c48e87befb1417927753f5e3c02d4ec81748540c6b0ca50bb95da'
EXPECTED_ROW_SET_SHA = '92011d1090c65a99e1472b61e81a5ad5d5f3753e37c93d6538fc01f92eb23c8c'
EXPECTED_STUDENT_BANK_SHA = 'e65180ae17971184fc84428a05dd42378aad68e9f0cf70b28b0f60062dee4e1c'
EXPECTED_TEACHER_BANK_SHA = '297a504770abd9bd2e6cad8c986aa83bf6ffad3978c8d7d1207e40d591f06a74'
EXPECTED_ARM_SHA = {
    'S': '2e37662fb949a43004d9842eb68166c751b2fd97273e6dc408d4b911a7a12308',
    'T': 'd68bff648787e93da34b550bb78b4e552c0c7bdff8f7b5db7138d36f215a87d1',
}

DIRECT_PROMPT = (
    'Translate the following text into English without additional explanations:'
    '\n\n{source}\n\n'
)
DIRECT_PROMPT_SHA256 = '63101d0739ff2ee11b0e5d548b85dcd18be7a241777893326140f12778d9a8b8'

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


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    raw = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(f'{path}:{ln}: {exc}') from exc
    return rows


def render_prompt_ids(tok, user_text: str) -> list[int]:
    rendered = tok.apply_chat_template(
        [{'role': 'user', 'content': user_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return [int(x) for x in tok(rendered, add_special_tokens=False)['input_ids']]


def write_parquet_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    pq.write_table(pa.Table.from_pylist(rows), tmp)
    os.replace(tmp, path)


def build_arm(
    *,
    arm: str,
    pairs: list[dict[str, Any]],
    student_tok,
    teacher_tok,
    run: Path,
) -> dict[str, Any]:
    output_rows: list[dict[str, Any]] = []
    response_lists: list[list[int]] = []
    row_ids: list[str] = []

    for index, pair in enumerate(pairs):
        require(pair['status'] == 'PASS', f'pair status fail index={index}')
        require(pair['domain'] == 'chemistry', f'domain drift index={index}')
        require(pair['row_id'] == f"chemistry:{pair['job_id']}", f'row_id drift index={index}')

        row_id = str(pair['row_id'])
        src_text = str(pair['src_text'])
        src_term = str(pair['src_term'])
        knowledge = str(pair['knowledge'])
        m_i = int(pair['m_i'])

        response_ids = [int(x) for x in pair[arm]['used_prefix_ids']]
        response_sha = str(pair[arm]['used_prefix_sha256'])
        require(len(response_ids) == m_i, f'{row_id} m_i mismatch')
        require(sha256_json(response_ids) == response_sha, f'{row_id} response SHA mismatch')

        student_user = DIRECT_PROMPT.format(source=src_text)
        teacher_user = TEACHER_CHEM_TEMPLATE.format(
            src_term=src_term,
            en_name=knowledge,
            source=src_text,
        )
        student_prompt_ids = render_prompt_ids(student_tok, student_user)
        teacher_prompt_ids = render_prompt_ids(teacher_tok, teacher_user)

        require(knowledge in teacher_user, f'{row_id} Teacher hint absent')
        require(student_user == DIRECT_PROMPT.format(source=src_text), f'{row_id} Student prompt drift')

        extra_info = {
            'index': index,
            'row_id': row_id,
            'job_id': int(pair['job_id']),
            'domain': 'chemistry',
            'arm': arm,
            'src_text': src_text,
            'src_term': src_term,
            'knowledge': knowledge,
            'm_i': m_i,
            'frozen_response_ids': response_ids,
            'frozen_response_sha256': response_sha,
            'student_user_text': student_user,
            'teacher_user_text': teacher_user,
            'student_prompt_ids': student_prompt_ids,
            'teacher_prompt_ids': teacher_prompt_ids,
            'student_prompt_sha256': sha256_json(student_prompt_ids),
            'teacher_prompt_sha256': sha256_json(teacher_prompt_ids),
            'teacher_hint_present': True,
            'student_hint_present': False,
            'chemistry_seed1_authorized': True,
            'scientific_class': 'DIAGNOSTIC ONLY / OFFLINE PREFIX-SUPPORT REPLAY',
        }

        output_rows.append(
            {
                'data_source': 'default',
                'prompt': [{'role': 'user', 'content': student_user}],
                'ability': 'translation',
                'reward_model': {'style': 'rule', 'ground_truth': 'UNUSED_ZERO_REWARD'},
                'extra_info': extra_info,
            }
        )
        response_lists.append(response_ids)
        row_ids.append(row_id)

    require(len(output_rows) == ROWS, f'{arm} rows={len(output_rows)}')
    require(sha256_json(row_ids) == EXPECTED_ROW_ORDER_SHA, f'{arm} row-order SHA drift')

    aggregate_sha = sha256_json(response_lists)
    require(aggregate_sha == EXPECTED_ARM_SHA[arm], f'{arm} aggregate response SHA mismatch got={aggregate_sha}')

    batches: list[dict[str, Any]] = []
    reconstructed: list[list[int]] = []
    for batch_idx, begin in enumerate(range(0, ROWS, BATCH_SIZE)):
        end = begin + BATCH_SIZE
        ids_batch = response_lists[begin:end]
        rows_batch = row_ids[begin:end]
        require(len(ids_batch) == BATCH_SIZE, f'{arm} partial batch at {batch_idx}')
        reconstructed.extend(ids_batch)
        batches.append(
            {
                'batch_index_zero_based': batch_idx,
                'row_start': begin,
                'row_end_exclusive': end,
                'rows': BATCH_SIZE,
                'row_ids': rows_batch,
                'row_ids_sha256': sha256_json(rows_batch),
                'response_ids_sha256': sha256_json(ids_batch),
                'effective_supervised_tokens': sum(len(x) for x in ids_batch),
            }
        )
    require(len(batches) == BATCHES_PER_PASS, f'{arm} batch count drift')
    require(sha256_json(reconstructed) == aggregate_sha, f'{arm} batch reconstruction SHA drift')

    parquet = run / 'inputs' / f'chemistry_{arm}_full1000.parquet'
    write_parquet_atomic(parquet, output_rows)
    back = pq.read_table(parquet).to_pylist()
    require(len(back) == ROWS, f'{arm} parquet row count after readback')
    back_ids = [[int(x) for x in r['extra_info']['frozen_response_ids']] for r in back]
    back_row_ids = [str(r['extra_info']['row_id']) for r in back]
    require(sha256_json(back_ids) == aggregate_sha, f'{arm} parquet response IDs changed')
    require(sha256_json(back_row_ids) == EXPECTED_ROW_ORDER_SHA, f'{arm} parquet row order changed')

    return {
        'arm': arm,
        'rows': ROWS,
        'parquet': str(parquet),
        'parquet_sha256': sha256_file(parquet),
        'aggregate_response_ids_sha256': aggregate_sha,
        'row_order_sha256': sha256_json(row_ids),
        'row_set_sha256': EXPECTED_ROW_SET_SHA,
        'effective_supervised_tokens_per_pass': sum(len(x) for x in response_lists),
        'batches_per_pass': BATCHES_PER_PASS,
        'batch_size': BATCH_SIZE,
        'batches': batches,
        'student_hint_present': False,
        'teacher_hint_present': True,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    args = ap.parse_args()
    run = Path(args.run)

    require(ROWS % BATCH_SIZE == 0, 'formal batch must divide 1000 exactly')
    require(BATCHES_PER_PASS * BATCH_SIZE == ROWS, 'batch geometry drift')
    require(PASSES * BATCHES_PER_PASS == TOTAL_UPDATES, 'update geometry drift')

    for path in (PAIR, STUDENT_BANK, TEACHER_BANK, H5):
        require(path.is_file(), f'missing frozen source: {path}')
    require(sha256_file(PAIR) == EXPECTED_PAIR_SHA, 'chemistry pair file SHA drift')
    require(sha256_file(STUDENT_BANK) == EXPECTED_STUDENT_BANK_SHA, 'Student bank SHA drift')
    require(sha256_file(TEACHER_BANK) == EXPECTED_TEACHER_BANK_SHA, 'Teacher bank SHA drift')
    require(sha256_file(H5) == EXPECTED_H5_SHA, 'H5 source SHA drift')
    require(
        hashlib.sha256(DIRECT_PROMPT.encode('utf-8')).hexdigest() == DIRECT_PROMPT_SHA256,
        'direct prompt SHA drift',
    )

    pairs = read_jsonl(PAIR)
    require(len(pairs) == ROWS, f'chemistry pair rows={len(pairs)}')
    row_ids = [str(x['row_id']) for x in pairs]
    require(sha256_json(row_ids) == EXPECTED_ROW_ORDER_SHA, 'full1000 row-order SHA drift')

    s_lists = [[int(v) for v in x['S']['used_prefix_ids']] for x in pairs]
    t_lists = [[int(v) for v in x['T']['used_prefix_ids']] for x in pairs]
    require(sha256_json(s_lists) == EXPECTED_ARM_SHA['S'], 'full1000 matched S SHA drift')
    require(sha256_json(t_lists) == EXPECTED_ARM_SHA['T'], 'full1000 matched T SHA drift')

    student_tok = AutoTokenizer.from_pretrained(STUDENT, local_files_only=True)
    teacher_tok = AutoTokenizer.from_pretrained(TEACHER, local_files_only=True)
    require(student_tok.pad_token_id is not None, 'Student pad token missing')
    require(teacher_tok.pad_token_id is not None, 'Teacher pad token missing')
    require(student_tok.eos_token_id == teacher_tok.eos_token_id, 'Student/Teacher EOS mismatch')

    arms = {
        arm: build_arm(
            arm=arm,
            pairs=pairs,
            student_tok=student_tok,
            teacher_tok=teacher_tok,
            run=run,
        )
        for arm in ('S', 'T')
    }
    require(
        arms['S']['effective_supervised_tokens_per_pass']
        == arms['T']['effective_supervised_tokens_per_pass'],
        'matched token budget diverged between S/T',
    )

    manifest = {
        'status': 'PASS_FORMAL_CHEMISTRY_INPUTS_FROZEN',
        'scientific_class': 'DIAGNOSTIC ONLY / OFFLINE PREFIX-SUPPORT REPLAY',
        'authorization_scope': 'Chemistry S/T seed1 under frozen prereg v2 only',
        'chemistry_seed1_authorized': True,
        'parameter_updates_authorized': True,
        'source_pair_file': str(PAIR),
        'source_pair_sha256': EXPECTED_PAIR_SHA,
        'row_order_sha256': EXPECTED_ROW_ORDER_SHA,
        'row_set_sha256': EXPECTED_ROW_SET_SHA,
        'student_prefix_bank_sha256': EXPECTED_STUDENT_BANK_SHA,
        'teacher_prefix_bank_sha256': EXPECTED_TEACHER_BANK_SHA,
        'matched_response_ids_sha256': EXPECTED_ARM_SHA,
        'direct_prompt_sha256': DIRECT_PROMPT_SHA256,
        'h5_source_sha256': EXPECTED_H5_SHA,
        'rows': ROWS,
        'domain': 'chemistry',
        'batch_size': BATCH_SIZE,
        'batches_per_pass': BATCHES_PER_PASS,
        'passes': PASSES,
        'total_optimizer_updates_per_arm': TOTAL_UPDATES,
        'effective_supervised_tokens_per_pass': arms['S']['effective_supervised_tokens_per_pass'],
        'effective_supervised_tokens_total_per_arm': (
            PASSES * arms['S']['effective_supervised_tokens_per_pass']
        ),
        'arms': arms,
    }
    atomic_json(run / 'input_manifest.json', manifest)
    print('FORMAL1000_INPUT_BUILD=PASS')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
