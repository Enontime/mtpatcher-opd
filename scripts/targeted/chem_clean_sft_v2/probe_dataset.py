#!/usr/bin/env python3
"""Exercise the *actual* Verl custom dataset and collator before NPU use."""
import argparse
import json
from pathlib import Path

ROOT=Path('/workspace/mtpatcher')
RAW=ROOT/'data/chem_clean_sft_matched_v1/teacher_targets.jsonl'
DATASET=ROOT/'repo/MT-Patcher-Reproduction-Ascend/scripts/data/verl_mt_response_sft_dataset.py'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--resolved',required=True)
    ap.add_argument('--data',required=True)
    args=ap.parse_args()
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl.utils.import_utils import load_extern_object
    from verl.trainer.sft_trainer import SFTTensorCollator
    cfg=OmegaConf.load(args.resolved)
    cls=load_extern_object(str(DATASET),'MTResponseOnlySFTDataset')
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen3-0.6B'),local_files_only=True)
    ds=cls(parquet_files=[str(args.data)],tokenizer=tok,config=cfg.data,processor=None,max_samples=-1)
    assert len(ds) in (80,5500),f'DATASET_LEN_{len(ds)}'
    collate=SFTTensorCollator(cfg.data.pad_mode)
    sample_a=ds[0]
    sample_b=ds[min(1,len(ds)-1)]
    print('DATASET_REAL_SAMPLE_KEYS',sorted(sample_a.keys()))
    # At the actual unpadded example boundary, the final token must be the
    # tokenizer EOS (<|im_end|> for this Qwen3 chat family). Explicitly fail
    # if the dataset does not expose input_ids for this check.
    ids = sample_a['input_ids']
    ids = ids.tolist() if hasattr(ids, 'tolist') else list(ids)
    if ids and isinstance(ids[0], list):
        assert len(ids)==1, 'UNEXPECTED_NESTED_EXAMPLE'
        ids=ids[0]
    assert ids and ids[-1] == tok.eos_token_id, f'FINAL_EOS_MISSING last={ids[-1] if ids else None} eos={tok.eos_token_id}'
    # Source prompt has no lexical hint, by construction of the frozen original.
    # Confirm the actual sample's loss mask does not cover its rendered prompt.
    import pyarrow.parquet as pq
    original=pq.read_table(ROOT/'data/chem_onpolicy_verl_v1_20261016/chem_onpolicy_full5500.parquet', columns=['prompt','extra_info']).to_pylist()[0]
    rendered=tok.apply_chat_template(original['prompt'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    prompt_ids=tok(rendered,add_special_tokens=False)['input_ids']
    assert ids[:len(prompt_ids)] == prompt_ids, 'DATASET_PROMPT_TOKEN_PREFIX_MISMATCH'
    assert len(ids)>len(prompt_ids)+1,'NO_RESPONSE_TOKENS_AFTER_PROMPT'
    if 'loss_mask' in sample_a:
        m=sample_a['loss_mask']
        m=m.tolist() if hasattr(m,'tolist') else list(m)
        assert len(m)==len(ids), 'LOSS_MASK_INPUT_LENGTH_MISMATCH'
        assert not any(m[:len(prompt_ids)]), 'MASK_SUPERVISES_PROMPT'
        assert all(m[len(prompt_ids):]), 'MASK_OMITS_RESPONSE_OR_EOS'
        print('RESPONSE_AND_EOS_BOUNDARY_VERIFIED', len(prompt_ids), len(ids), sum(m))
    elif 'labels' in sample_a:
        l=sample_a['labels']
        l=l.tolist() if hasattr(l,'tolist') else list(l)
        assert len(l)==len(ids), 'LABELS_INPUT_LENGTH_MISMATCH'
        assert all(x==-100 for x in l[:len(prompt_ids)]),'LABELS_SUPERVISE_PROMPT'
        assert all(x!=-100 for x in l[len(prompt_ids):]),'LABELS_OMIT_RESPONSE_OR_EOS'
        print('RESPONSE_LABELS_AND_EOS_BOUNDARY_VERIFIED')
    else:
        raise RuntimeError('NO_EXPLICIT_RESPONSE_ONLY_SUPERVISION_IN_DATASET_SAMPLE')
    batch=collate([sample_a,sample_b])
    print('COLLATOR_REAL_KEYS',sorted(batch.keys()))
    assert 'input_ids' in batch, 'MISSING_INPUT_IDS'
    # This class must label only Teacher response tokens plus EOS. Failing a
    # recognizable mask is safer than silently accepting full-prompt CE.
    if 'loss_mask' in batch:
        mask=batch['loss_mask']
        assert mask.numel()>0 and mask.sum().item()>0,'EMPTY_SUPERVISION_MASK'
        # Verify no supervision at the first prompt token. Native implementations
        # may shift response labels by one; all masks must be tested explicitly.
        # Native Verl returns a ragged NestedTensor from the collator.
        # Unbind batch dimension instead of indexing the ragged dimension.
        mask_rows = list(mask.unbind(0))
        assert len(mask_rows) == 2, 'COLLATOR_BATCH_SIZE_MISMATCH'
        for i, (row, sample) in enumerate(zip(mask_rows, (sample_a, sample_b))):
            actual = row.detach().cpu().tolist()
            expected = sample['loss_mask'].detach().cpu().tolist()
            assert actual == expected, f'COLLATOR_LOSS_MASK_CHANGED_SAMPLE_{i}'
            assert actual and actual[0] == 0, f'PROMPT_SUPERVISION_LEAK_SAMPLE_{i}'
            assert actual[-1] == 1, f'EOS_SUPERVISION_MISSING_SAMPLE_{i}'
        print('MASK_RESPONSE_ONLY_CANDIDATE  first_token=0  supervised_tokens=',mask.sum().item())
    elif 'labels' in batch:
        labels=batch['labels']
        assert (labels==-100).any() and (labels!=-100).any(),'LABELS_NOT_RESPONSE_ONLY'
        assert all(row[0].item() == -100 for row in labels.unbind(0)), 'PROMPT_SUPERVISION_LEAK'
        print('LABELS_RESPONSE_ONLY_CANDIDATE supervised_tokens=',(labels!=-100).sum().item())
    else:
        raise RuntimeError('NO_EXPLICIT_RESPONSE_MASK_OR_LABELS: inspect native dataset + collator before training')
    # Check mixed length collate runs and source prompt/target remain separated.
    print('REAL_VERL_DATASET_AND_COLLATOR_BASIC_PASS')
    print('NATIVE_LOSS_SHIFT_SEMANTICS_STILL_REQUIRE_CODE_REVIEW; probe passes prompt/response boundary and EOS')

if __name__=='__main__':
    main()
