# Chemistry knowledge-conditioned SFT parity v2

This workflow preserves the already matched 5,500-source, 3-epoch,
batch-4 × GA-4, 1,032-update, AdamW/BF16, seed, warmup, scheduler, and
LR-2e-5/LR-3e-5 contracts. Historical C2, A1, OPD, diagnostics, and the source
lexicon remain read-only.

## Scientific arms and freeze contracts

- `SFT_OLDTARGET_MATCHED_LR2E5` is the historical-target trainer control.
- `KC_SFT_MATCHED_PROMPT_LR2E5/3E5` is the one-shot raw-distribution control.
  All 5,500 outputs use the byte-identical frozen OPD Chemistry Teacher prompt.
  Canonical-term misses, meta-discourse, CJK residue, and other content-quality
  flags are measured but never used to retry, select, replace, or block a
  technically trainable freeze.
- `KC_SFT_SELECTIVE_CLEAN_LR2E5/3E5` is a mixed-source quality control.
  Independently flagged historical targets may use the fixed prompt-variant
  sequence. Freezing requires zero technical anomalies, zero content-quality
  failures, and strict sampled manual-review PASS.

MatchedPrompt never receives an `ALL_TARGETS_SEMANTICALLY_CORRECT` claim. Its
frozen manifest uses:

```text
technical_validity_status: PASS
target_quality_status: RAW_DISTRIBUTION_AUDITED
content_quality_gate: NOT_APPLICABLE_RAW_DISTRIBUTION
all_targets_semantically_correct: false
```

SelectiveClean instead requires:

```text
technical_validity_status: PASS
target_quality_status: PASS
content_quality_gate: STRICT_PASS
```

For both datasets, source/row identity errors hard-block before Teacher
inference. Empty output, generation exception, and training-length overflow are
recorded once as technical anomalies and block a trainable freeze. They are not
silently skipped. The candidate and frozen manifests record every anomaly count
and bind the complete trainable target-id list by SHA256.

OPD still evaluates Teacher logits on Student-generated prefixes. Offline SFT
generates a complete target and uses teacher-forced response-token CE. Prompt
and lexical-knowledge parity does not imply identical objective, trajectory,
target quality, token count, or compute.

## Read-only gates

```bash
source /workspace/mtpatcher/project_env.sh
cd /workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend

python scripts/targeted/clean_sft_parity_v1.py preflight \
  --json-out /workspace/mtpatcher/runs/targeted/chemistry_clean_sft_parity_20261015/preflight_v2.json
```

The output must include:

```text
SOURCE_IDENTITY_PASS
TEACHER_CONDITION_PARITY_PASS
CONFIG_PARITY_PASS
```

Audit-only performs no writes and does not create or overwrite
`generation_contract.json`, `candidate_manifest.json`, candidate data, review
templates, or generation history:

```bash
python scripts/targeted/clean_sft_parity_v1.py materialize \
  --dataset matched_prompt --audit-only

python scripts/targeted/clean_sft_parity_v1.py materialize \
  --dataset selective_clean --audit-only
```

## Generation contract and recovery

Before the first Teacher forward, `generation_contract.json` binds:

- context and historical-target SHA256;
- dataset/artifact version;
- Teacher model files and tokenizer identity;
- every Teacher prompt SHA256;
- `enable_thinking`, sampling, maximum generated tokens, and attempt limit;
- the fixed technical-failure and content-retry policies.

Every generation-history row carries this contract SHA. A changed prompt,
Teacher identity, decoding setting, data hash, or data version blocks recovery.
MatchedPrompt permits exactly one recorded semantic attempt per target. A
normal restart reuses completed target IDs and never generates them again.

The commands below are intentionally documented but must only run after manual
approval:

```bash
python scripts/targeted/clean_sft_parity_v1.py materialize \
  --dataset matched_prompt --device npu:0

python scripts/targeted/clean_sft_parity_v1.py materialize \
  --dataset selective_clean --device npu:0
```

MatchedPrompt can freeze without a manual-review submission because review is
observational, not a selection gate:

```bash
python scripts/targeted/clean_sft_parity_v1.py materialize \
  --dataset matched_prompt --finalize
```

If a MatchedPrompt review is supplied, its row identity and boolean completeness
are validated, but PASS/FAIL outcomes are reported rather than used to filter
the dataset. SelectiveClean requires every frozen sample row to have exact
source/target identity and all strict fields set to PASS/true:

```bash
python scripts/targeted/clean_sft_parity_v1.py materialize \
  --dataset selective_clean --finalize \
  --manual-review-jsonl /path/to/reviewed_selective_clean.jsonl
```

## Training and evaluation isolation

Training binds arm, training SHA, freeze status, base model, optimizer, LR,
weight decay, scheduler, warmup, seed, shuffle, and loss reduction in
`run_contract.json`. The Student forward remains BF16; response CE sum is
computed from FP32 logits, accumulated over the GA group, divided by the exact
group valid-response-token count, then clipped before AdamW/scheduler steps.

Checkpoint state explicitly records the local PyTorch version and uses trusted
local `torch.load(..., weights_only=False)` with a compatibility fallback for
older PyTorch. Resume verifies pointer/meta/state identity, model checkpoint,
optimizer, scheduler, RNG, completed epoch/update counts, and the contiguous
update ledger.

Evaluation cache identity remains bound to model files, inputs, prompt,
decoding, and scoring version.

## CPU-only verification

```bash
python scripts/targeted/test_clean_sft_parity_v1.py
python scripts/targeted/test_clean_sft_parity_contracts_v1.py
python scripts/targeted/test_clean_sft_parity_v2_contracts.py

python -m py_compile \
  scripts/targeted/clean_sft_parity_v1.py \
  scripts/targeted/_clean_sft_parity_core_v1.py \
  scripts/targeted/_clean_sft_parity_kc_v1.py \
  scripts/targeted/_clean_sft_materialize_v2.py \
  scripts/targeted/_clean_sft_checkpoint_v2.py \
  scripts/targeted/_clean_sft_optimization_contract_v2.py \
  scripts/targeted/_clean_sft_train_contract_v1.py \
  scripts/targeted/clean_sft_parity_runtime_v1.py
```
