# Chemistry Clean-SFT parity v1

> Superseded for formal knowledge-conditioned controls by
> `README_chemistry_kc_sft_parity_v2.md`. The v2 workflow requires an explicit
> `--dataset`, exact frozen OPD Teacher-template parity, strict review fields,
> and immutable training/evaluation contracts.

This workflow creates new matched SFT controls without modifying historical C2,
A1, or OPD artifacts.

Entry point:

    scripts/targeted/clean_sft_parity_v1.py

New artifacts are restricted to:

    data/mtpatcher_v3_full6565_20260823/targeted_clean_sft_targets_qwen3_8b_v1_20261015/
    runs/targeted/chemistry_clean_sft_parity_20261015/

## 1. Read-only checks

    source /workspace/mtpatcher/project_env.sh
    cd /workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend

    python scripts/targeted/clean_sft_parity_v1.py preflight \
      --json-out /workspace/mtpatcher/runs/targeted/chemistry_clean_sft_parity_20261015/preflight.json

    python scripts/targeted/clean_sft_parity_v1.py materialize --audit-only

The first command must print SOURCE_IDENTITY_PASS and CONFIG_PARITY_PASS. The
audit command writes only new audit artifacts; it does not load a model.

## 2. Materialize and freeze clean targets

Materialization is intentionally two-stage:

    python scripts/targeted/clean_sft_parity_v1.py materialize --device npu:0

This selectively regenerates independently flagged historical targets, validates
all 5,500 rows, and writes a deterministic human-review template. It leaves the
dataset in CANDIDATE_AWAITING_MANUAL_REVIEW state.

After filling every sampled row with review_status: PASS and completing the
review fields:

    python scripts/targeted/clean_sft_parity_v1.py materialize --finalize \
      --manual-review-jsonl /path/to/reviewed_manual_review.jsonl

Training rejects Clean targets until the manifest is FROZEN and
target_quality_status is PASS.

## 3. Training order

Run one arm at a time and review its artifacts before continuing:

    python scripts/targeted/clean_sft_parity_v1.py train \
      --arm SFT_OLDTARGET_MATCHED_LR2E5 --device npu:0

    python scripts/targeted/clean_sft_parity_v1.py train \
      --arm SFT_CLEAN_LR2E5 --device npu:0

    python scripts/targeted/clean_sft_parity_v1.py train \
      --arm SFT_CLEAN_LR3E5 --device npu:0

The trainer uses Python random.Random(seed + epoch - 1).shuffle, microbatch 4,
GA 4, and response-token CE sum. At each update boundary it divides every
accumulated gradient by the exact group response-token count before gradient
clipping and AdamW/scheduler steps. Per-update token counts, CE, gradient norm,
LR, and elapsed time are saved.

Each epoch is committed transactionally with model, optimizer, scheduler, RNG,
and completed-update state. Evaluation can target an epoch checkpoint by passing
its hf/ directory through --model.

## 4. Evaluation and paired comparison

    python scripts/targeted/clean_sft_parity_v1.py eval \
      --arm SFT_CLEAN_LR2E5 --device npu:0

    python scripts/targeted/clean_sft_parity_v1.py compare \
      --arm SFT_CLEAN_LR2E5

Evaluation preserves the frozen direct prompt, greedy decoding, 512 generated
tokens, diagnostic2000, and the existing sacreBLEU BLEU/chrF calls. The strict
canonical substring score remains primary. C.I. spacing-normalized, meta-free,
UC/UW, non-C.I., and paired GAIN/LOSS metrics are supplemental.

compare writes a blind semantic lexical review file and a separate key.
Semantic accuracy remains PENDING_BLIND_HUMAN_REVIEW unless a complete reviewed
file is supplied with --semantic-judgments.

Equal optimizer-step counts do not imply equal compute. The SFT manifest records
supervised response tokens and target-materialization provenance; OPD remains an
on-policy forward-KL method with different prefixes, Teacher calls, and response
tokens.

## 5. CPU-only verification

    python -m unittest scripts/targeted/test_clean_sft_parity_v1.py
    python -m py_compile \
      scripts/targeted/clean_sft_parity_v1.py \
      scripts/targeted/_clean_sft_parity_core_v1.py \
      scripts/targeted/clean_sft_parity_runtime_v1.py
