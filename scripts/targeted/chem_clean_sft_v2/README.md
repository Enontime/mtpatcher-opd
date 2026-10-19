# Chemistry Native OPD vs Raw SeqKD — SFT v2

**Maintained source:** `/workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend/scripts/targeted/chem_clean_sft_v2/`. No Codex needed. No Teacher generation, no old-run deletion. This package is an implementation that still **must pass checks against the real server's Native Verl and Stage1 source**; it has not been run on your 8 Ascend NPUs in this chat.

## Frozen experimental contract

- C0 Qwen3-0.6B Student, Qwen3-8B Teacher; source identity: existing 5500 Chemistry sources. Teacher hints only in Teacher; Student sees the original no-hint Chinese-to-English prompt.
- Teacher raw complete translations: **5500 rows**, SHA256 `cf1d23edad79ba23c90bbb00d00d9011207955b56da3803e49165372c8096615`; do not modify/filter the 1114 canonical strict misses.
- Original OPD source Parquet SHA256 `489db3fae79bfde3ba7a59ba43ec177eedf4502e928212d759f2d9dd75f81caf`.
- Native Verl `/workspace/mtpatcher/repo/verl-v0.9.0`; `verl.trainer.sft_trainer`; native Tracking and checkpoints; 8 NPU data-parallel, BF16 full-FT (verify actual resolved config), global batch 16, 3 epochs, LR 3e-5, weight decay .01, cosine schedule, warmup ratio .03, scheduler planned steps=1032. Expected actual optimizer updates=1029; fail if observed geometry differs.
- **Primary comparison**: OPD960 vs SFT960. SFT1029 supplemental only if the corresponding OPD1029 checkpoint can really be evaluated. Do not select a checkpoint on the repeatedly inspected Chemistry1000 test set.
- The comparison is **matched source population/knowledge/optimizer budget**, not controlled token trajectory / pure KL vs CE comparison. Source order is *not claimed identical* without historical OPD per-step proof.

## Directory layout (mandatory)

```text
/workspace/mtpatcher/smoke/chem_clean_sft_v2/<UTC-run-id>/
    smoke80.parquet
    template_path.txt
    provenance.txt
    phaseA/{config.yaml,resolved_config.yaml,train.log,tensorboard/,checkpoints/,dataset_probe.log}
    phaseB/{config.yaml,resolved_config.yaml,train.log,tensorboard/,checkpoints/,dataset_probe.log}
    SMOKE_A_PASS
    SMOKE_V2_PASS

/workspace/mtpatcher/runs/formal/chem_clean_sft_matched_v2/
    config.yaml  resolved_config.yaml  train.log  provenance.txt
    tensorboard/  checkpoints/  evaluation/
```

No smoke/debug/TensorBoard event under `runs/formal`, and no smoke in `paper_view`.

## Installation

Windows PowerShell:

```powershell
scp .\chem_clean_sft_v2_20261017.zip mtpatcher-ascend:/workspace/mtpatcher/
ssh mtpatcher-ascend
```

Remote Bash:

```bash
source /workspace/mtpatcher/project_env.sh
ROOT=/workspace/mtpatcher
PROJECT=$ROOT/repo/MT-Patcher-Reproduction-Ascend
PY=$ROOT/envs/mtpatcher-npu-py311/bin/python
"$PY" -m zipfile -e "$ROOT/chem_clean_sft_v2_20261017.zip" "$PROJECT"
SCRIPT=$PROJECT/scripts/targeted/chem_clean_sft_v2/run.sh
```

This creates new versioned source files only (not data, smoke or formal until later commands). Before installing, confirm destination `scripts/targeted/chem_clean_sft_v2/` does not exist or review version collision; the extraction command does not perform a Git commit.

## Execution order

**1. Reuse Stage1 builder, audit frozen sources, but never regenerate Teacher targets.**

```bash
bash "$SCRIPT" data
```

The build command reuses `chem_clean_sft_data_v1.py build` and checks immutable input hashes, identities, independent prompt+response+EOS lengths, and the 5500-row Parquet. The exact Stage1 row encoding and native response-only loss still need the *actual Dataset probe* performed before smoke. Existing Parquet is not overwritten.

**2. Get the real Stage1-generated YAML configuration template.** This archive intentionally does **not** invent missing Stage1 source code or Hydra keys. The user's previous Stage1 has `config-formal` and `config-smoke` operations; inspect its actual help / output before running:

```bash
"$PY" "$ROOT/native_chem12h_v2/clean_sft_stage1/chem_clean_sft_setup_v1.py" --help
"$PY" "$ROOT/native_chem12h_v2/clean_sft_stage1/chem_clean_sft_setup_v1.py" config-formal --help
```

Use `config-formal` as implemented by that local script and locate its emitted YAML **containing an `overrides:` list**. Set `TEMPLATE=/absolute/path/to/generated_stage1_formal.yaml`. This v2 then rewrites/fails on all key scientific values, always changes smoke paths away from the old Stage1 `runs/formal/.../smoke` location, and composes the resolved config using the real Native Verl tree. If template CLI isn't self-describing, provide the original `chem_clean_sft_setup_v1.py` or the emitted YAML for exact integration rather than guessing values.

**3. Review and explicitly apply one narrow checkpoint fix.**

```bash
bash "$SCRIPT" patch-inspect
bash "$SCRIPT" patch-apply
```

`patch-apply` is **opt-in**; it refuses a preexisting diff in Native `sft_trainer.py`, prints the Git diff, creates a backup, and modifies only the end-of-last-epoch checkpoint predicate. LR horizon remains 1032, but a 1029-update job now saves its true terminal checkpoint. The smoke-A epoch-end checkpoint at step5 tests this exact behavior.

**4. Eight-NPU smoke: 5 optimizer updates and resume one additional update.**

```bash
bash "$SCRIPT" smoke-a "$TEMPLATE"
# Copy the SMOKE_A_PASS run root printed above:
SMOKE=/workspace/mtpatcher/smoke/chem_clean_sft_v2/<ACTUAL-UTC-RUN-ID>
bash "$SCRIPT" smoke-b "$SMOKE"
```

- Smoke A uses 80 dataset rows, 1 epoch=5 global batches, scheduler horizon=6, save frequency=2; checkpoint `global_step_5` must be created **only by the terminal-step patch**.
- Native TensorBoard under smoke's `phaseA/tensorboard` must contain exactly 1..5 for `train/loss`, `train/lr`, `train/grad_norm`; both model and optimizer shard files must exist; model shards between checkpoints 2 and 5 must differ.
- Smoke B resumes from Step5 with `total_epochs=2`, `total_training_steps=6`; it must produce the native TensorBoard step6, not reinitialize at step1. Resume markers in original logs are required. **Optimizer/scheduler restoration is additionally to be inspected in the native checkpoint handler source/logs**; a directory existing does not constitute restoration proof.
- Check real raw logs (`tail -n 120 -F "$SMOKE/phaseA/train.log"`) — do not create a filtered watch that can hide tracebacks. Check event 1 as soon as the first optimizer update has logged:

```bash
PYVERL=/workspace/mtpatcher/envs/verl-v0.9.0-a3/bin/python
"$PYVERL" "$PROJECT/scripts/targeted/chem_clean_sft_v2/check_tb.py" "$SMOKE/phaseA/tensorboard" --last 1
```

The `--last 1` first-step check is best run in a **second SSH terminal while smoke-A is running**. It will fail if the TB event has not yet been flushed; retry after the actual optimizer step, never substitute a synthesized event.

**5. Formal is gated by the preceding smoke and an explicit human authorization token.**

```bash
bash "$SCRIPT" formal "$TEMPLATE" "$SMOKE" YES_START_FORMAL
```

This prepares a new formal root and launches the raw native Verl 8-NPU training detached. It **refuses to overwrite any existing formal root**. It never marks a just-started formal job as completed or passed. Show the original log continuously (no colored/filtered pretty watch):

```bash
tail -n 120 -F /workspace/mtpatcher/runs/formal/chem_clean_sft_matched_v2/train.log
```

Upon completion, check actual step1029 checkpoint and the real TensorBoard events 1..1029. Verify model, optimizer and loader restore state from native handler; record the actual LR series and sample order limitations. The run's `train/global_tokens` counts prompt+response, not supervised target tokens.

**6. Evaluation, after formal completion:** merge each SFT checkpoint with Native Verl's own merger:

```bash
bash "$PROJECT/scripts/targeted/chem_clean_sft_v2/merge_sft_step.sh" 640
bash "$PROJECT/scripts/targeted/chem_clean_sft_v2/merge_sft_step.sh" 800
bash "$PROJECT/scripts/targeted/chem_clean_sft_v2/merge_sft_step.sh" 960
```

For true parity, invoke the exact **existing** `/workspace/mtpatcher/native_chem12h_v2/chemistry1000_native_parity_eval_v1.py` interface with the merged model path; do not invent flags before reading its actual `--help` or script. Compare C0, OPD and SFT at the same steps using the same no-hint Student prompt, greedy decoding, 1000 frozen IDs, UC500/UW500 and canonical casefold-substring hit. Save 1000 case-level outputs per checkpoint and do paired comparisons; never choose the winning checkpoint based on test.

## Important unresolved validations (these are not silently considered PASS)

1. This chat has terminal excerpts, **not the actual Stage1 two Python files, `verl_mt_response_sft_dataset.py`, or Native Verl resolved YAML**. The package composes and probes the **real server files** and fails if an interface does not match. The exact `config-formal` invocation and the full `resume_mode` semantics can only be finalized against these source files.
2. Five updates + step6 continuation, train logs, TB, checkpoint structure are necessary but not sufficient to prove bit-identical optimizer/scheduler restoration. Check actual loader and optimizer state at resume.
3. Full teacher weight hash wasn't frozen in available conversation (only model path and teacher prompt token hashes); record current model fingerprint before claiming exact weight identity.
4. OPD and SFT optimizer steps may have same count without fully identical per-step source order. Both are currently source-population and budget matched only.
5. General WMT BLEU is secondary to Chemistry task knowledge-transfer, not the stopping criterion.

## Scope / what the package intentionally does NOT do

No Git reset, checkout, commit, push, broad `pkill`, PID1 zombie attach, TensorBoard event fabrication, automatic test-score tuning, new Teacher generation, live OPD v4 modification, or smoke assets under a formal directory. Formal is never launched merely by uploading this package.

**After formal has genuinely finished**, run `bash "$PROJECT/scripts/targeted/chem_clean_sft_v2/finish_formal.sh"`. It requires true native TensorBoard events at **every step 1..1029**, changed model shards and a terminal checkpoint. This does **not** substitute for Chemistry1000 evaluation or auditable optimizer resume.
