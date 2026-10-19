# Frozen Chemistry1000 Clean-SFT seven-checkpoint evaluation

This package **does not replace** `chemistry1000_native_parity_eval_v1.py` and **does not perform training**. It invokes the previously verified frozen evaluator with `eval --arm SFT_HIST --model <absolute HF merge> --out <unique directory>`, preserving the same dataset, student-only direct prompt, Qwen3 `enable_thinking=False`, greedy decoding, `max_new_tokens=512`, and canonical English casefold-substring criterion. It checks the formal data hash and seven native FSDP checkpoint directories. Each checkpoint is processed serially; merged Step160 is reused.

Server installation after Windows `scp` of this ZIP:

```bash
source /workspace/mtpatcher/project_env.sh
ROOT=/workspace/mtpatcher
PROJECT=$ROOT/repo/MT-Patcher-Reproduction-Ascend
PY=$ROOT/envs/mtpatcher-npu-py311/bin/python
"$PY" -m zipfile -e "$ROOT/chem_sft_eval_v1_20261017.zip" "$PROJECT"
EVALSCRIPT=$PROJECT/scripts/targeted/chem_sft_eval_v1/chem_sft_eval_v1.py
"$PY" "$EVALSCRIPT" inspect
```

First run one actual evaluation, which provides a strong interface test before 6000 more generations. It uses the **real** evaluator and no fabricated values:

```bash
"$PY" "$EVALSCRIPT" run --steps 160
```

If output schema parsing succeeds, it prints `FROZEN_EVAL_VERIFIED 160 ...` with the actual hit counts. Then run all remaining six (or all seven, as previously verified Step160 is skipped):

```bash
"$PY" "$EVALSCRIPT" run
```

Actual outputs remain separate under `/workspace/mtpatcher/runs/formal/chem_clean_sft_matched_v2/evaluation/chemistry1000_sft_curve_v1/step_<N>/frozen_evaluator_output/`. Source/model/provenance and verified counts are stored under each step. Logs are raw and unfiltered in `evaluator.log` and `merge_<N>.log`. If the evaluator output schema differs from known forms, the script **fails safely after the first step** and preserves the full evaluation output for inspection; it never guesses accuracy or writes a fake curve. Once the parser is adapted, repeating the run reuses previously completed evaluator outputs without re-generating translations.

After **all seven** complete, the script writes new and separate TensorBoard under `.../chemistry1000_sft_curve_v1/tensorboard_eval/`, with exact checkpoint step axis, tags `eval/strict_accuracy`, `eval/uc_accuracy`, and `eval/uw_accuracy` (fractions, 0–1) and separate hit count tags. Also generates `curve_summary.csv` and `curve_summary.json`, and re-reads native TensorBoard event files to assert seven-step coverage. The original `RUN/tensorboard/` training events remain untouched. It refuses duplicate event generation on reruns.

If the frozen evaluator fails, inspect the log but **do not rerun blindly**:

```bash
RUN=/workspace/mtpatcher/runs/formal/chem_clean_sft_matched_v2
BASE=$RUN/evaluation/chemistry1000_sft_curve_v1
tail -90 "$BASE/step_160/evaluator.log"
find "$BASE/step_160/frozen_evaluator_output" -maxdepth 3 -type f -print
```

Do not assert percentages from this package prior to seven actual evaluator executions. `OPD` source-comparable points exist at Step640/800/960; compare primarily at Step960, with separate UC500/UW500 and case-level paired differences. Never invent OPD values at untested steps.
