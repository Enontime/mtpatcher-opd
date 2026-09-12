# Targeted WA / OPD Research Index

This directory is the human and machine-readable index for the
current targeted MT-PATCHER / OPD research line.

It intentionally does **not** move historical data, checkpoints,
or run directories. Paths remain stable for provenance.

## Scientific chain

| No. | Experiment | Class | Status | TensorBoard |
|---|---|---|---|---|
| 00 | C0 targeted baseline | Baseline | PASS | NONE |
| 01 | C1/C2/C3 SFT positive control | Positive control | PASS | NONE |
| 02 | O1/O2/O3 knowledge-conditioned OPD | Lab adaptation | PASS | NONE |
| 03 | Matched trainset audits | Diagnostic | PASS | NONE |
| 04 | A0 OPD horizon-5 | Ablation | PASS | NONE |
| 05 | A4 KL signal localization | Mechanism diagnostic | PASS | NONE |
| 06 | A4b semantic lexical audit | Evaluator diagnostic | PASS with caveat | NONE |
| 07 | A1 SFT horizon-5 | Ablation/control | PASS | NONE |
| 08 | Prefix-Support Swap | Mechanism diagnostic | NEXT | TBD |

## Current scientific story

1. C0 establishes the frozen targeted baseline.
2. SFT demonstrates that the targeted lexical knowledge is learnable.
3. Knowledge-conditioned OPD transfers only a small fraction of that knowledge.
4. Matched-train audits show the gap already exists on training examples.
5. A0 shows that simply extending OPD from 3 to 5 passes does not close the gap.
6. A4 shows Teacher hint-induced distributional change overlaps strongly with OPD KL.
7. A4b shows evaluator mismatch is real but does not explain the OPD-vs-SFT gap.
8. A1 shows direct sequence supervision acquires most of the knowledge within 1-2 passes.
9. The next causal question is prefix/trajectory support.

## Asset policy

- Repository:
  method code, configs, recipes, manifests, tests, conclusions.

- `/workspace/mtpatcher/data`:
  frozen and generated datasets.

- `/workspace/mtpatcher/runs`:
  checkpoints, translations, metrics, judge outputs, run-local launchers.

- Historical paths are not physically reorganized merely for aesthetics.

- Invalid or aborted runs are retained and explicitly labelled for provenance.

## TensorBoard

The current targeted custom Torch-NPU line did not emit TensorBoard
event files. Training history is recorded through:

- `progress.json`
- `manifest.json`
- `summary.json`
- `master.log`
- arm-specific train/eval logs
- JSONL translation/judge outputs

Future formal training experiments should emit TensorBoard in addition
to durable JSON/log artifacts.

## Entry points

Human index:

    manifests/experiments/targeted/README.md

Machine index:

    manifests/experiments/targeted/INDEX.json

Individual experiment manifests are numbered in scientific order,
not chronological filesystem order.
