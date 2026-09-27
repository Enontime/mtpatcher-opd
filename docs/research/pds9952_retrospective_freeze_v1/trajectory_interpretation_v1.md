# PDS9952 Retrospective Trajectory Freeze v1

Status: **PASS**

## Frozen scope

- Paired trajectory: `0,100,...,3700,3732`.
- 39 points per arm.
- 3231 validation rows per point.
- 10 `compare/*` metrics per point.
- Primary endpoint remains **step3732**.
- Retrospective trajectory MUST NOT redefine the endpoint.

## Evidence construction

- SeqKD `0..3700`: persistent native FSDP replay.
- SeqKD `3732`: frozen endpoint-validation artifact.
- OPD `0..3732`: persistent retrospective replay.
- OPD `3732`: cross-checked against frozen endpoint validation.

## Gates

- `seqkd_generation_inventory`: **PASS**
- `opd_generation_inventory`: **PASS**
- `seqkd_replay_pass_count`: **37/37**
- `seqkd_fatal_gate`: **PASS**
- `raw_generation_rows`: **PASS**
- `all_input_multisets`: **PASS**
- `step0_pair_multiset_parity`: **PASS**
- `opd_3732_endpoint_pair_multiset_parity`: **PASS**
- `seqkd_3732_frozen_sha`: **PASS**
- `metric_extraction_39x2`: **PASS**
- `opd_3732_metric_endpoint_parity`: **PASS**

## Descriptive trajectory geometry

### macro_bleu

- TensorBoard tag: `compare/benchmark/macro_bleu`
- Endpoint SeqKD: `19.427635193`
- Endpoint OPD: `19.455373764`
- Endpoint OPD−SeqKD: `+0.027738571`
- Mean OPD−SeqKD over nonzero trajectory points: `+0.005432079`
- Median OPD−SeqKD over nonzero trajectory points: `+0.013714790`
- Positive / negative / tie points: `20 / 18 / 0`
- Sign crossings: `17`
- Minimum observed gap: `-0.406002045` at step `900`
- Maximum observed gap: `+0.385892868` at step `800`

These extrema describe trajectory geometry only. They do not select a checkpoint.

### macro_chrf

- TensorBoard tag: `compare/benchmark/macro_chrf`
- Endpoint SeqKD: `49.945316315`
- Endpoint OPD: `50.097225189`
- Endpoint OPD−SeqKD: `+0.151908875`
- Mean OPD−SeqKD over nonzero trajectory points: `+0.114780225`
- Median OPD−SeqKD over nonzero trajectory points: `+0.118705750`
- Positive / negative / tie points: `31 / 7 / 0`
- Sign crossings: `12`
- Minimum observed gap: `-0.156303406` at step `600`
- Maximum observed gap: `+0.322410583` at step `3400`

These extrema describe trajectory geometry only. They do not select a checkpoint.

## Interpretation boundary

This artifact describes single-seed learning dynamics. It does not establish general superiority, significance, or a causal mechanism.

The paper-facing primary endpoint remains the pre-frozen step3732 comparison.

