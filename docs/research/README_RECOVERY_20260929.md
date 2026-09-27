# MT-PATCHER pre-maintenance recovery note — 2026-09-29

This document records the frozen recovery state prepared before the
Huawei/Ascend maintenance window.

## Git state

Branch:

```text
infra/verl-ascend-parity
```

Pre-final-archive baseline:

```text
52591c02de627d96c46ed775a8c9a481b28190c0
```

The baseline repository was exported as:

```text
project_after_archive.bundle
SHA256 0efaa0e38b164b241c870b6aea1d056d6685c3a2a78f9a2205d1cc6d5da1c68e
```

The branch was also pushed to:

```text
https://github.com/Enontime/mtpatcher-opd.git
infra/verl-ascend-parity
```

## Off-server recovery archives

### Formal run evidence

```text
formal_runs_evidence_p0_20260927.tar.zst
SHA256 1c8d666d8d2a226c3261323c0f11e5302fa1a9f3694a57f4e90fdc6bce1ff339
```

Contains logs, TensorBoard, validation generations, status,
provenance and other evidence while excluding large training checkpoints.

### PDS9952 retrospective and endpoint evidence

```text
pds9952_retrospective_final_20260927.tar.zst
SHA256 51548ca28b27c1e9dc56543ee5f50c0a0d8d9c498a7b0807ea0ed77387180c08
```

Contains the completed 39-point SeqKD/OPD retrospective evidence,
freeze_v1, and endpoint_validation_v4.

### Critical inference-ready endpoint models

```text
critical_hf_endpoints_p0_20260927.tar.zst
SHA256 e17823ee2ce682c41fa82555ab5645458f5d28f280dfee2b58afba576def5b04
```

Contains selected HuggingFace/merged endpoint models for the canonical
Broad20k, matched20k-v2, PE11792 and PDS9952 formal lines.

### Frozen scientific data

```text
frozen_data_p0_20260927.tar.zst
SHA256 1bf540282d2343d168195638b003674a75d9c8f46a3ce5b2f263110789778ce6
```

Includes the frozen PE/PDS and Broad20k prepared data plus the
PDS structural accepted source asset.

If the value above is MISSING, that archive had not yet been finalized
when this recovery note was generated.

## PDS9952 retrospective frozen result

Status:

```text
PDS9952_RETROSPECTIVE_FREEZE_V1=PASS
```

Protocol:

```text
paired steps = 0,100,...,3700,3732
39 points per arm
3231 validation rows per point
10 compare/* metrics per point
primary endpoint = step3732
best-checkpoint reselection = forbidden
```

The SeqKD step3732 endpoint remains the previously frozen endpoint
artifact. The persistent SeqKD transport was accepted only after the
step3732 positive-control parity check.

Primary endpoint:

```text
Macro BLEU:
  SeqKD = 19.427635192871094
  OPD   = 19.455373764038086
  delta = +0.027738571166992188

Macro chrF:
  SeqKD = 49.945316314697266
  OPD   = 50.097225189208984
  delta = +0.15190887451171875
```

Retrospective interpretation is descriptive and single-seed:
BLEU shows frequent trajectory crossings and an endpoint near-tie;
chrF shows a more persistent small OPD-positive offset.
This does not establish general superiority, statistical significance,
or a causal mechanism.

## Recovery order

Recommended reconstruction order:

1. Restore/clone the Git repository.
2. Restore frozen scientific data.
3. Restore critical HuggingFace endpoint models.
4. Restore formal evidence and PDS9952 retrospective evidence.
5. Verify archive and per-file SHA256 manifests before reuse.

Large native optimizer/FSDP checkpoint trees were intentionally not
required for the P0 off-server backup. Completed endpoint models and
scientific evidence were prioritized over exact optimizer-state resume.
