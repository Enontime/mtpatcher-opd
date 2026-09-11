# Targeted Experiments

This directory contains the current targeted knowledge-transfer experiments for
the MT-PATCHER / OPD research line.

## Scientific roles

### Data construction

Files beginning with:

- `generate_targeted_*`
- `repair_targeted_*`
- `finalize_targeted_*`
- `materialize_targeted_*`

construct and freeze Chemistry / Idiom targeted datasets and SFT targets.

These scripts are provenance assets. They are usually not rerun during ordinary
method development.

### Baseline

`run_c0_targeted_diagnostic.py`

Freezes and evaluates the Qwen3-0.6B C0 baseline.

### SFT positive control

`targeted_sft_c123_pipeline.py`

Arms:

- C1: Idiom SFT
- C2: Chemistry SFT
- C3: combined SFT

Purpose: establish that the targeted knowledge and evaluators are learnable.

### OPD

Reference 3-pass implementation:

`targeted_wa_opd_overnight_v3.py`

Current horizon ablation:

`targeted_wa_opd_horizon5_o12_v2.py`

Arms:

- O1: Idiom
- O2: Chemistry
- O3: combined

Scientific label:

`LAB ADAPTATION / KNOWLEDGE-CONDITIONED OPD`

Student sees source only.
Teacher sees the same student trajectory prefix plus lexical side information.

### Training-process audits

- `wa_opd_trainset_audit_v1.py`
- `wa_sft_trainset_audit_c123_v1.py`

These evaluate task-level learning on frozen deterministic train subsets.

## Superseded OPD implementations

- `targeted_wa_opd_overnight_v1.py`
  Historical failed semantic/numerical gate implementation.

- `targeted_wa_opd_overnight_v2.py`
  Intermediate implementation; not a scientific result.

Do not use v1/v2 for new experiments.

## Current scientific flow

C0
-> C1/C2/C3 SFT positive control
-> O1/O2/O3 3-pass OPD
-> train-set task audit
-> O1/O2 five-pass horizon ablation
-> KL-mass diagnostic
-> localized OPD
-> small PDS pilot.
