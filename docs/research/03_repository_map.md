# Repository Map

This repository contains several generations of MT-PATCHER experiments.
Directory names encode historical development and should not be interpreted as
the current scientific hierarchy.

## Start here

- `README_ASCEND.md`
  Ascend/runtime overview.

- `docs/research/00_research_overview.md`
  Earlier research overview.

- `docs/research/01_current_status.md`
  Earlier project status.

- `docs/research/02_experiment_registry.md`
  **Current human-readable experiment registry and scientific status.**

- `docs/research/03_repository_map.md`
  This file.

- `VERSIONING.md`
  Versioning conventions.

## Active research code

### Targeted mechanism experiments

`/scripts/targeted/`

Current key files:

- `targeted_sft_c123_pipeline.py`
  Positive-control SFT experiment.

- `targeted_wa_opd_overnight_v3.py`
  Validated 3-pass knowledge-conditioned OPD implementation.

- `targeted_wa_opd_horizon5_o12_v2.py`
  Current five-pass horizon ablation derived from validated v3.

- `wa_opd_trainset_audit_v1.py`
  OPD train-set task-level audit.

- `wa_sft_trainset_audit_c123_v1.py`
  SFT matched train-set positive-control audit.

The v1/v2 OPD files are historical failed/intermediate implementations and are
not the current reference implementation.

## Historical research generations

Directories such as:

- `scripts/mtpatcher_v3`
- `scripts/mtpatcher_v4`
- ...
- `scripts/mtpatcher_v14`

represent chronological experiment generations.

They are retained for provenance. New research should not infer the preferred
implementation merely from the largest version number.

A historical timeline/index will be added before any large-scale physical move.

## Infrastructure

- `scripts/infra/`
  Runtime recovery, launch, and infrastructure utilities.

- `configs/`
  Declarative training configurations.

- `recipes/`
  Reproducible shell-level experiment recipes.

- `manifests/`
  Frozen experiment/runtime/data identities.

- `patches/`
  Local framework patches.

## External state

The following intentionally live outside Git:

- `/workspace/mtpatcher/data/`
  Datasets and generated training material.

- `/workspace/mtpatcher/runs/`
  Logs, checkpoints, progress state, evaluations, and run-specific artifacts.

Repository files should contain only compact manifests, summaries, scripts,
configuration, and documentation needed to reconstruct or interpret a run.
