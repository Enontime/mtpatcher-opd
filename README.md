# MT-Patcher Reproduction / OPD Research

This repository contains the Ascend reproduction and OPD adaptation work for
MT-PATCHER.

## Current research focus

The active research question is:

> Can on-policy distillation transfer targeted MT repair knowledge efficiently,
> and can better localization improve the quality of that supervision?

Current evidence:

- targeted SFT learns Chemistry and Idiom knowledge strongly;
- knowledge-conditioned OPD transfers some targeted knowledge;
- OPD targeted learning is much weaker than SFT even on seen train contexts;
- OPD causes substantially less broad MT degradation than targeted SFT;
- a five-pass horizon ablation is testing undertraining versus
  supervision-quality explanations.

## Start here

1. `docs/research/02_experiment_registry.md`
   Current experiments, results, and scientific decisions.

2. `docs/research/03_repository_map.md`
   Repository layout and provenance guide.

3. `scripts/targeted/README.md`
   Current targeted SFT / OPD implementation map.

4. `README_ASCEND.md`
   Ascend platform and runtime notes.

5. `VERSIONING.md`
   Versioning policy.

## Repository roles

- `configs/`: declarative experiment configurations
- `docs/`: human-readable research and infrastructure documentation
- `manifests/`: frozen provenance and experiment identities
- `patches/`: framework/runtime patches
- `recipes/`: reproducible launch recipes
- `scripts/`: experiment and infrastructure code
- `tests/`: implementation tests
- `vendor/`: external source snapshots; intentionally ignored by Git

Large mutable state intentionally lives outside this repository:

- `/workspace/mtpatcher/data/`: datasets and generated training data
- `/workspace/mtpatcher/runs/`: checkpoints, logs, evaluations, progress state
- `/workspace/mtpatcher/models/`: model weights

## Historical code

Directories such as:

- `scripts/mtpatcher_v3`
- `scripts/mtpatcher_v4`
- ...
- `scripts/mtpatcher_v14`

represent chronological research generations.

They are retained for provenance and should not be treated as the current API.

For current OPD work, start from `scripts/targeted/`.

## Git policy

- never use `git add .` for research commits;
- keep commits scientifically scoped;
- do not track checkpoints, generated results, caches, or model weights;
- retain experiment provenance even when an implementation is superseded;
- prefer documentation and indexing before physically reorganizing historical code.
