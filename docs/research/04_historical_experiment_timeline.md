# Historical Experiment Timeline

This document maps the older numbered experiment directories to their scientific
roles.

The numbered directories are historical generations, not a recommended API or
current experiment hierarchy.

## Phase 0 — Pilot

### `scripts/pilot_v2/`

Approximate period:

2026-08-20 to 2026-08-23

Main themes:

- Qwen3-0.6B supervised fine-tuning
- early GRPO experiments
- BLEU / chrF reward experiments
- early PEGRL-inspired training
- initial Ascend migration

Status:

Historical pilot code.

For current research, do not start here.

---

## Phase 1 — Initial OPD and strong reproduction

### `scripts/mtpatcher_v3/`

Approximate period:

2026-08-23 to 2026-09-01

Main themes:

- initial forward-KL OPD
- matched-data experiments
- broad20k experiments
- teacher SeqKD provenance locking
- strong reproduction dataset freezing
- PDS construction and validation
- WA reproduction
- paired WA experiments

This directory became the main strong-reproduction branch before the later
canonical Verl implementation.

Status:

Important historical provenance.

---

## Phase 2 — KL objective exploration

### `scripts/mtpatcher_v4/`

Approximate period:

2026-08-24

Main themes:

- reverse KL
- PGRKL
- audited PGRKL variants
- clean-room OPD implementations
- teacher-alignment diagnostics
- KL case extraction

Status:

Diagnostic and objective-design history.

---

## Phase 3 — Top-k and patch-aware OPD

### `scripts/mtpatcher_v5/`

Approximate period:

2026-08-24

Main themes:

- forward-KL top-k128
- entropy/temperature OPD variants
- patch-mask construction
- patch-aware data

Status:

Historical method exploration.

---

## Phase 4 — Correction training

### `scripts/mtpatcher_v6/`

Approximate period:

2026-08-24

Main themes:

- correction forward-KL
- HCCL debugging
- correction-model recovery

Status:

Historical method/infrastructure exploration.

---

### `scripts/mtpatcher_v7/`

Approximate period:

2026-08-24

Main themes:

- correction NLL

Status:

Historical baseline branch.

---

### `scripts/mtpatcher_v8/`

Approximate period:

2026-08-25

Main themes:

- PatchBoost
- weighted correction NLL

Status:

Historical method branch.

---

## Phase 5 — PatchBoost + OPD composition

### `scripts/mtpatcher_v9/`

Approximate period:

2026-08-25 to 2026-08-27

Main themes:

- PatchBoost followed by PGRKL
- post-correction data construction

Status:

Historical composition experiment.

---

## Phase 6 — RQ2 selection experiments

### `scripts/mtpatcher_v10/`

Approximate period:

2026-08-25

Main themes:

- selection-based OPD
- random3732 vs full6565
- full6565 PGRKL
- recovery/fix iterations

Status:

Historical RQ2 branch.

---

## Phase 7 — RQ3 PDS

### `scripts/mtpatcher_v11/`

Approximate period:

2026-08-25

Main themes:

- PDS job construction
- Qwen3-8B PDS generation
- repair and health checks
- conservative PDS filtering
- paper-budget PDS
- PE + PDS training/evaluation

Important naming note:

Some scripts inside this directory are named `v12` and `v13`.
Those are experiment revisions inside the v11 directory; there are no separate
`mtpatcher_v12/` or `mtpatcher_v13/` directories.

Status:

Historical RQ3 PDS pipeline.

---

## Phase 8 — RQ3 WA

### `scripts/mtpatcher_v14/`

Approximate period:

2026-08-25

Main themes:

- WA anchor jobs
- WA analog generation
- WA context generation
- repeated repair/salvage stages
- authoritative merge
- PE + PDS + WA training/evaluation

Status:

Historical RQ3 WA pipeline.

---

## Parallel historical branches

### `scripts/mtpatcher_rq0/`

Teacher headroom, NewsCrawl scaling, SeqKD50k, and high-NLL data-selection
experiments.

### `scripts/mtpatcher_paper_repro/`

Early paper-reproduction utilities.

### `scripts/mtpatcher_paper_faithful_v2/`

More faithful MT-PATCHER reproduction and patcher-quality experiments.

---

## Current research generation

Current work no longer follows the `mtpatcher_vN` naming scheme.

Use:

- `scripts/targeted/`
- `scripts/data/`
- `scripts/infra/`
- `scripts/opd/`
- `configs/`
- `recipes/`
- `manifests/`

Current scientific status is tracked in:

`docs/research/02_experiment_registry.md`

Current repository navigation is documented in:

`docs/research/03_repository_map.md`

## Policy going forward

Do not create new directories such as:

`mtpatcher_v15/`, `mtpatcher_v16/`, ...

New work should be named by scientific role, for example:

- `scripts/targeted/`
- `scripts/opd/`
- `scripts/pds/`
- `scripts/eval/`
- `scripts/diagnostics/`

Historical numbered generations remain frozen in place for provenance.
