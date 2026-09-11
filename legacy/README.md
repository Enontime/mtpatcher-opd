# Legacy / Historical Implementation Policy

Historical implementation is currently retained in place.

This directory is a logical boundary, not yet a physical archive.

## Historical areas currently retained in place

Examples include:

- `scripts/pilot_v2/`
- `scripts/mtpatcher_v3/`
- `scripts/mtpatcher_v4/`
- `scripts/mtpatcher_v5/`
- `scripts/mtpatcher_v6/`
- `scripts/mtpatcher_v7/`
- `scripts/mtpatcher_v8/`
- `scripts/mtpatcher_v9/`
- `scripts/mtpatcher_v10/`
- `scripts/mtpatcher_v11/`
- `scripts/mtpatcher_v14/`
- `scripts/mtpatcher_paper_repro/`
- `scripts/mtpatcher_paper_faithful_v2/`
- `scripts/mtpatcher_rq0/`
- older standalone OPD / recovery / mechanism probes under `scripts/`

These paths may remain useful for:

- historical result reconstruction;
- provenance audits;
- comparison against earlier custom implementations;
- mechanism diagnostics already referenced by research notes.

## Rule for new formal experiments

Do not add new canonical training infrastructure under historical paths.

New framework-native work should use the current structured areas:

    configs/
    recipes/
    scripts/data/
    scripts/opd/
    scripts/eval/
    scripts/analysis/
    scripts/infra/
    tests/
    manifests/

Only create a subdirectory when real code or artifacts require it.

## Physical migration

Do not move historical files merely for cosmetic cleanup.

Physical migration should happen only after:

1. active experiments are complete;
2. path dependencies have been audited;
3. historical manifests no longer depend on the old location;
4. the migration itself is captured by Git.

Until then, historical code remains frozen in place.
