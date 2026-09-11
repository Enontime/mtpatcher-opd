# Versioning Policy

## Core principle

Scientific artifact versions and source-code versions serve different roles.

Stable source files use stable semantic names and are versioned by Git.

Immutable scientific artifacts use dated artifact identifiers.

The intended chain is:

    stable source
        -> Git history
        -> formal scientific freeze
        -> dated immutable artifact

## Stable source names

Reusable source code, current configs, and current recipes should normally
keep stable semantic names.

Examples:

    scripts/data/verl_mt_response_sft_dataset.py
    scripts/opd/constant_zero_reward.py
    configs/opd/canonical_fkl_topk_broad20k_qwen3_06b_8b.yaml
    recipes/opd/canonical_fkl_topk_broad20k_qwen3_06b_8b.sh

Do not create chains such as:

    foo_v1.py
    foo_v2.py
    foo_final.py
    foo_final2.py

for ordinary source evolution.

Use Git commits for that history.

## Immutable scientific artifact IDs

A formal scientific snapshot uses:

    <semantic-name>_YYYYMMDD_vN

Examples:

    canonical_seqkd_vs_opd_20260903_v1
    opd_bridge_probe_20260908_v2
    wa_unseenword_repro_20260910_v1

`YYYYMMDD` is the UTC date on which the artifact was formally frozen.

`vN` is the formal revision number for that UTC date.

If another formal revision is created on the same UTC date:

    20260903_v1
    20260903_v2

If a new formal revision is created on a later date:

    20260904_v1

Runtime timestamps should use ISO 8601 with an explicit timezone.

## Artifacts that should be dated

Examples include:

- experiment manifests;
- frozen dataset snapshots or data manifests;
- audit CSV files;
- immutable analysis reports;
- release manifests;
- exported result-table snapshots;
- paper figure/table source snapshots;
- Git release tags.

## Files that normally should not be dated

Examples include:

- README files;
- VERSIONING.md;
- .gitignore;
- reusable Python modules;
- stable current configs;
- stable current recipes;
- stable top-level entrypoints.

## Immutability rule

Once a dated scientific artifact is declared frozen, do not edit it in place.

A correction creates a new dated version.

The new artifact should explicitly record which earlier artifact it supersedes.

## Live experiment protection

While a formal experiment is running:

- do not rename its referenced files;
- do not move them;
- do not reformat them;
- do not change comments;
- do not overwrite them;
- do not replace them with symlinks;
- do not change execution permissions unless required for recovery.

Record cryptographic hashes of protected live assets.

After the run finishes, verify those hashes before constructing the final
experiment manifest.

## Git tags

A formal science release tag should use:

    science-<semantic-name>-YYYYMMDD-vN

Example:

    science-canonical-seqkd-vs-opd-20260903-v1

A tag is created only after:

1. the experiment has finished;
2. protected hashes have been verified;
3. the final immutable experiment manifest exists;
4. the exact scientific source files have been committed;
5. the intended release commit is known.

Do not create a release tag merely because an experiment has started.

## Provenance honesty

Runtime-resolved framework defaults that were not explicitly preregistered
must be recorded as runtime-resolved values.

Do not retrospectively describe such values as preregistered.

Similarly, operational files frozen after launch must not be described as
launch-time scientific assets.
