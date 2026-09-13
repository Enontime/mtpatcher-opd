# Offline Prefix-Support Replay — Reviewer-Grade Preregistration v2

Frozen: `2026-09-13T07:34:29.809977+00:00`

## Endpoint decision

### Chemistry primary
**Canonical strict accuracy**.

Reason: it is deterministic and reproducible from frozen artifacts/scorer code.
The historical A4b semantic judged rows do not expose `system_prompt_sha256`, so
we do not invent one. A4b semantic correctness is frozen as a **secondary
robustness endpoint**.

- C0 strict: `0.097000`
- fixed C2 strict ceiling: `0.261000`

### Idiom primary
**Frozen external judge mean score**, with exact historical judge metadata.

- C0: `2.613000`
- fixed C1 ceiling: `3.132000`
- judge config SHA: `f159e8f9d83de9a9511467660efe6ed59a44371676e75a3fd7f3365acf4d7e90`

## Practical effect

`R=(M-M_C0)/(M_SFT-M_C0)`.

Positive screen requires both:

1. `DeltaR >= 0.20`
2. paired-bootstrap 95% CI excludes 0.

Equivalent primary-scale thresholds:

- Chemistry strict accuracy: `0.032800`
- Idiom judge score: `0.103800`

Historical per-example outputs are used only for uncertainty/power.

## Claim boundary

S/T alone supports at most:

> teacher-supported trajectory replay improves targeted soft-KL transfer.

Teacher-signal audit is auxiliary disambiguation; pure Student-prefix causality
is not identified by S/T alone.

## Seed policy

Positive seed1 -> independent confirmation seed.
Direction flip -> third seed.
Seed1 null is not a negative mechanism result.
A negative Chemistry mechanism claim requires at least two independent seeds.

## Historical sanity

Frozen-S must be reported beside historical O1/O2. Material departure restricts
the claim to the offline frozen-support replay regime.

## Governance

After Chemistry results, Idiom protocol, primary endpoint, judge configuration,
effect threshold, and seed policy are immutable except for a demonstrated
versioned engineering bug.

Stop budget: Chemistry seed1 -> Idiom seed1 -> required confirmation -> at most
one G/bridge follow-up -> return to PDS/general-MT unless evidence is clear.

## Current authorization

`formal_training_authorized=false`.

Remaining no-update gate: materialize/freeze full pre-update Teacher-signal
artifacts and event-covered secondary subsets. Then Chemistry seed1 may launch.
