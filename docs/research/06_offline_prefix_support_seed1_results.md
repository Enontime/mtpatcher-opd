# Offline Prefix-Support Replay — Seed1 Cross-Domain Result

**Scientific class:** `DIAGNOSTIC ONLY / OFFLINE PREFIX-SUPPORT REPLAY`

**Status:** `PASS / NO POSITIVE SEED1 SCREEN IN EITHER DOMAIN`

## Question

Holding the soft-KL objective and matched training dose fixed, does frozen Teacher-supported replay materially outperform frozen Student-supported replay on targeted knowledge transfer?

## Frozen decision rule

- Practical effect: `DeltaR >= 0.20`.
- Paired bootstrap 95% CI must exclude zero in the positive direction.
- P1/P2 are descriptive learning-curve checkpoints; P3 is the final trained state.
- Best-of-P1/P2/P3 selection is forbidden.

## Results

| Domain | Pass | S | T | T-S | DeltaR | 95% CI on endpoint difference | Positive screen |
|---|---:|---:|---:|---:|---:|---:|---:|
| Chemistry | P1 | 0.115000 | 0.116000 | +0.001000 | +0.006098 | [-0.007000, +0.009000] | FAIL |
| Chemistry | P2 | 0.123000 | 0.120000 | -0.003000 | -0.018293 | [-0.012000, +0.006000] | FAIL |
| Chemistry | P3 | 0.134000 | 0.125000 | -0.009000 | -0.054878 | [-0.018000, +0.000000] | FAIL |
| Idiom | P1 | 2.761000 | 2.779000 | +0.018000 | +0.034682 | [-0.023000, +0.058000] | FAIL |
| Idiom | P2 | 2.872000 | 2.846000 | -0.026000 | -0.050096 | [-0.074000, +0.022000] | FAIL |
| Idiom | P3 | 2.936000 | 2.903000 | -0.033000 | -0.063584 | [-0.084000, +0.018000] | FAIL |

## Primary outcome

Chemistry P3: `S=0.134`, `T=0.125`, `T-S=-0.009`, `DeltaR=-0.054878`; no positive screen.

Idiom P3: `S=2.936`, `T=2.903`, `T-S=-0.033`, `DeltaR=-0.063584`, endpoint-difference CI `[-0.084, 0.018]`; no positive screen.

The Student-supported arm itself learned substantially on Idiom: `2.613 -> 2.936`, recovering about `62.2%` of the frozen SFT headroom.

## Mechanistic interpretation

Pre-update T trajectories were much stronger semantically on Idiom (`4.232` vs `2.377`) and also had higher Teacher hint-gap signal (`0.8442` vs `0.8057`). That advantage did not become a downstream T-arm advantage.

Across Chemistry and Idiom, the result therefore weakens the explanation that poor Student trajectory/support alone is the dominant targeted-OPD bottleneck.

It does **not** establish a negative mechanism claim: one training seed per domain remains insufficient to prove prefix support irrelevant.

## Optimization-dose check

Idiom P3 parameter-update L2: `S=0.293354`, `T=0.286223`. The lack of T advantage is not explained by a grossly larger/smaller update magnitude.

Both formal domains retain non-empty TensorBoard event artifacts for S and T.

## Research decision

Do not launch a positive-confirmation prefix-support seed: the preregistered trigger did not fire.

Prioritize the next mechanism diagnostics around **where/how teacher supervision is applied** (critical-token / sparse or localized KL) and **optimization efficiency**, while preserving the current result as the frozen prefix-support baseline.

## Artifact roots

- Chemistry closure: `/workspace/mtpatcher/runs/targeted/offline_prefix_support_chemistry_seed1_posttrain_analysis_recovery_v2_20260913`
- Idiom closure: `/workspace/mtpatcher/runs/targeted/offline_prefix_support_idiom_seed1_posttrain_v1_20260914`
- Result manifest: `manifests/experiments/targeted/09_offline_prefix_support_seed1_results.json`
