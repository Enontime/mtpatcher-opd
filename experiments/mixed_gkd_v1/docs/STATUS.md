# Mixed-GKD v1 Status

## Scientific contract

Phase-1 studies two factors under the frozen MT-PATCHER PDS setting:

- trajectory source
  - lambda=1.0: ON_POLICY
  - lambda=0.5: deterministic 50% ON_POLICY / 50% FIXED whole-step routing
- divergence
  - projected FKL
  - projected JSD(beta=0.1)

Projected support is Teacher Top-32 + OTHER.

Formal probe budget:

- PDS2048
- 6 passes
- 12,288 source exposures
- global batch 16
- 768 optimizer steps
- lambda=0.5 mask: 384 ON_POLICY / 384 FIXED
- no runtime routing RNG

## Frozen science/assets

PASS:

- deterministic PDS2048 selection
- deterministic 768-step schedule
- lambda=0.5 384/384 step mask
- canonical fixed Teacher response IDs
- training/routing assets
- four-cell scientific configs
- config preflight

Frozen physical asset SHA256:

- fixed_response_bank_v1.jsonl:
  c995f56b2533bef831c4b88953910c06c0421acdb59250667aed86881e4f8711
- pds2048_onpolicy_v1.parquet:
  65495714a6c30ad23bce75712318cd8835ced7ec93a0d88944f31f2fcf19444a
- pds2048_lambda05_v1.parquet:
  e75120fcedc8bc872df2537cf71490120e9b52e0b447ba72f5891b5ac9c7301c
- lambda05_step_mask_v1.jsonl:
  7d94f521893538bfe10a3481109c07f5ebd226d84d12cbdc86ea9d39d6db70bd

## CPU/runtime closure

PASS:

- independent projected-loss reference
- projected FKL production kernel
- projected JSD(0.1) production kernel
- loss registration / dispatch
- terminal vLLM dummy compatibility regression
- fixed-response AgentLoop
- zero Student generation in FIXED branch
- deterministic routing assertions
- real Parquet -> RLHFDataset -> TensorDict -> TransferQueue transport
- FIXED model-version metadata regression

Latest complete CPU gate:

    37 passed

### FIXED model-version semantics

Verl v1 trainer `global_steps=N` denotes optimizer-step ordinal.

Synchronous rollout at trainer step N uses Student model version:

    N - 1

Therefore FIXED routing exports:

    global_steps     = N - 1
    min_global_steps = N - 1
    max_global_steps = N - 1

This matches the native ON_POLICY rollout-server bookkeeping.

The resulting synchronous trajectory metrics are:

    trajectory_spans = 1
    trajectory_staleness = 0
    trajectory_staleness_worst = 0

## NPU integration closure

### Smoke-C

- lambda=0.5
- projected FKL
- 2 optimizer steps
- step 1 = ON_POLICY
- step 2 = FIXED
- real forward/backward/update: PASS
- checkpoint steps 1 and 2: PASS
- post-step metrics: PASS
- trajectory staleness for both steps: 0
- Ray cleanup: PASS
- final smoke gate: PASS

Current train.log SHA256:

    0d89a306bffb7789721697244eb7046c4926f2eb0c2aca8f328b004c66638ae2

### Smoke-D

- lambda=0.5
- projected JSD(0.1)
- 2 optimizer steps
- step 1 = ON_POLICY
- step 2 = FIXED
- real forward/backward/update: PASS
- checkpoint steps 1 and 2: PASS
- post-step metrics: PASS
- trajectory staleness for both steps: 0
- Ray cleanup: PASS
- final smoke gate: PASS

Current train.log SHA256:

    f7b43d111c12ac84375d48bcfa32bd13b2cc2d5d3098f418a8dc1ebe719e2896

Pre-version-semantics Smoke-C/D runs are retained under:

    /workspace/mtpatcher/smoke/mixed_gkd_v1/npu/
      SMOKE-C_pre_version_semantics_fix
      SMOKE-D_pre_version_semantics_fix

They are provenance only and are superseded for final version-semantics verification.

## Candidate Verl provenance

Candidate base HEAD:

    ee2060725c3d5deff0d77ab4ffab4f30cd283dab

Current production patch SHA256:

    aeb31078322f2206bba053109a8502baa3495457d487b87b0d1a3cc37fa61a8a

Formal Verl remains at the same base HEAD and clean.

## Interpretation boundary

The engineering implementation is closed for the current v1 contract.

Smoke PASS establishes runtime correctness and integration coverage.
It does not establish MT generalization or a scientific advantage for any
Mixed-GKD cell.

## Execution boundary

Scientific probe specifications:

    experiments/mixed_gkd_v1/configs/probe/

Smoke outputs:

    /workspace/mtpatcher/smoke/mixed_gkd_v1/

FORMAL 4-CELL x 768-STEP TRAINING: NOT AUTHORIZED

FINAL_EVAL_RESERVED_v1: PASS

Primary post-hoc evaluation is frozen to the existing
matched20k_v2 common-validation contract over validation3231.
It is evaluated only after all four formal cells complete and
is not used for checkpoint selection or adaptive cell decisions.

Explicit human authorization for the GKD-A -> GKD-C formal chain is recorded in `manifests/FORMAL_PROBE_AUTHORIZATION_AC_v1.json`. GKD-B and GKD-D remain unauthorized. Primary final evaluation remains deferred until all four cells complete.

The smoke launcher exposes no formal/probe execution mode.
