# Mixed-GKD v1

MT-PATCHER × GKD/OPD Phase-1 development experiment.

## Research question

Under the frozen MT-PATCHER PDS setting, how do:

- trajectory source:
  - lambda = 1.0: pure on-policy
  - lambda = 0.5: deterministic 50% on-policy / 50% fixed Teacher trajectory

and

- divergence:
  - projected FKL
  - projected JSD(beta=0.1)

affect MT generalization?

## Scientific cells

| Cell | lambda | divergence |
|---|---:|---|
| GKD-A | 1.0 | projected FKL |
| GKD-B | 1.0 | projected JSD(0.1) |
| GKD-C | 0.5 | projected FKL |
| GKD-D | 0.5 | projected JSD(0.1) |

## Frozen support

Teacher Top-32 + OTHER.

Both Teacher and Student distributions are projected onto the
same K+1 categorical support.

## Frozen probe budget

- source population: PDS2048
- passes: 6
- source exposures: 12,288
- global batch: 16
- optimizer steps: 768
- Student rollout temperature: 1.0
- Teacher scoring temperature: 1.0
- aggregation: token-mean

## Lambda=0.5 schedule

Whole optimizer-step routing.

- ON_POLICY steps: 384
- FIXED steps: 384
- no runtime RNG

Mask semantic SHA256:

b3d7f2784401c939fe4dd8fc95c7aa5fb20c36068d8b3188649b039c872d9e8e

## Fixed response semantics

Frozen Teacher text is canonicalized as:

tokenize(target_translation.strip(), add_special_tokens=False) + [EOS]

Semantic response-bank SHA256:

2fc2c8066e46a7a69775de51e17d21f47b3abb84c8ba919610ca5135e1f421de

Physical response-bank SHA256:

c995f56b2533bef831c4b88953910c06c0421acdb59250667aed86881e4f8711

## Routing identities

lambda=1 route SHA256:

5328928c2d376d3437753d01f8e4c17c5163096dc633e3280bac6b9838d410d5

lambda=0.5 route SHA256:

cdaa103352a77a5798dcef6d162ec6f8cdc346bef75a56926ef1207438f47de6

## Current implementation state

PASS:

- deterministic PDS2048 asset construction
- deterministic lambda=0.5 routing mask
- fixed-response token audit
- independent projected-loss reference
- production projected FKL kernel
- production projected JSD(0.1) kernel
- loss-mode registration / dispatch
- legacy TopK regression
- terminal vLLM dummy compatibility regression
- fixed-response AgentLoop
- FIXED branch performs zero Student generation calls
- runtime step / routing assertions
- real Parquet -> RLHFDataset -> TensorDict -> TransferQueue transport
- four-cell Hydra configurations and preflight
- FIXED model-version bookkeeping:
  trainer step N -> Student model version N-1
- Smoke-C: lambda=0.5 projected FKL, ON_POLICY -> FIXED
- Smoke-D: lambda=0.5 projected JSD(0.1), ON_POLICY -> FIXED
- both NPU smokes complete forward/backward/update/checkpoint/metrics
- both NPU smokes report trajectory staleness 0 for both steps
- complete CPU gate: 37 passed

The pre-version-semantics C/D smoke runs are retained as superseded
provenance. Current C/D runs are the authoritative integration evidence.

NOT AUTHORIZED:

- formal four-cell 768-step probe

Passing CPU or NPU integration gates does not establish a scientific
generalization result.

## Authorization

FORMAL TRAINING: NOT AUTHORIZED

Passing CPU tests or NPU smoke tests does not authorize a formal run.

## Directory roles

- `src/`: candidate runtime code used by Mixed-GKD only
- `builders/`: deterministic asset builders
- `reference/`: independent mathematical oracle
- `tests/`: CPU/static regression tests
- `configs/`: Mixed-GKD-only configuration fragments
- `/workspace/mtpatcher/smoke/mixed_gkd_v1/`: generated smoke outputs only
- `/workspace/mtpatcher/data/verl_science_pe_pds/mixed_gkd_v1_probe/`:
  frozen data assets and manifests
- `/workspace/mtpatcher/repo/verl-v0.9.0-mixed-gkd-v1/`:
  isolated candidate Verl runtime
