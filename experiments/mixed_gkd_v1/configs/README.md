# Mixed-GKD v1 Configs

## probe/

Scientific Phase-1 specifications.

These files define the frozen four-cell 768-step experiment:

| Cell | lambda | loss |
|---|---:|---|
| GKD-A | 1.0 | projected FKL |
| GKD-B | 1.0 | projected JSD(0.1) |
| GKD-C | 0.5 | projected FKL |
| GKD-D | 0.5 | projected JSD(0.1) |

These configs are SPECIFICATION ONLY.

FORMAL EXECUTION IS NOT AUTHORIZED.

## smoke/

NPU integration checks only.

Each smoke runs two optimizer steps on lambda=0.5:

- step 1: ON_POLICY
- step 2: FIXED

Smoke-C exercises projected FKL.
Smoke-D exercises projected JSD(0.1).

Outputs must remain under:

/workspace/mtpatcher/smoke/mixed_gkd_v1/npu/

A smoke PASS does not authorize the 768-step probe.

## Shared runtime

- candidate Verl:
  /workspace/mtpatcher/repo/verl-v0.9.0-mixed-gkd-v1

- custom routing/fixed-response code:
  experiments/mixed_gkd_v1/src/

- AgentLoop registry:
  configs/agent_loops_v1.yaml

- frozen assets:
  /workspace/mtpatcher/data/verl_science_pe_pds/mixed_gkd_v1_probe/
