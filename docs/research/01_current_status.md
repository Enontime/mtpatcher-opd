# Current Research Status

Updated: 2026-09-11

## 1. Targeted SFT positive control

SFT establishes that both targeted tasks are learnable.

### Chemistry held-out

| Arm | Accuracy | Delta vs C0 |
|---|---:|---:|
| C0 | 0.097 | — |
| C1 Idiom SFT | 0.086 | -0.011 |
| C2 Chemistry SFT | 0.261 | +0.164 |
| C3 Combined SFT | 0.261 | +0.164 |

### Idiom train1000

| Arm | Mean score | Delta vs C0 |
|---|---:|---:|
| C0 | 2.680 | — |
| C1 Idiom SFT | 3.368 | +0.688 |
| C2 Chemistry SFT | 2.439 | -0.241 |
| C3 Combined SFT | 3.352 | +0.672 |

Conclusion:

Targeted knowledge is highly learnable and domain-specific.

---

## 2. Three-pass targeted OPD

### Idiom held-out

| Arm | Mean | Delta vs C0 |
|---|---:|---:|
| C0 | 2.613 | — |
| O1 | 2.721 | +0.108 |
| O2 | 2.639 | +0.026 |
| O3 | 2.716 | +0.103 |

### Chemistry held-out

| Arm | Accuracy | Delta vs C0 |
|---|---:|---:|
| C0 | 0.097 | — |
| O1 | 0.102 | +0.005 |
| O2 | 0.107 | +0.010 |
| O3 | 0.106 | +0.009 |

Conclusion:

OPD produces a real but weak targeted transfer signal.

---

## 3. Training-set audit

### Chemistry train1000

| Arm | Accuracy | Delta |
|---|---:|---:|
| C0 | 0.087 | — |
| O1 | 0.087 | +0.000 |
| O2 | 0.097 | +0.010 |
| O3 | 0.092 | +0.005 |

### Idiom train1000

| Arm | Mean | Delta |
|---|---:|---:|
| C0 | 2.680 | — |
| O1 | 2.746 | +0.066 |
| O2 | 2.692 | +0.012 |
| O3 | 2.726 | +0.046 |

Matched Idiom comparison:

- SFT C1: +0.688
- OPD O1: +0.066

OPD recovers only about 9.6% of the SFT train-set gain.

This strongly disfavors the explanation:

> OPD learns the training knowledge well but fails to generalize.

The task-level learning itself is weak.

---

## 4. General MT trade-off

Macro BLEU delta vs C0:

| Arm | Delta |
|---|---:|
| C1 | -0.813 |
| C2 | -2.172 |
| C3 | -0.842 |
| O1 | -0.586 |
| O2 | -0.094 |
| O3 | -0.408 |

Current interpretation:

OPD is substantially more conservative than SFT, but also injects
targeted knowledge much less effectively.

---

## 5. Active experiment

Run:

`/workspace/mtpatcher/runs/targeted/wa_opd_horizon5_o12_20260911`

Script:

`scripts/targeted/targeted_wa_opd_horizon5_o12_v2.py`

Purpose:

Test whether weak 3-pass OPD is primarily caused by insufficient
optimization horizon.

Only O1 and O2 are run.

The full scheduler is defined over five passes from C0.

Do not modify or relocate the active script/run directory while training.
