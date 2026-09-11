# Experiment Registry

Last major update: 2026-09-11

This file is the human-readable source of truth for the current research line.
Raw outputs remain under `/workspace/mtpatcher/runs/`.
Large datasets remain under `/workspace/mtpatcher/data/`.

## Current research question

Can on-policy distillation replace or improve the sequence-distillation component
of MT-PATCHER while retaining data efficiency and reducing pipeline complexity?

The current targeted experiments isolate one narrower question:

> Can OPD transfer targeted translation knowledge that is demonstrably learnable
> by ordinary supervised fine-tuning?

## Experiment matrix

| ID | Method | Target | Status | Main conclusion |
|---|---|---|---|---|
| C0 | Frozen Qwen3-0.6B | Baseline | PASS | Frozen targeted and general-MT baseline |
| C1 | SFT | Idiom | PASS | Strong idiom learning |
| C2 | SFT | Chemistry | PASS | Strong chemistry learning |
| C3 | SFT | Idiom + Chemistry | PASS | Learns both domains |
| O1 | 3-pass knowledge-conditioned OPD | Idiom | PASS | Positive but weak transfer |
| O2 | 3-pass knowledge-conditioned OPD | Chemistry | PASS | Very weak positive transfer |
| O3 | 3-pass knowledge-conditioned OPD | Combined | PASS | Weak positive transfer |
| A0 | 5-pass OPD horizon ablation | O1/O2 | RUNNING | Tests undertraining vs signal-quality hypothesis |
| A4 | KL-mass / gradient diagnostic | Idiom + Chemistry | PLANNED | Measures task-relevant supervision density |
| A3 | Localized OPD | Idiom + Chemistry | PLANNED | Dense vs informative vs random-matched token supervision |
| A1 | Exposure-matched SFT vs OPD | Idiom + Chemistry | PLANNED | Controls optimization/exposure explanation |
| A2 | Teacher-hint ablation | Idiom + Chemistry | PLANNED | Measures contribution of explicit teacher-side knowledge |
| B1 | PDS-OPD small pilot | PDS | BLOCKED | Starts only after targeted mechanism decision |
| C | Seeds + significance | Final candidate | BLOCKED | Only for final candidate method |

## Key established results

### Chemistry held-out

- C0: 0.097
- C2 SFT: 0.261, delta +0.164
- C3 SFT: 0.261, delta +0.164
- O2 OPD: 0.107, delta +0.010
- O3 OPD: 0.106, delta +0.009

### Idiom held-out

- C0: 2.613
- O1 OPD: 2.721, delta +0.108
- O3 OPD: 2.716, delta +0.103

### Idiom train1000

- C0: 2.680
- C1 SFT: 3.368, delta +0.688
- C3 SFT: 3.352, delta +0.672
- O1 OPD: 2.746, delta +0.066
- O3 OPD: 2.726, delta +0.046

The matched train-set comparison strongly indicates that the weak OPD result is
not primarily a held-out generalization failure. OPD under-learns targeted
knowledge even on seen training contexts.

### General MT cost

Macro BLEU delta vs C0:

- C1 SFT: -0.812508
- C2 SFT: -2.171510
- C3 SFT: -0.842484
- O1 OPD: -0.585807
- O2 OPD: -0.094100
- O3 OPD: -0.408422

Current interpretation:

> SFT injects targeted knowledge strongly but causes larger broad-MT damage.
> OPD is substantially more conservative but transfers targeted knowledge weakly.

## Current A0 experiment

Scientific class:

`ABLATION / KNOWLEDGE-CONDITIONED OPD HORIZON`

Question:

> Is 3-pass OPD weak because optimization horizon is insufficient, or because
> dense KL contains too little task-relevant supervision?

A0 runs O1 and O2 from C0 for five passes with a scheduler defined over the full
five-pass horizon.

Decision rule:

- P3 -> P4 -> P5 task improvement: undertraining remains plausible.
- Task-level plateau near P3 while KL continues changing: supervision quality
  becomes the primary explanation.
- Do not decide from P5 alone; use the P1-P5 learning curve.

## Next experiment order

A0 -> A4 -> A3 -> A1 -> A2 -> B1 -> final full pipeline -> seeds/significance.

Do not expand to large PDS runs while targeted OPD still fails to learn the
training knowledge efficiently.
