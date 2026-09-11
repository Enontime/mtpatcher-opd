# MT-Patcher / OPD Research Overview

## Goal

Study whether On-Policy Distillation (OPD) can replace or simplify
sequence-distillation-style components in the MT-Patcher pipeline,
especially for targeted translation error repair and knowledge transfer.

The current research question has narrowed to:

> Can OPD efficiently inject localized translation knowledge into a
> smaller Student, and if not, is the bottleneck optimization horizon
> or supervision quality?

---

## Models

### Student

Qwen3-0.6B

Canonical frozen base:

`/workspace/mtpatcher/models/Qwen3-0.6B`

### Teacher

Qwen3-8B

Canonical path:

`/workspace/mtpatcher/models/Qwen3-8B`

---

## Current targeted domains

### Idiom

Chinese idiom meaning realization in English translation.

### Chemistry

Canonical English lexical realization of chemistry-related terms.

---

## Experimental naming

### SFT controls

- C0: frozen base Student
- C1: Idiom SFT
- C2: Chemistry SFT
- C3: Idiom + Chemistry SFT

### OPD arms

- O0: frozen base Student
- O1: Idiom knowledge-conditioned OPD
- O2: Chemistry knowledge-conditioned OPD
- O3: combined knowledge-conditioned OPD

All OPD arms start from C0.

---

## Scientific classification

Current targeted OPD experiments are:

`LAB ADAPTATION / KNOWLEDGE-CONDITIONED OPD`

Student conditioning:

`source only`

Teacher conditioning:

`source + lexical knowledge + exact Student prefix`

Objective:

`renormalized top-k32 forward KL`

Temperature:

`1.0`

No PPO, reward, or advantage term is used.

---

## Current central observation

Targeted knowledge is strongly learnable with SFT.

However, 3-pass OPD transfers only a small fraction of the same
task-level knowledge, including on seen training contexts.

Therefore the main unresolved explanations are:

1. insufficient OPD optimization horizon;
2. low task-relevant supervision density / quality;
3. limitations of KL-based localized knowledge injection.

The current 5-pass experiment is designed to separate (1) from (2)/(3).
