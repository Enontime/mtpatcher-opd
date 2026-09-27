# MT-PATCHER × OPD — PDS9952 Scientific Contract v1 (FROZEN)

**Status:** FROZEN — D0–D7 explicitly approved by the human researcher on 2026-09-25. Implementation may proceed. Formal training remains gated on asset audits and one-step smoke closure.

**Date:** 2026-09-25

## 0. Experiment identity and claim scope

Formal experiment name:

```text
PDS9952 Parent-Diverse Low-Budget Adaptation
```

Scientific classification:

```text
ADAPTATION / local-stage practical transfer-replacement experiment
```

The experiment asks a conditional question:

> Given the same frozen low-budget MT-PATCHER-derived PDS source contexts, the same PE-OPD stage-entry Student, and the same source schedule/exposure budget, how does canonical fixed-target SeqKD compare with canonical on-policy distribution distillation at the PDS stage?

This experiment does **not** reconstruct the paper-exact PDS target supervision, does **not** isolate the causal value of PDS source expansion itself, and does **not** compare end-to-end all-SeqKD vs all-OPD pipelines.

---

## D0. Comparator semantics — FROZEN

Arm B is defined as:

```text
PDS-derived source X'
→ fresh Qwen3-8B deterministic full translation Y'_T
→ fixed target
→ response-only CE
```

Therefore Arm B is:

```text
PDS9952-SeqKD v1
= MT-PATCHER PDS-derived source/context selection
+ fresh Qwen3-8B fixed full-translation target
+ canonical response-only SeqKD training
```

It is a **practical fixed-target distillation comparator** on PDS-derived sources.

It is **not** a paper-exact reconstruction of the original MT-PATCHER PDS target supervision, because the historical PDS artifact already contains its own synthetic `target_translation`, while v1 deliberately replaces that target with a fresh deterministic translation from the same Qwen3-8B Translation Teacher used conceptually by the OPD arm.

The historical PDS `target_translation` is retained only as provenance / a possible later method-faithfulness anchor.

Allowed wording:

> canonical OPD vs canonical fixed-target SeqKD on frozen MT-PATCHER-derived PDS sources.

Disallowed wording from v1 alone:

> OPD directly replaces the paper-exact original MT-PATCHER PDS supervision.

A later optional anchor may train:

```text
PDS-original-Y' SFT
```

but this is outside the v1 primary experiment and must not be silently added before v1 closure.

---

## 1. Scientific question

Given a fixed, parent-diverse, low-budget set of MT-PATCHER PDS-expanded Chinese source contexts \(X'\), can canonical on-policy distribution distillation transfer knowledge more effectively than canonical fixed-target sequence distillation under the same source population, source order, source exposure budget, Student stage-entry weights, Teacher identity, translation prompt, and evaluation protocol?

This is a **method-level canonical-recipe comparison**. It is not strict loss-only causal isolation because canonical SeqKD and canonical OPD retain their already-validated method-specific optimization recipes.

---

## 2. Current evidence motivating the experiment

1. PE-level matched experiments provide positive single-seed evidence for OPD relative to same-source SeqKD.
2. Static patch-local FKL weighting produced transient early signal but no endpoint advantage, while patch-localized Teacher–Student disagreement remained positive at the endpoint.
3. Historical Strong-Reproduction PDS showed strong downstream utility relative to parent-matched repetition, motivating PDS as the next source-distribution setting for transfer-method comparison.
4. The existing Strong-Reproduction PDS pipeline follows the intended information-bottleneck structure: Sentence Analyzer produces topic/domain/style, and PDS generation is conditioned on those abstractions plus the bilingual error/correction pair.

Historical full-PDS utility is **motivation**, not a fresh causal control for this v1 experiment.

---

## 3. Competing explanations and falsifiable outcomes

### H_transfer

Conditional on the same frozen PDS9952 contexts, online Teacher supervision on the current Student trajectory transfers knowledge better than fixed Teacher sequences.

Prediction:

```text
PDS9952-OPD endpoint > PDS9952-SeqKD endpoint
```

on the preregistered evaluation suite.

### H_equivalent_transfer

Conditional on the same frozen PDS9952 contexts, both canonical transfer recipes produce similar endpoint quality.

Prediction:

```text
PDS9952-OPD endpoint ≈ PDS9952-SeqKD endpoint
```

within the resolution of the current single-seed screening run.

### H_seqkd_advantage

Conditional on the same frozen PDS9952 contexts, canonical fixed-target SeqKD produces a stronger endpoint than canonical OPD.

Prediction:

```text
PDS9952-SeqKD endpoint > PDS9952-OPD endpoint
```

### Secondary descriptive observation

Each arm may also be compared descriptively with the common PE-OPD stage-entry model at step 0.

However, improvement over step 0 must be phrased as:

> additional training on frozen PDS9952 contexts under method X improved the model relative to stage entry.

It must **not** be phrased as a causal estimate of the value of PDS context generation itself, because v1 has no no-PDS continuation / parent-repeat continuation control under the same stage and recipe.

---

## 4. Frozen upstream PDS source artifact

Canonical upstream artifact:

```text
/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/
strong_repro_pds_full_to_student_v2/
pds_structural_accepted_v2.jsonl
```

Observed frozen SHA256:

```text
078559128776c2dd2dd5d7ed2c9a8f22794457bf53382e3b4c4578404287f6f8
```

Observed population:

```text
accepted rows       = 57,125
unique parents      = 9,957
unique local pairs  = 18,468
unique synth source = 56,703
```

The upstream artifact is immutable.

---

## D5. PDS scope — FROZEN

PDS9952 is formally named:

```text
PDS9952 Parent-Diverse Low-Budget Adaptation
```

Construction:

```text
57,125 accepted historical PDS contexts
→ 9,957 eligible parents
→ exactly one deterministic accepted context per parent
→ deterministic parent ranking
→ drop exactly 5 parents solely to complete global batches
→ 9,952 selected contexts
```

This deliberately reduces the historical full-PDS expansion strength from many contexts per parent to one context per parent.

Therefore v1 is **not** a reproduction of the historical full-PDS mechanism. It is a low-budget adaptation designed to test whether OPD can make useful use of a roughly 10k, parent-diverse, Student-specific PDS source budget.

The number 9,952 has no semantic optimality claim. It is a mechanical consequence of:

```text
9,957 eligible parents
and
global batch = 16
```

because 9,952 is the largest integer ≤ 9,957 divisible by 16.

---

## D1. One context per parent — FROZEN

For each parent, select exactly one already-accepted PDS row by deterministic hash rank.

Do not use lexicographic first-row selection because the prefreeze audit showed strong positional bias toward `error_index=0` and `pds_slot=0`.

### Canonical hash serialization

For every candidate accepted row, construct exactly this JSON object:

```json
{
  "error_index": <integer>,
  "pair_id": <string>,
  "parent_index": <integer>,
  "pds_slot": <integer>,
  "salt": "MTP_PDS9952_ROW_V1",
  "source": <exact source string from parsed upstream JSON>
}
```

Serialize with Python-equivalent semantics:

```text
json.dumps(
    object,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":")
).encode("utf-8")
```

Rules:

```text
no Unicode normalization
no whitespace normalization
no source strip before hashing
source.strip() may be used only for non-empty validation
```

Compute:

```text
selection_row_sha256 = SHA256(canonical_utf8_bytes)
```

Select the minimum `selection_row_sha256` within each `parent_index`.

If an SHA256 tie ever occurs, break the tie by upstream JSONL line number ascending and record the event. No metric/model-based tiebreak is allowed.

For every selected row, the final population artifact and manifest must preserve at least:

```text
upstream_line_no
parent_index
error_index
pair_id
pds_slot
source
job_id if present
selection_row_sha256
```

This selection must not use BLEU, chrF, Teacher score, Student score, FKL, target quality score, reference overlap, or any test/validation behavior.

---

## D2. 9957 → 9952 parent freeze and training budget — FROZEN

For each of the 9,957 eligible parents, construct exactly:

```json
{
  "parent_index": <integer>,
  "salt": "MTP_PDS9952_PARENT_V1"
}
```

using the same canonical JSON/UTF-8 serialization rules as D1.

Compute `parent_rank_sha256`, sort ascending by `(parent_rank_sha256, parent_index)`, and retain exactly the first 9,952 parents.

Record:

```text
eligible parent count = 9,957
retained parent count = 9,952
excluded parent count = 5
exact excluded parent IDs
exact excluded parent rank hashes
final population SHA256
```

Training budget:

```text
population              = 9,952
passes                  = 6
global source batch     = 16
total source exposures  = 59,712
optimizer steps         = 3,732
schedule seed           = 20260820
```

Population invariants:

```text
rows                  = 9,952
unique parent_index   = 9,952
unique selected rows  = 9,952
unique source text    = 9,952
non-empty sources     = 9,952
```

Failure of any invariant is a global contract failure.

---

## 5. Stage-entry Student

Both arms initialize from the same PE-OPD formal endpoint:

```text
/workspace/mtpatcher/runs/science/pe_pds_v1/
pe_opd_formal_v1/checkpoints/global_step_4422/actor
```

Use the existing Verl FSDP merger:

```text
python -m verl.model_merger merge --backend fsdp ...
```

Create one frozen merged HF model and make both arms point to that exact model identity.

Only model weights / HF identity are inherited. PDS-stage training starts with fresh optimizer/scheduler state and `resume_mode=disable`.

Scientific scope of this stage boundary:

> this is a local PDS-stage replacement study on a Student that has already completed PE-OPD.

It cannot by itself answer an end-to-end claim comparing an all-SeqKD MT-PATCHER pipeline with an all-OPD MT-PATCHER pipeline.

---

## D3. Method-level comparison — FROZEN

Matched variables:

```text
PDS9952 source population
one selected X' per parent
source order
source exposure count
stage-entry Student weights
Teacher identity
translation prompt
thinking mode
evaluation suite
checkpoint/evaluation cadence
primary endpoint
```

Method-specific variables deliberately remain canonical:

```text
SeqKD: fixed Teacher full sequence + response-only CE
       AdamW, lr=2e-5, cosine, 3% warmup

OPD:   current Student rollout + Teacher top-k distribution + FKL
       actor lr=1e-6, canonical native-Verl OPD recipe
```

Therefore this is:

```text
source/exposure/init matched
canonical method-level recipe comparison
not strict loss-only causal isolation
not compute-matched
```

Do not force equal learning rates in v1.

---

## 6. Shared translation prompt

Use the existing frozen prompt:

```text
Translate the following text into English without additional explanations:

{source}

```

`enable_thinking=False` for Teacher-target generation and OPD Student/Teacher chat templating.

---

## 7. Arm B — PDS9952-SeqKD

Teacher target generation must reuse:

```text
scripts/mtpatcher_rq0/generate_seqkd50k_teacher_v1.py
```

Teacher contract:

```text
Teacher         = /workspace/mtpatcher/models/Qwen3-8B
prompt          = shared translation prompt
enable_thinking = false
do_sample       = false
max_new_tokens  = 512
```

Generate exactly one fixed Teacher target per unique PDS9952 source, freeze it once, and reuse the same target across all 6 passes.

Historical PDS `target_translation` must not be used as the main v1 Arm-B target.

Student training reuses the existing Verl response-only SFT path and matched dataset implementation.

Canonical SeqKD recipe:

```text
AdamW
lr                    = 2e-5
weight_decay          = 0.01
clip_grad             = 1.0
scheduler             = cosine
warmup ratio          = 0.03
response-only CE
global batch          = 16
max_length            = 1024
thinking              = false
trainer shuffle       = false
resume_mode           = disable
total_training_steps  = 3732
```

The materialized 6-pass schedule owns source ordering.

---

## 8. Arm C — PDS9952-OPD

Reuse the existing PE11792 / matched20k native-Verl OPD recipe.

Frozen OPD contract:

```text
Student stage-entry          = same merged PE-OPD step4422 model
Teacher                      = Qwen3-8B, frozen
Student rollout              = current Student, no_grad
rollout n                    = 1
rollout temperature          = 1.0
rollout top_p                = 1.0
rollout top_k                = -1
max prompt length            = 1024
max response length          = 256
loss                         = forward_kl_topk
Teacher top-k                = 32
loss aggregation             = token-mean
actor lr                     = 1e-6
ppo epochs                   = 1
use task rewards             = false
use policy gradient          = false
actor KL loss                = false
KL in reward                 = false
thinking                     = false
resume_mode                  = disable
total_training_steps         = 3732
```

OPD rows must contain source/prompt plus matched schedule metadata only.

Forbidden fields in the OPD training view:

```text
target_translation
reference
post_edit
student_translation
historical PDS target
```

---

## D6. Checkpoint and validation cadence — FROZEN

Formal checkpoint/evaluation steps are exactly:

```text
0,
100, 200, 300, ..., 3600, 3700,
3732
```

Step 0 is the single shared PE-OPD stage-entry model and may be evaluated once, then referenced by both arms.

Both B and C use the same:

```text
validation3231
scorer identity
generation settings
translation extraction
raw generation schema
case dump schema
metric computation
```

All nonzero listed checkpoints must be evaluated for both arms. Missing evaluation for one arm at a checkpoint makes that checkpoint unavailable for paired trajectory interpretation until repaired.

Primary endpoint remains step 3732. Intermediate checkpoints are trajectory diagnostics only and cannot replace step 3732 as the formal endpoint.

No retrospective best-checkpoint selection is allowed.

---

## D4. Primary endpoint — FROZEN

The unique primary endpoint is:

```text
step 3732
```

Primary evaluation suite:

```text
WMT24 zh→en 998
FLORES zh→en 1012
Challenge zh→en 197
BLEU
chrF
Macro BLEU
Macro chrF
```

The common stage-entry model is the descriptive step-0 baseline.

---

## D7. Single-seed claim scope and replication rule — FROZEN

v1 is one formal screening / primary run at the frozen training seed/schedule semantics.

Allowed from v1 alone:

```text
In the current single-seed setting,
PDS9952-OPD was higher / similar / lower than PDS9952-SeqKD
at the preregistered endpoint.
```

Not allowed from v1 alone:

```text
statistically significant across seeds
generally superior
stable across random seeds
robust replacement in general
```

Any claim intended to carry the paper's main superiority/robustness conclusion requires independent-seed replication. This requirement applies regardless of whether the single-seed endpoint gap appears large or small.

If the observed gap is modest, it must remain explicitly descriptive until replication; no significance language may be inferred from sample-level bootstrap alone.

Sample/case bootstrap, if used, quantifies sample uncertainty only and does not substitute for training-seed replication.

---

## 9. Required provenance

The frozen population/manifest/run must preserve at minimum:

```text
project git HEAD + dirty-state snapshot
Verl tree/path + HEAD/hash if available
upstream PDS path + SHA
canonical row-hash serialization rule
ROW_SALT
canonical parent-hash serialization rule
PARENT_SALT
per-selected-row selection_row_sha256
exact excluded parent IDs + rank hashes
PDS9952 population path + SHA
source-order manifest + SHA
Teacher generator path + SHA
Teacher model config/tokenizer hashes
frozen SeqKD target asset + SHA
stage-entry merged model path + model/config/tokenizer hashes
resolved SeqKD config
resolved OPD config
training seed/schedule seed
optimizer steps
source exposures
TensorBoard / structured logs
checkpoint identity
validation asset identity/hash
evaluation commands
raw generations
case dumps
per-dataset metrics
final status
```

---

## 10. Failure and repair policy

- Upstream frozen PDS is immutable.
- Any repair creates a versioned child artifact with provenance and fresh SHA.
- Fewer than 9,952 valid parent-diverse sources => stop; do not silently backfill with another policy.
- Teacher-generation local failures => preserve raw output, rerun failed source IDs only, deterministically reassemble, then refreeze the Teacher target manifest.
- Source/order mismatch between arms => global contract failure.
- OPD target/provenance leakage => global contract failure.
- Missing required paired checkpoint evaluation => mark that checkpoint pending/invalid for trajectory comparison; do not silently drop it and compare asymmetric curves.
- Training exit code 0 is insufficient; endpoint step, checkpoint contents, TensorBoard step, source consumption, and evaluation outputs must pass audit.

---

## 11. Explicit exclusions from v1

Do not add:

```text
historical PDS Y' as a third primary arm
patch-local KL weighting
error-span weighted loss
privileged Teacher context
Teacher intervention / corrected-prefix takeover
dynamic gap threshold
dynamic patch retirement
online Feedbacker regeneration
WA
xCOMET verifier
multi-Agent control
LR sweep
PDS ratio sweep
best-checkpoint selection
new Verl loss implementation
```

---

## 12. Claim boundaries

### Primary claim that v1 can answer

```text
Conditional on the frozen PDS9952 Parent-Diverse Low-Budget contexts,
common PE-OPD stage entry, and matched source schedule/exposures,
canonical OPD performs higher / similarly / lower than canonical fixed-target SeqKD
at the preregistered step3732 endpoint in the current single-seed setting.
```

### Secondary descriptive statement that v1 may report

```text
Additional training on PDS9952 under method B/C changed endpoint quality
relative to the common PE-OPD stage-entry model.
```

### Claims that v1 cannot establish

```text
causal value of PDS context generation itself
paper-exact replacement of original MT-PATCHER PDS supervision
end-to-end all-SeqKD vs all-OPD pipeline superiority
compute-matched superiority
strict loss-only causal isolation
general cross-seed superiority
that PDS generation no longer needs target generation
```

A no-PDS continuation / parent-repeat continuation control is a separate future experiment if causal PDS-source value becomes necessary.

A true source-only PDS-OPD pipeline is a separate future simplification experiment because the reused accepted PDS pool was created through a parallel-context generation/acceptance pipeline.

---

## 13. Frozen decision block

The complete D0–D7 decision set was explicitly approved on 2026-09-25 and is frozen for v1:

```text
D0  Comparator semantics:
    fresh Qwen3-8B fixed target is a practical SeqKD comparator,
    not paper-exact original PDS target supervision.

D1  Context selection:
    canonical UTF-8 JSON SHA256 hash-min row per parent;
    per-row selection hash recorded.

D2  Budget:
    9,957 eligible parents → deterministic drop of exactly 5 for batch divisibility
    → 9,952 parents × 6 passes = 59,712 exposures = 3,732 steps.

D3  Comparison type:
    canonical method-level SeqKD vs canonical method-level OPD;
    source/exposure/init matched, not loss-only and not compute-matched.

D4  Primary endpoint:
    step3732 only.

D5  PDS scope:
    Parent-Diverse Low-Budget Adaptation, one context per parent;
    not historical full-PDS reproduction.

D6  Validation cadence:
    step0, every 100 through 3700, and step3732;
    same validation3231/scorer/generation/case-dump contract for B/C.

D7  Single-seed scope:
    v1 permits descriptive single-seed conclusions only;
    any paper-level superiority/robustness claim requires independent-seed replication.
```

Implementation may proceed to builder/manifest/config/recipe and smoke. Formal training remains gated on asset audits and one-step smoke closure. Any semantic change to D0–D7 requires a new versioned contract rather than in-place mutation.


## 14. Freeze provenance

```text
freeze_date = 2026-09-25
freeze_scope = D0-D7 + claim boundaries + failure/repair policy
source_candidate = MT_PATCHER_OPD_PDS_SCIENTIFIC_CONTRACT_v1_RC1.md
change_policy = no in-place semantic edits after freeze; create v2 for any scientific change
formal_training_gate = prepare/finalize asset audit + common stage-entry audit + one-step SeqKD smoke + one-step OPD smoke
```
