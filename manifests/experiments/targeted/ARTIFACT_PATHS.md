# MT-PATCHER Targeted Research Artifact Paths

This file is a human-oriented map from each scientific experiment to its data, source code, run outputs, logs, checkpoints/evaluation artifacts, and TensorBoard status.

Canonical machine-readable provenance remains in the adjacent JSON manifests.

Repository root:

    /workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend

Research root:

    /workspace/mtpatcher

## Quick navigation

| ID | Experiment | Manifest |
|---|---|---|
| 00 | C0 Frozen Baseline | `00_c0_baseline.json` |
| 01 | C1/C2/C3 SFT Positive Control | `01_sft_positive_control.json` |
| 02 | O1/O2/O3 Knowledge-Conditioned OPD | `02_knowledge_conditioned_opd.json` |
| 03 | Matched Train-Set Audits | `03_trainset_audits.json` |
| 04 | A0 OPD Horizon-5 | `04_a0_opd_horizon5.json` |
| 05 | A4 KL Signal Localization | `05_a4_kl_localization.json` |
| 06 | A4b Semantic Lexical Audit | `06_a4b_semantic_audit.json` |
| 07 | A1 SFT Horizon-5 | `07_a1_sft_horizon5.json` |

## 00 — C0 Frozen Baseline

**Status:** `PASS`

**Scientific class:** `BASELINE / FROZEN STUDENT`

**Question:** What targeted Chemistry and Idiom performance does the frozen C0 Student achieve before adaptation?

**Frozen source commit:** `38d43788a830f85526b1a6ba961639f5dc01e9c5`

### Data

| Role | Path | Exists at manifest creation |
|---|---|---|
| `data.contexts` | `/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/targeted_section43_contexts_qwen3_8b_final_20260910` | `True` |
| `data.diagnostic` | `/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/targeted_diagnostic1000_v1_20260910` | `True` |

### Models

- **student:** `/workspace/mtpatcher/models/Qwen3-0.6B`
- **teacher:** `None`

### Scripts

| Role | Script | Exists | SHA-256 |
|---|---|---|---|
| `launcher` | `scripts/targeted/run_c0_targeted_diagnostic.sh` | `True` | `4cc8cf8218bfbbb5d10b0e92c36af3bc1aa99c85e084997c7d79bbb7141e324f` |
| `script` | `scripts/targeted/run_c0_targeted_diagnostic.py` | `True` | `f6e1d210f9760f4d31238efc71377122e28f367f8ff9cf0be8d1cf0e8f3e838b` |

### Run / logs / evaluation artifacts

| Artifact | Path | Exists at manifest creation |
|---|---|---|
| `artifacts.idiom_judge` | `/workspace/mtpatcher/runs/targeted/c0_targeted_diagnostic1000_20260910/idiom_deepseek_judge/summary.json` | `True` |
| `artifacts.manifest` | `/workspace/mtpatcher/runs/targeted/c0_targeted_diagnostic1000_20260910/manifest.json` | `True` |
| `artifacts.progress` | `/workspace/mtpatcher/runs/targeted/c0_targeted_diagnostic1000_20260910/progress.json` | `True` |
| `artifacts.run_dir` | `/workspace/mtpatcher/runs/targeted/c0_targeted_diagnostic1000_20260910` | `True` |

### TensorBoard

NONE — This targeted custom Torch-NPU experiment did not emit TensorBoard event files.

### Key recorded result

```json
{
  "chemistry_strict": 0.097,
  "idiom_mean_0_to_5": 2.613
}
```

### Interpretation

Frozen baseline for all targeted SFT/OPD comparisons.

## 01 — C1/C2/C3 SFT Positive Control

**Status:** `PASS`

**Scientific class:** `LAB ADAPTATION / TARGETED WA-SFT POSITIVE CONTROL`

**Question:** Is the frozen targeted lexical knowledge actually learnable by the C0 Student under direct sequence supervision?

**Frozen source commit:** `38d43788a830f85526b1a6ba961639f5dc01e9c5`

### Data

| Role | Path | Exists at manifest creation |
|---|---|---|
| `data.contexts` | `/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/targeted_section43_contexts_qwen3_8b_final_20260910` | `True` |
| `data.diagnostic` | `/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/targeted_diagnostic1000_v1_20260910` | `True` |
| `data.targets` | `/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/targeted_sft_positive_control_targets_qwen3_8b_v1_20260910` | `True` |

### Models

- **student_init:** `/workspace/mtpatcher/models/Qwen3-0.6B`
- **teacher_target_generator:** `/workspace/mtpatcher/models/Qwen3-8B`

### Scripts

| Role | Script | Exists | SHA-256 |
|---|---|---|---|
| `launcher` | `scripts/targeted/run_targeted_sft_c123_fullchain.sh` | `True` | `d284ff52b5962bd83dcd942c68fe3f3352b2094f51edaae058b305c6b7c75670` |
| `pipeline` | `scripts/targeted/targeted_sft_c123_pipeline.py` | `True` | `8a33cb039f142546aa98b7fd620b1dee869946867ad935db312ee2944b60fd0d` |
| `target_materializer` | `scripts/targeted/materialize_targeted_sft_positive_control_targets.py` | `True` | `5190b6510c3158fd2b1c30fe4d8acca86a4fac01f4a4792ec856eff0ac672c40` |

### Run / logs / evaluation artifacts

| Artifact | Path | Exists at manifest creation |
|---|---|---|
| `artifacts.idiom_judge` | `/workspace/mtpatcher/runs/targeted/wa_sft_positive_control_c123_20260910/idiom_deepseek_judge/summary.json` | `True` |
| `artifacts.master_log` | `/workspace/mtpatcher/runs/targeted/wa_sft_positive_control_c123_20260910/master.log` | `True` |
| `artifacts.run_dir` | `/workspace/mtpatcher/runs/targeted/wa_sft_positive_control_c123_20260910` | `True` |
| `artifacts.state` | `/workspace/mtpatcher/runs/targeted/wa_sft_positive_control_c123_20260910/state.json` | `True` |

### TensorBoard

NONE — This targeted custom Torch-NPU experiment did not emit TensorBoard event files.

### Key recorded result

```json
{
  "chemistry": {
    "C0": 0.097,
    "C2": 0.261,
    "C2_delta": 0.164,
    "C3": 0.261
  },
  "idiom": {
    "C0": 2.613,
    "C1": 3.132,
    "C1_delta": 0.519,
    "C3": 3.141,
    "C3_delta": 0.528
  }
}
```

### Interpretation

Strong domain-specific positive control: targeted lexical knowledge is learnable by C0.

## 02 — O1/O2/O3 Knowledge-Conditioned OPD

**Status:** `PASS`

**Scientific class:** `LAB ADAPTATION / KNOWLEDGE-CONDITIONED OPD`

**Question:** Can on-policy forward-KL transfer the same targeted lexical knowledge demonstrated learnable by SFT?

**Frozen source commit:** `38d43788a830f85526b1a6ba961639f5dc01e9c5`

### Data

| Role | Path | Exists at manifest creation |
|---|---|---|
| `data.contexts` | `/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/targeted_section43_contexts_qwen3_8b_final_20260910` | `True` |
| `data.diagnostic` | `/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/targeted_diagnostic1000_v1_20260910` | `True` |

### Models

- **student_init:** `/workspace/mtpatcher/models/Qwen3-0.6B`
- **teacher:** `/workspace/mtpatcher/models/Qwen3-8B`

### Scripts

| Role | Script | Exists | SHA-256 |
|---|---|---|---|
| `script` | `scripts/targeted/targeted_wa_opd_overnight_v3.py` | `True` | `8250a41e93a6d2f0f552a84ee70aeca695cd0ac3eb9de3d1d41139d6afe26ae1` |

### Run / logs / evaluation artifacts

| Artifact | Path | Exists at manifest creation |
|---|---|---|
| `artifacts.idiom_judge` | `/workspace/mtpatcher/runs/targeted/wa_opd_knowledge_conditioned_o123_20260911/idiom_deepseek_judge/summary.json` | `True` |
| `artifacts.master_log` | `/workspace/mtpatcher/runs/targeted/wa_opd_knowledge_conditioned_o123_20260911/nohup.out` | `True` |
| `artifacts.run_dir` | `/workspace/mtpatcher/runs/targeted/wa_opd_knowledge_conditioned_o123_20260911` | `True` |
| `artifacts.state` | `/workspace/mtpatcher/runs/targeted/wa_opd_knowledge_conditioned_o123_20260911/state.json` | `True` |

### TensorBoard

NONE — This targeted custom Torch-NPU experiment did not emit TensorBoard event files.

### Key recorded result

```json
{
  "O1_idiom_delta": 0.108,
  "O2_chemistry_delta": 0.01,
  "O3_chemistry_delta": 0.009,
  "O3_idiom_delta": 0.103
}
```

### Interpretation

Real targeted transfer exists, but is far weaker than the corresponding SFT positive control.

## 03 — Matched Train-Set Audits

**Status:** `PASS`

**Scientific class:** `DIAGNOSTIC ONLY / MATCHED TRAINSET AUDIT`

**Question:** Is the OPD-vs-SFT gap already present on the training examples, or mainly a held-out generalization problem?

**Frozen source commit:** `38d43788a830f85526b1a6ba961639f5dc01e9c5`

### Data

No separate data path recorded.

### Models

No model paths recorded.

### Scripts

| Role | Script | Exists | SHA-256 |
|---|---|---|---|
| `common` | `scripts/targeted/trainset_audit_common.py` | `True` | `f8dcf735e844a597d15fdfd16b8053c6b27ff3e07901e82d7e7816169d3d1ebb` |
| `opd` | `scripts/targeted/wa_opd_trainset_audit_v1.py` | `True` | `63ac5588be0728ab9552b9b92e67ad2ba876ea8ed1b012b2dc93596094b93d19` |
| `sft` | `scripts/targeted/wa_sft_trainset_audit_c123_v1.py` | `True` | `c883963b6384a53f80e1c9df3381616f2e6d54a15a8f4c3012b2340de61697f5` |

### Run / logs / evaluation artifacts

| Artifact | Path | Exists at manifest creation |
|---|---|---|
| `artifacts.opd_master_log` | `/workspace/mtpatcher/runs/targeted/wa_opd_trainset_audit_v1_20260911/master.log` | `True` |
| `artifacts.opd_run` | `/workspace/mtpatcher/runs/targeted/wa_opd_trainset_audit_v1_20260911` | `True` |
| `artifacts.sft_master_log` | `/workspace/mtpatcher/runs/targeted/wa_sft_trainset_audit_c123_20260911/master.log` | `True` |
| `artifacts.sft_run` | `/workspace/mtpatcher/runs/targeted/wa_sft_trainset_audit_c123_20260911` | `True` |

### TensorBoard

NONE — This targeted custom Torch-NPU experiment did not emit TensorBoard event files.

### Key recorded result

```json
{
  "chemistry_train": {
    "C0": 0.087,
    "O2": 0.097,
    "O2_delta": 0.01
  },
  "idiom_train": {
    "C0": 2.68,
    "C1": 3.368,
    "C1_delta": 0.688,
    "O1": 2.746,
    "O1_delta": 0.066,
    "OPD_recovery_vs_SFT": 0.0959
  }
}
```

### Interpretation

The weakness is already visible on the training set; it is not mainly a held-out generalization failure.

## 04 — A0 OPD Horizon-5

**Status:** `PASS`

**Scientific class:** `ABLATION / KNOWLEDGE-CONDITIONED OPD HORIZON`

**Question:** Is weak targeted OPD mainly caused by insufficient optimization horizon?

**Frozen source commit:** `38d43788a830f85526b1a6ba961639f5dc01e9c5`

### Data

No separate data path recorded.

### Models

No model paths recorded.

### Scripts

| Role | Script | Exists | SHA-256 |
|---|---|---|---|
| `matched_train_eval` | `scripts/targeted/horizon5_matched_traincurve_eval_v1.py` | `True` | `54a0c683e47e57e46b26f79672594624c94701e39d85976885ad41868ebd16ac` |
| `training` | `scripts/targeted/targeted_wa_opd_horizon5_o12_v2.py` | `True` | `38ac62c0a93f1b762b7536db60421dd8c1e7aca49a57c25851306c3cb76b6e16` |

### Run / logs / evaluation artifacts

| Artifact | Path | Exists at manifest creation |
|---|---|---|
| `artifacts.final_summary` | `/workspace/mtpatcher/runs/targeted/wa_opd_horizon5_o12_20260911/learning_curve/train_matched/a0_horizon5_final_summary_v1.json` | `True` |
| `artifacts.heldout_log` | `/workspace/mtpatcher/runs/targeted/wa_opd_horizon5_o12_20260911/learning_curve/heldout/master.log` | `True` |
| `artifacts.run_dir` | `/workspace/mtpatcher/runs/targeted/wa_opd_horizon5_o12_20260911` | `True` |
| `artifacts.train_matched_log` | `/workspace/mtpatcher/runs/targeted/wa_opd_horizon5_o12_20260911/learning_curve/train_matched/master.log` | `True` |

### TensorBoard

NONE — This targeted custom Torch-NPU experiment did not emit TensorBoard event files.

### Key recorded result

```json
{
  "chemistry_heldout_P5_delta": 0.009,
  "chemistry_train_P5_delta": 0.009,
  "idiom_heldout": {
    "P1_delta": 0.12,
    "P5_delta": 0.141
  },
  "idiom_train_best_recovery_vs_SFT": 0.167
}
```

### Interpretation

Longer optimization contributes modestly to Idiom fitting but does not explain the OPD-vs-SFT gap, and does not rescue Chemistry.

## 05 — A4 KL Signal Localization

**Status:** `PASS`

**Scientific class:** `DIAGNOSTIC ONLY / KL SIGNAL LOCALIZATION`

**Question:** Does dense OPD place substantial optimization signal on positions where Teacher lexical knowledge changes the target distribution?

**Frozen source commit:** `38d43788a830f85526b1a6ba961639f5dc01e9c5`

### Data

No separate data path recorded.

### Models

No model paths recorded.

### Scripts

| Role | Script | Exists | SHA-256 |
|---|---|---|---|
| `script` | `scripts/targeted/a4_kl_signal_localization_chemistry_v1.py` | `True` | `ebaaae102177ec3f076a4ad83437ef4f1a57348ee34462d700224d09ce397fa1` |

### Run / logs / evaluation artifacts

| Artifact | Path | Exists at manifest creation |
|---|---|---|
| `artifacts.invalid_duplicate_run` | `/workspace/mtpatcher/runs/targeted/a4_kl_signal_localization_chemistry_v1_20260912_DUPLICATE_ABORTED_20260912_004123` | `True` |
| `artifacts.master_log` | `/workspace/mtpatcher/runs/targeted/a4_kl_signal_localization_chemistry_v1_20260912/master.log` | `True` |
| `artifacts.run_dir` | `/workspace/mtpatcher/runs/targeted/a4_kl_signal_localization_chemistry_v1_20260912` | `True` |
| `artifacts.summary` | `/workspace/mtpatcher/runs/targeted/a4_kl_signal_localization_chemistry_v1_20260912/summary.json` | `True` |

### TensorBoard

NONE — This targeted custom Torch-NPU experiment did not emit TensorBoard event files.

### Key recorded result

```json
{
  "pearson_hint_vs_opd_kl": 0.5954808,
  "spearman_hint_vs_opd_kl": 0.7288818,
  "top10pct_hint_sensitive_hint_gap_mass": 0.9321,
  "top10pct_hint_sensitive_opd_kl_mass": 0.4914
}
```

### Interpretation

Teacher-hint-induced distributional change strongly overlaps with the actual OPD KL signal. Simple signal dilution is not a sufficient explanation.

## 06 — A4b Semantic Lexical Audit

**Status:** `PASS_WITH_EVALUATOR_NOISE_CAVEAT`

**Scientific class:** `DIAGNOSTIC ONLY / NLP SEMANTIC LEXICAL AUDIT`

**Question:** Does strict canonical substring scoring materially underestimate semantic lexical correctness, and can that explain the OPD-vs-SFT gap?

**Frozen source commit:** `38d43788a830f85526b1a6ba961639f5dc01e9c5`

### Data

No separate data path recorded.

### Models

No model paths recorded.

### Scripts

| Role | Script | Exists | SHA-256 |
|---|---|---|---|
| `full1000_builder` | `scripts/targeted/a4b_build_chem_semantic_audit_full1000_v2.py` | `True` | `71f0f542393092353b03686a15b18cf9443a4f259f1e37037b53469fc81460b3` |
| `pilot_builder` | `scripts/targeted/a4b_build_chem_semantic_audit_v1.py` | `True` | `b7bd8bd7df8f63576dfc7ac35ac0220255902c883ff2d98ebf7cc1f1e42c192e` |

### Run / logs / evaluation artifacts

| Artifact | Path | Exists at manifest creation |
|---|---|---|
| `artifacts.c2_meta_audit` | `/workspace/mtpatcher/runs/targeted/a4b_chem_semantic_audit_full1000_v2_20260912/c2_target_meta_audit_v1.json` | `True` |
| `artifacts.judge_summary` | `/workspace/mtpatcher/runs/targeted/a4b_chem_semantic_audit_full1000_v2_20260912/deepseek_semantic_judge_full1000_v2/summary.json` | `True` |
| `artifacts.judged_jsonl` | `/workspace/mtpatcher/runs/targeted/a4b_chem_semantic_audit_full1000_v2_20260912/deepseek_semantic_judge_full1000_v2/chem_semantic_full3000_judged.jsonl` | `True` |
| `artifacts.manifest` | `/workspace/mtpatcher/runs/targeted/a4b_chem_semantic_audit_full1000_v2_20260912/manifest.json` | `True` |
| `artifacts.posthoc_consistency_audit` | `/workspace/mtpatcher/runs/targeted/a4b_chem_semantic_audit_full1000_v2_20260912/posthoc_consistency_audit_v1.json` | `True` |
| `artifacts.run_dir` | `/workspace/mtpatcher/runs/targeted/a4b_chem_semantic_audit_full1000_v2_20260912` | `True` |

### TensorBoard

NONE — This targeted custom Torch-NPU experiment did not emit TensorBoard event files.

### Key recorded result

```json
{
  "C0_semantic": 0.183,
  "C2_delta": 0.326,
  "C2_semantic": 0.509,
  "O2P5_delta": 0.017,
  "O2P5_semantic": 0.2,
  "O2_recovery_vs_C2": 0.0521,
  "clean_C2_minus_C0_delta": 0.3157303370786517
}
```

### Interpretation

Canonical scoring underestimates semantic correctness, but evaluator mismatch does not explain the large OPD-vs-SFT gap.

## 07 — A1 SFT Horizon-5

**Status:** `PASS`

**Scientific class:** `ABLATION / SFT HORIZON-EXPOSURE CONTROL`

**Question:** How rapidly does direct sequence supervision acquire the same targeted knowledge across P1-P5?

**Frozen source commit:** `38d43788a830f85526b1a6ba961639f5dc01e9c5`

### Data

| Role | Path | Exists at manifest creation |
|---|---|---|
| `data.diagnostic` | `/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/targeted_diagnostic1000_v1_20260910` | `True` |
| `data.targets` | `/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/targeted_sft_positive_control_targets_qwen3_8b_v1_20260910` | `True` |

### Models

- **student_init:** `/workspace/mtpatcher/models/Qwen3-0.6B`

### Scripts

| Role | Script | Exists | SHA-256 |
|---|---|---|---|
| `runtime_arm_launcher` | `/workspace/mtpatcher/runs/targeted/a1_sft_horizon5_c12_v3_20260912/run_arm.sh` | `True` | `n/a` |
| `runtime_master_launcher` | `/workspace/mtpatcher/runs/targeted/a1_sft_horizon5_c12_v3_20260912/run_master.sh` | `True` | `n/a` |
| `trainer` | `scripts/targeted/targeted_sft_c123_pipeline.py` | `True` | `8a33cb039f142546aa98b7fd620b1dee869946867ad935db312ee2944b60fd0d` |

### Run / logs / evaluation artifacts

| Artifact | Path | Exists at manifest creation |
|---|---|---|
| `artifacts.C1_summary` | `/workspace/mtpatcher/runs/targeted/a1_sft_horizon5_c12_v3_20260912/C1/summary.json` | `True` |
| `artifacts.C2_summary` | `/workspace/mtpatcher/runs/targeted/a1_sft_horizon5_c12_v3_20260912/C2/summary.json` | `True` |
| `artifacts.idiom_judge` | `/workspace/mtpatcher/runs/targeted/a1_sft_horizon5_c12_v3_20260912/idiom_deepseek_judge/summary.json` | `True` |
| `artifacts.invalid_preflight_run` | `/workspace/mtpatcher/runs/targeted/a1_sft_horizon5_c12_20260912_INVALID_PREFLIGHT_DUPLICATE_20260912_020223` | `True` |
| `artifacts.invalid_shell_control_run` | `/workspace/mtpatcher/runs/targeted/a1_sft_horizon5_c12_v2_20260912_INVALID_SHELL_CONTROL_20260912_021458` | `True` |
| `artifacts.master_log` | `/workspace/mtpatcher/runs/targeted/a1_sft_horizon5_c12_v3_20260912/master.log` | `True` |
| `artifacts.run_dir` | `/workspace/mtpatcher/runs/targeted/a1_sft_horizon5_c12_v3_20260912` | `True` |

### TensorBoard

NONE — This targeted custom Torch-NPU experiment did not emit TensorBoard event files.

### Key recorded result

```json
{
  "chemistry": {
    "C0": 0.097,
    "P1": 0.233,
    "P2": 0.293,
    "P5": 0.298
  },
  "idiom": {
    "P1_delta": 0.441,
    "P2_delta": 0.536,
    "P5_delta": 0.55
  }
}
```

### Interpretation

The same targeted knowledge is acquired rapidly under SFT, typically within 1-2 passes. Insufficient update horizon is therefore not a convincing common explanation for weak OPD transfer.

## Existing TensorBoard assets outside the targeted line

The targeted custom Torch-NPU experiments above did not emit TensorBoard events. The following earlier SFT/framework runs do.

| Experiment | TensorBoard logdir |
|---|---|
| human6565 SFT | `/workspace/mtpatcher/runs/sft/human6565_qwen3_06b_20260902_114434/tensorboard_log` |
| canonical SeqKD broad20k — 20260903_033659 | `/workspace/mtpatcher/runs/sft/canonical_seqkd_broad20k_qwen3_06b_8b_20260903_033659/tensorboard_log` |
| canonical SeqKD broad20k — 20260903_133457 | `/workspace/mtpatcher/runs/sft/canonical_seqkd_broad20k_qwen3_06b_8b_20260903_133457/tensorboard_log` |

Small Gate-4 smoke TensorBoard directories also exist under `/workspace/mtpatcher/runs/sft/`, but their event files are only roughly 0.6-1.1 KB and are mainly useful as smoke provenance.

## Viewing TensorBoard

On the Ascend server, choose one log directory and run:

```bash
tensorboard --logdir <LOGDIR> --host 127.0.0.1 --port 6006
```

From Windows, in another terminal:

```powershell
ssh -N -L 6006:127.0.0.1:6006 mtpatcher-ascend
```

Then open `http://127.0.0.1:6006` locally.

If `tensorboard` is not installed in the active server environment, the event paths above remain the authoritative artifacts; do not install packages on the offline node merely to view them.
