# Targeted knowledge-transfer experiments

## 文件和结果在哪

代码：

```text
scripts/targeted/
```

结果：

```text
/workspace/mtpatcher/runs/targeted
```

数据：

```text
/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823
```

每个实验的 data / script / run / log：

```text
manifests/experiments/targeted/ARTIFACT_PATHS.md
```

实验状态：

```text
manifests/experiments/targeted/README.md
```

整个项目：

```text
docs/research/00_research_overview.md
docs/research/01_current_status.md
```

## 这条目录在整个项目里的位置

`targeted/` 只负责当前 Idiom / Chemistry 的 knowledge-transfer 机制实验。

整个 MT-PATCHER / OPD 项目还包括：

```text
Strong Reproduction
PE / PDS / WA
Canonical SeqKD / Full OPD
Random vs Selective OPD
PDS-OPD
```

## 主要脚本

### C0 baseline

```text
run_c0_targeted_diagnostic.py
run_c0_targeted_diagnostic.sh
```

### Targeted context / dataset construction

```text
generate_targeted_contexts_qwen3_8b_v1.py
generate_targeted_contexts_qwen3_8b_v2.py
generate_targeted_contexts_qwen3_8b_v2r1.py
repair_targeted_contexts_qwen3_8b_v2r2.py
repair_targeted_contexts_qwen3_8b_v2r3.py
finalize_targeted_contexts_v2r4.py
finalize_targeted_contexts_v2r5.py
finalize_targeted_context_near_duplicates.py
```

这些用于构造并冻结 Idiom / Chemistry targeted contexts。

### SFT positive control

```text
materialize_targeted_sft_positive_control_targets.py
targeted_sft_c123_pipeline.py
run_targeted_sft_c123_fullchain.sh
```

Arm：

```text
C1 Idiom
C2 Chemistry
C3 Combined
```

### Knowledge-conditioned OPD

```text
targeted_wa_opd_overnight_v3.py
```

Arm：

```text
O1 Idiom
O2 Chemistry
O3 Combined
```

当前语义：

```text
Student: source only
Teacher: source + lexical side information
trajectory: Student-generated
objective: top-k forward KL
```

这是 adaptation / mechanism experiment，不能和 canonical same-source OPD 混称完全相同 treatment。

### Train-set audit

```text
trainset_audit_common.py
wa_opd_trainset_audit_v1.py
wa_sft_trainset_audit_c123_v1.py
```

用途：看 OPD-SFT gap 是否在 seen examples 上就存在。

结论：存在。

### A0：OPD Horizon-5

```text
targeted_wa_opd_horizon5_o12_v2.py
horizon5_matched_traincurve_eval_v1.py
```

结论：

```text
Idiom 略有继续学习
Chemistry 基本不变
```

### A4：KL Signal Localization

```text
a4_kl_signal_localization_chemistry_v1.py
```

用途：

```text
Teacher lexical hint 改变 distribution 的 token
是否同时承载较大 OPD KL
```

结论：明显重合。

### A4b：Semantic Audit

```text
a4b_build_chem_semantic_audit_v1.py
a4b_build_chem_semantic_audit_full1000_v2.py
```

用途：区分 strict canonical mismatch 与真正 semantic error。

结论：evaluator noise 存在，但不足以解释 OPD-SFT gap。

### A1：SFT Horizon-5

当前有效 run：

```text
/workspace/mtpatcher/runs/targeted/a1_sft_horizon5_c12_v3_20260912
```

旧的两个 run 已标 INVALID，不进入结论。

A1 说明：

```text
Idiom / Chemistry 大部分 SFT gain 在 P1-P2 已经出现
```

### 下一步：Prefix-Support Swap

下一项正式机制实验不继续塞进历史 `targeted_wa_opd_overnight_v*.py`。

新实现优先放：

```text
scripts/opd/
scripts/analysis/
scripts/eval/
recipes/opd/
manifests/experiments/targeted/
```

核心比较：

```text
Student-prefix soft-KL
vs
Teacher-supported-prefix soft-KL
```

## `archive/`

```text
scripts/targeted/archive/
```

这里保存被后续实现替代的 OPD v1 / v2，仅用于 provenance，不从这里启动新实验。

## 当前最重要结果

```text
SFT:
Idiom +.519
Chemistry +.164

OPD:
Idiom +.108
Chemistry +.010
```

Train-set 上也存在同样差距。

因此 targeted 线当前要解释：

> 为什么 soft on-policy supervision 对明确 lexical knowledge 的写入效率远低于 direct sequence supervision？

## 说明

这条目录来自 9 月 10 日之后的机制实验快速推进，所以历史脚本较集中。

现有脚本保持原路径以保证 run / hash / manifest 可追溯；后续正式方法实现转向更清晰的 `scripts/opd` / `analysis` / `eval` 分层。
