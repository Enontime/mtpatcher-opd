# Targeted experiments

## 先看文件

当前目录：

```text
scripts/targeted/
```

实验结果：

```text
/workspace/mtpatcher/runs/targeted
```

targeted 数据：

```text
/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823
```

每个实验具体对应的数据、run、日志：

```text
manifests/experiments/targeted/ARTIFACT_PATHS.md
```

---

## 主要脚本

### C0 baseline

```text
run_c0_targeted_diagnostic.py
run_c0_targeted_diagnostic.sh
```

生成冻结 C0 在 targeted diagnostic set 上的翻译和基线结果。

### SFT positive control

```text
materialize_targeted_sft_positive_control_targets.py
targeted_sft_c123_pipeline.py
run_targeted_sft_c123_fullchain.sh
```

用于构造 SFT target，并训练：

- C1：Idiom；
- C2：Chemistry；
- C3：Combined。

### Knowledge-conditioned OPD

```text
targeted_wa_opd_overnight_v3.py
```

当前主要的 targeted OPD 实现。

Student 只看 source。

Teacher 除了 source，还获得对应的 lexical side information。

训练仍然使用 Student-generated trajectory 上的 top-k forward KL。

### Train-set audit

```text
trainset_audit_common.py
wa_opd_trainset_audit_v1.py
wa_sft_trainset_audit_c123_v1.py
```

检查 OPD / SFT 在训练样本上到底学到了多少。

### A0 horizon

```text
targeted_wa_opd_horizon5_o12_v2.py
horizon5_matched_traincurve_eval_v1.py
```

把 O1 / O2 延长到 5 个 pass，并看 held-out 和 matched train curve。

### A4 KL localization

```text
a4_kl_signal_localization_chemistry_v1.py
```

分析 Teacher lexical hint 改变分布的位置，和实际 OPD KL signal 是否重合。

### A4b semantic audit

```text
a4b_build_chem_semantic_audit_v1.py
a4b_build_chem_semantic_audit_full1000_v2.py
```

构造 Chemistry semantic audit，用来区分：

- strict canonical miss；
- 合理同义表达；
- 部分正确；
- 真正错误。

---

## 数据构造相关脚本

这一目录里还保留了 targeted context 和 SFT target 的生成、repair、freeze 脚本，例如：

```text
generate_targeted_contexts_qwen3_8b_v1.py
generate_targeted_contexts_qwen3_8b_v2.py
repair_targeted_contexts_qwen3_8b_v2r2.py
repair_targeted_contexts_qwen3_8b_v2r3.py
finalize_targeted_contexts_v2r4.py
finalize_targeted_contexts_v2r5.py
finalize_targeted_context_near_duplicates.py
```

这些脚本主要用于重建当前 frozen dataset 的生成过程。

---

## `archive/`

```text
scripts/targeted/archive/
```

这里放已经被后续版本替代，但仍需要保留的旧 targeted OPD 脚本。

当前实验不要从 archive 里启动。

---

## 新实验放哪

现有 targeted 脚本先保持原路径。

新的正式方法实现尽量放：

```text
scripts/opd/
```

新的分析：

```text
scripts/analysis/
```

新的评估：

```text
scripts/eval/
```

启动 recipe：

```text
recipes/
```

实验索引：

```text
manifests/experiments/targeted/
```

---

## 说明

`targeted/` 是这几天快速推进实验时自然形成的一组代码。

目前不再为了目录整齐去移动已经跑过的脚本，因为很多 run、hash 和文档已经引用这些路径。

后续新实验会逐步使用更清晰的 `scripts/opd`、`scripts/analysis`、`scripts/eval` 分层。
