# Targeted experiments index

## 先看文件

实验结果怎么串起来：

```text
docs/research/02_experiments.md
```

每个实验的数据、脚本、run、日志、结果在哪：

```text
manifests/experiments/targeted/ARTIFACT_PATHS.md
```

机器可读记录：

```text
00_c0_baseline.json
01_sft_positive_control.json
02_knowledge_conditioned_opd.json
03_trainset_audits.json
04_a0_opd_horizon5.json
05_a4_kl_localization.json
06_a4b_semantic_audit.json
07_a1_sft_horizon5.json
INDEX.json
```

---

## 当前实验顺序

| 编号 | 实验 | 当前状态 |
|---|---|---|
| 00 | C0 baseline | 完成 |
| 01 | SFT positive control | 完成 |
| 02 | knowledge-conditioned OPD | 完成 |
| 03 | matched train-set audit | 完成 |
| 04 | A0 OPD horizon-5 | 完成 |
| 05 | A4 KL signal localization | 完成 |
| 06 | A4b semantic audit | 完成 |
| 07 | A1 SFT horizon-5 | 完成 |
| 08 | Prefix-Support Swap | 下一步 |

---

## 一句话结果

```text
SFT 能快速学到 targeted knowledge
→ 当前 OPD 只能迁移一小部分
→ horizon 不足不是主要原因
→ KL signal 也并非完全落错位置
→ evaluator 误差不足以解释差距
→ 下一步直接检查 prefix / trajectory support
```

---

## 说明

这里主要用于快速定位实验。

需要理解实验为什么做、结果意味着什么时，看：

```text
docs/research/02_experiments.md
```

需要复现实验或找具体 artifact 时，看：

```text
ARTIFACT_PATHS.md
```
