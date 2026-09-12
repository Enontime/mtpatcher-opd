# 定向实验代码

本目录保存当前 MT-PATCHER / OPD 定向知识-transfer 研究线的实现。

## 数据构造

以下前缀的脚本主要用于构造和冻结 化学 / 成语 定向数据：

- `generate_targeted_*`
- `repair_targeted_*`
- `finalize_targeted_*`
- `materialize_targeted_*`
- `merge_targeted_*`

这些文件主要属于 来源追溯 资产。

在日常 方法开发 中通常不需要重新运行。

## C0 基线

`run_c0_targeted_diagnostic.py`

用于冻结并评测 Qwen3-0.6B C0 baseline。

## SFT 正对照

主要实现：

`targeted_sft_c123_pipeline.py`

实验 arm：

- C1：成语 SFT；
- C2：化学 SFT；
- C3：成语 + 化学 combined SFT。

目的：

证明当前 定向知识 与 evaluator 确实具有可学习信号。

## 知识条件 OPD

3-pass 参考实现：

`targeted_wa_opd_overnight_v3.py`

5-pass horizon 实验实现：

`targeted_wa_opd_horizon5_o12_v2.py`

实验 arm：

- O1：成语；
- O2：化学；
- O3：Combined。

科学标签：

`LAB ADAPTATION / KNOWLEDGE-CONDITIONED OPD`

语义：

- 学生模型 只看到 source；
- 教师模型 在相同 学生模型 轨迹前缀 上额外看到 词汇侧信息；
- 教师模型 与 学生模型 通过 软 KL 监督 交互。

## 训练过程与机制诊断

### 训练集审计s

- `wa_opd_trainset_audit_v1.py`
- `wa_sft_trainset_audit_c123_v1.py`
- `trainset_audit_common.py`

作用：

在冻结、确定性的 训练子集 上评测 task-level learning，
判断 OPD 的弱结果是否已经出现在训练数据本身。

### A0：Horizon-5

- `targeted_wa_opd_horizon5_o12_v2.py`
- `horizon5_matched_traincurve_eval_v1.py`

结论：

单纯把 OPD 从 3 pass 延长到 5 pass，
不能显著解决 定向知识 uptake 过弱的问题。

### A4：KL Signal Localization

`a4_kl_signal_localization_chemistry_v1.py`

作用：

分析 教师模型 词汇提示 引发的 distributional shift
与实际 OPD KL 信号之间的空间重叠。

### A4b：Semantic Lexical Audit

- `a4b_build_chem_semantic_audit_v1.py`
- `a4b_build_chem_semantic_audit_full1000_v2.py`

作用：

审计 化学 strict canonical substring evaluator 的语义噪声，
区分真实实体修复、canonical paraphrase 与真实退化。

### A1：SFT Horizon-5

对应评测资产记录在 定向实验 manifests 中。

结论：

SFT 对同样 定向知识 的大部分学习通常在 P1–P2 已经完成，
因此训练轮数不足不再是 OPD 弱学习的主要解释。

## 下一步：08 Prefix-Support Swap

下一项机制诊断：

`Prefix-Support Swap`

问题：

> 学生模型生成的前缀支持 是否限制了 软 KL
> 对 定向知识 的有效传递？

该实验属于 诊断，
不应被描述为 canonical OPD method。

如果 教师模型 / 参考译文支持的前缀 明显提高 软 KL knowledge uptake，
后续才进入 localized correction bridge + resumed 学生模型 rollout 的方法实验。

## 已废弃 OPD 实现

历史文件：

- `archive/targeted_wa_opd_overnight_v1.py`
- `archive/targeted_wa_opd_overnight_v2.py`

v1：

历史 semantic / numerical gate 失败实现。

v2：

中间实现，不作为正式科学结果。

新的实验不要从 v1 / v2 继续开发。

## 当前科学流程

C0 baseline

→ C1 / C2 / C3 SFT 正对照

→ O1 / O2 / O3 3-pass 知识条件 OPD

→ matched 训练集 audit

→ A0 5-pass horizon 诊断

→ A4 KL 信号定位

→ A4b 语义评测器审计

→ A1 SFT horizon control

→ 08 Prefix-Support Swap

→ 若 support hypothesis 成立，再进入 local correction bridge

→ 小规模 PDS-OPD pilot。
