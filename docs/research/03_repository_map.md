# 仓库地图

本仓库包含多个阶段的 MT-PATCHER 复现与 OPD 实验。

历史目录名称主要记录开发时间与实验演化，
不能直接当作当前科学结构。

## 推荐阅读顺序

### 当前实验状态

`manifests/experiments/targeted/README.md`

当前 定向 实验的主要人工入口。

### 资产路径

`manifests/experiments/targeted/ARTIFACT_PATHS.md`

用于查询数据、代码、run、日志、检查点 和评测结果。

### 定向实现代码

`scripts/targeted/README.md`

用于理解当前 定向 SFT / OPD / 诊断 实现。

### 科研总结

`docs/research/02_experiment_registry.md`

人工科研总结。

正式实验状态与 来源追溯 仍以 定向 manifests 为准。

### Ascend 环境

`README_ASCEND.md`

### 版本规则

`VERSIONING.md`

## 当前 当前研究代码

### 定向机制实验

目录：

`scripts/targeted/`

当前关键实现包括：

- `targeted_sft_c123_pipeline.py`
  - 定向 SFT 正对照；

- `targeted_wa_opd_overnight_v3.py`
  - 已验证 3-pass 知识条件 OPD；

- `targeted_wa_opd_horizon5_o12_v2.py`
  - 已完成的 A0 five-pass horizon ablation；

- `wa_opd_trainset_audit_v1.py`
  - OPD 训练集 audit；

- `wa_sft_trainset_audit_c123_v1.py`
  - SFT matched 训练集 audit；

- `a4_kl_signal_localization_chemistry_v1.py`
  - A4 KL 信号定位；

- `a4b_build_chem_semantic_audit_full1000_v2.py`
  - A4b 语义评测器审计。

历史 v1 / v2 OPD 实现已经移动到：

`scripts/targeted/archive/`

不要用于新的正式实验。

## 当前实验索引

定向 experiment manifests：

`manifests/experiments/targeted/`

其中：

- `00_c0_baseline.json`
- `01_sft_positive_control.json`
- `02_knowledge_conditioned_opd.json`
- `03_trainset_audits.json`
- `04_a0_opd_horizon5.json`
- `05_a4_kl_localization.json`
- `06_a4b_semantic_audit.json`
- `07_a1_sft_horizon5.json`

下一实验：

`08 — Prefix-Support Swap`

## 历史科研代码

例如：

- `scripts/pilot_v2/`
- `scripts/mtpatcher_v3/`
- `scripts/mtpatcher_v4/`
- ...
- `scripts/mtpatcher_v14/`
- `scripts/mtpatcher_paper_repro/`
- `scripts/mtpatcher_paper_faithful_v2/`
- `scripts/mtpatcher_rq0/`

这些目录代表过去不同阶段的实现。

保留它们的主要原因是 来源追溯。

新的研究不要通过最大版本号寻找“最新方法”。

## 历史时间线

已有历史时间线：

`docs/research/04_historical_experiment_timeline.md`

## 基础设施

- `scripts/infra/`
  - runtime recovery、launcher、Ascend 基础设施工具；

- `configs/`
  - 声明式训练配置；

- `recipes/`
  - shell 级可复现实验 recipe；

- `manifests/`
  - 实验、runtime、数据身份；

- `patches/`
  - 新的正式 framework patch；

- `tests/`
  - 实现级测试。

历史 patch 如果仍位于旧目录，
可以继续保留用于 来源追溯。

## Git 外部状态

以下大型状态有意放在 Git 仓库之外：

### 数据

`/workspace/mtpatcher/data/`

包括数据集和生成训练材料。

### Run

`/workspace/mtpatcher/runs/`

包括：

- logs；
- 检查点s；
- progress state；
- evaluation；
- 评测模型输出；
- 运行专属产物。

### 模型

`/workspace/mtpatcher/models/`

仓库内部只保留能够解释或重建实验所需的：

- manifests；
- summaries；
- scripts；
- configs；
- recipes；
- documentation；
- tests。

## results/

仓库内的：

`results/`

只用于保存少量历史分析摘要或需要 Git 来源追溯 的 compact outputs。

完整生成结果仍以：

`/workspace/mtpatcher/runs/`

为主。
