# MT-PATCHER 复现与 OPD 研究

本仓库保存 MT-PATCHER 在 Ascend 环境上的复现工作，以及基于
On-Policy Distillation（OPD）的扩展研究。

## 当前研究重点

当前总问题是：

> 能否用 OPD 替代或改造 MT-PATCHER 中原有的序列蒸馏过程，
> 在保持数据效率的同时简化流程，并提高错误定位后监督信号的有效性？

当前 定向 机制实验进一步聚焦于：

> 对于已经证明可以通过直接监督学习的翻译知识，
> 为什么 知识条件 OPD 的知识吸收效率明显更低？

目前已经得到的主要证据：

- 定向 SFT 可以明显学习 化学 与 成语 知识；
- 知识条件 OPD 存在真实但较弱的知识转移；
- OPD 的弱学习现象在训练集本身就已经存在，因此不主要是 留出集泛化问题；
- 将 OPD 从 3 pass 延长到 5 pass 仍不能明显缩小差距；
- A4 表明 教师模型 词汇提示 引发的分布变化与实际 OPD KL 信号有较强重叠；
- A4b 表明 化学 strict evaluator 存在一定 规范化匹配 噪声，但不足以解释 OPD 与 SFT 的巨大差距；
- A1 表明同样的 定向 知识在 SFT 下通常 1–2 pass 就能吸收大部分收益。

当前下一步：

`08 — Prefix-Support Swap`

它是一个机制诊断实验，用于判断 学生模型生成的前缀支持
是否限制了 软 KL 对 定向知识 的有效传递。

## 从哪里开始看

### 1. 当前实验状态与科学链

`manifests/experiments/targeted/README.md`

这是当前 定向研究线的**人工可读权威入口**。

### 2. 每个实验的数据、代码、日志和结果路径

`manifests/experiments/targeted/ARTIFACT_PATHS.md`

用于查找：

- 数据；
- 模型；
- 训练脚本；
- 运行目录；
- 日志；
- 检查点；
- 评测结果；
- 评测模型输出；
- TensorBoard 状态。

### 3. 当前 定向 代码说明

`scripts/targeted/README.md`

用于理解当前 定向 SFT / OPD / 诊断 脚本分别负责什么。

### 4. 研究总览与历史记录

`docs/research/`

其中：

- `00_research_overview.md`：较早的研究总览；
- `01_current_status.md`：较早阶段状态记录；
- `02_experiment_registry.md`：人工可读的实验总结；
- `03_repository_map.md`：仓库结构和 来源追溯 说明；
- `04_historical_experiment_timeline.md`：历史实验时间线。

当前实验的最终状态与 来源追溯 以
`manifests/experiments/targeted/` 为准。

### 5. Ascend 运行环境

`README_ASCEND.md`

### 6. 版本管理规则

`VERSIONING.md`

## 仓库目录职责

- `configs/`：声明式实验配置；
- `docs/`：人工可读的科研与基础设施文档；
- `manifests/`：冻结的实验身份、数据身份和 来源追溯；
- `patches/`：新的正式 framework/runtime patch；
- `recipes/`：可复现的启动脚本与实验 recipe；
- `scripts/`：实验代码和基础设施代码；
- `tests/`：实现级测试；
- `results/`：少量需要长期保留在 Git 中的历史分析摘要；
- `legacy/`：历史实现管理规则；
- `vendor/`：外部源码快照，本地存在但不作为本仓库当前实现。

大型可变状态不会直接存入 Git：

- `/workspace/mtpatcher/data/`：数据集和生成的训练数据；
- `/workspace/mtpatcher/runs/`：检查点、日志、评测和运行状态；
- `/workspace/mtpatcher/models/`：模型权重。

完整 generated 运行结果应优先在 `/workspace/mtpatcher/runs/` 查找。

## 历史代码

以下目录代表过去不同阶段的科研实现：

- `scripts/pilot_v2/`
- `scripts/mtpatcher_v3/`
- `scripts/mtpatcher_v4/`
- ...
- `scripts/mtpatcher_v14/`
- `scripts/mtpatcher_paper_repro/`
- `scripts/mtpatcher_paper_faithful_v2/`
- `scripts/mtpatcher_rq0/`

这些目录为了 来源追溯 暂时保留原路径。

它们不代表当前推荐 API，也不要通过“版本号最大”判断当前实现。

当前 定向 OPD 研究优先从：

`scripts/targeted/`

进入。

## Git 规则

- 科研提交不要使用 `git add .`；
- 每个 commit 尽量对应一个清楚的科研或工程目的；
- 不把 检查点、大型生成结果、cache、模型权重提交进 Git；
- 即使实现被替代，也要保留能够解释历史实验的 来源追溯；
- 历史路径优先通过文档和 manifest 索引，不为了视觉整洁随意迁移。
