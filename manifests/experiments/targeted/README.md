# 定向 WA / OPD 实验索引

本目录是当前 MT-PATCHER / OPD 定向研究线的人工可读与机器可读索引。

这里不会为了目录美观去移动历史数据、检查点 或 运行目录。
历史路径保持稳定，以保证 来源追溯 可追溯。

## 当前科学实验链

| 编号 | 实验 | 类型 | 状态 | TensorBoard |
|---|---|---|---|---|
| 00 | C0 定向 baseline | 基线 | PASS | NONE |
| 01 | C1/C2/C3 SFT 正对照 | 正对照 | PASS | NONE |
| 02 | O1/O2/O3 知识条件 OPD | 实验室适配 | PASS | NONE |
| 03 | Matched 训练集 audits | 诊断 | PASS | NONE |
| 04 | A0 OPD horizon-5 | 消融 | PASS | NONE |
| 05 | A4 KL 信号定位 | Mechanism 诊断 | PASS | NONE |
| 06 | A4b 语义词汇审计 | Evaluator 诊断 | PASS with caveat | NONE |
| 07 | A1 SFT horizon-5 | 消融 / 对照 | PASS | NONE |
| 08 | Prefix-Support Swap | Mechanism 诊断 | NEXT | TBD |

## 当前科学结论链

1. C0 固定了 定向 baseline。
2. SFT 证明 化学 与 成语 定向知识 对当前 学生模型 是可学习的。
3. Knowledge-conditioned OPD 能传递一部分知识，但幅度明显弱于 SFT。
4. Matched 训练集 audit 证明这个差距在训练样本本身已经存在。
5. A0 表明单纯把 OPD 从 3 pass 延长到 5 pass 不能解决问题。
6. A4 表明 教师模型 词汇提示 引发的分布变化与 OPD KL 信号有明显重叠。
7. A4b 表明 评测器偏差 确实存在，但不足以解释 OPD 与 SFT 的巨大差距。
8. A1 表明直接 序列监督 通常在 1–2 pass 内就能吸收大部分 定向知识。
9. 下一步机制问题是 前缀 / 轨迹支持。

## 当前权威关系

当前 定向 实验状态与 来源追溯：

`manifests/experiments/targeted/`

其中：

- `README.md`：人工可读的当前实验链；
- `INDEX.json`：机器可读索引；
- `00_*.json` 到 `07_*.json`：逐实验 frozen manifest；
- `ARTIFACT_PATHS.md`：数据、代码、run、日志和评测资产路径表。

旧的 `docs/research/02_experiment_registry.md`
只作为人工科研总结，不再承担唯一 唯一权威来源。

## 资产策略

### Git 仓库

保存：

- 方法代码；
- configs；
- recipes；
- manifests；
- tests；
- 关键科研结论与文档。

### `/workspace/mtpatcher/data`

保存：

- 冻结数据；
- 生成数据；
- 训练集与评测集资产。

### `/workspace/mtpatcher/runs`

保存：

- 检查点s；
- translations；
- metrics；
- 评测模型输出；
- 运行目录内启动脚本；
- progress / state；
- logs。

历史路径不会仅仅因为“目录不好看”而迁移。

失败或中止的 run 如果具有 来源追溯 价值，会明确标记后继续保留。

## TensorBoard

当前 定向 custom Torch-NPU 实验没有生成 TensorBoard event。

训练历史主要通过以下资产记录：

- `progress.json`
- `manifest.json`
- `summary.json`
- `master.log`
- 各 arm 的 train / eval 日志
- JSONL translation / judge 输出

未来新的正式训练实验，除 durable JSON / log 之外，原则上同时生成 TensorBoard。

## 入口

人工实验索引：

`manifests/experiments/targeted/README.md`

资产路径索引：

`manifests/experiments/targeted/ARTIFACT_PATHS.md`

机器索引：

`manifests/experiments/targeted/INDEX.json`

每个实验 manifest 按科学逻辑编号，而不是按 filesystem 创建时间编号。
