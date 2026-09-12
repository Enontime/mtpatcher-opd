# MT-PATCHER Reproduction / OPD Research

## 先看文件

当前研究状态：

```text
docs/research/01_current_status.md
```

整个研究问题和各条线关系：

```text
docs/research/00_research_overview.md
docs/research/01_research_overview.md
```

按实验顺序看我们做过什么、为什么做、得到什么：

```text
docs/research/02_experiments.md
```

查实验登记、关键结果和对应目录：

```text
docs/research/02_experiment_registry.md
```

看仓库结构和服务器数据 / run 在哪：

```text
docs/research/03_repository_map.md
```

回顾从 8 月到现在的研究路线：

```text
docs/research/04_historical_experiment_timeline.md
```

只查当前 targeted 机制实验：

```text
manifests/experiments/targeted/README.md
manifests/experiments/targeted/ARTIFACT_PATHS.md
scripts/targeted/README.md
```

Ascend 环境、框架版本和版本管理：

```text
README_ASCEND.md
framework.lock.yaml
VERSIONING.md
```

## 服务器上的主要位置

```text
仓库
/workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend

数据
/workspace/mtpatcher/data

模型
/workspace/mtpatcher/models

训练与评测结果
/workspace/mtpatcher/runs

额外日志
/workspace/mtpatcher/logs
```

常用 run 目录：

```text
/workspace/mtpatcher/runs/sft
/workspace/mtpatcher/runs/opd
/workspace/mtpatcher/runs/science
/workspace/mtpatcher/runs/diagnostics
/workspace/mtpatcher/runs/targeted
/workspace/mtpatcher/runs/overnight
```

## 当前研究版图

| 部分 | 当前状态 | 最重要结论 |
|---|---|---|
| Strong Reproduction | 已有可靠基线 | PDS 有明确正效果；WA 在通用 MT 上无稳健增益 |
| Full SeqKD / Full OPD | 基础结果已建立 | Full OPD 19.1036，接近 Full SeqKD 19.1652 |
| 低预算 source selection | 当前 K2 路线不成立 | Selective OPD4960 17.9600，Random OPD4960 17.9861 |
| Targeted knowledge transfer | 机制诊断已推进 | SFT 能快速学知识，OPD 只迁移其中一小部分 |
| 下一步 | Prefix-Support Swap | 直接检查 Student prefix support 是否限制 soft-KL 迁移 |

最常用的通用 MT 数字：

```text
Base                              17.3485
K1All / PE                        17.5139
Random SeqKD 4960                 18.2658
Same-source SeqKD 4960            18.4151
K1All + PDS                       18.5696
Random OPD 4960                   17.9861
Selective OPD 4960                17.9600
Full OPD 20000                    19.1036
Historical Full SeqKD 20000       19.1652
```

现在需要同时记住两件事：

1. Full-budget OPD 已表现出很强的知识迁移能力，基本追平 Full SeqKD。
2. 当前 K2 error selection 并没有让 OPD 在低 source budget 下更高效。

因此下一阶段的关键不再是继续缩 K2 budget，而是理解 **什么样的 source / state / prefix 才适合通过 OPD 传递 MT-PATCHER 找到的知识**。

## 仓库骨架

```text
MT-Patcher-Reproduction-Ascend/
├── configs/        # 正式实验配置
├── recipes/        # 调用 Verl / 启动实验
├── manifests/      # 实验版本、路径和来源记录
├── scripts/        # 数据、训练、评测、诊断与历史实现
├── docs/           # 给人看的研究与基础设施文档
├── tests/
├── patches/
├── vendor/
├── legacy/
└── results/
```

`scripts/mtpatcher_v3` 到 `scripts/mtpatcher_v14` 是研究历史代际，不能直接当成今天的科学层级。新的正式实验优先进入 `configs/`、`recipes/`、`scripts/data|opd|eval|analysis|infra` 和 `manifests/`；旧路径保留用于追溯已有 run。

## 项目介绍

本项目研究 MT-PATCHER 的 selective / extendable knowledge distillation 与 On-Policy Distillation（OPD）之间的关系。

原始 MT-PATCHER 先定位 Student 真正不会的翻译知识，再通过 PE、PDS、WA 扩展并强化这些知识。当前工作进一步研究：能否把原来的 offline sequence distillation / SFT 式知识注入替换或扩展成 OPD，并在较小 source budget 下保持效果，同时减少 pipeline 复杂度。

目前最简洁的研究叙事是：

> **MT-PATCHER 决定 WHAT to teach，OPD 决定 HOW to teach；真正的问题是怎样让两者在低预算下协同。**
