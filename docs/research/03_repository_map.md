# 仓库结构

## 先看文件

Git 仓库：

```text
/workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend
```

实验数据：

```text
/workspace/mtpatcher/data
```

模型：

```text
/workspace/mtpatcher/models
```

实验结果：

```text
/workspace/mtpatcher/runs
```

额外日志：

```text
/workspace/mtpatcher/logs
```

如果要查某个 targeted 实验具体在哪：

```text
manifests/experiments/targeted/ARTIFACT_PATHS.md
```

---

## Git 仓库里各目录放什么

```text
MT-Patcher-Reproduction-Ascend/
├── README.md
├── configs/
├── recipes/
├── scripts/
├── manifests/
├── docs/
├── tests/
├── legacy/
└── vendor/
```

### `configs/`

训练和实验配置。

当前主要包括：

```text
configs/opd/
configs/sft/
```

### `recipes/`

可直接启动的实验 recipe / launcher。

主要按方法分：

```text
recipes/opd/
recipes/sft/
```

### `scripts/`

主要代码。

```text
scripts/data/
scripts/opd/
scripts/eval/
scripts/analysis/
scripts/infra/
scripts/targeted/
```

其中：

- `data/`：数据准备；
- `opd/`：新的 OPD 方法代码；
- `eval/`：评估脚本；
- `analysis/`：分析脚本；
- `infra/`：环境、恢复、框架兼容和基础设施；
- `targeted/`：当前 targeted 研究线已经跑过的实现和分析代码。

### `manifests/`

实验索引和来源信息。

当前 targeted 实验主要看：

```text
manifests/experiments/targeted/
```

其中：

```text
README.md
ARTIFACT_PATHS.md
00_c0_baseline.json
01_sft_positive_control.json
02_knowledge_conditioned_opd.json
03_trainset_audits.json
04_a0_opd_horizon5.json
05_a4_kl_localization.json
06_a4b_semantic_audit.json
07_a1_sft_horizon5.json
```

### `docs/research/`

给人看的科研说明。

主要看：

```text
01_research_overview.md
02_experiments.md
03_repository_map.md
```

### `tests/`

当前框架和数据处理相关测试。

### `legacy/`

历史代码和旧实验实现。

这些代码仍然保留，因为旧结果有时需要回溯到当时的实现。

### `vendor/`

外部依赖、第三方代码或 patch 相关内容。

---

## 仓库外的大文件

Git 仓库只放代码、配置、文档和索引。

真正的大数据、模型和训练结果在：

```text
/workspace/mtpatcher/data
/workspace/mtpatcher/models
/workspace/mtpatcher/runs
/workspace/mtpatcher/logs
```

当前 `runs/` 体积已经很大，所以不会为了目录好看去移动历史 run。

---

## targeted 这一条线

主要结果目录：

```text
/workspace/mtpatcher/runs/targeted
```

主要代码：

```text
scripts/targeted/
```

主要数据：

```text
/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/
```

主要实验索引：

```text
manifests/experiments/targeted/
```

这四个位置一起看，基本就能把当前研究线完整还原出来。

---

## 说明

仓库现在同时包含早期复现代码、框架排错代码和当前研究代码。

没有继续大规模移动旧文件，主要是为了保留历史 run 与代码之间的对应关系。

以后新增实验尽量按新的目录分层组织；旧实验保持原路径。
