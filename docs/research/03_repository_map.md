# 仓库结构

## 先看路径

Git 仓库：

```text
/workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend
```

数据：

```text
/workspace/mtpatcher/data
```

模型：

```text
/workspace/mtpatcher/models
```

训练 / 评测结果：

```text
/workspace/mtpatcher/runs
```

额外日志：

```text
/workspace/mtpatcher/logs
```

当前几个大 run 家族：

```text
/workspace/mtpatcher/runs/sft          ~111G
/workspace/mtpatcher/runs/opd           ~73G
/workspace/mtpatcher/runs/science       ~27G
/workspace/mtpatcher/runs/targeted      ~86G
/workspace/mtpatcher/runs/diagnostics  ~165G
/workspace/mtpatcher/runs/overnight    ~116G
```

整个 `/workspace/mtpatcher/runs` 已约 1.3T，不为了目录美观去移动历史 run。

## Git 仓库顶层

```text
MT-Patcher-Reproduction-Ascend/
├── README.md
├── README_ASCEND.md
├── VERSIONING.md
├── framework.lock.yaml
├── configs/
├── recipes/
├── manifests/
├── scripts/
├── docs/
├── tests/
├── patches/
├── vendor/
├── legacy/
└── results/
```

## 当前正式实验入口

### `configs/`

回答“跑什么”。

```text
configs/sft/
configs/opd/
```

Canonical：

```text
configs/sft/canonical_seqkd_broad20k_qwen3_06b_8b.yaml
configs/opd/canonical_fkl_topk_broad20k_qwen3_06b_8b.yaml
```

### `recipes/`

回答“怎么调用训练框架”。

```text
recipes/sft/
recipes/opd/
```

Canonical：

```text
recipes/sft/canonical_seqkd_broad20k_qwen3_06b_8b.sh
recipes/opd/canonical_fkl_topk_broad20k_qwen3_06b_8b.sh
```

### `manifests/`

记录一次具体实验到底用了哪个 config、commit、data、run path 和 hash。

当前：

```text
manifests/experiments/canonical_seqkd_vs_opd_launch_20260903_v1.yaml
manifests/experiments/targeted/
manifests/releases/
```

Targeted 的人工路径表：

```text
manifests/experiments/targeted/ARTIFACT_PATHS.md
```

## `scripts/` 怎么看

当前目录既有新结构，也保留研究历史：

```text
scripts/
├── data/
├── eval/
├── infer/
├── infra/
├── opd/
├── rewards/
├── train/
├── utils/
├── targeted/
├── pilot_v2/
├── mtpatcher_paper_repro/
├── mtpatcher_paper_faithful_v2/
├── mtpatcher_rq0/
├── mtpatcher_v3/
├── mtpatcher_v4/
├── mtpatcher_v5/
├── mtpatcher_v6/
├── mtpatcher_v7/
├── mtpatcher_v8/
├── mtpatcher_v9/
├── mtpatcher_v10/
├── mtpatcher_v11/
└── mtpatcher_v14/
```

### 新正式代码

```text
scripts/data/
scripts/opd/
scripts/eval/
scripts/analysis/   # 后续分析优先放这里
scripts/infra/
```

当前 `scripts/eval/` 还是空目录；很多历史 evaluator 仍在旧 run / infra 脚本中。

### `scripts/targeted/`

这是 9 月 10 日以后针对 Idiom / Chemistry 做的 knowledge-transfer 机制分支，包括 context construction、SFT positive control、knowledge-conditioned OPD、train-set audit、horizon、KL localization 和 semantic audit。

它是当前机制研究的子目录，不代表整个 MT-PATCHER 项目。

### `scripts/mtpatcher_paper_repro/`

早期 paper reproduction 代码。

### `scripts/mtpatcher_paper_faithful_v2/`

更系统的 paper-faithful reproduction / patcher calibration / full-chain 尝试。

### `scripts/mtpatcher_rq0/`

更大 source pool、Teacher headroom、SeqKD scaling / top-NLL 等实验。

### `scripts/mtpatcher_v3/`

Strong Reproduction 的关键代际之一，包含 Broad20k、PE、PDS、WA 和 early forward-KL OPD。

### `scripts/mtpatcher_v4` 到 `v9`

主要记录早期 OPD / correction / KL / local repair 等多轮机制探索。

这些文件用于追溯旧 run，不是今天的正式方法 API。

### `scripts/mtpatcher_v10/`

selection OPD / RQ2 相关实现。

### `scripts/mtpatcher_v11/`

PDS generation / PE+PDS 相关实现。

### `scripts/mtpatcher_v14/`

WA generation / repair / final merge / train-eval 相关实现。

### `scripts/pilot_v2/`

更早期的 human SFT、SeqKD、GRPO / OPD feasibility pilot。

## `runs/` 怎么看

### `/runs/sft`

主要包括 human6565、canonical SeqKD，以及 SeqKD reduction / scheduler / order reconciliation。

重要 run：

```text
/workspace/mtpatcher/runs/sft/canonical_seqkd_broad20k_qwen3_06b_8b_20260903_133457
```

### `/runs/opd`

主要包括 canonical full OPD、OPD recovery、selective OPD4960、random OPD4960。

```text
/workspace/mtpatcher/runs/opd/canonical_fkl_topk_broad20k_qwen3_06b_8b_20260903_034058
/workspace/mtpatcher/runs/opd/opd_recovery_20260905_v2
/workspace/mtpatcher/runs/opd/selective_opd_k2_4960_20260909_172516
/workspace/mtpatcher/runs/opd/random_opd_4960_20260909_223550
```

### `/runs/science`

统一评测、positive control 和科学对照：

```text
canonical_seqkd_vs_opd_eval_20260906
seqkd_p1_rope_fix_final_20260909_v1
opd_rope_fix_reeval_20260909_v1
seqkd_bleu_curve_hist_vs_verl_20260909_v2
selective_opd4960_preview_20260910
```

### `/runs/diagnostics`

保留 SeqKD execution / optimizer / gradient audits、RoPE semantic audit、OPD temperature / frozen-rollout / microbatch audits。

这些是排错证据，不直接当主实验。

### `/runs/targeted`

```text
c0_targeted_diagnostic1000_20260910
wa_sft_positive_control_c123_20260910
wa_opd_knowledge_conditioned_o123_20260911
wa_opd_trainset_audit_v1_20260911
wa_sft_trainset_audit_c123_20260911
wa_opd_horizon5_o12_20260911
a4_kl_signal_localization_chemistry_v1_20260912
a4b_chem_semantic_audit_full1000_v2_20260912
a1_sft_horizon5_c12_v3_20260912
```

其中还保留明确标记 INVALID / ABORTED 的旧 A1 / A4 run，不能混入正式结论。

## `docs/research/` 怎么看

```text
00_research_overview.md
    整个项目地图

01_current_status.md
    今天知道什么

01_research_overview.md
    Q0 / Q1 / Q2 与方法关系

02_experiment_registry.md
    实验登记、路径、关键数字

02_experiments.md
    人类可读的实验故事

03_repository_map.md
    物理仓库与 runs/data 结构

04_historical_experiment_timeline.md
    时间线与旧结论如何被更新
```

保留两个 `01_*` 和两个 `02_*` 是因为用途不同；暂时不为了改名制造额外 path churn。

## 外部源码

```text
vendor/MT-Patcher-official
vendor/peft0200
vendor/trl-v1.0.0
```

Canonical Verl：

```text
/workspace/mtpatcher/repo/verl-v0.9.0
```

框架版本看：

```text
framework.lock.yaml
```

## 历史代码为什么还在原位置

`scripts/mtpatcher_v3` 到 `v14` 代表真实研究代际，很多旧 run、日志和总结直接引用这些路径。

所以现在采用：

```text
历史代码：原地保留
新实验：进入新结构
```

而不是大规模 `mv`。

历史区域的维护边界看：

```text
legacy/README.md
```

## 说明

这个仓库同时承担三件事：

1. 保存 MT-PATCHER reproduction 的历史证据；
2. 保存 canonical SeqKD / OPD 的可复现实验入口；
3. 支撑正在进行的 OPD 机制研究。

因此目标不是让所有目录看起来完全统一，而是让人能从 README 和 `docs/research/` 快速找到当前科学结论，同时仍能沿 manifest / Git / run path 回到具体实验。
