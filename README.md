# MT-Patcher Reproduction / OPD Experiments

## 先看文件

如果你只是想知道“现在做到哪了”，按这个顺序看：

```text
README.md
docs/research/02_experiments.md
manifests/experiments/targeted/ARTIFACT_PATHS.md
scripts/targeted/README.md
```

如果你想找某个实验的数据、脚本、日志或结果文件：

```text
manifests/experiments/targeted/ARTIFACT_PATHS.md
```

如果你想看仓库各目录分别放什么：

```text
docs/research/03_repository_map.md
```

如果你想看 targeted 这条线的具体实现代码：

```text
scripts/targeted/README.md
```

实验结果和大文件不放在 Git 仓库里，主要在：

```text
/workspace/mtpatcher/data
/workspace/mtpatcher/runs
/workspace/mtpatcher/models
/workspace/mtpatcher/logs
```

---

## 现在的主线

目前主要在研究一个问题：

> 当学生模型缺少某个明确的翻译知识时，On-Policy Distillation 能不能像 SFT 一样把这部分知识有效地教进去？

为了把问题做得尽量可分析，当前主要看两个场景：

- 化学术语；
- 成语翻译。

当前已经完成的实验链大致是：

```text
C0 baseline
→ SFT positive control
→ knowledge-conditioned OPD
→ train-set audit
→ A0: horizon
→ A4: KL signal localization
→ A4b: semantic audit
→ A1: SFT horizon
→ next: Prefix-Support Swap
```

---

## 当前最重要的结果

### 1. 这些知识本身是能学会的

SFT 正对照已经说明：

- Chemistry：C2 相比 C0 有明显提升；
- Idiom：C1 / C3 也有明显提升。

所以后面 OPD 效果弱，不能简单解释成“0.6B 学生模型学不会”。

### 2. 当前 OPD 只能吸收一小部分收益

knowledge-conditioned OPD 的确有提升，但明显小于 SFT：

- Idiom O1：约 `+0.108`
- Chemistry O2：约 `+0.010`

而对应的 SFT 提升明显更大。

### 3. 单纯延长训练没有解决问题

A0 把 OPD 拉到 5 个 pass：

- Idiom 有一些继续提升，但很快饱和；
- Chemistry 仍然基本不动。

### 4. KL 并没有完全“打偏”

A4 发现，Teacher 词汇提示真正改变分布的位置，和 OPD KL 较大的位置有明显重合。

这说明“有用信号全被无关 token 淹没”不是一个充分解释。

### 5. 评测误差存在，但解释不了大差距

A4b 用语义判断重新检查 Chemistry：

- strict evaluator 的确会漏掉一部分合理同义表达；
- 但修正后 OPD 的提升仍然很小；
- SFT 的大幅提升仍然存在。

### 6. SFT 学得非常快

A1 进一步看 1–5 pass 的学习曲线：

- Idiom：P1 已经拿到大部分收益，P2 基本接近最终结果；
- Chemistry：P1 明显提升，P2 后基本饱和。

因此“OPD 只是训练得不够久”已经不是主要解释。

---

## 下一步

下一步准备做 Prefix-Support Swap。

核心问题很直接：

> 如果保持相同的 soft-KL，只改变训练时所在的前缀轨迹，Teacher-supported prefix 是否会显著提高知识迁移效率？

这个实验用来区分：

- 问题主要出在 student 自己生成的轨迹支持；
- 还是 soft-KL / update 本身就不够有效。

---

## 项目介绍

这个仓库最早用于复现和改造 MT-PATCHER，并逐步加入 SeqKD、OPD、WA/PDS、targeted knowledge transfer 等实验。

现在仓库同时保留两部分内容：

1. 早期复现、排错和框架实验；
2. 当前围绕 targeted OPD 机制问题展开的新实验。

旧代码和旧 run 仍然保留，主要用于复盘历史结果；新的实验尽量通过更明确的脚本、manifest 和结果目录来组织。
