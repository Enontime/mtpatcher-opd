# 研究问题与方法关系

## 先看文件

当前结果：

```text
docs/research/01_current_status.md
```

完整实验过程：

```text
docs/research/02_experiments.md
```

实验登记：

```text
docs/research/02_experiment_registry.md
```

历史路线：

```text
docs/research/04_historical_experiment_timeline.md
```

Targeted 机制实验：

```text
manifests/experiments/targeted/README.md
```

## 三个核心问题

### Q0：MT-PATCHER 在当前模型上还能不能成立

主要看：

```text
PE
PDS
WA
selective efficiency
```

也就是：

> Qwen3 Student 已经会大部分普通翻译以后，挑出真正不会的知识再做 patch，是否比普通蒸馏更有效？

当前状态：

- PE 单独提升较弱；
- PDS 有明确正证据；
- WA general-MT 增益没有稳定复现；
- source selection density 仍高，论文级 selective efficiency 尚未复现。

### Q1：Full OPD 能不能替代 Full SeqKD

在相同 Student、Teacher、Broad20k source population 和 3 source passes 下比较：

```text
offline sequence distillation
vs
on-policy Teacher-top-k forward KL
```

当前结果：

```text
Full OPD P3      19.1036
Full SeqKD P3    19.1652  (historical frozen anchor)
```

因此 OPD 在当前 MT setting 下具备很强的 full-budget replacement 潜力。

### Q2：为什么低预算 / targeted OPD 仍不够高效

已经看到：

```text
K2 Selective OPD4960 ≈ Random OPD4960
```

以及：

```text
targeted SFT >> targeted OPD
```

所以问题从：

> OPD 能不能工作？

收缩成：

> 哪些 source / state / prefix 才真正适合通过 OPD 把局部 MT knowledge 传给 Student？

## WHAT to teach 与 HOW to teach

### MT-PATCHER：WHAT to teach

PE：

```text
找到 Student 真正翻错的地方
```

PDS：

```text
围绕已知错误重新生成相关 context
```

WA：

```text
从已知错误扩展到相似或潜在错误
```

它们主要解决：

> 哪些知识值得继续教？

### SeqKD / OPD：HOW to teach

SeqKD：

```text
Teacher 先生成完整目标序列
Student 在 Teacher sequence 上做 supervised learning
```

OPD：

```text
Student 自己 rollout
Teacher 沿 Student trajectory 提供 soft distribution
Student 用 KL 更新
```

它们主要解决：

> 找到知识以后，用什么监督路径把它写入 Student？

Full OPD 的成功说明 HOW 这一侧不是根本失效。

Low-budget / targeted OPD 的弱结果说明：

> WHAT 与 HOW 的接口还没有设计好。

## 为什么 K2 selection 对 OPD 可能不够

SeqKD 与 OPD 实际学习的 state 不同：

```text
SeqKD:
(x, y^T_<t)

OPD:
(x, y^S_<t)
```

所以：

```text
“这个 source 上 Student 有错误”
```

不自动等价于：

```text
“Student 当前 on-policy trajectory 上有高价值 Teacher signal”
```

这与：

```text
Selective OPD4960 17.9600
Random OPD4960    17.9861
```

基本持平是一致的。

以后真正的 OPD-aware selection 可能需要看：

```text
Teacher-Student disagreement
trajectory KL
Student uncertainty
recoverability
local lexical error state
```

但应先由机制实验决定什么 signal 真正重要。

## 为什么 targeted 分支重要

Idiom / Chemistry 分支把“BLEU 为什么没涨”缩成可定位的知识迁移问题。

它先确认：

```text
这些知识 SFT 能不能学？
```

答案是能，而且很快。

随后确认：

```text
同样知识用 OPD 能不能学？
```

答案是能一点，但远弱于 SFT。

再逐个排除：

```text
是不是 pass 太少？       → 不是主要原因
是不是 KL 全落错位置？   → 不是
是不是 evaluator 误判？  → 只能解释一小部分
```

下一步 Prefix-Support Swap 直接控制 prefix trajectory。

这条分支的地位是：

> 为 Q2 提供可解释、低成本的机制实验。

它不是整个项目的新主线，也不是 WA 项目的替代品。

## PDS-OPD 为什么仍值得做

PDS 已经证明，围绕 error knowledge 构造新 context 有明显 downstream utility。

自然的组合是：

```text
error / correction
→ PDS 构造 X'
→ Student 在 X' 上 rollout
→ Teacher 沿 Student trajectory 给 soft supervision
```

问题变成：

> PDS 扩展的新 context 是否更容易把 Student 带到有价值的 on-policy states？

也就是：

```text
Extend WHAT to teach
+
Improve HOW to teach
```

但它应该建立在 prefix / trajectory bottleneck 更清楚之后，而不是直接把大规模 PDS 数据塞进 OPD。

## 项目背景

MT-PATCHER 的核心动机是：强 Student 的错误通常稀疏而局部，因此不必对所有数据做等量蒸馏，可以定位错误、扩展相关知识，再有针对性地 patch。

当前项目把这一思想迁移到 Qwen3 Student / Teacher，并进一步研究 OPD 是否能成为更自然的知识传递方式。

最终希望回答的是：

> **Selective / extendable translation knowledge 应该在什么 state 上、用什么 supervision 传给 Student，才能真正获得数据效率。**
