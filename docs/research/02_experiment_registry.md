# 实验登记与科研总结

最后一次主要更新：2026-09-12

本文是人工可读的科研总结。

当前 定向 实验的**正式状态与 来源追溯**
以：

`manifests/experiments/targeted/`

中的 README、INDEX 和逐实验 manifest 为准。

原始运行结果位于：

`/workspace/mtpatcher/runs/`

大型数据位于：

`/workspace/mtpatcher/data/`

## 总研究问题

能否用 On-Policy Distillation（OPD）替代或改造 MT-PATCHER
中的 sequence distillation，在保持数据效率的同时减少 流程 复杂度？

当前 定向 机制实验进一步隔离一个更窄的问题：

> 对于已经证明能被 ordinary SFT 学会的 定向翻译知识，
> OPD 为什么只能吸收其中很小一部分？

## 当前主要实验矩阵

| ID | 方法 | Target | 状态 | 主要结论 |
|---|---|---|---|---|
| C0 | Frozen Qwen3-0.6B | 基线 | PASS | 固定 定向 与 通用机器翻译 baseline |
| C1 | SFT | 成语 | PASS | 成语知识可明显学习 |
| C2 | SFT | 化学 | PASS | 化学知识可明显学习 |
| C3 | SFT | 成语 + 化学 | PASS | 两个域均可学习 |
| O1 | 3-pass 知识条件 OPD | 成语 | PASS | 有正向但较弱的 transfer |
| O2 | 3-pass 知识条件 OPD | 化学 | PASS | 正向极弱 |
| O3 | 3-pass 知识条件 OPD | Combined | PASS | 正向但明显弱于 SFT |
| A0 | 5-pass OPD horizon ablation | O1/O2 | PASS | 增加 pass 不能解决主要差距 |
| A4 | KL 信号定位 | 化学 | PASS | 教师模型 hint shift 与 OPD KL 有较强重叠 |
| A4b | Semantic lexical audit | 化学 | PASS with caveat | evaluator 有噪声，但不是主要解释 |
| A1 | SFT horizon-5 control | 成语 + 化学 | PASS | SFT 在 P1–P2 已吸收大部分收益 |
| 08 | Prefix-Support Swap | Mechanism 诊断 | NEXT | 下一步机制因果诊断 |

## 已建立的关键结果

### 化学 留出集

- C0：0.097
- C2 SFT：0.261，delta +0.164
- C3 SFT：0.261，delta +0.164
- O2 OPD：0.107，delta +0.010
- O3 OPD：0.106，delta +0.009

### 成语 留出集

- C0：2.613
- O1 OPD：2.721，delta +0.108
- O3 OPD：2.716，delta +0.103

### 成语 train1000

- C0：2.680
- C1 SFT：3.368，delta +0.688
- C3 SFT：3.352，delta +0.672
- O1 OPD：2.746，delta +0.066
- O3 OPD：2.726，delta +0.046

Matched 训练集 comparison 表明：

OPD 的弱学习已经出现在训练样本上，
因此主要问题不是 留出集泛化失败。

### General MT cost

相对 C0 的 Macro BLEU delta：

- C1 SFT：-0.812508
- C2 SFT：-2.171510
- C3 SFT：-0.842484
- O1 OPD：-0.585807
- O2 OPD：-0.094100
- O3 OPD：-0.408422

当前可以概括为：

> SFT 能更强地注入 定向知识，但对 通用机器翻译 的破坏更大。

> OPD 更保守，对 通用机器翻译 的副作用更小，但 定向知识 uptake 明显偏弱。

## Horizon 诊断结论

A0 已完成。

主要结果：

- 从 3 pass 增加到 5 pass，没有显著缩小 定向 OPD 与 SFT 的差距；
- 因此单纯增加训练轮数不再是主要方向。

A1 同样已经完成。

SFT 在 化学 与 成语 上都显示：

- P1 已获得大部分最终收益；
- P2 基本接近最终饱和。

这说明当前 定向知识 本身可以快速被模型吸收。

## 当前机制判断

现有证据提高了以下假说的优先级：

> OPD 的瓶颈可能位于 supervision pathway，
> 尤其是 软 KL 被施加在哪些 学生模型 trajectory / prefix states 上。

但这一机制尚未被正式证明。

## 下一步

下一实验：

`08 — Prefix-Support Swap`

科学类别：

`DIAGNOSTIC ONLY`

目的：

在尽量保持相同 软 KL 条件下，
比较 学生模型-supported 与 教师模型/reference-supported prefix replay，
判断 prefix support 是否会显著影响 定向知识 transfer。

如果 support hypothesis 成立：

进入 localized correction bridge + resumed 学生模型 rollout。

如果不成立：

转向 objective / gradient effectiveness 的诊断。

大规模 PDS-OPD 暂不优先启动。
