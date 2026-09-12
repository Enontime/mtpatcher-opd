# 当前研究概览

## 先看文件

当前研究相关的主要入口：

```text
docs/research/02_experiments.md
manifests/experiments/targeted/ARTIFACT_PATHS.md
scripts/targeted/README.md
```

实验数据和运行结果：

```text
/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823
/workspace/mtpatcher/runs/targeted
```

学生模型：

```text
/workspace/mtpatcher/models/Qwen3-0.6B
```

Teacher：

```text
/workspace/mtpatcher/models/Qwen3-8B
```

---

## 研究问题

目前最关心的是：

> OPD 能不能把局部、明确、可验证的翻译知识有效地迁移给学生模型？

我们希望把这个问题从“最终 BLEU 有没有涨”进一步拆开。

当前主要看两个知识域：

- Chemistry：化学术语；
- Idiom：中文成语。

这样做的好处是，很多错误可以具体定位到词汇或短语，而不是把所有变化都混在一个整体 BLEU 里。

---

## 目前得到的认识

### 知识可学习

SFT 正对照已经证明这些知识可以被 C0 学生模型学习，而且通常学得很快。

### OPD 的知识吸收明显更弱

knowledge-conditioned OPD 能带来真实提升，但只获得 SFT 的一小部分收益。

### 训练轮数不是主要原因

把 OPD 继续训练到 5 个 pass 并没有明显改善 Chemistry，Idiom 也很快饱和。

### KL 信号不是完全没有落到关键位置

A4 的 token-level 分析表明，Teacher hint 改变分布较大的位置，也承载了相当多 OPD KL。

### evaluator 不是主因

A4b 证明 strict substring evaluator 的确有误差，但改成语义判断后，OPD 与 SFT 的差距依旧很大。

### 当前最值得查的是 prefix / trajectory support

SFT 在 Teacher-supported target trajectory 上学习，而 OPD 主要在 Student 自己生成的 trajectory 上做 KL。

接下来需要直接控制这个变量，而不是继续泛化地加训练轮数。

---

## 下一步

下一步做 Prefix-Support Swap：

```text
same rows
same Teacher lexical knowledge
same top-k soft KL
same token budget
same optimizer / updates
only swap prefix trajectory
```

如果 Teacher-supported prefix 明显更有效，说明 trajectory support 很可能是主要瓶颈之一。

如果仍然无效，就应该继续查 soft-KL update 本身的优化效率。

---

## 背景

这个方向来自 MT-PATCHER 的 sequence distillation / patching 思路。

当前工作已经不再局限于逐行复刻原实现，而是更关注：

- 哪些机制真的产生收益；
- OPD 能否替代一部分 sequence distillation；
- 局部翻译知识到底怎样进入学生模型；
- 怎样在较小数据量下获得更高效的迁移。
