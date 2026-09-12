# 实验记录

## 先看文件

每个实验的数据、脚本、run、日志位置统一看：

```text
manifests/experiments/targeted/ARTIFACT_PATHS.md
```

当前实验 run 主要在：

```text
/workspace/mtpatcher/runs/targeted
```

当前主要脚本在：

```text
scripts/targeted/
```

机器可读的实验记录在：

```text
manifests/experiments/targeted/
```

---

## 0. C0 baseline

C0 是冻结的 Qwen3-0.6B 学生模型。

主要基线：

- Chemistry strict：`0.097`
- Idiom mean：`2.613`

这组结果是后续 SFT / OPD 的共同参照。

---

## 1. SFT positive control

### 目的

先确认这些定向知识到底能不能被学生模型学会。

### 结果

Chemistry：

- C0：`0.097`
- C2：`0.261`
- 提升：`+0.164`

Idiom：

- C0：`2.613`
- C1：`3.132`
- 提升：`+0.519`

Combined C3 的 Idiom 结果约为：

- C3：`3.141`
- 提升：`+0.528`

### 怎么理解

这些知识本身是可学习的。

后面如果 OPD 提升很小，就不能简单解释成模型容量不足或数据完全无效。

---

## 2. Knowledge-conditioned OPD

### 目的

让 Teacher 获得额外词汇知识，Student 仍然只看原始 source，然后在 Student 自己生成的轨迹上做 forward KL。

### 结果

- O1 Idiom：`+0.108`
- O2 Chemistry：`+0.010`
- O3 Idiom：`+0.103`
- O3 Chemistry：`+0.009`

### 怎么理解

OPD 确实能迁移一些知识，但幅度明显小于 SFT。

这成为后续机制实验的主要问题：

> Teacher 明明知道正确知识，为什么 Student 只能吸收很小一部分？

---

## 3. Train-set audit

### 目的

确认 OPD 的弱提升是否只是 held-out 泛化问题。

### 结果

Idiom train：

- C0：`2.680`
- O1：`2.746`
- O1 提升：`+0.066`
- C1：`3.368`
- C1 提升：`+0.688`

Chemistry train：

- C0：`0.087`
- O2：`0.097`
- O2 提升：`+0.010`

### 怎么理解

差距在训练集上就已经存在。

所以问题不是“训练集学得很好，只是 held-out 泛化差”。

---

## 4. A0：OPD horizon-5

### 目的

检查 OPD 是否只是训练轮数不够。

### 结果

Idiom held-out：

- P1：约 `+0.120`
- P5：约 `+0.141`

Chemistry：

- held-out P5：约 `+0.009`
- train P5：约 `+0.009`

### 怎么理解

延长训练对 Idiom 有一点帮助，但很快饱和。

Chemistry 基本没有被救回来。

因此 horizon 不是主要公共瓶颈。

---

## 5. A4：KL signal localization

### 目的

检查有用知识是不是根本没有得到足够的 OPD KL 信号。

### 结果

在 256 个 Chemistry 样本上：

- Pearson：约 `0.596`
- Spearman：约 `0.729`

Teacher hint 最敏感的前 10% token：

- 承担约 `49.1%` 的 OPD KL mass；
- 承担约 `93.2%` 的 hint-gap mass。

### 怎么理解

有用知识相关位置并没有完全被 dense KL 淹没。

因此“KL 信号根本没落在关键位置”不是一个充分解释。

---

## 6. A4b：Chemistry semantic audit

### 目的

检查 strict canonical substring evaluator 是否低估了真实语义正确率。

### 结果

语义评测：

- C0：`0.183`
- O2P5：`0.200`
- C2：`0.509`

对应提升：

- O2P5：`+0.017`
- C2：`+0.326`

另外还发现少量 judge consistency noise，但规模不足以解释 OPD 与 SFT 的巨大差距。

### 实际翻译里看到什么

有些 strict miss 其实是合理同义表达，例如 canonical 名称和常用英文名称不同。

同时也存在真实修复，例如：

- `methylamine sulfoxide → methyl sulfonamide`
- `benzoic acid esters → benzoic acid anhydride`

也存在真实退化，例如：

- `Lutetium fluoride → lusite`
- `Montmorillonite → chlorite`

所以后续分析不能只看一个 accuracy 数字，需要继续读实际翻译。

---

## 7. A1：SFT horizon-5

### 目的

看 SFT 学同样知识需要多少 pass。

### Idiom

- P1：`+0.441`
- P2：`+0.536`
- P5：`+0.550`

P1 已经拿到绝大多数最终收益，P2 基本接近最终结果。

### Chemistry

- C0：`0.097`
- P1：`0.233`
- P2：`0.293`
- P5：`0.298`

Chemistry 也是前两轮学习最快。

### 怎么理解

这些知识在直接序列监督下非常容易被学到。

因此“OPD 只是需要更多轮”已经不再是一个有说服力的解释。

---

## 8. 下一步：Prefix-Support Swap

接下来直接控制 prefix trajectory。

比较：

```text
Student-prefix soft-KL
vs
Teacher-supported-prefix soft-KL
```

其他条件尽量保持一致：

```text
same rows
same token count
same Teacher lexical knowledge
same top-k KL
same optimizer
same update count
same C0 initialization
```

这个实验主要用来回答：

> OPD 的低效，究竟有多少来自 Student trajectory 本身不在正确 lexical support 上？

如果 Teacher-supported prefix 明显更强，再考虑做局部 lexical bridge。

如果仍然很弱，就继续查 soft-KL / gradient update 本身。

---

## 说明

这里记录的是实验之间的逻辑和主要结果。

具体路径、hash、日志、checkpoint、judge output 等信息不在这里展开，统一放在：

```text
manifests/experiments/targeted/ARTIFACT_PATHS.md
```
