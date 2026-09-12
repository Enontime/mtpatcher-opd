# 当前研究状态

## 先看结果在哪

通用 SFT / SeqKD：

```text
/workspace/mtpatcher/runs/sft
```

正式 OPD：

```text
/workspace/mtpatcher/runs/opd
```

统一评测与科学对照：

```text
/workspace/mtpatcher/runs/science
```

机制诊断：

```text
/workspace/mtpatcher/runs/diagnostics
```

Targeted knowledge transfer：

```text
/workspace/mtpatcher/runs/targeted
```

Targeted 的精确 artifact：

```text
manifests/experiments/targeted/ARTIFACT_PATHS.md
```

当前 frozen config：

```text
configs/sft/canonical_seqkd_broad20k_qwen3_06b_8b.yaml
configs/opd/canonical_fkl_topk_broad20k_qwen3_06b_8b.yaml
```

## 一页状态表

| 研究块 | 状态 | 当前结论 |
|---|---|---|
| Base | 完成 | Macro BLEU 17.3485 |
| Full SeqKD | 完成 | Historical P3 19.1652 |
| Full OPD | 完成 | P3 19.1036，基本追平 Full SeqKD |
| PE / source selection | 已评估 | K2 density 24.8%，未复现论文级 selective efficiency |
| PDS | 强正结果 | 18.5696；比 parent-matched repeat 高 +1.8104 |
| WA general MT | 弱 / 零结果 | 两 seed mean 约 0；unseen-word mechanism 仍待 targeted 验证 |
| Random vs Selective OPD4960 | 完成 | 17.9861 vs 17.9600；K2 selection 对 OPD 没有价值证据 |
| Targeted SFT | 完成 | Idiom / Chemistry 都能明显学习 |
| Targeted OPD | 完成 | 有真实迁移，但远弱于 SFT |
| A0 horizon | 完成 | 延长到 P5 不能救 Chemistry |
| A4 KL localization | 完成 | hint-sensitive token 与 OPD KL 显著重合 |
| A4b semantic audit | 完成 | evaluator noise 存在，但不是主因 |
| A1 SFT horizon | 完成 | SFT 1–2 pass 已拿到大部分收益 |
| Prefix-Support Swap | 下一步 | 检查 prefix / trajectory support |

## 1. Foundation

### Base

```text
Macro BLEU = 17.3485
```

### Full SeqKD

Historical Full SeqKD Broad20k：

```text
Macro BLEU ≈ 19.1652
gain vs Base ≈ +1.8167
```

RoPE compatibility 修复后，Formal SeqKD P1 恢复到 `18.5177`，与 historical P1 `18.5203` 只差约 `0.0026` BLEU，说明 comparator / evaluator stack 已重新对齐。

### Full OPD

| pass | Macro BLEU | vs Base |
|---|---:|---:|
| P1 | 18.3390 | +0.9905 |
| P2 | 18.6950 | +1.3465 |
| P3 | 19.1036 | +1.7551 |

```text
Full OPD 19.1036
Historical Full SeqKD 19.1652
gap ≈ 0.06
```

Full-budget OPD 已经是可靠的强结果。

## 2. Strong Reproduction：PE / PDS / WA

### PE / K1 / K2

```text
K1 = 11,792 / 20,000
K2 =  4,960 / 20,000
```

K2 仍占 24.8%，没有达到原论文更激进的 low-density selection。

| 方法 | Macro BLEU |
|---|---:|
| K1All / PE | 17.5139 |
| Random SeqKD 4960 | 18.2658 |
| Same-source SeqKD 4960 | 18.4151 |
| Full SeqKD 20000 | 19.1652 |

当前证据说明：source selection 有一定信息，但仅靠 source selection 解释不了 SeqKD 的大部分收益，target treatment 很重要。

### PDS

```text
K1All + PDS                  18.5696
Parent-matched repeat        16.7592
PDS - Repeat                 +1.8104
PDS - Base                   +1.2211
```

这条结果是 strong reproduction 里最稳定的机制正证据。

数据量：

```text
PE parents       11,792
PDS rows         57,125
total rows       68,917
```

所以不能把这条结果写成“比 20k Full SeqKD 更省训练 rows”。

### WA

```text
seed1 ≈ -0.0127
seed2 ≈ +0.0131
mean  ≈ 0
```

当前结论只到：

> WA 在 general-MT benchmark 上没有稳健 incremental BLEU。

原论文 targeted unseen-word / error-anticipation mechanism 仍需要专门测试。

## 3. Low-budget OPD / Selection

| 方法 | Macro BLEU | gain vs Base |
|---|---:|---:|
| Selective OPD 4960 | 17.9600 | +0.6115 |
| Random OPD 4960 | 17.9861 | +0.6376 |
| Random SeqKD 4960 | 18.2658 | +0.9173 |
| Same-source SeqKD 4960 | 18.4151 | +1.0666 |

关键差值：

```text
Selective OPD - Random OPD = -0.0261
Random SeqKD - Random OPD  ≈ +0.2797
Same-source SeqKD - Selective OPD ≈ +0.4551
```

所以：

1. K2 selection 没有提高 OPD 的低预算效率；
2. 4960-source OPD 当前比相同预算的 SeqKD 更弱；
3. Full OPD 很强，因此问题集中在 **低预算 source selection / trajectory coverage**，不是 OPD 整体无效。

## 4. Targeted knowledge transfer

详细路径：

```text
manifests/experiments/targeted/ARTIFACT_PATHS.md
```

### SFT positive control

Chemistry：

```text
C0 = 0.097
C2 = 0.261
Δ  = +0.164
```

Idiom：

```text
C0 = 2.613
C1 = 3.132
Δ  = +0.519

C3 = 3.141
Δ  = +0.528
```

SFT general-MT cost：

```text
C1 Δ Macro BLEU ≈ -0.813
C2 Δ Macro BLEU ≈ -2.172
C3 Δ Macro BLEU ≈ -0.842
```

### Knowledge-conditioned OPD

```text
O1 Idiom      +0.108
O2 Chemistry  +0.010
O3 Idiom      +0.103
O3 Chemistry  +0.009
```

General MT cost：

```text
O1 Δ Macro BLEU ≈ -0.586
O2 Δ Macro BLEU ≈ -0.094
O3 Δ Macro BLEU ≈ -0.408
```

稳定对比：

> SFT 写入 targeted knowledge 更强，但副作用更大；OPD 更保守，但知识写入明显更弱。

### Train-set audit

```text
Idiom train1000:
C0 2.680
O1 2.746 (+.066)
C1 3.368 (+.688)

Chemistry train1000:
C0 .087
O2 .097 (+.010)
```

弱 OPD learning 在 seen examples 上就存在。

### A0：OPD horizon

Idiom held-out：

```text
P1 +.120
P2 +.128
P3 +.135
P4 +.127
P5 +.141
```

Chemistry 到 P5 仍只有约 `+.009`。单纯增加 pass 不是共同主瓶颈。

### A4：KL signal localization

256 个 Chemistry 样本、6929 tokens：

```text
Pearson(H, KL)  ≈ .596
Spearman(H, KL) ≈ .729
```

hint-sensitive top10% token 承担约：

```text
49.1% OPD KL mass
93.2% hint-gap mass
```

所以“有用知识完全被 dense KL 淹没”不是主要解释。

### A4b：semantic audit

```text
C0 semantic       .183
O2P5 semantic     .200   (+.017)
C2 semantic       .509   (+.326)
```

consistency correction 后，O2P5 约 `+.020`。评测噪声存在，但无法解释 OPD 与 SFT 的巨大差距。

### A1：SFT horizon

Idiom：

```text
P1 +.441
P2 +.536
P5 +.550
```

Chemistry：

```text
C0 .097
P1 .233
P2 .293
P5 .298
```

直接 sequence supervision 在 1–2 pass 就能吸收大部分知识。

## 5. 当前下一步

下一步：

```text
Prefix-Support Swap
```

同一批数据、同一 lexical hint、同一 top-k forward KL，只改变 prefix trajectory：

```text
S arm: Student prefix
T arm: Teacher-supported prefix
```

匹配：

```text
rows
token budget
optimizer
LR
top-k
update count
C0 initialization
```

解释：

```text
T >> S
→ prefix / trajectory support 是重要瓶颈

T ≈ S
→ 更应该查 soft-KL objective / update 本身
```

它回答以后，再决定是否进入 local lexical bridge 或 PDS-OPD。

## 如何理解这页

这页只记录“今天知道什么”。

历史上被修正的结论放在：

```text
docs/research/04_historical_experiment_timeline.md
```

更详细的实验逻辑放在：

```text
docs/research/02_experiments.md
```
