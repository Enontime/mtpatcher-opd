# 研究总览

## 文件和路径

当前状态：

```text
docs/research/01_current_status.md
```

研究问题与方法关系：

```text
docs/research/01_research_overview.md
```

完整实验过程：

```text
docs/research/02_experiments.md
```

实验登记：

```text
docs/research/02_experiment_registry.md
```

历史时间线：

```text
docs/research/04_historical_experiment_timeline.md
```

仓库与服务器路径：

```text
docs/research/03_repository_map.md
```

Targeted 机制实验的精确 artifact：

```text
manifests/experiments/targeted/ARTIFACT_PATHS.md
```

正式 SeqKD / OPD 配置和 recipe：

```text
configs/sft/canonical_seqkd_broad20k_qwen3_06b_8b.yaml
configs/opd/canonical_fkl_topk_broad20k_qwen3_06b_8b.yaml

recipes/sft/canonical_seqkd_broad20k_qwen3_06b_8b.sh
recipes/opd/canonical_fkl_topk_broad20k_qwen3_06b_8b.sh
```

## 整个项目怎么分

```text
MT-PATCHER / OPD
│
├── A. Strong Reproduction
│   ├── Base / Full SeqKD
│   ├── PE / source selection
│   ├── PDS
│   └── WA
│
├── B. Canonical SeqKD / OPD
│   ├── Verl migration
│   ├── SeqKD positive control
│   ├── Full OPD
│   └── evaluator / RoPE compatibility closure
│
├── C. Data-efficient OPD
│   ├── Random SeqKD
│   ├── Selective SeqKD
│   ├── Random OPD
│   └── Selective OPD
│
├── D. Targeted mechanism diagnostics
│   ├── Idiom / Chemistry
│   ├── SFT positive control
│   ├── knowledge-conditioned OPD
│   ├── A0 / A4 / A4b / A1
│   └── Prefix-Support Swap
│
└── E. Future method extension
    ├── OPD-aware selection
    ├── PDS-OPD
    └── local / lexical bridge（若机制证据支持）
```

## A. Strong Reproduction

这部分回答：把 MT-PATCHER 放到当前 Qwen3 Student / Teacher 上，原方法里的关键机制还能不能成立。

冻结的通用 MT 结果：

| 方法 | Macro BLEU | 相对 Base |
|---|---:|---:|
| Base | 17.3485 | — |
| K1All / PE | 17.5139 | +0.1654 |
| Random SeqKD 4960 | 18.2658 | +0.9173 |
| Same-source SeqKD 4960 | 18.4151 | +1.0666 |
| K1All + PDS | 18.5696 | +1.2211 |
| Historical Full SeqKD 20000 | 19.1652 | +1.8167 |

Broad20k 上：

```text
K1 eligible = 11,792 / 20,000 = 58.96%
K2 strict   =  4,960 / 20,000 = 24.8%
```

这仍明显高于原论文更激进的 selective density，所以当前 strong reproduction 没有复现出论文级 source efficiency。

### PDS

PDS 是目前 strong reproduction 中最稳定的正结果：

```text
K1All + PDS                    18.5696
Parent-matched PE repetition   16.7592
difference                     +1.8104
```

它支持 PDS treatment 的实际 downstream utility。

但这条结果对应：

```text
11,792 PE parents
+ 57,125 accepted PDS rows
= 68,917 Student SFT rows
```

已经大于 Full SeqKD 20k，因此不能宣称 row efficiency。

### WA

General-MT 两 seed：

```text
约 -0.0127
约 +0.0131
mean 约 0
```

所以当前只能说：

> WA 在当前 general-MT benchmark 上没有稳健 incremental BLEU。

原论文的 controlled unseen-word / error-anticipation mechanism 仍未被这一结果否定。

## B. Canonical SeqKD / OPD

这部分回答：在相同 Student、Teacher、source population 和统一 evaluator 下，offline SeqKD 与 on-policy FKL 表现怎样。

SeqKD / OPD 一度同时出现异常低 BLEU，最后定位到 Transformers 新旧版本对 RoPE config schema 的解析不一致。修复 evaluator 侧的 `rope_theta` 语义后，SeqKD positive control 恢复。

Full OPD：

```text
P1 18.3390
P2 18.6950
P3 19.1036
```

Historical Full SeqKD P3：

```text
19.1652
```

Full-budget 上两者只差约 `0.06` BLEU。

这建立了一个关键地基：

> OPD 在 full source budget 下可以完成很强的 MT knowledge transfer。

## C. Data-efficient OPD

Full OPD 成立以后，问题转向：能不能只在 Student 更值得学的 source 上做 OPD。

4960-source 对照：

| 方法 | Macro BLEU |
|---|---:|
| Selective OPD 4960 | 17.9600 |
| Random OPD 4960 | 17.9861 |
| Random SeqKD 4960 | 18.2658 |
| Same-source SeqKD 4960 | 18.4151 |

Selective 与 Random OPD 差 `-0.0261` BLEU，基本持平。

因此当前已经可以关闭一个旧假设：

> K2 error selection 本身并没有提高 OPD 的低预算数据效率。

这不否定 Full OPD；它说明 selection criterion 与 OPD 的 on-policy state 之间还没有对齐。

## D. Targeted mechanism diagnostics

Targeted 线来自 WA / low-budget OPD 问题，是一个 **机制诊断分支**，不代表整个项目。

之所以选 Idiom 和 Chemistry，是因为知识缺口更容易定位和评价。

目前已经知道：

- 直接 SFT 可以快速把 targeted knowledge 写入 Student；
- knowledge-conditioned OPD 有真实提升，但明显更弱；
- 弱点在 seen training examples 上已经存在；
- 延长 OPD 到 5 pass 没有救 Chemistry；
- OPD KL 并没有完全避开知识相关 token；
- evaluator noise 也解释不了巨大差距；
- SFT 在 1–2 pass 内就能吸收大部分收益。

下一步 Prefix-Support Swap 直接比较：

```text
Student-prefix soft-KL
vs
Teacher-supported-prefix soft-KL
```

如果 Teacher-supported prefix 明显更强，才有证据继续发展 local lexical / correction bridge。

## E. 下一阶段

当前最有信息量的顺序：

```text
Prefix-Support Swap
    ↓
判断 trajectory support 是否是主要瓶颈
    ↓
若支持：local lexical / correction bridge
若不支持：soft-KL objective / update efficiency
    ↓
再回到 PDS-OPD / OPD-aware selection
```

PDS-OPD 仍是重要长期问题：PDS 负责扩展 WHAT to teach，OPD 负责 HOW to teach；要验证新 context 是否能把 Student 带到更有价值的 on-policy states。

## 项目背景

MT-PATCHER 的核心 premise 是：强 Student 已经会翻大部分普通数据，因此应优先定位真正不足的知识，再有针对性地扩展与蒸馏。

当前项目把这一思想迁移到 Qwen3 Student / Teacher，并研究 OPD 是否能成为更自然的知识传递方式。最终目标不是证明某个 trainer 更强，而是回答：

> **Selective / extendable translation knowledge 应该在什么 state 上、用什么 supervision 传给 Student，才能真正获得数据效率。**
