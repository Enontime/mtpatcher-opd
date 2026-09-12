# 实验登记

## 路径入口

通用训练 / 评测：

```text
/workspace/mtpatcher/runs/sft
/workspace/mtpatcher/runs/opd
/workspace/mtpatcher/runs/science
```

诊断：

```text
/workspace/mtpatcher/runs/diagnostics
/workspace/mtpatcher/runs/overnight
```

Strong reproduction / 历史 pipeline：

```text
/workspace/mtpatcher/runs/mtpatcher_v3_full6565_20260823
scripts/mtpatcher_v3
scripts/mtpatcher_v10
scripts/mtpatcher_v11
scripts/mtpatcher_v14
```

Targeted：

```text
/workspace/mtpatcher/runs/targeted
manifests/experiments/targeted/ARTIFACT_PATHS.md
```

正式配置：

```text
configs/sft/
configs/opd/
recipes/sft/
recipes/opd/
manifests/experiments/
```

## A. Foundation

| ID | 实验 | 主要路径 | 结果 / 状态 |
|---|---|---|---|
| F0 | Base Student | `/workspace/mtpatcher/models/Qwen3-0.6B` | Macro BLEU 17.3485 |
| F1 | Historical Full SeqKD20k | historical / `runs/sft` | P3 ≈ 19.1652 |
| F2 | Recovered Formal SeqKD P1 | `runs/science/seqkd_p1_rope_fix_final_20260909_v1` | 18.5177，恢复正控制 |
| F3 | Canonical Full OPD | `runs/opd/canonical_fkl_topk_broad20k_qwen3_06b_8b_20260903_034058` + recovery | P3 19.1036 |
| F4 | Canonical SeqKD vs OPD eval | `runs/science/` | evaluator / RoPE root cause 已关闭 |

Foundation 结论：

```text
Full OPD 20000   19.1036
Full SeqKD 20000 19.1652
```

Full-budget OPD 已基本恢复 Full SeqKD 的收益。

## B. Strong Reproduction / Selection

| ID | 实验 | Source / treatment | Macro BLEU | 结论 |
|---|---|---|---:|---|
| S0 | Base | none | 17.3485 | anchor |
| S1 | K1All / PE | 11,792 selected parents | 17.5139 | PE 单独提升较弱 |
| S2 | K2 | 4,960 strict selected | 17.3887 | K2 source utility 很弱 |
| S3 | RandomK1 | matched historical control | 17.3127 | selection 对照 |
| S4 | Random SeqKD | random 4,960 | 18.2658 | equal-budget KD |
| S5 | Same-source SeqKD | K2 4,960 | 18.4151 | 同 source Teacher-target control |
| S6 | Full SeqKD | 20,000 | 19.1652 | full anchor |

Broad20k：

```text
K1 = 58.96%
K2 = 24.8%
```

当前 selective density 仍高于论文目标量级。

## C. PDS / WA

### PDS

| ID | 实验 | Rows | Macro BLEU |
|---|---|---:|---:|
| P0 | K1All / PE | 11,792 parents | 17.5139 |
| P1 | K1All + PDS | 68,917 | 18.5696 |
| P2 | Parent-matched repeat | 68,917 | 16.7592 |

主要对比：

```text
PDS - Repeat  = +1.8104
PDS - Base    = +1.2211
PDS - K1All   = +1.0557
```

PDS treatment 是 strong reproduction 中最明确的正机制。

### WA

General-MT 两 seed：

```text
seed1 ≈ -0.0127
seed2 ≈ +0.0131
mean  ≈ 0
```

登记结论：

```text
general-MT WA incremental BLEU: no robust gain
targeted unseen-word / error-anticipation mechanism: unresolved
```

## D. Low-budget OPD

| ID | 实验 | Source budget | Macro BLEU | 结论 |
|---|---|---:|---:|---|
| O-FULL | Full OPD | 20,000 | 19.1036 | strong |
| O-SEL | Selective OPD | 4,960 | 17.9600 | 34.8% Full-OPD gain recovery |
| O-RND | Random OPD | 4,960 | 17.9861 | 36.3% recovery |

```text
Selective - Random = -0.0261
```

当前 K2 selection 对 OPD 没有可见价值。

低预算 OPD 同时弱于 SeqKD：

```text
Random SeqKD - Random OPD ≈ +0.2797
Same-source SeqKD - Selective OPD ≈ +0.4551
```

## E. Targeted knowledge transfer

精确 artifact：

```text
manifests/experiments/targeted/ARTIFACT_PATHS.md
```

### C0 / C1 / C2 / C3

| Arm | 方法 | Target | 主要结果 |
|---|---|---|---|
| C0 | frozen Student | baseline | Chemistry .097; Idiom 2.613 |
| C1 | SFT | Idiom | Idiom 3.132 (+.519) |
| C2 | SFT | Chemistry | Chemistry .261 (+.164) |
| C3 | SFT | combined | Idiom 3.141 (+.528); Chemistry .261 |

### O1 / O2 / O3

| Arm | 方法 | Target | 主要结果 |
|---|---|---|---|
| O1 | knowledge-conditioned OPD | Idiom | +.108 |
| O2 | knowledge-conditioned OPD | Chemistry | +.010 |
| O3 | knowledge-conditioned OPD | combined | Idiom +.103; Chemistry +.009 |

### Train-set audits

```text
Idiom:
C0 2.680
O1 2.746 (+.066)
C1 3.368 (+.688)

Chemistry:
C0 .087
O2 .097 (+.010)
```

弱 OPD learning 在 train examples 上就存在。

### A0：OPD Horizon-5

```text
Idiom P1..P5:
+.120, +.128, +.135, +.127, +.141

Chemistry P5:
约 +.009
```

结论：多训几轮不是共同主因。

### A4：KL Signal Localization

```text
256 Chemistry rows
6929 tokens
Pearson  ≈ .596
Spearman ≈ .729
```

top10% hint-sensitive tokens：

```text
49.14% OPD KL mass
93.21% hint-gap mass
```

结论：简单 dense-KL dilution 解释不够。

### A4b：Semantic Audit

```text
C0    .183
O2P5  .200
C2    .509
```

结论：string evaluator / judge noise 存在，但不能解释 OPD-SFT gap。

### A1：SFT Horizon-5

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

结论：direct sequence supervision 在 1–2 pass 就吸收大部分知识。

### 下一项：Prefix-Support Swap

```text
planned / not launched
```

比较：

```text
Student-prefix soft-KL
vs
Teacher-supported-prefix soft-KL
```

## F. Historical / closed branches

这些保留用于解释研究路线，不进入当前主结果表：

```text
scripts/pilot_v2/
scripts/mtpatcher_v4/
scripts/mtpatcher_v5/
scripts/mtpatcher_v6/
scripts/mtpatcher_v7/
scripts/mtpatcher_v8/
scripts/mtpatcher_v9/
top-level mtpatcher_ecropd_* / recovery probes
```

其中包括 custom full-vocab FKL / RKL、EC-ROPD、correction / repair、teacher-leg / recovery probes、早期 GRPO / PEGRL-inspired controls。

这些实验帮助发现 state / trajectory / correction 相关问题，但和当前 canonical Verl top-k FKL 的算法语义不同，不能混成一条结果。

## 说明

这个文件只负责“有哪些实验、在哪里、结果是什么”。

为什么这样设计、结果怎样连接，看：

```text
docs/research/02_experiments.md
```

历史顺序看：

```text
docs/research/04_historical_experiment_timeline.md
```
