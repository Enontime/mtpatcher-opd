# Targeted experiment index

## 先查路径

所有 targeted experiment 的 data / model / script / run / log：

```text
manifests/experiments/targeted/ARTIFACT_PATHS.md
```

机器可读 manifest：

```text
00_c0_baseline.json
01_sft_positive_control.json
02_knowledge_conditioned_opd.json
03_trainset_audits.json
04_a0_opd_horizon5.json
05_a4_kl_localization.json
06_a4b_semantic_audit.json
07_a1_sft_horizon5.json
INDEX.json
```

代码说明：

```text
scripts/targeted/README.md
```

整个项目状态：

```text
docs/research/01_current_status.md
```

## 当前顺序

| 编号 | 实验 | 状态 | 一句话结果 |
|---|---|---|---|
| 00 | C0 baseline | 完成 | frozen targeted baseline |
| 01 | SFT positive control | 完成 | Idiom / Chemistry 都可明显学习 |
| 02 | knowledge-conditioned OPD | 完成 | 有真实迁移，但明显弱于 SFT |
| 03 | matched train-set audits | 完成 | gap 在 seen train examples 上已存在 |
| 04 | A0 OPD horizon-5 | 完成 | 延长 pass 不能救 Chemistry |
| 05 | A4 KL localization | 完成 | hint-sensitive token 与 OPD KL 显著重合 |
| 06 | A4b semantic audit | 完成 | evaluator noise 不是主因 |
| 07 | A1 SFT horizon-5 | 完成 | SFT 1–2 pass 已吸收大部分知识 |
| 08 | Prefix-Support Swap | 下一步 | 检查 prefix / trajectory support |

## 核心结果

SFT positive control：

```text
Chemistry:
C0 .097
C2 .261
Δ +.164

Idiom:
C0 2.613
C1 3.132
Δ +.519
```

Knowledge-conditioned OPD：

```text
O1 Idiom      +.108
O2 Chemistry  +.010
O3 Idiom      +.103
O3 Chemistry  +.009
```

Train audit：

```text
Idiom:
C0 2.680
O1 2.746 (+.066)
C1 3.368 (+.688)

Chemistry:
C0 .087
O2 .097 (+.010)
```

A0：

```text
Idiom best improvement around +.14
Chemistry P5 around +.009
```

A4：

```text
Pearson  ≈ .596
Spearman ≈ .729
```

A4b：

```text
C0 semantic    .183
O2P5 semantic  .200
C2 semantic    .509
```

A1：

```text
Idiom SFT:
P1 +.441
P2 +.536
P5 +.550

Chemistry SFT:
C0 .097
P1 .233
P2 .293
P5 .298
```

## 这条线不是什么

`targeted` 不是整个 MT-PATCHER 项目。

它是从下面几个问题中派生出来的机制分支：

```text
WA general-MT signal weak
+
low-budget OPD weak
+
SFT / OPD knowledge-transfer gap
```

完整研究还包括：

```text
Strong Reproduction
PE / PDS / WA
Full SeqKD
Full OPD
Random / Selective OPD
PDS-OPD
```

完整结构看：

```text
docs/research/00_research_overview.md
```

## 下一步

Prefix-Support Swap 保持：

```text
same rows
same lexical hint
same top-k FKL
same token budget
same optimizer
same update count
same C0
```

只改变 prefix trajectory。

它决定后续应该优先走：

```text
local lexical bridge
```

还是继续检查：

```text
soft-KL objective / update efficiency
```
