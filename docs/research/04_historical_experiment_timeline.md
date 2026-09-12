# 历史实验时间线

## 历史资产在哪

早期 pilot：

```text
scripts/pilot_v2/
/workspace/mtpatcher/runs/pilot_v2_*
```

Strong reproduction：

```text
scripts/mtpatcher_v3/
scripts/mtpatcher_v11/
scripts/mtpatcher_v14/
/workspace/mtpatcher/runs/mtpatcher_v3_full6565_20260823
```

早期 OPD / correction：

```text
scripts/mtpatcher_v4/
scripts/mtpatcher_v5/
scripts/mtpatcher_v6/
scripts/mtpatcher_v7/
scripts/mtpatcher_v8/
scripts/mtpatcher_v9/
```

Selection OPD：

```text
scripts/mtpatcher_v10/
```

Canonical SeqKD / OPD：

```text
configs/
recipes/
/workspace/mtpatcher/runs/sft
/workspace/mtpatcher/runs/opd
/workspace/mtpatcher/runs/science
```

Targeted mechanism：

```text
scripts/targeted/
/workspace/mtpatcher/runs/targeted
manifests/experiments/targeted/
```

## 8 月中下旬：Pilot-V2 与最初 OPD

最早阶段主要确认 Qwen3-0.6B Student 能否训练、Teacher / Student 的翻译质量差距、human SFT / SeqKD 是否有提升，以及 OPD / GRPO 是否值得继续。

这一阶段留下：

```text
scripts/pilot_v2
runs/pilot_v2_sft
runs/pilot_v2_grpo
```

今天应把它们看作 feasibility / history，不是最终 comparator。

## 8 月 23–27 日：PE / PDS / WA pipeline 与 custom OPD

围绕：

```text
PE → PDS → WA
```

做数据构造、Feedbacker、repair、train/eval；同时尝试 custom Torch-NPU forward KL、reverse KL、PGRKL 等。

重要认识：

- pipeline 能跑；
- Student 可以通过 direct supervision 提升；
- early custom OPD full-scale 不稳定；
- trainer/debug 问题与 method 问题必须分开。

## 8 月 27–29 日：EC-ROPD / repair 与第一次重置

重点探索 error-conditioned state、local correction、resume、teacher leg。

matched-512 有过小正信号，但 full-scale 没形成稳定主效应。

8 月 29 日路线重置回：

```text
先做可靠 reproduction / positive control
再回答 OPD replacement
```

当时的 6565 positive control 包括：

```text
Full SeqKD6565       +1.465 BLEU
Selected SeqKD3732   +1.182
Equal-budget3732     +1.050
```

这说明 Student 能学，但 selective efficiency 尚未达到论文水平。

## 8 月 30–31 日：PDS 强正结果

Broad20k / K1 / K2 逐渐冻结：

```text
Base                  17.3485
K1All                 17.5139
Random SeqKD4960      18.2658
SameSource SeqKD4960  18.4151
Full SeqKD20k         19.1652
```

PDS 对照：

```text
K1All + PDS           18.5696
Matched Repeat        16.7592
difference            +1.8104
```

PDS 因而成为 strong reproduction 中最明确的机制正证据。

同时发现：

```text
PE+PDS = 68,917 Student rows
```

已经大于 Full SeqKD20k，因此 row-efficiency 叙事必须放弃。

## 9 月初：WA 通用 MT 没有稳定增益

WA 两 seed：

```text
约 -0.0127
约 +0.0131
```

均值接近 0。

旧的“WA mechanism failed”说法应废弃。当前准确表述是：

```text
general-MT WA gain 未复现
targeted unseen-word mechanism 仍 unresolved
```

## 9 月 2–3 日：迁移到 Verl

师兄建议后，项目从大量 custom trainer 转向：

```text
Verl
+
thin MT adapter
+
config / recipe / manifest
```

正式建立：

```text
configs/
recipes/
manifests/
scripts/data/
scripts/infra/
tests/
```

Canonical SeqKD 完成 3 pass。

Canonical OPD 第一次 run 在 Pass1 后中断，保留 `global_step_1250`。

## 9 月 4–5 日：Canonical OPD recovery

逐项确认 model、optimizer、scheduler、RNG、dataloader position 能通过 Verl 原生 checkpoint/resume 恢复。

随后从 P1 checkpoint 继续到 P3。

这一步把“OPD 能不能完成 formal run”从工程问题变成可回答的科学问题。

## 9 月 6–9 日：SeqKD parity 深挖，最后发现 RoPE evaluator bug

经历 reduction、scheduler、order、optimizer、gradient、FSDP、microbatch 等诊断后，真正的关键来自一个矛盾：

> step400 权重 exact，但 greedy translation 大量不同。

最后定位到 Transformers 新旧版本 RoPE config schema：

```text
new checkpoint:
rope_parameters.rope_theta = 1e6

old evaluator:
未正确消费
→ rope_theta = 1e4
```

只修 config 语义后：

```text
Formal SeqKD P1     18.5177
Historical P1       18.5203
delta               ≈ -0.0026
```

因此旧的“大 BLEU gap 来自 trainer execution stack”解释被更新：主 gap 实际是 evaluator / config compatibility。

## 9 月 9 日：Full OPD 重新评测后恢复

同样修 RoPE semantic 后：

```text
OPD P1 18.3390
OPD P2 18.6950
OPD P3 19.1036
```

P3 相对 Base `+1.7551`，与 Historical Full SeqKD `19.1652` 只差约 0.06。

从这时起，“OPD 整体无效”不再是合理主叙事。

## 9 月 9–10 日：低预算 OPD 与 K2 selection

先得到：

```text
Selective OPD4960 = 17.9600
```

它低于：

```text
Same-source SeqKD4960 = 18.4151
```

随后 Random OPD4960：

```text
17.9861
```

最终：

```text
Selective - Random = -0.0261
```

K2 selection 没有为 OPD 提供额外价值。

问题因此改写成：

> static error-selected source 是否和 on-policy high-value state 对齐？

## 9 月 10 日：导师建议 Idiom / Chemistry

导师建议直接在成语和化学术语上检查 WA knowledge，同时观察 OPD 训练过程。

因此 targeted 分支启动。

它的定位是：

```text
WA / OPD knowledge-transfer mechanism diagnostic
```

不是整个项目的新名字。

## 9 月 10–11 日：Targeted SFT positive control

```text
Chemistry +.164
Idiom     +.519
```

说明这些 knowledge gap 对 Student 是可学习的，后续 OPD 弱不能再归因于 Student capacity、数据完全无效或 test 不敏感。

## 9 月 11 日：Knowledge-conditioned OPD

```text
Idiom     +.108
Chemistry +.010
```

真实迁移存在，但远小于 SFT。

Train-set audit 又显示差距在 seen examples 上已经存在。

## 9 月 11–12 日：A0 / A4 / A4b / A1

A0：延长 OPD 到 5 pass，Idiom 略增后饱和，Chemistry 基本不动。

A4：Teacher hint 改变 distribution 的位置与 OPD KL 显著重合，简单 dense-KL dilution 解释不成立。

A4b：semantic audit 找到 evaluator noise 和同义表达，但 OPD 的真实提升仍小。

A1：SFT 在 P1–P2 已拿到大部分最终 gain，说明知识本身容易被 direct sequence supervision 写入。

## 9 月 12 日：当前状态

当前最有价值的机制问题已经收缩到：

```text
Student-generated prefix
是否限制 soft Teacher knowledge 的有效迁移？
```

下一实验：

```text
Prefix-Support Swap
```

它先区分：

```text
prefix / trajectory support
vs
soft-KL objective / update
```

然后才决定是否进入 local lexical bridge、PDS-OPD 或 OPD-aware selection。

## 这条时间线怎么读

旧结论如果和后期更强证据冲突，以后期为准。

重要更新包括：

```text
“OPD 可能整体无效”
→ Full OPD 19.1036 后不再成立

“SeqKD/OPD 低 BLEU 是 trainer execution gap”
→ RoPE evaluator bug 关闭后修正

“WA general BLEU≈0 所以 WA mechanism failed”
→ 改为 targeted unseen-word mechanism unresolved

“K2 selected source 应该适合 OPD”
→ Random vs Selective OPD4960 后缺乏支持
```

保留历史文件和 run，是为了让今天的结论能够追溯到它怎样被证据一步步修改出来。
