# 实验记录

## 文件在哪

当前状态：

```text
docs/research/01_current_status.md
```

实验登记：

```text
docs/research/02_experiment_registry.md
```

服务器结果：

```text
/workspace/mtpatcher/runs
```

Targeted 精确路径：

```text
manifests/experiments/targeted/ARTIFACT_PATHS.md
```

历史代码代际：

```text
scripts/mtpatcher_paper_repro/
scripts/mtpatcher_paper_faithful_v2/
scripts/mtpatcher_rq0/
scripts/mtpatcher_v3/
...
scripts/mtpatcher_v14/
```

下面按“这个实验回答了什么”记录，不按脚本数量罗列。

## 1. SeqKD positive control：先确认 Student 能学

最先要回答的是：

> Qwen3-0.6B 能不能从更强 Teacher 的翻译监督中明显提升？

答案是能。

```text
Base                   17.3485
Historical FullSeqKD   19.1652
gain                   +1.8167
```

因此后续 OPD / PE / PDS 的弱结果不能简单归因于“Student 太小，学不动”。

## 2. PE / source selection：挑错句本身够不够

Broad20k：

```text
K1 11,792 / 20,000
K2  4,960 / 20,000
```

冻结结果：

```text
K1All               17.5139
K2                  17.3887
RandomK1            17.3127
RandomSeqKD4960     18.2658
SameSourceSeqKD4960 18.4151
FullSeqKD20000      19.1652
```

最有信息量的差异是：

```text
SameSourceSeqKD4960 - K2 ≈ +1.026
```

同一批 source，只换成 Teacher sequence target，效果就大幅变好。

所以问题从“source 选得对不对”推进成“找到 source 以后，target treatment / sequence supervision 怎样才有效”。

## 3. PDS：第一次明确看到“扩展知识”有效

PDS 围绕已发现的 error/correction pair 构造新 bilingual context。

为避免把“更多 parent exposure”误当成 PDS 效果，使用 parent-matched repeat control：

```text
K1All + PDS          18.5696
Matched Repeat       16.7592
difference           +1.8104
```

三个 benchmark 同方向，chrF 也同步提高。

这支持：

> PDS treatment 带来的 context / target / knowledge re-instantiation 有实际 utility。

但它没有隔离出“context diversity alone”，因为 PDS 同时改变 source context、target 和 correction 的重新实例化。

数据量也必须一起记：

```text
11,792 PE parents
+ 57,125 PDS rows
= 68,917 rows
```

因此这不是 row-efficiency 结果。

## 4. WA：通用 BLEU 没给出稳健增益

当前 general-MT 两 seed：

```text
-0.0127
+0.0131
```

平均基本为 0。

能支持的结论：

> 当前 adaptation 下，WA 在 WMT24 / FLORES / Challenge 上没有稳健 incremental BLEU。

不能写成“WA mechanism failed”，因为原论文最强 WA 证据来自 controlled unseen-word setup。

后来转向 Idiom / Chemistry targeted test，也正是为了避免通用 benchmark 对 lexical / factual knowledge 不敏感。

## 5. 早期 OPD：custom trainer 暴露了 state / trajectory 问题

迁到 Verl 前做过：

```text
full-vocab FKL
RKL
EC-ROPD
correction / repair
teacher-leg
recovery horizon
```

多数 full-scale 结果 weak / null / negative，少数 local recovery / teacher-leg probe 有正 signal。

这批实验最重要的价值，是让我们看到下面这些变量可能很关键：

```text
Student trajectory
prefix failure
corrected state
Teacher / Student support
```

但旧 custom objective 与 canonical Verl teacher-top-k FKL 不同，因此只作为历史机制证据，不进入 formal OPD 主结论。

## 6. 迁到 Verl：把训练基础设施从研究问题中剥离

师兄建议后，SFT / OPD 通用训练尽量交给成熟框架。

当前 formal stack 采用 Verl，项目代码保留 MT-specific 部分：

```text
MT dataset adapter
response-only mask
PE / PDS / WA
selection
evaluation
analysis
config / manifest
```

SFT response-only supervision、checkpoint、FSDP 等逐步通过验证。

这一步让后续研究可以回到 method，而不是继续维护自建 trainer。

## 7. SeqKD / OPD 同时掉分：真正根因是 evaluator 的 RoPE 配置

迁移后 SeqKD 与 OPD 曾同时出现异常低 BLEU。

因为 SeqKD 是正控制，所以两条线一起异常时，优先检查共同 execution/evaluation contract。

最终定位到：

```text
new Transformers checkpoint:
rope_parameters.rope_theta = 1e6

old evaluator:
未正确消费该 schema
→ resolved rope_theta = 1e4
```

同一组权重因此生成不同翻译。

只修 evaluator 侧 `rope_theta=1e6` 后：

```text
Recovered Formal SeqKD P1 = 18.5177
Historical SeqKD P1       = 18.5203
delta                     ≈ -0.0026
```

正控制恢复。

这一步也更新了旧解释：此前的大 BLEU gap 不能继续归因于 data order / scheduler / FSDP 等训练执行细节。

## 8. Full OPD：full-budget replacement 基本成立

修复评测后，Canonical Full OPD：

```text
P1 18.3390
P2 18.6950
P3 19.1036
```

Historical Full SeqKD P3：

```text
19.1652
```

只差约 `0.06 BLEU`。

因此当前可以说：

> 在 full Broad20k source budget 下，OPD 基本恢复了 Full SeqKD 的 Student improvement。

这一步把研究推进到真正重要的问题：怎样让 OPD 在低预算、selected knowledge 上也高效。

## 9. Selective OPD4960：第一轮 data-efficient OPD 不成立

K2 selected sources：

```text
4960 / 20000 = 24.8%
```

Selective OPD：

```text
17.9600
```

同 source SeqKD：

```text
18.4151
```

差：

```text
-0.4551
```

这是明确负信号，但单看这一条还无法区分“selection 不好”与“低预算 OPD 本来就弱”。

## 10. Random OPD4960：把 selection 问题真正收口

matched Random OPD4960：

```text
Random OPD4960    17.9861
Selective OPD4960 17.9600
difference        -0.0261
```

两者基本持平。

所以当前 K2 criterion 对 OPD 没有价值证据。

同时：

```text
Random SeqKD4960  18.2658
Random OPD4960    17.9861

SameSource SeqKD  18.4151
Selective OPD     17.9600
```

低预算下 SeqKD 当前更强。

研究问题因此改写为：

```text
旧问题：
能不能把 MT-PATCHER K2 直接接到 OPD？

新问题：
什么样的 on-policy state 才是真正高价值的训练位置？
```

## 11. Targeted SFT positive control：先确认知识本身能不能写进去

导师建议直接用 Idiom / Chemistry test 检查 WA knowledge。

第一步先做 SFT positive control。

Chemistry：

```text
C0 .097
C2 .261
+ .164
```

Idiom：

```text
C0 2.613
C1 3.132
+ .519
```

这排除了 Student capacity、数据完全无效、target test 完全不敏感等简单解释。

## 12. Knowledge-conditioned OPD：真实迁移存在，但远弱于 SFT

Teacher 侧提供 lexical knowledge、Student 仍只看 source：

```text
O1 Idiom      +.108
O2 Chemistry  +.010
O3 Idiom      +.103
O3 Chemistry  +.009
```

OPD 的确迁移了一些 targeted knowledge，但效率远低于 SFT。

## 13. Train-set audit：差距不是 held-out 才出现

Idiom train1000：

```text
C0 2.680
O1 2.746 (+.066)
C1 3.368 (+.688)
```

Chemistry train1000：

```text
C0 .087
O2 .097 (+.010)
```

OPD 在 seen examples 上就没有把知识充分写进去，因此“训练集学得很好，只是泛化差”不成立。

## 14. A0：多训到 5 pass 也救不了 Chemistry

Idiom：

```text
P1 +.120
P5 +.141
```

Chemistry：

```text
P5 约 +.009
```

3 pass 不够不是两个 domain 的共同主因。

## 15. A4：KL 没有完全落错位置

256 条 Chemistry、6929 tokens：

```text
Pearson  ≈ .596
Spearman ≈ .729
```

hint-sensitive top10% token 承担约：

```text
49.1% OPD KL mass
93.2% hint-gap mass
```

“dense KL 把有用 lexical signal 全稀释掉”这个解释因此被大幅削弱。

## 16. A4b：semantic evaluator 能修一点，但修不了主结论

```text
C0    .183
O2P5  .200
C2    .509
```

OPD 从 strict 到 semantic 后回收了一点真实提升，但仍远小于 SFT。

实际翻译同时存在真实 lexical repair 与 regression，因此后续必须继续读具体例子，不能只看 aggregate accuracy。

## 17. A1：SFT 前 1–2 pass 已经学得差不多

Idiom：

```text
P1 +.441
P2 +.536
P5 +.550
```

Chemistry：

```text
P1 .233
P2 .293
P5 .298
```

目标知识对 Student 并不难写入。

但 A1 不能单独证明“prefix support 是因果瓶颈”，因为 SFT 与 OPD 同时改变 trajectory 和 hard CE / soft KL。

## 18. 下一步：Prefix-Support Swap

只改变：

```text
Student prefix
vs
Teacher-supported prefix
```

两边保持：

```text
same rows
same lexical hint
same top-k forward KL
same token budget
same optimizer
same LR
same updates
same C0
```

如果：

```text
Teacher-prefix >> Student-prefix
```

说明 trajectory support 是重要瓶颈。

如果：

```text
Teacher-prefix ≈ Student-prefix
```

就把注意力转向 soft-KL objective / update efficiency。

结果出来后再决定要不要进入：

```text
local lexical bridge
PDS-OPD
OPD-aware selection
```

## 总结

到目前为止：

```text
Full OPD 可以工作
K2 selection 不能让低预算 OPD 高效
PDS 可以扩展有用知识
targeted SFT 可以快速写入知识
targeted OPD 写入效率低
```

下一步要解释的是：

> **Student 自己的 on-policy prefix 是否阻止了 soft Teacher knowledge 真正进入它不会的 lexical state。**
