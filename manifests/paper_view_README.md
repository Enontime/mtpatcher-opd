# MT-PATCHER Paper View

本目录是 MT-PATCHER 当前研究结果的整理视图，用于论文写作、结果核对和后续复现。

`paper_view` 只保留已经确认、仍参与当前科学论证的实验资产。原始 run、失败尝试、调试记录和完整 provenance 继续保存在原始目录中。

## 1. 目录说明

| 目录 | 内容 | 主要用途 |
|---|---|---|
| `experiments/00_foundation/` | Full SeqKD、Full OPD | 主基线与完整训练对照 |
| `experiments/01_mtpatcher/` | PE、PDS、PDS repeat、WA | MT-PATCHER 主流程结果 |
| `experiments/02_data_efficiency/` | 4960 条数据下的 SeqKD / OPD 对照 | 数据效率与 source-matching 分析 |
| `experiments/03_targeted/` | Targeted baseline、SFT control、knowledge-conditioned OPD | 定向错误与知识条件实验 |
| `analysis/` | learning curve、KL localization、prefix 系列、semantic robustness | 机制分析与补充证据 |
| `tensorboard/` | 经过整理的正式训练曲线 | 训练过程检查与作图 |

推荐阅读顺序：

```text
00_foundation
→ 01_mtpatcher
→ 02_data_efficiency
→ 03_targeted
→ analysis
```

## 2. 实验资产

### Foundation

`experiments/00_foundation/` 保存 Full SeqKD 与 Full OPD 的基础对照。

当前正式对照为 `matched20k_v2_7500`：

| 项目 | 设置 |
|---|---:|
| Unique training sources | 20,000 |
| Source passes | 6 |
| Total source exposures | 120,000 |
| Global batch size | 16 |
| Formal optimizer steps | 7,500 |

SeqKD 和 OPD 使用相同的 source population、source order 和 source exposure budget。

7500-step 正式结果：

| Method | WMT24 BLEU | FLORES BLEU | Challenge BLEU | Macro BLEU | Macro chrF |
|---|---:|---:|---:|---:|---:|
| Full SeqKD | 17.175 | 21.762 | 19.302 | 19.413 | 49.879 |
| Full OPD | 17.225 | 22.539 | 19.727 | 19.830 | 50.260 |

因此正式 matched endpoint 下，Full OPD 相比 Full SeqKD：

- Macro BLEU: `+0.417`
- Macro chrF: `+0.381`

训练曲线显示 SeqKD 前期提升更快，OPD 后期追上。

另外将 OPD 从 7500 延长到 10000 steps 做了训练充分性诊断。
该实验标记为 `DIAGNOSTIC_BUDGET_EXTENSION`，不改变正式 7500-step endpoint。
在 extension 中 Teacher-probe 继续明显提高，而 benchmark 已进入平台和振荡区。

### MT-PATCHER

`experiments/01_mtpatcher/` 保存：

- `pe_all_11792`
- `pds_augmented_68917`
- `pds_parent_repeat_control_68917`
- `wa_general_mt_seed1`
- `wa_general_mt_seed2`

这里用于比较 PE、PDS、WA 各阶段对最终翻译性能的贡献。

### Data Efficiency

`experiments/02_data_efficiency/` 保存：

- random SeqKD 4960
- same-source SeqKD 4960
- random OPD 4960
- selective OPD K2 4960

这组实验用于比较低数据预算下的 SeqKD / OPD，并区分数据量、source matching 与选择策略的影响。

### Targeted Experiments

`experiments/03_targeted/` 保存：

- targeted baseline
- SFT positive controls
- knowledge-conditioned OPD
- Chemistry / Idiom 的逐样本评测结果
- general-MT sanity evaluation

聚合指标应能够下钻到对应的 case-level 输出。

## 3. Analysis

`analysis/` 只保留当前仍参与论文论证的分析，不作为历史诊断实验全集。

当前包括：

| 分析 | 目的 |
|---|---|
| `opd_learning_curve_5pass/` | 比较 OPD 随训练 pass 的变化 |
| `sft_learning_curve_5pass/` | 与 SFT learning curve 对照 |
| `kl_signal_localization/` | 检查 KL 信号集中位置 |
| `semantic_robustness/` | 检查 targeted 结论的语义稳健性 |
| `prefix_support/` | 检查不同 prefix state 下的支持信号 |
| `prefix_uptake/` | 检查 student 是否吸收 teacher signal |
| `prefix_release/` | 检查移除 prefix 后能力是否保留 |

## 4. TensorBoard

当前默认只展示：

- Full SeqKD
- Full OPD

`tensorboard/` 中的 clean event 只用于科研阅读。原始 TensorBoard event 保持不变，并通过 provenance 记录来源与 SHA256。

## 5. 原始资产与追溯

真实实验资产保存在：

```text
/workspace/mtpatcher/runs/
/workspace/mtpatcher/data/
/workspace/mtpatcher/models/
```

`paper_view` 中的稳定名称通过：

```text
manifests/paper_view.json
```

映射到真实物理路径。

需要追溯一个结果时，按下面顺序：

```text
paper_view 中的实验名
→ manifests/paper_view.json
→ 原始 run
→ config / checkpoint / evaluation output
```

## 6. 不进入 Paper View 的内容

默认不展示：

- smoke / failed / invalid run
- engineering gate / preflight / debug
- closure / reconciliation
- PID、lock、nohup 等运行态文件
- worker logs、临时 shards、resume state
- demo / Human6565
- 被后续版本替代的中间 run
- 只有 preregistration、尚无正式结果的实验

这些内容可以继续保留在原始目录和 Git 历史中用于追溯。

## 7. 实验文档约定

面向阅读的实验文档保持简洁：

- 正文直接说明问题、设置、结果和结论；
- 主要实验设置和结果优先使用表格；
- 逐样本结果单独保存，不堆进正文；
- 完整 prompt 放在附录，正文只说明 prompt 的用途和版本；
- prompt 必须完整保留，不用省略号截断；
- 工程排错过程不写成长篇实验叙事；
- 一个结论对应一组清楚的证据，不重复解释同一件事。

目标是让研究者能够快速回答三个问题：

1. 这个实验在验证什么？
2. 结果是什么？
3. 对应的原始证据在哪里？

## 8. 生成与检查

`paper_view` 是生成目录，不应手工维护。

重新生成：

```bash
python scripts/tools/sync_paper_view.py --apply
```

完整性检查：

```bash
python scripts/tools/sync_paper_view.py --check
```

只有 `--check` 通过后，当前 `paper_view` 才视为有效。
