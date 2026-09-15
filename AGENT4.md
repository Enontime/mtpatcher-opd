# AGENT4.md

> MT-PATCHER / OPD 项目的第四版操作合同。
> 更新日期：2026-09-15。
>
> 优先级：
>
> AGENTS.md < AGENT2.md < AGENT3.md < AGENT4.md
>
> AGENT4 不替代 AGENT3 的科研合同。
> AGENT3 中关于科学规格、reproduction/adaptation 分离、artifact freeze、
> evaluation contract、Git、安全、长任务可观测性与恢复性的规则继续全部有效。
>
> 当规则冲突时，执行更严格的规则；项目管理和论文级代码组织方面以 AGENT4 为准。

---

## 0. 最高操作优先级：先维护好项目，再继续扩张实验

当前项目已经进入需要长期维护和最终论文交付的阶段。

因此，在决定“下一步先做什么”时，项目管理、代码组织、实验资产可读性和最终论文视图具有最高操作优先级。

这条优先级不允许覆盖科学诚信、实验规格和安全约束；它约束的是工作顺序和项目组织。

如果当前存在以下问题，应先处理：

- 正式实验与 smoke/debug 混杂；
- 同一实验有多个难以辨认的版本入口；
- TensorBoard 无法区分正式实验和工程测试；
- 最终结果找不到对应 run/checkpoint；
- 代码、数据、run、结果缺少稳定入口；
- 人类打开项目后无法快速看懂当前论文主线；
- paper-facing 资产中混入失败、调试或历史噪声。

除非用户明确结束项目/目录审计，否则不要自行切换到新的科研实验或其他审计任务。

---

## 1. 最终目标：项目应像“提交论文时的代码仓库”

默认给研究者、审稿人或未来接手者看的项目视图，应尽量接近最终论文代码发布状态。

默认视图只应突出：

- 最终正式实验；
- 论文正文或附录需要的 matched controls；
- 与科学结论直接相关的 analysis / ablation；
- 最终 evaluation results；
- 有解释价值的 TensorBoard 曲线；
- 复现这些结果所需的正式代码、配置和说明。

内部工程历史不应污染默认视图。

---

## 2. 四层真源

### 2.1 代码真源

唯一代码真源：

/workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend

所有新实现继续进入正常仓库结构，例如：

- scripts/
- configs/
- recipes/
- manifests/
- docs/

禁止为了整理、展示或 paper view 再复制一套源码。

### 2.2 实验资产真源

真实实验资产继续保存在：

- /workspace/mtpatcher/runs/
- /workspace/mtpatcher/data/
- /workspace/mtpatcher/models/

这些物理目录可以包含历史版本、失败尝试和工程实验。

它们是 provenance 真源，不等于默认人类视图。

### 2.3 论文收录真源

决定哪些资产进入论文级视图的唯一 allowlist：

manifests/paper_view.json

禁止建立第二套并行的人工论文资产清单。

### 2.4 人类论文视图

生成后的论文级视图：

/workspace/mtpatcher/paper_view

paper_view 是生成物。

禁止手工修改 paper_view。

---

## 3. paper_view 的唯一维护方式

paper_view 只能通过：

manifests/paper_view.json

和：

scripts/tools/sync_paper_view.py

维护。

修改 paper_view manifest 后必须执行：

python scripts/tools/sync_paper_view.py --apply
python scripts/tools/sync_paper_view.py --check

只有 --check PASS，论文视图才视为有效。

任何 Agent 都不得直接：

- mkdir 一个新的 paper_view 实验；
- 手工创建 paper_view symlink；
- 手工复制文件进入 paper_view；
- 手工删除 paper_view 中的实验；
- 通过修改生成目录绕过 manifest。

---

## 4. 新代码不会自动改变论文视图

正常开发流程：

代码修改
→ 工程验证
→ 正式实验
→ 结果评测
→ 科学解释
→ 决定是否进入论文
→ 如需进入，修改 paper_view manifest
→ apply
→ check

代码完成、训练完成、checkpoint 存在，都不足以让实验进入 paper_view。

paper_view 是科学接纳层，不是运行目录索引。

---

## 5. Paper-facing admission gate

只有同时满足以下条件的资产才可以进入 paper_view：

- 实验科学问题明确；
- scientific specification 已冻结；
- 数据和 comparator 合法；
- evaluator semantics 合法；
- 最终结果已经确认；
- 这是该实验族最终科学有效版本；
- 论文正文或附录可能引用；
- 读者需要它理解或复现一个科学结论。

加入前必须问：

“如果今天就提交论文，读者是否需要这个资产来理解或复现论文中的一个科学 claim？”

答案为否：不加入。

答案不确定：默认不加入，并等待显式科学决策。

---

## 6. 永远不进入 paper_view 的内容

以下内容不进入最终论文视图：

- smoke；
- failed run；
- invalid run；
- engineering gate；
- preflight；
- debug；
- execution closure；
- reconciliation；
- 单纯用于证明某个工程异常的实验；
- demo；
- Human6565 substrate/demo；
- 只有 preregistration、尚无正式结果的实验；
- superseded intermediate version；
- parser/debug/repair 的中间产物；
- 临时诊断目录；
- 无论文解释价值的 TensorBoard。

这些资产可以继续留在真实物理目录、日志和 Git 历史中。

“不进入 paper_view”不等于删除 provenance。

---

## 7. 一个科学实验族只暴露一个稳定身份

真实物理目录可以存在：

v1
v2
v3
recovery
recovery2
final
final2

但论文视图应尽量只看到：

一个稳定、人类可读、代表最终科学结果的名字。

例如：

selective_opd4960

而不是：

selective_opd4960_v3_final_recovery2

如果旧版本被新正式版本取代：

- 保留旧物理 run；
- 更新 manifest target；
- paper_view 稳定名字不变。

只有版本差异本身构成科学变量时，才允许同时暴露多个版本。

---

## 8. Analysis 的收录标准同样严格

analysis 目录不是“所有诊断实验集合”。

只有可能出现在论文正文、图表、附录或论证链中的最终分析才进入 paper_view/analysis。

例如，一个分析如果回答：

- 为什么 targeted OPD 弱于 targeted SFT；
- KL signal 位于哪里；
- prefix support 是否解释 gap；
- 一个机制假设是否被实验排除；

并且该结果仍参与当前科学论证，则可以保留。

如果一个实验只用于：

- 修复 scheduler；
- 查 reduction；
- 查 RoPE 工程异常；
- 验证 launcher；
- 查 shell；
- 查 checkpoint merge；

其结论应写进日志、current status 或历史记录，不进入 paper-facing analysis。

---

## 9. TensorBoard 采用同样的论文标准

默认 TensorBoard 视图只暴露：

- 正式训练；
- 对论文解释有实际价值的曲线。

不得把：

- smoke；
- failed；
- debug；
- preflight；
- historical engineering attempts；

混入默认论文 TensorBoard。

未来正式训练应优先使用：

<RUN>/tensorboard

作为该 run 独有目录。

不同：

- arm；
- seed；
- recovery；

不得混写同一 event 目录。

---

## 10. 原仓库必须始终保持可开发

paper_view 只是视图，不能反过来控制源码布局。

Agent 修改方法时继续正常修改：

- scripts/
- configs/
- recipes/
- manifests/
- tests/
- docs/

不要为了 paper_view：

- 改源码路径；
- 复制 trainer；
- 创建 paper 专用重复代码；
- 让 symlink 成为代码依赖。

正式复现入口应最终来自正常仓库中的清晰 script/config/recipe。

---

## 11. 项目管理变更必须低风险

项目整理优先使用：

- symlink；
- manifest；
- generated view；
- README；
- workspace view。

在仅为提高可读性时，优先不移动真实历史资产。

未经用户明确授权，不要因为“整理目录”而：

- 大规模 mv runs；
- 删除 checkpoint；
- 删除历史结果；
- git clean；
- git reset；
- 重写 frozen artifact。

项目视图和物理 provenance 必须解耦。

---

## 12. Git 中区分三类变化

原则上区分：

1. 方法实现变化；
2. 实验结果/科学状态变化；
3. paper_view 收录变化。

例如：

方法代码完成
→ 一个 implementation commit

正式实验完成并冻结结果
→ 一个 research/result commit

决定进入论文视图
→ 一个 paper-view promotion commit

不要把大量无关整理、实验代码和结果收录揉成一个提交。

继续遵守 AGENT3：

禁止 git add .

只显式暂存需要提交的文件。

---

## 13. 文档职责必须稳定

默认权威入口：

AGENT4.md
→ 当前最高操作合同

AGENT3.md
→ 科研和工程硬合同

docs/research/01_current_status.md
→ 当前科学状态、当前工作和下一步

docs/research/02_experiments.md
→ 实验设计和科学解释

docs/research/03_repository_map.md
→ 原始仓库目录职责

docs/research/04_historical_experiment_timeline.md
→ 历史演进

manifests/paper_view.json
→ 最终论文视图收录名单

不要创建多个职责重叠的“current status”“asset index”“final index”。

---

## 14. AGENT3 的当前状态章节视为历史快照

AGENT3 中长期科研和工程规则继续有效。

但 AGENT3 的：

- Current scientific state；
- Current targeted-data state；
- Required experiment matrix；
- Current milestone order；

记录的是 2026-09-10 的状态。

它们不得覆盖更新后的项目状态。

当前科学状态优先读取：

docs/research/01_current_status.md

如果存在更晚 HANDOFF，则继续读取 HANDOFF。

实际服务器状态最终以实时文件系统、日志、进程和 frozen artifact 为准。

不要因为 AGENT3 的旧 milestone 仍写着某实验“下一步”，就重新启动已经完成、被否定或被替代的研究分支。

---

## 15. 当前工作上下文的恢复顺序

新的 Agent 或新的上下文窗口进入项目时，应优先读取：

1. AGENT4.md
2. AGENT3.md
3. docs/research/01_current_status.md
4. 最新 HANDOFF
5. 与当前任务直接相关的 manifest / spec / result
6. 必要时检查服务器实时状态

不要仅凭聊天摘要或文件名推测项目状态。

---

## 16. 项目管理优先原则

当项目可读性已经明显下降时：

先恢复结构和真源关系
→ 再继续科学实验。

一个项目只有在以下问题清楚时才算可维护：

- 代码在哪里；
- 数据在哪里；
- 最终 run 是哪个；
- 最终 checkpoint 是哪个；
- 结果在哪里；
- 哪些是论文资产；
- 哪些只是内部 provenance；
- TensorBoard 哪些值得看；
- 新实验如何进入最终视图；
- 被替代版本如何退出默认视图。

目录大小不是首要指标。

人类可读性、科学身份和长期维护性优先。

---

## 17. 面向最终论文代码的判断原则

整理项目时，始终假设：

“明天要把这个仓库交给论文审稿人和未来研究者。”

如果一个目录、名字或资产在这种情况下会让读者困惑，就应该重新设计默认视图。

但不要为了好看破坏原始 provenance。

目标是：

内部历史完整
+
默认视图简单
+
科学身份唯一
+
代码真源唯一
+
最终结果可复现。

---

## 18. AGENT3 科研规则继续具有硬约束

以下 AGENT3 原则不得因项目管理优先而弱化：

- 实验前写 Question / Competing explanations / Falsifiable prediction / Decision after result；
- reproduction 与 adaptation 分开；
- 禁止 hidden protocol drift；
- positive control 是 gate；
- evaluation semantics 是 checkpoint contract 的组成部分；
- frozen artifact 不原地修改；
- Engineering PASS 不等于 Scientific PASS；
- 正式实验必须保存 provenance；
- 长任务必须 observable、resumable；
- single-writer safety；
- shell / server safety；
- evidence over plausible explanation。

项目管理服务于这些科学规则，而不是绕过这些规则。

---

## 19. 默认决策

当不确定一个东西应该进入哪里时：

正式论文 claim 所需
→ paper_view

当前科学工作所需，但尚未成为最终结果
→ 正常 repo / runs / data / models

工程测试
→ 原始工程位置，不进入 paper_view

失败或无效实验
→ 保留必要 provenance，不进入 paper_view

旧版本
→ 保留物理资产，不进入默认视图

纯排错 closure
→ 写入日志/历史，不进入论文资产

当不确定是否应该进入 paper_view：
默认不进入。

---

## 20. 最终目标

目标不是建立最大的实验索引。

目标是让任何新的 Agent、合作者或未来的自己，在几分钟内回答：

- 项目当前在研究什么；
- 正式代码在哪里；
- 最重要的实验有哪些；
- 论文结果来自哪些 run；
- 当前证据支持什么；
- 下一步为什么值得做；
- 怎样添加新代码而不重新污染项目；
- 怎样把一个正式结果安全地晋升到论文视图。

项目管理应降低未来科研的认知负担，而不是制造新的目录体系。
