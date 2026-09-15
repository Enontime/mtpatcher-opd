<!-- AGENT4_POINTER_V1 -->

> 当前最高级项目合同：`AGENT4.md`
>
> 规则优先级：
>
> `AGENTS.md < AGENT2.md < AGENT3.md < AGENT4.md`
>
> 新 Agent 进入项目后，应先阅读 `AGENT4.md`，再阅读 `AGENT3.md` 和
> `docs/research/01_current_status.md`。

# MT-PATCHER Agent 长期规则

所有进入本仓库工作的 Agent，在修改代码、启动实验、整理结果之前，都必须先阅读本文件。

## 1. 项目的四层真源

### 代码真源

所有实现代码只维护在：

/workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend

新增代码应进入正常仓库目录，例如：

- scripts/
- configs/
- recipes/
- manifests/
- docs/

禁止为了展示或整理，再复制一套代码到其他目录。

### 实验资产真源

实验产生的原始资产继续保存在：

- /workspace/mtpatcher/runs/
- /workspace/mtpatcher/data/
- /workspace/mtpatcher/models/

这些目录保存真实实验、数据、模型、日志和 checkpoint。

### 论文视图清单

决定哪些资产进入最终论文视图的唯一清单是：

manifests/paper_view.json

### 论文视图

最终给人看的论文级视图是：

/workspace/mtpatcher/paper_view

paper_view 是自动生成目录，禁止手工修改。

维护命令：

python scripts/tools/sync_paper_view.py --apply
python scripts/tools/sync_paper_view.py --check

## 2. 什么可以进入 paper_view

一个实验完成，并不代表它自动进入 paper_view。

只有满足下面条件的最终版本才可以进入：

- 科学结果已经确认；
- 实验规格已经冻结；
- 是该实验族最终有效版本；
- 论文正文或附录可能需要引用；
- 读者需要它来理解或复现科学结论。

判断标准：

如果今天就提交论文，这个资产是否应该随论文代码一起交给读者？

如果答案不明确，默认不加入 paper_view。

## 3. 什么不能进入 paper_view

以下内容禁止进入 paper_view：

- smoke；
- failed / invalid attempt；
- engineering gate；
- preflight；
- debug；
- execution closure；
- reconciliation；
- demo / Human6565；
- 只有 preregistration、尚无正式结果的实验；
- 已被后续版本取代的中间版本；
- 单纯用于排查工程异常的实验。

这些内容可以继续留在原始 runs、日志和 Git 历史中用于追溯，但不出现在默认论文视图。

## 4. 一个实验族只保留一个稳定名字

同一个科学实验如果经历多个版本：

v1、v2、recovery、final、final2 等物理目录可以继续存在。

paper_view 中只暴露：

- 最终科学有效版本；
- 一个稳定、人类可读的实验名。

禁止把版本迭代历史直接暴露给论文读者，除非版本差异本身就是科学变量。

## 5. 新实验的标准流程

以后新增实验统一遵循：

代码修改
→ 工程验证
→ 正式实验
→ 结果评测
→ 科学判断
→ 决定是否进入论文
→ 如需进入，再修改 manifests/paper_view.json
→ 重新生成并检查 paper_view

任何新 run 都不得自动加入 paper_view。

## 6. Git 规则

禁止：

- git add .
- git clean
- 为整理目录做大范围 git reset
- 把无关文件一起暂存

必须显式指定研究文件。

代码实现、实验结果确认、论文视图收录，原则上分开处理。

## 7. 科研规格规则

工程问题、评测问题、科学方法变化必须明确区分。

修复工程问题时，不得偷偷修改 scientific specification。

Engineering PASS 不等于 Scientific PASS。

重要正式实验至少要能追溯：

- 数据身份；
- Student / Teacher；
- 代码版本；
- resolved config；
- seed；
- 训练预算；
- checkpoint；
- 评测输出；
- 最终结果。

## 8. 最终目录原则

整个项目最终应当像“提交论文时的代码仓库”。

默认给人看的内容只包括：

- 最终正式实验；
- 与论文结论直接相关的 analysis / ablation；
- 最终 results；
- 有用的 TensorBoard 曲线。

工程历史、失败尝试和排错过程留在后台追溯，不污染默认视图。
