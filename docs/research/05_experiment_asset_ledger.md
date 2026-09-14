# MT-PATCHER / OPD 实验资产清单

**整理日期：2026-09-12**
**当前证据截止：2026-09-12 HANDOFF + AGENT3 + 服务器枚举批次 5 / GLOBAL READ-ONLY CLOSURE AUDIT v2 （17:55 UTC）**
**项目根目录：`/workspace/mtpatcher`**
**代码仓库：`/workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend`**
**分支：`infra/verl-ascend-parity`**
**本轮 closure audit 基线 HEAD（ledger 冻结提交前）：`8441ee484f52c38502ce11e1f94df66d24686ca6`**
**方向：zh→en**
**Student：`/workspace/mtpatcher/models/Qwen3-0.6B`**
**Teacher：`/workspace/mtpatcher/models/Qwen3-8B`**

> 用途：按实验独立记录数据、脚本、配置、run/checkpoint、TensorBoard 或替代日志，以及关键结果。
> 原则：服务器实时文件系统优先于本清单；本文没有根据命名规律虚构缺失路径。

## 0. 文档职责与阅读入口

本文件是稳定的**实验资产入口**，回答“某个实验的数据、代码、run、checkpoint 和日志在哪里”。它不替代以下材料：

| 需求 | 权威入口 |
|---|---|
| 当前研究状态与下一步 | `docs/research/01_current_status.md` |
| 实验设计、对照与因果解释 | `docs/research/02_experiments.md` |
| 仓库目录职责 | `docs/research/03_repository_map.md` |
| 历史演进顺序 | `docs/research/04_historical_experiment_timeline.md` |
| Targeted 叶级资产 | `manifests/experiments/targeted/ARTIFACT_PATHS.md` |
| 当天过程、失败与决策 | 当日实验日志；格式见[每日实验日志规范](#j-每日实验日志规范) |
| 服务器即时状态 | 服务器文件系统、进程树与 status artifact |

快速导航：

- [总索引](#2-总索引)
- [Foundation](#a-foundation)
- [Strong Reproduction](#b-strong-reproduction)
- [Matched-budget](#c-matched-budget)
- [Targeted SFT / OPD](#d-targeted-sft--opd)
- [Targeted diagnostics](#e-targeted-mechanism-diagnostics)
- [RoPE / SeqKD closure](#f-rope--seqkd-execution-closure)
- [Pre-Verl / Custom Torch-NPU](#g-pre-verl--custom-torch-npu)
- [TensorBoard](#h-tensorboard)
- [待补字段](#i-待补字段)
- [每日实验日志规范](#j-每日实验日志规范)
- [对外汇报格式](#k-对外汇报格式)
- [科学摘要](#l-科学摘要)

截至当前证据批次，共记录 **51 个独立实验条目**。先从总索引定位实验，再进入对应条目查路径；不要全文搜索一个过于宽泛的词后直接选首个结果。

## 1. 路径可信度标记

- **[E] 精确路径**：上传材料明确写出完整路径或仓库内精确相对路径。
- **[D] 目录级路径**：实验目录已确认，但目录内具体文件名尚未枚举。
- **[F] 家族路径**：只确认共同根目录或历史脚本代际。
- **[V] 待服务器核验**：材料中没有足够证据；不得把猜测值发给师兄。
- **[I] 无效/历史**：保留追溯，不进入正式科学结论。

## 2. 总索引

| 分类 | 实验 | 当前状态 | 核心结果 |
|---|---|---|---|
| Foundation | Base Qwen3-0.6B | FROZEN | Macro BLEU 17.3485 |
| Foundation | Human SFT6565 | 历史 substrate | 有 TensorBoard；不是 Broad20k 主复现池 |
| Foundation | Full SeqKD20k | SCIENTIFIC POSITIVE CONTROL | P3 19.1652 |
| Foundation | Full OPD20k | SCIENTIFIC RUN COMPLETED | P3 19.1036，距 SeqKD 约 -0.06 |
| Strong Reproduction | PE / K1All | COMPLETE | 17.5139 |
| Strong Reproduction | PDS | STRONG PASS | 18.5696；比 parent-repeat +1.8104 |
| Strong Reproduction | General-MT WA | COMPLETE / NULL GENERAL-MT EFFECT | 两 seed 平均约 0 |
| Matched-budget | Random SeqKD4960 | COMPLETE | 18.2658 |
| Matched-budget | Same-source SeqKD4960 | COMPLETE | 18.4151 |
| Data-efficient OPD | Random OPD4960 | COMPLETE | 17.9861 |
| Data-efficient OPD | Selective OPD4960 | NEGATIVE RESULT | 17.9600 |
| Targeted | C0 baseline | COMPLETE | Idiom 2.613；Chemistry 0.097 |
| Targeted | C1/C2/C3 SFT | POSITIVE CONTROL PASS | targeted uptake 强，general MT cost 较大 |
| Targeted | O1/O2/O3 OPD | COMPLETE | targeted uptake 弱，general MT cost 较小 |
| Targeted diagnostic | Train-set audit | COMPLETE | gap 在 seen examples 上已存在 |
| Targeted diagnostic | A0 OPD Horizon-5 | COMPLETE | 多训几轮不是共同主瓶颈 |
| Targeted diagnostic | A4 KL localization | COMPLETE | lexical signal 并未被 dense KL 完全稀释 |
| Targeted diagnostic | A4b semantic audit | COMPLETE | evaluator noise 不能解释 OPD-SFT gap |
| Targeted diagnostic | A1 SFT Horizon-5 | COMPLETE | 1–2 pass 已吸收大部分知识 |
| Targeted diagnostic | Prefix-Support Swap | NOT STARTED | 当前最高优先级 |
| Historical custom | PE-SFT3732 initialization | FROZEN / COMPLETE | 约 +0.390 Avg BLEU |
| Historical custom | SeqKD Equal/Selected/Full/News50k | COMPLETE | +1.050 / +1.182 / +1.465 / +2.250 |
| Historical custom | Exact FKL/RKL/PG-RKL | CLOSED | 均约等于 Base |
| Historical custom | Top128/Entropy-aware OPD | CLOSED | +0.062 / -0.028 |
| Historical custom | Correction-trajectory FKL | CLOSED | 约等于 Base |
| Historical custom | Correction-NLL controls | COMPLETE | Patch-only 有害；FullCorr +0.285 |
| Historical custom | PatchBoost controls | COMPLETE | PB4 +0.374；比 Random4 +0.123 |
| Historical custom | PB4→PG-RKL | CLOSED | 未超过 PB4 初始化 |
| Historical custom | RQ2 Selected/Random/Full OPD | COMPLETE | selection 仅弱信号 |
| EC-ROPD | matched512 old RNG | CONFOUNDED | 正信号含 RNG 混杂 |
| EC-ROPD | matched512 RNG-paired | COMPLETE | e3 +0.1408 BLEU |
| EC-ROPD | Full3732 scaling | HISTORICAL / NEEDS LOG AUDIT | 报告只确认 Vanilla 启动 |
| Historical extension | PDS v13 | COMPLETE | 旧口径约 +0.395，存在冲突记录 |
| Historical extension | WA v14 generation | COMPLETE PIPELINE | 与 Broad20k WA 分开 |
| Patcher | Qwen3-8B full FT | ENGINEERING PASS | checkpoint-4752 |
| Legacy pilot | Qwen3.5 preliminary reproduction | COMPLETE | 25.93→26.97 |

---

# A. Foundation

## A0. Base Qwen3-0.6B

- **性质**：FROZEN BASELINE。
- **数据**：
  - 通用评测为 WMT24 / FLORES / Challenge；精确评测文件名 **[V]**。
  - 共享数据根：`/workspace/mtpatcher/data` **[F]**。
- **模型/checkpoint**：`/workspace/mtpatcher/models/Qwen3-0.6B` **[E]**。
- **脚本**：统一 evaluator 的精确脚本路径 **[V]**。
- **Run**：Base formal evaluation 的精确 run **[V]**。
- **TensorBoard**：无训练，不适用。
- **替代日志**：Base 逐集生成 JSONL 与评分 summary **[V]**。
- **关键结果**：Macro BLEU `17.3485`。
- **注意**：早期旧 evaluator 下约 `15.x` 的 Base 属于旧口径，不能与当前正式表直接混用。

## A1. Human SFT6565

- **性质**：历史 translation substrate / positive-control 资产；不是 Broad20k 主复现池。
- **数据**：Human6565，约 6.5k WMT17–20 平行语对；精确数据文件 **[V]**。
- **训练脚本**：`scripts/pilot_v2/train_qwen3_06b_full_sft_ascend.py` **[E，历史 source trainer]**。
- **Run**：`/workspace/mtpatcher/runs/sft/human6565_qwen3_06b_20260902_114434` **[E]**。
- **Checkpoint**：run 内具体 checkpoint **[V]**。
- **TensorBoard**：`/workspace/mtpatcher/runs/sft/human6565_qwen3_06b_20260902_114434/tensorboard_log` **[E]**。
- **替代日志**：run 内 train log / trainer state **[D]**。
- **关键结果**：当前 HANDOFF 未给该 run 的正式统一分数；早期 `Human-SFT6565 (e3)` 结果属于旧评测口径，不并入当前主表。

## A2. Full SeqKD20k — Historical anchor

- **性质**：LAB REPRODUCTION / SCIENTIFIC POSITIVE CONTROL。
- **数据**：
  - Broad20k Teacher targets：`/workspace/mtpatcher/data/verl_science_broad20k/seqkd_broad20k.parquet` **[E]**。
  - source 为 WMT NewsCrawl 2023 zh 固定 Broad20k。
- **脚本**：
  - Config：`configs/sft/canonical_seqkd_broad20k_qwen3_06b_8b.yaml` **[E]**。
  - Recipe：`recipes/sft/canonical_seqkd_broad20k_qwen3_06b_8b.sh` **[E]**。
  - Dataset adapter：`scripts/data/verl_mt_response_sft_dataset.py` **[E]**。
- **正式 Run**：`/workspace/mtpatcher/runs/sft/canonical_seqkd_broad20k_qwen3_06b_8b_20260903_133457` **[E]**。
- **Checkpoint**：P1/P2/P3 对应 checkpoint 叶目录 **[V]**。
- **TensorBoard**：`/workspace/mtpatcher/runs/sft/canonical_seqkd_broad20k_qwen3_06b_8b_20260903_133457/tensorboard_log` **[E]**。
- **较早 canonical run**：`/workspace/mtpatcher/runs/sft/canonical_seqkd_broad20k_qwen3_06b_8b_20260903_033659/tensorboard_log` **[E，勿与正式成功 run 混淆]**。
- **评测 Run**：`/workspace/mtpatcher/runs/science/canonical_seqkd_vs_opd_eval_20260906` **[E，阶段性 canonical eval]**。
- **RoPE 修复后的正式 P1 评测**：`/workspace/mtpatcher/runs/science/seqkd_p1_rope_fix_final_20260909_v1` **[E]**；Macro BLEU `18.5177`，与 historical P1 `18.5203` 相差约 `-0.0026`。
- **关键结果**：Historical Full SeqKD P3 Macro BLEU `19.1652`。
- **协议**：20,000 sources；3 passes；每 pass 1,250 updates；总计 3,750 updates；fixed Teacher target；response-only CE。

## A3. Full OPD20k — Canonical FKL top-k32

- **性质**：LAB REPRODUCTION / canonical OPD scientific run。
- **数据**：Broad20k source；精确 OPD source parquet **[V]**。
- **脚本**：
  - Config：`configs/opd/canonical_fkl_topk_broad20k_qwen3_06b_8b.yaml` **[E]**。
  - Recipe：`recipes/opd/canonical_fkl_topk_broad20k_qwen3_06b_8b.sh` **[E]**。
  - Zero reward：`scripts/opd/constant_zero_reward.py` **[E]**。
- **Canonical Run**：`/workspace/mtpatcher/runs/opd/canonical_fkl_topk_broad20k_qwen3_06b_8b_20260903_034058` **[E]**。
- **正式 Recovery Run**：`/workspace/mtpatcher/runs/opd/opd_recovery_20260905_v2` **[E]**。
- **Checkpoint**：正式从 canonical `global_step_1250`（P1）用 Verl `resume_path` 恢复；P1/P2/P3 精确目录 **[V]**。
- **TensorBoard contract**：
  - Canonical overrides 明确包含 `trainer.logger=['console','tensorboard']` **[E]**。
  - Canonical `train.log` 与 recovery `launcher.log/train.log` 均打印 `Saving tensorboard log to tensorboard_log/mtpatcher-opd/canonical-fkl-topk-broad20k-qwen3-06b-8b` **[E，运行时初始化证据]**。
  - 物理目录：`/workspace/mtpatcher/repo/verl-v0.9.0/tensorboard_log/mtpatcher-opd/canonical-fkl-topk-broad20k-qwen3-06b-8b` **[E]**。
  - Event 1：`events.out.tfevents.1788505998.8442009ba1f5.476464.0`，`3,900,526` bytes **[E，具体 arm 映射待核验]**。
  - Event 2：`events.out.tfevents.1788587310.8442009ba1f5.1473680.0`，`15,165,214` bytes **[E，PID 与 recovery 日志一致]**。
  - Canonical 初始化日志记录 PID `2373346`，当前没有找到同 PID event 文件；需核验 Event 1 的实际来源 **[V]**。
  - Canonical 与 recovery 使用相同 project/experiment 名并落入同一目录；TensorBoard 会把该目录视为同一 logical run，现有曲线必须先做 event-level provenance audit 才能解释。
- **替代日志**：recovery run 内 launcher/status/resolved config 等 **[D]**。
- **关键结果**：
  - P1 `18.3390`
  - P2 `18.6950`
  - P3 `19.1036`
- **协议**：temperature=1；Teacher top-k=32；Student rollout 冻结；Teacher 在相同 source + exact Student prefix 上给分；renormalized forward KL；仅更新 Student。
- **注意**：原 attempt 的 step1251–1549 不可作为正式 recovery source。

---

# B. Strong Reproduction

本节包括 PE/K1All、K2 selection、PDS 及 General-MT WA。

## B0. 共享资产根

- **数据根**：`/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823` **[E]**。
- **Run 根**：`/workspace/mtpatcher/runs/mtpatcher_v3_full6565_20260823` **[E]**。
- **历史脚本家族**：`scripts/mtpatcher_v3` 至 `scripts/mtpatcher_v14` **[F]**。
- **正式说明**：这些目录是开发代际，不等于科学实验层级；历史路径不得为了整洁随意移动。

## B1. PE / K1All

- **性质**：LAB REPRODUCTION；当前最接近论文主 PE selection 的 adaptation。
- **数据**：Broad20k 中第一轮 Feedbacker 判为有错的 `11,792` 个 parent；精确 leaf artifact **[V]**。
- **脚本**：PE generation / filtering / Student SFT 的精确脚本 **[V]**；位于历史 `scripts/mtpatcher_v*` 家族 **[F]**。
- **Run/checkpoint**：Strong Repro Student6 V1 下 K1All 对应 run/checkpoint **[V]**。
- **TensorBoard**：无已知 TensorBoard 地址。
- **替代日志**：对应 master log、manifest、评测 JSONL/summary **[V]**。
- **关键结果**：Macro BLEU `17.5139`，相对 Base `+0.1654`。

## B2. K2-consistent selected subset

- **性质**：探索性 selection control；不能称为论文主 PE。
- **数据**：`4,960 / 20,000 = 24.8%`；由 K1 post-edit 经严格二次一致性过滤；精确 leaf artifact **[V]**。
- **脚本**：K1→K2 construct/filter 的精确脚本 **[V]**。
- **Run/log**：Broad20k K1→K2 generation run **[V]**。
- **TensorBoard**：无，数据生成任务不适用。
- **关键结果**：K2 SFT Macro BLEU `17.3887`。
- **注意**：只能称 `K2-consistent targets`，不能称 `verified clean`。

## B3. PDS — K1All + PDS

- **性质**：LAB REPRODUCTION；`PDS downstream treatment = STRONG PASS`。
- **数据**：
  - 11,792 PE parents。
  - 25,000 local error pairs。
  - 22,605 repo-mechanical pass pairs。
  - 90,420 raw PDS jobs（每 error pair ×4）。
  - 57,125 structurally accepted PDS rows。
  - Student SFT 合计 68,917 rows。
  - 精确 leaf artifacts **[V]**，共享数据根见 B0。
- **脚本**：Analyzer / CaseGeneration / parser / Student SFT 精确文件 **[V]**。
- **Run/checkpoint**：Strong Repro PDS → Student BLEU V2 对应目录 **[V]**。
- **TensorBoard**：无已知地址。
- **替代日志**：finish `2026-08-31 22:49:06 CST`，total elapsed `15,654s`；具体 master log / summary **[V]**。
- **关键结果**：Macro BLEU `18.5696`，相对 Base `+1.2211`，相对 K1All `+1.0557`。
- **边界**：68,917 rows > 20,000，不能声称 row-efficient。

## B4. PDS parent-matched repetition control

- **性质**：ABLATION / matched exposure control。
- **数据**：68,917 rows；每个 PE parent 的 exposure 与 PDS arm 精确匹配；`PARENT_EXPOSURE_MAX_DIFF=0`。
- **脚本**：精确构造/训练脚本 **[V]**。
- **Run/checkpoint**：K1_PARENT_MATCHED_REPEAT 对应目录 **[V]**。
- **TensorBoard**：无已知地址。
- **替代日志**：与 B3 同一 Strong Repro PDS V2 结果家族 **[F]**。
- **关键结果**：Macro BLEU `16.7592`；PDS 相对该 control `+1.8104`，三个 benchmark 同方向。

## B5. General-MT WA — two seeds

- **性质**：LAB REPRODUCTION / adaptation；Raw-WA reproduction evidence 冻结，不得原地修改。
- **数据**：PE+PDS+WA 的精确 final rows/path **[V]**。
- **脚本**：WA analogy / context generation / Student SFT / evaluation 精确脚本 **[V]**。
- **Run/checkpoint**：seed1、seed2 精确 run **[V]**。
- **TensorBoard**：无已知地址。
- **替代日志**：seed-specific master logs / manifests / eval summaries **[V]**。
- **关键结果**：seed1 约 `-0.0127`，seed2 约 `+0.0131`，mean 约 `0`。
- **解释边界**：只能写“当前 general-MT incremental BLEU 未稳健复现”；不能写“WA mechanism failed”。

---

# C. Matched-budget

本节比较 4,960-row SeqKD 与 OPD，并区分 random 和 same-source/selected 对照。

## C1. Random SeqKD4960

- **性质**：matched-budget control。
- **数据**：Broad20k 随机 4,960 sources；精确 subset/Teacher target path **[V]**。
- **脚本**：精确 data builder / SFT recipe **[V]**。
- **Run/checkpoint**：Strong Repro Student6 V1 对应目录 **[V]**。
- **TensorBoard**：无已知地址。
- **替代日志**：eval summary / JSONL **[V]**。
- **关键结果**：Macro BLEU `18.2658`。

## C2. Same-source SeqKD4960

- **性质**：matched-source target-treatment control。
- **数据**：与 K2/Selective OPD 完全相同的 4,960 sources + fixed Teacher targets；精确路径 **[V]**。
- **脚本**：精确 builder/recipe **[V]**。
- **Run/checkpoint**：Strong Repro Student6 V1 对应目录 **[V]**。
- **TensorBoard**：无已知地址。
- **替代日志**：eval summary / JSONL **[V]**。
- **关键结果**：Macro BLEU `18.4151`；比 K2 SFT 高约 `1.026`。

## C3. Random OPD4960

- **性质**：matched-budget OPD control。
- **数据**：Broad20k random 4,960 sources；精确 subset path **[V]**。
- **脚本/config/recipe**：精确路径 **[V]**。
- **Run/checkpoint**：精确 run **[V]**。
- **TensorBoard**：材料未给地址 **[V]**。
- **替代日志**：Verl/launcher/eval logs **[V]**。
- **关键结果**：Macro BLEU `17.9861`。

## C4. Selective OPD4960

- **性质**：ADAPTATION / NEGATIVE RESULT；当前 K2 selection-only OPD 分支不再机械重跑。
- **数据**：K2-consistent 4,960 sources；精确 subset path **[V]**。
- **脚本/config/recipe**：精确路径 **[V]**。
- **Run/checkpoint**：精确 run **[V]**。
- **TensorBoard**：材料未给地址 **[V]**。
- **替代日志**：Verl/launcher/eval logs **[V]**。
- **关键结果**：Macro BLEU `17.9600`；相对 Random OPD `-0.0261`；相对 Same-source SeqKD `-0.4551`。
- **结论**：当前 K2 error selection 没有提高低预算 OPD source efficiency。

---

# D. Targeted SFT / OPD

本节记录 Idiom、Chemistry 及二者联合设置下的 SFT positive control 与 knowledge-conditioned OPD。

## D0. Frozen targeted data assets

- **Chemistry lexical freeze**：`/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/wa_section43_chemistry_source_v1/chemistry_section43_6000_20260910_v1` **[E]**。
- **Idiom lexical freeze**：`/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/wa_section43_idiom_source_v1` **[E]**。
- **Final targeted contexts**：`/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/targeted_section43_contexts_qwen3_8b_final_20260910` **[E]**。
- **SFT positive-control targets**：`/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/targeted_sft_positive_control_targets_qwen3_8b_v1_20260910` **[E]**。
- **Diagnostic1000**：`/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/targeted_diagnostic1000_v1_20260910` **[E]**。
- **共同结构**：Idiom 与 Chemistry 均为 6,000 total / 5,500 Seen / 500 Unseen。
- **构造性质**：Qwen3-8B synthesis 是相对论文 GPT-4 synthesis 的 ADAPTATION。

## D1. C0 targeted baseline

- **性质**：FROZEN BASELINE / evaluator anchor。
- **数据**：D0 的 frozen contexts + Diagnostic1000。
- **脚本**：
  - `scripts/targeted/run_c0_targeted_diagnostic.sh` **[E]**；SHA256 `4cc8cf8218bfbbb5d10b0e92c36af3bc1aa99c85e084997c7d79bbb7141e324f`。
  - `scripts/targeted/run_c0_targeted_diagnostic.py` **[E]**；SHA256 `f6e1d210f9760f4d31238efc71377122e28f367f8ff9cf0be8d1cf0e8f3e838b`。
- **Manifest**：`manifests/experiments/targeted/00_c0_baseline.json` **[E]**。
- **Run**：`/workspace/mtpatcher/runs/targeted/c0_targeted_diagnostic1000_20260910` **[E]**。
- **Checkpoint**：`/workspace/mtpatcher/models/Qwen3-0.6B` **[E]**。
- **TensorBoard**：无，纯评测。
- **替代日志**：
  - `/workspace/mtpatcher/runs/targeted/c0_targeted_diagnostic1000_20260910/manifest.json` **[E]**。
  - `/workspace/mtpatcher/runs/targeted/c0_targeted_diagnostic1000_20260910/progress.json` **[E]**。
  - `/workspace/mtpatcher/runs/targeted/c0_targeted_diagnostic1000_20260910/c0_targeted_diagnostic2000_translations.jsonl` **[E]**。
  - `/workspace/mtpatcher/runs/targeted/c0_targeted_diagnostic1000_20260910/c0_chemistry_diagnostic1000_scored.jsonl` **[E]**。
  - `/workspace/mtpatcher/runs/targeted/c0_targeted_diagnostic1000_20260910/idiom_deepseek_judge/summary.json` **[E]**。
- **关键结果**：Idiom `2.613`；Chemistry `0.097`。

## D2. C1 Idiom-WA SFT

- **性质**：TARGETED SFT POSITIVE CONTROL。
- **数据**：Idiom SFT target 叶文件位于 D0 的 SFT targets 目录 **[D]**。
- **脚本**：
  - `scripts/targeted/run_targeted_sft_c123_fullchain.sh` **[E]**；SHA256 `d284ff52b5962bd83dcd942c68fe3f3352b2094f51edaae058b305c6b7c75670`。
  - `scripts/targeted/targeted_sft_c123_pipeline.py` **[E]**；SHA256 `8a33cb039f142546aa98b7fd620b1dee869946867ad935db312ee2944b60fd0d`。
  - `scripts/targeted/materialize_targeted_sft_positive_control_targets.py` **[E]**；SHA256 `5190b6510c3158fd2b1c30fe4d8acca86a4fac01f4a4792ec856eff0ac672c40`。
- **Manifest**：`manifests/experiments/targeted/01_sft_positive_control.json` **[E]**。
- **Run**：`/workspace/mtpatcher/runs/targeted/wa_sft_positive_control_c123_20260910` **[E，共享 run family]**。
- **Checkpoint**：C1 leaf checkpoint **[V]**。
- **TensorBoard**：无。
- **替代日志**：
  - `/workspace/mtpatcher/runs/targeted/wa_sft_positive_control_c123_20260910/master.log` **[E]**。
  - `/workspace/mtpatcher/runs/targeted/wa_sft_positive_control_c123_20260910/state.json` **[E]**。
  - `/workspace/mtpatcher/runs/targeted/wa_sft_positive_control_c123_20260910/C1/train/{manifest.json,progress.json}` **[E，集合记法]**。
  - `/workspace/mtpatcher/runs/targeted/wa_sft_positive_control_c123_20260910/C1/eval/{manifest.json,progress.json,c1_diagnostic2000_translations.jsonl,c1_chemistry_diagnostic1000_scored.jsonl,c1_idiom_diagnostic1000_for_judge.jsonl}` **[E，集合记法]**。
  - `/workspace/mtpatcher/runs/targeted/wa_sft_positive_control_c123_20260910/idiom_deepseek_judge/summary.json` **[E，共享 C1/C2/C3 judge summary]**。
- **关键结果**：Idiom `3.132`，相对 C0 `+0.519`；General-MT Macro BLEU 约 `-0.813`。

## D3. C2 Chemistry-WA SFT

- **性质**：TARGETED SFT POSITIVE CONTROL。
- **数据**：Chemistry SFT target 叶文件位于 D0 的 SFT targets 目录 **[D]**。
- **脚本/Manifest**：与 D2 共用三份精确脚本和 `01_sft_positive_control.json` **[E]**。
- **Run**：`/workspace/mtpatcher/runs/targeted/wa_sft_positive_control_c123_20260910` **[E，共享 run family]**。
- **Checkpoint**：C2 leaf checkpoint **[V]**。
- **TensorBoard**：无。
- **替代日志**：
  - `/workspace/mtpatcher/runs/targeted/wa_sft_positive_control_c123_20260910/C2/train/{manifest.json,progress.json}` **[E，集合记法]**。
  - `/workspace/mtpatcher/runs/targeted/wa_sft_positive_control_c123_20260910/C2/eval/{manifest.json,progress.json,c2_diagnostic2000_translations.jsonl,c2_chemistry_diagnostic1000_scored.jsonl,c2_idiom_diagnostic1000_for_judge.jsonl}` **[E，集合记法]**。
  - 共享 `master.log`、`state.json`、Idiom judge summary 见 D2。
- **关键结果**：Chemistry `0.261`，相对 C0 `+0.164`；General-MT Macro BLEU 约 `-2.172`。

## D4. C3 Idiom+Chemistry WA SFT

- **性质**：TARGETED SFT POSITIVE CONTROL。
- **数据**：combined SFT target 叶文件位于 D0 的 SFT targets 目录 **[D]**。
- **脚本/Manifest**：与 D2 共用三份精确脚本和 `01_sft_positive_control.json` **[E]**。
- **Run**：`/workspace/mtpatcher/runs/targeted/wa_sft_positive_control_c123_20260910` **[E，共享 run family]**。
- **Checkpoint**：C3 leaf checkpoint **[V]**。
- **TensorBoard**：无。
- **替代日志**：
  - `/workspace/mtpatcher/runs/targeted/wa_sft_positive_control_c123_20260910/C3/train/{manifest.json,progress.json}` **[E，集合记法]**。
  - `/workspace/mtpatcher/runs/targeted/wa_sft_positive_control_c123_20260910/C3/eval/{manifest.json,progress.json,c3_diagnostic2000_translations.jsonl,c3_chemistry_diagnostic1000_scored.jsonl,c3_idiom_diagnostic1000_for_judge.jsonl}` **[E，集合记法]**。
  - 共享 `master.log`、`state.json`、Idiom judge summary 见 D2。
- **关键结果**：Idiom `3.141`，相对 C0 `+0.528`；Chemistry `0.261`；General-MT Macro BLEU 约 `-0.842`。

## D5. O1 Idiom knowledge-conditioned OPD

- **性质**：LAB ADAPTATION / KNOWLEDGE-CONDITIONED OPD。
- **数据**：Idiom targeted source + lexical hint；精确 leaf file **[V]**。
- **脚本**：`scripts/targeted/targeted_wa_opd_overnight_v3.py` **[E]**；SHA256 `8250a41e93a6d2f0f552a84ee70aeca695cd0ac3eb9de3d1d41139d6afe26ae1`。
- **Manifest**：`manifests/experiments/targeted/02_knowledge_conditioned_opd.json` **[E]**。
- **Run**：`/workspace/mtpatcher/runs/targeted/wa_opd_knowledge_conditioned_o123_20260911` **[E，共享 run family]**。
- **Checkpoint**：O1 P1/P2/P3 leaf paths **[V]**。
- **TensorBoard**：无。
- **替代日志**：
  - `/workspace/mtpatcher/runs/targeted/wa_opd_knowledge_conditioned_o123_20260911/nohup.out` **[E，共享 master log]**。
  - `/workspace/mtpatcher/runs/targeted/wa_opd_knowledge_conditioned_o123_20260911/state.json` **[E]**。
  - `/workspace/mtpatcher/runs/targeted/wa_opd_knowledge_conditioned_o123_20260911/O1/train/{manifest.json,progress.json}` **[E，集合记法]**。
  - `/workspace/mtpatcher/runs/targeted/wa_opd_knowledge_conditioned_o123_20260911/O1/targeted_eval/{targeted_diagnostic2000_translations.jsonl,o1_idiom1000_for_judge.jsonl}` **[E，集合记法]**。
  - `/workspace/mtpatcher/runs/targeted/wa_opd_knowledge_conditioned_o123_20260911/general_bleu_opd/O1/{summary.json,wmt24_predictions.jsonl,flores_predictions.jsonl,challenge_predictions.jsonl}` **[E，集合记法]**。
- **关键结果**：Idiom `+0.108`；General-MT Macro BLEU 约 `-0.586`。

## D6. O2 Chemistry knowledge-conditioned OPD

- **性质**：LAB ADAPTATION / KNOWLEDGE-CONDITIONED OPD。
- **数据**：Chemistry targeted source + lexical hint；精确 leaf file **[V]**。
- **脚本/Manifest**：与 D5 共用 `targeted_wa_opd_overnight_v3.py` 和 `02_knowledge_conditioned_opd.json` **[E]**。
- **Run**：`/workspace/mtpatcher/runs/targeted/wa_opd_knowledge_conditioned_o123_20260911` **[E，共享 run family]**。
- **Checkpoint**：O2 P1/P2/P3 leaf paths **[V]**。
- **TensorBoard**：无。
- **替代日志**：
  - `/workspace/mtpatcher/runs/targeted/wa_opd_knowledge_conditioned_o123_20260911/O2/train/{manifest.json,progress.json}` **[E，集合记法]**。
  - `/workspace/mtpatcher/runs/targeted/wa_opd_knowledge_conditioned_o123_20260911/O2/targeted_eval/{targeted_diagnostic2000_translations.jsonl,o2_idiom1000_for_judge.jsonl}` **[E，集合记法]**。
  - `/workspace/mtpatcher/runs/targeted/wa_opd_knowledge_conditioned_o123_20260911/general_bleu_opd/O2/{summary.json,wmt24_predictions.jsonl,flores_predictions.jsonl,challenge_predictions.jsonl}` **[E，集合记法]**。
- **关键结果**：Chemistry `+0.010`；General-MT Macro BLEU 约 `-0.094`。

## D7. O3 Idiom+Chemistry knowledge-conditioned OPD

- **性质**：LAB ADAPTATION / KNOWLEDGE-CONDITIONED OPD。
- **数据**：combined targeted source + lexical hint；精确 leaf file **[V]**。
- **脚本/Manifest**：与 D5 共用 `targeted_wa_opd_overnight_v3.py` 和 `02_knowledge_conditioned_opd.json` **[E]**。
- **Run**：`/workspace/mtpatcher/runs/targeted/wa_opd_knowledge_conditioned_o123_20260911` **[E，共享 run family]**。
- **Checkpoint**：O3 P1/P2/P3 leaf paths **[V]**。
- **TensorBoard**：无。
- **替代日志**：
  - `/workspace/mtpatcher/runs/targeted/wa_opd_knowledge_conditioned_o123_20260911/O3/train/{manifest.json,progress.json}` **[E，集合记法]**。
  - `/workspace/mtpatcher/runs/targeted/wa_opd_knowledge_conditioned_o123_20260911/O3/targeted_eval/{targeted_diagnostic2000_translations.jsonl,o3_idiom1000_for_judge.jsonl}` **[E，集合记法]**。
  - `/workspace/mtpatcher/runs/targeted/wa_opd_knowledge_conditioned_o123_20260911/general_bleu_opd/O3/{summary.json,wmt24_predictions.jsonl,flores_predictions.jsonl,challenge_predictions.jsonl}` **[E，集合记法]**。
  - `/workspace/mtpatcher/runs/targeted/wa_opd_knowledge_conditioned_o123_20260911/idiom_deepseek_judge/summary.json` **[E，共享 O1/O2/O3 judge summary]**。
- **关键结果**：Idiom `+0.103`；Chemistry `+0.009`；General-MT Macro BLEU 约 `-0.408`。
- **Run lineage**：同一 run family 中保留 early semantic-preflight v1 failure；随后 corrected `preflight_v3` 通过，并完成正式 O1/O2/O3 train/eval、general BLEU、targeted eval 与 frozen judge。旧 `nohup.out` 的 `O123_TRAIN_FAIL` 仅属于早期 attempt，不覆盖后续 formal PASS。

---

# E. Targeted mechanism diagnostics

## E1. OPD train-set audit

- **性质**：DIAGNOSTIC ONLY。
- **数据**：
  - `/workspace/mtpatcher/runs/targeted/wa_opd_trainset_audit_v1_20260911/frozen_train_subset/idiom_train1000.jsonl` **[E]**。
  - `/workspace/mtpatcher/runs/targeted/wa_opd_trainset_audit_v1_20260911/frozen_train_subset/chemistry_train1000.jsonl` **[E]**。
- **脚本**：
  - `scripts/targeted/trainset_audit_common.py` **[E]**；SHA256 `f8dcf735e844a597d15fdfd16b8053c6b27ff3e07901e82d7e7816169d3d1ebb`。
  - `scripts/targeted/wa_opd_trainset_audit_v1.py` **[E]**；SHA256 `63ac5588be0728ab9552b9b92e67ad2ba876ea8ed1b012b2dc93596094b93d19`。
- **Manifest**：`manifests/experiments/targeted/03_trainset_audits.json` **[E，共享]**。
- **Run**：`/workspace/mtpatcher/runs/targeted/wa_opd_trainset_audit_v1_20260911` **[E]**。
- **TensorBoard**：无。
- **替代日志**：
  - `/workspace/mtpatcher/runs/targeted/wa_opd_trainset_audit_v1_20260911/master.log` **[E]**。
  - `/workspace/mtpatcher/runs/targeted/wa_opd_trainset_audit_v1_20260911/manifest.json` **[E]**。
  - `/workspace/mtpatcher/runs/targeted/wa_opd_trainset_audit_v1_20260911/labels/{C0,O1,O2,O3}/{summary.json,progress.json,train2000_translations.jsonl}` **[E，集合记法]**。
  - `/workspace/mtpatcher/runs/targeted/wa_opd_trainset_audit_v1_20260911/idiom_train_deepseek_judge/summary.json` **[E]**。
- **关键结果**：Idiom train1000：C0 `2.680`，O1 `2.746`（`+0.066`）；Chemistry train1000：C0 `.087`，O2 `.097`（`+.010`）。

## E2. SFT train-set audit

- **性质**：DIAGNOSTIC ONLY。
- **数据**：Diagnostic1000，见 D0。
- **脚本**：
  - `scripts/targeted/trainset_audit_common.py` **[E]**。
  - `scripts/targeted/wa_sft_trainset_audit_c123_v1.py` **[E]**；SHA256 `c883963b6384a53f80e1c9df3381616f2e6d54a15a8f4c3012b2340de61697f5`。
- **Manifest**：`manifests/experiments/targeted/03_trainset_audits.json` **[E，共享]**。
- **Run**：`/workspace/mtpatcher/runs/targeted/wa_sft_trainset_audit_c123_20260911` **[E]**。
- **TensorBoard**：无。
- **替代日志**：
  - `/workspace/mtpatcher/runs/targeted/wa_sft_trainset_audit_c123_20260911/master.log` **[E]**。
  - `/workspace/mtpatcher/runs/targeted/wa_sft_trainset_audit_c123_20260911/{C1,C2,C3}/{summary.json,progress.json,train2000_translations.jsonl}` **[E，集合记法]**。
- **关键结果**：Idiom train1000：C1 `3.368`（相对 C0 `+0.688`）。

## E3. A0 — OPD Horizon-5

- **性质**：DIAGNOSTIC ONLY / optimization-horizon ablation。
- **数据**：O1 Idiom / O2 Chemistry 与 frozen Diagnostic1000。
- **脚本**：
  - `scripts/targeted/targeted_wa_opd_horizon5_o12_v2.py` **[E]**。
  - `scripts/targeted/horizon5_matched_traincurve_eval_v1.py` **[E]**；SHA256 `54a0c683e47e57e46b26f79672594624c94701e39d85976885ad41868ebd16ac`。
- **Manifest**：`manifests/experiments/targeted/04_a0_opd_horizon5.json` **[E]**。
- **脚本 SHA256**：`38ac62c0a93f1b762b7536db60421dd8c1e7aca49a57c25851306c3cb766e16` **[E]**。
- **3-pass validated script SHA256**：`8250a41e93a6d2f0f552a84ee70aeca695cd0ac3eb9de3d1d41139d6afe26ae1` **[E]**。
- **Run**：`/workspace/mtpatcher/runs/targeted/wa_opd_horizon5_o12_20260911` **[E]**。
- **Checkpoint**：O1/O2 P1–P5 leaf paths **[V]**。
- **TensorBoard**：无。
- **替代日志**：
  - `/workspace/mtpatcher/runs/targeted/wa_opd_horizon5_o12_20260911/master_h5.log` **[E]**。
  - `/workspace/mtpatcher/runs/targeted/wa_opd_horizon5_o12_20260911/O1/train/{manifest.json,progress.json}` **[E，集合记法]**。
  - `/workspace/mtpatcher/runs/targeted/wa_opd_horizon5_o12_20260911/O2/train/{manifest.json,progress.json}` **[E，集合记法]**。
  - `/workspace/mtpatcher/runs/targeted/wa_opd_horizon5_o12_20260911/learning_curve/train_matched/a0_horizon5_final_summary_v1.json` **[E]**。
  - `/workspace/mtpatcher/runs/targeted/wa_opd_horizon5_o12_20260911/learning_curve/{heldout,train_matched}/master.log` **[E，集合记法]**。
  - 失败启动：`.../failed_launch_single_visible_device_20260911_125949/master_h5.log` **[I]**。
- **关键结果**：Idiom gain P1/P2/P3/P4/P5 = `+.120 / +.128 / +.135 / +.127 / +.141`；Chemistry P5 约 `+.009`。
- **结论**：更多 pass 不是两个 domain 的共同主瓶颈。

## E4. A4 — Chemistry KL Signal Localization

- **性质**：DIAGNOSTIC ONLY。
- **数据**：Chemistry 256 rows / 6,929 valid tokens；精确 leaf input **[V]**。
- **脚本**：`scripts/targeted/a4_kl_signal_localization_chemistry_v1.py` **[E]**；SHA256 `ebaaae102177ec3f076a4ad83437ef4f1a57348ee34462d700224d09ce397fa1`。
- **Manifest**：`manifests/experiments/targeted/05_a4_kl_localization.json` **[E]**。
- **Run**：`/workspace/mtpatcher/runs/targeted/a4_kl_signal_localization_chemistry_v1_20260912` **[E]**。
- **TensorBoard**：无。
- **替代日志**：
  - `/workspace/mtpatcher/runs/targeted/a4_kl_signal_localization_chemistry_v1_20260912/master.log` **[E]**。
  - `/workspace/mtpatcher/runs/targeted/a4_kl_signal_localization_chemistry_v1_20260912/summary.json` **[E]**。
  - `/workspace/mtpatcher/runs/targeted/a4_kl_signal_localization_chemistry_v1_20260912/rowwise_token_signals.jsonl` **[E]**。
  - `/workspace/mtpatcher/runs/targeted/a4_kl_signal_localization_chemistry_v1_20260912/top100_hint_sensitive_tokens.jsonl` **[E]**。
  - duplicate-aborted run：`/workspace/mtpatcher/runs/targeted/a4_kl_signal_localization_chemistry_v1_20260912_DUPLICATE_ABORTED_20260912_004123` **[I]**。
- **关键结果**：Pearson(H, KL)≈`.596`；Spearman(H, KL)≈`.729`；hint-sensitive top10% tokens 吸收约 `49.1%` OPD KL mass 和 `93.2%` hint-gap mass。
- **结论**：dense KL 完全稀释 lexical signal 不是充分解释。

## E5. A4b — Chemistry Semantic Audit

- **性质**：DIAGNOSTIC ONLY / evaluator robustness。
- **数据**：Chemistry full1000；精确 input file **[V]**。
- **脚本**：
  - `scripts/targeted/a4b_build_chem_semantic_audit_full1000_v2.py` **[E，正式 full1000]**；SHA256 `71f0f542393092353b03686a15b18cf9443a4f259f1e37037b53469fc81460b3`。
  - `scripts/targeted/a4b_build_chem_semantic_audit_v1.py` **[E，早期版本]**；SHA256 `b7bd8bd7df8f63576dfc7ac35ac0220255902c883ff2d98ebf7cc1f1e42c192e`。
- **Manifest**：`manifests/experiments/targeted/06_a4b_semantic_audit.json` **[E]**。
- **Run**：`/workspace/mtpatcher/runs/targeted/a4b_chem_semantic_audit_full1000_v2_20260912` **[E]**。
- **TensorBoard**：无。
- **替代日志**：
  - `/workspace/mtpatcher/runs/targeted/a4b_chem_semantic_audit_full1000_v2_20260912/manifest.json` **[E]**。
  - `/workspace/mtpatcher/runs/targeted/a4b_chem_semantic_audit_full1000_v2_20260912/chem_semantic_audit_full3000.jsonl` **[E]**。
  - `/workspace/mtpatcher/runs/targeted/a4b_chem_semantic_audit_full1000_v2_20260912/deepseek_semantic_judge_full1000_v2/chem_semantic_full3000_judged.jsonl` **[E]**。
  - `/workspace/mtpatcher/runs/targeted/a4b_chem_semantic_audit_full1000_v2_20260912/deepseek_semantic_judge_full1000_v2/summary.json` **[E]**。
  - `/workspace/mtpatcher/runs/targeted/a4b_chem_semantic_audit_full1000_v2_20260912/c2_target_meta_audit_v1.json` **[E]**。
  - `/workspace/mtpatcher/runs/targeted/a4b_chem_semantic_audit_full1000_v2_20260912/posthoc_consistency_audit_v1.json` **[E]**。
  - v1 artifacts：`/workspace/mtpatcher/runs/targeted/a4b_chem_semantic_audit_v1_20260912/{chem_semantic_audit_input.jsonl,manifest.json}` **[I，早期版本]**。
- **关键结果**：C0 `.183`；O2P5 `.200`；C2 `.509`；consistency correction 后 O2 gain 约 `+.020`。
- **结论**：evaluator noise 存在，但解释不了 OPD-SFT gap。

## E6. A1 — SFT Horizon-5

- **性质**：DIAGNOSTIC ONLY / direct sequence-supervision horizon。
- **数据**：
  - `/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/targeted_diagnostic1000_v1_20260910` **[E]**。
  - `/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/targeted_sft_positive_control_targets_qwen3_8b_v1_20260910` **[E]**。
- **脚本**：
  - Trainer：`scripts/targeted/targeted_sft_c123_pipeline.py` **[E]**；SHA256 `8a33cb039f142546aa98b7fd620b1dee869946867ad935db312ee2944b60fd0d`。
  - Runtime arm launcher：`/workspace/mtpatcher/runs/targeted/a1_sft_horizon5_c12_v3_20260912/run_arm.sh` **[E，run 内冻结副本]**。
  - Runtime master launcher：`/workspace/mtpatcher/runs/targeted/a1_sft_horizon5_c12_v3_20260912/run_master.sh` **[E，run 内冻结副本]**。
- **Manifest**：`manifests/experiments/targeted/07_a1_sft_horizon5.json` **[E]**。
- **有效 Run**：`/workspace/mtpatcher/runs/targeted/a1_sft_horizon5_c12_v3_20260912` **[E]**。
- **Checkpoint**：
  - `/workspace/mtpatcher/runs/targeted/a1_sft_horizon5_c12_v3_20260912/C1/checkpoints` **[E，P1–P5 父目录]**。
  - `/workspace/mtpatcher/runs/targeted/a1_sft_horizon5_c12_v3_20260912/C2/checkpoints` **[E，P1–P5 父目录]**。
- **TensorBoard**：无。
- **替代日志**：
  - `/workspace/mtpatcher/runs/targeted/a1_sft_horizon5_c12_v3_20260912/master.log` **[E]**。
  - `/workspace/mtpatcher/runs/targeted/a1_sft_horizon5_c12_v3_20260912/C1/{summary.json,train/manifest.json,train/progress.json}` **[E，集合记法]**。
  - `/workspace/mtpatcher/runs/targeted/a1_sft_horizon5_c12_v3_20260912/C2/{summary.json,train/manifest.json,train/progress.json}` **[E，集合记法]**。
  - `/workspace/mtpatcher/runs/targeted/a1_sft_horizon5_c12_v3_20260912/a1_c1c2_p1p5_idiom10000_for_judge.jsonl` **[E]**。
  - `/workspace/mtpatcher/runs/targeted/a1_sft_horizon5_c12_v3_20260912/idiom_deepseek_judge/{a1_c1c2_p1p5_idiom10000_judged.jsonl,summary.json}` **[E，集合记法]**。
- **关键结果**：
  - Idiom：P1 `+.441`，P2 `+.536`，P5 `+.550`。
  - Chemistry：C0 `.097`，P1 `.233`，P2 `.293`，P5 `.298`。
- **结论**：direct sequence supervision 在 1–2 pass 已吸收大部分 targeted knowledge。
- **无效 runs [I]**：
  - `/workspace/mtpatcher/runs/targeted/a1_sft_horizon5_c12_20260912_INVALID_PREFLIGHT_DUPLICATE_20260912_020223`
  - `/workspace/mtpatcher/runs/targeted/a1_sft_horizon5_c12_v2_20260912_INVALID_SHELL_CONTROL_20260912_021458`
  - 两者仅供事故追溯，不进入正式结论。

## E7. Offline Prefix-Support Replay

- **性质**：`DIAGNOSTIC ONLY / OFFLINE PREFIX-SUPPORT REPLAY`。
- **状态**：`SPEC FROZEN / NOT STARTED`；这是 OPD 机制诊断，不是新的 OPD 方法。
- **正式 Manifest**：`manifests/experiments/targeted/08_offline_prefix_support_replay.json` **[E]**。
- **Question**：在保持 soft-KL objective、rows、hint、Student initialization 与训练预算不变时，用 frozen Teacher-supported trajectory replay 替换 frozen Student-supported trajectory replay，是否显著提高 targeted knowledge transfer？
- **Competing explanations**：
  1. Student-prefix / trajectory state support 限制 targeted knowledge 被 soft-KL 吸收；
  2. soft-KL objective / update efficiency 本身较低；
  3. Teacher 在 Teacher prefix 上给出的目标分布本身更强，因此 `T >> S` 可能同时混入 Teacher-target-quality 变化。
- **Prefix banks**：C0 时一次性生成 Student prefix bank 与 Teacher prefix bank，随后冻结并 replay；训练过程中禁止重新 rollout。
- **Primary population**：全部 frozen rows；`m_i=min(|Y_i^S|,|Y_i^T|)` 只用于 matched token budget，不解释为 semantic alignment。
- **Secondary analysis**：预先记录 lexical event 是否落入 shared window；event-covered subset 只作为 secondary analysis，不得用于训练后选择 rows。
- **Teacher signal audit**：任何参数更新前，分别计算 `Δ_T^S` 与 `Δ_T^T`。若 `T >> S` 且 Teacher signal `S≈T`，才构成较强 prefix/state-support 证据；若 Teacher signal `T >> S`，则 support 与 target-distribution quality 混杂。
- **训练匹配**：同一 C0 initialization、teacher-top-k32 forward KL、temperature、optimizer、LR、3-pass update budget、row order、batch grouping；Teacher frozen，仅 Student 更新。
- **RNG**：`prefix_generation_seed` 与 `training_seed` 分离；S/T training seed paired；冻结 Student-prefix SHA、Teacher-prefix SHA、row-set SHA、row-order SHA。
- **SFT ceiling**：Chemistry 固定原 C2 positive-control checkpoint；Idiom 固定原 C1 positive-control checkpoint；不得事后从 A1 P1–P5 中挑最好 checkpoint。
- **统计**：paired bootstrap 只表示 example-sampling uncertainty；若 screening 通过，至少再跑一个独立 training seed 确认方向；seed 翻转时再补第三 seed。
- **64-row smoke hard gates**：
  1. S arm 真正读取 frozen `Y^S`；
  2. T arm 真正读取 frozen `Y^T`；
  3. lexical hint 不进入 Student prompt；
  4. replay 前后 prefix SHA 完全不变。
- **Objective gates**：`forward_kl_topk=32`、`policy_gradient=false`、Teacher `requires_grad=false`、Student `requires_grad=true`。
- **Smoke logging**：打印 `row_id / arm / prefix_sha / prefix_token_count / first-last token ids / teacher_hint_present / student_hint_present=False`；人工检查 5–10 rows。
- **TensorBoard**：所有更新参数的 arm 使用独立绝对目录：`<RUN>/chemistry/S/tensorboard`、`<RUN>/chemistry/T/tensorboard`、`<RUN>/idiom/S/tensorboard`、`<RUN>/idiom/T/tensorboard`；第一步后必须验证非空 event。
- **执行顺序**：Teacher-signal audit → 64-row engineering smoke → Chemistry S/T → Idiom S/T → paired-bootstrap screening → 必要时 independent training-seed confirmation。
- **解释边界**：`T >> S` 第一层只能写 “teacher-supported replay improves targeted soft-KL transfer”；不得自动写成 “Student prefix support is the causal bottleneck”。
- **后续决策**：
  - `T >> S` 且 Teacher signal `S≈T` → strong support evidence，优先 Local Correction Bridge；
  - `T >> S` 且 Teacher signal `T>>S` → support / Teacher target quality confounded；
  - `T≈S` 且 Teacher signal strong → 转向 weighted / sparse / contrastive KL；
  - `T≈S` 且 Teacher signal weak → 检查 Teacher supervision construction；
  - Chemistry / Idiom 不一致 → seed confirmation + instance taxonomy。
- **PDS-OPD**：在本诊断给出方向前继续暂缓。

---

# F. RoPE / SeqKD execution closure

本节为历史诊断资产，用于证明 evaluator 与 execution contract 已闭环。

这些实验解释了为何早期 Verl SeqKD/OPD 出现异常低 BLEU。结论已经关闭，不应再次无证据重开。

## F1. Reduction reconciliation

- **Run**：`/workspace/mtpatcher/runs/sft/seqkd_reduction_reconciliation_20260906_v1` **[E]**。
- **结果**：P1 `16.0317`；`REJECTED_AS_MAIN_CAUSE`。
- **TensorBoard/脚本/数据**：精确地址 **[V]**。

## F2. Scheduler reconciliation

- **Run**：`/workspace/mtpatcher/runs/sft/seqkd_scheduler_reconciliation_20260906_v1_recovery1` **[E]**。
- **结果**：P1 `16.2007`；`REJECTED_AS_MAIN_CAUSE`。
- **TensorBoard/脚本/数据**：精确地址 **[V]**。

## F3. Historical-order reconciliation

- **Run**：`/workspace/mtpatcher/runs/sft/seqkd_historical_order_p1_20260906_v1` **[E]**。
- **Data**：`/workspace/mtpatcher/data/verl_science_broad20k/seqkd_broad20k_historical_p1_transport_v1.parquet` **[E]**。
- **结果**：P1 `16.4618`；`REJECTED_AS_MAIN_CAUSE`。
- **TensorBoard/脚本**：精确地址 **[V]**。

## F4. Known-stack O+R+S joint arm

- **Worktree**：`/workspace/mtpatcher/repo/verl-v0.9.0-seqkd-known-stack-p1-20260906-v1` **[E]**。
- **Train Run**：`/workspace/mtpatcher/runs/sft/seqkd_known_stack_p1_20260906_v1` **[E]**。
- **Eval Run**：`/workspace/mtpatcher/runs/science/seqkd_known_stack_p1_eval_20260906_v1` **[E]**。
- **结果**：P1 `16.1211`；联合解释仍不足。
- **TensorBoard**：精确地址 **[V]**。

## F5. Historical execution + user-only prompt repair

- **Trainer**：`scripts/pilot_v2/train_qwen3_06b_full_sft_ascend.py` **[E]**。
- **Train Run**：`/workspace/mtpatcher/runs/sft/seqkd_historical_execution_useronly_p1_20260907_v1` **[E]**。
- **Eval Run**：`/workspace/mtpatcher/runs/science/seqkd_historical_execution_useronly_p1_eval_20260907_v1` **[E]**。
- **Oracle**：`/workspace/mtpatcher/runs/sft/seqkd_historical_execution_useronly_p1_20260907_v1/first_update_oracle.json` **[E]**。
- **Summary**：`/workspace/mtpatcher/runs/science/seqkd_historical_execution_useronly_p1_eval_20260907_v1/diagnostic_summary.json` **[E]**。
- **结果**：Macro BLEU `18.52034208134914`；Macro chrF `49.25786025304718`。
- **结论**：Historical SeqKD P1 positive control reproduced；最终根因转向 evaluator RoPE schema compatibility。

---

# G. Pre-Verl / Custom Torch-NPU

本节补齐 2026-08-25 至 2026-08-29 的非 targeted 实验。它们使用旧的 Human6565 / PE3732 设置与 custom Torch-NPU trainer，和后来的 Broad20k canonical Verl 实验语义不同，不能混成同一张正式主表。

## G0. 共享协议与资产根

- **实验名**：`mtpatcher_v3_full6565_20260823`。
- **数据根**：`/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823` **[E]**。
- **Run 根**：`/workspace/mtpatcher/runs/mtpatcher_v3_full6565_20260823` **[E]**。
- **评测**：WMT24 998 / FLORES 1012 / Challenge 197；旧统一 greedy evaluator；SacreBLEU + chrF。
- **TensorBoard**：这些 custom Torch-NPU 实验没有已确认 TensorBoard 地址；以文本日志、epoch checkpoint 和评测 JSONL/summary 为准。
- **解释边界**：下列 `ΔBLEU` 均属于该历史 evaluation stack，不能与 RoPE 修复后的 Broad20k `17.3485` Base 直接相加。

## G1. PE-SFT3732 initialization

- **性质**：HISTORICAL LAB ADAPTATION / shared initialization。
- **数据**：`/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/pe_k1_clean3732.jsonl` **[E]**。
- **脚本**：精确 trainer/launcher **[V]**。
- **Run/Checkpoint**：`/workspace/mtpatcher/runs/mtpatcher_v3_full6565_20260823/pe_k1_sft3732_b4ga4/epoch3` **[E]**。
- **TensorBoard**：无已知地址。
- **替代日志**：`pe_k1_sft3732_b4ga4` run 内 train/eval log **[D]**。
- **关键结果**：约 `+0.390 Avg BLEU`；旧口径平均约 `17.738 BLEU / 48.024 chrF`。
- **状态**：FROZEN / COMPLETE。

## G2. Historical SeqKD controls：Equal3732 / Selected3732 / Full6565 / News50k

- **性质**：HISTORICAL POSITIVE CONTROLS。
- **数据**：Human6565 / PE-selected3732 / random-equal3732 / random News50k 的精确叶文件 **[V]**。
- **脚本**：历史 `scripts/pilot_v2/` 与 `scripts/mtpatcher_v*` 家族 **[F]**。
- **Run/Checkpoint**：各 arm 精确 run **[V]**。
- **TensorBoard**：无已知地址；Human6565 的后续迁移 run 另有 TensorBoard，见 H。
- **关键结果**：
  - SeqKD-Equal3732：约 `+1.050 Avg BLEU`。
  - SeqKD-Selected3732：约 `+1.182`。
  - SeqKD-Full6565：约 `+1.465`。
  - Random News SeqKD50k：约 `+2.250`。
- **解释边界**：属于旧 Human6565/custom stack；后续 Broad20k Full SeqKD `19.1652` 才是当前主 anchor。

## G3. Vanilla selected-source OPD：Exact FKL / Exact RKL / Clean PG-RKL

- **性质**：HISTORICAL ADAPTATION / RQ1。
- **数据**：PE-selected3732；精确 dataset path 见 G1。
- **脚本**：历史 `mtpatcher_ecropd_*` 与 `scripts/mtpatcher_v*` 家族 **[F]**；各 arm 精确文件 **[V]**。
- **Run/Checkpoint**：各 arm/epoch 精确目录 **[V]**。
- **TensorBoard**：无已知地址。
- **替代日志**：KL/runtime audit、epoch checkpoint、三套 held-out eval **[V]**。
- **关键结果**：Exact FKL `≈-0.003`；Exact RKL `≈+0.009`；Clean PG-RKL `≈-0.027 Avg ΔBLEU`。
- **结论**：只对当时 custom implementation/setup 成立；不能外推为 OPD 一般无效。

## G4. Top128 FKL / Entropy-aware OPD

- **性质**：HISTORICAL ABLATION。
- **数据**：PE-selected sources；精确叶文件 **[V]**。
- **脚本/Run/Checkpoint**：精确路径 **[V]**，历史脚本家族见 G3。
- **TensorBoard**：无已知地址。
- **关键结果**：Top128 FKL `+0.062`；Entropy-aware OPD `≈-0.028 Avg ΔBLEU`。
- **结论**：Teacher top128 mass≈1，tail truncation 不是主瓶颈；该分支已停止。

## G5. Correction-trajectory FKL

- **性质**：HISTORICAL DIAGNOSTIC。
- **问题**：Student 错 prefix 是否导致 Teacher 无法有效传递知识？
- **数据/脚本/Run**：精确路径 **[V]**。
- **TensorBoard**：无已知地址。
- **关键结果**：仍约等于 Base。
- **结论**：只把 trajectory 换为 corrected trajectory 不足以恢复提升。

## G6. Correction-NLL localization controls

- **性质**：HISTORICAL ABLATION。
- **Arms**：Patch-only / Random / Halo1 / FullCorr。
- **数据**：PE corrections；精确叶文件 **[V]**。
- **脚本/Run/Checkpoint**：精确路径 **[V]**。
- **TensorBoard**：无已知地址。
- **关键结果**：
  - Patch-NLL：`-0.644432 BLEU / -0.290579 chrF`。
  - Random-NLL：`+0.150531 / +0.375910`。
  - Halo1-NLL：`+0.099279 / +0.272608`。
  - FullCorr-NLL：`+0.285069 / +0.475952`。
- **结论**：hard patch-only support 明显有害；完整 correction trajectory 更稳。

## G7. PatchBoost2 / PatchBoost4 / RandomBoost4

- **性质**：HISTORICAL ABLATION / soft localization。
- **数据/脚本/Run/Checkpoint**：精确路径 **[V]**。
- **TensorBoard**：无已知地址。
- **关键结果**：
  - PatchBoost2：`+0.2672 BLEU / +0.5319 chrF`。
  - PatchBoost4：`+0.3743 / +0.6407`。
  - RandomBoost4：`+0.2509 / +0.4079`。
  - PB4 - Random4：约 `+0.123 BLEU`。
- **结论**：错误定位更适合作为 full-trajectory 上的 soft weighting，而不是 hard mask。

## G8. PatchBoost4 → Clean PG-RKL

- **性质**：HISTORICAL DIAGNOSTIC / supervised-init test。
- **初始化**：PatchBoost4 checkpoint；精确 path **[V]**。
- **数据/脚本/Run**：精确路径 **[V]**。
- **TensorBoard**：无已知地址。
- **关键结果**：PB4 init `+0.374298`；PG-RKL e1 `+0.282733`、e2 `+0.349546`、e3 `+0.313631`。
- **结论**：OPD 没有超过 PB4 初始化；“OPD 只缺 supervised cold start”未获支持。

## G9. RQ2 Historical selection OPD：Selected3732 / Random3732 / Full6565

- **性质**：HISTORICAL MATCHED-CONTROL ADAPTATION。
- **数据**：Selected3732 / Random3732 / Full6565；Random 与 Selected 重叠 2,088/3,732（55.95%）；精确 random/full leaf **[V]**。
- **脚本**：Full6565 修复入口 `scripts/mtpatcher_v10/fix_master_only_and_finish_rq2_v3.sh` **[E]**；训练/eval 精确脚本 **[V]**。
- **Run/Checkpoint**：精确目录 **[V]**。
- **TensorBoard**：无已知地址。
- **关键结果**：
  - Selected3732：WMT `15.4897` / FLORES `19.8777` / Challenge `16.5960` / Avg ΔBLEU `-0.0274`。
  - Random3732：`15.4493 / 19.9171 / 16.2723 / -0.1356`。
  - Full6565：`15.5901 / 19.8814 / 16.2974 / -0.0922`。
  - Selected - Random：约 `+0.108 BLEU`；Full - Selected：约 `-0.065`。
- **结论**：旧 setup 下 selection 只有弱信号，coverage 扩大也未救活 custom OPD。

## G10. EC-ROPD matched512 — old global RNG

- **性质**：HISTORICAL ADAPTATION / CONFOUNDED。
- **数据**：PE clean3732 的 frozen matched512 子集；精确 subset **[V]**。
- **脚本/Run**：EC v2 local-retoken matched512 的精确路径 **[V]**。
- **TensorBoard**：无已知地址。
- **关键结果**：EC-v2 vs Vanilla e1/e2/e3 = `+0.1057 / +0.2440 / +0.0271 BLEU`。
- **解释边界**：额外 resumed draws 消耗同一 global RNG stream，存在 current-rollout RNG 混杂；不可作为最终因果证据。

## G11. EC-ROPD matched512 — RNG-paired

- **性质**：HISTORICAL ADAPTATION / MATCHED CAUSAL PROBE。
- **数据**：与 G10 相同 matched512。
- **脚本**：matched512 Vanilla/EC 精确脚本 **[V]**。
- **Run/Checkpoint**：
  - `/workspace/mtpatcher/runs/mtpatcher_v3_full6565_20260823/rq_matched512_vanilla_rngpaired_v1/epoch{1,2,3}` **[E，集合记法]**。
  - `/workspace/mtpatcher/runs/mtpatcher_v3_full6565_20260823/rq_matched512_ecropd_rngpaired_v1/epoch{1,2,3}` **[E，集合记法]**。
- **评测 Run**：`/workspace/mtpatcher/runs/mtpatcher_v3_full6565_20260823/rq_matched512_rngpaired_eval_v1` **[E]**。
- **TensorBoard**：无已知地址。
- **关键结果**：EC - Vanilla e1 `-0.0336`、e2 `+0.1257`、e3 `+0.1408 BLEU`；e3 chrF `+0.1962`。
- **结论**：小规模正信号保留，但效应小且 Challenge 贡献明显；不能宣称机制已建立。

## G12. EC-ROPD Full3732 scaling

- **性质**：HISTORICAL ADAPTATION / scaling test。
- **数据**：`/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/pe_k1_clean3732.jsonl` **[E]**。
- **脚本**：
  - `mtpatcher_ecropd_full3732_vanilla_fkl_rngpaired_v1.py` **[E，仅文件名已确认]**；SHA256 `a9bd6a2585bd132946b4c1aae3970eb51cb1985601f9d7414d3c4143823f9eac`。
  - `mtpatcher_ecropd_full3732_train_rngpaired_v1.py` **[E，仅文件名已确认]**；SHA256 `b073ac03e57be47aeef1cc541e23f1657d5672b6ffa26607bae1df2948e51768`。
- **Run**：`/workspace/mtpatcher/runs/mtpatcher_v3_full6565_20260823/rq_full3732_vanilla_rngpaired_v1` **[E]**；EC/full completion run 的精确路径 **[V]**。
- **TensorBoard**：无已知地址。
- **状态**：2026-08-28 报告时 Vanilla 仅确认启动；后续旧 RQ2 汇总给出 Full6565 结果。需要服务器原日志判定 Full3732 最终状态，不能用后续 Full6565 替代。

## G13. Historical PDS v13

- **性质**：HISTORICAL LAB ADAPTATION；与后来的 Broad20k K1All+PDS 68,917-row Strong Reproduction 分开。
- **数据**：
  - PDS：`/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/rq3_pds_v13_paperbudget.jsonl` **[E]**。
  - PE+PDS：`/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/rq3_pe_plus_pds_v13_paperbudget.jsonl` **[E]**。
- **Run/Checkpoint**：`/workspace/mtpatcher/runs/mtpatcher_v3_full6565_20260823/rq3_pe_pds_v13_paperbudget_b4ga4/epoch3` **[E]**。
- **Summary**：`/workspace/mtpatcher/runs/mtpatcher_v3_full6565_20260823/rq3_pe_pds_v13_paperbudget_b4ga4_eval_epoch3/rq3_pe_pds_v13_summary.json` **[E]**。
- **脚本**：`scripts/mtpatcher_v13` 或邻近代际的精确文件 **[V]**。
- **TensorBoard**：无已知地址。
- **历史结果**：PE+PDS 相对 PE 约 `+0.395 Avg BLEU`；旧报告另出现约 `+0.7846` 口径，引用前必须以 summary 原文件核验。

## G14. Historical WA v14 generation chain

- **性质**：HISTORICAL LAB ADAPTATION / generation pipeline。
- **数据**：
  - Anchors：`/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/rq3_wa_anchor_jobs_v14.jsonl` **[E]**。
  - Original analog shards：`/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/rq3_wa_analogs_v14/device_*.jsonl` **[E，glob]**。
  - Repaired base：`/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/rq3_wa_analogs_repaired_v14` **[E]**。
  - Final analog target：`/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/rq3_wa_analogs_final_v14` **[E]**。
  - Context jobs：`/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/rq3_wa_context_jobs_v14.jsonl` **[E]**。
  - Final context shards：`/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/rq3_wa_contexts_final_v14/device_*.jsonl` **[E，glob]**。
- **脚本**：`scripts/mtpatcher_v14` **[F]**；具体 generation/repair/freeze scripts **[V]**。
- **Run/checkpoint/log**：精确路径 **[V]**。
- **TensorBoard**：数据生成任务不适用；Student WA SFT 的 TensorBoard 未确认。
- **解释边界**：这是旧 3,732 anchor / 14,928 context-job pipeline，不能与 Broad20k General-MT WA two-seed 最终结果混写。

## G15. Specialized Qwen3-8B Patcher full fine-tuning

- **性质**：SYSTEM ADAPTATION / Patcher-format learning。
- **数据**：Formal20k feedback demonstrations；精确 leaf path **[V]**。
- **代码**：`scripts/mtpatcher_paper_faithful_v2/` **[F]**，已知包括：
  - `paper_repro_v2.py`
  - `run_patcher_quality_gate_v1.sh`
  - `paper_repro_eval512_v1.py`
  - `run_patcher_core_overnight_v1.sh`
- **Run/Checkpoint**：`/workspace/mtpatcher/runs/mtpatcher_v3_full6565_20260823/patcher_qwen3_8b_paperfaith_fullft_v2/checkpoint-4752` **[E]**；另有 checkpoint-1584/3168。
- **TensorBoard**：未确认。
- **关键结果**：3 epochs / 4,752 steps；runtime≈17,053.7s；train_loss≈`0.1967`；末期 loss≈`0.10–0.12`；无 NaN/OOM。
- **解释边界**：训练拟合成功不等价于 held-out correction quality 已建立。

## G16. 早期 Qwen3.5 preliminary reproduction

- **性质**：LEGACY PILOT / METHOD-LEVEL RECONSTRUCTION。
- **模型**：Qwen3.5-4B Student；Qwen3.5-9B Feedbacker；DeepSeek V4 Flash 用于 Teacher/PDS/WA generation。
- **数据**：News Commentary 18.1 zh→en；PE 99，PDS 416，WA 412，独立 test 200；精确本地路径 **[V]**。
- **脚本/Run/Checkpoint/TensorBoard**：精确路径 **[V]**。
- **关键结果**：Base `25.93`；PE `26.22`；PE+PDS `26.33`；PE+PDS+WA `26.97`。
- **解释边界**：模型、数据、evaluator 与当前 Qwen3-0.6B/Broad20k 主线不同；仅作早期方法跑通记录。

---

# H. TensorBoard

## 已确认地址

```text
/workspace/mtpatcher/runs/sft/human6565_qwen3_06b_20260902_114434/tensorboard_log
/workspace/mtpatcher/runs/sft/canonical_seqkd_broad20k_qwen3_06b_8b_20260903_033659/tensorboard_log
/workspace/mtpatcher/runs/sft/canonical_seqkd_broad20k_qwen3_06b_8b_20260903_133457/tensorboard_log
/workspace/mtpatcher/repo/verl-v0.9.0/tensorboard_log/mtpatcher-opd/canonical-fkl-topk-broad20k-qwen3-06b-8b
```

其中第三个是当前 Full SeqKD 正式成功 run；第二个保留历史追溯。第四个包含至少两个 OPD event 文件，当前存在 canonical/recovery provenance 混写，不能直接当作单一干净 run。

## 启动与转发

服务器：

```bash
tensorboard --logdir <LOGDIR> --host 127.0.0.1 --port 6006
```

Windows：

```powershell
ssh -N -L 6006:127.0.0.1:6006 mtpatcher-ascend
```

浏览器：`http://127.0.0.1:6006`

## 没有 TensorBoard 的实验

Targeted 与 Pre-Verl custom Torch-NPU 实验大多没有已确认的 TensorBoard。查看顺序：

1. `summary.json`
2. `manifest.json`
3. `progress.json`
4. `master.log` / train log
5. raw/parsed JSONL outputs

不得为了补 TensorBoard 重跑任何已完成实验。

## 后续实验的 TensorBoard 硬约束

所有会更新模型参数的后续训练实验必须启用 TensorBoard。能够产生有意义时序指标的诊断实验也应启用；纯数据构造、静态审计或只评测任务可标为“不适用”，但必须保留 structured logs。

每个新 run 至少满足：

1. 使用该 run 独有的绝对目录，例如 `<RUN>/tensorboard`。
2. Verl 配置包含 `trainer.logger=['console','tensorboard']`。
3. 启动前显式设置 `TENSORBOARD_DIR=<RUN>/tensorboard`；禁止依赖 Ray worker 的相对工作目录。
4. 第一个训练 step 后验证存在非空 `events.out.tfevents.*`；验证失败时在昂贵训练继续前修复。
5. `manifest.json` 同时记录 TensorBoard 目录、event 文件、project name 和 experiment name。
6. TensorBoard 之外继续保存 `master/train.log`、`progress.json`、`manifest.json`、`summary.json` 与 raw outputs。

Canonical、recovery、不同 seed 和不同 arm 必须使用不同 TensorBoard 目录，不得因复用 `experiment_name` 而混写 event。

---

# I. 待补字段

以下缺口是本清单下一版的明确输入，不代表实验不存在：

1. Strong Reproduction PE/K1All、PDS、General-MT WA 两个 seed 的精确 data leaf、train/eval script、run、checkpoint、summary 与 logger。
2. Random/Same-source SeqKD4960、Random/Selective OPD4960 的精确 config、recipe、run、checkpoint、summary 与 logger。
3. Canonical Full OPD TensorBoard provenance：确认 Event 1 对应的启动/run，并解释 canonical 初始化 PID `2373346` 与现存 event PID `476464` 的差异。
4. Full SeqKD20k 与 Full OPD20k 的 P1/P2/P3 checkpoint 叶目录。
5. Pre-Verl custom Torch-NPU 各实验的源脚本、run、summary 叶路径；目前只对 PE、PDS、WA、EC-ROPD、Patcher 的部分资产获得精确路径。
6. C1/C2/C3、O1/O2/O3、A0、A1 的每个 arm/pass checkpoint 叶目录。
7. WMT24/FLORES/Challenge frozen test JSONL 的精确路径与 SHA256。
8. 早期 Qwen3.5 pilot 的数据、脚本、run、checkpoint 与日志路径。

## 服务器枚举批次 2 说明

- 时间：`2026-09-12 15:14 UTC`。
- 已补齐：Targeted C0、C1/C2/C3、O1/O2/O3、train-set audits、A0、A4、A4b、A1 的主要数据、脚本、manifest、日志、JSONL；A1 trainer、run 内 launcher 与 checkpoint 父目录也已确认。
- 已补入非 targeted 历史证据：PE、SeqKD controls、Vanilla OPD、Correction/PatchBoost、RQ2、EC-ROPD、PDS v13、WA v14、Patcher 与 Qwen3.5 pilot，共 17 个独立条目。
- 非 targeted 服务器枚举没有真正执行成功：终端明确返回 `rg: command not found`。因此不能把“没有输出”解释成“没有实验资产”。
- 原命令中的 `scripts/analysis` 不存在；这只是规划目录，不影响现有实验资产。

## 服务器枚举批次 3 说明

- 时间：`2026-09-12 15:50 UTC`。
- Canonical Full OPD 与 recovery 均确认启用 `console + tensorboard`，运行日志确认 TensorBoard adapter 已初始化。
- 当前只搜索了 `$RUNS/opd`、主仓库下的 `tensorboard_log` 与 `$ROOT/tensorboard_log`，没有覆盖 Verl 源码目录、Ray 临时工作目录及其他启动 cwd。
- 尚未发现物理 event 文件，因此当前状态是“已配置、已初始化、落盘地址待定位”，不能写成“未启用 TensorBoard”。

## 服务器枚举批次 4 说明

- 时间：`2026-09-12 16:04 UTC`。
- 在 Verl 源码目录下找到两个非空 OPD event 文件，TensorBoard 实际落盘已确认。
- PID `1473680` 的 event 与 recovery 日志一致，可映射到 recovery。
- PID `476464` 的 event 尚未完成 run 映射；canonical 初始化日志中出现的是 PID `2373346`。
- 两个 event 文件共用同一 logical run 目录，验证了相对 `TENSORBOARD_DIR` 会破坏 run 级隔离；后续实验必须使用 `<RUN>/tensorboard` 绝对路径。

## 服务器枚举批次 5 / Global closure audit v2

- 时间：`2026-09-12 17:53–17:55 UTC`。
- 审计基线：branch `infra/verl-ascend-parity`，HEAD `8441ee484f52c38502ce11e1f94df66d24686ca6`。
- Git：审计时唯一 worktree 变化为未跟踪的 `docs/research/05_experiment_asset_ledger.md`；staged / unstaged `git diff --check` 均通过。
- Ledger：服务器文件 SHA256 `da7ac557cdeed54efea9005f770112c46555a92a0886d0d4996b58e74eac3957`。
- 全项目主科学数字一致性检查通过：Base、K1All、Random/Same-source SeqKD4960、PDS、Random/Selective OPD4960、Full OPD20k、Historical Full SeqKD20k 均在当前文档体系中找到一致记录。
- Full OPD recovery：`global_step_3750` 存在，closure gate 通过。
- O123：最终 `state.json` 为 PASS；正式 lineage 为 early semantic-preflight v1 failure → corrected `preflight_v3` PASS → O1/O2/O3 formal train/eval completion。旧 `nohup.out` 中 FAIL 属于早期 attempt 追溯证据，不能覆盖后续 formal result。
- NPU：16 个 chip 均无 running process；当前没有正式训练占用设备。
- 残留 multiprocessing resource-tracker / forkserver 为 PPID=1、CPU=0 的 orphan；其启动时间与两个 INVALID A1 attempts 高度吻合，但缺少直接 parent provenance，因此不作为科学 blocker，也不在本次 closure 中主动 kill。
- 文档 absolute-path validator 报 5 个 missing token，其中已确认至少包含 glob `device_*.jsonl` 与集合路径 `epoch{1,2,3}` 被正则截断形成的假阳性；不得把该计数直接解释成 5 个真实资产丢失。
- closure v2 的 broad TensorBoard scan 没有枚举出 OPD event；本轮又对历史精确 OPD TensorBoard 路径执行 direct recheck。direct recheck 发现 event 行数：`6`。详细证据保存在本次冻结提交前的 `/tmp/opd_tensorboard_direct_recheck_20260912.txt`。
- 结论：旧实验在科学结果层面 closure PASS；当前 provenance 收尾任务是把本 ledger 与下一实验 spec 纳入 Git。Prefix-Support 新训练在本次提交中不会启动。

服务器上最优先读取的权威索引：

```text
/workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend/manifests/experiments/targeted/ARTIFACT_PATHS.md
/workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend/manifests/experiments/targeted/README.md
/workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend/scripts/targeted/README.md
/workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend/docs/research/02_experiment_registry.md
/workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend/docs/research/03_repository_map.md
```

## 非 targeted 资产补录命令（只读，兼容无 `rg` 环境）

```bash
cd /workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend

RUNS=/workspace/mtpatcher/runs
DATA=/workspace/mtpatcher/data

echo '=== NON-TARGETED RUN / CHECKPOINT DIRS ==='
find \
  "$RUNS/sft" \
  "$RUNS/opd" \
  "$RUNS/science" \
  "$RUNS/diagnostics" \
  "$RUNS/overnight" \
  "$RUNS/mtpatcher_v3_full6565_20260823" \
  -mindepth 1 -maxdepth 5 -type d 2>/dev/null \
| grep -Eiv '/targeted/' \
| grep -Ei 'seqkd|opd|strong|repro|k1|k2|pds|wa|ecropd|patchboost|correction|patcher|broad20k|checkpoint|epoch|global_step' \
| sort

echo '=== NON-TARGETED LOGS / SUMMARIES / TENSORBOARD ==='
find "$RUNS" -maxdepth 7 -type f \
  \( -name 'summary*.json' -o -name '*summary.json' -o -name 'manifest.json' \
     -o -name 'progress.json' -o -name 'master*.log' -o -name '*.log' \
     -o -name 'events.out.tfevents.*' \) 2>/dev/null \
| grep -Eiv '/targeted/' \
| grep -Ei 'seqkd|opd|strong|repro|k1|k2|pds|wa|ecropd|patchboost|correction|patcher|broad20k|human6565' \
| sort

echo '=== NON-TARGETED SCRIPTS / CONFIGS / RECIPES / MANIFESTS ==='
find scripts configs recipes manifests -type f \
  \( -name '*.py' -o -name '*.sh' -o -name '*.yaml' -o -name '*.yml' -o -name '*.json' \) 2>/dev/null \
| grep -Ev '^scripts/targeted/' \
| grep -Ei 'seqkd|opd|strong|repro|k1|k2|pds|wa|ecropd|patchboost|correction|patcher|broad20k' \
| sort

echo '=== NON-TARGETED DATA ==='
find \
  "$DATA/verl_science_broad20k" \
  "$DATA/mtpatcher_v3_full6565_20260823" \
  -maxdepth 4 -type f 2>/dev/null \
| grep -Eiv '/targeted_' \
| grep -Ei 'seqkd|opd|k1|k2|pds|wa|ecropd|patchboost|correction|human|selected|random|full' \
| sort
```

该命令不修改文件、不启动训练，只输出其他实验的现存路径。

---

# J. 每日实验日志规范

## J1. 日志与清单的边界

- **每日实验日志**：按时间追加当天的问题、操作、失败、结果与决策；保留探索过程。
- **本资产清单**：只登记可追溯的稳定入口；实验状态或资产冻结后再更新。
- **Manifest / structured logs**：保存机器可读协议、进度、指标与哈希；Markdown 不替代它们。
- **服务器状态**：判断进程是否存活时，以进程树、设备占用与 status artifact 为准。

沿用仓库已经存在的每日日志位置与命名规则。若尚无稳定入口，先在 `docs/research/03_repository_map.md` 冻结目录职责，再创建新目录；不要并行建立多个名称相近的日志体系。

## J2. 每个关键实验的最小记录

实验启动前写清：

```text
时间：<ISO-8601，含时区>
实验 ID：<唯一且稳定>
分类：<PAPER-FAITHFUL REPRODUCTION | LAB REPRODUCTION | ADAPTATION |
       ABLATION | DIAGNOSTIC ONLY | INVALID RUN>
状态：<PENDING | RUNNING | PASS | FAIL | SCIENTIFICALLY_BLOCKED | INTERRUPTED>

Question：
Competing explanations：
Falsifiable prediction：
Decision after result：

输入数据：<path + SHA256>
Teacher / Student：<role + checkpoint identity>
配置与命令：<resolved config path + exact command>
代码身份：<branch + commit + worktree status>
环境：<关键软件版本与设备>
预算：<seed + optimizer steps + source/token exposure>
Run：<versioned run directory>
日志：<text log + structured progress + TensorBoard/equivalent>
```

实验结束后追加：

```text
Problem：
Result：<逐数据集指标；同时给 matched comparator>
Interpretation：<结论成立范围与已知混杂>
Next step：<继续、停止、补诊断或等待决策>

最终状态：
Checkpoint：
Summary / raw outputs：
Artifact hashes：
Evaluation command：
```

禁止只写“跑完了”“有效”或一个 aggregate metric。Engineering completion 与 Scientific PASS 必须分开。

## J3. 长任务的可观察性与恢复

长任务至少每个 bounded batch 或约 30–60 秒刷新一次以下字段：

```text
timestamp, phase, done/total, percentage, success, failure,
elapsed, recent throughput, average throughput, ETA,
last durable job/shard
```

同时满足：

1. 使用 ISO-8601 时间并注明 UTC 或 UTC+8。
2. 文本日志及时 flush；维护原子更新的 `progress.json` 或等价状态文件。
3. 每个 unit 使用稳定 `job_id/source_id/row_id`，有效完成项可跳过，失败项可重试。
4. 输入哈希、checkpoint、prompt/config、代码版本或生成参数不一致时，禁止把恢复输出混入原 run。
5. 保存失败与 retry 记录；全局 contract 违规时立即 fail closed。

## J4. 每日收尾

1. 对照进程树、设备状态和日志末尾，更新每个 run 的真实状态。
2. 将失败、重复或污染的 run 标为 `INVALID RUN`，保留原因，不并入正式结果。
3. 核对当天产生的 data、script、config、run、checkpoint、summary 和日志路径。
4. 只有通过相应 gate 的资产才记录哈希并冻结；冻结资产不得原地修改。
5. 把耐久结果同步到实验 registry、本资产清单和 current status；过程性噪声只留在每日日志。
6. 检查 Git diff，避免暂存数据、模型、checkpoint、生成输出和临时修复文件。

---

# K. 对外汇报格式

每个实验保持同一顺序：

```text
实验名 / 编号：
性质与状态：
问题：
数据：
脚本：
Config / Recipe：
Run / Checkpoint：
TensorBoard：
替代日志：
关键结果：
解释边界：
```

若没有 TensorBoard，直接写：

```text
TensorBoard：无
替代日志：<精确 progress / manifest / summary / master log 路径>
```

不要把共享父目录冒充具体文件地址，也不要把 `INVALID_RUN` 混入正式结果。

---

# L. 科学摘要

- Full OPD20k `19.1036` 已基本追平 Full SeqKD20k `19.1652`，不能再说“OPD 整体不工作”。
- K2 Selective OPD4960 `17.9600` 没有超过 Random OPD4960 `17.9861`，当前 selection-only 低预算 OPD 没有优势。
- PDS 有强 downstream utility，但其 68,917 Student rows 已超过 Full SeqKD20k，不能宣称 row efficiency。
- Targeted SFT 能快速写入 Idiom/Chemistry knowledge；knowledge-conditioned OPD 能学到少量，但明显更弱。
- A0/A4/A4b/A1 已排除“只需多训几轮”“lexical signal 被完全稀释”“纯 evaluator noise”等充分解释。
- 下一项最高信息量实验是 Offline Prefix-Support Replay；spec 已冻结，训练尚未启动。

<!-- OFFLINE_PREFIX_SUPPORT_SEED1_RESULT_V1 -->
## E8. Offline Prefix-Support Replay — Seed1 Completed Result

- **性质**：`DIAGNOSTIC ONLY / OFFLINE PREFIX-SUPPORT REPLAY`。
- **状态**：`PASS / NO POSITIVE SEED1 SCREEN IN CHEMISTRY OR IDIOM`。
- **结果 Manifest**：`manifests/experiments/targeted/09_offline_prefix_support_seed1_results.json`。
- **结果说明**：`docs/research/06_offline_prefix_support_seed1_results.md`。
- **Chemistry P3**：S `0.134`，T `0.125`，T-S `-0.009`，DeltaR `-0.054878`，positive screen = `False`。
- **Idiom P3**：S `2.936`，T `2.903`，T-S `-0.033`，DeltaR `-0.063584`，95% CI `[-0.084, 0.018]`，positive screen = `False`。
- **Pre-update caveat**：Idiom T-prefix semantic quality `4.232` vs S `2.377`; Teacher hint-gap T `0.8442` vs S `0.8057`。
- **解释边界**：跨两个 domain 均未观察到 preregistered T-arm downstream advantage；这削弱“Student trajectory/support 单独构成主要瓶颈”的解释，但单 seed/domain 不足以形成负机制结论。
- **Seed governance**：不触发 positive-confirmation seed；禁止 best-of-pass 选择。
- **下一优先级**：监督位置/critical-token/sparse-or-localized KL 与 optimization efficiency。
