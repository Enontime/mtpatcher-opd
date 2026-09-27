# 2026-09-29 华为平台维护前 Git 审计

审计执行时间：2026-09-27 UTC。计划维护日期：2026-09-29。

## 审计上下文

- Project: `/workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend`
- Branch: `infra/verl-ascend-parity`
- PRE_HEAD: `42442473c7481306aa00b756e168bcf0a6a12833`
- 修改前快照：`/workspace/mtpatcher/maintenance_20260929/`
- 修改前 bundle: `project_before_archive.bundle`，`git bundle verify` PASS，包含完整 history。
- Git remote: 未配置；因此本地归档可完成，但无法 push。
- 活跃任务：PDS9952 SeqKD native persistent retrospective 正在运行；未 kill、未 restart，也未改写其依赖源码。
- 控制文档异常：`AGENT4.md` 和 `AGENTS.md` 均要求读取 `AGENT3.md`，但当前工作树不存在该文件。

## 分类定义

- `KEEP_AND_COMMIT`: 小型、可解释、可复现的科研代码/配置/协议/文档。
- `KEEP_BUT_NOT_COMMIT`: 必须保留在服务器或异地备份，但不进入 Git 的大体积实验资产。
- `GENERATED_IGNORE`: 可再生的编译、缓存、临时输出。
- `SENSITIVE_IGNORE`: 即使当前为空也不得误提交的敏感路径。
- `UNKNOWN_REVIEW`: 尚不能确定价值或安全性；本次审计无此类剩余项。

## Modified / untracked 文件逐项分类

### Git 安全与 persistent validation

| Path | Classification | Reason | Formal relation | Action |
|---|---|---|---|---|
| `.gitignore` | KEEP_AND_COMMIT | 明确隔离根级 `ssh` 和 `torch_compile_debug/` | 全仓库安全门 | 第三提交 |
| `scripts/matched20k_v2/opd_trainer.py` | KEEP_AND_COMMIT | 将 non-thinking marker 从硬失败改为记录告警；与 endpoint v4 和 persistent replay 的实际语义一致 | matched20k/PDS retrospective | 按已有内容归档；未重写运行中源码 |
| `docs/research/PRE_MAINTENANCE_20260929_GIT_AUDIT.md` | KEEP_AND_COMMIT | 本次逐文件 Git 审计 | 维护恢复 | 第三提交 |
| `docs/research/PRE_MAINTENANCE_20260929_FORMAL_RUN_LEDGER.md` | KEEP_AND_COMMIT | formal/canonical run 恢复账本 | 维护恢复 | 第三提交 |

### PDS9952 formal 主线

| Path | Classification | Reason | Formal relation | Action |
|---|---|---|---|---|
| `docs/research/MT_PATCHER_OPD_PDS_SCIENTIFIC_CONTRACT_v1.md` | KEEP_AND_COMMIT | frozen D0–D7 科研合同 | PDS9952 formal | commit `22d5093` |
| `manifests/pe_pds/pds9952_matched_formal_protocol_v1.json` | KEEP_AND_COMMIT | population/order/stage-entry/evaluation contract | PDS9952 formal | commit `22d5093` |
| `configs/sft/pds9952_seqkd_formal_v1.yaml` | KEEP_AND_COMMIT | formal SeqKD 可复现配置 | PDS9952 SeqKD | commit `22d5093` |
| `configs/opd/pds9952_opd_formal_v1.yaml` | KEEP_AND_COMMIT | formal OPD 可复现配置 | PDS9952 OPD | commit `22d5093` |
| `scripts/pe_pds/build_pds9952_matched_assets.py` | KEEP_AND_COMMIT | deterministic population/schedule/asset builder | PDS9952 formal | commit `22d5093` |
| `recipes/pe_pds/run_pds9952_formal_chain_v1.sh` | KEEP_AND_COMMIT | stage-entry 与 formal chain | PDS9952 formal | commit `22d5093` |
| `recipes/pe_pds/run_pds9952_formal_training_v1.sh` | KEEP_AND_COMMIT | 首版正式训练执行 provenance | PDS9952 formal history | commit `22d5093` |
| `recipes/pe_pds/run_pds9952_formal_training_v2.sh` | KEEP_AND_COMMIT | 实际完成 run 的修复版训练入口 | PDS9952 formal primary | commit `22d5093` |
| `recipes/pe_pds/run_pds9952_smoke1_v1.sh` | KEEP_AND_COMMIT | formal training 的前置工程 gate，可复现但不是 formal result | PDS9952 smoke | commit `22d5093` |
| `recipes/pe_pds/run_pds9952_endpoint_validation_v1.sh` | KEEP_AND_COMMIT | evaluator 修复历史 | PDS9952 formal evaluation history | commit `22d5093` |
| `recipes/pe_pds/run_pds9952_endpoint_validation_v3.sh` | KEEP_AND_COMMIT | evaluator 修复历史 | PDS9952 formal evaluation history | commit `22d5093` |
| `recipes/pe_pds/run_pds9952_endpoint_validation_v4.sh` | KEEP_AND_COMMIT | 当前 PASS 的 endpoint evaluator | PDS9952 formal evaluation | commit `22d5093` |
| `recipes/pe_pds/run_pds9952_seqkd_retrospective_v1.sh` | KEEP_AND_COMMIT | checkpoint trajectory replay 入口 | PDS9952 formal retrospective | commit `22d5093` |

### EAEC / source-pivot / shift-control 历史科研代码

这些文件有明确的机制研究与失败/诊断 provenance，但不属于当前 PDS9952 formal primary；统一单独归档在 commit `3478b1c`。

| Path | Classification | Reason | Formal relation | Action |
|---|---|---|---|---|
| `configs/opd/pe11792_eaec_opd_r2_v1.yaml` | KEEP_AND_COMMIT | EAEC method config | historical/diagnostic | commit `3478b1c` |
| `configs/opd/pe11792_shiftctrl_pilot100_v2.yaml` | KEEP_AND_COMMIT | causal pilot control | diagnostic | commit `3478b1c` |
| `configs/opd/pe11792_shiftctrl_pilot700_v1.yaml` | KEEP_AND_COMMIT | extended pilot control | diagnostic | commit `3478b1c` |
| `configs/opd/pe11792_shiftctrl_smoke1_v2.yaml` | KEEP_AND_COMMIT | engineering gate config | smoke only | commit `3478b1c` |
| `configs/opd/pe11792_sourcepivot_pilot100_v2.yaml` | KEEP_AND_COMMIT | source-pivot pilot | diagnostic | commit `3478b1c` |
| `configs/opd/pe11792_sourcepivot_pilot700_v1.yaml` | KEEP_AND_COMMIT | extended source-pivot pilot | diagnostic | commit `3478b1c` |
| `configs/opd/pe11792_sourcepivot_tq_smoke_v1.yaml` | KEEP_AND_COMMIT | transport/quality smoke config | smoke only | commit `3478b1c` |
| `configs/opd/pe11792_sourcepivot_uniform_pilot700_v1.yaml` | KEEP_AND_COMMIT | uniform comparator | diagnostic | commit `3478b1c` |
| `recipes/pe_pds/run_eaec_alignment_recovery1024_v1.sh` | KEEP_AND_COMMIT | alignment recovery recipe | diagnostic | commit `3478b1c` |
| `recipes/pe_pds/run_eaec_one_step_smoke_v1.sh` | KEEP_AND_COMMIT | EAEC one-step gate | smoke only | commit `3478b1c` |
| `recipes/pe_pds/run_eaec_persistence1024_v1.sh` | KEEP_AND_COMMIT | persistence probe | diagnostic | commit `3478b1c` |
| `recipes/pe_pds/run_eaec_strict_core_holdout1024_v2.sh` | KEEP_AND_COMMIT | strict-core holdout | diagnostic | commit `3478b1c` |
| `recipes/pe_pds/run_localization_benchmark128_v1.sh` | KEEP_AND_COMMIT | localization benchmark recipe | diagnostic | commit `3478b1c` |
| `recipes/pe_pds/run_pe11792_shiftctrl_pilot100_v2.sh` | KEEP_AND_COMMIT | shift-control pilot launcher | diagnostic | commit `3478b1c` |
| `recipes/pe_pds/run_pe11792_shiftctrl_pilot700_v1.sh` | KEEP_AND_COMMIT | shift-control extended launcher | diagnostic | commit `3478b1c` |
| `recipes/pe_pds/run_pe11792_shiftctrl_smoke1_v2.sh` | KEEP_AND_COMMIT | shift-control gate | smoke only | commit `3478b1c` |
| `recipes/pe_pds/run_pe11792_sourcepivot_pilot100_v2.sh` | KEEP_AND_COMMIT | source-pivot pilot launcher | diagnostic | commit `3478b1c` |
| `recipes/pe_pds/run_pe11792_sourcepivot_pilot700_v1.sh` | KEEP_AND_COMMIT | source-pivot extended launcher | diagnostic | commit `3478b1c` |
| `recipes/pe_pds/run_pe11792_sourcepivot_tq_smoke_v1.sh` | KEEP_AND_COMMIT | source-pivot transport gate | smoke only | commit `3478b1c` |
| `recipes/pe_pds/run_pe11792_sourcepivot_uniform_pilot700_v1.sh` | KEEP_AND_COMMIT | uniform comparator launcher | diagnostic | commit `3478b1c` |
| `recipes/pe_pds/run_sourcepivot_causal_pilot100_chain_v2.sh` | KEEP_AND_COMMIT | matched causal pilot chain | diagnostic | commit `3478b1c` |
| `scripts/pe_pds_v1/__init__.py` | KEEP_AND_COMMIT | package identity | historical support | commit `3478b1c` |
| `scripts/pe_pds_v1/build_eaec_patchbank_v1.py` | KEEP_AND_COMMIT | deterministic EAEC patchbank builder | historical/diagnostic | commit `3478b1c` |
| `scripts/pe_pds_v1/eaec_agent_loop_manager_v1.py` | KEEP_AND_COMMIT | EAEC execution implementation | historical/diagnostic | commit `3478b1c` |
| `scripts/pe_pds_v1/eaec_alignment_diag_manager_v1.py` | KEEP_AND_COMMIT | alignment diagnostic manager | diagnostic | commit `3478b1c` |
| `scripts/pe_pds_v1/eaec_alignment_v1.py` | KEEP_AND_COMMIT | frozen alignment logic | diagnostic support | commit `3478b1c` |
| `scripts/pe_pds_v1/eaec_core_refine_dev_v1.py` | KEEP_AND_COMMIT | edit-core refinement research | diagnostic | commit `3478b1c` |
| `scripts/pe_pds_v1/eaec_core_v1.py` | KEEP_AND_COMMIT | EAEC core implementation | historical/diagnostic | commit `3478b1c` |
| `scripts/pe_pds_v1/eaec_strict_core_diag_manager_v2.py` | KEEP_AND_COMMIT | strict-core diagnostic execution | diagnostic | commit `3478b1c` |
| `scripts/pe_pds_v1/eaec_strict_surviving_core_v2.py` | KEEP_AND_COMMIT | surviving-core extraction | diagnostic | commit `3478b1c` |
| `scripts/pe_pds_v1/eval_source_pivot_alignment_dev_v1.py` | KEEP_AND_COMMIT | source-pivot alignment evaluator | diagnostic | commit `3478b1c` |
| `scripts/pe_pds_v1/eval_ter_beam_projection_dev_v1.py` | KEEP_AND_COMMIT | TER projection evaluator | diagnostic | commit `3478b1c` |
| `scripts/pe_pds_v1/eval_xcomet_verifier_dev_v1.py` | KEEP_AND_COMMIT | xCOMET DEV verifier | diagnostic | commit `3478b1c` |
| `scripts/pe_pds_v1/eval_xcomet_verifier_test_frozen_v1.py` | KEEP_AND_COMMIT | frozen TEST verifier | diagnostic | commit `3478b1c` |
| `scripts/pe_pds_v1/localization_benchmark_export_manager_v1.py` | KEEP_AND_COMMIT | localization benchmark exporter | diagnostic | commit `3478b1c` |
| `scripts/pe_pds_v1/patch_verl_distillation_weights_v1.py` | KEEP_AND_COMMIT | exact Verl patch generator | historical transport support | commit `3478b1c` |
| `scripts/pe_pds_v1/test_eaec_core_v1.py` | KEEP_AND_COMMIT | EAEC core executable test | diagnostic support | commit `3478b1c` |

### 不提交项

| Path | Classification | Reason | Formal relation | Action |
|---|---|---|---|---|
| `ssh` | SENSITIVE_IGNORE | 根级敏感名称；当前是 0-byte regular file，SHA256 为 empty-file hash，但绝不提交 | none | `.gitignore` 增加 `/ssh`；保留原文件，不读取内容 |
| `torch_compile_debug/.../fx_graph_readable.py` | GENERATED_IGNORE | TorchInductor debug output | none | `.gitignore` 增加 `/torch_compile_debug/` |
| `torch_compile_debug/.../fx_graph_runnable.py` | GENERATED_IGNORE | TorchInductor debug output | none | ignore，不删除 |
| `torch_compile_debug/.../fx_graph_transformed.py` | GENERATED_IGNORE | TorchInductor debug output | none | ignore，不删除 |
| `torch_compile_debug/.../ir_post_fusion.txt` | GENERATED_IGNORE | TorchInductor debug output | none | ignore，不删除 |
| `torch_compile_debug/.../ir_pre_fusion.txt` | GENERATED_IGNORE | TorchInductor debug output | none | ignore，不删除 |
| `torch_compile_debug/.../output_code.py` | GENERATED_IGNORE | TorchInductor debug output | none | ignore，不删除 |
| `/workspace/mtpatcher/runs/**` | KEEP_BUT_NOT_COMMIT | checkpoints, TensorBoard, raw generations, logs, provenance | all experiments | read/index/backup only |
| `/workspace/mtpatcher/data/**` | KEEP_BUT_NOT_COMMIT | frozen/materialized data assets | all experiments | read/index/backup only |
| `/workspace/mtpatcher/models/**` | KEEP_BUT_NOT_COMMIT | base/teacher/merged model weights | all experiments | read/index/backup only |

## Secret / large-file gate

- Candidate MIME/type/size audit: PASS; no candidate exceeded 20 MiB and no model/checkpoint/TensorBoard/parquet/JSONL entered staging.
- Private-key/cloud-token/API-secret signature scan: no hits in staged candidates.
- Shell syntax: PASS for all candidate `.sh` files.
- Python AST, JSON, YAML parse: PASS for all 52 then-visible candidate paths.
- `git diff --cached --check`: PASS before every commit.
- Explicit staging only; `git add .` was never used.

## Unresolved items

1. Repository has no configured remote, so `git push origin infra/verl-ascend-parity` cannot run.
2. `AGENT3.md` is referenced as a hard contract but absent from the current tree.
3. PDS9952 SeqKD persistent retrospective was active during the archive. Its mutable run directory must be backed up only after a completion/status snapshot is taken.
4. matched20k SeqKD formal run has a complete step-7500 checkpoint and 0–7500 validation status records but no top-level `final_status.txt`.
5. `seqkd_p1_rope_fix_final_20260909_v1` is scientifically useful evaluator-repair evidence, but its literal exact-equality gate is `FAIL` because 18.5177147 differs from historical 18.5203421 by 0.0026274 BLEU.
