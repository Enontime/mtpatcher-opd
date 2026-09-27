# 2026-09-29 维护前 Formal Run / Code / Artifact Ledger

审计时间：2026-09-27 UTC。此文档是服务器维护恢复账本，不改变 `manifests/paper_view.json` 的论文收录权威地位。

## 分类与备份优先级

- `FORMAL_PRIMARY`: frozen scientific contract 下的正式主结果。
- `FORMAL_FAILED_BUT_EVIDENCE`: 正式尝试失败，但失败本身是科研/工程证据。
- `CANONICAL_HISTORICAL`: 历史 canonical anchor 或 strong reproduction 结果。
- `FORMAL_EVALUATION`: 正式训练的评测、replay 或 retrospective。
- `DIAGNOSTIC_ONLY`: 有科学解释价值，但不能替代 primary endpoint。
- `SMOKE_ONLY`: 仅工程 gate。
- `TEMPORARY`: 可再生临时产物。

备份优先级：`P0` 必须在维护前异地备份；`P1` 强烈建议；`P2` 容量允许时备份。

## 核心 formal / canonical 账本

| Experiment | Classification / scientific role | Run path, status, terminal checkpoint | Config / recipe | Data asset / manifest | TensorBoard / validation / generation | Final status / provenance | Backup recommendation / notes |
|---|---|---|---|---|---|---|---|
| Canonical Broad20k Full SeqKD | CANONICAL_HISTORICAL; 20k sources × 3 passes 的 Full SeqKD anchor | `/workspace/mtpatcher/runs/sft/canonical_seqkd_broad20k_qwen3_06b_8b_20260903_133457`; complete; `global_step_3750`; 11G | `configs/sft/canonical_seqkd_broad20k_qwen3_06b_8b.yaml`; `recipes/sft/canonical_seqkd_broad20k_qwen3_06b_8b.sh`; dataset adapter `scripts/data/verl_mt_response_sft_dataset.py` | canonical Broad20k Verl assets built by `scripts/data/build_canonical_broad20k_verl_assets.py` | run `tensorboard_log/`; paper-clean TB at `runs/science/paper_tensorboard_clean_20260915/seqkd_full_20k` | `final_exit_status.txt=0`, compose/contract/torchrun all 0; resolved config and train log in run | P0: step3750, resolved/experiment config, log, TB. Historical P3 Macro BLEU ≈19.1652 |
| Canonical Broad20k Full OPD + recovery | FORMAL_PRIMARY across original segment and recovery; Full OPD anchor | Original `/runs/opd/canonical_fkl_topk_broad20k_qwen3_06b_8b_20260903_034058` stopped at step1250 with status 1, 7.3G. Recovery `/runs/opd/opd_recovery_20260905_v2` completed step3750, status 0, 15G | `configs/opd/canonical_fkl_topk_broad20k_qwen3_06b_8b.yaml`; `recipes/opd/canonical_fkl_topk_broad20k_qwen3_06b_8b.sh`; `scripts/opd/constant_zero_reward.py`; `scripts/infra/opd_recovery_20260905_v2.sh` | canonical Broad20k source parquet; launch manifest `manifests/experiments/canonical_seqkd_vs_opd_launch_20260903_v1.yaml` | paper-clean TB split under `runs/science/paper_tensorboard_clean_20260915/opd_full_20k`; formal evaluation artifacts under `runs/science/canonical_seqkd_vs_opd_20260903_033654` | original runtime science provenance; recovery manifest, override diff, patch, SHA256, resolved config | P0: both step1250 continuity checkpoint and recovery step3750 endpoint plus provenance/TB. P3 Macro BLEU 19.1036 |
| matched20k-v2 Full SeqKD formal_v1 | FORMAL_PRIMARY; matched 120k exposures, 7500 updates, 100-step trajectory | `/runs/science/matched20k_v2/seqkd_formal_v1`; complete to `global_step_7500`; 357G | `configs/sft/matched20k_v2_seqkd_formal_v1.yaml`; `recipes/matched20k_v2/run_seqkd_formal_v1.sh`; replay `run_seqkd_validation_replay_v1.sh` | `manifests/matched20k_v2/formal_protocol_v1.json`; `model_provenance_v1.json`; matched assets and `validation3231` paths/hashes in protocol | run `tensorboard/`; `validation/{generations,logs,status,configs}`; step7500 status exists with Macro BLEU 19.4129753 | provenance contains formal/resolved config and runtime manifest; no top-level `final_status.txt` | P0: step7500 + provenance + complete validation/TB. P1: all 100-step checkpoints because frozen retrospective convergence selection depends on them |
| matched20k-v2 OPD formal_v1 | FORMAL_FAILED_BUT_EVIDENCE; first formal attempt | `/runs/science/matched20k_v2/opd_formal_v1`; `final_status.txt=1`; no checkpoint; 248K | `configs/opd/matched20k_v2_opd_formal_v1.yaml`; `recipes/matched20k_v2/run_opd_formal_v1.sh` | same matched20k protocol/data | one TensorBoard event; no generations | five provenance files and failed final status | P0: whole small run. Preserve as evidence; never overwrite or relabel as smoke |
| matched20k-v2 OPD formal_v2 | FORMAL_PRIMARY; Ascend-compatible repaired formal OPD | `/runs/science/matched20k_v2/opd_formal_v2`; `final_status.txt=0`; `global_step_7500`; 545G | `configs/opd/matched20k_v2_opd_formal_v2.yaml`; `recipes/matched20k_v2/run_opd_formal_v2.sh`; replay `run_opd_validation_replay_v1.sh` | same protocol/data; `manifests/matched20k_v2/opd_formal_contract_v1.json` | run TB plus persistent replay under `validation/`; continuation generations through 7500; diagnostic extension summary at `analysis/seqkd_vs_opd_with_extension_v1` | formal protocol/config/runtime provenance; formal step7500 Macro BLEU 19.8303871. Step10000 extension is DIAGNOSTIC_ONLY | P0: step7500 + provenance + validation/TB. P1: all formal 100-step checkpoints. Do not substitute diagnostic best step9500 |
| PE11792 SFT formal_v1 | FORMAL_PRIMARY; PE-selected fixed post-edit SFT comparator | `/runs/science/pe_pds_v1/pe_sft_formal_v1`; status 0; `global_step_4422`; 214G | `configs/sft/pe11792_sft_formal_v1.yaml`; `recipes/pe_pds/run_pe11792_sft_formal_v1.sh`; chain `run_pe11792_formal_chain_v1.sh` | `manifests/pe_pds/pe11792_matched_formal_protocol_v1.json`; run provenance `asset_manifest.json` | run `tensorboard/`; 45 generation-like files | `final_status.txt=0`; formal protocol/config/asset manifest in provenance | P0: step4422, config/manifest/provenance/TB/generations. P1: 100-step trajectory checkpoints |
| PE11792 OPD formal_v1 | FORMAL_PRIMARY; identical PE source trajectory, online top-k FKL; PDS common stage-entry source | `/runs/science/pe_pds_v1/pe_opd_formal_v1`; status 0; `global_step_4422`; 327G | `configs/opd/pe11792_opd_formal_v1.yaml`; `recipes/pe_pds/run_pe11792_opd_formal_v1.sh`; chain `run_pe11792_formal_chain_v1.sh` | PE11792 protocol and run `asset_manifest.json` | run `tensorboard/`; 45 generation-like files | `final_status.txt=0`; 11 provenance files | P0 highest: step4422 actor + merged stage entry `/runs/science/pe_pds_v1/pds9952_stage_entry_v1/pe_opd_step4422_merged_hf`, provenance/TB. P1: trajectory checkpoints |
| PDS9952 SeqKD formal_v1 | FORMAL_PRIMARY; fixed-target arm on frozen parent-diverse sources | `/runs/science/pe_pds_v1/pds9952_seqkd_formal_v1`; PASS; `global_step_3732`; 181G | `configs/sft/pds9952_seqkd_formal_v1.yaml`; `recipes/pe_pds/run_pds9952_formal_training_v2.sh` | `manifests/pe_pds/pds9952_matched_formal_protocol_v1.json`; builder `scripts/pe_pds/build_pds9952_matched_assets.py` | run `tensorboard/`; 38 generation-like files; endpoint and retrospective paths below | `PASS`, `formal_status.txt=0`, latest checkpoint 3732, three provenance files | P0 highest: step3732 HF model + config/provenance/TB. P1: all 100-step checkpoints for paired trajectory |
| PDS9952 OPD formal_v1 | FORMAL_PRIMARY; online forward-KL arm on same PDS9952 sources/order/init | `/runs/science/pe_pds_v1/pds9952_opd_formal_v1`; PASS; `global_step_3732`; 276G | `configs/opd/pds9952_opd_formal_v1.yaml`; `recipes/pe_pds/run_pds9952_formal_training_v2.sh` | same PDS9952 protocol/asset manifest | run `tensorboard/`; 38 generation-like files | `PASS`, `formal_status.txt=0`, latest checkpoint 3732, three provenance files | P0 highest: step3732 actor shards and merged endpoint, config/provenance/TB. P1: all 100-step checkpoints |
| PDS9952 endpoint_validation_v4 | FORMAL_EVALUATION; preregistered step0 vs step3732 endpoint comparison | `/runs/science/pe_pds_v1/pds9952_endpoint_validation_v4`; PASS; no training checkpoint; 1.2G | `recipes/pe_pds/run_pds9952_endpoint_validation_v4.sh`; shared validator `scripts/matched20k_v2/{run_opd.py,opd_trainer.py,shared_mt_validation.py}` | `validation3231` plus model hashes in `provenance/` | three arms each have raw generation + TB + status; `endpoint_summary.json` | PASS. Macro BLEU: step0 19.3197136; SeqKD 19.4276352; OPD 19.4553738; OPD-SeqKD +0.0277386. Single-seed descriptive only | P0: whole 1.2G directory, especially summary, raw generations, TB, source-state and model SHA256 provenance |
| PDS9952 retrospective_v1 | FORMAL_EVALUATION, IN_PROGRESS at audit | `/runs/science/pe_pds_v1/pds9952_retrospective_v1`; 117M at audit. OPD generations exist through 3732 (39 JSONL). Active SeqKD native persistent replay had reached step1400 (15 JSONL) | SeqKD launcher `recipes/pe_pds/run_pds9952_seqkd_retrospective_v1.sh`; persistent trainer/validator code above | formal checkpoints from both PDS arms | `opd/`, `seqkd/`, and native persistent repair variants; TB/provenance present | no final PASS marker at audit; active PID used evalmap tree | P0 after completion: immutable completion snapshot of all configs/generations/TB/provenance/status. Do not back up a live mutable directory without recording timestamp and process state |
| Recovered Formal SeqKD P1 RoPE evaluator check | FORMAL_EVALUATION / repair evidence, literal exact gate FAIL | `/runs/science/seqkd_p1_rope_fix_final_20260909_v1`; 2.0M; evaluates formal step1250 model | evaluator repair tooling documented in research docs; no new training recipe | WMT24/FLORES/Challenge predictions and metric JSON | `eval/*.pred.jsonl`, logs, metrics | `final_status=FAIL`: recovered 18.5177147 vs historical 18.5203421, delta -0.0026274 under unrealistic 1e-6 equality tolerance | P0: whole small evaluation directory. Scientifically demonstrates near recovery; must retain literal FAIL semantics |

## Additional scientifically important discovered families

| Experiment family | Classification / role | Exact primary paths | Status / endpoint | Backup recommendation |
|---|---|---|---|---|
| Strong reproduction PE/PDS | CANONICAL_HISTORICAL; paper-aligned adaptation evidence | `/runs/mtpatcher_v3_full6565_20260823/strong_repro_student6_v1/K1_ALL_PE11792`; `/runs/mtpatcher_v3_full6565_20260823/strong_repro_pds_student2_v2/K1ALL_PDS`; matched repeat sibling `K1_PARENT_MATCHED_REPEAT` | Complete frozen results; each displayed model dir ~3.4G. PDS 18.5696 vs repeat 16.7592 | P0: three endpoint model/result dirs plus upstream PDS accepted asset and manifests. These support the strongest PDS mechanism claim |
| Strong reproduction WA two-seed | CANONICAL_HISTORICAL; null/weak general-MT result | `/runs/mtpatcher_v3_full6565_20260823/strong_repro_pe_pds_wa_full_v1`; `/runs/mtpatcher_v3_full6565_20260823/wa_paired_seed2_v1/{PE_PDS_WA,PE_PDS}` | two seeds, mean incremental BLEU ≈0 | P1: results, generation/scoring outputs, manifests; endpoint weights P2 unless capacity permits |
| Low-budget Selected/Random SeqKD comparators | CANONICAL_HISTORICAL | `/runs/mtpatcher_v3_full6565_20260823/strong_repro_student6_v1/{RANDOM_SEQKD4960,SAME_SOURCE_SEQKD4960}` | Complete; Macro BLEU 18.2658 / 18.4151 | P1 endpoints/results/provenance |
| Low-budget Selected/Random OPD4960 | FORMAL_PRIMARY comparator family | `/runs/opd/selective_opd_k2_4960_20260909_172516`; `/runs/opd/random_opd_4960_20260909_223550` | both final status 0, terminal `global_step_930`, ~22G each; Macro BLEU 17.9600 / 17.9861 | P0 endpoints, resolved configs, logs, final status; P1 intermediate checkpoints/TB |
| Targeted SFT / knowledge-conditioned OPD | FORMAL_PRIMARY within targeted adaptation line | `/runs/targeted/wa_sft_positive_control_c123_20260910`; `/runs/targeted/wa_opd_knowledge_conditioned_o123_20260911` | Complete; exact assets listed in `manifests/experiments/targeted/` | P1: manifest-listed final summaries, translations, judged/scored files, training manifests; endpoint weights P2 |
| Offline Prefix-Support Replay seed1 | DIAGNOSTIC_ONLY, frozen result | Chemistry `/runs/targeted/offline_prefix_support_chemistry_st_seed1_native_verl_v3_20260913`; Idiom `/runs/targeted/offline_prefix_support_idiom_st_seed1_native_verl_v1_20260913`; posttrain analysis dirs recorded in result manifest | PASS execution, no positive screen; cannot support a general negative claim | P1: all SHA-listed artifacts and TB from `manifests/experiments/targeted/09_offline_prefix_support_seed1_results.json` |
| matched20k OPD extension to 10000 | DIAGNOSTIC_ONLY | `/runs/diagnostics/matched20k_v2/opd_formal_v2_budget_extension8_v1`; summary under formal analysis dir | step10000 worse than formal step7500 by -0.10095 Macro BLEU; diagnostic best step9500 only | P2: summary/provenance/TB; checkpoint backup lower priority because it cannot replace formal endpoint |
| SeqKD engineering/formalization trees whose internal dirs contain `formal` | DIAGNOSTIC_ONLY, not primary | `/runs/diagnostics/seqkd_*`, `/runs/overnight/seqkd_*` | construction, scheduler, FSDP, gradient, RoPE, optimizer-boundary and cross-NPU diagnostics | P2: compact provenance/status/decision files; large checkpoints only if a unique root-cause proof is not reproducible elsewhere |
| GRPO `bleu1024_formal` pilot | CANONICAL_HISTORICAL pilot, not current MT-PATCHER/OPD primary | `/runs/pilot_v2_grpo/bleu1024_formal_ascend_20260822`; evaluation sibling under `pilot_v2_grpo_eval` | historical pilot | P2: compact result/provenance only |
| PE/PDS source-pivot, shift-control, EAEC runs | DIAGNOSTIC_ONLY / SMOKE_ONLY | `/runs/diagnostics/pe_pds_v1/*` | causal pilots, alignment/localization probes and smoke gates; code archived separately | P2: final summaries/audits/provenance; do not prioritize checkpoints over formal endpoints |

## Code identity: formal Verl versus evalmap Verl

### Formal Verl

- Path: `/workspace/mtpatcher/repo/verl-v0.9.0-matched20k-v2-formal`
- Branch: `matched20k-v2-formal-runtime`
- HEAD: `ee2060725c3d5deff0d77ab4ffab4f30cd283dab`
- Worktree: clean.

### Evaluation-only evalmap Verl

- Path: `/workspace/mtpatcher/repo/verl-v0.9.0-matched20k-v2-formal-evalmap-v1-20260927`
- Branch: `matched20k-v2-formal-runtime`
- HEAD: same `ee2060725c3d5deff0d77ab4ffab4f30cd283dab`
- Worktree: modified `verl/utils/checkpoint/fsdp_checkpoint_manager.py`; generated `__pycache__` differs but is not source.
- Exact source diff: the `torch.load(local_model_path, weights_only=False)` call at the model-shard load site adds `map_location="npu:0"`.
- Formal file SHA256: `72afa38272a99962ae9850afabe5b7353fc558b24cce9aaf77825bbe86bc749e`.
- Evalmap file SHA256: `e23075f9a393aef98a5e112aec20815ed911ff96c2511b27c8fcb81123cec44e`.
- This evalmap tree is evaluation-only and must not be copied wholesale into project Git. P0 backup: exact diff, both hashes, HEAD/status, and patched source file.

## Off-server backup manifest by priority

### P0 — must back up

1. Git state: post-archive repository bundle, PRE/POST heads, commit list, and this ledger.
2. Final checkpoints: canonical SeqKD 3750; canonical OPD original 1250 + recovery 3750; matched20k SeqKD/OPD 7500; PE SFT/OPD 4422; PDS9952 SeqKD/OPD 3732; PDS stage-entry merged PE-OPD step4422.
3. PDS9952 endpoint validation v4 entire directory, including raw generations, summary, TensorBoard and provenance hashes.
4. matched20k/PDS/PE protocols, asset manifests, resolved configs, source-order manifests and validation assets with hashes.
5. Strong-reproduction PDS treatment + parent-repeat endpoints and the frozen upstream accepted PDS artifact.
6. Failed formal evidence: matched20k OPD formal_v1 and recovered SeqKD P1 exact-gate FAIL directory.
7. The exact evalmap patch/hashes and both Verl HEAD/status records.

### P1 — strongly recommended

1. All 100-step checkpoints for matched20k and PE/PDS formal runs because trajectory/convergence analysis is part of the frozen contract.
2. All formal TensorBoard event files and validation raw generations/case dumps.
3. Random/Selective OPD4960 endpoints plus resolved configs/logs; Random/Same-source SeqKD comparator endpoints.
4. All SHA-listed targeted result artifacts and judged/scored generation files.
5. Completed PDS9952 retrospective directory after final status is frozen.

### P2 — capacity permitting

1. Diagnostic checkpoint trees under `runs/diagnostics` and `runs/overnight`.
2. Smoke outputs, temporary Hydra outputs, Ray logs/caches and torch compile debug are not backup priorities unless needed for a unique incident investigation.

## Recovery order after maintenance

1. Verify Git bundle and checkout `infra/verl-ascend-parity` at POST_HEAD.
2. Verify project status and reapply/configure the missing Git remote before any push.
3. Verify formal Verl HEAD/clean state and separately reconstruct/verify the one-line evalmap patch by SHA256.
4. Verify P0 checkpoint/model hashes and manifest-referenced data assets before running evaluation.
5. Inspect the PDS9952 retrospective final process/status snapshot; resume only from its documented state, never by overwriting existing outputs.
6. Do not start new scientific experiments as part of recovery.
