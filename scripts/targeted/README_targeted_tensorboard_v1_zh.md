# Targeted OPD/SFT TensorBoard 监控 v1

本模块为 Chemistry/Idiom targeted 实验提供两条互不污染的路径：

1. `tensorboard_bridge_v1.py` 只读解析历史日志与已有评测结果，在独立版本化目录生成 event；
2. `_targeted_tensorboard_v1.py` 为未来训练提供默认关闭、仅主进程写入的原生记录接口。

冻结 OPD runner、旧 checkpoint、训练数据和评测产物均不会被修改。

## 1. 只读检查历史资产

```bash
source /workspace/mtpatcher/project_env.sh

python scripts/targeted/tensorboard_bridge_v1.py inspect \
  --preset chemistry
```

`chemistry` 预设覆盖：

- Chemistry OPD 1e-6 Horizon-5 O2；
- Chemistry OPD 1e-5、2e-5、3e-5 O2；
- 历史 C2-SFT；
- A1-SFT Horizon-5 的可发现 arm 与 P1–P5；
- 已完成的 `KC_SFT_MATCHED_PROMPT_LR2E5`。

也可以追加或覆盖目录：

```bash
python scripts/targeted/tensorboard_bridge_v1.py inspect \
  --preset chemistry \
  --run my_idiom_run=/absolute/path/to/run
```

`inspect` 不导入 TensorBoard，不创建目录。它会报告每个实验实际发现的日志、标量、SHA、缺失项和 `NOT_AVAILABLE` 机制诊断。为绑定 checkpoint 身份，它会只读计算模型权重 SHA；首次扫描多个 P1–P5 checkpoint 可能耗时较长，但不会加载模型或使用 NPU。

## 2. 历史日志转换

```bash
python scripts/targeted/tensorboard_bridge_v1.py convert \
  --preset chemistry \
  --output /workspace/mtpatcher/runs/targeted/tensorboard_bridge_v1_20261015
```

每个 logical run 写到：

```text
<output>/targeted_tensorboard_bridge_v1/<run_id>/<conversion_sha_prefix>/
  event_manifest.json
  events/events.out.tfevents.*
```

转换目录由源文件 SHA、标量内容、checkpoint/eval 身份和转换器版本共同寻址。同一输入重复转换返回 `ALREADY_CURRENT`，不会向原 event 追加重复曲线。源日志始终只读。

验证 event：

```bash
python scripts/targeted/tensorboard_bridge_v1.py validate \
  --output /workspace/mtpatcher/runs/targeted/tensorboard_bridge_v1_20261015
```

也可以验证单个转换目录：

```bash
python scripts/targeted/tensorboard_bridge_v1.py validate \
  --event-dir /absolute/path/to/<conversion_sha_prefix>
```

## 3. 打开 TensorBoard

```bash
tensorboard \
  --logdir /workspace/mtpatcher/runs/targeted/tensorboard_bridge_v1_20261015/targeted_tensorboard_bridge_v1 \
  --host 127.0.0.1 \
  --port 6006
```

远程服务器建议通过 SSH 端口转发访问，不要把服务绑定到公网地址。

## 4. LR 对比

在 TensorBoard Scalars 中选择以下 OPD 与 Matched-SFT run：

```text
opd_lr1e6_horizon5__O2
opd_lr1e5__O2
opd_lr2e5__O2
opd_lr3e5__O2
kc_sft_matched_prompt_lr2e5__KC_SFT_MATCHED_PROMPT_LR2E5
```

重点查看：

- `opd/kl_recent100`
- `optim/lr`
- `optim/grad_norm_preclip`
- `optim/grad_clip_scale`
- `train/rollout_length_recent200`
- `sft/ce_per_response_token`
- `train/response_tokens`

global step 是日志中实际 `optimizer update`，不是 epoch 编号。历史文本日志通常每 10 个 update 才打印一次，因此桥接器只写入确实存在的点，不插值、不补造每步指标。最终评测 step 还必须由 `resume_meta.json`、PASS `progress.json` 或带 SHA 的连续 update ledger 证明，不能只使用计划步数。OPD Horizon-5 的 P1–P5 知识曲线读取 `learning_curve/heldout/O2`，step 由冻结 `updates_per_epoch × pass_number` 得出，并同时验证 snapshot 日志与对应 checkpoint；`train_matched` 曲线不会冒充 heldout 指标。该历史产物没有逐 pass BLEU/chrF，因此不生成相应质量曲线。

### 已完成 MatchedPrompt 的历史读取合同

`KC_SFT_MATCHED_PROMPT_LR2E5` 直接读取冻结的 `train/update_metrics.jsonl`，逐 optimizer update 写入：

- `sft/ce_per_response_token`：`group_token_mean_ce`；
- `optim/grad_norm_preclip`：`grad_norm_before_clip`；
- `optim/lr`：`lr_after_step`；
- `train/response_tokens`：`group_response_tokens`。

桥接器不会从旧文本 `loss50` 重建这些值。正式转换前还会验证：update ledger SHA 和连续 1–1032 step、PASS progress/latest checkpoint/meta、训练合同 SHA、final/durable checkpoint 文件 SHA、BOUND `eval_contract.json`、eval manifest 中相同的合同 payload/SHA、冻结评测输入 SHA、两个 summary 的合同 SHA，以及 general prediction 文件 SHA。知识与 BLEU/chrF 只从冻结 summary 读取，不重新评分。

## 5. 未来 Clean-SFT 原生记录

默认关闭。显式启用：

```bash
python scripts/targeted/clean_sft_parity_v1.py train \
  --arm KC_SFT_MATCHED_PROMPT_LR2E5 \
  --run-dir /absolute/path/to/new_run \
  --tensorboard \
  --tensorboard-logdir /absolute/path/to/new_run/tensorboard \
  --tensorboard-write-every 1
```

原生 SFT 记录：

- `sft/ce_per_response_token`
- `optim/lr`
- `optim/grad_norm_preclip`
- `optim/grad_clip_scale`
- `train/response_tokens`
- `train/step_seconds`

记录发生在实际 optimizer/scheduler update 后、`group_tokens` 重置前。不开启 `--tensorboard` 时不导入 TensorBoard，也不创建 logdir。logdir 合同绑定 arm 与训练合同 SHA；若同一目录已属于另一实验，训练会在写 event 前阻断。

恢复训练时，以 durable checkpoint/update ledger 为边界，`SummaryWriter(purge_step=durable_step+1)` 隐藏中断后未提交的 event，再从下一 durable update 继续，避免可见曲线重复。

冻结 OPD runner 未作任何修改。`NativeTensorBoard.log_opd_update()` 已提供 OPD 原生接口；接入时必须复制成新的版本化 OPD runner，并在新实验 manifest 中记录 runner SHA，不得修改历史 runner。

## 6. 知识与翻译质量曲线

桥接器只读取已有评测，不触发推理：

- `knowledge/chemistry_strict`
- `knowledge/chemistry_spacing_normalized`（源文件明确提供时）
- `knowledge/chemistry_ci`
- `knowledge/chemistry_non_ci`
- `knowledge/chemistry_uc`
- `knowledge/chemistry_uw`
- `knowledge/idiom_strict`（源文件明确提供时）
- `quality/bleu`
- `quality/chrf`

C.I./非 C.I. 读取冻结 `slice_summary.json`，或读取新 Clean-SFT `chemistry_summary.json` 中已冻结的 `slices.ci/non_ci`；不重新分组。A1 P1–P5 的 step 读取 `P*_resume_meta.json`，并绑定对应 checkpoint 文件身份。缺失 step、checkpoint 身份或指标时不生成曲线。

需要 GAIN/LOSS 时，可传入冻结 eval index：

```bash
python scripts/targeted/tensorboard_bridge_v1.py convert \
  --preset chemistry \
  --eval-index /absolute/path/to/frozen_eval_index.json \
  --output /absolute/path/to/output
```

每条 index 记录必须包含相同 `eval_id`、实际 optimizer step、可读取的 checkpoint 路径及其校验 identity、评测文件 SHA、评测数据 SHA、prompt SHA、解码配置 SHA 和评测版本；评测数据文件本身必须列入 `source_files`，使该 SHA 可现场校验；GAIN/LOSS 还必须包含对照模型 identity。身份不完整时拒绝写入。历史 `slice_summary.json` 虽可能含 gain/loss 数字，但未明确绑定对照 checkpoint identity，因此不会自动转成 GAIN/LOSS 曲线。

内置自动适配器优先覆盖已有 Chemistry 产物。Idiom 运行可先用 `--run` 接入训练曲线；`knowledge/idiom_strict` 仅通过明确提供该字段及身份信息的冻结 eval index 写入，现阶段不会从 judge 文件推断。

## 7. 科学可比性

可以直接比较：

- 同一 OPD 实现和指标定义下，不同 LR 的 `opd/kl_recent100`、LR、梯度范数和 rollout 窗口；
- 同一冻结评测集、prompt、解码与评分版本下，不同 checkpoint 的知识/质量指标；
- 使用相同 token-normalized 合同的新 SFT `sft/ce_per_response_token`。

不能直接比较：

- OPD KL 与 SFT CE；
- 历史 SFT `loss50` 与新 SFT 精确 token-normalized CE；
- `kl100` 与完整数据集精确 token-level KL；
- `rollout_len200` 与单个 update 的 rollout mean；
- 不同实验的 epoch 编号；
- eval_id、checkpoint、prompt、解码或评分版本不同的 GAIN/LOSS。

因此历史 SFT 使用独立名称 `sft/ce_microbatch_mean_recent50`，历史 OPD rollout 使用 `train/rollout_length_recent200`。

## 8. P1 机制诊断接口

event manifest 预留以下标准字段，但现有文件不足时统一为 `NOT_AVAILABLE`：

- Student-prefix 关键术语概率；
- Teacher-prefix 关键术语概率；
- 术语/非术语位置 KL；
- Teacher-prefix/Student-prefix 有效术语事件覆盖率；
- C.I./非 C.I. 分组机制统计。

不得从最终 accuracy 或总 KL 反推这些指标。

## 9. 用户执行的测试

本次实现未代替用户运行测试。建议执行：

```bash
source /workspace/mtpatcher/project_env.sh

python -m py_compile \
  scripts/targeted/_targeted_tensorboard_v1.py \
  scripts/targeted/tensorboard_bridge_v1.py \
  scripts/targeted/clean_sft_parity_runtime_v1.py \
  scripts/targeted/test_targeted_tensorboard_v1.py

python scripts/targeted/test_targeted_tensorboard_v1.py
python scripts/targeted/test_clean_sft_parity_v1.py
python scripts/targeted/test_clean_sft_parity_contracts_v1.py
python scripts/targeted/test_clean_sft_parity_v2_contracts.py
```

若 `convert` 或 `validate` 报 TensorBoard 缺失，应切换到已包含 TensorBoard 的项目环境；本任务未安装任何软件包。

