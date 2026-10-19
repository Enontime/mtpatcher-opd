# OPD 学习曲线与评测速度诊断（2026-10-16）

## 目的及边界

这不是新的 OPD trainer，也不应取代 verl。它只是两个**只读诊断工具**，用于在下一次正式多 checkpoint 训练前，确认当前资产到底保存了什么、旧评测函数慢在哪里。脚本本身不训练、不改模型、不修改已有评测、不碰 Git，也不改变 `paper_view`。

根据仓库 `infra/verl-ascend-parity`（仅 GitHub 分支版本）审查：`scripts/targeted/targeted_wa_opd_overnight_v3.py` 的 `generate_one_translation` 是逐句 `generate`，BLEU/targeted evaluator 每条预测 `fsync`；原生 verl 路径参见 `recipes/targeted/run_offline_prefix_support_chemistry_seed1_native_verl_v1.sh`、`configs/targeted/offline_prefix_support_chemistry_seed1_native_verl_v1.yaml`。**服务器当前代码与 GitHub 分支可能不同**，两份脚本均记录原代码/模型身份，不将不同版本实验混为一谈。

## 传到 Ascend

Windows PowerShell 中（下载 ZIP 后按真实下载路径调整）：

```powershell
scp "$HOME\Downloads\opd_training_curve_toolkit_20261016.zip" mtpatcher-ascend:/workspace/mtpatcher/
ssh mtpatcher-ascend
```

Ascend shell：

```bash
source /workspace/mtpatcher/project_env.sh
cd /workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend
mkdir -p scripts/targeted/opd_curve_toolkit_20261016
/workspace/mtpatcher/envs/mtpatcher-npu-py311/bin/python -m zipfile -e \
  /workspace/mtpatcher/opd_training_curve_toolkit_20261016.zip \
  scripts/targeted/opd_curve_toolkit_20261016
```

## Step 1: CPU 只读资产审计（应先运行）

```bash
PY=/workspace/mtpatcher/envs/mtpatcher-npu-py311/bin/python
$PY scripts/targeted/opd_curve_toolkit_20261016/pipeline_audit.py
```

生成：

```
/workspace/mtpatcher/runs/targeted/opd_curve_audit_20261016/audit.json
/workspace/mtpatcher/runs/targeted/opd_curve_audit_20261016/audit.md
```

报告逐实验列出 checkpoint 及 optimizer state 可见性、manifest、训练 update JSONL（若有）、TensorBoard 标量 tag 与真实 step 区间。区分历史桥接事件与原生 event 文件。

**不能得出**：`resume_hf` 是中途 checkpoint；只有训练 loss 曲线就等于生成正确率曲线；旧 TensorBoard bridge 是实际训练产生的原生 event。

如果报告显示 `TENSORBOARD_PACKAGE_MISSING`，先尝试原生 verl Python 执行相同命令；无需安装新库：

```bash
/workspace/mtpatcher/envs/verl-v0.9.0-a3/bin/python \
  scripts/targeted/opd_curve_toolkit_20261016/pipeline_audit.py
```

## Step 2：NPU 性能诊断（只测旧方法，不替换正式评测）

`eval_speed_probe.py` 从原始 Chemistry2000 里固定、均匀选择 16 条，**直接导入原评测函数**作为 serial baseline；第二阶段尝试 `batch_size=4`，在同一冻结 checkpoint 上逐样本比较字符串。

```bash
PY=/workspace/mtpatcher/envs/mtpatcher-npu-py311/bin/python
$PY scripts/targeted/opd_curve_toolkit_20261016/eval_speed_probe.py \
  --n 16 --batch-size 4
```

报告：

```
/workspace/mtpatcher/runs/targeted/opd_eval_speed_probe_20261016/probe_n16_batch4.json
```

重点看 `baseline_serial_seconds`、`batch_seconds`、`speedup`、`serial_output_tokens`、`mismatches`。

> 该测速不包括评测产物逐条 `fsync` 开销，因此如果批处理仍慢，下一步需要单独对 I/O 做剖析。 `max_new_tokens=512` 保持与已检出的旧评测函数一致，不为速度擅自修改。异 batch 造成的 BF16 差异都视为安全性问题，必须先解决评测语义，不能未经验证替换正式 scorer。

必要时先单独跑 `--n 4 --batch-size 2`，确认 Ascend NPU 不会 OOM。

## 下一步决策（收到服务器两个 JSON 后才能决定）

1. OPD 原生日志如果已含 loss、Student mass、LR、grad norm、rollout token 数，直接用已有 verl/TensorBoard；缺什么再补，不重写已有 trainer。
2. 若 batch4 有加速且输出完全相同，再扩大到 32–64 条做一致性验证，正式评测改动需明确新版本、新 hash、新对照；旧正式结果不可无声明替换。
3. 正式 OPD 多 checkpoint 应优先使用现有 verl 路径。**但当前 Chemistry LR 扫描来自自定义 runner**；需要前向 KL/梯度/rollout/mask 等价性审计后才可将新 verl 实验称作同规格比较。
4. 高频曲线：训练中真实在线 rollout strict hit、KL（明确 top-32 定义）、Student mass、LR、grad norm、有效 token、长度、teacher/student 计算耗时。所有值从真实训练路径写入独立原生 TensorBoard，不伪造。
5. 中频：固定训练样本和未见诊断子集的自然生成；正式 BLEU 仅在少量关键 checkpoint 做。训练集指标与 Student temperature=1 rollout 的命中率必须使用不同 tag。

当前工具只是低风险诊断，不将其纳入 `paper_view`，不产生论文科学结果。
