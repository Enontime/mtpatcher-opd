# 正式训练与评测合同

更新日期：2026-09-15

## 1. SFT / OPD matched training

正式 SFT 与 OPD comparison 必须保证：

- 相同训练 source pool；
- global batch size = 16；
- 每个 optimizer step 使用完全相同的 16 条 source；
- source 顺序完全一致；
- optimizer-step budget 一致；
- initial Student checkpoint 一致；
- evaluation prompt / tokenizer / generation contract 一致。

不得再依赖两个 trainer 各自的 sampler 行为来假定顺序一致。

正式 comparison 应冻结 source-order manifest，并能够由 step id 恢复该 step 使用的 source ids。

历史 Full SeqKD / Full OPD：

- dataset identity：matched；
- global batch size：matched = 16；
- exposure：matched；
- step-wise source order：not matched。

因此历史结果不得宣称 strict step-wise pairing。

## 2. 每 100 optimizer steps 评测

每 100 optimizer steps 固定执行两类 evaluation。

### Train probe

用于观察训练数据吸收情况。

TensorBoard namespace：

- train_probe/bleu
- train_probe/chrf
- train_probe/macro_bleu
- train_probe/macro_chrf

Targeted 实验可增加：

- train_probe/chemistry_accuracy
- train_probe/idiom_score

### Benchmark validation

在固定 benchmark evaluation sets 上执行：

- WMT24
- FLORES
- Challenge

TensorBoard namespace：

- benchmark_val/wmt24_bleu
- benchmark_val/wmt24_chrf
- benchmark_val/flores_bleu
- benchmark_val/flores_chrf
- benchmark_val/challenge_bleu
- benchmark_val/challenge_chrf
- benchmark_val/macro_bleu
- benchmark_val/macro_chrf

这些 benchmark validation 结果允许用于判断训练是否收敛。

如果它们参与 checkpoint selection 或 early stopping，论文和 provenance 中必须明确说明。

## 3. 收敛与停止

默认每 100 steps 检查一次 benchmark validation。

推荐同时记录：

- best metric；
- best step；
- delta from best；
- consecutive no-improvement evaluations。

禁止只根据单个 checkpoint 的轻微波动立即停止。

early-stopping 参数必须在实验启动前冻结，不得看结果后修改。

## 4. Checkpoint

正式训练至少应保存：

- periodic checkpoint；
- best checkpoint；
- final checkpoint。

benchmark validation 对应的 checkpoint 必须可追溯。

训练结束后必须保存：

- best step；
- final step；
- stopping reason；
- benchmark validation history。

## 5. Case-level evaluation

每个正式 checkpoint 的评测必须保存逐样本输出，至少包括：

- sample id；
- source；
- reference；
- prediction；
- dataset；
- checkpoint step；
- model/checkpoint identity；
- generation config；
- tokenizer identity。

Targeted Chemistry 额外保存：

- source term/entity；
- canonical target term；
- accepted aliases；
- hit/miss。

Targeted Idiom 额外保存：

- source idiom；
- source sentence；
- reference meaning；
- prediction；
- judge raw output；
- parsed score。

任何 aggregate metric 都应能够下钻到具体 case。

## 6. Tokenizer / scorer

模型 tokenizer 用于：

- training；
- rollout；
- generation；
- checkpoint evaluation。

必须冻结：

- tokenizer path；
- tokenizer class；
- vocab hash；
- chat-template hash；
- BOS/EOS/PAD；
- enable_thinking。

OPD Teacher / Student token-level KL 必须验证 tokenizer/vocab compatibility。

BLEU / chrF 由冻结的 SacreBLEU scorer 计算，不使用 Qwen tokenizer 手工切词。

必须保存：

- sacrebleu version；
- BLEU scorer configuration/signature；
- chrF scorer configuration/signature。

## 7. TensorBoard

Raw TensorBoard 永久保留。

默认论文 TensorBoard 只展示科研有意义的指标。

推荐 namespace：

- train/
- distill/
- train_probe/
- benchmark_val/
- system/

Verl 通用 bookkeeping、全零 critic metrics、debug/perf 细节不进入默认 paper-facing TensorBoard。

Derived clean TensorBoard 必须保存 raw-event SHA256 和 tag mapping。
