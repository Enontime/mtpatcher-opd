# MT-PATCHER 3090 → Ascend Migration

## Source code

Expected old root:

/home/zfs02/shenyz/chentc/code/MT-Patcher-Reproduction-2080Ti

Priority:

1. scripts/qwen35
2. training scripts
3. evaluation scripts
4. reward implementations
5. configs
6. required vendor modifications only

## Frozen datasets

Required:

- pilot_v2_qwen3_06b
- pilot_v2_grpo/train1024_seed20260821.jsonl
- full 6565 training data
- wmt24_zh_en998.jsonl
- FLORES test
- Challenge test

## Frozen experimental facts

GRPO pilot1024 SHA256:

e93b84a995d903cc6cddfd7a87088a9e1b164a7f532e6b920123650a33f1a49c

Manifest SHA256:

b111b3c4475322bcfe6ef7033fb7a0e5b4aba7e352afc8fb5d00921d61eb124b

## Do not migrate as runtime dependencies

- old CUDA PyTorch
- old conda environment
- CUDA-specific binaries
- old Hugging Face caches
- large obsolete checkpoints
