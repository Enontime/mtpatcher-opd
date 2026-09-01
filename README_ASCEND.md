# MT-PATCHER Reproduction — Ascend

Managed workspace:

    /workspace/mtpatcher

Repository:

    /workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend

External resources:

    /workspace/mtpatcher/data
    /workspace/mtpatcher/models
    /workspace/mtpatcher/runs
    /workspace/mtpatcher/logs
    /workspace/mtpatcher/snapshots

Runtime environment:

    /workspace/mtpatcher/envs/mtpatcher-npu-py311

Entry point:

    source /workspace/mtpatcher/project_env.sh

Validated hardware/runtime:

- 16 x Ascend910_9362
- torch 2.9.0
- torch_npu 2.9.0rc1
- HCCL 2/4/8/16 NPU PASS
- Qwen3-0.6B BF16 inference PASS

Project rule:

Large datasets, pretrained models, generated checkpoints,
experiment outputs, and caches must remain outside Git.
