#!/usr/bin/env python3

import os

import torch
import torch.distributed as dist
import torch_npu


local_rank = int(
    os.environ["LOCAL_RANK"]
)

rank = int(
    os.environ["RANK"]
)

world = int(
    os.environ["WORLD_SIZE"]
)


torch.npu.set_device(
    local_rank
)

device = torch.device(
    f"npu:{local_rank}"
)


dist.init_process_group(
    backend="hccl"
)


x = torch.tensor(
    [
        float(
            rank + 1
        )
    ],
    dtype=torch.float32,
    device=device,
)


dist.all_reduce(
    x,
    op=dist.ReduceOp.SUM,
)


expected = (
    world
    * (
        world + 1
    )
    / 2
)


actual = float(
    x.item()
)


if abs(
    actual - expected
) > 1.0e-4:

    raise RuntimeError(
        f"HCCL float32 sum mismatch: "
        f"actual={actual} expected={expected}"
    )


if rank == 0:

    print(
        "HCCL_FLOAT32_ALLREDUCE_AUDIT",
        {
            "world_size":
                world,

            "dtype":
                str(
                    x.dtype
                ),

            "actual_sum":
                actual,

            "expected_sum":
                expected,
        },
        flush=True,
    )

    print(
        "HCCL_FLOAT32_ALLREDUCE_PASS",
        flush=True,
    )


dist.barrier()

dist.destroy_process_group()
