#!/usr/bin/env python3
"""Executable Hydra shim for the synchronous native-Verl RayPPOTrainer path.

main_ppo.py owns the Hydra CLI/config composition.
main_ppo_v0.py owns the synchronous TaskRunner -> RayPPOTrainer implementation.
We replace only main_ppo.TaskRunnerV1 before invoking its decorated main().
"""

from __future__ import annotations

import sys

VERL_ROOT = "/workspace/mtpatcher/repo/verl-v0.9.0"
PROJECT_ROOT = "/workspace/mtpatcher/repo/MT-Patcher-Reproduction-Ascend"

for path in (PROJECT_ROOT, VERL_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from verl.trainer import main_ppo as main_ppo  # noqa: E402
from verl.trainer.main_ppo_v0 import TaskRunner as SyncTaskRunner  # noqa: E402


def main() -> None:
    # main_ppo.main() is Hydra-decorated and resolves the normal trainer config.
    # Its body calls run_ppo(config, task_runner_class=TaskRunnerV1), so swapping
    # this one module global routes execution into the synchronous native trainer.
    main_ppo.TaskRunnerV1 = SyncTaskRunner
    main_ppo.main()


if __name__ == "__main__":
    main()
