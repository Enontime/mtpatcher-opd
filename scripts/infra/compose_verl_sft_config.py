#!/usr/bin/env python3
"""Compose Verl SFT Hydra config for machine-readable contract tests.

This is config inspection only.
Actual training still goes through Verl's native Hydra CLI entrypoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--output", required=True, type=Path)

    # Everything after "--" is passed verbatim to Hydra compose().
    args, overrides = parser.parse_known_args()

    if overrides and overrides[0] == "--":
        overrides = overrides[1:]

    config_dir = str(Path(args.config_dir).resolve())

    with initialize_config_dir(
        version_base=None,
        config_dir=config_dir,
        job_name="mtpatcher_sft_config_audit",
    ):
        cfg = compose(
            config_name="sft_trainer_engine",
            overrides=overrides,
        )

    # Resolve interpolations here, while cfg is still a real DictConfig.
    OmegaConf.resolve(cfg)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OmegaConf.save(
        config=cfg,
        f=args.output,
        resolve=True,
    )

    # Immediate round-trip validation.
    loaded = OmegaConf.load(args.output)

    if OmegaConf.to_container(
        cfg,
        resolve=True,
    ) != OmegaConf.to_container(
        loaded,
        resolve=True,
    ):
        raise AssertionError(
            "Saved resolved config failed round-trip equality"
        )

    print("HYDRA_COMPOSE_CONFIG_PASS")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
