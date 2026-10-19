#!/usr/bin/env python3
"""CLI entry point for Chemistry Clean-SFT parity v1.

Implementation is split into a data/contract core and a training/eval runtime
so CPU-only audits never import Torch/NPU dependencies.
"""
from clean_sft_parity_runtime_v1 import main


if __name__ == "__main__":
    main()
