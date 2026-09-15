#!/usr/bin/env python3

# Importing this module registers "matched20k_sync".
from scripts.matched20k_v2.opd_trainer import (  # noqa: F401
    Matched20kPPOTrainerSync,
)

from verl.trainer.main_ppo import main


if __name__ == "__main__":
    main()
