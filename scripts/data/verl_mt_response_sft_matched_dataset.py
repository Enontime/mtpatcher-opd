#!/usr/bin/env python3
from __future__ import annotations

from scripts.data.verl_mt_response_sft_dataset import (
    MTResponseOnlySFTDataset,
)


class MatchedMTResponseOnlySFTDataset(
    MTResponseOnlySFTDataset
):
    """
    MTResponseOnlySFTDataset plus matched20k provenance.

    Training semantics are inherited unchanged.
    The extra fields exist only so the trainer can verify
    which frozen source schedule was actually consumed.
    """

    REQUIRED_METADATA = (
        "source_id",
        "schedule_position",
        "matched_global_step",
        "matched_pass",
        "position_in_global_batch",
    )

    def __getitem__(self, item: int):
        out = super().__getitem__(item)

        row = self.dataframe.iloc[item].to_dict()

        for key in self.REQUIRED_METADATA:
            if key not in row:
                raise KeyError(
                    f"row {item}: missing matched metadata {key}"
                )

            value = row[key]

            if value is None:
                raise ValueError(
                    f"row {item}: {key} is None"
                )

            out[key] = int(value)

        return out
