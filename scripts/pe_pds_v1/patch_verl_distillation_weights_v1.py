#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = (
    "MTP_EAEC_DISTILLATION_"
    "TOKEN_WEIGHTS_V1"
)


BLOCK = r'''
    # MTP_EAEC_DISTILLATION_TOKEN_WEIGHTS_V1
    # Optional response-token weights supplied by EAEC.
    #
    # response_mask remains unchanged, so token-mean keeps the
    # original full response-token denominator.
    token_weights = data.get(
        "distillation_token_weights",
        None,
    )

    if token_weights is not None:
        if token_weights.is_nested:
            token_weights = (
                token_weights
                .to_padded_tensor(0.0)
            )

        if response_mask.is_nested:
            response_mask_for_weights = (
                response_mask
                .bool()
                .to_padded_tensor(False)
            )
        else:
            response_mask_for_weights = (
                response_mask.bool()
            )

        assert (
            token_weights.shape
            == distillation_losses.shape
            == response_mask_for_weights.shape
        )

        assert not token_weights.requires_grad
        assert torch.isfinite(token_weights).all()
        assert (token_weights >= 0).all()
        assert (
            token_weights[
                ~response_mask_for_weights
            ] == 0
        ).all()

        token_weights = (
            token_weights
            .detach()
            .to(
                device=
                    distillation_losses.device,
                dtype=
                    distillation_losses.dtype,
            )
        )

        valid_weights = token_weights[
            response_mask_for_weights
        ]

        if valid_weights.numel() == 0:
            raise RuntimeError(
                "EAEC got empty response weights"
            )

        distillation_metrics[
            "distillation/token_weight_mean"
        ] = (
            valid_weights
            .float()
            .mean()
            .item()
        )

        distillation_metrics[
            "distillation/token_weight_min"
        ] = (
            valid_weights
            .float()
            .min()
            .item()
        )

        distillation_metrics[
            "distillation/token_weight_max"
        ] = (
            valid_weights
            .float()
            .max()
            .item()
        )

        distillation_losses = (
            distillation_losses
            * token_weights
        )
'''


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "losses_py"
    )

    args = ap.parse_args()

    path = Path(
        args.losses_py
    )

    text = path.read_text(
        encoding="utf-8"
    )

    if MARKER in text:
        print(
            "EAEC_VERL_PATCH_ALREADY_PRESENT=PASS"
        )
        return

    needle = (
        '    response_mask = data["response_mask"]\n'
        "    loss_agg_mode = config.loss_agg_mode\n"
    )

    count = text.count(
        needle
    )

    if count != 1:
        raise RuntimeError(
            "expected exactly one distillation "
            f"aggregation insertion point, got {count}"
        )

    new_text = text.replace(
        needle,
        needle + BLOCK + "\n",
        1,
    )

    if MARKER not in new_text:
        raise RuntimeError(
            "patch marker missing after replacement"
        )

    path.write_text(
        new_text,
        encoding="utf-8",
    )

    print(
        "EAEC_VERL_PATCH_APPLIED=PASS"
    )


if __name__ == "__main__":
    main()
