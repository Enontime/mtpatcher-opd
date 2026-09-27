#!/usr/bin/env python3

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from verl.experimental.agent_loop import (
    AgentLoopManager,
)
from verl.utils.ray_utils import (
    auto_await,
)

from scripts.pe_pds_v1.eaec_core_v1 import (
    build_sample_weights,
    load_patchbank,
)

from scripts.pe_pds_v1.eaec_alignment_v1 import (
    classify_and_try_alignment,
    decode_response_text,
)


def _python_scalar(x: Any):
    if isinstance(
        x,
        np.ndarray,
    ) and x.shape == ():
        return x.item()

    if hasattr(
        x,
        "item",
    ) and not isinstance(
        x,
        (
            str,
            bytes,
            dict,
            list,
        ),
    ):
        try:
            return x.item()
        except Exception:
            pass

    return x


def _extract_source_ids(
    prompts,
) -> list[int]:
    nt = prompts.non_tensor_batch

    for key in (
        "source_id",
        "index",
    ):
        if key in nt:
            vals = nt[key]

            return [
                int(
                    _python_scalar(v)
                )
                for v in vals
            ]

    if "extra_info" in nt:
        out = []

        for raw in nt[
            "extra_info"
        ]:
            x = _python_scalar(
                raw
            )

            if not isinstance(
                x,
                dict,
            ):
                raise RuntimeError(
                    "extra_info is not dict"
                )

            if "index" not in x:
                raise RuntimeError(
                    "extra_info missing index"
                )

            out.append(
                int(x["index"])
            )

        return out

    raise RuntimeError(
        "EAEC requires source_id/index "
        "provenance in prompt batch"
    )


def append_jsonl(
    path: Path,
    row: dict,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )

        f.flush()


class LocalizationBenchmarkExportManagerV1(
    AgentLoopManager
):
    """
    Fresh Student rollout + exact-active MT-PATCHER edit-core weighting.

    This does NOT replace the rollout and does NOT construct fixed targets.

    It calls native Verl fresh rollout first, then attaches:
        distillation_token_weights

    Teacher scoring and forward-KL remain native Verl behavior.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        path = os.environ.get(
            "EAEC_PATCHBANK_JSONL"
        )

        if not path:
            raise RuntimeError(
                "EAEC_PATCHBANK_JSONL "
                "is required"
            )

        self.patchbank = (
            load_patchbank(path)
        )

        self.core_ratio = float(
            os.environ.get(
                "EAEC_CORE_RATIO",
                "2.0",
            )
        )

        if self.core_ratio < 1.0:
            raise RuntimeError(
                "EAEC_CORE_RATIO "
                "must be >= 1"
            )

        # AgentLoopManager itself does not expose the tokenizer
        # used inside AgentLoopWorker.  EAEC needs deterministic
        # decode -> exact-string-match -> offset-mapping after the
        # native fresh rollout has completed, so keep a local
        # CPU-side tokenizer loaded from the exact Student model.
        #
        # This tokenizer never generates tokens and never touches
        # Teacher/Student logits.  It is localization-only.
        model_path = str(
            self.config
            .actor_rollout_ref
            .model
            .path
        )

        self.eaec_tokenizer = (
            AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True,
                use_fast=True,
            )
        )

        if not getattr(
            self.eaec_tokenizer,
            "is_fast",
            False,
        ):
            raise RuntimeError(
                "EAEC requires a fast tokenizer "
                "for exact offset mapping"
            )

        if (
            self.eaec_tokenizer.eos_token_id
            is None
        ):
            raise RuntimeError(
                "EAEC tokenizer missing EOS"
            )

        print(
            "EAEC_TOKENIZER_INIT_PASS "
            f"class={type(self.eaec_tokenizer).__name__} "
            f"model={model_path} "
            f"eos={self.eaec_tokenizer.eos_token_id} "
            f"pad={self.eaec_tokenizer.pad_token_id}",
            flush=True,
        )

        rollout_n = int(
            self.config
            .actor_rollout_ref
            .rollout
            .n
        )

        if rollout_n != 1:
            raise RuntimeError(
                "EAEC v1 requires "
                "rollout.n=1 for exact "
                "source_id alignment"
            )

        default_dir = Path(
            str(
                self.config
                .trainer
                .default_local_dir
            )
        )

        self.audit_path = Path(
            os.environ.get(
                "EAEC_AUDIT_JSONL",
                str(
                    default_dir
                    .parent
                    / "eaec_runtime_audit.jsonl"
                ),
            )
        )

        self.eaec_calls = 0

        self.alignment_examples_path = (
            self.audit_path.parent
            / "eaec_alignment_examples.jsonl"
        )

        self.alignment_examples_written = 0

        # Pure benchmark export.
        # Gold labels are NOT written here.
        self.benchmark_rollouts_path = (
            self.audit_path.parent
            / "localization_benchmark_rollouts_v1.jsonl"
        )

        self.benchmark_rollouts_written = 0

        print(
            "EAEC_MANAGER_INIT_PASS "
            f"patchbank_sources={len(self.patchbank)} "
            f"core_ratio={self.core_ratio}",
            flush=True,
        )

    @auto_await
    async def generate_sequences(
        self,
        prompts,
    ):
        validate = bool(
            prompts.meta_info.get(
                "validate",
                False,
            )
        )

        source_ids = (
            _extract_source_ids(
                prompts
            )
        )

        # Call the real Verl fresh Student rollout.
        maybe = (
            AgentLoopManager
            .generate_sequences(
                self,
                prompts,
            )
        )

        if inspect.isawaitable(
            maybe
        ):
            out = await maybe
        else:
            out = maybe

        # Validation does not enter OPD training.
        if validate:
            return out

        responses = out.batch[
            "responses"
        ]

        response_mask = out.batch[
            "response_mask"
        ]

        if (
            responses.ndim != 2
            or response_mask.ndim != 2
        ):
            raise RuntimeError(
                "unexpected rollout tensor rank"
            )

        if (
            responses.shape
            != response_mask.shape
        ):
            raise RuntimeError(
                "responses/response_mask "
                "shape mismatch"
            )

        bsz = responses.shape[0]

        if len(
            source_ids
        ) != bsz:
            raise RuntimeError(
                "source_id / rollout batch "
                f"mismatch {len(source_ids)} "
                f"!= {bsz}"
            )

        weights = torch.zeros_like(
            response_mask,
            dtype=torch.float32,
        )

        aggregate = {
            "samples": bsz,
            "samples_with_patches": 0,
            "candidate_patches": 0,
            "missing_patchbank_sources":
                0,
            "active_patches": 0,
            "resolved_patches": 0,
            "ambiguous_patches": 0,
            "unlocated_patches": 0,
            "core_tokens": 0,
            "valid_response_tokens": 0,
            "tokenization_fallback":
                0,
            "projection_empty": 0,

            # Diagnostic-only trajectory alignment counters.
            "diag_string_active": 0,
            "diag_string_resolved": 0,
            "diag_string_ambiguous": 0,
            "diag_string_unlocated": 0,
            "diag_strict_recovered": 0,

            "diag_reject_historical_anchor_not_unique": 0,
            "diag_reject_empty_word_sequence": 0,
            "diag_reject_historical_word_projection_empty": 0,
            "diag_reject_no_left_anchor": 0,
            "diag_reject_no_right_anchor": 0,
            "diag_reject_left_anchor_not_unique": 0,
            "diag_reject_right_anchor_not_unique": 0,
            "diag_reject_empty_or_reversed_current_region": 0,
            "diag_reject_current_region_too_wide": 0,
        }

        for i, sid in enumerate(
            source_ids
        ):
            row = self.patchbank.get(
                int(sid)
            )

            if row is None:
                aggregate[
                    "missing_patchbank_sources"
                ] += 1

                patches = []
            else:
                patches = row[
                    "patches"
                ]

            aggregate[
                "candidate_patches"
            ] += len(patches)

            if patches:
                aggregate[
                    "samples_with_patches"
                ] += 1

            # -------------------------------------------------
            # DIAGNOSTIC ONLY:
            # classify every patch from current rollout text and
            # test conservative historical->current alignment.
            #
            # This does NOT alter distillation_token_weights.
            # -------------------------------------------------

            current_text = decode_response_text(
                self.eaec_tokenizer,
                responses[i],
                response_mask[i],
            )

            append_jsonl(
                self.benchmark_rollouts_path,
                {
                    "source_id": int(source_ids[i]),
                    "current_translation": current_text,
                    "rollout_call": int(self.eaec_calls),
                    "position_in_batch": int(i),
                },
            )

            self.benchmark_rollouts_written += 1

            base_text = (
                row["base_student_translation"]
                if row is not None
                else ""
            )

            for patch in patches:
                d = classify_and_try_alignment(
                    base_text=base_text,
                    current_text=current_text,
                    patch=patch,
                )

                state = d["state"]

                if state == "ACTIVE":
                    aggregate[
                        "diag_string_active"
                    ] += 1

                elif state == "RESOLVED":
                    aggregate[
                        "diag_string_resolved"
                    ] += 1

                elif state == "AMBIGUOUS":
                    aggregate[
                        "diag_string_ambiguous"
                    ] += 1

                elif state == "UNLOCATED":
                    aggregate[
                        "diag_string_unlocated"
                    ] += 1

                    if d.get(
                        "recovered",
                        False,
                    ):
                        aggregate[
                            "diag_strict_recovered"
                        ] += 1

                        if (
                            self.alignment_examples_written
                            < 100
                        ):
                            append_jsonl(
                                self.alignment_examples_path,
                                {
                                    "source_id":
                                        int(sid),
                                    "old_span":
                                        patch[
                                            "old_span"
                                        ],
                                    "correction":
                                        patch[
                                            "correction"
                                        ],
                                    "error_type":
                                        patch.get(
                                            "error_type",
                                            "",
                                        ),
                                    "base_student_translation":
                                        base_text,
                                    "current_translation":
                                        current_text,
                                    "projected_current_region":
                                        d.get(
                                            "current_region"
                                        ),
                                    "left_anchor":
                                        d.get(
                                            "left_anchor"
                                        ),
                                    "right_anchor":
                                        d.get(
                                            "right_anchor"
                                        ),
                                },
                            )

                            self.alignment_examples_written += 1

                    else:
                        reason = d.get(
                            "reason",
                            "unknown",
                        )

                        key = (
                            "diag_reject_"
                            + reason
                        )

                        if key in aggregate:
                            aggregate[key] += 1

            w, stats = (
                build_sample_weights(
                    tokenizer=
                        self.eaec_tokenizer,
                    response_ids=
                        responses[i],
                    response_mask=
                        response_mask[i],
                    patches=
                        patches,
                    core_ratio=
                        self.core_ratio,
                )
            )

            weights[i] = w.to(
                device=weights.device
            )

            for key in (
                "active_patches",
                "resolved_patches",
                "ambiguous_patches",
                "unlocated_patches",
                "core_tokens",
                "valid_response_tokens",
                "tokenization_fallback",
                "projection_empty",
            ):
                aggregate[key] += int(
                    stats[key]
                )

        valid = response_mask.bool()

        if not torch.isfinite(
            weights
        ).all():
            raise RuntimeError(
                "non-finite EAEC weights"
            )

        if bool(
            (
                weights[
                    ~valid
                ] != 0
            ).any()
        ):
            raise RuntimeError(
                "EAEC padding weights "
                "must be zero"
            )

        if bool(valid.any()):
            valid_mean = (
                weights[valid]
                .float()
                .mean()
            )

            if not torch.allclose(
                valid_mean,
                torch.tensor(
                    1.0,
                    device=
                        valid_mean.device,
                ),
                atol=1e-6,
                rtol=0,
            ):
                # Every sequence is independently normalized.
                # Therefore the batch token mean must also be one.
                raise RuntimeError(
                    "EAEC batch mean "
                    f"weight != 1: "
                    f"{valid_mean.item()}"
                )

            weight_min = float(
                weights[valid]
                .min()
                .item()
            )

            weight_max = float(
                weights[valid]
                .max()
                .item()
            )
        else:
            valid_mean = torch.tensor(
                0.0
            )

            weight_min = 0.0
            weight_max = 0.0

        weights = weights.detach()
        weights.requires_grad_(False)

        out.batch[
            "distillation_token_weights"
        ] = weights

        self.eaec_calls += 1

        denom = max(
            1,
            aggregate[
                "valid_response_tokens"
            ],
        )

        summary = {
            "status":
                "PASS_EAEC_RUNTIME_BATCH",
            "call":
                self.eaec_calls,
            "core_ratio":
                self.core_ratio,
            "core_token_ratio":
                aggregate[
                    "core_tokens"
                ]
                / denom,
            "token_weight_mean":
                float(
                    valid_mean.item()
                ),
            "token_weight_min":
                weight_min,
            "token_weight_max":
                weight_max,
            **aggregate,
        }

        append_jsonl(
            self.audit_path,
            summary,
        )

        print(
            "EAEC_RUNTIME_BATCH_PASS "
            f"call={self.eaec_calls} "
            f"active={aggregate['active_patches']} "
            f"resolved={aggregate['resolved_patches']} "
            f"ambiguous={aggregate['ambiguous_patches']} "
            f"unlocated={aggregate['unlocated_patches']} "
            f"core_ratio={summary['core_token_ratio']:.6f} "
            f"weight_mean={summary['token_weight_mean']:.6f} "
            f"weight_min={weight_min:.6f} "
            f"weight_max={weight_max:.6f}",
            flush=True,
        )

        return out
