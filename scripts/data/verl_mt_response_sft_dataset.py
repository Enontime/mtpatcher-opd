#!/usr/bin/env python3
"""MT response-only SFT dataset adapter for Verl.

Scope is deliberately narrow:
- text-only
- exactly one user turn followed by one assistant turn
- Qwen3 non-thinking mode
- response-only supervision
- training tokens are exactly:
    prompt_ids + target_ids + [eos]
- prompt is masked; target + EOS are supervised

The whole chat-template rendering is used only as a consistency oracle.
Training infrastructure remains owned by Verl.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


class MTResponseOnlySFTDataset(MultiTurnSFTDataset):
    """Strict single-turn MT SFT adapter.

    This intentionally fails closed when the input falls outside the
    experiment definition instead of silently generalizing to other tasks.
    """

    def __init__(
        self,
        parquet_files,
        tokenizer,
        config,
        processor=None,
        max_samples=-1,
    ):
        if processor is not None:
            raise ValueError(
                "MTResponseOnlySFTDataset supports text-only SFT; "
                "processor must be None."
            )

        super().__init__(
            parquet_files=parquet_files,
            tokenizer=tokenizer,
            config=config,
            processor=processor,
            max_samples=max_samples,
        )

        if self.pad_mode != "no_padding":
            raise ValueError(
                f"pad_mode must be 'no_padding', got {self.pad_mode!r}"
            )

        if self.truncation != "error":
            raise ValueError(
                "truncation must be 'error'; silent truncation is forbidden."
            )

        if self.tokenizer.eos_token_id is None:
            raise ValueError("tokenizer.eos_token_id is None")

        # This adapter has one frozen semantic meaning.
        if self.enable_thinking_default not in (False, None):
            raise ValueError(
                "enable_thinking_default must be False (or absent)."
            )

        configured = dict(self.apply_chat_template_kwargs)
        if configured.get("enable_thinking", False) is not False:
            raise ValueError(
                "apply_chat_template_kwargs.enable_thinking must be False."
            )

    @staticmethod
    def _require_text_message(message: Any, index: int) -> None:
        if not isinstance(message, dict):
            raise TypeError(
                f"message {index} must be dict, got {type(message)!r}"
            )
        if not isinstance(message.get("role"), str):
            raise TypeError(f"message {index} has invalid role")
        if not isinstance(message.get("content"), str):
            raise TypeError(
                f"message {index} must contain plain text content"
            )

    def _tokenize_plain(self, text: str) -> list[int]:
        ids = self.tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]

        if not isinstance(ids, list) or not all(
            isinstance(x, int) for x in ids
        ):
            raise TypeError("tokenizer returned invalid input_ids")

        return ids

    def _render_prompt_ids(self, user_messages: list[dict]) -> list[int]:
        # Deliberately mirrors the historical Human-SFT prompt construction:
        # apply_chat_template(tokenize=False) -> tokenizer(..., no specials).
        prompt_text = self.tokenizer.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return self._tokenize_plain(prompt_text)

    def _render_full_ids(self, messages: list[dict]) -> list[int]:
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        return self._tokenize_plain(full_text)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        row = self.dataframe.iloc[item].to_dict()
        messages = self._build_messages(row)

        # ----- experiment contract -----
        if len(messages) != 2:
            raise ValueError(
                f"row {item}: expected exactly 2 messages, got {len(messages)}"
            )

        self._require_text_message(messages[0], 0)
        self._require_text_message(messages[1], 1)

        roles = [m["role"] for m in messages]
        if roles != ["user", "assistant"]:
            raise ValueError(
                f"row {item}: expected roles ['user','assistant'], got {roles}"
            )

        if self.tools is not None:
            tools = self.tools[item]
            if tools not in (None, []):
                raise ValueError(
                    f"row {item}: tools are unsupported in MT SFT"
                )

        if self.enable_thinking is not None:
            thinking = self.enable_thinking[item]
            if thinking not in (None, False):
                raise ValueError(
                    f"row {item}: enable_thinking must be False"
                )

        target_raw = row.get("target_translation")
        if not isinstance(target_raw, str):
            raise TypeError(
                f"row {item}: target_translation must be str"
            )

        # Historical Human-SFT explicitly strips the target.
        target = target_raw.strip()
        if not target:
            raise ValueError(
                f"row {item}: empty target_translation"
            )

        assistant_raw = messages[1]["content"]
        if assistant_raw.strip() != target:
            raise ValueError(
                f"row {item}: assistant content != target_translation"
            )

        # Work on a local copy only.
        messages = deepcopy(messages)
        messages[1]["content"] = target

        prompt_ids = self._render_prompt_ids(messages[:-1])
        target_ids = self._tokenize_plain(target)

        if not target_ids:
            raise ValueError(
                f"row {item}: target tokenization is empty"
            )

        eos = int(self.tokenizer.eos_token_id)

        # Exact historical Human-SFT training sequence.
        input_ids_list = prompt_ids + target_ids + [eos]

        # ----- whole-template consistency oracle -----
        #
        # Qwen3 non-thinking generation prompt includes the empty thinking
        # block. The filled whole conversation must begin with precisely that
        # prompt and then the clean MT target.
        full_ids = self._render_full_ids(messages)

        if len(full_ids) < len(prompt_ids):
            raise AssertionError(
                f"row {item}: full rendering shorter than prompt"
            )

        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise AssertionError(
                f"row {item}: prompt is not an exact prefix of whole render"
            )

        response = full_ids[len(prompt_ids) :]

        if response[: len(target_ids)] != target_ids:
            raise AssertionError(
                f"row {item}: target does not start exactly at response boundary"
            )

        eos_pos = len(target_ids)
        if len(response) <= eos_pos or response[eos_pos] != eos:
            got = response[eos_pos] if len(response) > eos_pos else None
            raise AssertionError(
                f"row {item}: expected EOS {eos} immediately after target, "
                f"got {got}"
            )

        # Suffix such as Qwen chat-template trailing newline is intentionally
        # excluded from the actual training sequence. This gives exact parity
        # with the historical objective: prompt + target + EOS.

        if len(input_ids_list) > self.max_length:
            raise ValueError(
                f"row {item}: sequence_length={len(input_ids_list)} "
                f"exceeds max_length={self.max_length}"
            )

        input_ids = torch.tensor(
            input_ids_list,
            dtype=torch.long,
        )

        loss_mask = torch.zeros(
            len(input_ids_list),
            dtype=torch.long,
        )
        loss_mask[len(prompt_ids) :] = 1

        position_ids = torch.arange(
            len(input_ids_list),
            dtype=torch.long,
        )

        # Basic invariants. Fail here rather than inside a training worker.
        if not (
            input_ids.shape
            == loss_mask.shape
            == position_ids.shape
        ):
            raise AssertionError(
                f"row {item}: tensor shape mismatch"
            )

        if loss_mask[: len(prompt_ids)].any():
            raise AssertionError(
                f"row {item}: prompt token accidentally supervised"
            )

        expected_supervised = len(target_ids) + 1
        actual_supervised = int(loss_mask.sum().item())

        if actual_supervised != expected_supervised:
            raise AssertionError(
                f"row {item}: supervised token count "
                f"{actual_supervised} != {expected_supervised}"
            )

        if input_ids[-1].item() != eos or loss_mask[-1].item() != 1:
            raise AssertionError(
                f"row {item}: EOS must be the final supervised token"
            )

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
