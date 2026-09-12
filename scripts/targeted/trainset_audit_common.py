#!/usr/bin/env python3
"""Stable building blocks shared by the current train-set audits.

This module is deliberately narrower than a general experiment framework.  It
contains only behavior that is identical in both the OPD and SFT train-set
audits: the frozen student-facing translation prompt, durable JSON/JSONL I/O,
canonical Chemistry target lookup, model loading, and greedy translation.

Experiment identity, input hashes, arm ordering, output paths, progress/state
schemas, and summary construction stay in the experiment entrypoints because
those details are part of each audit's provenance contract.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


DIRECT_TRANSLATION_PROMPT = (
    "Translate the following text into English without additional explanations:"
    "\n\n{source}\n\n"
)
DIRECT_TRANSLATION_PROMPT_SHA256 = (
    "63101d0739ff2ee11b0e5d548b85dcd18be7a241777893326140f12778d9a8b8"
)


def sha256_utf8(text: str) -> str:
    """Return the SHA-256 identity of UTF-8 scientific text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading a potentially large artifact into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    """Raise a visible contract error instead of relying on disabled assertions."""
    if not condition:
        raise RuntimeError(message)


def read_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL records in their frozen on-disk order."""
    records = []
    with open(path, "r", encoding="utf-8") as file_handle:
        for line in file_handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_text_atomic(path: str | Path, content: str) -> None:
    """Durably replace a text artifact through its adjacent ``.tmp`` path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(content)
        file_handle.flush()
        os.fsync(file_handle.fileno())
    os.replace(temporary_path, path)


def write_json_atomic(path: str | Path, value: Any) -> None:
    """Write the audit JSON format atomically, preserving its byte contract."""
    write_text_atomic(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
    )


def write_jsonl_atomic(
    path: str | Path,
    records: Iterable[dict[str, Any]],
) -> None:
    """Atomically replace an ordered JSONL artifact with durable bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as file_handle:
        for record in records:
            file_handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
        file_handle.flush()
        os.fsync(file_handle.fileno())
    os.replace(temporary_path, path)


def append_jsonl_durable(path: str | Path, record: dict[str, Any]) -> None:
    """Append one resumable prediction record and force it to durable storage."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file_handle:
        file_handle.write(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
        file_handle.flush()
        os.fsync(file_handle.fileno())


def canonical_chemistry_english_name(
    row: dict[str, Any],
    *,
    missing_message: str,
) -> str:
    """Resolve the explicit Chemistry target used by the substring metric."""
    direct_name = row.get("en_name")
    if isinstance(direct_name, str) and direct_name.strip():
        return direct_name.strip()

    lexical_record = row.get("lexical_record")
    if isinstance(lexical_record, dict):
        lexical_name = lexical_record.get("en_name")
        if isinstance(lexical_name, str) and lexical_name.strip():
            return lexical_name.strip()

    # The caller supplies its historical error wording so invalid-input
    # diagnostics remain stable while the lookup logic has one implementation.
    raise RuntimeError(missing_message)


def translation_contains_chemistry_target(
    translation: str,
    canonical_english_name: str,
) -> bool:
    """Apply the frozen case-insensitive Chemistry substring metric."""
    return canonical_english_name.casefold() in translation.casefold()


def load_trainset_audit_model(model_path: str | Path, device):
    """Load tokenizer and BF16 causal LM under the audits' frozen settings."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return tokenizer, model


def generate_greedy_direct_translation(
    model,
    tokenizer,
    device,
    source: str,
) -> str:
    """Translate one source under the frozen train-set audit generation contract."""
    import torch

    user_text = DIRECT_TRANSLATION_PROMPT.format(source=source)
    rendered_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    tokenized_prompt = tokenizer(
        rendered_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    tokenized_prompt = {
        name: tensor.to(device)
        for name, tensor in tokenized_prompt.items()
    }
    prompt_length = tokenized_prompt["input_ids"].shape[1]

    with torch.inference_mode():
        generated_ids = model.generate(
            **tokenized_prompt,
            do_sample=False,
            max_new_tokens=512,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    response_ids = generated_ids[0, prompt_length:]
    return tokenizer.decode(
        response_ids,
        skip_special_tokens=True,
    ).strip()
