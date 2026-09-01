#!/usr/bin/env python3

import argparse
import importlib.util
import json
import os
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


# ============================================================
# Reuse frozen Vanilla OPD implementation for:
# - JsonlDataset
# - set_seed
# - exact translation prompt construction
# - checkpoint saving
# ============================================================

ROOT = Path(
    os.environ["ROOT"]
)

FROZEN_PATH = (
    ROOT
    / "scripts"
    / "mtpatcher_v3"
    / "train_opd_forwardkl_torchnpu.py"
)

spec = importlib.util.spec_from_file_location(
    "frozen_vanilla_fkl",
    FROZEN_PATH,
)

frozen = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    frozen
)


SYSTEM_PROMPT = """You are a high-precision machine translation error annotator.

Your task is to evaluate a Chinese-to-English translation draft.

Focus on translation correctness and faithfulness:
- mistranslation
- omission
- addition
- terminology
- named entities
- numbers
- grammar that changes or obscures meaning
- serious fluency problems

Do not rewrite an already correct translation merely because you prefer another wording.
Do not invent errors.
You do not have access to a human reference translation.

Return JSON only.
"""


def build_feedback_messages(
    source,
    draft,
):

    user = f"""Source Chinese text:
{source}

Student English translation:
{draft}

Determine whether the Student translation contains a substantive translation error.

Return exactly one JSON object with this schema:

{{
  "has_error": true,
  "errors": [
    {{
      "source_span": "the relevant span copied from the Chinese source",
      "translation_span": "the problematic span copied from the Student translation",
      "error_type": "Mistranslation | Omission | Addition | Terminology | Named Entity | Number | Grammar | Fluency | Other",
      "explanation": "why this is an error",
      "correction": "the corrected English rendering for this error"
    }}
  ],
  "post_edit": "the complete corrected English translation"
}}

If the translation is already accurate, complete, and natural enough, return:

{{
  "has_error": false,
  "errors": [],
  "post_edit": "{draft}"
}}

When has_error is false, post_edit must copy the Student translation exactly.
When has_error is true, post_edit must be a complete translation integrating all listed corrections.
Do not output markdown or any text outside the JSON object."""

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user,
        },
    ]


def parse_json(raw):

    text = raw.strip()

    if text.startswith("```"):

        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if (
            lines
            and lines[-1]
            .strip()
            .startswith("```")
        ):
            lines = lines[:-1]

        text = "\n".join(
            lines
        ).strip()


    candidates = [text]

    left = text.find("{")
    right = text.rfind("}")

    if (
        left >= 0
        and right > left
    ):
        candidates.append(
            text[left:right + 1]
        )


    obj = None

    for candidate in candidates:

        try:

            value = json.loads(
                candidate
            )

            if isinstance(
                value,
                dict,
            ):
                obj = value
                break

        except Exception:
            pass


    if obj is None:

        return {
            "parse_ok": False,
            "has_error": None,
            "errors": [],
            "post_edit": None,
        }


    has_error = obj.get(
        "has_error"
    )

    if isinstance(
        has_error,
        str,
    ):

        z = (
            has_error
            .strip()
            .lower()
        )

        if z in {
            "true",
            "yes",
            "1",
        }:
            has_error = True

        elif z in {
            "false",
            "no",
            "0",
        }:
            has_error = False

        else:
            has_error = None


    if not isinstance(
        has_error,
        bool,
    ):
        has_error = None


    errors = obj.get(
        "errors",
        [],
    )

    if not isinstance(
        errors,
        list,
    ):
        errors = []


    clean_errors = [
        x
        for x in errors
        if isinstance(
            x,
            dict,
        )
    ]


    post_edit = obj.get(
        "post_edit"
    )

    if not isinstance(
        post_edit,
        str,
    ):
        post_edit = None

    elif not post_edit.strip():
        post_edit = None

    else:
        post_edit = post_edit.strip()


    parse_ok = (
        isinstance(
            has_error,
            bool,
        )
        and post_edit is not None
        and isinstance(
            errors,
            list,
        )
    )


    return {
        "parse_ok":
            parse_ok,

        "has_error":
            has_error,

        "errors":
            clean_errors,

        "post_edit":
            post_edit,
    }


def read_source_map(path):

    result = {}

    with open(
        path,
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            x = json.loads(line)

            if set(x.keys()) != {
                "index",
                "source",
            }:
                raise RuntimeError(
                    "source-map leakage: "
                    f"unexpected keys={list(x.keys())}"
                )

            idx = int(
                x["index"]
            )

            if idx in result:
                raise RuntimeError(
                    f"duplicate source index={idx}"
                )

            result[idx] = (
                x["source"]
            )

    return result


def decode_ids(
    tokenizer,
    ids,
):

    if not ids:
        return ""

    return tokenizer.decode(
        ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def trim_at_eos(
    ids,
    eos_id,
):

    ids = list(ids)

    if eos_id in ids:

        return ids[
            :ids.index(eos_id)
        ]

    return ids


def find_unique(
    text,
    needle,
):

    if not needle:
        return None

    first = text.find(
        needle
    )

    if first < 0:
        return None

    if (
        text.find(
            needle,
            first + 1,
        )
        >= 0
    ):
        return None

    return first


def find_token_boundary(
    tokenizer,
    response_ids,
    prefix_text,
):

    for k in range(
        len(response_ids) + 1
    ):

        if (
            decode_ids(
                tokenizer,
                response_ids[:k],
            )
            == prefix_text
        ):
            return k

    return None


def valid_errors(
    parsed,
    source,
    draft,
    response_ids,
    tokenizer,
):

    rejects = Counter()

    if not parsed[
        "parse_ok"
    ]:

        rejects[
            "parse_fail"
        ] += 1

        return [], rejects


    if not parsed[
        "has_error"
    ]:

        return [], rejects


    post_edit = (
        parsed[
            "post_edit"
        ]
        or ""
    )


    errors = parsed[
        "errors"
    ]


    span_corrections = {}

    for e in errors:

        ts = str(
            e.get(
                "translation_span",
                "",
            )
            or ""
        ).strip()

        correction = str(
            e.get(
                "correction",
                "",
            )
            or ""
        ).strip()

        if ts:

            span_corrections.setdefault(
                ts,
                set(),
            ).add(
                correction
            )


    conflicting = {
        span
        for span, cs
        in span_corrections.items()
        if len(
            {
                c
                for c in cs
                if c
            }
        ) > 1
    }


    valid = []


    for error_id, e in enumerate(
        errors
    ):

        source_span = str(
            e.get(
                "source_span",
                "",
            )
            or ""
        ).strip()

        translation_span = str(
            e.get(
                "translation_span",
                "",
            )
            or ""
        ).strip()

        correction = str(
            e.get(
                "correction",
                "",
            )
            or ""
        ).strip()


        if not source_span:
            rejects[
                "empty_source_span"
            ] += 1
            continue

        if not translation_span:
            rejects[
                "empty_translation_span"
            ] += 1
            continue

        if not correction:
            rejects[
                "empty_correction"
            ] += 1
            continue

        if (
            correction
            == translation_span
        ):
            rejects[
                "no_op_correction"
            ] += 1
            continue

        if (
            source_span
            not in source
        ):
            rejects[
                "source_span_not_exact"
            ] += 1
            continue

        if (
            translation_span
            in conflicting
        ):
            rejects[
                "conflicting_same_span"
            ] += 1
            continue

        if (
            correction
            not in post_edit
        ):
            rejects[
                "correction_not_in_postedit"
            ] += 1
            continue


        start = find_unique(
            draft,
            translation_span,
        )

        if start is None:
            rejects[
                "translation_span_not_unique"
            ] += 1
            continue


        prefix_before = (
            draft[:start]
        )


        # --------------------------------------------------
        # Semantic character boundary -> token intervention seam
        #
        # Exact case:
        #   preserve the raw Student response IDs directly.
        #
        # Non-exact case:
        #   preserve the longest complete raw Student token
        #   prefix whose decoded text is still a prefix of
        #   prefix_before, then re-tokenize only:
        #
        #       bridge + correction
        #
        # This avoids requiring the semantic error span to begin
        # exactly at a tokenizer boundary.
        # --------------------------------------------------

        boundary = (
            find_token_boundary(
                tokenizer,
                response_ids,
                prefix_before,
            )
        )

        bridge = ""

        if boundary is not None:

            raw_prefix_ids = (
                response_ids[
                    :boundary
                ]
            )

        else:

            safe_k = None
            safe_text = None

            for k in range(
                len(response_ids) + 1
            ):

                decoded_prefix = (
                    decode_ids(
                        tokenizer,
                        response_ids[:k],
                    )
                )

                if prefix_before.startswith(
                    decoded_prefix
                ):

                    safe_k = k
                    safe_text = decoded_prefix

            if (
                safe_k is None
                or safe_text is None
            ):

                rejects[
                    "local_retoken_fail"
                ] += 1

                continue

            raw_prefix_ids = (
                response_ids[
                    :safe_k
                ]
            )

            bridge = (
                prefix_before[
                    len(safe_text):
                ]
            )

        intervention_text = (
            bridge
            + correction
        )

        correction_ids = (
            tokenizer.encode(
                intervention_text,
                add_special_tokens=False,
            )
        )

        if not correction_ids:

            rejects[
                "empty_correction_tokens"
            ] += 1

            continue

        combined = (
            raw_prefix_ids
            + correction_ids
        )

        corrected_prefix = (
            prefix_before
            + correction
        )

        if (
            decode_ids(
                tokenizer,
                combined,
            )
            != corrected_prefix
        ):

            rejects[
                "local_retoken_fail"
            ] += 1

            continue


        valid.append({
            "error_id":
                error_id,

            "source_span":
                source_span,

            "translation_span":
                translation_span,

            "correction":
                correction,

            "error_type":
                str(
                    e.get(
                        "error_type",
                        "",
                    )
                    or ""
                ),

            "raw_prefix_ids":
                raw_prefix_ids,

            "correction_ids":
                correction_ids,

            "corrected_prefix":
                corrected_prefix,
        })


    return valid, rejects


def run_feedbacker(
    teacher,
    tokenizer,
    source,
    draft,
    device,
    max_prompt_tokens,
    max_new_tokens,
):

    messages = (
        build_feedback_messages(
            source,
            draft,
        )
    )


    rendered = (
        tokenizer
        .apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


    old_side = (
        tokenizer
        .truncation_side
    )

    tokenizer.truncation_side = (
        "right"
    )


    # Intentionally follows the historical
    # structured Feedbacker:
    # no add_special_tokens=False override.
    enc = tokenizer(
        rendered,
        return_tensors="pt",
        truncation=True,
        max_length=(
            max_prompt_tokens
        ),
    )


    tokenizer.truncation_side = (
        old_side
    )


    enc = {
        k: v.to(device)
        for k, v
        in enc.items()
    }


    prompt_len = (
        enc[
            "input_ids"
        ]
        .shape[1]
    )


    with torch.no_grad():

        out = teacher.generate(
            **enc,
            do_sample=False,
            max_new_tokens=(
                max_new_tokens
            ),
            use_cache=True,
            eos_token_id=(
                tokenizer.eos_token_id
            ),
            pad_token_id=(
                tokenizer.pad_token_id
            ),
        )


    raw_ids = (
        out[
            0,
            prompt_len:
        ]
        .tolist()
    )


    raw = (
        tokenizer.decode(
            raw_ids,
            skip_special_tokens=True,
        )
    )


    return (
        raw,
        parse_json(
            raw
        ),
    )


def reduce_sum(
    value,
    device,
):

    x = torch.tensor(
        float(value),
        device=device,
        # HCCL in the current Ascend stack does not
        # support float64 all_reduce. These tensors are
        # epoch telemetry only; float32 is sufficient.
        dtype=torch.float32,
    )

    dist.all_reduce(
        x,
        op=dist.ReduceOp.SUM,
    )

    return float(
        x.cpu()
    )


def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--student",
        required=True,
    )

    p.add_argument(
        "--teacher",
        required=True,
    )

    p.add_argument(
        "--train",
        required=True,
    )

    p.add_argument(
        "--source-map",
        required=True,
    )

    p.add_argument(
        "--output-dir",
        required=True,
    )

    p.add_argument(
        "--expected-rows",
        type=int,
        default=512,
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=1e-6,
    )

    p.add_argument(
        "--max-prompt-length",
        type=int,
        default=512,
    )

    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )

    p.add_argument(
        "--feedback-max-prompt-tokens",
        type=int,
        default=1536,
    )

    p.add_argument(
        "--feedback-max-new-tokens",
        type=int,
        default=768,
    )

    p.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    p.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )

    p.add_argument(
        "--top-p",
        type=float,
        default=0.8,
    )

    p.add_argument(
        "--top-k",
        type=int,
        default=20,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=20260824,
    )

    return p.parse_args()


def main():

    args = parse_args()

    local_rank = int(
        os.environ[
            "LOCAL_RANK"
        ]
    )

    rank = int(
        os.environ["RANK"]
    )

    world_size = int(
        os.environ[
            "WORLD_SIZE"
        ]
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


    frozen.set_seed(
        args.seed,
        rank,
    )


    if rank == 0:

        print("=" * 80)
        print(
            "MT-PATCHER EC-ROPD "
            "MATCHED512 V1"
        )
        print("=" * 80)

        print(
            "WORLD_SIZE =",
            world_size,
        )

        print(
            "STUDENT =",
            args.student,
        )

        print(
            "TEACHER_FEEDBACKER =",
            args.teacher,
        )

        print(
            "TRAIN =",
            args.train,
        )

        print(
            "SOURCE_MAP =",
            args.source_map,
        )

        print(
            "REFERENCE_USED = False"
        )

        print(
            "STALE_FEEDBACK_USED = False"
        )

        print(
            "OBJECTIVE = "
            "source-mean(error-mean("
            "resumed-token FKL))"
        )

        print(
            "ROLLOUT = "
            f"T={args.temperature} "
            f"top_p={args.top_p} "
            f"top_k={args.top_k}"
        )


    student_cfg = (
        AutoConfig
        .from_pretrained(
            args.student,
            local_files_only=True,
        )
    )

    teacher_cfg = (
        AutoConfig
        .from_pretrained(
            args.teacher,
            local_files_only=True,
        )
    )


    if (
        student_cfg.vocab_size
        != teacher_cfg.vocab_size
    ):
        raise RuntimeError(
            "Student/Teacher "
            "vocab mismatch"
        )


    student_tok = (
        AutoTokenizer
        .from_pretrained(
            args.student,
            local_files_only=True,
        )
    )

    teacher_tok = (
        AutoTokenizer
        .from_pretrained(
            args.teacher,
            local_files_only=True,
        )
    )


    if (
        student_tok.pad_token_id
        is None
    ):
        student_tok.pad_token = (
            student_tok.eos_token
        )


    if (
        teacher_tok.pad_token_id
        is None
    ):
        teacher_tok.pad_token = (
            teacher_tok.eos_token
        )


    if (
        student_tok.eos_token_id
        != teacher_tok.eos_token_id
    ):
        raise RuntimeError(
            "Student/Teacher EOS mismatch"
        )


    time.sleep(
        local_rank * 0.5
    )


    student = (
        AutoModelForCausalLM
        .from_pretrained(
            args.student,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=True,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        .to(device)
    )

    student.config.use_cache = False


    dist.barrier()


    time.sleep(
        local_rank * 0.5
    )


    teacher = (
        AutoModelForCausalLM
        .from_pretrained(
            args.teacher,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=True,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        .to(device)
    )

    teacher.eval()
    teacher.config.use_cache = False


    for parameter in (
        teacher.parameters()
    ):
        parameter.requires_grad_(
            False
        )


    dist.barrier()


    student = DDP(
        student,
        device_ids=[
            local_rank
        ],
        output_device=(
            local_rank
        ),
        broadcast_buffers=False,
        find_unused_parameters=False,
    )


    dataset = (
        frozen.JsonlDataset(
            args.train
        )
    )


    if (
        len(dataset)
        != args.expected_rows
    ):
        raise RuntimeError(
            "Unexpected training rows: "
            f"{len(dataset)}"
        )


    source_map = (
        read_source_map(
            args.source_map
        )
    )


    for row in dataset.rows:

        if "reference" in row:
            raise RuntimeError(
                "Reference leakage "
                f"index={row.get('index')}"
            )

        idx = int(
            row["index"]
        )

        if idx not in source_map:
            raise RuntimeError(
                f"Missing source index={idx}"
            )


    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )


    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=0,
        collate_fn=lambda x: x,
    )


    total_source_steps = (
        len(loader)
        * args.epochs
    )


    warmup_steps = int(
        total_source_steps
        * args.warmup_ratio
    )


    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=(
            args.weight_decay
        ),
    )


    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=(
                warmup_steps
            ),
            num_training_steps=(
                total_source_steps
            ),
        )
    )


    if rank == 0:

        print(
            "TRAIN_ROWS =",
            len(dataset),
        )

        print(
            "LOCAL_STEPS_PER_EPOCH =",
            len(loader),
        )

        print(
            "TOTAL_SOURCE_STEPS =",
            total_source_steps,
        )

        print(
            "GLOBAL_BATCH =",
            world_size,
        )

        print(
            "WARMUP_STEPS =",
            warmup_steps,
        )

        print(
            "EC_ROPD_TRAINING_START",
            flush=True,
        )


    optimizer_updates = 0
    source_step = 0


    reject_keys = [
        "parse_fail",
        "empty_source_span",
        "empty_translation_span",
        "empty_correction",
        "no_op_correction",
        "source_span_not_exact",
        "conflicting_same_span",
        "correction_not_in_postedit",
        "translation_span_not_unique",
        "token_boundary_not_exact",
        "local_retoken_fail",
        "empty_correction_tokens",
        "corrected_prefix_roundtrip_fail",
    ]


    for epoch in range(
        1,
        args.epochs + 1,
    ):

        sampler.set_epoch(
            epoch
        )

        student.train()

        epoch_start = time.time()

        local_sources = 0
        local_parse_ok = 0
        local_has_error = 0
        local_active_sources = 0
        local_reported_errors = 0
        local_valid_states = 0
        local_resumed_tokens = 0

        local_source_loss_sum = 0.0
        local_active_source_loss_sum = 0.0
        local_token_kl_sum = 0.0

        local_rejects = Counter()


        for local_step, batch in enumerate(
            loader,
            start=1,
        ):

            source_step += 1

            row = batch[0]

            idx = int(
                row["index"]
            )

            source = (
                source_map[idx]
            )


            prompt_ids = (
                frozen.get_prompt(
                    row,
                    student_tok,
                    args.max_prompt_length,
                )
                .to(device)
            )


            attention_mask = (
                torch.ones_like(
                    prompt_ids,
                    device=device,
                )
            )


            prompt_len = (
                prompt_ids.shape[1]
            )


            # ----------------------------------------------
            # Current Student translation.
            # Same rollout policy as frozen Vanilla OPD.
            # ----------------------------------------------

            student.module.eval()

            with torch.no_grad():

                generated = (
                    student.module.generate(
                        input_ids=(
                            prompt_ids
                        ),
                        attention_mask=(
                            attention_mask
                        ),
                        max_new_tokens=(
                            args.max_new_tokens
                        ),
                        do_sample=True,
                        temperature=(
                            args.temperature
                        ),
                        top_p=(
                            args.top_p
                        ),
                        top_k=(
                            args.top_k
                        ),
                        pad_token_id=(
                            student_tok
                            .pad_token_id
                        ),
                        eos_token_id=(
                            student_tok
                            .eos_token_id
                        ),
                        use_cache=True,
                    )
                )


            student.module.train()


            response_ids_all = (
                generated[
                    0,
                    prompt_len:
                ]
                .tolist()
            )


            response_ids = (
                trim_at_eos(
                    response_ids_all,
                    student_tok
                    .eos_token_id,
                )
            )


            draft = decode_ids(
                student_tok,
                response_ids,
            )


            local_sources += 1


            if not draft.strip():

                parsed = {
                    "parse_ok":
                        False,
                    "has_error":
                        None,
                    "errors":
                        [],
                    "post_edit":
                        None,
                }

                valid = []

                rejects = Counter({
                    "parse_fail":
                        1
                })

            else:

                _, parsed = (
                    run_feedbacker(
                        teacher,
                        teacher_tok,
                        source,
                        draft,
                        device,
                        args.feedback_max_prompt_tokens,
                        args.feedback_max_new_tokens,
                    )
                )


                valid, rejects = (
                    valid_errors(
                        parsed,
                        source,
                        draft,
                        response_ids,
                        student_tok,
                    )
                )


            local_rejects.update(
                rejects
            )


            if parsed[
                "parse_ok"
            ]:
                local_parse_ok += 1


            if parsed[
                "has_error"
            ] is True:

                local_has_error += 1

                local_reported_errors += (
                    len(
                        parsed[
                            "errors"
                        ]
                    )
                )


            # ----------------------------------------------
            # Build all independent resumed states
            # BEFORE any update.
            # Every valid error object is symmetric.
            # ----------------------------------------------

            states = []


            for error in valid:

                corrected_response_ids = (
                    error[
                        "raw_prefix_ids"
                    ]
                    + error[
                        "correction_ids"
                    ]
                )


                corrected_input_ids = (
                    torch.cat(
                        [
                            prompt_ids,

                            torch.tensor(
                                [
                                    corrected_response_ids
                                ],
                                dtype=torch.long,
                                device=device,
                            ),
                        ],
                        dim=1,
                    )
                )


                corrected_len = (
                    corrected_input_ids
                    .shape[1]
                )


                corrected_mask = (
                    torch.ones_like(
                        corrected_input_ids,
                        device=device,
                    )
                )


                student.module.eval()

                with torch.no_grad():

                    resumed = (
                        student.module.generate(
                            input_ids=(
                                corrected_input_ids
                            ),
                            attention_mask=(
                                corrected_mask
                            ),
                            max_new_tokens=(
                                args.max_new_tokens
                            ),
                            do_sample=True,
                            temperature=(
                                args.temperature
                            ),
                            top_p=(
                                args.top_p
                            ),
                            top_k=(
                                args.top_k
                            ),
                            pad_token_id=(
                                student_tok
                                .pad_token_id
                            ),
                            eos_token_id=(
                                student_tok
                                .eos_token_id
                            ),
                            use_cache=True,
                        )
                    )


                student.module.train()


                resumed_len = (
                    resumed.shape[1]
                    - corrected_len
                )


                if resumed_len <= 0:
                    continue


                # Normal detached token tensor for
                # subsequent autograd forward.
                full_ids = (
                    resumed
                    .detach()
                    .clone()
                )


                states.append({
                    "full_ids":
                        full_ids,

                    "corrected_len":
                        corrected_len,

                    "resumed_len":
                        resumed_len,

                    "error_id":
                        error[
                            "error_id"
                        ],
                })


            n_states = len(
                states
            )


            if n_states > 0:
                local_active_sources += 1
                local_valid_states += n_states


            # Need to know whether the entire
            # global batch has zero usable states.
            active = torch.tensor(
                [
                    1
                    if n_states > 0
                    else 0
                ],
                dtype=torch.int64,
                device=device,
            )

            dist.all_reduce(
                active,
                op=dist.ReduceOp.SUM,
            )

            global_active_sources = int(
                active.item()
            )


            optimizer.zero_grad(
                set_to_none=True
            )


            source_loss_value = 0.0


            if global_active_sources > 0:

                if n_states > 0:

                    for state_no, state in enumerate(
                        states
                    ):

                        sync_context = (
                            student.no_sync()
                            if state_no
                            < n_states - 1
                            else nullcontext()
                        )


                        with sync_context:

                            full_ids = (
                                state[
                                    "full_ids"
                                ]
                            )

                            full_mask = torch.ones(
                                full_ids.shape,
                                dtype=torch.long,
                                device=device,
                            )


                            student_out = student(
                                input_ids=(
                                    full_ids
                                ),
                                attention_mask=(
                                    full_mask
                                ),
                                use_cache=False,
                            )


                            start = (
                                state[
                                    "corrected_len"
                                ]
                                - 1
                            )


                            s_logits = (
                                student_out.logits[
                                    :,
                                    start:-1,
                                    :,
                                ]
                            )


                            with torch.no_grad():

                                teacher_out = (
                                    teacher(
                                        input_ids=(
                                            full_ids
                                        ),
                                        attention_mask=(
                                            full_mask
                                        ),
                                        use_cache=False,
                                    )
                                )


                                t_logits = (
                                    teacher_out.logits[
                                        :,
                                        start:-1,
                                        :,
                                    ]
                                )


                                t_logp = (
                                    F.log_softmax(
                                        t_logits.float(),
                                        dim=-1,
                                    )
                                )

                                t_prob = (
                                    t_logp.exp()
                                )


                            s_logp = (
                                F.log_softmax(
                                    s_logits.float(),
                                    dim=-1,
                                )
                            )


                            token_kl = (
                                t_prob
                                * (
                                    t_logp
                                    - s_logp
                                )
                            ).sum(
                                dim=-1
                            )


                            error_loss = (
                                token_kl.mean()
                            )


                            if not torch.isfinite(
                                error_loss
                            ):
                                raise RuntimeError(
                                    "non-finite EC-ROPD loss "
                                    f"index={idx}"
                                )


                            scaled_loss = (
                                error_loss
                                / n_states
                            )


                            scaled_loss.backward()


                            source_loss_value += (
                                float(
                                    error_loss
                                    .detach()
                                    .cpu()
                                )
                                / n_states
                            )


                            local_token_kl_sum += (
                                float(
                                    token_kl
                                    .detach()
                                    .sum()
                                    .cpu()
                                )
                            )


                            local_resumed_tokens += (
                                token_kl.numel()
                            )


                            del (
                                student_out,
                                teacher_out,
                                s_logits,
                                t_logits,
                                t_logp,
                                t_prob,
                                s_logp,
                                token_kl,
                                error_loss,
                                scaled_loss,
                            )


                else:

                    # This source contributes exactly zero.
                    # A zero DDP backward is still required
                    # because other ranks in this global
                    # source batch may have active states.

                    dummy_out = student(
                        input_ids=(
                            prompt_ids
                        ),
                        attention_mask=(
                            attention_mask
                        ),
                        use_cache=False,
                    )


                    zero_loss = (
                        dummy_out.logits[
                            :,
                            -1:,
                            :
                        ]
                        .sum()
                        * 0.0
                    )


                    zero_loss.backward()


                    del (
                        dummy_out,
                        zero_loss,
                    )


                grad_norm = (
                    torch.nn.utils
                    .clip_grad_norm_(
                        student.parameters(),
                        args.max_grad_norm,
                    )
                )


                if not torch.isfinite(
                    grad_norm
                ):

                    raise RuntimeError(
                        "non-finite gradient norm"
                    )


                optimizer.step()
                scheduler.step()

                optimizer_updates += 1


            # If all 16 sources have no usable
            # error state, no parameter/weight-decay
            # update is performed.


            local_source_loss_sum += (
                source_loss_value
            )


            if n_states > 0:

                local_active_source_loss_sum += (
                    source_loss_value
                )


            if (
                rank == 0
                and (
                    local_step == 1
                    or local_step % 8 == 0
                )
            ):

                print(
                    f"epoch={epoch} "
                    f"local_step={local_step}/"
                    f"{len(loader)} "
                    f"source_step={source_step}/"
                    f"{total_source_steps} "
                    f"index={idx} "
                    f"valid_states={n_states} "
                    f"global_active="
                    f"{global_active_sources}/"
                    f"{world_size} "
                    f"source_loss="
                    f"{source_loss_value:.6f} "
                    f"lr="
                    f"{optimizer.param_groups[0]['lr']:.8g} "
                    f"elapsed="
                    f"{time.time()-epoch_start:.1f}s",
                    flush=True,
                )


        # ==================================================
        # Epoch aggregate statistics across all 16 ranks.
        # ==================================================

        global_sources = int(
            reduce_sum(
                local_sources,
                device,
            )
        )

        global_parse_ok = int(
            reduce_sum(
                local_parse_ok,
                device,
            )
        )

        global_has_error = int(
            reduce_sum(
                local_has_error,
                device,
            )
        )

        global_active_sources = int(
            reduce_sum(
                local_active_sources,
                device,
            )
        )

        global_reported_errors = int(
            reduce_sum(
                local_reported_errors,
                device,
            )
        )

        global_valid_states = int(
            reduce_sum(
                local_valid_states,
                device,
            )
        )

        global_resumed_tokens = int(
            reduce_sum(
                local_resumed_tokens,
                device,
            )
        )

        global_source_loss_sum = (
            reduce_sum(
                local_source_loss_sum,
                device,
            )
        )

        global_active_source_loss_sum = (
            reduce_sum(
                local_active_source_loss_sum,
                device,
            )
        )

        global_token_kl_sum = (
            reduce_sum(
                local_token_kl_sum,
                device,
            )
        )


        reject_global = {}

        for key in reject_keys:

            reject_global[key] = int(
                reduce_sum(
                    local_rejects[
                        key
                    ],
                    device,
                )
            )


        source_mean_loss = (
            global_source_loss_sum
            / max(
                global_sources,
                1,
            )
        )


        active_source_mean_loss = (
            global_active_source_loss_sum
            / max(
                global_active_sources,
                1,
            )
        )


        token_mean_kl = (
            global_token_kl_sum
            / max(
                global_resumed_tokens,
                1,
            )
        )


        if rank == 0:

            print()
            print(
                f"EPOCH_{epoch}_COMPLETE"
            )

            print(
                "sources =",
                global_sources,
            )

            print(
                "feedback_parse_ok =",
                global_parse_ok,
            )

            print(
                "feedback_has_error =",
                global_has_error,
            )

            print(
                "active_sources =",
                global_active_sources,
            )

            print(
                "reported_errors =",
                global_reported_errors,
            )

            print(
                "valid_error_states =",
                global_valid_states,
            )

            print(
                "resumed_kl_tokens =",
                global_resumed_tokens,
            )

            print(
                "source_mean_ecropd_loss =",
                source_mean_loss,
            )

            print(
                "active_source_mean_ecropd_loss =",
                active_source_mean_loss,
            )

            print(
                "token_mean_forward_kl =",
                token_mean_kl,
            )

            print(
                "reject_counts =",
                json.dumps(
                    reject_global,
                    sort_keys=True,
                ),
            )


        checkpoint = (
            Path(
                args.output_dir
            )
            / f"epoch{epoch}"
        )


        frozen.save_model(
            student,
            student_tok,
            checkpoint,
            rank,
        )


        if rank == 0:

            print(
                "CHECKPOINT_SAVED =",
                checkpoint,
                flush=True,
            )


    if rank == 0:

        print(
            "OPTIMIZER_UPDATES =",
            optimizer_updates,
        )

        print(
            "MTPATCHER_EC_ROPD_MATCHED512_V2_PASS",
            flush=True,
        )


    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
