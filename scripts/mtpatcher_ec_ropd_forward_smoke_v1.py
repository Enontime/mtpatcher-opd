#!/usr/bin/env python3

import json
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


EXP = os.environ["EXP"]

RUN_ROOT = Path(os.environ["RUN_ROOT"])
DATA_ROOT = Path(os.environ["DATA_ROOT"])
MODEL_ROOT = Path(os.environ["MODEL_ROOT"])


PE_MODEL = (
    RUN_ROOT
    / EXP
    / "pe_k1_sft3732_b4ga4"
    / "epoch3"
)

STUDENT_TOKENIZER_PATH = (
    MODEL_ROOT
    / "Qwen3-0.6B"
)

TEACHER_MODEL = (
    MODEL_ROOT
    / "Qwen3-8B"
)

PE_DATA = (
    DATA_ROOT
    / EXP
    / "pe_k1_clean3732.jsonl"
)

OLD_FEEDBACK = (
    DATA_ROOT
    / EXP
    / "feedback_qwen3_8b_merged6565.jsonl"
)

OUT_DIR = (
    RUN_ROOT
    / EXP
    / "ec_ropd_forward_smoke_v1"
)

OUT = (
    OUT_DIR
    / "result.json"
)


SEED = 20260827

SCAN_ROWS = 12

MAX_PROMPT_TOKENS = 1024
MAX_RESPONSE_TOKENS = 128

FEEDBACK_MAX_PROMPT_TOKENS = 1536
FEEDBACK_MAX_NEW_TOKENS = 768

TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = 0


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


    candidates = [
        text
    ]

    left = text.find("{")
    right = text.rfind("}")

    if (
        left >= 0
        and right > left
    ):
        candidates.append(
            text[
                left:right + 1
            ]
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
        post_edit = (
            post_edit.strip()
        )


    parse_ok = (
        isinstance(
            has_error,
            bool,
        )
        and post_edit
        is not None
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


def read_jsonl(path):

    with path.open(
        encoding="utf-8",
    ) as f:

        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def set_seed(seed):

    random.seed(seed)

    torch.manual_seed(
        seed
    )

    if hasattr(
        torch,
        "npu",
    ):
        torch.npu.manual_seed(
            seed
        )


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

        p = ids.index(
            eos_id
        )

        return ids[:p]

    return ids


def translation_prompt_ids(
    row,
    tokenizer,
):

    messages = row.get(
        "messages"
    )

    if not isinstance(
        messages,
        list,
    ):
        raise RuntimeError(
            "missing messages"
        )


    for message in messages:

        if (
            message.get(
                "role"
            )
            == "assistant"
        ):
            raise RuntimeError(
                "assistant leakage "
                "in translation prompt"
            )


    text = (
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
        "left"
    )


    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=(
            MAX_PROMPT_TOKENS
        ),
        add_special_tokens=False,
    )


    tokenizer.truncation_side = (
        old_side
    )


    return (
        enc["input_ids"],
        text,
    )


def run_feedbacker(
    model,
    tokenizer,
    source,
    draft,
    device,
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
        "left"
    )


    enc = tokenizer(
        rendered,
        return_tensors="pt",
        truncation=True,
        max_length=(
            FEEDBACK_MAX_PROMPT_TOKENS
        ),
        add_special_tokens=False,
    )


    tokenizer.truncation_side = (
        old_side
    )


    enc = {
        k: v.to(
            device
        )
        for k, v
        in enc.items()
    }


    prompt_len = (
        enc[
            "input_ids"
        ]
        .shape[1]
    )


    with torch.inference_mode():

        out = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=(
                FEEDBACK_MAX_NEW_TOKENS
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


    raw = decode_ids(
        tokenizer,
        raw_ids,
    )


    return (
        raw,
        parse_json(
            raw
        ),
    )


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


    second = text.find(
        needle,
        first + 1,
    )

    if second >= 0:
        return None


    return first


def token_boundary(
    tokenizer,
    response_ids,
    prefix_text,
):

    # Exact preservation requirement:
    #
    # Find k such that decoding the
    # REAL Student rollout tokens
    #
    # response_ids[:k]
    #
    # equals the text immediately
    # before the Patcher error span.

    for k in range(
        len(response_ids) + 1
    ):

        text = decode_ids(
            tokenizer,
            response_ids[:k],
        )

        if text == prefix_text:
            return k

    return None


def valid_errors(
    parsed,
    source,
    draft,
    response_ids,
    tokenizer,
):

    if not parsed[
        "parse_ok"
    ]:
        return []


    if not parsed[
        "has_error"
    ]:
        return []


    post_edit = (
        parsed[
            "post_edit"
        ]
        or ""
    )


    errors = parsed[
        "errors"
    ]


    # Detect conflicting corrections
    # for the exact same Student span.

    span_corrections = {}

    for e in errors:

        ts = str(
            e.get(
                "translation_span",
                "",
            )
            or ""
        ).strip()

        c = str(
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
            ).add(c)


    conflicting = {
        span
        for span, cs
        in span_corrections.items()
        if len(
            {
                x
                for x in cs
                if x
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
            continue

        if not translation_span:
            continue

        if not correction:
            continue

        if (
            correction
            == translation_span
        ):
            continue

        if (
            source_span
            not in source
        ):
            continue

        if (
            translation_span
            in conflicting
        ):
            continue

        if (
            correction
            not in post_edit
        ):
            continue


        start = find_unique(
            draft,
            translation_span,
        )

        if start is None:
            continue


        prefix_before = (
            draft[:start]
        )


        k = token_boundary(
            tokenizer,
            response_ids,
            prefix_before,
        )

        if k is None:
            continue


        raw_prefix_ids = (
            response_ids[:k]
        )


        correction_ids = (
            tokenizer.encode(
                correction,
                add_special_tokens=False,
            )
        )


        if not correction_ids:
            continue


        combined_ids = (
            raw_prefix_ids
            + correction_ids
        )


        combined_text = (
            decode_ids(
                tokenizer,
                combined_ids,
            )
        )


        corrected_prefix = (
            prefix_before
            + correction
        )


        if (
            combined_text
            != corrected_prefix
        ):
            continue


        valid.append(
            {
                "error_id":
                    error_id,

                "error_type":
                    str(
                        e.get(
                            "error_type",
                            "",
                        )
                        or ""
                    ),

                "source_span":
                    source_span,

                "translation_span":
                    translation_span,

                "correction":
                    correction,

                "explanation":
                    str(
                        e.get(
                            "explanation",
                            "",
                        )
                        or ""
                    ),

                "char_start":
                    start,

                "token_boundary":
                    k,

                "prefix_before":
                    prefix_before,

                "corrected_prefix":
                    corrected_prefix,

                "raw_prefix_ids":
                    raw_prefix_ids,

                "correction_ids":
                    correction_ids,
            }
        )


    return valid


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    print("=" * 100)
    print(
        "EC-ROPD FORWARD-ONLY SMOKE V1"
    )
    print("=" * 100)


    print(
        "PE_MODEL =",
        PE_MODEL,
    )

    print(
        "TEACHER_FEEDBACKER =",
        TEACHER_MODEL,
    )

    print(
        "PE_DATA =",
        PE_DATA,
    )

    print(
        "OBJECTIVE = "
        "implementation validation only"
    )

    print(
        "OPTIMIZER_STEP = False"
    )

    print(
        "BACKWARD = False"
    )

    print(
        "STUDENT_ROLLOUT = sampled"
    )

    print(
        "temperature =",
        TEMPERATURE,
    )

    print(
        "top_p =",
        TOP_P,
    )

    print(
        "top_k =",
        TOP_K,
    )


    if not torch.npu.is_available():

        raise RuntimeError(
            "NPU unavailable"
        )


    device = torch.device(
        "npu:0"
    )


    set_seed(
        SEED
    )


    student_tok = (
        AutoTokenizer
        .from_pretrained(
            STUDENT_TOKENIZER_PATH,
            local_files_only=True,
        )
    )


    teacher_tok = (
        AutoTokenizer
        .from_pretrained(
            TEACHER_MODEL,
            local_files_only=True,
        )
    )


    if (
        student_tok.eos_token_id
        != teacher_tok.eos_token_id
    ):
        raise RuntimeError(
            "EOS mismatch"
        )


    if (
        student_tok.get_vocab()
        != teacher_tok.get_vocab()
    ):
        raise RuntimeError(
            "Student/Teacher vocab mismatch"
        )


    if (
        student_tok.chat_template
        != teacher_tok.chat_template
    ):
        raise RuntimeError(
            "Student/Teacher chat template mismatch"
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


    print(
        "TOKENIZER_COMPATIBILITY_PASS"
    )


    print()
    print(
        "Loading PE Student..."
    )


    student = (
        AutoModelForCausalLM
        .from_pretrained(
            PE_MODEL,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
    )


    student.to(
        device
    )

    student.eval()


    print(
        "PE Student loaded"
    )


    print()
    print(
        "Loading shared "
        "Qwen3-8B Teacher/Feedbacker..."
    )


    teacher = (
        AutoModelForCausalLM
        .from_pretrained(
            TEACHER_MODEL,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
    )


    teacher.to(
        device
    )

    teacher.eval()


    print(
        "Teacher/Feedbacker loaded"
    )


    pe_rows = read_jsonl(
        PE_DATA
    )


    feedback_rows = read_jsonl(
        OLD_FEEDBACK
    )


    source_map = {
        int(x["index"]):
            x["source"]
        for x in feedback_rows
    }


    rng = random.Random(
        SEED
    )


    candidates_rows = list(
        pe_rows
    )

    rng.shuffle(
        candidates_rows
    )


    chosen = None

    scan_log = []


    for row in candidates_rows[
        :SCAN_ROWS
    ]:

        idx = int(
            row["index"]
        )


        source = source_map.get(
            idx
        )


        if not source:

            scan_log.append(
                {
                    "index":
                        idx,

                    "status":
                        "missing_source",
                }
            )

            continue


        prompt_ids_cpu, prompt_text = (
            translation_prompt_ids(
                row,
                student_tok,
            )
        )


        prompt_ids = (
            prompt_ids_cpu.to(
                device
            )
        )


        attention_mask = (
            torch.ones_like(
                prompt_ids,
                device=device,
            )
        )


        set_seed(
            SEED + idx
        )


        with torch.inference_mode():

            generated = (
                student.generate(
                    input_ids=(
                        prompt_ids
                    ),
                    attention_mask=(
                        attention_mask
                    ),
                    max_new_tokens=(
                        MAX_RESPONSE_TOKENS
                    ),
                    do_sample=True,
                    temperature=(
                        TEMPERATURE
                    ),
                    top_p=TOP_P,
                    top_k=TOP_K,
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


        prompt_len = (
            prompt_ids
            .shape[1]
        )


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


        if not draft.strip():

            scan_log.append(
                {
                    "index":
                        idx,

                    "status":
                        "empty_student_draft",
                }
            )

            continue


        raw_feedback, parsed = (
            run_feedbacker(
                teacher,
                teacher_tok,
                source,
                draft,
                device,
            )
        )


        valid = valid_errors(
            parsed,
            source,
            draft,
            response_ids,
            student_tok,
        )


        scan_log.append(
            {
                "index":
                    idx,

                "student_translation":
                    draft,

                "feedback_parse_ok":
                    parsed[
                        "parse_ok"
                    ],

                "feedback_has_error":
                    parsed[
                        "has_error"
                    ],

                "reported_errors":
                    len(
                        parsed[
                            "errors"
                        ]
                    ),

                "structurally_valid_errors":
                    len(valid),
            }
        )


        print()
        print(
            f"SCAN index={idx} "
            f"parse_ok={parsed['parse_ok']} "
            f"has_error={parsed['has_error']} "
            f"errors={len(parsed['errors'])} "
            f"valid={len(valid)}"
        )


        if not valid:
            continue


        # Smoke-only deterministic selection.
        # This carries NO algorithmic
        # priority semantics.

        local_rng = random.Random(
            SEED + idx + 999
        )


        selected = (
            local_rng.choice(
                valid
            )
        )


        chosen = {
            "row":
                row,

            "index":
                idx,

            "source":
                source,

            "prompt_ids":
                prompt_ids,

            "prompt_text":
                prompt_text,

            "response_ids":
                response_ids,

            "student_translation":
                draft,

            "raw_feedback":
                raw_feedback,

            "parsed":
                parsed,

            "valid_errors":
                valid,

            "selected":
                selected,
        }

        break


    if chosen is None:

        result = {
            "status":
                "NO_VALID_ERROR_FOUND",

            "scan_rows":
                SCAN_ROWS,

            "scan_log":
                scan_log,
        }


        OUT.write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


        print()
        print(
            "NO_VALID_ERROR_FOUND"
        )

        print(
            "RESULT=",
            OUT,
        )

        return


    idx = chosen[
        "index"
    ]

    selected = chosen[
        "selected"
    ]


    raw_prefix_ids = (
        selected[
            "raw_prefix_ids"
        ]
    )

    correction_ids = (
        selected[
            "correction_ids"
        ]
    )


    corrected_response_prefix_ids = (
        raw_prefix_ids
        + correction_ids
    )


    corrected_input_ids = (
        torch.cat(
            [
                chosen[
                    "prompt_ids"
                ],
                torch.tensor(
                    [
                        corrected_response_prefix_ids
                    ],
                    dtype=torch.long,
                    device=device,
                ),
            ],
            dim=1,
        )
    )


    corrected_prefix_len = (
        corrected_input_ids
        .shape[1]
    )


    corrected_attention = (
        torch.ones_like(
            corrected_input_ids,
            device=device,
        )
    )


    set_seed(
        SEED + idx + 100000
    )


    with torch.inference_mode():

        resumed = student.generate(
            input_ids=(
                corrected_input_ids
            ),
            attention_mask=(
                corrected_attention
            ),
            max_new_tokens=(
                MAX_RESPONSE_TOKENS
            ),
            do_sample=True,
            temperature=(
                TEMPERATURE
            ),
            top_p=TOP_P,
            top_k=TOP_K,
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


    resumed_ids = (
        resumed[
            0,
            corrected_prefix_len:
        ]
        .tolist()
    )


    if not resumed_ids:

        raise RuntimeError(
            "Student produced zero "
            "resumed tokens"
        )


    resumed_text = decode_ids(
        student_tok,
        resumed_ids,
    )


    full_translation = (
        selected[
            "corrected_prefix"
        ]
        + resumed_text
    )


    # ------------------------------------------------------------
    # Exact FKL on resumed Student states only.
    #
    # correction tokens themselves are
    # conditioning context, not loss tokens.
    # ------------------------------------------------------------

    full_ids = resumed


    full_mask = (
        torch.ones_like(
            full_ids,
            device=device,
        )
    )


    with torch.inference_mode():

        student_out = student(
            input_ids=full_ids,
            attention_mask=full_mask,
            use_cache=False,
        )


        teacher_out = teacher(
            input_ids=full_ids,
            attention_mask=full_mask,
            use_cache=False,
        )


    s_logits = (
        student_out.logits[
            :,
            corrected_prefix_len - 1:
            -1,
            :,
        ]
    )


    t_logits = (
        teacher_out.logits[
            :,
            corrected_prefix_len - 1:
            -1,
            :,
        ]
    )


    expected_tokens = (
        full_ids.shape[1]
        - corrected_prefix_len
    )


    if (
        s_logits.shape[1]
        != expected_tokens
    ):
        raise RuntimeError(
            "Student FKL shift mismatch"
        )


    if (
        t_logits.shape
        != s_logits.shape
    ):
        raise RuntimeError(
            "Teacher/student logits "
            "shape mismatch"
        )


    t_logp = F.log_softmax(
        t_logits.float(),
        dim=-1,
    )


    t_prob = (
        t_logp.exp()
    )


    s_logp = F.log_softmax(
        s_logits.float(),
        dim=-1,
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


    loss = (
        token_kl.mean()
    )


    if not torch.isfinite(
        loss
    ):

        raise RuntimeError(
            "Non-finite FKL"
        )


    token_kl_values = (
        token_kl[
            0
        ]
        .cpu()
        .tolist()
    )


    result = {
        "status":
            "PASS",

        "method":
            "EC_ROPD_FORWARD_ONLY_SMOKE_V1",

        "purpose":
            (
                "implementation_validation_only"
            ),

        "optimizer_step":
            False,

        "backward":
            False,

        "student_model":
            str(
                PE_MODEL
            ),

        "teacher_feedbacker_model":
            str(
                TEACHER_MODEL
            ),

        "student_rollout_protocol": {
            "do_sample":
                True,

            "temperature":
                TEMPERATURE,

            "top_p":
                TOP_P,

            "top_k":
                TOP_K,

            "max_new_tokens":
                MAX_RESPONSE_TOKENS,
        },

        "feedbacker_protocol": {
            "do_sample":
                False,

            "max_prompt_tokens":
                FEEDBACK_MAX_PROMPT_TOKENS,

            "max_new_tokens":
                FEEDBACK_MAX_NEW_TOKENS,

            "reference_available":
                False,
        },

        "selection": {
            "index":
                idx,

            "selection_method":
                (
                    "seeded smoke-only "
                    "choice among structurally "
                    "valid error objects"
                ),

            "source":
                chosen[
                    "source"
                ],

            "student_translation":
                chosen[
                    "student_translation"
                ],

            "feedback_parse_ok":
                chosen[
                    "parsed"
                ][
                    "parse_ok"
                ],

            "feedback_has_error":
                chosen[
                    "parsed"
                ][
                    "has_error"
                ],

            "reported_error_count":
                len(
                    chosen[
                        "parsed"
                    ][
                        "errors"
                    ]
                ),

            "valid_error_count":
                len(
                    chosen[
                        "valid_errors"
                    ]
                ),

            "selected_error": {
                k: v
                for k, v
                in selected.items()
                if k not in {
                    "raw_prefix_ids",
                    "correction_ids",
                }
            },
        },

        "state": {
            "prefix_before_error":
                selected[
                    "prefix_before"
                ],

            "correction":
                selected[
                    "correction"
                ],

            "corrected_prefix":
                selected[
                    "corrected_prefix"
                ],

            "original_prefix_tokens":
                len(
                    raw_prefix_ids
                ),

            "correction_tokens":
                len(
                    correction_ids
                ),

            "resumed_tokens":
                len(
                    resumed_ids
                ),

            "resumed_text":
                resumed_text,

            "full_translation":
                full_translation,
        },

        "fkl": {
            "objective":
                (
                    "exact_full_vocab_"
                    "forward_kl"
                ),

            "loss_scope":
                (
                    "student_resumed_"
                    "tokens_only"
                ),

            "correction_tokens_in_loss":
                False,

            "tokens":
                expected_tokens,

            "mean":
                float(
                    loss.cpu()
                ),

            "min":
                float(
                    min(
                        token_kl_values
                    )
                ),

            "max":
                float(
                    max(
                        token_kl_values
                    )
                ),

            "first_16_token_kl":
                token_kl_values[
                    :16
                ],
        },

        "scan_log":
            scan_log,
    }


    OUT.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    print()
    print("=" * 100)
    print(
        "SELECTED ONLINE ERROR STATE"
    )
    print("=" * 100)

    print(
        "INDEX =",
        idx,
    )

    print(
        "SOURCE =",
        chosen[
            "source"
        ],
    )

    print(
        "CURRENT STUDENT =",
        chosen[
            "student_translation"
        ],
    )

    print(
        "ERROR_TYPE =",
        selected[
            "error_type"
        ],
    )

    print(
        "SOURCE_SPAN =",
        repr(
            selected[
                "source_span"
            ]
        ),
    )

    print(
        "TRANSLATION_SPAN =",
        repr(
            selected[
                "translation_span"
            ]
        ),
    )

    print(
        "CORRECTION =",
        repr(
            selected[
                "correction"
            ]
        ),
    )

    print(
        "CORRECTED_PREFIX =",
        repr(
            selected[
                "corrected_prefix"
            ]
        ),
    )

    print(
        "STUDENT_RESUMED =",
        resumed_text,
    )

    print(
        "FULL_RESUMED_TRANSLATION =",
        full_translation,
    )


    print()
    print("=" * 100)
    print(
        "FKL"
    )
    print("=" * 100)

    print(
        "RESUMED_TOKENS =",
        expected_tokens,
    )

    print(
        "MEAN_FKL =",
        float(
            loss.cpu()
        ),
    )

    print(
        "CORRECTION_TOKENS_IN_LOSS = False"
    )


    print()
    print(
        "RESULT =",
        OUT,
    )

    print(
        "EC_ROPD_FORWARD_SMOKE_V1_PASS"
    )


if __name__ == "__main__":
    main()
