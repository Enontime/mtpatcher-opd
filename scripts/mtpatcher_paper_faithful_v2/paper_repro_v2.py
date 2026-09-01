#!/usr/bin/env python3

"""
MT-PATCHER — paper-faithful adapted reproduction v2.

Fidelity policy
===============

PAPER-EXACT:
- Section 3.3: random 20k monolingual examples for demonstrations
- Four demonstration tasks:
    feedback
    sentence analysis
    word analogy
    parallel data synthesis
- Appendix B:
    full finetuning
    3 epochs
    learning rate = 1e-5
    global batch size = 64
    response-only next-token loss

REPO-EXACT:
- Prompt strings are imported directly from the official repository.
- Feedback: temperature 0.1, beam_size 1
- Sentence analysis: temperature 0.2, beam_size 1
- Word analogy: temperature 1.0, beam_size 1
- PDS: temperature 1.0, num_case 4, beam_size 1
- Generation cap = 256
- HF release path uses top_p = 0.9
- Translation prompt imported from official repository

ADAPTATION:
- Student: Qwen3-0.6B
- demonstration annotator: Qwen3-8B
- Patcher backbone: Qwen3-8B
- source corpus: WMT NewsCrawl 2023 zh fixed pool
- Qwen chat-template wrapping + enable_thinking=False
- Ascend/NPU execution
- structured parsing adapter for raw feedback;
  raw feedback itself is preserved unchanged and used as SFT target

UNRESOLVED:
- original Baichuan/Llama Base vs Chat checkpoint identity
- release discrepancy concerning multi-error PDS:
  current llama_case_generation.py applies 4 cases PER parsed error;
  an older case_generation/run.py path only keeps prompt_now[0].
  This implementation follows current official data_manager behavior:
  all parsed errors × 4.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
import unicodedata
from pathlib import Path

import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def norm(s):
    return " ".join(
        unicodedata.normalize(
            "NFKC",
            str(s),
        ).split()
    ).strip()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_official(official_root):
    base = (
        Path(official_root)
        / "pipeline"
        / "data_manager"
    )

    translation = load_module(
        "off_translation",
        base / "translation.py",
    )

    feedback = load_module(
        "off_feedback",
        base / "llama_feedback.py",
    )

    analyzer = load_module(
        "off_analyzer",
        base / "llama_sentence_analyzer.py",
    )

    analogy = load_module(
        "off_analogy",
        base / "llama_word_analogy.py",
    )

    casegen = load_module(
        "off_casegen",
        base / "llama_case_generation.py",
    )

    return {
        "translation":
            translation.TranslationDataManager,

        "feedback":
            feedback.FeedbackDataManager,

        "analysis":
            analyzer.SentenceAnalyzerDataManager,

        "analogy":
            analogy.WordAnalogyDataManager,

        "pds":
            casegen.CaseGenerationDataManager,
    }


def fill_lang_prompt(
    prompt,
    srclang="Chinese",
    tgtlang="English",
):
    return (
        prompt
        .replace("<srclang>", srclang)
        .replace("<tgtlang>", tgtlang)
    )


###############################################################################
# §3.3 — random sample
###############################################################################

def cmd_sample(args):
    rows = read_jsonl(args.input)

    if len(rows) < args.n:
        raise RuntimeError(
            f"requested={args.n}, available={len(rows)}"
        )

    rng = random.Random(args.seed)

    selected_positions = rng.sample(
        range(len(rows)),
        args.n,
    )

    out = []

    for demo_id, pos in enumerate(
        selected_positions
    ):
        src = norm(
            rows[pos].get("source", "")
        )

        if not src:
            raise RuntimeError(
                f"empty source position={pos}"
            )

        out.append(
            {
                "demo_id": demo_id,
                "source_pool_position": pos,
                "source": src,
            }
        )

    write_jsonl(
        args.output,
        out,
    )

    print(
        f"PAPER_RANDOM_SAMPLE_ROWS={len(out)}"
    )

    print(
        f"PAPER_RANDOM_SAMPLE_SHA256="
        f"{sha256(args.output)}"
    )

    print(
        "PAPER_RANDOM_SAMPLE_PASS"
    )


###############################################################################
# Official prompt jobs
###############################################################################

def cmd_translation_jobs(args):
    dm = load_official(args.official)[
        "translation"
    ]

    template = dm.prompt

    rows = read_jsonl(args.pool)

    jobs = []

    for x in rows:
        prompt = (
            template
            .replace(
                "<src_text>",
                x["source"],
            )
            .replace(
                "<srclang>",
                "Chinese",
            )
            .replace(
                "<tgtlang>",
                "English",
            )
        )

        jobs.append(
            {
                "job_id":
                    int(x["demo_id"]),

                "demo_id":
                    int(x["demo_id"]),

                "task":
                    "student_translation",

                "source":
                    x["source"],

                "prompt":
                    prompt,

                # Official vLLM default CLI temp = 0.
                # On Qwen/HF we implement this as greedy.
                "temperature":
                    0.0,

                "top_p":
                    1.0,

                "max_new_tokens":
                    256,
            }
        )

    write_jsonl(
        args.output,
        jobs,
    )

    print(
        f"TRANSLATION_JOBS={len(jobs)}"
    )


def cmd_feedback_jobs(args):
    dm = load_official(args.official)[
        "feedback"
    ]

    rows = read_jsonl(
        args.student
    )

    jobs = []

    for x in rows:
        prompt = fill_lang_prompt(
            dm.prompt
        )

        prompt = (
            prompt
            .replace(
                "<srctext>",
                x["source"],
            )
            .replace(
                "<tgttext>",
                x["response"].strip(),
            )
        )

        jobs.append(
            {
                "job_id":
                    int(x["demo_id"]),

                "demo_id":
                    int(x["demo_id"]),

                "task":
                    "feedback",

                "source":
                    x["source"],

                "student_translation":
                    x["response"].strip(),

                "prompt":
                    prompt,

                "temperature":
                    float(dm.temperature),

                "top_p":
                    0.9,

                "max_new_tokens":
                    256,
            }
        )

    write_jsonl(
        args.output,
        jobs,
    )

    print(
        f"FEEDBACK_JOBS={len(jobs)}"
    )


def cmd_analysis_jobs(args):
    dm = load_official(args.official)[
        "analysis"
    ]

    rows = read_jsonl(
        args.student
    )

    jobs = []

    for x in rows:
        prompt = fill_lang_prompt(
            dm.prompt
        )

        prompt = prompt.replace(
            "<src_text>",
            x["source"],
        )

        jobs.append(
            {
                "job_id":
                    int(x["demo_id"]),

                "demo_id":
                    int(x["demo_id"]),

                "task":
                    "sentence_analysis",

                "source":
                    x["source"],

                "prompt":
                    prompt,

                "temperature":
                    float(dm.temperature),

                "top_p":
                    0.9,

                "max_new_tokens":
                    256,
            }
        )

    write_jsonl(
        args.output,
        jobs,
    )

    print(
        f"ANALYSIS_JOBS={len(jobs)}"
    )


###############################################################################
# Engineering parser adapter
#
# Raw Feedback remains unchanged.
# This stage ONLY converts the natural-language Feedback output into the
# structure required by the released downstream CaseGeneration manager.
###############################################################################

PARSER_PROMPT = """You are a deterministic information extractor.

You will receive:
1. a Chinese source sentence;
2. a student's English translation;
3. a translation assessment produced by another model.

Do not independently judge translation quality.
Only extract information explicitly stated in the assessment.

Return exactly one JSON object:

{
  "has_error": true,
  "errors": [
    {
      "error_source": "exact Chinese source span",
      "correction": "the English correction stated in the assessment"
    }
  ],
  "post_edit": "the final good translation stated in the assessment"
}

If the assessment explicitly says there is no error, return:

{
  "has_error": false,
  "errors": [],
  "post_edit": ""
}

Rules:
- error_source must be copied from the Chinese source sentence;
- keep every distinct error mentioned by the assessment;
- do not invent additional errors;
- output JSON only.

Chinese source:
{source}

Student translation:
{student}

Assessment:
{assessment}
"""


def render_parser_prompt(source, student, assessment):
    """
    Render the engineering-only feedback parser prompt.

    Deliberately avoids str.format(), because PARSER_PROMPT contains
    literal JSON braces that must remain literal.
    """
    return (
        PARSER_PROMPT
        .replace("{source}", str(source))
        .replace("{student}", str(student))
        .replace("{assessment}", str(assessment))
    )


def cmd_parser_jobs(args):
    feedback = read_jsonl(
        args.feedback
    )

    jobs = []

    for x in feedback:
        jobs.append(
            {
                "job_id":
                    int(x["demo_id"]),

                "demo_id":
                    int(x["demo_id"]),

                "task":
                    "feedback_structured_extraction",

                "source":
                    x["source"],

                "student_translation":
                    x["student_translation"],

                "assessment":
                    x["response"],

                "prompt":
                    render_parser_prompt(
                        source=x["source"],
                        student=x[
                            "student_translation"
                        ],
                        assessment=x[
                            "response"
                        ],
                    ),

                "temperature":
                    0.0,

                "top_p":
                    1.0,

                "max_new_tokens":
                    256,
            }
        )

    write_jsonl(
        args.output,
        jobs,
    )

    print(
        f"PARSER_JOBS={len(jobs)}"
    )


def parse_json_text(text):
    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.I,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    a = text.find("{")
    b = text.rfind("}")

    if a < 0 or b < a:
        raise ValueError(
            "no json object"
        )

    return json.loads(
        text[a:b + 1]
    )


def cmd_validate_parser(args):
    rows = read_jsonl(
        args.parser
    )

    parsed = []
    parse_fail = 0
    invalid_span = 0
    error_rows = 0
    no_error_rows = 0
    total_errors = 0

    for x in rows:
        try:
            obj = parse_json_text(
                x["response"]
            )

            has_error = bool(
                obj.get(
                    "has_error",
                    False,
                )
            )

            errors = []

            if has_error:
                for e in obj.get(
                    "errors",
                    [],
                ):
                    s = norm(
                        e.get(
                            "error_source",
                            "",
                        )
                    )

                    c = norm(
                        e.get(
                            "correction",
                            "",
                        )
                    )

                    if not s or not c:
                        continue

                    # Hard hallucination check:
                    # the extracted source span must occur in X.
                    if s not in norm(
                        x["source"]
                    ):
                        invalid_span += 1
                        continue

                    errors.append(
                        {
                            "error_source": s,
                            "correction": c,
                        }
                    )

            if has_error and not errors:
                parse_fail += 1
                continue

            if errors:
                error_rows += 1
                total_errors += len(errors)
            else:
                no_error_rows += 1

            parsed.append(
                {
                    "demo_id":
                        int(x["demo_id"]),

                    "source":
                        x["source"],

                    "student_translation":
                        x[
                            "student_translation"
                        ],

                    "assessment":
                        x["assessment"],

                    "has_error":
                        bool(errors),

                    "model_assessment_parsed":
                        errors,

                    "post_edit":
                        str(
                            obj.get(
                                "post_edit",
                                "",
                            )
                        ).strip(),
                }
            )

        except Exception:
            parse_fail += 1

    write_jsonl(
        args.output,
        parsed,
    )

    audit = {
        "input_rows":
            len(rows),

        "valid_rows":
            len(parsed),

        "parse_fail":
            parse_fail,

        "invalid_extracted_source_spans":
            invalid_span,

        "has_error_rows":
            error_rows,

        "no_error_rows":
            no_error_rows,

        "total_errors":
            total_errors,

        "valid_rate":
            (
                len(parsed)
                / max(1, len(rows))
            ),
    }

    Path(
        str(args.output)
        + ".audit.json"
    ).write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        )
    )

    # Engineering adaptation only.
    #
    # The paper and released repository do not specify a minimum
    # structured-feedback parsing rate. Therefore parser validity
    # is recorded as an AUDIT metric and must not be turned into an
    # arbitrary reproduction stopping criterion.
    #
    # Invalid structured rows have already been excluded from
    # downstream WA/PDS inputs above. Raw natural-language Feedback
    # remains untouched and is still retained for Patcher SFT.
    if len(parsed) == 0:
        raise RuntimeError(
            "feedback structured extraction produced zero valid rows"
        )

    print(
        f"FEEDBACK_PARSE_VALID_RATE={audit['valid_rate']:.6f}"
    )

    print(
        f"FEEDBACK_PARSE_DROPPED={audit['input_rows'] - audit['valid_rows']}"
    )

    print(
        "FEEDBACK_PARSE_ADAPTER_PASS"
    )


###############################################################################
# WA jobs — current official release semantics:
# one analogy request for every parsed error.
###############################################################################

def cmd_analogy_jobs(args):
    dm = load_official(args.official)[
        "analogy"
    ]

    rows = read_jsonl(
        args.parsed
    )

    jobs = []

    for x in rows:
        for error_id, e in enumerate(
            x["model_assessment_parsed"]
        ):
            prompt = fill_lang_prompt(
                dm.prompt
            )

            prompt = (
                prompt
                .replace(
                    "<src_text>",
                    x["source"],
                )
                .replace(
                    "<error_word>",
                    e["error_source"],
                )
            )

            jobs.append(
                {
                    "job_id":
                        len(jobs),

                    "demo_id":
                        int(x["demo_id"]),

                    "error_id":
                        error_id,

                    "task":
                        "word_analogy",

                    "source":
                        x["source"],

                    "error_source":
                        e["error_source"],

                    "correction":
                        e["correction"],

                    "prompt":
                        prompt,

                    "temperature":
                        float(dm.temperature),

                    "top_p":
                        0.9,

                    "max_new_tokens":
                        256,
                }
            )

    write_jsonl(
        args.output,
        jobs,
    )

    print(
        f"ANALOGY_JOBS={len(jobs)}"
    )


###############################################################################
# PDS jobs — current official data_manager:
# every parsed error × num_case(4)
###############################################################################

def cmd_pds_jobs(args):
    dm = load_official(args.official)[
        "pds"
    ]

    parsed = {
        int(x["demo_id"]): x
        for x in read_jsonl(
            args.parsed
        )
    }

    analysis = {
        int(x["demo_id"]): x
        for x in read_jsonl(
            args.analysis
        )
    }

    jobs = []

    for demo_id, x in parsed.items():
        if demo_id not in analysis:
            continue

        raw_analysis = analysis[
            demo_id
        ]["response"].strip()

        for error_id, e in enumerate(
            x["model_assessment_parsed"]
        ):
            # PAPER §3.3 / Appendix demonstration construction:
            # one PDS demonstration generates one parallel pair.
            #
            # IMPORTANT:
            # dm.num_case == 4 belongs to the RELEASE Step-4
            # Student patch-generation pipeline and will be used later
            # after the specialized MT-PATCHER has been trained.
            demo_num_case = 1

            for case_id in range(
                demo_num_case
            ):
                prompt = fill_lang_prompt(
                    dm.prompt
                )

                prompt = (
                    prompt
                    .replace(
                        "<domain_topic_style>",
                        raw_analysis,
                    )
                    .replace(
                        "<word_pair>",
                        "{}({})".format(
                            e["error_source"],
                            e["correction"],
                        ),
                    )
                )

                jobs.append(
                    {
                        "job_id":
                            len(jobs),

                        "demo_id":
                            demo_id,

                        "error_id":
                            error_id,

                        "case_id":
                            case_id,

                        "task":
                            "parallel_data_synthesis",

                        "error_source":
                            e["error_source"],

                        "correction":
                            e["correction"],

                        "sentence_analysis":
                            raw_analysis,

                        "prompt":
                            prompt,

                        "temperature":
                            float(dm.temperature),

                        "top_p":
                            0.9,

                        "max_new_tokens":
                            256,
                    }
                )

    write_jsonl(
        args.output,
        jobs,
    )

    print(
        f"PDS_JOBS={len(jobs)}"
    )

    print(
        "PDS_DEMO_NUM_CASE_PER_ERROR=1"
    )

    print(
        f"PDS_RELEASE_STEP4_NUM_CASE_PER_ERROR="
        f"{int(dm.num_case)}"
    )


###############################################################################
# Generic 16-way generation worker
###############################################################################

def qwen_format(
    tokenizer,
    prompt,
):
    """
    ADAPTATION:
    Official prompts are kept byte-for-byte from release after placeholder
    substitution, but Qwen3 requires its own chat serialization.
    """
    return tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def cmd_generate(args):
    torch.npu.set_device(
        f"npu:{args.device}"
    )

    device = torch.device(
        f"npu:{args.device}"
    )

    rows = read_jsonl(
        args.jobs
    )

    assigned = [
        x
        for x in rows
        if int(x["job_id"])
        % args.world_size
        == args.device
    ]

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    done = {}

    if output.exists():
        for x in read_jsonl(output):
            done[int(x["job_id"])] = x

    pending = [
        x
        for x in assigned
        if int(x["job_id"])
        not in done
    ]

    print(
        f"DEVICE={args.device} "
        f"ASSIGNED={len(assigned)} "
        f"DONE={len(done)} "
        f"PENDING={len(pending)}",
        flush=True,
    )

    if not pending:
        print(
            f"DEVICE={args.device} "
            f"GENERATION_WORKER_PASS",
            flush=True,
        )
        return

    tok = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
    )

    tok.padding_side = "left"

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )

    model.to(device)
    model.eval()

    with output.open(
        "a",
        encoding="utf-8",
    ) as fout:

        for start in range(
            0,
            len(pending),
            args.batch_size,
        ):
            batch = pending[
                start:
                start + args.batch_size
            ]

            prompts = [
                qwen_format(
                    tok,
                    x["prompt"],
                )
                for x in batch
            ]

            enc = tok(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )

            enc = {
                k: v.to(
                    device,
                    non_blocking=True,
                )
                for k, v in enc.items()
            }

            width = (
                enc["input_ids"].shape[1]
            )

            # A stage has one task, therefore same decoding config.
            temperature = float(
                batch[0]["temperature"]
            )

            top_p = float(
                batch[0]["top_p"]
            )

            max_new_tokens = int(
                batch[0]["max_new_tokens"]
            )

            generation_kwargs = {
                "max_new_tokens":
                    max_new_tokens,

                "pad_token_id":
                    tok.pad_token_id,

                "eos_token_id":
                    tok.eos_token_id,
            }

            if temperature <= 0:
                # Translation / parser:
                # Qwen-compatible implementation of release vLLM temp=0.
                generation_kwargs[
                    "do_sample"
                ] = False

            else:
                # Mirrors official HF release path.
                generation_kwargs.update(
                    {
                        "do_sample":
                            True,

                        "temperature":
                            temperature,

                        "top_p":
                            top_p,
                    }
                )

            with torch.inference_mode():
                generated = model.generate(
                    **enc,
                    **generation_kwargs,
                )

            new_tokens = generated[
                :,
                width:
            ]

            texts = tok.batch_decode(
                new_tokens,
                skip_special_tokens=True,
            )

            for row, text, ids in zip(
                batch,
                texts,
                new_tokens,
            ):
                z = dict(row)

                z["response"] = (
                    text.strip()
                )

                z["generated_token_slots"] = int(
                    ids.numel()
                )

                fout.write(
                    json.dumps(
                        z,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                fout.flush()

            complete = min(
                start + len(batch),
                len(pending),
            )

            if (
                complete % 200 == 0
                or complete
                == len(pending)
            ):
                print(
                    f"DEVICE={args.device} "
                    f"PROGRESS={complete}/"
                    f"{len(pending)}",
                    flush=True,
                )

    print(
        f"DEVICE={args.device} "
        f"GENERATION_WORKER_PASS",
        flush=True,
    )


###############################################################################
# Merge
###############################################################################

def cmd_merge(args):
    rows = []

    for d in range(
        args.world_size
    ):
        path = (
            Path(args.shard_dir)
            / f"{args.prefix}_{d}.jsonl"
        )

        if not path.exists():
            raise RuntimeError(
                f"missing shard {path}"
            )

        rows.extend(
            read_jsonl(path)
        )

    merged = {}

    for row in rows:
        jid = int(
            row["job_id"]
        )

        if jid in merged:
            if (
                merged[jid]["response"]
                != row["response"]
            ):
                raise RuntimeError(
                    f"non-identical duplicate "
                    f"job_id={jid}"
                )

            continue

        merged[jid] = row

    out = [
        merged[k]
        for k in sorted(
            merged.keys()
        )
    ]

    expected = read_jsonl(
        args.jobs
    )

    if len(out) != len(expected):
        raise RuntimeError(
            f"merged={len(out)}, "
            f"jobs={len(expected)}"
        )

    write_jsonl(
        args.output,
        out,
    )

    print(
        f"MERGED_ROWS={len(out)}"
    )

    print(
        f"MERGED_SHA256="
        f"{sha256(args.output)}"
    )


###############################################################################
# SFT demonstration corpus
###############################################################################

def cmd_build_sft(args):
    feedback = read_jsonl(
        args.feedback
    )

    analysis = read_jsonl(
        args.analysis
    )

    analogy = read_jsonl(
        args.analogy
    )

    pds = read_jsonl(
        args.pds
    )

    combined = []

    def add(rows, task):
        for x in rows:
            response = str(
                x.get(
                    "response",
                    "",
                )
            ).strip()

            if not response:
                continue

            combined.append(
                {
                    "sft_id":
                        len(combined),

                    "task":
                        task,

                    "demo_id":
                        int(x["demo_id"]),

                    "prompt":
                        x["prompt"],

                    "response":
                        response,
                }
            )

    add(
        feedback,
        "feedback",
    )

    add(
        analysis,
        "sentence_analysis",
    )

    add(
        analogy,
        "word_analogy",
    )

    add(
        pds,
        "parallel_data_synthesis",
    )

    rng = random.Random(
        args.seed
    )

    rng.shuffle(
        combined
    )

    # Re-number after shuffle.
    for i, x in enumerate(
        combined
    ):
        x["sft_id"] = i

    write_jsonl(
        args.output,
        combined,
    )

    counts = {}

    for x in combined:
        counts[x["task"]] = (
            counts.get(
                x["task"],
                0,
            )
            + 1
        )

    manifest = {
        "protocol":
            "MT_PATCHER_PAPER_FAITHFUL_ADAPTED_V2",

        "rows":
            len(combined),

        "task_counts":
            counts,

        "sha256":
            sha256(args.output),

        "fidelity": {
            "random_20k_demonstration_protocol":
                "PAPER_EXACT",

            "four_demonstration_tasks":
                "PAPER_EXACT",

            "prompt_text":
                "REPO_EXACT imported from frozen official commit",

            "feedback_temperature_0.1":
                "REPO_EXACT",

            "analysis_temperature_0.2":
                "REPO_EXACT",

            "analogy_temperature_1.0":
                "REPO_EXACT",

            "pds_temperature_1.0":
                "REPO_EXACT",

            "pds_demo_num_case_1":
                "PAPER_APPENDIX_DEMONSTRATION_SEMANTICS",

            "pds_release_step4_num_case_4":
                "REPO_EXACT_CURRENT_DATA_MANAGER; reserved for downstream patch generation",

            "generation_max_new_tokens_256":
                "REPO_EXACT",

            "hf_top_p_0.9":
                "REPO_EXACT",

            "student_model":
                "ADAPTATION: Qwen3-0.6B",

            "demonstration_annotator":
                "ADAPTATION: Qwen3-8B replaces GPT-4",

            "patcher_backbone":
                "ADAPTATION: Qwen3-8B replaces Baichuan2-13B",

            "source_corpus":
                "ADAPTATION: WMT NewsCrawl 2023 zh",

            "qwen_chat_serialization":
                "ADAPTATION",

            "feedback_structured_extraction":
                "ENGINEERING_ADAPTATION; raw feedback is unchanged",

            "base_vs_chat":
                "UNRESOLVED",

            "legacy_vs_current_multi_error_pds":
                "UNRESOLVED_RELEASE_DISCREPANCY; current data_manager chosen",
        },
    }

    Path(
        str(args.output)
        + ".manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "PATCHER_SFT_CORPUS_PASS"
    )


###############################################################################
# CLI
###############################################################################

def main():
    ap = argparse.ArgumentParser()

    sub = ap.add_subparsers(
        dest="cmd",
        required=True,
    )

    p = sub.add_parser(
        "sample"
    )
    p.add_argument(
        "--input",
        required=True,
    )
    p.add_argument(
        "--output",
        required=True,
    )
    p.add_argument(
        "--n",
        required=True,
        type=int,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=20260826,
    )
    p.set_defaults(
        func=cmd_sample
    )

    for name, func in [
        (
            "translation-jobs",
            cmd_translation_jobs,
        ),
        (
            "feedback-jobs",
            cmd_feedback_jobs,
        ),
        (
            "analysis-jobs",
            cmd_analysis_jobs,
        ),
    ]:
        p = sub.add_parser(name)
        p.add_argument(
            "--official",
            required=True,
        )

        if name == "translation-jobs":
            p.add_argument(
                "--pool",
                required=True,
            )
        else:
            p.add_argument(
                "--student",
                required=True,
            )

        p.add_argument(
            "--output",
            required=True,
        )
        p.set_defaults(
            func=func
        )

    p = sub.add_parser(
        "parser-jobs"
    )
    p.add_argument(
        "--feedback",
        required=True,
    )
    p.add_argument(
        "--output",
        required=True,
    )
    p.set_defaults(
        func=cmd_parser_jobs
    )

    p = sub.add_parser(
        "validate-parser"
    )
    p.add_argument(
        "--parser",
        required=True,
    )
    p.add_argument(
        "--output",
        required=True,
    )
    p.set_defaults(
        func=cmd_validate_parser
    )

    p = sub.add_parser(
        "analogy-jobs"
    )
    p.add_argument(
        "--official",
        required=True,
    )
    p.add_argument(
        "--parsed",
        required=True,
    )
    p.add_argument(
        "--output",
        required=True,
    )
    p.set_defaults(
        func=cmd_analogy_jobs
    )

    p = sub.add_parser(
        "pds-jobs"
    )
    p.add_argument(
        "--official",
        required=True,
    )
    p.add_argument(
        "--parsed",
        required=True,
    )
    p.add_argument(
        "--analysis",
        required=True,
    )
    p.add_argument(
        "--output",
        required=True,
    )
    p.set_defaults(
        func=cmd_pds_jobs
    )

    p = sub.add_parser(
        "generate"
    )
    p.add_argument(
        "--jobs",
        required=True,
    )
    p.add_argument(
        "--output",
        required=True,
    )
    p.add_argument(
        "--model",
        required=True,
    )
    p.add_argument(
        "--device",
        required=True,
        type=int,
    )
    p.add_argument(
        "--world-size",
        type=int,
        default=16,
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    p.set_defaults(
        func=cmd_generate
    )

    p = sub.add_parser(
        "merge"
    )
    p.add_argument(
        "--jobs",
        required=True,
    )
    p.add_argument(
        "--shard-dir",
        required=True,
    )
    p.add_argument(
        "--prefix",
        required=True,
    )
    p.add_argument(
        "--output",
        required=True,
    )
    p.add_argument(
        "--world-size",
        type=int,
        default=16,
    )
    p.set_defaults(
        func=cmd_merge
    )

    p = sub.add_parser(
        "build-sft"
    )
    p.add_argument(
        "--feedback",
        required=True,
    )
    p.add_argument(
        "--analysis",
        required=True,
    )
    p.add_argument(
        "--analogy",
        required=True,
    )
    p.add_argument(
        "--pds",
        required=True,
    )
    p.add_argument(
        "--output",
        required=True,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    p.set_defaults(
        func=cmd_build_sft
    )

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
