#!/usr/bin/env bash
set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

SCRIPT_DIR="$ROOT/scripts/mtpatcher_v14"
EXP_DATA="$DATA_ROOT/$EXP"
EXP_LOG="$LOG_ROOT/$EXP"

MODEL="$MODEL_ROOT/Qwen3-8B"

mkdir -p \
    "$SCRIPT_DIR" \
    "$EXP_DATA/rq3_wa_analogs_v14" \
    "$EXP_DATA/rq3_wa_contexts_v14" \
    "$EXP_LOG/rq3_wa_v14"

###############################################################################
# 1. BUILD ANALOG JOBS
###############################################################################

cat > "$SCRIPT_DIR/build_wa_anchor_jobs_v14.py" <<'PY'
import argparse
import hashlib
import json
import unicodedata
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)

    return h.hexdigest()


def text(x):
    return x.strip() if isinstance(x, str) else ""


def canonical(x):
    x = unicodedata.normalize(
        "NFKC",
        text(x),
    ).casefold()

    return "".join(
        c for c in x
        if unicodedata.category(c)[0] in {"L", "N"}
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--pe", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--audit", required=True)

    args = ap.parse_args()

    pe_path = Path(args.pe)
    out_path = Path(args.output)
    audit_path = Path(args.audit)

    rows = []

    with pe_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    if len(rows) != 3732:
        raise RuntimeError(
            f"Expected frozen PE3732, got {len(rows)}"
        )

    jobs = []

    parent_span_exact = 0
    parent_span_nonexact = 0
    missing_anchor = 0

    examples = []

    for row_pos, row in enumerate(rows):
        errors = row.get("feedback_errors")

        if not isinstance(errors, list) or not errors:
            raise RuntimeError(
                f"Invalid feedback_errors row_pos={row_pos}"
            )

        err = errors[0]

        if not isinstance(err, dict):
            raise RuntimeError(
                f"Invalid first error row_pos={row_pos}"
            )

        source = text(row.get("source"))
        source_span = text(err.get("source_span"))
        correction = text(err.get("correction"))

        if not source_span:
            missing_anchor += 1
            raise RuntimeError(
                f"Missing first-error source_span row_pos={row_pos}"
            )

        exact = (
            canonical(source_span)
            in canonical(source)
        )

        if exact:
            parent_span_exact += 1
        else:
            parent_span_nonexact += 1

            if len(examples) < 30:
                examples.append(
                    {
                        "row_pos":
                            row_pos,

                        "index":
                            row.get("index"),

                        "source":
                            source,

                        "source_span":
                            source_span,

                        "correction":
                            correction,
                    }
                )

        jobs.append(
            {
                "analog_job_id":
                    row_pos,

                "parent_row_pos":
                    row_pos,

                "parent_index":
                    row.get("index"),

                "source":
                    source,

                "student_translation":
                    text(
                        row.get(
                            "student_translation"
                        )
                    ),

                "source_span":
                    source_span,

                "correction":
                    correction,

                "error_type":
                    text(
                        err.get("error_type")
                    ),

                "explanation":
                    text(
                        err.get("explanation")
                    ),

                "parent_span_exact":
                    exact,

                "construction_method":
                    "MT_PATCHER_WA_ANCHOR_V14",
            }
        )

    if len(jobs) != 3732:
        raise RuntimeError(
            f"Expected 3732 WA jobs, got {len(jobs)}"
        )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with out_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in jobs:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    audit = {
        "pe_rows":
            len(rows),

        "wa_anchor_jobs":
            len(jobs),

        "anchor":
            "feedback_errors[0]",

        "parent_span_exact":
            parent_span_exact,

        "parent_span_nonexact":
            parent_span_nonexact,

        "missing_anchor":
            missing_anchor,

        "mismatch_examples":
            examples,

        "pe_sha256":
            sha256(pe_path),

        "jobs_sha256":
            sha256(out_path),

        "protocol":
            "MT_PATCHER_WA_V14",
    }

    audit_path.write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("PE_ROWS =", len(rows))
    print("WA_ANCHOR_JOBS =", len(jobs))
    print(
        "PARENT_SPAN_EXACT =",
        parent_span_exact,
    )
    print(
        "PARENT_SPAN_NONEXACT =",
        parent_span_nonexact,
    )
    print(
        "JOBS_SHA256 =",
        sha256(out_path),
    )

    print("WA_ANCHOR_JOB_BUILD_PASS")


if __name__ == "__main__":
    main()
PY

###############################################################################
# 2. GENERATE ANALOGOUS WORD PAIRS
###############################################################################

cat > "$SCRIPT_DIR/generate_wa_analogs_qwen3_8b_v14.py" <<'PY'
import argparse
import json
import random
import re
from pathlib import Path

import torch
import torch_npu
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


def load_jsonl(path):
    rows = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def load_completed(path):
    done = set()

    if not path.exists():
        return done

    with path.open(
        encoding="utf-8",
        errors="replace",
    ) as f:
        for line in f:
            try:
                row = json.loads(line)
                done.add(int(row["analog_job_id"]))
            except Exception:
                continue

    return done


def extract_json(text):
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

    start = text.find("{")
    end = text.rfind("}")

    if start < 0 or end <= start:
        raise ValueError("JSON object not found")

    return json.loads(
        text[start:end + 1]
    )


def validate(obj, anchor):
    if not isinstance(obj, dict):
        raise ValueError("root not dict")

    out = {}

    seen_src = set()

    for key in ("category", "semantics"):
        arr = obj.get(key)

        if not isinstance(arr, list):
            raise ValueError(
                f"{key} is not list"
            )

        if len(arr) != 2:
            raise ValueError(
                f"{key} requires exactly two pairs"
            )

        clean = []

        for x in arr:
            if not isinstance(x, dict):
                raise ValueError(
                    f"{key} entry not dict"
                )

            src = x.get("source")
            tgt = x.get("target")

            if not isinstance(src, str):
                raise ValueError("source invalid")

            if not isinstance(tgt, str):
                raise ValueError("target invalid")

            src = src.strip()
            tgt = tgt.strip()

            if not src or not tgt:
                raise ValueError(
                    "empty bilingual pair"
                )

            norm = "".join(src.split()).casefold()

            if norm == "".join(
                anchor.split()
            ).casefold():
                raise ValueError(
                    "analog equals original anchor"
                )

            if norm in seen_src:
                raise ValueError(
                    "duplicate analogous source"
                )

            seen_src.add(norm)

            clean.append(
                {
                    "source": src,
                    "target": tgt,
                }
            )

        out[key] = clean

    if len(seen_src) != 4:
        raise ValueError(
            "expected four distinct analogs"
        )

    return out


def build_prompt(row):
    return f"""Assume you are a Chinese-English language expert with broad knowledge and strong associative ability.

A Chinese machine-translation student made an error involving the following Chinese word or phrase P in sentence X.

Associate rare and challenging Chinese words or phrases from exactly two perspectives:

1. Category:
   Words or phrases belonging to the same category or type as P.

2. Semantics:
   Words or phrases that frequently co-occur with P or naturally occur in closely related semantic contexts.

Generate exactly TWO Chinese-English bilingual pairs for Category and exactly TWO for Semantics.

Requirements:
- All four Chinese entries must be different from P and from each other.
- Prefer relatively rare or challenging translation knowledge.
- Each entry should be a word or short phrase, not a full sentence.
- Give a natural English translation for every Chinese entry.
- Return ONLY one JSON object in the exact structure below.
- Do not add markdown or explanations.

{{
  "category": [
    {{"source": "Chinese phrase 1", "target": "English translation 1"}},
    {{"source": "Chinese phrase 2", "target": "English translation 2"}}
  ],
  "semantics": [
    {{"source": "Chinese phrase 3", "target": "English translation 3"}},
    {{"source": "Chinese phrase 4", "target": "English translation 4"}}
  ]
}}

X: {row["source"]}
P: {row["source_span"]}
"""


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--jobs", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True)

    ap.add_argument(
        "--device-id",
        type=int,
        required=True,
    )

    ap.add_argument(
        "--world-size",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )

    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=20260825,
    )

    args = ap.parse_args()

    device_id = args.device_id

    torch.npu.set_device(device_id)

    device = f"npu:{device_id}"

    random.seed(args.seed + device_id)
    torch.manual_seed(args.seed + device_id)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )

    model.to(device)
    model.eval()

    jobs = load_jsonl(args.jobs)

    assigned = [
        x
        for x in jobs
        if int(x["analog_job_id"])
        % args.world_size
        == device_id
    ]

    out_path = Path(args.output)

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed = load_completed(out_path)

    pending = [
        x for x in assigned
        if int(x["analog_job_id"])
        not in completed
    ]

    print(
        f"DEVICE={device_id} "
        f"ASSIGNED={len(assigned)} "
        f"COMPLETED={len(completed)} "
        f"PENDING={len(pending)}",
        flush=True,
    )

    print(
        f"WA_ANALOG_MODEL_READY device={device_id}",
        flush=True,
    )

    generated = 0

    with out_path.open(
        "a",
        encoding="utf-8",
    ) as fout:

        for start in range(
            0,
            len(pending),
            args.batch_size,
        ):
            batch = pending[
                start:start + args.batch_size
            ]

            prompts = [
                build_prompt(row)
                for row in batch
            ]

            chats = [
                [
                    {
                        "role": "user",
                        "content": p,
                    }
                ]
                for p in prompts
            ]

            rendered = [
                tokenizer.apply_chat_template(
                    c,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for c in chats
            ]

            enc = tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )

            enc = {
                k: v.to(device)
                for k, v in enc.items()
            }

            with torch.inference_mode():
                outputs = model.generate(
                    **enc,
                    do_sample=True,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            prompt_len = enc[
                "input_ids"
            ].shape[1]

            for row, output in zip(
                batch,
                outputs,
            ):
                raw = tokenizer.decode(
                    output[prompt_len:],
                    skip_special_tokens=True,
                ).strip()

                parse_ok = False
                parsed = None
                parse_error = ""

                try:
                    parsed = validate(
                        extract_json(raw),
                        row["source_span"],
                    )
                    parse_ok = True
                except Exception as exc:
                    parse_error = (
                        type(exc).__name__
                        + ": "
                        + str(exc)
                    )

                fout.write(
                    json.dumps(
                        {
                            **row,

                            "raw_analogy":
                                raw,

                            "parse_ok":
                                parse_ok,

                            "parse_error":
                                parse_error,

                            "analogs":
                                parsed,

                            "temperature":
                                args.temperature,

                            "analog_model":
                                args.model,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            fout.flush()

            generated += len(batch)

            if (
                generated == len(batch)
                or generated % 80 == 0
                or generated == len(pending)
            ):
                print(
                    f"DEVICE={device_id} "
                    f"GENERATED={generated}/"
                    f"{len(pending)}",
                    flush=True,
                )

    print(
        f"WA_ANALOG_DEVICE_{device_id}_COMPLETE",
        flush=True,
    )


if __name__ == "__main__":
    main()
PY

###############################################################################
# 3. MERGE ANALOGS + BUILD 14928 CONTEXT JOBS
###############################################################################

cat > "$SCRIPT_DIR/merge_wa_analogs_build_context_jobs_v14.py" <<'PY'
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path):
    rows = []

    with path.open(
        encoding="utf-8",
        errors="replace",
    ) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)

    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--merged", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--context-jobs", required=True)

    args = ap.parse_args()

    shard_dir = Path(args.shard_dir)

    rows = []

    seen_job = set()

    for device in range(16):
        p = shard_dir / f"device_{device}.jsonl"

        if not p.exists():
            raise RuntimeError(
                f"Missing analog shard {p}"
            )

        for row in load_jsonl(p):
            jid = int(row["analog_job_id"])

            if jid in seen_job:
                raise RuntimeError(
                    f"Duplicate analog job id {jid}"
                )

            seen_job.add(jid)
            rows.append(row)

    rows.sort(
        key=lambda x: int(
            x["analog_job_id"]
        )
    )

    if len(rows) != 3732:
        raise RuntimeError(
            f"Expected 3732 analog results, "
            f"got {len(rows)}"
        )

    parse_ok = sum(
        bool(x.get("parse_ok"))
        for x in rows
    )

    parse_fail = len(rows) - parse_ok

    merged_path = Path(args.merged)

    with merged_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    if parse_fail:
        print(
            "WA_ANALOG_PARSE_FAILURE "
            f"count={parse_fail}"
        )

        raise RuntimeError(
            "WA analog generation must produce "
            "exactly four valid pairs for all "
            "3732 PE examples before the "
            "paper-budget context stage."
        )

    context_jobs = []

    category_count = 0
    semantics_count = 0

    context_job_id = 0

    for row in rows:
        analogs = row["analogs"]

        for aspect in (
            "category",
            "semantics",
        ):
            for rank, pair in enumerate(
                analogs[aspect]
            ):
                context_jobs.append(
                    {
                        "wa_context_job_id":
                            context_job_id,

                        "analog_job_id":
                            row["analog_job_id"],

                        "parent_row_pos":
                            row["parent_row_pos"],

                        "parent_index":
                            row["parent_index"],

                        "aspect":
                            aspect,

                        "analog_rank":
                            rank,

                        "original_source":
                            row["source"],

                        "original_error_span":
                            row["source_span"],

                        "analog_source":
                            pair["source"],

                        "analog_target":
                            pair["target"],

                        "construction_method":
                            "MT_PATCHER_WA_CONTEXT_JOB_V14",
                    }
                )

                context_job_id += 1

                if aspect == "category":
                    category_count += 1
                else:
                    semantics_count += 1

    if len(context_jobs) != 14928:
        raise RuntimeError(
            f"Expected 14928 WA context jobs, "
            f"got {len(context_jobs)}"
        )

    if category_count != 7464:
        raise RuntimeError(
            f"Category count mismatch "
            f"{category_count}"
        )

    if semantics_count != 7464:
        raise RuntimeError(
            f"Semantics count mismatch "
            f"{semantics_count}"
        )

    context_path = Path(
        args.context_jobs
    )

    with context_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in context_jobs:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    audit = {
        "analog_rows":
            len(rows),

        "parse_ok":
            parse_ok,

        "parse_fail":
            parse_fail,

        "category_pairs":
            category_count,

        "semantics_pairs":
            semantics_count,

        "context_jobs":
            len(context_jobs),

        "contexts_per_pe":
            4,

        "analog_merged_sha256":
            sha256(merged_path),

        "context_jobs_sha256":
            sha256(context_path),

        "protocol":
            "MT_PATCHER_WA_V14_PAPER_BUDGET",
    }

    Path(args.audit).write_text(
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

    print("WA_ANALOG_MERGE_PASS")
    print("WA_CONTEXT_14928_JOB_BUILD_PASS")


if __name__ == "__main__":
    main()
PY

###############################################################################
# 4. GENERATE ONE CONTEXT FOR EACH ANALOG PAIR
###############################################################################

cat > "$SCRIPT_DIR/generate_wa_contexts_qwen3_8b_v14.py" <<'PY'
import argparse
import json
import random
import re
from pathlib import Path

import torch
import torch_npu
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


def load_jsonl(path):
    rows = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def load_completed(path):
    done = set()

    if not path.exists():
        return done

    with path.open(
        encoding="utf-8",
        errors="replace",
    ) as f:
        for line in f:
            try:
                row = json.loads(line)

                done.add(
                    int(
                        row[
                            "wa_context_job_id"
                        ]
                    )
                )
            except Exception:
                continue

    return done


def parse_pair(raw):
    zh = None
    en = None

    for line in raw.splitlines():
        line = line.strip()

        if line.startswith("中文句子:"):
            zh = line.split(
                ":",
                1,
            )[1].strip()

        elif line.startswith("中文句子："):
            zh = line.split(
                "：",
                1,
            )[1].strip()

        elif line.startswith("英文句子:"):
            en = line.split(
                ":",
                1,
            )[1].strip()

        elif line.startswith("英文句子："):
            en = line.split(
                "：",
                1,
            )[1].strip()

    if not zh or not en:
        return None, None

    return zh, en


def prompt(row):
    return f"""You are a Chinese-English parallel-data synthesizer.

Use the ORIGINAL SOURCE only as a loose guide for domain, register and style.

Create ONE new Chinese-English parallel sentence pair containing the given bilingual ANALOG WORD PAIR.

Requirements:
- The new Chinese sentence must naturally use the Chinese analog phrase.
- The English sentence must naturally express its given English translation.
- Preserve approximately the same domain/register/style as the original source.
- The new sentence should describe a different situation or semantic content from the original source.
- Produce fluent natural Chinese and English.
- Do not mention placeholders P or Q.
- Output exactly two lines and nothing else:

中文句子: <new Chinese sentence>
英文句子: <new English translation>

ORIGINAL SOURCE:
{row["original_source"]}

ANALOG WORD PAIR:
Chinese: {row["analog_source"]}
English: {row["analog_target"]}
"""


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--jobs", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True)

    ap.add_argument(
        "--device-id",
        type=int,
        required=True,
    )

    ap.add_argument(
        "--world-size",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=192,
    )

    ap.add_argument(
        "--temperature",
        type=float,
        default=1.5,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=20260825,
    )

    args = ap.parse_args()

    torch.npu.set_device(
        args.device_id
    )

    device = f"npu:{args.device_id}"

    random.seed(
        args.seed + args.device_id
    )

    torch.manual_seed(
        args.seed + args.device_id
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )

    model.to(device)
    model.eval()

    all_jobs = load_jsonl(args.jobs)

    assigned = [
        x
        for x in all_jobs
        if int(
            x["wa_context_job_id"]
        ) % args.world_size
        == args.device_id
    ]

    out_path = Path(args.output)

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed = load_completed(
        out_path
    )

    pending = [
        x for x in assigned
        if int(
            x["wa_context_job_id"]
        ) not in completed
    ]

    print(
        f"DEVICE={args.device_id} "
        f"ASSIGNED={len(assigned)} "
        f"COMPLETED={len(completed)} "
        f"PENDING={len(pending)}",
        flush=True,
    )

    print(
        "WA_CONTEXT_MODEL_READY "
        f"device={args.device_id}",
        flush=True,
    )

    generated = 0

    with out_path.open(
        "a",
        encoding="utf-8",
    ) as fout:

        for start in range(
            0,
            len(pending),
            args.batch_size,
        ):
            batch = pending[
                start:start
                + args.batch_size
            ]

            rendered = []

            for row in batch:
                chat = [
                    {
                        "role": "user",
                        "content": prompt(row),
                    }
                ]

                rendered.append(
                    tokenizer.apply_chat_template(
                        chat,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )

            enc = tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )

            enc = {
                k: v.to(device)
                for k, v in enc.items()
            }

            with torch.inference_mode():
                outputs = model.generate(
                    **enc,
                    do_sample=True,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            prompt_len = enc[
                "input_ids"
            ].shape[1]

            for row, output in zip(
                batch,
                outputs,
            ):
                raw = tokenizer.decode(
                    output[prompt_len:],
                    skip_special_tokens=True,
                ).strip()

                zh, en = parse_pair(raw)

                fout.write(
                    json.dumps(
                        {
                            **row,

                            "raw_generation":
                                raw,

                            "parse_ok":
                                bool(zh and en),

                            "synthesized_source":
                                zh,

                            "synthesized_target":
                                en,

                            "temperature":
                                args.temperature,

                            "synthesis_model":
                                args.model,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            fout.flush()

            generated += len(batch)

            if (
                generated == len(batch)
                or generated % 80 == 0
                or generated == len(pending)
            ):
                print(
                    f"DEVICE={args.device_id} "
                    f"GENERATED={generated}/"
                    f"{len(pending)}",
                    flush=True,
                )

    print(
        f"WA_CONTEXT_DEVICE_"
        f"{args.device_id}_COMPLETE",
        flush=True,
    )


if __name__ == "__main__":
    main()
PY

###############################################################################
# 5. MERGE WA CONTEXTS + BUILD PE+PDS+WA
###############################################################################

cat > "$SCRIPT_DIR/merge_wa_contexts_build_full_v14.py" <<'PY'
import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
import random


def load_jsonl(path):
    rows = []

    with path.open(
        encoding="utf-8",
        errors="replace",
    ) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)

    return h.hexdigest()


def text(x):
    return x.strip() if isinstance(x, str) else ""


def canonical(x):
    x = unicodedata.normalize(
        "NFKC",
        text(x),
    ).casefold()

    return "".join(
        ch
        for ch in x
        if unicodedata.category(ch)[0]
        in {"L", "N"}
    )


def contains_surface(needle, haystack):
    n = canonical(needle)
    h = canonical(haystack)

    return bool(n and h and n in h)


def pair_key(src, tgt):
    return (
        text(src),
        text(tgt),
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--shard-dir",
        required=True,
    )

    ap.add_argument(
        "--existing-pe-pds",
        required=True,
    )

    ap.add_argument(
        "--wa-output",
        required=True,
    )

    ap.add_argument(
        "--wa-audit",
        required=True,
    )

    ap.add_argument(
        "--combined",
        required=True,
    )

    ap.add_argument(
        "--combined-audit",
        required=True,
    )

    args = ap.parse_args()

    shard_dir = Path(
        args.shard_dir
    )

    generated = []

    seen_ids = set()

    for device in range(16):
        p = shard_dir / (
            f"device_{device}.jsonl"
        )

        if not p.exists():
            raise RuntimeError(
                f"Missing WA context shard {p}"
            )

        for row in load_jsonl(p):
            jid = int(
                row[
                    "wa_context_job_id"
                ]
            )

            if jid in seen_ids:
                raise RuntimeError(
                    f"Duplicate context job {jid}"
                )

            seen_ids.add(jid)
            generated.append(row)

    generated.sort(
        key=lambda x: int(
            x["wa_context_job_id"]
        )
    )

    if len(generated) != 14928:
        raise RuntimeError(
            f"Expected 14928 generated "
            f"contexts, got {len(generated)}"
        )

    parse_fail = 0
    duplicate_wa = 0

    quality = Counter()

    seen_pairs = set()

    valid = []

    for row in generated:
        src = text(
            row.get(
                "synthesized_source"
            )
        )

        tgt = text(
            row.get(
                "synthesized_target"
            )
        )

        if (
            not row.get("parse_ok")
            or not src
            or not tgt
        ):
            parse_fail += 1
            continue

        key = pair_key(src, tgt)

        if key in seen_pairs:
            duplicate_wa += 1
            continue

        seen_pairs.add(key)

        p_ok = contains_surface(
            row["analog_source"],
            src,
        )

        q_ok = contains_surface(
            row["analog_target"],
            tgt,
        )

        extended = (
            canonical(src)
            != canonical(
                row["original_source"]
            )
        )

        literal_placeholder = bool(
            re.search(
                r"(^|[\s:：])P"
                r"($|[\s,，。.;；:：])",
                src,
            )
            or re.search(
                r"(^|[\s:：])Q"
                r"($|[\s,，。.;；:：])",
                tgt,
            )
        )

        quality[
            "generated_contains_analog_source"
            if p_ok
            else
            "generated_missing_analog_source"
        ] += 1

        quality[
            "generated_contains_analog_target"
            if q_ok
            else
            "generated_missing_analog_target"
        ] += 1

        quality[
            "source_extended"
            if extended
            else
            "source_not_extended"
        ] += 1

        if literal_placeholder:
            quality[
                "literal_placeholder_suspect"
            ] += 1

        valid.append(
            {
                "index":
                    f"wa_v14_"
                    f"{row['wa_context_job_id']}",

                "source":
                    src,

                "messages":
                    [
                        {
                            "role": "user",
                            "content":
                                "Translate the following text into English "
                                "without additional explanations:\n\n"
                                + src
                                + "\n\n",
                        }
                    ],

                "target_translation":
                    tgt,

                "student_translation":
                    "",

                "feedback_errors":
                    [
                        {
                            "source_span":
                                row[
                                    "analog_source"
                                ],

                            "translation_span":
                                "",

                            "error_type":
                                "WA_"
                                + row[
                                    "aspect"
                                ].upper(),

                            "explanation":
                                "Word-analogy knowledge extension",

                            "correction":
                                row[
                                    "analog_target"
                                ],
                        }
                    ],

                "construction_method":
                    "MT_PATCHER_WA_QWEN3_8B_V14",

                "rq3_data_component":
                    "WA",

                "parent_index":
                    row["parent_index"],

                "parent_row_pos":
                    row["parent_row_pos"],

                "aspect":
                    row["aspect"],

                "analog_rank":
                    row["analog_rank"],

                "quality_audit":
                    {
                        "contains_P":
                            p_ok,

                        "contains_Q":
                            q_ok,

                        "source_extended":
                            extended,

                        "literal_placeholder_suspect":
                            literal_placeholder,
                    },
            }
        )

    wa_output = Path(
        args.wa_output
    )

    with wa_output.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in valid:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    wa_audit = {
        "requested_contexts":
            14928,

        "generated_contexts":
            len(generated),

        "parse_fail":
            parse_fail,

        "duplicate_wa_pair":
            duplicate_wa,

        "wa_after_postprocess":
            len(valid),

        "keep_ratio":
            len(valid) / 14928,

        "quality_flags":
            dict(quality),

        "wa_sha256":
            sha256(wa_output),

        "protocol":
            "MT_PATCHER_WA_V14_PAPER_STYLE",
    }

    Path(
        args.wa_audit
    ).write_text(
        json.dumps(
            wa_audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    ###########################################################################
    # Preserve the frozen PE+PDS-v13 baseline EXACTLY.
    ###########################################################################

    existing_path = Path(
        args.existing_pe_pds
    )

    existing = load_jsonl(
        existing_path
    )

    if len(existing) != 18610:
        raise RuntimeError(
            f"Expected frozen PE+PDS=18610, "
            f"got {len(existing)}"
        )

    existing_pairs = {
        pair_key(
            x["source"],
            x["target_translation"],
        )
        for x in existing
    }

    combined = list(existing)

    wa_overlap_existing = 0
    wa_kept_final = 0

    for row in valid:
        key = pair_key(
            row["source"],
            row["target_translation"],
        )

        if key in existing_pairs:
            wa_overlap_existing += 1
            continue

        existing_pairs.add(key)

        combined.append(row)

        wa_kept_final += 1

    rng = random.Random(
        20260825
    )

    rng.shuffle(combined)

    combined_path = Path(
        args.combined
    )

    with combined_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in combined:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    comp_counts = Counter(
        x.get(
            "rq3_data_component"
        )
        for x in combined
    )

    combined_audit = {
        "frozen_pe_pds_input_rows":
            len(existing),

        "wa_postprocessed":
            len(valid),

        "wa_overlap_existing_removed":
            wa_overlap_existing,

        "wa_kept":
            wa_kept_final,

        "combined_rows":
            len(combined),

        "component_counts":
            dict(comp_counts),

        "existing_pe_pds_sha256":
            sha256(existing_path),

        "wa_sha256":
            sha256(wa_output),

        "combined_sha256":
            sha256(combined_path),

        "protocol":
            "RQ3_PE_PDS_WA_V14",
    }

    Path(
        args.combined_audit
    ).write_text(
        json.dumps(
            combined_audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            wa_audit,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()

    print(
        json.dumps(
            combined_audit,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("WA_CONTEXT_MERGE_PASS")
    print(
        "FROZEN_PE_PDS_V13_PRESERVED_PASS"
    )
    print(
        "RQ3_PE_PDS_WA_V14_DATA_READY"
    )


if __name__ == "__main__":
    main()
PY

###############################################################################
# 6. MASTER RUNNER
###############################################################################

cat > "$SCRIPT_DIR/run_rq3_wa_v14.sh" <<'BASH2'
#!/usr/bin/env bash
set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

SCRIPT_DIR="$ROOT/scripts/mtpatcher_v14"
EXP_DATA="$DATA_ROOT/$EXP"
EXP_LOG="$LOG_ROOT/$EXP"

MODEL="$MODEL_ROOT/Qwen3-8B"

PE="$EXP_DATA/pe_k1_clean3732.jsonl"

PE_PDS="$EXP_DATA/rq3_pe_plus_pds_v13_paperbudget.jsonl"

ANCHOR_JOBS="$EXP_DATA/rq3_wa_anchor_jobs_v14.jsonl"
ANCHOR_AUDIT="$EXP_DATA/rq3_wa_anchor_jobs_v14_audit.json"

ANALOG_DIR="$EXP_DATA/rq3_wa_analogs_v14"
ANALOG_MERGED="$EXP_DATA/rq3_wa_analogs_merged_v14.jsonl"
ANALOG_AUDIT="$EXP_DATA/rq3_wa_analogs_audit_v14.json"

CONTEXT_JOBS="$EXP_DATA/rq3_wa_context_jobs_v14.jsonl"
CONTEXT_DIR="$EXP_DATA/rq3_wa_contexts_v14"

WA_VALID="$EXP_DATA/rq3_wa_valid_v14.jsonl"
WA_AUDIT="$EXP_DATA/rq3_wa_audit_v14.json"

FULL="$EXP_DATA/rq3_pe_pds_wa_v14.jsonl"
FULL_AUDIT="$EXP_DATA/rq3_pe_pds_wa_v14_audit.json"

echo "======================================================================"
echo "RQ3 WA V14 — PAPER-BUDGET KNOWLEDGE EXTENSION"
echo "======================================================================"

###############################################################################
# BUILD 3732 ANALOG ANCHORS
###############################################################################

python \
"$SCRIPT_DIR/build_wa_anchor_jobs_v14.py" \
    --pe "$PE" \
    --output "$ANCHOR_JOBS" \
    --audit "$ANCHOR_AUDIT"

###############################################################################
# ANALOG GENERATION — 16 NPU
###############################################################################

mkdir -p "$ANALOG_DIR"

PIDS=()

for DEVICE in $(seq 0 15); do
    LOG="$EXP_LOG/rq3_wa_v14/analog_device_${DEVICE}.log"

    python \
    "$SCRIPT_DIR/generate_wa_analogs_qwen3_8b_v14.py" \
        --jobs "$ANCHOR_JOBS" \
        --output "$ANALOG_DIR/device_${DEVICE}.jsonl" \
        --model "$MODEL" \
        --device-id "$DEVICE" \
        --world-size 16 \
        --batch-size 8 \
        --max-new-tokens 256 \
        --temperature 1.0 \
        --seed 20260825 \
        > "$LOG" 2>&1 &

    PIDS+=("$!")
done

FAIL=0

for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        FAIL=1
    fi
done

if [ "$FAIL" -ne 0 ]; then
    echo "WA_ANALOG_WORKER_FAILURE"
    false
fi

echo "WA_ALL_16_ANALOG_WORKERS_COMPLETE"

###############################################################################
# MERGE ANALOGS AND REQUIRE EXACT 4×3732
###############################################################################

python \
"$SCRIPT_DIR/merge_wa_analogs_build_context_jobs_v14.py" \
    --shard-dir "$ANALOG_DIR" \
    --merged "$ANALOG_MERGED" \
    --audit "$ANALOG_AUDIT" \
    --context-jobs "$CONTEXT_JOBS"

###############################################################################
# WA → PDS CONTEXT GENERATION — 16 NPU
###############################################################################

mkdir -p "$CONTEXT_DIR"

PIDS=()

for DEVICE in $(seq 0 15); do
    LOG="$EXP_LOG/rq3_wa_v14/context_device_${DEVICE}.log"

    python \
    "$SCRIPT_DIR/generate_wa_contexts_qwen3_8b_v14.py" \
        --jobs "$CONTEXT_JOBS" \
        --output "$CONTEXT_DIR/device_${DEVICE}.jsonl" \
        --model "$MODEL" \
        --device-id "$DEVICE" \
        --world-size 16 \
        --batch-size 8 \
        --max-new-tokens 192 \
        --temperature 1.5 \
        --seed 20260825 \
        > "$LOG" 2>&1 &

    PIDS+=("$!")
done

FAIL=0

for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        FAIL=1
    fi
done

if [ "$FAIL" -ne 0 ]; then
    echo "WA_CONTEXT_WORKER_FAILURE"
    false
fi

echo "WA_ALL_16_CONTEXT_WORKERS_COMPLETE"

###############################################################################
# PAPER-STYLE MERGE + BUILD FULL BASELINE
###############################################################################

python \
"$SCRIPT_DIR/merge_wa_contexts_build_full_v14.py" \
    --shard-dir "$CONTEXT_DIR" \
    --existing-pe-pds "$PE_PDS" \
    --wa-output "$WA_VALID" \
    --wa-audit "$WA_AUDIT" \
    --combined "$FULL" \
    --combined-audit "$FULL_AUDIT"

echo
echo "======================================================================"
echo "FINAL FILES"
echo "======================================================================"

wc -l \
    "$PE" \
    "$PE_PDS" \
    "$WA_VALID" \
    "$FULL"

echo

sha256sum \
    "$WA_VALID" \
    "$FULL"

echo
echo "RQ3_WA_V14_ALL_PASS"
BASH2

chmod +x \
"$SCRIPT_DIR/run_rq3_wa_v14.sh"

###############################################################################
# COMPILE
###############################################################################

python -m py_compile \
    "$SCRIPT_DIR/build_wa_anchor_jobs_v14.py" \
    "$SCRIPT_DIR/generate_wa_analogs_qwen3_8b_v14.py" \
    "$SCRIPT_DIR/merge_wa_analogs_build_context_jobs_v14.py" \
    "$SCRIPT_DIR/generate_wa_contexts_qwen3_8b_v14.py" \
    "$SCRIPT_DIR/merge_wa_contexts_build_full_v14.py"

echo "RQ3_WA_V14_PY_COMPILE_PASS"

###############################################################################
# LAUNCH DETACHED
###############################################################################

MASTER_LOG="$EXP_LOG/rq3_wa_v14_master.log"

nohup setsid bash \
"$SCRIPT_DIR/run_rq3_wa_v14.sh" \
> "$MASTER_LOG" 2>&1 < /dev/null &

PID="$!"

echo "RQ3_WA_V14_STARTED"
echo "PID=$PID"
echo "LOG=$MASTER_LOG"

sleep 15

echo
echo "========== PROCESS =========="

pgrep -af \
'run_rq3_wa_v14|generate_wa_analogs_qwen3_8b_v14|generate_wa_contexts_qwen3_8b_v14' \
|| true

echo
echo "========== MASTER LOG =========="

tail -n 120 "$MASTER_LOG" || true

echo
echo "RQ3_WA_V14_DETACHED_SAFE"
