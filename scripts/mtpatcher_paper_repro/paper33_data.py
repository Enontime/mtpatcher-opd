#!/usr/bin/env python3
"""
MT-PATCHER paper-aligned demonstration construction.

Paper mapping
-------------
§3.1:
    Feedback f = (c, {(s_i, e_i, t_i)}, p)

§3.2:
    Sentence Analyzer -> domain/topic/style
    PDS -> synthesize new (X',Y') containing (s,c)
    WA -> category + semantic/co-occurrence analogies

§3.3:
    randomly sample 20k monolingual sentences
    Student translates
    annotator executes four MT-PATCHER tasks
    collected demonstrations -> MT-PATCHER SFT
"""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

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
        for x in rows:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                ) + "\n"
            )


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for b in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(b)

    return h.hexdigest()


def parse_json_response(text):
    """
    Serialization adaptation only.

    Paper uses natural-language assessment.
    We preserve exactly the semantic fields but request JSON
    so that error spans can be audited deterministically.
    """
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

    p = text.find("{")
    q = text.rfind("}")

    if p < 0 or q < p:
        raise ValueError("no JSON object")

    return json.loads(text[p:q + 1])


# ----------------------------------------------------------------------
# Paper Appendix A semantic-equivalent prompts.
#
# We do not alter the tasks:
# - feedback
# - sentence analysis
# - word analogy
# - parallel data synthesis
# ----------------------------------------------------------------------

def translation_prompt(source):
    return (
        "Translate the following Chinese sentence into English. "
        "Return only the translation, with no explanation.\n\n"
        f"Chinese: {source}"
    )


def feedback_prompt(source, student_translation):
    # Paper §3.1:
    # c = whether there is an error
    # s = source span
    # e = explanation
    # t = correction
    # p = final post-edited translation
    #
    # Appendix Table 7 additionally requires:
    # - identify all errors
    # - smallest possible erroneous spans
    # - concise assessment
    return f"""
You are an expert Chinese-English translation reviewer.

Assess the student's translation below.

Chinese source:
{source}

Student English translation:
{student_translation}

Identify all genuine translation errors.

For each error:
- classify its error type;
- locate the smallest possible corresponding span in the Chinese source;
- locate the corresponding erroneous English span;
- explain why it is wrong;
- provide the correct English translation of that Chinese span.

Avoid labeling an entire sentence as one error when a smaller span can be identified.

Return exactly one JSON object with this schema:

{{
  "has_error": true,
  "overall": "brief overall assessment",
  "errors": [
    {{
      "error_type": "...",
      "source_span": "...",
      "translation_span": "...",
      "reason": "...",
      "correction": "..."
    }}
  ],
  "post_edit": "complete corrected English translation"
}}

If the translation has no genuine translation error, return:

{{
  "has_error": false,
  "overall": "No error.",
  "errors": [],
  "post_edit": "{student_translation}"
}}

Do not output anything outside the JSON object.
""".strip()


def analysis_prompt(source):
    # Paper §3.2 / Appendix Table 8:
    # information bottleneck = domain/topic/style
    return f"""
You are a Chinese-English language expert.

Analyze the Chinese sentence only at the level of:
1. domain,
2. topic,
3. style.

Chinese sentence:
{source}

Return exactly one JSON object:

{{
  "domain": "...",
  "topic": "...",
  "style": "..."
}}

Do not reproduce or paraphrase the full semantic content of the sentence.
Do not output anything outside the JSON object.
""".strip()


def analogy_prompt(source, error_span):
    # Paper §3.2:
    # two association axes:
    #   category
    #   semantics/co-occurrence
    #
    # Appendix prompt constructs several analogous words.
    # Demonstration collection uses 3 per axis.
    # Main MT-PATCHER experiment later uses 2 per axis.
    return f"""
You are a Chinese-English bilingual language expert.

Chinese sentence:
{source}

The student mistranslated this Chinese word or short phrase:
{error_span}

Generate analogous Chinese words or short phrases from two perspectives:

1. Category:
   items belonging to the same meaningful category.

2. Semantics:
   items that frequently occur in similar semantic contexts.

Prefer rare and challenging items that a machine translation system
may plausibly mistranslate.

For each item provide its correct English translation.

Return exactly one JSON object:

{{
  "category": [
    {{"source": "...", "target": "..."}},
    {{"source": "...", "target": "..."}},
    {{"source": "...", "target": "..."}}
  ],
  "semantics": [
    {{"source": "...", "target": "..."}},
    {{"source": "...", "target": "..."}},
    {{"source": "...", "target": "..."}}
  ]
}}

Do not output anything outside the JSON object.
""".strip()


def pds_prompt(
    domain,
    topic,
    style,
    error_span,
    correction,
):
    # Paper §3.2:
    # synthesize a new parallel pair containing (s,c)
    # while preserving domain/topic/style.
    #
    # The original semantic content is deliberately omitted.
    # This is the paper's information bottleneck.
    return f"""
You are a Chinese-English bilingual data synthesizer.

Generate ONE new Chinese-English parallel sentence pair.

Required attributes:
Domain: {domain}
Topic: {topic}
Style: {style}

Required bilingual phrase pair:
Chinese phrase: {error_span}
English translation: {correction}

Requirements:
- the Chinese sentence must naturally contain the Chinese phrase;
- the English sentence must naturally contain its stated English translation;
- preserve the requested domain, topic and style;
- create a genuinely new semantic context;
- the pair must be fluent and mutually faithful.

Return exactly one JSON object:

{{
  "source": "...",
  "target": "..."
}}

Do not output anything outside the JSON object.
""".strip()


# ----------------------------------------------------------------------
# Paper §3.3 line 242-244:
# randomly select 20,000 monolingual sentences.
# ----------------------------------------------------------------------

def cmd_sample(args):
    rows = read_jsonl(args.input)

    if len(rows) < args.n:
        raise RuntimeError(
            f"need {args.n}, have {len(rows)}"
        )

    rng = random.Random(args.seed)

    chosen_idx = rng.sample(
        range(len(rows)),
        args.n,
    )

    out = []

    for demo_id, pos in enumerate(chosen_idx):
        x = rows[pos]

        source = str(
            x.get("source", "")
        ).strip()

        if not source:
            raise RuntimeError(
                f"missing source at position={pos}"
            )

        out.append(
            {
                "demo_id": demo_id,
                "source_pool_position": pos,
                "source": source,
            }
        )

    write_jsonl(
        args.output,
        out,
    )

    print(
        f"PAPER33_RANDOM_MONOLINGUAL_ROWS={len(out)}"
    )
    print(
        f"PAPER33_RANDOM_MONOLINGUAL_SHA256="
        f"{sha256(args.output)}"
    )
    print(
        "PAPER33_RANDOM_SAMPLE_PASS"
    )


# ----------------------------------------------------------------------
# Paper §3.3:
# Student translation jobs.
# ----------------------------------------------------------------------

def cmd_student_jobs(args):
    pool = read_jsonl(args.pool)

    jobs = []

    for x in pool:
        jobs.append(
            {
                "job_id": int(x["demo_id"]),
                "demo_id": int(x["demo_id"]),
                "task": "student_translation",
                "source": x["source"],
                "prompt":
                    translation_prompt(
                        x["source"]
                    ),
            }
        )

    write_jsonl(
        args.output,
        jobs,
    )

    print(
        f"STUDENT_TRANSLATION_JOBS={len(jobs)}"
    )


# ----------------------------------------------------------------------
# Paper §3.3 item (1):
# feedback f given (X,Y)
# ----------------------------------------------------------------------

def cmd_feedback_jobs(args):
    student = read_jsonl(args.student)

    jobs = []

    for x in student:
        jobs.append(
            {
                "job_id": int(x["demo_id"]),
                "demo_id": int(x["demo_id"]),
                "task": "feedback",
                "source": x["source"],
                "student_translation":
                    x["response"].strip(),
                "prompt":
                    feedback_prompt(
                        x["source"],
                        x["response"].strip(),
                    ),
            }
        )

    write_jsonl(
        args.output,
        jobs,
    )

    print(
        f"FEEDBACK_JOBS={len(jobs)}"
    )


# ----------------------------------------------------------------------
# Paper §3.3 item (2):
# analyze domain/topic/style.
# ----------------------------------------------------------------------

def cmd_analysis_jobs(args):
    student = read_jsonl(args.student)

    jobs = []

    for x in student:
        jobs.append(
            {
                "job_id": int(x["demo_id"]),
                "demo_id": int(x["demo_id"]),
                "task": "sentence_analysis",
                "source": x["source"],
                "prompt":
                    analysis_prompt(
                        x["source"]
                    ),
            }
        )

    write_jsonl(
        args.output,
        jobs,
    )

    print(
        f"ANALYSIS_JOBS={len(jobs)}"
    )


# ----------------------------------------------------------------------
# Paper §3.3 item (3):
# make analogies given X and an erroneous word s.
#
# Public code uses the first parsed error for downstream case generation.
# We retain all errors in feedback, but use first error for one
# demonstration of WA/PDS per source.
# ----------------------------------------------------------------------

def first_error_from_feedback(x):
    obj = parse_json_response(
        x["response"]
    )

    if not obj.get("has_error", False):
        return None

    errors = obj.get(
        "errors",
        [],
    )

    if not errors:
        return None

    e = errors[0]

    s = str(
        e.get("source_span", "")
    ).strip()

    c = str(
        e.get("correction", "")
    ).strip()

    if not s or not c:
        return None

    return s, c, obj


def cmd_analogy_jobs(args):
    feedback = read_jsonl(
        args.feedback
    )

    jobs = []
    parse_fail = 0

    for x in feedback:
        try:
            item = first_error_from_feedback(x)
        except Exception:
            parse_fail += 1
            continue

        if item is None:
            continue

        s, c, _ = item

        job_id = len(jobs)

        jobs.append(
            {
                "job_id": job_id,
                "demo_id":
                    int(x["demo_id"]),
                "task":
                    "word_analogy",
                "source":
                    x["source"],
                "error_span":
                    s,
                "correction":
                    c,
                "prompt":
                    analogy_prompt(
                        x["source"],
                        s,
                    ),
            }
        )

    write_jsonl(
        args.output,
        jobs,
    )

    print(
        f"ANALOGY_JOBS={len(jobs)}"
    )
    print(
        f"FEEDBACK_PARSE_FAIL={parse_fail}"
    )


# ----------------------------------------------------------------------
# Paper §3.3 item (4):
# synthesize parallel data containing (s,c)
# and preserving (domain,topic,style).
# ----------------------------------------------------------------------

def cmd_pds_jobs(args):
    feedback = {
        int(x["demo_id"]): x
        for x in read_jsonl(
            args.feedback
        )
    }

    analysis = {
        int(x["demo_id"]): x
        for x in read_jsonl(
            args.analysis
        )
    }

    jobs = []
    parse_fail = 0

    for demo_id, fx in feedback.items():
        if demo_id not in analysis:
            continue

        try:
            item = first_error_from_feedback(
                fx
            )

            if item is None:
                continue

            s, c, _ = item

            a = parse_json_response(
                analysis[demo_id]["response"]
            )

            d = str(
                a["domain"]
            ).strip()

            t = str(
                a["topic"]
            ).strip()

            st = str(
                a["style"]
            ).strip()

        except Exception:
            parse_fail += 1
            continue

        jobs.append(
            {
                "job_id": len(jobs),
                "demo_id": demo_id,
                "task":
                    "parallel_data_synthesis",
                "error_span": s,
                "correction": c,
                "domain": d,
                "topic": t,
                "style": st,
                "prompt":
                    pds_prompt(
                        d,
                        t,
                        st,
                        s,
                        c,
                    ),
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
        f"PDS_INPUT_PARSE_FAIL={parse_fail}"
    )


# ----------------------------------------------------------------------
# Merge deterministic 16-way shards.
# ----------------------------------------------------------------------

def cmd_merge(args):
    rows = []

    shard_dir = Path(
        args.shard_dir
    )

    for i in range(args.world_size):
        p = (
            shard_dir
            / f"{args.prefix}_{i}.jsonl"
        )

        if not p.exists():
            raise RuntimeError(
                f"missing shard={p}"
            )

        rows.extend(
            read_jsonl(p)
        )

    by_id = {}

    for x in rows:
        jid = int(
            x["job_id"]
        )

        if jid in by_id:
            if (
                by_id[jid]["response"]
                != x["response"]
            ):
                raise RuntimeError(
                    f"non-identical duplicate job={jid}"
                )

            continue

        by_id[jid] = x

    merged = [
        by_id[k]
        for k in sorted(by_id)
    ]

    write_jsonl(
        args.output,
        merged,
    )

    print(
        f"MERGED_ROWS={len(merged)}"
    )
    print(
        f"MERGED_SHA256="
        f"{sha256(args.output)}"
    )


# ----------------------------------------------------------------------
# Build the four-task SFT corpus.
#
# Paper Appendix B:
# response-token loss only.
#
# Here we store prompt and response separately;
# training code masks the entire prompt.
# ----------------------------------------------------------------------

def cmd_build_sft(args):
    task_files = [
        ("feedback", args.feedback),
        ("sentence_analysis", args.analysis),
        ("word_analogy", args.analogy),
        ("parallel_data_synthesis", args.pds),
    ]

    out = []
    counts = {}
    parse_fail = {}

    for task, path in task_files:
        rows = read_jsonl(path)

        kept = 0
        failed = 0

        for x in rows:
            try:
                parse_json_response(
                    x["response"]
                )
            except Exception:
                failed += 1
                continue

            out.append(
                {
                    "sft_id":
                        len(out),

                    "task":
                        task,

                    "demo_id":
                        int(
                            x["demo_id"]
                        ),

                    "prompt":
                        x["prompt"],

                    "response":
                        x["response"].strip(),
                }
            )

            kept += 1

        counts[task] = kept
        parse_fail[task] = failed

    random.Random(
        args.seed
    ).shuffle(out)

    write_jsonl(
        args.output,
        out,
    )

    manifest = {
        "protocol":
            "MT_PATCHER_PAPER33_SPECIALIZATION_V1",

        "paper_correspondence": {
            "feedback":
                "Section 3.1 and Section 3.3 item 1",

            "sentence_analysis":
                "Section 3.2 and Section 3.3 item 2",

            "word_analogy":
                "Section 3.2 and Section 3.3 item 3",

            "parallel_data_synthesis":
                "Section 3.2 and Section 3.3 item 4",

            "training":
                "Appendix B: full FT, 3 epochs, lr=1e-5, batch=64, response-only loss",
        },

        "adaptations": {
            "student":
                "Qwen3-0.6B",

            "paper_student":
                "NLLB-3.3B / ParroT-7B",

            "demonstration_annotator":
                "Qwen3-8B",

            "paper_demonstration_annotator":
                "GPT-4",

            "patcher_backbone":
                "Qwen3-8B",

            "paper_patcher_backbone":
                "Baichuan2-13B for zh-en",

            "feedback_serialization":
                "JSON encoding of the paper's c,{s,e,t},p semantics",
        },

        "rows":
            len(out),

        "task_counts":
            counts,

        "parse_fail":
            parse_fail,

        "sha256":
            sha256(args.output),
    }

    manifest_path = (
        str(args.output)
        + ".manifest.json"
    )

    Path(
        manifest_path
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
        "PAPER33_PATCHER_SFT_DATA_PASS"
    )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(
        dest="cmd",
        required=True,
    )

    p = sub.add_parser("sample")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, default=20260826)
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("student-jobs")
    p.add_argument("--pool", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_student_jobs)

    p = sub.add_parser("feedback-jobs")
    p.add_argument("--student", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_feedback_jobs)

    p = sub.add_parser("analysis-jobs")
    p.add_argument("--student", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_analysis_jobs)

    p = sub.add_parser("analogy-jobs")
    p.add_argument("--feedback", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_analogy_jobs)

    p = sub.add_parser("pds-jobs")
    p.add_argument("--feedback", required=True)
    p.add_argument("--analysis", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_pds_jobs)

    p = sub.add_parser("merge")
    p.add_argument("--shard-dir", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--world-size", type=int, default=16)
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("build-sft")
    p.add_argument("--feedback", required=True)
    p.add_argument("--analysis", required=True)
    p.add_argument("--analogy", required=True)
    p.add_argument("--pds", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_build_sft)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
