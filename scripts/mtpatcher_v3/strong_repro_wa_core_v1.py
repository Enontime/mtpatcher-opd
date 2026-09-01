import argparse
import ast
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path


PROMPT_PREFIX = (
    "Translate the following text into English "
    "without additional explanations:\n\n"
)


def load_jsonl(path):
    rows = []
    path = Path(path)

    with path.open(
        encoding="utf-8-sig",
        errors="replace",
    ) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(
                    f"{path}:{n}: {e}"
                )

    return rows


def dump_jsonl(path, rows):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for x in rows:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )


def norm(x):
    return " ".join(
        str(x).strip().split()
    )


def get_raw(row):
    for key in (
        "raw_generation",
        "response",
        "generation",
        "output",
        "word_analogy",
        "synthesized_case",
    ):
        value = row.get(key)

        if (
            isinstance(value, str)
            and value.strip()
        ):
            return value

    return ""


def merge_shards(shard_dir):
    shard_dir = Path(shard_dir)

    paths = sorted(
        shard_dir.glob(
            "device_*.jsonl"
        )
    )

    if len(paths) != 16:
        raise RuntimeError(
            f"expected16 shards got={len(paths)}"
        )

    by_id = {}
    physical = 0
    duplicates = 0

    for path in paths:
        for row in load_jsonl(path):
            physical += 1

            jid = int(
                row["job_id"]
            )

            if jid in by_id:
                duplicates += 1
                continue

            by_id[jid] = row

    return by_id, physical, duplicates


###############################################################################
# 1. BUILD WORD-ANALOGY JOBS
###############################################################################

def cmd_build_analog_jobs(args):
    k1 = load_jsonl(
        args.k1
    )

    analyses = {
        int(x["parent_index"]):
            x["sentence_analysis"]
        for x in load_jsonl(
            args.analysis
        )
    }

    if len(k1) != 11792:
        raise RuntimeError(
            f"expected frozen K1=11792 got={len(k1)}"
        )

    sys.path.insert(
        0,
        str(Path(args.official)),
    )

    from pipeline.data_manager.llama_word_analogy import (
        WordAnalogyDataManager,
    )

    dm = WordAnalogyDataManager

    if float(dm.temperature) != 1.0:
        raise RuntimeError(
            f"WA repo temperature expected1.0 got={dm.temperature}"
        )

    if int(dm.beam_size) != 1:
        raise RuntimeError(
            f"WA repo beam_size expected1 got={dm.beam_size}"
        )

    repo_prompt = str(
        dm.prompt
    )

    ###########################################################################
    # SPEC ASSERTION.
    #
    # Paper main experiment / Table-1-aligned treatment: 2 per aspect.
    # Released current prompt says 3 per aspect.
    ###########################################################################

    needle = (
        "generate three words similar to X for each aspect"
    )

    needle_count = repo_prompt.count(
        needle
    )

    print(
        "WA_CARDINALITY_REPAIR_NEEDLE_COUNT =",
        needle_count,
    )

    if needle_count != 1:
        raise RuntimeError(
            "WA cardinality repair target string "
            "not found exactly once"
        )

    repo_prompt = repo_prompt.replace(
        needle,
        "generate two words similar to X for each aspect",
        1,
    )

    if needle in repo_prompt:
        raise RuntimeError(
            "WA 3->2 cardinality repair did not eliminate old instruction"
        )

    if (
        "generate two words similar to X for each aspect"
        not in repo_prompt
    ):
        raise RuntimeError(
            "WA 2-per-aspect instruction absent after repair"
        )

    ###########################################################################
    # Output-format adaptation ONLY.
    ###########################################################################

    json_instruction = r"""
Return ONLY one valid JSON object with this schema:
{
  "category": [
    {"source": "<Chinese analogous word/phrase 1>", "target": "<English translation 1>"},
    {"source": "<Chinese analogous word/phrase 2>", "target": "<English translation 2>"}
  ],
  "semantics": [
    {"source": "<Chinese co-occurring/semantically associated word/phrase 1>", "target": "<English translation 1>"},
    {"source": "<Chinese co-occurring/semantically associated word/phrase 2>", "target": "<English translation 2>"}
  ]
}
Do not output explanations outside the JSON.
"""

    if "[/INST]" in repo_prompt:
        before, after = repo_prompt.rsplit(
            "[/INST]",
            1,
        )

        repo_prompt = (
            before
            + "\n"
            + json_instruction
            + "\n[/INST]"
            + after
        )
    else:
        repo_prompt += (
            "\n"
            + json_instruction
        )

    eligible = []

    counters = Counter()

    for row in k1:
        parent_index = int(
            row["index"]
        )

        if parent_index not in analyses:
            counters[
                "NO_CANONICAL_ANALYSIS"
            ] += 1
            continue

        errors = row.get(
            "feedback_errors"
        )

        if (
            not isinstance(errors, list)
            or not errors
        ):
            counters[
                "NO_FEEDBACK_ERROR"
            ] += 1
            continue

        first_error = errors[0]

        if not isinstance(
            first_error,
            dict,
        ):
            counters[
                "FIRST_ERROR_NOT_DICT"
            ] += 1
            continue

        source = str(
            row["source"]
        ).strip()

        source_span = str(
            first_error.get(
                "source_span",
                "",
            )
        ).strip()

        correction = str(
            first_error.get(
                "correction",
                "",
            )
        ).strip()

        if not source or not source_span:
            counters[
                "EMPTY_SOURCE_OR_ANCHOR"
            ] += 1
            continue

        eligible.append({
            "parent_index":
                parent_index,

            "source":
                source,

            "source_span":
                source_span,

            "correction":
                correction,

            "error_type":
                str(
                    first_error.get(
                        "error_type",
                        "",
                    )
                ),

            "sentence_analysis":
                analyses[
                    parent_index
                ],
        })

    if args.sample_size > 0:
        if len(eligible) < args.sample_size:
            raise RuntimeError(
                f"eligible={len(eligible)} < sample={args.sample_size}"
            )

        rng = random.Random(
            args.seed
        )

        selected = rng.sample(
            eligible,
            args.sample_size,
        )
    else:
        selected = eligible

    jobs = []

    span_exact = 0

    for row in selected:
        if (
            row["source_span"]
            in
            row["source"]
        ):
            span_exact += 1

        prompt = (
            repo_prompt
            .replace(
                "<srclang>",
                "Chinese",
            )
            .replace(
                "<tgtlang>",
                "English",
            )
            .replace(
                "<src_text>",
                row["source"],
            )
            .replace(
                "<error_word>",
                row["source_span"],
            )
        )

        jobs.append({
            "job_id":
                len(jobs),

            **row,

            "prompt":
                prompt,

            "wa_anchor_policy":
                "FIRST_ERROR_PER_PE_PARENT_ADAPTATION",

            "wa_cardinality":
                "PAPER_MAIN_EXPERIMENT_2_CATEGORY_PLUS_2_SEMANTIC",
        })

    dump_jsonl(
        args.output,
        jobs,
    )

    report = {
        "K1_rows":
            len(k1),

        "eligible_parents":
            len(eligible),

        "sample_size":
            len(jobs),

        "seed":
            args.seed,

        "source_span_exact_in_parent":
            span_exact,

        "skip_counts":
            dict(counters),

        "repo_WA_temperature":
            float(dm.temperature),

        "repo_WA_beam_size":
            int(dm.beam_size),

        "cardinality_repair_needle_count":
            needle_count,

        "cardinality":
            {
                "category":
                    2,

                "semantics":
                    2,

                "contexts_per_analog":
                    1,
            },

        "claim_classification":
            {
                "2_plus_2":
                    "PAPER_MAIN_EXPERIMENT_CARDINALITY",

                "category_semantics_axes":
                    "PAPER_AND_RELEASED_PROMPT_SEMANTICS",

                "first_error_anchor":
                    "ADAPTATION_HISTORICAL_CONVENTION",

                "json_output":
                    "OUTPUT_FORMAT_ADAPTATION",
            },
    }

    Path(
        args.report
    ).write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "WA_ELIGIBLE_PARENTS =",
        len(eligible),
    )

    print(
        "WA_SELECTED_ANCHORS =",
        len(jobs),
    )

    print(
        "WA_ANCHOR_SPAN_EXACT_IN_PARENT =",
        span_exact,
    )

    print(
        "WA_REPO_TEMPERATURE =",
        dm.temperature,
    )

    print(
        "WA_REPO_BEAM_SIZE =",
        dm.beam_size,
    )

    print(
        "WA_2PLUS2_CARDINALITY_ASSERTION_PASS"
    )

    print(
        "BUILD_ANALOG_JOBS_PASS"
    )


###############################################################################
# 2. PARSE ANALOG OUTPUTS
###############################################################################

def extract_json(text):
    text = str(text).strip()

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

    decoder = json.JSONDecoder()

    for i, ch in enumerate(text):
        if ch != "{":
            continue

        try:
            obj, _ = decoder.raw_decode(
                text[i:]
            )
        except Exception:
            continue

        if isinstance(obj, dict):
            return obj

    return None


def parse_pair_item(item):
    if isinstance(item, dict):
        source = ""
        target = ""

        for key in (
            "source",
            "src",
            "chinese",
            "source_word",
            "word",
        ):
            value = item.get(key)

            if value:
                source = str(
                    value
                ).strip()
                break

        for key in (
            "target",
            "tgt",
            "english",
            "translation",
            "target_word",
        ):
            value = item.get(key)

            if value:
                target = str(
                    value
                ).strip()
                break

        if source and target:
            return source, target

    elif isinstance(
        item,
        (list, tuple),
    ):
        if len(item) >= 2:
            source = str(
                item[0]
            ).strip()

            target = str(
                item[1]
            ).strip()

            if source and target:
                return source, target

    return None


def parse_analogy(raw):
    obj = extract_json(
        raw
    )

    if not obj:
        return None

    category = (
        obj.get("category")
        or
        obj.get("categories")
        or
        []
    )

    semantics = (
        obj.get("semantics")
        or
        obj.get("semantic")
        or
        obj.get("cooccurrence")
        or
        obj.get("context")
        or
        []
    )

    out = {
        "category": [],
        "semantics": [],
    }

    for aspect, values in (
        ("category", category),
        ("semantics", semantics),
    ):
        if not isinstance(
            values,
            list,
        ):
            continue

        for item in values:
            parsed = parse_pair_item(
                item
            )

            if parsed:
                out[
                    aspect
                ].append(
                    parsed
                )

    return out


def cmd_parse_analogs(args):
    jobs = {
        int(x["job_id"]): x
        for x in load_jsonl(
            args.jobs
        )
    }

    (
        results,
        physical,
        duplicates,
    ) = merge_shards(
        args.shards
    )

    missing = (
        set(jobs)
        -
        set(results)
    )

    if missing:
        raise RuntimeError(
            f"analogy result missing={len(missing)} "
            f"first={sorted(missing)[:20]}"
        )

    json_ok = 0
    both_aspects = 0
    complete4 = 0

    pairs = []

    aspect_counts = Counter()

    duplicate_source_removed = 0
    original_source_removed = 0

    for jid in sorted(jobs):
        job = jobs[jid]

        raw = get_raw(
            results[jid]
        )

        parsed = parse_analogy(
            raw
        )

        if parsed is None:
            continue

        json_ok += 1

        if (
            parsed["category"]
            and
            parsed["semantics"]
        ):
            both_aspects += 1

        selected = []

        seen = set()

        original_norm = (
            norm(
                job["source_span"]
            )
            .casefold()
        )

        for aspect in (
            "category",
            "semantics",
        ):
            kept = 0

            for source, target in parsed[
                aspect
            ]:
                if kept >= 2:
                    break

                source = source.strip()
                target = target.strip()

                source_norm = (
                    norm(source)
                    .casefold()
                )

                if not source_norm:
                    continue

                if (
                    source_norm
                    ==
                    original_norm
                ):
                    original_source_removed += 1
                    continue

                if source_norm in seen:
                    duplicate_source_removed += 1
                    continue

                seen.add(
                    source_norm
                )

                selected.append(
                    (
                        aspect,
                        source,
                        target,
                    )
                )

                kept += 1

                aspect_counts[
                    aspect
                ] += 1

        if len(selected) == 4:
            complete4 += 1

        for aspect, source, target in selected:
            pairs.append({
                "analog_pair_id":
                    len(pairs),

                "analog_job_id":
                    jid,

                "parent_index":
                    int(
                        job[
                            "parent_index"
                        ]
                    ),

                "original_source":
                    job["source"],

                "original_error_span":
                    job[
                        "source_span"
                    ],

                "sentence_analysis":
                    job[
                        "sentence_analysis"
                    ],

                "aspect":
                    aspect,

                "analog_source":
                    source,

                "analog_target":
                    target,
            })

    dump_jsonl(
        args.output,
        pairs,
    )

    n = len(jobs)

    report = {
        "anchor_jobs":
            n,

        "physical_results":
            physical,

        "duplicate_physical_results":
            duplicates,

        "json_object_ok":
            json_ok,

        "json_object_rate":
            json_ok / n,

        "both_aspects_nonempty":
            both_aspects,

        "both_aspects_rate":
            both_aspects / n,

        "anchors_complete_4":
            complete4,

        "complete_4_rate":
            complete4 / n,

        "valid_analog_pairs":
            len(pairs),

        "mean_valid_pairs_per_anchor":
            len(pairs) / n,

        "aspect_counts":
            dict(aspect_counts),

        "original_anchor_repeats_removed":
            original_source_removed,

        "within_anchor_duplicate_source_removed":
            duplicate_source_removed,
    }

    Path(
        args.report
    ).write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "ANALOG_JSON_OK =",
        f"{json_ok}/{n}",
    )

    print(
        "ANALOG_JSON_RATE =",
        f"{json_ok/n:.6f}",
    )

    print(
        "ANALOG_BOTH_ASPECTS =",
        f"{both_aspects}/{n}",
    )

    print(
        "ANALOG_BOTH_ASPECT_RATE =",
        f"{both_aspects/n:.6f}",
    )

    print(
        "ANALOG_COMPLETE4 =",
        f"{complete4}/{n}",
    )

    print(
        "ANALOG_VALID_PAIRS =",
        len(pairs),
    )

    print(
        "ANALOG_MEAN_PAIRS_PER_ANCHOR =",
        f"{len(pairs)/n:.6f}",
    )

    print(
        "ANALOG_ASPECT_COUNTS =",
        dict(aspect_counts),
    )

    if not pairs:
        raise RuntimeError(
            "zero valid analog pairs"
        )

    print(
        "PARSE_ANALOGS_PASS"
    )


###############################################################################
# 3. BUILD 1 CONTEXT JOB / ANALOG
###############################################################################

def cmd_build_context_jobs(args):
    analogs = load_jsonl(
        args.analogs
    )

    sys.path.insert(
        0,
        str(Path(args.official)),
    )

    from pipeline.data_manager.llama_case_generation import (
        CaseGenerationDataManager,
    )

    dm = CaseGenerationDataManager

    if float(dm.temperature) != 1.0:
        raise RuntimeError(
            f"CaseGeneration temperature expected1.0 got={dm.temperature}"
        )

    if int(dm.beam_size) != 1:
        raise RuntimeError(
            f"CaseGeneration beam expected1 got={dm.beam_size}"
        )

    prompt_template = str(
        dm.prompt
    )

    jobs = []

    for row in analogs:
        P = str(
            row["analog_source"]
        ).strip()

        Q = str(
            row["analog_target"]
        ).strip()

        analysis = str(
            row[
                "sentence_analysis"
            ]
        ).strip()

        prompt = (
            prompt_template
            .replace(
                "<domain_topic_style>",
                analysis,
            )
            .replace(
                "<word_pair>",
                f"{P}({Q})",
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

        jobs.append({
            "job_id":
                len(jobs),

            **row,

            "prompt":
                prompt,

            "contexts_per_analog":
                1,
        })

    dump_jsonl(
        args.output,
        jobs,
    )

    print(
        "WA_CONTEXT_JOBS =",
        len(jobs),
    )

    print(
        "WA_CONTEXTS_PER_ANALOG = 1"
    )

    print(
        "CASEGEN_REPO_TEMPERATURE =",
        dm.temperature,
    )

    if not jobs:
        raise RuntimeError(
            "zero context jobs"
        )

    print(
        "BUILD_CONTEXT_JOBS_PASS"
    )


###############################################################################
# 4. STRUCTURAL ACCEPTANCE + SFT SCHEMA
###############################################################################

def load_frozen_parser(path):
    path = Path(path)

    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        )
    )

    wanted_assign = {
        "ZH_LABEL",
        "EN_LABEL",
        "META_HEAD",
    }

    body = []

    for node in tree.body:
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
            ),
        ):
            body.append(node)

        elif isinstance(
            node,
            ast.Assign,
        ):
            names = {
                target.id
                for target
                in node.targets
                if isinstance(
                    target,
                    ast.Name,
                )
            }

            if names & wanted_assign:
                body.append(node)

    ns = {}

    module = ast.Module(
        body=body,
        type_ignores=[],
    )

    exec(
        compile(
            module,
            str(path),
            "exec",
        ),
        ns,
        ns,
    )

    if (
        "parse_pair"
        not in ns
        or
        "norm"
        not in ns
    ):
        raise RuntimeError(
            "frozen structural parser functions unavailable"
        )

    return (
        ns["parse_pair"],
        ns["norm"],
    )


def cmd_accept_contexts(args):
    jobs = {
        int(x["job_id"]): x
        for x in load_jsonl(
            args.jobs
        )
    }

    (
        results,
        physical,
        duplicates,
    ) = merge_shards(
        args.shards
    )

    missing = (
        set(jobs)
        -
        set(results)
    )

    if missing:
        raise RuntimeError(
            f"context result missing={len(missing)} "
            f"first={sorted(missing)[:20]}"
        )

    (
        parse_pair,
        parser_norm,
    ) = load_frozen_parser(
        args.parser
    )

    accepted = []

    rejects = Counter()
    methods = Counter()

    q_exact = 0

    for jid in sorted(jobs):
        job = jobs[jid]

        P = str(
            job[
                "analog_source"
            ]
        ).strip()

        Q = str(
            job[
                "analog_target"
            ]
        ).strip()

        parent = str(
            job[
                "original_source"
            ]
        ).strip()

        raw = get_raw(
            results[jid]
        )

        parsed = parse_pair(
            raw,
            P,
        )

        if (
            parsed["status"]
            !=
            "PARSEABLE"
        ):
            rejects[
                parsed["status"]
            ] += 1
            continue

        X = str(
            parsed[
                "X_prime"
            ]
        ).strip()

        Y = str(
            parsed[
                "Y_prime"
            ]
        ).strip()

        methods[
            parsed["method"]
        ] += 1

        p_in_x = (
            bool(parser_norm(P))
            and
            parser_norm(P)
            in
            parser_norm(X)
        )

        parent_copy = (
            bool(parser_norm(parent))
            and
            parser_norm(parent)
            in
            parser_norm(X)
        )

        q_in_y = (
            bool(parser_norm(Q))
            and
            parser_norm(Q)
            in
            parser_norm(Y)
        )

        if q_in_y:
            q_exact += 1

        if parent_copy:
            rejects[
                "PARENT_COPY_X"
            ] += 1
            continue

        if not p_in_x:
            rejects[
                "ANALOG_SOURCE_MISSING_X"
            ] += 1
            continue

        accepted.append({
            "job_id":
                jid,

            "parent_index":
                int(
                    job[
                        "parent_index"
                    ]
                ),

            "analog_pair_id":
                int(
                    job[
                        "analog_pair_id"
                    ]
                ),

            "aspect":
                job["aspect"],

            "source":
                X,

            "target_translation":
                Y,

            "analog_source":
                P,

            "analog_target":
                Q,

            "Q_exact_in_Y_DIAGNOSTIC_ONLY":
                q_in_y,

            "parse_method":
                parsed[
                    "method"
                ],
        })

    dump_jsonl(
        args.accepted,
        accepted,
    )

    sft_rows = []

    for row in accepted:
        src = row["source"]
        tgt = row[
            "target_translation"
        ]

        sft_rows.append({
            "index":
                f"wa_smoke_{row['job_id']}",

            "source":
                src,

            "messages":
                [
                    {
                        "role":
                            "user",

                        "content":
                            PROMPT_PREFIX
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
                            (
                                "WORD_ANALOGY_"
                                + row[
                                    "aspect"
                                ].upper()
                            ),

                        "explanation":
                            "Word-analogy knowledge extension",

                        "correction":
                            row[
                                "analog_target"
                            ],
                    }
                ],

            "construction_method":
                "STRONG_REPRO_WA_SMOKE128_V1",

            "parent_index":
                row[
                    "parent_index"
                ],

            "wa_aspect":
                row[
                    "aspect"
                ],
        })

    dump_jsonl(
        args.sft,
        sft_rows,
    )

    n = len(jobs)

    report = {
        "context_jobs":
            n,

        "physical_results":
            physical,

        "duplicate_physical_results":
            duplicates,

        "accepted":
            len(accepted),

        "accept_rate":
            len(accepted) / n,

        "reject_counts":
            dict(rejects),

        "parse_methods":
            dict(methods),

        "Q_exact_diagnostic_count":
            q_exact,

        "Q_exact_diagnostic_rate":
            q_exact / n,

        "SFT_rows":
            len(sft_rows),
    }

    Path(
        args.report
    ).write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "WA_CONTEXT_RESULTS =",
        f"{physical}/{n}",
    )

    print(
        "WA_STRUCTURAL_ACCEPTED =",
        len(accepted),
    )

    print(
        "WA_STRUCTURAL_ACCEPT_RATE =",
        f"{len(accepted)/n:.6f}",
    )

    print(
        "WA_STRUCTURAL_REJECTS =",
        dict(rejects),
    )

    print(
        "WA_Q_EXACT_RATE_DIAGNOSTIC =",
        f"{q_exact/n:.6f}",
    )

    print(
        "WA_SFT_ROWS =",
        len(sft_rows),
    )

    if not accepted:
        raise RuntimeError(
            "zero accepted WA contexts"
        )

    print(
        "ACCEPT_CONTEXTS_PASS"
    )


###############################################################################
# 5. SFT SCHEMA / TOKENIZATION SMOKE
###############################################################################

def cmd_validate_sft(args):
    rows = load_jsonl(
        args.sft
    )

    if not rows:
        raise RuntimeError(
            "no SFT rows"
        )

    from transformers import (
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
    )

    bad = 0
    total_source_tokens = 0
    total_target_tokens = 0

    required = {
        "source",
        "messages",
        "target_translation",
    }

    for i, row in enumerate(rows):
        if not required.issubset(
            row
        ):
            bad += 1
            continue

        source = str(
            row["source"]
        )

        target = str(
            row[
                "target_translation"
            ]
        )

        if (
            not source.strip()
            or
            not target.strip()
        ):
            bad += 1
            continue

        messages = row[
            "messages"
        ]

        if (
            not isinstance(messages, list)
            or not messages
        ):
            bad += 1
            continue

        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        source_ids = tokenizer(
            rendered,
            add_special_tokens=False,
            truncation=False,
        )[
            "input_ids"
        ]

        target_ids = tokenizer(
            target,
            add_special_tokens=False,
            truncation=False,
        )[
            "input_ids"
        ]

        if (
            not source_ids
            or
            not target_ids
        ):
            bad += 1
            continue

        total_source_tokens += len(
            source_ids
        )

        total_target_tokens += len(
            target_ids
        )

    print(
        "SFT_SCHEMA_ROWS =",
        len(rows),
    )

    print(
        "SFT_SCHEMA_BAD_ROWS =",
        bad,
    )

    print(
        "SFT_SOURCE_TOKENS =",
        total_source_tokens,
    )

    print(
        "SFT_TARGET_TOKENS =",
        total_target_tokens,
    )

    if bad != 0:
        raise RuntimeError(
            f"SFT schema/tokenization bad rows={bad}"
        )

    print(
        "SFT_SCHEMA_TOKENIZATION_PASS"
    )


###############################################################################
# CLI
###############################################################################

ap = argparse.ArgumentParser()

sub = ap.add_subparsers(
    dest="cmd",
    required=True,
)

p = sub.add_parser(
    "build-analog-jobs"
)

p.add_argument(
    "--k1",
    required=True,
)

p.add_argument(
    "--analysis",
    required=True,
)

p.add_argument(
    "--official",
    required=True,
)

p.add_argument(
    "--output",
    required=True,
)

p.add_argument(
    "--report",
    required=True,
)

p.add_argument(
    "--sample-size",
    type=int,
    default=0,
)

p.add_argument(
    "--seed",
    type=int,
    default=20260831,
)

p.set_defaults(
    func=cmd_build_analog_jobs
)


p = sub.add_parser(
    "parse-analogs"
)

p.add_argument(
    "--jobs",
    required=True,
)

p.add_argument(
    "--shards",
    required=True,
)

p.add_argument(
    "--output",
    required=True,
)

p.add_argument(
    "--report",
    required=True,
)

p.set_defaults(
    func=cmd_parse_analogs
)


p = sub.add_parser(
    "build-context-jobs"
)

p.add_argument(
    "--analogs",
    required=True,
)

p.add_argument(
    "--official",
    required=True,
)

p.add_argument(
    "--output",
    required=True,
)

p.set_defaults(
    func=cmd_build_context_jobs
)


p = sub.add_parser(
    "accept-contexts"
)

p.add_argument(
    "--jobs",
    required=True,
)

p.add_argument(
    "--shards",
    required=True,
)

p.add_argument(
    "--parser",
    required=True,
)

p.add_argument(
    "--accepted",
    required=True,
)

p.add_argument(
    "--sft",
    required=True,
)

p.add_argument(
    "--report",
    required=True,
)

p.set_defaults(
    func=cmd_accept_contexts
)


p = sub.add_parser(
    "validate-sft"
)

p.add_argument(
    "--sft",
    required=True,
)

p.add_argument(
    "--model",
    required=True,
)

p.set_defaults(
    func=cmd_validate_sft
)


args = ap.parse_args()
args.func(args)
