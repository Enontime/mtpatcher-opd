#!/usr/bin/env python3

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import sacrebleu


EXP = "mtpatcher_v3_full6565_20260823"

DEFAULT_HELD = (
    Path("/workspace/mtpatcher/data")
    / EXP
    / "patcher_quality_gate_v1"
)

DEFAULT_OVERNIGHT = (
    Path("/workspace/mtpatcher/data")
    / EXP
    / "patcher_core_overnight_v1"
)

DEFAULT_MODEL = (
    Path("/workspace/mtpatcher/models")
    / "Qwen3-0.6B"
)


###############################################################################
# IO
###############################################################################

def load_jsonl(path):
    rows = []

    with Path(path).open(
        encoding="utf-8"
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def write_jsonl(path, rows):
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


###############################################################################
# Reproduce PATCHER_CORE_OVERNIGHT_V1 raw-feedback scorer
###############################################################################

def raw_feedback(raw, draft):
    text = str(raw).strip()

    has_no = bool(
        re.search(
            r"(?i)\bNo\s+error\.?\b",
            text,
        )
    )

    explicit_error = bool(
        re.search(
            r"(?i)"
            r"(Error\s*(?:Type|Details|\d+)"
            r"|Chinese Segment"
            r"|Correct Translation)",
            text,
        )
    )

    if (
        has_no
        and not explicit_error
    ):
        return {
            "has_error": False,
            "post_edit": draft,
            "extract_ok": True,
            "route": "no_error",
        }

    patterns = [
        r"(?is)"
        r"\*\*Good Translation\*\*\s*:?\s*(.+?)\s*$",

        r"(?is)"
        r"Good Translation\s*:?\s*(.+?)\s*$",

        r"(?is)"
        r"Good translation\s*:?\s*(.+?)\s*$",

        r"(?is)"
        r"\*\*Final Translation\*\*\s*:?\s*(.+?)\s*$",

        r"(?is)"
        r"Final Translation\s*:?\s*(.+?)\s*$",
    ]

    candidate = None

    for pattern in patterns:
        m = re.search(
            pattern,
            text,
        )

        if m:
            candidate = (
                m.group(1).strip()
            )
            break

    if candidate:
        candidate = re.sub(
            r"^\s*[-*]\s*",
            "",
            candidate,
        ).strip()

        return {
            "has_error": True,
            "post_edit": candidate,
            "extract_ok": True,
            "route": "post_edit",
        }

    return {
        "has_error": True,
        "post_edit": draft,
        "extract_ok": False,
        "route": "fallback_draft",
    }


###############################################################################
# Translation-token alignment
###############################################################################

TOKEN_RE = re.compile(
    r"\w+(?:['’\-]\w+)*|[^\w\s]",
    flags=re.UNICODE,
)


def text_tokens(text):
    """
    Return:
        token strings
        [(char_start, char_end), ...]
    """
    toks = []
    spans = []

    for m in TOKEN_RE.finditer(text):
        toks.append(
            m.group(0)
        )

        spans.append(
            (
                m.start(),
                m.end(),
            )
        )

    return toks, spans


def local_edit(draft, fixed):
    dtok, dspan = text_tokens(
        draft
    )

    ftok, fspan = text_tokens(
        fixed
    )

    sm = SequenceMatcher(
        None,
        dtok,
        ftok,
        autojunk=False,
    )

    ops = sm.get_opcodes()

    changed = [
        x
        for x in ops
        if x[0] != "equal"
    ]

    if len(changed) != 1:
        return None

    tag, i1, i2, j1, j2 = (
        changed[0]
    )

    # Need a genuine replacement-like region.
    # Pure deletion/insertion can be studied later.
    if i1 == i2 or j1 == j2:
        return None

    # Common target suffix must exist so there is
    # meaningful continuation after the repair.
    suffix_tokens = (
        len(dtok) - i2
    )

    if suffix_tokens <= 0:
        return None

    # End the forced prefix at the beginning of the
    # first common suffix token.  This preserves the
    # whitespace between edit and continuation.
    d_cut = (
        dspan[i2][0]
        if i2 < len(dspan)
        else len(draft)
    )

    f_cut = (
        fspan[j2][0]
        if j2 < len(fspan)
        else len(fixed)
    )

    raw_prefix = draft[:d_cut]
    fix_prefix = fixed[:f_cut]

    raw_edit = (
        draft[
            dspan[i1][0]:
            dspan[i2 - 1][1]
        ]
    )

    fix_edit = (
        fixed[
            fspan[j1][0]:
            fspan[j2 - 1][1]
        ]
    )

    raw_suffix = draft[d_cut:]
    fix_suffix = fixed[f_cut:]

    return {
        "tag": tag,

        "draft_tokens":
            len(dtok),

        "fixed_tokens":
            len(ftok),

        "prefix_tokens":
            i1,

        "raw_edit_tokens":
            i2 - i1,

        "fix_edit_tokens":
            j2 - j1,

        "suffix_tokens":
            suffix_tokens,

        "raw_prefix":
            raw_prefix,

        "fix_prefix":
            fix_prefix,

        "raw_edit":
            raw_edit,

        "fix_edit":
            fix_edit,

        "raw_original_suffix":
            raw_suffix,

        "fix_original_suffix":
            fix_suffix,

        "opcodes":
            ops,
    }


###############################################################################
# Metrics
###############################################################################

def sent_chrf(hyp, ref):
    return float(
        sacrebleu.sentence_chrf(
            hyp,
            [ref],
        ).score
    )


def corpus_metrics(hyp, ref):
    if not hyp:
        return {
            "bleu": 0.0,
            "chrf": 0.0,
        }

    return {
        "bleu":
            float(
                sacrebleu.corpus_bleu(
                    hyp,
                    [ref],
                ).score
            ),

        "chrf":
            float(
                sacrebleu.corpus_chrf(
                    hyp,
                    [ref],
                ).score
            ),
    }


###############################################################################
# Candidate construction
###############################################################################

def load_echo_ids(path):
    path = Path(path)

    if not path.exists():
        return set()

    obj = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    return {
        int(x["job_id"])
        for x in obj.get(
            "rows",
            [],
        )
    }


def build_candidates(args):
    held = Path(
        args.held_dir
    )

    overnight = Path(
        args.overnight_dir
    )

    student_rows = load_jsonl(
        held / "student.jsonl"
    )

    zero_rows = load_jsonl(
        overnight
        / "held2207_zero512.jsonl"
    )

    job_rows = load_jsonl(
        held / "student_jobs.jsonl"
    )

    student = {
        int(x["job_id"]): x
        for x in student_rows
    }

    zero = {
        int(x["job_id"]): x
        for x in zero_rows
    }

    jobs = {
        int(x["job_id"]): x
        for x in job_rows
    }

    echo_ids = load_echo_ids(
        held
        / "student_prompt_echo_ids.json"
    )

    rejects = Counter()
    out = []

    common_ids = sorted(
        set(student)
        & set(zero)
        & set(jobs)
    )

    for jid in common_ids:
        base = student[jid]
        raw = zero[jid]
        job = jobs[jid]

        if jid in echo_ids:
            rejects[
                "prompt_echo"
            ] += 1
            continue

        source = str(
            base["source"]
        )

        if source.startswith(
            "CANARY GUID"
        ):
            rejects[
                "canary"
            ] += 1
            continue

        draft = str(
            base[
                "student_translation"
            ]
        ).strip()

        ref = str(
            base[
                "reference"
            ]
        ).strip()

        parsed = raw_feedback(
            raw.get(
                "response",
                "",
            ),
            draft,
        )

        if not parsed[
            "has_error"
        ]:
            rejects[
                "no_error"
            ] += 1
            continue

        if not parsed[
            "extract_ok"
        ]:
            rejects[
                "extract_fail"
            ] += 1
            continue

        fixed = str(
            parsed[
                "post_edit"
            ]
        ).strip()

        if (
            not fixed
            or fixed == draft
        ):
            rejects[
                "no_change"
            ] += 1
            continue

        edit = local_edit(
            draft,
            fixed,
        )

        if edit is None:
            rejects[
                "not_single_local_edit"
            ] += 1
            continue

        if (
            edit[
                "raw_edit_tokens"
            ]
            > args.max_edit_tokens
            or
            edit[
                "fix_edit_tokens"
            ]
            > args.max_edit_tokens
        ):
            rejects[
                "edit_too_long"
            ] += 1
            continue

        if (
            edit[
                "suffix_tokens"
            ]
            < args.min_suffix_tokens
        ):
            rejects[
                "suffix_too_short"
            ] += 1
            continue

        max_len = max(
            edit[
                "draft_tokens"
            ],
            edit[
                "fixed_tokens"
            ],
            1,
        )

        edit_ratio = max(
            edit[
                "raw_edit_tokens"
            ],
            edit[
                "fix_edit_tokens"
            ],
        ) / max_len

        if (
            edit_ratio
            > args.max_edit_ratio
        ):
            rejects[
                "edit_ratio_too_large"
            ] += 1
            continue

        student_q = sent_chrf(
            draft,
            ref,
        )

        fixed_q = sent_chrf(
            fixed,
            ref,
        )

        gain = (
            fixed_q
            - student_q
        )

        if (
            gain
            < args.min_ref_gain
        ):
            rejects[
                "no_reference_gain"
            ] += 1
            continue

        prompt = str(
            job.get(
                "prompt",
                "",
            )
        )

        if not prompt:
            rejects[
                "missing_prompt"
            ] += 1
            continue

        out.append({
            "job_id":
                jid,

            "dataset":
                base.get(
                    "dataset",
                    "",
                ),

            "local_id":
                base.get(
                    "local_id",
                    None,
                ),

            "source":
                source,

            "reference":
                ref,

            "student_translation":
                draft,

            "post_edit":
                fixed,

            "raw_feedback":
                raw.get(
                    "response",
                    "",
                ),

            "prompt":
                prompt,

            "student_sentence_chrf":
                student_q,

            "post_edit_sentence_chrf":
                fixed_q,

            "reference_gain_chrf":
                gain,

            "edit_ratio":
                edit_ratio,

            **edit,
        })

    write_jsonl(
        args.candidates,
        out,
    )

    summary = {
        "input_common_rows":
            len(common_ids),

        "candidate_rows":
            len(out),

        "rejects":
            dict(rejects),

        "datasets":
            dict(
                Counter(
                    x[
                        "dataset"
                    ]
                    for x in out
                )
            ),

        "reference_gain_mean":
            (
                sum(
                    x[
                        "reference_gain_chrf"
                    ]
                    for x in out
                )
                / len(out)
                if out
                else 0.0
            ),

        "min_ref_gain":
            args.min_ref_gain,

        "max_edit_tokens":
            args.max_edit_tokens,

        "min_suffix_tokens":
            args.min_suffix_tokens,

        "max_edit_ratio":
            args.max_edit_ratio,
    }

    manifest = Path(
        args.candidates
    ).with_suffix(
        ".summary.json"
    )

    manifest.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "PREFIX_FAILURE_V0_BUILD_PASS"
    )


###############################################################################
# Sampling
###############################################################################

def stratified_sample(
    rows,
    limit,
    seed,
):
    if (
        limit is None
        or limit <= 0
        or len(rows) <= limit
    ):
        return rows

    rng = random.Random(
        seed
    )

    groups = defaultdict(
        list
    )

    for x in rows:
        groups[
            x.get(
                "dataset",
                "unknown",
            )
        ].append(x)

    for g in groups.values():
        rng.shuffle(g)

    names = sorted(
        groups
    )

    selected = []

    while (
        len(selected) < limit
        and any(
            groups[n]
            for n in names
        )
    ):
        for n in names:
            if (
                groups[n]
                and len(
                    selected
                ) < limit
            ):
                selected.append(
                    groups[n].pop()
                )

    return selected


###############################################################################
# Student causal continuation
###############################################################################

def run_probe(args):
    import torch

    try:
        import torch_npu  # noqa: F401
    except Exception:
        pass

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    rows = load_jsonl(
        args.candidates
    )

    rows = stratified_sample(
        rows,
        args.limit,
        args.seed,
    )

    if not rows:
        raise RuntimeError(
            "no probe candidates"
        )

    device = (
        f"npu:{args.device}"
    )

    if hasattr(
        torch,
        "npu",
    ):
        torch.npu.set_device(
            device
        )

    print(
        f"MODEL={args.student_model}"
    )

    print(
        f"DEVICE={device}"
    )

    print(
        f"PROBE_ROWS={len(rows)}"
    )

    tok = AutoTokenizer.from_pretrained(
        args.student_model,
        trust_remote_code=True,
        use_fast=True,
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            args.student_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    )

    model.to(
        device
    )

    model.eval()

    eos = (
        tok.eos_token_id
    )

    def encode_text(text):
        return tok(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )[
            "input_ids"
        ][0]

    def generate_suffix(
        prompt,
        target_prefix,
    ):
        # Important:
        # the next autoregressive token usually carries the
        # inter-word leading space itself.  The word-level
        # alignment prefix often ends immediately before the
        # next lexical token and therefore contains that space.
        #
        # Forcing it here would produce e.g.
        #   "Economic " + " Industry"
        # -> "Economic  Industry"
        # and immediately leave the original Student trajectory.
        target_prefix = target_prefix.rstrip()

        prompt_ids = encode_text(
            prompt
        )

        prefix_ids = encode_text(
            target_prefix
        )

        input_ids = torch.cat(
            [
                prompt_ids,
                prefix_ids,
            ],
            dim=0,
        ).unsqueeze(0)

        input_ids = input_ids.to(
            device
        )

        attn = torch.ones_like(
            input_ids
        )

        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                do_sample=False,
                max_new_tokens=
                    args.max_new_tokens,
                pad_token_id=(
                    tok.pad_token_id
                    if tok.pad_token_id
                    is not None
                    else eos
                ),
                eos_token_id=eos,
            )

        new_ids = generated[
            0,
            input_ids.shape[1]:
        ].detach().cpu()

        prefix_ids_cpu = (
            prefix_ids.detach().cpu()
        )

        full_ids = torch.cat(
            [
                prefix_ids_cpu,
                new_ids,
            ],
            dim=0,
        )

        return {
            "prefix_ids":
                prefix_ids_cpu.tolist(),

            "suffix_ids":
                new_ids.tolist(),

            "full_text":
                tok.decode(
                    full_ids,
                    skip_special_tokens=True,
                ).strip(),

            "suffix_text":
                tok.decode(
                    new_ids,
                    skip_special_tokens=True,
                ),
        }

    results = []

    for idx, row in enumerate(
        rows
    ):
        raw_branch = (
            generate_suffix(
                row["prompt"],
                row["raw_prefix"],
            )
        )

        fix_branch = (
            generate_suffix(
                row["prompt"],
                row["fix_prefix"],
            )
        )

        raw_prefix_ids = (
            raw_branch[
                "prefix_ids"
            ]
        )

        fix_prefix_ids = (
            fix_branch[
                "prefix_ids"
            ]
        )

        raw_suffix_ids = (
            raw_branch[
                "suffix_ids"
            ]
        )

        fix_suffix_ids = (
            fix_branch[
                "suffix_ids"
            ]
        )

        def decode_combo(
            prefix_ids,
            suffix_ids,
        ):
            return tok.decode(
                prefix_ids
                + suffix_ids,
                skip_special_tokens=True,
            ).strip()

        rr = decode_combo(
            raw_prefix_ids,
            raw_suffix_ids,
        )

        rf = decode_combo(
            raw_prefix_ids,
            fix_suffix_ids,
        )

        fr = decode_combo(
            fix_prefix_ids,
            raw_suffix_ids,
        )

        ff = decode_combo(
            fix_prefix_ids,
            fix_suffix_ids,
        )

        ref = row[
            "reference"
        ]

        q_rr = sent_chrf(
            rr,
            ref,
        )

        q_rf = sent_chrf(
            rf,
            ref,
        )

        q_fr = sent_chrf(
            fr,
            ref,
        )

        q_ff = sent_chrf(
            ff,
            ref,
        )

        total_effect = (
            q_ff - q_rr
        )

        suffix_effect = 0.5 * (
            (q_rf - q_rr)
            +
            (q_ff - q_fr)
        )

        prefix_effect = 0.5 * (
            (q_fr - q_rr)
            +
            (q_ff - q_rf)
        )

        interaction = (
            q_ff
            - q_fr
            - q_rf
            + q_rr
        )

        replay_exact = (
            rr.strip()
            ==
            row[
                "student_translation"
            ].strip()
        )

        replay_chrf = sent_chrf(
            rr,
            row[
                "student_translation"
            ],
        )

        result = {
            **row,

            "rr_rawprefix_rawsuffix":
                rr,

            "rf_rawprefix_fixsuffix":
                rf,

            "fr_fixprefix_rawsuffix":
                fr,

            "ff_fixprefix_fixsuffix":
                ff,

            "raw_generated_suffix":
                raw_branch[
                    "suffix_text"
                ],

            "fix_generated_suffix":
                fix_branch[
                    "suffix_text"
                ],

            "q_rr_chrf":
                q_rr,

            "q_rf_chrf":
                q_rf,

            "q_fr_chrf":
                q_fr,

            "q_ff_chrf":
                q_ff,

            "total_effect_chrf":
                total_effect,

            "suffix_effect_chrf":
                suffix_effect,

            "prefix_effect_chrf":
                prefix_effect,

            "interaction_chrf":
                interaction,

            "raw_replay_exact":
                replay_exact,

            "raw_replay_chrf":
                replay_chrf,
        }

        results.append(
            result
        )

        if (
            (idx + 1) % 10 == 0
            or idx + 1
            == len(rows)
        ):
            print(
                f"PROGRESS="
                f"{idx + 1}/"
                f"{len(rows)}",
                flush=True,
            )

    write_jsonl(
        args.output,
        results,
    )

    summarize_probe(
        results,
        args,
    )


###############################################################################
# Statistical summary
###############################################################################

def mean(xs):
    return (
        sum(xs) / len(xs)
        if xs
        else 0.0
    )


def bootstrap_ci(
    values,
    seed,
    reps=3000,
):
    if not values:
        return [
            0.0,
            0.0,
        ]

    rng = random.Random(
        seed
    )

    n = len(values)

    stats = []

    for _ in range(reps):
        stats.append(
            mean(
                [
                    values[
                        rng.randrange(n)
                    ]
                    for _ in range(n)
                ]
            )
        )

    stats.sort()

    lo = stats[
        int(
            0.025
            * len(stats)
        )
    ]

    hi = stats[
        min(
            len(stats) - 1,
            int(
                0.975
                * len(stats)
            ),
        )
    ]

    return [
        lo,
        hi,
    ]


def summarize_probe(
    rows,
    args,
):
    suffix = [
        x[
            "suffix_effect_chrf"
        ]
        for x in rows
    ]

    total = [
        x[
            "total_effect_chrf"
        ]
        for x in rows
    ]

    prefix = [
        x[
            "prefix_effect_chrf"
        ]
        for x in rows
    ]

    interaction = [
        x[
            "interaction_chrf"
        ]
        for x in rows
    ]

    replay = [
        x[
            "raw_replay_chrf"
        ]
        for x in rows
    ]

    exact_rate = mean(
        [
            float(
                x[
                    "raw_replay_exact"
                ]
            )
            for x in rows
        ]
    )

    ref = [
        x[
            "reference"
        ]
        for x in rows
    ]

    rr = [
        x[
            "rr_rawprefix_rawsuffix"
        ]
        for x in rows
    ]

    ff = [
        x[
            "ff_fixprefix_fixsuffix"
        ]
        for x in rows
    ]

    summary = {
        "protocol":
            "MT_PREFIX_FAILURE_PROBE_V0",

        "rows":
            len(rows),

        "student_model":
            str(
                args.student_model
            ),

        "raw_replay":
            {
                "exact_rate":
                    exact_rate,

                "mean_chrf_to_original":
                    mean(replay),
            },

        "actual_branches":
            {
                "raw":
                    corpus_metrics(
                        rr,
                        ref,
                    ),

                "repaired":
                    corpus_metrics(
                        ff,
                        ref,
                    ),
            },

        "causal_factorial_chrf":
            {
                "total_effect_mean":
                    mean(total),

                "total_effect_ci95":
                    bootstrap_ci(
                        total,
                        args.seed + 1,
                    ),

                "suffix_effect_mean":
                    mean(suffix),

                "suffix_effect_ci95":
                    bootstrap_ci(
                        suffix,
                        args.seed + 2,
                    ),

                "prefix_effect_mean":
                    mean(prefix),

                "prefix_effect_ci95":
                    bootstrap_ci(
                        prefix,
                        args.seed + 3,
                    ),

                "interaction_mean":
                    mean(
                        interaction
                    ),
            },

        "positive_suffix_fraction":
            mean(
                [
                    float(x > 0)
                    for x in suffix
                ]
            ),

        "datasets":
            dict(
                Counter(
                    x[
                        "dataset"
                    ]
                    for x in rows
                )
            ),
    }

    summary_path = Path(
        args.output
    ).with_suffix(
        ".summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()

    # Replay is the validity check for forcing
    # the recorded Student prefix.
    if (
        exact_rate
        < args.min_replay_exact
        and mean(replay)
        < args.min_replay_chrf
    ):
        raise RuntimeError(
            "RAW_REPLAY_VALIDITY_FAIL: "
            f"exact_rate={exact_rate:.4f}, "
            f"mean_chrf={mean(replay):.4f}"
        )

    print(
        "PREFIX_FAILURE_V0_PROBE_PASS"
    )


###############################################################################
# CLI
###############################################################################

def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--mode",
        choices=[
            "build",
            "probe",
        ],
        required=True,
    )

    p.add_argument(
        "--held-dir",
        default=str(
            DEFAULT_HELD
        ),
    )

    p.add_argument(
        "--overnight-dir",
        default=str(
            DEFAULT_OVERNIGHT
        ),
    )

    p.add_argument(
        "--student-model",
        default=str(
            DEFAULT_MODEL
        ),
    )

    p.add_argument(
        "--candidates",
        default=(
            "/workspace/mtpatcher/data/"
            f"{EXP}/"
            "prefix_failure_probe_v0/"
            "candidates.jsonl"
        ),
    )

    p.add_argument(
        "--output",
        default=(
            "/workspace/mtpatcher/runs/"
            f"{EXP}/"
            "prefix_failure_probe_v0/"
            "results.jsonl"
        ),
    )

    p.add_argument(
        "--min-ref-gain",
        type=float,
        default=2.0,
    )

    p.add_argument(
        "--max-edit-tokens",
        type=int,
        default=10,
    )

    p.add_argument(
        "--min-suffix-tokens",
        type=int,
        default=4,
    )

    p.add_argument(
        "--max-edit-ratio",
        type=float,
        default=0.35,
    )

    p.add_argument(
        "--limit",
        type=int,
        default=0,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=20260827,
    )

    p.add_argument(
        "--device",
        type=int,
        default=0,
    )

    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )

    p.add_argument(
        "--min-replay-exact",
        type=float,
        default=0.70,
    )

    p.add_argument(
        "--min-replay-chrf",
        type=float,
        default=90.0,
    )

    args = p.parse_args()

    if args.mode == "build":
        build_candidates(
            args
        )

    elif args.mode == "probe":
        run_probe(
            args
        )


if __name__ == "__main__":
    main()
