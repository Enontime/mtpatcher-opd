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

DEFAULT_CANDIDATES = (
    Path("/workspace/mtpatcher/data")
    / EXP
    / "prefix_failure_probe_v0"
    / "candidates.jsonl"
)

DEFAULT_MODEL = (
    Path("/workspace/mtpatcher/models")
    / "Qwen3-0.6B"
)


TOKEN_RE = re.compile(
    r"\w+(?:['’\-]\w+)*|[^\w\s]",
    flags=re.UNICODE,
)


def load_jsonl(path):
    with Path(path).open(
        encoding="utf-8"
    ) as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


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


def qwen_format(
    tokenizer,
    prompt,
):
    # Exact ADAPTATION used by
    # paper_repro_eval512_v1.py.
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


def text_tokens(text):
    toks = []
    spans = []

    for m in TOKEN_RE.finditer(
        text
    ):
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


def single_edit_info(
    draft,
    fixed,
):
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

    changed = [
        x
        for x in sm.get_opcodes()
        if x[0] != "equal"
    ]

    if len(changed) != 1:
        return None

    tag, i1, i2, j1, j2 = (
        changed[0]
    )

    if (
        i1 == i2
        or j1 == j2
    ):
        return None

    raw_start = (
        dspan[i1][0]
    )

    raw_end = (
        dspan[i2 - 1][1]
    )

    fix_start = (
        fspan[j1][0]
    )

    fix_end = (
        fspan[j2 - 1][1]
    )

    # For the local-only control we require
    # byte/text identity before and after
    # the one intervention block.
    if (
        draft[:raw_start]
        != fixed[:fix_start]
    ):
        return None

    if (
        draft[raw_end:]
        != fixed[fix_end:]
    ):
        return None

    return {
        "tag":
            tag,

        "raw_char_start":
            raw_start,

        "raw_char_end":
            raw_end,

        "fix_char_start":
            fix_start,

        "fix_char_end":
            fix_end,

        "shared_prefix_text":
            draft[:raw_start],

        "raw_edit_text":
            draft[
                raw_start:
                raw_end
            ],

        "fix_edit_text":
            fixed[
                fix_start:
                fix_end
            ],

        "desired_suffix_text":
            draft[raw_end:],
    }


def trim_at_eos(
    ids,
    eos_id,
):
    ids = list(ids)

    if eos_id in ids:
        k = ids.index(
            eos_id
        )

        return (
            ids[:k],
            True,
        )

    return (
        ids,
        False,
    )


def find_text_boundary(
    tok,
    ids,
    full_text,
    char_index,
):
    """
    Map a character boundary in the stripped
    target text to an exact tokenizer boundary.

    No approximation is allowed.
    """

    decoded = tok.decode(
        ids,
        skip_special_tokens=True,
    )

    if (
        decoded.strip()
        != full_text.strip()
    ):
        return None

    # Locate the stored stripped target inside
    # the raw decoded target.
    start = decoded.find(
        full_text
    )

    if start < 0:
        stripped = decoded.strip()

        if stripped != full_text:
            return None

        start = (
            len(decoded)
            - len(
                decoded.lstrip()
            )
        )

    absolute = (
        start
        + char_index
    )

    for k in range(
        len(ids) + 1
    ):
        prefix = tok.decode(
            ids[:k],
            skip_special_tokens=True,
        )

        if len(prefix) == absolute:
            return k

    return None


def sent_chrf(
    hyp,
    ref,
):
    return float(
        sacrebleu.sentence_chrf(
            hyp,
            [ref],
        ).score
    )


def mean(xs):
    if not xs:
        return 0.0

    return (
        sum(xs)
        / len(xs)
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


def stratified_sample(
    rows,
    limit,
    seed,
):
    if (
        limit <= 0
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

    for x in groups.values():
        rng.shuffle(x)

    names = sorted(
        groups
    )

    out = []

    while (
        len(out) < limit
        and any(
            groups[n]
            for n in names
        )
    ):
        for name in names:
            if (
                groups[name]
                and len(out) < limit
            ):
                out.append(
                    groups[name].pop()
                )

    return out


def run(args):
    import torch
    import torch_npu

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    candidates = load_jsonl(
        args.candidates
    )

    candidates = (
        stratified_sample(
            candidates,
            args.limit,
            args.seed,
        )
    )

    held = Path(
        args.held_dir
    )

    jobs = {
        int(x["job_id"]): x
        for x in load_jsonl(
            held
            / "student_jobs.jsonl"
        )
    }

    stored = {
        int(x["job_id"]): x
        for x in load_jsonl(
            held
            / "student_raw.jsonl"
        )
    }

    device = (
        f"npu:{args.device}"
    )

    torch.npu.set_device(
        device
    )

    tok = AutoTokenizer.from_pretrained(
        args.student_model,
        local_files_only=True,
    )

    tok.padding_side = "left"

    if tok.pad_token_id is None:
        tok.pad_token = (
            tok.eos_token
        )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            args.student_model,
            dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
    )

    model.to(device)
    model.eval()

    def encode_prompt(
        raw_prompt,
    ):
        serialized = qwen_format(
            tok,
            raw_prompt,
        )

        enc = tok(
            [serialized],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )

        return (
            serialized,
            enc,
        )

    def full_rollout(
        raw_prompt,
    ):
        serialized, enc = (
            encode_prompt(
                raw_prompt
            )
        )

        enc = {
            k:
                v.to(
                    device,
                    non_blocking=True,
                )
            for k, v
            in enc.items()
        }

        width = (
            enc[
                "input_ids"
            ].shape[1]
        )

        with torch.inference_mode():
            generated = (
                model.generate(
                    **enc,
                    max_new_tokens=
                        args.max_new_tokens,
                    pad_token_id=
                        tok.pad_token_id,
                    eos_token_id=
                        tok.eos_token_id,
                    do_sample=False,
                )
            )

        new_ids = (
            generated[
                0,
                width:
            ]
            .detach()
            .cpu()
            .tolist()
        )

        active, eos_seen = (
            trim_at_eos(
                new_ids,
                tok.eos_token_id,
            )
        )

        prompt_ids = (
            enc[
                "input_ids"
            ][0]
            .detach()
            .cpu()
            .tolist()
        )

        text = tok.decode(
            active,
            skip_special_tokens=True,
        ).strip()

        return {
            "serialized":
                serialized,

            "prompt_ids":
                prompt_ids,

            "target_ids":
                active,

            "text":
                text,

            "eos_seen":
                eos_seen,
        }

    def continue_from_prefix(
        prompt_ids,
        forced_prefix_ids,
    ):
        remaining = (
            args.max_new_tokens
            - len(
                forced_prefix_ids
            )
        )

        if remaining <= 0:
            raise RuntimeError(
                "forced prefix reaches "
                "max_new_tokens"
            )

        combined = (
            prompt_ids
            + forced_prefix_ids
        )

        input_ids = torch.tensor(
            [combined],
            dtype=torch.long,
            device=device,
        )

        attention_mask = (
            torch.ones_like(
                input_ids
            )
        )

        width = (
            input_ids.shape[1]
        )

        with torch.inference_mode():
            generated = (
                model.generate(
                    input_ids=
                        input_ids,

                    attention_mask=
                        attention_mask,

                    max_new_tokens=
                        remaining,

                    pad_token_id=
                        tok.pad_token_id,

                    eos_token_id=
                        tok.eos_token_id,

                    do_sample=False,
                )
            )

        new_ids = (
            generated[
                0,
                width:
            ]
            .detach()
            .cpu()
            .tolist()
        )

        active, eos_seen = (
            trim_at_eos(
                new_ids,
                tok.eos_token_id,
            )
        )

        return {
            "suffix_ids":
                active,

            "eos_seen":
                eos_seen,

            "suffix_text":
                tok.decode(
                    active,
                    skip_special_tokens=True,
                ),
        }

    rejects = Counter()
    results = []

    for pos, row in enumerate(
        candidates,
        start=1,
    ):
        jid = int(
            row["job_id"]
        )

        draft = str(
            row[
                "student_translation"
            ]
        ).strip()

        fixed = str(
            row[
                "post_edit"
            ]
        ).strip()

        ref = str(
            row[
                "reference"
            ]
        ).strip()

        edit = single_edit_info(
            draft,
            fixed,
        )

        if edit is None:
            rejects[
                "local_text_invariant_fail"
            ] += 1
            continue

        job = jobs[jid]

        rollout = full_rollout(
            str(
                job["prompt"]
            )
        )

        expected = str(
            stored[jid][
                "response"
            ]
        ).strip()

        full_exact = (
            rollout[
                "text"
            ]
            == expected
            == draft
        )

        if not full_exact:
            rejects[
                "full_regeneration_fail"
            ] += 1
            continue

        raw_ids = (
            rollout[
                "target_ids"
            ]
        )

        # ------------------------------------------------------------------
        # TOKEN-LEVEL LOCAL INTERVENTION ALIGNMENT
        #
        # The previous implementation required the character-level edit
        # boundaries to land exactly on tokenizer boundaries.  That rejects
        # valid local repairs whenever an edit starts/ends inside a subword.
        #
        # Here the true generated Student token trajectory is authoritative.
        # We independently tokenize the intended corrected sentence, then
        # find:
        #
        #   longest common token prefix
        #   longest common token suffix
        #
        # Everything between them is the minimal legal token intervention.
        # ------------------------------------------------------------------

        raw_text_ids = tok(
            draft,
            add_special_tokens=False,
        )[
            "input_ids"
        ]

        # Because the raw branch must represent the actual on-policy
        # trajectory, silently substituting a retokenized trajectory is not
        # allowed.
        if (
            raw_text_ids
            != raw_ids
        ):
            rejects[
                "raw_retokenization_mismatch"
            ] += 1
            continue

        fix_ids = tok(
            fixed,
            add_special_tokens=False,
        )[
            "input_ids"
        ]

        fix_decoded = tok.decode(
            fix_ids,
            skip_special_tokens=True,
        )

        if (
            fix_decoded.strip()
            != fixed
        ):
            rejects[
                "fix_roundtrip_fail"
            ] += 1
            continue

        # Longest common token prefix.
        common_prefix = 0

        while (
            common_prefix
            < len(raw_ids)
            and common_prefix
            < len(fix_ids)
            and raw_ids[
                common_prefix
            ]
            == fix_ids[
                common_prefix
            ]
        ):
            common_prefix += 1

        # Longest common token suffix, without overlapping the prefix.
        common_suffix = 0

        max_suffix = min(
            len(raw_ids)
            - common_prefix,
            len(fix_ids)
            - common_prefix,
        )

        while (
            common_suffix
            < max_suffix
            and raw_ids[
                len(raw_ids)
                - 1
                - common_suffix
            ]
            == fix_ids[
                len(fix_ids)
                - 1
                - common_suffix
            ]
        ):
            common_suffix += 1

        raw_start_k = (
            common_prefix
        )

        fix_start_k = (
            common_prefix
        )

        raw_end_k = (
            len(raw_ids)
            - common_suffix
        )

        fix_end_k = (
            len(fix_ids)
            - common_suffix
        )

        if (
            raw_start_k
            == raw_end_k
            and fix_start_k
            == fix_end_k
        ):
            rejects[
                "no_token_change"
            ] += 1
            continue

        # Keep this first mechanism set replacement-only.
        # Insertions/deletions can be handled in a later intervention study.
        if (
            raw_start_k
            == raw_end_k
            or fix_start_k
            == fix_end_k
        ):
            rejects[
                "token_insert_delete"
            ] += 1
            continue

        # These are the causal invariants for Local-only controls.
        if (
            raw_ids[
                :raw_start_k
            ]
            != fix_ids[
                :fix_start_k
            ]
        ):
            rejects[
                "pre_edit_token_history_mismatch"
            ] += 1
            continue

        expected_raw_suffix = (
            raw_ids[
                raw_end_k:
            ]
        )

        intended_fix_suffix = (
            fix_ids[
                fix_end_k:
            ]
        )

        if (
            expected_raw_suffix
            != intended_fix_suffix
        ):
            rejects[
                "post_edit_token_suffix_mismatch"
            ] += 1
            continue

        raw_edit_token_ids = (
            raw_ids[
                raw_start_k:
                raw_end_k
            ]
        )

        fix_edit_token_ids = (
            fix_ids[
                fix_start_k:
                fix_end_k
            ]
        )

        raw_token_edit_text = (
            tok.decode(
                raw_edit_token_ids,
                skip_special_tokens=True,
            )
        )

        fix_token_edit_text = (
            tok.decode(
                fix_edit_token_ids,
                skip_special_tokens=True,
            )
        )

        # RAW:
        # keep the true Student trajectory through the erroneous block.
        raw_forced = (
            raw_ids[
                :raw_end_k
            ]
        )

        # REPAIR:
        # preserve the exact real Student token history before intervention,
        # then insert only the corrected minimal token block.
        fix_forced = (
            raw_ids[
                :raw_start_k
            ]
            + fix_edit_token_ids
        )

        raw_branch = (
            continue_from_prefix(
                rollout[
                    "prompt_ids"
                ],
                raw_forced,
            )
        )

        raw_cont_exact = (
            raw_branch[
                "suffix_ids"
            ]
            == expected_raw_suffix
        )

        if not raw_cont_exact:
            rejects[
                "raw_continuation_replay_fail"
            ] += 1

            results.append({
                **row,
                **edit,

                "full_regeneration_exact":
                    True,

                "raw_continuation_exact":
                    False,

                "valid_for_causal":
                    False,
            })

            continue

        repair_branch = (
            continue_from_prefix(
                rollout[
                    "prompt_ids"
                ],
                fix_forced,
            )
        )

        raw_full = tok.decode(
            raw_forced
            + raw_branch[
                "suffix_ids"
            ],
            skip_special_tokens=True,
        ).strip()

        repair_full = tok.decode(
            fix_forced
            + repair_branch[
                "suffix_ids"
            ],
            skip_special_tokens=True,
        ).strip()

        # For Local-only control the target suffix is defined directly
        # from the true original Student token trajectory.
        desired_suffix = tok.decode(
            expected_raw_suffix,
            skip_special_tokens=True,
        )

        repair_suffix = (
            repair_branch[
                "suffix_text"
            ]
        )

        # Strongest preservation test: token-exact downstream continuation.
        suffix_exact = (
            repair_branch[
                "suffix_ids"
            ]
            == expected_raw_suffix
        )

        suffix_chrf = sent_chrf(
            repair_suffix.strip(),
            desired_suffix.strip(),
        )

        student_ref = sent_chrf(
            draft,
            ref,
        )

        postedit_ref = sent_chrf(
            fixed,
            ref,
        )

        repair_ref = sent_chrf(
            repair_full,
            ref,
        )

        repair_to_postedit = (
            sent_chrf(
                repair_full,
                fixed,
            )
        )

        results.append({
            **row,
            **edit,

            "raw_start_token":
                raw_start_k,

            "raw_end_token":
                raw_end_k,

            "fix_start_token":
                fix_start_k,

            "fix_end_token":
                fix_end_k,

            "raw_edit_token_ids":
                raw_edit_token_ids,

            "fix_edit_token_ids":
                fix_edit_token_ids,

            "raw_token_edit_text":
                raw_token_edit_text,

            "fix_token_edit_text":
                fix_token_edit_text,

            "common_prefix_tokens":
                common_prefix,

            "common_suffix_tokens":
                common_suffix,

            "full_regeneration_exact":
                True,

            "raw_continuation_exact":
                True,

            "valid_for_causal":
                True,

            "raw_full":
                raw_full,

            "repair_full":
                repair_full,

            "raw_suffix":
                raw_branch[
                    "suffix_text"
                ],

            "repair_suffix":
                repair_suffix,

            "desired_suffix":
                desired_suffix,

            "repair_suffix_exact_desired":
                suffix_exact,

            "repair_suffix_chrf_desired":
                suffix_chrf,

            "student_ref_chrf":
                student_ref,

            "postedit_ref_chrf":
                postedit_ref,

            "repair_ref_chrf":
                repair_ref,

            "repair_total_delta_chrf":
                repair_ref
                - student_ref,

            "repair_vs_postedit_chrf":
                repair_to_postedit,

            # Same local correction is already present in the static
            # post-edit.  Therefore this difference is a diagnostic for
            # the additional effect introduced by Student re-generation
            # of the downstream continuation.
            "downstream_delta_ref_chrf":
                repair_ref
                - postedit_ref,
        })

        if (
            pos % 4 == 0
            or pos
            == len(candidates)
        ):
            print(
                f"PROGRESS="
                f"{pos}/"
                f"{len(candidates)}",
                flush=True,
            )

    write_jsonl(
        args.output,
        results,
    )

    valid = [
        x
        for x in results
        if x.get(
            "valid_for_causal",
            False,
        )
    ]

    suffix_chrf = [
        x[
            "repair_suffix_chrf_desired"
        ]
        for x in valid
    ]

    total_delta = [
        x[
            "repair_total_delta_chrf"
        ]
        for x in valid
    ]

    summary = {
        "protocol":
            "MT_PREFIX_FAILURE_PROBE_V1_LOCAL_CONTROL",

        "requested_rows":
            len(candidates),

        "result_rows":
            len(results),

        "valid_rows":
            len(valid),

        "rejects":
            dict(rejects),

        "hard_invariants":
            {
                "full_regeneration_exact":
                    all(
                        x.get(
                            "full_regeneration_exact",
                            False,
                        )
                        for x in valid
                    ),

                "raw_continuation_exact":
                    all(
                        x.get(
                            "raw_continuation_exact",
                            False,
                        )
                        for x in valid
                    ),
            },

        "repair_suffix_exact_fraction":
            mean(
                [
                    float(
                        x[
                            "repair_suffix_exact_desired"
                        ]
                    )
                    for x in valid
                ]
            ),

        "repair_suffix_chrf_to_desired_mean":
            mean(
                suffix_chrf
            ),

        "repair_total_delta_chrf_mean":
            mean(
                total_delta
            ),

        "repair_total_delta_chrf_ci95":
            bootstrap_ci(
                total_delta,
                args.seed + 1,
            ),

        "repair_vs_postedit_chrf_mean":
            mean(
                [
                    x[
                        "repair_vs_postedit_chrf"
                    ]
                    for x in valid
                ]
            ),

        "downstream_effect":
            {
                "mean_ref_chrf_delta":
                    mean(
                        [
                            x[
                                "downstream_delta_ref_chrf"
                            ]
                            for x in valid
                        ]
                    ),

                "strong_positive_fraction_gt1":
                    mean(
                        [
                            float(
                                x[
                                    "downstream_delta_ref_chrf"
                                ]
                                > 1.0
                            )
                            for x in valid
                        ]
                    ),

                "strong_negative_fraction_lt_minus1":
                    mean(
                        [
                            float(
                                x[
                                    "downstream_delta_ref_chrf"
                                ]
                                < -1.0
                            )
                            for x in valid
                        ]
                    ),

                "neutral_fraction_abs_le1":
                    mean(
                        [
                            float(
                                abs(
                                    x[
                                        "downstream_delta_ref_chrf"
                                    ]
                                )
                                <= 1.0
                            )
                            for x in valid
                        ]
                    ),
            },

        "datasets":
            dict(
                Counter(
                    x.get(
                        "dataset",
                        "unknown",
                    )
                    for x in valid
                )
            ),
    }

    summary_path = (
        Path(args.output)
        .with_suffix(
            ".summary.json"
        )
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()

    if len(valid) != len(
        candidates
    ):
        print(
            "LOCAL_CONTROL_REPLAY_GATE_PARTIAL"
        )
    else:
        print(
            "LOCAL_CONTROL_REPLAY_GATE_PASS"
        )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--held-dir",
        default=str(
            DEFAULT_HELD
        ),
    )

    p.add_argument(
        "--candidates",
        default=str(
            DEFAULT_CANDIDATES
        ),
    )

    p.add_argument(
        "--student-model",
        default=str(
            DEFAULT_MODEL
        ),
    )

    p.add_argument(
        "--output",
        required=True,
    )

    p.add_argument(
        "--limit",
        type=int,
        default=0,
    )

    p.add_argument(
        "--device",
        type=int,
        default=0,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=20260827,
    )

    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )

    args = p.parse_args()

    run(args)


if __name__ == "__main__":
    main()
