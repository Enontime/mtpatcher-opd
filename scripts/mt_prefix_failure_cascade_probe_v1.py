#!/usr/bin/env python3

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import sacrebleu


EXP = "mtpatcher_v3_full6565_20260823"

DEFAULT_HELD = (
    Path("/workspace/mtpatcher/data")
    / EXP
    / "patcher_quality_gate_v1"
)

DEFAULT_CAND = (
    Path("/workspace/mtpatcher/data")
    / EXP
    / "prefix_failure_probe_v1"
    / "cascade_candidates_strict_v1.jsonl"
)

DEFAULT_MODEL = (
    Path("/workspace/mtpatcher/models")
    / "Qwen3-0.6B"
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
    p = Path(path)

    p.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with p.open(
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


def median(xs):
    if not xs:
        return 0.0

    xs = sorted(xs)
    n = len(xs)

    if n % 2:
        return xs[n // 2]

    return (
        xs[n // 2 - 1]
        + xs[n // 2]
    ) / 2


def bootstrap_ci(
    values,
    seed,
    reps=5000,
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

    return [
        stats[
            int(
                0.025
                * len(stats)
            )
        ],
        stats[
            min(
                len(stats) - 1,
                int(
                    0.975
                    * len(stats)
                ),
            )
        ],
    ]


def run(args):
    import torch
    import torch_npu

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    rows = load_jsonl(
        args.candidates
    )

    if (
        args.limit > 0
        and len(rows) > args.limit
    ):
        rows = rows[
            :args.limit
        ]

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

    tok = (
        AutoTokenizer
        .from_pretrained(
            args.student_model,
            local_files_only=True,
        )
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
        serialized = (
            qwen_format(
                tok,
                raw_prompt,
            )
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

        target_ids, eos_seen = (
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
            target_ids,
            skip_special_tokens=True,
        ).strip()

        return {
            "serialized":
                serialized,

            "prompt_ids":
                prompt_ids,

            "target_ids":
                target_ids,

            "text":
                text,

            "eos_seen":
                eos_seen,
        }


    def continue_from_prefix(
        prompt_ids,
        forced_ids,
    ):
        remaining = (
            args.max_new_tokens
            - len(forced_ids)
        )

        if remaining <= 0:
            raise RuntimeError(
                "prefix exceeds generation budget"
            )

        combined = (
            prompt_ids
            + forced_ids
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

        suffix_ids, eos_seen = (
            trim_at_eos(
                new_ids,
                tok.eos_token_id,
            )
        )

        return {
            "suffix_ids":
                suffix_ids,

            "suffix_text":
                tok.decode(
                    suffix_ids,
                    skip_special_tokens=True,
                ),

            "eos_seen":
                eos_seen,
        }


    rejects = Counter()
    results = []

    prompt_pattern = re.compile(
        r"(?im)"
        r"(^|\n)\s*"
        r"(Input|Output)\s*:"
        r"|Translate the following"
    )


    for pos, x in enumerate(
        rows,
        start=1,
    ):
        jid = int(
            x["job_id"]
        )

        draft = str(
            x[
                "student_translation"
            ]
        ).strip()

        first_only = str(
            x[
                "first_only_translation"
            ]
        ).strip()

        postedit = str(
            x[
                "post_edit"
            ]
        ).strip()

        ref = str(
            x[
                "reference"
            ]
        ).strip()

        raw_first = str(
            x.get(
                "first_raw_text",
                "",
            )
        )

        fix_first = str(
            x.get(
                "first_fix_text",
                "",
            )
        )


        # ------------------------------------------------------------
        # Objective hygiene filters fixed before causal evaluation.
        # ------------------------------------------------------------

        if (
            prompt_pattern.search(
                draft
            )
            or prompt_pattern.search(
                first_only
            )
        ):
            rejects[
                "prompt_contamination"
            ] += 1
            continue

        if (
            "\n" in raw_first
            or "\n" in fix_first
        ):
            rejects[
                "multiline_first_edit"
            ] += 1
            continue

        # If draft has balanced ordinary double quotes,
        # the first local intervention may not create
        # an unmatched quote that is only closed by a later edit.
        if (
            draft.count('"') % 2 == 0
            and first_only.count('"') % 2 != 0
        ):
            rejects[
                "first_only_unbalanced_quote"
            ] += 1
            continue


        # ------------------------------------------------------------
        # Exact on-policy trajectory recovery.
        # ------------------------------------------------------------

        rollout = full_rollout(
            str(
                jobs[jid][
                    "prompt"
                ]
            )
        )

        expected = str(
            stored[jid][
                "response"
            ]
        ).strip()

        if not (
            rollout["text"]
            == expected
            == draft
        ):
            rejects[
                "full_regeneration_fail"
            ] += 1
            continue

        raw_ids = (
            rollout[
                "target_ids"
            ]
        )

        # Canonical retokenization must recover the
        # exact true trajectory for this analysis subset.
        raw_text_ids = tok(
            draft,
            add_special_tokens=False,
        )[
            "input_ids"
        ]

        if (
            raw_text_ids
            != raw_ids
        ):
            rejects[
                "raw_retokenization_mismatch"
            ] += 1
            continue


        first_ids = tok(
            first_only,
            add_special_tokens=False,
        )[
            "input_ids"
        ]

        first_roundtrip = (
            tok.decode(
                first_ids,
                skip_special_tokens=True,
            ).strip()
        )

        if (
            first_roundtrip
            != first_only
        ):
            rejects[
                "first_only_roundtrip_fail"
            ] += 1
            continue


        # ------------------------------------------------------------
        # Minimal legal token replacement between
        # raw trajectory and static first-only counterfactual.
        # ------------------------------------------------------------

        common_prefix = 0

        while (
            common_prefix
            < len(raw_ids)
            and common_prefix
            < len(first_ids)
            and raw_ids[
                common_prefix
            ]
            == first_ids[
                common_prefix
            ]
        ):
            common_prefix += 1


        common_suffix = 0

        max_suffix = min(
            len(raw_ids)
            - common_prefix,

            len(first_ids)
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
            == first_ids[
                len(first_ids)
                - 1
                - common_suffix
            ]
        ):
            common_suffix += 1


        raw_end = (
            len(raw_ids)
            - common_suffix
        )

        fix_end = (
            len(first_ids)
            - common_suffix
        )


        if (
            common_prefix
            == raw_end
            or common_prefix
            == fix_end
        ):
            rejects[
                "token_insert_delete"
            ] += 1
            continue


        raw_edit_ids = (
            raw_ids[
                common_prefix:
                raw_end
            ]
        )

        fix_edit_ids = (
            first_ids[
                common_prefix:
                fix_end
            ]
        )


        # RAW control:
        # preserve the actual student trajectory through first error.
        raw_forced = (
            raw_ids[
                :raw_end
            ]
        )

        expected_raw_suffix = (
            raw_ids[
                raw_end:
            ]
        )


        raw_branch = (
            continue_from_prefix(
                rollout[
                    "prompt_ids"
                ],
                raw_forced,
            )
        )

        if (
            raw_branch[
                "suffix_ids"
            ]
            != expected_raw_suffix
        ):
            rejects[
                "raw_continuation_replay_fail"
            ] += 1
            continue


        # REPAIR:
        # exact true history before first edit,
        # then only the corrected first token span.
        fix_forced = (
            raw_ids[
                :common_prefix
            ]
            + fix_edit_ids
        )


        repair_branch = (
            continue_from_prefix(
                rollout[
                    "prompt_ids"
                ],
                fix_forced,
            )
        )


        repair_full = (
            tok.decode(
                fix_forced
                + repair_branch[
                    "suffix_ids"
                ],

                skip_special_tokens=True,
            )
            .strip()
        )


        raw_suffix_text = (
            tok.decode(
                expected_raw_suffix,
                skip_special_tokens=True,
            )
        )

        repair_suffix_text = (
            repair_branch[
                "suffix_text"
            ]
        )


        # ------------------------------------------------------------
        # Metrics
        # ------------------------------------------------------------

        student_ref = (
            sent_chrf(
                draft,
                ref,
            )
        )

        first_ref = float(
            x[
                "first_only_ref_chrf"
            ]
        )

        full_postedit_ref = float(
            x[
                "full_postedit_ref_chrf"
            ]
        )

        later_gain = float(
            x[
                "later_gain"
            ]
        )

        repair_ref = (
            sent_chrf(
                repair_full,
                ref,
            )
        )


        downstream_delta = (
            repair_ref
            - first_ref
        )


        recovery_ratio = (
            downstream_delta
            / later_gain
            if later_gain > 0
            else None
        )


        first_to_postedit = (
            sent_chrf(
                first_only,
                postedit,
            )
        )

        repair_to_postedit = (
            sent_chrf(
                repair_full,
                postedit,
            )
        )

        toward_postedit_delta = (
            repair_to_postedit
            - first_to_postedit
        )


        suffix_changed = (
            repair_branch[
                "suffix_ids"
            ]
            != expected_raw_suffix
        )


        results.append({
            **x,

            "full_regeneration_exact":
                True,

            "raw_continuation_exact":
                True,

            "valid_for_causal":
                True,

            "token_common_prefix":
                common_prefix,

            "token_common_suffix":
                common_suffix,

            "raw_first_token_text":
                tok.decode(
                    raw_edit_ids,
                    skip_special_tokens=True,
                ),

            "fix_first_token_text":
                tok.decode(
                    fix_edit_ids,
                    skip_special_tokens=True,
                ),

            "raw_suffix":
                raw_suffix_text,

            "repair_suffix":
                repair_suffix_text,

            "repair_suffix_changed":
                suffix_changed,

            "repair_full":
                repair_full,

            "student_ref_chrf_recomputed":
                student_ref,

            "first_only_ref_chrf_used":
                first_ref,

            "full_postedit_ref_chrf_used":
                full_postedit_ref,

            "repair_ref_chrf":
                repair_ref,

            "downstream_delta_ref_chrf":
                downstream_delta,

            "available_later_gain":
                later_gain,

            "recovery_ratio":
                recovery_ratio,

            "first_only_to_postedit_chrf":
                first_to_postedit,

            "repair_to_postedit_chrf":
                repair_to_postedit,

            "toward_postedit_delta_chrf":
                toward_postedit_delta,
        })


        if (
            pos % 4 == 0
            or pos == len(rows)
        ):
            print(
                f"PROGRESS="
                f"{pos}/"
                f"{len(rows)}",
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


    effects = [
        x[
            "downstream_delta_ref_chrf"
        ]
        for x in valid
    ]

    ratios = [
        x[
            "recovery_ratio"
        ]
        for x in valid
        if x[
            "recovery_ratio"
        ] is not None
    ]

    toward = [
        x[
            "toward_postedit_delta_chrf"
        ]
        for x in valid
    ]


    summary = {
        "protocol":
            "MT_PREFIX_FAILURE_CASCADE_PROBE_V1",

        "requested_rows":
            len(rows),

        "valid_rows":
            len(valid),

        "rejects":
            dict(rejects),

        "datasets":
            dict(
                Counter(
                    x[
                        "dataset"
                    ]
                    for x in valid
                )
            ),

        "hard_invariants":
            {
                "full_regeneration_exact":
                    all(
                        x[
                            "full_regeneration_exact"
                        ]
                        for x in valid
                    ),

                "raw_continuation_exact":
                    all(
                        x[
                            "raw_continuation_exact"
                        ]
                        for x in valid
                    ),
            },

        "suffix_changed_fraction":
            mean(
                [
                    float(
                        x[
                            "repair_suffix_changed"
                        ]
                    )
                    for x in valid
                ]
            ),

        "downstream_effect":
            {
                "mean_ref_chrf_delta":
                    mean(
                        effects
                    ),

                "median_ref_chrf_delta":
                    median(
                        effects
                    ),

                "ci95":
                    bootstrap_ci(
                        effects,
                        args.seed,
                    ),

                "positive_fraction_gt_0":
                    mean(
                        [
                            float(
                                v > 0
                            )
                            for v in effects
                        ]
                    ),

                "strong_positive_fraction_gt_1":
                    mean(
                        [
                            float(
                                v > 1
                            )
                            for v in effects
                        ]
                    ),

                "strong_negative_fraction_lt_minus1":
                    mean(
                        [
                            float(
                                v < -1
                            )
                            for v in effects
                        ]
                    ),

                "neutral_fraction_abs_le_1":
                    mean(
                        [
                            float(
                                abs(v) <= 1
                            )
                            for v in effects
                        ]
                    ),
            },

        "recovery_ratio":
            {
                "mean":
                    mean(
                        ratios
                    ),

                "median":
                    median(
                        ratios
                    ),

                "fraction_gt_0":
                    mean(
                        [
                            float(
                                v > 0
                            )
                            for v in ratios
                        ]
                    ),

                "fraction_ge_025":
                    mean(
                        [
                            float(
                                v >= 0.25
                            )
                            for v in ratios
                        ]
                    ),

                "fraction_ge_050":
                    mean(
                        [
                            float(
                                v >= 0.50
                            )
                            for v in ratios
                        ]
                    ),

                "fraction_ge_100":
                    mean(
                        [
                            float(
                                v >= 1.00
                            )
                            for v in ratios
                        ]
                    ),
            },

        "toward_full_postedit":
            {
                "mean_delta_chrf":
                    mean(
                        toward
                    ),

                "median_delta_chrf":
                    median(
                        toward
                    ),

                "positive_fraction":
                    mean(
                        [
                            float(
                                v > 0
                            )
                            for v in toward
                        ]
                    ),
            },
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
    print(
        "CASCADE_PROBE_V1_DONE"
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
            DEFAULT_CAND
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
