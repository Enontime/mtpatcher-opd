#!/usr/bin/env python3

import json
import random
from collections import Counter
from pathlib import Path

import sacrebleu
import torch
import torch_npu

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


EXP = "mtpatcher_v3_full6565_20260823"

ROOT = Path(
    "/workspace/mtpatcher"
)

HELD = (
    ROOT
    / "data"
    / EXP
    / "patcher_quality_gate_v1"
)

LOCAL = (
    ROOT
    / "runs"
    / EXP
    / "prefix_failure_probe_v1"
    / "local_control_full58.jsonl"
)

OUT = (
    ROOT
    / "runs"
    / EXP
    / "prefix_failure_probe_v1"
    / "recovery_horizon_local40_v1.jsonl"
)

MODEL = (
    ROOT
    / "models"
    / "Qwen3-0.6B"
)

DEVICE = "npu:0"
MAX_NEW_TOKENS = 256

HORIZONS = [
    0,
    1,
    2,
    4,
    8,
]


def load_jsonl(path):
    with path.open(
        encoding="utf-8"
    ) as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def mean(xs):
    if not xs:
        return 0.0

    return sum(xs) / len(xs)


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


def trim_eos(
    ids,
    eos_id,
):
    ids = list(ids)

    if eos_id in ids:
        ids = ids[
            :ids.index(
                eos_id
            )
        ]

    return ids


torch.npu.set_device(
    DEVICE
)

tok = (
    AutoTokenizer
    .from_pretrained(
        MODEL,
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
        MODEL,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
)

model.to(
    DEVICE
)

model.eval()


jobs = {
    int(x["job_id"]): x
    for x in load_jsonl(
        HELD
        / "student_jobs.jsonl"
    )
}

stored = {
    int(x["job_id"]): x
    for x in load_jsonl(
        HELD
        / "student_raw.jsonl"
    )
}

rows = [
    x
    for x in load_jsonl(
        LOCAL
    )
    if x.get(
        "valid_for_causal",
        True,
    )
]


def full_rollout(
    raw_prompt,
):
    prompt = qwen_format(
        tok,
        raw_prompt,
    )

    enc = tok(
        [prompt],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )

    enc = {
        k:
            v.to(
                DEVICE
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
        output = model.generate(
            **enc,

            max_new_tokens=
                MAX_NEW_TOKENS,

            pad_token_id=
                tok.pad_token_id,

            eos_token_id=
                tok.eos_token_id,

            do_sample=False,
        )

    target = (
        output[
            0,
            width:
        ]
        .detach()
        .cpu()
        .tolist()
    )

    target = trim_eos(
        target,
        tok.eos_token_id,
    )

    prompt_ids = (
        enc[
            "input_ids"
        ][0]
        .detach()
        .cpu()
        .tolist()
    )

    return (
        prompt_ids,
        target,
        tok.decode(
            target,
            skip_special_tokens=True,
        ).strip(),
    )


def continue_from(
    prompt_ids,
    forced,
):
    remaining = (
        MAX_NEW_TOKENS
        - len(forced)
    )

    if remaining <= 0:
        return []

    combined = (
        prompt_ids
        + forced
    )

    input_ids = torch.tensor(
        [combined],
        dtype=torch.long,
        device=DEVICE,
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
        output = model.generate(
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

    suffix = (
        output[
            0,
            width:
        ]
        .detach()
        .cpu()
        .tolist()
    )

    return trim_eos(
        suffix,
        tok.eos_token_id,
    )


results = []

rejects = Counter()


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

    fixed = str(
        x[
            "post_edit"
        ]
    ).strip()

    prompt_ids, raw_ids, text = (
        full_rollout(
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
        text
        == expected
        == draft
    ):
        rejects[
            "full_regeneration_fail"
        ] += 1
        continue


    raw_text_ids = tok(
        draft,
        add_special_tokens=False,
    )[
        "input_ids"
    ]

    if raw_text_ids != raw_ids:
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


    # --------------------------------------------------------
    # Minimal token-level replacement.
    # --------------------------------------------------------

    cp = 0

    while (
        cp < len(raw_ids)
        and cp < len(fix_ids)
        and raw_ids[cp]
        == fix_ids[cp]
    ):
        cp += 1


    cs = 0

    max_cs = min(
        len(raw_ids) - cp,
        len(fix_ids) - cp,
    )

    while (
        cs < max_cs
        and raw_ids[
            len(raw_ids)
            - 1
            - cs
        ]
        == fix_ids[
            len(fix_ids)
            - 1
            - cs
        ]
    ):
        cs += 1


    raw_end = (
        len(raw_ids)
        - cs
    )

    fix_end = (
        len(fix_ids)
        - cs
    )


    if (
        cp == raw_end
        or cp == fix_end
    ):
        rejects[
            "insert_delete"
        ] += 1
        continue


    expected_suffix = (
        raw_ids[
            raw_end:
        ]
    )

    fixed_suffix = (
        fix_ids[
            fix_end:
        ]
    )


    # Local-only invariant:
    # Patcher left the downstream suffix unchanged.
    if (
        expected_suffix
        != fixed_suffix
    ):
        rejects[
            "not_local_only_token_suffix"
        ] += 1
        continue


    fix_edit = (
        fix_ids[
            cp:fix_end
        ]
    )


    corrected_prefix = (
        raw_ids[:cp]
        + fix_edit
    )


    row_result = {
        "job_id":
            jid,

        "dataset":
            x.get(
                "dataset"
            ),

        "student_translation":
            draft,

        "post_edit":
            fixed,

        "suffix_tokens":
            len(
                expected_suffix
            ),

        "horizons":
            {},
    }


    for h in HORIZONS:
        # Require at least four genuinely generated
        # downstream tokens after handoff.
        if (
            len(
                expected_suffix
            )
            - h
            < 4
        ):
            continue


        anchor = (
            expected_suffix[
                :h
            ]
        )

        forced = (
            corrected_prefix
            + anchor
        )

        generated = (
            continue_from(
                prompt_ids,
                forced,
            )
        )


        desired = (
            expected_suffix[
                h:
            ]
        )


        exact = (
            generated
            == desired
        )


        gen_text = tok.decode(
            generated,
            skip_special_tokens=True,
        )

        desired_text = tok.decode(
            desired,
            skip_special_tokens=True,
        )


        if (
            gen_text.strip()
            or desired_text.strip()
        ):
            chrf = float(
                sacrebleu.sentence_chrf(
                    gen_text.strip(),
                    [
                        desired_text.strip()
                    ],
                ).score
            )
        else:
            chrf = 100.0


        row_result[
            "horizons"
        ][str(h)] = {
            "exact":
                exact,

            "chrf":
                chrf,

            "generated":
                gen_text,

            "desired":
                desired_text,
        }


    results.append(
        row_result
    )


    if (
        pos % 5 == 0
        or pos == len(rows)
    ):
        print(
            f"PROGRESS="
            f"{pos}/"
            f"{len(rows)}",
            flush=True,
        )


OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUT.open(
    "w",
    encoding="utf-8",
) as f:
    for x in results:
        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )


summary = {
    "protocol":
        "MT_RECOVERY_HORIZON_LOCAL_CONTROL_V1",

    "input_rows":
        len(rows),

    "valid_rows":
        len(results),

    "rejects":
        dict(rejects),

    "horizons":
        {},
}


for h in HORIZONS:
    key = str(h)

    vals = [
        x[
            "horizons"
        ][key]
        for x in results
        if key
        in x[
            "horizons"
        ]
    ]

    if not vals:
        continue

    exact_rate = mean(
        [
            float(
                y[
                    "exact"
                ]
            )
            for y in vals
        ]
    )

    chrf_mean = mean(
        [
            y[
                "chrf"
            ]
            for y in vals
        ]
    )


    # Among rows that failed at h=0 and
    # are also evaluable at this horizon,
    # how many become exactly synchronized?
    if h > 0:
        paired = [
            x
            for x in results
            if (
                "0"
                in x[
                    "horizons"
                ]
                and key
                in x[
                    "horizons"
                ]
                and not x[
                    "horizons"
                ][
                    "0"
                ][
                    "exact"
                ]
            )
        ]

        resync = mean(
            [
                float(
                    x[
                        "horizons"
                    ][key][
                        "exact"
                    ]
                )
                for x in paired
            ]
        )

        failed0_n = len(
            paired
        )

    else:
        resync = None
        failed0_n = None


    summary[
        "horizons"
    ][key] = {
        "n":
            len(vals),

        "exact_fraction":
            exact_rate,

        "mean_chrf":
            chrf_mean,

        "h0_failed_paired_n":
            failed0_n,

        "resync_fraction_among_h0_failures":
            resync,
    }


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
    "OUTPUT=",
    OUT
)

print(
    "RECOVERY_HORIZON_PROBE_V1_DONE"
)
