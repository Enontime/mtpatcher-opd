#!/usr/bin/env python3

import json
import os
import statistics
from pathlib import Path

import sacrebleu
import torch
import torch_npu

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


EXP = "mtpatcher_v3_full6565_20260823"

RUN_ROOT = Path(os.environ["RUN_ROOT"])
MODEL_ROOT = Path(os.environ["MODEL_ROOT"])

INPUT = (
    RUN_ROOT
    / EXP
    / "prefix_failure_probe_v1"
    / "teacher_leg_probe_common26_v3.jsonl"
)

LOCAL = (
    RUN_ROOT
    / EXP
    / "prefix_failure_probe_v1"
    / "local_control_full58.jsonl"
)

OUT = (
    RUN_ROOT
    / EXP
    / "prefix_failure_probe_v1"
    / "teacher_exact_divergence_handoff_v1.jsonl"
)

MODEL = (
    MODEL_ROOT
    / "Qwen3-0.6B"
)

DEVICE = "npu:0"

MAX_RESPONSE_TOKENS = 256
PAD_ID = 151643
EOS_ID = 151645


def load_jsonl(path):
    with path.open(
        encoding="utf-8",
    ) as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


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


def prompt_ids(
    tokenizer,
    prompt,
):
    rendered = qwen_format(
        tokenizer,
        prompt,
    )

    return tokenizer(
        rendered,
        add_special_tokens=False,
    )["input_ids"]


def generate_from(
    model,
    prompt,
    forced,
):
    remaining = (
        MAX_RESPONSE_TOKENS
        - len(forced)
    )

    if remaining <= 0:
        return []

    combined = (
        prompt
        + forced
    )

    ids = torch.tensor(
        [combined],
        dtype=torch.long,
        device=DEVICE,
    )

    mask = torch.ones_like(
        ids
    )

    width = ids.shape[1]

    with torch.inference_mode():
        out = model.generate(
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=remaining,
            pad_token_id=PAD_ID,
            eos_token_id=EOS_ID,
            do_sample=False,
        )

    return (
        out[
            0,
            width:
        ]
        .detach()
        .cpu()
        .tolist()
    )


def trim_eos(ids):
    ids = list(ids)

    if EOS_ID in ids:
        ids = ids[
            :ids.index(EOS_ID)
        ]

    return ids


def score_chrf(
    hyp,
    ref,
):
    return float(
        sacrebleu.sentence_chrf(
            hyp.strip(),
            [ref.strip()],
        ).score
    )


def mean(xs):
    return (
        sum(xs) / len(xs)
        if xs
        else float("nan")
    )


def median(xs):
    return (
        statistics.median(xs)
        if xs
        else float("nan")
    )


def summarize(
    name,
    xs,
):
    print(
        f"{name}: "
        f"n={len(xs)} "
        f"mean={mean(xs):+.4f} "
        f"median={median(xs):+.4f} "
        f">+1={sum(x > 1 for x in xs)}/{len(xs)} "
        f"<-1={sum(x < -1 for x in xs)}/{len(xs)}"
    )


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

if (
    tok.pad_token_id != PAD_ID
    or tok.eos_token_id != EOS_ID
):
    raise RuntimeError(
        "Unexpected tokenizer special IDs"
    )


print(
    "LOADING_STUDENT",
    flush=True,
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
    .to(DEVICE)
    .eval()
)


rows = load_jsonl(
    INPUT
)

if len(rows) != 26:
    raise RuntimeError(
        f"Expected 26 rows, got {len(rows)}"
    )


local = {
    int(x["job_id"]): x
    for x in load_jsonl(
        LOCAL
    )
}


results = []


for pos, x in enumerate(
    rows,
    start=1,
):
    jid = int(
        x["job_id"]
    )

    lx = local[jid]

    draft = str(
        lx["student_translation"]
    ).strip()

    fixed = str(
        lx["post_edit"]
    ).strip()

    reference = str(
        lx["reference"]
    ).strip()

    raw_ids = tok(
        draft,
        add_special_tokens=False,
    )["input_ids"]

    fix_ids = tok(
        fixed,
        add_special_tokens=False,
    )["input_ids"]


    # Exact existing correction reconstruction.
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
            len(raw_ids) - 1 - cs
        ]
        == fix_ids[
            len(fix_ids) - 1 - cs
        ]
    ):
        cs += 1


    raw_end = (
        len(raw_ids) - cs
    )

    fix_end = (
        len(fix_ids) - cs
    )

    fix_edit = fix_ids[
        cp:fix_end
    ]

    corrected_prefix = (
        raw_ids[:cp]
        + fix_edit
    )


    # Frozen Local artifact assertions.
    if (
        int(lx["raw_start_token"]) != cp
        or int(lx["raw_end_token"]) != raw_end
        or int(lx["fix_start_token"]) != cp
        or int(lx["fix_end_token"]) != fix_end
        or [
            int(v)
            for v
            in lx["fix_edit_token_ids"]
        ] != fix_edit
    ):
        raise RuntimeError(
            f"Correction provenance mismatch "
            f"job={jid}"
        )


    pids = prompt_ids(
        tok,
        lx["prompt"],
    )


    # --------------------------------------------------------
    # Replay H0 exactly.
    # --------------------------------------------------------
    new_h0_raw = generate_from(
        model,
        pids,
        corrected_prefix,
    )

    stored_h0_raw = [
        int(v)
        for v
        in x[
            "h0"
        ][
            "raw_generated_ids"
        ]
    ]

    if (
        new_h0_raw
        != stored_h0_raw
    ):
        raise RuntimeError(
            f"H0 raw replay mismatch job={jid}"
        )


    h0_text = x[
        "h0"
    ][
        "full"
    ]

    h0_score = float(
        x[
            "h0"
        ][
            "ref_chrf"
        ]
    )


    teacher_cont = [
        int(v)
        for v
        in x[
            "teacher"
        ][
            "continuation_ids_no_eos"
        ]
    ]

    if len(
        teacher_cont
    ) < 8:
        raise RuntimeError(
            f"Teacher continuation <8 "
            f"job={jid}"
        )


    d = x[
        "online_teacher_student"
    ][
        "first_divergence_position_1based"
    ]

    if d is None:
        raise RuntimeError(
            f"Unexpected no divergence "
            f"job={jid}"
        )

    d = int(d)


    h_results = {}


    for h in range(
        1,
        9,
    ):
        bridge = teacher_cont[
            :h
        ]

        generated_raw = generate_from(
            model,
            pids,
            corrected_prefix
            + bridge,
        )

        generated = trim_eos(
            generated_raw
        )

        final_ids = (
            corrected_prefix
            + bridge
            + generated
        )

        final_text = (
            tok.decode(
                final_ids,
                skip_special_tokens=True,
            )
            .strip()
        )

        score = score_chrf(
            final_text,
            reference,
        )

        h_results[
            str(h)
        ] = {
            "full":
                final_text,

            "ref_chrf":
                score,

            "delta_vs_h0":
                score - h0_score,
        }


        # Before first Teacher/Student disagreement,
        # Teacher prefix is exactly what Student would have
        # generated, so deterministic continuation MUST equal H0.
        if h < d:
            if (
                final_text
                != h0_text
            ):
                raise RuntimeError(
                    f"Pre-divergence invariant "
                    f"failed job={jid} "
                    f"h={h} d={d}"
                )


    # --------------------------------------------------------
    # Existing T2/T4/T8 exact replay gates.
    # --------------------------------------------------------
    for h in [
        2,
        4,
        8,
    ]:
        old = x[
            "teacher"
        ][
            "horizons"
        ][
            str(h)
        ][
            "full"
        ]

        new = h_results[
            str(h)
        ][
            "full"
        ]

        if old != new:
            raise RuntimeError(
                f"T{h} replay mismatch "
                f"job={jid}"
            )


    result = {
        "job_id":
            jid,

        "dataset":
            x["dataset"],

        "first_ts_divergence":
            d,

        "h0_ref_chrf":
            h0_score,

        "teacher_full_ref_chrf":
            float(
                x[
                    "teacher"
                ][
                    "full_ref_chrf"
                ]
            ),

        "horizons":
            h_results,
    }

    results.append(
        result
    )


    print(
        f"JOB={jid} "
        f"D={d} "
        f"T_D={h_results[str(min(d,8))]['delta_vs_h0']:+.3f} "
        f"T8={h_results['8']['delta_vs_h0']:+.3f}",
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


print()
print("=" * 110)
print("EXACT FIRST-DIVERGENCE HANDOFF ANALYSIS")
print("=" * 110)


groups = {
    "by2": [],
    "by4": [],
    "by8": [],
    "after8": [],
}

for x in results:

    d = x[
        "first_ts_divergence"
    ]

    if d <= 2:
        groups["by2"].append(x)

    elif d <= 4:
        groups["by4"].append(x)

    elif d <= 8:
        groups["by8"].append(x)

    else:
        groups[
            "after8"
        ].append(x)


print(
    "GROUP_COUNTS=",
    {
        k: len(v)
        for k, v
        in groups.items()
    },
)

print()


for name, xs in groups.items():

    print("=" * 110)
    print(
        f"GROUP={name} "
        f"N={len(xs)}"
    )
    print("=" * 110)

    if not xs:
        continue


    if name != "after8":

        td = [
            float(
                x[
                    "horizons"
                ][
                    str(
                        x[
                            "first_ts_divergence"
                        ]
                    )
                ][
                    "delta_vs_h0"
                ]
            )
            for x in xs
        ]

        summarize(
            "T_AT_FIRST_DIVERGENCE - H0",
            td,
        )


        t8_minus_td = [
            float(
                x[
                    "horizons"
                ][
                    "8"
                ][
                    "ref_chrf"
                ]
            )
            -
            float(
                x[
                    "horizons"
                ][
                    str(
                        x[
                            "first_ts_divergence"
                        ]
                    )
                ][
                    "ref_chrf"
                ]
            )
            for x in xs
        ]

        summarize(
            "CONTINUE_FIRST_DIVERGENCE_TO_8",
            t8_minus_td,
        )


    for x in xs:

        d = x[
            "first_ts_divergence"
        ]

        vals = []

        start = min(
            d,
            8,
        )

        for h in range(
            start,
            9,
        ):
            vals.append(
                f"T{h}="
                f"{x['horizons'][str(h)]['delta_vs_h0']:+.3f}"
            )

        print(
            f"JOB={x['job_id']} "
            f"D={d} "
            + " ".join(vals)
        )

    print()


print("=" * 110)
print("MARGINAL CONTINUATION AFTER FIRST DIVERGENCE")
print("=" * 110)

for offset in [
    1,
    2,
    3,
    4,
]:

    vals = []

    for x in results:

        d = x[
            "first_ts_divergence"
        ]

        h = (
            d + offset
        )

        if (
            d <= 8
            and h <= 8
        ):
            base = float(
                x[
                    "horizons"
                ][
                    str(d)
                ][
                    "ref_chrf"
                ]
            )

            later = float(
                x[
                    "horizons"
                ][
                    str(h)
                ][
                    "ref_chrf"
                ]
            )

            vals.append(
                later - base
            )

    summarize(
        f"T_D+{offset} - T_D",
        vals,
    )


print()
print(
    f"OUTPUT={OUT}"
)

print(
    "EXACT_FIRST_DIVERGENCE_HANDOFF_PROBE_PASS"
)
