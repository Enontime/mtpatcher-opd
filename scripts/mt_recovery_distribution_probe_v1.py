#!/usr/bin/env python3

import json
import math
import statistics
from pathlib import Path

import torch
import torch_npu
import torch.nn.functional as F

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


EXP = "mtpatcher_v3_full6565_20260823"

BASE = Path(
    "/workspace/mtpatcher"
)

HELD = (
    BASE
    / "data"
    / EXP
    / "patcher_quality_gate_v1"
)

LOCAL = (
    BASE
    / "runs"
    / EXP
    / "prefix_failure_probe_v1"
    / "local_control_full58.jsonl"
)

RH = (
    BASE
    / "runs"
    / EXP
    / "prefix_failure_probe_v1"
    / "recovery_horizon_local40_v1.jsonl"
)

OUT = (
    BASE
    / "runs"
    / EXP
    / "prefix_failure_probe_v1"
    / "recovery_distribution_common26_v1.jsonl"
)

MODEL = (
    BASE
    / "models"
    / "Qwen3-0.6B"
)

DEVICE = "npu:0"

HORIZONS = [
    0,
    1,
    2,
    4,
    8,
]


def load(path):
    with Path(path).open(
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

    return (
        sum(xs)
        / len(xs)
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


# ------------------------------------------------------------
# Use exactly the 26 examples evaluable at all
# behavioral recovery horizons.
# ------------------------------------------------------------

rh_rows = load(RH)

common_ids = {
    int(x["job_id"])
    for x in rh_rows
    if all(
        str(h)
        in x["horizons"]
        for h in HORIZONS
    )
}

local = {
    int(x["job_id"]): x
    for x in load(LOCAL)
}

jobs = {
    int(x["job_id"]): x
    for x in load(
        HELD
        / "student_jobs.jsonl"
    )
}


print("=" * 100)
print("RECOVERY DISTRIBUTION PROBE V1")
print("=" * 100)

print(
    "COMMON_IDS=",
    len(common_ids),
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


def prompt_ids_for(
    raw_prompt,
):
    text = qwen_format(
        tok,
        raw_prompt,
    )

    return tok(
        text,
        add_special_tokens=False,
    )[
        "input_ids"
    ]


def next_logits(
    context_ids,
):
    ids = torch.tensor(
        [
            context_ids
        ],
        dtype=torch.long,
        device=DEVICE,
    )

    mask = torch.ones_like(
        ids
    )

    with torch.inference_mode():
        out = model(
            input_ids=ids,
            attention_mask=mask,
            use_cache=False,
        )

    logits = (
        out.logits[
            0,
            -1,
            :
        ]
        .float()
        .cpu()
    )

    del out

    return logits


def compare_distributions(
    raw_logits,
    fix_logits,
    desired_id,
):
    logp = F.log_softmax(
        raw_logits,
        dim=-1,
    )

    logq = F.log_softmax(
        fix_logits,
        dim=-1,
    )

    p = logp.exp()
    q = logq.exp()

    m = (
        0.5
        * (
            p + q
        )
    )

    logm = torch.log(
        m.clamp_min(
            1e-30
        )
    )

    js = float(
        (
            0.5
            * torch.sum(
                p
                * (
                    logp
                    - logm
                )
            )
            +
            0.5
            * torch.sum(
                q
                * (
                    logq
                    - logm
                )
            )
        ).item()
    )

    kl_raw_fix = float(
        torch.sum(
            p
            * (
                logp
                - logq
            )
        ).item()
    )

    kl_fix_raw = float(
        torch.sum(
            q
            * (
                logq
                - logp
            )
        ).item()
    )

    raw_top1 = int(
        torch.argmax(
            raw_logits
        ).item()
    )

    fix_top1 = int(
        torch.argmax(
            fix_logits
        ).item()
    )

    raw_top5 = set(
        torch.topk(
            raw_logits,
            k=5,
        ).indices.tolist()
    )

    fix_top5 = set(
        torch.topk(
            fix_logits,
            k=5,
        ).indices.tolist()
    )

    top5_overlap = (
        len(
            raw_top5
            & fix_top5
        )
        / 5.0
    )

    raw_entropy = float(
        (
            -torch.sum(
                p * logp
            )
        ).item()
    )

    fix_entropy = float(
        (
            -torch.sum(
                q * logq
            )
        ).item()
    )

    raw_desired_logp = float(
        logp[
            desired_id
        ].item()
    )

    fix_desired_logp = float(
        logq[
            desired_id
        ].item()
    )

    return {
        "js":
            js,

        "kl_raw_fix":
            kl_raw_fix,

        "kl_fix_raw":
            kl_fix_raw,

        "top1_same":
            (
                raw_top1
                == fix_top1
            ),

        "top5_overlap":
            top5_overlap,

        "raw_entropy":
            raw_entropy,

        "fix_entropy":
            fix_entropy,

        "raw_desired_logp":
            raw_desired_logp,

        "fix_desired_logp":
            fix_desired_logp,

        "desired_logp_delta_fix_minus_raw":
            (
                fix_desired_logp
                - raw_desired_logp
            ),
    }


results = []


for pos, jid in enumerate(
    sorted(
        common_ids
    ),
    start=1,
):
    x = local[jid]

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

    raw_ids = tok(
        draft,
        add_special_tokens=False,
    )[
        "input_ids"
    ]

    fix_ids = tok(
        fixed,
        add_special_tokens=False,
    )[
        "input_ids"
    ]


    # --------------------------------------------------------
    # Minimal token replacement.
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
        raise RuntimeError(
            f"unexpected insert/delete "
            f"for job {jid}"
        )


    suffix = (
        raw_ids[
            raw_end:
        ]
    )

    fix_suffix = (
        fix_ids[
            fix_end:
        ]
    )

    if suffix != fix_suffix:
        raise RuntimeError(
            f"local-only suffix mismatch "
            f"for job {jid}"
        )


    if len(suffix) <= 8:
        raise RuntimeError(
            f"suffix too short "
            f"for job {jid}"
        )


    fix_edit = (
        fix_ids[
            cp:
            fix_end
        ]
    )


    raw_base = (
        raw_ids[
            :raw_end
        ]
    )

    fix_base = (
        raw_ids[
            :cp
        ]
        + fix_edit
    )


    pids = prompt_ids_for(
        jobs[jid][
            "prompt"
        ]
    )


    row = {
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
                suffix
            ),

        "horizons":
            {},
    }


    for h in HORIZONS:
        desired_id = int(
            suffix[h]
        )

        anchor = (
            suffix[
                :h
            ]
        )

        raw_context = (
            pids
            + raw_base
            + anchor
        )

        fix_context = (
            pids
            + fix_base
            + anchor
        )


        raw_logits = next_logits(
            raw_context
        )

        fix_logits = next_logits(
            fix_context
        )


        stats = (
            compare_distributions(
                raw_logits,
                fix_logits,
                desired_id,
            )
        )

        stats[
            "desired_token_id"
        ] = desired_id

        stats[
            "desired_token"
        ] = tok.decode(
            [
                desired_id
            ],
            skip_special_tokens=True,
        )

        row[
            "horizons"
        ][str(h)] = stats


    results.append(
        row
    )


    if (
        pos % 5 == 0
        or pos
        == len(common_ids)
    ):
        print(
            f"PROGRESS="
            f"{pos}/"
            f"{len(common_ids)}",
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
        "MT_RECOVERY_DISTRIBUTION_COMMON26_V1",

    "n":
        len(results),

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
    ]

    js = [
        y["js"]
        for y in vals
    ]

    logp_delta = [
        y[
            "desired_logp_delta_fix_minus_raw"
        ]
        for y in vals
    ]

    top1 = [
        float(
            y[
                "top1_same"
            ]
        )
        for y in vals
    ]

    top5 = [
        y[
            "top5_overlap"
        ]
        for y in vals
    ]


    if h == 0:
        js_ratio = None
        below_h0 = None
        below_half = None

    else:
        ratios = []
        below = []
        half = []

        for x in results:
            j0 = float(
                x[
                    "horizons"
                ][
                    "0"
                ][
                    "js"
                ]
            )

            jh = float(
                x[
                    "horizons"
                ][key][
                    "js"
                ]
            )

            if j0 > 1e-12:
                ratios.append(
                    jh / j0
                )

            below.append(
                float(
                    jh < j0
                )
            )

            half.append(
                float(
                    jh
                    < 0.5 * j0
                )
            )

        js_ratio = mean(
            ratios
        )

        below_h0 = mean(
            below
        )

        below_half = mean(
            half
        )


    summary[
        "horizons"
    ][key] = {
        "js_mean":
            mean(
                js
            ),

        "js_median":
            statistics.median(
                js
            ),

        "mean_js_ratio_to_h0":
            js_ratio,

        "fraction_js_below_h0":
            below_h0,

        "fraction_js_below_half_h0":
            below_half,

        "top1_agreement":
            mean(
                top1
            ),

        "top5_overlap_mean":
            mean(
                top5
            ),

        "desired_logp_delta_fix_minus_raw_mean":
            mean(
                logp_delta
            ),

        "desired_logp_delta_fix_minus_raw_median":
            statistics.median(
                logp_delta
            ),
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
    "RECOVERY_DISTRIBUTION_PROBE_V1_DONE"
)
