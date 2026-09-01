#!/usr/bin/env python3

import csv
import json
import math
import os
import re
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


EXP = "mtpatcher_v3_full6565_20260823"

RUN_ROOT = Path(os.environ["RUN_ROOT"])
MODEL_ROOT = Path(os.environ["MODEL_ROOT"])

BASE = (
    RUN_ROOT
    / EXP
    / "prefix_failure_probe_v1"
)

V3 = (
    BASE
    / "teacher_leg_probe_common26_v3.jsonl"
)

HANDOFF = (
    BASE
    / "teacher_exact_divergence_handoff_v1.jsonl"
)

LOCAL = (
    BASE
    / "local_control_full58.jsonl"
)

OUT_JSONL = (
    BASE
    / "local_trust_signal_audit_v1.jsonl"
)

OUT_CSV = (
    BASE
    / "local_trust_signal_audit_v1.csv"
)

STUDENT_PATH = (
    MODEL_ROOT
    / "Qwen3-0.6B"
)

TEACHER_PATH = (
    MODEL_ROOT
    / "Qwen3-8B"
)

STUDENT_DEVICE = "npu:1"
TEACHER_DEVICE = "npu:0"

PAD_ID = 151643
EOS_ID = 151645

MAX_H = 8


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


def get_prompt_ids(
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


def reconstruct_correction(
    tokenizer,
    row,
):
    draft = str(
        row["student_translation"]
    ).strip()

    fixed = str(
        row["post_edit"]
    ).strip()

    raw_ids = tokenizer(
        draft,
        add_special_tokens=False,
    )["input_ids"]

    fix_ids = tokenizer(
        fixed,
        add_special_tokens=False,
    )["input_ids"]


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
        len(raw_ids)
        - cs
    )

    fix_end = (
        len(fix_ids)
        - cs
    )

    fix_edit = fix_ids[
        cp:fix_end
    ]

    corrected_prefix = (
        raw_ids[:cp]
        + fix_edit
    )


    # Exact provenance assertions.
    if (
        int(row["raw_start_token"]) != cp
        or int(row["raw_end_token"]) != raw_end
        or int(row["fix_start_token"]) != cp
        or int(row["fix_end_token"]) != fix_end
    ):
        raise RuntimeError(
            f"Correction boundary mismatch "
            f"job={row['job_id']}"
        )


    stored_fix = [
        int(x)
        for x
        in row["fix_edit_token_ids"]
    ]

    if stored_fix != fix_edit:
        raise RuntimeError(
            f"Correction token mismatch "
            f"job={row['job_id']}"
        )


    return corrected_prefix


def extract_error_type(
    raw_feedback,
):
    text = str(
        raw_feedback or ""
    )

    patterns = [
        r"Error\s*Type\s*:\s*([^\n\r;]+)",
        r"Error\s*Category\s*:\s*([^\n\r;]+)",
    ]

    for pat in patterns:
        m = re.search(
            pat,
            text,
            flags=re.IGNORECASE,
        )

        if m:
            return (
                m.group(1)
                .strip()
                [:120]
            )

    return "UNKNOWN"


def selected_logits(
    model,
    device,
    input_ids,
    positions,
):
    ids = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=device,
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
                positions,
                :
            ]
            .float()
            .cpu()
        )

    del out
    del ids
    del mask

    return logits


def distribution_features(
    student_logits,
    teacher_logits,
    saved_teacher_token,
):
    s_logp = F.log_softmax(
        student_logits,
        dim=-1,
    )

    t_logp = F.log_softmax(
        teacher_logits,
        dim=-1,
    )

    s_p = s_logp.exp()
    t_p = t_logp.exp()

    m = (
        0.5
        * (
            s_p
            + t_p
        )
    )

    log_m = torch.log(
        m.clamp_min(
            1e-30
        )
    )


    js = float(
        (
            0.5
            * (
                t_p
                * (
                    t_logp
                    - log_m
                )
            ).sum()
            +
            0.5
            * (
                s_p
                * (
                    s_logp
                    - log_m
                )
            ).sum()
        ).item()
    )


    h_s = float(
        (
            -s_p
            * s_logp
        ).sum().item()
    )

    h_t = float(
        (
            -t_p
            * t_logp
        ).sum().item()
    )


    vocab_n = int(
        s_p.numel()
    )

    log_vocab = math.log(
        vocab_n
    )


    s_top = torch.topk(
        s_p,
        k=2,
    )

    t_top = torch.topk(
        t_p,
        k=2,
    )


    s_top1_id = int(
        s_top.indices[0]
    )

    t_top1_id = int(
        t_top.indices[0]
    )


    s_conf = float(
        s_top.values[0]
    )

    t_conf = float(
        t_top.values[0]
    )


    s_margin = float(
        s_top.values[0]
        - s_top.values[1]
    )

    t_margin = float(
        t_top.values[0]
        - t_top.values[1]
    )


    teacher_token = int(
        saved_teacher_token
    )

    p_s_teacher = float(
        s_p[
            teacher_token
        ]
    )

    p_t_teacher = float(
        t_p[
            teacher_token
        ]
    )

    lp_s_teacher = float(
        s_logp[
            teacher_token
        ]
    )

    lp_t_teacher = float(
        t_logp[
            teacher_token
        ]
    )


    return {
        "js":
            js,

        "student_entropy":
            h_s,

        "teacher_entropy":
            h_t,

        "student_entropy_norm":
            h_s
            / log_vocab,

        "teacher_entropy_norm":
            h_t
            / log_vocab,

        "student_conf":
            s_conf,

        "teacher_conf":
            t_conf,

        "student_margin":
            s_margin,

        "teacher_margin":
            t_margin,

        "student_top1_id":
            s_top1_id,

        "teacher_top1_id":
            t_top1_id,

        "top1_agree":
            int(
                s_top1_id
                == t_top1_id
            ),

        "saved_teacher_token":
            teacher_token,

        "teacher_top1_matches_saved":
            int(
                t_top1_id
                == teacher_token
            ),

        "p_student_teacher_token":
            p_s_teacher,

        "p_teacher_teacher_token":
            p_t_teacher,

        "teacher_student_logprob_gap":
            lp_t_teacher
            - lp_s_teacher,

        # Simple literature-inspired
        # diagnostic scores.
        "js_x_teacher_conf":
            js
            * t_conf,

        "js_x_student_entropy":
            js
            * (
                h_s
                / log_vocab
            ),

        "trust_need_score":
            js
            * t_conf
            * (
                h_s
                / log_vocab
            ),
    }


def mean(xs):
    return (
        sum(xs)
        / len(xs)
        if xs
        else float("nan")
    )


def median(xs):
    return (
        statistics.median(xs)
        if xs
        else float("nan")
    )


def average_ranks(xs):
    order = sorted(
        range(len(xs)),
        key=lambda i:
            xs[i],
    )

    ranks = [
        0.0
    ] * len(xs)

    p = 0

    while p < len(xs):
        q = p + 1

        while (
            q < len(xs)
            and xs[
                order[q]
            ]
            == xs[
                order[p]
            ]
        ):
            q += 1

        r = (
            (
                p + 1
                + q
            )
            / 2.0
        )

        for k in range(
            p,
            q,
        ):
            ranks[
                order[k]
            ] = r

        p = q

    return ranks


def pearson(
    xs,
    ys,
):
    if (
        len(xs) < 2
        or len(xs)
        != len(ys)
    ):
        return float("nan")

    mx = mean(xs)
    my = mean(ys)

    num = sum(
        (x - mx)
        * (y - my)
        for x, y
        in zip(
            xs,
            ys,
        )
    )

    dx = math.sqrt(
        sum(
            (x - mx) ** 2
            for x in xs
        )
    )

    dy = math.sqrt(
        sum(
            (y - my) ** 2
            for y in ys
        )
    )

    if (
        dx == 0
        or dy == 0
    ):
        return float("nan")

    return (
        num
        / (
            dx
            * dy
        )
    )


def spearman(
    xs,
    ys,
):
    return pearson(
        average_ranks(xs),
        average_ranks(ys),
    )


def auc(
    scores,
    labels,
):
    pos = [
        s
        for s, y
        in zip(
            scores,
            labels,
        )
        if y
    ]

    neg = [
        s
        for s, y
        in zip(
            scores,
            labels,
        )
        if not y
    ]

    if (
        not pos
        or not neg
    ):
        return float("nan")


    win = 0.0
    total = 0

    for p in pos:
        for n in neg:
            total += 1

            if p > n:
                win += 1.0

            elif p == n:
                win += 0.5

    return (
        win
        / total
    )


# ============================================================
# Load artifacts.
# ============================================================

v3_rows = load_jsonl(
    V3
)

handoff_rows = load_jsonl(
    HANDOFF
)

local_rows = load_jsonl(
    LOCAL
)


if len(v3_rows) != 26:
    raise RuntimeError(
        f"Expected v3 common26, "
        f"got {len(v3_rows)}"
    )

if len(handoff_rows) != 26:
    raise RuntimeError(
        f"Expected handoff26, "
        f"got {len(handoff_rows)}"
    )


v3 = {
    int(x["job_id"]):
        x
    for x in v3_rows
}

handoff = {
    int(x["job_id"]):
        x
    for x in handoff_rows
}

local = {
    int(x["job_id"]):
        x
    for x in local_rows
}


# ============================================================
# Tokenizer compatibility.
# ============================================================

student_tok = (
    AutoTokenizer
    .from_pretrained(
        STUDENT_PATH,
        local_files_only=True,
    )
)

teacher_tok = (
    AutoTokenizer
    .from_pretrained(
        TEACHER_PATH,
        local_files_only=True,
    )
)


if (
    student_tok.get_vocab()
    != teacher_tok.get_vocab()
):
    raise RuntimeError(
        "Student/Teacher vocab mismatch"
    )

if (
    student_tok.chat_template
    != teacher_tok.chat_template
):
    raise RuntimeError(
        "Student/Teacher chat template mismatch"
    )

if (
    student_tok.pad_token_id
    != PAD_ID
    or teacher_tok.pad_token_id
    != PAD_ID
    or student_tok.eos_token_id
    != EOS_ID
    or teacher_tok.eos_token_id
    != EOS_ID
):
    raise RuntimeError(
        "Special-token protocol mismatch"
    )


# ============================================================
# Load models.
# ============================================================

torch.npu.set_device(
    TEACHER_DEVICE
)

print(
    "LOADING_TEACHER",
    flush=True,
)

teacher = (
    AutoModelForCausalLM
    .from_pretrained(
        TEACHER_PATH,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    .to(
        TEACHER_DEVICE
    )
    .eval()
)


torch.npu.set_device(
    STUDENT_DEVICE
)

print(
    "LOADING_STUDENT",
    flush=True,
)

student = (
    AutoModelForCausalLM
    .from_pretrained(
        STUDENT_PATH,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    .to(
        STUDENT_DEVICE
    )
    .eval()
)


records = []
teacher_top1_mismatches = []


# ============================================================
# 26 cases × one forward/model/case.
# ============================================================

for pos, jid in enumerate(
    sorted(v3),
    start=1,
):
    x = v3[jid]
    hx = handoff[jid]
    lx = local[jid]


    corrected_prefix = (
        reconstruct_correction(
            student_tok,
            lx,
        )
    )


    prompt_ids = get_prompt_ids(
        student_tok,
        lx["prompt"],
    )

    teacher_prompt_ids = (
        get_prompt_ids(
            teacher_tok,
            lx["prompt"],
        )
    )

    if (
        prompt_ids
        != teacher_prompt_ids
    ):
        raise RuntimeError(
            f"Prompt IDs mismatch "
            f"job={jid}"
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
    ) < MAX_H:
        raise RuntimeError(
            f"Teacher continuation <8 "
            f"job={jid}"
        )


    # Full sequence contains future teacher tokens,
    # but causal masking guarantees earlier logits
    # cannot see them.
    full_ids = (
        prompt_ids
        + corrected_prefix
        + teacher_cont[
            :MAX_H
        ]
    )


    base_len = (
        len(prompt_ids)
        + len(
            corrected_prefix
        )
    )


    # State after h Teacher tokens:
    # logits at base_len+h-1 predict token h+1.
    positions = [
        base_len
        + h
        - 1
        for h in range(
            0,
            MAX_H
        )
    ]


    s_logits = selected_logits(
        student,
        STUDENT_DEVICE,
        full_ids,
        positions,
    )

    t_logits = selected_logits(
        teacher,
        TEACHER_DEVICE,
        full_ids,
        positions,
    )


    h0_score = float(
        hx[
            "h0_ref_chrf"
        ]
    )


    error_type = (
        extract_error_type(
            lx.get(
                "raw_feedback",
                "",
            )
        )
    )


    for h in range(
        0,
        MAX_H,
    ):
        if h == 0:
            current_score = (
                h0_score
            )
        else:
            current_score = float(
                hx[
                    "horizons"
                ][
                    str(h)
                ][
                    "ref_chrf"
                ]
            )


        next_score = float(
            hx[
                "horizons"
            ][
                str(h + 1)
            ][
                "ref_chrf"
            ]
        )


        marginal_gain = (
            next_score
            - current_score
        )


        feat = (
            distribution_features(
                s_logits[h],
                t_logits[h],
                teacher_cont[h],
            )
        )


        if not feat[
            "teacher_top1_matches_saved"
        ]:
            teacher_top1_mismatches.append(
                (
                    jid,
                    h,
                    feat[
                        "teacher_top1_id"
                    ],
                    teacher_cont[h],
                )
            )


        record = {
            "job_id":
                jid,

            "dataset":
                x["dataset"],

            # h = number of Teacher recovery
            # tokens already injected.
            "h":
                h,

            # next Teacher token is at
            # 1-based distance h+1.
            "next_distance_from_correction":
                h + 1,

            "first_ts_divergence":
                int(
                    hx[
                        "first_ts_divergence"
                    ]
                ),

            "error_type":
                error_type,

            "reference_gain_chrf":
                float(
                    lx.get(
                        "reference_gain_chrf",
                        0.0,
                    )
                ),

            "fix_edit_token_count":
                len(
                    lx[
                        "fix_edit_token_ids"
                    ]
                ),

            "current_ref_chrf":
                current_score,

            "next_ref_chrf":
                next_score,

            # Oracle label:
            # Is ONE more Teacher token
            # useful at this state?
            "marginal_gain_one_teacher_token":
                marginal_gain,

            "benefit_positive":
                int(
                    marginal_gain
                    > 0.0
                ),

            "benefit_gt1":
                int(
                    marginal_gain
                    > 1.0
                ),

            "harm_lt_minus1":
                int(
                    marginal_gain
                    < -1.0
                ),
        }

        record.update(
            feat
        )

        records.append(
            record
        )


    print(
        f"PROGRESS={pos}/26 "
        f"JOB={jid}",
        flush=True,
    )


# ============================================================
# Save.
# ============================================================

OUT_JSONL.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUT_JSONL.open(
    "w",
    encoding="utf-8",
) as f:
    for x in records:
        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )


fieldnames = list(
    records[0].keys()
)

with OUT_CSV.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(
        records
    )


# ============================================================
# Analysis.
# ============================================================

print()
print("=" * 120)
print("ERROR-LOCALIZED TRUST SIGNAL AUDIT")
print("=" * 120)

print(
    f"RECORDS={len(records)}"
)

print(
    f"TEACHER_TOP1_MISMATCHES="
    f"{len(teacher_top1_mismatches)}"
)

if teacher_top1_mismatches:
    print(
        "MISMATCH_DETAIL=",
        teacher_top1_mismatches,
    )


gains = [
    float(
        x[
            "marginal_gain_one_teacher_token"
        ]
    )
    for x in records
]

positive = [
    bool(
        x["benefit_positive"]
    )
    for x in records
]

gt1 = [
    bool(
        x["benefit_gt1"]
    )
    for x in records
]


print()
print(
    "ALL MARGINAL ONE-TOKEN GAINS: "
    f"mean={mean(gains):+.4f} "
    f"median={median(gains):+.4f} "
    f">0={sum(positive)}/{len(records)} "
    f">1={sum(gt1)}/{len(records)} "
    f"<-1="
    f"{sum(x < -1 for x in gains)}/{len(records)}"
)


print()
print("=" * 120)
print("BY DISTANCE FROM LOCALIZED CORRECTION")
print("=" * 120)

for d in range(
    1,
    MAX_H + 1,
):
    xs = [
        x
        for x in records
        if int(
            x[
                "next_distance_from_correction"
            ]
        ) == d
    ]

    gs = [
        float(
            x[
                "marginal_gain_one_teacher_token"
            ]
        )
        for x in xs
    ]

    disagree = sum(
        not bool(
            x[
                "top1_agree"
            ]
        )
        for x in xs
    )

    print(
        f"D={d} "
        f"n={len(xs)} "
        f"gain_mean={mean(gs):+.4f} "
        f"gain_median={median(gs):+.4f} "
        f">+1={sum(g > 1 for g in gs)}/{len(gs)} "
        f"<-1={sum(g < -1 for g in gs)}/{len(gs)} "
        f"TOP1_DISAGREE={disagree}/{len(xs)}"
    )


feature_names = [
    "js",
    "student_entropy_norm",
    "teacher_entropy_norm",
    "student_conf",
    "teacher_conf",
    "student_margin",
    "teacher_margin",
    "p_student_teacher_token",
    "p_teacher_teacher_token",
    "teacher_student_logprob_gap",
    "js_x_teacher_conf",
    "js_x_student_entropy",
    "trust_need_score",
    "next_distance_from_correction",
]


ranking = []

for name in feature_names:
    values = [
        float(
            x[name]
        )
        for x in records
    ]

    rho = spearman(
        values,
        gains,
    )

    a0 = auc(
        values,
        positive,
    )

    a1 = auc(
        values,
        gt1,
    )

    ranking.append(
        (
            name,
            rho,
            a0,
            a1,
        )
    )


ranking.sort(
    key=lambda x:
        (
            -1
            if math.isnan(x[1])
            else abs(x[1])
        ),
    reverse=True,
)


print()
print("=" * 120)
print("SIGNAL RANKING")
print("=" * 120)

print(
    "FEATURE\t"
    "SPEARMAN_WITH_GAIN\t"
    "AUC_GAIN_GT0\t"
    "AUC_GAIN_GT1"
)

for (
    name,
    rho,
    a0,
    a1,
) in ranking:
    print(
        f"{name}\t"
        f"{rho:+.4f}\t"
        f"{a0:.4f}\t"
        f"{a1:.4f}"
    )


print()
print("=" * 120)
print("TOP1 AGREEMENT CONDITIONAL")
print("=" * 120)

for agree in [
    1,
    0,
]:
    xs = [
        float(
            x[
                "marginal_gain_one_teacher_token"
            ]
        )
        for x in records
        if int(
            x[
                "top1_agree"
            ]
        ) == agree
    ]

    print(
        f"TOP1_AGREE={agree} "
        f"n={len(xs)} "
        f"mean={mean(xs):+.4f} "
        f"median={median(xs):+.4f} "
        f">+1={sum(x > 1 for x in xs)}/{len(xs)} "
        f"<-1={sum(x < -1 for x in xs)}/{len(xs)}"
    )


print()
print(
    f"JSONL={OUT_JSONL}"
)

print(
    f"CSV={OUT_CSV}"
)

print(
    "LOCAL_TRUST_SIGNAL_AUDIT_V1_PASS"
)
