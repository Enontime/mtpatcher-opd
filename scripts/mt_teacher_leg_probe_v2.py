#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path

import sacrebleu
import torch
import torch_npu

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


EXP = "mtpatcher_v3_full6565_20260823"

ROOT = Path("/workspace/mtpatcher")

LOCAL = (
    ROOT
    / "runs"
    / EXP
    / "prefix_failure_probe_v1"
    / "local_control_full58.jsonl"
)

RH = (
    ROOT
    / "runs"
    / EXP
    / "prefix_failure_probe_v1"
    / "recovery_horizon_local40_v1.jsonl"
)

STUDENT_MODEL = (
    ROOT
    / "models"
    / "Qwen3-0.6B"
)

TEACHER_MODEL = (
    ROOT
    / "models"
    / "Qwen3-8B"
)

STUDENT_DEVICE = "npu:1"
TEACHER_DEVICE = "npu:0"

MAX_RESPONSE_TOKENS = 256
HORIZONS = [2, 4, 8]


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
    tok,
    prompt,
):
    return tok.apply_chat_template(
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
    tok,
    prompt,
):
    rendered = qwen_format(
        tok,
        prompt,
    )

    return tok(
        rendered,
        add_special_tokens=False,
    )["input_ids"]


def generate_new_tokens(
    model,
    device,
    prompt_ids,
    forced,
):
    remaining = (
        MAX_RESPONSE_TOKENS
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
        device=device,
    )

    attention_mask = torch.ones_like(
        input_ids
    )

    width = input_ids.shape[1]

    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=remaining,
            pad_token_id=151643,
            eos_token_id=151645,
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


def trim_at_eos(
    ids,
    eos_id,
):
    ids = list(ids)

    if eos_id in ids:
        p = ids.index(
            eos_id
        )

        return (
            ids[:p],
            True,
            p,
        )

    return (
        ids,
        False,
        None,
    )


def chrf(
    hyp,
    ref,
):
    return float(
        sacrebleu.sentence_chrf(
            hyp.strip(),
            [ref.strip()],
        ).score
    )


def lcp_len(
    a,
    b,
):
    n = min(
        len(a),
        len(b),
    )

    i = 0

    while (
        i < n
        and a[i] == b[i]
    ):
        i += 1

    return i


parser = argparse.ArgumentParser()

parser.add_argument(
    "--limit",
    type=int,
    default=0,
)

parser.add_argument(
    "--output",
    type=Path,
    required=True,
)

args = parser.parse_args()


random.seed(20260827)
torch.manual_seed(20260827)


student_tok = (
    AutoTokenizer
    .from_pretrained(
        STUDENT_MODEL,
        local_files_only=True,
    )
)

teacher_tok = (
    AutoTokenizer
    .from_pretrained(
        TEACHER_MODEL,
        local_files_only=True,
    )
)

student_tok.padding_side = "left"
teacher_tok.padding_side = "left"


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
    student_tok.eos_token_id
    != teacher_tok.eos_token_id
):
    raise RuntimeError(
        "Student/Teacher EOS mismatch"
    )


# Freeze generation special-token protocol to the already audited
# Qwen generator protocol. Model config may leave pad_token_id unset.
student_pad_id = student_tok.pad_token_id
student_eos_id = student_tok.eos_token_id
teacher_pad_id = teacher_tok.pad_token_id
teacher_eos_id = teacher_tok.eos_token_id

if (
    student_pad_id != 151643
    or teacher_pad_id != 151643
    or student_eos_id != 151645
    or teacher_eos_id != 151645
):
    raise RuntimeError(
        "Unexpected audited special-token IDs"
    )


print(
    "LOADING_TEACHER",
    flush=True,
)

teacher = (
    AutoModelForCausalLM
    .from_pretrained(
        TEACHER_MODEL,
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

print(
    "LOADING_STUDENT",
    flush=True,
)

student = (
    AutoModelForCausalLM
    .from_pretrained(
        STUDENT_MODEL,
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


local_rows = {
    int(x["job_id"]):
        x
    for x in load_jsonl(
        LOCAL
    )
    if x.get(
        "valid_for_causal",
        True,
    )
}

rh_rows = load_jsonl(
    RH
)


# Primary diagnostic population:
# exactly the previously defined common26.
common = [
    x
    for x in rh_rows
    if all(
        str(h)
        in x.get(
            "horizons",
            {},
        )
        for h in [
            0,
            1,
            2,
            4,
            8,
        ]
    )
]

if len(common) != 26:
    raise RuntimeError(
        f"Expected common26, "
        f"got {len(common)}"
    )

if args.limit > 0:
    common = common[
        :args.limit
    ]


results = []


for pos, rh in enumerate(
    common,
    start=1,
):
    jid = int(
        rh["job_id"]
    )

    x = local_rows[jid]

    draft = str(
        x["student_translation"]
    ).strip()

    fixed = str(
        x["post_edit"]
    ).strip()

    reference = str(
        x["reference"]
    ).strip()

    prompt = str(
        x["prompt"]
    )


    raw_ids = student_tok(
        draft,
        add_special_tokens=False,
    )["input_ids"]

    fix_ids = student_tok(
        fixed,
        add_special_tokens=False,
    )["input_ids"]


    # --------------------------------------------------------
    # Reconstruct the exact existing Local-control replacement.
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


    if (
        cp == raw_end
        or cp == fix_end
    ):
        raise RuntimeError(
            f"Unexpected insert/delete "
            f"job={jid}"
        )


    expected_suffix = raw_ids[
        raw_end:
    ]

    fixed_suffix = fix_ids[
        fix_end:
    ]


    if (
        expected_suffix
        != fixed_suffix
    ):
        raise RuntimeError(
            f"Local-only suffix invariant "
            f"failed job={jid}"
        )


    fix_edit = fix_ids[
        cp:fix_end
    ]

    corrected_prefix = (
        raw_ids[:cp]
        + fix_edit
    )


    # --------------------------------------------------------
    # Assert against the already frozen Local artifact.
    # --------------------------------------------------------

    if int(
        x["raw_start_token"]
    ) != cp:
        raise RuntimeError(
            f"raw_start mismatch job={jid}"
        )

    if int(
        x["raw_end_token"]
    ) != raw_end:
        raise RuntimeError(
            f"raw_end mismatch job={jid}"
        )

    if int(
        x["fix_start_token"]
    ) != cp:
        raise RuntimeError(
            f"fix_start mismatch job={jid}"
        )

    if int(
        x["fix_end_token"]
    ) != fix_end:
        raise RuntimeError(
            f"fix_end mismatch job={jid}"
        )

    if [
        int(v)
        for v in x[
            "fix_edit_token_ids"
        ]
    ] != fix_edit:
        raise RuntimeError(
            f"fix_edit IDs mismatch job={jid}"
        )


    student_prompt_ids = get_prompt_ids(
        student_tok,
        prompt,
    )

    teacher_prompt_ids = get_prompt_ids(
        teacher_tok,
        prompt,
    )


    if (
        student_prompt_ids
        != teacher_prompt_ids
    ):
        raise RuntimeError(
            f"Prompt token mismatch job={jid}"
        )


    # --------------------------------------------------------
    # Replay h0.
    # --------------------------------------------------------

    h0_raw = generate_new_tokens(
        student,
        STUDENT_DEVICE,
        student_prompt_ids,
        corrected_prefix,
    )

    h0_generated, _, _ = trim_at_eos(
        h0_raw,
        student_tok.eos_token_id,
    )

    h0_generated_text = student_tok.decode(
        h0_generated,
        skip_special_tokens=True,
    )


    if (
        h0_generated_text
        != rh[
            "horizons"
        ][
            "0"
        ][
            "generated"
        ]
    ):
        raise RuntimeError(
            f"h0 replay mismatch job={jid}"
        )


    h0_full_ids = (
        corrected_prefix
        + h0_generated
    )

    h0_full_text = student_tok.decode(
        h0_full_ids,
        skip_special_tokens=True,
    ).strip()

    h0_ref_chrf = chrf(
        h0_full_text,
        reference,
    )


    # --------------------------------------------------------
    # Replay Common-2/4/8 before introducing Teacher.
    # --------------------------------------------------------

    common_results = {}

    for h in HORIZONS:

        anchor = expected_suffix[
            :h
        ]

        common_raw = generate_new_tokens(
            student,
            STUDENT_DEVICE,
            student_prompt_ids,
            corrected_prefix
            + anchor,
        )

        common_generated, _, _ = trim_at_eos(
            common_raw,
            student_tok.eos_token_id,
        )

        common_generated_text = (
            student_tok.decode(
                common_generated,
                skip_special_tokens=True,
            )
        )


        if (
            common_generated_text
            != rh[
                "horizons"
            ][
                str(h)
            ][
                "generated"
            ]
        ):
            raise RuntimeError(
                f"Common-{h} replay mismatch "
                f"job={jid}"
            )


        common_full_ids = (
            corrected_prefix
            + anchor
            + common_generated
        )

        common_full_text = (
            student_tok.decode(
                common_full_ids,
                skip_special_tokens=True,
            )
            .strip()
        )

        common_results[
            str(h)
        ] = {
            "anchor_ids":
                anchor,

            "full":
                common_full_text,

            "ref_chrf":
                chrf(
                    common_full_text,
                    reference,
                ),
        }


    # --------------------------------------------------------
    # ONE Teacher trajectory.
    #
    # Teacher-2/4/8 are nested prefixes from this same rollout.
    # --------------------------------------------------------

    teacher_raw = generate_new_tokens(
        teacher,
        TEACHER_DEVICE,
        teacher_prompt_ids,
        corrected_prefix,
    )

    teacher_cont, teacher_eos, teacher_eos_pos = (
        trim_at_eos(
            teacher_raw,
            teacher_tok.eos_token_id,
        )
    )


    teacher_full_ids = (
        corrected_prefix
        + teacher_cont
    )

    teacher_full_text = (
        teacher_tok.decode(
            teacher_full_ids,
            skip_special_tokens=True,
        )
        .strip()
    )

    teacher_full_ref_chrf = chrf(
        teacher_full_text,
        reference,
    )


    teacher_h_results = {}


    for h in HORIZONS:

        # Teacher ended before the requested handoff point.
        if len(
            teacher_cont
        ) < h:

            final_ids = (
                corrected_prefix
                + teacher_cont
            )

            final_text = (
                student_tok.decode(
                    final_ids,
                    skip_special_tokens=True,
                )
                .strip()
            )

            bridge = list(
                teacher_cont
            )

            student_resumed = False

        else:
            bridge = teacher_cont[
                :h
            ]

            student_raw = generate_new_tokens(
                student,
                STUDENT_DEVICE,
                student_prompt_ids,
                corrected_prefix
                + bridge,
            )

            student_generated, _, _ = (
                trim_at_eos(
                    student_raw,
                    student_tok.eos_token_id,
                )
            )

            final_ids = (
                corrected_prefix
                + bridge
                + student_generated
            )

            final_text = (
                student_tok.decode(
                    final_ids,
                    skip_special_tokens=True,
                )
                .strip()
            )

            student_resumed = True


        common_anchor = expected_suffix[
            :h
        ]

        teacher_h_results[
            str(h)
        ] = {
            "bridge_ids":
                bridge,

            "bridge_text":
                student_tok.decode(
                    bridge,
                    skip_special_tokens=True,
                ),

            "bridge_exact_common":
                bridge
                == common_anchor,

            "bridge_lcp_common":
                lcp_len(
                    bridge,
                    common_anchor,
                ),

            "student_resumed":
                student_resumed,

            "full":
                final_text,

            "ref_chrf":
                chrf(
                    final_text,
                    reference,
                ),

            "delta_vs_h0":
                chrf(
                    final_text,
                    reference,
                )
                - h0_ref_chrf,

            "delta_vs_common":
                chrf(
                    final_text,
                    reference,
                )
                - common_results[
                    str(h)
                ][
                    "ref_chrf"
                ],
        }


    result = {
        "job_id":
            jid,

        "dataset":
            x["dataset"],

        "source":
            x["source"],

        "reference":
            reference,

        "student_translation":
            draft,

        "post_edit":
            fixed,

        "corrected_prefix_text":
            student_tok.decode(
                corrected_prefix,
                skip_special_tokens=True,
            ),

        "h0": {
            "full":
                h0_full_text,

            "ref_chrf":
                h0_ref_chrf,
        },

        "common":
            common_results,

        "teacher": {
            "continuation_text":
                teacher_tok.decode(
                    teacher_cont,
                    skip_special_tokens=True,
                ),

            "ended_with_eos":
                teacher_eos,

            "eos_position":
                teacher_eos_pos,

            "full":
                teacher_full_text,

            "full_ref_chrf":
                teacher_full_ref_chrf,

            "horizons":
                teacher_h_results,
        },
    }

    results.append(
        result
    )


    print()
    print("=" * 100)
    print(
        f"JOB={jid} "
        f"DATASET={x['dataset']}"
    )
    print("=" * 100)

    print(
        "CORRECTED_PREFIX:",
        result[
            "corrected_prefix_text"
        ],
    )

    print(
        "H0:",
        f"{h0_ref_chrf:.3f}",
        h0_full_text,
    )

    for h in HORIZONS:
        c = common_results[
            str(h)
        ]

        t = teacher_h_results[
            str(h)
        ]

        print()
        print(
            f"h={h} "
            f"COMMON={c['ref_chrf']:.3f} "
            f"TEACHER={t['ref_chrf']:.3f} "
            f"ΔT-H0={t['delta_vs_h0']:+.3f} "
            f"ΔT-C={t['delta_vs_common']:+.3f}"
        )

        print(
            "  BRIDGE:",
            repr(
                t[
                    "bridge_text"
                ]
            ),
        )

        print(
            "  BRIDGE_EXACT_COMMON=",
            t[
                "bridge_exact_common"
            ],
            "LCP=",
            t[
                "bridge_lcp_common"
            ],
            "RESUMED=",
            t[
                "student_resumed"
            ],
        )

        print(
            "  FINAL:",
            t[
                "full"
            ],
        )

    print()
    print(
        "TEACHER_FULL:",
        f"{teacher_full_ref_chrf:.3f}",
        teacher_full_text,
    )


args.output.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with args.output.open(
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
print(
    f"VALID_ROWS={len(results)}"
)

print(
    f"OUTPUT={args.output}"
)

print(
    "TEACHER_LEG_PROBE_SMOKE_PASS"
)
