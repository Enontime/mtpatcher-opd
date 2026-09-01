#!/usr/bin/env python3

import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path

from transformers import AutoTokenizer


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

SIGNAL = (
    BASE
    / "local_trust_signal_audit_v1.jsonl"
)

OUT_MD = (
    BASE
    / "teacher_semantic_event_audit_v1.md"
)

OUT_JSONL = (
    BASE
    / "teacher_semantic_event_audit_v1.jsonl"
)

TOKENIZER = (
    MODEL_ROOT
    / "Qwen3-0.6B"
)


def load_jsonl(path):
    with path.open(
        encoding="utf-8",
    ) as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def lexical_tokens(text):
    # English-target-oriented display tokenizer.
    # Only used for human-readable edit visualization.
    return re.findall(
        r"""
        [A-Za-z0-9]+
        (?:['’\-][A-Za-z0-9]+)*
        |
        [^\w\s]
        """,
        text,
        flags=re.VERBOSE,
    )


def join_tokens(xs):
    if not xs:
        return ""

    s = " ".join(xs)

    # Human-readable punctuation cleanup only.
    s = re.sub(
        r"\s+([,.;:!?%\)])",
        r"\1",
        s,
    )

    s = re.sub(
        r"([\(\[\{])\s+",
        r"\1",
        s,
    )

    s = re.sub(
        r'\s+(["”’])',
        r"\1",
        s,
    )

    return s


def render_diff(old, new):
    a = lexical_tokens(old)
    b = lexical_tokens(new)

    sm = SequenceMatcher(
        None,
        a,
        b,
        autojunk=False,
    )

    pieces = []

    opcodes = []

    for (
        tag,
        i1,
        i2,
        j1,
        j2,
    ) in sm.get_opcodes():

        old_piece = join_tokens(
            a[i1:i2]
        )

        new_piece = join_tokens(
            b[j1:j2]
        )

        if tag == "equal":
            pieces.append(
                old_piece
            )

        elif tag == "replace":
            pieces.append(
                f"[-{old_piece}-]"
                f"{{+{new_piece}+}}"
            )

        elif tag == "delete":
            pieces.append(
                f"[-{old_piece}-]"
            )

        elif tag == "insert":
            pieces.append(
                f"{{+{new_piece}+}}"
            )

        if tag != "equal":
            opcodes.append(
                {
                    "tag": tag,
                    "old_start": i1,
                    "old_end": i2,
                    "new_start": j1,
                    "new_end": j2,
                    "old_text": old_piece,
                    "new_text": new_piece,
                }
            )

    return (
        " ".join(
            p
            for p in pieces
            if p
        ),
        opcodes,
    )


v3_rows = load_jsonl(
    V3
)

handoff_rows = load_jsonl(
    HANDOFF
)

local_rows = load_jsonl(
    LOCAL
)

signal_rows = (
    load_jsonl(SIGNAL)
    if SIGNAL.exists()
    else []
)


v3 = {
    int(x["job_id"]): x
    for x in v3_rows
}

handoff = {
    int(x["job_id"]): x
    for x in handoff_rows
}

local = {
    int(x["job_id"]): x
    for x in local_rows
}

signal = {
    (
        int(x["job_id"]),
        int(x["h"]),
    ): x
    for x in signal_rows
}


if len(v3) != 26:
    raise RuntimeError(
        f"Expected 26 v3 rows, got {len(v3)}"
    )

if len(handoff) != 26:
    raise RuntimeError(
        f"Expected 26 handoff rows, "
        f"got {len(handoff)}"
    )


tok = (
    AutoTokenizer
    .from_pretrained(
        TOKENIZER,
        local_files_only=True,
    )
)


all_results = []

md = []

md.append(
    "# Teacher Semantic/Edit Event Audit"
)

md.append("")

md.append(
    "This file is descriptive. "
    "chrF / JSD / log-prob gap are shown "
    "only as auxiliary evidence and do not "
    "define intervention boundaries."
)

md.append("")


for jid in sorted(v3):

    x = v3[jid]
    hx = handoff[jid]
    lx = local[jid]

    source = str(
        lx.get(
            "source",
            "",
        )
    )

    reference = str(
        lx.get(
            "reference",
            "",
        )
    )

    student_translation = str(
        lx.get(
            "student_translation",
            "",
        )
    )

    post_edit = str(
        lx.get(
            "post_edit",
            "",
        )
    )

    raw_edit = str(
        lx.get(
            "raw_edit_text",
            lx.get(
                "raw_edit",
                "",
            ),
        )
    )

    fix_edit = str(
        lx.get(
            "fix_edit_text",
            lx.get(
                "fix_edit",
                "",
            ),
        )
    )

    corrected_prefix = str(
        x.get(
            "corrected_prefix_text",
            "",
        )
    )

    h0_text = str(
        x["h0"]["full"]
    )

    h0_score = float(
        x["h0"][
            "ref_chrf"
        ]
    )

    teacher_full = str(
        x["teacher"]["full"]
    )

    teacher_full_score = float(
        x["teacher"][
            "full_ref_chrf"
        ]
    )

    teacher_ids = [
        int(v)
        for v in x[
            "teacher"
        ][
            "continuation_ids_no_eos"
        ]
    ]


    events = []

    previous_text = h0_text
    previous_score = h0_score


    for h in range(
        1,
        9,
    ):

        current = hx[
            "horizons"
        ][str(h)]

        current_text = str(
            current["full"]
        )

        current_score = float(
            current["ref_chrf"]
        )

        if (
            current_text
            != previous_text
        ):

            diff_text, opcodes = (
                render_diff(
                    previous_text,
                    current_text,
                )
            )

            one_token = tok.decode(
                [
                    teacher_ids[
                        h - 1
                    ]
                ],
                skip_special_tokens=True,
            )

            bridge_text = tok.decode(
                teacher_ids[:h],
                skip_special_tokens=True,
            )

            sig = signal.get(
                (
                    jid,
                    h - 1,
                ),
                {},
            )

            event = {
                "h": h,

                "teacher_token_text":
                    one_token,

                "teacher_bridge_text":
                    bridge_text,

                "step_delta_chrf":
                    current_score
                    - previous_score,

                "cumulative_delta_chrf":
                    current_score
                    - h0_score,

                "before":
                    previous_text,

                "after":
                    current_text,

                "word_level_diff":
                    diff_text,

                "edit_opcodes":
                    opcodes,

                # Diagnostic only.
                "js":
                    sig.get(
                        "js"
                    ),

                "teacher_student_logprob_gap":
                    sig.get(
                        "teacher_student_logprob_gap"
                    ),

                "top1_agree":
                    sig.get(
                        "top1_agree"
                    ),
            }

            events.append(
                event
            )


        previous_text = (
            current_text
        )

        previous_score = (
            current_score
        )


    result = {
        "job_id":
            jid,

        "dataset":
            x["dataset"],

        "source":
            source,

        "reference":
            reference,

        "student_translation":
            student_translation,

        "post_edit":
            post_edit,

        "raw_edit":
            raw_edit,

        "fix_edit":
            fix_edit,

        "corrected_prefix":
            corrected_prefix,

        "h0":
            h0_text,

        "h0_ref_chrf":
            h0_score,

        "teacher_full":
            teacher_full,

        "teacher_full_ref_chrf":
            teacher_full_score,

        "events":
            events,
    }

    all_results.append(
        result
    )


    md.append(
        f"## JOB {jid} "
        f"({x['dataset']})"
    )

    md.append("")

    md.append(
        f"**Source:** {source}"
    )

    md.append("")

    md.append(
        f"**Reference:** {reference}"
    )

    md.append("")

    md.append(
        f"**Student:** "
        f"{student_translation}"
    )

    md.append("")

    md.append(
        f"**Patcher post-edit:** "
        f"{post_edit}"
    )

    md.append("")

    md.append(
        f"**Localized edit:** "
        f"`{raw_edit}` → `{fix_edit}`"
    )

    md.append("")

    md.append(
        f"**Corrected prefix:** "
        f"`{corrected_prefix}`"
    )

    md.append("")

    md.append(
        f"**H0 ({h0_score:.3f} chrF):** "
        f"{h0_text}"
    )

    md.append("")

    md.append(
        f"**Teacher-full "
        f"({teacher_full_score:.3f} chrF):** "
        f"{teacher_full}"
    )

    md.append("")


    if not events:

        md.append(
            "**No final-translation change "
            "within Teacher h=1..8.**"
        )

        md.append("")

    else:

        md.append(
            "### Translation-changing events"
        )

        md.append("")


        for ev in events:

            md.append(
                f"#### h={ev['h']}"
            )

            md.append("")

            md.append(
                f"Teacher new token: "
                f"`{ev['teacher_token_text']}`"
            )

            md.append("")

            md.append(
                f"Teacher bridge so far: "
                f"`{ev['teacher_bridge_text']}`"
            )

            md.append("")

            md.append(
                "Quality reference only: "
                f"step ΔchrF="
                f"{ev['step_delta_chrf']:+.3f}, "
                f"cumulative="
                f"{ev['cumulative_delta_chrf']:+.3f}"
            )

            md.append("")

            md.append(
                f"Before: {ev['before']}"
            )

            md.append("")

            md.append(
                f"After: {ev['after']}"
            )

            md.append("")

            md.append(
                f"Word/edit diff: "
                f"{ev['word_level_diff']}"
            )

            md.append("")


            if (
                ev["js"]
                is not None
            ):

                md.append(
                    "Diagnostics only: "
                    f"JSD={ev['js']:.6f}, "
                    f"logprob-gap="
                    f"{ev['teacher_student_logprob_gap']:+.4f}, "
                    f"top1_agree="
                    f"{ev['top1_agree']}"
                )

                md.append("")


    md.append("---")

    md.append("")


with OUT_JSONL.open(
    "w",
    encoding="utf-8",
) as f:

    for x in all_results:

        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )


OUT_MD.write_text(
    "\n".join(md),
    encoding="utf-8",
)


print(
    "ROWS=",
    len(all_results),
)

print(
    "TOTAL_TRANSLATION_CHANGE_EVENTS=",
    sum(
        len(x["events"])
        for x in all_results
    ),
)

for x in all_results:

    hs = [
        e["h"]
        for e in x["events"]
    ]

    print(
        f"JOB={x['job_id']} "
        f"EVENT_H={hs}"
    )


print(
    "MD=",
    OUT_MD,
)

print(
    "JSONL=",
    OUT_JSONL,
)

print(
    "TEACHER_SEMANTIC_EVENT_AUDIT_V1_PASS"
)
