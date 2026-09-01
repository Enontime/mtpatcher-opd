# SCORE
###############################################################################

python - <<'PY'
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import sacrebleu

work = Path(os.environ["WORK"])


def load(name):
    p = work / name

    with p.open(encoding="utf-8") as f:
        return [
            json.loads(x)
            for x in f
            if x.strip()
        ]


def jid(x):
    return int(
        x.get(
            "job_id",
            x.get(
                "demo_id",
                x.get("id"),
            ),
        )
    )


def response(x):
    for k in [
        "response",
        "generated_text",
        "output",
        "prediction",
    ]:
        if k in x:
            return str(x[k]).strip()

    raise RuntimeError(
        f"response field missing: {list(x.keys())}"
    )


def clean_translation(s):
    s = s.strip()

    if s.lower().startswith("output:"):
        s = s.split(":", 1)[1].strip()

    s = re.sub(
        r"^\s*(?:Translation|English)\s*:\s*",
        "",
        s,
        flags=re.I,
    )

    if (
        len(s) >= 2
        and s[0] == s[-1]
        and s[0] in "\"'"
    ):
        s = s[1:-1].strip()

    return s


def raw_feedback_to_postedit(raw, draft):
    text = raw.strip()

    # Conservative No-Error detection.
    no_error = bool(
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

    if no_error and not explicit_error:
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
    ]

    candidate = None

    for pattern in patterns:
        m = re.search(
            pattern,
            text,
        )

        if m:
            candidate = m.group(1).strip()
            break

    if candidate:
        candidate = re.sub(
            r"^\s*[-*]\s*",
            "",
            candidate,
        ).strip()

        if (
            len(candidate) >= 2
            and candidate[0] == candidate[-1]
            and candidate[0] in "\"'"
        ):
            candidate = (
                candidate[1:-1].strip()
            )

        return {
            "has_error": True,
            "post_edit": candidate,
            "extract_ok": True,
            "route": "error_good_translation",
        }

    return {
        "has_error": True,
        "post_edit": draft,
        "extract_ok": False,
        "route": "error_extract_fail_fallback_draft",
    }


student = load("student.jsonl")
special = load(
    "specialized_feedback_raw.jsonl"
)
combo = load(
    "base8b_combo_raw.jsonl"
)

N = len(student)

if N != 2207:
    raise RuntimeError(
        f"N != 2207: {N}"
    )

student_by_id = {
    int(x["job_id"]): x
    for x in student
}

special_by_id = {
    jid(x): x
    for x in special
}

zero_raw = {}
teacher_raw = {}

for x in combo:
    i = jid(x)

    if i < N:
        zero_raw[i] = x
    else:
        teacher_raw[i - N] = x

expected = set(range(N))

for name, obj in [
    ("special", special_by_id),
    ("zero", zero_raw),
    ("teacher", teacher_raw),
]:
    if set(obj) != expected:
        raise RuntimeError(
            f"{name} id coverage mismatch: "
            f"{len(obj)}/{N}"
        )


materialized = []

for i in range(N):
    s = student_by_id[i]

    draft = s["student_translation"]

    sp_raw = response(
        special_by_id[i]
    )

    z_raw = response(
        zero_raw[i]
    )

    t_raw = clean_translation(
        response(
            teacher_raw[i]
        )
    )

    sp = raw_feedback_to_postedit(
        sp_raw,
        draft,
    )

    z = raw_feedback_to_postedit(
        z_raw,
        draft,
    )

    materialized.append({
        **s,

        "zero_raw_feedback":
            z_raw,

        "zero_has_error":
            z["has_error"],

        "zero_extract_ok":
            z["extract_ok"],

        "zero_post_edit":
            z["post_edit"],

        "specialized_raw_feedback":
            sp_raw,

        "specialized_has_error":
            sp["has_error"],

        "specialized_extract_ok":
            sp["extract_ok"],

        "specialized_post_edit":
            sp["post_edit"],

        "direct_teacher":
            t_raw,
    })


out = work / "quality_gate_materialized.jsonl"

with out.open(
    "w",
    encoding="utf-8",
) as f:
    for x in materialized:
        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )


def metrics(rows, key):
    hyp = [
        x[key]
        for x in rows
    ]

    ref = [
        x["reference"]
        for x in rows
    ]

    bleu = sacrebleu.corpus_bleu(
        hyp,
        [ref],
    ).score

    chrf = sacrebleu.corpus_chrf(
        hyp,
        [ref],
    ).score

    return {
        "bleu": bleu,
        "chrf": chrf,
    }


summary = {
    "rows": N,
    "checkpoint":
        str(
            Path(os.environ["RUN_ROOT"])
            / os.environ["EXP"]
            / "patcher_qwen3_8b_paperfaith_fullft_v2"
            / "checkpoint-4752"
        ),

    "datasets": {},
}

for dataset in [
    "wmt24",
    "flores",
    "challenge",
]:
    rows = [
        x for x in materialized
        if x["dataset"] == dataset
    ]

    d = {
        "rows": len(rows),

        "student":
            metrics(
                rows,
                "student_translation",
            ),

        "zero_shot_patcher":
            metrics(
                rows,
                "zero_post_edit",
            ),

        "specialized_patcher":
            metrics(
                rows,
                "specialized_post_edit",
            ),

        "direct_teacher":
            metrics(
                rows,
                "direct_teacher",
            ),

        "zero_has_error_rate":
            sum(
                x["zero_has_error"]
                for x in rows
            )
            / len(rows),

        "specialized_has_error_rate":
            sum(
                x["specialized_has_error"]
                for x in rows
            )
            / len(rows),

        "zero_extract_rate":
            sum(
                x["zero_extract_ok"]
                for x in rows
            )
            / len(rows),

        "specialized_extract_rate":
            sum(
                x["specialized_extract_ok"]
                for x in rows
            )
            / len(rows),
    }

    d["specialized_minus_zero_bleu"] = (
        d["specialized_patcher"]["bleu"]
        - d["zero_shot_patcher"]["bleu"]
    )

    d["specialized_minus_student_bleu"] = (
        d["specialized_patcher"]["bleu"]
        - d["student"]["bleu"]
    )

    d["teacher_minus_specialized_bleu"] = (
        d["direct_teacher"]["bleu"]
        - d["specialized_patcher"]["bleu"]
    )

    summary["datasets"][dataset] = d


all_rows = materialized

summary["overall_micro"] = {
    "student":
        metrics(
            all_rows,
            "student_translation",
        ),

    "zero_shot_patcher":
        metrics(
            all_rows,
            "zero_post_edit",
        ),

    "specialized_patcher":
        metrics(
            all_rows,
            "specialized_post_edit",
        ),

    "direct_teacher":
        metrics(
            all_rows,
            "direct_teacher",
        ),

    "zero_has_error_rate":
        sum(
            x["zero_has_error"]
            for x in all_rows
        )
        / N,

    "specialized_has_error_rate":
        sum(
            x["specialized_has_error"]
            for x in all_rows
        )
        / N,

    "zero_extract_rate":
        sum(
            x["zero_extract_ok"]
            for x in all_rows
        )
        / N,

    "specialized_extract_rate":
        sum(
            x["specialized_extract_ok"]
            for x in all_rows
        )
        / N,
}

o = summary["overall_micro"]

o["specialized_minus_zero_bleu"] = (
    o["specialized_patcher"]["bleu"]
    - o["zero_shot_patcher"]["bleu"]
)

o["specialized_minus_student_bleu"] = (
    o["specialized_patcher"]["bleu"]
    - o["student"]["bleu"]
)

o["teacher_minus_specialized_bleu"] = (
    o["direct_teacher"]["bleu"]
    - o["specialized_patcher"]["bleu"]
)

summary_path = (
    work
    / "quality_gate_summary.json"
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
print("============================================================")
print("SPECIALIZED PATCHER QUALITY GATE RESULTS")
print("============================================================")

for dataset, d in summary["datasets"].items():
    print()
    print(dataset)

    for k in [
        "student",
        "zero_shot_patcher",
        "specialized_patcher",
        "direct_teacher",
    ]:
        print(
            f"  {k:24s} "
            f"BLEU={d[k]['bleu']:.4f} "
            f"chrF={d[k]['chrf']:.4f}"
        )

    print(
        "  specialized-zero BLEU = "
        f"{d['specialized_minus_zero_bleu']:+.4f}"
    )

    print(
        "  specialized-student BLEU = "
        f"{d['specialized_minus_student_bleu']:+.4f}"
    )

    print(
        "  teacher-specialized BLEU = "
        f"{d['teacher_minus_specialized_bleu']:+.4f}"
    )

    print(
        "  zero has_error = "
        f"{d['zero_has_error_rate']:.4%}"
    )

    print(
        "  specialized has_error = "
        f"{d['specialized_has_error_rate']:.4%}"
    )

    print(
        "  zero extract = "
        f"{d['zero_extract_rate']:.4%}"
    )

    print(
        "  specialized extract = "
        f"{d['specialized_extract_rate']:.4%}"
    )


print()
print("OVERALL MICRO")

for k in [
    "student",
    "zero_shot_patcher",
    "specialized_patcher",
    "direct_teacher",
]:
    print(
        f"  {k:24s} "
        f"BLEU={o[k]['bleu']:.4f} "
        f"chrF={o[k]['chrf']:.4f}"
    )

print(
    "  specialized-zero BLEU = "
    f"{o['specialized_minus_zero_bleu']:+.4f}"
)

print(
    "  specialized-student BLEU = "
    f"{o['specialized_minus_student_bleu']:+.4f}"
)

print(
    "  teacher-specialized BLEU = "
    f"{o['teacher_minus_specialized_bleu']:+.4f}"
)

print(
    "  zero has_error = "
    f"{o['zero_has_error_rate']:.4%}"
)

print(
    "  specialized has_error = "
    f"{o['specialized_has_error_rate']:.4%}"
)

print(
    "  zero extract = "
    f"{o['zero_extract_rate']:.4%}"
)

print(
    "  specialized extract = "
    f"{o['specialized_extract_rate']:.4%}"
)

print()
print(
    "SUMMARY_FILE=",
    summary_path,
)

print(
    "MATERIALIZED_FILE=",
    out,
)

print(
    "PATCHER_QUALITY_GATE_V1_PASS"
)
PY
