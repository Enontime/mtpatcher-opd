#!/usr/bin/env bash
set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

SCRIPT_DIR="$ROOT/scripts/mtpatcher_v14"
EXP_DATA="$DATA_ROOT/$EXP"
EXP_LOG="$LOG_ROOT/$EXP"

MODEL="$MODEL_ROOT/Qwen3-8B"

ANCHORS="$EXP_DATA/rq3_wa_anchor_jobs_v14.jsonl"

BASE_REPAIRED="$EXP_DATA/rq3_wa_analogs_repaired_v14"
PAIRWISE_DIR="$EXP_DATA/rq3_wa_analog_finalretry_v14"
LAST31_DIR="$EXP_DATA/rq3_wa_last31_v14"

MISSING8="$EXP_DATA/rq3_wa_final_missing8_v14.jsonl"
LAST8_DIR="$EXP_DATA/rq3_wa_last8_pool_v14"

FINAL_ANALOG_DIR="$EXP_DATA/rq3_wa_analogs_final_v14"

ANALOG_MERGED="$EXP_DATA/rq3_wa_analogs_merged_v14.jsonl"
ANALOG_AUDIT="$EXP_DATA/rq3_wa_analogs_audit_v14.json"

CONTEXT_JOBS="$EXP_DATA/rq3_wa_context_jobs_v14.jsonl"
CONTEXT_DIR="$EXP_DATA/rq3_wa_contexts_final_v14"

PE_PDS="$EXP_DATA/rq3_pe_plus_pds_v13_paperbudget.jsonl"

WA_VALID="$EXP_DATA/rq3_wa_valid_v14.jsonl"
WA_AUDIT="$EXP_DATA/rq3_wa_audit_v14.json"

FULL="$EXP_DATA/rq3_pe_pds_wa_v14.jsonl"
FULL_AUDIT="$EXP_DATA/rq3_pe_pds_wa_v14_audit.json"

mkdir -p \
    "$LAST8_DIR" \
    "$EXP_LOG/rq3_wa_v14_last8"

###############################################################################
# 1. BUILD AUTHORITATIVE 3724 + EXACT MISSING 8
###############################################################################

cat > "$SCRIPT_DIR/build_wa_missing8_v14.py" <<'PY'
import argparse
import json
from pathlib import Path


def load_jsonl(path):
    rows = []

    path = Path(path)

    if not path.exists():
        return rows

    with path.open(
        encoding="utf-8",
        errors="replace",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))
            except Exception:
                pass

    return rows


def valid(row):
    if not row.get("parse_ok"):
        return False

    a = row.get("analogs")

    if not isinstance(a, dict):
        return False

    seen = set()

    anchor = "".join(
        str(
            row.get(
                "source_span",
                "",
            )
        ).split()
    ).casefold()

    for aspect in (
        "category",
        "semantics",
    ):
        arr = a.get(aspect)

        if (
            not isinstance(arr, list)
            or len(arr) != 2
        ):
            return False

        for item in arr:
            if not isinstance(item, dict):
                return False

            src = item.get("source")
            tgt = item.get("target")

            if (
                not isinstance(src, str)
                or not src.strip()
                or not isinstance(tgt, str)
                or not tgt.strip()
            ):
                return False

            n = "".join(
                src.split()
            ).casefold()

            if n == anchor:
                return False

            if n in seen:
                return False

            seen.add(n)

    return len(seen) == 4


def read_dir(path):
    out = {}

    path = Path(path)

    if not path.exists():
        return out

    for f in sorted(
        path.glob("device_*.jsonl")
    ):
        for row in load_jsonl(f):
            if valid(row):
                out[
                    int(row["analog_job_id"])
                ] = row

    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--anchors", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--pairwise", required=True)
    ap.add_argument("--last31", required=True)
    ap.add_argument("--output", required=True)

    args = ap.parse_args()

    anchors = {
        int(x["analog_job_id"]): x
        for x in load_jsonl(args.anchors)
    }

    base = read_dir(args.base)
    pairwise = read_dir(args.pairwise)
    last31 = read_dir(args.last31)

    resolved = dict(base)

    counts = {
        "base": len(resolved),
        "pairwise_added": 0,
        "last31_added": 0,
    }

    for jid, row in pairwise.items():
        if jid not in resolved:
            resolved[jid] = row
            counts["pairwise_added"] += 1

    for jid, row in last31.items():
        if jid not in resolved:
            resolved[jid] = row
            counts["last31_added"] += 1

    missing = sorted(
        set(anchors)
        - set(resolved)
    )

    print("COUNTS =", counts)
    print(
        "RESOLVED_BEFORE_LAST8 =",
        len(resolved),
    )
    print(
        "MISSING8_COUNT =",
        len(missing),
    )
    print(
        "MISSING8_IDS =",
        missing,
    )

    expected = [
        318,
        360,
        362,
        796,
        1328,
        1839,
        3151,
        3261,
    ]

    if len(resolved) != 3724:
        raise RuntimeError(
            f"Expected 3724 resolved, "
            f"got {len(resolved)}"
        )

    if missing != expected:
        raise RuntimeError(
            f"Unexpected missing set: "
            f"{missing}"
        )

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as f:
        for jid in missing:
            f.write(
                json.dumps(
                    anchors[jid],
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("WA_EXACT_MISSING8_AUDIT_PASS")


if __name__ == "__main__":
    main()
PY

###############################################################################
# 2. LAST 8: CANDIDATE-POOL GENERATOR
###############################################################################

cat > "$SCRIPT_DIR/generate_wa_last8_pool_v14.py" <<'PY'
import argparse
import json
import random
import re
from pathlib import Path

import torch
import torch_npu

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


def load_jsonl(path):
    rows = []

    path = Path(path)

    if not path.exists():
        return rows

    with path.open(
        encoding="utf-8",
        errors="replace",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))
            except Exception:
                pass

    return rows


def norm(x):
    return "".join(
        str(x).strip().split()
    ).casefold()


def render(tokenizer, prompt):
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


def generate(
    model,
    tokenizer,
    device,
    prompt,
    do_sample,
    temperature=1.0,
    max_new_tokens=256,
):
    text = render(
        tokenizer,
        prompt,
    )

    enc = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    enc = {
        k: v.to(device)
        for k, v in enc.items()
    }

    kwargs = {
        "max_new_tokens":
            max_new_tokens,

        "pad_token_id":
            tokenizer.pad_token_id,

        "eos_token_id":
            tokenizer.eos_token_id,

        "do_sample":
            do_sample,
    }

    if do_sample:
        kwargs.update(
            temperature=temperature,
            top_p=0.95,
        )

    with torch.inference_mode():
        out = model.generate(
            **enc,
            **kwargs,
        )

    prompt_len = (
        enc["input_ids"].shape[1]
    )

    return tokenizer.decode(
        out[0][prompt_len:],
        skip_special_tokens=True,
    ).strip()


def clean_candidate(line):
    x = line.strip()

    if not x:
        return ""

    x = re.sub(
        r"^[\-\*\•\·\s]+",
        "",
        x,
    )

    x = re.sub(
        r"^\d+\s*[\.\)、\):：]\s*",
        "",
        x,
    )

    for prefix in (
        "候选词：",
        "候选词:",
        "候选短语：",
        "候选短语:",
        "中文：",
        "中文:",
        "短语：",
        "短语:",
        "答案：",
        "答案:",
    ):
        if x.startswith(prefix):
            x = x[len(prefix):].strip()

    x = x.strip(
        " \"'“”‘’`"
    )

    # Some models occasionally append explanations.
    for sep in (
        "\t",
        " —— ",
        " -- ",
        " -> ",
        " → ",
    ):
        if sep in x:
            x = x.split(
                sep,
                1,
            )[0].strip()

    return x


def parse_pool(raw):
    raw = re.sub(
        r"```.*?",
        "",
        raw,
        flags=re.I,
    )

    raw = raw.replace(
        "```",
        "",
    )

    out = []

    for line in raw.splitlines():
        x = clean_candidate(line)

        if not x:
            continue

        # Keep phrase-level material.
        if len(x) > 80:
            continue

        if x in {
            "Category",
            "Semantics",
            "类别",
            "语义",
        }:
            continue

        out.append(x)

    return out


def pool_prompt(
    row,
    aspect,
    forbidden,
):
    forbidden_text = "\n".join(
        f"- {x}"
        for x in forbidden
    )

    if aspect == "category":
        relation = """请给出 8 个与原短语属于相同类别、
相同实体类型、相同概念类别或相似术语类别的中文词或短语。"""
    else:
        relation = """请给出 8 个与原短语语义相关、
经常共现、属于同一事件场景或自然出现在相近语境中的中文词或短语。"""

    return f"""你是一名中英机器翻译专家。

{relation}

优先选择：
- 相对少见；
- 对机器翻译具有一定难度；
- 可以作为独立翻译知识的词或短语。

要求：
1. 每行只输出一个中文词或短语。
2. 总共输出 8 行。
3. 不输出英文。
4. 不解释。
5. 不写完整句子。
6. 不得输出下列禁用短语。

原始句子：
{row["source"]}

原始错误短语：
{row["source_span"]}

禁用短语：
{forbidden_text}

现在直接输出 8 个中文候选：
"""


def translation_prompt(phrase):
    return f"""Translate the following Chinese word or short phrase into natural English.

Return only the English translation.
Do not explain.
Do not use quotation marks.

Chinese:
{phrase}
"""


def clean_translation(raw):
    lines = [
        x.strip()
        for x in raw.splitlines()
        if x.strip()
    ]

    if not lines:
        return ""

    x = lines[0]

    for prefix in (
        "English:",
        "Translation:",
        "English translation:",
        "英文：",
        "英文:",
        "翻译：",
        "翻译:",
    ):
        if x.startswith(prefix):
            x = x[len(prefix):].strip()

    x = x.strip(
        " \"'“”‘’`"
    )

    return x


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--jobs", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True)

    ap.add_argument(
        "--device-id",
        type=int,
        required=True,
    )

    ap.add_argument(
        "--world-size",
        type=int,
        default=2,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=20260825,
    )

    args = ap.parse_args()

    torch.npu.set_device(
        args.device_id
    )

    device = (
        f"npu:{args.device_id}"
    )

    random.seed(
        args.seed
        + args.device_id
    )

    torch.manual_seed(
        args.seed
        + args.device_id
    )

    jobs = load_jsonl(
        args.jobs
    )

    assigned = [
        row
        for row in jobs
        if int(
            row["analog_job_id"]
        ) % args.world_size
        == args.device_id
    ]

    output = Path(
        args.output
    )

    previous = {
        int(x["analog_job_id"]): x
        for x in load_jsonl(output)
        if x.get("parse_ok")
    }

    pending = [
        row
        for row in assigned
        if int(
            row["analog_job_id"]
        ) not in previous
    ]

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            args.model,
            local_files_only=True,
            trust_remote_code=True,
        )
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id
        )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            args.model,
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
    )

    model.to(device)
    model.eval()

    print(
        f"LAST8_DEVICE={args.device_id} "
        f"ASSIGNED={len(assigned)} "
        f"PENDING={len(pending)}",
        flush=True,
    )

    final = dict(previous)

    for pos, row in enumerate(
        pending,
        1,
    ):
        anchor = row["source_span"]

        selected = []

        analogs = {
            "category": [],
            "semantics": [],
        }

        errors = []

        for aspect in (
            "category",
            "semantics",
        ):
            candidates = []

            for attempt in range(
                1,
                13,
            ):
                raw = generate(
                    model,
                    tokenizer,
                    device,
                    pool_prompt(
                        row,
                        aspect,
                        [
                            anchor,
                            *selected,
                            *candidates,
                        ],
                    ),
                    do_sample=True,
                    temperature=(
                        0.8
                        if attempt <= 6
                        else 1.1
                    ),
                    max_new_tokens=320,
                )

                pool = parse_pool(raw)

                for candidate in pool:
                    n = norm(candidate)

                    if not n:
                        continue

                    if n == norm(anchor):
                        continue

                    if any(
                        n == norm(x)
                        for x in selected
                    ):
                        continue

                    if any(
                        n == norm(x)
                        for x in candidates
                    ):
                        continue

                    candidates.append(
                        candidate
                    )

                if len(candidates) >= 2:
                    break

            if len(candidates) < 2:
                errors.append(
                    f"{aspect}:"
                    f"only_{len(candidates)}_candidates"
                )
                break

            chosen = candidates[:2]

            for phrase in chosen:
                english = ""

                for tr_attempt in range(
                    1,
                    5,
                ):
                    raw_en = generate(
                        model,
                        tokenizer,
                        device,
                        translation_prompt(
                            phrase
                        ),
                        do_sample=False,
                        max_new_tokens=96,
                    )

                    english = (
                        clean_translation(
                            raw_en
                        )
                    )

                    if english:
                        break

                if not english:
                    errors.append(
                        f"{aspect}:"
                        f"translation_failed:"
                        f"{phrase}"
                    )
                    break

                analogs[
                    aspect
                ].append(
                    {
                        "source":
                            phrase,

                        "target":
                            english,
                    }
                )

                selected.append(
                    phrase
                )

            if len(
                analogs[aspect]
            ) != 2:
                break

        unique = {
            norm(x["source"])
            for arr
            in analogs.values()
            for x in arr
        }

        parse_ok = (
            len(
                analogs["category"]
            ) == 2
            and
            len(
                analogs["semantics"]
            ) == 2
            and
            len(unique) == 4
            and
            norm(anchor)
            not in unique
        )

        result = {
            **row,

            "parse_ok":
                parse_ok,

            "parse_error":
                ""
                if parse_ok
                else ";".join(
                    errors
                ),

            "analogs":
                analogs
                if parse_ok
                else None,

            "wa_recovery_origin":
                "last8_candidate_pool",

            "construction_method":
                "MT_PATCHER_WA_LAST8_POOL_V14",
        }

        final[
            int(
                row["analog_job_id"]
            )
        ] = result

        print(
            f"LAST8_DEVICE={args.device_id} "
            f"JOB={row['analog_job_id']} "
            f"OK={parse_ok} "
            f"PROGRESS={pos}/{len(pending)}",
            flush=True,
        )

    tmp = Path(
        str(output) + ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as f:
        for jid in sorted(final):
            f.write(
                json.dumps(
                    final[jid],
                    ensure_ascii=False,
                )
                + "\n"
            )

    tmp.replace(output)

    failures = sum(
        not bool(x.get("parse_ok"))
        for x in final.values()
    )

    print(
        f"LAST8_DEVICE_{args.device_id}_COMPLETE "
        f"FAILURES={failures}",
        flush=True,
    )


if __name__ == "__main__":
    main()
PY

###############################################################################
# 3. FINAL AUTHORITATIVE MERGE
###############################################################################

cat > "$SCRIPT_DIR/merge_wa_final3732_v14.py" <<'PY'
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def load_jsonl(path):
    rows = []

    path = Path(path)

    if not path.exists():
        return rows

    with path.open(
        encoding="utf-8",
        errors="replace",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))
            except Exception:
                pass

    return rows


def valid(row):
    if not row.get("parse_ok"):
        return False

    a = row.get("analogs")

    if not isinstance(a, dict):
        return False

    seen = set()

    anchor = "".join(
        str(
            row.get(
                "source_span",
                "",
            )
        ).split()
    ).casefold()

    for aspect in (
        "category",
        "semantics",
    ):
        arr = a.get(aspect)

        if (
            not isinstance(arr, list)
            or len(arr) != 2
        ):
            return False

        for item in arr:
            if not isinstance(
                item,
                dict,
            ):
                return False

            src = item.get("source")
            tgt = item.get("target")

            if (
                not isinstance(src, str)
                or not src.strip()
                or not isinstance(tgt, str)
                or not tgt.strip()
            ):
                return False

            n = "".join(
                src.split()
            ).casefold()

            if n == anchor:
                return False

            if n in seen:
                return False

            seen.add(n)

    return len(seen) == 4


def read_dir(path):
    result = {}

    p = Path(path)

    if not p.exists():
        return result

    for f in sorted(
        p.glob("device_*.jsonl")
    ):
        for row in load_jsonl(f):
            if valid(row):
                result[
                    int(
                        row[
                            "analog_job_id"
                        ]
                    )
                ] = row

    return result


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base", required=True)
    ap.add_argument("--pairwise", required=True)
    ap.add_argument("--last31", required=True)
    ap.add_argument("--last8", required=True)
    ap.add_argument("--output", required=True)

    args = ap.parse_args()

    sources = [
        (
            "repaired_base",
            read_dir(args.base),
        ),
        (
            "pairwise",
            read_dir(args.pairwise),
        ),
        (
            "last31",
            read_dir(args.last31),
        ),
        (
            "last8_pool",
            read_dir(args.last8),
        ),
    ]

    final = {}
    origin = {}

    for name, rows in sources:
        for jid, row in rows.items():
            if jid not in final:
                final[jid] = row
                origin[jid] = name

    missing = sorted(
        set(range(3732))
        - set(final)
    )

    counts = Counter(
        origin.values()
    )

    print(
        "FINAL_SOURCE_COUNTS =",
        dict(counts),
    )

    print(
        "FINAL_VALID_ANALOG_ROWS =",
        len(final),
    )

    print(
        "FINAL_MISSING_IDS =",
        missing,
    )

    if missing:
        raise RuntimeError(
            f"WA still missing IDs: "
            f"{missing}"
        )

    if len(final) != 3732:
        raise RuntimeError(
            f"Expected 3732, "
            f"got {len(final)}"
        )

    out = Path(args.output)

    tmp = out.with_name(
        out.name + ".tmp"
    )

    if tmp.exists():
        shutil.rmtree(tmp)

    tmp.mkdir(
        parents=True,
        exist_ok=True,
    )

    handles = {
        i: (
            tmp
            / f"device_{i}.jsonl"
        ).open(
            "w",
            encoding="utf-8",
        )
        for i in range(16)
    }

    try:
        for jid in range(3732):
            row = dict(
                final[jid]
            )

            row[
                "wa_final_origin"
            ] = origin[jid]

            handles[
                jid % 16
            ].write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )
    finally:
        for h in handles.values():
            h.close()

    if out.exists():
        shutil.rmtree(out)

    tmp.rename(out)

    print(
        "WA_FINAL_3732_AUTHORITATIVE_PASS"
    )


if __name__ == "__main__":
    main()
PY

python -m py_compile \
    "$SCRIPT_DIR/build_wa_missing8_v14.py" \
    "$SCRIPT_DIR/generate_wa_last8_pool_v14.py" \
    "$SCRIPT_DIR/merge_wa_final3732_v14.py"

echo "WA_LAST8_COMPILE_PASS"

###############################################################################
# 4. BUILD THE EXACT 8
###############################################################################

python \
"$SCRIPT_DIR/build_wa_missing8_v14.py" \
    --anchors "$ANCHORS" \
    --base "$BASE_REPAIRED" \
    --pairwise "$PAIRWISE_DIR" \
    --last31 "$LAST31_DIR" \
    --output "$MISSING8"

###############################################################################
# 5. GENERATE ONLY 8 — 2 NPUs
###############################################################################

PIDS=()

for DEVICE in 0 1; do

    LOG="$EXP_LOG/rq3_wa_v14_last8/device_${DEVICE}.log"

    python \
    "$SCRIPT_DIR/generate_wa_last8_pool_v14.py" \
        --jobs "$MISSING8" \
        --output "$LAST8_DIR/device_${DEVICE}.jsonl" \
        --model "$MODEL" \
        --device-id "$DEVICE" \
        --world-size 2 \
        --seed 20260825 \
        > "$LOG" 2>&1 &

    PIDS+=("$!")

done

FAIL=0

for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        FAIL=1
    fi
done

if [ "$FAIL" -ne 0 ]; then
    echo "WA_LAST8_WORKER_FAILURE"
    false
fi

echo "WA_LAST8_WORKERS_COMPLETE"

###############################################################################
# 6. FINAL 3732 MERGE
###############################################################################

python \
"$SCRIPT_DIR/merge_wa_final3732_v14.py" \
    --base "$BASE_REPAIRED" \
    --pairwise "$PAIRWISE_DIR" \
    --last31 "$LAST31_DIR" \
    --last8 "$LAST8_DIR" \
    --output "$FINAL_ANALOG_DIR"

###############################################################################
# 7. EXACT 14928 CONTEXT JOBS
###############################################################################

python \
"$SCRIPT_DIR/merge_wa_analogs_build_context_jobs_v14.py" \
    --shard-dir "$FINAL_ANALOG_DIR" \
    --merged "$ANALOG_MERGED" \
    --audit "$ANALOG_AUDIT" \
    --context-jobs "$CONTEXT_JOBS"

N=$(
    wc -l < "$CONTEXT_JOBS"
)

echo "FINAL_WA_CONTEXT_JOBS=$N"

if [ "$N" -ne 14928 ]; then
    echo "WA_CONTEXT_BUDGET_FAILURE"
    false
fi

echo "WA_EXACT_14928_CONTEXT_BUDGET_PASS"

###############################################################################
# 8. START FROM CLEAN CONTEXT OUTPUT
###############################################################################

rm -rf "$CONTEXT_DIR"

mkdir -p "$CONTEXT_DIR"

###############################################################################
# 9. GENERATE 14928 WA CONTEXTS
###############################################################################

PIDS=()

for DEVICE in $(seq 0 15); do

    LOG="$EXP_LOG/rq3_wa_v14_last8/context_device_${DEVICE}.log"

    python \
    "$SCRIPT_DIR/generate_wa_contexts_qwen3_8b_v14.py" \
        --jobs "$CONTEXT_JOBS" \
        --output "$CONTEXT_DIR/device_${DEVICE}.jsonl" \
        --model "$MODEL" \
        --device-id "$DEVICE" \
        --world-size 16 \
        --batch-size 8 \
        --max-new-tokens 192 \
        --temperature 1.5 \
        --seed 20260825 \
        > "$LOG" 2>&1 &

    PIDS+=("$!")

done

FAIL=0

for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        FAIL=1
    fi
done

if [ "$FAIL" -ne 0 ]; then
    echo "WA_CONTEXT_WORKER_FAILURE"
    false
fi

echo "WA_ALL_16_CONTEXT_WORKERS_COMPLETE"

###############################################################################
# 10. POSTPROCESS + BUILD PE+PDS+WA
###############################################################################

python \
"$SCRIPT_DIR/merge_wa_contexts_build_full_v14.py" \
    --shard-dir "$CONTEXT_DIR" \
    --existing-pe-pds "$PE_PDS" \
    --wa-output "$WA_VALID" \
    --wa-audit "$WA_AUDIT" \
    --combined "$FULL" \
    --combined-audit "$FULL_AUDIT"

###############################################################################
# 11. FINAL AUDIT
###############################################################################

echo
echo "======================================================================"
echo "ANALOG AUDIT"
echo "======================================================================"

cat "$ANALOG_AUDIT"

echo
echo "======================================================================"
echo "WA CONTEXT AUDIT"
echo "======================================================================"

cat "$WA_AUDIT"

echo
echo "======================================================================"
echo "FULL DATA AUDIT"
echo "======================================================================"

cat "$FULL_AUDIT"

echo
echo "======================================================================"
echo "CARDINALITY"
echo "======================================================================"

wc -l \
    "$PE_PDS" \
    "$WA_VALID" \
    "$FULL"

echo
echo "======================================================================"
echo "SHA256"
echo "======================================================================"

sha256sum \
    "$WA_VALID" \
    "$FULL"

echo
echo "RQ3_WA_V14_LAST8_ALL_PASS"
