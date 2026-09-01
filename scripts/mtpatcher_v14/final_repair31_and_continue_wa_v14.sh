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

# 3699 条稳定结果
BASE_REPAIRED="$EXP_DATA/rq3_wa_analogs_repaired_v14"

# 上次 pairwise，其中已有 2 条成功
PAIRWISE_DIR="$EXP_DATA/rq3_wa_analog_finalretry_v14"

MISSING="$EXP_DATA/rq3_wa_final_missing_v14.jsonl"

FINAL_RETRY_DIR="$EXP_DATA/rq3_wa_last31_v14"

# 新建最终 canonical analog 目录，成功前绝不覆盖历史数据
FINAL_ANALOG_DIR="$EXP_DATA/rq3_wa_analogs_final_v14"

ANALOG_MERGED="$EXP_DATA/rq3_wa_analogs_merged_v14.jsonl"
ANALOG_AUDIT="$EXP_DATA/rq3_wa_analogs_audit_v14.json"

CONTEXT_JOBS="$EXP_DATA/rq3_wa_context_jobs_v14.jsonl"

# 使用新目录，避免任何历史 partial shard
CONTEXT_DIR="$EXP_DATA/rq3_wa_contexts_final_v14"

PE_PDS="$EXP_DATA/rq3_pe_plus_pds_v13_paperbudget.jsonl"

WA_VALID="$EXP_DATA/rq3_wa_valid_v14.jsonl"
WA_AUDIT="$EXP_DATA/rq3_wa_audit_v14.json"

FULL="$EXP_DATA/rq3_pe_pds_wa_v14.jsonl"
FULL_AUDIT="$EXP_DATA/rq3_pe_pds_wa_v14_audit.json"

mkdir -p \
    "$FINAL_RETRY_DIR" \
    "$EXP_LOG/rq3_wa_v14_last31"

###############################################################################
# 1. BUILD REAL SET DIFFERENCE
###############################################################################

cat > "$SCRIPT_DIR/build_wa_last_missing_v14.py" <<'PY'
import argparse
import json
from pathlib import Path


def load(path):
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

        for x in arr:
            if not isinstance(x, dict):
                return False

            s = x.get("source")
            t = x.get("target")

            if (
                not isinstance(s, str)
                or not s.strip()
                or not isinstance(t, str)
                or not t.strip()
            ):
                return False

            n = "".join(
                s.split()
            ).casefold()

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
        for row in load(f):
            if valid(row):
                result[
                    int(row["analog_job_id"])
                ] = row

    return result


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--anchors",
        required=True,
    )

    ap.add_argument(
        "--base",
        required=True,
    )

    ap.add_argument(
        "--pairwise",
        required=True,
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    args = ap.parse_args()

    anchors = {
        int(x["analog_job_id"]): x
        for x in load(args.anchors)
    }

    if len(anchors) != 3732:
        raise RuntimeError(
            f"anchors={len(anchors)}"
        )

    base = read_dir(args.base)
    pairwise = read_dir(
        args.pairwise
    )

    resolved = dict(base)

    pairwise_added = []

    for jid, row in pairwise.items():
        if jid not in resolved:
            resolved[jid] = row
            pairwise_added.append(jid)

    missing = sorted(
        set(anchors)
        - set(resolved)
    )

    print(
        "BASE_VALID =",
        len(base),
    )

    print(
        "PAIRWISE_VALID =",
        len(pairwise),
    )

    print(
        "PAIRWISE_NEW_IDS =",
        pairwise_added,
    )

    print(
        "RESOLVED_BEFORE_LAST_RETRY =",
        len(resolved),
    )

    print(
        "REAL_MISSING_COUNT =",
        len(missing),
    )

    print(
        "REAL_MISSING_IDS =",
        missing,
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

    if len(resolved) != 3701:
        raise RuntimeError(
            "Expected 3699 repaired "
            "+ 2 pairwise = 3701"
        )

    if len(missing) != 31:
        raise RuntimeError(
            f"Expected 31 real missing, "
            f"got {len(missing)}"
        )

    print(
        "WA_REAL_31_MISSING_AUDIT_PASS"
    )


if __name__ == "__main__":
    main()
PY

###############################################################################
# 2. ULTRA-ROBUST TWO-STAGE GENERATOR
###############################################################################

cat > "$SCRIPT_DIR/generate_wa_last31_twostage_v14.py" <<'PY'
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


def load(path):
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
                rows.append(
                    json.loads(line)
                )
            except Exception:
                pass

    return rows


def norm(x):
    return "".join(
        str(x).strip().split()
    ).casefold()


def clean_phrase(raw):
    raw = raw.strip()

    raw = re.sub(
        r"^```.*?\n?",
        "",
        raw,
        flags=re.I,
    )

    raw = re.sub(
        r"\n?```$",
        "",
        raw,
    )

    lines = [
        x.strip()
        for x in raw.splitlines()
        if x.strip()
    ]

    if not lines:
        return ""

    x = lines[0]

    x = re.sub(
        r"^[\-\*\d\.\)\s]+",
        "",
        x,
    )

    # Common labels.
    for prefix in (
        "中文短语：",
        "中文短语:",
        "短语：",
        "短语:",
        "答案：",
        "答案:",
        "Phrase:",
        "Chinese:",
    ):
        if x.startswith(prefix):
            x = x[len(prefix):].strip()

    x = x.strip(
        " \t\r\n\"'“”‘’`"
    )

    # JSON fallback.
    if x.startswith("{"):
        try:
            obj = json.loads(x)

            for key in (
                "source",
                "phrase",
                "chinese",
            ):
                v = obj.get(key)

                if (
                    isinstance(v, str)
                    and v.strip()
                ):
                    return v.strip()
        except Exception:
            pass

    return x.strip()


def clean_translation(raw):
    raw = raw.strip()

    raw = re.sub(
        r"^```.*?\n?",
        "",
        raw,
        flags=re.I,
    )

    raw = re.sub(
        r"\n?```$",
        "",
        raw,
    )

    lines = [
        x.strip()
        for x in raw.splitlines()
        if x.strip()
    ]

    if not lines:
        return ""

    x = lines[0]

    for prefix in (
        "English translation:",
        "English:",
        "Translation:",
        "英文翻译：",
        "英文翻译:",
        "答案：",
        "答案:",
    ):
        if x.startswith(prefix):
            x = x[len(prefix):].strip()

    x = x.strip(
        " \t\r\n\"'“”‘’`"
    )

    if x.startswith("{"):
        try:
            obj = json.loads(x)

            for key in (
                "target",
                "translation",
                "english",
            ):
                v = obj.get(key)

                if (
                    isinstance(v, str)
                    and v.strip()
                ):
                    return v.strip()
        except Exception:
            pass

    return x.strip()


def phrase_prompt(
    row,
    aspect,
    selected,
):
    anchor = row["source_span"]

    forbidden = [
        anchor,
        *selected,
    ]

    forbidden_text = "、".join(
        forbidden
    )

    if aspect == "category":
        instruction = (
            "请给出一个与原短语属于同一类别、"
            "同一事物类型或同一概念类别的"
            "较少见、较有翻译难度的中文词或短语。"
        )
    else:
        instruction = (
            "请给出一个与原短语在语义上紧密相关、"
            "经常共现或自然出现在相近语境中的"
            "较少见、较有翻译难度的中文词或短语。"
        )

    return f"""你是一名中英机器翻译专家。

{instruction}

要求：
1. 只输出一个中文词或短语。
2. 不要输出完整句子。
3. 不要解释。
4. 不要加编号。
5. 不要输出英文。
6. 不得与禁用短语相同。
7. 尽量选择对机器翻译具有挑战性的表达。

原句：
{row["source"]}

原错误短语：
{anchor}

禁用短语：
{forbidden_text}

只输出新的中文词或短语：
"""


def translation_prompt(
    phrase,
):
    return f"""Translate the following Chinese word or short phrase into natural English.

Output only the English translation.
Do not explain.
Do not add quotation marks.

Chinese phrase:
{phrase}
"""


def render(
    tokenizer,
    prompt,
):
    return tokenizer.apply_chat_template(
        [
            {
                "role":
                    "user",

                "content":
                    prompt,
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
    sample,
    temperature=1.0,
):
    rendered = render(
        tokenizer,
        prompt,
    )

    enc = tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
    )

    enc = {
        k: v.to(device)
        for k, v in enc.items()
    }

    kwargs = dict(
        max_new_tokens=96,
        pad_token_id=
            tokenizer.pad_token_id,
        eos_token_id=
            tokenizer.eos_token_id,
    )

    if sample:
        kwargs.update(
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
        )
    else:
        kwargs.update(
            do_sample=False,
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


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--jobs",
        required=True,
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    ap.add_argument(
        "--model",
        required=True,
    )

    ap.add_argument(
        "--device-id",
        type=int,
        required=True,
    )

    ap.add_argument(
        "--world-size",
        type=int,
        default=4,
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

    jobs = load(
        args.jobs
    )

    assigned = [
        x
        for x in jobs
        if int(
            x["analog_job_id"]
        ) % args.world_size
        == args.device_id
    ]

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    previous = {
        int(x["analog_job_id"]): x
        for x in load(output)
        if x.get("parse_ok")
    }

    pending = [
        x
        for x in assigned
        if int(
            x["analog_job_id"]
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
        f"DEVICE={args.device_id} "
        f"ASSIGNED={len(assigned)} "
        f"PENDING={len(pending)}",
        flush=True,
    )

    final = dict(
        previous
    )

    for pos, row in enumerate(
        pending,
        1,
    ):
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
            for rank in range(2):

                phrase = None

                # Diversity is needed here, so phrase
                # generation remains sampled.
                for attempt in range(
                    1,
                    17,
                ):
                    temp = (
                        0.7
                        if attempt <= 8
                        else 1.0
                    )

                    raw = generate(
                        model,
                        tokenizer,
                        device,
                        phrase_prompt(
                            row,
                            aspect,
                            selected,
                        ),
                        sample=True,
                        temperature=temp,
                    )

                    candidate = (
                        clean_phrase(raw)
                    )

                    n = norm(candidate)

                    if not candidate:
                        errors.append(
                            f"{aspect}/{rank}:"
                            "empty_phrase"
                        )
                        continue

                    if len(candidate) > 80:
                        errors.append(
                            f"{aspect}/{rank}:"
                            "phrase_too_long"
                        )
                        continue

                    if n == norm(
                        row["source_span"]
                    ):
                        errors.append(
                            f"{aspect}/{rank}:"
                            "equals_anchor"
                        )
                        continue

                    if any(
                        n == norm(x)
                        for x in selected
                    ):
                        errors.append(
                            f"{aspect}/{rank}:"
                            "duplicate"
                        )
                        continue

                    phrase = candidate
                    break

                if phrase is None:
                    break

                # Translation is deterministic.
                raw_en = generate(
                    model,
                    tokenizer,
                    device,
                    translation_prompt(
                        phrase
                    ),
                    sample=False,
                )

                english = (
                    clean_translation(
                        raw_en
                    )
                )

                if not english:
                    errors.append(
                        f"{aspect}/{rank}:"
                        "empty_translation"
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

        parse_ok = (
            len(
                analogs["category"]
            ) == 2
            and
            len(
                analogs["semantics"]
            ) == 2
            and
            len({
                norm(x["source"])
                for arr
                in analogs.values()
                for x in arr
            }) == 4
        )

        final[
            int(
                row[
                    "analog_job_id"
                ]
            )
        ] = {
            **row,

            "parse_ok":
                parse_ok,

            "parse_error":
                ""
                if parse_ok
                else ";".join(
                    errors[-20:]
                ),

            "analogs":
                analogs
                if parse_ok
                else None,

            "wa_recovery_origin":
                "last31_twostage",

            "construction_method":
                "MT_PATCHER_WA_TWOSTAGE_RECOVERY_V14",
        }

        print(
            f"DEVICE={args.device_id} "
            f"JOB={row['analog_job_id']} "
            f"OK={parse_ok} "
            f"{pos}/{len(pending)}",
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

    fail = sum(
        not x.get("parse_ok")
        for x in final.values()
    )

    print(
        f"LAST31_DEVICE_"
        f"{args.device_id}_COMPLETE "
        f"FAILURES={fail}",
        flush=True,
    )


if __name__ == "__main__":
    main()
PY

###############################################################################
# 3. MERGE WITHOUT RE-PARSING HISTORY
###############################################################################

cat > "$SCRIPT_DIR/merge_wa_authoritative_v14.py" <<'PY'
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def load(path):
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
                rows.append(
                    json.loads(line)
                )
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

        for x in arr:
            if not isinstance(
                x,
                dict,
            ):
                return False

            src = x.get("source")
            tgt = x.get("target")

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
        for row in load(f):
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

    ap.add_argument(
        "--base",
        required=True,
    )

    ap.add_argument(
        "--pairwise",
        required=True,
    )

    ap.add_argument(
        "--last31",
        required=True,
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    args = ap.parse_args()

    base = read_dir(args.base)
    pairwise = read_dir(
        args.pairwise
    )
    last31 = read_dir(
        args.last31
    )

    result = dict(base)

    source = {
        jid: "authoritative_repaired"
        for jid in result
    }

    for jid, row in pairwise.items():
        if jid not in result:
            result[jid] = row
            source[jid] = (
                "pairwise_final"
            )

    for jid, row in last31.items():
        if jid not in result:
            result[jid] = row
            source[jid] = (
                "last31_twostage"
            )

    missing = sorted(
        set(range(3732))
        - set(result)
    )

    counts = Counter(
        source.values()
    )

    print(
        "AUTHORITATIVE_COUNTS =",
        dict(counts),
    )

    print(
        "FINAL_VALID_ANALOG_ROWS =",
        len(result),
    )

    print(
        "FINAL_MISSING_ANALOG_IDS =",
        missing,
    )

    if missing:
        raise RuntimeError(
            "Still missing WA analog jobs"
        )

    if len(result) != 3732:
        raise RuntimeError(
            f"Expected 3732, got "
            f"{len(result)}"
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
                result[jid]
            )

            row[
                "wa_authoritative_source"
            ] = source[jid]

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
        "WA_AUTHORITATIVE_3732_MERGE_PASS"
    )


if __name__ == "__main__":
    main()
PY

python -m py_compile \
    "$SCRIPT_DIR/build_wa_last_missing_v14.py" \
    "$SCRIPT_DIR/generate_wa_last31_twostage_v14.py" \
    "$SCRIPT_DIR/merge_wa_authoritative_v14.py"

echo "WA_LAST31_COMPILE_PASS"

###############################################################################
# 4. BUILD MISSING SET
###############################################################################

python \
"$SCRIPT_DIR/build_wa_last_missing_v14.py" \
    --anchors "$ANCHORS" \
    --base "$BASE_REPAIRED" \
    --pairwise "$PAIRWISE_DIR" \
    --output "$MISSING"

###############################################################################
# 5. GENERATE ONLY REAL 31
###############################################################################

PIDS=()

for DEVICE in 0 1 2 3; do

    LOG="$EXP_LOG/rq3_wa_v14_last31/device_${DEVICE}.log"

    python \
    "$SCRIPT_DIR/generate_wa_last31_twostage_v14.py" \
        --jobs "$MISSING" \
        --output "$FINAL_RETRY_DIR/device_${DEVICE}.jsonl" \
        --model "$MODEL" \
        --device-id "$DEVICE" \
        --world-size 4 \
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
    echo "WA_LAST31_WORKER_FAILURE"
    false
fi

echo "WA_LAST31_WORKERS_COMPLETE"

###############################################################################
# 6. AUTHORITATIVE MERGE
###############################################################################

python \
"$SCRIPT_DIR/merge_wa_authoritative_v14.py" \
    --base "$BASE_REPAIRED" \
    --pairwise "$PAIRWISE_DIR" \
    --last31 "$FINAL_RETRY_DIR" \
    --output "$FINAL_ANALOG_DIR"

###############################################################################
# 7. BUILD EXACT 14928 CONTEXT JOBS
###############################################################################

python \
"$SCRIPT_DIR/merge_wa_analogs_build_context_jobs_v14.py" \
    --shard-dir "$FINAL_ANALOG_DIR" \
    --merged "$ANALOG_MERGED" \
    --audit "$ANALOG_AUDIT" \
    --context-jobs "$CONTEXT_JOBS"

N_CONTEXT_JOBS=$(
    wc -l < "$CONTEXT_JOBS"
)

echo \
    "FINAL_WA_CONTEXT_JOBS=$N_CONTEXT_JOBS"

if [ "$N_CONTEXT_JOBS" -ne 14928 ]; then
    echo "WA_CONTEXT_BUDGET_FAILURE"
    false
fi

echo "WA_EXACT_14928_CONTEXT_BUDGET_PASS"

###############################################################################
# 8. CLEAN NEW CONTEXT OUTPUT DIR BEFORE FIRST REAL LAUNCH
###############################################################################

if [ -d "$CONTEXT_DIR" ]; then
    rm -rf "$CONTEXT_DIR"
fi

mkdir -p "$CONTEXT_DIR"

###############################################################################
# 9. 16-NPU CONTEXT SYNTHESIS
###############################################################################

PIDS=()

for DEVICE in $(seq 0 15); do

    LOG="$EXP_LOG/rq3_wa_v14_last31/context_device_${DEVICE}.log"

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
# 10. POSTPROCESS + BUILD FULL PE+PDS+WA
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
echo "WA AUDIT"
echo "======================================================================"

cat "$WA_AUDIT"

echo
echo "======================================================================"
echo "PE+PDS+WA AUDIT"
echo "======================================================================"

cat "$FULL_AUDIT"

echo
echo "======================================================================"
echo "FINAL CARDINALITY"
echo "======================================================================"

wc -l \
    "$PE_PDS" \
    "$WA_VALID" \
    "$FULL"

echo
echo "======================================================================"
echo "FINAL SHA256"
echo "======================================================================"

sha256sum \
    "$WA_VALID" \
    "$FULL"

echo
echo "RQ3_WA_V14_LAST31_ALL_PASS"
