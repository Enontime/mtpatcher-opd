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

ORIGINAL_DIR="$EXP_DATA/rq3_wa_analogs_v14"
OLD_RETRY_DIR="$EXP_DATA/rq3_wa_analog_retry_v14"

FINAL_JOBS="$EXP_DATA/rq3_wa_analog_retry_jobs_v14.jsonl"
FINAL_RETRY_DIR="$EXP_DATA/rq3_wa_analog_finalretry_v14"

REPAIRED_DIR="$EXP_DATA/rq3_wa_analogs_repaired_v14"

ANALOG_MERGED="$EXP_DATA/rq3_wa_analogs_merged_v14.jsonl"
ANALOG_AUDIT="$EXP_DATA/rq3_wa_analogs_audit_v14.json"

CONTEXT_JOBS="$EXP_DATA/rq3_wa_context_jobs_v14.jsonl"
CONTEXT_DIR="$EXP_DATA/rq3_wa_contexts_v14"

PE_PDS="$EXP_DATA/rq3_pe_plus_pds_v13_paperbudget.jsonl"

WA_VALID="$EXP_DATA/rq3_wa_valid_v14.jsonl"
WA_AUDIT="$EXP_DATA/rq3_wa_audit_v14.json"

FULL="$EXP_DATA/rq3_pe_pds_wa_v14.jsonl"
FULL_AUDIT="$EXP_DATA/rq3_pe_pds_wa_v14_audit.json"

mkdir -p \
    "$FINAL_RETRY_DIR" \
    "$REPAIRED_DIR" \
    "$CONTEXT_DIR" \
    "$EXP_LOG/rq3_wa_v14_final"

###############################################################################
# 1. FINAL 33: GENERATE ONE ANALOG PAIR AT A TIME
###############################################################################

cat > "$SCRIPT_DIR/retry_wa_pairwise_v14.py" <<'PY'
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

    if not Path(path).exists():
        return rows

    with open(
        path,
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


def extract_pair(raw):
    raw = raw.strip()

    raw = re.sub(
        r"^```(?:json)?\s*",
        "",
        raw,
        flags=re.I,
    )

    raw = re.sub(
        r"\s*```$",
        "",
        raw,
    )

    start = raw.find("{")
    end = raw.rfind("}")

    if (
        start < 0
        or end <= start
    ):
        raise ValueError(
            "JSON object missing"
        )

    obj = json.loads(
        raw[start:end + 1]
    )

    if not isinstance(obj, dict):
        raise ValueError(
            "root is not dict"
        )

    src = obj.get("source")
    tgt = obj.get("target")

    if (
        not isinstance(src, str)
        or not isinstance(tgt, str)
    ):
        raise ValueError(
            "source/target invalid"
        )

    src = src.strip()
    tgt = tgt.strip()

    if not src or not tgt:
        raise ValueError(
            "empty bilingual pair"
        )

    return {
        "source": src,
        "target": tgt,
    }


def build_prompt(
    row,
    aspect,
    selected,
):
    original = row["source_span"]

    forbidden = [
        original,
        *selected,
    ]

    forbidden_text = "\n".join(
        f"- {x}"
        for x in forbidden
    )

    if aspect == "category":
        relation = (
            "belong to the same category or type "
            "as the original Chinese phrase"
        )
    else:
        relation = (
            "be semantically associated with, "
            "frequently co-occur with, or naturally "
            "appear in a closely related context to "
            "the original Chinese phrase"
        )

    return f"""You are a Chinese-English language expert.

Generate ONE Chinese-English bilingual phrase pair.

The new Chinese phrase should {relation}.

Prefer relatively rare or translation-challenging knowledge.

Rules:
- Output a word or short phrase, not a full sentence.
- Do not reuse the original phrase.
- Do not reuse any previously selected phrase.
- Give a natural English translation.
- Return ONLY valid JSON.
- Use exactly two keys: source and target.
- No markdown.
- No explanation.

Forbidden Chinese phrases:
{forbidden_text}

Original sentence:
{row["source"]}

Original problematic phrase:
{original}

Required JSON:
{{"source":"中文短语","target":"English translation"}}
"""


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

    device_id = args.device_id

    torch.npu.set_device(
        device_id
    )

    device = f"npu:{device_id}"

    random.seed(
        args.seed + device_id
    )

    torch.manual_seed(
        args.seed + device_id
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
        == device_id
    ]

    output_path = Path(
        args.output
    )

    previous = {}

    for row in load_jsonl(
        output_path
    ):
        if row.get("parse_ok"):
            previous[
                int(row["analog_job_id"])
            ] = row

    pending = [
        row
        for row in assigned
        if int(row["analog_job_id"])
        not in previous
    ]

    print(
        f"DEVICE={device_id} "
        f"ASSIGNED={len(assigned)} "
        f"PREVIOUS={len(previous)} "
        f"PENDING={len(pending)}",
        flush=True,
    )

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
        f"PAIRWISE_MODEL_READY device={device_id}",
        flush=True,
    )

    final_rows = dict(
        previous
    )

    for pos, row in enumerate(
        pending,
        1,
    ):
        selected = []
        result = {
            "category": [],
            "semantics": [],
        }

        failure = ""

        for aspect in (
            "category",
            "semantics",
        ):
            for rank in range(2):

                pair = None

                for attempt in range(
                    1,
                    9,
                ):
                    chat = [
                        {
                            "role": "user",
                            "content":
                                build_prompt(
                                    row,
                                    aspect,
                                    selected,
                                ),
                        }
                    ]

                    rendered = (
                        tokenizer
                        .apply_chat_template(
                            chat,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    )

                    enc = tokenizer(
                        rendered,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )

                    enc = {
                        k: v.to(device)
                        for k, v
                        in enc.items()
                    }

                    with torch.inference_mode():
                        out = model.generate(
                            **enc,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            max_new_tokens=128,
                            pad_token_id=
                                tokenizer.pad_token_id,
                            eos_token_id=
                                tokenizer.eos_token_id,
                        )

                    prompt_len = (
                        enc["input_ids"]
                        .shape[1]
                    )

                    raw = tokenizer.decode(
                        out[0][prompt_len:],
                        skip_special_tokens=True,
                    ).strip()

                    try:
                        candidate = (
                            extract_pair(raw)
                        )

                        n = norm(
                            candidate["source"]
                        )

                        if n == norm(
                            row["source_span"]
                        ):
                            raise ValueError(
                                "equals original anchor"
                            )

                        if any(
                            n == norm(x)
                            for x in selected
                        ):
                            raise ValueError(
                                "duplicate selected analog"
                            )

                        pair = candidate
                        break

                    except Exception as exc:
                        failure = (
                            f"{aspect}/{rank}/"
                            f"attempt{attempt}: "
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        )

                if pair is None:
                    break

                result[
                    aspect
                ].append(pair)

                selected.append(
                    pair["source"]
                )

            if len(
                result[aspect]
            ) != 2:
                break

        parse_ok = (
            len(result["category"]) == 2
            and
            len(result["semantics"]) == 2
            and
            len({
                norm(x["source"])
                for aspect in result.values()
                for x in aspect
            }) == 4
        )

        final_rows[
            int(row["analog_job_id"])
        ] = {
            **row,

            "parse_ok":
                parse_ok,

            "parse_error":
                "" if parse_ok
                else failure,

            "analogs":
                result if parse_ok
                else None,

            "wa_recovery_origin":
                "pairwise_final_retry",

            "construction_method":
                "MT_PATCHER_WA_PAIRWISE_RETRY_V14",
        }

        print(
            f"DEVICE={device_id} "
            f"JOB={row['analog_job_id']} "
            f"OK={parse_ok} "
            f"PROGRESS={pos}/{len(pending)}",
            flush=True,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = output_path.with_suffix(
        ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as f:
        for jid in sorted(
            final_rows
        ):
            f.write(
                json.dumps(
                    final_rows[jid],
                    ensure_ascii=False,
                )
                + "\n"
            )

    tmp.replace(
        output_path
    )

    failures = sum(
        not bool(
            row.get("parse_ok")
        )
        for row
        in final_rows.values()
    )

    print(
        f"PAIRWISE_DEVICE_{device_id}_COMPLETE "
        f"FAILURES={failures}",
        flush=True,
    )


if __name__ == "__main__":
    main()
PY

###############################################################################
# 2. FINAL MERGER:
# ORIGINAL + ORIGINAL SALVAGE + OLD RETRY + PAIRWISE RETRY
###############################################################################

cat > "$SCRIPT_DIR/final_merge_wa_analogs_v14.py" <<'PY'
import argparse
import ast
import json
import re
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


def validate(obj, anchor):
    if not isinstance(obj, dict):
        return None

    seen = set()
    clean = {}

    for aspect in (
        "category",
        "semantics",
    ):
        arr = obj.get(aspect)

        if (
            not isinstance(arr, list)
            or len(arr) != 2
        ):
            return None

        result = []

        for item in arr:
            if not isinstance(
                item,
                dict,
            ):
                return None

            src = item.get("source")
            tgt = item.get("target")

            if (
                not isinstance(src, str)
                or
                not isinstance(tgt, str)
            ):
                return None

            src = src.strip()
            tgt = tgt.strip()

            if not src or not tgt:
                return None

            n = norm(src)

            if n == norm(anchor):
                return None

            if n in seen:
                return None

            seen.add(n)

            result.append(
                {
                    "source": src,
                    "target": tgt,
                }
            )

        clean[aspect] = result

    if len(seen) != 4:
        return None

    return clean


def salvage_raw(row):
    raw = str(
        row.get(
            "raw_analogy",
            "",
        )
    ).strip()

    raw = (
        raw
        .replace("“", '"')
        .replace("”", '"')
    )

    raw = re.sub(
        r"^```(?:json)?\s*",
        "",
        raw,
        flags=re.I,
    )

    raw = re.sub(
        r"\s*```$",
        "",
        raw,
    )

    start = raw.find("{")
    end = raw.rfind("}")

    if (
        start < 0
        or end <= start
    ):
        return None

    chunk = raw[
        start:end + 1
    ]

    obj = None

    for parser in (
        json.loads,
        ast.literal_eval,
    ):
        try:
            obj = parser(chunk)
            break
        except Exception:
            pass

    if not isinstance(obj, dict):
        return None

    lowered = {
        str(k).strip().casefold():
            v
        for k, v in obj.items()
    }

    category = lowered.get(
        "category"
    )

    semantics = (
        lowered.get("semantics")
        or lowered.get("semantic")
    )

    candidate = {
        "category":
            category,

        "semantics":
            semantics,
    }

    return validate(
        candidate,
        row["source_span"],
    )


def load_map(directory):
    result = {}

    directory = Path(
        directory
    )

    if not directory.exists():
        return result

    for p in directory.glob(
        "device_*.jsonl"
    ):
        for row in load_jsonl(p):
            jid = int(
                row["analog_job_id"]
            )

            result[jid] = row

    return result


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--original-dir",
        required=True,
    )

    ap.add_argument(
        "--old-retry-dir",
        required=True,
    )

    ap.add_argument(
        "--final-retry-dir",
        required=True,
    )

    ap.add_argument(
        "--output-dir",
        required=True,
    )

    args = ap.parse_args()

    original = load_map(
        args.original_dir
    )

    old_retry = load_map(
        args.old_retry_dir
    )

    final_retry = load_map(
        args.final_retry_dir
    )

    if len(original) != 3732:
        raise RuntimeError(
            f"Original rows={len(original)}"
        )

    resolved = {}
    stats = Counter()

    unresolved = []

    for jid in range(3732):
        base = original[jid]

        analogs = None
        source = None

        if base.get("parse_ok"):
            analogs = validate(
                base.get("analogs"),
                base["source_span"],
            )

            if analogs is not None:
                source = "original"

        if analogs is None:
            analogs = salvage_raw(
                base
            )

            if analogs is not None:
                source = (
                    "original_salvage"
                )

        if (
            analogs is None
            and jid in old_retry
        ):
            candidate = (
                old_retry[jid]
            )

            if candidate.get(
                "parse_ok"
            ):
                analogs = validate(
                    candidate.get(
                        "analogs"
                    ),
                    base["source_span"],
                )

                if analogs is not None:
                    base = candidate
                    source = "old_retry"

        if (
            analogs is None
            and jid in final_retry
        ):
            candidate = (
                final_retry[jid]
            )

            if candidate.get(
                "parse_ok"
            ):
                analogs = validate(
                    candidate.get(
                        "analogs"
                    ),
                    original[jid][
                        "source_span"
                    ],
                )

                if analogs is not None:
                    base = candidate
                    source = (
                        "pairwise_final"
                    )

        if analogs is None:
            unresolved.append(
                jid
            )
            continue

        row = dict(base)

        row["analogs"] = analogs
        row["parse_ok"] = True
        row["parse_error"] = ""
        row[
            "wa_final_resolution"
        ] = source

        resolved[jid] = row

        stats[source] += 1

    print(
        "FINAL_RESOLUTION_COUNTS =",
        dict(stats),
    )

    print(
        "FINAL_RESOLVED =",
        len(resolved),
    )

    print(
        "FINAL_UNRESOLVED =",
        len(unresolved),
    )

    print(
        "UNRESOLVED_IDS =",
        unresolved,
    )

    if unresolved:
        raise RuntimeError(
            "WA final pairwise repair "
            "still has unresolved jobs"
        )

    if len(resolved) != 3732:
        raise RuntimeError(
            "WA resolution cardinality "
            "mismatch"
        )

    out_dir = Path(
        args.output_dir
    )

    tmp = out_dir.with_name(
        out_dir.name + ".finaltmp"
    )

    if tmp.exists():
        shutil.rmtree(tmp)

    tmp.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = {
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
            device = jid % 16

            files[
                device
            ].write(
                json.dumps(
                    resolved[jid],
                    ensure_ascii=False,
                )
                + "\n"
            )
    finally:
        for f in files.values():
            f.close()

    if out_dir.exists():
        shutil.rmtree(
            out_dir
        )

    tmp.rename(
        out_dir
    )

    print(
        "WA_ALL_3732_FINAL_REPAIR_PASS"
    )


if __name__ == "__main__":
    main()
PY

python -m py_compile \
    "$SCRIPT_DIR/retry_wa_pairwise_v14.py" \
    "$SCRIPT_DIR/final_merge_wa_analogs_v14.py"

echo "WA_FINAL_REPAIR_COMPILE_PASS"

###############################################################################
# 3. RUN THE 33-JOB PAIRWISE REPAIR ON 4 NPUs
###############################################################################

FINAL_COUNT=$(
    wc -l < "$FINAL_JOBS"
)

echo "FINAL_REPAIR_JOBS=$FINAL_COUNT"

if [ "$FINAL_COUNT" -ne 33 ]; then
    echo "WARNING_EXPECTED_33_GOT_$FINAL_COUNT"
fi

PIDS=()

for DEVICE in 0 1 2 3; do

    LOG="$EXP_LOG/rq3_wa_v14_final/pairwise_device_${DEVICE}.log"

    python \
    "$SCRIPT_DIR/retry_wa_pairwise_v14.py" \
        --jobs "$FINAL_JOBS" \
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
    echo "WA_PAIRWISE_RETRY_WORKER_FAILURE"
    false
fi

echo "WA_PAIRWISE_RETRY_WORKERS_COMPLETE"

###############################################################################
# 4. MERGE ALL THREE SOURCES INTO EXACT 3732 VALID ANALOG JOBS
###############################################################################

python \
"$SCRIPT_DIR/final_merge_wa_analogs_v14.py" \
    --original-dir "$ORIGINAL_DIR" \
    --old-retry-dir "$OLD_RETRY_DIR" \
    --final-retry-dir "$FINAL_RETRY_DIR" \
    --output-dir "$REPAIRED_DIR"

###############################################################################
# 5. BUILD EXACT 14928 CONTEXT JOBS
###############################################################################

python \
"$SCRIPT_DIR/merge_wa_analogs_build_context_jobs_v14.py" \
    --shard-dir "$REPAIRED_DIR" \
    --merged "$ANALOG_MERGED" \
    --audit "$ANALOG_AUDIT" \
    --context-jobs "$CONTEXT_JOBS"

CONTEXT_JOB_COUNT=$(
    wc -l < "$CONTEXT_JOBS"
)

echo "WA_CONTEXT_JOB_COUNT=$CONTEXT_JOB_COUNT"

if [ "$CONTEXT_JOB_COUNT" -ne 14928 ]; then
    echo "WA_CONTEXT_JOB_CARDINALITY_FAILURE"
    false
fi

echo "WA_14928_CONTEXT_JOBS_PASS"

###############################################################################
# 6. GENERATE ALL WA CONTEXTS — RESUME SAFE
###############################################################################

PIDS=()

for DEVICE in $(seq 0 15); do

    LOG="$EXP_LOG/rq3_wa_v14_final/context_device_${DEVICE}.log"

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
# 7. PAPER-STYLE POSTPROCESS + BUILD PE+PDS+WA
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
# 8. FINAL AUDIT
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
echo "PE + PDS + WA AUDIT"
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
echo "RQ3_WA_V14_FINAL_ALL_PASS"
