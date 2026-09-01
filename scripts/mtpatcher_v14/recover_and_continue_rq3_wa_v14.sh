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

ANCHOR_JOBS="$EXP_DATA/rq3_wa_anchor_jobs_v14.jsonl"

ORIGINAL_ANALOG_DIR="$EXP_DATA/rq3_wa_analogs_v14"

REPAIRED_ANALOG_DIR="$EXP_DATA/rq3_wa_analogs_repaired_v14"

RETRY_JOBS="$EXP_DATA/rq3_wa_analog_retry_jobs_v14.jsonl"
RETRY_DIR="$EXP_DATA/rq3_wa_analog_retry_v14"

REPAIR_AUDIT="$EXP_DATA/rq3_wa_analog_repair_audit_v14.json"

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
    "$REPAIRED_ANALOG_DIR" \
    "$RETRY_DIR" \
    "$CONTEXT_DIR" \
    "$EXP_LOG/rq3_wa_v14_recovery"

###############################################################################
# 1. SALVAGE EXISTING RAW ANALOG OUTPUTS
###############################################################################

cat > "$SCRIPT_DIR/salvage_wa_analogs_v14.py" <<'PY'
import argparse
import ast
import json
import re
import shutil
from collections import Counter
from pathlib import Path


def load_jsonl(path):
    rows = []

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


def normalize_source(x):
    if not isinstance(x, str):
        return ""

    return "".join(
        x.strip().split()
    ).casefold()


def extract_container(raw):
    raw = (
        raw.strip()
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
        start >= 0
        and end > start
    ):
        raw = raw[
            start:end + 1
        ]

    parsers = [
        lambda x: json.loads(x),
        lambda x: ast.literal_eval(x),
    ]

    for parser in parsers:
        try:
            obj = parser(raw)

            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    return None


def find_group(obj, names):
    lowered = {
        str(k).strip().casefold():
            v
        for k, v in obj.items()
    }

    for name in names:
        if name in lowered:
            return lowered[name]

    return None


def parse_item(x):
    if isinstance(x, dict):
        lower = {
            str(k).strip().casefold():
                v
            for k, v in x.items()
        }

        src = None
        tgt = None

        for key in (
            "source",
            "chinese",
            "zh",
            "word",
            "phrase",
        ):
            if key in lower:
                src = lower[key]
                break

        for key in (
            "target",
            "english",
            "en",
            "translation",
        ):
            if key in lower:
                tgt = lower[key]
                break

        if (
            isinstance(src, str)
            and isinstance(tgt, str)
        ):
            src = src.strip()
            tgt = tgt.strip()

            if src and tgt:
                return {
                    "source": src,
                    "target": tgt,
                }

    if (
        isinstance(x, (list, tuple))
        and len(x) == 2
        and isinstance(x[0], str)
        and isinstance(x[1], str)
    ):
        return {
            "source":
                x[0].strip(),

            "target":
                x[1].strip(),
        }

    return None


def clean_group(
    arr,
    anchor,
    globally_seen,
):
    if not isinstance(
        arr,
        (list, tuple),
    ):
        return None

    clean = []

    anchor_n = normalize_source(
        anchor
    )

    for item in arr:
        pair = parse_item(item)

        if pair is None:
            continue

        src_n = normalize_source(
            pair["source"]
        )

        if not src_n:
            continue

        if src_n == anchor_n:
            continue

        if src_n in globally_seen:
            continue

        globally_seen.add(src_n)
        clean.append(pair)

        if len(clean) == 2:
            break

    if len(clean) != 2:
        return None

    return clean


def strict_existing(row):
    if not row.get("parse_ok"):
        return None

    obj = row.get("analogs")

    if not isinstance(obj, dict):
        return None

    seen = set()

    category = clean_group(
        obj.get("category"),
        row["source_span"],
        seen,
    )

    semantics = clean_group(
        obj.get("semantics"),
        row["source_span"],
        seen,
    )

    if (
        category is None
        or semantics is None
    ):
        return None

    return {
        "category": category,
        "semantics": semantics,
    }


def salvage(row):
    obj = extract_container(
        row.get(
            "raw_analogy",
            "",
        )
    )

    if obj is None:
        return None

    category_raw = find_group(
        obj,
        (
            "category",
            "categories",
            "categorical",
        ),
    )

    semantics_raw = find_group(
        obj,
        (
            "semantics",
            "semantic",
            "semantic association",
            "semantic associations",
            "co-occurrence",
            "cooccurrence",
        ),
    )

    seen = set()

    category = clean_group(
        category_raw,
        row["source_span"],
        seen,
    )

    semantics = clean_group(
        semantics_raw,
        row["source_span"],
        seen,
    )

    if (
        category is None
        or semantics is None
    ):
        return None

    return {
        "category": category,
        "semantics": semantics,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--original-dir",
        required=True,
    )

    ap.add_argument(
        "--retry-dir",
    )

    ap.add_argument(
        "--anchor-jobs",
        required=True,
    )

    ap.add_argument(
        "--repaired-dir",
        required=True,
    )

    ap.add_argument(
        "--retry-jobs",
        required=True,
    )

    ap.add_argument(
        "--audit",
        required=True,
    )

    args = ap.parse_args()

    original_dir = Path(
        args.original_dir
    )

    retry_dir = (
        Path(args.retry_dir)
        if args.retry_dir
        else None
    )

    repaired_dir = Path(
        args.repaired_dir
    )

    anchors = {
        int(x["analog_job_id"]): x
        for x in load_jsonl(
            Path(args.anchor_jobs)
        )
    }

    if len(anchors) != 3732:
        raise RuntimeError(
            f"Expected 3732 anchors, "
            f"got {len(anchors)}"
        )

    original = {}

    malformed_json_lines = 0

    for device in range(16):
        p = (
            original_dir
            / f"device_{device}.jsonl"
        )

        if not p.exists():
            raise RuntimeError(
                f"Missing original shard {p}"
            )

        rows = load_jsonl(p)

        for row in rows:
            jid = int(
                row["analog_job_id"]
            )

            if jid in original:
                raise RuntimeError(
                    f"Duplicate original "
                    f"analog_job_id={jid}"
                )

            original[jid] = row

    if len(original) != 3732:
        raise RuntimeError(
            f"Expected 3732 original outputs, "
            f"got {len(original)}"
        )

    retry_success = {}

    if (
        retry_dir is not None
        and retry_dir.exists()
    ):
        for device in range(16):
            p = (
                retry_dir
                / f"device_{device}.jsonl"
            )

            for row in load_jsonl(p):
                if not row.get("parse_ok"):
                    continue

                analogs = strict_existing(
                    row
                )

                if analogs is None:
                    continue

                retry_success[
                    int(
                        row[
                            "analog_job_id"
                        ]
                    )
                ] = (
                    row,
                    analogs,
                )

    stats = Counter()
    parse_errors = Counter()

    repaired = {}
    unresolved = []

    for jid in range(3732):
        row = original[jid]

        analogs = strict_existing(
            row
        )

        origin = None

        if analogs is not None:
            stats[
                "original_strict_valid"
            ] += 1

            origin = "original"

        else:
            stats[
                "original_invalid"
            ] += 1

            parse_errors[
                str(
                    row.get(
                        "parse_error",
                        "UNKNOWN",
                    )
                ).split(
                    ":",
                    1,
                )[0]
            ] += 1

            analogs = salvage(row)

            if analogs is not None:
                stats[
                    "salvaged_without_model"
                ] += 1

                origin = "salvage"

        if (
            analogs is None
            and jid in retry_success
        ):
            retry_row, analogs = (
                retry_success[jid]
            )

            row = retry_row

            stats[
                "recovered_by_retry"
            ] += 1

            origin = "retry"

        if analogs is None:
            unresolved.append(
                anchors[jid]
            )
            continue

        out = dict(row)

        out["parse_ok"] = True
        out["parse_error"] = ""
        out["analogs"] = analogs

        out[
            "wa_recovery_origin"
        ] = origin

        repaired[jid] = out

    stats[
        "resolved_total"
    ] = len(repaired)

    stats[
        "unresolved_total"
    ] = len(unresolved)

    # Rebuild repaired shards atomically.
    tmp_dir = repaired_dir.with_name(
        repaired_dir.name + ".tmp"
    )

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    tmp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = {
        i: (
            tmp_dir
            / f"device_{i}.jsonl"
        ).open(
            "w",
            encoding="utf-8",
        )
        for i in range(16)
    }

    try:
        for jid in sorted(repaired):
            device = jid % 16

            files[device].write(
                json.dumps(
                    repaired[jid],
                    ensure_ascii=False,
                )
                + "\n"
            )
    finally:
        for f in files.values():
            f.close()

    if repaired_dir.exists():
        shutil.rmtree(
            repaired_dir
        )

    tmp_dir.rename(
        repaired_dir
    )

    retry_jobs_path = Path(
        args.retry_jobs
    )

    with retry_jobs_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in unresolved:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    audit = {
        "original_outputs":
            len(original),

        "stats":
            dict(stats),

        "parse_error_type_counts":
            dict(parse_errors),

        "unresolved_ids_first100":
            [
                int(x["analog_job_id"])
                for x in unresolved[:100]
            ],

        "protocol":
            "RQ3_WA_V14_FORMAT_RECOVERY",
    }

    Path(args.audit).write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "WA_ANALOG_SALVAGE_PASS"
    )

    print(
        "WA_ANALOG_RETRY_REQUIRED =",
        len(unresolved),
    )


if __name__ == "__main__":
    main()
PY

###############################################################################
# 2. RETRY GENERATOR — ONLY UNRESOLVED WA ANALOG JOBS
###############################################################################

cat > "$SCRIPT_DIR/retry_wa_analogs_qwen3_8b_v14.py" <<'PY'
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
                continue

    return rows


def normalize(x):
    return "".join(
        x.strip().split()
    ).casefold()


def validate(obj, anchor):
    if not isinstance(obj, dict):
        raise ValueError(
            "root_not_dict"
        )

    seen = set()

    result = {}

    for key in (
        "category",
        "semantics",
    ):
        arr = obj.get(key)

        if not isinstance(arr, list):
            raise ValueError(
                f"{key}_not_list"
            )

        if len(arr) != 2:
            raise ValueError(
                f"{key}_count_{len(arr)}"
            )

        clean = []

        for item in arr:
            if not isinstance(
                item,
                dict,
            ):
                raise ValueError(
                    "item_not_dict"
                )

            src = item.get("source")
            tgt = item.get("target")

            if not isinstance(
                src,
                str,
            ):
                raise ValueError(
                    "source_invalid"
                )

            if not isinstance(
                tgt,
                str,
            ):
                raise ValueError(
                    "target_invalid"
                )

            src = src.strip()
            tgt = tgt.strip()

            if not src or not tgt:
                raise ValueError(
                    "empty_pair"
                )

            n = normalize(src)

            if n == normalize(anchor):
                raise ValueError(
                    "analog_equals_anchor"
                )

            if n in seen:
                raise ValueError(
                    "duplicate_analog"
                )

            seen.add(n)

            clean.append(
                {
                    "source": src,
                    "target": tgt,
                }
            )

        result[key] = clean

    if len(seen) != 4:
        raise ValueError(
            "not_four_unique"
        )

    return result


def extract_json(raw):
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
            "json_object_missing"
        )

    return json.loads(
        raw[start:end + 1]
    )


def prompt(row):
    return f"""Generate exactly four Chinese-English analogous phrase pairs for the translation weakness below.

You MUST return valid JSON only.

There are exactly two groups:
- category: exactly 2 pairs from the same category/type
- semantics: exactly 2 semantically associated or commonly co-occurring pairs

All four Chinese phrases must:
- be different from the original phrase
- be different from each other
- preferably be relatively rare or translation-challenging
- be words or short phrases, not sentences

Use EXACTLY this schema:

{{
  "category": [
    {{"source": "中文短语1", "target": "English phrase 1"}},
    {{"source": "中文短语2", "target": "English phrase 2"}}
  ],
  "semantics": [
    {{"source": "中文短语3", "target": "English phrase 3"}},
    {{"source": "中文短语4", "target": "English phrase 4"}}
  ]
}}

Do not output markdown.
Do not output commentary.
Do not add extra keys.

Original Chinese sentence:
{row["source"]}

Original problematic Chinese phrase:
{row["source_span"]}
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
        default=16,
    )

    ap.add_argument(
        "--max-attempts",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=384,
    )

    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
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
        Path(args.jobs)
    )

    assigned = [
        x
        for x in jobs
        if int(
            x["analog_job_id"]
        ) % args.world_size
        == args.device_id
    ]

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Preserve only previous successful retries.
    previous_success = {}

    for row in load_jsonl(
        output_path
    ):
        if row.get("parse_ok"):
            previous_success[
                int(
                    row[
                        "analog_job_id"
                    ]
                )
            ] = row

    pending = [
        row
        for row in assigned
        if int(
            row["analog_job_id"]
        ) not in previous_success
    ]

    print(
        f"DEVICE={args.device_id} "
        f"ASSIGNED={len(assigned)} "
        f"PREVIOUS_SUCCESS="
        f"{len(previous_success)} "
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

    tokenizer.padding_side = (
        "left"
    )

    if (
        tokenizer.pad_token_id
        is None
    ):
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
        "WA_RETRY_MODEL_READY "
        f"device={args.device_id}",
        flush=True,
    )

    final_rows = dict(
        previous_success
    )

    for pos, row in enumerate(
        pending,
        1,
    ):
        success = None
        last_raw = ""
        last_error = ""

        for attempt in range(
            1,
            args.max_attempts + 1,
        ):
            chat = [
                {
                    "role": "user",
                    "content":
                        prompt(row),
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
                output = model.generate(
                    **enc,
                    do_sample=True,
                    temperature=
                        args.temperature,
                    max_new_tokens=
                        args.max_new_tokens,
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
                output[0][prompt_len:],
                skip_special_tokens=True,
            ).strip()

            last_raw = raw

            try:
                analogs = validate(
                    extract_json(raw),
                    row["source_span"],
                )

                success = {
                    **row,

                    "raw_analogy":
                        raw,

                    "parse_ok":
                        True,

                    "parse_error":
                        "",

                    "analogs":
                        analogs,

                    "retry_attempt":
                        attempt,

                    "analog_model":
                        args.model,

                    "temperature":
                        args.temperature,

                    "construction_method":
                        "MT_PATCHER_WA_ANALOG_RETRY_V14",
                }

                break

            except Exception as exc:
                last_error = (
                    type(exc).__name__
                    + ": "
                    + str(exc)
                )

        if success is None:
            success = {
                **row,

                "raw_analogy":
                    last_raw,

                "parse_ok":
                    False,

                "parse_error":
                    last_error,

                "analogs":
                    None,

                "retry_attempt":
                    args.max_attempts,

                "construction_method":
                    "MT_PATCHER_WA_ANALOG_RETRY_V14",
            }

        final_rows[
            int(
                row[
                    "analog_job_id"
                ]
            )
        ] = success

        if (
            pos == 1
            or pos % 10 == 0
            or pos == len(pending)
        ):
            ok_now = sum(
                bool(x.get("parse_ok"))
                for x in
                final_rows.values()
            )

            print(
                f"DEVICE={args.device_id} "
                f"PROCESSED={pos}/"
                f"{len(pending)} "
                f"SUCCESS_TOTAL={ok_now}",
                flush=True,
            )

    tmp = output_path.with_suffix(
        ".jsonl.tmp"
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
            x.get("parse_ok")
        )
        for x
        in final_rows.values()
    )

    print(
        f"WA_RETRY_DEVICE_"
        f"{args.device_id}_COMPLETE "
        f"FAILURES={failures}",
        flush=True,
    )


if __name__ == "__main__":
    main()
PY

###############################################################################
# 3. INITIAL SALVAGE
###############################################################################

python \
"$SCRIPT_DIR/salvage_wa_analogs_v14.py" \
    --original-dir "$ORIGINAL_ANALOG_DIR" \
    --anchor-jobs "$ANCHOR_JOBS" \
    --repaired-dir "$REPAIRED_ANALOG_DIR" \
    --retry-jobs "$RETRY_JOBS" \
    --audit "$REPAIR_AUDIT"

RETRY_COUNT=$(
    wc -l < "$RETRY_JOBS"
)

echo
echo "WA_RETRY_COUNT=$RETRY_COUNT"

###############################################################################
# 4. RETRY ONLY TRUE UNRESOLVED FAILURES
###############################################################################

if [ "$RETRY_COUNT" -gt 0 ]; then

    echo
    echo "======================================================================"
    echo "RETRY UNRESOLVED WA ANALOGS"
    echo "======================================================================"

    mkdir -p "$RETRY_DIR"

    PIDS=()

    for DEVICE in $(seq 0 15); do
        LOG="$EXP_LOG/rq3_wa_v14_recovery/retry_device_${DEVICE}.log"

        python \
        "$SCRIPT_DIR/retry_wa_analogs_qwen3_8b_v14.py" \
            --jobs "$RETRY_JOBS" \
            --output "$RETRY_DIR/device_${DEVICE}.jsonl" \
            --model "$MODEL" \
            --device-id "$DEVICE" \
            --world-size 16 \
            --max-attempts 4 \
            --max-new-tokens 384 \
            --temperature 1.0 \
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
        echo "WA_RETRY_WORKER_FAILURE"
        false
    fi

    echo "WA_ALL_RETRY_WORKERS_COMPLETE"

fi

###############################################################################
# 5. SECOND SALVAGE — OVERLAY SUCCESSFUL RETRIES
###############################################################################

python \
"$SCRIPT_DIR/salvage_wa_analogs_v14.py" \
    --original-dir "$ORIGINAL_ANALOG_DIR" \
    --retry-dir "$RETRY_DIR" \
    --anchor-jobs "$ANCHOR_JOBS" \
    --repaired-dir "$REPAIRED_ANALOG_DIR" \
    --retry-jobs "$RETRY_JOBS" \
    --audit "$REPAIR_AUDIT"

FINAL_UNRESOLVED=$(
    wc -l < "$RETRY_JOBS"
)

echo
echo "FINAL_UNRESOLVED=$FINAL_UNRESOLVED"

if [ "$FINAL_UNRESOLVED" -ne 0 ]; then
    echo "WA_ANALOG_RECOVERY_INCOMPLETE"
    cat "$REPAIR_AUDIT"
    false
fi

echo "WA_ALL_3732_ANALOGS_VALID_PASS"

###############################################################################
# 6. REBUILD EXACT 14928 CONTEXT JOBS
###############################################################################

python \
"$SCRIPT_DIR/merge_wa_analogs_build_context_jobs_v14.py" \
    --shard-dir "$REPAIRED_ANALOG_DIR" \
    --merged "$ANALOG_MERGED" \
    --audit "$ANALOG_AUDIT" \
    --context-jobs "$CONTEXT_JOBS"

ROWS=$(
    wc -l < "$CONTEXT_JOBS"
)

echo "WA_CONTEXT_JOBS=$ROWS"

if [ "$ROWS" -ne 14928 ]; then
    echo "WA_CONTEXT_JOB_CARDINALITY_FAILURE"
    false
fi

echo "WA_CONTEXT_EXACT_PAPER_BUDGET_PASS"

###############################################################################
# 7. GENERATE WA CONTEXTS
###############################################################################

echo
echo "======================================================================"
echo "GENERATE 14928 WA CONTEXTS"
echo "======================================================================"

mkdir -p "$CONTEXT_DIR"

PIDS=()

for DEVICE in $(seq 0 15); do
    LOG="$EXP_LOG/rq3_wa_v14_recovery/context_device_${DEVICE}.log"

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
# 8. MERGE WA + FROZEN PE+PDS-V13
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
# 9. FINAL AUDIT
###############################################################################

echo
echo "======================================================================"
echo "WA FINAL AUDIT"
echo "======================================================================"

cat "$REPAIR_AUDIT"

echo

cat "$ANALOG_AUDIT"

echo

cat "$WA_AUDIT"

echo

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
echo "RQ3_WA_V14_RECOVERY_ALL_PASS"
