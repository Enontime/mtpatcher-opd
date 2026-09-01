#!/usr/bin/env bash
set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="mtpatcher_v3_full6565_20260823"

SCRIPT_DIR="$ROOT/scripts/mtpatcher_v11"
EXP_DATA="$DATA_ROOT/$EXP"
EXP_RUN="$RUN_ROOT/$EXP"
EXP_LOG="$LOG_ROOT/$EXP"

PE="$EXP_DATA/pe_k1_clean3732.jsonl"
JOBS="$EXP_DATA/rq3_pds_jobs_v11.jsonl"

PDS_DIR="$EXP_DATA/rq3_pds_v11"
PDS_VALID="$EXP_DATA/rq3_pds_valid_v11.jsonl"
PDS_AUDIT="$EXP_DATA/rq3_pds_audit_v11.json"

PE_PDS="$EXP_DATA/rq3_pe_plus_pds_v11.jsonl"
PE_PDS_AUDIT="$EXP_DATA/rq3_pe_plus_pds_audit_v11.json"

MODEL="$MODEL_ROOT/Qwen3-8B"

RUNNER="$SCRIPT_DIR/run_rq3_pds_generation_v11.sh"
MASTER_LOG="$EXP_LOG/rq3_pds_generation_v11.log"

mkdir -p "$SCRIPT_DIR" "$PDS_DIR" "$EXP_RUN" "$EXP_LOG"

###############################################################################
# 1. Build PDS jobs
###############################################################################

cat > "$SCRIPT_DIR/build_pds_jobs_v11.py" <<'PY'
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def clean_text(x):
    if not isinstance(x, str):
        return ""
    return x.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pe", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--repeat", type=int, default=4)
    args = ap.parse_args()

    pe_path = Path(args.pe)
    output_path = Path(args.output)

    rows = []
    with pe_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    if len(rows) != 3732:
        raise RuntimeError(
            f"Expected frozen PE3732 input, got {len(rows)}"
        )

    jobs = []
    error_occurrences = 0
    invalid_errors = 0
    multi_error_rows = 0
    error_type_counts = {}

    job_id = 0

    for row_pos, row in enumerate(rows):
        index = row.get("index")
        source = clean_text(row.get("source"))
        student_translation = clean_text(
            row.get("student_translation")
        )

        errors = row.get("feedback_errors")

        if not isinstance(errors, list) or not errors:
            raise RuntimeError(
                f"PE row has no feedback_errors: row_pos={row_pos}"
            )

        if len(errors) > 1:
            multi_error_rows += 1

        for error_index, err in enumerate(errors):
            if not isinstance(err, dict):
                invalid_errors += 1
                continue

            source_span = clean_text(err.get("source_span"))
            correction = clean_text(err.get("correction"))
            translation_span = clean_text(
                err.get("translation_span")
            )
            error_type = clean_text(err.get("error_type"))
            explanation = clean_text(err.get("explanation"))

            if not source_span or not correction:
                invalid_errors += 1
                continue

            if source_span not in source:
                invalid_errors += 1
                continue

            error_occurrences += 1

            error_type_counts[error_type] = (
                error_type_counts.get(error_type, 0) + 1
            )

            for slot in range(args.repeat):
                jobs.append(
                    {
                        "job_id": job_id,
                        "parent_row_pos": row_pos,
                        "parent_index": index,
                        "error_index": error_index,
                        "pds_slot": slot,
                        "source": source,
                        "student_translation":
                            student_translation,
                        "source_span": source_span,
                        "translation_span":
                            translation_span,
                        "correction": correction,
                        "error_type": error_type,
                        "explanation": explanation,
                        "construction_method":
                            "MT_PATCHER_PDS_QWEN3_8B_V11",
                    }
                )
                job_id += 1

    if invalid_errors:
        raise RuntimeError(
            f"Invalid feedback errors found: {invalid_errors}"
        )

    if not jobs:
        raise RuntimeError("No PDS jobs constructed")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open("w", encoding="utf-8") as f:
        for job in jobs:
            f.write(
                json.dumps(
                    job,
                    ensure_ascii=False
                ) + "\n"
            )

    print("PE_ROWS =", len(rows))
    print("ERROR_OCCURRENCES =", error_occurrences)
    print("MULTI_ERROR_ROWS =", multi_error_rows)
    print("PDS_REPEAT =", args.repeat)
    print("PDS_TOTAL_JOBS =", len(jobs))
    print("ERROR_TYPE_COUNTS =", error_type_counts)
    print("PE_SHA256 =", sha256(pe_path))
    print("JOBS_SHA256 =", sha256(output_path))
    print("PDS_JOB_BUILD_PASS")


if __name__ == "__main__":
    main()
PY

###############################################################################
# 2. Parallel Qwen3-8B PDS generator
###############################################################################

cat > "$SCRIPT_DIR/generate_pds_qwen3_8b_v11.py" <<'PY'
import argparse
import json
import os
import random
import re
from pathlib import Path

import torch
import torch_npu
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


def clean_text(x):
    if not isinstance(x, str):
        return ""
    return x.strip()


def parse_completion(text):
    src = ""
    tgt = ""

    m_src = re.search(
        r"中文句子\s*[:：]\s*(.+)",
        text
    )
    m_tgt = re.search(
        r"英文句子\s*[:：]\s*(.+)",
        text
    )

    if m_src:
        src = m_src.group(1).strip()

    if m_tgt:
        tgt = m_tgt.group(1).strip()

    return src, tgt


def build_prompt(job):
    original = job["source"]
    source_span = job["source_span"]
    correction = job["correction"]
    slot = int(job["pds_slot"]) + 1

    return f"""你是一名高质量的中英平行语料合成器。

下面给出一个学生翻译模型曾经出错的中文短语 P，
以及它在英语中的正确翻译 Q。

原始中文句子：
{original}

P：{source_span}
Q：{correction}

请生成一个新的中英平行句对，用于让学生在新的语境中学习这一翻译知识。

要求：
1. 新中文句子必须原样包含 P。
2. 新英文句子必须自然地包含 Q。
3. 中英文必须语义完全对应。
4. 新句子应与原句保持大致相似的领域、语体或风格。
5. 新句子的具体语义和场景应与原句明显不同，不能只是改几个词。
6. 英文必须自然、完整、符合母语表达。
7. 这是针对该知识点生成的第 {slot} 个独立语境，请尽量避免与其他语境雷同。
8. 不要解释，不要输出分析过程。

严格只输出两行：

中文句子: <新的中文句子>
英文句子: <对应英文翻译>
"""


def load_jobs(path, device_id, world_size):
    jobs = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            jid = int(row["job_id"])

            if jid % world_size == device_id:
                jobs.append(row)

    return jobs


def load_completed(output):
    completed = set()

    if not Path(output).exists():
        return completed

    with open(output, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                completed.add(int(row["job_id"]))
            except Exception:
                continue

    return completed


def apply_chat(tokenizer, prompt):
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device-id", type=int, required=True)
    ap.add_argument("--world-size", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--seed", type=int, default=20260825)
    args = ap.parse_args()

    device_id = args.device_id
    device = f"npu:{device_id}"

    torch.npu.set_device(device_id)

    seed = args.seed + device_id
    random.seed(seed)
    torch.manual_seed(seed)

    jobs = load_jobs(
        args.jobs,
        device_id,
        args.world_size
    )

    completed = load_completed(args.output)

    pending = [
        j for j in jobs
        if int(j["job_id"]) not in completed
    ]

    print(
        f"DEVICE={device_id} "
        f"ASSIGNED={len(jobs)} "
        f"COMPLETED={len(completed)} "
        f"PENDING={len(pending)}",
        flush=True
    )

    if not pending:
        print(
            f"PDS_DEVICE_{device_id}_ALREADY_COMPLETE",
            flush=True
        )
        return

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )

    model.to(device)
    model.eval()

    print(
        f"PDS_MODEL_READY device={device_id}",
        flush=True
    )

    Path(args.output).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    done_now = 0

    with open(
        args.output,
        "a",
        encoding="utf-8"
    ) as fout:

        for start in range(
            0,
            len(pending),
            args.batch_size
        ):
            batch_jobs = pending[
                start:start + args.batch_size
            ]

            prompts = [
                apply_chat(
                    tokenizer,
                    build_prompt(job)
                )
                for job in batch_jobs
            ]

            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=768,
            )

            encoded = {
                k: v.to(device)
                for k, v in encoded.items()
            }

            input_len = encoded[
                "input_ids"
            ].shape[1]

            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=1.5,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

            new_tokens = generated[
                :,
                input_len:
            ]

            decoded = tokenizer.batch_decode(
                new_tokens,
                skip_special_tokens=True,
            )

            for job, raw in zip(
                batch_jobs,
                decoded
            ):
                synthesized_source, synthesized_target = (
                    parse_completion(raw)
                )

                result = dict(job)

                result.update(
                    {
                        "generator_model":
                            "Qwen3-8B",
                        "generator_temperature":
                            1.5,
                        "enable_thinking":
                            False,
                        "raw_generation":
                            raw,
                        "synthesized_source":
                            clean_text(
                                synthesized_source
                            ),
                        "synthesized_target":
                            clean_text(
                                synthesized_target
                            ),
                        "parse_ok":
                            bool(
                                synthesized_source
                                and synthesized_target
                            ),
                    }
                )

                fout.write(
                    json.dumps(
                        result,
                        ensure_ascii=False
                    ) + "\n"
                )

                fout.flush()
                done_now += 1

            if (
                done_now == len(batch_jobs)
                or done_now % 80 == 0
            ):
                print(
                    f"DEVICE={device_id} "
                    f"GENERATED={done_now}/"
                    f"{len(pending)}",
                    flush=True
                )

    print(
        f"PDS_DEVICE_{device_id}_COMPLETE "
        f"generated={done_now}",
        flush=True
    )


if __name__ == "__main__":
    main()
PY

###############################################################################
# 3. Merge and audit all generated PDS pairs
###############################################################################

cat > "$SCRIPT_DIR/merge_and_audit_pds_v11.py" <<'PY'
import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def norm(x):
    x = unicodedata.normalize(
        "NFKC",
        str(x)
    )
    x = re.sub(r"\s+", " ", x)
    return x.strip().casefold()


def norm_no_space(x):
    return re.sub(
        r"\s+",
        "",
        unicodedata.normalize(
            "NFKC",
            str(x)
        )
    ).casefold()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--world-size", type=int, default=16)
    args = ap.parse_args()

    jobs = {}

    with open(
        args.jobs,
        encoding="utf-8"
    ) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                jobs[int(row["job_id"])] = row

    results = {}

    shard_paths = []

    for device_id in range(args.world_size):
        path = (
            Path(args.shard_dir)
            / f"device_{device_id}.jsonl"
        )

        shard_paths.append(path)

        if not path.exists():
            raise RuntimeError(
                f"Missing PDS shard: {path}"
            )

        with path.open(
            encoding="utf-8"
        ) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                results[int(row["job_id"])] = row

    missing = sorted(
        set(jobs) - set(results)
    )

    if missing:
        raise RuntimeError(
            f"Missing generated jobs: "
            f"{len(missing)}, first={missing[:20]}"
        )

    reject_counts = Counter()
    coverage = defaultdict(int)

    valid = []
    seen_pairs = set()

    for jid in sorted(jobs):
        row = results[jid]

        src = str(
            row.get(
                "synthesized_source",
                ""
            )
        ).strip()

        tgt = str(
            row.get(
                "synthesized_target",
                ""
            )
        ).strip()

        original = str(
            row.get(
                "source",
                ""
            )
        ).strip()

        source_span = str(
            row.get(
                "source_span",
                ""
            )
        ).strip()

        correction = str(
            row.get(
                "correction",
                ""
            )
        ).strip()

        reason = None

        if not row.get("parse_ok"):
            reason = "parse_fail"

        elif not src or not tgt:
            reason = "empty_pair"

        elif (
            norm_no_space(source_span)
            not in norm_no_space(src)
        ):
            reason = "source_span_missing"

        elif norm(correction) not in norm(tgt):
            reason = "correction_missing"

        elif norm_no_space(src) == norm_no_space(
            original
        ):
            reason = "source_not_extended"

        elif norm(src) == norm(tgt):
            reason = "src_tgt_identical"

        pair_key = (
            norm(src),
            norm(tgt)
        )

        if reason is None and pair_key in seen_pairs:
            reason = "duplicate_pair"

        if reason is not None:
            reject_counts[reason] += 1
            continue

        seen_pairs.add(pair_key)

        parent_key = (
            row.get("parent_index"),
            row.get("error_index"),
        )

        coverage[parent_key] += 1

        valid.append(
            {
                "job_id": jid,
                "parent_index":
                    row.get("parent_index"),
                "parent_row_pos":
                    row.get("parent_row_pos"),
                "error_index":
                    row.get("error_index"),
                "pds_slot":
                    row.get("pds_slot"),
                "source":
                    src,
                "target_translation":
                    tgt,
                "source_span":
                    source_span,
                "correction":
                    correction,
                "error_type":
                    row.get("error_type"),
                "original_source":
                    original,
                "construction_method":
                    "MT_PATCHER_PDS_QWEN3_8B_V11",
            }
        )

    coverage_hist = Counter(
        coverage.values()
    )

    all_parent_keys = {
        (
            row.get("parent_index"),
            row.get("error_index")
        )
        for row in jobs.values()
    }

    zero_coverage = (
        len(all_parent_keys)
        - len(coverage)
    )

    coverage_hist[0] = zero_coverage

    Path(args.output).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        args.output,
        "w",
        encoding="utf-8"
    ) as f:
        for row in valid:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                ) + "\n"
            )

    valid_ratio = (
        len(valid) / len(jobs)
        if jobs else 0.0
    )

    audit = {
        "expected_jobs": len(jobs),
        "generated_jobs": len(results),
        "valid_pairs": len(valid),
        "valid_ratio": valid_ratio,
        "reject_counts":
            dict(reject_counts),
        "error_occurrences":
            len(all_parent_keys),
        "coverage_histogram":
            {
                str(k): coverage_hist[k]
                for k in sorted(
                    coverage_hist
                )
            },
        "jobs_sha256":
            sha256(args.jobs),
        "output_sha256":
            sha256(args.output),
        "world_size":
            args.world_size,
        "method":
            "MT_PATCHER_PDS_QWEN3_8B_V11",
    }

    with open(
        args.audit,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            audit,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False
        )
    )

    if valid_ratio < 0.50:
        raise RuntimeError(
            "PDS valid ratio below 0.50; "
            "do not train on this dataset"
        )

    print("PDS_MERGE_AUDIT_PASS")


if __name__ == "__main__":
    main()
PY

###############################################################################
# 4. Build PE + PDS dataset in the exact PE SFT schema
###############################################################################

cat > "$SCRIPT_DIR/build_pe_plus_pds_v11.py" <<'PY'
import argparse
import hashlib
import json
import random
from pathlib import Path


PROMPT_PREFIX = (
    "Translate the following text into English "
    "without additional explanations:\n\n"
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def pair_key(source, target):
    return (
        str(source).strip(),
        str(target).strip()
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pe", required=True)
    ap.add_argument("--pds", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--seed", type=int, default=20260825)
    args = ap.parse_args()

    pe_rows = []

    with open(
        args.pe,
        encoding="utf-8"
    ) as f:
        for line in f:
            if line.strip():
                pe_rows.append(
                    json.loads(line)
                )

    if len(pe_rows) != 3732:
        raise RuntimeError(
            f"Expected PE3732, got "
            f"{len(pe_rows)}"
        )

    pds_rows = []

    with open(
        args.pds,
        encoding="utf-8"
    ) as f:
        for line in f:
            if line.strip():
                pds_rows.append(
                    json.loads(line)
                )

    combined = []
    seen = set()

    pe_kept = 0
    pds_kept = 0
    duplicates = 0

    for row in pe_rows:
        source = str(
            row["source"]
        ).strip()

        target = str(
            row["target_translation"]
        ).strip()

        key = pair_key(
            source,
            target
        )

        if key in seen:
            duplicates += 1
            continue

        seen.add(key)

        new_row = dict(row)
        new_row[
            "rq3_data_component"
        ] = "PE"

        combined.append(new_row)
        pe_kept += 1

    for row in pds_rows:
        source = str(
            row["source"]
        ).strip()

        target = str(
            row["target_translation"]
        ).strip()

        key = pair_key(
            source,
            target
        )

        if key in seen:
            duplicates += 1
            continue

        seen.add(key)

        message = {
            "role": "user",
            "content":
                PROMPT_PREFIX
                + source
                + "\n\n",
        }

        new_row = {
            "index":
                f"pds_v11_{row['job_id']}",
            "source":
                source,
            "messages":
                [message],
            "target_translation":
                target,
            "student_translation":
                "",
            "feedback_errors":
                [
                    {
                        "source_span":
                            row["source_span"],
                        "translation_span":
                            "",
                        "error_type":
                            row.get(
                                "error_type",
                                ""
                            ),
                        "explanation":
                            "PDS synthesized context",
                        "correction":
                            row["correction"],
                    }
                ],
            "construction_method":
                "MT_PATCHER_PDS_QWEN3_8B_V11",
            "rq3_data_component":
                "PDS",
            "parent_index":
                row["parent_index"],
            "parent_row_pos":
                row["parent_row_pos"],
            "error_index":
                row["error_index"],
            "pds_slot":
                row["pds_slot"],
        }

        combined.append(new_row)
        pds_kept += 1

    rng = random.Random(args.seed)
    rng.shuffle(combined)

    Path(args.output).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        args.output,
        "w",
        encoding="utf-8"
    ) as f:
        for row in combined:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                ) + "\n"
            )

    audit = {
        "pe_input_rows":
            len(pe_rows),
        "pds_input_rows":
            len(pds_rows),
        "pe_kept":
            pe_kept,
        "pds_kept":
            pds_kept,
        "combined_rows":
            len(combined),
        "duplicates_removed":
            duplicates,
        "shuffle_seed":
            args.seed,
        "pe_sha256":
            sha256(args.pe),
        "pds_sha256":
            sha256(args.pds),
        "combined_sha256":
            sha256(args.output),
        "method":
            "MT_PATCHER_PE_PLUS_PDS_QWEN3_V11",
    }

    with open(
        args.audit,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            audit,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False
        )
    )

    print("PE_PLUS_PDS_BUILD_PASS")


if __name__ == "__main__":
    main()
PY

###############################################################################
# 5. Detached 16-NPU runner
###############################################################################

cat > "$RUNNER" <<BASH2
#!/usr/bin/env bash
set -euo pipefail

source /workspace/mtpatcher/project_env.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXP="$EXP"

SCRIPT_DIR="$SCRIPT_DIR"
EXP_DATA="$EXP_DATA"
EXP_LOG="$EXP_LOG"

PE="$PE"
JOBS="$JOBS"

PDS_DIR="$PDS_DIR"
PDS_VALID="$PDS_VALID"
PDS_AUDIT="$PDS_AUDIT"

PE_PDS="$PE_PDS"
PE_PDS_AUDIT="$PE_PDS_AUDIT"

MODEL="$MODEL"

mkdir -p "\$PDS_DIR" "\$EXP_LOG"

echo "======================================================================"
echo "RQ3 PDS V11 — BUILD JOBS"
echo "======================================================================"

python "\$SCRIPT_DIR/build_pds_jobs_v11.py" \
    --pe "\$PE" \
    --output "\$JOBS" \
    --repeat 4

echo
echo "======================================================================"
echo "RQ3 PDS V11 — 16 NPU GENERATION"
echo "======================================================================"

PIDS=()

for DEVICE_ID in \$(seq 0 15); do
    DEVICE_LOG="\$EXP_LOG/rq3_pds_v11_device_\${DEVICE_ID}.log"
    DEVICE_OUT="\$PDS_DIR/device_\${DEVICE_ID}.jsonl"

    python "\$SCRIPT_DIR/generate_pds_qwen3_8b_v11.py" \
        --jobs "\$JOBS" \
        --output "\$DEVICE_OUT" \
        --model "\$MODEL" \
        --device-id "\$DEVICE_ID" \
        --world-size 16 \
        --batch-size 8 \
        --max-new-tokens 192 \
        --seed 20260825 \
        > "\$DEVICE_LOG" 2>&1 &

    PIDS+=("\$!")
done

FAIL=0

for PID in "\${PIDS[@]}"; do
    if ! wait "\$PID"; then
        FAIL=1
    fi
done

if [ "\$FAIL" -ne 0 ]; then
    echo "PDS_GENERATION_WORKER_FAILURE"
    false
fi

echo "PDS_ALL_16_WORKERS_COMPLETE"

echo
echo "======================================================================"
echo "RQ3 PDS V11 — MERGE + SCIENTIFIC AUDIT"
echo "======================================================================"

python "\$SCRIPT_DIR/merge_and_audit_pds_v11.py" \
    --jobs "\$JOBS" \
    --shard-dir "\$PDS_DIR" \
    --output "\$PDS_VALID" \
    --audit "\$PDS_AUDIT" \
    --world-size 16

echo
echo "======================================================================"
echo "RQ3 PDS V11 — BUILD PE + PDS SFT DATA"
echo "======================================================================"

python "\$SCRIPT_DIR/build_pe_plus_pds_v11.py" \
    --pe "\$PE" \
    --pds "\$PDS_VALID" \
    --output "\$PE_PDS" \
    --audit "\$PE_PDS_AUDIT" \
    --seed 20260825

echo
echo "======================================================================"
echo "RQ3 PE+PDS DATA READY"
echo "======================================================================"

cat "\$PDS_AUDIT"
echo
cat "\$PE_PDS_AUDIT"

echo
echo "PDS_GENERATION_ALL_PASS"
echo "PE_PLUS_PDS_DATA_ALL_PASS"
echo "RQ3_PDS_BASELINE_DATA_READY"
BASH2

chmod +x "$RUNNER"

###############################################################################
# 6. Static checks before launch
###############################################################################

python -m py_compile \
    "$SCRIPT_DIR/build_pds_jobs_v11.py" \
    "$SCRIPT_DIR/generate_pds_qwen3_8b_v11.py" \
    "$SCRIPT_DIR/merge_and_audit_pds_v11.py" \
    "$SCRIPT_DIR/build_pe_plus_pds_v11.py"

echo "RQ3_PDS_V11_PY_COMPILE_PASS"

if [ ! -f "$PE" ]; then
    echo "MISSING_PE=$PE"
    false
fi

if [ ! -d "$MODEL" ]; then
    echo "MISSING_MODEL=$MODEL"
    false
fi

echo "RQ3_PDS_V11_STATIC_PASS"

###############################################################################
# 7. Launch detached
###############################################################################

nohup setsid bash "$RUNNER" \
    > "$MASTER_LOG" 2>&1 < /dev/null &

PID="$!"

echo "RQ3_PDS_V11_STARTED"
echo "PID=$PID"
echo "LOG=$MASTER_LOG"

sleep 10

echo
echo "========== INITIAL PROCESS CHECK =========="
pgrep -af \
'rq3_pds_generation_v11|generate_pds_qwen3_8b_v11' \
|| true

echo
echo "========== INITIAL MASTER LOG =========="
tail -n 80 "$MASTER_LOG" || true

echo
echo "RQ3_PDS_V11_DETACHED_SAFE"
