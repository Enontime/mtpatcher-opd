#!/usr/bin/env python3
"""Clean Chemistry SFT controls matched to completed LR2e-5/LR3e-5 OPD.

Commands: preflight, materialize, train, eval, compare.
Historical C2/A1/OPD files are read-only; all writes use new artifact roots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

TZ8 = timezone(timedelta(hours=8))
ROOT = Path("/workspace/mtpatcher")
DATA = ROOT / "data/mtpatcher_v3_full6565_20260823"
CONTEXT = DATA / "targeted_section43_contexts_qwen3_8b_final_20260910/chemistry_train.jsonl"
OLD_TARGETS = DATA / "targeted_sft_positive_control_targets_qwen3_8b_v1_20260910/chemistry_train5500_sft.jsonl"
CLEAN_DIR = DATA / "targeted_clean_sft_targets_qwen3_8b_v1_20261015"
CLEAN_TARGETS = CLEAN_DIR / "chemistry_train5500_clean_sft.jsonl"
DIAG = DATA / "targeted_diagnostic1000_v1_20260910/targeted_diagnostic2000.jsonl"
C0 = ROOT / "models/Qwen3-0.6B"
TEACHER = ROOT / "models/Qwen3-8B"
RUN_ROOT = ROOT / "runs/targeted/chemistry_clean_sft_parity_20261015"
PARITY_REPORT = Path("/workspace/chemistry_sft_parity_report.json")
OPD_RUNS = {
    "2e-5": ROOT / "runs/targeted/wa_opd_lr2e5_chemistry_20261014",
    "3e-5": ROOT / "runs/targeted/wa_opd_lr3e5_chemistry_20261014",
}
OPD_MANIFESTS = {k: p / "O2/train/manifest.json" for k, p in OPD_RUNS.items()}
OPD_RUNNERS = {
    "2e-5": OPD_RUNS["2e-5"] / "runner/targeted_wa_opd_lr2e5.py",
    "3e-5": OPD_RUNS["3e-5"] / "runner/targeted_wa_opd_lr3e5.py",
}
GENERAL_EVAL = {
    "WMT24": ROOT / "data/pilot_v2_qwen3_06b/wmt24_zh_en998.jsonl",
    "FLORES": ROOT / "data/pilot_v2_qwen3_06b/flores_zh_en1012.jsonl",
    "Challenge": ROOT / "data/pilot_v2_qwen3_06b/challenge_zh_en197.jsonl",
}
GENERAL_ROWS = {"WMT24": 998, "FLORES": 1012, "Challenge": 197}
BASE_GENERAL_MACRO_BLEU = 17.348521468

EXPECTED_CONTEXT_SHA = "ca6532cacce64f24f14226a4dca509b494aaa5f5bbeecbc0f1d2f157c317044f"
EXPECTED_OLD_TARGET_SHA = "9703cee06bf97ee15afa4c79d0fe935b4acf59e9b60a9d33abc31f58d348f914"
EXPECTED_DIAG_SHA = "7c5082217de8071bd0b519e83cac255896f704d0b41d9bd49ae2cd5b41af6db2"
EXPECTED_C0_CONFIG_SHA = "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"
EXPECTED_C0_TOKENIZER_SHA = "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"
EXPECTED_DIRECT_PROMPT_SHA = "63101d0739ff2ee11b0e5d548b85dcd18be7a241777893326140f12778d9a8b8"
DIRECT_PROMPT = (
    "Translate the following text into English without additional explanations:"
    "\n\n{source}\n\n"
)
DIRECT_PROMPT_SHA = hashlib.sha256(DIRECT_PROMPT.encode()).hexdigest()

TEACHER_CLEAN_TEMPLATES = [
    """You are translating a Chinese sentence into English.

Teacher-only lexical knowledge:
Chinese chemistry term: {src_term}
Canonical English registry term: {en_name}

Use the exact canonical English registry term in a faithful, natural translation
of the complete Chinese source. Output only the translation: no labels, preface,
quotation marks, notes, alternatives, or explanation.

Chinese source:
{source}""",
    """Translate the complete Chinese source into one clean English translation.

Required terminology (Teacher-only): {src_term} -> {en_name}

Preserve the source meaning and use the required English term. Do not mention
these instructions, terminology, a source term, or a canonical term. Do not add
a heading or commentary. Return only the translation.

Chinese source:
{source}""",
    """Produce only a faithful English translation of the Chinese sentence below.
Use {en_name} for {src_term}. Never echo the prompt, describe the terminology,
or explain the answer. Do not wrap the answer in quotes.

Chinese source:
{source}""",
]
TEACHER_PROMPT_FAMILY = "opd_chem_knowledge_clean_translation_v1"
TEACHER_PROMPT_SHAS = [
    hashlib.sha256(x.encode()).hexdigest() for x in TEACHER_CLEAN_TEMPLATES
]
EPOCHS, MICRO_BATCH, GRAD_ACCUM, SEED = 3, 4, 4, 20260820
WEIGHT_DECAY, WARMUP_RATIO, MAX_GRAD_NORM = 0.01, 0.03, 1.0
MAX_TOTAL_LENGTH = 1024
ARM_CONFIG = {
    "SFT_OLDTARGET_MATCHED_LR2E5": {
        "lr": 2e-5, "targets": OLD_TARGETS, "opd": "2e-5", "clean": False
    },
    "SFT_CLEAN_LR2E5": {
        "lr": 2e-5, "targets": CLEAN_TARGETS, "opd": "2e-5", "clean": True
    },
    "SFT_CLEAN_LR3E5": {
        "lr": 3e-5, "targets": CLEAN_TARGETS, "opd": "3e-5", "clean": True
    },
}

# Frozen v1 contamination rules. A rule change requires a new data version.
META_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("designated_chemistry_term", re.compile(r"designated\s+chemistry\s+term", re.I)),
    ("canonical_english_term", re.compile(r"canonical\s+english(?:\s+registry)?\s+term", re.I)),
    ("chinese_term_label", re.compile(r"\bchinese\s+(?:chemistry\s+)?term\s*:", re.I)),
    ("source_term_label", re.compile(r"\bsource\s+term\s*:", re.I)),
    ("target_term_label", re.compile(r"\btarget\s+term\s*:", re.I)),
    ("translation_label", re.compile(r"(?:^|\n)\s*(?:natural\s+english\s+)?translation\s*:", re.I)),
    ("role_label", re.compile(r"(?:^|\n)\s*(?:system|assistant|user)\s*:", re.I)),
    ("instruction_echo", re.compile(r"when\s+translating|must\s+use\s+the\s+exact|term\s+shown\s+above|preserving\s+its\s+spelling", re.I)),
    ("return_only_echo", re.compile(r"return\s+only\s+the\s+english|output\s+only\s+the\s+translation", re.I)),
    ("prompt_echo", re.compile(r"translate\s+the\s+following\s+text|without\s+additional\s+explanations", re.I)),
    ("markdown_fence", re.compile(r"\x60{3}")),
)
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
CI_RE = re.compile(r"c\s*\.\s*i\s*\.", re.I)


def now() -> str:
    return datetime.now(TZ8).isoformat(timespec="seconds")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    raw = json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(f"{path}:{line_no}: {exc}") from exc
    return rows


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def translation_of(row: dict[str, Any]) -> str:
    for key in (
        "model_translation", "translation", "student_translation",
        "hypothesis", "prediction",
    ):
        value = row.get(key)
        if isinstance(value, str):
            return value.strip()
    return ""


def strict_hit(canonical: str, text: str) -> bool:
    return canonical.casefold() in text.casefold()


def ci_spacing_hit(canonical: str, text: str) -> bool:
    if not CI_RE.search(canonical):
        return False
    def normalize(value: str) -> str:
        value = unicodedata.normalize("NFKC", value)
        return re.sub(r"\s+", " ", CI_RE.sub("CI", value)).casefold()
    return normalize(canonical) in normalize(text)


def is_ci_row(row: dict[str, Any]) -> bool:
    return bool(
        CI_RE.search(str(row.get("src_term", "")))
        or CI_RE.search(str(row.get("canonical_en_target", "")))
    )


def repeated_sentence(text: str) -> bool:
    pieces = [
        re.sub(r"\s+", " ", x).strip().casefold()
        for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()
    ]
    return len(pieces) != len(set(pieces))


def audit_translation(
    text: str, canonical: str, alias: str = ""
) -> dict[str, Any]:
    stripped = text.strip()
    flags = [name for name, pattern in META_RULES if pattern.search(stripped)]
    if not stripped:
        flags.append("empty")
    if CONTROL_RE.search(stripped):
        flags.append("control_character")
    if CJK_RE.search(stripped):
        flags.append("cjk_residual")
    if "\n\n" in stripped:
        flags.append("multiple_paragraphs")
    if (
        (stripped.startswith('"') and stripped.endswith('"'))
        or (stripped.startswith("'") and stripped.endswith("'"))
    ):
        flags.append("wrapped_quote")
    if repeated_sentence(stripped):
        flags.append("repeated_sentence")
    strict = strict_hit(canonical, stripped) if canonical else False
    ci_hit = ci_spacing_hit(canonical, stripped) if canonical else False
    alias_hit = bool(alias and strict_hit(alias, stripped))
    if not (strict or ci_hit):
        flags.append("canonical_term_missing")
    flags = sorted(set(flags))
    return {
        "flags": flags,
        "format_clean": not flags,
        "strict_canonical_hit": strict,
        "ci_spacing_normalized_hit": ci_hit,
        "alias_hit": alias_hit,
        "contains_cjk": bool(CJK_RE.search(stripped)),
        "character_count": len(stripped),
        "paragraph_count": len(
            [x for x in re.split(r"\n\s*\n", stripped) if x.strip()]
        ),
    }


def source_identity(
    contexts: list[dict[str, Any]], targets: list[dict[str, Any]]
) -> dict[str, Any]:
    issues = []
    if len(contexts) != 5500 or len(targets) != 5500:
        issues.append(f"row_count context={len(contexts)} targets={len(targets)}")
    if len({int(x["job_id"]) for x in contexts}) != len(contexts):
        issues.append("duplicate_context_job_id")
    if len({int(x["job_id"]) for x in targets}) != len(targets):
        issues.append("duplicate_target_job_id")
    for index, (src, target) in enumerate(zip(contexts, targets)):
        canonical = src.get("lexical_record", {}).get("en_name")
        expected_prompt = DIRECT_PROMPT.format(source=src.get("src_text"))
        messages = target.get("messages")
        checks = (
            int(src.get("job_id", -1)) == int(target.get("job_id", -2)),
            src.get("entity_key") == target.get("entity_key"),
            src.get("src_text") == target.get("source"),
            src.get("src_term") == target.get("src_term"),
            canonical == target.get("canonical_en_target"),
            isinstance(messages, list)
            and len(messages) == 1
            and messages[0] == {"role": "user", "content": expected_prompt},
            target.get("student_prompt_sha256") == EXPECTED_DIRECT_PROMPT_SHA,
        )
        if not all(checks):
            issues.append(
                f"row_identity index={index} job_id={src.get('job_id')}"
            )
            if len(issues) >= 25:
                break
    return {
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "rows": len(contexts),
    }


def epoch_order(rows: int, epoch: int) -> list[int]:
    order = list(range(rows))
    random.Random(SEED + epoch - 1).shuffle(order)
    return order


def epoch_order_sha(rows: int, epoch: int) -> str:
    return sha256_json(epoch_order(rows, epoch))


def frozen_input_checks() -> dict[str, Any]:
    expected = {
        CONTEXT: EXPECTED_CONTEXT_SHA,
        OLD_TARGETS: EXPECTED_OLD_TARGET_SHA,
        DIAG: EXPECTED_DIAG_SHA,
        C0 / "config.json": EXPECTED_C0_CONFIG_SHA,
        C0 / "tokenizer_config.json": EXPECTED_C0_TOKENIZER_SHA,
    }
    checks = {}
    for path, wanted in expected.items():
        got = sha256_file(path) if path.is_file() else None
        checks[str(path)] = {
            "expected": wanted, "actual": got,
            "status": "PASS" if got == wanted else "FAIL",
        }
    checks["direct_prompt"] = {
        "expected": EXPECTED_DIRECT_PROMPT_SHA,
        "actual": DIRECT_PROMPT_SHA,
        "status": (
            "PASS" if DIRECT_PROMPT_SHA == EXPECTED_DIRECT_PROMPT_SHA
            else "FAIL"
        ),
    }
    for name, path in GENERAL_EVAL.items():
        count = len(read_jsonl(path)) if path.is_file() else None
        checks[str(path)] = {
            "expected_rows": GENERAL_ROWS[name], "actual_rows": count,
            "status": "PASS" if count == GENERAL_ROWS[name] else "FAIL",
        }
    return checks


def inspect_opd_contract(label: str) -> dict[str, Any]:
    manifest_path = OPD_MANIFESTS[label]
    runner_path = OPD_RUNNERS[label]
    if not manifest_path.is_file() or not runner_path.is_file():
        return {"status": "FAIL", "reason": "missing manifest or frozen runner"}
    manifest = read_json(manifest_path)
    training = manifest.get("training", {})
    fields = {
        "status": manifest.get("status") == "PASS",
        "init_model": manifest.get("init_model") == str(C0),
        "teacher_model": manifest.get("teacher_model") == str(TEACHER),
        "rows": manifest.get("rows") == 5500,
        "context_sha": (
            manifest.get("frozen_inputs", {}).get("context_hashes", {})
            .get("chemistry_train.jsonl") == EXPECTED_CONTEXT_SHA
        ),
        "c0_config_sha": (
            manifest.get("frozen_inputs", {}).get("c0_config_sha256")
            == EXPECTED_C0_CONFIG_SHA
        ),
        "c0_tokenizer_sha": (
            manifest.get("frozen_inputs", {}).get(
                "c0_tokenizer_config_sha256"
            ) == EXPECTED_C0_TOKENIZER_SHA
        ),
        "prompt_sha": (
            manifest.get("frozen_inputs", {}).get("direct_prompt_sha256")
            == EXPECTED_DIRECT_PROMPT_SHA
        ),
        "epochs": training.get("epochs") == EPOCHS,
        "micro_batch": training.get("micro_batch") == MICRO_BATCH,
        "grad_accum": training.get("gradient_accumulation") == GRAD_ACCUM,
        "updates_per_epoch": training.get("updates_per_epoch") == 344,
        "total_updates": training.get("total_updates") == 1032,
        "optimizer": training.get("optimizer") == "AdamW(foreach=False)",
        "lr": training.get("lr") == float(label),
        "weight_decay": training.get("weight_decay") == WEIGHT_DECAY,
        "warmup_ratio": training.get("warmup_ratio") == WARMUP_RATIO,
        "scheduler": training.get("scheduler") == "linear",
        "max_grad_norm": training.get("max_grad_norm") == MAX_GRAD_NORM,
        "precision": training.get("precision") == "bfloat16",
        "seed": training.get("seed") == SEED,
        "loss_reduction": (
            training.get("loss_reduction")
            == "global valid response-token mean across accumulated microbatches"
        ),
    }
    runner_sha = sha256_file(runner_path)
    source_sha = manifest.get(
        "e2e_runtime_preflight_v3", {}
    ).get("script_sha256")
    source = runner_path.read_text(encoding="utf-8")
    divide_at = source.find("p.grad.div_(group_tokens)")
    clip_at = source.find("clip_grad_norm_", divide_at)
    source_evidence = {
        "runner_sha_matches_manifest": runner_sha == source_sha,
        "python_shuffle_exact": (
            "random.Random(SEED + epoch - 1).shuffle(order)" in source
        ),
        "gradient_token_division": divide_at >= 0,
        "clip_after_division": divide_at >= 0 and clip_at > divide_at,
        "warmup_int_rule": "int(total_updates * WARMUP_RATIO)" in source,
        "max_total_length_1024": "MAX_TOTAL_LENGTH = 1024" in source,
    }
    ok = all(fields.values()) and all(source_evidence.values())
    return {
        "status": "PASS" if ok else "FAIL",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "runner": str(runner_path),
        "runner_sha256": runner_sha,
        "fields": fields,
        "source_evidence": source_evidence,
        "epoch_order_sha256": {
            str(e): epoch_order_sha(5500, e) for e in range(1, 4)
        },
        "warmup_steps_derived": int(1032 * WARMUP_RATIO),
    }


def cmd_preflight(args: argparse.Namespace) -> None:
    contexts, old = read_jsonl(CONTEXT), read_jsonl(OLD_TARGETS)
    frozen = frozen_input_checks()
    counts: Counter[str] = Counter()
    flagged = 0
    for src, row in zip(contexts, old):
        audit = audit_translation(
            row.get("target_translation", ""),
            src["lexical_record"]["en_name"],
            src["lexical_record"].get("en_alias", ""),
        )
        counts.update(audit["flags"])
        flagged += bool(audit["flags"])
    parity_file = read_json(PARITY_REPORT) if PARITY_REPORT.is_file() else None
    parity_ok = bool(
        parity_file and parity_file.get("ready_for_implementation")
        and not parity_file.get("issues")
    )
    report = {
        "status": "PASS",
        "created": now(),
        "read_only": True,
        "frozen_inputs": frozen,
        "source_identity": source_identity(contexts, old),
        "provided_parity_report": {
            "path": str(PARITY_REPORT),
            "status": "PASS" if parity_ok else "NOT_VERIFIED",
            "sha256": (
                sha256_file(PARITY_REPORT) if PARITY_REPORT.is_file() else None
            ),
        },
        "opd_contracts": {
            label: inspect_opd_contract(label) for label in OPD_MANIFESTS
        },
        "epoch_order_sha256": {
            str(e): epoch_order_sha(5500, e) for e in range(1, 4)
        },
        "old_target_audit": {
            "flagged_rows": flagged, "flag_counts": dict(counts),
            "status": "HISTORICAL_CONTAMINATION_CONFIRMED",
        },
        "clean_target": {"status": "NOT_MATERIALIZED"},
        "claim_boundary": (
            "Base optimization/input contracts only; offline CE and on-policy "
            "forward-KL remain intentionally different."
        ),
    }
    clean_manifest = CLEAN_DIR / "manifest.json"
    if clean_manifest.is_file() and CLEAN_TARGETS.is_file():
        cm = read_json(clean_manifest)
        report["clean_target"] = {
            "status": cm.get("status"), "path": str(CLEAN_TARGETS),
            "sha256": sha256_file(CLEAN_TARGETS),
            "source_identity": source_identity(
                contexts, read_jsonl(CLEAN_TARGETS)
            ),
        }
    if (
        any(x.get("status") == "FAIL" for x in frozen.values())
        or report["source_identity"]["status"] != "PASS"
        or any(x["status"] != "PASS" for x in report["opd_contracts"].values())
        or not parity_ok
    ):
        report["status"] = "BLOCKED"
    if args.json_out:
        atomic_json(Path(args.json_out), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        "SOURCE_IDENTITY_PASS"
        if report["source_identity"]["status"] == "PASS"
        else "SOURCE_IDENTITY_FAIL"
    )
    print(
        "CONFIG_PARITY_PASS"
        if report["status"] == "PASS" else "CONFIG_PARITY_BLOCKED"
    )


def render_prompt_ids(tokenizer: Any, user_text: str) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return tokenizer(rendered, add_special_tokens=False)["input_ids"]


def generate_clean_target(
    model: Any, tokenizer: Any, src: dict[str, Any], device: str,
    variant: int, max_new_tokens: int,
) -> str:
    import torch
    prompt = TEACHER_CLEAN_TEMPLATES[variant].format(
        src_term=src["src_term"], en_name=src["lexical_record"]["en_name"],
        source=src["src_text"],
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    batch = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    batch = {k: v.to(device) for k, v in batch.items()}
    width = batch["input_ids"].shape[1]
    with torch.no_grad():
        output = model.generate(
            **batch, do_sample=False, max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id, use_cache=True,
        )
    return tokenizer.decode(
        output[0, width:].detach().cpu().tolist(),
        skip_special_tokens=True,
    ).strip()


def validate_candidate_rows(
    contexts: list[dict[str, Any]], rows: list[dict[str, Any]],
    tokenizer: Any | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audits, failures = [], []
    for src, row in zip(contexts, rows):
        text = row.get("target_translation", "")
        audit = audit_translation(
            text, src["lexical_record"]["en_name"],
            src["lexical_record"].get("en_alias", ""),
        )
        if tokenizer is not None:
            prompt_ids = render_prompt_ids(
                tokenizer, DIRECT_PROMPT.format(source=src["src_text"])
            )
            target_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            total = len(prompt_ids) + len(target_ids) + 1
            audit["total_training_tokens"] = total
            if total > MAX_TOTAL_LENGTH:
                audit["flags"] = sorted(
                    set(audit["flags"] + ["max_total_length_exceeded"])
                )
                audit["format_clean"] = False
        record = {
            "target_id": row.get("target_id"), "job_id": row.get("job_id"),
            **audit,
        }
        audits.append(record)
        if audit["flags"]:
            failures.append(record)
    return audits, failures


def deterministic_review_sample(
    rows: list[dict[str, Any]], size: int
) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            ("clean-sft-v1-review|" + str(row["target_id"])).encode()
        ).hexdigest(),
    )
    return ranked[:min(size, len(ranked))]


def finalize_clean_targets(args: argparse.Namespace) -> None:
    candidate = CLEAN_DIR / "candidate/chemistry_train5500_clean_sft.jsonl"
    require(candidate.is_file(), f"candidate missing: {candidate}")
    require(
        args.manual_review_jsonl,
        "--manual-review-jsonl is required with --finalize",
    )
    review_path = Path(args.manual_review_jsonl)
    require(review_path.is_file(), f"manual review missing: {review_path}")
    template = read_jsonl(CLEAN_DIR / "manual_review_template.jsonl")
    reviewed = {str(x["target_id"]): x for x in read_jsonl(review_path)}
    missing = [
        x["target_id"] for x in template if str(x["target_id"]) not in reviewed
    ]
    failed = [
        x["target_id"] for x in template
        if str(x["target_id"]) in reviewed
        and str(reviewed[str(x["target_id"])].get("review_status", "")).upper()
        != "PASS"
    ]
    require(
        not missing and not failed,
        f"MANUAL_REVIEW_BLOCKED missing={missing[:10]} nonpass={failed[:10]}",
    )
    contexts, rows = read_jsonl(CONTEXT), read_jsonl(candidate)
    identity = source_identity(contexts, rows)
    require(
        identity["status"] == "PASS",
        f"SOURCE_IDENTITY_FAIL {identity['issues']}",
    )
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(C0)
    audits, failures = validate_candidate_rows(contexts, rows, tokenizer)
    require(
        not failures,
        f"TARGET_QUALITY_FAIL n={len(failures)} first={failures[:3]}",
    )
    if CLEAN_TARGETS.exists():
        existing_manifest = CLEAN_DIR / "manifest.json"
        require(
            existing_manifest.is_file()
            and read_json(existing_manifest).get("status") != "FROZEN",
            "refusing to overwrite frozen clean targets",
        )
    atomic_jsonl(CLEAN_TARGETS, rows)
    attempts_path = CLEAN_DIR / "candidate/regeneration_attempts.jsonl"
    attempts = read_jsonl(attempts_path) if attempts_path.is_file() else []
    repaired_ids = sorted({
        str(x["target_id"]) for x in attempts if x.get("status") == "PASS"
    })
    successful_attempts = [
        x for x in attempts if x.get("status") == "PASS"
    ]
    retry_distribution = dict(Counter(
        str(x["variant"]) for x in successful_attempts
    ))
    materialization_cost = {
        "attempts_jsonl": str(attempts_path),
        "attempts_sha256": sha256_file(attempts_path),
        "teacher_generate_calls": len(attempts),
        "failed_attempts": sum(
            x.get("status") != "PASS" for x in attempts
        ),
        "successful_targets": len(successful_attempts),
        "successful_retry_variant_counts": retry_distribution,
    }
    manifest = {
        "status": "FROZEN",
        "target_quality_status": "PASS",
        "artifact_version": "targeted-clean-sft-qwen3-8b-v1-20261015",
        "created": now(),
        "inputs": {
            "context": {"path": str(CONTEXT), "sha256": sha256_file(CONTEXT)},
            "old_targets": {
                "path": str(OLD_TARGETS), "sha256": sha256_file(OLD_TARGETS)
            },
        },
        "output": {
            "path": str(CLEAN_TARGETS), "rows": len(rows),
            "sha256": sha256_file(CLEAN_TARGETS),
        },
        "source_identity": identity,
        "repaired_target_ids": repaired_ids,
        "repaired_count": len(repaired_ids),
        "unchanged_count": len(rows) - len(repaired_ids),
        "target_materialization": materialization_cost,
        "teacher": {
            "model": str(TEACHER), "prompt_family": TEACHER_PROMPT_FAMILY,
            "prompt_sha256_by_retry": TEACHER_PROMPT_SHAS,
            "config_sha256": sha256_file(TEACHER / "config.json"),
            "tokenizer_config_sha256": sha256_file(TEACHER / "tokenizer_config.json"),
            "enable_thinking": False, "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "mixed_source_dataset": True,
            "student_user_prompt_sha256": EXPECTED_DIRECT_PROMPT_SHA,
        },
        "validation": {
            "rows": len(audits), "failures": 0,
            "rules_sha256": sha256_json([
                (name, pattern.pattern) for name, pattern in META_RULES
            ]),
            "canonical_is_not_sole_quality_check": True,
        },
        "manual_review": {
            "file": str(review_path), "sha256": sha256_file(review_path),
            "rows": len(template), "all_pass": True,
        },
        "claim_boundary": (
            "Mixed-source clean-SFT: unflagged historical targets retained; "
            "only independently flagged targets regenerated. Fixed-sample "
            "human review is not proof of semantic perfection."
        ),
    }
    atomic_json(CLEAN_DIR / "manifest.json", manifest)
    print(f"TARGET_QUALITY_PASS rows={len(rows)} repaired={len(repaired_ids)}")
    print(f"CLEAN_TARGETS_FROZEN={CLEAN_TARGETS}")


def cmd_materialize(args: argparse.Namespace) -> None:
    if args.finalize:
        finalize_clean_targets(args)
        return
    frozen_manifest = CLEAN_DIR / "manifest.json"
    require(
        not frozen_manifest.exists()
        or read_json(frozen_manifest).get("status") != "FROZEN",
        "clean target artifact is already FROZEN",
    )
    contexts, old = read_jsonl(CONTEXT), read_jsonl(OLD_TARGETS)
    require(
        source_identity(contexts, old)["status"] == "PASS",
        "historical source-target identity failed",
    )
    initial_audit, flagged_ids = [], set()
    for src, row in zip(contexts, old):
        audit = audit_translation(
            row["target_translation"], src["lexical_record"]["en_name"],
            src["lexical_record"].get("en_alias", ""),
        )
        record = {
            "target_id": row["target_id"], "job_id": row["job_id"],
            "flags": audit["flags"],
            "old_target_sha256": hashlib.sha256(
                row["target_translation"].encode()
            ).hexdigest(),
        }
        initial_audit.append(record)
        if audit["flags"]:
            flagged_ids.add(str(row["target_id"]))
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(CLEAN_DIR / "initial_target_audit.jsonl", initial_audit)
    atomic_json(
        CLEAN_DIR / "candidate_manifest.json",
        {
            "status": "AUDITED", "flagged_count": len(flagged_ids),
            "flag_counts": dict(Counter(
                flag for row in initial_audit for flag in row["flags"]
            )),
            "rules_sha256": sha256_json([
                (name, pattern.pattern) for name, pattern in META_RULES
            ]),
            "created": now(),
        },
    )
    if args.audit_only:
        print(
            f"TARGET_AUDIT_COMPLETE flagged={len(flagged_ids)} total={len(old)}"
        )
        return

    import torch
    import torch_npu  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = args.device
    torch.npu.set_device(device)
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER)
    student_tokenizer = AutoTokenizer.from_pretrained(C0)
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device).eval()
    attempts_path = CLEAN_DIR / "candidate/regeneration_attempts.jsonl"
    completed = {}
    if attempts_path.is_file():
        for record in read_jsonl(attempts_path):
            if record.get("status") == "PASS":
                completed[str(record["target_id"])] = record
    started = time.time()
    for src, old_row in zip(contexts, old):
        target_id = str(old_row["target_id"])
        if target_id not in flagged_ids or target_id in completed:
            continue
        success = None
        for variant in range(min(args.max_retries, len(TEACHER_CLEAN_TEMPLATES))):
            text = generate_clean_target(
                model, teacher_tokenizer, src, device, variant,
                args.max_new_tokens,
            )
            audit = audit_translation(
                text, src["lexical_record"]["en_name"],
                src["lexical_record"].get("en_alias", ""),
            )
            total_tokens = (
                len(render_prompt_ids(
                    student_tokenizer,
                    DIRECT_PROMPT.format(source=src["src_text"]),
                ))
                + len(student_tokenizer(
                    text, add_special_tokens=False
                )["input_ids"]) + 1
            )
            if total_tokens > MAX_TOTAL_LENGTH:
                audit["flags"] = sorted(set(
                    audit["flags"] + ["max_total_length_exceeded"]
                ))
            record = {
                "status": "PASS" if not audit["flags"] else "FAIL",
                "target_id": target_id, "job_id": src["job_id"],
                "variant": variant,
                "teacher_prompt_sha256": TEACHER_PROMPT_SHAS[variant],
                "target_translation": text,
                "target_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "validation": audit, "total_training_tokens": total_tokens,
                "timestamp": now(),
            }
            append_jsonl(attempts_path, record)
            if record["status"] == "PASS":
                success, completed[target_id] = record, record
                break
        if success is None:
            raise RuntimeError(
                f"TARGET_REGENERATION_FAIL target_id={target_id}; "
                f"see {attempts_path}"
            )
        if len(completed) % 25 == 0:
            rate = len(completed) / max(time.time() - started, 1e-9)
            print(
                f"{now()} materialize regenerated={len(completed)}/"
                f"{len(flagged_ids)} rate={rate:.3f}/s", flush=True,
            )
    missing = sorted(flagged_ids - set(completed))
    require(not missing, f"missing regenerated targets: {missing[:20]}")
    candidate_rows = []
    for row in old:
        copied, target_id = dict(row), str(row["target_id"])
        old_text = row["target_translation"]
        if target_id in flagged_ids:
            record = completed[target_id]
            new_text = record["target_translation"]
            copied["target_translation"] = copied["reference"] = new_text
            copied["clean_sft_provenance"] = {
                "source": "regenerated_qwen3_8b", "repaired": True,
                "old_target_sha256": hashlib.sha256(old_text.encode()).hexdigest(),
                "new_target_sha256": hashlib.sha256(new_text.encode()).hexdigest(),
                "teacher_prompt_family": TEACHER_PROMPT_FAMILY,
                "teacher_prompt_variant": record["variant"],
                "teacher_prompt_sha256": record["teacher_prompt_sha256"],
            }
        else:
            target_sha = hashlib.sha256(old_text.encode()).hexdigest()
            copied["clean_sft_provenance"] = {
                "source": "historical_target_retained", "repaired": False,
                "old_target_sha256": target_sha,
                "new_target_sha256": target_sha,
                "teacher_prompt_family": row.get("teacher_prompt_family"),
                "teacher_prompt_variant": row.get("teacher_prompt_variant"),
            }
        candidate_rows.append(copied)
    candidate_path = CLEAN_DIR / "candidate/chemistry_train5500_clean_sft.jsonl"
    atomic_jsonl(candidate_path, candidate_rows)
    audits, failures = validate_candidate_rows(
        contexts, candidate_rows, student_tokenizer
    )
    atomic_jsonl(CLEAN_DIR / "candidate/full_validation.jsonl", audits)
    require(
        not failures,
        f"TARGET_QUALITY_FAIL n={len(failures)} first={failures[:3]}",
    )
    repaired_rows = [
        x for x in candidate_rows if x["clean_sft_provenance"]["repaired"]
    ]
    review = [{
        "target_id": row["target_id"], "job_id": row["job_id"],
        "source": row["source"], "src_term": row["src_term"],
        "canonical_en_target": row["canonical_en_target"],
        "candidate_translation": row["target_translation"],
        "review_status": "PENDING",
        "faithful_complete_translation": None,
        "terminology_correct": None, "clean_output_only": None,
        "reviewer_notes": "",
    } for row in deterministic_review_sample(
        repaired_rows, args.review_sample_size
    )]
    atomic_jsonl(CLEAN_DIR / "manual_review_template.jsonl", review)
    atomic_json(
        CLEAN_DIR / "candidate_manifest.json",
        {
            "status": "CANDIDATE_AWAITING_MANUAL_REVIEW",
            "rows": len(candidate_rows), "flagged_count": len(flagged_ids),
            "candidate": str(candidate_path),
            "candidate_sha256": sha256_file(candidate_path),
            "automatic_validation_failures": 0,
            "manual_review_rows": len(review), "created": now(),
        },
    )
    print(
        f"CLEAN_SFT_CANDIDATE_PASS rows={len(candidate_rows)} "
        f"repaired={len(flagged_ids)}"
    )
    print(
        f"NEXT=review {CLEAN_DIR / 'manual_review_template.jsonl'} then "
        "rerun materialize --finalize --manual-review-jsonl <reviewed-file>"
    )
