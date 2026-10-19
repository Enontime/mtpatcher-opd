#!/usr/bin/env python3
"""Raw-distribution and selective-clean materialization contracts for KC-SFT."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import _clean_sft_parity_core_v1 as c
import _clean_sft_parity_kc_v1 as kc


GENERATION_CONTRACT_VERSION = "chemistry_kc_sft_generation_contract_v2"
TECHNICAL_POLICY = {
    "version": "raw_teacher_output_technical_policy_v1",
    "empty_output": "record_once_and_block_trainable_freeze",
    "max_total_length_exceeded": "record_once_and_block_trainable_freeze",
    "generation_exception": "record_once_and_block_trainable_freeze",
    "source_or_row_identity_error": "hard_block_before_teacher_inference",
    "silent_skip_allowed": False,
    "automatic_technical_retry_allowed": False,
}


def prompt_templates(dataset: str, max_retries: int) -> list[str]:
    if dataset == "matched_prompt":
        return [kc.OPD_TEACHER_CHEM_TEMPLATE]
    c.require(dataset == "selective_clean", f"unknown dataset {dataset}")
    return c.TEACHER_CLEAN_TEMPLATES[:max_retries]


def build_generation_contract(
    dataset: str,
    max_new_tokens: int,
    max_retries: int,
    teacher_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = kc.DATASETS[dataset]
    templates = prompt_templates(dataset, max_retries)
    if teacher_identity is None:
        teacher_identity = kc.model_identity(c.TEACHER)
    return {
        "contract_version": GENERATION_CONTRACT_VERSION,
        "dataset": dataset,
        "artifact_version": spec["artifact_version"],
        "inputs": {
            "context": {
                "path": str(c.CONTEXT), "sha256": c.sha256_file(c.CONTEXT),
            },
            "old_targets": {
                "path": str(c.OLD_TARGETS),
                "sha256": c.sha256_file(c.OLD_TARGETS),
            },
        },
        "teacher": {
            "path": str(c.TEACHER),
            "identity": teacher_identity,
            "config_sha256": c.sha256_file(c.TEACHER / "config.json"),
            "tokenizer_config_sha256": c.sha256_file(
                c.TEACHER / "tokenizer_config.json"
            ),
        },
        "prompts": {
            "sha256_by_variant": [
                hashlib.sha256(template.encode()).hexdigest()
                for template in templates
            ],
            "matched_prompt_is_byte_identical_to_opd": (
                dataset == "matched_prompt"
            ),
        },
        "decoding": {
            "do_sample": False,
            "enable_thinking": False,
            "max_new_tokens": max_new_tokens,
            "semantic_attempt_limit": len(templates),
            "content_retry_policy": (
                "none_raw_distribution"
                if dataset == "matched_prompt"
                else "fixed_variant_sequence_until_strict_content_pass"
            ),
        },
        "technical_policy": TECHNICAL_POLICY,
        "student_evaluation_used_for_selection": False,
    }


def classify_output(
    dataset: str,
    text: str,
    content_audit: dict[str, Any],
    total_training_tokens: int | None,
    generation_exception: str | None = None,
) -> dict[str, Any]:
    technical_flags: list[str] = []
    if generation_exception is not None:
        technical_flags.append("generation_exception")
    if not text.strip():
        technical_flags.append("empty_output")
    if (
        total_training_tokens is not None
        and total_training_tokens > c.MAX_TOTAL_LENGTH
    ):
        technical_flags.append("max_total_length_exceeded")
    content_flags = [
        flag for flag in content_audit.get("flags", [])
        if flag != "empty"
    ]
    technical_flags = sorted(set(technical_flags))
    content_flags = sorted(set(content_flags))
    technical_valid = not technical_flags
    content_quality_pass = not content_flags
    if not technical_valid:
        status = "TECHNICAL_INVALID"
    elif dataset == "matched_prompt":
        status = "RAW_OUTPUT_RECORDED"
    elif content_quality_pass:
        status = "PASS"
    else:
        status = "CONTENT_REJECTED"
    return {
        "status": status,
        "technical_valid": technical_valid,
        "technical_flags": technical_flags,
        "content_quality_pass": content_quality_pass,
        "content_quality_flags": content_flags,
    }


def summarize_admission(
    dataset: str,
    records: list[dict[str, Any]],
    expected_ids: set[str],
    identity_status: str,
) -> dict[str, Any]:
    by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    for record in records:
        target_id = str(record["target_id"])
        if target_id in by_id:
            duplicate_ids.append(target_id)
        by_id[target_id] = record
    missing = sorted(expected_ids - set(by_id))
    extra = sorted(set(by_id) - expected_ids)
    technical_counts = Counter(
        flag for row in by_id.values() for flag in row.get("technical_flags", [])
    )
    content_counts = Counter(
        flag
        for row in by_id.values()
        for flag in row.get("content_quality_flags", [])
    )
    trainable_ids = sorted(
        target_id for target_id, row in by_id.items()
        if row.get("technical_valid") is True
    )
    technical_pass = (
        identity_status == "PASS" and not duplicate_ids
        and not missing and not extra and len(trainable_ids) == len(expected_ids)
    )
    content_pass = all(
        row.get("content_quality_pass") is True for row in by_id.values()
    )
    admitted = technical_pass and (
        dataset == "matched_prompt" or content_pass
    )
    return {
        "dataset": dataset,
        "admitted": admitted,
        "technical_validity_status": "PASS" if technical_pass else "FAIL",
        "content_quality_status": (
            "OBSERVED_NOT_GATED"
            if dataset == "matched_prompt"
            else "PASS" if content_pass else "FAIL"
        ),
        "identity_status": identity_status,
        "expected_count": len(expected_ids),
        "record_count": len(by_id),
        "duplicate_target_ids": sorted(set(duplicate_ids)),
        "missing_target_ids": missing,
        "extra_target_ids": extra,
        "technical_anomaly_counts": dict(technical_counts),
        "content_quality_flag_counts": dict(content_counts),
        "content_quality_flagged_count": sum(
            bool(row.get("content_quality_flags")) for row in by_id.values()
        ),
        "trainable_target_ids": trainable_ids,
        "trainable_count": len(trainable_ids),
        "all_targets_semantically_correct": False,
    }


def validate_observational_review(
    template_rows: list[dict[str, Any]], reviewed_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    template_ids = [str(row.get("target_id")) for row in template_rows]
    reviewed_ids = [str(row.get("target_id")) for row in reviewed_rows]
    errors: list[str] = []
    if len(set(template_ids)) != len(template_ids):
        errors.append("duplicate target_id in review template")
    if len(set(reviewed_ids)) != len(reviewed_ids):
        errors.append("duplicate target_id in review submission")
    if set(template_ids) != set(reviewed_ids):
        errors.append("review target_id set differs from frozen template")
    by_id = {str(row.get("target_id")): row for row in reviewed_rows}
    immutable = (
        "source", "target_id", "canonical_en_target", "candidate_translation"
    )
    decisions = (
        "faithful_complete_translation", "terminology_correct",
        "clean_output_only",
    )
    quality_counts: Counter[str] = Counter()
    for frozen in template_rows:
        target_id = str(frozen.get("target_id"))
        submitted = by_id.get(target_id)
        if submitted is None:
            continue
        for field in immutable:
            if submitted.get(field) != frozen.get(field):
                errors.append(f"target_id={target_id} immutable field changed: {field}")
        if submitted.get("review_status") not in {"PASS", "FAIL"}:
            errors.append(f"target_id={target_id} review_status must be PASS or FAIL")
        else:
            quality_counts[str(submitted["review_status"])] += 1
        for field in decisions:
            if not isinstance(submitted.get(field), bool):
                errors.append(f"target_id={target_id} {field} is not boolean")
            elif submitted[field]:
                quality_counts[f"{field}_true"] += 1
            else:
                quality_counts[f"{field}_false"] += 1
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "rows": len(template_rows),
        "quality_counts": dict(quality_counts),
        "gates_freezing": False,
    }


def inspect_history(
    attempts: list[dict[str, Any]],
    dataset: str,
    contract_sha256: str,
    generation_ids: set[str],
    attempt_limit: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in attempts:
        c.require(
            record.get("generation_contract_sha256") == contract_sha256,
            "GENERATION_CONTRACT_MISMATCH in historical attempt record",
        )
        target_id = str(record.get("target_id"))
        c.require(target_id in generation_ids, f"unexpected attempt target_id={target_id}")
        grouped[target_id].append(record)
    completed: dict[str, dict[str, Any]] = {}
    next_variant: dict[str, int] = {}
    for target_id, rows in grouped.items():
        variants = [int(row["variant"]) for row in rows]
        c.require(
            variants == list(range(len(rows))),
            f"non-contiguous or duplicate generation variants target_id={target_id}",
        )
        c.require(len(rows) <= attempt_limit, f"too many attempts target_id={target_id}")
        if dataset == "matched_prompt":
            c.require(len(rows) == 1, f"MatchedPrompt retry detected target_id={target_id}")
            completed[target_id] = rows[0]
            continue
        technical = [row for row in rows if row.get("technical_valid") is False]
        passed = [row for row in rows if row.get("status") == "PASS"]
        c.require(len(technical) <= 1 and len(passed) <= 1, f"invalid history target_id={target_id}")
        if technical:
            completed[target_id] = technical[0]
        elif passed:
            c.require(passed[0] is rows[-1], f"records after PASS target_id={target_id}")
            completed[target_id] = passed[0]
        elif len(rows) == attempt_limit:
            completed[target_id] = rows[-1]
        else:
            next_variant[target_id] = len(rows)
    return completed, next_variant


def _initial_audit(
    contexts: list[dict[str, Any]], old: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], set[str]]:
    records, flagged_ids = [], set()
    for src, row in zip(contexts, old):
        audit = c.audit_translation(
            row["target_translation"], src["lexical_record"]["en_name"],
            src["lexical_record"].get("en_alias", ""),
        )
        records.append({
            "target_id": row["target_id"], "job_id": row["job_id"],
            "flags": audit["flags"],
            "old_target_sha256": hashlib.sha256(
                row["target_translation"].encode()
            ).hexdigest(),
        })
        if audit["flags"]:
            flagged_ids.add(str(row["target_id"]))
    return records, flagged_ids


def audit_only_report(
    dataset: str,
    contexts: list[dict[str, Any]],
    old: list[dict[str, Any]],
) -> dict[str, Any]:
    initial, flagged_ids = _initial_audit(contexts, old)
    generation_count = len(old) if dataset == "matched_prompt" else len(flagged_ids)
    return {
        "status": "AUDIT_ONLY_READ_ONLY",
        "dataset": dataset,
        "rows": len(old),
        "historical_flagged_count": len(flagged_ids),
        "generation_count": generation_count,
        "flag_counts": dict(Counter(
            flag for record in initial for flag in record["flags"]
        )),
        "writes_performed": 0,
    }


def _record_for_output(
    dataset: str,
    target_id: str,
    src: dict[str, Any],
    variant: int,
    template: str,
    text: str,
    student_tokenizer: Any,
    generation_contract_sha256: str,
    generation_exception: str | None = None,
) -> dict[str, Any]:
    content_audit = c.audit_translation(
        text, src["lexical_record"]["en_name"],
        src["lexical_record"].get("en_alias", ""),
    )
    total_tokens = None
    if generation_exception is None:
        total_tokens = (
            len(c.render_prompt_ids(
                student_tokenizer,
                c.DIRECT_PROMPT.format(source=src["src_text"]),
            ))
            + len(student_tokenizer(text, add_special_tokens=False)["input_ids"])
            + 1
        )
    classification = classify_output(
        dataset, text, content_audit, total_tokens, generation_exception
    )
    return {
        **classification,
        "dataset": dataset,
        "target_id": target_id,
        "job_id": src["job_id"],
        "variant": variant,
        "semantic_attempt_index": variant + 1,
        "generation_contract_sha256": generation_contract_sha256,
        "teacher_prompt_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "raw_target_translation": text,
        "target_translation": text,
        "target_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "content_audit": content_audit,
        "total_training_tokens": total_tokens,
        "generation_exception": generation_exception,
        "timestamp": c.now(),
    }


def _build_candidate_rows(
    dataset: str,
    old: list[dict[str, Any]],
    generation_ids: set[str],
    completed: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    candidate_rows = []
    for row in old:
        copied, target_id = dict(row), str(row["target_id"])
        old_text = row["target_translation"]
        if target_id in generation_ids:
            record = completed[target_id]
            new_text = record["target_translation"]
            copied["target_translation"] = copied["reference"] = new_text
            copied["clean_sft_provenance"] = {
                "source": (
                    "qwen3_8b_exact_opd_prompt_raw"
                    if dataset == "matched_prompt"
                    else "qwen3_8b_selective_repair"
                ),
                "generated": True,
                "repaired": dataset == "selective_clean",
                "old_target_sha256": hashlib.sha256(old_text.encode()).hexdigest(),
                "new_target_sha256": hashlib.sha256(new_text.encode()).hexdigest(),
                "teacher_prompt_sha256": record["teacher_prompt_sha256"],
                "generation_contract_sha256": record["generation_contract_sha256"],
                "technical_valid": record["technical_valid"],
                "technical_flags": record["technical_flags"],
                "content_quality_pass": record["content_quality_pass"],
                "content_quality_flags": record["content_quality_flags"],
                "raw_output_preserved_without_replacement": (
                    dataset == "matched_prompt"
                ),
            }
        else:
            target_sha = hashlib.sha256(old_text.encode()).hexdigest()
            copied["clean_sft_provenance"] = {
                "source": "historical_target_retained",
                "generated": False, "repaired": False,
                "old_target_sha256": target_sha,
                "new_target_sha256": target_sha,
            }
        candidate_rows.append(copied)
    return candidate_rows


def _candidate_records(
    dataset: str,
    contexts: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    student_tokenizer: Any,
) -> list[dict[str, Any]]:
    evidence = []
    for src, row in zip(contexts, rows):
        audit = c.audit_translation(
            row.get("target_translation", ""),
            src["lexical_record"]["en_name"],
            src["lexical_record"].get("en_alias", ""),
        )
        text = row.get("target_translation", "")
        total = (
            len(c.render_prompt_ids(
                student_tokenizer,
                c.DIRECT_PROMPT.format(source=src["src_text"]),
            ))
            + len(student_tokenizer(text, add_special_tokens=False)["input_ids"])
            + 1
        )
        classified = classify_output(dataset, text, audit, total)
        provenance = row.get("clean_sft_provenance", {})
        if provenance.get("generated"):
            technical_flags = sorted(set(
                classified["technical_flags"]
                + provenance.get("technical_flags", [])
            ))
            content_flags = sorted(set(
                classified["content_quality_flags"]
                + provenance.get("content_quality_flags", [])
            ))
            classified["technical_flags"] = technical_flags
            classified["technical_valid"] = not technical_flags
            classified["content_quality_flags"] = content_flags
            classified["content_quality_pass"] = not content_flags
            classified["status"] = (
                "TECHNICAL_INVALID" if technical_flags
                else "RAW_OUTPUT_RECORDED" if dataset == "matched_prompt"
                else "CONTENT_REJECTED" if content_flags
                else "CLEAN_OUTPUT_ACCEPTED"
            )
        evidence.append({
            "target_id": str(row["target_id"]),
            "job_id": row["job_id"],
            **classified,
            "total_training_tokens": total,
            "content_audit": audit,
        })
    return evidence


def _write_review_template(
    directory: Path,
    rows: list[dict[str, Any]],
    sample_size: int,
) -> tuple[Path, list[dict[str, Any]]]:
    generated = [
        row for row in rows if row["clean_sft_provenance"]["generated"]
    ]
    retained = [
        row for row in rows if not row["clean_sft_provenance"]["generated"]
    ]
    review = [
        kc._review_row(row, "generated_or_repaired")
        for row in c.deterministic_review_sample(generated, sample_size)
    ]
    review.extend(
        kc._review_row(row, "historical_unflagged_retained")
        for row in c.deterministic_review_sample(retained, sample_size)
    )
    path = directory / "manual_review_template.jsonl"
    c.atomic_jsonl(path, review)
    return path, review


def _candidate_manifest_is_complete(
    directory: Path, candidate_path: Path, generation_contract_sha256: str
) -> bool:
    manifest_path = directory / "candidate_manifest.json"
    if not manifest_path.is_file() or not candidate_path.is_file():
        return False
    manifest = c.read_json(manifest_path)
    c.require(
        manifest.get("generation_contract_sha256")
        == generation_contract_sha256,
        "GENERATION_CONTRACT_MISMATCH in candidate manifest",
    )
    c.require(
        manifest.get("candidate_sha256") == c.sha256_file(candidate_path),
        "candidate hash differs from candidate manifest",
    )
    return manifest.get("status") in {
        "CANDIDATE_RAW_DISTRIBUTION_READY",
        "CANDIDATE_AWAITING_MANUAL_REVIEW",
        "CANDIDATE_TECHNICAL_INVALID",
        "CANDIDATE_CONTENT_QUALITY_FAILED",
    }


def cmd_materialize(args: argparse.Namespace) -> None:
    if args.finalize:
        finalize_targets(args)
        return
    spec = kc.DATASETS[args.dataset]
    directory = spec["directory"]
    frozen_manifest = directory / "manifest.json"
    if not args.audit_only:
        c.require(
            not frozen_manifest.exists()
            or c.read_json(frozen_manifest).get("status") != "FROZEN",
            "target artifact is already FROZEN",
        )
    contexts, old = c.read_jsonl(c.CONTEXT), c.read_jsonl(c.OLD_TARGETS)
    identity = c.source_identity(contexts, old)
    c.require(identity["status"] == "PASS", "historical source-target identity failed")
    if args.audit_only:
        print(json.dumps(
            audit_only_report(args.dataset, contexts, old),
            ensure_ascii=False, indent=2,
        ))
        return

    teacher_audit = kc.teacher_condition_parity()
    c.require(teacher_audit["status"] == "PASS", "TEACHER_CONDITION_PARITY_BLOCKED")
    generation_contract = build_generation_contract(
        args.dataset, args.max_new_tokens, args.max_retries
    )
    directory.mkdir(parents=True, exist_ok=True)
    generation_contract_sha256 = kc.ensure_immutable_contract(
        directory / "generation_contract.json", generation_contract
    )
    initial_audit, flagged_ids = _initial_audit(contexts, old)
    initial_path = directory / "initial_target_audit.jsonl"
    if initial_path.is_file():
        c.require(
            c.read_jsonl(initial_path) == initial_audit,
            "initial target audit changed during resume",
        )
    else:
        c.atomic_jsonl(initial_path, initial_audit)
    generation_ids = (
        {str(row["target_id"]) for row in old}
        if args.dataset == "matched_prompt" else flagged_ids
    )
    templates = prompt_templates(args.dataset, args.max_retries)
    attempts_path = directory / "candidate/regeneration_attempts.jsonl"
    attempts = c.read_jsonl(attempts_path) if attempts_path.is_file() else []
    completed, next_variant = inspect_history(
        attempts, args.dataset, generation_contract_sha256,
        generation_ids, len(templates),
    )
    candidate_path = directory / "candidate" / spec["candidate_name"]
    if (
        set(completed) == generation_ids
        and _candidate_manifest_is_complete(
            directory, candidate_path, generation_contract_sha256
        )
    ):
        print(f"CANDIDATE_ALREADY_COMPLETE dataset={args.dataset}")
        return

    missing_ids = generation_ids - set(completed)
    teacher_tokenizer = student_tokenizer = model = None
    if missing_ids:
        import torch
        import torch_npu  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.npu.set_device(args.device)
        teacher_tokenizer = AutoTokenizer.from_pretrained(c.TEACHER)
        student_tokenizer = AutoTokenizer.from_pretrained(c.C0)
        model = AutoModelForCausalLM.from_pretrained(
            c.TEACHER, dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).to(args.device).eval()
    started = time.time()
    for src, old_row in zip(contexts, old):
        target_id = str(old_row["target_id"])
        if target_id not in missing_ids:
            continue
        start_variant = next_variant.get(target_id, 0)
        selected = None
        for variant in range(start_variant, len(templates)):
            template = templates[variant]
            prompt = template.format(
                src_term=src["src_term"],
                en_name=src["lexical_record"]["en_name"],
                source=src["src_text"],
            )
            generation_exception = None
            try:
                text = kc._generate(
                    model, teacher_tokenizer, prompt,
                    args.device, args.max_new_tokens,
                )
            except Exception as exc:
                text = ""
                generation_exception = f"{type(exc).__name__}: {exc}"
            record = _record_for_output(
                args.dataset, target_id, src, variant, template, text,
                student_tokenizer, generation_contract_sha256,
                generation_exception,
            )
            c.append_jsonl(attempts_path, record)
            attempts.append(record)
            if args.dataset == "matched_prompt":
                selected = record
                break
            if record["technical_valid"] is False or record["status"] == "PASS":
                selected = record
                break
            if variant + 1 == len(templates):
                selected = record
        c.require(selected is not None, f"no terminal record target_id={target_id}")
        completed[target_id] = selected
        if len(completed) % 25 == 0:
            rate = len(completed) / max(time.time() - started, 1e-9)
            print(
                f"{c.now()} materialize dataset={args.dataset} "
                f"completed={len(completed)}/{len(generation_ids)} "
                f"rate={rate:.3f}/s", flush=True,
            )
    c.require(
        set(completed) == generation_ids,
        f"missing terminal generation records={sorted(generation_ids-set(completed))[:20]}",
    )
    candidate_rows = _build_candidate_rows(
        args.dataset, old, generation_ids, completed
    )
    c.atomic_jsonl(candidate_path, candidate_rows)
    if student_tokenizer is None:
        from transformers import AutoTokenizer
        student_tokenizer = AutoTokenizer.from_pretrained(c.C0)
    candidate_identity = c.source_identity(contexts, candidate_rows)
    evidence = _candidate_records(
        args.dataset, contexts, candidate_rows, student_tokenizer
    )
    evidence_path = directory / "candidate/full_validation.jsonl"
    c.atomic_jsonl(evidence_path, evidence)
    expected_ids = {str(row["target_id"]) for row in old}
    admission = summarize_admission(
        args.dataset, evidence, expected_ids, candidate_identity["status"]
    )
    trainable_path = directory / "candidate/trainable_target_ids.json"
    c.atomic_json(trainable_path, {
        "count": admission["trainable_count"],
        "target_ids": admission["trainable_target_ids"],
    })
    review_path = None
    review: list[dict[str, Any]] = []
    if admission["technical_validity_status"] == "PASS":
        review_path, review = _write_review_template(
            directory, candidate_rows, args.review_sample_size
        )
    if admission["technical_validity_status"] != "PASS":
        status = "CANDIDATE_TECHNICAL_INVALID"
    elif args.dataset == "selective_clean" and not admission["admitted"]:
        status = "CANDIDATE_CONTENT_QUALITY_FAILED"
    elif args.dataset == "matched_prompt":
        status = "CANDIDATE_RAW_DISTRIBUTION_READY"
    else:
        status = "CANDIDATE_AWAITING_MANUAL_REVIEW"
    candidate_manifest = {
        "status": status,
        "dataset": args.dataset,
        "artifact_version": spec["artifact_version"],
        "generation_contract_sha256": generation_contract_sha256,
        "generation_contract": generation_contract,
        "candidate": str(candidate_path),
        "candidate_sha256": c.sha256_file(candidate_path),
        "rows": len(candidate_rows),
        "historical_flagged_count": len(flagged_ids),
        "generated_count": len(generation_ids),
        "teacher_generate_calls_recorded": len(attempts),
        "source_identity": candidate_identity,
        "admission": admission,
        "trainable_target_ids": {
            "path": str(trainable_path),
            "sha256": c.sha256_file(trainable_path),
            "count": admission["trainable_count"],
        },
        "full_validation": {
            "path": str(evidence_path),
            "sha256": c.sha256_file(evidence_path),
        },
        "manual_review": {
            "required_for_freeze": args.dataset == "selective_clean",
            "template": str(review_path) if review_path else None,
            "template_sha256": c.sha256_file(review_path) if review_path else None,
            "rows": len(review),
            "strata": dict(Counter(row["review_stratum"] for row in review)),
        },
        "raw_output_policy": {
            "content_quality_used_for_retry_or_selection": (
                args.dataset == "selective_clean"
            ),
            "matched_prompt_raw_outputs_preserved": (
                args.dataset == "matched_prompt"
            ),
            "all_targets_semantically_correct": False,
        },
        "teacher_condition_parity": teacher_audit,
        "created": c.now(),
    }
    c.atomic_json(directory / "candidate_manifest.json", candidate_manifest)
    if status == "CANDIDATE_TECHNICAL_INVALID":
        print(
            f"TECHNICAL_VALIDITY_BLOCKED dataset={args.dataset} "
            f"counts={admission['technical_anomaly_counts']}"
        )
    elif status == "CANDIDATE_CONTENT_QUALITY_FAILED":
        print(
            f"TARGET_QUALITY_BLOCKED dataset={args.dataset} "
            f"flagged={admission['content_quality_flagged_count']}"
        )
    else:
        print(
            f"TARGET_CANDIDATE_READY dataset={args.dataset} "
            f"content_flagged={admission['content_quality_flagged_count']}"
        )


def _load_candidate_state(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], Path, list[dict[str, Any]]]:
    spec = kc.DATASETS[args.dataset]
    directory = spec["directory"]
    candidate_path = directory / "candidate" / spec["candidate_name"]
    manifest_path = directory / "candidate_manifest.json"
    c.require(candidate_path.is_file(), f"candidate missing: {candidate_path}")
    c.require(manifest_path.is_file(), f"candidate manifest missing: {manifest_path}")
    manifest = c.read_json(manifest_path)
    c.require(manifest.get("dataset") == args.dataset, "candidate dataset identity mismatch")
    c.require(
        manifest.get("candidate_sha256") == c.sha256_file(candidate_path),
        "candidate hash differs from manifest",
    )
    generation_contract_path = directory / "generation_contract.json"
    bound = c.read_json(generation_contract_path)
    c.require(
        bound.get("contract_sha256") == manifest.get("generation_contract_sha256")
        and bound.get("contract") == manifest.get("generation_contract"),
        "generation contract differs from candidate manifest",
    )
    return spec, manifest, candidate_path, c.read_jsonl(candidate_path)


def finalize_targets(args: argparse.Namespace) -> None:
    spec, candidate_manifest, candidate_path, rows = _load_candidate_state(args)
    directory, output = spec["directory"], spec["targets"]
    admission = candidate_manifest["admission"]
    c.require(
        admission.get("technical_validity_status") == "PASS",
        "TECHNICAL_VALIDITY_BLOCKED",
    )
    contexts = c.read_jsonl(c.CONTEXT)
    identity = c.source_identity(contexts, rows)
    c.require(identity["status"] == "PASS", f"SOURCE_IDENTITY_FAIL {identity['issues']}")
    trainable_path = Path(candidate_manifest["trainable_target_ids"]["path"])
    c.require(
        trainable_path.is_file()
        and c.sha256_file(trainable_path)
        == candidate_manifest["trainable_target_ids"]["sha256"]
        and candidate_manifest["trainable_target_ids"]["count"] == 5500,
        "TRAINABLE_SAMPLE_IDENTITY_BLOCKED",
    )
    review_info: dict[str, Any]
    template_path = Path(candidate_manifest["manual_review"]["template"])
    c.require(
        template_path.is_file()
        and c.sha256_file(template_path)
        == candidate_manifest["manual_review"]["template_sha256"],
        "frozen manual review template identity mismatch",
    )
    template = c.read_jsonl(template_path)
    if args.dataset == "selective_clean":
        c.require(
            admission.get("content_quality_status") == "PASS",
            "TARGET_QUALITY_BLOCKED",
        )
        c.require(
            args.manual_review_jsonl,
            "--manual-review-jsonl is required for SelectiveClean finalize",
        )
        review_path = Path(args.manual_review_jsonl)
        c.require(review_path.is_file(), f"manual review missing: {review_path}")
        result = kc.validate_manual_review(template, c.read_jsonl(review_path))
        c.require(result["status"] == "PASS", f"MANUAL_REVIEW_BLOCKED {result['errors'][:10]}")
        review_info = {
            "mode": "STRICT_GATE", "required": True,
            "submission": str(review_path),
            "submission_sha256": c.sha256_file(review_path),
            **result,
        }
        target_quality_status = "PASS"
        content_gate = "STRICT_PASS"
    else:
        c.require(
            admission.get("content_quality_status") == "OBSERVED_NOT_GATED",
            "MatchedPrompt content policy mismatch",
        )
        if args.manual_review_jsonl:
            review_path = Path(args.manual_review_jsonl)
            c.require(review_path.is_file(), f"manual review missing: {review_path}")
            result = validate_observational_review(
                template, c.read_jsonl(review_path)
            )
            c.require(result["status"] == "PASS", f"OBSERVATIONAL_REVIEW_INVALID {result['errors'][:10]}")
            review_info = {
                "mode": "OBSERVATIONAL_NOT_GATING", "required": False,
                "submission": str(review_path),
                "submission_sha256": c.sha256_file(review_path),
                **result,
            }
        else:
            review_info = {
                "mode": "NOT_SUBMITTED", "required": False,
                "status": "NOT_REVIEWED", "gates_freezing": False,
                "rows": 0,
            }
        target_quality_status = "RAW_DISTRIBUTION_AUDITED"
        content_gate = "NOT_APPLICABLE_RAW_DISTRIBUTION"
    if output.exists():
        manifest_path = directory / "manifest.json"
        c.require(
            manifest_path.is_file()
            and c.read_json(manifest_path).get("status") != "FROZEN",
            "refusing to overwrite frozen targets",
        )
    c.atomic_jsonl(output, rows)
    generation_contract_path = directory / "generation_contract.json"
    attempts_path = directory / "candidate/regeneration_attempts.jsonl"
    teacher = kc.teacher_condition_parity()
    c.require(teacher["status"] == "PASS", "TEACHER_CONDITION_PARITY_BLOCKED")
    manifest = {
        "status": "FROZEN",
        "dataset": args.dataset,
        "artifact_version": spec["artifact_version"],
        "technical_validity_status": "PASS",
        "target_quality_status": target_quality_status,
        "content_quality_gate": content_gate,
        "teacher_condition_parity_status": (
            "PASS" if args.dataset == "matched_prompt"
            else "QUALITY_CONTROL_NOT_PROMPT_MATCHED"
        ),
        "formal_matched_sft_baseline": args.dataset == "matched_prompt",
        "all_targets_semantically_correct": False,
        "created": c.now(),
        "output": {
            "path": str(output), "rows": len(rows),
            "sha256": c.sha256_file(output),
        },
        "source_identity": identity,
        "generation_contract": {
            "path": str(generation_contract_path),
            "sha256": c.sha256_file(generation_contract_path),
            "contract_sha256": candidate_manifest["generation_contract_sha256"],
        },
        "target_materialization": {
            "attempts_jsonl": str(attempts_path),
            "attempts_sha256": c.sha256_file(attempts_path),
            "teacher_generate_calls_recorded": len(c.read_jsonl(attempts_path)),
            "content_quality_used_for_matched_retry_or_selection": False,
            "student_evaluation_used_for_selection": False,
        },
        "admission": admission,
        "trainable_target_ids": candidate_manifest["trainable_target_ids"],
        "manual_review": {
            "template": str(template_path),
            "template_sha256": c.sha256_file(template_path),
            **review_info,
        },
        "teacher_condition_parity": teacher,
        "claim_boundary": (
            "MatchedPrompt preserves the one-shot raw Teacher distribution: "
            "content-quality flags are reported and never used for retry, selection, "
            "replacement, or freezing. SelectiveClean instead requires strict content "
            "and manual-review gates. Neither arm claims all targets are semantically correct."
        ),
    }
    c.atomic_json(directory / "manifest.json", manifest)
    if args.dataset == "matched_prompt":
        print(
            "RAW_DISTRIBUTION_FREEZE_PASS "
            f"content_flagged={admission['content_quality_flagged_count']}"
        )
    else:
        print("TARGET_QUALITY_PASS dataset=selective_clean")
    print(f"TARGETS_FROZEN={output}")
