#!/usr/bin/env python3
"""Knowledge-condition and artifact-isolation layer for Clean-SFT parity v1."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import _clean_sft_parity_core_v1 as c


OPD_TEACHER_CHEM_TEMPLATE = """You are translating a Chinese sentence into English.

Teacher-only lexical knowledge:
Chinese chemistry term: {src_term}
Canonical English registry term: {en_name}

Use the canonical English registry term when translating that designated
chemistry term. Translate the complete source sentence accurately and
naturally. Do not explain the instruction.

Chinese source:
{source}

Return only the English translation."""

MATCHED_DIR = (
    c.DATA / "targeted_kc_sft_matched_prompt_qwen3_8b_v1_20261015"
)
MATCHED_TARGETS = MATCHED_DIR / "chemistry_train5500_kc_sft_matched_prompt.jsonl"
SELECTIVE_DIR = c.CLEAN_DIR
SELECTIVE_TARGETS = c.CLEAN_TARGETS

DATASETS: dict[str, dict[str, Any]] = {
    "matched_prompt": {
        "directory": MATCHED_DIR,
        "targets": MATCHED_TARGETS,
        "candidate_name": "chemistry_train5500_kc_sft_matched_prompt.jsonl",
        "artifact_version": "kc-sft-matched-prompt-qwen3-8b-v1-20261015",
        "all_rows_generated": True,
        "mixed_source_dataset": False,
        "formal_prompt_match": True,
    },
    "selective_clean": {
        "directory": SELECTIVE_DIR,
        "targets": SELECTIVE_TARGETS,
        "candidate_name": "chemistry_train5500_clean_sft.jsonl",
        "artifact_version": "kc-sft-selective-clean-qwen3-8b-v1-20261015",
        "all_rows_generated": False,
        "mixed_source_dataset": True,
        "formal_prompt_match": False,
    },
}

ARM_CONFIG = {
    "SFT_OLDTARGET_MATCHED_LR2E5": {
        "lr": 2e-5, "targets": c.OLD_TARGETS, "opd": "2e-5",
        "dataset": "old_target", "formal_matched_baseline": False,
    },
    "KC_SFT_MATCHED_PROMPT_LR2E5": {
        "lr": 2e-5, "targets": MATCHED_TARGETS, "opd": "2e-5",
        "dataset": "matched_prompt", "formal_matched_baseline": True,
    },
    "KC_SFT_MATCHED_PROMPT_LR3E5": {
        "lr": 3e-5, "targets": MATCHED_TARGETS, "opd": "3e-5",
        "dataset": "matched_prompt", "formal_matched_baseline": True,
    },
    "KC_SFT_SELECTIVE_CLEAN_LR2E5": {
        "lr": 2e-5, "targets": SELECTIVE_TARGETS, "opd": "2e-5",
        "dataset": "selective_clean", "formal_matched_baseline": False,
    },
    "KC_SFT_SELECTIVE_CLEAN_LR3E5": {
        "lr": 3e-5, "targets": SELECTIVE_TARGETS, "opd": "3e-5",
        "dataset": "selective_clean", "formal_matched_baseline": False,
    },
}


def _literal_assignment(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(x, ast.Name) and x.id == name for x in targets):
                value = ast.literal_eval(node.value)
                c.require(isinstance(value, str), f"{name} is not a string in {path}")
                return value
    raise RuntimeError(f"missing {name} in {path}")


def teacher_condition_parity() -> dict[str, Any]:
    runner_templates: dict[str, str] = {}
    runner_checks: dict[str, Any] = {}
    for label, path in c.OPD_RUNNERS.items():
        source = path.read_text(encoding="utf-8")
        template = _literal_assignment(path, "TEACHER_CHEM_TEMPLATE")
        runner_templates[label] = template
        checks = {
            "teacher_model_identity": (
                "teacher = AutoModelForCausalLM.from_pretrained(" in source
                and "TEACHER," in source
            ),
            "teacher_tokenizer_identity": (
                "AutoTokenizer.from_pretrained(TEACHER)" in source
            ),
            "thinking_disabled": "enable_thinking=False" in source,
            "source_field_exact": 'source=row["src_text"]' in source,
            "src_term_field_exact": 'src_term=row["src_term"]' in source,
            "canonical_field_exact": (
                'row["lexical_record"]["en_name"]' in source
                and "en_name=en_name" in source
            ),
            "student_prefix_teacher_logits": (
                "seqs.append(p + resp)" in source
                and "logits[t] predicts token[t+1]" in source
            ),
        }
        manifest = c.read_json(c.OPD_MANIFESTS[label])
        checks["manifest_teacher_model"] = manifest.get("teacher_model") == str(c.TEACHER)
        runner_checks[label] = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
            "runner_sha256": c.sha256_file(path),
        }
    unique_templates = set(runner_templates.values())
    opd_template = next(iter(unique_templates)) if len(unique_templates) == 1 else ""
    opd_sha = hashlib.sha256(opd_template.encode()).hexdigest() if opd_template else None
    matched_sha = hashlib.sha256(OPD_TEACHER_CHEM_TEMPLATE.encode()).hexdigest()
    contexts = c.read_jsonl(c.CONTEXT)
    row_identity = [{
        "source": row["src_text"],
        "src_term": row["src_term"],
        "en_name": row["lexical_record"]["en_name"],
    } for row in contexts]
    exact = (
        len(unique_templates) == 1
        and opd_template == OPD_TEACHER_CHEM_TEMPLATE
        and all(x["status"] == "PASS" for x in runner_checks.values())
        and len(contexts) == 5500
    )
    return {
        "status": "PASS" if exact else "FAIL",
        "gate": "TEACHER_CONDITION_PARITY",
        "opd_template_sha256": opd_sha,
        "sft_matched_prompt_template_sha256": matched_sha,
        "templates_byte_identical": opd_template == OPD_TEACHER_CHEM_TEMPLATE,
        "opd_runners_template_identical": len(unique_templates) == 1,
        "runners": runner_checks,
        "teacher_model": str(c.TEACHER),
        "teacher_config_sha256": c.sha256_file(c.TEACHER / "config.json"),
        "teacher_tokenizer_config_sha256": c.sha256_file(
            c.TEACHER / "tokenizer_config.json"
        ),
        "row_field_reuse": {
            "rows": len(contexts),
            "source_field": "src_text",
            "src_term_field": "src_term",
            "canonical_field": "lexical_record.en_name",
            "tuple_sha256": c.sha256_json(row_identity),
            "exact": len(contexts) == 5500,
        },
        "wording_difference_audit": {
            "matched_prompt": {
                "difference_from_opd_source_template": "NONE",
                "extra_source_side_knowledge": False,
            },
            "selective_clean": {
                "difference_from_opd_source_template": (
                    "output-format/fidelity wording and retry variants"
                ),
                "extra_source_side_knowledge": False,
                "same_knowledge_fields": True,
                "distribution_equivalent_claim_allowed": False,
            },
        },
        "generation_and_selection": {
            "matched_prompt": {
                "semantic_attempts_per_row": 1,
                "decoding": "greedy",
                "content_based_retry": False,
                "student_evaluation_selection": False,
                "raw_translation_persisted": True,
                "content_quality_gates_freezing": False,
                "technical_validity_gates_trainable_freezing": True,
            },
            "selective_clean": {
                "historical_unflagged_targets_retained": True,
                "flagging_independent_of_student_evaluation": True,
                "fixed_prompt_variants_may_retry": True,
                "student_evaluation_selection": False,
                "mixed_source_dataset": True,
            },
        },
        "necessary_supervision_difference": (
            "OPD queries Teacher logits on Student-generated response prefixes; "
            "KC-SFT generates a complete offline Teacher target and trains with "
            "teacher-forced response-token CE."
        ),
    }


def cmd_preflight(args: argparse.Namespace) -> None:
    contexts, old = c.read_jsonl(c.CONTEXT), c.read_jsonl(c.OLD_TARGETS)
    frozen = c.frozen_input_checks()
    teacher = teacher_condition_parity()
    parity_file = c.read_json(c.PARITY_REPORT) if c.PARITY_REPORT.is_file() else None
    parity_ok = bool(
        parity_file and parity_file.get("ready_for_implementation")
        and not parity_file.get("issues")
    )
    counts: Counter[str] = Counter()
    flagged = 0
    for src, row in zip(contexts, old):
        audit = c.audit_translation(
            row.get("target_translation", ""),
            src["lexical_record"]["en_name"],
            src["lexical_record"].get("en_alias", ""),
        )
        counts.update(audit["flags"])
        flagged += bool(audit["flags"])
    report = {
        "status": "PASS", "created": c.now(), "read_only": True,
        "frozen_inputs": frozen,
        "source_identity": c.source_identity(contexts, old),
        "provided_parity_report": {
            "status": "PASS" if parity_ok else "NOT_VERIFIED",
            "path": str(c.PARITY_REPORT),
            "sha256": c.sha256_file(c.PARITY_REPORT) if c.PARITY_REPORT.is_file() else None,
        },
        "opd_contracts": {
            label: c.inspect_opd_contract(label) for label in c.OPD_MANIFESTS
        },
        "teacher_condition_parity": teacher,
        "old_target_audit": {
            "flagged_rows": flagged, "flag_counts": dict(counts),
            "status": "HISTORICAL_CONTAMINATION_CONFIRMED",
        },
        "datasets": {},
        "claim_boundary": (
            "Matched knowledge condition does not equate offline target quality, "
            "screening, objective, trajectory, response tokens, or compute."
        ),
    }
    for name, spec in DATASETS.items():
        manifest_path = spec["directory"] / "manifest.json"
        if manifest_path.is_file() and spec["targets"].is_file():
            manifest = c.read_json(manifest_path)
            report["datasets"][name] = {
                "status": manifest.get("status"),
                "target_quality_status": manifest.get("target_quality_status"),
                "path": str(spec["targets"]),
                "sha256": c.sha256_file(spec["targets"]),
            }
        else:
            report["datasets"][name] = {"status": "NOT_MATERIALIZED"}
    if (
        any(x.get("status") == "FAIL" for x in frozen.values())
        or report["source_identity"]["status"] != "PASS"
        or any(x["status"] != "PASS" for x in report["opd_contracts"].values())
        or teacher["status"] != "PASS" or not parity_ok
    ):
        report["status"] = "BLOCKED"
    if args.json_out:
        c.atomic_json(Path(args.json_out), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("SOURCE_IDENTITY_PASS" if report["source_identity"]["status"] == "PASS" else "SOURCE_IDENTITY_FAIL")
    print("TEACHER_CONDITION_PARITY_PASS" if teacher["status"] == "PASS" else "TEACHER_CONDITION_PARITY_BLOCKED")
    print("CONFIG_PARITY_PASS" if report["status"] == "PASS" else "CONFIG_PARITY_BLOCKED")


def _generate(model: Any, tokenizer: Any, prompt: str, device: str, max_new: int) -> str:
    import torch
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    batch = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    batch = {key: value.to(device) for key, value in batch.items()}
    width = batch["input_ids"].shape[1]
    with torch.no_grad():
        output = model.generate(
            **batch, do_sample=False, max_new_tokens=max_new,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id, use_cache=True,
        )
    return tokenizer.decode(
        output[0, width:].detach().cpu().tolist(), skip_special_tokens=True
    ).strip()


def _review_row(row: dict[str, Any], stratum: str) -> dict[str, Any]:
    return {
        "target_id": row["target_id"], "job_id": row["job_id"],
        "review_stratum": stratum,
        "source": row["source"], "src_term": row["src_term"],
        "canonical_en_target": row["canonical_en_target"],
        "candidate_translation": row["target_translation"],
        "review_status": "PENDING",
        "faithful_complete_translation": None,
        "terminology_correct": None,
        "clean_output_only": None,
        "reviewer_notes": "",
    }


def validate_manual_review(
    template_rows: list[dict[str, Any]], reviewed_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    template_ids = [str(x.get("target_id")) for x in template_rows]
    reviewed_ids = [str(x.get("target_id")) for x in reviewed_rows]
    errors: list[str] = []
    if len(set(template_ids)) != len(template_ids):
        errors.append("duplicate target_id in frozen review template")
    if len(set(reviewed_ids)) != len(reviewed_ids):
        errors.append("duplicate target_id in submitted review")
    missing = sorted(set(template_ids) - set(reviewed_ids))
    extra = sorted(set(reviewed_ids) - set(template_ids))
    if missing:
        errors.append(f"missing target_ids={missing[:10]}")
    if extra:
        errors.append(f"unexpected target_ids={extra[:10]}")
    by_id = {str(row.get("target_id")): row for row in reviewed_rows}
    immutable = (
        "source", "target_id", "canonical_en_target", "candidate_translation"
    )
    required_true = (
        "faithful_complete_translation", "terminology_correct", "clean_output_only"
    )
    for frozen in template_rows:
        target_id = str(frozen.get("target_id"))
        submitted = by_id.get(target_id)
        if submitted is None:
            continue
        for field in immutable:
            if submitted.get(field) != frozen.get(field):
                errors.append(f"target_id={target_id} immutable field changed: {field}")
        if submitted.get("review_status") != "PASS":
            errors.append(f"target_id={target_id} review_status is not PASS")
        for field in required_true:
            if submitted.get(field) is not True:
                errors.append(f"target_id={target_id} {field} is not literal true")
    strata = Counter(str(x.get("review_stratum")) for x in template_rows)
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors, "rows": len(template_rows),
        "strata": dict(strata),
    }


def _dataset(args: argparse.Namespace) -> dict[str, Any]:
    c.require(args.dataset in DATASETS, f"unknown dataset {args.dataset}")
    return DATASETS[args.dataset]


def _candidate_path(spec: dict[str, Any]) -> Path:
    return spec["directory"] / "candidate" / spec["candidate_name"]


def finalize_targets(args: argparse.Namespace) -> None:
    spec = _dataset(args)
    directory, output = spec["directory"], spec["targets"]
    candidate = _candidate_path(spec)
    c.require(candidate.is_file(), f"candidate missing: {candidate}")
    c.require(args.manual_review_jsonl, "--manual-review-jsonl is required with --finalize")
    review_path = Path(args.manual_review_jsonl)
    c.require(review_path.is_file(), f"manual review missing: {review_path}")
    template_path = directory / "manual_review_template.jsonl"
    candidate_state = c.read_json(directory / "candidate_manifest.json")
    c.require(
        candidate_state.get("candidate_sha256") == c.sha256_file(candidate)
        and candidate_state.get("manual_review_template_sha256")
        == c.sha256_file(template_path),
        "FROZEN_REVIEW_TEMPLATE_OR_CANDIDATE_IDENTITY_MISMATCH",
    )
    template = c.read_jsonl(template_path)
    review_result = validate_manual_review(template, c.read_jsonl(review_path))
    c.require(
        review_result["status"] == "PASS",
        f"MANUAL_REVIEW_BLOCKED {review_result['errors'][:10]}",
    )
    contexts, rows = c.read_jsonl(c.CONTEXT), c.read_jsonl(candidate)
    identity = c.source_identity(contexts, rows)
    c.require(identity["status"] == "PASS", f"SOURCE_IDENTITY_FAIL {identity['issues']}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(c.C0)
    audits, failures = c.validate_candidate_rows(contexts, rows, tokenizer)
    c.require(not failures, f"TARGET_QUALITY_FAIL n={len(failures)} first={failures[:3]}")
    candidate_manifest = c.read_json(directory / "candidate_manifest.json")
    c.require(
        candidate_manifest.get("automatic_validation_failures") == 0,
        "TARGET_QUALITY_FAIL candidate manifest records automatic failures",
    )
    if output.exists():
        manifest_path = directory / "manifest.json"
        c.require(
            manifest_path.is_file() and c.read_json(manifest_path).get("status") != "FROZEN",
            "refusing to overwrite frozen targets",
        )
    c.atomic_jsonl(output, rows)
    attempts_path = directory / "candidate/regeneration_attempts.jsonl"
    attempts = c.read_jsonl(attempts_path)
    generated_ids = {str(x["target_id"]) for x in attempts}
    repaired_ids = {
        str(row["target_id"]) for row in rows
        if row.get("clean_sft_provenance", {}).get("repaired")
    }
    teacher = teacher_condition_parity()
    c.require(teacher["status"] == "PASS", "TEACHER_CONDITION_PARITY_BLOCKED")
    manifest = {
        "status": "FROZEN", "target_quality_status": "PASS",
        "teacher_condition_parity_status": (
            "PASS" if spec["formal_prompt_match"] else "QUALITY_CONTROL_NOT_PROMPT_MATCHED"
        ),
        "formal_matched_sft_baseline": bool(spec["formal_prompt_match"]),
        "dataset": args.dataset, "artifact_version": spec["artifact_version"],
        "created": c.now(),
        "inputs": {
            "context": {"path": str(c.CONTEXT), "sha256": c.sha256_file(c.CONTEXT)},
            "old_targets": {"path": str(c.OLD_TARGETS), "sha256": c.sha256_file(c.OLD_TARGETS)},
        },
        "output": {"path": str(output), "rows": len(rows), "sha256": c.sha256_file(output)},
        "source_identity": identity,
        "generated_count": len(generated_ids), "repaired_count": len(repaired_ids),
        "unchanged_count": len(rows) - len(repaired_ids),
        "target_materialization": {
            "attempts_jsonl": str(attempts_path),
            "attempts_sha256": c.sha256_file(attempts_path),
            "teacher_generate_calls": len(attempts),
            "raw_automatic_pass_count": sum(x.get("status") == "PASS" for x in attempts),
            "raw_automatic_pass_rate": sum(x.get("status") == "PASS" for x in attempts) / len(attempts),
            "student_evaluation_used_for_selection": False,
        },
        "teacher": {
            "model": str(c.TEACHER), "enable_thinking": False,
            "do_sample": False, "max_new_tokens": args.max_new_tokens,
            "mixed_source_dataset": spec["mixed_source_dataset"],
            "teacher_condition_parity": teacher,
        },
        "validation": {
            "automatic_rows": len(audits), "automatic_failures": 0,
            "rules_sha256": c.sha256_json([(n, p.pattern) for n, p in c.META_RULES]),
            "canonical_is_not_sole_quality_check": True,
        },
        "manual_review": {
            "template": str(template_path), "template_sha256": c.sha256_file(template_path),
            "submission": str(review_path), "submission_sha256": c.sha256_file(review_path),
            **review_result,
        },
        "claim_boundary": (
            "Automatic rules cover all rows; manual semantic/terminology review covers only "
            "the deterministic sample and does not prove that all 5,500 translations are "
            "semantically correct. SelectiveClean is mixed-source and is not prompt-distribution matched."
        ),
    }
    c.atomic_json(directory / "manifest.json", manifest)
    print(f"TARGET_QUALITY_PASS dataset={args.dataset} rows={len(rows)}")
    if spec["formal_prompt_match"]:
        print("TEACHER_CONDITION_PARITY_PASS")
    print(f"TARGETS_FROZEN={output}")


def cmd_materialize(args: argparse.Namespace) -> None:
    if args.finalize:
        finalize_targets(args)
        return
    spec = _dataset(args)
    directory = spec["directory"]
    frozen_manifest = directory / "manifest.json"
    c.require(
        not frozen_manifest.exists() or c.read_json(frozen_manifest).get("status") != "FROZEN",
        "target artifact is already FROZEN",
    )
    contexts, old = c.read_jsonl(c.CONTEXT), c.read_jsonl(c.OLD_TARGETS)
    c.require(c.source_identity(contexts, old)["status"] == "PASS", "historical source-target identity failed")
    flagged_ids: set[str] = set()
    initial_audit = []
    for src, row in zip(contexts, old):
        audit = c.audit_translation(
            row["target_translation"], src["lexical_record"]["en_name"],
            src["lexical_record"].get("en_alias", ""),
        )
        initial_audit.append({
            "target_id": row["target_id"], "job_id": row["job_id"],
            "flags": audit["flags"],
            "old_target_sha256": hashlib.sha256(row["target_translation"].encode()).hexdigest(),
        })
        if audit["flags"]:
            flagged_ids.add(str(row["target_id"]))
    generation_ids = (
        {str(row["target_id"]) for row in old}
        if spec["all_rows_generated"] else flagged_ids
    )
    directory.mkdir(parents=True, exist_ok=True)
    c.atomic_jsonl(directory / "initial_target_audit.jsonl", initial_audit)
    c.atomic_json(directory / "candidate_manifest.json", {
        "status": "AUDITED", "dataset": args.dataset,
        "historical_flagged_count": len(flagged_ids),
        "generation_count": len(generation_ids),
        "flag_counts": dict(Counter(flag for row in initial_audit for flag in row["flags"])),
        "created": c.now(),
    })
    if args.audit_only:
        print(f"TARGET_AUDIT_COMPLETE dataset={args.dataset} historical_flagged={len(flagged_ids)} to_generate={len(generation_ids)}")
        return
    teacher_audit = teacher_condition_parity()
    c.require(teacher_audit["status"] == "PASS", "TEACHER_CONDITION_PARITY_BLOCKED")
    import torch
    import torch_npu  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.npu.set_device(args.device)
    teacher_tokenizer = AutoTokenizer.from_pretrained(c.TEACHER)
    student_tokenizer = AutoTokenizer.from_pretrained(c.C0)
    model = AutoModelForCausalLM.from_pretrained(
        c.TEACHER, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(args.device).eval()
    attempts_path = directory / "candidate/regeneration_attempts.jsonl"
    attempts = c.read_jsonl(attempts_path) if attempts_path.is_file() else []
    completed: dict[str, dict[str, Any]] = {}
    for record in attempts:
        target_id = str(record["target_id"])
        if spec["all_rows_generated"] or record.get("status") == "PASS":
            c.require(target_id not in completed, f"duplicate completed generation target_id={target_id}")
            completed[target_id] = record
    started = time.time()
    for src, old_row in zip(contexts, old):
        target_id = str(old_row["target_id"])
        if target_id not in generation_ids or target_id in completed:
            continue
        templates = (
            [OPD_TEACHER_CHEM_TEMPLATE]
            if spec["all_rows_generated"]
            else c.TEACHER_CLEAN_TEMPLATES[:args.max_retries]
        )
        selected = None
        for variant, template in enumerate(templates):
            prompt = template.format(
                src_term=src["src_term"],
                en_name=src["lexical_record"]["en_name"], source=src["src_text"],
            )
            text = _generate(model, teacher_tokenizer, prompt, args.device, args.max_new_tokens)
            audit = c.audit_translation(
                text, src["lexical_record"]["en_name"],
                src["lexical_record"].get("en_alias", ""),
            )
            total_tokens = len(c.render_prompt_ids(
                student_tokenizer, c.DIRECT_PROMPT.format(source=src["src_text"])
            )) + len(student_tokenizer(text, add_special_tokens=False)["input_ids"]) + 1
            if total_tokens > c.MAX_TOTAL_LENGTH:
                audit["flags"] = sorted(set(audit["flags"] + ["max_total_length_exceeded"]))
            record = {
                "status": "PASS" if not audit["flags"] else "FAIL",
                "dataset": args.dataset, "target_id": target_id,
                "job_id": src["job_id"], "variant": variant,
                "teacher_prompt_sha256": hashlib.sha256(template.encode()).hexdigest(),
                "raw_target_translation": text, "target_translation": text,
                "target_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "validation": audit, "total_training_tokens": total_tokens,
                "semantic_attempt_index": variant + 1, "timestamp": c.now(),
            }
            c.append_jsonl(attempts_path, record)
            if spec["all_rows_generated"] or record["status"] == "PASS":
                selected, completed[target_id] = record, record
                break
        if selected is None:
            raise RuntimeError(f"TARGET_REGENERATION_FAIL target_id={target_id}; see {attempts_path}")
        if len(completed) % 25 == 0:
            rate = len(completed) / max(time.time() - started, 1e-9)
            print(f"{c.now()} materialize dataset={args.dataset} generated={len(completed)}/{len(generation_ids)} rate={rate:.3f}/s", flush=True)
    missing = sorted(generation_ids - set(completed))
    c.require(not missing, f"missing generated targets: {missing[:20]}")
    candidate_rows = []
    for row in old:
        copied, target_id = dict(row), str(row["target_id"])
        old_text = row["target_translation"]
        if target_id in generation_ids:
            record = completed[target_id]
            new_text = record["target_translation"]
            copied["target_translation"] = copied["reference"] = new_text
            copied["clean_sft_provenance"] = {
                "source": "qwen3_8b_exact_opd_prompt" if spec["all_rows_generated"] else "qwen3_8b_selective_repair",
                "generated": True, "repaired": not spec["all_rows_generated"],
                "old_target_sha256": hashlib.sha256(old_text.encode()).hexdigest(),
                "new_target_sha256": hashlib.sha256(new_text.encode()).hexdigest(),
                "teacher_prompt_sha256": record["teacher_prompt_sha256"],
                "automatic_status": record["status"],
            }
        else:
            target_sha = hashlib.sha256(old_text.encode()).hexdigest()
            copied["clean_sft_provenance"] = {
                "source": "historical_target_retained", "repaired": False,
                "generated": False,
                "old_target_sha256": target_sha, "new_target_sha256": target_sha,
            }
        candidate_rows.append(copied)
    candidate_path = _candidate_path(spec)
    c.atomic_jsonl(candidate_path, candidate_rows)
    audits, failures = c.validate_candidate_rows(contexts, candidate_rows, student_tokenizer)
    c.atomic_jsonl(directory / "candidate/full_validation.jsonl", audits)
    generated_passes = sum(completed[x].get("status") == "PASS" for x in generation_ids)
    status = "CANDIDATE_AWAITING_MANUAL_REVIEW" if not failures else "CANDIDATE_AUTOMATIC_QA_FAILED"
    generated_rows = [
        x for x in candidate_rows if x["clean_sft_provenance"]["generated"]
    ]
    retained_rows = [
        x for x in candidate_rows if not x["clean_sft_provenance"]["generated"]
    ]
    review = []
    if not failures:
        review.extend(_review_row(x, "generated_or_repaired") for x in c.deterministic_review_sample(generated_rows, args.review_sample_size))
        if retained_rows:
            review.extend(_review_row(x, "historical_unflagged_retained") for x in c.deterministic_review_sample(retained_rows, args.review_sample_size))
        c.atomic_jsonl(directory / "manual_review_template.jsonl", review)
    c.atomic_json(directory / "candidate_manifest.json", {
        "status": status, "dataset": args.dataset, "rows": len(candidate_rows),
        "candidate": str(candidate_path), "candidate_sha256": c.sha256_file(candidate_path),
        "historical_flagged_count": len(flagged_ids), "generated_count": len(generation_ids),
        "raw_automatic_pass_count": generated_passes,
        "raw_automatic_pass_rate": generated_passes / len(generation_ids),
        "automatic_validation_failures": len(failures),
        "manual_review_rows": len(review),
        "manual_review_template_sha256": (
            c.sha256_file(directory / "manual_review_template.jsonl") if not failures else None
        ),
        "manual_review_strata": dict(Counter(x["review_stratum"] for x in review)),
        "student_evaluation_used_for_selection": False,
        "teacher_condition_parity": teacher_audit,
        "created": c.now(),
    })
    if failures:
        print(f"TARGET_QUALITY_BLOCKED dataset={args.dataset} failures={len(failures)} raw_pass_rate={generated_passes / len(generation_ids):.6f}")
        return
    print(f"TARGET_CANDIDATE_PASS dataset={args.dataset} rows={len(candidate_rows)} raw_pass_rate={generated_passes / len(generation_ids):.6f}")
    print(f"NEXT=review {directory / 'manual_review_template.jsonl'} then rerun materialize --dataset {args.dataset} --finalize --manual-review-jsonl <reviewed-file>")


def ensure_immutable_contract(path: Path, contract: dict[str, Any]) -> str:
    digest = c.sha256_json(contract)
    payload = {"status": "BOUND", "contract_sha256": digest, "contract": contract}
    if path.is_file():
        existing = c.read_json(path)
        c.require(
            existing.get("contract_sha256") == digest
            and existing.get("contract") == contract,
            f"ARTIFACT_IDENTITY_MISMATCH {path}",
        )
    else:
        c.atomic_json(path, payload)
    return digest


def model_identity(path: Path) -> dict[str, Any]:
    c.require(path.is_dir(), f"model missing: {path}")
    files = []
    for item in sorted(path.iterdir()):
        if item.is_file() and (
            item.name in {"config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json"}
            or item.suffix == ".safetensors"
        ):
            files.append({"name": item.name, "size": item.stat().st_size, "sha256": c.sha256_file(item)})
    c.require(any(x["name"] == "config.json" for x in files), "model config missing")
    c.require(any(x["name"].endswith(".safetensors") for x in files), "model weights missing")
    return {"resolved_path": str(path.resolve()), "files": files, "identity_sha256": c.sha256_json(files)}


def resolve_latest_checkpoint(run_dir: Path, contract_sha256: str) -> Path | None:
    root = run_dir / "checkpoints"
    complete = []
    for path in sorted(root.glob("epoch_[0-9][0-9][0-9]")):
        if not ((path / "meta.json").is_file() and (path / "state.pt").is_file() and (path / "hf").is_dir()):
            continue
        meta = c.read_json(path / "meta.json")
        c.require(meta.get("run_contract_sha256") == contract_sha256, f"RESUME_CONTRACT_MISMATCH {path}")
        complete.append(path)
    if not complete:
        return None
    newest = complete[-1]
    meta = c.read_json(newest / "meta.json")
    c.atomic_json(run_dir / "latest_resume.json", {"checkpoint": str(newest), **meta})
    return newest


def pass_map(path: Path, key: str, eval_contract_sha256: str) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    result = {}
    for row in c.read_jsonl(path):
        if row.get("status") != "PASS":
            continue
        c.require(
            row.get("eval_contract_sha256") == eval_contract_sha256,
            f"EVAL_CACHE_IDENTITY_MISMATCH {path}",
        )
        value = str(row[key])
        c.require(value not in result, f"duplicate cached prediction key={value} at {path}")
        result[value] = row
    return result
