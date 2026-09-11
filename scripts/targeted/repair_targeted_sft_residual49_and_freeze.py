#!/usr/bin/env python3
"""
Repair only the 49 residual targeted SFT target-synthesis failures, then freeze
the full 11k positive-control target artifact.

Parent artifact:
  targeted_sft_positive_control_targets_qwen3_8b_v1_20260910

Scientific class:
  LAB ADAPTATION / LOCAL RESIDUAL REPAIR

This script never mutates frozen Chemistry/Idiom source contexts and never
regenerates the 10,951 already-successful Teacher targets.

Repair strategy:
  Chemistry (19 rows):
    Qwen3-8B rewrites the source using an explicit exact canonical-English-term
    constraint. A stronger final variant requires the output to begin with the
    canonical term, making exact terminology insertion easy to verify.

  Idiom (30 rows):
    Qwen3-8B rewrites from source + dictionary definition under an English-only
    constraint and an explicit prohibition on copying the Chinese idiom.

All repaired rows are written durably as they succeed. The final merge happens
only after 49/49 durable repairs exist.

No UC/UW/evaluator/Student outputs are read.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

TZ8 = timezone(timedelta(hours=8))
ARTIFACT_ID = "targeted-sft-positive-control-targets-qwen3-8b-v1-20260910"
REPAIR_ID = "targeted-sft-positive-control-residual49-repair-v1-20260910"

STUDENT_PROMPT = (
    "Translate the following text into English without additional explanations:"
    "\n\n{source}\n\n"
)
STUDENT_PROMPT_SHA = hashlib.sha256(STUDENT_PROMPT.encode("utf-8")).hexdigest()
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def now():
    return datetime.now(TZ8).isoformat(timespec="seconds")


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"{path}:{ln}: {e}") from e
    return rows


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_jsonl_atomic(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_successes(parent):
    success = {}
    for p in sorted((parent / "worker_shards").glob("part_*.jsonl")):
        for row in read_jsonl(p):
            if row.get("status") != "PASS":
                continue
            tid = row["target_id"]
            if tid in success:
                raise SystemExit(f"DUP_PARENT_SUCCESS {tid}")
            success[tid] = row
    return success


def load_latest_failures(parent):
    latest = {}
    for p in sorted((parent / "worker_failures").glob("part_*.jsonl")):
        for row in read_jsonl(p):
            latest[row["target_id"]] = row
    return latest


def load_jobs(parent):
    jobs = read_jsonl(parent / "jobs11000.jsonl")
    by_id = {r["target_id"]: r for r in jobs}
    if len(by_id) != 11000:
        raise SystemExit(f"JOBS_IDENTITY_FAIL unique={len(by_id)}")
    return jobs, by_id


def valid(domain, job, text):
    if not isinstance(text, str):
        return False, "non_string"
    text = text.strip()
    if not text:
        return False, "empty"
    if len(text) < 4:
        return False, "too_short"
    if len(text) > 2500:
        return False, "too_long"

    if domain == "chemistry":
        canonical = job["_canonical_en_target"]
        if canonical.casefold() not in text.casefold():
            return False, "canonical_term_missing"
        return True, "ok"

    src_term = job["src_term"]
    if src_term and src_term in text:
        return False, "source_idiom_copied"
    cjk = len(CJK_RE.findall(text))
    if cjk:
        return False, f"cjk_remaining_{cjk}"
    return True, "ok"


def chemistry_prompts(job):
    term = job["src_term"]
    canonical = job["_canonical_en_target"]
    source = job["src_text"]
    return [
f"""Translate the Chinese source sentence into one natural English sentence.

The designated chemistry expression is:
Chinese: {term}

Its exact canonical English registry name is:
{canonical}

Hard constraint: the final English translation MUST contain the exact canonical
English registry name above character-for-character. Do not abbreviate it,
paraphrase it, singularize/pluralize it, change punctuation, or replace any
piece with a synonym.

Source sentence:
{source}

Output only the final English translation.""",

f"""Rewrite the source as faithful English while inserting one frozen technical
name verbatim.

FROZEN STRING -- copy exactly, without any edits:
<<<{canonical}>>>

This string is the English translation of the Chinese chemistry term
"{term}" in the source. The final answer must literally contain the entire
frozen string between <<< >>> above, but do NOT include the angle brackets.

Chinese source:
{source}

Return only a natural English translation. No notes or explanations.""",

f"""You must produce an English translation satisfying an exact-string test.

Required substring:
{canonical}

Before answering, silently verify that your answer contains exactly the full
required substring above. If necessary, reorganize the sentence so the required
substring can be copied unchanged.

Chinese term represented by that substring:
{term}

Chinese source:
{source}

Output only one complete English translation.""",

f"""Translate the source into English.

Your answer MUST START with this exact frozen chemistry name:
{canonical}

After that exact prefix, continue with grammatical English so that the complete
answer faithfully conveys the source sentence. It is acceptable to front the
designated chemistry entity to satisfy the terminology constraint.

Chinese source:
{source}

Do not explain. Output only the English sentence, starting with the exact frozen
name."""
    ]


def idiom_prompts(job):
    term = job["src_term"]
    definition = job["definition"]
    source = job["src_text"]
    return [
f"""Translate this Chinese sentence into fluent English.

Designated idiom: {term}
Dictionary meaning: {definition}

Hard constraints:
1. Convey the idiom's contextual MEANING, not its Chinese surface form.
2. The final answer must contain ZERO Chinese characters.
3. Never copy, quote, transliterate, or mention the idiom "{term}".
4. Do not say "the idiom", "the phrase", or explain the translation.

Chinese source:
{source}

Output only the complete English translation.""",

f"""Produce an English-only translation of the source.

The Chinese idiom "{term}" means:
{definition}

Paraphrase that meaning naturally in context. Remove all Chinese text from the
answer. The literal Chinese idiom itself is forbidden in the output.

Source:
{source}

Return only one fluent English sentence.""",

f"""Rewrite the whole sentence in English using semantic paraphrase.

Source sentence:
{source}

Semantic hint for the source idiom "{term}":
{definition}

Mandatory output checks:
- English only;
- no Chinese characters anywhere;
- do not reproduce "{term}";
- no meta-commentary about an idiom or translation;
- preserve the source meaning.

Final English translation only:"""
    ]


def completed_map(path):
    p = Path(path)
    if not p.exists():
        return {}
    out = {}
    for r in read_jsonl(p):
        if r.get("status") == "PASS":
            out[r["target_id"]] = r
    return out


def worker(args):
    import torch
    import torch_npu  # noqa: F401
    from transformers import AutoTokenizer, AutoModelForCausalLM

    jobs = read_jsonl(args.repair_jobs)
    assigned = [
        r for i, r in enumerate(jobs)
        if i % args.num_workers == args.worker_id
    ]

    done_map = completed_map(args.output)
    device = f"npu:{args.device}"
    torch.npu.set_device(device)

    tok = AutoTokenizer.from_pretrained(args.teacher)
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    initial = sum(r["target_id"] in done_map for r in assigned)
    done = initial
    total = len(assigned)
    started = time.time()

    print(
        f"{now()} REPAIR_WORKER_READY worker={args.worker_id} "
        f"device={device} resume={initial}/{total}",
        flush=True,
    )

    def infer(prompt):
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        batch = tok(rendered, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            ids = model.generate(
                **batch,
                do_sample=False,
                max_new_tokens=512,
                use_cache=True,
            )
        gen = ids[:, batch["input_ids"].shape[1]:]
        return tok.batch_decode(gen, skip_special_tokens=True)[0].strip()

    for job in assigned:
        tid = job["target_id"]
        if tid in done_map:
            continue

        domain = job["domain"]
        prompts = chemistry_prompts(job) if domain == "chemistry" else idiom_prompts(job)
        attempts = []
        accepted = None
        accepted_variant = None

        for variant, prompt in enumerate(prompts):
            try:
                text = infer(prompt)
                ok, reason = valid(domain, job, text)
                attempts.append({
                    "variant": variant,
                    "ok": ok,
                    "reason": reason,
                    "text": text,
                })
                if ok:
                    accepted = text
                    accepted_variant = variant
                    break
            except Exception as e:
                attempts.append({
                    "variant": variant,
                    "ok": False,
                    "reason": "exception",
                    "error": repr(e),
                })

        if accepted is None:
            append_jsonl(
                args.failure,
                {
                    "status": "FAIL",
                    "repair_id": REPAIR_ID,
                    "target_id": tid,
                    "domain": domain,
                    "job_id": job["job_id"],
                    "src_term": job["src_term"],
                    "attempts": attempts,
                    "timestamp": now(),
                },
            )
            print(
                f"{now()} REPAIR_ROW_FAIL worker={args.worker_id} "
                f"target_id={tid}",
                flush=True,
            )
            continue

        student_user = STUDENT_PROMPT.format(source=job["src_text"])
        rec = {
            "status": "PASS",
            "repair_id": REPAIR_ID,
            "repair_class": "LOCAL_RESIDUAL_REPAIR",
            "parent_target_id": tid,
            "target_id": tid,
            "domain": domain,
            "job_id": job["job_id"],
            "entity_key": job.get("entity_key"),
            "src_term": job["src_term"],
            "definition": job.get("definition"),
            "source": job["src_text"],
            "reference": accepted,
            "target_translation": accepted,
            "messages": [{"role": "user", "content": student_user}],
            "student_prompt_sha256": STUDENT_PROMPT_SHA,
            "teacher_model": args.teacher,
            "teacher_enable_thinking": False,
            "teacher_do_sample": False,
            "teacher_max_new_tokens": 512,
            "teacher_prompt_family": (
                "chemistry_residual_exact_term_repair_v1"
                if domain == "chemistry"
                else "idiom_residual_english_only_semantic_repair_v1"
            ),
            "teacher_prompt_variant": accepted_variant,
            "canonical_en_target": (
                job.get("_canonical_en_target") if domain == "chemistry" else None
            ),
            "validation": {
                "chemistry_canonical_term_present": (
                    job["_canonical_en_target"].casefold() in accepted.casefold()
                    if domain == "chemistry" else None
                ),
                "idiom_source_string_absent": (
                    job["src_term"] not in accepted if domain == "idiom" else None
                ),
                "idiom_cjk_count": (
                    len(CJK_RE.findall(accepted)) if domain == "idiom" else None
                ),
            },
            "timestamp": now(),
        }

        append_jsonl(args.output, rec)
        done += 1

        elapsed = max(time.time() - started, 1e-9)
        fresh = done - initial
        rate = fresh / elapsed if fresh else 0.0
        eta = (total - done) / rate if rate else None
        print(
            f"{now()} worker={args.worker_id} done={done}/{total} "
            f"pct={100*done/total:.1f}% rate={rate:.3f}/s "
            f"ETA={'?' if eta is None else f'{eta:.1f}s'} "
            f"target_id={tid}",
            flush=True,
        )

    print(
        f"{now()} REPAIR_WORKER_FINISHED worker={args.worker_id}",
        flush=True,
    )


def count_pass(path):
    return len(completed_map(path))


def master(args):
    parent = Path(args.parent)
    repair_dir = parent / "residual49_repair_v1"
    repair_dir.mkdir(parents=True, exist_ok=True)

    parent_success = load_successes(parent)
    failures = load_latest_failures(parent)
    all_jobs, jobs_by_id = load_jobs(parent)

    missing = [tid for tid in jobs_by_id if tid not in parent_success]
    if len(parent_success) != 10951 or len(missing) != 49:
        raise SystemExit(
            f"PARENT_CONTRACT_FAIL success={len(parent_success)} missing={len(missing)}"
        )

    missing_domains = Counter(jobs_by_id[tid]["domain"] for tid in missing)
    if missing_domains != Counter({"chemistry": 19, "idiom": 30}):
        raise SystemExit(f"RESIDUAL_DOMAIN_CONTRACT_FAIL {dict(missing_domains)}")

    if set(missing) != set(failures):
        only_missing = sorted(set(missing) - set(failures))[:20]
        only_fail = sorted(set(failures) - set(missing))[:20]
        raise SystemExit(
            f"FAILURE_IDENTITY_MISMATCH only_missing={only_missing} only_fail={only_fail}"
        )

    repair_jobs = []
    for tid in missing:
        job = dict(jobs_by_id[tid])
        job["parent_failure"] = failures[tid]
        repair_jobs.append(job)

    jobs_path = repair_dir / "repair_jobs49.jsonl"
    if jobs_path.exists():
        if read_jsonl(jobs_path) != repair_jobs:
            raise SystemExit("IMMUTABLE_REPAIR_JOBS_MISMATCH")
    else:
        write_jsonl_atomic(jobs_path, repair_jobs)

    shard_dir = repair_dir / "worker_shards"
    fail_dir = repair_dir / "worker_failures"
    log_dir = repair_dir / "worker_logs"
    shard_dir.mkdir(exist_ok=True)
    fail_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    procs = []
    handles = []
    for wid in range(args.num_workers):
        outp = shard_dir / f"part_{wid:02d}.jsonl"
        failp = fail_dir / f"part_{wid:02d}.jsonl"
        logp = log_dir / f"worker_{wid:02d}.log"
        fh = open(logp, "a", encoding="utf-8")
        handles.append(fh)

        cmd = [
            sys.executable, "-u", str(Path(__file__).resolve()),
            "--worker",
            "--worker-id", str(wid),
            "--device", str(wid),
            "--num-workers", str(args.num_workers),
            "--repair-jobs", str(jobs_path),
            "--output", str(outp),
            "--failure", str(failp),
            "--teacher", args.teacher,
        ]
        p = subprocess.Popen(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        procs.append((wid, p, outp, failp, logp))

    started = time.time()
    last = None
    while True:
        done = sum(count_pass(outp) for _, _, outp, _, _ in procs)
        running = sum(p.poll() is None for _, p, _, _, _ in procs)
        elapsed = max(time.time() - started, 1e-9)
        rate = done / elapsed if done else 0.0
        eta = (49 - done) / rate if rate else None

        atomic_json(
            repair_dir / "progress.json",
            {
                "status": "RUNNING" if running else "FINALIZING",
                "done": done,
                "total": 49,
                "percentage": round(100 * done / 49, 2),
                "workers_running": running,
                "workers_total": args.num_workers,
                "rate_rows_per_sec": round(rate, 3),
                "eta_seconds": round(eta, 1) if eta is not None else None,
                "updated": now(),
            },
        )

        state = (done, running)
        if state != last:
            print(
                f"{now()} phase=residual49_repair done={done}/49 "
                f"pct={100*done/49:.1f}% workers={running}/{args.num_workers} "
                f"rate={rate:.2f}/s "
                f"ETA={'?' if eta is None else f'{eta:.1f}s'}",
                flush=True,
            )
            last = state

        if running == 0:
            break
        time.sleep(2)

    for fh in handles:
        fh.close()

    bad_workers = [
        {"worker": wid, "rc": p.returncode, "log": str(logp)}
        for wid, p, _, _, logp in procs if p.returncode != 0
    ]
    if bad_workers:
        atomic_json(repair_dir / "worker_process_failures.json", bad_workers)
        raise SystemExit(f"REPAIR_WORKER_PROCESS_FAIL {bad_workers}")

    repaired = {}
    for _, _, outp, _, _ in procs:
        if not outp.exists():
            continue
        for row in read_jsonl(outp):
            if row.get("status") != "PASS":
                continue
            tid = row["target_id"]
            if tid in repaired:
                raise SystemExit(f"DUP_REPAIR_SUCCESS {tid}")
            repaired[tid] = row

    if len(repaired) != 49 or set(repaired) != set(missing):
        left = sorted(set(missing) - set(repaired))
        atomic_json(
            repair_dir / "progress.json",
            {
                "status": "FAIL",
                "done": len(repaired),
                "total": 49,
                "missing": left,
                "updated": now(),
            },
        )
        print(
            f"RESIDUAL_REPAIR_INCOMPLETE pass={len(repaired)}/49 "
            f"missing={len(left)}",
            flush=True,
        )
        print("RERUN_SAME_COMMAND_TO_RETRY_ONLY_UNFINISHED_REPAIRS", flush=True)
        raise SystemExit(2)

    # Compose final target map: immutable 10951 parent + 49 repair rows.
    final = dict(parent_success)
    overlap = set(final) & set(repaired)
    if overlap:
        raise SystemExit(f"UNEXPECTED_PARENT_REPAIR_OVERLAP n={len(overlap)}")
    final.update(repaired)

    if len(final) != 11000:
        raise SystemExit(f"FINAL_CARDINALITY_FAIL {len(final)}")

    ordered = [final[r["target_id"]] for r in all_jobs]
    chem = [r for r in ordered if r["domain"] == "chemistry"]
    idiom = [r for r in ordered if r["domain"] == "idiom"]

    if len(chem) != 5500 or len(idiom) != 5500:
        raise SystemExit(
            f"DOMAIN_CARDINALITY_FAIL chemistry={len(chem)} idiom={len(idiom)}"
        )

    chem_bad = [
        r["target_id"] for r in chem
        if not r.get("canonical_en_target")
        or r["canonical_en_target"].casefold()
        not in r["target_translation"].casefold()
    ]
    if chem_bad:
        raise SystemExit(f"CHEM_FINAL_INVARIANT_FAIL n={len(chem_bad)}")

    idiom_bad_copy = [
        r["target_id"] for r in idiom
        if r["src_term"] and r["src_term"] in r["target_translation"]
    ]
    idiom_bad_cjk = [
        r["target_id"] for r in idiom
        if CJK_RE.search(r["target_translation"])
    ]
    if idiom_bad_copy or idiom_bad_cjk:
        raise SystemExit(
            f"IDIOM_FINAL_INVARIANT_FAIL copied={len(idiom_bad_copy)} "
            f"cjk={len(idiom_bad_cjk)}"
        )

    # Student tokenizer max-length hard check.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.student_tokenizer)
    max_len = 0
    overlong = []

    for r in ordered:
        prompt = tok.apply_chat_template(
            r["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        pids = tok(prompt, add_special_tokens=False)["input_ids"]
        tids = tok(r["target_translation"], add_special_tokens=False)["input_ids"]
        n = len(pids) + len(tids) + 1
        max_len = max(max_len, n)
        if n > 1024:
            overlong.append({"target_id": r["target_id"], "tokens": n})

    if overlong:
        write_jsonl_atomic(repair_dir / "overlong_gt1024.jsonl", overlong)
        raise SystemExit(
            f"FINAL_MAX_LENGTH_FAIL n={len(overlong)} max={max_len}"
        )

    chem_path = parent / "chemistry_train5500_sft.jsonl"
    idiom_path = parent / "idiom_train5500_sft.jsonl"
    combined_path = parent / "combined_train11000_sft.jsonl"

    # Parent was not frozen because the first run stopped before finalization.
    # These files are now written once from the validated 10951+49 composition.
    write_jsonl_atomic(chem_path, chem)
    write_jsonl_atomic(idiom_path, idiom)
    write_jsonl_atomic(combined_path, ordered)

    idiom_qc100 = sorted(
        idiom,
        key=lambda r: hashlib.sha256(
            ("idiom-target-qc100-v1|" + r["target_id"]).encode("utf-8")
        ).hexdigest(),
    )[:100]
    qc_path = parent / "idiom_teacher_target_qc100.jsonl"
    write_jsonl_atomic(qc_path, idiom_qc100)

    manifest = {
        "status": "FROZEN",
        "artifact_id": ARTIFACT_ID,
        "scientific_class": "LAB ADAPTATION / POSITIVE-CONTROL TARGET SYNTHESIS",
        "composition": {
            "original_qwen3_8b_pass": 10951,
            "local_residual_repair": 49,
            "repair_id": REPAIR_ID,
            "repair_scientific_class": "LAB ADAPTATION / LOCAL RESIDUAL REPAIR",
        },
        "target_synthesis_contract": {
            "teacher_model": args.teacher,
            "teacher_enable_thinking": False,
            "teacher_do_sample": False,
            "original_teacher_max_new_tokens": 384,
            "repair_teacher_max_new_tokens": 512,
            "chemistry_teacher_hint": "canonical English term; Teacher only",
            "idiom_teacher_hint": "dictionary definition; Teacher only",
            "student_receives_lexical_hint": False,
            "student_prompt": STUDENT_PROMPT,
            "student_prompt_sha256": STUDENT_PROMPT_SHA,
        },
        "data_access_contract": {
            "train_only": True,
            "uc_access": False,
            "uw_access": False,
            "student_output_used": False,
            "evaluator_score_used": False,
            "downstream_result_used": False,
        },
        "repair_contract": {
            "parent_successes_preserved_byte_identically": True,
            "only_parent_missing_49_repaired": True,
            "residual_domain_counts": {"chemistry": 19, "idiom": 30},
            "observed_parent_failure_reasons": {
                "chemistry": "canonical_term_missing",
                "idiom": "source_idiom_copied / residual Chinese",
            },
        },
        "outputs": {
            chem_path.name: {"rows": 5500, "sha256": sha256_file(chem_path)},
            idiom_path.name: {"rows": 5500, "sha256": sha256_file(idiom_path)},
            combined_path.name: {"rows": 11000, "sha256": sha256_file(combined_path)},
            qc_path.name: {"rows": 100, "sha256": sha256_file(qc_path)},
        },
        "hard_checks": {
            "chemistry_canonical_term_present_all_5500": True,
            "idiom_source_term_absent_all_5500": True,
            "idiom_zero_cjk_all_5500": True,
            "all_student_examples_le_1024": True,
            "student_rendered_max_tokens": max_len,
        },
        "planned_sft_exposure_contract": {
            "C1": "Idiom5500 x 3 epochs from C0",
            "C2": "Chemistry5500 x 3 epochs from C0",
            "C3": "Combined11000 x 3 epochs from C0",
            "matching_rule": (
                "same per-example exposure; C3 intentionally has ~2x total "
                "examples/updates versus a single-domain arm"
            ),
        },
        "created": now(),
    }

    manifest_path = parent / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("status") == "FROZEN":
            if old != manifest:
                raise SystemExit("FROZEN_MANIFEST_ALREADY_EXISTS_DIFFERENT")
        else:
            atomic_json(manifest_path, manifest)
    else:
        atomic_json(manifest_path, manifest)

    atomic_json(
        parent / "progress.json",
        {
            "status": "PASS",
            "done": 11000,
            "total": 11000,
            "percentage": 100.0,
            "updated": now(),
        },
    )
    atomic_json(
        repair_dir / "progress.json",
        {
            "status": "PASS",
            "done": 49,
            "total": 49,
            "percentage": 100.0,
            "updated": now(),
        },
    )

    print()
    print("RESIDUAL49_REPAIR=PASS", flush=True)
    print("PARENT_10951_PRESERVED=PASS", flush=True)
    print("FINAL_11000_HARD_CHECKS=PASS", flush=True)
    print(f"CHEMISTRY_TARGETS=5500 sha={sha256_file(chem_path)}", flush=True)
    print(f"IDIOM_TARGETS=5500 sha={sha256_file(idiom_path)}", flush=True)
    print(f"COMBINED_TARGETS=11000 sha={sha256_file(combined_path)}", flush=True)
    print(f"IDIOM_QC100={qc_path}", flush=True)
    print(f"STUDENT_RENDERED_MAX_TOKENS={max_len}", flush=True)
    print("TARGETED_SFT_TARGETS=FROZEN_CLOSED", flush=True)
    print("FINAL_RESULT=TARGETED_SFT_POSITIVE_CONTROL_TARGETS_FROZEN", flush=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-id", type=int)
    ap.add_argument("--device", type=int)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--repair-jobs")
    ap.add_argument("--output")
    ap.add_argument("--failure")
    ap.add_argument("--teacher")
    ap.add_argument("--student-tokenizer")
    ap.add_argument("--parent")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.worker:
        for name in (
            "worker_id", "device", "repair_jobs", "output", "failure", "teacher"
        ):
            if getattr(args, name) is None:
                raise SystemExit(f"MISSING_WORKER_ARG {name}")
        worker(args)
    else:
        for name in ("teacher", "student_tokenizer", "parent"):
            if not getattr(args, name):
                raise SystemExit(f"MISSING_MASTER_ARG {name}")
        master(args)
