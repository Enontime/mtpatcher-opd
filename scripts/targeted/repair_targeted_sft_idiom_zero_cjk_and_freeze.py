#!/usr/bin/env python3
"""
Repair all residual Chinese-character contamination in frozen-target candidates,
then freeze the 11k targeted SFT positive-control target artifact.

Scientific class:
    LAB ADAPTATION / LOCAL RESIDUAL REPAIR

Why this exists:
    The original Idiom target validator rejected only outputs with >4 CJK
    characters. Therefore some rows with 1--4 residual Chinese characters were
    incorrectly accepted. A later final hard check correctly required zero CJK
    and found 86 affected Idiom rows.

This script:
    - composes the current 11,000 candidate targets from:
        original successes (10951)
        residual49 repair successes (45)
        residual4 repair successes (4)
    - selects exactly the Idiom rows whose accepted target still contains >=1
      CJK character;
    - repairs only those rows;
    - never regenerates any already-clean target;
    - writes every repaired row durably and is resume-safe;
    - performs full 11k hard checks before freezing.

Teacher-only repair:
    Replace the designated idiom in the original frozen source with its
    dictionary definition, then translate the resulting semantic paraphrase
    into English-only text. The Student still sees the original frozen Chinese
    source and never sees the definition or transformed Teacher input.
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
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")

ARTIFACT_ID = "targeted-sft-positive-control-targets-qwen3-8b-v1-20260910"
REPAIR_ID = "targeted-sft-idiom-zero-cjk-repair-v1-20260910"

STUDENT_PROMPT = (
    "Translate the following text into English without additional explanations:"
    "\n\n{source}\n\n"
)
STUDENT_PROMPT_SHA = hashlib.sha256(STUDENT_PROMPT.encode("utf-8")).hexdigest()


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


def load_pass_dir(path):
    out = {}
    for p in sorted(Path(path).glob("part_*.jsonl")):
        for row in read_jsonl(p):
            if row.get("status") != "PASS":
                continue
            tid = row["target_id"]
            if tid in out:
                raise SystemExit(f"DUP_PASS target_id={tid} path={path}")
            out[tid] = row
    return out


def compose_current(parent):
    original = load_pass_dir(parent / "worker_shards")
    r49 = load_pass_dir(parent / "residual49_repair_v1" / "worker_shards")
    r4 = load_pass_dir(parent / "residual4_final_repair_v1" / "worker_shards")

    if len(original) != 10951:
        raise SystemExit(f"ORIGINAL_PASS_CONTRACT_FAIL got={len(original)}")
    if len(r49) != 45:
        raise SystemExit(f"REPAIR49_PASS_CONTRACT_FAIL got={len(r49)}")
    if len(r4) != 4:
        raise SystemExit(f"REPAIR4_PASS_CONTRACT_FAIL got={len(r4)}")

    ids = set()
    for label, part in (("original", original), ("r49", r49), ("r4", r4)):
        overlap = ids & set(part)
        if overlap:
            raise SystemExit(f"COMPOSITION_OVERLAP label={label} n={len(overlap)}")
        ids |= set(part)

    current = {}
    current.update(original)
    current.update(r49)
    current.update(r4)

    if len(current) != 11000:
        raise SystemExit(f"CURRENT_COMPOSITION_FAIL got={len(current)}")

    return current, {
        "original": len(original),
        "residual49_repair": len(r49),
        "residual4_repair": len(r4),
    }


def completed_map(path):
    p = Path(path)
    if not p.exists():
        return {}
    out = {}
    for row in read_jsonl(p):
        if row.get("status") == "PASS":
            out[row["target_id"]] = row
    return out


def transform_source(job):
    source = job["source"]
    term = job["src_term"]
    definition = (job.get("definition") or "").strip()

    if not definition:
        raise RuntimeError(f"missing definition: {job['target_id']}")

    # Usually exact term is present. If it is not, still make the semantic hint
    # explicit without altering the Student-side source.
    if term and term in source:
        return source.replace(term, definition, 1)

    return f"{source}\n\n语义说明：{definition}"


def valid(job, text):
    if not isinstance(text, str):
        return False, "non_string"
    text = text.strip()
    if not text:
        return False, "empty"
    if len(text) < 4:
        return False, "too_short"
    if len(text) > 2500:
        return False, "too_long"
    if job["src_term"] and job["src_term"] in text:
        return False, "source_idiom_copied"

    n_cjk = len(CJK_RE.findall(text))
    if n_cjk:
        return False, f"cjk_remaining_{n_cjk}"

    return True, "ok"


def prompts(job):
    transformed = transform_source(job)
    definition = job["definition"].strip()

    return [
f"""Translate the following semantic paraphrase into fluent English.

The original Chinese idiom has already been replaced by its dictionary meaning.
Translate ALL remaining Chinese text.

Source:
{transformed}

Hard constraints:
- English only;
- ZERO Chinese characters;
- no Chinese words, punctuation phrases, or parenthetical Chinese;
- do not quote or mention the original idiom;
- no explanation or meta-commentary;
- preserve the full sentence meaning.

Output only one complete English translation.""",

f"""Produce one fully English sentence from the source below.

Source:
{transformed}

The inserted semantic phrase corresponds to this meaning:
{definition}

Mandatory validation before answering:
1. Scan your answer and remove/translate every Chinese character.
2. Do not include the original Chinese idiom.
3. Do not discuss the translation process.
4. Keep the meaning faithful.

Return only the final English sentence.""",

f"""English-only rewrite task.

Convert every part of this source to English:
{transformed}

Your final response must use ASCII/Latin-script English words and normal English
punctuation only. No CJK characters are allowed anywhere. Preserve the intended
semantic meaning.

Final English sentence only:"""
    ]


def worker(args):
    import torch
    import torch_npu  # noqa: F401
    from transformers import AutoTokenizer, AutoModelForCausalLM

    jobs = read_jsonl(args.jobs)
    assigned = [
        row for i, row in enumerate(jobs)
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

    total = len(assigned)
    initial = sum(r["target_id"] in done_map for r in assigned)
    done = initial
    started = time.time()

    print(
        f"{now()} ZERO_CJK_WORKER_READY worker={args.worker_id} "
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

        transformed = transform_source(job)
        attempts = []
        accepted = None
        accepted_variant = None

        for variant, prompt in enumerate(prompts(job)):
            try:
                text = infer(prompt)
                ok, reason = valid(job, text)
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


        # SECONDARY_ENGLISH_ONLY_REWRITE_V1
        #
        # Some rare rows still leave a tiny Chinese fragment after the
        # source+definition translation. Do not delete characters mechanically.
        # Take the best primary candidate and ask the same frozen Teacher to
        # semantically rewrite it into fully English text.
        if accepted is None:
            candidate_texts = [
                a["text"]
                for a in attempts
                if isinstance(a.get("text"), str) and a["text"].strip()
            ]

            if candidate_texts:
                seed = min(
                    candidate_texts,
                    key=lambda x: (
                        job["src_term"] in x,
                        len(CJK_RE.findall(x)),
                        len(x),
                    ),
                )

                cleanup_prompts = [
                    f"""Rewrite the candidate translation below into completely fluent English.

Candidate:
{seed}

Dictionary meaning of the designated source idiom:
{job['definition']}

Translate every remaining Chinese fragment into English.
Do not copy or quote the Chinese idiom "{job['src_term']}".

Hard constraints:
- zero Chinese/CJK characters;
- preserve the candidate's intended meaning;
- no explanation;
- output only one complete English sentence.""",

                    f"""Convert this mixed-language translation into English-only text:

{seed}

The intended idiomatic meaning is:
{job['definition']}

Preserve all correct English content, but translate or paraphrase every
remaining non-English fragment. The final answer must contain no Chinese
characters and must not contain "{job['src_term']}".

Return only the corrected English translation.""",

                    f"""Produce a fresh English-only translation.

Original Chinese source:
{job['source']}

Meaning of the designated idiom:
{job['definition']}

Express the idiom semantically rather than literally.

Before returning the answer, verify:
1. every part is translated into English;
2. there are zero CJK characters;
3. the Chinese string "{job['src_term']}" does not appear;
4. there is no meta-commentary.

Output only the final English sentence."""
                ]

                for cleanup_variant, cleanup_prompt in enumerate(cleanup_prompts):
                    try:
                        text = infer(cleanup_prompt)
                        ok, reason = valid(job, text)

                        attempts.append({
                            "variant": 100 + cleanup_variant,
                            "stage": "secondary_english_only_rewrite",
                            "ok": ok,
                            "reason": reason,
                            "text": text,
                        })

                        if ok:
                            accepted = text
                            accepted_variant = 100 + cleanup_variant
                            break

                    except Exception as e:
                        attempts.append({
                            "variant": 100 + cleanup_variant,
                            "stage": "secondary_english_only_rewrite",
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
                    "job_id": job["job_id"],
                    "src_term": job["src_term"],
                    "teacher_only_transformed_source": transformed,
                    "attempts": attempts,
                    "timestamp": now(),
                },
            )
            print(
                f"{now()} ZERO_CJK_ROW_FAIL worker={args.worker_id} "
                f"target_id={tid}",
                flush=True,
            )
            continue

        rec = {
            "status": "PASS",
            "repair_id": REPAIR_ID,
            "repair_class": "LOCAL_RESIDUAL_REPAIR",
            "target_id": tid,
            "domain": "idiom",
            "job_id": job["job_id"],
            "entity_key": job.get("entity_key"),
            "src_term": job["src_term"],
            "definition": job["definition"],
            "source": job["source"],
            "reference": accepted,
            "target_translation": accepted,
            "messages": [
                {
                    "role": "user",
                    "content": STUDENT_PROMPT.format(source=job["source"]),
                }
            ],
            "student_prompt_sha256": STUDENT_PROMPT_SHA,
            "teacher_model": args.teacher,
            "teacher_enable_thinking": False,
            "teacher_do_sample": False,
            "teacher_max_new_tokens": 512,
            "teacher_prompt_family": "idiom_zero_cjk_semantic_rewrite_v1",
            "teacher_prompt_variant": accepted_variant,
            "teacher_only_transformed_source": transformed,
            "canonical_en_target": None,
            "validation": {
                "idiom_source_string_absent": job["src_term"] not in accepted,
                "idiom_cjk_count": len(CJK_RE.findall(accepted)),
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
            f"pct={100*done/total:.1f}% "
            f"ETA={'?' if eta is None else f'{eta:.1f}s'} "
            f"target_id={tid}",
            flush=True,
        )

    print(
        f"{now()} ZERO_CJK_WORKER_FINISHED worker={args.worker_id}",
        flush=True,
    )


def master(args):
    parent = Path(args.parent)
    repair_dir = parent / "idiom_zero_cjk_repair_v1"
    repair_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = parent / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("status") == "FROZEN":
            print("ARTIFACT_ALREADY_FROZEN", flush=True)
            print(json.dumps(old, ensure_ascii=False, indent=2), flush=True)
            return

    current, composition = compose_current(parent)

    jobs = read_jsonl(parent / "jobs11000.jsonl")
    job_ids = [r["target_id"] for r in jobs]
    if len(job_ids) != 11000 or len(set(job_ids)) != 11000:
        raise SystemExit("JOBS_IDENTITY_FAIL")

    bad = []
    for tid in job_ids:
        row = current[tid]
        if row["domain"] != "idiom":
            continue
        cjk_count = len(CJK_RE.findall(row["target_translation"]))
        if cjk_count:
            x = dict(row)
            x["_pre_repair_cjk_count"] = cjk_count
            bad.append(x)

    print(
        f"{now()} ZERO_CJK_SELECTION rows={len(bad)} "
        f"counts={dict(Counter(r['_pre_repair_cjk_count'] for r in bad))}",
        flush=True,
    )

    # The observed run is expected to produce exactly 86. Refuse silent drift.
    if len(bad) != 86:
        raise SystemExit(
            f"ZERO_CJK_SELECTION_CONTRACT_FAIL expected=86 got={len(bad)}"
        )

    jobs_path = repair_dir / "repair_jobs86.jsonl"
    if jobs_path.exists():
        if read_jsonl(jobs_path) != bad:
            raise SystemExit("IMMUTABLE_REPAIR86_JOBS_MISMATCH")
    else:
        write_jsonl_atomic(jobs_path, bad)

    shard_dir = repair_dir / "worker_shards"
    fail_dir = repair_dir / "worker_failures"
    log_dir = repair_dir / "worker_logs"
    shard_dir.mkdir(exist_ok=True)
    fail_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    workers = min(args.num_workers, 8)
    procs = []
    handles = []

    for wid in range(workers):
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
            "--num-workers", str(workers),
            "--jobs", str(jobs_path),
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
        repaired = load_pass_dir(shard_dir)
        done = len(repaired)
        running = sum(p.poll() is None for _, p, _, _, _ in procs)
        elapsed = max(time.time() - started, 1e-9)
        rate = done / elapsed if done else 0.0
        eta = (86 - done) / rate if rate else None

        atomic_json(
            repair_dir / "progress.json",
            {
                "status": "RUNNING" if running else "FINALIZING",
                "done": done,
                "total": 86,
                "percentage": round(100 * done / 86, 2),
                "workers_running": running,
                "workers_total": workers,
                "elapsed_seconds": round(elapsed, 1),
                "rate_rows_per_sec": round(rate, 3),
                "eta_seconds": round(eta, 1) if eta is not None else None,
                "updated": now(),
            },
        )

        state = (done, running)
        if state != last:
            print(
                f"{now()} phase=idiom_zero_cjk_repair done={done}/86 "
                f"pct={100*done/86:.1f}% workers={running}/{workers} "
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
        for wid, p, _, _, logp in procs
        if p.returncode != 0
    ]
    if bad_workers:
        atomic_json(repair_dir / "worker_process_failures.json", bad_workers)
        raise SystemExit(f"REPAIR86_WORKER_PROCESS_FAIL {bad_workers}")

    repaired = load_pass_dir(shard_dir)
    selected_ids = {r["target_id"] for r in bad}

    if len(repaired) != 86 or set(repaired) != selected_ids:
        missing = sorted(selected_ids - set(repaired))
        print(
            f"ZERO_CJK_REPAIR_INCOMPLETE pass={len(repaired)}/86 "
            f"missing={len(missing)}",
            flush=True,
        )
        print("RERUN_SAME_COMMAND_TO_RETRY_ONLY_UNFINISHED_ROWS", flush=True)
        raise SystemExit(2)

    # Replace exactly the 86 dirty Idiom targets in the candidate composition.
    final = dict(current)
    for tid, row in repaired.items():
        final[tid] = row

    if len(final) != 11000:
        raise SystemExit(f"FINAL_COMPOSITION_FAIL got={len(final)}")

    ordered = [final[tid] for tid in job_ids]
    chem = [r for r in ordered if r["domain"] == "chemistry"]
    idiom = [r for r in ordered if r["domain"] == "idiom"]

    if len(chem) != 5500 or len(idiom) != 5500:
        raise SystemExit(
            f"DOMAIN_COUNT_FAIL chem={len(chem)} idiom={len(idiom)}"
        )

    chem_bad = [
        r["target_id"] for r in chem
        if not r.get("canonical_en_target")
        or r["canonical_en_target"].casefold()
        not in r["target_translation"].casefold()
    ]
    idiom_copy = [
        r["target_id"] for r in idiom
        if r["src_term"] and r["src_term"] in r["target_translation"]
    ]
    idiom_cjk = [
        r["target_id"] for r in idiom
        if CJK_RE.search(r["target_translation"])
    ]

    if chem_bad or idiom_copy or idiom_cjk:
        raise SystemExit(
            f"FINAL_TEXT_INVARIANT_FAIL chem={len(chem_bad)} "
            f"idiom_copy={len(idiom_copy)} idiom_cjk={len(idiom_cjk)}"
        )

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.student_tokenizer)

    max_len = 0
    overlong = []
    for r in ordered:
        rendered = tok.apply_chat_template(
            r["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        pids = tok(rendered, add_special_tokens=False)["input_ids"]
        tids = tok(r["target_translation"], add_special_tokens=False)["input_ids"]
        n = len(pids) + len(tids) + 1
        max_len = max(max_len, n)
        if n > 1024:
            overlong.append({"target_id": r["target_id"], "tokens": n})

    if overlong:
        write_jsonl_atomic(repair_dir / "overlong_gt1024.jsonl", overlong)
        raise SystemExit(f"FINAL_MAX_LENGTH_FAIL n={len(overlong)} max={max_len}")

    chem_path = parent / "chemistry_train5500_sft.jsonl"
    idiom_path = parent / "idiom_train5500_sft.jsonl"
    combined_path = parent / "combined_train11000_sft.jsonl"

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
        "composition_before_zero_cjk_repair": composition,
        "zero_cjk_repair": {
            "repair_id": REPAIR_ID,
            "selected_rows": 86,
            "selection_rule": (
                "Idiom accepted target contains at least one CJK character"
            ),
            "cause": (
                "original validator allowed <=4 CJK characters; final hard "
                "contract requires zero CJK"
            ),
            "only_selected_86_replaced": True,
            "teacher_transform": (
                "replace designated idiom with dictionary definition in "
                "Teacher-only source, then translate to English-only"
            ),
            "student_source_or_prompt_changed": False,
        },
        "student_information_boundary": {
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
                "same per-example exposure; C3 intentionally has about 2x "
                "total examples/updates versus a single-domain arm"
            ),
        },
        "created": now(),
    }
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
            "done": 86,
            "total": 86,
            "percentage": 100.0,
            "updated": now(),
        },
    )

    print()
    print("IDIOM_ZERO_CJK_REPAIR=PASS", flush=True)
    print("REPAIRED_ROWS=86", flush=True)
    print("CHEMISTRY_5500_HARD_CHECK=PASS", flush=True)
    print("IDIOM_5500_ZERO_CJK=PASS", flush=True)
    print("IDIOM_5500_SOURCE_TERM_ABSENT=PASS", flush=True)
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
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--jobs")
    ap.add_argument("--output")
    ap.add_argument("--failure")
    ap.add_argument("--teacher")
    ap.add_argument("--student-tokenizer")
    ap.add_argument("--parent")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.worker:
        for name in ("worker_id", "device", "jobs", "output", "failure", "teacher"):
            if getattr(args, name) is None:
                raise SystemExit(f"MISSING_WORKER_ARG {name}")
        worker(args)
    else:
        for name in ("teacher", "student_tokenizer", "parent"):
            if not getattr(args, name):
                raise SystemExit(f"MISSING_MASTER_ARG {name}")
        master(args)
