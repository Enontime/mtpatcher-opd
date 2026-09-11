#!/usr/bin/env python3
"""
Final repair of the 4 residual rows left after residual49_repair_v1, then freeze
the full 11k targeted SFT positive-control target artifact.

Scientific class:
    LAB ADAPTATION / LOCAL RESIDUAL REPAIR

Parent state expected:
    original Teacher synthesis successes: 10951
    residual49_repair_v1 successes:        45
    remaining:                               4

Key design:
- Never regenerate the 10,951 original successes.
- Never regenerate the 45 successful residual49 repairs.
- Identify the remaining 4 by stable target_id set difference.
- Repair only those 4 with a stronger semantics-preserving source transform:
    Chemistry:
      Replace the designated Chinese chemistry term in the Teacher-only source
      with the exact canonical English registry term, then translate while
      preserving that embedded English string verbatim.
    Idiom:
      Replace the designated idiom in the Teacher-only source with its Chinese
      dictionary definition, then translate the resulting semantic paraphrase
      into English. This removes the stubborn idiom surface form from the
      Teacher input.
- Student still sees the original frozen Chinese source, not the transformed
  Teacher-only source and not any lexical hint.
- Durable per-row writes, resume-safe, explicit progress.
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
REPAIR_ID = "targeted-sft-positive-control-residual4-final-repair-v1-20260910"

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

    if job["src_term"] in text:
        return False, "source_idiom_copied"
    cjk = len(CJK_RE.findall(text))
    if cjk:
        return False, f"cjk_remaining_{cjk}"
    return True, "ok"


def transform_source(job):
    src = job["src_text"]
    term = job["src_term"]

    if term not in src:
        raise RuntimeError(
            f"designated term not found in frozen source: {job['target_id']}"
        )

    if job["domain"] == "chemistry":
        replacement = job["_canonical_en_target"]
    else:
        replacement = job["definition"].strip()

    return src.replace(term, replacement, 1)


def prompts_for(job):
    transformed = transform_source(job)

    if job["domain"] == "chemistry":
        canonical = job["_canonical_en_target"]
        return [
f"""Translate the mixed Chinese-English source below into one natural English sentence.

One chemistry entity has already been replaced by its exact frozen English
registry name. Preserve that embedded English registry name EXACTLY, including
all punctuation, spacing, capitalization, abbreviations, and pluralization.

Exact frozen registry name:
<<<{canonical}>>>

Mixed source:
{transformed}

Output only the complete English translation. The exact frozen registry name
must occur verbatim in your answer.""",

f"""Convert the following mixed-language sentence to fluent English.

DO NOT EDIT this exact substring:
{canonical}

It is already the correct English translation of the designated chemistry
entity. Translate everything else around it while copying that substring
character-for-character.

Source:
{transformed}

Return only the final English sentence.""",

f"""Required exact substring:
{canonical}

Write one faithful English translation of the source below. Your output MUST
contain the required exact substring character-for-character.

Source:
{transformed}

If needed, start the sentence with the required substring and then express the
rest of the source naturally. English translation only."""
        ]

    definition = job["definition"].strip()
    return [
f"""Translate the following Chinese sentence into fluent English.

The original idiom has already been replaced by its dictionary meaning so that
you should translate the semantic content directly.

Source with semantic replacement:
{transformed}

Hard output constraints:
- English only;
- zero Chinese characters;
- no commentary about idioms or translation;
- preserve the full sentence meaning.

Output only one complete English translation.""",

f"""Rewrite this semantic paraphrase as one natural English sentence:

{transformed}

The inserted Chinese phrase is the dictionary meaning of the original idiom:
{definition}

Translate all content into English. Do not leave any Chinese characters.
Return only the final English sentence.""",

f"""English-only translation task.

Source:
{transformed}

Translate every Chinese part into English and preserve the intended meaning.
Your final answer must contain zero Chinese characters and no meta-commentary.

Final English sentence only:"""
    ]


def completed_map(path):
    p = Path(path)
    if not p.exists():
        return {}
    out = {}
    for row in read_jsonl(p):
        if row.get("status") == "PASS":
            out[row["target_id"]] = row
    return out


def worker(args):
    import torch
    import torch_npu  # noqa: F401
    from transformers import AutoTokenizer, AutoModelForCausalLM

    jobs = read_jsonl(args.jobs)
    assigned = [
        row for i, row in enumerate(jobs)
        if i % args.num_workers == args.worker_id
    ]
    completed = completed_map(args.output)

    device = f"npu:{args.device}"
    torch.npu.set_device(device)

    tok = AutoTokenizer.from_pretrained(args.teacher)
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    initial = sum(r["target_id"] in completed for r in assigned)
    done = initial
    total = len(assigned)
    started = time.time()

    print(
        f"{now()} FINAL4_WORKER_READY worker={args.worker_id} "
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
        if tid in completed:
            continue

        transformed = transform_source(job)
        attempts = []
        accepted = None
        accepted_variant = None

        for variant, prompt in enumerate(prompts_for(job)):
            try:
                text = infer(prompt)
                ok, reason = valid(job["domain"], job, text)
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
                    "domain": job["domain"],
                    "job_id": job["job_id"],
                    "src_term": job["src_term"],
                    "teacher_only_transformed_source": transformed,
                    "attempts": attempts,
                    "timestamp": now(),
                },
            )
            print(
                f"{now()} FINAL4_ROW_FAIL worker={args.worker_id} target_id={tid}",
                flush=True,
            )
            continue

        rec = {
            "status": "PASS",
            "repair_id": REPAIR_ID,
            "repair_class": "LOCAL_RESIDUAL_REPAIR",
            "target_id": tid,
            "domain": job["domain"],
            "job_id": job["job_id"],
            "entity_key": job.get("entity_key"),
            "src_term": job["src_term"],
            "definition": job.get("definition"),
            "source": job["src_text"],
            "reference": accepted,
            "target_translation": accepted,
            "messages": [
                {
                    "role": "user",
                    "content": STUDENT_PROMPT.format(source=job["src_text"]),
                }
            ],
            "student_prompt_sha256": STUDENT_PROMPT_SHA,
            "teacher_model": args.teacher,
            "teacher_enable_thinking": False,
            "teacher_do_sample": False,
            "teacher_max_new_tokens": 512,
            "teacher_prompt_family": (
                "chemistry_teacher_source_term_replaced_by_canonical_v1"
                if job["domain"] == "chemistry"
                else "idiom_teacher_source_term_replaced_by_definition_v1"
            ),
            "teacher_prompt_variant": accepted_variant,
            "teacher_only_transformed_source": transformed,
            "canonical_en_target": (
                job.get("_canonical_en_target")
                if job["domain"] == "chemistry"
                else None
            ),
            "validation": {
                "chemistry_canonical_term_present": (
                    job["_canonical_en_target"].casefold() in accepted.casefold()
                    if job["domain"] == "chemistry"
                    else None
                ),
                "idiom_source_string_absent": (
                    job["src_term"] not in accepted
                    if job["domain"] == "idiom"
                    else None
                ),
                "idiom_cjk_count": (
                    len(CJK_RE.findall(accepted))
                    if job["domain"] == "idiom"
                    else None
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
            f"pct={100*done/total:.1f}% "
            f"ETA={'?' if eta is None else f'{eta:.1f}s'} target_id={tid}",
            flush=True,
        )

    print(
        f"{now()} FINAL4_WORKER_FINISHED worker={args.worker_id}",
        flush=True,
    )


def master(args):
    parent = Path(args.parent)
    r49 = parent / "residual49_repair_v1"
    r4 = parent / "residual4_final_repair_v1"
    r4.mkdir(parents=True, exist_ok=True)

    manifest_path = parent / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("status") == "FROZEN":
            print("ARTIFACT_ALREADY_FROZEN", flush=True)
            print(json.dumps(old, ensure_ascii=False, indent=2), flush=True)
            return

    jobs = read_jsonl(parent / "jobs11000.jsonl")
    jobs_by_id = {r["target_id"]: r for r in jobs}
    if len(jobs_by_id) != 11000:
        raise SystemExit(f"JOBS_CONTRACT_FAIL unique={len(jobs_by_id)}")

    original = load_pass_dir(parent / "worker_shards")
    repair49 = load_pass_dir(r49 / "worker_shards")

    if len(original) != 10951:
        raise SystemExit(f"ORIGINAL_PASS_CONTRACT_FAIL got={len(original)}")
    if len(repair49) != 45:
        raise SystemExit(f"REPAIR49_PASS_CONTRACT_FAIL got={len(repair49)}")

    overlap = set(original) & set(repair49)
    if overlap:
        raise SystemExit(f"ORIGINAL_REPAIR49_OVERLAP n={len(overlap)}")

    solved = set(original) | set(repair49)
    missing = [tid for tid in jobs_by_id if tid not in solved]

    if len(missing) != 4:
        raise SystemExit(
            f"FINAL4_IDENTITY_CONTRACT_FAIL missing={len(missing)} ids={missing}"
        )

    domains = Counter(jobs_by_id[tid]["domain"] for tid in missing)
    print(
        f"{now()} FINAL4_SELECTION=PASS ids={missing} domains={dict(domains)}",
        flush=True,
    )

    repair_jobs = [jobs_by_id[tid] for tid in missing]
    jobs_path = r4 / "repair_jobs4.jsonl"
    if jobs_path.exists():
        if read_jsonl(jobs_path) != repair_jobs:
            raise SystemExit("IMMUTABLE_FINAL4_JOBS_MISMATCH")
    else:
        write_jsonl_atomic(jobs_path, repair_jobs)

    shard_dir = r4 / "worker_shards"
    fail_dir = r4 / "worker_failures"
    log_dir = r4 / "worker_logs"
    shard_dir.mkdir(exist_ok=True)
    fail_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    procs = []
    handles = []

    workers = min(args.num_workers, 4)
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
        final4 = load_pass_dir(shard_dir)
        done = len(final4)
        running = sum(p.poll() is None for _, p, _, _, _ in procs)
        elapsed = max(time.time() - started, 1e-9)
        rate = done / elapsed if done else 0.0
        eta = (4 - done) / rate if rate else None

        atomic_json(
            r4 / "progress.json",
            {
                "status": "RUNNING" if running else "FINALIZING",
                "done": done,
                "total": 4,
                "percentage": round(25 * done, 1),
                "workers_running": running,
                "elapsed_seconds": round(elapsed, 1),
                "eta_seconds": round(eta, 1) if eta is not None else None,
                "updated": now(),
            },
        )

        state = (done, running)
        if state != last:
            print(
                f"{now()} phase=residual4_final_repair done={done}/4 "
                f"pct={25*done:.1f}% workers={running}/{workers} "
                f"ETA={'?' if eta is None else f'{eta:.1f}s'}",
                flush=True,
            )
            last = state

        if running == 0:
            break
        time.sleep(1)

    for fh in handles:
        fh.close()

    bad_workers = [
        {"worker": wid, "rc": p.returncode, "log": str(logp)}
        for wid, p, _, _, logp in procs
        if p.returncode != 0
    ]
    if bad_workers:
        atomic_json(r4 / "worker_process_failures.json", bad_workers)
        raise SystemExit(f"FINAL4_WORKER_PROCESS_FAIL {bad_workers}")

    final4 = load_pass_dir(shard_dir)
    if len(final4) != 4 or set(final4) != set(missing):
        left = sorted(set(missing) - set(final4))
        print(
            f"FINAL4_REPAIR_INCOMPLETE pass={len(final4)}/4 missing={left}",
            flush=True,
        )
        raise SystemExit(2)

    final = {}
    final.update(original)
    final.update(repair49)
    final.update(final4)

    if len(final) != 11000:
        raise SystemExit(f"FINAL_COMPOSE_FAIL got={len(final)}")

    ordered = [final[r["target_id"]] for r in jobs]
    chem = [r for r in ordered if r["domain"] == "chemistry"]
    idiom = [r for r in ordered if r["domain"] == "idiom"]

    if len(chem) != 5500 or len(idiom) != 5500:
        raise SystemExit(
            f"FINAL_DOMAIN_COUNT_FAIL chemistry={len(chem)} idiom={len(idiom)}"
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
        write_jsonl_atomic(r4 / "overlong_gt1024.jsonl", overlong)
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
        "composition": {
            "original_qwen3_8b_pass": 10951,
            "residual49_repair_v1_pass": 45,
            "residual4_final_repair_v1_pass": 4,
            "total": 11000,
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
        "repair4_contract": {
            "repair_id": REPAIR_ID,
            "original_10951_preserved": True,
            "repair49_success_45_preserved": True,
            "only_final_missing_4_repaired": True,
            "chemistry_teacher_transform": (
                "replace designated Chinese term by exact canonical English "
                "term in Teacher-only source"
            ),
            "idiom_teacher_transform": (
                "replace designated idiom by dictionary definition in "
                "Teacher-only source"
            ),
            "student_source_transformed": False,
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
        r4 / "progress.json",
        {
            "status": "PASS",
            "done": 4,
            "total": 4,
            "percentage": 100.0,
            "updated": now(),
        },
    )

    print()
    print("FINAL4_REPAIR=PASS", flush=True)
    print("ORIGINAL_10951_PRESERVED=PASS", flush=True)
    print("RESIDUAL49_SUCCESS45_PRESERVED=PASS", flush=True)
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
