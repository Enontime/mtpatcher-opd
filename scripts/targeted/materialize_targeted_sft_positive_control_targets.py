#!/usr/bin/env python3
"""
Materialize frozen positive-control SFT sequence targets for targeted Chemistry/Idiom.

Scientific class:
    LAB ADAPTATION / POSITIVE-CONTROL TARGET SYNTHESIS

Purpose:
    Build high-quality offline sequence targets for the already-frozen Seen/train
    source contexts, before C1/C2/C3 SFT.

Critical separation:
    - Teacher target synthesis MAY use frozen lexical knowledge:
        Chemistry: canonical English term.
        Idiom: dictionary definition.
    - Student training prompt NEVER sees that lexical hint.
      Student receives only the canonical direct-translation prompt + source.
    - UC/UW files are never read by this script.
    - No C0/C1/C2/C3 output or evaluator score is used for target selection.

Teacher:
    /workspace/mtpatcher/models/Qwen3-8B
    greedy, enable_thinking=False

Outputs:
    chemistry_train5500_sft.jsonl
    idiom_train5500_sft.jsonl
    combined_train11000_sft.jsonl
    manifest.json
    progress.json

Engineering:
    - 16 independent NPU workers by default
    - one durable shard per worker
    - each successful row fsyncs immediately
    - failed rows are durable
    - rerun resumes by target_id
    - master progress + ETA every ~5 seconds
    - immutable final artifact once manifest status=FROZEN
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

STUDENT_PROMPT = (
    "Translate the following text into English without additional explanations:"
    "\n\n{source}\n\n"
)
STUDENT_PROMPT_SHA = hashlib.sha256(STUDENT_PROMPT.encode("utf-8")).hexdigest()

CHEM_PROMPTS = [
    """Translate the Chinese sentence into natural English.

The designated chemistry term is:
Chinese term: {src_term}
Canonical English term: {canonical}

When translating that designated term, you MUST use the exact canonical English
term shown above, preserving its spelling. Translate the rest of the sentence
naturally.

Chinese sentence:
{source}

Output only the complete English translation. Do not explain.""",

    """Produce one faithful English translation of the Chinese sentence below.

Mandatory terminology constraint:
"{src_term}" -> "{canonical}"

The final English sentence MUST literally contain the exact string
"{canonical}". Do not translate that term with a synonym or alternate name.

Source:
{source}

Return only the English translation.""",

    """Chinese-to-English translation task.

Use this frozen bilingual chemistry mapping exactly:
SOURCE TERM: {src_term}
TARGET TERM: {canonical}

Translate the whole source sentence. The TARGET TERM must appear verbatim in
your English output.

SOURCE SENTENCE:
{source}

English translation only:""",
]

IDIOM_PROMPTS = [
    """Translate the Chinese sentence into natural English.

A designated Chinese idiom appears in the sentence:
Idiom: {src_term}
Dictionary meaning: {definition}

Render the idiom according to its intended contextual meaning. Do not translate
the idiom word-for-word when that would lose the idiomatic meaning.

Chinese sentence:
{source}

Output only the complete English translation. Do not explain.""",

    """Produce a faithful and natural English translation.

The phrase "{src_term}" is an idiom. Its dictionary meaning is:
{definition}

Use that meaning to translate the idiom appropriately in this sentence. The
English output should express the intended meaning rather than preserve the
Chinese wording.

Source:
{source}

Return only the English translation.""",
]

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def now():
    return datetime.now(TZ8).isoformat(timespec="seconds")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
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


def canonical_chem_target(row):
    candidates = [
        row.get("en_name"),
        row.get("target_term"),
        row.get("tgt_word"),
        row.get("target"),
    ]
    lex = row.get("lexical_record")
    if isinstance(lex, dict):
        candidates.extend([
            lex.get("en_name"),
            lex.get("target_term"),
            lex.get("tgt_word"),
            lex.get("target"),
        ])
    for x in candidates:
        if isinstance(x, str) and x.strip():
            return x.strip()
    return None


def target_id(row):
    return f"{row['domain']}:{row['job_id']}"


def load_input(data_dir):
    # Deliberately read only the train files. UC/UW paths never appear here.
    chem_path = data_dir / "chemistry_train.jsonl"
    idiom_path = data_dir / "idiom_train.jsonl"

    chem = read_jsonl(chem_path)
    idiom = read_jsonl(idiom_path)

    if len(chem) != 5500 or len(idiom) != 5500:
        raise SystemExit(
            f"TRAIN_CARDINALITY_FAIL chemistry={len(chem)} idiom={len(idiom)}"
        )

    rows = []
    for domain, src_rows in (("chemistry", chem), ("idiom", idiom)):
        seen = set()
        for r in src_rows:
            x = dict(r)
            if x.get("split") != "train":
                raise SystemExit(
                    f"NONTRAIN_ROW_READ domain={domain} job_id={x.get('job_id')} "
                    f"split={x.get('split')}"
                )
            x["domain"] = domain
            tid = target_id(x)
            if tid in seen:
                raise SystemExit(f"DUP_TARGET_ID {tid}")
            seen.add(tid)
            x["target_id"] = tid

            if domain == "chemistry":
                canonical = canonical_chem_target(x)
                if not canonical:
                    raise SystemExit(f"MISSING_CHEM_CANONICAL {tid}")
                x["_canonical_en_target"] = canonical
            else:
                definition = x.get("definition")
                if not isinstance(definition, str) or not definition.strip():
                    raise SystemExit(f"MISSING_IDIOM_DEFINITION {tid}")

            rows.append(x)

    if len(rows) != 11000:
        raise SystemExit(f"EXPECTED_11000 got={len(rows)}")

    ids = [r["target_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("GLOBAL_TARGET_ID_DUP")

    return rows, {
        "chemistry_train.jsonl": {
            "rows": len(chem),
            "sha256": sha256_file(chem_path),
        },
        "idiom_train.jsonl": {
            "rows": len(idiom),
            "sha256": sha256_file(idiom_path),
        },
    }


def completed_map(path):
    p = Path(path)
    if not p.exists():
        return {}
    out = {}
    for r in read_jsonl(p):
        if r.get("status") == "PASS":
            out[r["target_id"]] = r
    return out


def valid_translation(domain, row, text):
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
        canonical = row["_canonical_en_target"]
        if canonical.casefold() not in text.casefold():
            return False, "canonical_term_missing"
        return True, "ok"

    # Idiom: reject obvious untranslated source idiom and strongly Chinese output.
    idiom = row["src_term"]
    if idiom and idiom in text:
        return False, "source_idiom_copied"
    cjk = len(CJK_RE.findall(text))
    if cjk > 4:
        return False, f"too_many_cjk_{cjk}"
    return True, "ok"


def worker_main(args):
    import torch
    import torch_npu  # noqa: F401
    from transformers import AutoTokenizer, AutoModelForCausalLM

    rows = read_jsonl(args.jobs)
    assigned = [
        r for i, r in enumerate(rows)
        if i % args.num_workers == args.worker_id
    ]

    output = Path(args.output)
    failure = Path(args.failure)
    completed = completed_map(output)

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
        f"{now()} WORKER_READY worker={args.worker_id} device={device} "
        f"resume={initial}/{total}",
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
                max_new_tokens=384,
                use_cache=True,
            )
        gen = ids[:, batch["input_ids"].shape[1]:]
        return tok.batch_decode(gen, skip_special_tokens=True)[0].strip()

    for r in assigned:
        tid = r["target_id"]
        if tid in completed:
            continue

        domain = r["domain"]
        attempts = []
        success_text = None
        success_variant = None

        prompts = CHEM_PROMPTS if domain == "chemistry" else IDIOM_PROMPTS
        # Cycle the finite frozen prompt variants; deterministic generation each time.
        max_attempts = len(prompts)

        for variant in range(max_attempts):
            if domain == "chemistry":
                prompt = prompts[variant].format(
                    src_term=r["src_term"],
                    canonical=r["_canonical_en_target"],
                    source=r["src_text"],
                )
            else:
                prompt = prompts[variant].format(
                    src_term=r["src_term"],
                    definition=r["definition"],
                    source=r["src_text"],
                )

            try:
                text = infer(prompt)
                ok, reason = valid_translation(domain, r, text)
                attempts.append({
                    "variant": variant,
                    "ok": ok,
                    "reason": reason,
                    "text_preview": text[:240],
                })
                if ok:
                    success_text = text
                    success_variant = variant
                    break
            except Exception as e:
                attempts.append({
                    "variant": variant,
                    "ok": False,
                    "reason": "exception",
                    "error": repr(e),
                })

        if success_text is None:
            append_jsonl(
                failure,
                {
                    "status": "FAIL",
                    "target_id": tid,
                    "domain": domain,
                    "job_id": r["job_id"],
                    "src_term": r["src_term"],
                    "attempts": attempts,
                    "timestamp": now(),
                },
            )
            print(
                f"{now()} TARGET_FAIL worker={args.worker_id} target_id={tid} "
                f"attempts={len(attempts)}",
                flush=True,
            )
            # Continue: one bad row must not erase successful work.
            continue

        # The Student prompt deliberately excludes lexical hints.
        student_user = STUDENT_PROMPT.format(source=r["src_text"])

        rec = {
            "status": "PASS",
            "target_id": tid,
            "domain": domain,
            "job_id": r["job_id"],
            "entity_key": r.get("entity_key"),
            "src_term": r["src_term"],
            "definition": r.get("definition"),
            "source": r["src_text"],
            "reference": success_text,
            "target_translation": success_text,
            "messages": [
                {"role": "user", "content": student_user}
            ],
            "student_prompt_sha256": STUDENT_PROMPT_SHA,
            "teacher_model": args.teacher,
            "teacher_enable_thinking": False,
            "teacher_do_sample": False,
            "teacher_max_new_tokens": 384,
            "teacher_prompt_family": (
                "chemistry_canonical_term_guided_v1"
                if domain == "chemistry"
                else "idiom_definition_guided_v1"
            ),
            "teacher_prompt_variant": success_variant,
            "canonical_en_target": (
                r.get("_canonical_en_target") if domain == "chemistry" else None
            ),
            "validation": {
                "chemistry_canonical_term_present": (
                    r["_canonical_en_target"].casefold() in success_text.casefold()
                    if domain == "chemistry" else None
                ),
                "idiom_source_string_absent": (
                    r["src_term"] not in success_text
                    if domain == "idiom" else None
                ),
            },
            "timestamp": now(),
        }

        append_jsonl(output, rec)
        done += 1

        fresh = done - initial
        elapsed = max(time.time() - started, 1e-9)
        rate = fresh / elapsed if fresh else 0.0
        eta = (total - done) / rate if rate else None

        print(
            f"{now()} worker={args.worker_id} done={done}/{total} "
            f"pct={100*done/total:.1f}% rate={rate:.3f}/s "
            f"ETA={'?' if eta is None else f'{eta/60:.1f}m'} "
            f"target_id={tid}",
            flush=True,
        )

    print(
        f"{now()} WORKER_FINISHED worker={args.worker_id} "
        f"durable_pass={len(completed_map(output))}/{total}",
        flush=True,
    )


def count_pass(path):
    p = Path(path)
    if not p.exists():
        return 0
    return sum(1 for r in read_jsonl(p) if r.get("status") == "PASS")


def main_master(args):
    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frozen_manifest = out / "manifest.json"
    if frozen_manifest.exists():
        old = json.loads(frozen_manifest.read_text(encoding="utf-8"))
        if old.get("status") == "FROZEN":
            print("ARTIFACT_ALREADY_FROZEN", flush=True)
            print(json.dumps(old, ensure_ascii=False, indent=2), flush=True)
            return

    rows, source_files = load_input(data)

    # Durable frozen job snapshot, independent of later source-file mutation.
    jobs = out / "jobs11000.jsonl"
    if jobs.exists():
        existing = read_jsonl(jobs)
        if existing != rows:
            raise SystemExit("IMMUTABLE_JOB_SNAPSHOT_MISMATCH")
    else:
        write_jsonl_atomic(jobs, rows)

    shard_dir = out / "worker_shards"
    fail_dir = out / "worker_failures"
    log_dir = out / "worker_logs"
    shard_dir.mkdir(exist_ok=True)
    fail_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    procs = []
    handles = []

    for wid in range(args.num_workers):
        shard = shard_dir / f"part_{wid:02d}.jsonl"
        failure = fail_dir / f"part_{wid:02d}.jsonl"
        log = log_dir / f"worker_{wid:02d}.log"
        fh = open(log, "a", encoding="utf-8")
        handles.append(fh)

        cmd = [
            sys.executable, "-u", str(Path(__file__).resolve()),
            "--worker",
            "--worker-id", str(wid),
            "--device", str(wid),
            "--num-workers", str(args.num_workers),
            "--jobs", str(jobs),
            "--output", str(shard),
            "--failure", str(failure),
            "--teacher", args.teacher,
        ]
        p = subprocess.Popen(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        procs.append((wid, p, shard, failure, log))

    started = time.time()
    last = None

    while True:
        done = sum(count_pass(shard) for _, _, shard, _, _ in procs)
        running = sum(p.poll() is None for _, p, _, _, _ in procs)
        elapsed = max(time.time() - started, 1e-9)
        rate = done / elapsed if done else 0.0
        eta = (11000 - done) / rate if rate else None

        state = {
            "status": "RUNNING" if running else "FINALIZING",
            "done": done,
            "total": 11000,
            "percentage": round(100 * done / 11000, 2),
            "workers_running": running,
            "workers_total": args.num_workers,
            "elapsed_seconds": round(elapsed, 1),
            "rate_rows_per_sec": round(rate, 3),
            "eta_seconds": round(eta, 1) if eta is not None else None,
            "updated": now(),
        }
        atomic_json(out / "progress.json", state)

        key = (done, running)
        if key != last:
            print(
                f"{now()} phase=teacher_target_synthesis "
                f"done={done}/11000 pct={100*done/11000:.1f}% "
                f"workers={running}/{args.num_workers} "
                f"rate={rate:.2f}/s "
                f"ETA={'?' if eta is None else f'{eta/60:.1f}m'}",
                flush=True,
            )
            last = key

        if running == 0:
            break
        time.sleep(5)

    for fh in handles:
        fh.close()

    bad_workers = [
        {"worker": wid, "rc": p.returncode, "log": str(log)}
        for wid, p, _, _, log in procs
        if p.returncode != 0
    ]
    if bad_workers:
        atomic_json(out / "worker_process_failures.json", bad_workers)
        raise SystemExit(f"WORKER_PROCESS_FAIL {bad_workers}")

    # Merge durable successes by identity.
    merged = {}
    for _, _, shard, _, _ in procs:
        if not shard.exists():
            continue
        for r in read_jsonl(shard):
            if r.get("status") != "PASS":
                continue
            tid = r["target_id"]
            if tid in merged:
                raise SystemExit(f"DUP_SUCCESS_TARGET_ID {tid}")
            merged[tid] = r

    expected_ids = [r["target_id"] for r in rows]
    missing = [tid for tid in expected_ids if tid not in merged]

    # Aggregate failure records for inspection, but do not use them to rewrite jobs.
    failure_rows = []
    for _, _, _, fp, _ in procs:
        if fp.exists():
            failure_rows.extend(read_jsonl(fp))
    if failure_rows:
        write_jsonl_atomic(out / "failures_all.jsonl", failure_rows)

    if missing:
        atomic_json(
            out / "progress.json",
            {
                "status": "FAIL",
                "done": len(merged),
                "total": 11000,
                "missing": len(missing),
                "missing_examples": missing[:30],
                "updated": now(),
            },
        )
        print(
            f"TARGET_SYNTHESIS_INCOMPLETE pass={len(merged)}/11000 "
            f"missing={len(missing)}",
            flush=True,
        )
        print("RERUN_SAME_COMMAND_TO_RETRY_ONLY_MISSING_ROWS", flush=True)
        raise SystemExit(2)

    ordered = [merged[tid] for tid in expected_ids]
    chem = [r for r in ordered if r["domain"] == "chemistry"]
    idiom = [r for r in ordered if r["domain"] == "idiom"]

    if len(chem) != 5500 or len(idiom) != 5500:
        raise SystemExit(
            f"MERGED_CARDINALITY_FAIL chemistry={len(chem)} idiom={len(idiom)}"
        )

    # Hard Chemistry target invariant.
    chem_bad = [
        r["target_id"] for r in chem
        if not r["canonical_en_target"]
        or r["canonical_en_target"].casefold()
        not in r["target_translation"].casefold()
    ]
    if chem_bad:
        raise SystemExit(f"CHEM_TARGET_INVARIANT_FAIL n={len(chem_bad)}")

    # Student-tokenization preflight: target must fit canonical max_length=1024.
    # Use C0 tokenizer because C0/C1/C2/C3 share tokenizer identity.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.student_tokenizer)

    overlong = []
    max_len = 0
    for r in ordered:
        rendered_prompt = tok.apply_chat_template(
            r["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = tok(rendered_prompt, add_special_tokens=False)["input_ids"]
        target_ids = tok(
            r["target_translation"],
            add_special_tokens=False,
        )["input_ids"]
        # Match response-only training intent: prompt + supervised target + eos.
        n = len(prompt_ids) + len(target_ids) + 1
        max_len = max(max_len, n)
        if n > 1024:
            overlong.append({"target_id": r["target_id"], "tokens": n})

    if overlong:
        write_jsonl_atomic(out / "overlong_gt1024.jsonl", overlong)
        raise SystemExit(f"MAX_LENGTH_FAIL n={len(overlong)} max={max_len}")

    chem_path = out / "chemistry_train5500_sft.jsonl"
    idiom_path = out / "idiom_train5500_sft.jsonl"
    combined_path = out / "combined_train11000_sft.jsonl"

    write_jsonl_atomic(chem_path, chem)
    write_jsonl_atomic(idiom_path, idiom)
    write_jsonl_atomic(combined_path, ordered)

    # Fixed, outcome-independent 100-row Idiom target-quality sample.
    idiom_qc100 = sorted(
        idiom,
        key=lambda r: hashlib.sha256(
            ("idiom-target-qc100-v1|" + r["target_id"]).encode("utf-8")
        ).hexdigest(),
    )[:100]
    qc_path = out / "idiom_teacher_target_qc100.jsonl"
    write_jsonl_atomic(qc_path, idiom_qc100)

    manifest = {
        "status": "FROZEN",
        "artifact_id": ARTIFACT_ID,
        "scientific_class": "LAB ADAPTATION / POSITIVE-CONTROL TARGET SYNTHESIS",
        "question": (
            "Can C0 learn frozen Seen Chemistry/Idiom knowledge when supplied "
            "with explicit high-quality offline sequence supervision?"
        ),
        "target_synthesis_contract": {
            "teacher_model": args.teacher,
            "teacher_enable_thinking": False,
            "teacher_do_sample": False,
            "teacher_max_new_tokens": 384,
            "chemistry_teacher_hint": (
                "canonical English term supplied only to Teacher target synthesis"
            ),
            "idiom_teacher_hint": (
                "dictionary definition supplied only to Teacher target synthesis"
            ),
            "student_receives_lexical_hint": False,
            "student_prompt": STUDENT_PROMPT,
            "student_prompt_sha256": STUDENT_PROMPT_SHA,
        },
        "data_access_contract": {
            "read_train_only": True,
            "uc_access": False,
            "uw_access": False,
            "student_output_used": False,
            "evaluator_score_used": False,
            "downstream_result_used": False,
        },
        "source_files": source_files,
        "outputs": {
            chem_path.name: {"rows": 5500, "sha256": sha256_file(chem_path)},
            idiom_path.name: {"rows": 5500, "sha256": sha256_file(idiom_path)},
            combined_path.name: {"rows": 11000, "sha256": sha256_file(combined_path)},
            qc_path.name: {"rows": 100, "sha256": sha256_file(qc_path)},
        },
        "hard_checks": {
            "chemistry_all_targets_contain_canonical_term": True,
            "student_rendered_max_tokens": max_len,
            "all_student_examples_le_1024": True,
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
        "note": (
            "This target synthesis is a positive-control adaptation. It is not "
            "claimed to reproduce the paper's exact GPT-4/PDS target synthesis."
        ),
        "created": now(),
    }
    atomic_json(frozen_manifest, manifest)
    atomic_json(
        out / "progress.json",
        {
            "status": "PASS",
            "done": 11000,
            "total": 11000,
            "percentage": 100.0,
            "updated": now(),
        },
    )

    print()
    print("TARGETED_SFT_TARGETS=FROZEN", flush=True)
    print(f"CHEMISTRY_TARGETS=5500 sha={manifest['outputs'][chem_path.name]['sha256']}", flush=True)
    print(f"IDIOM_TARGETS=5500 sha={manifest['outputs'][idiom_path.name]['sha256']}", flush=True)
    print(f"COMBINED_TARGETS=11000 sha={manifest['outputs'][combined_path.name]['sha256']}", flush=True)
    print(f"IDIOM_QC100={qc_path}", flush=True)
    print(f"STUDENT_RENDERED_MAX_TOKENS={max_len}", flush=True)
    print("FINAL_RESULT=TARGETED_SFT_POSITIVE_CONTROL_TARGETS_FROZEN", flush=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-id", type=int)
    ap.add_argument("--device", type=int)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--jobs")
    ap.add_argument("--output")
    ap.add_argument("--failure")
    ap.add_argument("--teacher")
    ap.add_argument("--student-tokenizer")
    ap.add_argument("--data")
    ap.add_argument("--out")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.worker:
        for name in ("worker_id", "device", "jobs", "output", "failure", "teacher"):
            if getattr(args, name) is None:
                raise SystemExit(f"MISSING_WORKER_ARG {name}")
        worker_main(args)
    else:
        for name in ("teacher", "student_tokenizer", "data", "out"):
            if not getattr(args, name):
                raise SystemExit(f"MISSING_MASTER_ARG {name}")
        main_master(args)
