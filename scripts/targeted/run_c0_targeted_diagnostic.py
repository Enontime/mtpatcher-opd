#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

TZ8 = timezone(timedelta(hours=8))
SAMPLE_TAG = "targeted-diagnostic1000-v1-20260910"
PROMPT = (
    "Translate the following text into English without additional explanations:"
    "\n\n{source}\n\n"
)
EXPECTED_PROMPT_SHA256 = "63101d0739ff2ee11b0e5d548b85dcd18be7a241777893326140f12778d9a8b8"


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


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


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


def stable_rank(row, domain, split):
    s = (
        f"{SAMPLE_TAG}|{domain}|{split}|"
        f"{row.get('entity_key')}|{row.get('job_id')}"
    )
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def canonical_target(row):
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


def eval_id(domain, split, row):
    return f"{domain}:{split}:{row['job_id']}"


def freeze_diagnostic(data_dir, diag_dir):
    diag_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    combined_rows = []

    for domain in ("chemistry", "idiom"):
        uc = read_jsonl(data_dir / f"{domain}_uc.jsonl")
        uw = read_jsonl(data_dir / f"{domain}_uw.jsonl")
        if len(uc) != 5500 or len(uw) != 500:
            raise SystemExit(
                f"INPUT_CARDINALITY_FAIL domain={domain} uc={len(uc)} uw={len(uw)}"
            )

        uc500 = sorted(uc, key=lambda r: stable_rank(r, domain, "uc"))[:500]
        uw500 = sorted(
            uw,
            key=lambda r: (str(r.get("entity_key", "")), int(r.get("job_id", -1))),
        )

        domain_rows = []
        for split, subset in (("uc", uc500), ("uw", uw500)):
            out_subset = []
            for r in subset:
                x = dict(r)
                x["eval_domain"] = domain
                x["eval_split"] = split
                x["eval_id"] = eval_id(domain, split, r)
                if domain == "chemistry":
                    x["canonical_en_target"] = canonical_target(r)
                    if not x["canonical_en_target"]:
                        raise SystemExit(f"MISSING_CHEM_TARGET {x['eval_id']}")
                out_subset.append(x)
                domain_rows.append(x)
                combined_rows.append(x)

            p = diag_dir / f"{domain}_{split}500.jsonl"
            if p.exists():
                if read_jsonl(p) != out_subset:
                    raise SystemExit(f"IMMUTABLE_DIAGNOSTIC_MISMATCH {p}")
            else:
                write_jsonl_atomic(p, out_subset)
            files[p.name] = {"rows": len(out_subset), "sha256": sha256_file(p)}

        merged = diag_dir / f"{domain}_diagnostic1000.jsonl"
        if merged.exists():
            if read_jsonl(merged) != domain_rows:
                raise SystemExit(f"IMMUTABLE_DIAGNOSTIC_MISMATCH {merged}")
        else:
            write_jsonl_atomic(merged, domain_rows)
        files[merged.name] = {"rows": len(domain_rows), "sha256": sha256_file(merged)}

    combined = diag_dir / "targeted_diagnostic2000.jsonl"
    if combined.exists():
        if read_jsonl(combined) != combined_rows:
            raise SystemExit(f"IMMUTABLE_DIAGNOSTIC_MISMATCH {combined}")
    else:
        write_jsonl_atomic(combined, combined_rows)
    files[combined.name] = {"rows": len(combined_rows), "sha256": sha256_file(combined)}

    manifest = {
        "status": "FROZEN",
        "scientific_class": "DIAGNOSTIC ONLY / LAB ADAPTATION",
        "artifact_id": SAMPLE_TAG,
        "source_artifact": str(data_dir),
        "selection": {
            "chemistry_uc": "deterministic SHA rank first 500",
            "chemistry_uw": "all frozen 500",
            "idiom_uc": "deterministic SHA rank first 500",
            "idiom_uw": "all frozen 500",
            "student_output_used": False,
            "evaluator_score_used": False,
            "downstream_outcome_used": False,
        },
        "reuse_contract": "Reuse byte-identically for C0/C1/C2/C3.",
        "files": files,
    }

    mp = diag_dir / "manifest.json"
    if mp.exists():
        old = json.loads(mp.read_text(encoding="utf-8"))
        if old != manifest:
            raise SystemExit("IMMUTABLE_DIAGNOSTIC_MANIFEST_MISMATCH")
    else:
        atomic_json(mp, manifest)

    print(
        f"{now()} DIAGNOSTIC_FREEZE=PASS rows={len(combined_rows)} "
        f"sha256={files[combined.name]['sha256']}",
        flush=True,
    )
    return combined, manifest


def completed_map(path):
    if not Path(path).exists():
        return {}
    out = {}
    for r in read_jsonl(path):
        out[r["eval_id"]] = r
    return out


def worker_main(args):
    import torch
    import torch_npu  # noqa: F401
    from transformers import AutoTokenizer, AutoModelForCausalLM

    rows = read_jsonl(args.input)
    assigned = [
        r for i, r in enumerate(rows)
        if i % args.num_workers == args.worker_id
    ]
    done_map = completed_map(args.output)

    device = f"npu:{args.device}"
    torch.npu.set_device(device)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    total = len(assigned)
    initial = sum(r["eval_id"] in done_map for r in assigned)
    done = initial
    started = time.time()

    print(
        f"{now()} WORKER_READY worker={args.worker_id} device={args.device} "
        f"resume={done}/{total}",
        flush=True,
    )

    for r in assigned:
        if r["eval_id"] in done_map:
            continue

        user_text = PROMPT.format(source=r["src_text"])
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": user_text}],
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
        hyp = tok.batch_decode(gen, skip_special_tokens=True)[0].strip()

        rec = {
            "eval_id": r["eval_id"],
            "eval_domain": r["eval_domain"],
            "eval_split": r["eval_split"],
            "job_id": r["job_id"],
            "entity_key": r.get("entity_key"),
            "src_term": r["src_term"],
            "definition": r.get("definition"),
            "src_text": r["src_text"],
            "canonical_en_target": r.get("canonical_en_target"),
            "translation": hyp,
            "model_path": args.model,
            "prompt_sha256": EXPECTED_PROMPT_SHA256,
            "enable_thinking": False,
            "do_sample": False,
            "max_new_tokens": 512,
            "worker_id": args.worker_id,
            "device": args.device,
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
            f"eval_id={r['eval_id']}",
            flush=True,
        )

    print(f"{now()} WORKER_PASS worker={args.worker_id}", flush=True)


def count_lines(path):
    p = Path(path)
    if not p.exists():
        return 0
    with open(p, "rb") as f:
        return sum(1 for line in f if line.strip())


def master_generate(args, input_path, diag_manifest):
    out = Path(args.out)
    shard_dir = out / "generation_shards"
    log_dir = out / "worker_logs"
    shard_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    source_rows = read_jsonl(input_path)
    if len(source_rows) != 2000:
        raise SystemExit(f"EXPECTED_2000 got={len(source_rows)}")

    if hashlib.sha256(PROMPT.encode("utf-8")).hexdigest() != EXPECTED_PROMPT_SHA256:
        raise SystemExit("PROMPT_SHA_CONTRACT_FAIL")

    procs = []
    handles = []

    for wid in range(args.num_workers):
        shard = shard_dir / f"part_{wid:02d}.jsonl"
        log = log_dir / f"worker_{wid:02d}.log"
        fh = open(log, "a", encoding="utf-8")
        handles.append(fh)

        cmd = [
            sys.executable, "-u", str(Path(__file__).resolve()),
            "--worker",
            "--worker-id", str(wid),
            "--device", str(wid),
            "--num-workers", str(args.num_workers),
            "--input", str(input_path),
            "--output", str(shard),
            "--model", args.model,
        ]
        p = subprocess.Popen(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        procs.append((wid, p, shard, log))

    started = time.time()
    last_report = None

    while True:
        done = sum(count_lines(shard) for _, _, shard, _ in procs)
        running = sum(p.poll() is None for _, p, _, _ in procs)
        elapsed = max(time.time() - started, 1e-9)
        rate = done / elapsed if done else 0.0
        eta = (2000 - done) / rate if rate else None

        atomic_json(
            out / "progress.json",
            {
                "status": "RUNNING" if running else "FINALIZING",
                "done": done,
                "total": 2000,
                "percentage": round(done / 20, 2),
                "workers_running": running,
                "workers_total": args.num_workers,
                "elapsed_seconds": round(elapsed, 1),
                "rate_rows_per_sec": round(rate, 3),
                "eta_seconds": round(eta, 1) if eta is not None else None,
                "updated": now(),
            },
        )

        key = (done, running)
        if key != last_report:
            print(
                f"{now()} phase=c0_generate done={done}/2000 "
                f"pct={done/20:.1f}% workers={running}/{args.num_workers} "
                f"rate={rate:.2f}/s "
                f"ETA={'?' if eta is None else f'{eta/60:.1f}m'}",
                flush=True,
            )
            last_report = key

        if running == 0:
            break
        time.sleep(5)

    for fh in handles:
        fh.close()

    bad = [(wid, p.returncode, str(log)) for wid, p, _, log in procs if p.returncode != 0]
    if bad:
        atomic_json(out / "worker_failures.json", bad)
        raise SystemExit(f"WORKER_FAILURES {bad}")

    merged = {}
    for _, _, shard, _ in procs:
        for r in read_jsonl(shard):
            if r["eval_id"] in merged:
                raise SystemExit(f"DUPLICATE_GENERATION {r['eval_id']}")
            merged[r["eval_id"]] = r

    source_ids = [r["eval_id"] for r in source_rows]
    if set(merged) != set(source_ids):
        missing = sorted(set(source_ids) - set(merged))[:20]
        extra = sorted(set(merged) - set(source_ids))[:20]
        raise SystemExit(f"GENERATION_IDENTITY_FAIL missing={missing} extra={extra}")

    rows = [merged[eid] for eid in source_ids]
    merged_path = out / "c0_targeted_diagnostic2000_translations.jsonl"
    write_jsonl_atomic(merged_path, rows)

    chem_scored = []
    for r in rows:
        if r["eval_domain"] != "chemistry":
            continue
        target = r["canonical_en_target"]
        hit = target.casefold() in r["translation"].casefold()
        x = dict(r)
        x["canonical_target_hit"] = hit
        chem_scored.append(x)

    chem_path = out / "c0_chemistry_diagnostic1000_scored.jsonl"
    write_jsonl_atomic(chem_path, chem_scored)

    def chem_stats(split=None):
        xs = chem_scored if split is None else [
            r for r in chem_scored if r["eval_split"] == split
        ]
        hits = sum(bool(r["canonical_target_hit"]) for r in xs)
        return {"n": len(xs), "hits": hits, "accuracy": hits / len(xs)}

    chem_summary = {
        "metric": "case-insensitive canonical English target substring accuracy",
        "scientific_role": "primary Chemistry targeted diagnostic metric",
        "overall": chem_stats(),
        "uc500": chem_stats("uc"),
        "uw500": chem_stats("uw"),
    }
    atomic_json(out / "c0_chemistry_summary.json", chem_summary)

    idiom = []
    for r in rows:
        if r["eval_domain"] != "idiom":
            continue
        if not r.get("definition"):
            raise SystemExit(f"IDIOM_MISSING_DEFINITION {r['eval_id']}")
        idiom.append({
            "eval_id": r["eval_id"],
            "job_id": r["job_id"],
            "split": r["eval_split"],
            "src_term": r["src_term"],
            "definition": r["definition"],
            "src_text": r["src_text"],
            "model_translation": r["translation"],
        })

    if Counter(r["split"] for r in idiom) != Counter({"uc": 500, "uw": 500}):
        raise SystemExit("IDIOM_SPLIT_CONTRACT_FAIL")

    idiom_path = out / "c0_idiom_diagnostic1000_for_judge.jsonl"
    write_jsonl_atomic(idiom_path, idiom)

    manifest = {
        "status": "PASS",
        "scientific_class": "DIAGNOSTIC ONLY / LAB ADAPTATION",
        "model": args.model,
        "generation_contract": {
            "prompt": PROMPT,
            "prompt_sha256": EXPECTED_PROMPT_SHA256,
            "enable_thinking": False,
            "do_sample": False,
            "max_new_tokens": 512,
        },
        "diagnostic_manifest": diag_manifest,
        "outputs": {
            merged_path.name: sha256_file(merged_path),
            chem_path.name: sha256_file(chem_path),
            "c0_chemistry_summary.json": sha256_file(out / "c0_chemistry_summary.json"),
            idiom_path.name: sha256_file(idiom_path),
        },
    }
    atomic_json(out / "manifest.json", manifest)
    atomic_json(
        out / "progress.json",
        {"status": "PASS", "done": 2000, "total": 2000, "percentage": 100.0, "updated": now()},
    )

    print("C0_TARGETED_DIAGNOSTIC_GENERATION=PASS", flush=True)
    print("CHEMISTRY_DIAGNOSTIC=" + json.dumps(chem_summary, ensure_ascii=False), flush=True)
    print(f"IDIOM_JUDGE_INPUT={idiom_path}", flush=True)
    print(f"IDIOM_JUDGE_INPUT_SHA256={sha256_file(idiom_path)}", flush=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-id", type=int)
    ap.add_argument("--device", type=int)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--input")
    ap.add_argument("--output")
    ap.add_argument("--model")
    ap.add_argument("--data")
    ap.add_argument("--diagnostic-dir")
    ap.add_argument("--out")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.worker:
        worker_main(args)
    else:
        for name in ("model", "data", "diagnostic_dir", "out"):
            if not getattr(args, name):
                raise SystemExit(f"MISSING_ARG --{name.replace('_', '-')}")
        inp, diag_manifest = freeze_diagnostic(Path(args.data), Path(args.diagnostic_dir))
        master_generate(args, inp, diag_manifest)
