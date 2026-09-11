#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/workspace/mtpatcher")

CONTEXT_ROOT = (
    ROOT
    / "data/mtpatcher_v3_full6565_20260823"
    / "targeted_section43_contexts_qwen3_8b_final_20260910"
)

RUN = (
    ROOT
    / "runs/targeted"
    / "wa_opd_trainset_audit_v1_20260911"
)

CHEM_SOURCE = CONTEXT_ROOT / "chemistry_train.jsonl"
IDIOM_SOURCE = CONTEXT_ROOT / "idiom_train.jsonl"

EXPECTED_SOURCE_SHA = {
    "chemistry": "ca6532cacce64f24f14226a4dca509b494aaa5f5bbeecbc0f1d2f157c317044f",
    "idiom": "804df445752caa48ee95a6d2d6ce9403b03e8aba6eaa994b380b5bc9653d0739",
}

MODELS = {
    "C0": ROOT / "models/Qwen3-0.6B",
    "O1": (
        ROOT
        / "runs/targeted/wa_opd_knowledge_conditioned_o123_20260911"
        / "O1/train/final_hf"
    ),
    "O2": (
        ROOT
        / "runs/targeted/wa_opd_knowledge_conditioned_o123_20260911"
        / "O2/train/final_hf"
    ),
    "O3": (
        ROOT
        / "runs/targeted/wa_opd_knowledge_conditioned_o123_20260911"
        / "O3/train/final_hf"
    ),
}

DEVICES = {
    "C0": "0",
    "O1": "1",
    "O2": "2",
    "O3": "3",
}

SAMPLE_N = 1000

PROMPT = (
    "Translate the following text into English without additional explanations:"
    "\n\n{source}\n\n"
)

EXPECTED_PROMPT_SHA = (
    "63101d0739ff2ee11b0e5d548b85dcd18be7a241777893326140f12778d9a8b8"
)


def now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_bytes(x):
    return hashlib.sha256(x).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json(path, obj):
    atomic_text(
        path,
        json.dumps(
            obj,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
    )


def write_jsonl_atomic(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl_durable(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
        f.flush()
        os.fsync(f.fileno())


def selection_key(domain, row):
    payload = {
        "domain": domain,
        "job_id": row.get("job_id"),
        "entity_key": row.get("entity_key"),
        "src_term": row.get("src_term"),
        "src_text": row.get("src_text"),
    }
    b = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def freeze_subset(domain, source, out_path):
    rows = read_jsonl(source)
    require(len(rows) == 5500, f"{domain}: expected 5500 rows, got {len(rows)}")

    ranked = sorted(
        rows,
        key=lambda r: (
            selection_key(domain, r),
            str(r.get("job_id")),
        ),
    )

    selected = ranked[:SAMPLE_N]

    for r in selected:
        require(r.get("src_text"), f"{domain}: missing src_text")
        require(r.get("src_term"), f"{domain}: missing src_term")

        if domain == "chemistry":
            lexical = r.get("lexical_record") or {}
            en_name = r.get("en_name") or lexical.get("en_name")
            require(
                isinstance(en_name, str) and en_name.strip(),
                f"chemistry job={r.get('job_id')}: missing explicit en_name",
            )

        if domain == "idiom":
            require(
                isinstance(r.get("definition"), str)
                and r["definition"].strip(),
                f"idiom job={r.get('job_id')}: missing definition",
            )

    write_jsonl_atomic(out_path, selected)
    return selected


def prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    subset_dir = RUN / "frozen_train_subset"
    subset_dir.mkdir(parents=True, exist_ok=True)

    prompt_sha = sha256_bytes(PROMPT.encode("utf-8"))
    require(
        prompt_sha == EXPECTED_PROMPT_SHA,
        f"prompt SHA mismatch: {prompt_sha}",
    )

    for domain, p in {
        "chemistry": CHEM_SOURCE,
        "idiom": IDIOM_SOURCE,
    }.items():
        require(p.is_file(), f"missing source: {p}")
        got = sha256_file(p)
        require(
            got == EXPECTED_SOURCE_SHA[domain],
            f"{domain} source SHA mismatch: {got}",
        )

    chem_out = subset_dir / "chemistry_train1000.jsonl"
    idiom_out = subset_dir / "idiom_train1000.jsonl"

    chem = freeze_subset("chemistry", CHEM_SOURCE, chem_out)
    idiom = freeze_subset("idiom", IDIOM_SOURCE, idiom_out)

    manifest = {
        "status": "FROZEN",
        "scientific_class": "DIAGNOSTIC ONLY / TRAINING-PROCESS AUDIT",
        "created": now(),
        "selection": (
            "deterministic SHA256 rank over "
            "domain/job_id/entity_key/src_term/src_text; first 1000"
        ),
        "downstream_outcome_used": False,
        "student_output_used": False,
        "evaluator_score_used": False,
        "student_prompt_contains_target_knowledge": False,
        "sample_n_per_domain": SAMPLE_N,
        "generation": {
            "prompt": PROMPT,
            "prompt_sha256": prompt_sha,
            "enable_thinking": False,
            "do_sample": False,
            "max_new_tokens": 512,
        },
        "sources": {
            "chemistry": {
                "path": str(CHEM_SOURCE),
                "sha256": sha256_file(CHEM_SOURCE),
            },
            "idiom": {
                "path": str(IDIOM_SOURCE),
                "sha256": sha256_file(IDIOM_SOURCE),
            },
        },
        "subsets": {
            "chemistry": {
                "path": str(chem_out),
                "rows": len(chem),
                "sha256": sha256_file(chem_out),
            },
            "idiom": {
                "path": str(idiom_out),
                "rows": len(idiom),
                "sha256": sha256_file(idiom_out),
            },
        },
    }

    atomic_json(RUN / "manifest.json", manifest)

    print(
        "TRAINSET_AUDIT_FREEZE=PASS "
        f"chem_sha={manifest['subsets']['chemistry']['sha256']} "
        f"idiom_sha={manifest['subsets']['idiom']['sha256']}",
        flush=True,
    )


def extract_en_name(row):
    direct = row.get("en_name")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    lexical = row.get("lexical_record")
    if isinstance(lexical, dict):
        x = lexical.get("en_name")
        if isinstance(x, str) and x.strip():
            return x.strip()

    raise RuntimeError(
        f"missing explicit canonical en_name job={row.get('job_id')}"
    )


def translate(model, tokenizer, device, source):
    import torch

    user = PROMPT.format(source=source)

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    batch = tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
    )

    batch = {
        k: v.to(device)
        for k, v in batch.items()
    }

    input_len = batch["input_ids"].shape[1]

    with torch.inference_mode():
        out = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=512,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    gen = out[0, input_len:]
    text = tokenizer.decode(
        gen,
        skip_special_tokens=True,
    ).strip()

    return text


def worker(label):
    import torch
    import torch_npu
    from transformers import AutoModelForCausalLM, AutoTokenizer

    require(label in MODELS, f"unknown label {label}")

    model_path = MODELS[label]
    require(model_path.is_dir(), f"missing model {model_path}")

    device = torch.device("npu:0")
    torch.npu.set_device(device)

    label_dir = RUN / "labels" / label
    label_dir.mkdir(parents=True, exist_ok=True)

    out_path = label_dir / "train2000_translations.jsonl"
    progress_path = label_dir / "progress.json"
    summary_path = label_dir / "summary.json"

    chem = read_jsonl(
        RUN / "frozen_train_subset/chemistry_train1000.jsonl"
    )
    idiom = read_jsonl(
        RUN / "frozen_train_subset/idiom_train1000.jsonl"
    )

    jobs = [
        ("chemistry", r)
        for r in chem
    ] + [
        ("idiom", r)
        for r in idiom
    ]

    completed = {}

    if out_path.exists():
        for r in read_jsonl(out_path):
            completed[r["eval_id"]] = r

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)

    model.eval()

    start = time.time()
    total = len(jobs)

    for i, (domain, row) in enumerate(jobs, 1):
        eval_id = (
            f"{label}|{domain}|train|{row['job_id']}"
        )

        if eval_id in completed:
            continue

        hyp = translate(
            model,
            tokenizer,
            device,
            row["src_text"],
        )

        result = {
            "eval_id": eval_id,
            "label": label,
            "domain": domain,
            "split": "train",
            "job_id": row["job_id"],
            "entity_key": row.get("entity_key"),
            "src_term": row["src_term"],
            "src_text": row["src_text"],
            "model_translation": hyp,
        }

        if domain == "chemistry":
            en_name = extract_en_name(row)
            result["en_name"] = en_name
            result["chemistry_hit"] = (
                en_name.casefold()
                in hyp.casefold()
            )

        else:
            result["definition"] = row["definition"]

        append_jsonl_durable(out_path, result)
        completed[eval_id] = result

        done = len(completed)

        if done % 20 == 0 or done == total:
            elapsed = max(time.time() - start, 1e-6)

            newly_done = max(
                1,
                done,
            )

            rate = newly_done / elapsed
            remain = total - done
            eta = remain / rate if rate > 0 else None

            progress = {
                "status": "RUNNING" if done < total else "PASS",
                "label": label,
                "done": done,
                "total": total,
                "percentage": 100.0 * done / total,
                "rate_rows_per_sec": rate,
                "eta_seconds": eta,
                "updated": now(),
            }
            atomic_json(progress_path, progress)

            eta_m = (
                f"{eta / 60:.1f}m"
                if eta is not None
                else "?"
            )

            print(
                f"{now()} label={label} "
                f"done={done}/{total} "
                f"pct={100*done/total:.1f}% "
                f"rate={rate:.2f}/s "
                f"ETA={eta_m}",
                flush=True,
            )

    rows = read_jsonl(out_path)
    require(len(rows) == 2000, f"{label}: expected 2000 rows")

    chem_rows = [
        r for r in rows
        if r["domain"] == "chemistry"
    ]
    idiom_rows = [
        r for r in rows
        if r["domain"] == "idiom"
    ]

    require(len(chem_rows) == 1000, f"{label}: chemistry != 1000")
    require(len(idiom_rows) == 1000, f"{label}: idiom != 1000")

    hits = sum(bool(r["chemistry_hit"]) for r in chem_rows)

    summary = {
        "status": "PASS",
        "label": label,
        "model": str(model_path),
        "rows": 2000,
        "chemistry_train": {
            "n": 1000,
            "hits": hits,
            "accuracy": hits / 1000.0,
            "metric": (
                "case-insensitive explicit canonical "
                "English en_name substring accuracy"
            ),
        },
        "idiom_train": {
            "n": 1000,
            "status": "WAITING_FOR_FROZEN_DEEPSEEK_JUDGE",
        },
        "translations_sha256": sha256_file(out_path),
        "updated": now(),
    }

    atomic_json(summary_path, summary)

    atomic_json(
        progress_path,
        {
            "status": "PASS",
            "label": label,
            "done": 2000,
            "total": 2000,
            "percentage": 100.0,
            "updated": now(),
        },
    )

    print(
        f"{label}_TRAINSET_GENERATION=PASS "
        f"chem_acc={hits/1000.0:.6f}",
        flush=True,
    )


def combine():
    all_rows = []
    chemistry = {}

    for label in ("C0", "O1", "O2", "O3"):
        label_dir = RUN / "labels" / label

        summary = json.load(
            open(
                label_dir / "summary.json",
                "r",
                encoding="utf-8",
            )
        )

        require(summary["status"] == "PASS", f"{label}: not PASS")
        chemistry[label] = summary["chemistry_train"]

        rows = read_jsonl(
            label_dir / "train2000_translations.jsonl"
        )

        idiom = [
            r
            for r in rows
            if r["domain"] == "idiom"
        ]
        require(len(idiom) == 1000, f"{label}: idiom count != 1000")
        all_rows.extend(idiom)

    require(len(all_rows) == 4000, "combined idiom != 4000")

    idiom_path = RUN / "idiom_train4000_for_judge.jsonl"
    write_jsonl_atomic(idiom_path, all_rows)

    c0 = chemistry["C0"]["accuracy"]

    chem_summary = {
        "status": "PASS",
        "scientific_class": (
            "DIAGNOSTIC ONLY / TRAINING-PROCESS AUDIT"
        ),
        "metric": (
            "case-insensitive explicit canonical "
            "English en_name substring accuracy"
        ),
        "C0": chemistry["C0"],
        "O1": chemistry["O1"],
        "O2": chemistry["O2"],
        "O3": chemistry["O3"],
        "delta_vs_C0": {
            label: chemistry[label]["accuracy"] - c0
            for label in ("O1", "O2", "O3")
        },
        "idiom_judge_input": {
            "path": str(idiom_path),
            "rows": 4000,
            "sha256": sha256_file(idiom_path),
        },
        "created": now(),
    }

    atomic_json(
        RUN / "chemistry_train_comparison.json",
        chem_summary,
    )

    print("=" * 72)
    print("CHEMISTRY TRAIN1000", flush=True)

    for label in ("C0", "O1", "O2", "O3"):
        acc = chemistry[label]["accuracy"]
        print(
            f"{label}: accuracy={acc:.6f} "
            f"delta_vs_C0={acc-c0:+.6f}",
            flush=True,
        )

    print(
        f"IDIOM_TRAIN4000_JUDGE_INPUT={idiom_path}",
        flush=True,
    )
    print(
        f"IDIOM_TRAIN4000_SHA256={sha256_file(idiom_path)}",
        flush=True,
    )
    print("SERVER_TRAINSET_AUDIT_STAGE=PASS", flush=True)
    print("NEXT=RUN_FROZEN_DEEPSEEK_IDIOM_TRAIN4000_JUDGE", flush=True)
    print("=" * 72)


def master():
    print("=" * 72)
    print("WA-OPD TRAIN-SET TASK-LEVEL AUDIT v1")
    print(
        "SCIENTIFIC_CLASS="
        "DIAGNOSTIC_ONLY_TRAINING_PROCESS_AUDIT"
    )
    print(
        "Question: Is weak 3-pass WA-OPD caused by "
        "insufficient fitting of targeted train knowledge?"
    )
    print(
        "Competing explanations: under-training vs "
        "generalization gap vs low-quality dense KL."
    )
    print(
        "Falsifiable prediction: matched O1/O2/O3 must "
        "show clear train-set task gains if targeted knowledge "
        "was actually learned."
    )
    print(
        "Decision after result: use train metric to decide "
        "whether a 5-pass learning-curve ablation is justified."
    )
    print("=" * 72, flush=True)

    prepare()

    script = Path(__file__).resolve()

    processes = {}

    for label in ("C0", "O1", "O2", "O3"):
        log_path = RUN / "logs" / f"{label}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["ASCEND_RT_VISIBLE_DEVICES"] = DEVICES[label]
        env["PYTHONUNBUFFERED"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"

        log = open(log_path, "a", encoding="utf-8")

        p = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(script),
                "worker",
                "--label",
                label,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )

        processes[label] = (p, log)

        print(
            f"START label={label} "
            f"physical_npu={DEVICES[label]} "
            f"pid={p.pid}",
            flush=True,
        )

    while True:
        states = []
        alive = False

        for label, (p, _) in processes.items():
            rc = p.poll()

            progress_path = (
                RUN
                / "labels"
                / label
                / "progress.json"
            )

            if rc is None:
                alive = True
                status = "RUNNING"

                if progress_path.exists():
                    try:
                        x = json.load(
                            open(
                                progress_path,
                                "r",
                                encoding="utf-8",
                            )
                        )
                        status += (
                            f":{x.get('done', 0)}/"
                            f"{x.get('total', 2000)}"
                        )
                    except Exception:
                        pass

            else:
                status = f"RC={rc}"

            states.append(f"{label}={status}")

        print(
            f"{now()} phase=trainset_audit "
            + " | ".join(states),
            flush=True,
        )

        if not alive:
            break

        time.sleep(20)

    failed = []

    for label, (p, log) in processes.items():
        rc = p.wait()
        log.close()
        if rc != 0:
            failed.append((label, rc))

    if failed:
        atomic_json(
            RUN / "state.json",
            {
                "status": "FAIL",
                "failed": failed,
                "updated": now(),
            },
        )
        raise RuntimeError(f"workers failed: {failed}")

    combine()

    atomic_json(
        RUN / "state.json",
        {
            "status": "PASS",
            "server_stage": "PASS",
            "idiom_judge": "PENDING",
            "updated": now(),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("master")

    p = sub.add_parser("worker")
    p.add_argument(
        "--label",
        required=True,
        choices=["C0", "O1", "O2", "O3"],
    )

    sub.add_parser("combine")

    args = parser.parse_args()

    if args.cmd == "master":
        master()
    elif args.cmd == "worker":
        worker(args.label)
    elif args.cmd == "combine":
        combine()
    else:
        raise RuntimeError(args.cmd)


if __name__ == "__main__":
    main()
