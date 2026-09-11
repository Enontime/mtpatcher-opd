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

SRC_RUN = (
    ROOT
    / "runs/targeted"
    / "wa_opd_trainset_audit_v1_20260911"
)

RUN = (
    ROOT
    / "runs/targeted"
    / "wa_sft_trainset_audit_c123_20260911"
)

CHEM = SRC_RUN / "frozen_train_subset/chemistry_train1000.jsonl"
IDIOM = SRC_RUN / "frozen_train_subset/idiom_train1000.jsonl"

EXPECTED_CHEM_SHA = (
    "069b17ad9e0fa2438a846ddca161dbf6a20e20417cd0a7f39e34a15ad0a5b69d"
)

EXPECTED_IDIOM_SHA = (
    "ca633196cdcd2a7707b13f78fcead6a75fa9a7558a28db4b4e02ee535952f809"
)

MODELS = {
    "C1": (
        ROOT
        / "runs/targeted"
        / "wa_sft_positive_control_c123_20260910"
        / "C1/train/final_hf"
    ),
    "C2": (
        ROOT
        / "runs/targeted"
        / "wa_sft_positive_control_c123_20260910"
        / "C2/train/final_hf"
    ),
    "C3": (
        ROOT
        / "runs/targeted"
        / "wa_sft_positive_control_c123_20260910"
        / "C3/train/final_hf"
    ),
}

DEVICES = {
    "C1": "0",
    "C2": "1",
    "C3": "2",
}

PROMPT = (
    "Translate the following text into English without additional explanations:"
    "\n\n{source}\n\n"
)

EXPECTED_PROMPT_SHA = (
    "63101d0739ff2ee11b0e5d548b85dcd18be7a241777893326140f12778d9a8b8"
)


def now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


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
        ) + "\n",
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
                ) + "\n"
            )
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
            ) + "\n"
        )
        f.flush()
        os.fsync(f.fileno())


def get_en_name(row):
    x = row.get("en_name")

    if isinstance(x, str) and x.strip():
        return x.strip()

    lexical = row.get("lexical_record")

    if isinstance(lexical, dict):
        x = lexical.get("en_name")

        if isinstance(x, str) and x.strip():
            return x.strip()

    raise RuntimeError(
        f"missing explicit en_name job_id={row.get('job_id')}"
    )


def validate_inputs():
    require(CHEM.is_file(), f"missing {CHEM}")
    require(IDIOM.is_file(), f"missing {IDIOM}")

    require(
        sha256_file(CHEM) == EXPECTED_CHEM_SHA,
        f"CHEM SHA mismatch: {sha256_file(CHEM)}",
    )

    require(
        sha256_file(IDIOM) == EXPECTED_IDIOM_SHA,
        f"IDIOM SHA mismatch: {sha256_file(IDIOM)}",
    )

    require(
        sha256_text(PROMPT) == EXPECTED_PROMPT_SHA,
        f"prompt SHA mismatch: {sha256_text(PROMPT)}",
    )

    chem = read_jsonl(CHEM)
    idiom = read_jsonl(IDIOM)

    require(len(chem) == 1000, f"chem rows={len(chem)}")
    require(len(idiom) == 1000, f"idiom rows={len(idiom)}")

    for row in chem:
        require(row.get("src_text"), "chem missing src_text")
        require(row.get("src_term"), "chem missing src_term")
        get_en_name(row)

    for row in idiom:
        require(row.get("src_text"), "idiom missing src_text")
        require(row.get("src_term"), "idiom missing src_term")
        require(
            isinstance(row.get("definition"), str)
            and row["definition"].strip(),
            f"idiom missing definition job={row.get('job_id')}",
        )

    print(
        "INPUT_CONTRACT=PASS "
        f"chem_sha={EXPECTED_CHEM_SHA} "
        f"idiom_sha={EXPECTED_IDIOM_SHA}",
        flush=True,
    )


def generate_one(model, tokenizer, device, source):
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

    return tokenizer.decode(
        gen,
        skip_special_tokens=True,
    ).strip()


def worker(label):
    import torch
    import torch_npu
    from transformers import AutoModelForCausalLM, AutoTokenizer

    require(label in MODELS, label)

    model_path = MODELS[label]

    require(
        model_path.is_dir(),
        f"missing model: {model_path}",
    )

    device = torch.device("npu:0")
    torch.npu.set_device(device)

    out_dir = RUN / label
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "train2000_translations.jsonl"
    progress_file = out_dir / "progress.json"
    summary_file = out_dir / "summary.json"

    chem = read_jsonl(CHEM)
    idiom = read_jsonl(IDIOM)

    jobs = (
        [("chemistry", x) for x in chem]
        + [("idiom", x) for x in idiom]
    )

    completed = {}

    if out_file.exists():
        for row in read_jsonl(out_file):
            completed[row["eval_id"]] = row

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
    start_done = len(completed)

    total = len(jobs)

    for domain, row in jobs:

        eval_id = (
            f"{label}|{domain}|train|{row['job_id']}"
        )

        if eval_id in completed:
            continue

        hyp = generate_one(
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
            en_name = get_en_name(row)

            result["en_name"] = en_name

            result["chemistry_hit"] = (
                en_name.casefold()
                in hyp.casefold()
            )

        else:
            result["definition"] = row["definition"]

        append_jsonl(out_file, result)
        completed[eval_id] = result

        done = len(completed)

        if done % 20 == 0 or done == total:

            elapsed = max(time.time() - start, 1e-6)

            new_done = max(
                done - start_done,
                1,
            )

            rate = new_done / elapsed
            remain = total - done

            eta = (
                remain / rate
                if rate > 0
                else None
            )

            atomic_json(
                progress_file,
                {
                    "status": (
                        "RUNNING"
                        if done < total
                        else "PASS"
                    ),
                    "label": label,
                    "done": done,
                    "total": total,
                    "percentage": 100.0 * done / total,
                    "rate": rate,
                    "eta_seconds": eta,
                    "updated": now(),
                },
            )

            eta_text = (
                f"{eta/60:.1f}m"
                if eta is not None
                else "?"
            )

            print(
                f"{now()} "
                f"label={label} "
                f"done={done}/{total} "
                f"pct={100*done/total:.1f}% "
                f"rate={rate:.2f}/s "
                f"ETA={eta_text}",
                flush=True,
            )

    rows = read_jsonl(out_file)

    require(
        len(rows) == 2000,
        f"{label}: rows={len(rows)}",
    )

    chem_rows = [
        r
        for r in rows
        if r["domain"] == "chemistry"
    ]

    idiom_rows = [
        r
        for r in rows
        if r["domain"] == "idiom"
    ]

    require(
        len(chem_rows) == 1000,
        f"{label}: chem={len(chem_rows)}",
    )

    require(
        len(idiom_rows) == 1000,
        f"{label}: idiom={len(idiom_rows)}",
    )

    hits = sum(
        int(bool(r["chemistry_hit"]))
        for r in chem_rows
    )

    summary = {
        "status": "PASS",
        "scientific_class": (
            "DIAGNOSTIC ONLY / TRAINING-PROCESS POSITIVE CONTROL"
        ),
        "label": label,
        "model": str(model_path),
        "chemistry_train": {
            "n": 1000,
            "hits": hits,
            "accuracy": hits / 1000.0,
        },
        "idiom_train": {
            "n": 1000,
            "status": "PENDING_FROZEN_DEEPSEEK_JUDGE",
        },
        "output_sha256": sha256_file(out_file),
        "updated": now(),
    }

    atomic_json(
        summary_file,
        summary,
    )

    atomic_json(
        progress_file,
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
        f"{label}_SFT_TRAINSET=PASS "
        f"chem_acc={hits/1000.0:.6f}",
        flush=True,
    )


def combine():
    rows = []
    chem = {}

    for label in ("C1", "C2", "C3"):

        summary_path = (
            RUN
            / label
            / "summary.json"
        )

        require(
            summary_path.exists(),
            f"missing {summary_path}",
        )

        summary = json.load(
            open(
                summary_path,
                "r",
                encoding="utf-8",
            )
        )

        require(
            summary["status"] == "PASS",
            f"{label} summary not PASS",
        )

        chem[label] = summary["chemistry_train"]

        label_rows = read_jsonl(
            RUN
            / label
            / "train2000_translations.jsonl"
        )

        idiom = [
            r
            for r in label_rows
            if r["domain"] == "idiom"
        ]

        require(
            len(idiom) == 1000,
            f"{label}: idiom={len(idiom)}",
        )

        rows.extend(idiom)

    require(
        len(rows) == 3000,
        f"idiom combined={len(rows)}",
    )

    idiom_out = (
        RUN
        / "c123_idiom_train3000_for_judge.jsonl"
    )

    write_jsonl_atomic(
        idiom_out,
        rows,
    )

    c0_chem = 0.087

    chem_out = {
        "status": "PASS",
        "scientific_class": (
            "DIAGNOSTIC ONLY / TRAINING-PROCESS POSITIVE CONTROL"
        ),
        "C0": {
            "accuracy": c0_chem,
            "source": (
                "/workspace/mtpatcher/runs/targeted/"
                "wa_opd_trainset_audit_v1_20260911/"
                "chemistry_train_comparison.json"
            ),
        },
        "C1": chem["C1"],
        "C2": chem["C2"],
        "C3": chem["C3"],
        "delta_vs_C0": {
            x: chem[x]["accuracy"] - c0_chem
            for x in ("C1", "C2", "C3")
        },
        "idiom_judge_input": {
            "path": str(idiom_out),
            "rows": 3000,
            "sha256": sha256_file(idiom_out),
        },
        "created": now(),
    }

    atomic_json(
        RUN / "chemistry_train_comparison.json",
        chem_out,
    )

    print("=" * 72)
    print("SFT TRAIN1000 CHEMISTRY RESULTS")

    print(
        f"C0 accuracy={c0_chem:.6f}"
    )

    for label in ("C1", "C2", "C3"):
        acc = chem[label]["accuracy"]

        print(
            f"{label} accuracy={acc:.6f} "
            f"delta_vs_C0={acc-c0_chem:+.6f}"
        )

    print(
        f"C123_IDIOM_TRAIN3000_JUDGE_INPUT={idiom_out}"
    )

    print(
        f"C123_IDIOM_TRAIN3000_SHA256="
        f"{sha256_file(idiom_out)}"
    )

    print(
        "SERVER_SFT_TRAINSET_AUDIT=PASS"
    )

    print(
        "NEXT=RUN_FROZEN_DEEPSEEK_C123_TRAIN3000"
    )

    print("=" * 72)


def master():
    print("=" * 72)
    print("SFT C1/C2/C3 TRAINSET POSITIVE-CONTROL AUDIT v1")
    print(
        "Question: Are the same frozen train contexts "
        "easy to learn under supervised targets?"
    )
    print(
        "Competing explanations: data intrinsically hard vs "
        "OPD objective/optimization weak."
    )
    print(
        "Falsifiable prediction: matched SFT arms should show "
        "substantially larger train task gains than OPD."
    )
    print(
        "Decision after result: if SFT >> OPD on identical train1000, "
        "proceed to 5-pass OPD horizon test."
    )
    print("=" * 72, flush=True)

    validate_inputs()

    atomic_json(
        RUN / "state.json",
        {
            "status": "RUNNING",
            "phase": "generation",
            "updated": now(),
        },
    )

    script = Path(__file__).resolve()

    procs = {}

    for label in ("C1", "C2", "C3"):

        log_path = (
            RUN
            / "logs"
            / f"{label}.log"
        )

        log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        log = open(
            log_path,
            "a",
            encoding="utf-8",
        )

        env = os.environ.copy()

        env["ASCEND_RT_VISIBLE_DEVICES"] = DEVICES[label]
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["PYTHONUNBUFFERED"] = "1"

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

        procs[label] = (
            p,
            log,
        )

        print(
            f"START label={label} "
            f"npu={DEVICES[label]} "
            f"pid={p.pid}",
            flush=True,
        )

    while True:

        alive = False
        parts = []

        for label, (p, _) in procs.items():

            rc = p.poll()

            if rc is None:
                alive = True

                progress = (
                    RUN
                    / label
                    / "progress.json"
                )

                if progress.exists():
                    try:
                        x = json.load(
                            open(
                                progress,
                                "r",
                                encoding="utf-8",
                            )
                        )

                        parts.append(
                            f"{label}=RUNNING:"
                            f"{x.get('done',0)}/"
                            f"{x.get('total',2000)}"
                        )

                    except Exception:
                        parts.append(
                            f"{label}=RUNNING"
                        )

                else:
                    parts.append(
                        f"{label}=RUNNING"
                    )

            else:
                parts.append(
                    f"{label}=RC={rc}"
                )

        print(
            f"{now()} phase=sft_trainset_audit "
            + " | ".join(parts),
            flush=True,
        )

        if not alive:
            break

        time.sleep(20)

    failed = []

    for label, (p, log) in procs.items():

        rc = p.wait()
        log.close()

        if rc != 0:
            failed.append(
                [label, rc]
            )

    if failed:

        atomic_json(
            RUN / "state.json",
            {
                "status": "FAIL",
                "failed": failed,
                "updated": now(),
            },
        )

        raise RuntimeError(
            f"worker failures: {failed}"
        )

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

    print(
        "FINAL_RESULT=SFT_TRAINSET_SERVER_PASS",
        flush=True,
    )


def main():

    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(
        dest="cmd",
        required=True,
    )

    sub.add_parser("master")

    p = sub.add_parser("worker")

    p.add_argument(
        "--label",
        choices=["C1", "C2", "C3"],
        required=True,
    )

    sub.add_parser("combine")

    args = parser.parse_args()

    if args.cmd == "master":
        master()

    elif args.cmd == "worker":
        worker(args.label)

    elif args.cmd == "combine":
        combine()


if __name__ == "__main__":
    main()
