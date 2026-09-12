#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path("/workspace/mtpatcher")

REPO = ROOT / "repo/MT-Patcher-Reproduction-Ascend"

H5_RUN = (
    ROOT
    / "runs/targeted/wa_opd_horizon5_o12_20260911"
)

OLD_RUN = (
    ROOT
    / "runs/targeted/wa_opd_trainset_audit_v1_20260911"
)

CONTEXT_ROOT = (
    ROOT
    / "data/mtpatcher_v3_full6565_20260823"
    / "targeted_section43_contexts_qwen3_8b_final_20260910"
)

H5_SOURCE = (
    REPO
    / "scripts/targeted/targeted_wa_opd_horizon5_o12_v2.py"
)

EXPECTED_H5_SHA = (
    "38ac62c0a93f1b762b7536db60421dd8"
    "c1e7aca49a57c25851306c3cb76b6e16"
)

OUT = H5_RUN / "learning_curve/train_matched"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)

    return h.hexdigest()


def read_jsonl(path: Path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def append_jsonl(path: Path, row):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
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


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(path.name + ".tmp")

    tmp.write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(tmp, path)


def load_h5_module():
    got = sha256_file(H5_SOURCE)

    if got != EXPECTED_H5_SHA:
        raise RuntimeError(
            f"H5 source SHA mismatch got={got}"
        )

    spec = importlib.util.spec_from_file_location(
        "h5_source",
        H5_SOURCE,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load H5 source")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


def frozen_context(domain: str):
    path = CONTEXT_ROOT / f"{domain}_train.jsonl"

    rows = read_jsonl(path)

    if len(rows) != 5500:
        raise RuntimeError(
            f"{domain} frozen train count={len(rows)}"
        )

    out = {}

    for r in rows:
        jid = int(r["job_id"])

        if jid in out:
            raise RuntimeError(
                f"duplicate job_id domain={domain} jid={jid}"
            )

        out[jid] = r

    return out


def row_label_is_c0(row, path: Path):
    label = str(
        row.get("label")
        or row.get("arm")
        or ""
    )

    eid = str(row.get("eval_id") or "")

    path_parts = set(path.parts)

    return (
        label == "C0"
        or eid.startswith("C0|")
        or "C0" in path_parts
    )


def discover_exact_old_sample(domain: str):
    context = frozen_context(domain)

    selected = {}

    for path in OLD_RUN.rglob("*.jsonl"):
        try:
            rows = read_jsonl(path)
        except Exception:
            continue

        for r in rows:
            if not row_label_is_c0(r, path):
                continue

            jid_raw = r.get("job_id")

            if jid_raw is None:
                continue

            try:
                jid = int(jid_raw)
            except Exception:
                continue

            base = context.get(jid)

            if base is None:
                continue

            src = r.get("src_text")

            if isinstance(src, str):
                if src.strip() != str(base["src_text"]).strip():
                    continue

            selected[jid] = base

    if len(selected) != 1000:
        report = {
            "status": "FAIL",
            "domain": domain,
            "discovered": len(selected),
            "expected": 1000,
            "job_ids": sorted(selected),
        }

        atomic_json(
            OUT / f"{domain}_sample_discovery_fail.json",
            report,
        )

        raise RuntimeError(
            f"old frozen {domain} sample discovery "
            f"got={len(selected)} expected=1000"
        )

    rows = [
        selected[j]
        for j in sorted(selected)
    ]

    manifest = {
        "status": "PASS",
        "domain": domain,
        "n": 1000,
        "job_ids": [
            int(r["job_id"])
            for r in rows
        ],
    }

    manifest_bytes = json.dumps(
        manifest["job_ids"],
        separators=(",", ":"),
    ).encode()

    manifest["job_id_sha256"] = hashlib.sha256(
        manifest_bytes
    ).hexdigest()

    atomic_json(
        OUT / f"{domain}_sample_manifest.json",
        manifest,
    )

    return rows


def existing_map(path: Path):
    if not path.exists():
        return {}

    out = {}

    for r in read_jsonl(path):
        if r.get("status") == "PASS":
            out[r["job_id"]] = r

    return out


def canonical_chem_target(row):
    lr = row.get("lexical_record")

    if not isinstance(lr, dict):
        raise RuntimeError(
            f"missing lexical_record job_id={row['job_id']}"
        )

    target = lr.get("en_name")

    if not isinstance(target, str) or not target.strip():
        raise RuntimeError(
            f"missing strict en_name job_id={row['job_id']}"
        )

    return target.strip()


def worker(args):
    import torch
    import torch_npu  # noqa: F401

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    arm = args.arm
    pass_id = args.pass_id

    if arm == "O1":
        domain = "idiom"
    elif arm == "O2":
        domain = "chemistry"
    else:
        raise RuntimeError(arm)

    label = f"{arm}P{pass_id}"

    model_path = (
        H5_RUN
        / arm
        / "train/epoch_checkpoints"
        / f"pass_{pass_id}_hf"
    )

    if not model_path.exists():
        raise RuntimeError(
            f"missing checkpoint {model_path}"
        )

    rows = discover_exact_old_sample(domain)

    od = OUT / arm / f"P{pass_id}"
    od.mkdir(parents=True, exist_ok=True)

    pred_path = od / "train1000_translations.jsonl"

    done = existing_map(pred_path)

    h5 = load_h5_module()

    device = "npu:0"
    torch.npu.set_device(device)

    tok = AutoTokenizer.from_pretrained(
        model_path
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    started = time.time()
    initial = len(done)

    for r in rows:
        jid = int(r["job_id"])

        if jid in done:
            continue

        hyp = h5.generate_one_translation(
            model,
            tok,
            r["src_text"],
            device,
        )

        rec = {
            "status": "PASS",
            "label": label,
            "arm": label,
            "domain": domain,
            "split": "train",
            "eval_id":
                f"{label}|{domain}|train|{jid}",
            "job_id": jid,
            "entity_key": r.get("entity_key"),
            "src_term": r["src_term"],
            "definition": r.get("definition"),
            "src_text": r["src_text"],
            "model_translation": hyp,
        }

        if domain == "chemistry":
            target = canonical_chem_target(r)

            rec["canonical_en_target"] = target

            rec["canonical_target_hit"] = (
                target.casefold()
                in hyp.casefold()
            )

        append_jsonl(
            pred_path,
            rec,
        )

        done[jid] = rec

        n = len(done)

        if n % 20 == 0 or n == 1000:
            fresh = n - initial

            elapsed = max(
                time.time() - started,
                1e-9,
            )

            rate = (
                fresh / elapsed
                if fresh
                else 0.0
            )

            eta = (
                (1000 - n) / rate
                if rate
                else None
            )

            print(
                f"{label} "
                f"domain={domain} "
                f"done={n}/1000 "
                f"pct={100*n/1000:.1f}% "
                f"rate={rate:.2f}/s "
                f"ETA="
                + (
                    "?"
                    if eta is None
                    else f"{eta/60:.1f}m"
                ),
                flush=True,
            )

    ordered = [
        done[int(r["job_id"])]
        for r in rows
    ]

    if len(ordered) != 1000:
        raise RuntimeError(
            f"{label} final n={len(ordered)}"
        )

    if domain == "chemistry":
        hits = sum(
            bool(r["canonical_target_hit"])
            for r in ordered
        )

        summary = {
            "status": "PASS",
            "label": label,
            "domain": domain,
            "n": 1000,
            "hits": hits,
            "accuracy": hits / 1000,
        }

        atomic_json(
            od / "chemistry_summary.json",
            summary,
        )

        print(
            f"{label} CHEM_TRAIN "
            f"accuracy={hits/1000:.6f}",
            flush=True,
        )

    else:
        judge_path = (
            od
            / f"{label.lower()}_idiom_train1000_for_judge.jsonl"
        )

        tmp = judge_path.with_name(
            judge_path.name + ".tmp"
        )

        with tmp.open(
            "w",
            encoding="utf-8",
        ) as f:
            for r in ordered:
                f.write(
                    json.dumps(
                        r,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )

            f.flush()
            os.fsync(f.fileno())

        os.replace(
            tmp,
            judge_path,
        )

        print(
            f"{label} IDIOM_TRAIN_JUDGE_INPUT="
            f"{judge_path}",
            flush=True,
        )

    atomic_json(
        od / "state.json",
        {
            "status": "PASS",
            "label": label,
            "domain": domain,
            "n": 1000,
        },
    )

    print(
        f"FINAL_RESULT={label}_TRAIN1000_PASS",
        flush=True,
    )


def sample_audit(_):
    chem = discover_exact_old_sample(
        "chemistry"
    )

    idiom = discover_exact_old_sample(
        "idiom"
    )

    print(
        "TRAIN_SAMPLE_AUDIT=PASS "
        f"chem={len(chem)} idiom={len(idiom)}"
    )


def main():
    p = argparse.ArgumentParser()

    sp = p.add_subparsers(
        dest="cmd",
        required=True,
    )

    a = sp.add_parser("sample-audit")
    a.set_defaults(func=sample_audit)

    w = sp.add_parser("worker")

    w.add_argument(
        "--arm",
        choices=["O1", "O2"],
        required=True,
    )

    w.add_argument(
        "--pass-id",
        type=int,
        choices=[1, 2, 3, 4, 5],
        required=True,
    )

    w.set_defaults(func=worker)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
