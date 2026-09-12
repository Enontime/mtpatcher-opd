#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path("/workspace/mtpatcher")
REPO = ROOT / "repo/MT-Patcher-Reproduction-Ascend"
H5_SCRIPT = REPO / "scripts/targeted/targeted_wa_opd_horizon5_o12_v2.py"
SPEC = REPO / "manifests/experiments/targeted/08_offline_prefix_support_replay.json"

EXPECTED_REPO_HEAD = "0ac7d1e979f72af623e4731a1b8058cd98c24f6a"
EXPECTED_SPEC_SHA = "f9ba67c7da7f1b7fea324186401ff2fdf96200caf1e9b245a2b7ccd88d4dd621"
EXPECTED_H5_SHA = "38ac62c0a93f1b762b7536db60421dd8c1e7aca49a57c25851306c3cb76b6e16"

PREFIX_GENERATION_SEED = 20260912
TRAINING_SEED = 20260820
MAX_NEW_TOKENS = 256
TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = 0
TEACHER_TOPK = 32


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    raw = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl_fsync(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"{path}:{ln}: {e}") from e
    return rows


def load_module():
    spec = importlib.util.spec_from_file_location(
        "mtpatcher_targeted_h5_frozen_source",
        H5_SCRIPT,
    )
    require(spec is not None and spec.loader is not None, "cannot import H5 source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def repo_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def static_contract() -> dict[str, Any]:
    require(REPO.is_dir(), f"repo missing: {REPO}")
    require(SPEC.is_file(), f"spec missing: {SPEC}")
    require(H5_SCRIPT.is_file(), f"H5 source missing: {H5_SCRIPT}")
    require(repo_head() == EXPECTED_REPO_HEAD, f"repo HEAD drift: {repo_head()}")
    require(sha256_file(SPEC) == EXPECTED_SPEC_SHA, "prefix-support spec SHA drift")
    require(sha256_file(H5_SCRIPT) == EXPECTED_H5_SHA, "H5 source SHA drift")

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    require(
        spec["scientific_class"] == "DIAGNOSTIC ONLY / OFFLINE PREFIX-SUPPORT REPLAY",
        "wrong scientific class",
    )
    require(spec["status"] == "SPEC_FROZEN_NOT_STARTED", "spec status drift")
    require(spec["objective_contract"]["teacher_top_k"] == 32, "top-k drift")
    require(spec["objective_contract"]["policy_gradient"] is False, "PG drift")
    require(
        spec["teacher_signal_audit"]["required_before_parameter_update"] is True,
        "teacher-signal gate missing",
    )

    return {
        "repo_head": EXPECTED_REPO_HEAD,
        "spec_sha256": EXPECTED_SPEC_SHA,
        "h5_source_sha256": EXPECTED_H5_SHA,
        "prefix_generation_seed": PREFIX_GENERATION_SEED,
        "training_seed_reserved": TRAINING_SEED,
        "generation": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "max_new_tokens": MAX_NEW_TOKENS,
            "enable_thinking": False,
        },
        "teacher_top_k": TEACHER_TOPK,
    }


def row_id(row: dict[str, Any]) -> str:
    return f"{row['domain']}:{int(row['job_id'])}"


def stable_seed(row: dict[str, Any], role: str) -> int:
    require(role in {"S", "T"}, f"bad role={role}")
    key = (
        f"{PREFIX_GENERATION_SEED}|{role}|"
        f"{row['domain']}|{int(row['job_id'])}"
    )
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:12], 16) % 2_000_000_000


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2**32 - 1))
    except Exception:
        pass

    import torch
    torch.manual_seed(seed)
    try:
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)
    except Exception:
        pass


def load_domain_rows(h5, domain: str, n: int) -> list[dict[str, Any]]:
    if domain == "chemistry":
        rows = h5.load_arm_rows("O2")
    elif domain == "idiom":
        rows = h5.load_arm_rows("O1")
    else:
        raise ValueError(domain)

    require(1 <= n <= len(rows), f"n={n} outside 1..{len(rows)}")
    rows = rows[:n]

    ids = [row_id(r) for r in rows]
    require(len(ids) == len(set(ids)), f"duplicate row ids in {domain}")
    return rows


def prompt_for_role(h5, row: dict[str, Any], role: str) -> str:
    if role == "S":
        return h5.student_user_text(row)
    if role == "T":
        return h5.teacher_user_text(row)
    raise ValueError(role)


def generate_one(model, tok, h5, row: dict[str, Any], role: str, device: str) -> dict[str, Any]:
    import torch

    seed = stable_seed(row, role)
    set_seed(seed)

    user_text = prompt_for_role(h5, row, role)
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    batch = tok(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    prompt_ids = batch["input_ids"][0].detach().cpu().tolist()
    prompt_width = len(prompt_ids)

    with torch.no_grad():
        seq = model.generate(
            **batch,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            use_cache=True,
        )

    response_ids = seq[0, prompt_width:].detach().cpu().tolist()
    response_ids = h5.trim_response(response_ids, tok.eos_token_id)
    require(response_ids, f"empty prefix role={role} row={row_id(row)}")

    response_text = tok.decode(
        response_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return {
        "status": "PASS",
        "row_id": row_id(row),
        "domain": row["domain"],
        "job_id": int(row["job_id"]),
        "role": role,
        "model": str(h5.C0 if role == "S" else h5.TEACHER),
        "seed": seed,
        "source_sha256": hashlib.sha256(
            row["src_text"].encode("utf-8")
        ).hexdigest(),
        "prompt_ids_sha256": sha256_json(prompt_ids),
        "prompt_token_count": len(prompt_ids),
        "prefix_ids": [int(x) for x in response_ids],
        "prefix_token_count": len(response_ids),
        "prefix_sha256": sha256_json([int(x) for x in response_ids]),
        "prefix_text": response_text,
        "first_token_ids": [int(x) for x in response_ids[:8]],
        "last_token_ids": [int(x) for x in response_ids[-8:]],
        "enable_thinking": False,
        "sampling": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "created_utc": now_utc(),
    }


def existing_bank(path: Path, role: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for rec in read_jsonl(path):
        require(rec.get("status") == "PASS", f"non-PASS row in {path}")
        require(rec.get("role") == role, f"role drift in {path}")
        rid = str(rec["row_id"])
        require(rid not in out, f"duplicate bank row {rid} in {path}")
        require(
            rec["prefix_sha256"] == sha256_json(rec["prefix_ids"]),
            f"prefix SHA corrupt row={rid} path={path}",
        )
        out[rid] = rec
    return out


def fill_bank(
    model,
    tok,
    h5,
    rows: list[dict[str, Any]],
    role: str,
    device: str,
    path: Path,
) -> list[dict[str, Any]]:
    done = existing_bank(path, role)
    started = time.time()
    initial = len(done)

    for idx, row in enumerate(rows, 1):
        rid = row_id(row)
        if rid not in done:
            rec = generate_one(model, tok, h5, row, role, device)
            append_jsonl_fsync(path, rec)
            done[rid] = rec

        if idx % 8 == 0 or idx == len(rows):
            elapsed = max(time.time() - started, 1e-9)
            fresh = len(done) - initial
            rate = fresh / elapsed if fresh else 0.0
            remain = len(rows) - idx
            eta = remain / rate if rate else None
            print(
                f"{now_utc()} BANK role={role} domain={row['domain']} "
                f"progress={idx}/{len(rows)} fresh={fresh} "
                f"rate={rate:.3f}/s ETA="
                + ("?" if eta is None else f"{eta/60:.1f}m"),
                flush=True,
            )

    ordered = [done[row_id(r)] for r in rows]
    require(len(ordered) == len(rows), f"bank count mismatch role={role}")
    return ordered


def find_subsequence(haystack: list[int], needle: list[int]) -> int | None:
    if not needle or len(needle) > len(haystack):
        return None
    limit = len(haystack) - len(needle) + 1
    for i in range(limit):
        if haystack[i:i + len(needle)] == needle:
            return i
    return None


def lexical_event(tok, en_name: str, response_ids: list[int]) -> dict[str, Any]:
    candidates: list[tuple[str, list[int]]] = []
    seen: set[tuple[int, ...]] = set()

    for form in (en_name, " " + en_name):
        ids = [
            int(x)
            for x in tok(
                form,
                add_special_tokens=False,
            )["input_ids"]
        ]
        key = tuple(ids)
        if ids and key not in seen:
            seen.add(key)
            candidates.append((form, ids))

    matches = []
    for form, ids in candidates:
        pos = find_subsequence(response_ids, ids)
        if pos is not None:
            matches.append((pos, len(ids), form, ids))

    if not matches:
        return {
            "covered": False,
            "en_name": en_name,
            "candidate_tokenizations": [
                {"form": form, "ids": ids}
                for form, ids in candidates
            ],
        }

    matches.sort(key=lambda x: (x[0], x[1]))
    pos, length, form, ids = matches[0]
    return {
        "covered": True,
        "en_name": en_name,
        "position": pos,
        "length": length,
        "matched_form": form,
        "matched_ids": ids,
    }


def teacher_signal_one(
    teacher,
    tok,
    h5,
    row: dict[str, Any],
    response_ids: list[int],
    device: str,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    require(response_ids, f"empty audit response {row_id(row)}")

    hint_prompt = h5.render_prompt_ids(
        tok,
        h5.teacher_user_text(row),
    )
    nohint_prompt = h5.render_prompt_ids(
        tok,
        h5.student_user_text(row),
    )

    def forward(prompt_ids: list[int]):
        require(
            len(prompt_ids) + len(response_ids) <= h5.MAX_TOTAL_LENGTH,
            f"teacher audit overflow row={row_id(row)}",
        )
        seq = prompt_ids + response_ids
        ids = torch.tensor([seq], dtype=torch.long, device=device)
        mask = torch.ones_like(ids)
        with torch.no_grad():
            logits = teacher(
                input_ids=ids,
                attention_mask=mask,
            ).logits[0].float()
        start = len(prompt_ids) - 1
        end = start + len(response_ids)
        return logits[start:end, :]

    hlog = forward(hint_prompt)
    nlog = forward(nohint_prompt)

    require(
        hlog.shape[0] == nlog.shape[0] == len(response_ids),
        f"audit token alignment fail row={row_id(row)}",
    )

    hvals, hinds = torch.topk(
        hlog,
        k=TEACHER_TOPK,
        dim=-1,
    )
    hlogp_top = F.log_softmax(hvals, dim=-1)
    hprob_top = hlogp_top.exp()

    nlog_lse = torch.logsumexp(
        nlog,
        dim=-1,
        keepdim=True,
    )
    nlogp_on_hint_support = torch.gather(
        nlog,
        dim=-1,
        index=hinds,
    ) - nlog_lse

    token_gap = (
        hprob_top
        * (hlogp_top - nlogp_on_hint_support)
    ).sum(dim=-1)

    result: dict[str, Any] = {
        "hint_gap_top32_mean": float(token_gap.mean().cpu()),
        "hint_gap_top32_sum": float(token_gap.sum().cpu()),
        "tokens": len(response_ids),
    }

    if row["domain"] == "chemistry":
        en_name = row["lexical_record"]["en_name"].strip()
        event = lexical_event(tok, en_name, response_ids)
        result["lexical_event"] = event

        if event["covered"]:
            pos = int(event["position"])
            length = int(event["length"])
            actual = torch.tensor(
                response_ids[pos:pos + length],
                dtype=torch.long,
                device=device,
            ).unsqueeze(-1)

            hfull = F.log_softmax(
                hlog[pos:pos + length, :],
                dim=-1,
            )
            nfull = F.log_softmax(
                nlog[pos:pos + length, :],
                dim=-1,
            )

            h_lp = torch.gather(
                hfull,
                dim=-1,
                index=actual,
            ).squeeze(-1)
            n_lp = torch.gather(
                nfull,
                dim=-1,
                index=actual,
            ).squeeze(-1)

            result["lexical_event"].update(
                {
                    "hint_logp_sum": float(h_lp.sum().cpu()),
                    "nohint_logp_sum": float(n_lp.sum().cpu()),
                    "delta_logp": float((h_lp - n_lp).sum().cpu()),
                    "delta_logp_mean_per_target_token": float(
                        (h_lp - n_lp).mean().cpu()
                    ),
                }
            )

    del hlog, nlog
    return result


def summarize_signal(rows: list[dict[str, Any]], domain: str) -> dict[str, Any]:
    import statistics

    by_arm: dict[str, dict[str, Any]] = {}
    for arm in ("S", "T"):
        arm_rows = [r for r in rows if r["arm"] == arm]
        require(arm_rows, f"no audit rows for arm={arm}")

        gaps = [float(r["signal"]["hint_gap_top32_mean"]) for r in arm_rows]
        summary: dict[str, Any] = {
            "rows": len(arm_rows),
            "mean_hint_gap_top32": statistics.fmean(gaps),
            "median_hint_gap_top32": statistics.median(gaps),
        }

        if domain == "chemistry":
            covered = [
                r for r in arm_rows
                if r["signal"].get("lexical_event", {}).get("covered")
            ]
            deltas = [
                float(r["signal"]["lexical_event"]["delta_logp"])
                for r in covered
            ]
            summary.update(
                {
                    "lexical_event_covered_rows": len(covered),
                    "lexical_event_coverage": len(covered) / len(arm_rows),
                    "mean_lexical_delta_logp":
                        statistics.fmean(deltas) if deltas else None,
                    "median_lexical_delta_logp":
                        statistics.median(deltas) if deltas else None,
                }
            )

        by_arm[arm] = summary

    return {
        "status": "MEASURED_NO_CAUSAL_DECISION",
        "domain": domain,
        "arms": by_arm,
        "T_minus_S_mean_hint_gap_top32":
            by_arm["T"]["mean_hint_gap_top32"]
            - by_arm["S"]["mean_hint_gap_top32"],
        "interpretation":
            "Measurement only. No comparability threshold is invented here.",
    }


def build_pairs(
    run: Path,
    domain: str,
    rows: list[dict[str, Any]],
    s_bank: list[dict[str, Any]],
    t_bank: list[dict[str, Any]],
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    require(len(rows) == len(s_bank) == len(t_bank), "pair bank length mismatch")

    pairs: list[dict[str, Any]] = []
    for row, srec, trec in zip(rows, s_bank, t_bank):
        rid = row_id(row)
        require(srec["row_id"] == trec["row_id"] == rid, f"pair row mismatch {rid}")

        sids = [int(x) for x in srec["prefix_ids"]]
        tids = [int(x) for x in trec["prefix_ids"]]
        m_i = min(len(sids), len(tids))
        require(m_i > 0, f"zero matched prefix row={rid}")

        s_used = sids[:m_i]
        t_used = tids[:m_i]

        pairs.append(
            {
                "status": "PASS",
                "row_id": rid,
                "domain": domain,
                "job_id": int(row["job_id"]),
                "src_text": row["src_text"],
                "src_term": row["src_term"],
                "knowledge": (
                    row["lexical_record"]["en_name"].strip()
                    if domain == "chemistry"
                    else row["definition"].strip()
                ),
                "m_i": m_i,
                "S": {
                    "full_prefix_sha256": srec["prefix_sha256"],
                    "full_prefix_token_count": len(sids),
                    "used_prefix_ids": s_used,
                    "used_prefix_sha256": sha256_json(s_used),
                },
                "T": {
                    "full_prefix_sha256": trec["prefix_sha256"],
                    "full_prefix_token_count": len(tids),
                    "used_prefix_ids": t_used,
                    "used_prefix_sha256": sha256_json(t_used),
                },
            }
        )

    pair_path = run / "paired" / f"{domain}_pairs_first{len(rows)}.jsonl"
    write_jsonl_atomic(pair_path, pairs)

    ids = [p["row_id"] for p in pairs]
    hashes = {
        "domain": domain,
        "rows": len(rows),
        "row_set_sha256": sha256_json(sorted(ids)),
        "row_order_sha256": sha256_json(ids),
        "student_prefix_bank_sha256":
            sha256_file(run / "banks" / domain / "S.jsonl"),
        "teacher_prefix_bank_sha256":
            sha256_file(run / "banks" / domain / "T.jsonl"),
        "paired_file_sha256": sha256_file(pair_path),
        "matched_S_used_ids_sha256":
            sha256_json([p["S"]["used_prefix_ids"] for p in pairs]),
        "matched_T_used_ids_sha256":
            sha256_json([p["T"]["used_prefix_ids"] for p in pairs]),
    }
    atomic_json(
        run / "paired" / f"{domain}_hashes_first{len(rows)}.json",
        hashes,
    )
    return pair_path, pairs, hashes


def run_prepare(args) -> None:
    import torch
    import torch_npu  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer

    contract = static_contract()
    h5 = load_module()
    frozen = h5.verify_frozen_inputs()

    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)

    run_manifest_path = run / "prepare_manifest.json"
    expected_manifest = {
        "status": "RUNNING",
        "scientific_class": "DIAGNOSTIC ONLY / OFFLINE PREFIX-SUPPORT REPLAY",
        "parameter_updates": False,
        "repo_head": EXPECTED_REPO_HEAD,
        "spec_sha256": EXPECTED_SPEC_SHA,
        "h5_source_sha256": EXPECTED_H5_SHA,
        "prefix_generation_seed": PREFIX_GENERATION_SEED,
        "training_seed_reserved": TRAINING_SEED,
        "n_per_domain": args.n,
        "domains": ["chemistry", "idiom"],
        "student_device": args.student_device,
        "teacher_device": args.teacher_device,
        "frozen_inputs": frozen,
        "contract": contract,
    }

    if run_manifest_path.exists():
        old = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        for key in (
            "scientific_class",
            "parameter_updates",
            "repo_head",
            "spec_sha256",
            "h5_source_sha256",
            "prefix_generation_seed",
            "training_seed_reserved",
            "n_per_domain",
            "domains",
            "student_device",
            "teacher_device",
        ):
            require(
                old.get(key) == expected_manifest.get(key),
                f"resume manifest drift key={key}: {old.get(key)!r} != "
                f"{expected_manifest.get(key)!r}",
            )
    else:
        expected_manifest["created_utc"] = now_utc()
        atomic_json(run_manifest_path, expected_manifest)

    print("Question: Does frozen Teacher-supported replay expose stronger transferable teacher signal than frozen Student-supported replay?", flush=True)
    print("Competing explanations: prefix/state support vs Teacher-target-strength confound vs objective inefficiency.", flush=True)
    print("Falsifiable prediction: T>S with comparable Teacher hint signal supports the support hypothesis; stronger T hint signal remains confounded.", flush=True)
    print("Decision after result: no formal training is authorized by this script.", flush=True)
    print("PARAMETER_UPDATES=FALSE", flush=True)

    torch.npu.set_device(args.student_device)

    st = AutoTokenizer.from_pretrained(h5.C0)
    tt = AutoTokenizer.from_pretrained(h5.TEACHER)
    require(h5.tokenizer_vocab_hash(st) == h5.tokenizer_vocab_hash(tt), "tokenizer vocab mismatch")
    require(st.eos_token_id == tt.eos_token_id, "EOS mismatch")

    student = AutoModelForCausalLM.from_pretrained(
        h5.C0,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(args.student_device).eval()
    student.config.use_cache = True
    for p in student.parameters():
        p.requires_grad_(False)

    teacher = AutoModelForCausalLM.from_pretrained(
        h5.TEACHER,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(args.teacher_device).eval()
    teacher.config.use_cache = True
    for p in teacher.parameters():
        p.requires_grad_(False)

    all_summaries: dict[str, Any] = {}

    for domain in ("chemistry", "idiom"):
        rows = load_domain_rows(h5, domain, args.n)
        bank_dir = run / "banks" / domain
        s_path = bank_dir / "S.jsonl"
        t_path = bank_dir / "T.jsonl"

        print(f"{now_utc()} START_BANK domain={domain} arm=S", flush=True)
        s_bank = fill_bank(
            student, st, h5, rows, "S", args.student_device, s_path
        )

        print(f"{now_utc()} START_BANK domain={domain} arm=T", flush=True)
        t_bank = fill_bank(
            teacher, tt, h5, rows, "T", args.teacher_device, t_path
        )

        pair_path, pairs, hashes = build_pairs(
            run, domain, rows, s_bank, t_bank
        )

        audit_path = run / "teacher_signal" / f"{domain}_first{args.n}.jsonl"
        existing = {
            (str(r["row_id"]), str(r["arm"])): r
            for r in read_jsonl(audit_path)
            if r.get("status") == "PASS"
        }

        started = time.time()
        total = len(pairs) * 2
        completed = 0

        for pair, row in zip(pairs, rows):
            for arm in ("S", "T"):
                key = (pair["row_id"], arm)
                if key not in existing:
                    used_ids = pair[arm]["used_prefix_ids"]
                    signal = teacher_signal_one(
                        teacher,
                        tt,
                        h5,
                        row,
                        used_ids,
                        args.teacher_device,
                    )
                    rec = {
                        "status": "PASS",
                        "row_id": pair["row_id"],
                        "domain": domain,
                        "job_id": pair["job_id"],
                        "arm": arm,
                        "prefix_sha256": pair[arm]["used_prefix_sha256"],
                        "prefix_token_count": pair["m_i"],
                        "first_token_ids": used_ids[:8],
                        "last_token_ids": used_ids[-8:],
                        "teacher_hint_present": True,
                        "student_hint_present": False,
                        "signal": signal,
                        "created_utc": now_utc(),
                    }
                    append_jsonl_fsync(audit_path, rec)
                    existing[key] = rec

                completed += 1
                if completed % 16 == 0 or completed == total:
                    elapsed = max(time.time() - started, 1e-9)
                    print(
                        f"{now_utc()} TEACHER_SIGNAL domain={domain} "
                        f"progress={completed}/{total} elapsed={elapsed/60:.1f}m",
                        flush=True,
                    )

        ordered_audit = [
            existing[(pair["row_id"], arm)]
            for pair in pairs
            for arm in ("S", "T")
        ]
        summary = summarize_signal(ordered_audit, domain)
        summary["hashes"] = hashes
        summary["paired_file"] = str(pair_path)
        summary["parameter_updates"] = False

        if domain == "idiom":
            judge_rows = []
            for pair, row, srec, trec in zip(pairs, rows, s_bank, t_bank):
                m_i = pair["m_i"]
                judge_rows.append(
                    {
                        "row_id": pair["row_id"],
                        "job_id": pair["job_id"],
                        "src_text": row["src_text"],
                        "src_term": row["src_term"],
                        "definition": row["definition"],
                        "S_prefix_text": st.decode(
                            pair["S"]["used_prefix_ids"],
                            skip_special_tokens=True,
                        ),
                        "T_prefix_text": tt.decode(
                            pair["T"]["used_prefix_ids"],
                            skip_special_tokens=True,
                        ),
                        "m_i": m_i,
                    }
                )
            judge_path = (
                run
                / "teacher_signal"
                / f"idiom_semantic_judge_payload_first{args.n}.jsonl"
            )
            write_jsonl_atomic(judge_path, judge_rows)
            summary["semantic_judge"] = {
                "status": "PENDING_EXTERNAL_FROZEN_JUDGE",
                "payload": str(judge_path),
                "claim_boundary":
                    "Internal Teacher hint-gap is recorded but is not substituted "
                    "for the preregistered semantic/judge analogue.",
            }

        atomic_json(
            run / "teacher_signal" / f"{domain}_summary_first{args.n}.json",
            summary,
        )
        all_summaries[domain] = summary

    final = {
        "status": "PASS_NO_PARAMETER_UPDATE",
        "scientific_class": "DIAGNOSTIC ONLY / OFFLINE PREFIX-SUPPORT REPLAY",
        "n_per_domain": args.n,
        "parameter_updates": False,
        "prefix_generation_seed": PREFIX_GENERATION_SEED,
        "training_seed_reserved": TRAINING_SEED,
        "chemistry_teacher_signal": all_summaries["chemistry"],
        "idiom_teacher_signal": all_summaries["idiom"],
        "formal_training_authorized": False,
        "formal_training_blockers": [
            "64-row native-Verl replay smoke not yet implemented/passed",
            "Idiom preregistered external semantic/judge analogue is pending",
            "practical-effect floor not yet computed from existing per-example outputs",
        ],
        "created_utc": now_utc(),
    }
    atomic_json(run / "prepare64_summary.json", final)

    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "PASS_NO_PARAMETER_UPDATE"
    manifest["completed_utc"] = now_utc()
    manifest["formal_training_authorized"] = False
    atomic_json(run_manifest_path, manifest)

    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    print("FINAL_RESULT=OFFLINE_PREFIX_SUPPORT_PREPARE64_PASS_NO_PARAMETER_UPDATE", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--student-device", default="npu:0")
    ap.add_argument("--teacher-device", default="npu:1")
    args = ap.parse_args()

    require(1 <= args.n <= 5500, f"bad n={args.n}")
    run_prepare(args)


if __name__ == "__main__":
    main()
