#!/usr/bin/env python3
"""Finalize the V2R4 targeted-context artifact after the 9-row construction gate.

Design goals:
- parent V2R4 is immutable;
- reconstruct V2R3 -> V2R4 final-row provenance before freezing;
- repair only rows selected by audit_bad_construction_v2r4.jsonl;
- persist each accepted repair immediately and atomically;
- print timestamp / done / total / throughput / ETA;
- maintain progress.json and append-only repair provenance;
- resume safely after interruption;
- re-run the full 23k hard gates and near-duplicate audit;
- never use Student outputs or downstream scores.
"""

import argparse
import hashlib
import json
import os
import re
import threading
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone, timedelta
from difflib import SequenceMatcher
from pathlib import Path

SLOT = "[[TERM]]"
TZ8 = timezone(timedelta(hours=8))
META = (
    "这个任务", "根据你的要求", "按照规则", "占位符", "我将",
    "需要构造", "生成一句", "请生成", "目标词", "具体字面形式",
)

CHEM_FRAMES = {
    "train": [
        "实验人员在本次检测中记录了[[TERM]]，并继续分析样品的相关性质。",
        "生产记录中提到了[[TERM]]，技术人员随后核对了对应的工艺参数。",
        "监测报告列出了[[TERM]]，并记录了样品中的相关检测结果。",
        "研究人员在材料分析中讨论了[[TERM]]，并比较了不同条件下的性能。",
    ],
    "uc": [
        "研究人员在另一项实验中检测了[[TERM]]，并记录了不同条件下的测量结果。",
        "技术报告在样品分析部分提到了[[TERM]]，随后讨论了相关检测指标。",
        "实验室在复核过程中记录了[[TERM]]，同时检查了相关的实验参数。",
        "研究报告在另一种应用条件下讨论了[[TERM]]，并给出了对应观测结果。",
    ],
    "uw": [
        "研究人员在测试样品时检测了[[TERM]]，并记录了相关实验结果。",
        "分析报告首次记录了[[TERM]]，随后对样品性质进行了进一步检测。",
        "实验室在该样品中发现了[[TERM]]，并继续核对相关测量数据。",
        "研究记录提到了[[TERM]]，技术人员随后完成了对应的分析步骤。",
    ],
}

IDIOM_SCENES = [
    "校园讨论", "职场沟通", "新闻评论", "家庭生活",
    "历史叙述", "社会观察", "文学叙事", "日常对话",
]


def now_iso():
    return datetime.now(TZ8).isoformat(timespec="seconds")


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def atomic_write_text(path, text):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_jsonl(path, rows):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
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


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_index(s, n):
    d = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(d[:8], "big") % n


def normalized_text(text):
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE).lower()


def looks_meta(text):
    return any(x in text for x in META)


def canonical_ok(row):
    term = row["src_term"]
    src = row.get("src_text", "")
    templ = row.get("context_template", "")
    return (
        bool(term)
        and src.count(term) == 1
        and SLOT not in src
        and templ.count(SLOT) == 1
        and term not in templ
        and not looks_meta(src)
    )


def build_candidate(row, template, mode):
    term = row["src_term"]
    if template.count(SLOT) != 1 or term in template or looks_meta(template):
        return None
    src = template.replace(SLOT, term, 1)
    nr = dict(row)
    nr.update(
        context_template=template,
        src_text=src,
        valid_placeholder=True,
        valid_exact_term=True,
        repair_mode=mode,
    )
    return nr if canonical_ok(nr) else None


def load_all(out, n=16):
    rows = []
    for sid in range(n):
        p = out / "shards" / f"part_{sid:02d}.jsonl"
        if not p.exists():
            raise SystemExit(f"MISSING_SHARD {p}")
        rows.extend(read_jsonl(p))
    rows.sort(key=lambda r: r["job_id"])
    return rows


def patch_one_row(out, job_id, new_row, n=16):
    hits = 0
    for sid in range(n):
        p = out / "shards" / f"part_{sid:02d}.jsonl"
        rows = read_jsonl(p)
        changed = False
        for i, r in enumerate(rows):
            if r["job_id"] == job_id:
                rows[i] = new_row
                hits += 1
                changed = True
        if changed:
            write_jsonl(p, rows)
            print(f"{now_iso()} durable_patch shard={sid} job_id={job_id}", flush=True)
    if hits != 1:
        raise SystemExit(f"PATCH_IDENTITY_FAIL job_id={job_id} hits={hits}")


def contamination_hits(row, unseen_terms):
    residual = row["src_text"].replace(row["src_term"], "", 1)
    return [u for u in unseen_terms if u and u in residual]


def exact_counterpart_texts(rows, row):
    return {
        r["src_text"]
        for r in rows
        if r["domain"] == row["domain"]
        and r["entity_key"] == row["entity_key"]
        and r["split"] != row["split"]
        and r["split"] in ("train", "uc")
    }


def dictionary_example_candidate(row):
    rec = row.get("lexical_record") or {}
    ex = str(rec.get("example") or "").strip()
    term = row["src_term"]
    if not ex or ex == "无":
        return None
    ex = ex.replace("～", term).replace("~", term)
    ex = re.split(r"[★☆]", ex, maxsplit=1)[0].strip().strip('“”"\' ')
    if ex.count(term) != 1 or SLOT in ex or looks_meta(ex):
        return None
    return build_candidate(
        row, ex.replace(term, SLOT, 1), "v2r5_dictionary_example"
    )


def parse_lr(raw):
    raw = raw.strip()
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    left, right = str(obj.get("left", "")).strip(), str(obj.get("right", "")).strip()
    if not left and not right:
        return None
    return left, right


def idiom_prompt(row, variant):
    definition = row.get("definition", "").strip()
    scene = IDIOM_SCENES[variant % len(IDIOM_SCENES)]
    mode = [
        "写成简洁自然的叙事句",
        "写成自然的人物评价句",
        "写成自然的事件描述句",
        "写成自然的评论句",
    ][(variant // len(IDIOM_SCENES)) % 4]
    return (
        "你要构造一句中文成语使用语境。我不会告诉你成语的字面形式。"
        f"它的字典释义是：{definition}。语境类型：{scene}。{mode}。"
        "把句子分成成语前面的 left 和成语后面的 right。"
        "不要写出或猜测成语，不要使用任何占位符，不要解释任务。"
        '只输出一行严格JSON：{"left":"成语前面的文字","right":"成语后面的文字"}'
    )


class Heartbeat:
    def __init__(self, label, every=30):
        self.label = label
        self.every = every
        self.stop_event = threading.Event()
        self.thread = None

    def __enter__(self):
        def run():
            started = time.time()
            while not self.stop_event.wait(self.every):
                print(
                    f"{now_iso()} heartbeat phase={self.label} "
                    f"elapsed_call={int(time.time()-started)}s",
                    flush=True,
                )
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1)


class IdiomGenerator:
    def __init__(self, model_path, device):
        import torch
        import torch_npu
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.dev = f"npu:{device}"
        torch.npu.set_device(self.dev)
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(self.dev).eval()

    def infer(self, prompt, label):
        rendered = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        batch = self.tok(rendered, return_tensors="pt")
        batch = {k: v.to(self.dev) for k, v in batch.items()}
        with Heartbeat(label):
            with self.torch.no_grad():
                ids = self.model.generate(
                    **batch,
                    max_new_tokens=128,
                    do_sample=False,
                    use_cache=True,
                )
        gen = ids[:, batch["input_ids"].shape[1]:]
        return self.tok.batch_decode(gen, skip_special_tokens=True)[0]


def regenerate_row(row, rows, unseen_terms, idiom_gen, max_attempts=16):
    forbidden = exact_counterpart_texts(rows, row)
    forbidden_norm = {normalized_text(x) for x in forbidden}

    if row["domain"] == "chemistry":
        frames = CHEM_FRAMES[row["split"]]
        start = stable_index(f"{row['job_id']}/{row['split']}", len(frames))
        for off in range(len(frames)):
            cand = build_candidate(
                row,
                frames[(start + off) % len(frames)],
                "v2r5_chemistry_frame",
            )
            if cand is None:
                continue
            if cand["src_text"] in forbidden or normalized_text(cand["src_text"]) in forbidden_norm:
                continue
            if contamination_hits(cand, unseen_terms):
                continue
            return cand
        return None

    ex = dictionary_example_candidate(row)
    if ex is not None:
        if (
            ex["src_text"] not in forbidden
            and normalized_text(ex["src_text"]) not in forbidden_norm
            and not contamination_hits(ex, unseen_terms)
        ):
            return ex

    if idiom_gen is None:
        return None

    seed = stable_index(f"{row['job_id']}/{row['split']}", 1000000)
    for attempt in range(max_attempts):
        variant = seed + attempt
        raw = idiom_gen.infer(
            idiom_prompt(row, variant),
            f"repair_job_{row['job_id']}_attempt_{attempt+1}",
        )
        lr = parse_lr(raw)
        if not lr:
            continue
        left, right = lr
        if row["src_term"] in left or row["src_term"] in right:
            continue
        if SLOT in left or SLOT in right or looks_meta(left + right):
            continue
        cand = build_candidate(
            row, left + SLOT + right, "v2r5_hidden_term_left_right"
        )
        if cand is None:
            continue
        if cand["src_text"] in forbidden or normalized_text(cand["src_text"]) in forbidden_norm:
            continue
        if contamination_hits(cand, unseen_terms):
            continue
        if 6 <= len(cand["src_text"]) <= 220:
            cand["generation_attempts"] = attempt + 1
            return cand
    return None


def progress_write(out, status, phase, done, total, started, ok, failed, last_job=None):
    elapsed = max(time.time() - started, 1e-9)
    rate = done / elapsed if done else 0.0
    eta = ((total - done) / rate) if rate > 0 else None
    obj = {
        "status": status,
        "phase": phase,
        "done": done,
        "total": total,
        "percentage": round(100.0 * done / total, 2) if total else 100.0,
        "success": ok,
        "failed": failed,
        "elapsed_seconds": round(elapsed, 1),
        "rate_rows_per_sec": round(rate, 4),
        "eta_seconds": round(eta, 1) if eta is not None else None,
        "last_job_id": last_job,
        "last_update": now_iso(),
    }
    atomic_write_text(out / "progress.json", json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    eta_s = "?" if eta is None else f"{eta/60:.1f}m"
    print(
        f"{obj['last_update']} phase={phase} done={done}/{total} "
        f"pct={obj['percentage']:.2f}% ok={ok} fail={failed} "
        f"rate={rate:.3f} rows/s elapsed={elapsed/60:.1f}m ETA={eta_s} "
        f"last_job={last_job}",
        flush=True,
    )


def reconstruct_parent_diff(parent, out, n=16):
    prov = out / "repairs" / "v2r4_reconstructed_parent_diff.jsonl"
    if prov.exists():
        rows = read_jsonl(prov)
        print(f"{now_iso()} provenance_reuse v2r3_to_v2r4_rows={len(rows)}", flush=True)
        return rows

    pmap = {r["job_id"]: r for r in load_all(parent, n)}
    cmap = {r["job_id"]: r for r in load_all(out, n)}
    if set(pmap) != set(cmap):
        raise SystemExit("PARENT_CHILD_JOB_ID_MISMATCH")

    diffs = []
    fields = ("src_text", "context_template", "repair_mode", "generation_attempts")
    for jid in sorted(pmap):
        a, b = pmap[jid], cmap[jid]
        if any(a.get(k) != b.get(k) for k in fields):
            diffs.append({
                "job_id": jid,
                "domain": b["domain"],
                "split": b["split"],
                "entity_key": b["entity_key"],
                "src_term": b["src_term"],
                "parent_src_text": a.get("src_text", ""),
                "child_src_text": b.get("src_text", ""),
                "parent_context_template": a.get("context_template", ""),
                "child_context_template": b.get("context_template", ""),
                "child_repair_mode": b.get("repair_mode"),
                "provenance_kind": "reconstructed_final_parent_child_diff",
            })
    write_jsonl(prov, diffs)
    print(f"{now_iso()} provenance_written v2r3_to_v2r4_rows={len(diffs)}", flush=True)
    return diffs


def hard_checks(rows, out):
    if len(rows) != 23000 or len({r["job_id"] for r in rows}) != 23000:
        raise SystemExit("CARDINALITY_FAIL")

    bad = [r for r in rows if not canonical_ok(r)]
    if bad:
        write_jsonl(out / "audit_bad_construction_v2r5.jsonl", bad)
        raise SystemExit(f"CONSTRUCTION_FAIL n={len(bad)}")

    expected = {
        ("chemistry", "train"): 5500, ("chemistry", "uc"): 5500, ("chemistry", "uw"): 500,
        ("idiom", "train"): 5500, ("idiom", "uc"): 5500, ("idiom", "uw"): 500,
    }
    counts = Counter((r["domain"], r["split"]) for r in rows)
    if counts != Counter(expected):
        raise SystemExit(f"SPLIT_COUNT_FAIL got={dict(counts)}")

    exact_dup, norm_dup = [], []
    for domain in ("chemistry", "idiom"):
        tr = {r["entity_key"]: r for r in rows if r["domain"] == domain and r["split"] == "train"}
        uc = {r["entity_key"]: r for r in rows if r["domain"] == domain and r["split"] == "uc"}
        for key, u in uc.items():
            t = tr[key]
            if t["src_text"] == u["src_text"]:
                exact_dup.append(u)
            elif normalized_text(t["src_text"]) == normalized_text(u["src_text"]):
                norm_dup.append(u)
    if exact_dup:
        write_jsonl(out / "audit_train_uc_exact_duplicates_v2r5.jsonl", exact_dup)
        raise SystemExit(f"TRAIN_UC_EXACT_DUP_FAIL n={len(exact_dup)}")
    if norm_dup:
        write_jsonl(out / "audit_train_uc_normalized_duplicates_v2r5.jsonl", norm_dup)
        raise SystemExit(f"TRAIN_UC_NORMALIZED_DUP_FAIL n={len(norm_dup)}")

    contamination = []
    for domain in ("chemistry", "idiom"):
        unseen = sorted(
            {r["src_term"] for r in rows if r["domain"] == domain and r["split"] == "uw"},
            key=len, reverse=True,
        )
        for r in rows:
            if r["domain"] == domain and r["split"] in ("train", "uc"):
                hits = contamination_hits(r, unseen)
                if hits:
                    contamination.append({
                        "job_id": r["job_id"],
                        "domain": domain,
                        "split": r["split"],
                        "src_term": r["src_term"],
                        "src_text": r["src_text"],
                        "unseen_hits": hits[:20],
                    })
    if contamination:
        write_jsonl(out / "audit_seen_context_unseen_contamination_v2r5.jsonl", contamination)
        raise SystemExit(f"UNSEEN_CONTAMINATION_FAIL n={len(contamination)}")


def near_duplicate_audit(rows, out):
    items = []
    for domain in ("chemistry", "idiom"):
        tr = {r["entity_key"]: r for r in rows if r["domain"] == domain and r["split"] == "train"}
        uc = {r["entity_key"]: r for r in rows if r["domain"] == domain and r["split"] == "uc"}
        for key, u in uc.items():
            t = tr[key]
            a, b = normalized_text(t["src_text"]), normalized_text(u["src_text"])
            ratio = SequenceMatcher(None, a, b, autojunk=False).ratio()
            items.append({
                "domain": domain,
                "entity_key": key,
                "src_term": u["src_term"],
                "similarity_ratio": round(ratio, 6),
                "train_text": t["src_text"],
                "uc_text": u["src_text"],
            })
    items.sort(key=lambda x: x["similarity_ratio"], reverse=True)
    write_jsonl(out / "near_duplicate_audit_top100.jsonl", items[:100])
    vals = sorted(x["similarity_ratio"] for x in items)

    def q(p):
        i = min(len(vals)-1, max(0, round((len(vals)-1)*p)))
        return vals[i]

    summary = {
        "metric": "SequenceMatcher on NFKC + punctuation/whitespace stripped text",
        "policy": "report-only; no arbitrary threshold used for filtering",
        "pairs": len(items),
        "max": vals[-1],
        "p99": q(0.99),
        "p95": q(0.95),
        "p90": q(0.90),
        "median": q(0.50),
        "top100_file": "near_duplicate_audit_top100.jsonl",
    }
    atomic_write_text(
        out / "near_duplicate_audit_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    return summary


def export_manifest(rows, out, model, parent_diff_count, repair_count, near):
    for domain in ("chemistry", "idiom"):
        for split in ("train", "uc", "uw"):
            grp = [r for r in rows if r["domain"] == domain and r["split"] == split]
            write_jsonl(out / f"{domain}_{split}.jsonl", grp)

    manifest = {
        "status": "CONTEXTS_FROZEN",
        "artifact_version": "targeted-section43-contexts-qwen3-8b-v2r5",
        "total": 23000,
        "counts": {
            "chemistry/train": 5500, "chemistry/uc": 5500, "chemistry/uw": 500,
            "idiom/train": 5500, "idiom/uc": 5500, "idiom/uw": 500,
        },
        "frozen_lexical_identity_changed": False,
        "frozen_seen_unseen_split_changed": False,
        "generator_role": "Synthesis Model",
        "generator_model": model,
        "enable_thinking": False,
        "hard_checks": {
            "construction_failures": 0,
            "train_uc_exact_duplicates": 0,
            "train_uc_normalized_duplicates": 0,
            "seen_context_unseen_term_contamination": 0,
        },
        "near_duplicate_audit": near,
        "provenance": {
            "v2r3_to_v2r4_reconstructed_final_diffs": parent_diff_count,
            "v2r5_incremental_repairs": repair_count,
            "selection_basis": "protocol validity only; no Student/evaluator outcome used",
            "parent_diff_file": "repairs/v2r4_reconstructed_parent_diff.jsonl",
            "v2r5_repair_file": "repairs/v2r5_repair_provenance.jsonl",
        },
        "files": {},
    }

    for p in sorted(out.glob("*.jsonl")):
        if p.name.startswith("audit_"):
            continue
        manifest["files"][p.name] = sha256_file(p)
    for p in sorted(out.glob("*.json")):
        if p.name in ("context_freeze_manifest.json", "progress.json"):
            continue
        manifest["files"][p.name] = sha256_file(p)

    atomic_write_text(
        out / "context_freeze_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--parent-v2r3", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.out)
    parent = Path(args.parent_v2r3)
    audit = out / "audit_bad_construction_v2r4.jsonl"
    targets = read_jsonl(audit)
    if not (1 <= len(targets) <= 32):
        raise SystemExit(f"UNEXPECTED_CONSTRUCTION_AUDIT_SIZE n={len(targets)}")

    parent_diffs = reconstruct_parent_diff(parent, out, args.num_shards)

    repair_path = out / "repairs" / "v2r5_repair_provenance.jsonl"
    completed = set()
    if repair_path.exists():
        completed = {r["job_id"] for r in read_jsonl(repair_path)}

    target_map = {r["job_id"]: r for r in targets}
    rows = load_all(out, args.num_shards)

    unseen_by_domain = {}
    for domain in ("chemistry", "idiom"):
        unseen_by_domain[domain] = sorted(
            {r["src_term"] for r in rows if r["domain"] == domain and r["split"] == "uw"},
            key=len, reverse=True,
        )

    pending = [jid for jid in sorted(target_map) if jid not in completed]
    need_idiom_model = any(target_map[j]["domain"] == "idiom" for j in pending)
    idiom_gen = IdiomGenerator(args.model, args.device) if need_idiom_model else None

    started = time.time()
    total = len(targets)
    done = len(completed)
    ok = done
    failed = 0
    progress_write(out, "RUNNING", "repair_construction", done, total, started, ok, failed)

    for jid in pending:
        rows = load_all(out, args.num_shards)
        current = next(r for r in rows if r["job_id"] == jid)
        cand = regenerate_row(
            current, rows, unseen_by_domain[current["domain"]], idiom_gen
        )
        if cand is None:
            failed += 1
            rec = {
                "timestamp": now_iso(),
                "job_id": jid,
                "domain": current["domain"],
                "split": current["split"],
                "src_term": current["src_term"],
                "status": "FAILED",
                "reason": "no valid repair candidate",
            }
            append_jsonl(out / "repairs" / "v2r5_failed.jsonl", rec)
            done += 1
            progress_write(out, "RUNNING", "repair_construction", done, total, started, ok, failed, jid)
            continue

        before = {
            "src_text": current.get("src_text", ""),
            "context_template": current.get("context_template", ""),
            "repair_mode": current.get("repair_mode"),
        }
        patch_one_row(out, jid, cand, args.num_shards)
        append_jsonl(
            repair_path,
            {
                "timestamp": now_iso(),
                "job_id": jid,
                "domain": cand["domain"],
                "split": cand["split"],
                "entity_key": cand["entity_key"],
                "src_term": cand["src_term"],
                "before": before,
                "after": {
                    "src_text": cand["src_text"],
                    "context_template": cand["context_template"],
                    "repair_mode": cand.get("repair_mode"),
                },
                "selection_reason": "audit_bad_construction_v2r4",
            },
        )
        done += 1
        ok += 1
        progress_write(out, "RUNNING", "repair_construction", done, total, started, ok, failed, jid)

    if failed:
        progress_write(out, "FAIL", "repair_construction", done, total, started, ok, failed)
        raise SystemExit(f"REPAIR_FAILED n={failed}")

    rows = load_all(out, args.num_shards)
    progress_write(out, "RUNNING", "full_hard_checks", total, total, started, ok, failed)
    hard_checks(rows, out)

    # Hard checks passed. Archive stale audit artifacts inherited from failed parents
    # so the frozen root contains no misleading historical FAIL markers.
    legacy_dir = out / "repairs" / "legacy_audits"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    for apath in sorted(out.glob("audit_*.jsonl")):
        target = legacy_dir / apath.name
        if target.exists():
            target = legacy_dir / (apath.stem + ".archived" + apath.suffix)
        os.replace(apath, target)
        print(f"{now_iso()} archived_legacy_audit {apath.name} -> {target}", flush=True)

    progress_write(out, "RUNNING", "near_duplicate_audit", total, total, started, ok, failed)
    near = near_duplicate_audit(rows, out)

    repair_count = len(read_jsonl(repair_path)) if repair_path.exists() else 0
    manifest = export_manifest(
        rows, out, args.model, len(parent_diffs), repair_count, near
    )

    progress_write(out, "PASS", "frozen", total, total, started, ok, failed)
    print("FINAL_HARD_CHECKS=PASS", flush=True)
    print("NEAR_DUPLICATE_AUDIT=" + json.dumps(near, ensure_ascii=False), flush=True)
    print("TARGETED_CONTEXT_FREEZE_PASS", flush=True)
    print(json.dumps(manifest["provenance"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
