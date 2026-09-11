#!/usr/bin/env python3
"""Close the targeted Chemistry/Idiom train-vs-UC near-duplicate gate.

Scientific status:
- This is a LAB ADAPTATION, because the paper does not define an exact
  surface-similarity threshold for "near duplicate".
- The preregistered repair rule is:
    same-entity train vs UC, after masking the designated src_term,
    NFKC normalization and punctuation/whitespace removal,
    SequenceMatcher ratio >= 0.95.
- Only the UC context is regenerated.
- Frozen lexical identity, Seen/Unseen split, train contexts and UW contexts
  are never changed.
- Student outputs, evaluator scores and downstream performance are never used.

Engineering contract:
- immutable V2R5 parent;
- fixed selection snapshot from the parent;
- per-row atomic persistence;
- append-only provenance;
- resumable;
- timestamps, progress, throughput and ETA;
- heartbeat during slow model calls;
- full 23k revalidation before final freeze.
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
THRESHOLD = 0.95
TZ8 = timezone(timedelta(hours=8))
META = (
    "这个任务", "根据你的要求", "按照规则", "占位符", "我将",
    "需要构造", "生成一句", "请生成", "目标词", "具体字面形式",
)

CHEM_UC_FRAMES = [
    "研究人员在另一项实验中检测了[[TERM]]，随后记录了样品在不同条件下的测量结果。",
    "技术人员在新的质量检测中分析了[[TERM]]，并进一步核对了相关实验参数。",
    "实验室在复核样品时记录了[[TERM]]，之后比较了多组检测数据。",
    "研究报告在另一种应用条件下讨论了[[TERM]]，并给出了对应的观测结果。",
    "材料测试过程中涉及[[TERM]]，研究人员随后分析了不同处理条件产生的变化。",
    "分析人员在新的样品批次中检测了[[TERM]]，并将结果纳入后续质量评估。",
    "科研人员在独立实验中使用了[[TERM]]，随后对所得数据进行了统计分析。",
    "实验记录在另一项测试中提到了[[TERM]]，并描述了对应的检测流程。",
    "环境样品分析发现涉及[[TERM]]，技术人员随后复核了相关测量指标。",
    "产品评估过程中记录了[[TERM]]，实验人员进一步检查了其使用条件。",
    "工艺验证阶段提到了[[TERM]]，随后对关键参数和测试结果进行了复核。",
    "研究小组在后续实验中考察了[[TERM]]，并比较了不同样品之间的差异。",
]

IDIOM_SCENES = [
    "校园讨论", "职场沟通", "新闻评论", "家庭生活",
    "历史叙述", "社会观察", "文学叙事", "日常对话",
    "团队合作", "公共事务", "人物评价", "事件回顾",
]

def now_iso():
    return datetime.now(TZ8).isoformat(timespec="seconds")

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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

def norm(s):
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"[\s\W_]+", "", s, flags=re.UNICODE).lower()

def context_only(src, term):
    # The designated shared lexical item must not drive similarity.
    return norm(src.replace(term, "", 1))

def similarity(train_row, uc_row):
    a = context_only(train_row["src_text"], train_row["src_term"])
    b = context_only(uc_row["src_text"], uc_row["src_term"])
    return SequenceMatcher(None, a, b, autojunk=False).ratio()

def stable_index(s, n):
    d = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(d[:8], "big") % n

def looks_meta(text):
    return any(m in text for m in META)

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

def patch_one(out, job_id, row, n=16):
    hits = 0
    for sid in range(n):
        p = out / "shards" / f"part_{sid:02d}.jsonl"
        rows = read_jsonl(p)
        changed = False
        for i, r in enumerate(rows):
            if r["job_id"] == job_id:
                rows[i] = row
                hits += 1
                changed = True
        if changed:
            write_jsonl(p, rows)
            print(f"{now_iso()} durable_patch shard={sid} job_id={job_id}", flush=True)
    if hits != 1:
        raise SystemExit(f"PATCH_IDENTITY_FAIL job_id={job_id} hits={hits}")

def split_maps(rows, domain):
    tr = {r["entity_key"]: r for r in rows if r["domain"] == domain and r["split"] == "train"}
    uc = {r["entity_key"]: r for r in rows if r["domain"] == domain and r["split"] == "uc"}
    return tr, uc

def unseen_terms(rows, domain):
    return sorted(
        {r["src_term"] for r in rows if r["domain"] == domain and r["split"] == "uw"},
        key=len, reverse=True,
    )

def contamination_hits(row, unseen):
    residual = row["src_text"].replace(row["src_term"], "", 1)
    return [u for u in unseen if u and u in residual]

def make_parent_selection(parent, selection_path):
    rows = load_all(parent)
    selected = []
    for domain in ("chemistry", "idiom"):
        tr, uc = split_maps(rows, domain)
        for key in sorted(tr):
            s = similarity(tr[key], uc[key])
            if s >= THRESHOLD:
                selected.append({
                    "job_id": uc[key]["job_id"],
                    "domain": domain,
                    "entity_key": key,
                    "src_term": uc[key]["src_term"],
                    "parent_similarity": round(s, 6),
                    "train_text": tr[key]["src_text"],
                    "uc_text": uc[key]["src_text"],
                    "selection_rule": (
                        "same-entity term-masked context-only SequenceMatcher "
                        f"ratio >= {THRESHOLD}"
                    ),
                })
    selected.sort(key=lambda x: (-x["parent_similarity"], x["job_id"]))
    write_jsonl(selection_path, selected)
    return selected

def parse_lr(raw):
    m = re.search(r"\{.*\}", raw.strip(), re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    left = str(obj.get("left", "")).strip()
    right = str(obj.get("right", "")).strip()
    if not left and not right:
        return None
    return left, right

def idiom_prompt(row, variant):
    definition = row.get("definition", "").strip()
    scene = IDIOM_SCENES[variant % len(IDIOM_SCENES)]
    styles = [
        "写成自然的叙事句",
        "写成自然的人物评价句",
        "写成自然的事件描述句",
        "写成自然的评论句",
        "写成自然的对话转述句",
        "写成自然的社会观察句",
    ]
    style = styles[(variant // len(IDIOM_SCENES)) % len(styles)]
    return (
        "构造一句自然中文成语语境。我不会告诉你成语的字面形式。"
        f"字典释义：{definition}。语境类型：{scene}。{style}。"
        "请输出成语出现位置之前的 left 与之后的 right。"
        "不要写出或猜测成语本身，不要使用任何占位符，不要解释任务。"
        '只输出严格JSON：{"left":"成语前文字","right":"成语后文字"}'
    )

class Heartbeat:
    def __init__(self, label, every=30):
        self.label = label
        self.every = every
        self.stop = threading.Event()
        self.thread = None
    def __enter__(self):
        def run():
            start = time.time()
            while not self.stop.wait(self.every):
                print(
                    f"{now_iso()} heartbeat phase={self.label} "
                    f"call_elapsed={int(time.time()-start)}s",
                    flush=True,
                )
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        return self
    def __exit__(self, *args):
        self.stop.set()
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
                    **batch, max_new_tokens=128, do_sample=False, use_cache=True
                )
        gen = ids[:, batch["input_ids"].shape[1]:]
        return self.tok.batch_decode(gen, skip_special_tokens=True)[0]

def valid_against_contract(candidate, train_row, unseen):
    if not canonical_ok(candidate):
        return False, "construction"
    if candidate["src_text"] == train_row["src_text"]:
        return False, "exact_duplicate"
    if norm(candidate["src_text"]) == norm(train_row["src_text"]):
        return False, "normalized_duplicate"
    s = similarity(train_row, candidate)
    if s >= THRESHOLD:
        return False, f"near_duplicate_{s:.6f}"
    hits = contamination_hits(candidate, unseen)
    if hits:
        return False, "unseen_contamination"
    return True, f"similarity={s:.6f}"

def chemistry_candidate(uc_row, train_row, unseen):
    start = stable_index(f"{uc_row['entity_key']}/near-dup-final", len(CHEM_UC_FRAMES))
    for off in range(len(CHEM_UC_FRAMES)):
        cand = build_candidate(
            uc_row,
            CHEM_UC_FRAMES[(start + off) % len(CHEM_UC_FRAMES)],
            "final_near_dup_chemistry_uc_regeneration",
        )
        if cand is None:
            continue
        ok, why = valid_against_contract(cand, train_row, unseen)
        if ok:
            return cand, why
    return None, "no_valid_frame"

def idiom_candidate(uc_row, train_row, unseen, gen, max_attempts=24):
    seed = stable_index(f"{uc_row['entity_key']}/near-dup-final", 1000000)
    for attempt in range(max_attempts):
        raw = gen.infer(
            idiom_prompt(uc_row, seed + attempt),
            f"idiom_job_{uc_row['job_id']}_attempt_{attempt+1}",
        )
        lr = parse_lr(raw)
        if not lr:
            continue
        left, right = lr
        if uc_row["src_term"] in left or uc_row["src_term"] in right:
            continue
        if SLOT in left or SLOT in right or looks_meta(left + right):
            continue
        cand = build_candidate(
            uc_row,
            left + SLOT + right,
            "final_near_dup_idiom_uc_regeneration",
        )
        if cand is None:
            continue
        ok, why = valid_against_contract(cand, train_row, unseen)
        if ok and 6 <= len(cand["src_text"]) <= 220:
            cand["generation_attempts"] = attempt + 1
            return cand, why
    return None, "generation_exhausted"

def update_progress(out, status, phase, done, total, ok, failed, start, last_job=None):
    elapsed = max(time.time() - start, 1e-9)
    rate = done / elapsed if done else 0.0
    eta = (total - done) / rate if rate > 0 else None
    obj = {
        "status": status,
        "phase": phase,
        "done": done,
        "total": total,
        "percentage": round(100 * done / total, 2) if total else 100.0,
        "success": ok,
        "failed": failed,
        "elapsed_seconds": round(elapsed, 1),
        "rate_rows_per_sec": round(rate, 4),
        "eta_seconds": round(eta, 1) if eta is not None else None,
        "last_job_id": last_job,
        "last_update": now_iso(),
    }
    atomic_write_text(out / "progress.json", json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    eta_txt = "?" if eta is None else f"{eta/60:.1f}m"
    print(
        f"{obj['last_update']} phase={phase} done={done}/{total} "
        f"pct={obj['percentage']:.2f}% ok={ok} fail={failed} "
        f"elapsed={elapsed/60:.1f}m rate={rate:.3f} rows/s "
        f"ETA={eta_txt} last_job={last_job}",
        flush=True,
    )

def full_checks(rows, out):
    if len(rows) != 23000 or len({r["job_id"] for r in rows}) != 23000:
        raise SystemExit("CARDINALITY_FAIL")

    bad = [r for r in rows if not canonical_ok(r)]
    if bad:
        write_jsonl(out / "audit_bad_construction_final.jsonl", bad)
        raise SystemExit(f"CONSTRUCTION_FAIL n={len(bad)}")

    expected = Counter({
        ("chemistry", "train"): 5500, ("chemistry", "uc"): 5500, ("chemistry", "uw"): 500,
        ("idiom", "train"): 5500, ("idiom", "uc"): 5500, ("idiom", "uw"): 500,
    })
    got = Counter((r["domain"], r["split"]) for r in rows)
    if got != expected:
        raise SystemExit(f"SPLIT_COUNT_FAIL got={dict(got)}")

    exact, normalized, near = [], [], []
    for domain in ("chemistry", "idiom"):
        tr, uc = split_maps(rows, domain)
        for key in tr:
            t, u = tr[key], uc[key]
            if t["src_text"] == u["src_text"]:
                exact.append(u)
            elif norm(t["src_text"]) == norm(u["src_text"]):
                normalized.append(u)
            s = similarity(t, u)
            if s >= THRESHOLD:
                near.append({
                    "job_id": u["job_id"], "domain": domain, "entity_key": key,
                    "src_term": u["src_term"], "similarity": round(s, 6),
                    "train_text": t["src_text"], "uc_text": u["src_text"],
                })

    if exact:
        write_jsonl(out / "audit_exact_duplicates_final.jsonl", exact)
        raise SystemExit(f"EXACT_DUP_FAIL n={len(exact)}")
    if normalized:
        write_jsonl(out / "audit_normalized_duplicates_final.jsonl", normalized)
        raise SystemExit(f"NORMALIZED_DUP_FAIL n={len(normalized)}")
    if near:
        write_jsonl(out / "audit_near_duplicates_ge_095_final.jsonl", near)
        raise SystemExit(f"NEAR_DUP_GE_095_FAIL n={len(near)}")

    contamination = []
    for domain in ("chemistry", "idiom"):
        unseen = unseen_terms(rows, domain)
        for r in rows:
            if r["domain"] == domain and r["split"] in ("train", "uc"):
                hits = contamination_hits(r, unseen)
                if hits:
                    contamination.append({
                        "job_id": r["job_id"], "domain": domain, "split": r["split"],
                        "src_term": r["src_term"], "src_text": r["src_text"],
                        "unseen_hits": hits[:20],
                    })
    if contamination:
        write_jsonl(out / "audit_unseen_contamination_final.jsonl", contamination)
        raise SystemExit(f"UNSEEN_CONTAMINATION_FAIL n={len(contamination)}")

def near_audit(rows, out):
    recs = []
    for domain in ("chemistry", "idiom"):
        tr, uc = split_maps(rows, domain)
        for key in tr:
            s = similarity(tr[key], uc[key])
            recs.append({
                "domain": domain,
                "entity_key": key,
                "src_term": uc[key]["src_term"],
                "context_only_similarity": round(s, 6),
                "train_text": tr[key]["src_text"],
                "uc_text": uc[key]["src_text"],
            })
    recs.sort(key=lambda x: x["context_only_similarity"], reverse=True)
    write_jsonl(out / "near_duplicate_context_only_top100_final.jsonl", recs[:100])
    vals = sorted(r["context_only_similarity"] for r in recs)
    def q(p):
        return vals[min(len(vals)-1, max(0, round((len(vals)-1)*p)))]
    summary = {
        "metric": (
            "SequenceMatcher after designated src_term masking, then "
            "NFKC + punctuation/whitespace removal"
        ),
        "hard_repair_threshold": THRESHOLD,
        "threshold_status": "LAB ADAPTATION / preregistered before repair",
        "pairs": len(vals),
        "count_ge_threshold": sum(v >= THRESHOLD for v in vals),
        "max": vals[-1],
        "p99": q(0.99),
        "p95": q(0.95),
        "p90": q(0.90),
        "median": q(0.50),
        "top100_file": "near_duplicate_context_only_top100_final.jsonl",
    }
    atomic_write_text(
        out / "near_duplicate_context_only_summary_final.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    return summary

def export_splits(rows, out):
    for domain in ("chemistry", "idiom"):
        for split in ("train", "uc", "uw"):
            write_jsonl(
                out / f"{domain}_{split}.jsonl",
                [r for r in rows if r["domain"] == domain and r["split"] == split],
            )

def write_manifest(out, parent, model, selection, repairs, near):
    parent_files = {}
    for name in (
        "chemistry_train.jsonl", "chemistry_uc.jsonl", "chemistry_uw.jsonl",
        "idiom_train.jsonl", "idiom_uc.jsonl", "idiom_uw.jsonl",
    ):
        parent_files[name] = sha256_file(parent / name)

    final_files = {}
    for name in (
        "chemistry_train.jsonl", "chemistry_uc.jsonl", "chemistry_uw.jsonl",
        "idiom_train.jsonl", "idiom_uc.jsonl", "idiom_uw.jsonl",
        "near_duplicate_context_only_top100_final.jsonl",
        "near_duplicate_context_only_summary_final.json",
    ):
        final_files[name] = sha256_file(out / name)

    manifest = {
        "status": "TARGETED_CONTEXTS_FROZEN_CLOSED",
        "artifact_version": "targeted-section43-contexts-qwen3-8b-final-near-dup-repair-20260910",
        "scientific_class": "LAB ADAPTATION",
        "parent": str(parent),
        "generator_model": model,
        "lexical_identity_changed": False,
        "seen_unseen_split_changed": False,
        "train_contexts_changed_by_this_step": False,
        "uw_contexts_changed_by_this_step": False,
        "uc_contexts_selected_for_repair": len(selection),
        "uc_contexts_repaired": len(repairs),
        "selection_rule": {
            "metric": near["metric"],
            "threshold": THRESHOLD,
            "condition": "same-entity context-only similarity >= threshold",
            "paper_defined_threshold": False,
            "student_outputs_used": False,
            "evaluation_scores_used": False,
        },
        "hard_checks": {
            "construction_failures": 0,
            "exact_train_uc_duplicates": 0,
            "normalized_train_uc_duplicates": 0,
            "near_duplicates_ge_095": 0,
            "seen_context_unseen_term_contamination": 0,
        },
        "near_duplicate_audit": near,
        "parent_split_hashes": parent_files,
        "final_split_hashes": final_files,
        "selection_file": "repairs/near_dup_selection_from_v2r5.jsonl",
        "repair_provenance_file": "repairs/near_dup_repair_provenance.jsonl",
    }
    atomic_write_text(
        out / "context_freeze_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=16)
    args = ap.parse_args()

    parent = Path(args.parent)
    out = Path(args.out)
    selection_path = out / "repairs" / "near_dup_selection_from_v2r5.jsonl"
    provenance_path = out / "repairs" / "near_dup_repair_provenance.jsonl"

    if selection_path.exists():
        selection = read_jsonl(selection_path)
        print(f"{now_iso()} selection_reuse count={len(selection)} threshold={THRESHOLD}", flush=True)
    else:
        selection = make_parent_selection(parent, selection_path)
        print(
            f"{now_iso()} selection_frozen count={len(selection)} threshold={THRESHOLD}",
            flush=True,
        )

    completed = set()
    repairs = []
    if provenance_path.exists():
        repairs = read_jsonl(provenance_path)
        completed = {r["job_id"] for r in repairs}

    pending = [r for r in selection if r["job_id"] not in completed]
    rows = load_all(out, args.num_shards)

    idiom_needed = any(r["domain"] == "idiom" for r in pending)
    gen = IdiomGenerator(args.model, args.device) if idiom_needed else None

    start = time.time()
    total = len(selection)
    done = len(completed)
    ok = done
    failed = 0
    update_progress(out, "RUNNING", "near_dup_repair", done, total, ok, failed, start)

    for sel in pending:
        rows = load_all(out, args.num_shards)
        current = next(r for r in rows if r["job_id"] == sel["job_id"])
        tr, _ = split_maps(rows, current["domain"])
        train_row = tr[current["entity_key"]]
        unseen = unseen_terms(rows, current["domain"])

        if current["domain"] == "chemistry":
            cand, why = chemistry_candidate(current, train_row, unseen)
        else:
            cand, why = idiom_candidate(current, train_row, unseen, gen)

        if cand is None:
            failed += 1
            append_jsonl(
                out / "repairs" / "near_dup_failed.jsonl",
                {
                    "timestamp": now_iso(),
                    "job_id": current["job_id"],
                    "domain": current["domain"],
                    "entity_key": current["entity_key"],
                    "src_term": current["src_term"],
                    "reason": why,
                },
            )
            done += 1
            update_progress(
                out, "RUNNING", "near_dup_repair", done, total, ok, failed,
                start, current["job_id"]
            )
            continue

        new_sim = similarity(train_row, cand)
        before = current["src_text"]
        patch_one(out, current["job_id"], cand, args.num_shards)
        rec = {
            "timestamp": now_iso(),
            "job_id": current["job_id"],
            "domain": current["domain"],
            "entity_key": current["entity_key"],
            "src_term": current["src_term"],
            "parent_similarity": sel["parent_similarity"],
            "new_similarity": round(new_sim, 6),
            "old_uc_text": before,
            "new_uc_text": cand["src_text"],
            "repair_mode": cand["repair_mode"],
            "selection_rule": sel["selection_rule"],
        }
        append_jsonl(provenance_path, rec)
        repairs.append(rec)
        done += 1
        ok += 1
        update_progress(
            out, "RUNNING", "near_dup_repair", done, total, ok, failed,
            start, current["job_id"]
        )

    if failed:
        update_progress(out, "FAIL", "near_dup_repair", done, total, ok, failed, start)
        raise SystemExit(f"NEAR_DUP_REPAIR_FAILED n={failed}")

    rows = load_all(out, args.num_shards)
    update_progress(out, "RUNNING", "full_23k_hard_checks", total, total, ok, 0, start)
    full_checks(rows, out)

    update_progress(out, "RUNNING", "final_near_dup_audit", total, total, ok, 0, start)
    near = near_audit(rows, out)
    export_splits(rows, out)
    write_manifest(out, parent, args.model, selection, repairs, near)

    update_progress(out, "PASS", "FROZEN_CLOSED", total, total, ok, 0, start)
    print("FINAL_23K_HARD_CHECKS=PASS", flush=True)
    print("FINAL_NEAR_DUP_GE_095=0", flush=True)
    print("CHEMISTRY_CONTEXTS=FROZEN_CLOSED", flush=True)
    print("IDIOM_CONTEXTS=FROZEN_CLOSED", flush=True)
    print("TARGETED_CONTEXT_DATA=FROZEN_CLOSED", flush=True)
    print("NEAR_DUP_SUMMARY=" + json.dumps(near, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
