#!/usr/bin/env python3
"""Finalize targeted Section 4.3 contexts from V2R3.

Repairs only protocol-violating rows:
  1) train/UC duplicate UC rows,
  2) seen train/UC rows containing any exact Unseen-500 lexical item.

The frozen 6000/5500/500 identity and split are never changed.
No Student output, evaluator score, or downstream performance is used.

After repair, performs:
  - 23,000-row cardinality/integrity check,
  - exact frozen-term construction invariant,
  - exact and punctuation/whitespace-normalized train-vs-UC duplicate checks,
  - unseen-term contamination check,
  - report-only near-duplicate similarity audit (no arbitrary cutoff/filtering),
  - six split exports + final manifest.
"""

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

SLOT = "[[TERM]]"
META = (
    "这个任务", "根据你的要求", "按照规则", "占位符", "我将",
    "需要构造", "生成一句", "请生成", "目标词",
)

IDIOM_SCENES = [
    "校园讨论", "职场沟通", "新闻评论", "家庭生活",
    "历史叙述", "社会观察", "文学叙事", "日常对话",
]
CHEM_UC_FRAMES = [
    "研究人员在另一项实验中检测了[[TERM]]，并记录了不同条件下的测量结果。",
    "技术报告在样品分析部分提到了[[TERM]]，随后讨论了相关检测指标。",
    "实验记录显示本轮分析涉及[[TERM]]，研究人员进一步核对了样品性质。",
    "质量检测文件列出了[[TERM]]，并附上了与其相关的测试记录。",
    "科研人员在新的材料测试中使用了[[TERM]]，随后比较了多组实验数据。",
    "分析报告提到样品中涉及[[TERM]]，并对后续检测步骤作了说明。",
    "实验室在复核过程中记录了[[TERM]]，同时检查了相关的实验参数。",
    "研究报告在另一种应用条件下讨论了[[TERM]]，并给出了对应观测结果。",
]
CHEM_TRAIN_FRAMES = [
    "实验人员在本次检测中记录了[[TERM]]，并继续分析样品的相关性质。",
    "生产记录中提到了[[TERM]]，技术人员随后核对了对应的工艺参数。",
    "监测报告列出了[[TERM]]，并记录了样品中的相关检测结果。",
    "研究人员在材料分析中讨论了[[TERM]]，并比较了不同条件下的性能。",
    "安全评估文件提到了[[TERM]]，并要求进一步核查其使用条件。",
    "相关法规文件列出了[[TERM]]，并说明了对应的管理要求。",
    "工艺文件中记录了[[TERM]]，随后对相关流程参数进行了复核。",
    "论文在实验部分提到了[[TERM]]，并报告了相关测量结果。",
]


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def normalized_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)
    return text.lower()


def stable_index(text, n):
    d = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(d[:8], "big") % n


def build_from_template(row, template, mode):
    term = row["src_term"]
    if template.count(SLOT) != 1 or term in template:
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


def parse_lr(raw):
    raw = raw.strip()
    m = re.search(r"\{.*\}", raw, re.S)
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
    style = [
        "写成简洁自然的叙事句",
        "写成自然的人物评价句",
        "写成自然的事件描述句",
        "写成自然的评论句",
    ][(variant // len(IDIOM_SCENES)) % 4]
    return (
        "你要构造一句中文成语使用语境。我不会告诉你成语的字面形式。"
        f"它的字典释义是：{definition}。语境类型：{scene}。{style}。"
        "请把句子分成成语前面的 left 和成语后面的 right 两部分。"
        "不要写出或猜测成语本身，不要使用任何占位符，不要解释任务。"
        "只输出一行严格 JSON："
        '{"left":"成语前面的文字","right":"成语后面的文字"}'
    )


class IdiomGenerator:
    def __init__(self, model_path, device):
        import torch
        import torch_npu
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        dev = f"npu:{device}"
        torch.npu.set_device(dev)
        self.dev = dev
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(dev).eval()

    def infer(self, prompt):
        tok = self.tok
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        batch = tok(rendered, return_tensors="pt")
        batch = {k: v.to(self.dev) for k, v in batch.items()}
        with self.torch.no_grad():
            ids = self.model.generate(
                **batch,
                max_new_tokens=128,
                do_sample=False,
                use_cache=True,
            )
        gen = ids[:, batch["input_ids"].shape[1]:]
        return tok.batch_decode(gen, skip_special_tokens=True)[0]

    def regenerate(self, row, forbidden_terms, forbidden_texts, max_attempts=16):
        term = row["src_term"]
        seed = stable_index(row["entity_key"] + "/" + row["split"], 1000000)
        for attempt in range(max_attempts):
            variant = seed + attempt
            raw = self.infer(idiom_prompt(row, variant))
            lr = parse_lr(raw)
            if not lr:
                continue
            left, right = lr
            if term in left or term in right or SLOT in left or SLOT in right:
                continue
            if looks_meta(left + right):
                continue
            templ = left + SLOT + right
            candidate = build_from_template(
                row, templ, "v2r4_independent_idiom_regeneration"
            )
            if candidate is None:
                continue
            src = candidate["src_text"]
            if src in forbidden_texts or normalized_text(src) in {
                normalized_text(x) for x in forbidden_texts
            }:
                continue
            residual = src.replace(term, "", 1)
            if any(u and u in residual for u in forbidden_terms):
                continue
            if not (6 <= len(src) <= 220):
                continue
            candidate["generation_attempts"] = attempt + 1
            candidate["generation_variant"] = variant
            return candidate
        return None


def contamination_hits(row, unseen_terms):
    residual = row["src_text"].replace(row["src_term"], "", 1)
    return [u for u in unseen_terms if u and u in residual]


def pair_maps(rows, domain):
    tr = {r["entity_key"]: r for r in rows if r["domain"] == domain and r["split"] == "train"}
    uc = {r["entity_key"]: r for r in rows if r["domain"] == domain and r["split"] == "uc"}
    return tr, uc


def repair_row(row, reason, all_rows, unseen_terms, idiom_gen):
    domain, split = row["domain"], row["split"]
    tr, uc = pair_maps(all_rows, domain)
    counterpart = uc.get(row["entity_key"]) if split == "train" else tr.get(row["entity_key"])
    forbidden_texts = set()
    if counterpart:
        forbidden_texts.add(counterpart["src_text"])

    if domain == "chemistry":
        frames = CHEM_UC_FRAMES if split == "uc" else CHEM_TRAIN_FRAMES
        start = stable_index(row["entity_key"] + "/" + split + "/" + reason, len(frames))
        for off in range(len(frames)):
            cand = build_from_template(
                row,
                frames[(start + off) % len(frames)],
                f"v2r4_{reason}_chemistry_frame",
            )
            if cand is None:
                continue
            if cand["src_text"] in forbidden_texts:
                continue
            if normalized_text(cand["src_text"]) in {normalized_text(x) for x in forbidden_texts}:
                continue
            if contamination_hits(cand, unseen_terms):
                continue
            return cand
        return None

    if idiom_gen is None:
        return None
    return idiom_gen.regenerate(row, unseen_terms, forbidden_texts)


def replace_rows_in_shards(out, replacements, num_shards):
    patched = 0
    for sid in range(num_shards):
        p = out / "shards" / f"part_{sid:02d}.jsonl"
        rows = read_jsonl(p)
        changed = False
        for i, row in enumerate(rows):
            jid = row["job_id"]
            if jid in replacements:
                rows[i] = replacements[jid]
                patched += 1
                changed = True
        if changed:
            write_jsonl(p, rows)
            print(f"PATCHED_SHARD={sid}")
    if patched != len(replacements):
        raise SystemExit(f"PATCH_COUNT_FAIL patched={patched} expected={len(replacements)}")


def load_all(out, num_shards):
    rows = []
    for sid in range(num_shards):
        p = out / "shards" / f"part_{sid:02d}.jsonl"
        if not p.exists():
            raise SystemExit(f"MISSING_SHARD {p}")
        rows.extend(read_jsonl(p))
    rows.sort(key=lambda r: r["job_id"])
    return rows


def final_checks(rows, out):
    if len(rows) != 23000 or len({r["job_id"] for r in rows}) != 23000:
        raise SystemExit("CARDINALITY_FAIL")

    bad = [r for r in rows if not canonical_ok(r)]
    if bad:
        write_jsonl(out / "audit_bad_construction_v2r4.jsonl", bad)
        raise SystemExit(f"CONSTRUCTION_FAIL n={len(bad)}")

    counts = Counter((r["domain"], r["split"]) for r in rows)
    expected = {
        ("chemistry", "train"): 5500, ("chemistry", "uc"): 5500, ("chemistry", "uw"): 500,
        ("idiom", "train"): 5500, ("idiom", "uc"): 5500, ("idiom", "uw"): 500,
    }
    if counts != Counter(expected):
        raise SystemExit(f"SPLIT_COUNT_FAIL got={dict(counts)}")

    exact_dup, norm_dup = [], []
    for domain in ("chemistry", "idiom"):
        tr, uc = pair_maps(rows, domain)
        for key, u in uc.items():
            t = tr[key]
            if t["src_text"] == u["src_text"]:
                exact_dup.append(u)
            elif normalized_text(t["src_text"]) == normalized_text(u["src_text"]):
                norm_dup.append(u)

    if exact_dup:
        write_jsonl(out / "audit_train_uc_exact_duplicates_v2r4.jsonl", exact_dup)
        raise SystemExit(f"TRAIN_UC_EXACT_DUP_FAIL n={len(exact_dup)}")
    if norm_dup:
        write_jsonl(out / "audit_train_uc_normalized_duplicates_v2r4.jsonl", norm_dup)
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
                        "job_id": r["job_id"], "domain": domain, "split": r["split"],
                        "entity_key": r["entity_key"], "src_term": r["src_term"],
                        "src_text": r["src_text"], "unseen_hits": hits[:20],
                    })
    if contamination:
        write_jsonl(out / "audit_seen_context_unseen_contamination_v2r4.jsonl", contamination)
        raise SystemExit(f"UNSEEN_CONTAMINATION_FAIL n={len(contamination)}")


def make_near_dup_audit(rows, out):
    items = []
    for domain in ("chemistry", "idiom"):
        tr, uc = pair_maps(rows, domain)
        for key, u in uc.items():
            t = tr[key]
            a = normalized_text(t["src_text"])
            b = normalized_text(u["src_text"])
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
        idx = min(len(vals)-1, max(0, round((len(vals)-1)*p)))
        return vals[idx]
    summary = {
        "metric": "difflib.SequenceMatcher ratio over NFKC/punctuation-whitespace-stripped text",
        "gate_policy": "report-only; no arbitrary similarity threshold used for data filtering",
        "pairs": len(items),
        "max": vals[-1],
        "p99": q(0.99),
        "p95": q(0.95),
        "p90": q(0.90),
        "median": q(0.50),
        "top100_file": "near_duplicate_audit_top100.jsonl",
    }
    (out / "near_duplicate_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def export_and_manifest(rows, out, model, repair_log, near_summary):
    groups = {}
    for domain in ("chemistry", "idiom"):
        for split in ("train", "uc", "uw"):
            grp = [r for r in rows if r["domain"] == domain and r["split"] == split]
            groups[(domain, split)] = grp
            write_jsonl(out / f"{domain}_{split}.jsonl", grp)

    prov_path = out / "repairs" / "v2r4_repair_provenance.jsonl"
    write_jsonl(prov_path, repair_log)

    manifest = {
        "status": "CONTEXTS_FROZEN",
        "artifact_version": "targeted-section43-contexts-qwen3-8b-v2r4",
        "total": 23000,
        "counts": {f"{d}/{s}": len(v) for (d,s), v in groups.items()},
        "frozen_lexical_identity_changed": False,
        "frozen_seen_unseen_split_changed": False,
        "generator_role": "Synthesis Model",
        "generator_model": model,
        "enable_thinking": False,
        "decoding": "greedy",
        "hard_checks": {
            "construction_failures": 0,
            "train_uc_exact_duplicates": 0,
            "train_uc_normalized_duplicates": 0,
            "seen_context_unseen_term_contamination": 0,
        },
        "near_duplicate_audit": near_summary,
        "repair": {
            "rows": len(repair_log),
            "basis": "protocol violations only: train/UC duplicate or unseen-term contamination",
            "student_outputs_used": False,
            "evaluation_scores_used": False,
            "provenance_file": str(prov_path),
            "mode_counts": dict(Counter(x["repair_mode"] for x in repair_log)),
        },
        "files": {},
    }
    for p in sorted(out.glob("*.jsonl")):
        if p.name.startswith("audit_"):
            continue
        manifest["files"][p.name] = sha256_file(p)
    for p in sorted(out.glob("*.json")):
        if p.name == "context_freeze_manifest.json":
            continue
        manifest["files"][p.name] = sha256_file(p)

    mp = out / "context_freeze_manifest.json"
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.out)
    rows = load_all(out, args.num_shards)

    if len(rows) != 23000:
        raise SystemExit(f"INPUT_ROWS_FAIL n={len(rows)}")

    repair_log = []
    idiom_gen = None

    # At most four repair passes. Each pass fixes only currently observable protocol violations.
    for pass_id in range(1, 5):
        replacements = {}

        # 1) Train/UC duplicates: modify UC only.
        dup_rows = []
        for domain in ("chemistry", "idiom"):
            tr, uc = pair_maps(rows, domain)
            for key, u in uc.items():
                t = tr[key]
                if (
                    t["src_text"] == u["src_text"]
                    or normalized_text(t["src_text"]) == normalized_text(u["src_text"])
                ):
                    dup_rows.append(u)

        # 2) Seen-context contamination by exact unseen lexical items.
        contam_rows = []
        unseen_by_domain = {}
        for domain in ("chemistry", "idiom"):
            unseen = sorted(
                {r["src_term"] for r in rows if r["domain"] == domain and r["split"] == "uw"},
                key=len, reverse=True,
            )
            unseen_by_domain[domain] = unseen
            for r in rows:
                if r["domain"] == domain and r["split"] in ("train", "uc"):
                    if contamination_hits(r, unseen):
                        contam_rows.append(r)

        targets = {}
        for r in dup_rows:
            targets[r["job_id"]] = (r, "duplicate_uc")
        for r in contam_rows:
            targets[r["job_id"]] = (r, "unseen_contamination")

        print(
            f"PASS={pass_id} DUP_ROWS={len(dup_rows)} "
            f"CONTAM_ROWS={len(contam_rows)} UNIQUE_REPAIR_TARGETS={len(targets)}"
        )

        if not targets:
            break

        if len(targets) > 3000:
            raise SystemExit(f"REFUSE_MASS_REPAIR n={len(targets)}")

        if any(r["domain"] == "idiom" for r, _ in targets.values()) and idiom_gen is None:
            idiom_gen = IdiomGenerator(args.model, args.device)

        for jid, (row, reason) in targets.items():
            cand = repair_row(
                row, reason, rows, unseen_by_domain[row["domain"]], idiom_gen
            )
            if cand is None:
                raise SystemExit(
                    f"ROW_REPAIR_FAIL job_id={jid} domain={row['domain']} "
                    f"split={row['split']} term={row['src_term']!r} reason={reason}"
                )
            replacements[jid] = cand
            repair_log.append({
                "pass": pass_id,
                "job_id": jid,
                "domain": row["domain"],
                "split": row["split"],
                "entity_key": row["entity_key"],
                "src_term": row["src_term"],
                "reason": reason,
                "repair_mode": cand["repair_mode"],
                "old_src_text": row["src_text"],
                "new_src_text": cand["src_text"],
            })

        replace_rows_in_shards(out, replacements, args.num_shards)
        rows = load_all(out, args.num_shards)

    # Final hard checks.
    final_checks(rows, out)
    near = make_near_dup_audit(rows, out)
    manifest = export_and_manifest(rows, out, args.model, repair_log, near)

    print("FINAL_HARD_CHECKS=PASS")
    print("NEAR_DUPLICATE_AUDIT")
    print(json.dumps(near, ensure_ascii=False, indent=2))
    print("REPAIR_MODE_COUNTS", json.dumps(manifest["repair"]["mode_counts"], ensure_ascii=False))
    print("TARGETED_CONTEXT_FREEZE_PASS")


if __name__ == "__main__":
    main()
