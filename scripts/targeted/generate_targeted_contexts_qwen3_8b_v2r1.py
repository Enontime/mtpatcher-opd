
#!/usr/bin/env python3
"""Generate frozen targeted-MT source contexts for Chemistry/Idiom UC/UW experiments.

V2 uses a placeholder-slot construction: Qwen3-8B generates the context envelope
with [[TERM]], then the exact frozen lexical item is inserted deterministically.
This prevents copy/normalization failures on rare chemistry names and idioms.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path

SCENES = {
    "chemistry": [
        "实验室操作", "化工生产", "环境监测", "材料研究",
        "产品安全", "法规说明", "工业流程", "科研论文",
    ],
    "idiom": [
        "日常叙事", "新闻评论", "校园生活", "职场交流",
        "文学叙述", "社会观察", "人物对话", "议论文",
    ],
}

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def scene_for(domain: str, split: str, key: str):
    salt = f"{domain}|{split}|{key}".encode()
    idx = int.from_bytes(hashlib.sha256(salt).digest()[:4], "big") % len(SCENES[domain])
    return SCENES[domain][idx]

def build(args):
    chem = Path(args.chem)
    idiom = Path(args.idiom)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    specs = [
        ("chemistry", "train", chem / "seen5500.jsonl"),
        ("chemistry", "uc",    chem / "seen5500.jsonl"),
        ("chemistry", "uw",    chem / "unseen500.jsonl"),
        ("idiom",     "train", idiom / "seen5500_v1.jsonl"),
        ("idiom",     "uc",    idiom / "seen5500_v1.jsonl"),
        ("idiom",     "uw",    idiom / "unseen500_v1.jsonl"),
    ]

    jobs = []
    jid = 0
    source_hashes = {}
    for domain, split, path in specs:
        source_hashes[str(path)] = sha256(path)
        for row in read_jsonl(path):
            if domain == "chemistry":
                key = row["entity_key"]
                term = row["zh_name"].strip()
                definition = ""
            else:
                key = row["word"].strip()
                term = row["word"].strip()
                definition = row["definition"].strip()

            jobs.append({
                "job_id": jid,
                "domain": domain,
                "split": split,
                "entity_key": key,
                "src_term": term,
                "definition": definition,
                "scene": scene_for(domain, split, key),
                "lexical_record": row,
            })
            jid += 1

    assert len(jobs) == 23000, len(jobs)
    write_jsonl(out / "jobs.jsonl", jobs)
    (out / "build_manifest.json").write_text(
        json.dumps({
            "status": "BUILT",
            "jobs": len(jobs),
            "counts": {
                "chemistry/train": 5500,
                "chemistry/uc": 5500,
                "chemistry/uw": 500,
                "idiom/train": 5500,
                "idiom/uc": 5500,
                "idiom/uw": 500,
            },
            "generator_role": "Synthesis Model",
            "generator_model": args.model,
            "enable_thinking": False,
            "decoding": "greedy",
            "prompt_version": "targeted-context-slot-v2",
            "context_slot_strategy": "Qwen3-8B generates [[TERM]] context envelope; frozen src_term inserted deterministically",
            "source_hashes": source_hashes,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("BUILD_PASS jobs=23000")

def render_prompt(job, attempt=0):
    term = job["src_term"]
    scene = job["scene"]
    split = job["split"]
    purpose = {
        "train": "训练用单语源句",
        "uc": "未见上下文测试源句",
        "uw": "未见词测试源句",
    }[split]

    if job["domain"] == "chemistry":
        semantic = (
            f"目标化学名称是「{term}」。"
            "请根据这个名称构造自然、真实的中文使用语境。"
        )
    else:
        definition = job["definition"]
        semantic = (
            f"目标成语是「{term}」，释义是：{definition}。"
            "请让这个成语按该释义自然出现在句中。"
        )

    retry = "" if attempt == 0 else (
        f"这是第{attempt + 1}次生成。上一次没有严格遵守占位符规则。"
        "请逐字复制占位符 [[TERM]]，只能出现一次。"
    )

    return (
        "你在构造中文到英文机器翻译的受控数据。"
        f"场景：{scene}。用途：{purpose}。"
        f"{semantic}"
        "请写且只写一句中文句子。"
        "重要：在目标词真正应该出现的位置，不要输出目标词本身，"
        "而要逐字输出占位符 [[TERM]]；[[TERM]] 必须且只能出现一次。"
        "除 [[TERM]] 外不要使用任何方括号占位符。"
        "不要解释目标词，不要给英文翻译，不要标题、编号、引号或额外说明。"
        f"{retry}"
    )

def clean_template(text: str):
    text = text.strip()
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    containing = [x for x in lines if "[[TERM]]" in x]
    text = containing[0] if containing else (lines[0] if lines else "")
    text = re.sub(r"^(?:句子|输出|答案)\s*[:：]\s*", "", text).strip()
    text = text.strip('“”"\'')
    return text

def generate(args):
    import torch
    import torch_npu
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out)
    jobs = read_jsonl(out / "jobs.jsonl")
    shard_jobs = [x for x in jobs if x["job_id"] % args.num_shards == args.shard_id]

    device = f"npu:{args.device}"
    torch.npu.set_device(device)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    def infer(prompts):
        rendered = [
            tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for p in prompts
        ]
        batch = tok(rendered, return_tensors="pt", padding=True)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            ids = model.generate(
                **batch,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        gen = ids[:, batch["input_ids"].shape[1]:]
        return tok.batch_decode(gen, skip_special_tokens=True)

    results = []
    bs = args.batch_size
    for start in range(0, len(shard_jobs), bs):
        chunk = shard_jobs[start:start + bs]
        texts = infer([render_prompt(j, 0) for j in chunk])
        for job, raw in zip(chunk, texts):
            template = clean_template(raw)
            attempts = 1

            def template_ok(t):
                return t.count("[[TERM]]") == 1 and job["src_term"] not in t

            while not template_ok(template) and attempts < 5:
                raw = infer([render_prompt(job, attempts)])[0]
                template = clean_template(raw)
                attempts += 1

            valid_placeholder = template_ok(template)
            if valid_placeholder:
                text = template.replace("[[TERM]]", job["src_term"], 1)
            else:
                text = template

            results.append({
                **job,
                "context_template": template,
                "src_text": text,
                "generation_attempts": attempts,
                "valid_placeholder": valid_placeholder,
                "valid_exact_term": valid_placeholder and job["src_term"] in text,
            })
        if (start // bs) % 20 == 0:
            print(f"shard={args.shard_id} done={min(start+bs,len(shard_jobs))}/{len(shard_jobs)}", flush=True)

    shard_path = out / "shards" / f"part_{args.shard_id:02d}.jsonl"
    write_jsonl(shard_path, results)
    print(f"SHARD_PASS shard={args.shard_id} rows={len(results)} file={shard_path}")

def merge(args):
    out = Path(args.out)
    rows = []
    for i in range(args.num_shards):
        p = out / "shards" / f"part_{i:02d}.jsonl"
        if not p.exists():
            raise SystemExit(f"MISSING_SHARD {p}")
        rows.extend(read_jsonl(p))

    rows.sort(key=lambda x: x["job_id"])
    if len(rows) != 23000 or len({r["job_id"] for r in rows}) != 23000:
        raise SystemExit("CARDINALITY_FAIL")

    # V2R1 validator repair:
    # V2 mistakenly defined exact-term validity as
    # valid_placeholder AND exact-term presence.
    #
    # Some generations ignored [[TERM]] and copied the frozen term
    # directly. If it occurs exactly once, the surrounding sentence
    # is still usable. Canonicalize that row back to the uniform
    # [[TERM]] representation, then deterministically reinsert it.
    direct_copy_canonicalized = 0

    for r in rows:
        term = r["src_term"]
        text = r.get("src_text", "")

        if not r.get("valid_placeholder", False):
            if term and text.count(term) == 1:
                template = text.replace(term, "[[TERM]]", 1)

                r["context_template"] = template
                r["src_text"] = template.replace("[[TERM]]", term, 1)
                r["valid_placeholder"] = True
                r["repair_mode"] = "canonicalized_direct_exact_term_v2r1"
                direct_copy_canonicalized += 1

        # Final invariant is checked from the final source sentence.
        # The prompt contract requested exactly one designated term.
        r["valid_exact_term"] = (
            bool(term) and r.get("src_text", "").count(term) == 1
        )

    print(
        f"V2R1_DIRECT_COPY_CANONICALIZED "
        f"n={direct_copy_canonicalized}"
    )

    bad_term = [r for r in rows if not r["valid_exact_term"]]
    if bad_term:
        write_jsonl(out / "audit_bad_exact_term.jsonl", bad_term)
        raise SystemExit(f"EXACT_TERM_FAIL n={len(bad_term)}")

    expected = {
        ("chemistry", "train"): 5500,
        ("chemistry", "uc"): 5500,
        ("chemistry", "uw"): 500,
        ("idiom", "train"): 5500,
        ("idiom", "uc"): 5500,
        ("idiom", "uw"): 500,
    }
    groups = {}
    for k in expected:
        groups[k] = [r for r in rows if (r["domain"], r["split"]) == k]
        if len(groups[k]) != expected[k]:
            raise SystemExit(f"COUNT_FAIL {k}={len(groups[k])}")

    dup = []
    for domain in ("chemistry", "idiom"):
        train = {r["entity_key"]: r["src_text"] for r in groups[(domain, "train")]}
        for r in groups[(domain, "uc")]:
            if train.get(r["entity_key"]) == r["src_text"]:
                dup.append(r)
    if dup:
        write_jsonl(out / "audit_train_uc_exact_duplicates.jsonl", dup)
        raise SystemExit(f"TRAIN_UC_DUP_FAIL n={len(dup)}")

    contamination = []
    for domain in ("chemistry", "idiom"):
        unseen_terms = sorted({r["src_term"] for r in groups[(domain, "uw")]}, key=len, reverse=True)
        for split in ("train", "uc"):
            for r in groups[(domain, split)]:
                # Remove the designated seen term once before checking. This prevents
                # unavoidable substring relations inside the designated term itself
                # from being misclassified as generated-context leakage.
                residual = r["src_text"].replace(r["src_term"], "", 1)
                hits = [u for u in unseen_terms if u and u in residual]
                if hits:
                    contamination.append({
                        "domain": domain,
                        "split": split,
                        "entity_key": r["entity_key"],
                        "src_term": r["src_term"],
                        "src_text": r["src_text"],
                        "unseen_hits": hits[:20],
                    })
    if contamination:
        write_jsonl(out / "audit_seen_context_unseen_term_contamination.jsonl", contamination)
        raise SystemExit(f"UNSEEN_TERM_CONTAMINATION_FAIL n={len(contamination)}")

    for (domain, split), data in groups.items():
        write_jsonl(out / f"{domain}_{split}.jsonl", data)

    manifest = {
        "status": "CONTEXTS_FROZEN",
        "total": 23000,
        "counts": {f"{d}/{s}": len(v) for (d, s), v in groups.items()},
        "exact_term_failures": 0,
        "exact_term_validator": "final_src_text_exactly_once_v2r1",
        "direct_copy_canonicalized": direct_copy_canonicalized,
        "train_uc_exact_duplicates": 0,
        "seen_context_unseen_term_contamination": 0,
        "generator_role": "Synthesis Model",
        "generator_model": args.model,
        "enable_thinking": False,
        "decoding": "greedy",
        "prompt_version": "targeted-context-slot-v2",
        "context_slot_strategy": "Qwen3-8B generates [[TERM]] context envelope; frozen src_term inserted deterministically",
        "files": {},
    }
    for p in sorted(out.glob("*.jsonl")):
        if p.name.startswith("audit_"):
            continue
        manifest["files"][p.name] = sha256(p)
    (out / "context_freeze_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("TARGETED_CONTEXT_FREEZE_PASS")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "generate", "merge"])
    ap.add_argument("--chem")
    ap.add_argument("--idiom")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=16)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    args = ap.parse_args()
    {"build": build, "generate": generate, "merge": merge}[args.mode](args)

if __name__ == "__main__":
    main()
