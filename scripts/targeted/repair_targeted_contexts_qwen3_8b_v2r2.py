#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path

SLOT = "[[TERM]]"
META_MARKERS = (
    "这个任务", "根据你的要求", "按照规则", "占位符", "我将",
    "需要构造", "请生成", "生成一句", "目标词", "具体字面形式",
)

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

def looks_meta(text):
    return any(x in text for x in META_MARKERS)

def clean_template(text):
    text = text.strip()
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    containing = [x for x in lines if SLOT in x]
    text = containing[0] if containing else (lines[0] if lines else "")
    text = re.sub(r"^(?:句子|输出|答案)\s*[:：]\s*", "", text).strip()
    return text.strip('“”"\'')

def valid_canonical(row):
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

def safe_direct_copy(row):
    term = row["src_term"]
    src = row.get("src_text", "")
    templ = row.get("context_template", "")
    return (
        bool(term)
        and src.count(term) == 1
        and SLOT not in src
        and SLOT not in templ
        and not looks_meta(src)
    )

def render_hidden_term_prompt(row, attempt):
    split_name = {
        "train": "训练用单语源句",
        "uc": "未见上下文测试源句",
        "uw": "未见词测试源句",
    }[row["split"]]
    scene = row.get("scene", "").strip()
    definition = row.get("definition", "").strip()
    retry = ""
    if attempt:
        retry = (
            f"这是第{attempt + 1}次尝试。请换一个自然语境。"
            f"输出中必须逐字包含一次且仅一次 {SLOT}。"
        )
    return (
        "你在构造中文到英文机器翻译的受控数据。"
        f"用途：{split_name}。场景：{scene}。"
        f"某个中文成语或固定短语的字典释义是：{definition}。"
        f"请写且只写一句自然中文句子，并在这个成语或固定短语应出现的位置写 {SLOT}。"
        "不要猜测、恢复或写出该词的具体字面形式。"
        f"{SLOT} 必须且只能出现一次。"
        "不要解释任务，不要提到规则、要求或生成过程，不要给英文翻译、标题、编号或额外说明。"
        f"{retry}"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--max-attempts", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out)
    bad = read_jsonl(out / "audit_bad_exact_term.jsonl")
    print(f"REJECTED_ROWS_INPUT={len(bad)}")
    if not bad:
        raise SystemExit("NO_REJECTED_ROWS")
    if len(bad) > 512:
        raise SystemExit(f"REFUSE_LARGE_REPAIR n={len(bad)}")

    direct, regen = [], []
    for row in bad:
        (direct if safe_direct_copy(row) else regen).append(row)

    print(f"SAFE_DIRECT_CANONICALIZE={len(direct)}")
    print(f"REGENERATE_REQUIRED={len(regen)}")

    replacements = {}
    provenance = []

    for row in direct:
        term = row["src_term"]
        src = row["src_text"]
        templ = src.replace(term, SLOT, 1)
        new = dict(row)
        new.update({
            "context_template": templ,
            "src_text": src,
            "valid_placeholder": True,
            "valid_exact_term": True,
            "repair_mode": "v2r2_canonicalize_direct_exact_term",
        })
        assert valid_canonical(new)
        replacements[row["job_id"]] = new
        provenance.append({
            "job_id": row["job_id"],
            "domain": row["domain"],
            "split": row["split"],
            "entity_key": row["entity_key"],
            "src_term": term,
            "repair_mode": new["repair_mode"],
            "old_context_template": row.get("context_template", ""),
            "old_src_text": row.get("src_text", ""),
            "new_context_template": templ,
            "new_src_text": src,
        })

    if regen:
        import torch, torch_npu
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = f"npu:{args.device}"
        torch.npu.set_device(device)
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(device).eval()

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
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            gen = ids[:, batch["input_ids"].shape[1]:]
            return tok.batch_decode(gen, skip_special_tokens=True)[0]

        for row in regen:
            term = row["src_term"]
            chosen = None
            attempts_log = []
            for attempt in range(args.max_attempts):
                raw = infer(render_hidden_term_prompt(row, attempt))
                templ = clean_template(raw)
                attempts_log.append({
                    "attempt": attempt + 1,
                    "raw": raw,
                    "template": templ,
                })
                if (
                    templ.count(SLOT) == 1
                    and term not in templ
                    and not looks_meta(templ)
                    and 6 <= len(templ) <= 220
                ):
                    src = templ.replace(SLOT, term, 1)
                    candidate = dict(row)
                    candidate.update({
                        "context_template": templ,
                        "src_text": src,
                        "generation_attempts": attempt + 1,
                        "valid_placeholder": True,
                        "valid_exact_term": True,
                        "repair_mode": "v2r2_hidden_term_slot_regeneration",
                    })
                    if valid_canonical(candidate):
                        chosen = candidate
                        break

            if chosen is None:
                repair_dir = out / "repairs"
                repair_dir.mkdir(parents=True, exist_ok=True)
                write_jsonl(
                    repair_dir / "v2r2_failed_regeneration.jsonl",
                    [{
                        "job_id": row["job_id"],
                        "src_term": term,
                        "split": row["split"],
                        "attempts": attempts_log,
                    }],
                )
                raise SystemExit(
                    f"REPAIR_GENERATION_FAIL job_id={row['job_id']} term={term!r}"
                )

            replacements[row["job_id"]] = chosen
            provenance.append({
                "job_id": row["job_id"],
                "domain": row["domain"],
                "split": row["split"],
                "entity_key": row["entity_key"],
                "src_term": term,
                "repair_mode": chosen["repair_mode"],
                "old_context_template": row.get("context_template", ""),
                "old_src_text": row.get("src_text", ""),
                "new_context_template": chosen["context_template"],
                "new_src_text": chosen["src_text"],
                "repair_attempts": chosen["generation_attempts"],
            })
            print(
                f"REGENERATED job_id={row['job_id']} split={row['split']} "
                f"term={term!r} attempts={chosen['generation_attempts']}"
            )

    patched = 0
    for shard_id in range(args.num_shards):
        path = out / "shards" / f"part_{shard_id:02d}.jsonl"
        rows = read_jsonl(path)
        changed = False
        for i, row in enumerate(rows):
            jid = row["job_id"]
            if jid in replacements:
                rows[i] = replacements[jid]
                patched += 1
                changed = True
        if changed:
            write_jsonl(path, rows)
            print(f"PATCHED_SHARD={shard_id}")

    if patched != len(replacements):
        raise SystemExit(
            f"PATCH_CARDINALITY_FAIL patched={patched} expected={len(replacements)}"
        )

    repair_dir = out / "repairs"
    repair_dir.mkdir(parents=True, exist_ok=True)
    prov = repair_dir / "v2r2_repair_provenance.jsonl"
    write_jsonl(prov, provenance)

    print(f"REPAIR_PASS rows={patched}")
    print(f"PROVENANCE={prov}")

if __name__ == "__main__":
    main()
