#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path

SLOT = "[[TERM]]"
META = ("这个任务", "根据你的要求", "按照规则", "占位符", "我将", "需要构造", "生成一句")

def read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def write_jsonl(p, rows):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(p)

def meta(text):
    return any(x in text for x in META)

def safe_direct(r):
    t, s, templ = r["src_term"], r.get("src_text",""), r.get("context_template","")
    return bool(t) and s.count(t) == 1 and SLOT not in s and SLOT not in templ and not meta(s)

def canonical(term, template):
    src = template.replace(SLOT, term, 1)
    return (
        template.count(SLOT) == 1 and term not in template and
        src.count(term) == 1 and SLOT not in src and not meta(src)
    ), src

def chemistry_template(scene):
    frames = {
        "实验室操作": "实验人员在本次检测中记录了[[TERM]]，并继续分析样品的相关性质。",
        "化工生产": "生产记录中提到了[[TERM]]，技术人员随后核对了对应的工艺参数。",
        "环境监测": "监测报告列出了[[TERM]]，并记录了样品中的相关检测结果。",
        "材料研究": "研究人员在材料分析中讨论了[[TERM]]，并比较了不同条件下的性能。",
        "产品安全": "安全评估文件提到了[[TERM]]，并要求进一步核查其使用条件。",
        "法规说明": "相关法规文件列出了[[TERM]]，并说明了对应的管理要求。",
        "工业流程": "工艺文件中记录了[[TERM]]，随后对相关流程参数进行了复核。",
        "科研论文": "论文在实验部分提到了[[TERM]]，并报告了相关测量结果。",
    }
    return frames.get(scene, "技术报告中提到了[[TERM]]，并记录了与其相关的实验结果。")

def example_template(r):
    rec = r.get("lexical_record") or {}
    ex = str(rec.get("example") or "").strip()
    term = r["src_term"]
    if not ex or ex == "无":
        return None

    ex = ex.replace("～", "~")
    if "~" in ex:
        ex = ex.replace("~", term)

    # Remove trailing citation marker when possible.
    ex = re.split(r"[★☆]", ex, maxsplit=1)[0].strip()
    ex = ex.strip("“”\"' ")

    if ex.count(term) != 1 or SLOT in ex or meta(ex):
        return None
    templ = ex.replace(term, SLOT, 1)
    ok, _ = canonical(term, templ)
    return templ if ok else None

def parse_lr(text):
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            l = str(obj.get("left",""))
            r = str(obj.get("right",""))
            return l, r
        except Exception:
            pass

    ml = re.search(r"(?:LEFT|left|前半句)\s*[:：]\s*(.+)", text)
    mr = re.search(r"(?:RIGHT|right|后半句)\s*[:：]\s*(.+)", text)
    if ml and mr:
        return ml.group(1).strip(), mr.group(1).strip()
    return None

def lr_prompt(r, attempt):
    definition = r.get("definition","").strip()
    scene = r.get("scene","")
    modes = [
        "让这个成语自然修饰人物的行为或处境",
        "写成自然的叙事句",
        "写成自然的评论句",
        "写成自然的人物描写句",
        "写成自然的新闻或社会观察句",
        "写成自然的校园或职场语境句",
    ]
    mode = modes[attempt % len(modes)]
    return (
        "你要构造一句中文成语使用语境，但我不会告诉你成语的字面形式。"
        f"它的释义是：{definition}。场景：{scene}。{mode}。"
        "请把句子切成成语前面的 left 和成语后面的 right 两部分。"
        "不要写出成语本身，不要使用任何占位符，不要解释任务。"
        "只输出一行严格 JSON，格式为："
        '{"left":"成语前面的文字","right":"成语后面的文字"}'
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=16)
    ap.add_argument("--max-attempts", type=int, default=6)
    args = ap.parse_args()

    out = Path(args.out)
    bad = read_jsonl(out / "audit_bad_exact_term.jsonl")
    print(f"REJECTED_ROWS_INPUT={len(bad)}")
    if len(bad) != 145:
        raise SystemExit(f"EXPECTED_V2_REJECTS_145_GOT_{len(bad)}")

    replacements, prov, need_model = {}, [], []

    for r in bad:
        term = r["src_term"]

        if safe_direct(r):
            templ = r["src_text"].replace(term, SLOT, 1)
            ok, src = canonical(term, templ)
            assert ok
            nr = dict(r)
            nr.update(context_template=templ, src_text=src,
                      valid_placeholder=True, valid_exact_term=True,
                      repair_mode="v2r3_direct_canonicalize")
            replacements[r["job_id"]] = nr
            prov.append({"job_id":r["job_id"],"domain":r["domain"],"split":r["split"],
                         "src_term":term,"repair_mode":nr["repair_mode"]})
            continue

        if r["domain"] == "chemistry":
            templ = chemistry_template(r.get("scene",""))
            ok, src = canonical(term, templ)
            assert ok
            nr = dict(r)
            nr.update(context_template=templ, src_text=src,
                      valid_placeholder=True, valid_exact_term=True,
                      repair_mode="v2r3_deterministic_chemistry_frame")
            replacements[r["job_id"]] = nr
            prov.append({"job_id":r["job_id"],"domain":r["domain"],"split":r["split"],
                         "src_term":term,"repair_mode":nr["repair_mode"]})
            continue

        templ = example_template(r)
        if templ is not None:
            ok, src = canonical(term, templ)
            assert ok
            nr = dict(r)
            nr.update(context_template=templ, src_text=src,
                      valid_placeholder=True, valid_exact_term=True,
                      repair_mode="v2r3_dictionary_example")
            replacements[r["job_id"]] = nr
            prov.append({"job_id":r["job_id"],"domain":r["domain"],"split":r["split"],
                         "src_term":term,"repair_mode":nr["repair_mode"]})
            continue

        need_model.append(r)

    print(f"PRE_MODEL_REPAIRED={len(replacements)}")
    print(f"IDIOM_LR_GENERATION_REQUIRED={len(need_model)}")

    if need_model:
        import torch, torch_npu
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dev = f"npu:{args.device}"
        torch.npu.set_device(dev)
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16,
            trust_remote_code=True, low_cpu_mem_usage=True
        ).to(dev).eval()

        def infer(prompt):
            rendered = tok.apply_chat_template(
                [{"role":"user","content":prompt}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            b = tok(rendered, return_tensors="pt")
            b = {k:v.to(dev) for k,v in b.items()}
            with torch.no_grad():
                ids = model.generate(**b, max_new_tokens=128, do_sample=False, use_cache=True)
            g = ids[:, b["input_ids"].shape[1]:]
            return tok.batch_decode(g, skip_special_tokens=True)[0]

        failures = []
        for r in need_model:
            term = r["src_term"]
            chosen = None
            logs = []
            for a in range(args.max_attempts):
                raw = infer(lr_prompt(r, a))
                lr = parse_lr(raw)
                logs.append({"attempt":a+1,"raw":raw})
                if not lr:
                    continue
                left, right = (x.strip() for x in lr)
                if not left and not right:
                    continue
                if term in left or term in right or SLOT in left or SLOT in right:
                    continue
                if meta(left + right):
                    continue
                templ = left + SLOT + right
                ok, src = canonical(term, templ)
                if ok and 6 <= len(src) <= 220:
                    chosen = (templ, src, a+1)
                    break

            if chosen is None:
                failures.append({
                    "job_id":r["job_id"], "src_term":term, "split":r["split"],
                    "definition":r.get("definition",""), "attempts":logs
                })
                continue

            templ, src, att = chosen
            nr = dict(r)
            nr.update(context_template=templ, src_text=src,
                      generation_attempts=att,
                      valid_placeholder=True, valid_exact_term=True,
                      repair_mode="v2r3_hidden_term_left_right")
            replacements[r["job_id"]] = nr
            prov.append({"job_id":r["job_id"],"domain":r["domain"],"split":r["split"],
                         "src_term":term,"repair_mode":nr["repair_mode"],
                         "attempts":att})

        if failures:
            rd = out / "repairs"
            rd.mkdir(parents=True, exist_ok=True)
            write_jsonl(rd / "v2r3_unrepaired.jsonl", failures)
            raise SystemExit(f"V2R3_UNREPAIRED n={len(failures)}")

    if len(replacements) != len(bad):
        raise SystemExit(f"REPLACEMENT_COUNT_FAIL {len(replacements)} != {len(bad)}")

    patched = 0
    for sid in range(args.num_shards):
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

    if patched != 145:
        raise SystemExit(f"PATCH_COUNT_FAIL {patched}")

    rd = out / "repairs"
    rd.mkdir(parents=True, exist_ok=True)
    write_jsonl(rd / "v2r3_repair_provenance.jsonl", prov)
    print("REPAIR_PASS rows=145")

if __name__ == "__main__":
    main()
