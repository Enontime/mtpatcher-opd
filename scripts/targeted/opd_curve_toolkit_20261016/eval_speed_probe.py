#!/usr/bin/env python3
"""Read-only NPU evaluation throughput probe, reuses original serial evaluator.
A diagnostic ONLY: batched results are never written to frozen official evaluations.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

ROOT=Path("/workspace/mtpatcher")
DEFAULT_EVAL=ROOT/"data/mtpatcher_v3_full6565_20260823/targeted_diagnostic1000_v1_20260910/targeted_diagnostic2000.jsonl"
LEGACY=ROOT/"repo/MT-Patcher-Reproduction-Ascend/scripts/targeted/targeted_wa_opd_overnight_v3.py"
MODEL=ROOT/"runs/targeted/wa_opd_lr3e5_chemistry_20261014/O2/train/final_hf"


def sha256_file(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):
            h.update(b)
    return h.hexdigest()


def records(path: Path, domain: str, n: int):
    found=[]
    with path.open(encoding="utf8") as f:
        for line in f:
            if not line.strip():
                continue
            r=json.loads(line)
            if domain and r.get("eval_domain")!=domain:
                continue
            source=r.get("src_text") or r.get("source")
            if not isinstance(source,str) or not source:
                continue
            found.append((str(r.get("eval_id") or r.get("row_id") or len(found)),source))
    if not found:
        raise RuntimeError("No evaluation rows found")
    if len(found)<n:
        raise RuntimeError(f"Requested {n} from {len(found)} rows")
    # Fixed uniform sampling over frozen order, not cherry-picking.
    idx=[(i*len(found))//n for i in range(n)]
    return [found[i] for i in idx]


def load_legacy(path: Path):
    spec=importlib.util.spec_from_file_location("opd_legacy_eval_readonly",path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Failed to import original eval: {path}")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module,"generate_one_translation"), "Original eval function missing"
    assert hasattr(module,"DIRECT_PROMPT"), "Original DIRECT_PROMPT missing"
    return module


def batch_generate(sources, model, tok, device, source_mod, batch_size):
    import torch
    saved=tok.padding_side
    saved_pad=tok.pad_token_id
    tok.padding_side="left"
    if tok.pad_token_id is None:
        tok.pad_token=tok.eos_token
    out=[]
    try:
        for offset in range(0,len(sources),batch_size):
            subset=sources[offset:offset+batch_size]
            prompts=[tok.apply_chat_template(
                [{"role":"user","content":source_mod.DIRECT_PROMPT.format(source=s)}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
                for s in subset]
            inputs=tok(prompts,return_tensors="pt",padding=True,add_special_tokens=False)
            inputs={k:v.to(device) for k,v in inputs.items()}
            width=inputs["input_ids"].shape[1]
            with torch.inference_mode():
                sequences=model.generate(
                    **inputs,do_sample=False,max_new_tokens=512,
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.pad_token_id,use_cache=True)
            # Both paths must decode generated suffix, NEVER padded prefixes.
            for i in range(sequences.shape[0]):
                out.append(tok.decode(
                    sequences[i,width:].detach().cpu().tolist(),
                    skip_special_tokens=True).strip())
    finally:
        tok.padding_side=saved
        if saved_pad is None:
            tok.pad_token=None
    return out


def timed(sync, fn):
    sync()
    start=time.perf_counter()
    output=fn()
    sync()
    return output, time.perf_counter()-start


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",type=Path,default=MODEL)
    p.add_argument("--data",type=Path,default=DEFAULT_EVAL)
    p.add_argument("--legacy",type=Path,default=LEGACY)
    p.add_argument("--out",type=Path,default=ROOT/"runs/targeted/opd_eval_speed_probe_20261016")
    p.add_argument("--n",type=int,default=16)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--domain",default="chemistry")
    args=p.parse_args()
    if not 1<=args.n<=128 or not 1<=args.batch_size<=16:
        raise SystemExit("n must be 1..128, batch-size 1..16")
    for path in (args.model/"model.safetensors",args.data,args.legacy):
        if not path.is_file():
            raise SystemExit(f"Missing required file: {path}")
    mod=load_legacy(args.legacy)
    eval_rows=records(args.data,args.domain,args.n)
    from transformers import AutoModelForCausalLM,AutoTokenizer
    import torch
    import torch_npu  # noqa
    dev="npu:0"
    torch.npu.set_device(dev)
    tload=time.perf_counter()
    tok=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForCausalLM.from_pretrained(
        args.model,dtype=torch.bfloat16,low_cpu_mem_usage=True).to(dev).eval()
    load_seconds=time.perf_counter()-tload
    def sync():
        torch.npu.synchronize()
    # One warm-up example, not counted.
    _=mod.generate_one_translation(model,tok,eval_rows[0][1],dev)
    sync()
    sources=[row[1] for row in eval_rows]
    a,at=timed(sync,lambda:[
        mod.generate_one_translation(model,tok,s,dev) for s in sources])
    try:
        b,bt=timed(sync,lambda:batch_generate(
            sources,model,tok,dev,mod,args.batch_size))
        batch_error=None
    except Exception as e:
        b=None
        bt=None
        batch_error=f"{type(e).__name__}:{e}"
    mismatches=[]
    if b is not None:
        mismatches=[
            {"eval_id":eid,"serial":x,"batched":y}
            for (eid,_),x,y in zip(eval_rows,a,b) if x!=y
        ]
    lengths=[len(tok.encode(x,add_special_tokens=False)) for x in a]
    result={
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "status":"COMPARE_DONE" if b is not None else "BATCH_FAIL",
        "promotion_status":"CANDIDATE_EQUIVALENT_ON_SAMPLE" if b is not None
                          and not mismatches else "NOT_EQUIVALENT_DO_NOT_PROMOTE",
        "warning":"Microbenchmark only. NO official evaluation rewritten.",
        "model":str(args.model),"data":str(args.data),
        "data_sha256":sha256_file(args.data),
        "legacy_script":str(args.legacy),"legacy_sha256":sha256_file(args.legacy),
        "direct_prompt_sha256":hashlib.sha256(mod.DIRECT_PROMPT.encode()).hexdigest(),
        "eval_id_order":[x[0] for x in eval_rows],
        "n":args.n,"batch_size":args.batch_size,"model_load_seconds":load_seconds,
        "baseline_serial_seconds":at,
        "batch_seconds":bt,
        "serial_examples_per_second":args.n/at,
        "batched_examples_per_second":args.n/bt if bt else None,
        "speedup":at/bt if bt else None,
        "serial_generated_tokens":sum(lengths),
        "serial_tokens_per_second":sum(lengths)/at,
        "serial_output_tokens":{"min":min(lengths),"max":max(lengths),
                                "mean":sum(lengths)/len(lengths)},
        "mismatches":mismatches,
        "batch_error":batch_error,
    }
    args.out.mkdir(parents=True,exist_ok=True)
    f=args.out/f"probe_n{args.n}_batch{args.batch_size}.json"
    f.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print(f"PROBE={result['status']}")
    print(f"PROMOTION={result['promotion_status']}")
    print(f"SERIAL_SECONDS={at:.2f}")
    print(f"BATCH_SECONDS={bt if bt is not None else 'FAIL'}")
    print(f"SPEEDUP={result['speedup']}")
    print(f"EXACT_MISMATCHES={len(mismatches)}")
    print(f"REPORT={f}")


if __name__=="__main__":
    main()
