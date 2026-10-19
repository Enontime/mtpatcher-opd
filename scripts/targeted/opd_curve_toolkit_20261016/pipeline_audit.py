#!/usr/bin/env python3
"""Read-only OPD/SFT curve and checkpoint inventory. Does NOT run inference."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

DEFAULT_ROOT = Path("/workspace/mtpatcher")
ARMS = {
    "opd_lr1e5": "wa_opd_lr1e5_chemistry_20261014/O2",
    "opd_lr2e5": "wa_opd_lr2e5_chemistry_20261014/O2",
    "opd_lr3e5": "wa_opd_lr3e5_chemistry_20261014/O2",
    "sft_lr2e5": "chemistry_clean_sft_parity_20261015/KC_SFT_MATCHED_PROMPT_LR2E5",
    "sft_lr3e5": "chemistry_clean_sft_parity_20261015/KC_SFT_MATCHED_PROMPT_LR3E5",
}
METRIC_MARKERS = ("loss", "kl", "grad", "lr", "entropy", "mass", "accuracy", "hit",
                  "bleu", "chrf", "token", "throughput", "speed", "time", "norm")


def sha256(path: Path, limit: int = 20_000_000) -> str | None:
    if not path.is_file() or path.stat().st_size > limit:
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd(*args: str, cwd: Path | None = None) -> str:
    try:
        p = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=7, check=False)
        return (p.stdout or p.stderr).strip()[:16000]
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"NOT_AVAILABLE:{e}"


def find_small_metadata(base: Path) -> list[dict]:
    if not base.exists():
        return []
    out = []
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(base)
        if len(rel.parts) > 7:
            continue
        if path.name.endswith((".safetensors", ".pt", ".bin", ".lock", ".npy")):
            continue
        if not (path.suffix in {".json", ".jsonl", ".log", ".csv"}
                or "tfevents" in path.name):
            continue
        size = path.stat().st_size
        if size > 60_000_000:
            continue
        out.append({"path": str(rel), "size_bytes": size,
                    "sha256": sha256(path)})
    return out


def read_jsonl_metrics(path: Path, max_mb: int = 45) -> dict:
    out = {"path": str(path), "rows": 0, "invalid": 0,
           "steps": [], "metrics": {}, "raw_keys": []}
    if path.stat().st_size > max_mb * 1_000_000:
        out["error"] = "TOO_LARGE"
        return out
    tags = defaultdict(list)
    raw_keys = Counter()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    continue
            except ValueError:
                out["invalid"] += 1
                continue
            out["rows"] += 1
            if out["rows"] == 1:
                out["example_record"] = {k: str(v)[:120] for k,v in row.items()
                                         if k not in {"model_translation","source","reference"}}
            for k,v in row.items():
                raw_keys[k] += 1
                if k.lower() in ("step", "global_step", "optimizer_step", "update",
                                 "updates", "completed_updates") and isinstance(v, (int, float)):
                    if len(out["steps"]) < 100000:
                        out["steps"].append(v)
                if isinstance(v, (int,float)) and not isinstance(v,bool) and any(
                     marker in k.lower() for marker in METRIC_MARKERS):
                    z=tags[k]
                    if len(z)<100000:
                        z.append(float(v))
    out["raw_keys"] = [k for k,_ in raw_keys.most_common(65)]
    steps=out["steps"]
    out["step_range"] = [min(steps),max(steps)] if steps else None
    out["metrics"] = {k:{"points":len(v),"first":v[0],"last":v[-1],
                         "min":min(v),"max":max(v)}
                     for k,v in sorted(tags.items()) if v}
    del out["steps"]
    return out


def event_tags(path: Path) -> dict:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {"status":"TENSORBOARD_PACKAGE_MISSING", "path":str(path)}
    try:
        x=EventAccumulator(str(path), size_guidance={"scalars":3000,"tensors":1000})
        x.Reload()
        names=x.Tags()
        detail={}
        for tag in names.get("scalars",[]):
            ev=x.Scalars(tag)
            if ev:
                detail[tag]={"n_sampled":len(ev),"first_step":ev[0].step,
                             "last_step":ev[-1].step, "last_value":ev[-1].value}
        tensor_tags={}
        for tag in names.get("tensors",[]):
            ev=x.Tensors(tag)
            if ev:
                tensor_tags[tag]={"n_sampled":len(ev),
                                  "first_step":ev[0].step,"last_step":ev[-1].step}
        return {"status":"OK","path":str(path),"scalars":detail,
                "tensor_tags":tensor_tags}
    except Exception as e:
        return {"status":"ERROR","path":str(path),"reason":str(e)[:250]}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    ap.add_argument("--out",type=Path,default=None)
    args=ap.parse_args()
    root=args.root
    target=root/"runs/targeted"
    out=args.out or root/"runs/targeted/opd_curve_audit_20261016"
    if not target.exists():
        raise SystemExit(f"No run directory: {target}")
    repo=root/"repo/MT-Patcher-Reproduction-Ascend"
    obj={
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "status":"INVENTORY_ONLY",
        "root":str(root),
        "git_head":cmd("git","rev-parse","HEAD",cwd=repo),
        "git_branch":cmd("git","branch","--show-current",cwd=repo),
        "git_dirty":cmd("git","status","--short",cwd=repo),
        "verl_git_head":cmd("git","rev-parse","HEAD",cwd=root/"repo/verl-v0.9.0"),
        "environment_candidates":{
            "torch_npu_env":str(root/"envs/mtpatcher-npu-py311/bin/python"),
            "verl_env":str(root/"envs/verl-v0.9.0-a3/bin/python"),
            "verl_repo_exists":(root/"repo/verl-v0.9.0").exists()},
        "arms":{}, "tensorboard":[],
    }
    for arm,rel in ARMS.items():
        p=target/rel
        arm_record={"exists":p.exists(),"path":str(p)}
        if p.exists():
            train=p/"train"
            arm_record["checkpoint_dirs"]=[
                {"path":str(q.relative_to(p)), "has_model_weights":any(
                    (q/name).exists() for name in ("model.safetensors","pytorch_model.bin")),
                 "contains_optimizer_state":any("optim" in x.name.lower() for x in q.glob("*"))}
                for q in sorted(train.rglob("*"))
                if q.is_dir() and (q.name in ("final_hf","resume_hf","hf")
                                  or q.name.startswith(("epoch_","pass_")))]
            arm_record["metadata_files"]=find_small_metadata(train)[:200]
            candidate_jsonls=list(p.rglob("*metrics*.jsonl"))+list(p.rglob("*update*.jsonl"))
            candidate_jsonls=list(dict.fromkeys(x for x in candidate_jsonls
                if x.is_file() and "eval" not in str(x).lower()))
            arm_record["jsonl_metric_series"]=[
                read_jsonl_metrics(q) for q in candidate_jsonls[:15]]
            for q in [train/"manifest.json",train/"progress.json", train/"resume_meta.json",
                      train/"run_contract.json", p/"eval/general_summary.json"]:
                if q.exists() and q.stat().st_size<2_000_000:
                    try:
                        arm_record["json_"+q.name]=json.loads(q.read_text(encoding="utf8"))
                    except Exception as e:
                        arm_record["json_"+q.name+"_error"]=str(e)
        obj["arms"][arm]=arm_record
    # Read BOTH native tfevents and transformed TB bridge, distinguish their origin.
    bridge=target/"tensorboard_bridge_v1_20261015"
    paths=list(target.glob("offline_prefix_support_*/*/tensorboard/events.out.tfevents.*"))
    paths.extend(target.glob("offline_prefix_support_*/**/tensorboard/events.out.tfevents.*"))
    if bridge.exists():
        paths.extend(bridge.glob("**/events.out.tfevents.*"))
    paths=sorted(dict.fromkeys(x for x in paths if x.is_file()))
    obj["tfevent_file_count"]=len(paths)
    # Avoid huge all-experiment event traversal: include targeted arms + native verl + bridge.
    selected=[p for p in paths if ("opd_lr" in str(p) or "sft_matched" in str(p) or
               "kc_sft" in str(p) or "prefix_support_chemistry_st" in str(p))]
    for path in selected[:100]:
        result=event_tags(path)
        result["origin"]="HISTORICAL_BRIDGE" if "tensorboard_bridge_v1" in str(path) else "NATIVE_EVENT"
        obj["tensorboard"].append(result)
    out.mkdir(parents=True,exist_ok=True)
    (out/"audit.json").write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+"\n",
                                  encoding="utf8")
    lines=["# OPD training curve inventory","",f"Git: `{obj['git_head']}`",
           f"Dirty: `{obj['git_dirty'][:500]}`",""]
    for arm,x in obj["arms"].items():
        lines.extend([f"## {arm}","",f"- Run exists: {x['exists']}",
                      f"- Weight directories: {len(x.get('checkpoint_dirs',[]))}",
                      f"- Candidate metric JSONL: {len(x.get('jsonl_metric_series',[]))}"])
        for item in x.get("jsonl_metric_series",[]):
            lines.extend([f"- `{item['path']}` rows={item['rows']} step_range={item['step_range']}",
                          f"  Metrics: {', '.join(item['metrics'].keys()) or 'NONE'}"])
        lines.append("")
    lines.extend(["## TensorBoard","",f"Native + bridge files selected: {len(obj['tensorboard'])}"])
    for b in obj["tensorboard"]:
        lines.append(f"- [{b['origin']}] `{b['path']}` {b['status']} scalar_tags={len(b.get('scalars',{}))} tensor_tags={len(b.get('tensor_tags',{}))}")
        for tag,v in list(b.get("scalars",{}).items())[:60]:
            lines.append(f"  - `{tag}` step={v['first_step']}..{v['last_step']} points_sampled={v['n_sampled']}")
    lines.extend(["","## Boundaries","",
                  "- A model-only HF snapshot is NOT proof of exact optimizer/scheduler resume.",
                  "- Historical-bridge events are derived metrics, NOT original native verl logs.",
                  "- If no intermediate OPD weight snapshots exist, no intermediate free-generation accuracy can be reconstructed from loss scalars.",
                  "- No inference, training, or modification of frozen experimental assets was performed."])
    (out/"audit.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("AUDIT=PASS_INVENTORY_ONLY")
    print(f"JSON={out/'audit.json'}")
    print(f"REPORT={out/'audit.md'}")
    print(f"TFEVENT_FILES_SELECTED={len(obj['tensorboard'])}")
    for arm,a in obj["arms"].items():
        print(f"{arm}: exists={a['exists']} ckpt_dirs={len(a.get('checkpoint_dirs',[]))} metric_jsonl={len(a.get('jsonl_metric_series',[]))}")


if __name__=="__main__":
    main()
