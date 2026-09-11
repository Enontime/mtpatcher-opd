#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path

SLOT = "[[TERM]]"

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-shards", type=int, default=16)
    args = ap.parse_args()

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

    bad_term = []
    for r in rows:
        term = r["src_term"]
        src = r.get("src_text", "")
        templ = r.get("context_template", "")
        valid = (
            bool(term)
            and src.count(term) == 1
            and SLOT not in src
            and templ.count(SLOT) == 1
            and term not in templ
        )
        r["valid_placeholder"] = valid
        r["valid_exact_term"] = valid
        if not valid:
            bad_term.append(r)

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
        unseen_terms = sorted(
            {r["src_term"] for r in groups[(domain, "uw")]},
            key=len,
            reverse=True,
        )
        for split in ("train", "uc"):
            for r in groups[(domain, split)]:
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
        write_jsonl(
            out / "audit_seen_context_unseen_term_contamination.jsonl",
            contamination,
        )
        raise SystemExit(
            f"UNSEEN_TERM_CONTAMINATION_FAIL n={len(contamination)}"
        )

    for (domain, split), data in groups.items():
        write_jsonl(out / f"{domain}_{split}.jsonl", data)

    repair_prov = out / "repairs" / "v2r2_repair_provenance.jsonl"
    repairs = read_jsonl(repair_prov) if repair_prov.exists() else []

    manifest = {
        "status": "CONTEXTS_FROZEN",
        "artifact_version": "targeted-section43-contexts-qwen3-8b-v2r2",
        "total": 23000,
        "counts": {f"{d}/{s}": len(v) for (d, s), v in groups.items()},
        "exact_term_failures": 0,
        "construction_invariant": (
            "src has frozen term exactly once; no literal [[TERM]] remains; "
            "template has exactly one [[TERM]] and excludes frozen term"
        ),
        "train_uc_exact_duplicates": 0,
        "seen_context_unseen_term_contamination": 0,
        "generator_role": "Synthesis Model",
        "generator_model": args.model,
        "enable_thinking": False,
        "decoding": "greedy",
        "prompt_version": "targeted-context-slot-v2 + v2r2 residual repair",
        "repair": {
            "rows": len(repairs),
            "direct_canonicalized": sum(
                r.get("repair_mode") == "v2r2_canonicalize_direct_exact_term"
                for r in repairs
            ),
            "hidden_term_regenerated": sum(
                r.get("repair_mode") == "v2r2_hidden_term_slot_regeneration"
                for r in repairs
            ),
            "selection_basis": (
                "V2 construction-gate rejection only; "
                "no Student/eval/downstream outcome used"
            ),
            "provenance_file": str(repair_prov),
        },
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

if __name__ == "__main__":
    main()
