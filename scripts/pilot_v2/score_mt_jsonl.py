#!/usr/bin/env python3
# coding: utf-8

import argparse
import json
from pathlib import Path

import sacrebleu


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda x: int(x["index"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    indices = [int(x["index"]) for x in rows]
    if indices != list(range(len(rows))):
        raise RuntimeError("indices must be exactly 0..N-1")

    hyps = [x["student_translation"] for x in rows]
    refs = [x["reference"] for x in rows]

    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs])

    result = {
        "rows": len(rows),
        "BLEU": bleu.score,
        "chrF": chrf.score,
        "sacrebleu_version": getattr(sacrebleu, "__version__", "unknown"),
        "bleu_signature": str(bleu),
        "chrf_signature": str(chrf),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"ROWS={len(rows)}")
    print(f"BLEU={bleu.score:.6f}")
    print(f"CHRF={chrf.score:.6f}")
    print(f"METRICS={args.output}")


if __name__ == "__main__":
    main()
