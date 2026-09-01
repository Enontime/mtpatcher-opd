#!/usr/bin/env bash
set -euo pipefail

source /workspace/mtpatcher/project_env.sh

EXP="mtpatcher_v3_full6565_20260823"

RAW="$DATA_ROOT/rq0/news.2023.zh.shuffled.deduped.gz"

TRAIN="$DATA_ROOT/pilot_v2_qwen3_06b/human_train6565.jsonl"
WMT="$DATA_ROOT/pilot_v2_qwen3_06b/wmt24_zh_en998.jsonl"
FLORES="$DATA_ROOT/pilot_v2_qwen3_06b/flores_zh_en1012.jsonl"
CHALLENGE="$DATA_ROOT/pilot_v2_qwen3_06b/challenge_zh_en197.jsonl"

OUT="$DATA_ROOT/$EXP/rq0_newscrawl50k_sources_v1.jsonl"
MANIFEST="$DATA_ROOT/$EXP/rq0_newscrawl50k_sources_v1.manifest.json"

LOG="$LOG_ROOT/$EXP/rq0_build_newscrawl50k.log"

mkdir -p \
    "$(dirname "$OUT")" \
    "$(dirname "$LOG")"


echo "========== RQ0 NEWCRAWL50K BUILD =========="


echo "[1] gzip audit"

gzip -t "$RAW"

echo "GZIP_PASS"


echo "[2] create builder"

cat > /tmp/build_newscrawl50k_pool.py <<'PY'
import argparse
import gzip
import hashlib
import json
import random
import unicodedata
from pathlib import Path


def norm(s):
    return " ".join(
        unicodedata.normalize(
            "NFKC",
            str(s)
        ).split()
    ).strip()


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for x in iter(
            lambda:f.read(1024*1024),
            b""
        ):
            h.update(x)

    return h.hexdigest()


def load_sources(path):
    ans=set()

    with open(
        path,
        encoding="utf-8"
    ) as f:
        for line in f:
            try:
                obj=json.loads(line)

                for key in [
                    "source",
                    "src",
                    "zh",
                    "input"
                ]:
                    if key in obj:
                        s=norm(obj[key])
                        if s:
                            ans.add(s)
                            break
            except Exception:
                pass

    return ans


def zh_ratio(s):
    if not s:
        return 0

    zh=sum(
        1 for c in s
        if "\u4e00" <= c <= "\u9fff"
    )

    return zh/max(
        1,
        len(
            [
                c for c in s
                if not c.isspace()
            ]
        )
    )


def main():

    ap=argparse.ArgumentParser()

    ap.add_argument("--raw")
    ap.add_argument("--train")
    ap.add_argument("--wmt")
    ap.add_argument("--flores")
    ap.add_argument("--challenge")
    ap.add_argument("--out")
    ap.add_argument("--manifest")

    args=ap.parse_args()


    forbidden=set()

    for x in [
        args.train,
        args.wmt,
        args.flores,
        args.challenge
    ]:
        forbidden |= load_sources(x)


    print(
        "FORBIDDEN=",
        len(forbidden)
    )


    stats={
        "raw_lines":0,
        "accepted":0,
        "empty":0,
        "short":0,
        "long":0,
        "low_zh":0,
        "duplicate":0,
        "overlap":0
    }


    candidates=[]
    seen=set()


    with gzip.open(
        args.raw,
        "rt",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        for line in f:

            stats["raw_lines"]+=1

            s=norm(line)

            if not s:
                stats["empty"]+=1
                continue

            if len(s)<8:
                stats["short"]+=1
                continue

            if len(s)>400:
                stats["long"]+=1
                continue

            if zh_ratio(s)<0.2:
                stats["low_zh"]+=1
                continue

            if s in forbidden:
                stats["overlap"]+=1
                continue

            if s in seen:
                stats["duplicate"]+=1
                continue

            seen.add(s)
            candidates.append(s)

            if len(candidates)>=100000:
                break


    if len(candidates)<50000:
        raise RuntimeError(
            f"only {len(candidates)} candidates"
        )


    random.Random(
        20260825
    ).shuffle(candidates)


    selected=candidates[:50000]


    with open(
        args.out,
        "w",
        encoding="utf-8"
    ) as f:

        for i,s in enumerate(selected):

            row={
                "index":i,
                "source":s,
                "data_source":
                    "WMT_NewsCrawl_2023_ZH"
            }

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )+"\n"
            )


    stats["accepted"]=len(selected)

    manifest={

        "protocol":
            "RQ0_B_NEWCRAWL50K_V1",

        "rows":
            len(selected),

        "raw_sha256":
            sha256(args.raw),

        "output_sha256":
            sha256(args.out),

        "stats":
            stats,

        "forbidden_count":
            len(forbidden)
    }


    Path(
        args.manifest
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


    print(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False
        )
    )

    print(
        "RQ0_NEWCRAWL50K_PASS"
    )


if __name__=="__main__":
    main()
PY


echo "[3] build pool"


python /tmp/build_newscrawl50k_pool.py \
    --raw "$RAW" \
    --train "$TRAIN" \
    --wmt "$WMT" \
    --flores "$FLORES" \
    --challenge "$CHALLENGE" \
    --out "$OUT" \
    --manifest "$MANIFEST"


echo "[4] audit"

wc -l "$OUT"

sha256sum "$OUT"

cat "$MANIFEST"


echo "RQ0_NEWCRAWL50K_ALL_PASS"

