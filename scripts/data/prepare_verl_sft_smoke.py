#!/usr/bin/env python3

from pathlib import Path
import json
import pandas as pd

SRC = Path(
    "/workspace/mtpatcher/data/"
    "pilot_v2_qwen3_06b/human_train6565.jsonl"
)

OUT_DIR = Path(
    "/workspace/mtpatcher/data/"
    "verl_sft_qwen3_06b"
)

OUT = OUT_DIR / "human64.parquet"
N = 64

rows = []

with SRC.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= N:
            break

        x = json.loads(line)

        assert isinstance(x["messages"], list)
        assert len(x["messages"]) == 1
        assert x["messages"][0]["role"] == "user"

        target = x["target_translation"]

        assert isinstance(target, str)
        assert target.strip()

        messages = [
            dict(x["messages"][0]),
            {
                "role": "assistant",
                "content": target,
            },
        ]

        rows.append(
            {
                "index": x["index"],
                "source": x["source"],
                "reference": x["reference"],
                "target_translation": target,
                "messages": messages,
                "src_lang": x["src_lang"],
                "tgt_lang": x["tgt_lang"],
                "data_source": x["data_source"],
                "ability": x["ability"],
            }
        )

assert len(rows) == N

OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(rows)
df.to_parquet(OUT, index=False)

check = pd.read_parquet(OUT)

assert len(check) == N

for x in check["messages"]:
    assert len(x) == 2
    assert x[0]["role"] == "user"
    assert x[1]["role"] == "assistant"

print("source:", SRC)
print("output:", OUT)
print("rows:", len(check))
print("first_messages:", check.iloc[0]["messages"])

print("PREPARE_VERL_SFT_SMOKE_PASS")
