#!/usr/bin/env python3

import argparse
import json
import math
import statistics
from pathlib import Path

import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer


TARGET_KEYS = [
    "target_translation",
    "target",
    "response",
    "output",
    "translation",
]


def get_target(x):
    for k in TARGET_KEYS:
        v = x.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    raise RuntimeError(
        f"cannot find target; keys={list(x.keys())}"
    )


def get_messages(x):
    m = x.get("messages")

    if isinstance(m, list) and m:
        return m

    src = str(
        x.get("source", "")
    ).strip()

    if not src:
        raise RuntimeError(
            "missing source/messages"
        )

    return [
        {
            "role": "user",
            "content":
                "Translate the following text into English "
                "without additional explanations:\n\n"
                + src
                + "\n\n",
        }
    ]


def load_jsonl(path):
    rows = []

    with open(
        path,
        encoding="utf-8-sig",
    ) as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        required=True,
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    ap.add_argument(
        "--model",
        required=True,
    )

    ap.add_argument(
        "--device",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--max-length",
        type=int,
        default=1024,
    )

    args = ap.parse_args()

    torch.npu.set_device(
        f"npu:{args.device}"
    )

    device = torch.device(
        f"npu:{args.device}"
    )

    tok = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
    )

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )

    model.to(device)
    model.eval()

    rows = load_jsonl(
        args.input
    )

    print(
        f"ROWS={len(rows)}",
        flush=True,
    )

    out_path = Path(
        args.output
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for start in range(
        0,
        len(rows),
        args.batch_size,
    ):

        batch = rows[
            start:
            start + args.batch_size
        ]

        seqs = []
        labels_list = []
        meta = []

        for pos, x in enumerate(batch):

            messages = get_messages(x)
            target = get_target(x)

            prompt = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            pids = tok(
                prompt,
                add_special_tokens=False,
            )["input_ids"]

            tids = tok(
                target,
                add_special_tokens=False,
            )["input_ids"]

            if tok.eos_token_id is not None:
                tids = tids + [
                    tok.eos_token_id
                ]

            ids = pids + tids

            if len(ids) > args.max_length:
                # Preserve complete target where possible.
                overflow = (
                    len(ids)
                    - args.max_length
                )

                if overflow >= len(pids):
                    raise RuntimeError(
                        f"target itself too long "
                        f"row={start+pos} "
                        f"len={len(ids)}"
                    )

                pids = pids[
                    overflow:
                ]

                ids = pids + tids

            labels = (
                [-100] * len(pids)
                + tids
            )

            seqs.append(ids)
            labels_list.append(labels)

            meta.append(
                {
                    "row_position":
                        start + pos,

                    "index":
                        x.get(
                            "index",
                            start + pos,
                        ),

                    "source":
                        x.get(
                            "source",
                            "",
                        ),

                    "target_translation":
                        target,

                    "target_tokens":
                        len(tids),
                }
            )

        max_len = max(
            len(x)
            for x in seqs
        )

        input_ids = []
        attention_mask = []
        labels = []

        for ids, lab in zip(
            seqs,
            labels_list,
        ):
            pad = (
                max_len - len(ids)
            )

            input_ids.append(
                ids
                + [tok.pad_token_id]
                * pad
            )

            attention_mask.append(
                [1] * len(ids)
                + [0] * pad
            )

            labels.append(
                lab
                + [-100] * pad
            )

        input_ids = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=device,
        )

        attention_mask = torch.tensor(
            attention_mask,
            dtype=torch.long,
            device=device,
        )

        labels = torch.tensor(
            labels,
            dtype=torch.long,
            device=device,
        )

        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

        shift_logits = (
            logits[:, :-1, :]
            .float()
        )

        shift_labels = (
            labels[:, 1:]
        )

        vocab = (
            shift_logits.shape[-1]
        )

        token_loss = (
            torch.nn.functional.cross_entropy(
                shift_logits.reshape(
                    -1,
                    vocab,
                ),
                shift_labels.reshape(-1),
                reduction="none",
                ignore_index=-100,
            )
            .reshape(
                shift_labels.shape
            )
        )

        valid = (
            shift_labels != -100
        )

        loss_sum = (
            token_loss
            * valid
        ).sum(dim=1)

        token_count = (
            valid.sum(dim=1)
        )

        mean_nll = (
            loss_sum
            / token_count.clamp_min(1)
        )

        ppl = torch.exp(
            torch.clamp(
                mean_nll,
                max=20,
            )
        )

        for i, m in enumerate(meta):

            result = dict(m)

            result.update(
                {
                    "student_mean_nll":
                        float(
                            mean_nll[i].item()
                        ),

                    "student_ppl":
                        float(
                            ppl[i].item()
                        ),

                    "scored_tokens":
                        int(
                            token_count[i].item()
                        ),
                }
            )

            results.append(
                result
            )

        done = min(
            start + args.batch_size,
            len(rows),
        )

        if (
            done == len(rows)
            or done % 1000 == 0
        ):
            print(
                f"SCORED={done}/{len(rows)}",
                flush=True,
            )

    with out_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for x in results:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )

    vals = [
        x["student_mean_nll"]
        for x in results
    ]

    vals_sorted = sorted(vals)

    def pct(q):
        i = int(
            q
            * (len(vals_sorted) - 1)
        )
        return vals_sorted[i]

    summary = {
        "rows":
            len(results),

        "mean_nll":
            statistics.mean(vals),

        "median_nll":
            statistics.median(vals),

        "p10":
            pct(0.10),

        "p50":
            pct(0.50),

        "p90":
            pct(0.90),

        "p95":
            pct(0.95),

        "p99":
            pct(0.99),

        "max":
            max(vals),
    }

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "STUDENT_NLL_SCORING_PASS"
    )


if __name__ == "__main__":
    main()
