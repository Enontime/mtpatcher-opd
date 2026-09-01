#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

import sacrebleu
import torch
import torch_npu

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


def read_jsonl(
    path
):

    rows = []

    with Path(path).open(
        "r",
        encoding="utf-8-sig",
    ) as f:

        for line in f:

            if line.strip():

                rows.append(
                    json.loads(
                        line
                    )
                )

    rows.sort(
        key=lambda x: int(
            x[
                "index"
            ]
        )
    )

    return rows


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        required=True,
    )

    ap.add_argument(
        "--input",
        required=True,
        type=Path,
    )

    ap.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    ap.add_argument(
        "--metrics",
        required=True,
        type=Path,
    )

    ap.add_argument(
        "--method",
        required=True,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )

    args = ap.parse_args()


    torch.npu.set_device(
        0
    )

    device = torch.device(
        "npu:0"
    )


    rows = read_jsonl(
        args.input
    )


    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
    )


    tokenizer.padding_side = "left"


    if tokenizer.pad_token_id is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )


    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )


    model.to(
        device
    )

    model.eval()


    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    output_rows = []

    started = time.time()


    with torch.inference_mode():

        for start in range(
            0,
            len(rows),
            args.batch_size,
        ):

            batch = rows[
                start:
                start
                + args.batch_size
            ]


            prompts = [
                tokenizer.apply_chat_template(
                    row[
                        "messages"
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )

                for row in batch
            ]


            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )


            encoded = {
                k:
                    v.to(
                        device
                    )

                for k, v in encoded.items()
            }


            input_width = encoded[
                "input_ids"
            ].shape[
                1
            ]


            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=
                    args.max_new_tokens,
                pad_token_id=
                    tokenizer.pad_token_id,
                eos_token_id=
                    tokenizer.eos_token_id,
            )


            new_ids = generated[
                :,
                input_width:
            ]


            texts = tokenizer.batch_decode(
                new_ids,
                skip_special_tokens=True,
            )


            for row, text in zip(
                batch,
                texts,
            ):

                translation = (
                    text.strip()
                )


                if not translation:

                    raise RuntimeError(
                        f"empty translation "
                        f"index={row['index']}"
                    )


                output_rows.append(
                    {
                        "index":
                            int(
                                row[
                                    "index"
                                ]
                            ),

                        "source":
                            row[
                                "source"
                            ],

                        "reference":
                            row[
                                "reference"
                            ],

                        "student_translation":
                            translation,

                        "evaluation_method":
                            args.method,

                        "student_model_path":
                            args.model,
                    }
                )


            print(
                f"EVAL_PROGRESS "
                f"{len(output_rows)}/{len(rows)}",
                flush=True,
            )


    output_rows.sort(
        key=lambda x: int(
            x[
                "index"
            ]
        )
    )


    if [
        int(
            x[
                "index"
            ]
        )
        for x in output_rows
    ] != list(
        range(
            len(rows)
        )
    ):

        raise RuntimeError(
            "evaluation index coverage mismatch"
        )


    with args.output.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in output_rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(
                        ",",
                        ":",
                    ),
                )
                + "\n"
            )


    refs = [
        x[
            "reference"
        ]
        for x in output_rows
    ]

    hyps = [
        x[
            "student_translation"
        ]
        for x in output_rows
    ]


    bleu = sacrebleu.corpus_bleu(
        hyps,
        [
            refs
        ],
    ).score


    chrf = sacrebleu.corpus_chrf(
        hyps,
        [
            refs
        ],
    ).score


    result = {
        "rows":
            len(
                output_rows
            ),

        "BLEU":
            bleu,

        "chrF":
            chrf,

        "seconds":
            time.time()
            - started,

        "method":
            args.method,

        "model":
            args.model,
    }


    args.metrics.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    args.metrics.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    print(
        f"BLEU={bleu:.6f}"
    )

    print(
        f"CHRF={chrf:.6f}"
    )

    print(
        "CORRECTION_FKL_EVAL_PASS"
    )


if __name__ == "__main__":

    main()
