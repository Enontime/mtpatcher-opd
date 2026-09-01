#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

import torch
import torch_npu
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


IGNORE_INDEX = -100


class PatcherDataset(Dataset):

    def __init__(
        self,
        path,
        tokenizer,
        max_length=512,
        max_rows=None,
    ):
        self.rows = []

        with open(
            path,
            encoding="utf-8-sig",
        ) as f:
            for line in f:
                if line.strip():
                    self.rows.append(
                        json.loads(line)
                    )

        if max_rows is not None:
            self.rows = self.rows[
                :max_rows
            ]

        self.tokenizer = tokenizer
        self.max_length = max_length

        self.prompt_truncated = 0
        self.response_truncated = 0
        self.max_combined_length_seen = 0
        self.zero_response_examples = 0

        # Audit once at dataset initialization.
        #
        # OFFICIAL RELEASE src/data/sft_dataset.py:
        # prompt and response are tokenized/truncated INDEPENDENTLY
        # with max_length=512, and are concatenated afterwards.
        for x in self.rows:
            input_ids, labels = self.encode_row(
                x,
                audit_only=True,
            )

            self.max_combined_length_seen = max(
                self.max_combined_length_seen,
                len(input_ids),
            )

            response_label_count = sum(
                1
                for y in labels
                if y != IGNORE_INDEX
            )

            if response_label_count == 0:
                self.zero_response_examples += 1

        print(
            json.dumps(
                {
                    "rows":
                        len(self.rows),

                    "official_prompt_max_length":
                        512,

                    "official_response_max_length":
                        512,

                    "prompt_truncated_examples":
                        self.prompt_truncated,

                    "response_truncated_examples":
                        self.response_truncated,

                    "max_combined_length_seen":
                        self.max_combined_length_seen,

                    "zero_response_examples":
                        self.zero_response_examples,
                },
                indent=2,
            ),
            flush=True,
        )

        if self.zero_response_examples > 0:
            raise RuntimeError(
                "Some examples contain zero trainable response tokens."
            )

    def __len__(self):
        return len(self.rows)

    def encode_row(
        self,
        x,
        audit_only=False,
    ):
        # ------------------------------------------------------------
        # PROMPT SERIALIZATION
        #
        # ADAPTATION:
        # The prompt text itself comes from the frozen MT-PATCHER
        # release. Qwen3 requires its own chat serialization.
        # ------------------------------------------------------------

        prompt_text = (
            self.tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": x["prompt"],
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

        # ------------------------------------------------------------
        # REPO-EXACT PREPROCESSING SEMANTICS
        #
        # Official:
        #
        # tokenized_prompts =
        #   tokenizer(prompt, truncation=True, max_length=512)
        #
        # tokenized_responses =
        #   tokenizer(response, truncation=True, max_length=512)
        #
        # input = prompt + response + eos
        # label = -100*prompt + response + eos
        #
        # IMPORTANT:
        # 512 applies independently to prompt and response.
        # The final concatenated sequence can therefore reach 1025.
        # ------------------------------------------------------------

        prompt_full = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]

        response_full = self.tokenizer(
            x["response"],
            add_special_tokens=False,
        )["input_ids"]

        if audit_only:
            if len(prompt_full) > 512:
                self.prompt_truncated += 1

            if len(response_full) > 512:
                self.response_truncated += 1

        prompt_ids = prompt_full[:512]
        response_ids = response_full[:512]

        eos = []

        if self.tokenizer.eos_token_id is not None:
            eos = [
                self.tokenizer.eos_token_id
            ]

        input_ids = (
            prompt_ids
            + response_ids
            + eos
        )

        labels = (
            [IGNORE_INDEX]
            * len(prompt_ids)
            + response_ids
            + eos
        )

        return (
            input_ids,
            labels,
        )

    def __getitem__(self, idx):
        input_ids, labels = (
            self.encode_row(
                self.rows[idx]
            )
        )

        return {
            "input_ids":
                input_ids,

            "labels":
                labels,

            "attention_mask":
                [1] * len(input_ids),
        }


class Collator:

    def __init__(
        self,
        tokenizer,
    ):
        self.tokenizer = tokenizer

    def __call__(
        self,
        batch,
    ):
        max_len = max(
            len(x["input_ids"])
            for x in batch
        )

        input_ids = []
        labels = []
        attention_mask = []

        for x in batch:
            pad = (
                max_len
                - len(
                    x["input_ids"]
                )
            )

            input_ids.append(
                x["input_ids"]
                + [
                    self.tokenizer.pad_token_id
                ]
                * pad
            )

            labels.append(
                x["labels"]
                + [IGNORE_INDEX]
                * pad
            )

            attention_mask.append(
                x["attention_mask"]
                + [0] * pad
            )

        return {
            "input_ids":
                torch.tensor(
                    input_ids,
                    dtype=torch.long,
                ),

            "labels":
                torch.tensor(
                    labels,
                    dtype=torch.long,
                ),

            "attention_mask":
                torch.tensor(
                    attention_mask,
                    dtype=torch.long,
                ),
        }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        required=True,
    )

    ap.add_argument(
        "--train",
        required=True,
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    ap.add_argument(
        "--max-rows",
        type=int,
        default=None,
    )

    ap.add_argument(
        "--max-steps",
        type=int,
        default=-1,
    )

    ap.add_argument(
        "--smoke",
        action="store_true",
    )

    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
    )

    tok.padding_side = "right"

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    ds = PatcherDataset(
        path=args.train,
        tokenizer=tok,

        # Kept as metadata for compatibility.
        # Actual official preprocessing independently caps
        # prompt and response at 512 inside encode_row().
        max_length=512,

        max_rows=args.max_rows,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )

    model.config.use_cache = False

    #
    # PAPER Appendix B
    #
    epochs = 3
    lr = 1e-5

    #
    # Paper global batch = 64.
    #
    # Ascend realization:
    # 16 devices × micro1 × GA4 = 64.
    #
    per_device = 1
    ga = 4

    train_args = TrainingArguments(
        output_dir=
            args.output,

        num_train_epochs=
            epochs,

        learning_rate=
            lr,

        per_device_train_batch_size=
            per_device,

        gradient_accumulation_steps=
            ga,

        #
        # REPO-EXACT
        #
        lr_scheduler_type=
            "cosine",

        warmup_ratio=
            0.03,

        weight_decay=
            0.0,

        seed=
            42,

        bf16=
            True,

        logging_steps=
            10,

        save_strategy=
            "epoch"
            if not args.smoke
            else "no",

        report_to=
            "none",

        remove_unused_columns=
            False,

        gradient_checkpointing=
            False,

        max_grad_norm=
            1.0,

        max_steps=
            args.max_steps,

        #
        # SYSTEM ADAPTATION:
        # Official release uses DeepSpeed ZeRO-2.
        # Current Ascend reproduction uses FSDP full-shard.
        #
        # It remains FULL PARAMETER FINETUNING.
        #
        fsdp=
            "full_shard auto_wrap",

        fsdp_config={
            "transformer_layer_cls_to_wrap":
                [
                    "Qwen3DecoderLayer"
                ],

            "use_orig_params":
                True,

            "sync_module_states":
                True,

            "activation_checkpointing":
                True,
        },

        ddp_find_unused_parameters=
            False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        data_collator=Collator(tok),
        processing_class=tok,
    )

    trainer.train()

    if not args.smoke:
        trainer.save_model(
            args.output
        )

        tok.save_pretrained(
            args.output
        )

    if trainer.is_world_process_zero():
        manifest = {
            "protocol":
                "MT_PATCHER_PATCHER_FULLFT_PAPER_FAITHFUL_V2",

            "rows":
                len(ds),

            "fidelity": {
                "full_parameter_finetuning":
                    "PAPER_EXACT",

                "epochs_3":
                    "PAPER_EXACT",

                "learning_rate_1e-5":
                    "PAPER_EXACT",

                "effective_global_batch_64":
                    "PAPER_EXACT",

                "response_only_loss":
                    "PAPER_EXACT",

                "cosine_scheduler":
                    "REPO_EXACT",

                "warmup_ratio_0.03":
                    "REPO_EXACT",

                "weight_decay_0":
                    "REPO_EXACT",

                "seed_42":
                    "REPO_EXACT",

                "sft_prompt_max_length_512":
                    "REPO_EXACT: src/data/sft_dataset.py",

                "sft_response_max_length_512":
                    "REPO_EXACT: src/data/sft_dataset.py",

                "combined_sequence_length":
                    "REPO_EXACT SEMANTICS: prompt<=512 + response<=512 + EOS",

                "physical_batch":
                    "ADAPTATION: 16x micro1 x GA4",

                "distributed_optimizer":
                    "SYSTEM_ADAPTATION: FSDP full-shard replaces DeepSpeed ZeRO-2",

                "patcher_backbone":
                    "ADAPTATION: Qwen3-8B",

                "base_vs_chat":
                    "UNRESOLVED",
            },
        }

        Path(
            args.output
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        Path(
            args.output,
            "paper_fidelity_manifest.json",
        ).write_text(
            json.dumps(
                manifest,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        if args.smoke:
            print(
                "PATCHER_FULLFT_SMOKE_PASS",
                flush=True,
            )
        else:
            print(
                "PATCHER_FULLFT_PAPER_V2_PASS",
                flush=True,
            )


if __name__ == "__main__":
    main()
