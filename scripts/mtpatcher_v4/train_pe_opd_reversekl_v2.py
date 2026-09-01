#!/usr/bin/env python3

import os
import json
import argparse
import torch
import torch.distributed as dist

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


def load_jsonl(path, limit=None):

    rows=[]

    with open(path, "r", encoding="utf8") as f:

        for line in f:

            if not line.strip():
                continue

            rows.append(json.loads(line))

            if limit and len(rows)>=limit:
                break

    return rows



def build_text(item):

    source = item.get(
        "source",
        ""
    )

    draft = item.get(
        "student_translation",
        ""
    )

    target = item.get(
        "corrected_translation",
        ""
    )


    prompt = (
        "Translate the following sentence.\n\n"
        f"Source:\n{source}\n\n"
        f"Previous translation:\n{draft}\n\n"
        "Correct translation:\n"
    )


    return (
        prompt,
        prompt + target
    )



def main():

    parser=argparse.ArgumentParser()


    parser.add_argument(
        "--student",
        required=True
    )

    parser.add_argument(
        "--teacher",
        required=True
    )

    parser.add_argument(
        "--data",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )


    parser.add_argument(
        "--epochs",
        type=int,
        default=3
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-6
    )


    parser.add_argument(
        "--limit",
        type=int,
        default=None
    )


    args=parser.parse_args()



    rank=int(
        os.environ.get(
            "RANK",
            0
        )
    )


    world=int(
        os.environ.get(
            "WORLD_SIZE",
            1
        )
    )



    if world>1:

        dist.init_process_group(
            backend="hccl"
        )



    device=torch.device(
        f"npu:{rank}"
    )


    if rank==0:

        print("="*70)
        print(
            "PE CONDITIONED REVERSE KL OPD V2"
        )
        print("="*70)



    tokenizer=AutoTokenizer.from_pretrained(
        args.student,
        local_files_only=True
    )



    student=AutoModelForCausalLM.from_pretrained(
        args.student,
        dtype=torch.float16,
        local_files_only=True
    )


    teacher=AutoModelForCausalLM.from_pretrained(
        args.teacher,
        dtype=torch.float16,
        local_files_only=True
    )


    student.to(device)
    teacher.to(device)


    teacher.eval()


    for p in teacher.parameters():
        p.requires_grad=False



    rows=load_jsonl(
        args.data,
        args.limit
    )


    if rank==0:

        print(
            "TRAIN_ROWS=",
            len(rows)
        )



    optimizer=torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )



    student.train()



    step=0



    for epoch in range(args.epochs):


        for item in rows:


            prompt_text, full_text = build_text(item)



            full=tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False
            )


            prompt=tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False
            )



            input_ids=full.input_ids.to(device)

            attention_mask=(
                full.attention_mask.to(device)
            )


            prompt_len=(
                prompt.input_ids.shape[1]
            )



            with torch.no_grad():

                teacher_out=teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )



            student_out=student(
                input_ids=input_ids,
                attention_mask=attention_mask
            )



            # float32 KL
            student_logits=(
                student_out.logits
                .float()
            )

            teacher_logits=(
                teacher_out.logits
                .float()
            )



            student_logp=torch.log_softmax(
                student_logits,
                dim=-1
            )


            teacher_logp=torch.log_softmax(
                teacher_logits,
                dim=-1
            )


            student_prob=torch.softmax(
                student_logits,
                dim=-1
            )



            # reverse KL:
            # KL(student || teacher)

            token_kl=(

                student_prob
                *
                (
                    student_logp
                    -
                    teacher_logp
                )

            ).sum(
                dim=-1
            )



            # only response tokens

            mask=torch.zeros_like(
                token_kl
            )


            mask[:,prompt_len:]=1



            loss=(

                token_kl
                *
                mask

            ).sum() / mask.sum()



            if torch.isnan(loss) or torch.isinf(loss):

                print(
                    "NAN_SKIP",
                    step,
                    flush=True
                )

                optimizer.zero_grad()

                continue



            optimizer.zero_grad()


            loss.backward()



            torch.nn.utils.clip_grad_norm_(
                student.parameters(),
                1.0
            )


            optimizer.step()



            step+=1



            if rank==0 and step%50==0:

                print(
                    {
                        "epoch":epoch,
                        "step":step,
                        "loss":float(
                            loss.detach()
                        )
                    },
                    flush=True
                )



    if rank==0:

        os.makedirs(
            args.output,
            exist_ok=True
        )


        student.save_pretrained(
            args.output
        )


        tokenizer.save_pretrained(
            args.output
        )


        print(
            "PE_OPD_REVERSEKL_V2_FINISH"
        )



if __name__=="__main__":

    main()

